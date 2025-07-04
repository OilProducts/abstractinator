import os
import math
import sys
import time
from collections import defaultdict
import argparse
import importlib.util

import torch
from datasets import load_dataset, Dataset  # Import Dataset for the dummy data
from torch import optim
from torch.utils.data import DataLoader
from typing import List, Dict, Any
from transformers.optimization import get_scheduler

import mlflow  # Logging with MLflow

# Assuming HierarchicalAutoencoder is in abstractinator.py and has KPM updates
from components import HierarchicalAutoencoder
from components.utils import short_num, format_duration

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script for HierarchicalAutoencoder")
    parser.add_argument(
        "--config",
        type=str,
        default="config.py",
        help="Path to the configuration Python file",
    )
    args = parser.parse_args()

    def load_config(path: str):
        spec = importlib.util.spec_from_file_location("config_module", path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module

    config = load_config(args.config)
    DEVICE = config.DEVICE
    N_CPU = config.N_CPU
    exp_config = config.exp_config

    torch.set_float32_matmul_precision("high")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_printoptions(threshold=100_000)
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.recompile_limit = 128

    print(f"Using device: {DEVICE}")

    # --- Experiment Configuration ---
    # Loaded from the provided config file

    # --- MLflow Setup ---
    # See https://mlflow.org/docs/latest/python_api/mlflow.html for API details
    print(f"Initializing MLflow run: {exp_config.get('run_name', 'DefaultRun')}")
    if exp_config.get("project_name"):
        mlflow.set_tracking_uri(f"file:./mlruns/{exp_config['project_name']}")
        print(f"MLflow tracking URI set to: file:./mlruns/{exp_config['project_name']}")
    mlflow.set_experiment(exp_config.get("project_name", "DefaultExperiment"))
    mlflow_run = mlflow.start_run(run_name=exp_config.get("run_name", "DefaultRun"))

    # --- Model, Optimizer ---
    print("Initializing HierarchicalAutoencoder model...")
    model = HierarchicalAutoencoder(
        num_levels=exp_config["num_levels"],
        compressor_level_configs=exp_config["compressor_level_configs"],
        initial_vocab_size=exp_config["initial_vocab_size"],
        expander_dim_scale=exp_config["expander_dim_scale"],
        expander_num_enc_layers=exp_config["expander_num_enc_layers"],
        expander_num_dec_layers=exp_config["expander_num_dec_layers"],
        expander_heads_scale=exp_config["expander_heads_scale"],
        expander_eos_id=exp_config["expander_eos_id"],
        expander_max_len=exp_config["expander_max_len"],  # Pass expander_max_len
        use_decoder_only_expander=exp_config.get("use_decoder_only_expander", False),
        propagate_key_padding_mask=exp_config["propagate_key_padding_mask"],
        aux_lm_loss_weight=exp_config["aux_lm_loss_weight"],
        top_transformer_config=exp_config.get("top_transformer_config", None),  # <<< ADDED
        top_lm_loss_weight=exp_config.get("top_lm_loss_weight", 0.0),
        use_continuous_expander_inputs=exp_config.get("use_continuous_expander_inputs", False)
    ).to(DEVICE)

    # model = torch.compile(model, mode="default", dynamic=True)

    def _count_params(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    num_params = _count_params(model)
    print(f"Model initialized on {DEVICE} with {short_num(num_params)} trainable parameters.")

    # Print parameter counts for major components
    compressor_params = _count_params(model.compressors)
    expander_params = _count_params(model.expanders)
    print(f"  Compressors: {short_num(compressor_params)} parameters")
    print(f"  Expanders: {short_num(expander_params)} parameters")
    if getattr(model, "code_sequence_transformer", None) is not None:
        cst_params = _count_params(model.code_sequence_transformer)
        print(f"  CodeSequenceTransformer: {short_num(cst_params)} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=exp_config["learning_rate"])
    print(f"Optimizer AdamW initialized with learning rate: {exp_config['learning_rate']:.0e}")

    # --- Checkpoint Utilities ---
    def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": step,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    start_epoch = 0
    global_step = 0
    resume_path = exp_config.get("resume_from_checkpoint")
    if resume_path:
        if os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location=DEVICE)
            model.load_state_dict(ckpt.get("model_state", {}))
            optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
            start_epoch = ckpt.get("epoch", 0)
            global_step = ckpt.get("global_step", 0)
            print(f"Resumed from checkpoint '{resume_path}' at step {global_step}, epoch {start_epoch}")
        else:
            print(f"Checkpoint path '{resume_path}' not found. Starting from scratch.")


    # --- Tokenizer ---
    class ByteLevelTokenizer:
        """Simple byte-level tokenizer used for the demo training script.

        The tokenizer works directly on raw UTF-8 bytes.  Optionally BOS and EOS
        tokens are inserted when encoding text.  It keeps byte values (0‑255) as
        their own token IDs and reserves additional IDs for BOS, EOS and padding.

        Attributes:
            bos_id (int): Token ID prepended at the start of a sequence when
                ``add_bos`` is ``True``.
            eos_id (int): Token ID appended at the end of a sequence when
                ``add_eos`` is ``True``.
            pad_id (int): Token ID used for padding sequences.
            add_bos (bool): Whether ``encode`` adds ``bos_id`` by default.
            add_eos (bool): Whether ``encode`` adds ``eos_id`` by default.
            vocab_size (int): Size of the tokenizer vocabulary.
        """

        def __init__(self, bos_id: int = 256, eos_id: int = 257, pad_id: int = 258,
                     add_bos: bool = True, add_eos: bool = True):
            self.bos_id = bos_id
            self.eos_id = eos_id
            self.pad_id = pad_id
            self.add_bos = add_bos
            self.add_eos = add_eos
            # Ensure initial_vocab_size in exp_config matches this.
            # vocab_size here would be max(bos,eos,pad) + 1 = 259 for defaults.
            self.vocab_size = max(bos_id, eos_id, pad_id) + 1
            if self.vocab_size != exp_config["initial_vocab_size"]:
                print(f"Warning: Tokenizer vocab_size ({self.vocab_size}) does not match "
                      f"exp_config['initial_vocab_size'] ({exp_config['initial_vocab_size']}).")

        def encode(self, text: str, add_bos: bool | None = None,
                   add_eos: bool | None = None) -> torch.Tensor:
            if add_bos is None: add_bos = self.add_bos
            if add_eos is None: add_eos = self.add_eos
            raw_bytes = text.encode("utf-8", errors="ignore")
            tokens = []
            if add_bos: tokens.append(self.bos_id)
            tokens.extend(raw_bytes)
            if add_eos: tokens.append(self.eos_id)
            return torch.tensor(tokens, dtype=torch.int)  # Using int16 for tokens

        def decode(self, tokens: list[int] | torch.Tensor, cut_at_eos: bool = False) -> str:
            if isinstance(tokens, torch.Tensor): tokens = tokens.tolist()
            if cut_at_eos and (self.eos_id in tokens):
                try:
                    eos_index = tokens.index(self.eos_id)
                    tokens = tokens[:eos_index]
                except ValueError:
                    pass
            byte_list = [t for t in tokens if 0 <= t < 256]
            return bytes(byte_list).decode("utf-8", errors="ignore")


    tokenizer = ByteLevelTokenizer(add_bos=True,
                                   add_eos=True)  # Using defaults that match initial_vocab_size=259
    print(f"Tokenizer initialized. BOS ID: {tokenizer.bos_id}, EOS ID: {tokenizer.eos_id}, "
          f"PAD ID: {tokenizer.pad_id}, Effective Vocab Size: {tokenizer.vocab_size}")


    # --- Data Processing Function ---
    def tokenize_and_process_examples(
            examples: Dict[str, List[str]],
            sequence_length: int,  # From exp_config
            text_column="text"
    ) -> Dict[str, List[torch.Tensor]]:
        processed_input_ids_list = []
        # Labels are same as input_ids for autoencoding tasks
        # processed_labels_list = [] # Not explicitly needed if labels are input_ids
        processed_kpm_list = []

        for text_content in examples[text_column]:
            if not isinstance(text_content, str):
                text_content = str(text_content) if text_content is not None else ""

            # Encode tokens (BOS/EOS added by tokenizer based on its settings)
            encoded_tokens = tokenizer.encode(text_content)
            current_length = len(encoded_tokens)
            final_tokens: torch.Tensor

            if current_length == 0 and sequence_length == 0:
                final_tokens = torch.tensor([], dtype=torch.int16)
            elif current_length == 0 and sequence_length > 0:
                final_tokens = torch.full((sequence_length,), tokenizer.pad_id,
                                          dtype=torch.int16)
            elif sequence_length == 0 and current_length > 0:  # Truncate to empty
                final_tokens = torch.tensor([], dtype=torch.int16)
            elif current_length > sequence_length:
                # Truncate and ensure EOS if tokenizer adds it and there's space
                if tokenizer.add_eos:
                    # Truncate to sequence_length - 1 to make space for EOS
                    final_tokens = encoded_tokens[:sequence_length - 1]
                    final_tokens = torch.cat(
                        (final_tokens, torch.tensor([tokenizer.eos_id], dtype=torch.int16))
                    )
                else:
                    final_tokens = encoded_tokens[:sequence_length]
            elif current_length < sequence_length:
                # Pad
                padding_needed = sequence_length - current_length
                padding_tensor = torch.full((padding_needed,), tokenizer.pad_id,
                                            dtype=torch.int16)
                final_tokens = torch.cat((encoded_tokens, padding_tensor))
            else:  # current_length == sequence_length
                final_tokens = encoded_tokens

            # Ensure the final tensor has the exact sequence_length (important safety check)
            if len(final_tokens) != sequence_length:
                if len(final_tokens) > sequence_length:
                    final_tokens = final_tokens[:sequence_length]
                else:  # len < sequence_length
                    padding_needed = sequence_length - len(final_tokens)
                    final_tokens = torch.cat((final_tokens,
                                              torch.full((padding_needed,), tokenizer.pad_id,
                                                         dtype=torch.int16)))

            # Generate key_padding_mask (True for padded tokens)
            key_padding_mask = (final_tokens == tokenizer.pad_id)

            processed_input_ids_list.append(final_tokens)
            # processed_labels_list.append(final_tokens.clone()) # Labels are same as input
            processed_kpm_list.append(key_padding_mask)

        return {
            "input_ids": processed_input_ids_list,
            "labels": processed_input_ids_list,  # For autoencoder, labels are inputs
            "key_padding_mask": processed_kpm_list
        }


    # --- Dataset Loading and Processing ---
    print(f"\nLoading dataset '{exp_config['dataset_name']}'...")
    raw_dataset = None
    try:
        raw_dataset = load_dataset(
            exp_config["dataset_name"],
            name=exp_config.get("dataset_config"),  # Use .get() for optional config name
            split=exp_config["dataset_train_split"]
        )
        print(f"Dataset loaded. Number of examples: {len(raw_dataset)}")
    except Exception as e:
        print(f"Error loading dataset '{exp_config['dataset_name']}': {e}")
        sys.exit()

    print(
        f"\nTokenizing and processing dataset with sequence length {exp_config['sequence_length']}...")
    tokenized_dataset = raw_dataset.map(
        tokenize_and_process_examples,
        batched=True,
        fn_kwargs={
            "sequence_length": exp_config["sequence_length"],
            "text_column": exp_config["text_column_name"]
        },
        remove_columns=raw_dataset.column_names,
        num_proc=N_CPU  # Use multiple processes for mapping
    )
    print("Tokenization complete.")

    print("\nSetting dataset format to PyTorch tensors...")
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "labels", "key_padding_mask"]
    )
    print("Dataset format set.")

    # --- DataLoader ---
    print(f"\nCreating PyTorch DataLoader with batch size {exp_config['batch_size']}...")
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=exp_config["batch_size"],
        # shuffle=True,
        num_workers=N_CPU,  # Use N_CPU for DataLoader workers
        pin_memory=True if DEVICE == "cuda" else False  # pin_memory if using CUDA
    )
    print(f"DataLoader created with {N_CPU} workers.")


    # --- Scheduler Setup --- # <<< ADDED SECTION
    # Calculate the total number of training (optimizer) steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / exp_config["gradient_accumulation_steps"])
    if exp_config.get("max_steps") is not None:
        exp_config["num_training_steps"] = int(exp_config["max_steps"])
    else:
        exp_config["num_training_steps"] = exp_config["num_epochs"] * num_update_steps_per_epoch

    print(
        f"Creating learning rate scheduler: {exp_config['scheduler_type']} with {exp_config['warmup_steps']} warmup steps and {exp_config['num_training_steps']} total steps."
    )
    lr_scheduler = get_scheduler(
        name=exp_config['scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=exp_config['warmup_steps'],
        num_training_steps=exp_config['num_training_steps'],
        scheduler_specific_kwargs=exp_config["scheduler_specific_kwargs"] if "scheduler_specific_kwargs" in exp_config else {}
    )

    # --- Track final hyperparameters with MLflow ---
    print("Tracking hyperparameters with MLflow...")
    mlflow.log_params(exp_config)
    print("Hyperparameters tracked.")

    # --- Training Loop ---
    print("\nStarting training loop...")
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Reset gradients before training

    # Track overall training time and total bytes processed
    training_start_time = time.time()
    total_bytes_processed = 0

    # --- Metric Accumulators ---
    def reset_accumulators(num_levels):
        accumulators = {
            "total_loss": 0.0,
            "vq_loss": 0.0,
            "avg_reconstruction_loss": 0.0,
            "reconstruction_loss_details": defaultdict(float),
            "avg_aux_lm_loss": 0.0,
            "aux_lm_loss_details": defaultdict(float),
            "compression_ratios": [0.0] * num_levels,
            "input_seq_lengths_compressors": [0.0] * num_levels,
            "output_seq_lengths_compressors": [0.0] * num_levels,
            "all_codebook_perplexities": [0.0] * num_levels,
            "all_smoothed_perplexities": [0.0] * num_levels,  # For smooth perplexity if used
            "non_padded_tokens": 0,  # For tokens/sec
            "count": 0, # To track number of batches accumulated
            "avg_top_code_lm_loss": 0.0,
            "top_code_lm_loss_details": defaultdict(float),
            "top_code_mse": 0.0,
            "top_code_vq_loss": 0.0,
        }
        return accumulators

    accumulators = reset_accumulators(exp_config["num_levels"])
    time_of_last_optimizer_step_event = time.time() # Initialize timer for tokens/sec
    total_minibatches_in_epoch = len(train_dataloader)

    for epoch in range(start_epoch, exp_config["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{exp_config['num_epochs']} ---") # MODIFIED: Epoch start print
        epoch_start_time = time.time()
        model.train() # Ensure model is in train mode at start of epoch

        for i, batch in enumerate(train_dataloader):
            tokens = batch["input_ids"].to(DEVICE)
            # KPM is already boolean, just move to device
            kpm = batch["key_padding_mask"].to(DEVICE) if exp_config[
                "propagate_key_padding_mask"] else None

            # HierarchicalAutoencoder.forward now handles loss calculation internally
            # and expects KPM for the initial tokens.
            output_dict = model(tokens, key_padding_mask=kpm)
            total_loss = output_dict['total_loss']

            # Normalize loss to account for accumulation
            # Each batch contributes 1/N to the total gradient, so scale loss by 1/N
            # This ensures that the magnitude of the gradients is similar to non-accumulated training
            loss_for_backward = total_loss / exp_config["gradient_accumulation_steps"]
            loss_for_backward.backward()

            # --- Accumulate Metrics ---
            accumulators["total_loss"] += total_loss.item()
            accumulators["vq_loss"] += output_dict['vq_loss'].item()
            accumulators["avg_reconstruction_loss"] += output_dict['avg_reconstruction_loss'].item()
            accumulators["count"] += 1

            if kpm is not None:
                accumulators["non_padded_tokens"] += (~kpm).sum().item() # Add count of non-padded tokens
            else:
                accumulators["non_padded_tokens"] += tokens.numel() # All tokens are considered valid


            for key, value in output_dict['reconstruction_loss_details'].items():
                accumulators["reconstruction_loss_details"][key] += value.item()

            if 'avg_aux_lm_loss' in output_dict and exp_config.get("aux_lm_loss_weight", 0.0) > 0:
                accumulators["avg_aux_lm_loss"] += output_dict['avg_aux_lm_loss'].item()
                for key, value in output_dict['aux_lm_loss_details'].items():
                    accumulators["aux_lm_loss_details"][key] += value.item()

            if 'avg_top_code_lm_loss' in output_dict and exp_config.get("top_lm_loss_weight", 0.0) > 0:
                accumulators["avg_top_code_lm_loss"] += output_dict['avg_top_code_lm_loss'].item()
                for key, value in output_dict['top_code_lm_loss_details'].items():
                    accumulators["top_code_lm_loss_details"][key] += value.item()
                    if key == 'top_code_mse':
                        accumulators['top_code_mse'] += value.item()
                    elif key == 'top_code_vq_loss':
                        accumulators['top_code_vq_loss'] += value.item()

            if 'compression_ratios' in output_dict:
                for level_idx, ratio in enumerate(output_dict['compression_ratios']):
                    accumulators["compression_ratios"][level_idx] += ratio # Assuming ratio is already a float or .item() called
                for level_idx, length in enumerate(output_dict['input_seq_lengths_compressors']):
                    accumulators["input_seq_lengths_compressors"][level_idx] += length
                for level_idx, length in enumerate(output_dict['output_seq_lengths_compressors']):
                    accumulators["output_seq_lengths_compressors"][level_idx] += length

            if 'all_codebook_perplexities' in output_dict:
                for level_idx, perplexity in enumerate(output_dict['all_codebook_perplexities']):
                    accumulators["all_codebook_perplexities"][level_idx] += perplexity.item()

            if 'all_smoothed_perplexities' in output_dict:
                for level_idx, smooth_perplexity in enumerate(output_dict['all_smoothed_perplexities']):
                    accumulators["all_smoothed_perplexities"][level_idx] += smooth_perplexity.item()

            # --- End Accumulate Metrics ---

            if (i + 1) % exp_config["gradient_accumulation_steps"] == 0:
                if exp_config.get("gradient_clip_norm"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), exp_config["gradient_clip_norm"])
                optimizer.step()
                lr_scheduler.step()  # Step the scheduler after optimizer step
                optimizer.zero_grad()  # Reset gradients after accumulation

                # Calculate metrics for the completed accumulation window
                current_time = time.time()
                duration_accumulation_window = current_time - time_of_last_optimizer_step_event

                tokens_processed_this_window = accumulators["non_padded_tokens"]
                tokens_per_second = tokens_processed_this_window / duration_accumulation_window

                # Update global byte count and compute ETAs
                total_bytes_processed += tokens_processed_this_window

                total_progress = (global_step + 1) / exp_config["num_training_steps"]
                total_elapsed = current_time - training_start_time
                total_eta_sec = (total_elapsed / total_progress - total_elapsed) if total_progress > 0 else 0

                steps_accumulated = accumulators["count"]

                if steps_accumulated > 0:  # Ensure there's something to log
                    # --- Console Logging (replaces tqdm postfix and description updates) ---
                    console_log_parts = [
                        f"Epoch {epoch + 1}/{exp_config['num_epochs']}",
                        f"OptStep {global_step}",
                        f"MB {i + 1}/{total_minibatches_in_epoch}",  # Minibatch progress
                        f"Loss {accumulators['total_loss'] / steps_accumulated:.4f}",
                        f"VQ {accumulators['vq_loss'] / steps_accumulated:.4f}",
                        f"Reco {accumulators['avg_reconstruction_loss'] / steps_accumulated:.4f}",
                        f"Tok/s {short_num(tokens_per_second)}",
                        f"Bytes {short_num(total_bytes_processed)}",
                        f"ETAt {format_duration(total_eta_sec)}"
                    ]

                    # Mimicking the original postfix_dict logic for specific items
                    if 'avg_top_code_lm_loss' in output_dict and exp_config.get("top_lm_loss_weight", 0.0) > 0:
                        # TQDM showed the last batch's value for this
                        console_log_parts.append(f"TopLM {output_dict['avg_top_code_lm_loss'].item():.4f}")
                        if 'top_code_lm_loss_details' in output_dict:
                            tcld = output_dict['top_code_lm_loss_details']
                            if 'top_code_mse' in tcld:
                                console_log_parts.append(f"TopMSE {tcld['top_code_mse'].item():.4f}")
                            if 'top_code_vq_loss' in tcld:
                                console_log_parts.append(f"TopVQ {tcld['top_code_vq_loss'].item():.4f}")

                    if 'compression_ratios' in accumulators and len(accumulators["compression_ratios"]) == exp_config[
                        "num_levels"]:
                        # TQDM showed accumulated average for ratios
                        ratios_str = ", ".join([f"{r / steps_accumulated:.2f}" for r in accumulators['compression_ratios']])
                        console_log_parts.append(f"Ratios [{ratios_str}]")

                    if 'all_smoothed_perplexities' in accumulators and len(accumulators["all_smoothed_perplexities"]) == \
                            exp_config["num_levels"]:
                        # TQDM showed accumulated average for smooth perplexities
                        ppl_str = ", ".join(
                            [f"{p / steps_accumulated:.4f}" for p in accumulators['all_smoothed_perplexities']])
                        console_log_parts.append(f"SmoothPPL [{ppl_str}]")

                    if exp_config.get("aux_lm_loss_weight", 0.0) > 0 and "avg_aux_lm_loss" in accumulators:
                        # TQDM showed accumulated average for auxLM
                        console_log_parts.append(f"AuxLM {accumulators['avg_aux_lm_loss'] / steps_accumulated:.4f}")

                    print(" | ".join(console_log_parts), flush=True)


                    # --- Logging to MLflow (occurs every optimizer step) ---
                    if global_step % exp_config["log_interval"] == 0 and accumulators["count"] > 0:

                        mlflow.log_metric('loss/total_avg_accum', accumulators["total_loss"] / steps_accumulated, step=global_step)
                        mlflow.log_metric('loss/vq_avg_accum', accumulators["vq_loss"] / steps_accumulated, step=global_step)
                        mlflow.log_metric('loss/reconstruction_avg_accum', accumulators["avg_reconstruction_loss"] / steps_accumulated, step=global_step)
                        mlflow.log_metric('performance/tokens_per_sec', tokens_per_second, step=global_step)
                        mlflow.log_metric('loss/top_code_lm_avg_accum', accumulators["avg_top_code_lm_loss"] / steps_accumulated, step=global_step)
                        mlflow.log_metric('loss/top_code_mse_avg_accum', accumulators['top_code_mse'] / steps_accumulated, step=global_step)
                        mlflow.log_metric('loss/top_code_vq_avg_accum', accumulators['top_code_vq_loss'] / steps_accumulated, step=global_step)
                        for key, value in accumulators["top_code_lm_loss_details"].items():
                            mlflow.log_metric(f'loss_detail_avg_accum/{key}', value / steps_accumulated, step=global_step)

                        for key, value in accumulators["reconstruction_loss_details"].items():
                            mlflow.log_metric(f'loss_detail_avg_accum/{key}', value / steps_accumulated, step=global_step)

                        if exp_config.get("aux_lm_loss_weight", 0.0) > 0:
                            mlflow.log_metric('loss/aux_lm_avg_accum', accumulators["avg_aux_lm_loss"] / steps_accumulated, step=global_step)
                            for key, value in accumulators["aux_lm_loss_details"].items():
                                mlflow.log_metric(f'loss_detail_avg_accum/{key}', value / steps_accumulated, step=global_step)

                        if 'compression_ratios' in output_dict:  # Check if key exists in output_dict to ensure lists are correct length
                            for level_idx in range(exp_config["num_levels"]):
                                mlflow.log_metric(f'compression_avg/ratio_L{level_idx}',
                                                  accumulators["compression_ratios"][level_idx] / steps_accumulated,
                                                  step=global_step)
                                mlflow.log_metric(f'compression_avg/input_len_L{level_idx}',
                                                  accumulators["input_seq_lengths_compressors"][level_idx] / steps_accumulated,
                                                  step=global_step)
                                mlflow.log_metric(f'compression_avg/output_len_L{level_idx}',
                                                  accumulators["output_seq_lengths_compressors"][level_idx] / steps_accumulated,
                                                  step=global_step)

                        mlflow.log_metric('learning_rate', optimizer.param_groups[0]['lr'], step=global_step)

                        if 'all_codebook_perplexities' in output_dict:
                            for level_idx in range(exp_config["num_levels"]):
                                mlflow.log_metric(f'vq_metrics_avg/perplexity_L{level_idx}',
                                                  accumulators["all_codebook_perplexities"][level_idx] / steps_accumulated,
                                                  step=global_step)
                                codebook_size_L_i = exp_config["compressor_level_configs"][level_idx][
                                    "codebook_size"]
                                mlflow.log_metric(f'vq_metrics/codebook_size_L{level_idx}', codebook_size_L_i, step=global_step)

                        if 'all_smoothed_perplexities' in output_dict:
                            for level_idx in range(exp_config["num_levels"]):
                                mlflow.log_metric(f'vq_metrics_avg/smooth_perplexity_L{level_idx}',
                                                  accumulators["all_smoothed_perplexities"][level_idx] / steps_accumulated,
                                                  step=global_step)

                # Reset accumulators for the next window
                accumulators = reset_accumulators(exp_config["num_levels"])
                # Update the timer to mark the start of the next accumulation window's measurement period
                time_of_last_optimizer_step_event = time.time()

                # --- End Logging ---

                # --- Generation During Training ---
                if global_step > 0 and global_step % exp_config["generation_interval"] == 0:
                    print(f"\nStep {global_step}: Generating sample...")
                    model.eval()  # Switch to evaluation mode for generation
                    with torch.no_grad():
                        sample_text = exp_config["sample_prompt_for_generation"]
                        # Encode the prompt without padding to a fixed length
                        # BOS/EOS will be added by tokenizer.encode if configured
                        input_gen_tokens = tokenizer.encode(sample_text).unsqueeze(0).to(DEVICE)

                        # For unpadded, variable-length input to generate_bytes, KPM is all False
                        input_gen_kpm = None
                        if exp_config["propagate_key_padding_mask"]:
                            input_gen_kpm = torch.zeros_like(input_gen_tokens, dtype=torch.bool).to(DEVICE)

                        reconstructed_tokens = model.generate_bytes(
                            tokens=input_gen_tokens,
                            key_padding_mask=input_gen_kpm,  # Pass KPM if propagation is on
                            max_len_override=exp_config["generation_max_len_override"]
                        )
                        reconstructed_text = tokenizer.decode(reconstructed_tokens.squeeze(0).cpu(),
                                                              cut_at_eos=True)

                        print(f"--- Sample Generation at Step {global_step} ---")
                        print(f"Original Prompt:\n{sample_text}")
                        print(f"Reconstructed Text:\n{reconstructed_text}")
                        print("------------------------------------------")

                        # Log to MLflow
                        mlflow_text_log = f"Original:\n{sample_text}\n\nReconstructed:\n{reconstructed_text}"
                        mlflow.log_text(mlflow_text_log, f"sample_generation_step_{global_step}.txt")

                    model.train()  # Switch back to training mode
                if global_step > 0 and global_step % exp_config["checkpoint_interval"] == 0:
                    save_checkpoint(model, optimizer, epoch, global_step, exp_config["checkpoint_dir"])
                global_step += 1
                if exp_config.get("max_steps") is not None and global_step >= exp_config["max_steps"]:
                    print(f"Reached max_steps {exp_config['max_steps']}. Stopping training.")
                    break
        epoch_duration = time.time() - epoch_start_time
        print(f"--- Epoch {epoch + 1} finished. Duration: {epoch_duration:.2f}s ---") # MODIFIED: Epoch end print
        save_checkpoint(model, optimizer, epoch, global_step, exp_config["checkpoint_dir"])

        if exp_config.get("max_steps") is not None and global_step >= exp_config["max_steps"]:
            break

    print("Training finished.")
    mlflow.end_run()

    # Example of saving the model (optional)
    # final_model_path = f"{exp_config.get('run_name', 'model')}_final.pth"
    # torch.save(model.state_dict(), final_model_path)
    # print(f"Final model state dict saved to {final_model_path}")
