import os
import math
import sys
import time
from collections import defaultdict, deque
import argparse
import importlib.util
from dataclasses import asdict

import torch
from datasets import load_dataset, Dataset  # Import Dataset for the dummy data
from torch import optim
from torch.utils.data import DataLoader
from typing import List, Dict, Any
from transformers.optimization import get_scheduler

import mlflow  # Logging with MLflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric

# Assuming HierarchicalAutoencoder is in abstractinator.py and has KPM updates
from components import HierarchicalAutoencoder
from components.sliding_window_attention import _cached_cross_window_mask as _cached_cross_window_mask_cpu
from components.expander import _cached_causal_mask as _cached_causal_mask_cpu
from components.utils import short_num, format_duration
from components.tokenizer import ByteLevelTokenizer
from data_utils import tokenize_and_process_examples
from configs.base_config import (
    DEVICE as DEFAULT_DEVICE,
    N_CPU as DEFAULT_N_CPU,
    ExpConfig,
)


def save_base_components(model: HierarchicalAutoencoder, path: str) -> None:
    """Save only the compressor and expander weights."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "compressors": model.compressors.state_dict(),
        "expanders": model.expanders.state_dict(),
    }, path)
    print(f"Base components saved to {path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script for HierarchicalAutoencoder")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=("Path to the configuration Python file. If omitted, the config "
              "saved in a checkpoint will be used when resuming"),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--load_base_from",
        type=str,
        default=None,
        help="Load pretrained compressor/expander checkpoint",
    )
    args = parser.parse_args()

    def load_config(path: str):
        spec = importlib.util.spec_from_file_location("config_module", path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module

    if args.config:
        config = load_config(args.config)
        DEVICE = config.DEVICE
        N_CPU = config.N_CPU
        exp_config: ExpConfig = config.exp_config
    else:
        if not args.resume_from_checkpoint:
            raise ValueError("--config or --resume_from_checkpoint must be provided")
        ckpt = torch.load(args.resume_from_checkpoint, map_location="cpu")
        if "exp_config" not in ckpt:
            raise ValueError("Checkpoint missing exp_config; provide --config")
        cfg = ckpt["exp_config"]
        if isinstance(cfg, dict):
            exp_config = ExpConfig(**cfg)
        else:
            exp_config = cfg
        DEVICE = DEFAULT_DEVICE
        N_CPU = DEFAULT_N_CPU

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
    print(
        f"Initializing MLflow run: {getattr(exp_config, 'run_name', 'DefaultRun')}"
    )
    if getattr(exp_config, "project_name", None):
        mlflow.set_tracking_uri(f"file:./mlruns/{exp_config.project_name}")
        print(
            f"MLflow tracking URI set to: file:./mlruns/{exp_config.project_name}"
        )
    mlflow.set_experiment(getattr(exp_config, "project_name", "DefaultExperiment"))
    mlflow_run = mlflow.start_run(
        run_name=getattr(exp_config, "run_name", "DefaultRun")
    )
    mlflow_client = MlflowClient()
    mlflow_run_id = mlflow_run.info.run_id
    mlflow_metric_buffer = []

    # --- Model, Optimizer ---
    print("Initializing HierarchicalAutoencoder model...")
    model = HierarchicalAutoencoder(
        num_levels=exp_config.num_levels,
        compressor_level_configs=[asdict(c) for c in exp_config.compressor_level_configs],
        initial_vocab_size=exp_config.initial_vocab_size,
        expander_dim_scale=exp_config.expander_dim_scale,
        expander_num_enc_layers=exp_config.expander_num_enc_layers,
        expander_num_dec_layers=exp_config.expander_num_dec_layers,
        expander_heads_scale=exp_config.expander_heads_scale,
        expander_eos_id=exp_config.expander_eos_id,
        expander_max_len=exp_config.expander_max_len,  # Pass expander_max_len
        use_decoder_only_expander=exp_config.use_decoder_only_expander,
        propagate_key_padding_mask=exp_config.propagate_key_padding_mask,
        aux_lm_loss_weight=exp_config.aux_lm_loss_weight,
        top_transformer_config=(
            asdict(exp_config.top_transformer_config)
            if exp_config.top_transformer_config
            else None
        ),
        top_lm_loss_weight=exp_config.top_lm_loss_weight,
        use_continuous_expander_inputs=exp_config.use_continuous_expander_inputs,
    ).to(DEVICE)

    # model = torch.compile(model, mode="default", dynamic=True)

    if args.load_base_from:
        ckpt = torch.load(args.load_base_from, map_location=DEVICE)
        model.compressors.load_state_dict(ckpt.get("compressors", {}), strict=False)
        model.expanders.load_state_dict(ckpt.get("expanders", {}), strict=False)
        model.compressors.requires_grad_(False)
        model.expanders.requires_grad_(False)
        model.compressors.eval()
        model.expanders.eval()
        print(f"Loaded base components from {args.load_base_from}")

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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=exp_config.learning_rate)
    print(
        f"Optimizer AdamW initialized with learning rate: {exp_config.learning_rate:.0e}"
        f" and {short_num(sum(p.numel() for p in trainable_params))} trainable params"
    )

    # --- Checkpoint Utilities ---
    def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "exp_config": exp_config,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    start_epoch = 0
    global_step = 0
    resume_path = args.resume_from_checkpoint or exp_config.resume_from_checkpoint
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
    tokenizer = ByteLevelTokenizer(
        add_bos=True,
        add_eos=True,
        expected_vocab_size=exp_config.initial_vocab_size,
    )
    print(
        f"Tokenizer initialized. BOS ID: {tokenizer.bos_id}, EOS ID: {tokenizer.eos_id}, "
        f"PAD ID: {tokenizer.pad_id}, Effective Vocab Size: {tokenizer.vocab_size}"
    )

    # --- Dataset Loading and Processing ---
    print(f"\nLoading dataset '{exp_config.dataset_name}'...")
    raw_dataset = None
    try:
        raw_dataset = load_dataset(
            exp_config.dataset_name,
            name=exp_config.dataset_config,
            split=exp_config.dataset_train_split,
        )
        print(f"Dataset loaded. Number of examples: {len(raw_dataset)}")
    except Exception as e:
        print(f"Error loading dataset '{exp_config.dataset_name}': {e}")
        sys.exit()

    print(
        f"\nTokenizing and processing dataset with sequence length {exp_config.sequence_length}..."
    )
    tokenized_dataset = raw_dataset.map(
        tokenize_and_process_examples,
        batched=True,
        fn_kwargs={
            "sequence_length": exp_config.sequence_length,
            "tokenizer": tokenizer,
            "text_column": exp_config.text_column_name,
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
    print(f"\nCreating PyTorch DataLoader with batch size {exp_config.batch_size}...")
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=exp_config.batch_size,
        # shuffle=True,
        num_workers=N_CPU,  # Use N_CPU for DataLoader workers
        pin_memory=True if DEVICE == "cuda" else False  # pin_memory if using CUDA
    )
    print(f"DataLoader created with {N_CPU} workers.")


    # --- Scheduler Setup --- # <<< ADDED SECTION
    # Calculate the total number of training (optimizer) steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / exp_config.gradient_accumulation_steps
    )
    if exp_config.max_steps is not None:
        exp_config.num_training_steps = int(exp_config.max_steps)
    else:
        exp_config.num_training_steps = exp_config.num_epochs * num_update_steps_per_epoch

    print(
        f"Creating learning rate scheduler: {exp_config.scheduler_type} with {exp_config.warmup_steps} warmup steps and {exp_config.num_training_steps} total steps."
    )
    lr_scheduler = get_scheduler(
        name=exp_config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=exp_config.warmup_steps,
        num_training_steps=exp_config.num_training_steps,
        scheduler_specific_kwargs=exp_config.scheduler_specific_kwargs if hasattr(exp_config, "scheduler_specific_kwargs") else {}
    )

    # --- Track final hyperparameters with MLflow ---
    print("Tracking hyperparameters with MLflow...")
    mlflow.log_params(exp_config.as_dict())
    print("Hyperparameters tracked.")

    # --- Training Loop ---
    print("\nStarting training loop...")
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Reset gradients before training

    # Track overall training time and total bytes processed
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    training_start_time = time.time()
    total_bytes_processed = 0
    total_patches_processed_per_level = [0.0] * exp_config.num_levels

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

    accumulators = reset_accumulators(exp_config.num_levels)
    tok_s_deque = deque(maxlen=10)  # For moving average of tok/s
    time_of_last_optimizer_step_event = training_start_time # Initialize timer for tokens/sec
    total_minibatches_in_epoch = len(train_dataloader)

    for epoch in range(start_epoch, exp_config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{exp_config.num_epochs} ---")
        # Clear cached attention masks to avoid GPU memory buildup
        _cached_cross_window_mask_cpu.cache_clear()
        _cached_causal_mask_cpu.cache_clear()
        epoch_start_time = time.time()
        model.train() # Ensure model is in train mode at start of epoch

        for i, batch in enumerate(train_dataloader):
            tokens = batch["input_ids"].to(DEVICE)
            # KPM is already boolean, just move to device
            key_padding_mask = batch["key_padding_mask"].to(DEVICE)
            model_kpm = (
                key_padding_mask if exp_config.propagate_key_padding_mask else None
            )

            # HierarchicalAutoencoder.forward now handles loss calculation internally
            # and expects KPM for the initial tokens.
            output_dict = model(tokens, key_padding_mask=model_kpm)
            total_loss = output_dict['total_loss']

            # Normalize loss to account for accumulation
            # Each batch contributes 1/N to the total gradient, so scale loss by 1/N
            # This ensures that the magnitude of the gradients is similar to non-accumulated training
            loss_for_backward = total_loss / exp_config.gradient_accumulation_steps
            loss_for_backward.backward()

            # --- Accumulate Metrics ---
            accumulators["total_loss"] += total_loss.item()
            accumulators["vq_loss"] += output_dict['vq_loss'].item()
            accumulators["avg_reconstruction_loss"] += output_dict['avg_reconstruction_loss'].item()
            accumulators["count"] += 1

            # For tokens/sec calculation, always use the actual number of non-padded tokens.
            accumulators["non_padded_tokens"] += (~key_padding_mask).sum().item()


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
                    # ratio is already a float or tensor scalar
                    accumulators["compression_ratios"][level_idx] += ratio

                batch_sz = tokens.size(0)
                for level_idx, length in enumerate(output_dict['input_seq_lengths_compressors']):
                    # lengths from the model are per-sample averages
                    accumulators["input_seq_lengths_compressors"][level_idx] += length * batch_sz
                for level_idx, length in enumerate(output_dict['output_seq_lengths_compressors']):
                    accumulators["output_seq_lengths_compressors"][level_idx] += length * batch_sz

            if 'all_codebook_perplexities' in output_dict:
                for level_idx, perplexity in enumerate(output_dict['all_codebook_perplexities']):
                    accumulators["all_codebook_perplexities"][level_idx] += perplexity.item()

            if 'all_smoothed_perplexities' in output_dict:
                for level_idx, smooth_perplexity in enumerate(output_dict['all_smoothed_perplexities']):
                    accumulators["all_smoothed_perplexities"][level_idx] += smooth_perplexity.item()

            # --- End Accumulate Metrics ---

            if (i + 1) % exp_config.gradient_accumulation_steps == 0:
                if exp_config.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), exp_config.gradient_clip_norm)
                optimizer.step()
                lr_scheduler.step()  # Step the scheduler after optimizer step
                optimizer.zero_grad()  # Reset gradients after accumulation

                # Calculate metrics for the completed accumulation window
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                current_time = time.time()
                duration_accumulation_window = current_time - time_of_last_optimizer_step_event

                tokens_processed_this_window = accumulators["non_padded_tokens"]
                tokens_per_second = tokens_processed_this_window / duration_accumulation_window
                tok_s_deque.append(tokens_per_second)
                avg_tok_s = sum(tok_s_deque) / len(tok_s_deque)

                # Use output sequence lengths to measure patches processed
                # The top LM consumes the outputs of the compressors, so its
                # throughput should be based on these lengths.
                patches_processed_this_window = accumulators["output_seq_lengths_compressors"]
                patches_per_second = [
                    p / duration_accumulation_window for p in patches_processed_this_window
                ]
                for lvl, cnt in enumerate(patches_processed_this_window):
                    total_patches_processed_per_level[lvl] += cnt

                # Update global byte count and compute ETAs
                total_bytes_processed += tokens_processed_this_window
                total_progress = (global_step + 1) / exp_config.num_training_steps
                total_elapsed = current_time - training_start_time
                total_eta_sec = (total_elapsed / total_progress - total_elapsed) if total_progress > 0 else 0

                steps_accumulated = accumulators["count"]

                if steps_accumulated > 0:  # Ensure there's something to log
                    # --- Console Logging (replaces tqdm postfix and description updates) ---
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    console_log_parts = [
                        f"{timestamp}",
                        f"Epoch {epoch + 1}/{exp_config.num_epochs}",
                        f"OptStep {global_step}",
                        f"MB {i + 1}/{total_minibatches_in_epoch}",  # Minibatch progress
                        f"Loss {accumulators['total_loss'] / steps_accumulated:.4f}",
                        f"Reco {accumulators['avg_reconstruction_loss'] / steps_accumulated:.4f}",
                        f"VQ {accumulators['vq_loss'] / steps_accumulated:.4f}"
                    ]

                    if exp_config.get("aux_lm_loss_weight", 0.0) > 0 and "avg_aux_lm_loss" in accumulators:
                        console_log_parts.append(
                            f"AuxLM {accumulators['avg_aux_lm_loss'] / steps_accumulated:.4f}"
                        )

                    if 'avg_top_code_lm_loss' in output_dict and exp_config.get("top_lm_loss_weight", 0.0) > 0:
                        console_log_parts.append(
                            f"TopLM {output_dict['avg_top_code_lm_loss'].item():.4f}"
                        )
                        if 'top_code_lm_loss_details' in output_dict:
                            tcld = output_dict['top_code_lm_loss_details']
                            if 'top_code_mse' in tcld:
                                console_log_parts.append(f"TopMSE {tcld['top_code_mse'].item():.4f}")
                            if 'top_code_vq_loss' in tcld:
                                console_log_parts.append(f"TopVQ {tcld['top_code_vq_loss'].item():.4f}")

                    console_log_parts.extend([
                        f"Tok/s {short_num(avg_tok_s)}",
                        f"Bytes {short_num(total_bytes_processed)}",
                        f"ETAt {format_duration(total_eta_sec)}",
                    ])

                    patch_log_parts = []
                    for lvl in range(1, exp_config.num_levels - 1):
                        patch_log_parts.append(
                            f"L{lvl} {short_num(patches_per_second[lvl])}/s {short_num(total_patches_processed_per_level[lvl])}"
                        )
                    patch_log_parts.append(
                        f"Top {short_num(patches_per_second[-1])}/s {short_num(total_patches_processed_per_level[-1])}"
                    )
                    if patch_log_parts:
                        console_log_parts.append("Patches " + ", ".join(patch_log_parts))

                    # Additional metrics appended after the loss components
                    if 'compression_ratios' in accumulators and len(accumulators["compression_ratios"]) == exp_config.num_levels:
                        # TQDM showed accumulated average for ratios
                        ratios_str = ", ".join([f"{r / steps_accumulated:.2f}" for r in accumulators['compression_ratios']])
                        console_log_parts.append(f"Ratios [{ratios_str}]")

                    if 'all_smoothed_perplexities' in accumulators and len(accumulators["all_smoothed_perplexities"]) == exp_config.num_levels:
                        # TQDM showed accumulated average for smooth perplexities
                        ppl_str = ", ".join(
                            [f"{p / steps_accumulated:.4f}" for p in accumulators['all_smoothed_perplexities']])
                        console_log_parts.append(f"SmoothPPL [{ppl_str}]")

                    print(" | ".join(console_log_parts), flush=True)


                    # --- Logging to MLflow (occurs every optimizer step) ---
                    if global_step % exp_config.log_interval == 0 and accumulators["count"] > 0:
                        metrics_dict = {
                            'loss/total_avg_accum': accumulators["total_loss"] / steps_accumulated,
                            'loss/vq_avg_accum': accumulators["vq_loss"] / steps_accumulated,
                            'loss/reconstruction_avg_accum': accumulators["avg_reconstruction_loss"] / steps_accumulated,
                            'performance/tokens_per_sec': tokens_per_second,
                            'loss/top_code_lm_avg_accum': accumulators["avg_top_code_lm_loss"] / steps_accumulated,
                            'loss/top_code_mse_avg_accum': accumulators['top_code_mse'] / steps_accumulated,
                            'loss/top_code_vq_avg_accum': accumulators['top_code_vq_loss'] / steps_accumulated,
                            'learning_rate': optimizer.param_groups[0]['lr'],
                        }
                        for key, value in accumulators["top_code_lm_loss_details"].items():
                            metrics_dict[f'loss_detail_avg_accum/{key}'] = value / steps_accumulated

                        for key, value in accumulators["reconstruction_loss_details"].items():
                            metrics_dict[f'loss_detail_avg_accum/{key}'] = value / steps_accumulated

                        if exp_config.get("aux_lm_loss_weight", 0.0) > 0:
                            metrics_dict['loss/aux_lm_avg_accum'] = accumulators["avg_aux_lm_loss"] / steps_accumulated
                            for key, value in accumulators["aux_lm_loss_details"].items():
                                metrics_dict[f'loss_detail_avg_accum/{key}'] = value / steps_accumulated

                        if 'compression_ratios' in output_dict:  # Check if key exists in output_dict to ensure lists are correct length
                            for level_idx in range(exp_config.num_levels):
                                metrics_dict[f'compression_avg/ratio_L{level_idx}'] = (
                                    accumulators["compression_ratios"][level_idx] / steps_accumulated
                                )
                                metrics_dict[f'compression_avg/input_len_L{level_idx}'] = (
                                    accumulators["input_seq_lengths_compressors"][level_idx] / steps_accumulated
                                )
                                metrics_dict[f'compression_avg/output_len_L{level_idx}'] = (
                                    accumulators["output_seq_lengths_compressors"][level_idx] / steps_accumulated
                                )

                        if 'all_codebook_perplexities' in output_dict:
                            for level_idx in range(exp_config.num_levels):
                                metrics_dict[f'vq_metrics_avg/perplexity_L{level_idx}'] = (
                                    accumulators["all_codebook_perplexities"][level_idx] / steps_accumulated
                                )
                                codebook_size_L_i = exp_config.compressor_level_configs[level_idx].codebook_size
                                metrics_dict[f'vq_metrics/codebook_size_L{level_idx}'] = codebook_size_L_i

                        if 'all_smoothed_perplexities' in output_dict:
                            for level_idx in range(exp_config.num_levels):
                                metrics_dict[f'vq_metrics_avg/smooth_perplexity_L{level_idx}'] = (
                                    accumulators["all_smoothed_perplexities"][level_idx] / steps_accumulated
                                )

                        mlflow_metric_buffer.append((global_step, metrics_dict))
                        if len(mlflow_metric_buffer) >= exp_config.mlflow_batch_interval:
                            timestamp_ms = int(time.time() * 1000)
                            metrics_entities = []
                            for step_id, mdict in mlflow_metric_buffer:
                                metrics_entities.extend([
                                    Metric(key=k, value=v, timestamp=timestamp_ms, step=step_id)
                                    for k, v in mdict.items()
                                ])
                            mlflow_client.log_batch(mlflow_run_id, metrics=metrics_entities)
                            mlflow_metric_buffer = []

                # Reset accumulators for the next window
                accumulators = reset_accumulators(exp_config.num_levels)
                # Update the timer to mark the start of the next accumulation window's measurement period
                time_of_last_optimizer_step_event = current_time

                # --- End Logging ---

                # --- Generation During Training ---
                if global_step > 0 and global_step % exp_config.generation_interval == 0:
                    print(f"\nStep {global_step}: Generating sample...")
                    model.eval()  # Switch to evaluation mode for generation
                    with torch.no_grad():
                        sample_text = exp_config.sample_prompt_for_generation
                        # Encode the prompt without padding to a fixed length
                        # BOS/EOS will be added by tokenizer.encode if configured
                        input_gen_tokens = tokenizer.encode(sample_text).unsqueeze(0).to(DEVICE)

                        # For unpadded, variable-length input to generate_bytes, KPM is all False
                        input_gen_kpm = None
                        if exp_config.propagate_key_padding_mask:
                            input_gen_kpm = torch.zeros_like(input_gen_tokens, dtype=torch.bool).to(DEVICE)

                        reconstructed_tokens = model.generate_bytes(
                            tokens=input_gen_tokens,
                            key_padding_mask=input_gen_kpm,  # Pass KPM if propagation is on
                            max_len_override=exp_config.generation_max_len_override
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
                if global_step > 0 and global_step % exp_config.checkpoint_interval == 0:
                    save_checkpoint(model, optimizer, epoch, global_step, exp_config.checkpoint_dir)
                global_step += 1
                if exp_config.max_steps is not None and global_step >= exp_config.max_steps:
                    print(f"Reached max_steps {exp_config.max_steps}. Stopping training.")
                    break
        epoch_duration = time.time() - epoch_start_time
        print(f"--- Epoch {epoch + 1} finished. Duration: {epoch_duration:.2f}s ---") # MODIFIED: Epoch end print
        save_checkpoint(model, optimizer, epoch, global_step, exp_config.checkpoint_dir)

        if exp_config.max_steps is not None and global_step >= exp_config.max_steps:
            break

    print("Training finished.")
    if exp_config.save_base_components_path:
        save_base_components(model, exp_config.save_base_components_path)
    if mlflow_metric_buffer:
        timestamp_ms = int(time.time() * 1000)
        metrics_entities = []
        for step_id, mdict in mlflow_metric_buffer:
            metrics_entities.extend([
                Metric(key=k, value=v, timestamp=timestamp_ms, step=step_id)
                for k, v in mdict.items()
            ])
        mlflow_client.log_batch(mlflow_run_id, metrics=metrics_entities)
    mlflow.end_run()

    # Example of saving the model (optional)
    # final_model_path = f"{exp_config.get('run_name', 'model')}_final.pth"
    # torch.save(model.state_dict(), final_model_path)
    # print(f"Final model state dict saved to {final_model_path}")
