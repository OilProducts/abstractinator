import os
import math
import sys
import time
from collections import defaultdict, deque
import argparse
import importlib.util
from dataclasses import asdict
import logging

# Name for loggers created in this module so logs don't show '__main__'
LOGGER_NAME = "abstractinator.train"

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
from components.checkpoint_utils import save_base_components, load_base_components
from data_utils import tokenize_and_process_examples

from configs.base_config import (
    DEVICE as DEFAULT_DEVICE,
    N_CPU as DEFAULT_N_CPU,
    ExpConfig,
)

torch.backends.cuda.enable_flash_sdp(True)   # FlashAttn‑2*
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


def parse_args() -> argparse.Namespace:
    """Return command line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Training script for HierarchicalAutoencoder"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the configuration Python file. If omitted, the config saved in a checkpoint "
            "will be used when resuming"
        ),
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
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Logging level (debug, info, warning, error, critical)",
    )
    return parser.parse_args()


def load_experiment_config(args: argparse.Namespace) -> tuple[str, int, ExpConfig]:
    """Load the experiment configuration from a file or checkpoint."""

    def _load_config(path: str):
        spec = importlib.util.spec_from_file_location("config_module", path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module

    if args.config:
        config = _load_config(args.config)
        device = config.DEVICE
        n_cpu = config.N_CPU
        exp_cfg: ExpConfig = config.exp_config
    else:
        if not args.resume_from_checkpoint:
            raise ValueError("--config or --resume_from_checkpoint must be provided")
        ckpt = torch.load(args.resume_from_checkpoint, map_location="cpu")
        if "exp_config" not in ckpt:
            raise ValueError("Checkpoint missing exp_config; provide --config")
        cfg = ckpt["exp_config"]
        exp_cfg = ExpConfig(**cfg) if isinstance(cfg, dict) else cfg
        device = DEFAULT_DEVICE
        n_cpu = DEFAULT_N_CPU

    return device, n_cpu, exp_cfg


def initialize_model(
    args: argparse.Namespace, exp_config: ExpConfig, device: str
) -> tuple[HierarchicalAutoencoder, optim.Optimizer, int, int]:
    """Instantiate the model, optimizer and restore any checkpoints."""

    logger = logging.getLogger(LOGGER_NAME)

    model = HierarchicalAutoencoder(
        num_levels=exp_config.num_levels,
        compressor_level_configs=[asdict(c) for c in exp_config.compressor_level_configs],
        initial_vocab_size=exp_config.initial_vocab_size,
        expander_dim_scale=exp_config.expander.dim_scale,
        expander_num_enc_layers=exp_config.expander.num_enc_layers,
        expander_num_dec_layers=exp_config.expander.num_dec_layers,
        expander_heads_scale=exp_config.expander.heads_scale,
        expander_eos_id=exp_config.expander.eos_id,
        expander_max_len=exp_config.expander.max_len,
        use_decoder_only_expander=exp_config.expander.use_decoder_only,
        propagate_key_padding_mask=exp_config.propagate_key_padding_mask,
        aux_lm_loss_weight=exp_config.aux_lm_loss_weight,
        top_transformer_config=(
            asdict(exp_config.top_transformer_config)
            if exp_config.top_transformer_config
            else None
        ),
        top_lm_loss_weight=exp_config.top_lm_loss_weight,
        use_continuous_expander_inputs=exp_config.expander.use_continuous_inputs,
        top_lm_mse_weight=exp_config.top_lm_mse_weight,
        top_lm_ce_weight=exp_config.top_lm_ce_weight,
    ).to(device)

    if args.load_base_from:
        load_base_components(
            model,
            args.load_base_from,
            freeze=True,
            map_location=device,
        )

    def _count_params(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    num_params = _count_params(model)
    logger.info(
        "Model initialized on %s with %s trainable parameters.",
        device,
        short_num(num_params),
    )

    compressor_params = _count_params(model.compressors)
    expander_params = _count_params(model.expanders)
    logger.info("  Compressors: %s parameters", short_num(compressor_params))
    logger.info("  Expanders: %s parameters", short_num(expander_params))
    if getattr(model, "code_sequence_transformer", None) is not None:
        cst_params = _count_params(model.code_sequence_transformer)
        logger.info("  CodeSequenceTransformer: %s parameters", short_num(cst_params))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=exp_config.learning_rate)

    start_epoch = 0
    global_step = 0
    resume_path = args.resume_from_checkpoint or exp_config.resume_from_checkpoint
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", {}))
        optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        logger.info(
            "Resumed from checkpoint '%s' at step %s, epoch %s",
            resume_path,
            global_step,
            start_epoch,
        )
    elif resume_path:
        logger.warning("Checkpoint path '%s' not found. Starting from scratch.", resume_path)

    return model, optimizer, start_epoch, global_step


def save_checkpoint(
    model: HierarchicalAutoencoder,
    optimizer: optim.Optimizer,
    epoch: int,
    step: int,
    checkpoint_dir: str,
    exp_config: ExpConfig,
    logger: logging.Logger | None = None,
) -> None:
    """Persist training state to ``checkpoint_dir``."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "exp_config": exp_config,
        },
        checkpoint_path,
    )
    if logger:
        logger.info("Checkpoint saved to %s", checkpoint_path)


class Trainer:
    """Encapsulate training state and loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device, self.n_cpu, self.exp_config = load_experiment_config(args)
        self._setup_logging()

        torch.set_float32_matmul_precision("high")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_printoptions(threshold=100_000)
        torch._dynamo.config.capture_scalar_outputs = True
        # torch._dynamo.config.recompile_limit = 128

        (
            self.model,
            self.optimizer,
            self.start_epoch,
            self.global_step,
        ) = initialize_model(args, self.exp_config, self.device)

    def _setup_logging(self) -> None:
        """Configure the global logger."""
        log_level = getattr(logging, self.args.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )
        self.logger = logging.getLogger(LOGGER_NAME)
        self.logger.info("Using device: %s", self.device)

    def train(self) -> None:
        """Run the training loop."""
        train_loop(
            model=self.model,
            optimizer=self.optimizer,
            exp_config=self.exp_config,
            device=self.device,
            n_cpu=self.n_cpu,
            start_epoch=self.start_epoch,
            global_step=self.global_step,
            args=self.args,
        )


def train_loop(
    model: HierarchicalAutoencoder,
    optimizer: optim.Optimizer,
    exp_config: ExpConfig,
    device: str,
    n_cpu: int,
    start_epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    """Run the full training loop."""

    logger = logging.getLogger(LOGGER_NAME)

    mlflow.set_experiment(getattr(exp_config, "project_name", "DefaultExperiment"))
    mlflow_run = mlflow.start_run(run_name=getattr(exp_config, "run_name", "DefaultRun"))
    mlflow_client = MlflowClient()
    mlflow_run_id = mlflow_run.info.run_id
    mlflow_metric_buffer = []

    tokenizer = ByteLevelTokenizer(
        add_bos=True,
        add_eos=True,
        expected_vocab_size=exp_config.initial_vocab_size,
    )

    logger.info(
        "Tokenizer initialized. BOS ID: %s, EOS ID: %s, PAD ID: %s, Effective Vocab Size: %s",
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.pad_id,
        tokenizer.vocab_size,
    )

    logger.info("\nLoading dataset '%s'...", exp_config.dataset_name)
    raw_dataset = load_dataset(
        exp_config.dataset_name,
        name=exp_config.dataset_config,
        split=exp_config.dataset_train_split,
    )
    logger.info("Dataset loaded. Number of examples: %s", len(raw_dataset))

    logger.info(
        "\nTokenizing and processing dataset with sequence length %s...",
        exp_config.sequence_length,
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
        num_proc=n_cpu,
    )
    logger.info("Tokenization complete.")

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "labels", "key_padding_mask"],
    )

    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=exp_config.batch_size,
        num_workers=n_cpu,
        pin_memory=True if device == "cuda" else False,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / exp_config.gradient_accumulation_steps
    )
    if exp_config.max_steps is not None:
        exp_config.num_training_steps = int(exp_config.max_steps)
    else:
        exp_config.num_training_steps = exp_config.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=exp_config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=exp_config.warmup_steps,
        num_training_steps=exp_config.num_training_steps,
        scheduler_specific_kwargs=exp_config.scheduler_specific_kwargs if hasattr(exp_config, "scheduler_specific_kwargs") else {},
    )

    mlflow.log_params(exp_config.as_dict())

    model.train()
    if args.load_base_from:
        model.compressors.eval()
        model.expanders.eval()
    optimizer.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()
    training_start_time = time.time()
    total_bytes_processed = 0
    total_patches_processed_per_level = [0.0] * exp_config.num_levels

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
            "all_smoothed_perplexities": [0.0] * num_levels,
            "non_padded_tokens": 0,
            "count": 0,
            "avg_top_code_lm_loss": 0.0,
            "top_code_lm_loss_details": defaultdict(float),
            "top_code_mse": 0.0,
            "top_code_vq_loss": 0.0,
        }
        return accumulators

    accumulators = reset_accumulators(exp_config.num_levels)
    tok_s_deque = deque(maxlen=10)
    time_of_last_optimizer_step_event = training_start_time
    total_minibatches_in_epoch = len(train_dataloader)

    for epoch in range(start_epoch, exp_config.num_epochs):
        logger.info("\n--- Epoch %s/%s ---", epoch + 1, exp_config.num_epochs)
        _cached_cross_window_mask_cpu.cache_clear()
        _cached_causal_mask_cpu.cache_clear()
        epoch_start_time = time.time()
        model.train()
        if args.load_base_from:
            model.compressors.eval()
            model.expanders.eval()

        for i, batch in enumerate(train_dataloader):
            tokens = batch["input_ids"].to(device)
            key_padding_mask = batch["key_padding_mask"].to(device)
            model_kpm = key_padding_mask if exp_config.propagate_key_padding_mask else None

            output_dict = model(tokens, key_padding_mask=model_kpm)
            total_loss = output_dict["total_loss"]

            loss_for_backward = total_loss / exp_config.gradient_accumulation_steps
            loss_for_backward.backward()

            accumulators["total_loss"] += total_loss.item()
            accumulators["vq_loss"] += output_dict["vq_loss"].item()
            accumulators["avg_reconstruction_loss"] += output_dict["avg_reconstruction_loss"].item()
            accumulators["count"] += 1
            accumulators["non_padded_tokens"] += (~key_padding_mask).sum().item()

            for key, value in output_dict["reconstruction_loss_details"].items():
                accumulators["reconstruction_loss_details"][key] += value.item()

            if "avg_aux_lm_loss" in output_dict and exp_config.get("aux_lm_loss_weight", 0.0) > 0:
                accumulators["avg_aux_lm_loss"] += output_dict["avg_aux_lm_loss"].item()
                for key, value in output_dict["aux_lm_loss_details"].items():
                    accumulators["aux_lm_loss_details"][key] += value.item()

            if "avg_top_code_lm_loss" in output_dict and exp_config.get("top_lm_loss_weight", 0.0) > 0:
                accumulators["avg_top_code_lm_loss"] += output_dict["avg_top_code_lm_loss"].item()
                for key, value in output_dict["top_code_lm_loss_details"].items():
                    accumulators["top_code_lm_loss_details"][key] += value.item()
                    if key == "top_code_mse":
                        accumulators["top_code_mse"] += value.item()
                    elif key == "top_code_vq_loss":
                        accumulators["top_code_vq_loss"] += value.item()

            if "compression_ratios" in output_dict:
                for level_idx, ratio in enumerate(output_dict["compression_ratios"]):
                    accumulators["compression_ratios"][level_idx] += ratio

                batch_sz = tokens.size(0)
                for level_idx, length in enumerate(output_dict["input_seq_lengths_compressors"]):
                    accumulators["input_seq_lengths_compressors"][level_idx] += length * batch_sz
                for level_idx, length in enumerate(output_dict["output_seq_lengths_compressors"]):
                    accumulators["output_seq_lengths_compressors"][level_idx] += length * batch_sz

            if "all_codebook_perplexities" in output_dict:
                for level_idx, perplexity in enumerate(output_dict["all_codebook_perplexities"]):
                    accumulators["all_codebook_perplexities"][level_idx] += perplexity.item()

            if "all_smoothed_perplexities" in output_dict:
                for level_idx, smooth_perplexity in enumerate(output_dict["all_smoothed_perplexities"]):
                    accumulators["all_smoothed_perplexities"][level_idx] += smooth_perplexity.item()

            if (i + 1) % exp_config.gradient_accumulation_steps == 0:
                if exp_config.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), exp_config.gradient_clip_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if device == "cuda":
                    torch.cuda.synchronize()
                current_time = time.time()
                duration_accumulation_window = current_time - time_of_last_optimizer_step_event

                tokens_processed_this_window = accumulators["non_padded_tokens"]
                tokens_per_second = tokens_processed_this_window / duration_accumulation_window
                tok_s_deque.append(tokens_per_second)
                avg_tok_s = sum(tok_s_deque) / len(tok_s_deque)

                patches_processed_this_window = accumulators["output_seq_lengths_compressors"]
                patches_per_second = [p / duration_accumulation_window for p in patches_processed_this_window]
                for lvl, cnt in enumerate(patches_processed_this_window):
                    total_patches_processed_per_level[lvl] += cnt

                total_bytes_processed += tokens_processed_this_window
                total_progress = (global_step + 1) / exp_config.num_training_steps
                total_elapsed = current_time - training_start_time
                total_eta_sec = (total_elapsed / total_progress - total_elapsed) if total_progress > 0 else 0

                steps_accumulated = accumulators["count"]

                if steps_accumulated > 0:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    console_log_parts = [
                        f"{timestamp}",
                        f"Epoch {epoch + 1}/{exp_config.num_epochs}",
                        f"OptStep {global_step}",
                        f"MB {i + 1}/{total_minibatches_in_epoch}",
                        f"Loss {accumulators['total_loss'] / steps_accumulated:.4f}",
                        f"Reco {accumulators['avg_reconstruction_loss'] / steps_accumulated:.4f}",
                        f"VQ {accumulators['vq_loss'] / steps_accumulated:.4f}",
                    ]

                    if exp_config.get("aux_lm_loss_weight", 0.0) > 0 and "avg_aux_lm_loss" in accumulators:
                        console_log_parts.append(
                            f"AuxLM {accumulators['avg_aux_lm_loss'] / steps_accumulated:.4f}"
                        )

                    if "avg_top_code_lm_loss" in output_dict and exp_config.get("top_lm_loss_weight", 0.0) > 0:
                        console_log_parts.append(
                            f"TopLM {output_dict['avg_top_code_lm_loss'].item():.4f}"
                        )
                        if "top_code_lm_loss_details" in output_dict:
                            tcld = output_dict["top_code_lm_loss_details"]
                            if "top_code_mse" in tcld:
                                console_log_parts.append(f"TopMSE {tcld['top_code_mse'].item():.4f}")
                            if "top_code_vq_loss" in tcld:
                                console_log_parts.append(f"TopVQ {tcld['top_code_vq_loss'].item():.4f}")

                    console_log_parts.extend(
                        [
                            f"Tok/s {short_num(avg_tok_s)}",
                            f"Bytes {short_num(total_bytes_processed)}",
                            f"ETAt {format_duration(total_eta_sec)}",
                        ]
                    )

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

                    if "compression_ratios" in accumulators and len(accumulators["compression_ratios"]) == exp_config.num_levels:
                        ratios_str = ", ".join([f"{r / steps_accumulated:.2f}" for r in accumulators["compression_ratios"]])
                        console_log_parts.append(f"Ratios [{ratios_str}]")

                    if "all_smoothed_perplexities" in accumulators and len(accumulators["all_smoothed_perplexities"]) == exp_config.num_levels:
                        ppl_str = ", ".join([
                            f"{p / steps_accumulated:.4f}" for p in accumulators["all_smoothed_perplexities"]
                        ])
                        console_log_parts.append(f"SmoothPPL [{ppl_str}]")

                    logger.info(" | ".join(console_log_parts))

                    if global_step % exp_config.log_interval == 0 and accumulators["count"] > 0:
                        metrics_dict = {
                            "loss/total_avg_accum": accumulators["total_loss"] / steps_accumulated,
                            "loss/vq_avg_accum": accumulators["vq_loss"] / steps_accumulated,
                            "loss/reconstruction_avg_accum": accumulators["avg_reconstruction_loss"] / steps_accumulated,
                            "performance/tokens_per_sec": tokens_per_second,
                            "loss/top_code_lm_avg_accum": accumulators["avg_top_code_lm_loss"] / steps_accumulated,
                            "loss/top_code_mse_avg_accum": accumulators["top_code_mse"] / steps_accumulated,
                            "loss/top_code_vq_avg_accum": accumulators["top_code_vq_loss"] / steps_accumulated,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                        for key, value in accumulators["top_code_lm_loss_details"].items():
                            metrics_dict[f"loss_detail_avg_accum/{key}"] = value / steps_accumulated

                        for key, value in accumulators["reconstruction_loss_details"].items():
                            metrics_dict[f"loss_detail_avg_accum/{key}"] = value / steps_accumulated

                        if exp_config.get("aux_lm_loss_weight", 0.0) > 0:
                            metrics_dict["loss/aux_lm_avg_accum"] = accumulators["avg_aux_lm_loss"] / steps_accumulated
                            for key, value in accumulators["aux_lm_loss_details"].items():
                                metrics_dict[f"loss_detail_avg_accum/{key}"] = value / steps_accumulated

                        if "compression_ratios" in output_dict:
                            for level_idx in range(exp_config.num_levels):
                                metrics_dict[f"compression_avg/ratio_L{level_idx}"] = (
                                    accumulators["compression_ratios"][level_idx] / steps_accumulated
                                )
                                metrics_dict[f"compression_avg/input_len_L{level_idx}"] = (
                                    accumulators["input_seq_lengths_compressors"][level_idx] / steps_accumulated
                                )
                                metrics_dict[f"compression_avg/output_len_L{level_idx}"] = (
                                    accumulators["output_seq_lengths_compressors"][level_idx] / steps_accumulated
                                )

                        if "all_codebook_perplexities" in output_dict:
                            for level_idx in range(exp_config.num_levels):
                                metrics_dict[f"vq_metrics_avg/perplexity_L{level_idx}"] = (
                                    accumulators["all_codebook_perplexities"][level_idx] / steps_accumulated
                                )
                                codebook_size_L_i = exp_config.compressor_level_configs[level_idx].codebook_size
                                metrics_dict[f"vq_metrics/codebook_size_L{level_idx}"] = codebook_size_L_i

                        if "all_smoothed_perplexities" in output_dict:
                            for level_idx in range(exp_config.num_levels):
                                metrics_dict[f"vq_metrics_avg/smooth_perplexity_L{level_idx}"] = (
                                    accumulators["all_smoothed_perplexities"][level_idx] / steps_accumulated
                                )

                        mlflow_metric_buffer.append((global_step, metrics_dict))
                        if len(mlflow_metric_buffer) >= exp_config.mlflow_batch_interval:
                            timestamp_ms = int(time.time() * 1000)
                            metrics_entities = []
                            for step_id, mdict in mlflow_metric_buffer:
                                metrics_entities.extend(
                                    [Metric(key=k, value=v, timestamp=timestamp_ms, step=step_id) for k, v in mdict.items()]
                                )
                            mlflow_client.log_batch(mlflow_run_id, metrics=metrics_entities)
                            mlflow_metric_buffer = []

                accumulators = reset_accumulators(exp_config.num_levels)
                time_of_last_optimizer_step_event = current_time

                if global_step > 0 and global_step % exp_config.generation_interval == 0:
                    logger.info("\nStep %s: Generating sample...", global_step)
                    model.eval()
                    with torch.no_grad():
                        sample_text = exp_config.sample_prompt_for_generation
                        input_gen_tokens = tokenizer.encode(sample_text).unsqueeze(0).to(device)

                        input_gen_kpm = None
                        if exp_config.propagate_key_padding_mask:
                            input_gen_kpm = torch.zeros_like(input_gen_tokens, dtype=torch.bool).to(device)

                        reconstructed_tokens = model.generate_bytes(
                            tokens=input_gen_tokens,
                            key_padding_mask=input_gen_kpm,
                            max_len_override=exp_config.generation_max_len_override,
                        )
                        reconstructed_text = tokenizer.decode(
                            reconstructed_tokens.squeeze(0).cpu(), cut_at_eos=True
                        )

                        logger.info("--- Sample Generation at Step %s ---", global_step)
                        logger.info("Original Prompt:\n%s", sample_text)
                        logger.info("Reconstructed Text:\n%s", reconstructed_text)
                        logger.info("------------------------------------------")

                        mlflow_text_log = f"Original:\n{sample_text}\n\nReconstructed:\n{reconstructed_text}"
                        mlflow.log_text(mlflow_text_log, f"sample_generation_step_{global_step}.txt")

                    model.train()
                    if args.load_base_from:
                        model.compressors.eval()
                        model.expanders.eval()
                if global_step > 0 and global_step % exp_config.checkpoint_interval == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        exp_config.checkpoint_dir,
                        exp_config,
                        logger,
                    )

                global_step += 1
                if exp_config.max_steps is not None and global_step >= exp_config.max_steps:
                    logger.info("Reached max_steps %s. Stopping training.", exp_config.max_steps)
                    break
        epoch_duration = time.time() - epoch_start_time
        logger.info("--- Epoch %s finished. Duration: %.2fs ---", epoch + 1, epoch_duration)
        save_checkpoint(
            model,
            optimizer,
            epoch,
            global_step,
            exp_config.checkpoint_dir,
            exp_config,
            logger,

        )

        if exp_config.max_steps is not None and global_step >= exp_config.max_steps:
            break

    logger.info("Training finished.")
    if exp_config.save_base_components_path:
        save_base_components(model, exp_config.save_base_components_path)
    if mlflow_metric_buffer:
        timestamp_ms = int(time.time() * 1000)
        metrics_entities = []
        for step_id, mdict in mlflow_metric_buffer:
            metrics_entities.extend(
                [Metric(key=k, value=v, timestamp=timestamp_ms, step=step_id) for k, v in mdict.items()]
            )
        mlflow_client.log_batch(mlflow_run_id, metrics=metrics_entities)
    mlflow.end_run()


def main() -> None:
    """Entry point for command line execution."""

    args = parse_args()
    trainer = Trainer(args)
    trainer.train()



if __name__ == "__main__":
    main()

