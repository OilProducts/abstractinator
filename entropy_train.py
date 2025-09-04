"""
Standalone trainer for the entropy stack (embedding + shared layers + FlexibleEntropyModel).

Usage:
  python entropy_train.py --config experiments/regular_flex_512.py --save_entropy_to ./entropy_stack.pt

Notes:
  - Trains only the entropy stack. Compression layers, pooling, and quantizer
    are not used and are frozen.
  - The saved checkpoint can be loaded by config on an Abstractinator level via:
      level.c_entropy_load_path = "./entropy_stack.pt"
      level.c_entropy_freeze = True
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from dataclasses import replace
import math
import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from functools import partial

from components.config_types import EntropyModelConfig
from components.utils import short_num, format_duration
import mlflow
from components.segment_compressor import SegmentCompressor
from components.tokenizer import ByteLevelTokenizer
from components.checkpoint_utils import save_entropy_stack
from data_utils import tokenize_and_process_examples
from experiments.exp_config import DEVICE as DEFAULT_DEVICE

LOGGER = logging.getLogger("entropy_train")


def _load_config(path: str):
    spec = importlib.util.spec_from_file_location("config_module", path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Train only the entropy stack")
    ap.add_argument("--config", type=str, required=True, help="Path to experiment config .py")
    ap.add_argument("--save_entropy_to", type=str, required=True, help="Output file for entropy stack")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=8)
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    # Match main training performance toggles
    try:
        import torch._dynamo as dynamo  # noqa: F401
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.set_default_dtype(torch.bfloat16)
        torch.set_float32_matmul_precision("high")
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    cfg_mod = _load_config(args.config)
    exp_cfg = cfg_mod.exp_config
    # Resolve device from config or CLI default
    device = getattr(cfg_mod, "DEVICE", None) or getattr(exp_cfg, "device", None) or DEFAULT_DEVICE

    level_cfg = exp_cfg.pyramid_config.levels[-1]

    # CLI args overridden by exp_config when present
    batch_size = getattr(exp_cfg, "batch_size", args.batch_size)
    seq_len = getattr(exp_cfg, "sequence_length", args.seq_len)
    epochs = getattr(exp_cfg, "num_epochs", args.epochs)
    lr = getattr(exp_cfg, "learning_rate", args.lr)
    log_every = getattr(exp_cfg, "log_interval", args.log_every)
    max_steps = getattr(exp_cfg, "max_steps", None) if getattr(exp_cfg, "max_steps", None) is not None else args.max_steps
    num_workers = args.num_workers

    # Build compressor with only the pieces we train/use
    entropy_cfg: EntropyModelConfig | None = getattr(level_cfg, "c_entropy_config", None)
    compressor = SegmentCompressor(
        vocab_size=level_cfg.vocab_size,
        dim=level_cfg.D,
        heads=level_cfg.c_heads,
        window=level_cfg.c_window,
        head_dim=level_cfg.c_head_dim,
        kv_comp_dim=level_cfg.c_kv_comp_dim,
        q_comp_dim=level_cfg.c_q_comp_dim,
        retr_dim=level_cfg.c_retr_dim,
        num_encoder_layers=level_cfg.c_num_encoder_layers,
        num_shared_encoder_layers=level_cfg.c_num_shared_encoder_layers,
        num_lm_encoder_layers=level_cfg.c_num_lm_encoder_layers,
        num_compression_encoder_layers=0,
        num_queries=level_cfg.c_num_queries,
        entropy_delta=level_cfg.c_entropy_delta,
        entropy_abs_threshold=level_cfg.c_entropy_abs_threshold,
        output_length=level_cfg.c_output_length,
        quantizer=None,
        attention_config=level_cfg.compressor_attention,
        entropy_config=entropy_cfg,
    ).to(device)


    # Freeze unused paths
    compressor.compression_layers.requires_grad_(False)
    compressor.pooler.requires_grad_(False)
    if hasattr(compressor, "quantizer") and compressor.quantizer is not None:
        compressor.quantizer.requires_grad_(False)

    # Optimizer over embedding + shared + entropy_model
    params = list(compressor.embedding.parameters())
    params += list(compressor.shared_layers.parameters())
    params += list(compressor.entropy_model.parameters())
    optimizer = optim.AdamW(params, lr=lr)

    tokenizer = ByteLevelTokenizer()

    raw_dataset = load_dataset(
        exp_cfg.dataset_name,
        name=exp_cfg.dataset_config,
        split=exp_cfg.dataset_train_split,
    )

    tokenized_dataset = raw_dataset.map(
        partial(
            tokenize_and_process_examples,
            sequence_length=seq_len,
            tokenizer=tokenizer,
            text_column=exp_cfg.text_column_name,
        ),
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=num_workers,
        desc="Byte-tokenising for entropy-only",
    )

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "labels", "key_padding_mask"],
        output_all_columns=True,
    )

    loader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        shuffle=True,
    )

    # --- Summary logging (params, steps, bytes) ---
    def _count_params(mod: torch.nn.Module) -> int:
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)

    emb_params = _count_params(compressor.embedding)
    shared_params = _count_params(compressor.shared_layers)
    entropy_params = _count_params(compressor.entropy_model)
    total_trainable = emb_params + shared_params + entropy_params

    LOGGER.info(
        "Trainable params → embedding: %s | shared: %s | entropy: %s | total: %s",
        short_num(emb_params),
        short_num(shared_params),
        short_num(entropy_params),
        short_num(total_trainable),
    )

    # MLflow setup
    try:
        mlflow.set_experiment(getattr(exp_cfg, "project_name", "EntropyTraining"))
        run_name = f"entropy_{getattr(exp_cfg, 'run_name', 'run')}"
        mlflow.start_run(run_name=run_name)
        # Log run parameters
        try:
            mlflow.log_params(exp_cfg.as_dict())
        except Exception:
            pass
        mlflow.log_params({
            "entropy_lr": lr,
            "entropy_batch_size": batch_size,
            "entropy_seq_len": seq_len,
            "entropy_log_every": log_every,
            "entropy_max_steps": max_steps if max_steps is not None else -1,
        })
    except Exception:
        LOGGER.warning("MLflow not available; continuing without experiment logging.")

    step = 0
    total_batches_per_epoch = len(loader)
    total_steps_planned = max_steps if max_steps is not None else (epochs * total_batches_per_epoch)
    # Approximate total bytes planned (upper bound): steps * batch_size * seq_len
    approx_total_bytes = total_steps_planned * batch_size * seq_len
    LOGGER.info(
        "Planned run → steps: %s | epochs: %s | batches/epoch: %s | batch_size: %s | seq_len: %s | approx bytes: %s",
        short_num(total_steps_planned),
        short_num(epochs),
        short_num(total_batches_per_epoch),
        short_num(batch_size),
        short_num(seq_len),
        short_num(approx_total_bytes),
    )
    start_time = time.time()
    window_start_time = start_time
    toks_total = 0
    toks_window = 0
    bytes_total = 0
    bytes_window = 0


    # --- generation helper ---
    def _generate_once(prompt: str, max_new: int = 64, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> str:
        tok_local = tokenizer
        with torch.no_grad():
            compressor.eval()
            try:
                # Prompt ids (1,S)
                ids = tok_local.encode(prompt, add_eos=False).unsqueeze(0).to(device).to(torch.int64)
                cache, _ = compressor.stream_prefill_ids(ids, key_padding_mask=None)
                E = compressor.embedding.weight  # (V,D)

                generated_ids = []
                for _ in range(max_new):
                    next_id = None
                    # Prefer logits if available
                    if getattr(cache.entropy, "last_logits", None) is not None:
                        logits = cache.entropy.last_logits  # (B,V)
                        if temperature != 1.0 and temperature > 0:
                            logits = logits / temperature
                        if top_k and top_k < logits.size(-1):
                            v, idx = torch.topk(logits, top_k, dim=-1)
                            mask = torch.full_like(logits, float("-inf"))
                            logits = mask.scatter(-1, idx, v)
                        if 0.0 < top_p < 1.0:
                            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                            probs = torch.softmax(sorted_logits, dim=-1)
                            cum = probs.cumsum(dim=-1)
                            cutoff = (cum > top_p).float().argmax(dim=-1, keepdim=True)
                            mask = torch.full_like(sorted_logits, float("-inf"))
                            keep = torch.arange(sorted_logits.size(-1), device=logits.device)[None, :] <= cutoff
                            sorted_logits = torch.where(keep, sorted_logits, mask)
                            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
                        probs = torch.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
                    elif getattr(cache.entropy, "last_mu", None) is not None:
                        mu = cache.entropy.last_mu  # (B,D)
                        logvar = cache.entropy.last_logvar  # (B,D)
                        std = torch.exp(0.5 * logvar)
                        z = mu if temperature <= 0 else mu + (temperature * std) * torch.randn_like(std)
                        logits = torch.matmul(z, E.t())  # (B,V)
                        next_id = logits.argmax(dim=-1)
                    else:
                        break

                    compressor.stream_step_ids(cache, next_id.unsqueeze(1), key_padding_mask_new=None)
                    generated_ids.append(next_id)

                if generated_ids:
                    new_ids = torch.stack(generated_ids, dim=1).squeeze(0)  # (T)
                else:
                    new_ids = torch.empty(0, dtype=torch.long, device=device)
                return tok_local.decode(new_ids.cpu(), cut_at_eos=True)
            finally:
                compressor.train()

    for epoch in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True).to(torch.int64)
            key_padding_mask = batch["key_padding_mask"].to(device, non_blocking=True).to(torch.bool)
            out = compressor.entropy_loss_ids(input_ids, key_padding_mask)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            step += 1

            # Throughput accounting
            valid_mask = (~key_padding_mask).to(torch.int64)
            batch_tokens = int(valid_mask.sum().item())
            batch_bytes = int(((input_ids < 256).to(torch.int64) * valid_mask).sum().item())
            toks_total += batch_tokens
            toks_window += batch_tokens
            bytes_total += batch_bytes
            bytes_window += batch_bytes

            if step % log_every == 0:
                now = time.time()
                dt = max(1e-6, now - window_start_time)
                total_dt = max(1e-6, now - start_time)
                tok_s = toks_window / dt
                bytes_s = bytes_window / dt
                avg_tok_s = toks_total / total_dt
                avg_bytes_s = bytes_total / total_dt
                progress = step / total_steps_planned if total_steps_planned > 0 else 0.0
                eta_sec = (total_dt / progress - total_dt) if progress > 0 else 0.0
                LOGGER.info(
                    "step %d/%d | loss=%.4f | tok/s=%s (avg %s) | bytes/s=%s (avg %s) | total_bytes=%s | ETA=%s",
                    step,
                    total_steps_planned,
                    float(loss.item()),
                    short_num(tok_s),
                    short_num(avg_tok_s),
                    short_num(bytes_s),
                    short_num(avg_bytes_s),
                    short_num(bytes_total),
                    format_duration(eta_sec),
                )
                # MLflow metrics (best-effort)
                try:
                    mlflow.log_metrics(
                        {
                            "entropy_loss": float(loss.item()),
                            "tok_per_sec": float(tok_s),
                            "tok_per_sec_avg": float(avg_tok_s),
                            "bytes_per_sec": float(bytes_s),
                            "bytes_per_sec_avg": float(avg_bytes_s),
                            "total_bytes": float(bytes_total),
                            "eta_seconds": float(eta_sec),
                            "epoch": float(epoch),
                        },
                        step=step,
                    )
                except Exception:
                    pass
                window_start_time = now
                toks_window = 0
                bytes_window = 0

            # Periodic generation
            gen_interval = getattr(exp_cfg, "generation_interval", None)
            if gen_interval and step > 0 and (step % int(gen_interval) == 0):
                try:
                    prompt = getattr(exp_cfg, "sample_prompt_for_generation", "The purpose of education is to")
                    max_new = int(getattr(exp_cfg, "generation_max_len_override", 64) or 64)
                    sample = _generate_once(prompt, max_new=max_new, temperature=1.0, top_k=0, top_p=1.0)
                    LOGGER.info(
                        "--- Entropy Sample @ step %s ---\nPrompt: %s\nGenerated: %s\n----------------------------------",
                        short_num(step),
                        prompt,
                        prompt + sample,
                    )
                    try:
                        mlflow.log_text(f"Prompt:\n{prompt}\n\nGenerated:\n{prompt + sample}", f"entropy_sample_step_{step}.txt")
                    except Exception:
                        pass
                except Exception:
                    LOGGER.exception("Generation failed at step %d", step)

            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    os.makedirs(os.path.dirname(args.save_entropy_to) or ".", exist_ok=True)
    save_entropy_stack(compressor, args.save_entropy_to)
    LOGGER.info("Saved entropy stack to %s", args.save_entropy_to)
    try:
        mlflow.log_artifact(args.save_entropy_to)
        mlflow.end_run()
    except Exception:
        pass


if __name__ == "__main__":
    main()
