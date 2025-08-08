from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from components.utils import short_num, format_duration


@dataclass
class WindowThroughput:
    tokens_processed: int
    tokens_per_second: float
    avg_tokens_per_second: float
    patches_processed_per_level: List[float]
    patches_per_second_per_level: List[float]


class SmoothEMA:
    def __init__(self, size: int, alpha: float = 0.9) -> None:
        self.values = [0.0] * size
        self.ready = [False] * size
        self.alpha = alpha

    def update(self, index: int, value: float) -> None:
        if not self.ready[index]:
            self.values[index] = value
            self.ready[index] = True
        else:
            self.values[index] = self.alpha * self.values[index] + (1.0 - self.alpha) * value


class TrainingMetrics:
    """Accumulates metrics for a gradient accumulation window and provides summaries.

    This keeps the training loop readable while preserving the same metrics and keys.
    """

    def __init__(self, num_levels: int, aux_lm_enabled: bool, top_lm_enabled: bool, tok_s_window: int = 10) -> None:
        self.num_levels = num_levels
        self.aux_lm_enabled = aux_lm_enabled
        self.top_lm_enabled = top_lm_enabled
        self._tok_s_deque = deque(maxlen=tok_s_window)
        self._ppl_ema = SmoothEMA(num_levels, alpha=0.9)
        self.total_patches_processed_per_level = [0.0] * num_levels
        self.total_bytes_processed = 0
        self.reset_window()

    # -------- Window accumulation --------
    def reset_window(self) -> None:
        self.total_loss = 0.0
        self.vq_loss = 0.0
        self.avg_reconstruction_loss = 0.0
        self.reconstruction_loss_details: Dict[str, float] = defaultdict(float)

        self.avg_aux_lm_loss = 0.0
        self.aux_lm_loss_details: Dict[str, float] = defaultdict(float)

        self.avg_top_code_lm_loss = 0.0
        self.top_code_lm_loss_details: Dict[str, float] = defaultdict(float)
        self.top_code_mse = 0.0
        self.top_code_vq_loss = 0.0

        self.compression_ratios = [0.0] * self.num_levels
        self.input_seq_lengths = [0.0] * self.num_levels
        self.output_seq_lengths = [0.0] * self.num_levels

        self.all_codebook_perplexities = [0.0] * self.num_levels
        self.non_padded_tokens = 0
        self.count = 0

        self._saw_compression = False
        self._saw_perplexities = False

    def update_from_batch(self, output: Dict, key_padding_mask: torch.Tensor) -> None:
        self.total_loss += float(output["total_loss"])  # tensor -> python float
        self.vq_loss += float(output["vq_loss"])
        self.avg_reconstruction_loss += float(output["avg_reconstruction_loss"])
        self.count += 1
        self.non_padded_tokens += int((~key_padding_mask).sum().item())

        for k, v in output.get("reconstruction_loss_details", {}).items():
            self.reconstruction_loss_details[k] += float(v)

        if self.aux_lm_enabled and "avg_aux_lm_loss" in output:
            self.avg_aux_lm_loss += float(output["avg_aux_lm_loss"])
            for k, v in output.get("aux_lm_loss_details", {}).items():
                self.aux_lm_loss_details[k] += float(v)

        if self.top_lm_enabled and "avg_top_code_lm_loss" in output:
            self.avg_top_code_lm_loss += float(output["avg_top_code_lm_loss"])
            for k, v in output.get("top_code_lm_loss_details", {}).items():
                self.top_code_lm_loss_details[k] += float(v)
                if k == "top_code_mse":
                    self.top_code_mse += float(v)
                elif k == "top_code_vq_loss":
                    self.top_code_vq_loss += float(v)

        if "compression_ratios" in output:
            self._saw_compression = True
            for lvl, ratio in enumerate(output["compression_ratios"]):
                self.compression_ratios[lvl] += float(ratio)

            # Per-level input/output lengths from compression results
            comp_steps = output.get("compression_results", {}).get("steps", [])
            for lvl, step in enumerate(comp_steps):
                # (~padding).sum() counts non-padded tokens at compressor input
                self.input_seq_lengths[lvl] += float((~step.input_padding_mask).sum().item())
                self.output_seq_lengths[lvl] += float(step.patch_end_mask.sum().item())

        if "all_codebook_perplexities" in output:
            self._saw_perplexities = True
            for lvl, ppl in enumerate(output["all_codebook_perplexities"]):
                val = float(ppl) if isinstance(ppl, (float, int)) else float(ppl.item())
                self.all_codebook_perplexities[lvl] += val
                self._ppl_ema.update(lvl, val)

    # -------- Window summarization --------
    def throughput(self, duration_sec: float) -> WindowThroughput:
        tokens_per_second = self.non_padded_tokens / duration_sec if duration_sec > 0 else 0.0
        self._tok_s_deque.append(tokens_per_second)
        avg_tok_s = sum(self._tok_s_deque) / len(self._tok_s_deque) if self._tok_s_deque else 0.0

        patches_per_sec = [
            (self.output_seq_lengths[lvl] / duration_sec) if duration_sec > 0 else 0.0
            for lvl in range(self.num_levels)
        ]
        for lvl in range(self.num_levels):
            self.total_patches_processed_per_level[lvl] += self.output_seq_lengths[lvl]

        self.total_bytes_processed += self.non_padded_tokens

        return WindowThroughput(
            tokens_processed=self.non_padded_tokens,
            tokens_per_second=tokens_per_second,
            avg_tokens_per_second=avg_tok_s,
            patches_processed_per_level=self.output_seq_lengths[:],
            patches_per_second_per_level=patches_per_sec,
        )

    def console_parts(
        self,
        timestamp: str,
        epoch: int,
        num_epochs: int,
        global_step: int,
        minibatch_index: int,
        total_minibatches: int,
        avg_tok_s: float,
        total_eta_sec: float,
        patches_per_second: List[float],
    ) -> List[str]:
        if self.count == 0:
            return []
        parts = [
            f"{timestamp}",
            f"Epoch {epoch}/{num_epochs}",
            f"OptStep {global_step}",
            f"MB {minibatch_index}/{total_minibatches}",
            f"Loss {self.total_loss / self.count:.4f}",
            f"Reco {self.avg_reconstruction_loss / self.count:.4f}",
            f"VQ {self.vq_loss / self.count:.4f}",
            f"Tok/s {short_num(avg_tok_s)}",
            f"Bytes {short_num(self.total_bytes_processed)}",
            f"ETA {format_duration(total_eta_sec)}",
        ]

        if self.aux_lm_enabled:
            parts.append(f"AuxLM {self.avg_aux_lm_loss / self.count:.4f}")

        if self.top_lm_enabled:
            if self.count > 0:
                parts.append(f"TopLM {self.avg_top_code_lm_loss / self.count:.4f}")
            if self.top_code_mse:
                parts.append(f"TopMSE {self.top_code_mse / self.count:.4f}")
            if self.top_code_vq_loss:
                parts.append(f"TopVQ {self.top_code_vq_loss / self.count:.4f}")

        # Patches summary (skip level 0 in console as before)
        patch_log_parts = []
        for lvl in range(1, self.num_levels):
            label = "Top" if lvl == self.num_levels - 1 else f"L{lvl}"
            pps_rate = patches_per_second[lvl] if lvl < len(patches_per_second) else 0.0
            pps_total = self.total_patches_processed_per_level[lvl]
            patch_log_parts.append(
                f"{label} {short_num(pps_rate)}/s {short_num(pps_total)}"
            )
        if patch_log_parts:
            parts.append("Patches " + ", ".join(patch_log_parts))

        if self._saw_compression and len(self.compression_ratios) == self.num_levels:
            ratios_str = ", ".join([f"{r / self.count:.2f}" for r in self.compression_ratios])
            parts.append(f"Ratios [{ratios_str}]")

        ppl_str = ", ".join(
            f"{self._ppl_ema.values[lvl]:.4f}" if self._ppl_ema.ready[lvl] else "n/a"
            for lvl in range(self.num_levels)
        )
        parts.append(f"SmoothPPL [{ppl_str}]")

        for k, v in self.reconstruction_loss_details.items():
            parts.append(f"{k}:{v / self.count:.4f}")
        if self.aux_lm_enabled:
            for k, v in self.aux_lm_loss_details.items():
                parts.append(f"{k}:{v / self.count:.4f}")

        return parts

    def metrics_dict(
        self,
        learning_rate: float,
        tokens_per_second: float,
        patches_per_second: List[float],
        codebook_sizes: List[int] | None,
    ) -> Dict[str, float]:
        if self.count == 0:
            return {}

        md: Dict[str, float] = {
            "loss/total_avg_accum": self.total_loss / self.count,
            "loss/vq_avg_accum": self.vq_loss / self.count,
            "loss/reconstruction_avg_accum": self.avg_reconstruction_loss / self.count,
            "performance/tokens_per_sec": tokens_per_second,
            "loss/top_code_lm_avg_accum": self.avg_top_code_lm_loss / self.count if self.top_lm_enabled else 0.0,
            "loss/top_code_mse_avg_accum": self.top_code_mse / self.count if self.top_lm_enabled else 0.0,
            "loss/top_code_vq_avg_accum": self.top_code_vq_loss / self.count if self.top_lm_enabled else 0.0,
            "learning_rate": learning_rate,
        }

        # Per-level patches performance (skip level 0 as in original)
        for lvl in range(1, self.num_levels):
            md[f"performance/patches_per_sec_L{lvl}"] = patches_per_second[lvl]
            md[f"performance/patches_total_L{lvl}"] = self.total_patches_processed_per_level[lvl]

        for k, v in self.top_code_lm_loss_details.items():
            md[f"loss_detail_avg_accum/{k}"] = v / self.count
        for k, v in self.reconstruction_loss_details.items():
            md[f"loss_detail_avg_accum/{k}"] = v / self.count
        if self.aux_lm_enabled:
            md["loss/aux_lm_avg_accum"] = self.avg_aux_lm_loss / self.count
            for k, v in self.aux_lm_loss_details.items():
                md[f"loss_detail_avg_accum/{k}"] = v / self.count

        if self._saw_compression:
            for lvl in range(self.num_levels):
                md[f"compression_avg/ratio_L{lvl}"] = self.compression_ratios[lvl] / self.count
                md[f"compression_avg/input_len_L{lvl}"] = self.input_seq_lengths[lvl] / self.count
                md[f"compression_avg/output_len_L{lvl}"] = self.output_seq_lengths[lvl] / self.count

        if self._saw_perplexities:
            for lvl in range(self.num_levels):
                md[f"vq_metrics_avg/perplexity_L{lvl}"] = self.all_codebook_perplexities[lvl] / self.count
                if codebook_sizes is not None:
                    md[f"vq_metrics/codebook_size_L{lvl}"] = int(codebook_sizes[lvl])
        for lvl in range(self.num_levels):
            if self._ppl_ema.ready[lvl]:
                md[f"vq_metrics_ema/smooth_perplexity_L{lvl}"] = self._ppl_ema.values[lvl]

        return md

    # (No local formatting helpers; use shared utils)


class MlflowBatchLogger:
    """Thin wrapper to batch MLflow metric logging, mirroring original behavior."""

    def __init__(self, client, run_id: str, batch_interval: int) -> None:
        self.client = client
        self.run_id = run_id
        self.batch_interval = batch_interval
        self._buffer: List[Tuple[int, Dict[str, float]]] = []

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        self._buffer.append((step, metrics))
        if len(self._buffer) >= self.batch_interval:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        from mlflow.entities import Metric  # local import to keep module import-light

        timestamp_ms = int(time.time() * 1000)
        metrics_entities = []
        for step_id, mdict in self._buffer:
            metrics_entities.extend(
                [Metric(key=k, value=v, timestamp=timestamp_ms, step=step_id) for k, v in mdict.items()]
            )
        self.client.log_batch(self.run_id, metrics=metrics_entities)
        self._buffer.clear()
