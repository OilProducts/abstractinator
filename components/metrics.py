from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from components.utils import format_duration, short_num


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

    def __init__(self, num_levels: int, entropy_enabled: bool, top_lm_enabled: bool, tok_s_window: int = 10) -> None:
        self.num_levels = num_levels
        # Whether to display/track entropy-model loss explicitly in console logs
        self.entropy_enabled = entropy_enabled
        self.top_lm_enabled = top_lm_enabled
        self._tok_s_deque = deque(maxlen=tok_s_window)
        self._ppl_ema = SmoothEMA(num_levels, alpha=0.9)
        # Long-running totals (CPU)
        self.total_patches_processed_per_level = [0.0] * num_levels
        self.total_bytes_processed = 0
        self.total_top_lm_positions = 0  # total effective positions seen by top LM (teacher-forced)
        # Device for GPU accumulators will be set on first update
        self._device: torch.device | None = None
        # Last-window rates (for console + MLflow)
        self._top_lm_positions_per_sec: float = 0.0
        self.reset_window()

    # -------- Window accumulation --------
    def reset_window(self) -> None:
        # GPU accumulators (created lazily on first batch to match device)
        self.total_loss_t = None
        self.vq_loss_t = None
        self.avg_reconstruction_loss_t = None
        self.reconstruction_loss_details: Dict[str, torch.Tensor] = defaultdict(torch.Tensor)

        self.avg_entropy_loss_t = None  # entropy_model_loss accumulator
        self.entropy_loss_details: Dict[str, torch.Tensor] = defaultdict(torch.Tensor)

        self.avg_top_code_lm_loss_t = None
        self.top_code_lm_loss_details: Dict[str, torch.Tensor] = defaultdict(torch.Tensor)
        self.top_code_mse_t = None
        self.top_code_vq_loss_t = None

        self.compression_ratios_t: list[torch.Tensor | None] = [None] * self.num_levels
        self.input_seq_lengths_t: list[torch.Tensor | None] = [None] * self.num_levels
        self.output_seq_lengths_t: list[torch.Tensor | None] = [None] * self.num_levels

        self.all_codebook_perplexities_t: list[torch.Tensor | None] = [None] * self.num_levels
        # Sum for window-average of perplexity; None means "no signal this window"
        self._ppl_sum_t: list[torch.Tensor | None] = [None] * self.num_levels
        self.non_padded_tokens_t = None
        self.top_lm_positions_t = None  # window sum of effective positions seen by top LM
        self.count = 0

        self._saw_compression = False
        self._saw_perplexities = False

    def update_from_batch(self, output: Dict, key_padding_mask: torch.Tensor) -> None:
        # Lazily capture device and create zero scalars
        dev = output["loss_total"].device
        if self._device is None:
            self._device = dev
        if self.total_loss_t is None:
            zero = torch.zeros((), device=dev, dtype=torch.float32)
            self.total_loss_t = zero.clone()
            self.vq_loss_t = zero.clone()
            self.avg_reconstruction_loss_t = zero.clone()
            self.avg_entropy_loss_t = zero.clone()
            self.avg_top_code_lm_loss_t = zero.clone()
            self.top_code_mse_t = zero.clone()
            self.top_code_vq_loss_t = zero.clone()
            self.non_padded_tokens_t = zero.clone()
            # initialise per-level lists
            self.input_seq_lengths_t = [zero.clone() for _ in range(self.num_levels)]
            self.output_seq_lengths_t = [zero.clone() for _ in range(self.num_levels)]
            self.all_codebook_perplexities_t = [None for _ in range(self.num_levels)]
            self._ppl_sum_t = [None for _ in range(self.num_levels)]

        # Window counters on device; no host syncs here
        self.total_loss_t = self.total_loss_t + output["loss_total"].detach()
        # VQ loss at pyramid level
        self.vq_loss_t = self.vq_loss_t + output.get("loss_vq", torch.zeros((), device=dev)).detach()
        # Reconstruction/entropy losses are now per-level under comp_outs; fall back to
        # 0 if omitted in a given batch (we still report 0 to surface gaps).
        self.avg_reconstruction_loss_t = (
            self.avg_reconstruction_loss_t
            + (
                output.get("loss_digit_ce", torch.zeros((), device=dev))
                + output.get("loss_special_ce", torch.zeros((), device=dev))
            ).detach()
        )
        self.count += 1
        self.non_padded_tokens_t = self.non_padded_tokens_t + (~key_padding_mask).sum()

        for k, v in output.get("reconstruction_loss_details", {}).items():
            prev = self.reconstruction_loss_details.get(k)
            self.reconstruction_loss_details[k] = (
                prev if isinstance(prev, torch.Tensor) else torch.zeros((), device=dev)
            ) + v.detach()

        # Entropy model loss – aggregated below from comp_outs

        if self.top_lm_enabled and "avg_top_code_lm_loss" in output:
            self.avg_top_code_lm_loss_t = self.avg_top_code_lm_loss_t + output["avg_top_code_lm_loss"].detach()
            for k, v in output.get("top_code_lm_loss_details", {}).items():
                prev = self.top_code_lm_loss_details.get(k)
                self.top_code_lm_loss_details[k] = (
                    prev if isinstance(prev, torch.Tensor) else torch.zeros((), device=dev)
                ) + v.detach()
                if k == "top_code_mse":
                    self.top_code_mse_t = self.top_code_mse_t + v.detach()
                elif k == "top_code_vq_loss":
                    self.top_code_vq_loss_t = self.top_code_vq_loss_t + v.detach()

        # New structure: aggregate from per-level comp_outs
        if "comp_outs" in output:
            comp_outs = output["comp_outs"]
            for lvl, co in enumerate(comp_outs):
                # Reconstruction components per level
                if isinstance(co, dict):
                    if "loss_digit_ce" in co:
                        self.avg_reconstruction_loss_t = self.avg_reconstruction_loss_t + co["loss_digit_ce"].detach()
                    if "loss_special_ce" in co:
                        self.avg_reconstruction_loss_t = self.avg_reconstruction_loss_t + co["loss_special_ce"].detach()
                    # Entropy model loss (entropy CE)
                    if "loss_entropy_ce" in co:
                        if self.avg_entropy_loss_t is None:
                            self.avg_entropy_loss_t = torch.zeros((), device=dev, dtype=torch.float32)
                        self.avg_entropy_loss_t = self.avg_entropy_loss_t + co["loss_entropy_ce"].detach()

                    # Compression lengths from the compressor output
                    comp = co.get("comp_out")
                    if comp is not None:
                        if self.input_seq_lengths_t[lvl] is None:
                            self.input_seq_lengths_t[lvl] = torch.zeros((), device=dev, dtype=torch.float32)
                        if self.output_seq_lengths_t[lvl] is None:
                            self.output_seq_lengths_t[lvl] = torch.zeros((), device=dev, dtype=torch.float32)
                        if getattr(comp, "input_padding_mask", None) is not None:
                            self.input_seq_lengths_t[lvl] = (
                                self.input_seq_lengths_t[lvl] + (~comp.input_padding_mask).sum()
                            )
                        elif getattr(comp, "input_sequence", None) is not None:
                            # fallback: count full length if no mask provided
                            self.input_seq_lengths_t[lvl] = self.input_seq_lengths_t[
                                lvl
                            ] + comp.input_sequence.new_full((), comp.input_sequence.size(1)).to(torch.long)
                        if getattr(comp, "patch_end_mask", None) is not None:
                            self.output_seq_lengths_t[lvl] = self.output_seq_lengths_t[lvl] + comp.patch_end_mask.sum()
                        self._saw_compression = True

                        # Count top-LM effective positions from the top level's valid_mask
                        if self.top_lm_enabled and lvl == (self.num_levels - 1):
                            vm = getattr(comp, "valid_mask", None)
                            if vm is not None:
                                # Per sample top length Ŝ; effective positions = max(Ŝ - 1, 0)
                                eff_batch = (vm.sum(dim=1) - 1).clamp(min=0).sum()
                                if self.top_lm_positions_t is None:
                                    self.top_lm_positions_t = torch.zeros((), device=dev, dtype=torch.float32)
                                self.top_lm_positions_t = self.top_lm_positions_t + eff_batch.to(torch.float32)

        if "all_codebook_perplexities" in output:
            self._saw_perplexities = True
            for lvl, ppl in enumerate(output["all_codebook_perplexities"]):
                val = ppl if isinstance(ppl, torch.Tensor) else torch.tensor(float(ppl), device=dev)
                if self.all_codebook_perplexities_t[lvl] is None:
                    self.all_codebook_perplexities_t[lvl] = torch.zeros((), device=dev, dtype=torch.float32)
                self.all_codebook_perplexities_t[lvl] = self.all_codebook_perplexities_t[lvl] + val.detach()
                # Accumulate raw sum; apply EMA only at log time to avoid per‑batch syncs
                if self._ppl_sum_t[lvl] is None:
                    self._ppl_sum_t[lvl] = torch.zeros((), device=dev, dtype=torch.float32)
                self._ppl_sum_t[lvl] = self._ppl_sum_t[lvl] + val.detach()

    # -------- Window summarization --------
    def throughput(self, duration_sec: float) -> WindowThroughput:
        # Convert GPU scalars to CPU exactly once per window
        tokens_processed = int(self.non_padded_tokens_t.item()) if self.non_padded_tokens_t is not None else 0
        tokens_per_second = (tokens_processed / duration_sec) if duration_sec > 0 else 0.0
        self._tok_s_deque.append(tokens_per_second)
        avg_tok_s = sum(self._tok_s_deque) / len(self._tok_s_deque) if self._tok_s_deque else 0.0

        patches_processed_per_level: list[float] = []
        patches_per_sec: list[float] = []
        for lvl in range(self.num_levels):
            lvl_count = (
                float(self.output_seq_lengths_t[lvl].item()) if self.output_seq_lengths_t[lvl] is not None else 0.0
            )
            patches_processed_per_level.append(lvl_count)
            patches_per_sec.append((lvl_count / duration_sec) if duration_sec > 0 else 0.0)
            self.total_patches_processed_per_level[lvl] += lvl_count

        self.total_bytes_processed += tokens_processed
        # Top-LM positions (window + rate + totals)
        positions = int(self.top_lm_positions_t.item()) if self.top_lm_positions_t is not None else 0
        self._top_lm_positions_per_sec = (positions / duration_sec) if duration_sec > 0 else 0.0
        self.total_top_lm_positions += positions

        return WindowThroughput(
            tokens_processed=tokens_processed,
            tokens_per_second=tokens_per_second,
            avg_tokens_per_second=avg_tok_s,
            patches_processed_per_level=patches_processed_per_level,
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

        # gather the scalar tensors once
        scalars = [
            self.total_loss_t,
            self.avg_reconstruction_loss_t,
            self.vq_loss_t,
            (
                self.avg_entropy_loss_t
                if self.avg_entropy_loss_t is not None
                else torch.zeros((), device=self.total_loss_t.device)
            ),
            (self.avg_top_code_lm_loss_t if self.top_lm_enabled else torch.zeros((), device=self.total_loss_t.device)),
        ]
        vals = torch.stack(scalars).to(torch.float32) / max(1, self.count)
        loss_total, reco, vq, entropy_loss, top = vals.detach().cpu().tolist()

        parts = [
            f"{timestamp}",
            f"Epoch {epoch}/{num_epochs}",
            f"OptStep {global_step}",
            f"MB {minibatch_index}/{total_minibatches}",
            f"Loss {loss_total:.4f}",
            f"Reco {reco:.4f}",
            f"VQ {vq:.4f}",
            f"Tok/s {short_num(avg_tok_s)}",
            f"Bytes {short_num(self.total_bytes_processed)}",
            f"ETA {format_duration(total_eta_sec)}",
        ]
        # Include entropy-model loss only when enabled
        if self.entropy_enabled:
            parts.append(f"Entropy {entropy_loss:.4f}")

        if self.top_lm_enabled:
            if self.count > 0:
                parts.append(f"TopLM {(self.avg_top_code_lm_loss_t / self.count).item():.4f}")
            if self.top_code_mse_t is not None and float(self.top_code_mse_t.item()) != 0.0:
                parts.append(f"TopMSE {(self.top_code_mse_t / self.count).item():.4f}")
            if self.top_code_vq_loss_t is not None and float(self.top_code_vq_loss_t.item()) != 0.0:
                parts.append(f"TopVQ {(self.top_code_vq_loss_t / self.count).item():.4f}")
            # Top-level byte CE flowing through predicted top rows (if provided)
            def _maybe_append(name: str, label: str, fmt: str = ".3f") -> None:
                try:
                    if name in self.top_code_lm_loss_details:
                        val = (self.top_code_lm_loss_details[name] / max(1, self.count)).item()
                        parts.append(f"{label} {val:{fmt}}")
                except Exception:
                    pass

            _maybe_append("top_code_byte_ce", "TopCE", fmt=".4f")
            # Optional: Top-level code accuracy if provided
            if "top_code_acc" in self.top_code_lm_loss_details:
                try:
                    acc_avg = (self.top_code_lm_loss_details["top_code_acc"] / max(1, self.count)).item()
                    parts.append(f"TopAcc {acc_avg:.3f}")
                except Exception:
                    pass
            # Optional diagnostics (if provided by the model)

            # All-positions TopK only (no last-step fallback)
            _maybe_append("top_code_acc_top5_stage0", "TopK5")
            _maybe_append("top_code_acc_top10_stage0", "TopK10")
            _maybe_append("top_code_center_agree_stage0", "Center0")
            _maybe_append("top_code_margin_D", "MarginD")
            _maybe_append("top_code_mse_to_target_center_D", "MSE2C", fmt=".4f")
            # Effective positions seen by the top LM (teacher-forced)
            parts.append(
                f"TopSeen {short_num(self.total_top_lm_positions)} @ {short_num(self._top_lm_positions_per_sec)}/s"
            )

        # Patches summary (skip level 0 in console as before)
        patch_log_parts = []
        for lvl in range(1, self.num_levels):
            label = "Top" if lvl == self.num_levels - 1 else f"L{lvl}"
            pps_rate = patches_per_second[lvl] if lvl < len(patches_per_second) else 0.0
            pps_total = self.total_patches_processed_per_level[lvl]
            patch_log_parts.append(f"{label} {short_num(pps_rate)}/s {short_num(pps_total)}")
        if patch_log_parts:
            parts.append("Patches " + ", ".join(patch_log_parts))

        if self._saw_compression and len(self.output_seq_lengths_t) == self.num_levels and self.count > 0:
            ratios = []
            for lvl in range(self.num_levels):
                in_len = self.input_seq_lengths_t[lvl] if self.input_seq_lengths_t[lvl] is not None else None
                out_len = self.output_seq_lengths_t[lvl] if self.output_seq_lengths_t[lvl] is not None else None
                if in_len is None or out_len is None:
                    ratios.append("0.00")
                else:
                    ratios.append(f"{(out_len / (in_len + 1e-9)).item():.2f}")
            ratios_str = ", ".join(ratios)
            parts.append(f"Ratios [{ratios_str}]")

        # Update smoothed PPL from window averages (one sync per level per log event)
        if self._saw_perplexities:
            for lvl in range(self.num_levels):
                if self._ppl_sum_t[lvl] is not None and self.count > 0:
                    avg_val = (self._ppl_sum_t[lvl] / self.count).item()
                    self._ppl_ema.update(lvl, avg_val)
        ppl_str = ", ".join(
            f"{self._ppl_ema.values[lvl]:.4f}" if self._ppl_ema.ready[lvl] else "n/a" for lvl in range(self.num_levels)
        )
        parts.append(f"SmoothPPL [{ppl_str}]")

        for k, v in self.reconstruction_loss_details.items():
            if isinstance(v, torch.Tensor):
                parts.append(f"{k}:{(v / self.count).item():.4f}")
            else:
                parts.append(f"{k}:{float(v) / self.count:.4f}")
        for k, v in self.entropy_loss_details.items():
            if isinstance(v, torch.Tensor):
                parts.append(f"{k}:{(v / self.count).item():.4f}")
            else:
                parts.append(f"{k}:{float(v) / self.count:.4f}")

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

        zero_dev = torch.zeros((), device=self.total_loss_t.device)
        md: Dict[str, float] = {
            "loss/total_avg_accum": float((self.total_loss_t / self.count).item()),
            "loss/vq_avg_accum": float((self.vq_loss_t / self.count).item()),
            "loss/reconstruction_avg_accum": float((self.avg_reconstruction_loss_t / self.count).item()),
            "loss/entropy_model_avg_accum": float(
                ((self.avg_entropy_loss_t if self.avg_entropy_loss_t is not None else zero_dev) / max(1, self.count)).item()
            ),
            "performance/tokens_per_sec": tokens_per_second,
            "loss/top_code_lm_avg_accum": float((self.avg_top_code_lm_loss_t / self.count).item())
            if self.top_lm_enabled
            else 0.0,
            "loss/top_code_mse_avg_accum": float((self.top_code_mse_t / self.count).item())
            if self.top_lm_enabled
            else 0.0,
            "loss/top_code_vq_avg_accum": float((self.top_code_vq_loss_t / self.count).item())
            if self.top_lm_enabled
            else 0.0,
            "learning_rate": learning_rate,
        }

        # Top-LM effective positions (window rate, total, and window avg per batch)
        md["top_lm/effective_positions_per_sec"] = self._top_lm_positions_per_sec if self.top_lm_enabled else 0.0
        md["top_lm/effective_positions_total"] = float(self.total_top_lm_positions) if self.top_lm_enabled else 0.0
        md["top_lm/effective_positions_avg_accum"] = float(
            (
                (self.top_lm_positions_t if self.top_lm_positions_t is not None else zero_dev) / max(1, self.count)
            ).item()
        ) if self.top_lm_enabled else 0.0

        # Top-LM accuracy average (per-batch average across the window) if provided
        if self.top_lm_enabled and "top_code_acc" in self.top_code_lm_loss_details:
            try:
                md["top_lm/accuracy_avg_accum"] = float(
                    (self.top_code_lm_loss_details["top_code_acc"] / max(1, self.count)).item()
                )
            except Exception:
                md["top_lm/accuracy_avg_accum"] = 0.0

        # Per-level patches performance (skip level 0 as in original)
        for lvl in range(1, self.num_levels):
            md[f"performance/patches_per_sec_L{lvl}"] = patches_per_second[lvl]
            md[f"performance/patches_total_L{lvl}"] = self.total_patches_processed_per_level[lvl]

        for k, v in self.top_code_lm_loss_details.items():
            md[f"loss_detail_avg_accum/{k}"] = float(
                ((v if isinstance(v, torch.Tensor) else torch.tensor(v)) / self.count).item()
            )
        for k, v in self.reconstruction_loss_details.items():
            md[f"loss_detail_avg_accum/{k}"] = float(
                ((v if isinstance(v, torch.Tensor) else torch.tensor(v)) / self.count).item()
            )
        # Entropy model loss details (if any) are included under loss_detail_avg_accum/
        if self.avg_entropy_loss_t is not None:
            for k, v in self.entropy_loss_details.items():
                md[f"loss_detail_avg_accum/{k}"] = float(
                    ((v if isinstance(v, torch.Tensor) else torch.tensor(v)) / self.count).item()
                )

        if self._saw_compression:
            for lvl in range(self.num_levels):
                in_len_t = self.input_seq_lengths_t[lvl]
                out_len_t = self.output_seq_lengths_t[lvl]
                in_avg = float(((in_len_t / self.count) if in_len_t is not None else torch.tensor(0.0)).item())
                out_avg = float(((out_len_t / self.count) if out_len_t is not None else torch.tensor(0.0)).item())
                ratio_avg = float(
                    (
                        (out_len_t / (in_len_t + 1e-9))
                        if (in_len_t is not None and out_len_t is not None)
                        else torch.tensor(0.0)
                    ).item()
                )
                md[f"compression_avg/ratio_L{lvl}"] = ratio_avg
                md[f"compression_avg/input_len_L{lvl}"] = in_avg
                md[f"compression_avg/output_len_L{lvl}"] = out_avg

        if self._saw_perplexities:
            for lvl in range(self.num_levels):
                ppl_t = self.all_codebook_perplexities_t[lvl]
                ppl_avg = float(((ppl_t / self.count) if ppl_t is not None else torch.tensor(0.0)).item())
                md[f"vq_metrics_avg/perplexity_L{lvl}"] = ppl_avg
                if codebook_sizes is not None:
                    md[f"vq_metrics/codebook_size_L{lvl}"] = int(codebook_sizes[lvl])
        for lvl in range(self.num_levels):
            if self._ppl_ema.ready[lvl]:
                md[f"vq_metrics_ema/smooth_perplexity_L{lvl}"] = self._ppl_ema.values[lvl]

        return md


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
