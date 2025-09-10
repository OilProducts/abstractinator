from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from components.config_types import AbstractinatorConfig

from .expander import DecoderOnlyExpanderRVQ, DecoderExpander
from .segment_compressor import CompressorOutput, SegmentCompressor
from .vector_quantizer import ComposedIndexCodec, MultiStageResidualVQ, VectorQuantizer


class Abstractinator(nn.Module):
    """
    One-level unit that:
      1) Compresses a token stream into segment codes via SegmentCompressor (with MS-RVQ)
      2) Models those codes autoregressively conditioned on "memory" using DecoderOnlyExpanderRVQ,
         reusing the *same* MS-RVQ codebooks (no giant D×K_eff embeddings).
    """

    def __init__(self, cfg: AbstractinatorConfig, lo_vq: Optional[MultiStageResidualVQ] = None):
        super().__init__()
        self.cfg = cfg
        self._logger = logging.getLogger(__name__)

        # ---- 1) Compressor (produces vq_embeddings, vq_indices, seg_id, masks, etc.)
        # Optional high‑side quantizer to bottleneck segment memory.
        # Support both MultiStageResidualVQ (default) and standard VectorQuantizer.
        if bool(getattr(cfg, "c_use_standard_vq", False)):
            # Standard VQ in D-space, single stage
            self.hi_quantizer = VectorQuantizer(
                K=int(cfg.c_vq_K),
                D=int(cfg.D),
                beta=float(cfg.c_vq_beta),
                reset_interval=int(cfg.c_vq_reset_interval),
                bos_token_id=int(cfg.bos_id),
                eos_token_id=int(cfg.eos_id),
                padding_token_id=int(cfg.pad_id),
            )
        else:
            # Multi-stage residual VQ
            vq_d_c = cfg.c_vq_d_c if cfg.c_vq_d_c is not None else 64
            self.hi_quantizer = MultiStageResidualVQ(
                K=int(cfg.c_vq_K),
                D=int(cfg.D),
                depth=int(cfg.c_vq_depth),
                d_c=int(vq_d_c),
                beta=float(cfg.c_vq_beta),
                reset_interval=int(cfg.c_vq_reset_interval),
                bos_token_id=int(cfg.bos_id),
                eos_token_id=int(cfg.eos_id),
                padding_token_id=int(cfg.pad_id),
            )

        self.compressor = SegmentCompressor(
            vocab_size=cfg.vocab_size,
            dim=cfg.D,
            heads=cfg.c_heads,
            window=cfg.c_window,
            head_dim=cfg.c_head_dim,
            kv_comp_dim=cfg.c_kv_comp_dim,
            q_comp_dim=cfg.c_q_comp_dim,
            retr_dim=cfg.c_retr_dim,
            num_encoder_layers=cfg.c_num_encoder_layers,
            num_shared_encoder_layers=cfg.c_num_shared_encoder_layers,
            num_entropy_encoder_layers=cfg.c_num_entropy_encoder_layers,
            num_compression_encoder_layers=cfg.c_num_compression_encoder_layers,
            num_queries=cfg.c_num_queries,
            entropy_delta=cfg.c_entropy_delta,
            entropy_abs_threshold=cfg.c_entropy_abs_threshold,
            output_length=cfg.c_output_length,
            quantizer=self.hi_quantizer,
            attention_config=cfg.compressor_attention,
            entropy_config=cfg.c_entropy_config,
        )

        # Optionally preload and freeze entropy stack (embedding + shared + entropy)
        if getattr(cfg, "c_entropy_load_path", None):
            try:
                from .checkpoint_utils import load_entropy_stack

                # Inspect checkpoint for saved config and shapes to warn on mismatches
                try:
                    ckpt = torch.load(cfg.c_entropy_load_path, map_location="cpu", weights_only=False)
                    saved_cfg = ckpt.get("entropy_config", None)
                    emb_sd = ckpt.get("embedding", None)
                    ent_sd = ckpt.get("entropy_model", None)

                    saved_D = None
                    if isinstance(emb_sd, dict):
                        w = emb_sd.get("weight", None)
                        if isinstance(w, torch.Tensor) and w.ndim == 2:
                            saved_D = int(w.shape[1])

                    def _infer_layers(d: dict | None, prefix: str) -> int:
                        if not isinstance(d, dict):
                            return 0
                        seen = set()
                        for k in d.keys():
                            if not k.startswith(prefix):
                                continue
                            parts = k[len(prefix) :].lstrip(".").split(".")
                            if parts and parts[0].isdigit():
                                seen.add(int(parts[0]))
                        return (max(seen) + 1) if seen else 0

                    saved_layers = _infer_layers(ent_sd, prefix="layers")

                    # Current model settings
                    cur_D = int(self.cfg.D)
                    try:
                        cur_layers = len(getattr(self.compressor.entropy_model, "layers", []))
                    except Exception:
                        cur_layers = 0
                    try:
                        cur_heads = getattr(self.compressor.entropy_model.cfg, "n_heads", None)
                        cur_window = getattr(self.compressor.entropy_model.cfg, "window", None)
                    except Exception:
                        cur_heads = None
                        cur_window = None

                    # Saved settings from checkpoint config, if available
                    saved_heads = saved_cfg.get("n_heads") if isinstance(saved_cfg, dict) else None
                    saved_window = saved_cfg.get("window") if isinstance(saved_cfg, dict) else None
                    saved_n_layers = saved_cfg.get("n_layers") if isinstance(saved_cfg, dict) else None

                    msgs = []
                    if saved_D is not None and saved_D != cur_D:
                        msgs.append(f"D: ckpt={saved_D} vs level={cur_D}")
                    if saved_n_layers is not None and cur_layers and (int(saved_n_layers) != int(cur_layers)):
                        msgs.append(f"layers: ckpt={saved_n_layers} vs level={cur_layers}")
                    elif saved_layers and cur_layers and (int(saved_layers) != int(cur_layers)):
                        msgs.append(f"layers(inferred): ckpt={saved_layers} vs level={cur_layers}")
                    if (saved_heads is not None) and (cur_heads is not None) and (int(saved_heads) != int(cur_heads)):
                        msgs.append(f"heads: ckpt={saved_heads} vs level={cur_heads}")
                    if (saved_window is not None) and (cur_window is not None) and (int(saved_window) != int(cur_window)):
                        msgs.append(f"window: ckpt={saved_window} vs level={cur_window}")

                    if msgs:
                        self._logger.warning(
                            "Entropy checkpoint config mismatch for level: %s | %s",
                            cfg.c_entropy_load_path,
                            "; ".join(msgs),
                        )
                    else:
                        self._logger.info(
                            "Entropy checkpoint appears compatible (D=%s, layers≈%s).",
                            saved_D if saved_D is not None else cur_D,
                            saved_n_layers if saved_n_layers is not None else (saved_layers or cur_layers),
                        )
                except Exception:
                    self._logger.debug("Could not inspect entropy checkpoint at %s", cfg.c_entropy_load_path)

                map_loc = getattr(cfg, "device", None) or "cpu"
                load_entropy_stack(
                    self.compressor,
                    cfg.c_entropy_load_path,
                    freeze=bool(getattr(cfg, "c_entropy_freeze", True)),
                    map_location=map_loc,
                )
                self._logger.info(
                    "Loaded entropy stack for level from %s (freeze=%s)",
                    cfg.c_entropy_load_path,
                    bool(getattr(cfg, "c_entropy_freeze", True)),
                )
            except Exception:
                self._logger.exception("Failed to load entropy stack from %s", cfg.c_entropy_load_path)

        # ---- 2) Expander
        self.lo_vq: Optional[MultiStageResidualVQ] = lo_vq

        # Choose decoder head implementation
        if lo_vq is None and bool(getattr(cfg, "d_use_standard_vq", False)):
            # Bottom-level: standard VQ classifier head over K=byte vocab
            self.dec_vq = VectorQuantizer(
                K=int(cfg.vocab_size),
                D=int(cfg.D),
                beta=float(cfg.d_vq_beta),
                ema=bool(cfg.d_vq_ema),
                decay=float(cfg.d_vq_decay),
                reset_interval=int(cfg.d_vq_reset_interval),
                max_codes_to_reset_pct=float(cfg.d_vq_max_codes_to_reset_pct),
                replacement_buffer_size=int(cfg.d_vq_replacement_buffer_size),
                vectors_per_step_to_buffer=int(cfg.d_vq_vectors_per_step_to_buffer),
                bos_token_id=int(cfg.bos_id),
                eos_token_id=int(cfg.eos_id),
                padding_token_id=int(cfg.pad_id),
            )
            self.expander = DecoderExpander(
                vq_lo=self.dec_vq,
                D=int(cfg.D),
                N_dec=int(cfg.d_layers),
                H=int(cfg.d_heads),
                cross_window=int(cfg.d_cross_window),
                eos_id=int(cfg.eos_id),
                max_len=int(cfg.d_max_len),
                use_sqdist_logits=bool(cfg.d_use_sqdist_logits),
                device=getattr(cfg, "device", None),
                self_attn_config=cfg.decoder_self_attention,
                cross_attn_config=cfg.decoder_cross_attention,
            )
            # No ComposedIndexCodec in this path
            self.lo_codec = None
        else:
            # Default RVQ-based expander (factorized digits)
            if lo_vq is None:
                K_lo = cfg.vocab_size
                eos_id = cfg.eos_id
            else:
                K_lo = None
                eos_id = int(lo_vq.eos_token_id)

            self.expander = DecoderOnlyExpanderRVQ(
                lo_vq=self.lo_vq,
                hi_vq=None,
                K_lo=K_lo,
                D=cfg.D,
                N_dec=cfg.d_layers,
                H=cfg.d_heads,
                cross_window=cfg.d_cross_window,
                eos_id=eos_id,
                max_len=cfg.d_max_len,
                lo_d_c=cfg.d_lo_d_c,
                residual_conditioning=cfg.d_residual_conditioning,
                use_sqdist_logits=cfg.d_use_sqdist_logits,
                device=cfg.device,
                self_attn_config=cfg.decoder_self_attention,
                cross_attn_config=cfg.decoder_cross_attention,
            )

            # codec for the low side (factorized digits or degenerate depth=1)
            if lo_vq is not None:
                self.lo_codec = lo_vq.codec
            else:
                self.lo_codec = ComposedIndexCodec(
                    K=cfg.vocab_size,
                    depth=1,
                    bos=cfg.bos_id,
                    eos=cfg.eos_id,
                    pad=cfg.pad_id,
                )

        # Register useful ids
        self.eos_id = cfg.eos_id
        
        self.pad_id = cfg.pad_id

        # Optionally freeze the entire Abstractinator (compressor + expander)
        if bool(getattr(cfg, "a_freeze", False)):
            self.requires_grad_(False)
            self.eval()

    # ---------------------------
    # Public API
    # ---------------------------

    def train(self, mode: bool = True) -> "Abstractinator":  # type: ignore[override]
        """
        Override nn.Module.train() to respect frozen submodules.

        Any module whose parameters are all frozen (requires_grad == False)
        will be kept in eval() mode even when the parent is switched to train().
        If the entire Abstractinator is frozen, the whole module is kept in
        eval() regardless of the requested mode.
        """

        def _all_params_frozen(mod: nn.Module) -> bool:
            has_param = False
            for p in mod.parameters(recurse=True):
                has_param = True
                if p.requires_grad:
                    return False
            return has_param  # True only if it has params and all are frozen

        # If turning training off, defer to default behavior
        if not mode:
            super().train(False)
            return self

        # If this entire module is frozen, keep it (and children) in eval
        if _all_params_frozen(self):
            super().train(False)
            return self

        # Otherwise, enable training as usual first
        super().train(True)

        # Then force any fully-frozen submodules back to eval
        for m in self.modules():
            if m is self:
                continue
            if _all_params_frozen(m):
                m.eval()

        return self

    @torch.no_grad()
    def segment_only(
        self, token_ids: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper: entropy-based segmentation using the compressor's entropy_model
        branch, with embeddings computed at this level. Returns (seg_id, patch_end_mask),
        both (B, S).
        """
        return self.compressor.segment_only_ids(token_ids, key_padding_mask=key_padding_mask)

    def compress(self, token_ids: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> CompressorOutput:
        """
        Just run the compressor.
        Returns the original CompressorOutput with segment codes/embeddings.
        """
        return self.compressor.forward_ids(token_ids, key_padding_mask=key_padding_mask)

    def uncompress(
        self,
        memory: torch.Tensor,  # (B,S_hi,D) or (B,S_hi) if discrete & hi_vq provided
        *,
        target_low: Optional[torch.Tensor] = None,  # teacher-forced path when provided
        seg_ids: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        seed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inverse of compress (per level):
          - if target_low is given → teacher-forced logits (+ optional losses if you compute them here)
          - else → generate low tokens autoregressively (greedy)
        """
        if target_low is not None:
            logits = self.expander(
                codes_hi=memory,
                codes_lo=target_low,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                seg_ids=seg_ids,
            )
            return {"stage_logits": logits["stage_logits"]}
        else:
            if seed is None:
                seed = torch.full((memory.size(0), 1), self.eos_id, dtype=torch.long, device=memory.device)
            out = self.expander.generate(
                codes_hi=memory,
                codes_lo=seed,
                src_key_padding_mask=src_key_padding_mask,
                max_new_tokens=max_new_tokens,
                seg_ids=seg_ids,
            )
            return {"generated": out}

    def decode_logits(
        self,
        memory: torch.Tensor,  # (B, S_hi, D) *continuous* memory OR (B, S_hi) discrete if hi_vq provided
        codes_lo: torch.Tensor,  # (B, L) composed low codes to teacher-force
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, S_hi) True where memory is pad
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True where target token is pad
        seg_ids: Optional[torch.Tensor] = None,  # (B, L) mapping each low token to a memory segment index
    ) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
        """
        Run the expander to obtain stage-wise logits (and optional special head logits).
        """
        return self.expander(
            codes_hi=memory,
            codes_lo=codes_lo,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            seg_ids=seg_ids,
        )

    def compute_code_loss(
        self,
        logits: Dict[str, List[torch.Tensor] | torch.Tensor],  # from decode_logits()
        codes_lo: torch.Tensor,  # (B, L)
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage-wise cross-entropy on digits, + optional special-head CE.
        Masks out specials for digit CE, and uses specials only for special CE.
        """
        stage_logits: List[torch.Tensor] = logits["stage_logits"]  # list of (B, L, K)

        # If no codec (standard VQ head), use single-stage CE directly
        if getattr(self, "lo_codec", None) is None or len(stage_logits) == 1:
            lg = stage_logits[0]
            if tgt_key_padding_mask is None:
                ce = F.cross_entropy(lg.transpose(1, 2), codes_lo, reduction="mean")
            else:
                ce_all = F.cross_entropy(lg.transpose(1, 2), codes_lo, reduction="none")  # (B,L)
                valid = ~tgt_key_padding_mask
                denom = valid.sum().clamp(min=1)
                ce = (ce_all * valid).sum() / denom
            return {"digit_ce": ce, "special_ce": torch.zeros((), device=codes_lo.device)}

        # Factorized RVQ path (uses codec to decompose digits)
        teacher_digits, _ = self.lo_codec.decompose(codes_lo)  # list of (B,L)
        valid = torch.ones_like(codes_lo, dtype=torch.bool)
        if tgt_key_padding_mask is not None:
            valid &= ~tgt_key_padding_mask
        denom = valid.sum().clamp(min=1)
        digit_loss = codes_lo.new_zeros([], dtype=torch.float32)
        for s, lg in enumerate(stage_logits):
            ce = F.cross_entropy(lg.transpose(1, 2), teacher_digits[s], reduction="none")  # (B,L)
            digit_loss = digit_loss + (ce * valid).sum() / denom
        return {"digit_ce": digit_loss, "special_ce": torch.zeros((), device=codes_lo.device)}

    def forward(
        self,
        token_ids: torch.Tensor,  # (B,S_in) low stream
        key_padding_mask: Optional[torch.Tensor] = None,  # (B,S_in) True==PAD
        *,
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for one level:
          1) compress low stream → hi memory (continuous)
          3) decode (teacher-forced) with seg_ids + masks
          4) losses: factorized digit CE (+special), VQ, optional entropy-model CE
          5) report compression metrics
        """
        cfg = self.cfg
        token_ids.size(0)

        # ---- 1) Compress ----
        comp: CompressorOutput = self.compressor.forward_ids(token_ids, key_padding_mask=key_padding_mask)

        # hi memory for decoder (continuous)
        # Prefer continuous quantized memory (z_hat). Falls back to pre‑VQ embeddings if needed.
        memory = comp.vq_embeddings if comp.vq_embeddings is not None else comp.pre_vq_embeddings
        # src KPM (True==pad) over memory rows
        if comp.valid_mask is not None:
            if self.compressor.num_queries_per_segment == 1:
                src_kpm = ~comp.valid_mask  # (B, Ŝ)
            else:
                # Expand per-query; simplest policy = all queries are valid where segment is valid
                Lq = self.compressor.num_queries_per_segment
                src_kpm = ~comp.valid_mask.repeat_interleave(Lq, dim=1)  # (B, Ŝ*Lq)
        else:
            src_kpm = None

        # If the entire level is frozen, skip decoder-loss computation and zero losses.
        if cfg.a_freeze:
            ce = torch.zeros((), device=memory.device)
            entropy_loss = torch.zeros((), device=memory.device)
            vq_loss = torch.zeros((), device=memory.device)
            total = torch.zeros((), device=memory.device)
            include_entropy = False
            tgt = None  # suppress return_logits extras
            stage_logits = None
        else:
            # ---- 2) Targets ----
            # tgt = comp.input_sequence  # (B, S_in)
            tgt = comp.hidden_states
            tgt_kpm = comp.input_padding_mask  # (B, S_in) or None
            seg_ids = comp.seg_id  # (B, S_in)

            # If memory has multiple query rows per segment but seg_ids is per-segment,
            # map tokens to the *first* query row of each segment. (Custom policies welcome.)
            if self.compressor.num_queries_per_segment > 1:
                Lq = self.compressor.num_queries_per_segment
                seg_ids = seg_ids * Lq  # point to query row 0 for each segment

            # Safety: clip seg_ids to memory length
            if memory.dim() == 3:
                max_row = memory.size(1) - 1
                seg_ids = seg_ids.clamp(min=0, max=max_row)

            # ---- 3) Decode (teacher forcing) ----
            logits = self.expander(
                codes_hi=memory,
                codes_lo=tgt,
                src_key_padding_mask=src_kpm,
                tgt_key_padding_mask=tgt_kpm,
                seg_ids=seg_ids,
            )
            stage_logits: List[torch.Tensor] = logits["stage_logits"]  # len=depth, each (B,L,K)
            logits.get("special_logits")

            # Standard VQ head: next-token prediction (shifted)
            if getattr(self, "lo_codec", None) is None or len(stage_logits) == 1:
                lg = stage_logits[0]  # (B, L, K)
                tgt_full = comp.input_sequence  # (B, L)
                if tgt_full.numel() == 0 or lg.size(1) <= 1:
                    ce = torch.zeros((), device=memory.device)
                else:
                    # predict token t+1 from positions up to t
                    logits_used = lg[:, :-1, :]  # (B, L-1, K)
                    targets_used = tgt_full[:, 1:]  # (B, L-1)
                    if comp.input_padding_mask is None:
                        ce = F.cross_entropy(logits_used.transpose(1, 2), targets_used, reduction="mean")
                    else:
                        valid = ~comp.input_padding_mask[:, 1:]  # (B, L-1)
                        denom = valid.sum().clamp(min=1)
                        ce_all = F.cross_entropy(
                            logits_used.transpose(1, 2), targets_used, reduction="none"
                        )  # (B, L-1)
                        ce = (ce_all * valid).sum() / denom
            else:
                # Factorized RVQ path (unchanged here; handled via digits in compute_code_loss if needed)
                ce = F.cross_entropy(stage_logits[0].transpose(1, 2), comp.input_sequence)

            vq_loss = comp.vq_loss if comp.vq_loss is not None else torch.zeros((), device=memory.device)

            # If the entropy stack is loaded and configured as frozen, exclude its
            # loss from the objective and logging using config only (fast path).
            frozen_by_cfg = bool(getattr(cfg, "c_entropy_load_path", None)) and bool(
                getattr(cfg, "c_entropy_freeze", True)
            )
            include_entropy = (not frozen_by_cfg) and (float(cfg.w_entropy_ce) != 0.0)
            entropy_loss = comp.entropy_loss if comp.entropy_loss is not None else torch.zeros((), device=memory.device)

            total = cfg.w_code_ce * ce + (cfg.w_entropy_ce if include_entropy else 0.0) * entropy_loss + cfg.w_vq * vq_loss

        # ---- 5) Metrics (compression) ----
        stats = _compression_stats(comp, key_padding_mask, self.compressor.num_queries_per_segment)

        out: Dict[str, torch.Tensor] = {
            "comp_out": comp,
            "loss_total": total,
            "loss_digit_ce": ce.detach(),
            # Only report entropy-model loss when that stack is trainable; else 0
            "loss_entropy_ce": (entropy_loss.detach() if include_entropy else torch.zeros((), device=memory.device)),
            **stats,
        }
        if return_logits and (tgt is not None) and (stage_logits is not None):
            out["codes_lo_targets"] = tgt.detach()
            out["stage_logits_last"] = stage_logits[-1].detach()
        return out

    @torch.no_grad()
    def encode_for_next_level(
        self,
        comp: CompressorOutput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the ingredients typically fed into the *next* level:
          codes_hi_discrete: (B, S_hat) composed indices (lo_vq output)
          cont_hi_memory   : (B, S_hat, D) quantized continuous embeddings
          src_keypad_mask  : (B, S_hat) True where memory row is padding
          token_to_seg_ids : (B, S_in) seg id per input token (for cross mapping)
        """
        assert self.cfg.c_num_queries == 1, (
            "If num_queries>1, decide how you map tokens to query rows (e.g., first of L)."
        )
        codes_hi = comp.vq_indices  # (B, S_hat)
        cont_hi = comp.vq_embeddings  # (B, S_hat, D)
        src_kpm = ~comp.valid_mask  # (B, S_hat) True where padded
        return codes_hi, cont_hi, src_kpm, comp.seg_id

    @torch.no_grad()
    def generate_codes(
        self,
        hi_memory: torch.Tensor,  # (B, S_hi, D) continuous OR (B, S_hi) discrete
        seed_lo: torch.Tensor,  # (B, T0) seed codes (composed indices)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, S_hi)
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # compat; not used during generation
        max_new_tokens: Optional[int] = None,
        seg_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation that mirrors training inputs:

        - For the standard VQ decoder head (DecoderExpander), use the compressor's
          hidden states as the decoder inputs (continuous [B,T,D]), identical to
          teacher-forced training.
        - For the RVQ-based expander (DecoderOnlyExpanderRVQ), fall back to its
          internal generate() implementation for now.
        
        Returns only the newly generated tokens (B, steps).
        """
        # Fast path for RVQ expander: keep existing implementation
        if isinstance(self.expander, DecoderOnlyExpanderRVQ):
            return self.expander.generate(
                codes_hi=hi_memory,
                codes_lo=seed_lo,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                max_new_tokens=max_new_tokens,
                seg_ids=seg_ids,
            )

        # Standard VQ head: use hidden states from the compressor as decoder inputs
        assert isinstance(self.expander, DecoderExpander)

        device = hi_memory.device
        B = hi_memory.size(0)

        generated = seed_lo.clone()  # (B, T)
        kpm = (
            tgt_key_padding_mask.clone()
            if tgt_key_padding_mask is not None
            else torch.zeros_like(generated, dtype=torch.bool)
        )

        # steps budget: how many new tokens to produce
        max_len_hint = getattr(self.expander, "max_len", None)
        steps_budget = int(max_new_tokens) if max_new_tokens is not None else int((max_len_hint or 0) or 0)
        if steps_budget == 0:
            # If no explicit budget and no model max_len hint, default to 1 step
            steps_budget = 1

        def _align_seg_ids(curr_T: int) -> torch.Tensor:
            if seg_ids is None:
                # Default: map all queries to the last memory row (like training when padding)
                last_seg = hi_memory.size(1) - 1 if hi_memory.dim() >= 2 else 0
                return torch.full((B, curr_T), last_seg, dtype=torch.long, device=device)
            if seg_ids.size(1) < curr_T:
                pad = seg_ids[:, -1:].expand(-1, curr_T - seg_ids.size(1))
                return torch.cat([seg_ids, pad], dim=1)
            if seg_ids.size(1) > curr_T:
                return seg_ids[:, :curr_T]
            return seg_ids

        new_tokens = torch.zeros((B, 0), dtype=generated.dtype, device=device)
        for _ in range(steps_budget):
            T = generated.size(1)

            # Prepare decoder inputs as compressor hidden states, mirroring training
            inp_emb = self.compressor.embedding(generated)  # (B, T, D)
            h_shared = self.compressor._run_shared(inp_emb, kpm)
            dec_inp = self.compressor._run_compression(h_shared, kpm)  # (B, T, D)

            seg_ids_cur = _align_seg_ids(T)  # (B, T)

            # Forward once to obtain logits over the next id at position T-1
            out = self.expander(
                codes_hi=hi_memory,
                codes_lo=dec_inp,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=kpm,
                seg_ids=seg_ids_cur,
            )

            lg = out["stage_logits"][-1]  # (B, T, K)
            next_id = lg[:, -1:, :].argmax(dim=-1)  # (B,1)

            generated = torch.cat([generated, next_id], dim=1)
            kpm = F.pad(kpm, (0, 1), value=False)
            new_tokens = torch.cat([new_tokens, next_id], dim=1)

            # EOS early stop if defined on expander
            eos_id = getattr(self.expander, "eos_id", getattr(self, "eos_id", None))
            if eos_id is not None and (next_id == int(eos_id)).all():
                break

        return new_tokens


@torch.no_grad()
def _compression_stats(
    comp: CompressorOutput, input_kpm: Optional[torch.Tensor], num_queries: int
) -> Dict[str, torch.Tensor]:
    """
    Report batch-mean compression ratio and a few simple stats.
    ratio = (valid_segments * num_queries) / real_input_tokens
    """
    dev = comp.pre_vq_embeddings.device
    if input_kpm is not None:
        in_len = (~input_kpm).sum(dim=1).float()  # (B,)
    else:
        in_len = torch.full((comp.vq_indices.size(0),), comp.input_sequence.size(1), device=dev, dtype=torch.float32)

    if comp.valid_mask is not None:
        out_len = (comp.valid_mask.sum(dim=1) * max(1, num_queries)).float()  # (B,)
    else:
        out_len = torch.full_like(in_len, comp.vq_indices.size(1), dtype=torch.float32)

    ratio = out_len / (in_len + 1e-9)  # (B,)
    stats = {
        "compression_ratio_mean": ratio.mean(),
        "compression_ratio_min": ratio.min(),
        "compression_ratio_max": ratio.max(),
        "segments_per_seq_mean": comp.valid_mask.sum(dim=1).float().mean()
        if comp.valid_mask is not None
        else torch.tensor(0.0, device=dev),
    }
    return stats


def _factorized_code_ce(stage_logits, special_logits, targets, codec, tgt_kpm):
    digits, is_special = codec.decompose(targets)  # list[(B,L)], (B,L)

    # If no special head, we train specials via digits too
    mask_specials_in_digits = special_logits is not None

    valid = torch.ones_like(is_special, dtype=torch.bool)
    if tgt_kpm is not None:
        valid &= ~tgt_kpm
    if mask_specials_in_digits:
        valid &= ~is_special

    denom = valid.sum().clamp(min=1)
    digit_loss = torch.zeros((), device=targets.device)
    for s, lg in enumerate(stage_logits):
        ce = F.cross_entropy(lg.transpose(1, 2), digits[s], reduction="none")
        digit_loss = digit_loss + (ce * valid).sum() / denom

    special_loss = torch.zeros((), device=targets.device)
    if special_logits is not None:
        sp_local = codec.special_local_index(targets)
        sp_mask = is_special & (valid if tgt_kpm is None else ~tgt_kpm)
        if sp_mask.any():
            ce_sp = F.cross_entropy(special_logits.transpose(1, 2), sp_local, reduction="none")
            special_loss = (ce_sp * sp_mask).sum() / sp_mask.sum().clamp(min=1)

    return {"digit_ce": digit_loss, "special_ce": special_loss}
