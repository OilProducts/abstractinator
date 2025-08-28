from __future__ import annotations
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import AbstractinatorConfig
from .segment_compressor import SegmentCompressor, CompressorOutput
from .vector_quantizer import MultiStageResidualVQ, ComposedIndexCodec
from .expander import DecoderOnlyExpanderRVQ


class Abstractinator(nn.Module):
    """
    One-level unit that:
      1) Compresses a token stream into segment codes via SegmentCompressor (with MS-RVQ)
      2) Models those codes autoregressively conditioned on "memory" using DecoderOnlyExpanderRVQ,
         reusing the *same* MS-RVQ codebooks (no giant D×K_eff embeddings).
    """

    def __init__(
            self,
            cfg: AbstractinatorConfig,
            lo_vq: Optional[MultiStageResidualVQ] = None):
        super().__init__()
        self.cfg = cfg

        # ---- Shared embedding table (moved from SegmentCompressor)
        # Construct the token embedding at the Abstractinator level so it can be
        # owned/shared externally and referenced by the compressor.
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.D)

        # ---- 1) Compressor (produces vq_embeddings, vq_indices, seg_id, masks, etc.)
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
            num_lm_encoder_layers=cfg.c_num_lm_encoder_layers,
            num_compression_encoder_layers=cfg.c_num_compression_encoder_layers,
            num_queries=cfg.c_num_queries,
            entropy_delta=cfg.c_entropy_delta,
            entropy_abs_threshold=cfg.c_entropy_abs_threshold,
            output_length=cfg.c_output_length,
        )
        # Embedding is owned here; SegmentCompressor consumes precomputed embeddings.

        # ---- 2) Expander
        self.lo_vq: Optional[MultiStageResidualVQ] = lo_vq

        if lo_vq is None:
            K_lo = cfg.vocab_size
            eos_id = cfg.eos_id
            eop_id = cfg.eop_id
        else:
            K_lo = None
            eos_id = int(lo_vq.eos_token_id)
            eop_id = int(lo_vq.eop_token_id)

        self.expander = DecoderOnlyExpanderRVQ(
            lo_vq=self.lo_vq,
            hi_vq=None, #self.hi_vq,  # None at bottom level; non-None for upper levels
            K_lo=K_lo,
            D=cfg.D,
            N_dec=cfg.d_layers,
            H=cfg.d_heads,
            cross_window=cfg.d_cross_window,
            eos_id=eos_id,
            eop_id=eop_id,
            max_len=cfg.d_max_len,
            lo_d_c=cfg.d_lo_d_c,
            residual_conditioning=cfg.d_residual_conditioning,
            use_sqdist_logits=cfg.d_use_sqdist_logits,
            device=cfg.device,
        )

        # codec for the low side (factorized digits or degenerate depth=1)
        if lo_vq is not None:
            self.lo_codec = lo_vq.codec
        else:
            # depth=1 byte space; specials in this space are your byte specials
            self.lo_codec = ComposedIndexCodec(
                K=cfg.vocab_size, depth=1,
                bos=cfg.bos_id, eos=cfg.eos_id, pad=cfg.pad_id, eop=cfg.eop_id,
            )

        # Register useful ids
        self.eos_id = cfg.eos_id
        self.eop_id = cfg.eop_id
        self.pad_id = cfg.pad_id

    # ---------------------------
    # Public API
    # ---------------------------

    @torch.no_grad()
    def segment_only(self, token_ids: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper: entropy-based segmentation using the compressor's LM branch,
        with embeddings computed at this level.
        Returns (seg_id, patch_end_mask), both (B, S).
        """
        x = self.embedding(token_ids)
        return self.compressor.segment_only(x, key_padding_mask=key_padding_mask)

    def compress(self, token_ids: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> CompressorOutput:
        """
        Just run the compressor.
        Returns the original CompressorOutput with segment codes/embeddings.
        """
        x = self.embedding(token_ids)
        return self.compressor(x, key_padding_mask=key_padding_mask, token_ids=token_ids)

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
            insert_eop: bool = True,
            return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for one level:
          1) compress low stream → hi memory (continuous)
          2) build EOP-aware low targets (same length)
          3) decode (teacher-forced) with seg_ids + masks
          4) losses: factorized digit CE (+special), VQ, optional byte-LM CE
          5) report compression metrics
        """
        cfg = self.cfg
        B = token_ids.size(0)
        dev = token_ids.device

        embeddings = self.embedding(token_ids)  # (B, S_in, D)

        # ---- 1) Compress ----
        comp: CompressorOutput = self.compressor(embeddings, key_padding_mask=key_padding_mask, token_ids=token_ids)

        # hi memory for decoder (continuous)
        memory = comp.pre_vq_embeddings  # (B, Ŝ*L, D) when num_queries>1; (B,Ŝ,D) when L==1
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

        # ---- 2) Targets: NO EOP insertion ----
        tgt = comp.input_sequence  # (B, S_in)
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
        special_logits: Optional[torch.Tensor] = logits.get("special_logits")

        # standard CE on logits
        ce = F.cross_entropy(stage_logits[0].transpose(1, 2), tgt)

        total = (
                cfg.w_code_ce * ce
                + cfg.w_byte_lm_ce * comp.entropy_loss
                # + cfg.w_vq * vq_loss
        )

        # ---- 5) Metrics (compression) ----
        stats = _compression_stats(comp, key_padding_mask, self.compressor.num_queries_per_segment)

        out: Dict[str, torch.Tensor] = {
            "comp_out": comp,
            "loss_total": total,
            "loss_digit_ce": ce.detach(),
            "loss_byte_lm_ce": comp.entropy_loss.detach(),
            **stats,
        }
        if return_logits:
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
        assert self.cfg.c_num_queries == 1, "If num_queries>1, decide how you map tokens to query rows (e.g., first of L)."
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
        Autoregressive generation in the low code space (composed ids).
        """
        return self.expander.generate(
            codes_hi=hi_memory,
            codes_lo=seed_lo,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            max_new_tokens=max_new_tokens,
            seg_ids=seg_ids,
        )


@torch.no_grad()
def _compression_stats(comp: CompressorOutput, input_kpm: Optional[torch.Tensor], num_queries: int) -> Dict[
    str, torch.Tensor]:
    """
    Report batch-mean compression ratio and a few simple stats.
    ratio = (valid_segments * num_queries) / real_input_tokens
    """
    dev = comp.pre_vq_embeddings.device
    if input_kpm is not None:
        in_len = (~input_kpm).sum(dim=1).float()  # (B,)
    else:
        in_len = torch.full((comp.vq_indices.size(0),), comp.input_sequence.size(1), device=dev,
                            dtype=torch.float32)

    if comp.valid_mask is not None:
        out_len = (comp.valid_mask.sum(dim=1) * max(1, num_queries)).float()  # (B,)
    else:
        out_len = torch.full_like(in_len, comp.vq_indices.size(1), dtype=torch.float32)

    ratio = out_len / (in_len + 1e-9)  # (B,)
    stats = {
        "compression_ratio_mean": ratio.mean(),
        "compression_ratio_min": ratio.min(),
        "compression_ratio_max": ratio.max(),
        "segments_per_seq_mean": comp.valid_mask.sum(
            dim=1).float().mean() if comp.valid_mask is not None else torch.tensor(0.0, device=dev),
    }
    return stats


def _factorized_code_ce(stage_logits, special_logits, targets, codec, tgt_kpm):
    digits, is_special = codec.decompose(targets)  # list[(B,L)], (B,L)

    # If no special head, we train specials via digits too
    mask_specials_in_digits = (special_logits is not None)

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
