from __future__ import annotations
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import AbstractinatorConfig
from .byte_segment_compressor import ByteSegmentCompressor, CompressorOutput
from .vector_quantizer import MultiStageResidualVQ, ComposedIndexCodec
from .expander import DecoderOnlyExpanderRVQ

class Abstractinator(nn.Module):
    """
    One-level unit that:
      1) Compresses a token stream into segment codes via ByteSegmentCompressor (with MS-RVQ)
      2) Models those codes autoregressively conditioned on "memory" using DecoderOnlyExpanderRVQ,
         reusing the *same* MS-RVQ codebooks (no giant D×K_eff embeddings).
    """

    def __init__(
            self,
            cfg: AbstractinatorConfig,
            lo_vq: Optional[MultiStageResidualVQ] = None):
        super().__init__()
        self.cfg = cfg

        # ---- 1) Compressor (produces vq_embeddings, vq_indices, seg_id, masks, etc.)
        self.compressor = ByteSegmentCompressor(
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
            codebook_size=cfg.c_vq_K,
            vq_depth=cfg.c_vq_depth,
            vq_d_c=cfg.c_vq_d_c,
            beta=cfg.c_vq_beta,
            vq_reset_interval=cfg.c_vq_reset_interval,
            entropy_delta=cfg.c_entropy_delta,
            entropy_abs_threshold=cfg.c_entropy_abs_threshold,
            output_length=cfg.c_output_length,
        )

        # ---- 2) Expander
        self.lo_vq: Optional[MultiStageResidualVQ] = lo_vq
        self.hi_vq: MultiStageResidualVQ = self.compressor.vq

        # low side:
        #   - bottom (L0): bytes => lo_vq=None, K_lo=cfg.vocab_size (e.g., 260)
        #   - upper (L>0): child level codes => lo_vq=<prev_level.vq>
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
            hi_vq=self.hi_vq,  # None at bottom level; non-None for upper levels
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

    def compress(self, token_ids: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> CompressorOutput:
        """
        Just run the compressor.
        Returns the original CompressorOutput with segment codes/embeddings.
        """
        return self.compressor(token_ids, key_padding_mask=key_padding_mask)


    def uncompress(
        self,
        memory: torch.Tensor,                     # (B,S_hi,D) or (B,S_hi) if discrete & hi_vq provided
        *,
        target_low: Optional[torch.Tensor] = None,    # teacher-forced path when provided
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
        teacher_digits, _ = self.lo_codec.decompose(codes_lo)      # list of (B,L)

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
        token_ids: torch.Tensor,                            # (B,S_in) low stream
        key_padding_mask: Optional[torch.Tensor] = None,    # (B,S_in) True==PAD
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

        # ---- 1) Compress ----
        comp: CompressorOutput = self.compressor(token_ids, key_padding_mask=key_padding_mask)

        # hi memory for decoder (continuous)
        memory = comp.vq_embeddings                          # (B, Ŝ*L, D) when num_queries>1; (B,Ŝ,D) when L==1
        # src KPM (True==pad) over memory rows
        if comp.valid_mask is not None:
            if self.compressor.num_queries_per_segment == 1:
                src_kpm = ~comp.valid_mask                   # (B, Ŝ)
            else:
                # Expand per-query; simplest policy = all queries are valid where segment is valid
                Lq = self.compressor.num_queries_per_segment
                src_kpm = ~comp.valid_mask.repeat_interleave(Lq, dim=1)  # (B, Ŝ*Lq)
        else:
            src_kpm = None

        # # ---- 2) Build targets for low stream (teacher forcing), with optional EOP insertion ----
        # tgt = comp.input_sequence.clone()                    # (B, S_in)
        # tgt_kpm = comp.input_padding_mask.clone() if comp.input_padding_mask is not None else None
        # seg_ids = comp.seg_id.clone()                        # (B, S_in) maps each token to a segment id
        #
        # # if insert_eop and (comp.patch_end_mask is not None) and (tgt_kpm is not None):
        # eop_tgt, eop_tgt_kpm, eop_seg_ids = _insert_eop_fixed_len_with_seg(
        #     seq=tgt,
        #     end_mask=comp.patch_end_mask,
        #     kpm=tgt_kpm,
        #     seg=seg_ids,
        #     eop_id=cfg.eop_id,
        #     pad_id=cfg.pad_id,
        # )

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
        stage_logits: List[torch.Tensor] = logits["stage_logits"]      # len=depth, each (B,L,K)
        special_logits: Optional[torch.Tensor] = logits.get("special_logits")

        # ---- 4) Losses ----
        ce = _factorized_code_ce(
            stage_logits=stage_logits,
            special_logits=special_logits,
            targets=tgt,
            codec=self.lo_codec,
            tgt_kpm=tgt_kpm,
        )
        digit_ce = ce["digit_ce"]
        # special_ce = ce["special_ce"]

        # byte-LM CE (entropy branch)
        byte_lm_ce = torch.zeros((), device=dev)
        if cfg.w_byte_lm_ce > 0.0 and comp.entropy_model_logits.size(1) >= 2:
            lm_logits = comp.entropy_model_logits[:, :-1, :]          # predict next low token
            lm_target = comp.input_sequence[:, 1:]
            lm_kpm = comp.input_padding_mask[:, 1:] if comp.input_padding_mask is not None else None
            per_tok = F.cross_entropy(lm_logits.transpose(1, 2), lm_target, reduction="none")
            if lm_kpm is not None:
                m = ~lm_kpm
                byte_lm_ce = (per_tok * m).sum() / m.sum().clamp(min=1)
            else:
                byte_lm_ce = per_tok.mean()

        vq_loss = comp.vq_loss

        total = (
            cfg.w_code_ce * digit_ce
            + cfg.w_byte_lm_ce * byte_lm_ce
            + cfg.w_vq * vq_loss
        )

        # ---- 5) Metrics (compression) ----
        stats = _compression_stats(comp, key_padding_mask, self.compressor.num_queries_per_segment)

        out: Dict[str, torch.Tensor] = {
            "comp_out": comp,
            "loss_total": total,
            "loss_digit_ce": digit_ce.detach(),
            # "loss_special_ce": special_ce.detach(),
            "loss_byte_lm_ce": byte_lm_ce.detach(),
            "loss_vq": vq_loss.detach(),
            "ppl_vq": comp.vq_perplexity.detach(),
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
    dev = comp.vq_indices.device
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
    digits, is_special = codec.decompose(targets)   # list[(B,L)], (B,L)

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


# @torch._dynamo.disable()
# @torch.no_grad()
# def _insert_eop_fixed_len_with_seg(
#     seq: torch.Tensor,              # (B,S)
#     end_mask: torch.Tensor,         # (B,S) True at last token of segment
#     kpm: torch.Tensor,              # (B,S) True==PAD
#     seg: torch.Tensor,              # (B,S) segment id per token (0..Ŝ-1)
#     eop_id: int,
#     pad_id: int,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Length-preserving EOP insertion *and* seg-id update, without relying on PAD budget.
#     Always inserts an EOP after every kept segment-end token. If there is not enough
#     space, later elements are truncated. New EOP tokens inherit the segment id of the
#     token they follow.
#     """
#     import torch.nn.functional as F
#
#     B, S = seq.shape
#     device = seq.device
#
#     # Align end_mask to sequence length
#     if end_mask.size(1) != S:
#         end_mask = end_mask[:, :S] if end_mask.size(1) > S else F.pad(end_mask, (0, S - end_mask.size(1)), value=False)
#
#     real = ~kpm                                # real (non-PAD) tokens
#     end_mask = end_mask & real                 # only real tokens can end segments
#
#     # "Units" accounting: a normal token costs 1, an end-of-seg token costs 2 (token + EOP)
#     units = real.to(torch.int64) + end_mask.to(torch.int64)          # (B,S) in {0,1,2}
#     cum_units = torch.cumsum(units, dim=1)                            # inclusive prefix of units
#
#     # Greedy keep-from-left: include token i iff all items up to *and including* its EOP fit
#     keep_tok = real & (cum_units <= S)                                 # (B,S) bool
#
#     # Among kept tokens, EOPs are exactly those with end_mask
#     keep_eop = keep_tok & end_mask                                     # (B,S) bool
#
#     # Compute output positions using only the kept units:
#     # For token i, its output pos = (# of kept units strictly before i).
#     units_kept = keep_tok.to(torch.int64) + keep_eop.to(torch.int64)   # 1 or 2 at kept tokens, else 0
#     cum_kept = torch.cumsum(units_kept, dim=1)                          # inclusive
#     pos_tok = cum_kept - units_kept                                     # token lands here
#     pos_eop = pos_tok + 1                                               # eop lands immediately after its token
#
#     # Prepare outputs
#     out = seq.new_full((B, S), pad_id)
#     out_kpm = torch.ones_like(kpm, dtype=torch.bool)  # True==PAD
#     out_seg = torch.zeros_like(seg)
#
#     # Scatter kept tokens
#     if keep_tok.any():
#         batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(seq)
#         out[batch_idx[keep_tok], pos_tok[keep_tok]] = seq[keep_tok]
#         out_kpm[batch_idx[keep_tok], pos_tok[keep_tok]] = False
#         out_seg[batch_idx[keep_tok], pos_tok[keep_tok]] = seg[keep_tok]
#
#     # Scatter their EOPs (always adjacent; guaranteed in-range by the keep rule)
#     if keep_eop.any():
#         batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(seq)
#         out[batch_idx[keep_eop], pos_eop[keep_eop]] = eop_id
#         out_kpm[batch_idx[keep_eop], pos_eop[keep_eop]] = False
#         out_seg[batch_idx[keep_eop], pos_eop[keep_eop]] = seg[keep_eop]  # inherit seg id
#
#     # Fill seg ids for remaining PADs with last known seg id (safe filler)
#     last_seg = seg.masked_fill(kpm, -1).max(dim=1).values.clamp(min=0)  # (B,)
#     pad_mask = out_kpm
#     if pad_mask.any():
#         out_seg[pad_mask] = last_seg.unsqueeze(1).expand_as(out_seg)[pad_mask]
#
#     return out, out_kpm, out_seg
