# abstractinator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import AbstractinatorConfig
# Import from your project
# If these live under a package, adjust the imports accordingly.
from .byte_segment_compressor import ByteSegmentCompressor, CompressorOutput
from .vector_quantizer import MultiStageResidualVQ, ComposedIndexCodec
from .expander import DecoderOnlyExpanderRVQ  # your DecoderOnlyExpanderRVQ


# ---------------------------
# Abstractinator (one level)
# ---------------------------

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
            predict_specials=cfg.d_predict_specials,
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

    # @torch.no_grad()
    def compress(self, token_ids: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> CompressorOutput:
        """
        Just run the compressor.
        Returns the original CompressorOutput with segment codes/embeddings.
        """
        return self.compressor(token_ids, key_padding_mask=key_padding_mask)

    # def uncompress(self,
    #                memory: torch.Tensor,
    #                target_lo: torch.Tensor,
    #                seg_ids: Optional[torch.Tensor] = None,
    #                src_key_padding_mask: Optional[torch.Tensor] = None,
    #                tgt_key_padding_mask: Optional[torch.Tensor] = None
    #                ):
    #
    #     """
    #     Inverse of compress.
    #     """
    #     logits = self.expander(codes_hi=memory,
    #                            codes_lo=target_lo,
    #                            src_key_padding_mask=src_key_padding_mask,
    #                            tgt_key_padding_mask=tgt_key_padding_mask,
    #                            seg_ids=seg_ids)
    #     return  {"stage_logits": logits["stage_logits"],
    #              "special_logits": logits.get("special_logits", None)}



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
            return {"stage_logits": logits["stage_logits"], "special_logits": logits.get("special_logits")}
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
        special_logits: Optional[torch.Tensor] = logits.get("special_logits")  # (B, L, S) or None

        # Teacher digits and special mask
        teacher_digits, special_mask = self.lo_codec.decompose(codes_lo)  # list of (B,L), and (B,L) bool
        B, L = codes_lo.shape
        valid_mask = torch.ones(B, L, dtype=torch.bool, device=codes_lo.device)
        if tgt_key_padding_mask is not None:
            valid_mask &= ~tgt_key_padding_mask

        # Digit CE: ignore positions that are special or padded
        not_special = ~special_mask & valid_mask
        digit_loss_sum = 0.0
        denom = 0
        with torch.autocast(device_type=codes_lo.device.type, enabled=False) if hasattr(torch,
                                                                                        'autocast') else _nullcontext():
            for s, logits_s in enumerate(stage_logits):
                # (B, L, K) -> (B, K, L)
                ce_s = F.cross_entropy(
                    logits_s.transpose(1, 2),  # (B, K, L)
                    teacher_digits[s],  # (B, L)
                    reduction="none",
                )  # (B, L)
                if not_special.dtype != torch.bool:
                    not_special = not_special.bool()
                ce_s = ce_s * not_special.float()
                digit_loss_sum = digit_loss_sum + ce_s.sum()
                denom = denom + not_special.sum()

        digit_ce = digit_loss_sum / denom.clamp(min=1)

        # Special CE (only where special)
        special_ce = torch.zeros((), device=codes_lo.device)
        if special_logits is not None:
            # Map specials to local 0..S-1 head indices
            target_sp = self.lo_codec.special_local_index(codes_lo)  # (B, L)
            mask_sp = special_mask & valid_mask
            ce_sp = F.cross_entropy(
                special_logits.transpose(1, 2),  # (B, S, L)
                target_sp,
                reduction="none",
            )  # (B, L)
            special_ce = (ce_sp * mask_sp.float()).sum() / mask_sp.sum().clamp(min=1)

        return {"digit_ce": digit_ce, "special_ce": special_ce}




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

        # ---- 2) Build targets for low stream (teacher forcing), with optional EOP insertion ----
        tgt = comp.input_sequence.clone()                    # (B, S_in)
        tgt_kpm = comp.input_padding_mask.clone() if comp.input_padding_mask is not None else None
        seg_ids = comp.seg_id.clone()                        # (B, S_in) maps each token to a segment id

        # if insert_eop and (comp.patch_end_mask is not None) and (tgt_kpm is not None):
        eop_tgt, eop_tgt_kpm, eop_seg_ids = _insert_eop_fixed_len_with_seg(
            seq=tgt,
            end_mask=comp.patch_end_mask,
            kpm=tgt_kpm,
            seg=seg_ids,
            eop_id=cfg.eop_id,
            pad_id=cfg.pad_id,
        )

        # If memory has multiple query rows per segment but seg_ids is per-segment,
        # map tokens to the *first* query row of each segment. (Custom policies welcome.)
        if self.compressor.num_queries_per_segment > 1:
            Lq = self.compressor.num_queries_per_segment
            seg_ids = seg_ids * Lq  # point to query row 0 for each segment

        # Safety: clip seg_ids to memory length
        if memory.dim() == 3:
            max_row = memory.size(1) - 1
            seg_ids = seg_ids.clamp_(min=0, max=max_row)

        # ---- 3) Decode (teacher forcing) ----
        logits = self.expander(
            codes_hi=memory,
            codes_lo=eop_tgt,
            src_key_padding_mask=src_kpm,
            tgt_key_padding_mask=eop_tgt_kpm,
            seg_ids=eop_seg_ids,
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
        special_ce = ce["special_ce"]

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
            + cfg.w_special_ce * special_ce
            + cfg.w_byte_lm_ce * byte_lm_ce
            + cfg.w_vq * vq_loss
        )

        # ---- 5) Metrics (compression) ----
        stats = _compression_stats(comp, key_padding_mask, self.compressor.num_queries_per_segment)

        out: Dict[str, torch.Tensor] = {
            "comp_out": comp,
            "loss_total": total,
            "loss_digit_ce": digit_ce.detach(),
            "loss_special_ce": special_ce.detach(),
            "loss_byte_lm_ce": byte_lm_ce.detach(),
            "loss_vq": vq_loss.detach(),
            "ppl_vq": comp.vq_perplexity.detach(),
            **stats,
        }
        if return_logits:
            out["codes_lo_targets"] = tgt.detach()
            out["stage_logits_last"] = stage_logits[-1].detach()
        return out



    # def forward(
    #         self,
    #         token_ids: torch.Tensor,  # (B, S_in) input tokens at this level
    #         hi_memory: Optional[torch.Tensor] = None,
    #         # (B, S_hi, D) continuous or (B, S_hi) discrete if hi_vq used in expander
    #         key_padding_mask: Optional[torch.Tensor] = None,  # (B, S_in) True where padded
    #         compute_byte_lm_ce: Optional[bool] = None,  # override cfg.w_byte_lm_ce>0?
    #         stop_grad_through_memory: bool = True,
    #         return_logits: bool = False,
    #         insert_eop: bool = False,  # insert EOP token at segment ends
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     End-to-end training step for one level:
    #       - compress -> (codes_lo, cont_hi_mem, seg_id, masks)
    #       - decode (teacher forced) -> stage-wise logits
    #       - losses: code CE (+ specials), optional byte LM CE, VQ loss
    #     If hi_memory is provided, it overrides using local continuous memory for decoding.
    #     """
    #
    #     cfg = self.cfg
    #     comp: CompressorOutput = self.compressor(token_ids, key_padding_mask=key_padding_mask)
    #
    #     # TODO: experiment with pre_vq_embeddings
    #     memory = comp.vq_embeddings
    #     src_kpm = (~comp.valid_mask) if (comp.valid_mask is not None) else None
    #
    #
    #     # Low side to predict:
    #     codes_lo = comp.vq_indices  # (B, L_lo)
    #     tgt_kpm = key_padding_mask
    #     L_lo = codes_lo.size(1)
    #
    #     if insert_eop and (comp.patch_end_mask is not None) and (tgt_kpm is not None):
    #         eop_id = self.compressor.vq.eop_token_id  # for this level's low side
    #         pad_id = self.compressor.vq.padding_token_id
    #         codes_lo, tgt_kpm = insert_eop_fixed_len(
    #             seq=codes_lo,
    #             end_mask=comp.patch_end_mask,  # True at segment ends
    #             kpm=tgt_kpm,
    #             eop_id=eop_id,
    #             pad_id=pad_id,
    #         )
    #
    #     # Memory for decoder:
    #     #   - If hi_memory is given: use it directly.
    #     #   - Else: use *continuous* segment embeddings from this level as memory.
    #     #     We also pass seg_ids mapping each low token to its segment index.
    #     memory = hi_memory if hi_memory is not None else comp.vq_embeddings  # (B, S_hi, D)
    #     # Build memory KeyPaddingMask: True where memory row is pad/invalid.
    #
    #     if stop_grad_through_memory:
    #         memory = memory.detach()
    #     src_kpm = (~comp.valid_mask) if (comp.valid_mask is not None) else None
    #     seg_ids = comp.seg_id[:, :codes_lo.size(1)]  # (B, Llo)
    #
    #     if hi_memory is not None:
    #         if hi_memory.is_floating_point():
    #             # caller must provide src_key_padding_mask; fall back to "all real"
    #             src_kpm = None
    #         else:
    #             # Discrete memory: treat PAD id as padded
    #             src_kpm = (hi_memory == self.pad_id)
    #     else:
    #         # Using our own continuous memory; mask comes from valid segments (per segment, not per query)
    #         # NOTE: this assumes num_queries==1. If L>1, expand comp.valid_mask to query rows accordingly.
    #         assert self.cfg.c_num_queries == 1, "If num_queries>1, provide seg_ids and memory mask consistent with memory rows."
    #         src_kpm = ~comp.valid_mask  # (B, S_hat) True where padded segment slot
    #
    #     # Target KeyPaddingMask for low tokens:
    #     tgt_kpm = (token_ids == self.pad_id) if key_padding_mask is None else key_padding_mask
    #     # For the code targets, also mask positions where codes are special PAD
    #     tgt_kpm_codes = tgt_kpm | (codes_lo == self.pad_id)
    #
    #     # Segment mapping: (B, L_lo) tells decoder which memory row each low token belongs to.
    #     seg_ids = comp.seg_id  # (B, S_in)
    #     # If your compressor trimmed/reshaped lengths, ensure seg_ids matches codes_lo length (teacher forcing range).
    #     seg_ids = seg_ids[:, :L_lo]
    #
    #     # Decoder forward (teacher-forced residual conditioning)
    #     logits = self.decode_logits(
    #         memory=memory,
    #         codes_lo=codes_lo,
    #         src_key_padding_mask=src_kpm,
    #         tgt_key_padding_mask=tgt_kpm_codes,
    #         seg_ids=seg_ids,
    #     )
    #
    #     # Stage-wise & specials CE
    #     ce_dict = self.compute_code_loss(logits, codes_lo, tgt_key_padding_mask=tgt_kpm_codes)
    #     digit_ce = ce_dict["digit_ce"]
    #     special_ce = ce_dict["special_ce"]
    #
    #     # Optional byte LM CE (for entropy branch only; typical weight small or zero)
    #     use_lm = (self.cfg.w_byte_lm_ce > 0.0) if compute_byte_lm_ce is None else compute_byte_lm_ce
    #     byte_lm_ce = torch.zeros((), device=token_ids.device)
    #     if use_lm:
    #         byte_logits = comp.entropy_model_logits[:, :L_lo, :]  # (B, L, V)
    #         byte_targets = token_ids[:, :L_lo]  # (B, L)
    #         # Standard next-token CE: expander already does shift-on-input; we do not here.
    #         # Feel free to shift if your LM branch is trained with teacher forcing requiring EOS shift.
    #         # Mask padded tokens
    #         lm_mask = ~tgt_kpm
    #         ce = F.cross_entropy(byte_logits.transpose(1, 2), byte_targets, reduction="none")
    #         byte_lm_ce = (ce * lm_mask.float()).sum() / lm_mask.sum().clamp(min=1)
    #
    #     # VQ loss (scalar from compressor; already aggregated across stages)
    #     vq_loss = comp.vq_loss
    #
    #     # Total
    #     total = (
    #             self.cfg.w_code_ce * digit_ce
    #             + self.cfg.w_special_ce * special_ce
    #             + self.cfg.w_byte_lm_ce * byte_lm_ce
    #             + self.cfg.w_vq * vq_loss
    #     )
    #
    #     out: Dict[str, torch.Tensor] = {
    #         "loss_total": total,
    #         "loss_digit_ce": digit_ce.detach(),
    #         "loss_special_ce": special_ce.detach(),
    #         "loss_byte_lm_ce": byte_lm_ce.detach(),
    #         "loss_vq": vq_loss.detach(),
    #         "ppl_vq": comp.vq_perplexity.detach(),
    #     }
    #
    #     if return_logits:
    #         # Only small tensors into the dict to avoid autograd bloat;
    #         # callers can recompute logits via decode_logits if needed.
    #         out["codes_lo"] = codes_lo.detach()
    #         out["codes_hi_cont"] = memory.detach() if memory.is_floating_point() else None
    #         out["seg_ids"] = seg_ids.detach()
    #     return out

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


def _factorized_code_ce(
    stage_logits: List[torch.Tensor],               # len=depth, each (B,L,K)
    special_logits: Optional[torch.Tensor],         # (B,L,S) or None
    targets: torch.Tensor,                          # (B,L)
    codec: Optional[ComposedIndexCodec],            # None at bottom (depth=1 learned adapter)
    tgt_kpm: Optional[torch.Tensor],                # (B,L) True==PAD
) -> Dict[str, torch.Tensor]:
    """
    If codec is None (bottom, depth=1), treat as single stage CE.
    Otherwise, decompose into digits and sum CE over stages; specials via tiny head if provided.
    """
    device = targets.device
    if codec is None:
        # learned single-stage adapter case (bottom)
        lg = stage_logits[0]                        # (B,L,K)
        ce = F.cross_entropy(lg.transpose(1, 2), targets, reduction="none")  # (B,L)
        valid = ~tgt_kpm if tgt_kpm is not None else torch.ones_like(targets, dtype=torch.bool)
        denom = valid.sum().clamp(min=1)
        return {"digit_ce": (ce * valid).sum() / denom, "special_ce": torch.zeros((), device=device)}

    digits, is_special = codec.decompose(targets)   # list[(B,L)], (B,L)
    valid = ~is_special
    if tgt_kpm is not None:
        valid &= ~tgt_kpm
    denom = valid.sum().clamp(min=1)

    digit_loss = torch.zeros((), device=device)
    for s, lg in enumerate(stage_logits):
        ce = F.cross_entropy(lg.transpose(1, 2), digits[s], reduction="none")  # (B,L)
        digit_loss = digit_loss + (ce * valid).sum() / denom

    special_loss = torch.zeros((), device=device)
    if special_logits is not None:
        sp_local = codec.special_local_index(targets)
        sp_mask = is_special & (~tgt_kpm if tgt_kpm is not None else True)
        if sp_mask.any():
            ce_sp = F.cross_entropy(special_logits.transpose(1, 2), sp_local, reduction="none")
            special_loss = (ce_sp * sp_mask).sum() / sp_mask.sum().clamp(min=1)

    return {"digit_ce": digit_loss, "special_ce": special_loss}

@torch._dynamo.disable()
@torch.no_grad()
def _insert_eop_fixed_len_with_seg(
    seq: torch.Tensor,              # (B,S)
    end_mask: torch.Tensor,         # (B,S) True at last token of segment
    kpm: torch.Tensor,              # (B,S) True==PAD
    seg: torch.Tensor,              # (B,S) segment id per token (0..Ŝ-1)
    eop_id: int,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Length-preserving EOP insertion *and* seg-id update.
    New EOP tokens inherit the segment id of the token they follow.
    """
    B, S = seq.shape
    device = seq.device

    # align end_mask length
    if end_mask.size(1) != S:
        end_mask = end_mask[:, :S] if end_mask.size(1) > S else F.pad(end_mask, (0, S - end_mask.size(1)), value=False)

    # budget: how many PAD slots to convert to EOP per row
    pad_slots = kpm.sum(dim=1)                      # (B,)
    csum = torch.cumsum(end_mask.to(torch.int64), dim=1)
    keep = csum <= pad_slots.unsqueeze(1)
    end_mask = end_mask & keep

    # shift map
    m = end_mask.to(torch.int64)
    c = torch.cumsum(m, dim=1)
    shift = c - m

    real = ~kpm
    token_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(seq)

    out = seq.new_full((B, S), pad_id)
    out_kpm = torch.ones_like(kpm, dtype=torch.bool)
    out_seg = torch.zeros_like(seg)

    # move original tokens
    tgt_pos = token_idx + shift
    tgt_flat = tgt_pos[real]
    out[batch_idx[real], tgt_flat] = seq[real]
    out_kpm[batch_idx[real], tgt_flat] = False
    out_seg[batch_idx[real], tgt_flat] = seg[real]

    # place EOPs
    need_eop = real & end_mask
    eop_pos = token_idx[need_eop] + shift[need_eop] + 1
    out[batch_idx[need_eop], eop_pos] = eop_id
    out_kpm[batch_idx[need_eop], eop_pos] = False
    out_seg[batch_idx[need_eop], eop_pos] = seg[need_eop]  # inherit seg id

    # fill seg ids for remaining PADs with last known seg id (safe filler)
    # compute last seg per row among real tokens
    last_seg = seg.masked_fill(kpm, -1).max(dim=1).values.clamp(min=0)  # (B,)
    pad_mask = out_kpm
    out_seg[pad_mask] = last_seg.unsqueeze(1).expand_as(out_seg)[pad_mask]

    return out, out_kpm, out_seg


# Fallback context manager if autocast not present in your env
class _nullcontext:
    def __enter__(self): return self

    def __exit__(self, *_): return False

#
# @torch.no_grad()
# @torch._dynamo.disable()  # small, fast eager micro-kernel; keep the rest of your model compiled
# def insert_eop_fixed_len(
#         seq: torch.Tensor,  # (B, S) token ids
#         end_mask: torch.Tensor,  # (B, S) True at last token of a segment
#         kpm: torch.Tensor,  # (B, S) True == PAD (required; right-padded rows)
#         eop_id: int,
#         pad_id: int,
# ):
#     """
#     Insert EOPs without changing length S by shifting real tokens right.
#     Sparse: two cumsums + two sparse writes (tokens, EOPs). Very fast.
#
#     Returns:
#       out     : (B, S) with tokens shifted and EOPs inserted
#       out_kpm : (B, S) bool, True == PAD in `out`
#     """
#     B, S = seq.shape
#     if S == 0:
#         # fast path; nothing to do
#         return seq, kpm.clone() if kpm is not None else None
#
#     dev = seq.device
#     itype = seq.dtype
#
#     # ---- Align masks to S (cheap; eager so python branching is fine) ----
#     if end_mask.size(1) != S:
#         end_mask = end_mask[:, :S] if end_mask.size(1) > S else F.pad(end_mask, (0, S - end_mask.size(1)), value=False)
#     if kpm.size(1) != S:
#         kpm = kpm[:, :S] if kpm.size(1) > S else F.pad(kpm, (0, S - kpm.size(1)), value=True)
#
#     # Ensure contiguous (avoids hidden copies in masked_select/index_copy)
#     seq = seq.contiguous()
#     end_mask = end_mask.contiguous()
#     kpm = kpm.contiguous()
#
#     real = (~kpm).bool()  # True where token is real (non-PAD)
#     end_mask &= real  # count ends only on real tokens
#
#     # ---- Budget: prune ends beyond available PAD slots per row ----
#     pad_slots = kpm.sum(dim=1, dtype=torch.int32)  # (B,) int32
#     csum_end = torch.cumsum(end_mask.to(torch.int32), dim=1)  # (B,S) int32
#     keep = csum_end <= pad_slots.unsqueeze(1)  # (B,S) bool
#     end_mask &= keep
#
#     # ---- Exclusive prefix of kept ends: #EOP strictly before each position ----
#     ends32 = end_mask.to(torch.int32)  # (B,S)
#     prev32 = torch.cumsum(ends32, dim=1) - ends32  # (B,S) int32
#
#     # ---- Target positions (int32) ----
#     j32 = torch.arange(S, device=dev, dtype=torch.int32).view(1, S).expand(B, S)
#     idx_tok32 = j32 + prev32  # token dest: i + prev[i]
#     idx_eop32 = idx_tok32 + 1  # eop   dest: token_dest + 1
#
#     # By budget + right-padding: idx_tok32 < S for real, idx_eop32 < S for ends.
#     tok_ok = real
#     eop_ok = end_mask
#
#     # ---- Linearize indices without allocating (B,S) batch_idx ----
#     base32 = (torch.arange(B, device=dev, dtype=torch.int32) * S).view(B, 1).expand(B, S)
#     lin_tok32 = (base32 + idx_tok32)[tok_ok]  # (N_tok,) int32
#     lin_eop32 = (base32 + idx_eop32)[eop_ok]  # (N_eop,) int32
#
#     # ---- Gather the source tokens to write ----
#     src_tok = seq[tok_ok]  # (N_tok,)
#
#     # ---- Allocate outputs (prefill PADs) ----
#     out = seq.new_full((B, S), pad_id)  # (B,S)
#     out_flat = out.view(-1)
#     out_kpm = torch.ones_like(kpm, dtype=torch.bool)
#     out_kpm_f = out_kpm.view(-1)
#
#     # ---- Sparse writes (use the specialized index_* ops) ----
#     # NOTE: indices for PyTorch indexing must be int64
#     lin_tok = lin_tok32.to(torch.long)
#     out_flat.index_copy_(0, lin_tok, src_tok)  # write moved tokens
#     out_kpm_f.index_fill_(0, lin_tok, False)  # mark them as non-PAD
#
#     if lin_eop32.numel():
#         lin_eop = lin_eop32.to(torch.long)
#         out_flat.index_fill_(0, lin_eop, int(eop_id))  # write EOPs (constant)
#         out_kpm_f.index_fill_(0, lin_eop, False)  # mark EOPs as non-PAD
#
#     return out, out_kpm
