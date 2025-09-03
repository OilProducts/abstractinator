from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.base import SegmentContext
from .attention.cache import AttnCache
from .config_types import AttentionConfig
from .swiglu import SwiGLU
from .vector_quantizer import (
    ComposedIndexCodec,
    LearnedCodebookAdapter,
    MultiStageResidualVQ,
    RVQEmbeddingAdapter,
    RVQFactorizedHead,
)


@lru_cache(maxsize=64)
def _cached_causal_mask(length: int) -> torch.Tensor:
    """Return a causal mask computed on CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
    return mask


def get_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Return the cached causal mask on ``device``."""
    return _cached_causal_mask(length).to(device)


class SlidingDecoderBlock(nn.Module):
    """Decoder block using causal self-attention and sliding-window cross attention."""

    def __init__(
        self,
        idx: int,
        d_model: int,
        num_heads: int,
        cross_window: int,
        q_dim,
        kv_dim,
        *,
        cross_attn_config: "AttentionConfig" | None = None,
        self_attn_config: "AttentionConfig" | None = None,
    ):
        super().__init__()
        self.layer_id = f'L{idx}'
        self.norm1 = nn.RMSNorm(d_model)
        from .attention.factory import make_causal_self_block, make_segment_cross_attention

        self.self_attn = make_causal_self_block(
            dim=d_model,
            num_heads=num_heads,
            ffn_dim_multiplier=4,
            cfg=self_attn_config
            or AttentionConfig(
                variant="mla",
                kernel="flex",
                kv_comp_dim=d_model // 16,
                q_comp_dim=d_model // 32,
                retr_dim=d_model // num_heads,
            ),
        )
        self.norm2 = nn.RMSNorm(d_model)
        lookback = (
            cross_attn_config.lookback
            if (cross_attn_config and cross_attn_config.lookback is not None)
            else cross_window
        )
        cfg_cross = cross_attn_config or AttentionConfig(variant="regular", kernel="sdpa", lookback=lookback)
        self.cross_attn = make_segment_cross_attention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            d_attn=d_model,
            n_heads=num_heads,
            lookback=lookback,
            cfg=cfg_cross,
        )
        self.norm3 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, 4 * d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        q_pos_ids: torch.Tensor | None = None,
        kv_pos_ids: torch.Tensor | None = None,
        seg_ids: torch.Tensor | None = None,
        cache: AttnCache | None = None,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        # cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.norm1(x), key_padding_mask=tgt_key_padding_mask
        )  # attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)

        seg_ctx = SegmentContext(
            seg_id=seg_ids,
            q_pos_ids=q_pos_ids if q_pos_ids is not None else torch.arange(x.size(1), device=x.device),
            kv_pos_ids=kv_pos_ids if kv_pos_ids is not None else torch.arange(memory.size(1), device=memory.device),
            lookback=0,
            kv_mask=memory_key_padding_mask,
            q_pad_mask=tgt_key_padding_mask,
        )
        x = x + self.cross_attn(self.norm2(x), memory, segment=seg_ctx)
        x = x + self.ffn(self.norm3(x))
        return x


class SlidingDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        cross_window: int,
        q_dim: int,
        kv_dim: int,
        *,
        cross_attn_config: "AttentionConfig" | None = None,
        self_attn_config: "AttentionConfig" | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for l_idx in range(num_layers):
            layer = SlidingDecoderBlock(
                idx=l_idx,
                d_model=d_model,
                num_heads=num_heads,
                cross_window=cross_window,
                q_dim=q_dim,
                kv_dim=kv_dim,
                cross_attn_config=cross_attn_config,
                self_attn_config=self_attn_config,
            )
            self.layers.append(layer)

        self.final_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        q_pos_ids: torch.Tensor | None = None,
        kv_pos_ids: torch.Tensor | None = None,
        cache: AttnCache | None = None,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        # cross_attn_mask: Optional[torch.Tensor] = None,
        seg_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                memory,
                q_pos_ids=q_pos_ids,
                kv_pos_ids=kv_pos_ids,
                cache=cache,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                # cross_attn_mask=cross_attn_mask,
                seg_ids=seg_ids,
            )
        return self.final_norm(x)


class DecoderOnlyExpanderRVQ(nn.Module):
    """
    Decoder-only expander that:
      - embeds high codes by reusing the *high-level* MS-RVQ codebooks (no emb_hi),
      - embeds low codes likewise (no emb_lo),
      - outputs factorized stage-wise logits (no D×K_eff softmax).
    """

    def __init__(
        self,
        lo_vq: "MultiStageResidualVQ",  # MS-RVQ for the *target* code space
        hi_vq: Optional["MultiStageResidualVQ"],  # MS-RVQ for the *memory* code space
        K_lo: Optional[int] = None,  # if lo_vq is None
        D: int = 256,
        N_dec: int = 4,
        H: int = 8,
        cross_window: int = 128,
        eos_id: int = 257,
        eop_id: int = 259,
        max_len: int = 2048,
        lo_d_c: int = 64,  # Factorized rank for bytes
        residual_conditioning: bool = True,
        use_sqdist_logits: bool = False,
        device: Optional[torch.device] = None,
        *,
        cross_attn_config: Optional[AttentionConfig] = None,
        self_attn_config: Optional[AttentionConfig] = None,
    ):
        super().__init__()
        self.lo_vq = lo_vq
        self.hi_vq = hi_vq
        self.D = D
        self.eos_id = eos_id
        self.eop_id = eop_id
        self.max_len = max_len
        self.K_lo = K_lo

        # Reuse VQ projections and codebooks
        # ---- low side (what we predict) ----
        if lo_vq is not None:
            self.lo_adapt = RVQEmbeddingAdapter(lo_vq, target_D=D, use_shared_proj=True, tie_up_down=True)
            self.lo_codec = ComposedIndexCodec(
                K=lo_vq.K,
                depth=lo_vq.depth,
                bos=lo_vq.bos_token_id,
                eos=lo_vq.eos_token_id,
                pad=lo_vq.padding_token_id,
                eop=lo_vq.eop_token_id,
            )

        else:
            assert K_lo is not None, "Need K_lo for bottom level when lo_vq is None."
            self.lo_adapt = LearnedCodebookAdapter(
                K=K_lo,
                D=D,
                d_c=lo_d_c,
                device=device,
            )
            self.lo_codec = ComposedIndexCodec(K=K_lo, depth=1, bos=-1, eos=-1, pad=-1, eop=-1)

        # ---- high side (memory) ----
        self.hi_adapt = RVQEmbeddingAdapter(hi_vq, target_D=D, use_shared_proj=True) if hi_vq is not None else None

        # Decoder block with configurable attention forms/backends
        self.decoder = SlidingDecoder(
            N_dec,
            D,
            H,
            cross_window,
            q_dim=D,
            kv_dim=D,
            cross_attn_config=cross_attn_config,
            self_attn_config=self_attn_config,
        )

        # Factorized head
        self.head = RVQFactorizedHead(
            self.lo_adapt, residual_conditioning=residual_conditioning, use_sqdist_logits=use_sqdist_logits
        )

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return get_causal_mask(length, device)

    def _embed_memory(self, codes_hi: torch.Tensor) -> torch.Tensor:
        # Either high side is already continuous (D), or embed via hi_vq
        if codes_hi.is_floating_point():
            return codes_hi
        if self.hi_adapt is None:
            raise ValueError("codes_hi is discrete but hi_vq/hi_adapt not provided")
        return self.hi_adapt.embed_composed(codes_hi)

    def _embed_decoder_inputs(self, codes_lo: torch.Tensor) -> torch.Tensor:
        # Shift-right with EOS
        decoder_input_ids = F.pad(codes_lo[:, :-1], (1, 0), value=self.eos_id)
        return self.lo_adapt.embed_composed(decoder_input_ids)

    def forward(
        self,
        codes_hi: torch.Tensor,  # (B, S_hi) or (B, S_hi, D)
        codes_lo: torch.Tensor,  # (B, L)
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        device = codes_hi.device
        memory = self._embed_memory(codes_hi)  # (B, S_hi, D)
        dec_inp = self._embed_decoder_inputs(codes_lo)  # (B, L, D)

        L_q = codes_lo.size(1)
        L_kv = memory.size(1)
        q_pos_ids = torch.arange(L_q, device=device)
        kv_pos_ids = torch.arange(L_kv, device=device)

        tgt_mask = self._causal_mask(codes_lo.size(1), device)
        adjusted_tgt_kpm = None
        if tgt_key_padding_mask is not None:
            adjusted_tgt_kpm = F.pad(tgt_key_padding_mask[:, :-1], (1, 0), value=False)

        h = self.decoder(
            dec_inp,
            memory,
            q_pos_ids=q_pos_ids,
            kv_pos_ids=kv_pos_ids,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=adjusted_tgt_kpm,
            memory_key_padding_mask=src_key_padding_mask,
            seg_ids=seg_ids,
        )  # (B, L, D)

        # Teacher-forced residual conditioning uses *true* digits at (B,L)
        teacher_digits, _ = self.lo_codec.decompose(codes_lo)
        logits = self.head(h, teacher_digits=teacher_digits)  # dict

        return logits  # {"stage_logits": [B,L,K]*depth, "special_logits": (B,L,S) or None}

    @torch.no_grad()
    def generate(
        self,
        codes_hi: torch.Tensor,  # (B, S_hi) or (B, S_hi, D)
        codes_lo: torch.Tensor,  # (B, T0)  seed(s)
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # <- compat
        max_len: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        seg_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = codes_hi.device
        B = codes_hi.size(0)
        memory = self._embed_memory(codes_hi)  # (B, S_hi, D)

        kv_pos_ids = torch.arange(memory.size(1), device=device)

        generated = codes_lo.clone()  # (B, T)
        kpm = (
            tgt_key_padding_mask.clone()
            if tgt_key_padding_mask is not None
            else torch.zeros_like(generated, dtype=torch.bool)
        )

        cache = AttnCache()

        # steps budget: how many *new* tokens we’re allowed to produce
        steps_budget = int(max_new_tokens) if max_new_tokens is not None else int((max_len or self.max_len) - 1)
        steps_budget = max(0, steps_budget)

        # helper: align seg_ids length to current T
        def _align_seg_ids(curr_T: int) -> torch.Tensor:
            if seg_ids is None:
                last_seg = memory.size(1) - 1
                return torch.full((B, curr_T), last_seg, dtype=torch.long, device=device)
            if seg_ids.size(1) < curr_T:
                pad = seg_ids[:, -1:].expand(-1, curr_T - seg_ids.size(1))
                return torch.cat([seg_ids, pad], dim=1)
            if seg_ids.size(1) > curr_T:
                return seg_ids[:, :curr_T]
            return seg_ids

        depth = self.lo_codec.depth  # works for both MS-RVQ and learned single-stage
        new_tokens = torch.zeros((B, 0), dtype=generated.dtype, device=device)  # (B,0)
        for step in range(steps_budget):
            T = generated.size(1)
            q_pos_ids = torch.arange(T, device=device)

            dec_inp = self.lo_adapt.embed_composed(generated)  # (B, T, D)
            tgt_mask = self._causal_mask(T, device)
            seg_ids_cur = _align_seg_ids(T)  # (B, T)

            h = self.decoder(
                dec_inp,
                memory,
                q_pos_ids=q_pos_ids,
                kv_pos_ids=kv_pos_ids,
                cache=cache,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=kpm,
                memory_key_padding_mask=src_key_padding_mask,
                seg_ids=seg_ids_cur,
            )  # (B, T, D)

            last_h = h[:, -1, :]  # (B,1,D)
            r_dc = self.lo_adapt.down(last_h)  # (B,1,d_c)

            # Greedy stage-by-stage with residual updates
            digits = []
            for s in range(depth):
                W = self.lo_adapt.stage_codebook(s)  # (K, d_c)
                logit_s = self.head._stage_logits(r_dc, W)  # (B,1,K)
                idx_s = logit_s.argmax(dim=-1)  # (B,1)
                digits.append(idx_s)  # keep time dim
                r_dc = r_dc - F.embedding(idx_s, W)  # (B,1,d_c)

            next_id = self.lo_codec.compose(digits)  # (B,1) or (B,)
            if next_id.ndim == 1:
                next_id = next_id.unsqueeze(1)  # (B,1)

            # Append next_id to the sequence and extend masks accordingly
            generated = torch.cat([generated, next_id], dim=1)  # (B, T+1)
            kpm = F.pad(kpm, (0, 1), value=False)  # new token is real (not PAD)
            new_tokens = torch.cat([new_tokens, next_id], dim=1)
            # Early stop on EOS/EOP (after appending)
            if ((next_id == self.eos_id) | (next_id == self.eop_id)).all():
                break

        return new_tokens
