from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from components.rope import RoPECache, apply_rope
from ...backends.flex import run as flex_run


class SegmentCausalCrossAttentionFlex(nn.Module):
    """
    Regular segment cross-attention using FlexAttention.

    - Computes Q/K/V with linear projections and applies RoPE on Q and K using
      provided absolute position ids (SegmentContext.q_pos_ids, kv_pos_ids).
    - Enforces segment lookback window via a FlexAttention block mask:
        keep if (q_seg - lookback) <= k_index <= q_seg
    - Key padding is handled in score_mod.
    """

    def __init__(
        self,
        *,
        q_dim: int,
        kv_dim: int,
        d_attn: int,
        n_heads: int,
        lookback: int = 0,
        bias: bool = False,
        rope_max_seqlen: int = 8192,
    ) -> None:
        super().__init__()
        assert d_attn % n_heads == 0, "d_attn must be divisible by n_heads"
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.d_attn = d_attn
        self.n_heads = n_heads
        self.hdim = d_attn // n_heads
        self.lookback = int(lookback)

        self.q_proj = nn.Linear(q_dim, d_attn, bias=bias)
        self.kv_proj = nn.Linear(kv_dim, 2 * d_attn, bias=bias)
        self.o_proj = nn.Linear(d_attn, q_dim, bias=bias)

        # Build on CPU by default; slices are moved to the right device at use time
        self.rope = RoPECache(max_seqlen=rope_max_seqlen, head_dim=self.hdim, dtype=torch.bfloat16, device="cpu")
        assert (self.hdim % 2) == 0

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return (
            x.reshape(B, L, self.n_heads, self.hdim)
             .permute(0, 2, 1, 3)
             .contiguous()
        )

    def _build_block_mask(self, seg_id_q: torch.Tensor, Lk: int, pad: Optional[torch.Tensor]) -> object:
        # seg_id_q: [B, Lq] long; pad: [B, Lk] bool or None
        seg_id_q = seg_id_q.contiguous()
        if pad is not None:
            pad = pad.to(torch.bool).contiguous()

        def keep(b, h, q, k):
            ok = (k <= seg_id_q[b, q]) & (k >= (seg_id_q[b, q] - self.lookback))
            if pad is not None:
                ok = ok & (~pad[b, k])
            return ok

        B = seg_id_q.size(0)
        Lq = seg_id_q.size(1)
        return create_block_mask(keep, B=B, H=self.n_heads, Q_LEN=Lq, KV_LEN=Lk, BLOCK_SIZE=128, device=seg_id_q.device)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        *,
        segment=None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[object] = None,
    ) -> torch.Tensor:
        del cache
        B, Lq, _ = q.shape
        Lk = kv.size(1)

        # Q path
        qh = self._split_heads(self.q_proj(q))  # [B,H,Lq,Dh]
        # K/V path
        kv_lin = self.kv_proj(kv)
        k, v = kv_lin.split(self.d_attn, dim=-1)
        kh = self._split_heads(k)
        vh = self._split_heads(v)

        # RoPE
        # Gather absolute position ids from SegmentContext; accept (B,Lq) and (Lk,)
        q_pos_ids = getattr(segment, "q_pos_ids", None)
        kv_pos_ids = getattr(segment, "kv_pos_ids", None)
        if q_pos_ids is None or kv_pos_ids is None:
            raise AssertionError("SegmentContext.q_pos_ids and kv_pos_ids are required for regular+flex cross-attention")

        q_pos = q_pos_ids if q_pos_ids.dim() == 2 else q_pos_ids.unsqueeze(0).expand(B, -1)
        q_pos = q_pos.to(torch.long)
        kv_pos = kv_pos_ids.to(torch.long).view(1, 1, Lk, 1)

        cos_all, sin_all = self.rope.cos.to(device=qh.device, dtype=qh.dtype), self.rope.sin.to(device=qh.device, dtype=qh.dtype)
        Smax = cos_all.size(2)
        q_pos = torch.clamp(q_pos, 0, Smax - 1)
        q_cos = torch.take_along_dim(cos_all.expand(B, 1, -1, -1), q_pos[:, None, :, None], dim=2)
        q_sin = torch.take_along_dim(sin_all.expand(B, 1, -1, -1), q_pos[:, None, :, None], dim=2)
        qh = apply_rope(qh, q_cos, q_sin)

        kv_pos = torch.clamp(kv_pos, 0, Smax - 1)
        k_cos = torch.take_along_dim(cos_all.unsqueeze(3).expand(B, 1, -1, 1, cos_all.size(-1)), kv_pos, dim=2)
        k_sin = torch.take_along_dim(sin_all.unsqueeze(3).expand(B, 1, -1, 1, sin_all.size(-1)), kv_pos, dim=2)
        kh = apply_rope(kh, k_cos, k_sin)

        # Block mask from segment lookback + padding
        kv_mask = getattr(segment, "kv_mask", None)
        if key_padding_mask is not None:
            kv_mask = key_padding_mask if kv_mask is None else (kv_mask | key_padding_mask)
        block = self._build_block_mask(getattr(segment, "seg_id"), Lk, kv_mask)

        ctx = flex_run(qh, kh, vh, block_mask=block, score_mod=None)  # [B,H,Lq,Dh]
        out = ctx.transpose(1, 2).reshape(B, Lq, self.d_attn)
        return self.o_proj(out)

__all__ = ["SegmentCausalCrossAttentionFlex"]
