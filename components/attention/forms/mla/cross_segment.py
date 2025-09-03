from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .impl import MultiheadLatentAttention
from torch.nn.attention.flex_attention import create_block_mask


class MLASegmentCrossAttention(nn.Module):
    """
    Cross-attention for segment memories using MultiheadLatentAttention.

    Enforces segment lookback with a boolean block mask:
      allow keys where (q_seg - lookback) <= k_seg <= q_seg.
    """

    def __init__(
        self,
        *,
        q_dim: int,
        kv_dim: int,
        n_heads: int,
        lookback: int,
        head_dim: int,
        kv_comp_dim: int,
        q_comp_dim: int,
        retr_dim: int,
        use_flex_attention: bool,
    ) -> None:
        super().__init__()
        assert retr_dim % 2 == 0, "retr_dim must be even for RoPE"
        assert q_dim % n_heads == 0, "q_dim must be divisible by n_heads"

        self.lookback = int(lookback)
        self.kv_comp = nn.Linear(kv_dim, kv_comp_dim, bias=False)
        self.mla = MultiheadLatentAttention(
            dim_q=q_dim,
            num_heads=n_heads,
            head_dim=head_dim,
            kv_comp_dim=kv_comp_dim,
            q_comp_dim=q_comp_dim,
            retr_dim=retr_dim,
            use_flex_attention=use_flex_attention,
        )

    @staticmethod
    def _combine(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if a is None:
            return b
        if b is None:
            return a
        return a | b

    def _lookback_block(self, seg_id_q: torch.Tensor, Lk: int) -> torch.Tensor:
        B, Lq = seg_id_q.shape
        dev = seg_id_q.device
        k_seg = torch.arange(Lk, device=dev).view(1, 1, Lk)  # [1,1,Lk]
        q_seg = seg_id_q.view(B, Lq, 1)                      # [B,Lq,1]
        good = (k_seg <= q_seg) & (k_seg >= (q_seg - self.lookback))
        return (~good).unsqueeze(1)                          # [B,1,Lq,Lk]

    def _flex_lookback_block(self, seg_id_q: torch.Tensor, Lk: int, pad: Optional[torch.Tensor]) -> object:
        # Build a FlexAttention block mask for lookback (and optional padding) gating.
        # seg_id_q: [B,Lq] long; pad: [B,Lk] bool or None
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
        return create_block_mask(keep, B=B, H=self.mla.h, Q_LEN=Lq, KV_LEN=Lk, BLOCK_SIZE=128, device=seg_id_q.device)

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
        dev = q.device

        kv_c = self.kv_comp(kv)  # [B, Lk, c]

        seg_id = getattr(segment, "seg_id", None)
        if seg_id is None:
            raise AssertionError("SegmentContext.seg_id is required for MLASegmentCrossAttention")
        # Merge key padding from SegmentContext and/or function arg
        kv_mask = getattr(segment, "kv_mask", None)
        if key_padding_mask is not None:
            kv_mask = key_padding_mask if kv_mask is None else (kv_mask | key_padding_mask)
        seg_long = seg_id.to(device=dev, dtype=torch.long)
        if self.mla.use_flex_attention:
            block = self._flex_lookback_block(seg_long, Lk, kv_mask.to(torch.bool) if kv_mask is not None else None)
        else:
            block = self._lookback_block(seg_long, Lk)
            if kv_mask is not None:
                block = self._combine(block, kv_mask[:, None, None, :].to(device=dev, dtype=torch.bool))

        # Hint padding to MLA for score-side masking
        self.mla.set_pad_for_scores(kv_mask.to(torch.bool) if kv_mask is not None else None)

        out = self.mla(hidden_q=q, kv_c=kv_c, block_mask=block)  # [B, Lq, q_dim]

        q_pad = getattr(segment, "q_pad_mask", None)
        if q_pad is not None:
            out = out.masked_fill(q_pad.to(device=dev, dtype=torch.bool).unsqueeze(-1), 0.0)

        self.mla.set_pad_for_scores(None)
        return out


__all__ = ["MLASegmentCrossAttention"]
