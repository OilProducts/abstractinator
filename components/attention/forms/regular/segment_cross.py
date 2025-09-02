from __future__ import annotations

from typing import Optional, Any

import torch
import torch.nn as nn

from .adapter import SDPASegmentCrossAttention as _SDPASegmentCross
from .cross_segment_flex import SegmentCausalCrossAttentionFlex as _FlexSegmentCross
from ...base import SegmentContext


class SegmentCross(nn.Module):
    """
    Regular segment lookback cross-attention.

    backend: 'sdpa' | 'flex'
    """

    def __init__(
        self,
        *,
        q_dim: int,
        kv_dim: int,
        d_attn: int,
        n_heads: int,
        lookback: int = 0,
        backend: str = "sdpa",
    ) -> None:
        super().__init__()
        if backend == "flex":
            self.inner = _FlexSegmentCross(
                q_dim=q_dim,
                kv_dim=kv_dim,
                d_attn=d_attn,
                n_heads=n_heads,
                lookback=int(lookback),
                bias=False,
            )
        elif backend == "sdpa":
            self.inner = _SDPASegmentCross(
                q_dim=q_dim,
                kv_dim=kv_dim,
                d_attn=d_attn,
                n_heads=n_heads,
                lookback=int(lookback),
                bias=False,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        *,
        segment: Optional[SegmentContext] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        return self.inner(q, kv, segment=segment, attn_mask=attn_mask, key_padding_mask=key_padding_mask, cache=cache)

__all__ = ["SegmentCross"]

