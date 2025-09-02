from __future__ import annotations

from typing import Optional, Any

import torch
import torch.nn as nn

from ...base import SegmentContext
from .cross_segment_sdpa import SegmentCausalCrossAttention


class SDPASegmentCrossAttention(nn.Module):
    """
    Adapter exposing a CrossAttentionBase-style interface around the regular
    SegmentCausalCrossAttention implementation.
    """

    def __init__(
        self,
        *,
        q_dim: int,
        kv_dim: int,
        d_attn: int,
        n_heads: int,
        lookback: int = 0,
        dropout: float = 0.0,
        bias: bool = False,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.inner = SegmentCausalCrossAttention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            d_attn=d_attn,
            n_heads=n_heads,
            lookback=lookback,
            dropout=dropout,
            bias=bias,
            device=device,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        segment: Optional[SegmentContext] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        assert segment is not None, "SegmentContext is required for SDPASegmentCrossAttention"

        # Prefer masks from SegmentContext; OR with provided key_padding_mask if given
        kv_mask = segment.kv_mask
        if key_padding_mask is not None:
            kv_mask = key_padding_mask if kv_mask is None else (kv_mask | key_padding_mask)

        return self.inner(
            q=q,
            kv_src=kv,
            seg_id=segment.seg_id,
            q_pos_ids=segment.q_pos_ids,
            kv_pos_ids=segment.kv_pos_ids,
            kv_mask=kv_mask,
            q_pad_mask=segment.q_pad_mask,
            cache=cache,
        )

