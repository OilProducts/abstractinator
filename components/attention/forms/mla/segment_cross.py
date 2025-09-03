from __future__ import annotations

import torch.nn as nn

from .cross_segment import MLASegmentCrossAttention


class SegmentCross(nn.Module):
    """
    MLA segment lookback cross-attention.

    backend: 'flex' | 'fallback' (maps to use_flex_attention)
    """

    def __init__(
        self,
        *,
        q_dim: int,
        kv_dim: int,
        n_heads: int,
        lookback: int = 0,
        head_dim: int,
        kv_comp_dim: int,
        q_comp_dim: int,
        retr_dim: int,
        backend: str = "flex",
    ) -> None:
        super().__init__()
        use_flex = (backend == "flex")
        self.inner = MLASegmentCrossAttention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            n_heads=n_heads,
            lookback=lookback,
            head_dim=head_dim,
            kv_comp_dim=kv_comp_dim,
            q_comp_dim=q_comp_dim,
            retr_dim=retr_dim,
            use_flex_attention=use_flex,
        )

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

__all__ = ["SegmentCross"]
