from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .sliding_sdpa import CausalLocalSDPABlock as _SDPASlidingSelf
from .flex_self import CausalLocalSelfFlexBlock as _FlexSlidingSelf


class SlidingSelf(nn.Module):
    """
    Regular sliding-window causal self-attention.

    backend: 'sdpa' | 'flex'
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        ffn_dim_multiplier: int = 4,
        *,
        backend: str = "sdpa",
    ):
        super().__init__()
        if backend == "flex":
            self.inner = _FlexSlidingSelf(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                d_ff=d_model * ffn_dim_multiplier,
            )
        elif backend == "sdpa":
            self.inner = _SDPASlidingSelf(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                d_ff=d_model * ffn_dim_multiplier,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.inner(x, key_padding_mask=key_padding_mask)

__all__ = ["SlidingSelf"]

