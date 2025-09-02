from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .self_sdpa import TransformerBlock as _SDPAFullSelf
from .flex_self import CausalSelfFlexBlock as _FlexFullSelf


class FullSelf(nn.Module):
    """
    Regular causal self-attention over full history.

    backend: 'sdpa' | 'flex'
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim_multiplier: int = 4, *, backend: str = "sdpa"):
        super().__init__()
        if backend == "flex":
            self.inner = _FlexFullSelf(d_model, n_heads, d_ff=d_model * ffn_dim_multiplier)
        elif backend == "sdpa":
            self.inner = _SDPAFullSelf(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_model * ffn_dim_multiplier,
                attn_dropout=0.0,
                resid_dropout=0.0,
                prenorm=True,
                ln_eps=1e-5,
                bias=True,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.inner(x, key_padding_mask=key_padding_mask)

__all__ = ["FullSelf"]

