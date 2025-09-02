from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ...sdpa.block import TransformerBlock as SDPATransformerBlock


class CausalSelfRegularBlock(nn.Module):
    """Adapter exposing a causal self-attention block powered by SDPA backend."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim_multiplier: int = 4):
        super().__init__()
        self.block = SDPATransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * ffn_dim_multiplier,
            attn_dropout=0.0,
            resid_dropout=0.0,
            prenorm=True,
            ln_eps=1e-5,
            bias=True,
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.block(x, attn_mask=None, key_padding_mask=key_padding_mask, is_causal=True)


__all__ = ["CausalSelfRegularBlock"]

