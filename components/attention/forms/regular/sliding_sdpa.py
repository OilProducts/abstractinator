from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .block_impl import TransformerBlock as SDPATransformerBlock
from ...masks import merge_masks, additive_neg_inf


class CausalLocalSDPABlock(nn.Module):
    """
    Causal local self-attention using SDPA with a sliding window of radius
    `window_size` over past tokens only. Pre‑LN block structure.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        window_size: int,
        d_ff: int | None = None,
        ln_eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.window = int(window_size)
        self.block = SDPATransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff or (4 * d_model),
            attn_dropout=0.0,
            resid_dropout=0.0,
            prenorm=True,
            ln_eps=ln_eps,
            bias=bias,
        )

    def _causal_band_mask(self, L: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(L, device=device)[:, None]
        j = torch.arange(L, device=device)[None, :]
        # Disallow future (j > i) or past beyond window (i - j > window)
        return (j > i) | ((i - j) > self.window)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        band = self._causal_band_mask(L, x.device)  # (L, L) bool
        # Convert to additive mask now (block will merge KPM); pass is_causal=True, too.
        bias = torch.where(band, additive_neg_inf(x.dtype, x.device), torch.zeros(1, dtype=x.dtype, device=x.device))
        bias = bias.view(1, 1, L, L)  # (1,1,L,L)
        return self.block(x, attn_mask=bias, key_padding_mask=key_padding_mask, is_causal=True)

