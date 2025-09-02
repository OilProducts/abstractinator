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

    # Streaming API (recompute fallback)
    def prefill(self, x: torch.Tensor, *, pos_start: int = 0, key_padding_mask: Optional[torch.Tensor] = None):
        return {"x": x, "kpm": key_padding_mask}, self.forward(x, key_padding_mask)

    def step(
        self,
        new_x: torch.Tensor,
        cache,
        *,
        pos_start: int = 0,
        key_padding_mask_new: Optional[torch.Tensor] = None,
    ):
        old_x = cache.get("x")
        old_kpm = cache.get("kpm")
        x_cat = torch.cat([old_x, new_x], dim=1)
        if old_kpm is None and key_padding_mask_new is None:
            kpm_cat = None
        else:
            old_kpm = old_kpm if old_kpm is not None else torch.zeros_like(old_x[..., 0], dtype=torch.bool)
            new_kpm = key_padding_mask_new if key_padding_mask_new is not None else torch.zeros_like(new_x[..., 0], dtype=torch.bool)
            kpm_cat = torch.cat([old_kpm, new_kpm], dim=1)
        y_cat = self.forward(x_cat, kpm_cat)
        y_new = y_cat[:, -new_x.size(1) :, :]
        cache = {"x": x_cat, "kpm": kpm_cat}
        return cache, y_new

__all__ = ["SlidingSelf"]
