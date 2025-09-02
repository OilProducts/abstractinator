from __future__ import annotations

import torch.nn as nn

from .impl import SlidingWindowMLATransformerBlock


class SlidingSelf(nn.Module):
    """
    MLA sliding-window self-attention block.

    backend: 'flex' | 'fallback' (maps to use_flex_attention)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        *,
        head_dim: int,
        kv_comp_dim: int,
        q_comp_dim: int,
        retr_dim: int,
        ffn_dim_multiplier: int = 4,
        backend: str = "flex",
    ) -> None:
        super().__init__()
        use_flex = (backend == "flex")
        self.inner = SlidingWindowMLATransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            head_dim=head_dim,
            kv_comp_dim=kv_comp_dim,
            q_comp_dim=q_comp_dim,
            retr_dim=retr_dim,
            ffn_dim_multiplier=ffn_dim_multiplier,
            use_flex_attention=use_flex,
        )

    def forward(self, x, key_padding_mask=None):
        return self.inner(x, key_padding_mask=key_padding_mask)

    # Streaming API (native MLA)
    def prefill(self, x, *, pos_start: int = 0, key_padding_mask=None):
        return self.inner.prefill(x, pos_start=pos_start, key_padding_mask=key_padding_mask)

    def step(self, x_new, cache, *, pos_start: int, key_padding_mask_new=None):
        return self.inner.step(cache=cache, x_new=x_new, pos_start=pos_start, key_padding_mask_new=key_padding_mask_new)

__all__ = ["SlidingSelf"]
