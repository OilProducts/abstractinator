from __future__ import annotations

import torch.nn as nn

from .impl import CausalMLATransformerBlock


class FullSelf(nn.Module):
    """
    MLA causal self-attention block.

    backend: 'flex' | 'fallback' (maps to use_flex_attention)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
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
        self.inner = CausalMLATransformerBlock(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_comp_dim=kv_comp_dim,
            q_comp_dim=q_comp_dim,
            retr_dim=retr_dim,
            ffn_dim_multiplier=ffn_dim_multiplier,
            use_flex_attention=use_flex,
        )

    def forward(self, x, key_padding_mask=None):
        return self.inner(x, key_padding_mask=key_padding_mask)

__all__ = ["FullSelf"]

