from __future__ import annotations

import torch
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

    # Streaming API (recompute fallback; matches regular.FullSelf signature)
    def prefill(self, x, *, pos_start: int = 0, key_padding_mask=None):
        y = self.forward(x, key_padding_mask=key_padding_mask)
        return y, {"x": x, "kpm": key_padding_mask}

    def step(
        self,
        new_x,
        cache,
        *,
        pos_start: int = 0,
        key_padding_mask_new=None,
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
        y_cat = self.forward(x_cat, key_padding_mask=kpm_cat)
        y_new = y_cat[:, -new_x.size(1) :, :]
        cache = {"x": x_cat, "kpm": kpm_cat}
        return y_new, cache

__all__ = ["FullSelf"]
