from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config_types import AttentionConfig
from .sdpa.block import TransformerBlock as SDPATransformerBlock
from .sdpa.local import CausalLocalSDPABlock
from ..mla import CausalMLATransformerBlock, SlidingWindowMLATransformerBlock  # local import; avoid re-export cycles


class _SelfCausalSDPABlock(nn.Module):
    """Adapter to present a CausalMLATransformerBlock-like interface for SDPA blocks."""

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


def make_causal_self_block(
    *,
    dim: int,
    num_heads: int,
    ffn_dim_multiplier: int = 4,
    cfg: Optional[AttentionConfig] = None,
    ) -> nn.Module:
    cfg = cfg or AttentionConfig()
    if cfg.backend == "sdpa":
        return _SelfCausalSDPABlock(dim, num_heads, ffn_dim_multiplier)
    # default MLA
    head_dim = cfg.head_dim if cfg.head_dim is not None else (dim // num_heads)
    kv_comp_dim = cfg.kv_comp_dim
    q_comp_dim = cfg.q_comp_dim
    retr_dim = cfg.retr_dim
    use_flex = cfg.use_flex_attention
    return CausalMLATransformerBlock(
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        kv_comp_dim=kv_comp_dim,
        q_comp_dim=q_comp_dim,
        retr_dim=retr_dim,
        ffn_dim_multiplier=ffn_dim_multiplier,
        use_flex_attention=use_flex,
    )


def make_sliding_self_block(
    *,
    dim: int,
    num_heads: int,
    window_size: int,
    ffn_dim_multiplier: int = 4,
    cfg: Optional[AttentionConfig] = None,
) -> nn.Module:
    cfg = cfg or AttentionConfig()
    if cfg.backend == "sdpa":
        return CausalLocalSDPABlock(
            d_model=dim,
            n_heads=num_heads,
            window_size=window_size,
            d_ff=dim * ffn_dim_multiplier,
        )
    # default MLA
    head_dim = cfg.head_dim if cfg.head_dim is not None else (dim // num_heads)
    return SlidingWindowMLATransformerBlock(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        head_dim=head_dim,
        kv_comp_dim=cfg.kv_comp_dim,
        q_comp_dim=cfg.q_comp_dim,
        retr_dim=cfg.retr_dim,
        ffn_dim_multiplier=ffn_dim_multiplier,
        use_flex_attention=cfg.use_flex_attention,
    )
