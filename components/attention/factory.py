from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config_types import AttentionConfig
from .forms.regular import (
    CausalSelfRegularBlock,
    CausalLocalRegularBlock,
    SDPASegmentCrossAttention,
    CausalSelfFlexBlock,
    CausalLocalSelfFlexBlock,
    SegmentCausalCrossAttentionFlex,
)
from .forms.mla.self import (
    CausalMLATransformerBlock,
    SlidingWindowMLATransformerBlock,
)
from .forms.mla.cross_segment import MLASegmentCrossAttention


class _SelfCausalSDPABlock(CausalSelfRegularBlock):
    pass


def make_causal_self_block(
    *,
    dim: int,
    num_heads: int,
    ffn_dim_multiplier: int = 4,
    cfg: Optional[AttentionConfig] = None,
    ) -> nn.Module:
    cfg = cfg or AttentionConfig()
    if cfg.variant == "regular":
        if cfg.kernel == "flex":
            return CausalSelfFlexBlock(dim, num_heads, dim * ffn_dim_multiplier)
        # Regular attention via SDPA
        return _SelfCausalSDPABlock(dim, num_heads, ffn_dim_multiplier)
    # MLA path
    head_dim = cfg.head_dim if cfg.head_dim is not None else (dim // num_heads)
    kv_comp_dim = cfg.kv_comp_dim
    q_comp_dim = cfg.q_comp_dim
    retr_dim = cfg.retr_dim
    use_flex = (cfg.kernel == "flex")
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
    if cfg.variant == "regular":
        if cfg.kernel == "flex":
            return CausalLocalSelfFlexBlock(
                d_model=dim,
                n_heads=num_heads,
                window_size=window_size,
                d_ff=dim * ffn_dim_multiplier,
            )
        return CausalLocalRegularBlock(
            d_model=dim,
            n_heads=num_heads,
            window_size=window_size,
            d_ff=dim * ffn_dim_multiplier,
        )
    # MLA path
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
        use_flex_attention=(cfg.kernel == "flex"),
    )




def make_segment_cross_attention(
    *,
    q_dim: int,
    kv_dim: int,
    d_attn: int,
    n_heads: int,
    lookback: int = 0,
    cfg: Optional[AttentionConfig] = None,
) -> nn.Module:
    """
    Factory for cross-attention over segment memories.

    - cfg.variant == "regular" → SDPA implementation
    - cfg.variant == "mla"     → MLA with flex or fallback kernel
    """
    cfg = cfg or AttentionConfig()
    if cfg.variant == "regular":
        look = lookback if cfg.lookback is None else int(cfg.lookback)
        if cfg.kernel == "flex":
            return SegmentCausalCrossAttentionFlex(
                q_dim=q_dim,
                kv_dim=kv_dim,
                d_attn=d_attn,
                n_heads=n_heads,
                lookback=look,
                bias=False,
            )
        return SDPASegmentCrossAttention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            d_attn=d_attn,
            n_heads=n_heads,
            lookback=look,
            bias=False,
        )

    head_dim = cfg.head_dim if cfg.head_dim is not None else (q_dim // n_heads)
    return MLASegmentCrossAttention(
        q_dim=q_dim,
        kv_dim=kv_dim,
        n_heads=n_heads,
        lookback=lookback if cfg.lookback is None else int(cfg.lookback),
        head_dim=head_dim,
        kv_comp_dim=int(cfg.kv_comp_dim or (q_dim // 8)),
        q_comp_dim=int(cfg.q_comp_dim or (q_dim // 8)),
        retr_dim=int(cfg.retr_dim or head_dim),
        use_flex_attention=(cfg.kernel == "flex"),
    )
