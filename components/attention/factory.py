from __future__ import annotations

from typing import Optional

import torch.nn as nn

from ..config_types import AttentionConfig
from .forms.mla.full_self import FullSelf as MLAFullSelf
from .forms.mla.segment_cross import SegmentCross as MLASegmentCross
from .forms.mla.sliding_self import SlidingSelf as MLASlidingSelf
from .forms.regular.full_self import FullSelf as RegularFullSelf
from .forms.regular.segment_cross import SegmentCross as RegularSegmentCross
from .forms.regular.sliding_self import SlidingSelf as RegularSlidingSelf


class _SelfCausalSDPABlock(RegularFullSelf):
    def __init__(self, d_model: int, n_heads: int, ffn_dim_multiplier: int = 4):
        super().__init__(d_model, n_heads, ffn_dim_multiplier, backend="sdpa")


def make_causal_self_block(
    *,
    dim: int,
    num_heads: int,
    ffn_dim_multiplier: int = 4,
    cfg: Optional[AttentionConfig] = None,
) -> nn.Module:
    cfg = cfg or AttentionConfig()
    if cfg.variant == "regular":
        return RegularFullSelf(
            dim,
            num_heads,
            ffn_dim_multiplier,
            backend=(cfg.kernel or "sdpa"),
        )
    # MLA path
    head_dim = cfg.head_dim if cfg.head_dim is not None else (dim // num_heads)
    kv_comp_dim = cfg.kv_comp_dim
    q_comp_dim = cfg.q_comp_dim
    retr_dim = cfg.retr_dim
    use_flex = cfg.kernel == "flex"
    return MLAFullSelf(
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        kv_comp_dim=kv_comp_dim,
        q_comp_dim=q_comp_dim,
        retr_dim=retr_dim,
        ffn_dim_multiplier=ffn_dim_multiplier,
        backend=("flex" if use_flex else "fallback"),
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
        return RegularSlidingSelf(
            d_model=dim,
            n_heads=num_heads,
            window_size=window_size,
            ffn_dim_multiplier=ffn_dim_multiplier,
            backend=(cfg.kernel or "sdpa"),
        )
    # MLA path
    head_dim = cfg.head_dim if cfg.head_dim is not None else (dim // num_heads)
    return MLASlidingSelf(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        head_dim=head_dim,
        kv_comp_dim=cfg.kv_comp_dim,
        q_comp_dim=cfg.q_comp_dim,
        retr_dim=cfg.retr_dim,
        ffn_dim_multiplier=ffn_dim_multiplier,
        backend=("flex" if (cfg.kernel == "flex") else "fallback"),
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
        return RegularSegmentCross(
            q_dim=q_dim,
            kv_dim=kv_dim,
            d_attn=d_attn,
            n_heads=n_heads,
            lookback=look,
            backend=(cfg.kernel or "sdpa"),
        )

    head_dim = cfg.head_dim if cfg.head_dim is not None else (q_dim // n_heads)
    return MLASegmentCross(
        q_dim=q_dim,
        kv_dim=kv_dim,
        n_heads=n_heads,
        lookback=lookback if cfg.lookback is None else int(cfg.lookback),
        head_dim=head_dim,
        kv_comp_dim=int(cfg.kv_comp_dim or (q_dim // 8)),
        q_comp_dim=int(cfg.q_comp_dim or (q_dim // 8)),
        retr_dim=int(cfg.retr_dim or head_dim),
        backend=("flex" if (cfg.kernel == "flex") else "fallback"),
    )
