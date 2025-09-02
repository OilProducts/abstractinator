from __future__ import annotations

from typing import Optional, Callable

import torch
from torch.nn.attention.flex_attention import flex_attention


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_mask: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Execute attention using torch.nn.attention.flex_attention.

    Expects tensors shaped [B, H, L_q, d], [B, H, L_k, d], [B, H, L_k, d_v].
    Returns [B, H, L_q, d_v].
    """
    return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

