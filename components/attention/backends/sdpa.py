from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_bias: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Execute scaled dot-product attention using PyTorch SDPA backends.

    Expects tensors shaped [B, H, L_q, d], [B, H, L_k, d], [B, H, L_k, d_v].
    Returns [B, H, L_q, d_v].
    """
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_bias,
        is_causal=is_causal and (attn_bias is None),
        dropout_p=dropout_p if q.requires_grad else 0.0,
    )
