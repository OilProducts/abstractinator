from __future__ import annotations

from typing import Optional

import torch


def causal_mask(
    L_q: int, L_k: int | None = None, *, device: torch.device, dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """Return a causal mask with shape (L_q, L_k) as a boolean tensor (True = disallow)."""
    L_k = L_q if L_k is None else L_k
    m = torch.triu(torch.ones(L_q, L_k, device=device, dtype=torch.bool), diagonal=1)
    return m.to(dtype)


def additive_neg_inf(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)


def merge_masks(
    *,
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    B: int,
    H: int,
    L_q: int,
    L_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Normalize and merge boolean/additive masks into one additive mask broadcastable
    to (B, H, L_q, L_k). Returns None if no masking is needed.
    """
    final = None

    # Normalize attn_mask to additive float mask
    if attn_mask is not None:
        am = attn_mask
        if am.dtype == torch.bool:
            am = torch.where(am, additive_neg_inf(dtype, device), torch.zeros(1, dtype=dtype, device=device))
        else:
            am = am.to(dtype)

        if am.dim() == 2:  # (L_q, L_k)
            am = am.view(1, 1, L_q, L_k)
        elif am.dim() == 3:  # (B, L_q, L_k)
            am = am.view(B, 1, L_q, L_k)
        # else assume (B,H,L_q,L_k)

        final = am

    # Fold key_padding_mask into additive mask
    if key_padding_mask is not None:
        kpm = key_padding_mask
        if kpm.dtype == torch.bool:
            kpm = torch.where(kpm, additive_neg_inf(dtype, device), torch.zeros(1, dtype=dtype, device=device))
        else:
            kpm = kpm.to(dtype)
        kpm = kpm.view(B, 1, 1, L_k)
        final = kpm if final is None else final + kpm

    if final is not None and final.size(1) == 1 and H > 1:
        final = final.expand(B, H, L_q, L_k)

    return final
