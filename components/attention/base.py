from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch


@dataclass
class SegmentContext:
    """
    Segment-aware context for cross-attention.

    Fields
    -----
    seg_id:      (B, Lq) int – segment index per query token (monotonic non-decreasing)
    q_pos_ids:   (B, Lq) or (Lq,) int – absolute positions for query tokens
    kv_pos_ids:  (Lkv,) int – absolute positions for memory segments
    lookback:    int – how many segments back each query may attend to
    kv_mask:     (B, Lkv) bool – True => mask out memory item
    q_pad_mask:  (B, Lq) bool – True => query is padding (output is zeroed)
    """

    seg_id: torch.Tensor
    q_pos_ids: torch.Tensor
    kv_pos_ids: torch.Tensor
    lookback: int = 0
    kv_mask: Optional[torch.Tensor] = None
    q_pad_mask: Optional[torch.Tensor] = None


class SelfAttentionBase(Protocol):
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
        is_causal: bool = False,
    ) -> torch.Tensor: ...

    def prefill(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Any: ...

    def step(
        self,
        new_x: torch.Tensor,
        cache: Any,
        *,
        key_padding_mask_new: Optional[torch.Tensor] = None,
    ) -> tuple[Any, torch.Tensor]: ...


class CrossAttentionBase(Protocol):
    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        segment: Optional[SegmentContext] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor: ...
