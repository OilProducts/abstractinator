from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedQueryAttention(nn.Module):
    """
    Learned-query pooling over variable-length segments.

    This is a relocation of the previous `components/learned_query_attention.py`
    with identical behavior.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_queries_per_segment: int = 1,
        max_queries: int | None = None,
        num_heads: int = 8,
        use_flex_attention: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries_per_segment = int(num_queries_per_segment)
        self.max_queries = max_queries
        self.use_flex_attention = use_flex_attention

        self.query_bank = nn.Parameter(
            torch.randn(1, 1, self.num_queries_per_segment, embed_dim)
        )  # (1, S_seg, Q_per_seg, D) -> broadcast across segments

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

    def forward(
        self,
        x: torch.Tensor,             # (B, S_tokens, D)
        seg_id: torch.Tensor,         # (B, S_tokens) integers
        valid_mask: torch.Tensor | None = None,  # (B, S_tokens) bool
    ) -> torch.Tensor:
        """
        Returns pooled per-segment representations shaped (B, S_seg * Q_per_seg, D),
        optionally clipped by `max_queries`.
        """
        B, S, D = x.shape

        # Count segments per batch (assumes seg_id increases, non-negative)
        num_segs = seg_id.amax(dim=1) + 1  # (B,)
        S_seg_max = int(num_segs.max().item())

        # Build per-segment queries by broadcasting query_bank
        Q_per_seg = self.num_queries_per_segment
        queries = self.query_bank.expand(B, S_seg_max, Q_per_seg, D)

        # Project Q/K/V
        q = self.q_proj(queries).view(B, S_seg_max * Q_per_seg, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Multi-head split
        H = self.num_heads
        q = q.view(B, -1, H, self.head_dim).transpose(1, 2)  # (B,H, S_seg*Q, d)
        k = k.view(B, S, H, self.head_dim).transpose(1, 2)   # (B,H, S, d)
        v = v.view(B, S, H, self.head_dim).transpose(1, 2)   # (B,H, S, d)

        # Build attention mask: per query, allow attending to tokens with same seg_id
        # Create (B, S_seg*Q, S) mask
        seg_for_query = torch.arange(S_seg_max, device=seg_id.device)
        seg_for_query = seg_for_query.repeat_interleave(Q_per_seg)  # (S_seg*Q,)
        mask = seg_id.unsqueeze(1) != seg_for_query.view(1, -1, 1)  # (B, S_seg*Q, S)
        if valid_mask is not None:
            mask = mask | (~valid_mask.unsqueeze(1))

        mask = mask.unsqueeze(1)  # (B,1, S_seg*Q, S) broadcast over heads

        # Scaled dot product attention
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
        )  # (B,H, S_seg*Q, d)

        out = attn.transpose(1, 2).reshape(B, S_seg_max * Q_per_seg, D)

        # Optional clipping by max_queries
        if self.max_queries is not None:
            out = out[:, : self.max_queries, :]
        return self.out_proj(out)

