from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

from .utils import safe_softmax

class LearnedQueryAttention(nn.Module):
    """
    Multi-Head Attention where queries are derived from a set of learned vectors.

    This module performs multi-head attention using a key-value context `x` and
    a set of `queries`. It owns a `query_template` (L learned vectors), which is
    intended to be used by a calling module (e.g., by tiling or adapting it per
    segment) to construct the actual `queries` tensor passed to the `forward` method.

    The module handles attention masking (both a generic attention mask and a key
    padding mask) and applies RMS normalization to the input `x` (keys/values)
    and to the output of the attention mechanism before a final projection.

    Attributes:
        d_model (int): The embedding dimension.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head (embed_dim // num_heads).
        L (int): The number of unique learned query vectors in `query_template`.
                 This is set by `num_queries_per_segment` in the constructor.
        in_norm (nn.RMSNorm): RMS normalization applied to the input `x`.
        out_norm (nn.RMSNorm): RMS normalization applied to the attention output
                                 before the final projection.
        query_template (nn.Parameter): A learnable tensor of shape (L, D). This serves
                                      as a template for generating the actual queries
                                      used in the attention mechanism.
        q_proj (nn.Linear): Linear projection for queries.
        k_proj (nn.Linear): Linear projection for keys.
        v_proj (nn.Linear): Linear projection for values.
        out_proj (nn.Linear): Final linear projection for the output.
    """

    def __init__(
        self,
        embed_dim: int,
        num_queries_per_segment: int,  # Defines the size L of query_template
        num_heads: int,
        use_flex_attention: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads for multi-head attention.")

        self.d_model = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # L: Number of distinct learned query vectors in the template
        self.L = num_queries_per_segment

        self.in_norm = nn.RMSNorm(embed_dim)
        self.out_norm = nn.RMSNorm(embed_dim)

        # Learned query template – (L, D).
        # This template is intended to be used by the caller to construct the
        # `queries` tensor passed to the forward method. For example, in a
        # segmented attention scenario, these L queries might be repeated for each segment.
        self.query_template = nn.Parameter(torch.randn(self.L, embed_dim))

        # Linear projections for Q, K, V and output. Bias is often disabled
        # in transformer components but can be included if desired.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.use_flex_attention = use_flex_attention

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        # Initialize query_template with a small standard deviation
        nn.init.normal_(self.query_template, std=0.02)

        self._pad_for_scores: Optional[torch.Tensor]    = None  # [B, S] bool
        self._qseg_for_scores: Optional[torch.Tensor]   = None  # [B, Q] int (−1 = invalid/pad query)
        self._segid_for_scores: Optional[torch.Tensor]  = None  # [B, S] int

    def forward(
        self,
        x: torch.Tensor,  # Input context: (B, S, D) for keys & values
        queries: torch.Tensor,  # Pre-built queries: (B, Q_tot, D)
        q_seg: Optional[torch.Tensor] = None,  # Segment indices for queries: (B, Q_tot), int
        seg_id: Optional[torch.Tensor] = None,  # Segment indices for keys/
        attn_mask: Optional[torch.Tensor] = None,  # Attention mask: (B*H, Q_tot, S), boolean (True where masked)
        key_padding_mask: torch.Tensor | None = None,  # Key padding mask: (B, S), boolean (True where padded)
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the Learned Query Attention.

        The `queries` are expected to be constructed by the caller by utilizing
        `self.query_template`. For instance, if processing `N_seg` segments, `Q_tot`
        might be `N_seg * self.L`, where `self.L` is `num_queries_per_segment`.
        The `attn_mask` should be constructed to restrict attention appropriately,
        e.g., to within segments.

        Args:
            x (torch.Tensor): The input tensor providing keys and values.
                Shape: (batch_size, sequence_length_kv, embed_dim).
            queries (torch.Tensor): The pre-constructed query tensor.
                Shape: (batch_size, total_num_queries, embed_dim).
            attn_mask (torch.Tensor): Boolean attention mask. `True` indicates
                positions that should be masked (ignored) during attention.
                The shape (batch_size * num_heads, total_num_queries, sequence_length_kv)
                is expected to align with internal reshaping.
            key_padding_mask (Optional[torch.Tensor]): Boolean mask for padded
                elements in `x`. `True` indicates a padded key/value position.
                Shape: (batch_size, sequence_length_kv).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - attn_output (torch.Tensor): The output of the attention mechanism.
              Shape: (batch_size, total_num_queries, embed_dim).
            - attn_weights (torch.Tensor): The attention weights, averaged over heads.
              Shape: (batch_size, total_num_queries, sequence_length_kv).
              Suitable for logging or analysis (consider detaching if not used for gradients).
        """
        B, S, D_x = x.shape
        Q = queries.size(1)

        if D_x != self.d_model:
            raise ValueError(f"Input feature dimension {D_x} does not match model dimension {self.d_model}")
        if queries.size(2) != self.d_model:
            raise ValueError(
                f"Queries feature dimension {queries.size(2)} does not match model dimension {self.d_model}"
            )

        # Normalize the input context (keys and values) - Pre-LN style for K/V
        x_norm = self.in_norm(x)

        # Project queries, keys, and values
        q_proj = self.q_proj(queries)  # (B, Q_tot, D)
        k_proj = self.k_proj(x_norm)  # (B, S, D)
        v_proj = self.v_proj(x_norm)  # (B, S, D)

        # Reshape and permute for multi-head attention
        # q: (B, Q_tot, D) -> (B, Q_tot, H, d_h) -> (B, H, Q_tot, d_h)
        q = q_proj.view(B, Q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # k: (B, S, D) -> (B, S, H, d_h) -> (B, H, d_h, S) (transpose last two dims for matmul)
        k = k_proj.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # v: (B, S, D) -> (B, S, H, d_h) -> (B, H, S, d_h)
        v = v_proj.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)


        if self.use_flex_attention:
            # Wire small side tensors for score_mod
            self._pad_for_scores   = key_padding_mask.to(torch.bool).contiguous() if key_padding_mask is not None else None
            self._qseg_for_scores  = q_seg.contiguous() if q_seg is not None else None
            self._segid_for_scores = seg_id.contiguous() if seg_id is not None else None

            from torch.nn.attention.flex_attention import create_block_mask

            def _keep_factory(pad: torch.Tensor | None, qseg: torch.Tensor | None, kseg: torch.Tensor | None):
                pad = None if pad is None else pad.to(torch.bool).contiguous()
                qseg = None if qseg is None else qseg.contiguous()
                kseg = None if kseg is None else kseg.contiguous()

                def keep(b, h, q, k):
                    # True == keep (allowed)
                    ok = torch.ones_like(q == q, dtype=torch.bool)  # shape broadcastable to (Q,K)
                    if pad is not None:
                        ok &= ~pad[b, k]  # disallow padded keys
                    if (qseg is not None) and (kseg is not None):
                        ok &= (qseg[b, q] == kseg[b, k])  # require same segment
                    if qseg is not None:
                        ok &= (qseg[b, q] >= 0)  # drop invalid queries
                    return ok

                return keep

            keep = _keep_factory(key_padding_mask, q_seg, seg_id)
            block_mask = create_block_mask(
                keep,
                B=None, H=None, Q_LEN=q.size(2), KV_LEN=k.size(2),  # or explicit B/H if you prefer
                BLOCK_SIZE=128, device=q.device
            )

            ctx = flex_attention(q, k, v, block_mask=block_mask)  # no score_mod needed

            # hygiene
            self._pad_for_scores = self._qseg_for_scores = self._segid_for_scores = None

            # We generally don't need per-head weights in training; avoid materializing them
            attn_weights = queries.new_empty(0) if not return_attn else None  # (optional: compute via fallback if really needed)

        else:
            # Fallback: standard dot-prod attention, still mask padding cheaply,
            # and apply segment gating in manageable chunks (no huge (B·H,Q,S) mask).
            scale = 1.0 / math.sqrt(self.head_dim)
            # (B,H,Q,d) @ (B,H,d,S) → (B,H,Q,S)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # cheap broadcast for padding
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

            CHUNK = 128  # segment-gating chunk (Q axis)
            outs, atts = [], []
            for i in range(0, Q, CHUNK):
                sl = slice(i, min(i + CHUNK, Q))
                scores_i = scores[:, :, sl, :]  # (B,H,c,S)
                # if (q_seg is not None) and (seg_id is not None):
                    # (B,c,S) mismatch mask (small-ish) instead of full (B,H,c,S)
                mismatch = (seg_id[:, None, :] != q_seg[:, sl].unsqueeze(-1))
                    # scores_i = scores_i.masked_fill(mismatch[:, None, :, :], float("-inf"))
                attn_i = safe_softmax(scores_i, mismatch[:, None, :, :], dim=-1)  # (B,H,c,S)
                outs.append(torch.matmul(attn_i, v))           # (B,H,c,d)
                if return_attn:
                    atts.append(attn_i.mean(dim=1))            # (B,c,S)

            ctx = torch.cat(outs, dim=2)                        # (B,H,Q,d)
            attn_weights = torch.cat(atts, dim=1) if return_attn else queries.new_empty(0)

        # Concat heads → (B,Q,D)
        out = ctx.permute(0, 2, 1, 3).reshape(B, Q, self.d_model)
        out = self.out_norm(out)
        out = self.out_proj(out)

        return out, attn_weights
