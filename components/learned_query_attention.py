import torch
import torch.nn as nn
import torch.nn.functional as F # Required if safe_softmax uses F.softmax
from typing import Optional, Tuple, Callable, Union
import math

from .utils import safe_softmax

# @torch.compile
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
    def __init__(self,
                 embed_dim: int,
                 num_queries_per_segment: int, # Defines the size L of query_template
                 num_heads: int,
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

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initializes the weights of projection layers and the query template.
        Uses Xavier uniform for projection weights and normal distribution for the query template.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        # Initialize query_template with a small standard deviation
        nn.init.normal_(self.query_template, std=0.02)

    def forward(
        self,
        x: torch.Tensor,                            # Input context: (B, S, D) for keys & values
        queries: torch.Tensor,                      # Pre-built queries: (B, Q_tot, D)
        attn_mask: Union[torch.Tensor, Callable],   # Attention mask: (B*H, Q_tot, S), boolean (True where masked)
        key_padding_mask: Optional[torch.Tensor] = None # Key padding mask: (B, S), boolean (True where padded)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the Learned Query Attention.

        The `queries` are expected to be constructed by the caller, often by utilizing
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
        Q_tot = queries.size(1)

        if D_x != self.d_model:
            raise ValueError(f"Input feature dimension {D_x} does not match model dimension {self.d_model}")
        if queries.size(2) != self.d_model:
            raise ValueError(f"Queries feature dimension {queries.size(2)} does not match model dimension {self.d_model}")

        # Normalize the input context (keys and values) - Pre-LN style for K/V
        x_norm = self.in_norm(x)

        # Project queries, keys, and values
        q_proj = self.q_proj(queries)    # (B, Q_tot, D)
        k_proj = self.k_proj(x_norm)     # (B, S, D)
        v_proj = self.v_proj(x_norm)     # (B, S, D)

        # Reshape and permute for multi-head attention
        # q: (B, Q_tot, D) -> (B, Q_tot, H, d_h) -> (B, H, Q_tot, d_h)
        q = q_proj.view(B, Q_tot, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # k: (B, S, D) -> (B, S, H, d_h) -> (B, H, d_h, S) (transpose last two dims for matmul)
        k = k_proj.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        # v: (B, S, D) -> (B, S, H, d_h) -> (B, H, S, d_h)
        v = v_proj.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention scores
        # (B, H, Q_tot, d_h) @ (B, H, d_h, S) -> (B, H, Q_tot, S)
        scores = (q @ k) / math.sqrt(self.head_dim)

        # scores: (B,H,Q,S)
        if callable(attn_mask):
            # Build mask *on the fly* by giving indices to the closure
            q_idx = torch.arange(Q_tot, device=scores.device)[:, None]             # (Q,)
            k_idx = torch.arange(S,     device=scores.device)[None, :]             # (S,)
            # (B,Q,S) — stack calls for each batch
            mask_bqs = torch.stack(
                [attn_mask(b=i, h=0, q_idx=q_idx, k_idx=k_idx) for i in range(B)],
                dim=0
            )
            combined_mask = mask_bqs.unsqueeze(1)  # (1,1,Q,S)
            if key_padding_mask is not None:
                combined_mask = combined_mask | key_padding_mask[:, None, None, :]
        else:
            # Combine attention mask and key padding mask
            # Reshape attn_mask from (B*H, Q_tot, S) to (B, H, Q_tot, S) for broadcasting
            combined_mask = attn_mask.view(B, self.num_heads, Q_tot, S)
            if key_padding_mask is not None:
                # Expand key_padding_mask from (B, S) to (B, 1, 1, S) for broadcasting.
                # If a key is padded (True in key_padding_mask), or if attn_mask
                # already specifies masking (True in combined_mask), then mask.
                combined_mask = combined_mask | key_padding_mask[:, None, None, :] # (B,H,Q,S)

        # Apply softmax with the combined mask
        # `safe_softmax` is assumed to handle masking by setting masked scores to -inf before softmax
        attn_weights = safe_softmax(scores, combined_mask, dim=-1) # (B, H, Q_tot, S)

        # Compute weighted sum of values (applying attention)
        # (B, H, Q_tot, S) @ (B, H, S, d_h) -> (B, H, Q_tot, d_h)
        output = attn_weights @ v

        # Concatenate heads: (B, H, Q_tot, d_h) -> (B, Q_tot, H, d_h) -> (B, Q_tot, D)
        output = output.permute(0, 2, 1, 3).reshape(B, Q_tot, self.d_model)

        # Apply output normalization (Post-LN style for this block's output) and final projection
        output_norm = self.out_norm(output)
        attn_output = self.out_proj(output_norm)

        # Return final output and mean attention weights for logging/inspection
        # Mean over heads: (B, H, Q_tot, S) -> (B, Q_tot, S)
        return attn_output, attn_weights.mean(dim=1)

