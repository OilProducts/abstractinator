import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from .utils import safe_softmax
from torch.nn.attention.flex_attention import (
    flex_attention,           # fused kernel
    create_block_mask,        # helper to build sparse mask
    and_masks
)
from .rope import apply_rope
from .swiglu import SwiGLU
from typing import Optional, Tuple, Dict


@lru_cache(maxsize=64)
def _cached_cross_window_mask(q_len: int, kv_len: int, window: int) -> torch.Tensor:
    """Return a cross-window mask computed on CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q_idx = torch.arange(q_len, device=device)[:, None]
    kv_idx = torch.arange(kv_len, device=device)[None, :]
    rel_pos = kv_idx - q_idx
    mask = (rel_pos > 0) | (rel_pos < -window)
    return mask

def get_cross_window_mask(
    q_len: int,
    kv_len: int,
    window: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the cached mask on ``device``."""
    return _cached_cross_window_mask(q_len, kv_len, window).to(device)

# @torch.compile
class LocalSlidingWindowAttention(nn.Module):
    """
    Causal sliding‑window multi‑head attention in O(S·w) with FlexAttention.

    *   Works with torch.compile → single Flash‑class kernel.
    *   Key‑padding mask is applied to keys inside each window.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        bias: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window = window_size

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


    # ------------------------------------------------------------------ #
    # helper: memo‑cached BlockMask builder
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sliding_block(seq_len: int, window: int, device: torch.device, is_training: bool) -> torch.Tensor:
        cache_key = (seq_len, window, device.index if device.type == "cuda" else -1)
        cache = LocalSlidingWindowAttention._sliding_block.__dict__.setdefault(
            "cache", {}
        )
        if cache_key in cache:
            return cache[cache_key]

        # python callback: keep (q,k) if q‑window ≤ k ≤ q
        def mask_mod(b, h, q, k):
            return (k >= q - window) & (k <= q)

        blk = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            BLOCK_SIZE=128,
            _compile=False,           # pre‑compile sparse metadata
            device=device,
        )

        if is_training:
            cache[cache_key] = blk
        return blk

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,                            # (B, S, D)
        key_padding_mask: Optional[torch.Tensor] = None,   # (B, S) – True = pad
    ) -> torch.Tensor:
        B, S, D = x.shape
        H, d = self.num_heads, self.head_dim
        if D != self.embed_dim:
            raise ValueError("embed_dim mismatch")

        # project and reshape -------------------------------------------------
        q = self.q_proj(x).view(B, S, H, d).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(B, S, H, d).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, S, H, d).transpose(1, 2).contiguous()
        q = apply_rope(q)
        k = apply_rope(k)
        q = q * (d ** -0.5)

        # block mask ----------------------------------------------------------
        block_mask = self._sliding_block(S, self.window, x.device, self.training)
        if key_padding_mask is not None:
            # Recreate the mask with padding info since BlockMask.add_padding_mask
            # is not implemented in PyTorch.
            pad_mask = key_padding_mask.to(torch.bool)

            def _kpm_mod(b, h, q, k):
                return ~pad_mask[b, k]

            combined = and_masks(block_mask.mask_mod, _kpm_mod)
            block_mask = create_block_mask(
                combined,
                B=pad_mask.size(0),
                H=None,
                Q_LEN=block_mask.seq_lengths[0],
                KV_LEN=block_mask.seq_lengths[1],
                BLOCK_SIZE=block_mask.BLOCK_SIZE,
                device=pad_mask.device,
            )

        # fused attention -----------------------------------------------------
        ctx = flex_attention(
            query=q,
            key=k,
            value=v,
            block_mask=block_mask,   # ← mandatory kw‑arg
        )

        # merge heads + output proj -------------------------------------------
        out = ctx.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)
        return out

# @torch.compile
class SlidingWindowAttention(nn.Module):
    """Multi-head scaled-dot-product attention restricted to a fixed retrospective window.

    Each position `i` in the sequence can attend to tokens in the range
    `[max(0, i - window_size), i]`. This creates a causal attention mechanism
    where attention is limited to a recent context.

    Attributes:
        embed_dim (int): Total embedding dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head (`embed_dim // num_heads`).
        window_size (int): The number of previous tokens (relative to the current token)
                           that a query can attend to. The current token itself is always
                           included in the window. Thus, the total window length is `window_size + 1`.
        q_proj (nn.Linear): Linear projection for queries.
        k_proj (nn.Linear): Linear projection for keys.
        v_proj (nn.Linear): Linear projection for values.
        out_proj (nn.Linear): Final linear projection for the output.
    """

    def __init__(self, embed_dim: int, num_heads: int, window_size: int,
                 bias: bool = True) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size # Number of *previous* tokens query can attend to

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates a sliding window attention mask.

        The mask ensures that a token at position `i` can only attend to tokens
        in the range `[max(0, i - window_size), i]`.

        Args:
            seq_len (int): The length of the sequence.
            device (torch.device): The device to create the mask on.

        Returns:
            torch.Tensor: A boolean mask of shape (seq_len, seq_len) where `True`
                          indicates positions that should be *blocked* from attention.
        """
        # Create indices for sequence positions: [0, 1, ..., seq_len-1]
        idx = torch.arange(seq_len, device=device)

        # Calculate relative positions: rel_pos[i, j] = j - i
        # Rows are query positions (i), columns are key positions (j).
        # Shape: (seq_len, seq_len)
        rel_pos = idx[None, :] - idx[:, None]

        # Mask conditions:
        # 1. `rel_pos > 0`: Block attention to future tokens (j > i). This ensures causality.
        # 2. `rel_pos < -self.window_size`: Block attention to tokens too far in the past.
        #    This means j - i < -window_size, or j < i - window_size.
        #    So, tokens from `i - window_size` up to `i` are allowed.
        #    The window includes the current token and `self.window_size` previous tokens.
        mask = (rel_pos > 0) | (rel_pos < -self.window_size)
        return mask  # True means blocked

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes sliding-window multi-head attention.

        Args:
            x (torch.Tensor): Input tensor.
                Shape: (batch_size, sequence_length, embed_dim).
            attn_mask (Optional[torch.Tensor]): Additional attention mask, broadcastable
                to (sequence_length, sequence_length). `True` indicates positions
                where attention is **not permitted**.
            key_padding_mask (Optional[torch.Tensor]): Boolean mask indicating padded
                tokens in `x`. `True` for padding.
                Shape: (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor after attention.
                Shape: (batch_size, sequence_length, embed_dim).
        """
        B, S, D = x.shape # D is embed_dim
        if D != self.embed_dim:
            raise ValueError(f"Input embed_dim ({D}) does not match layer embed_dim ({self.embed_dim})")

        # Linear projections for Q, K, V
        q = self.q_proj(x)  # (B, S, D)
        k = self.k_proj(x)  # (B, S, D)
        v = self.v_proj(x)  # (B, S, D)

        # Reshape and transpose for multi-head attention
        # (B, S, D) -> (B, S, num_heads, head_dim) -> (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q)
        k = apply_rope(k)
        q = q * (self.head_dim ** -0.5)

        # Compute attention scores: (B, num_heads, S, S)
        # (B, H, S, d_h) @ (B, H, d_h, S) -> (B, H, S, S)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # --- Apply masks ---
        # 1. Sliding window mask (causal + fixed window)
        # This mask is (S, S) and applies to all heads and batch items uniformly.
        combined_mask = self._sliding_window_mask(S, x.device) # (S, S)

        # 2. Optional additional attention mask (e.g., for specific blocked patterns)
        if attn_mask is not None:
            # Ensure attn_mask is boolean and combine with OR logic
            combined_mask = combined_mask | attn_mask.to(device=x.device, dtype=torch.bool)

        # Unsqueeze combined_mask to be broadcastable for batch and heads: (1, 1, S, S)
        combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)

        # 3. Key padding mask (masks attention *to* padded keys)
        if key_padding_mask is not None:
            # Expand key_padding_mask from (B, S) to (B, 1, 1, S)
            # This masks columns (keys) corresponding to padded input tokens.
            expanded_kpm = key_padding_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
            # Combine with OR: if any mask says block, then block.
            # combined_mask (1,1,S,S) | expanded_kpm (B,1,1,S) -> (B,1,S,S) by broadcasting rules
            combined_mask = combined_mask | expanded_kpm

        # Apply the combined mask to attention scores
        # attn_scores: (B, H, S, S)
        # combined_mask can be (S,S), (1,1,S,S) or (B,1,S,S). It will broadcast.
        attn_scores = attn_scores.masked_fill(combined_mask, float('-inf'))

        # Compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Compute weighted sum of values
        # (B, H, S, S) @ (B, H, S, d_h) -> (B, H, S, d_h)
        output = torch.matmul(attn_probs, v)

        # Concatenate heads and apply final projection
        # (B, H, S, d_h) -> (B, S, H, d_h) -> (B, S, D)
        output = output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        output = self.out_proj(output)

        return output


class SlidingWindowCrossAttention(nn.Module):
    """Sliding-window cross attention with separate query and key/value sequences."""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int,
                 bias: bool = True) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _cross_window_mask(self, q_len: int, kv_len: int,
                           device: torch.device) -> torch.Tensor:
        """Mask restricting queries to attend within a retrospective window."""
        return get_cross_window_mask(q_len, kv_len, self.window_size, device)

    def forward(self,
                query: torch.Tensor,  # codes_lo
                key: torch.Tensor,  # codes_hi
                value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S_q, D = query.shape
        if D != self.embed_dim or key.size(2) != self.embed_dim or value.size(2) != self.embed_dim:
            raise ValueError("embed_dim mismatch")
        S_k = key.size(1)

        q = self.q_proj(query).view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)
        q = q * (self.head_dim ** -0.5)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        combined_mask = self._cross_window_mask(S_q, S_k, query.device)

        if attn_mask is not None:
            combined_mask = combined_mask | attn_mask.to(device=query.device, dtype=torch.bool)

        combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)

        if key_padding_mask is not None:
            expanded_kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)
            combined_mask = combined_mask | expanded_kpm

        attn_scores = attn_scores.masked_fill(combined_mask, float('-inf'))
        attn_probs = safe_softmax(attn_scores, combined_mask, dim=-1)
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(B, S_q, self.embed_dim)
        output = self.out_proj(output)

        return output

# @torch.compile
class SlidingWindowTransformerBlock(nn.Module):
    """
    A single Transformer block using LocalSlidingWindowAttention (Pre-LN variant).

    The block consists of two main sub-layers:
    1. Multi-Head Self-Attention (LocalSlidingWindowAttention) with Pre-RMSNorm.
    2. Feed-Forward Network (FFN) with Pre-RMSNorm.
    Residual connections are applied after each sub-layer.

    Attributes:
        norm1 (nn.RMSNorm): RMS normalization before the attention layer.
        attn (LocalSlidingWindowAttention): The sliding window self-attention mechanism.
        norm2 (nn.RMSNorm): RMS normalization before the FFN.
        ffn (nn.Sequential): The feed-forward network.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int,
                 ffn_dim_multiplier: int = 4,
                 use_flex_attention: bool = True):
        super().__init__()
        # First sub-block: Sliding Window Multi-Head Attention
        self.norm1 = nn.RMSNorm(dim)
        if use_flex_attention:
            self.attn = LocalSlidingWindowAttention(
                embed_dim=dim,
                num_heads=num_heads,
                window_size=window_size,
            )
        else:
            self.attn = SlidingWindowAttention(
                embed_dim=dim,
                num_heads=num_heads,
                window_size=window_size,
            )

        # Second sub-block: Feed-Forward Network
        self.norm2 = nn.RMSNorm(dim)
        ffn_hidden_dim = dim * ffn_dim_multiplier
        self.ffn = SwiGLU(dim, ffn_hidden_dim)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
                Shape: (batch_size, sequence_length, embed_dim).
            key_padding_mask (Optional[torch.Tensor]): Boolean mask indicating
                padded tokens. Passed to the attention layer.
                Shape: (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of the block.
                Shape: (batch_size, sequence_length, embed_dim).
        """
        # Attention sub-layer (Pre-LN)
        # x_norm1 = self.norm1(x) # Normalize input
        # attn_output = self.attn(x_norm1, key_padding_mask=key_padding_mask) # Apply attention
        attn_output = self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + attn_output

        # Feed-forward sub-layer (Pre-LN)
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output

        return x

# @torch.compile
class StackedSlidingWindowEncoder(nn.Module):
    """
    A Transformer-style encoder composed of a stack of SlidingWindowTransformerBlock layers.

    It includes token embeddings, stacked Transformer blocks with RoPE,
    a final RMS normalization, and a projection to logits.

    Attributes:
        embedding (nn.Embedding): Token embedding layer.
        layers (nn.ModuleList): List of SlidingWindowTransformerBlock layers.
        final_norm (nn.RMSNorm): RMS normalization applied to the output of the
                                 last Transformer block before logit projection.
        logit_proj (nn.Linear): Linear layer to project Transformer output to vocabulary logits.
        use_flex_attention (bool): If True, uses the FlexAttention implementation
            for sliding window attention. Should typically be True only when CUDA
            is available.
    """
    def __init__(self, vocab_size: int, dim: int, num_heads: int, window_size: int,
                 num_layers: int, ffn_dim_multiplier: int = 4,
                 use_flex_attention: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            SlidingWindowTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                ffn_dim_multiplier=ffn_dim_multiplier,
                use_flex_attention=use_flex_attention
            ) for _ in range(num_layers)
        ])

        self.final_norm = nn.RMSNorm(dim) # Normalization before output projection
        self.logit_proj = nn.Linear(dim, vocab_size) # Projects to vocabulary size

    def forward(self, token_ids: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the encoder.

        Args:
            token_ids (torch.Tensor): Input tensor of token IDs.
                Shape: (batch_size, sequence_length).
            key_padding_mask (Optional[torch.Tensor]): Boolean mask indicating
                padded tokens in `token_ids`.
                Shape: (batch_size, sequence_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - hidden_states (torch.Tensor): Output hidden states from the last
              Transformer block. Shape: (batch_size, sequence_length, embed_dim).
            - logits (torch.Tensor): Output logits over the vocabulary.
              Shape: (batch_size, sequence_length, vocab_size).
        """
        B, S = token_ids.shape

        # 1. Token Embeddings
        x = self.embedding(token_ids) # (B, S, D)

        # 2. (RoPE applied inside attention; no absolute positional encoding)

        # 3. Pass through Transformer Layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
            # x is continuously updated by each layer

        # `x` is now the output of the last Transformer block.
        # This can be used as hidden states for downstream tasks (e.g., pooling).
        hidden_states_for_pooling = x

        # 4. Final Normalization and Logit Projection
        normed_output = self.final_norm(hidden_states_for_pooling)
        logits = self.logit_proj(normed_output)  # (B, S, V)

        return hidden_states_for_pooling, logits

# class SegmentCausalCrossAttention(nn.Module):
#     """
#     Cross-attention where each query token attends to its compressed
#     segment and up to `lookback` previous segments (causal).
#
#     During *training* you pass the whole sequence in one call.
#     During *generation* you call `forward` with `incremental_state`
#     (a dict) and a single-token query.
#
#     Args
#     ----
#     d_model   : model width (must divide `n_heads`)
#     n_heads   : number of heads
#     lookback  : how many *earlier* segments each query may see
#     dropout   : attention-probability dropout
#     """
#
#     def __init__(
#         self,
#         d_model:  int,
#         n_heads:  int,
#         lookback: int  = 0,
#         dropout:  float = 0.0,
#         bias:     bool  = True,
#     ):
#         super().__init__()
#         assert d_model % n_heads == 0
#         self.d_model   = d_model
#         self.n_heads   = n_heads
#         self.hdim      = d_model // n_heads
#         self.scale     = self.hdim ** -0.5
#         self.lookback  = lookback
#
#         self.q_proj = nn.Linear(d_model, d_model, bias=bias)
#         self.k_proj = nn.Linear(d_model, d_model, bias=bias)
#         self.v_proj = nn.Linear(d_model, d_model, bias=bias)
#         self.o_proj = nn.Linear(d_model, d_model, bias=bias)
#         self.drop   = nn.Dropout(dropout)
#
#     # ------------------------------------------------------------------ #
#     def forward(
#         self,
#         q: torch.Tensor,                # (B, Lq,  D) – OR (B, 1, D) at decode
#         kv_src: torch.Tensor,           # (B, Lkv, D) – compressed reps
#         seg_id: torch.Tensor,           # (B, Lq) int – mapping q → kv row
#         kv_mask: Optional[torch.Tensor] = None,    # (B, Lkv) bool
#         q_pad_mask: Optional[torch.Tensor] = None,  # (B, Lq) bool
#         incremental_state: Optional[Dict[str, torch.Tensor]] = None,
#     ):
#         """
#         incremental_state keys (all optional on first call):
#
#             "k"       : (B, H, Lkv_cache, Dh)
#             "v"       : (B, H, Lkv_cache, Dh)
#             "seg_ptr" : (B,)  last segment index already in cache
#         """
#
#         if incremental_state is None or q.size(1) > 1:          # ----- training path
#             out = self._full_forward(q, kv_src, seg_id, kv_mask)
#             out.masked_fill_(q_pad_mask.unsqueeze(-1), 0.0)  # zero out padded queries
#             return out
#
#         # ----------------------------------------------------- #
#         # Incremental decoding – q is shape (B, 1, D)
#         B = q.size(0)
#         device = q.device
#         k_cache = incremental_state.get("k")      # maybe None on first token
#         v_cache = incremental_state.get("v")
#         seg_ptr = incremental_state.get("seg_ptr")  # (B,) or None
#
#         # Project query
#         q_proj = self._split_heads(self.q_proj(q))       # (B,H,1,Dh)
#
#         # If this token belongs to a *new* segment: append K/V row to cache
#         new_seg = (seg_ptr is None) | (seg_id[:, -1] != seg_ptr)
#         if new_seg.any():
#             # project *one* kv row per *new* segment
#             kv_new = kv_src[torch.arange(B, device=device), seg_id[:, -1]]  # (B,D)
#
#             k_new  = self._split_heads(self.k_proj(kv_new.unsqueeze(1)))     # (B,H,1,Dh)
#             v_new  = self._split_heads(self.v_proj(kv_new.unsqueeze(1)))
#
#             if k_cache is None:                      # first ever row
#                 k_cache = k_new
#                 v_cache = v_new
#             else:                                    # append along Lkv axis
#                 k_cache = torch.cat([k_cache, k_new], dim=2)
#                 v_cache = torch.cat([v_cache, v_new], dim=2)
#
#             seg_ptr = seg_id[:, -1]                  # update pointer
#
#         # Gather indices: own segment and `lookback` previous ones
#         max_Lkv = k_cache.size(2)
#         offsets = torch.arange(0, self.lookback + 1, device=device)   # 0..lookback
#         gather_idx = seg_id[:, -1:].unsqueeze(-1) - offsets           # (B,1,Kw)
#         gather_idx.clamp_(0, max_Lkv - 1)
#         Kw = gather_idx.size(-1)
#
#         # Expand for heads, gather keys / values
#         gather_idx = gather_idx.expand(B, self.n_heads, 1, Kw)        # (B,H,1,Kw)
#         k_slice = k_cache.gather(2, gather_idx.unsqueeze(-1).expand(-1,-1,-1,-1,self.hdim))
#         v_slice = v_cache.gather(2, gather_idx.unsqueeze(-1).expand(-1,-1,-1,-1,self.hdim))
#         # shapes (B,H,1,Kw,Dh)
#
#         # Attention
#         scores = (q_proj.unsqueeze(-2) * k_slice).sum(-1) * self.scale  # (B,H,1,Kw)
#
#         if kv_mask is not None:
#             mask_slice = kv_mask.gather(1, gather_idx[:,0,0,:]).view(B,1,1,Kw)
#             scores.masked_fill_(mask_slice, torch.finfo(scores.dtype).min)
#
#         probs  = self.drop(F.softmax(scores, dim=-1))                   # (B,H,1,Kw)
#         out    = (probs.unsqueeze(-1) * v_slice).sum(-2)               # (B,H,1,Dh)
#
#         out = out.transpose(1,2).reshape(B,1,self.d_model)
#         out = self.o_proj(out)
#
#         # stash updated cache
#         incremental_state["k"]       = k_cache
#         incremental_state["v"]       = v_cache
#         incremental_state["seg_ptr"] = seg_ptr
#
#         return out

    # ------------------------------------------------------------------ #
    # def _full_forward(self, q, kv, seg_id, kv_mask):
    #     """Training / teacher-forcing path (whole sequence)."""
    #     B, Lq, _  = q.shape
    #     device    = q.device
    #
    #     r         = self.lookback
    #     qh = self._split_heads(self.q_proj(q))      # (B,H,Lq,Dh)
    #     kh = self._split_heads(self.k_proj(kv))     # (B,H,Lkv,Dh)
    #     vh = self._split_heads(self.v_proj(kv))
    #
    #     B, H, Lkv, Dh = kh.shape
    #
    #     # build gather indices for every position
    #     offsets = torch.arange(0, r+1, device=device)          # (Kw,)
    #     gather  = seg_id.unsqueeze(-1) - offsets               # (B,Lq,Kw)
    #     gather.clamp_(0, kv.size(1)-1)
    #     Kw      = gather.size(-1)
    #
    #     gather = gather.unsqueeze(1).expand(B, self.n_heads, Lq, Kw)   # (B,H,Lq,Kw)
    #     idx_exp = gather.unsqueeze(-1).expand(-1, -1, -1, -1, Dh)  # (B,H,Lq,Kw,Dh)
    #
    #     k_sel = torch.take_along_dim(
    #         kh.unsqueeze(3).expand(-1, -1, -1, Kw, -1),  # (B,H,Lkv,Kw,Dh)
    #         idx_exp,  # (B,H,Lq,Kw,Dh)
    #         dim=2
    #     )
    #
    #     v_sel = torch.take_along_dim(
    #         vh.unsqueeze(3).expand(-1, -1, -1, Kw, -1),  # (B,H,Lkv,Kw,Dh)
    #         idx_exp,  # (B,H,Lq,Kw,Dh)
    #         dim=2
    #     )
    #
    #     # k_sel  = kh.unsqueeze(3).gather(2, gather.unsqueeze(-1).expand(-1,-1,-1,-1,self.hdim))
    #     # v_sel  = vh.unsqueeze(3).gather(2, gather.unsqueeze(-1).expand(-1,-1,-1,-1,self.hdim))
    #
    #     scores = (qh.unsqueeze(-2) * k_sel).sum(-1) * self.scale       # (B,H,Lq,Kw)
    #
    #     if kv_mask is not None:
    #         mask_sel = kv_mask.gather(1, gather[:,0,0,:]).view(B,1,Lq,Kw)
    #         scores.masked_fill_(mask_sel, torch.finfo(scores.dtype).min)
    #
    #     probs = self.drop(F.softmax(scores, dim=-1))
    #     out   = (probs.unsqueeze(-1) * v_sel).sum(-2)                  # (B,H,Lq,Dh)
    #     out   = out.transpose(1,2).reshape(B,Lq,self.d_model)
    #     return self.o_proj(out)
    #
    # def _full_forward(
    #         self,
    #         q: torch.Tensor,  # (B, Lq,  D)
    #         kv: torch.Tensor,  # (B, Lkv, D)
    #         seg_id: torch.Tensor,  # (B, Lq)   int
    #         kv_mask: Optional[torch.Tensor] = None  # (B, Lkv) bool
    # ) -> torch.Tensor:
    #     """
    #     Training / teacher-forcing path (whole sequence at once).
    #     Each query attends to its own compressed segment plus the previous
    #     `self.lookback` segments (causal).
    #
    #     Returns:
    #         out : (B, Lq, D)
    #     """
    #     B, Lq, _ = q.shape
    #     _, Lkv, _ = kv.shape
    #     R = self.lookback  # look-back radius
    #     Dh = self.hdim
    #     device = q.device
    #
    #     # 1. project + split heads
    #     qh = self._split_heads(self.q_proj(q))  # (B, H, Lq, Dh)
    #     kh = self._split_heads(self.k_proj(kv))  # (B, H, Lkv, Dh)
    #     vh = self._split_heads(self.v_proj(kv))  # (B, H, Lkv, Dh)
    #
    #     # 2. build (B, Lq, Kw) index tensor of segment rows to fetch
    #     offsets = torch.arange(0, R + 1, device=device)  # (Kw,)
    #     gather = seg_id.unsqueeze(-1) - offsets  # (B, Lq, Kw)
    #     gather.clamp_(0, Lkv - 1)  # keep in-range
    #     Kw = gather.size(-1)
    #
    #     # 3. expand for heads and Dh so we can gather k/v
    #     idx_exp = gather.unsqueeze(1).unsqueeze(-1)  # (B,1,Lq,Kw,1)
    #     idx_exp = idx_exp.expand(-1, self.n_heads, -1, -1, Dh)  # (B,H,Lq,Kw,Dh)
    #
    #     kh_exp = kh.unsqueeze(3).expand(-1, -1, -1, Kw, -1)  # (B,H,Lkv, Kw,Dh)
    #     vh_exp = vh.unsqueeze(3).expand(-1, -1, -1, Kw, -1)
    #
    #     k_sel = torch.gather(kh_exp, 2, idx_exp)  # (B,H,Lq,Kw,Dh)
    #     v_sel = torch.gather(vh_exp, 2, idx_exp)  # (B,H,Lq,Kw,Dh)
    #
    #     # 4. scaled dot-prod attention
    #     scores = (qh.unsqueeze(-2) * k_sel).sum(-1) * self.scale  # (B,H,Lq,Kw)
    #
    #     # 5. validity / padding mask on K/V rows (optional)
    #     if kv_mask is not None:
    #         # gather kv_mask for every (B,Lq,Kw)
    #         mask_flat = kv_mask.gather(  # (B, Lq*Kw)
    #             1,
    #             gather.reshape(B, -1)
    #         )
    #         mask_sel = mask_flat.reshape(B, Lq, Kw).unsqueeze(1)  # (B,1,Lq,Kw)
    #         scores.masked_fill_(mask_sel, torch.finfo(scores.dtype).min)
    #
    #     # 6. softmax & value mix
    #     probs = self.drop(torch.softmax(scores, dim=-1))  # (B,H,Lq,Kw)
    #     out = (probs.unsqueeze(-1) * v_sel).sum(-2)  # (B,H,Lq,Dh)
    #
    #     # 7. merge heads & final projection
    #     out = out.transpose(1, 2).reshape(B, Lq, self.d_model)  # (B,Lq,D)
    #     return self.o_proj(out)
    #
    # # ------------------------------------------------------------------ #
    # def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
    #     B, L, _ = x.shape
    #     return x.view(B, L, self.n_heads, self.hdim).transpose(1, 2)


class SegmentCausalCrossAttention(nn.Module):
    """
    Query width ≠ KV width version.
    q_dim     : embedding dim of query stream
    kv_dim    : embedding dim of compressed memory
    d_attn    : internal attention dim (must % n_heads == 0)
    n_heads   : number of heads
    lookback  : how many earlier segments a query can see
    """

    def __init__(
        self,
        q_dim:     int,
        kv_dim:    int,
        d_attn:    int,
        n_heads:   int,
        lookback:  int  = 0,
        dropout:   float = 0.0,
        bias:      bool  = True,
    ):
        super().__init__()
        assert d_attn % n_heads == 0, "d_attn must be divisible by n_heads"
        self.q_dim   = d_attn
        self.kv_dim  = d_attn
        self.d_attn  = d_attn
        self.n_heads = n_heads
        self.hdim    = d_attn // n_heads
        self.scale   = self.hdim ** -0.5
        self.lookback = lookback

        # separate input projections
        self.q_proj = nn.Linear(d_attn,  d_attn, bias=bias)
        self.k_proj = nn.Linear(d_attn, d_attn, bias=bias)
        self.v_proj = nn.Linear(d_attn, d_attn, bias=bias)

        # output goes back to the *query* dimensionality
        self.o_proj = nn.Linear(d_attn, d_attn, bias=bias)
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,                # (B, Lq,  D) – OR (B, 1, D) at decode
        kv_src: torch.Tensor,           # (B, Lkv, D) – compressed reps
        seg_id: torch.Tensor,           # (B, Lq) int – mapping q → kv row
        kv_mask: Optional[torch.Tensor] = None,    # (B, Lkv) bool
        q_pad_mask: Optional[torch.Tensor] = None,  # (B, Lq) bool
        incremental_state: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        incremental_state keys (all optional on first call):

            "k"       : (B, H, Lkv_cache, Dh)
            "v"       : (B, H, Lkv_cache, Dh)
            "seg_ptr" : (B,)  last segment index already in cache
        """

        if incremental_state is None or q.size(1) > 1:          # ----- training path
            out = self._full_forward(q, kv_src, seg_id, kv_mask)
            out.masked_fill_(q_pad_mask.unsqueeze(-1), 0.0)  # zero out padded queries
            return out

        # ----------------------------------------------------- #
        # Incremental decoding – q is shape (B, 1, D)
        B = q.size(0)
        device = q.device
        k_cache = incremental_state.get("k")      # maybe None on first token
        v_cache = incremental_state.get("v")
        seg_ptr = incremental_state.get("seg_ptr")  # (B,) or None

        # Project query
        q_proj = self._split_heads(self.q_proj(q))       # (B,H,1,Dh)

        # If this token belongs to a *new* segment: append K/V row to cache
        new_seg = (seg_ptr is None) | (seg_id[:, -1] != seg_ptr)
        if new_seg.any():
            # project *one* kv row per *new* segment
            kv_new = kv_src[torch.arange(B, device=device), seg_id[:, -1]]  # (B,D)

            k_new  = self._split_heads(self.k_proj(kv_new.unsqueeze(1)))     # (B,H,1,Dh)
            v_new  = self._split_heads(self.v_proj(kv_new.unsqueeze(1)))

            if k_cache is None:                      # first ever row
                k_cache = k_new
                v_cache = v_new
            else:                                    # append along Lkv axis
                k_cache = torch.cat([k_cache, k_new], dim=2)
                v_cache = torch.cat([v_cache, v_new], dim=2)

            seg_ptr = seg_id[:, -1]                  # update pointer

        # Gather indices: own segment and `lookback` previous ones
        max_Lkv = k_cache.size(2)
        offsets = torch.arange(0, self.lookback + 1, device=device)   # 0..lookback
        gather_idx = seg_id[:, -1:].unsqueeze(-1) - offsets           # (B,1,Kw)
        gather_idx.clamp_(0, max_Lkv - 1)
        Kw = gather_idx.size(-1)

        # Expand for heads, gather keys / values
        gather_idx = gather_idx.expand(B, self.n_heads, 1, Kw)        # (B,H,1,Kw)
        k_slice = k_cache.gather(2, gather_idx.unsqueeze(-1).expand(-1,-1,-1,-1,self.hdim))
        v_slice = v_cache.gather(2, gather_idx.unsqueeze(-1).expand(-1,-1,-1,-1,self.hdim))
        # shapes (B,H,1,Kw,Dh)

        # Attention
        scores = (q_proj.unsqueeze(-2) * k_slice).sum(-1) * self.scale  # (B,H,1,Kw)

        if kv_mask is not None:
            mask_slice = kv_mask.gather(1, gather_idx[:,0,0,:]).view(B,1,1,Kw)
            scores.masked_fill_(mask_slice, torch.finfo(scores.dtype).min)

        probs  = self.drop(F.softmax(scores, dim=-1))                   # (B,H,1,Kw)
        out    = (probs.unsqueeze(-1) * v_slice).sum(-2)               # (B,H,1,Dh)

        out = out.transpose(1,2).reshape(B,1,self.d_model)
        out = self.o_proj(out)

        # stash updated cache
        incremental_state["k"]       = k_cache
        incremental_state["v"]       = v_cache
        incremental_state["seg_ptr"] = seg_ptr

        return out

    # ----------------------------- training / teacher-forcing ---------- #
    def _full_forward(
        self,
        q:       torch.Tensor,              # (B, Lq,  q_dim)
        kv:      torch.Tensor,             # (B, Lkv, kv_dim)
        seg_id:  torch.Tensor,             # (B, Lq)
        kv_mask: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:

        B, Lq, _  = q.shape
        _, Lkv, _ = kv.shape
        R, H, Dh  = self.lookback, self.n_heads, self.hdim
        device    = q.device

        # 1-- projections
        qh = self._split_heads(self.q_proj(q))      # (B,H,Lq,Dh)
        kh = self._split_heads(self.k_proj(kv))     # (B,H,Lkv,Dh)
        vh = self._split_heads(self.v_proj(kv))

        # 2-- segment indices (B,Lq,Kw)
        offsets = torch.arange(0, R + 1, device=device)          # (Kw,)
        gather  = seg_id.unsqueeze(-1) - offsets                 # (B,Lq,Kw)
        gather.clamp_(0, Lkv-1)
        Kw      = gather.size(-1)

        # 3-- take_along_dim handles broadcasting
        idx = gather.unsqueeze(1).unsqueeze(-1)                  # (B,1,Lq,Kw,1)

        k_sel = torch.take_along_dim(                            # (B,H,Lq,Kw,Dh)
            kh.unsqueeze(3), idx.expand(-1,H,-1,-1,Dh), dim=2
        )
        v_sel = torch.take_along_dim(
            vh.unsqueeze(3), idx.expand(-1,H,-1,-1,Dh), dim=2
        )

        # 4-- attention
        scores = (qh.unsqueeze(-2) * k_sel).sum(-1) * self.scale # (B,H,Lq,Kw)

        if kv_mask is not None:
            mask_sel = torch.take_along_dim(
                kv_mask.unsqueeze(2), gather, dim=1              # (B,Lq,Kw)
            ).unsqueeze(1)                                       # (B,1,Lq,Kw)
            scores.masked_fill_(mask_sel, torch.finfo(scores.dtype).min)

        probs = self.drop(torch.softmax(scores, dim=-1))         # (B,H,Lq,Kw)
        out   = (probs.unsqueeze(-1) * v_sel).sum(-2)            # (B,H,Lq,Dh)

        # 5-- merge heads, project back to q_dim
        out = out.transpose(1, 2).reshape(B, Lq, self.d_attn)
        return self.o_proj(out)

    # ---------------------- helpers -------------------- #
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.hdim).transpose(1, 2)
