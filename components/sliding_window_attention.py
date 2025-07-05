import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import safe_softmax
from torch.nn.attention.flex_attention import (
    flex_attention,           # fused kernel
    create_block_mask         # helper to build sparse mask
)
from .rope import apply_rope
from .swiglu import SwiGLU
from typing import Optional, Tuple


def _cached_cross_window_mask(
    q_len: int, kv_len: int, window: int, device: torch.device
) -> torch.Tensor:
    """Return a cached mask for the usable slice of each query position."""

    cache_key = (q_len, kv_len, window, device.index if device.type == "cuda" else -1)
    cache = _cached_cross_window_mask.__dict__.setdefault("cache", {})
    if cache_key in cache:
        return cache[cache_key]

    q_pos = torch.arange(q_len, device=device).unsqueeze(1)
    offsets = torch.arange(-window, 1, device=device).unsqueeze(0)
    idx = q_pos + offsets
    mask = (idx < 0) | (idx >= kv_len)
    cache[cache_key] = mask
    return mask

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
        ).to(device)

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
        q = self.q_proj(x).view(B, S, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, d).transpose(1, 2)
        q = apply_rope(q)
        k = apply_rope(k)
        q = q * (d ** -0.5)

        # block mask ----------------------------------------------------------
        block_mask = self._sliding_block(S, self.window, x.device, self.training)
        if key_padding_mask is not None:
            block_mask = block_mask.add_padding_mask(key_padding_mask)

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
        return _cached_cross_window_mask(q_len, kv_len, self.window_size, device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S_q, D = query.shape
        if D != self.embed_dim or key.size(2) != self.embed_dim or value.size(2) != self.embed_dim:
            raise ValueError("embed_dim mismatch")
        S_k = key.size(1)

        H, d = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, S_q, H, d).transpose(1, 2)
        k = self.k_proj(key).view(B, S_k, H, d).transpose(1, 2)
        v = self.v_proj(value).view(B, S_k, H, d).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)
        q = q * (d ** -0.5)

        # --- gather sliding windows for K and V ---
        pad_left = self.window_size
        pad_right = max(S_q - S_k, 0)
        k_pad = F.pad(k, (0, 0, pad_left, pad_right))
        v_pad = F.pad(v, (0, 0, pad_left, pad_right))
        k_win = k_pad.unfold(2, pad_left + 1, 1)[:, :, :S_q].permute(0, 1, 2, 4, 3)
        v_win = v_pad.unfold(2, pad_left + 1, 1)[:, :, :S_q].permute(0, 1, 2, 4, 3)

        # Attention scores only over local windows
        attn_scores = (q.unsqueeze(-2) * k_win).sum(-1)

        # Build mask for positions outside the valid kv range
        local_mask = self._cross_window_mask(S_q, S_k, query.device)
        mask = local_mask.unsqueeze(0).expand(B, -1, -1)
        if key_padding_mask is not None:
            base = torch.arange(S_q, device=query.device).unsqueeze(1)
            offsets = torch.arange(-self.window_size, 1, device=query.device)
            gather_idx = (base + offsets).clamp(min=0, max=S_k - 1)
            kpm = key_padding_mask[:, gather_idx]
            mask = mask | kpm

        mask = mask.unsqueeze(1)  # (B,1,S_q,w)

        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_probs = safe_softmax(attn_scores, mask, dim=-1)

        ctx = (attn_probs.unsqueeze(-1) * v_win).sum(-2)
        ctx = ctx.transpose(1, 2).contiguous().view(B, S_q, self.embed_dim)
        ctx = self.out_proj(ctx)
        return ctx

# @torch.compile
class SlidingWindowTransformerBlock(nn.Module):
    """
    A single Transformer block using SlidingWindowAttention (Pre-LN variant).

    The block consists of two main sub-layers:
    1. Multi-Head Self-Attention (SlidingWindowAttention) with Pre-RMSNorm.
    2. Feed-Forward Network (FFN) with Pre-RMSNorm.
    Residual connections are applied after each sub-layer.

    Attributes:
        norm1 (nn.RMSNorm): RMS normalization before the attention layer.
        attn (SlidingWindowAttention): The sliding window self-attention mechanism.
        norm2 (nn.RMSNorm): RMS normalization before the FFN.
        ffn (nn.Sequential): The feed-forward network.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int,
                 ffn_dim_multiplier: int = 4):
        super().__init__()
        # First sub-block: Sliding Window Multi-Head Attention
        self.norm1 = nn.RMSNorm(dim)
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
        # x_norm2 = self.norm2(x) # Normalize input to FFN
        # ffn_output = self.ffn(x_norm2) # Apply FFN
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
    """
    def __init__(self, vocab_size: int, dim: int, num_heads: int, window_size: int,
                 num_layers: int, ffn_dim_multiplier: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            SlidingWindowTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                ffn_dim_multiplier=ffn_dim_multiplier
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
