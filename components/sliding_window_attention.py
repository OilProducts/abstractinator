from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,  # helper to build sparse mask
    flex_attention,  # fused kernel
)

from .rope import apply_rope




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
        cache = LocalSlidingWindowAttention._sliding_block.__dict__.setdefault("cache", {})
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
            _compile=False,  # pre‑compile sparse metadata
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
        x: torch.Tensor,  # (B, S, D)
        key_padding_mask: torch.Tensor | None = None,  # (B, S) – True = pad
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
        q = q * (d**-0.5)

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
            block_mask=block_mask,  # ← mandatory kw‑arg
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

    def __init__(self, embed_dim: int, num_heads: int, window_size: int, bias: bool = True) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size  # Number of *previous* tokens query can attend to

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

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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
        B, S, D = x.shape  # D is embed_dim
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
        q = q * (self.head_dim**-0.5)

        # Compute attention scores: (B, num_heads, S, S)
        # (B, H, S, d_h) @ (B, H, d_h, S) -> (B, H, S, S)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # --- Apply masks ---
        # 1. Sliding window mask (causal + fixed window)
        # This mask is (S, S) and applies to all heads and batch items uniformly.
        combined_mask = self._sliding_window_mask(S, x.device)  # (S, S)

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
            expanded_kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
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


class AttnCache:
    """Simple container so we don’t juggle nested dicts."""

    def __init__(self):
        self.self_k = self.self_v = None  # self-attn keys/vals
        self.cross = {}  # layer_name → {k,v,seg_ptr}


class SegmentCausalCrossAttention(nn.Module):
    """
    Cross-attention where queries attend to a *compressed* memory of segments.

    q_dim     : embedding dim of the query stream (original sequence)
    kv_dim    : embedding dim of the compressed memory stream
    d_attn    : internal attention width; must be divisible by n_heads
    n_heads   : number of attention heads
    lookback  : number of *previous* segments visible in addition to the current one
    """

    def __init__(
            self,
            q_dim: int,
            kv_dim: int,
            d_attn: int,
            n_heads: int,
            lookback: int = 0,
            dropout: float = 0.0,
            bias: bool = False,
    ):
        super().__init__()
        assert d_attn % n_heads == 0, "d_attn must be divisible by n_heads"

        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.d_attn = d_attn
        self.n_heads = n_heads
        self.hdim = d_attn // n_heads
        self.scale = self.hdim ** -0.5
        self.lookback = int(lookback)

        # Projections
        self.q_proj = nn.Linear(q_dim, d_attn, bias=bias)
        self.kv_proj = nn.Linear(kv_dim, 2 * d_attn, bias=bias)
        self.o_proj = nn.Linear(d_attn, q_dim, bias=bias)
        self.drop = nn.Dropout(dropout)

    # ----------------------------- public forward ----------------------------- #
    def forward(
            self,
            q: torch.Tensor,  # (B, Lq, q_dim)
            kv_src: torch.Tensor,  # (B, Lkv, kv_dim)
            seg_id: torch.Tensor,  # (B, Lq) int
            kv_mask: Optional[torch.Tensor] = None,  # (B, Lkv) bool – True means "mask out"
            q_pad_mask: Optional[torch.Tensor] = None,  # (B, Lq) bool – True means padding
            cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Cache protocol (optional, read-only inside compiled graph):
            cache["k"] : (B, H, Lkv, Dh)
            cache["v"] : (B, H, Lkv, Dh)

        Notes
        -----
        • To avoid graph breaks, we do not *write* into `cache` here. Populate/maintain it
          outside the compiled region if you need decode-time reuse.
        """
        B, Lq, _ = q.shape
        device = q.device
        H, Dh = self.n_heads, self.hdim

        # 1) Q projection & split heads: (B,H,Lq,Dh)
        qh = self._split_heads(self.q_proj(q))

        # 2) K/V: choose cached vs recompute without graph breaks.
        #    We build a tensor predicate and use torch.cond.
        # Build predicate as a 0-d bool tensor
        has_cache_py = (
                cache is not None
                and isinstance(cache.get("k", None), torch.Tensor)
                and isinstance(cache.get("v", None), torch.Tensor)
        )
        pred = q.new_tensor(has_cache_py, dtype=torch.bool)

        B, H, Dh = q.shape[0], self.n_heads, self.hdim
        Lkv_src = kv_src.size(1)

        # Provide operands explicitly so both branches have identical signatures.
        kh_in = cache["k"] if has_cache_py else q.new_empty((B, H, Lkv_src, Dh), dtype=q.dtype, device=q.device)
        vh_in = cache["v"] if has_cache_py else q.new_empty((B, H, Lkv_src, Dh), dtype=q.dtype, device=q.device)

        # def _true(kv_src_, kh_c, vh_c):
        #     # Use cached, but force a consistent contiguous layout
        #     return kh_c.contiguous(), vh_c.contiguous()
        #
        # # sliding_window_attention.py

        def _true(kv_src, kh_in, vh_in):
            # torch.cond requires outputs not to alias inputs; contiguous() may return self.
            # clone(..., contiguous_format) guarantees new storage and contiguous layout.
            kh_c = kh_in.clone(memory_format=torch.contiguous_format)
            vh_c = vh_in.clone(memory_format=torch.contiguous_format)
            return kh_c, vh_c

        def _false(kv_src_, kh_c, vh_c):
            # Recompute and return contiguous via _split_heads()
            kv = self.kv_proj(kv_src_)  # (B, Lkv, 2*D)
            k, v = kv.split(self.d_attn, dim=-1)  # (B, Lkv, D), (B, Lkv, D)
            return self._split_heads(k), self._split_heads(v)  # already contiguous

        # kh, vh = torch.cond(pred, _true, _false, (kv_src, kh_in, vh_in))
        if pred:
            kh, vh = _true(kv_src, kh_in, vh_in)
        else:
            kh, vh = _false(kv_src, kh_in, vh_in)

        Lkv = kh.size(2)
        Kw = self.lookback + 1

        # 3) Per-query segment window indices: (B, Lq, Kw)
        offsets = torch.arange(0, Kw, device=device)  # (Kw,)
        gather = seg_id.unsqueeze(-1) - offsets  # (B,Lq,Kw)
        neg_mask = gather < 0  # (B,Lq,Kw)
        gather_clamped = torch.clamp(gather, 0, Lkv - 1)  # (B,Lq,Kw)

        # 4) Select K/V windows with take_along_dim over Lkv.
        idx = gather_clamped.unsqueeze(1).unsqueeze(-1)  # (B,1,Lq,Kw,1)
        k_sel = torch.take_along_dim(kh.unsqueeze(3), idx.expand(-1, H, -1, -1, 1), dim=2)  # (B,H,Lq,Kw,Dh)
        v_sel = torch.take_along_dim(vh.unsqueeze(3), idx.expand(-1, H, -1, -1, 1), dim=2)  # (B,H,Lq,Kw,Dh)

        # 5) Attention scores over the window: (B,H,Lq,Kw)
        scores = (qh.unsqueeze(-2) * k_sel).sum(dim=-1) * self.scale

        # 6) Masks — always applied (no Python conditionals).
        #    a) segment padding mask projected into window space: (B,Lq,Kw)
        if kv_mask is None:
            kvm_win = torch.zeros((B, Lq, Kw), dtype=torch.bool, device=device)
        else:
            kv_mask_b = kv_mask.to(device=device, dtype=torch.bool)  # (B,Lkv)
            b_idx = torch.arange(B, device=device)[:, None, None]  # (B,1,1)
            kvm_win = kv_mask_b[b_idx, gather_clamped]  # (B,Lq,Kw)

        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(kvm_win.unsqueeze(1), neg_inf)  # (B,H,Lq,Kw)
        #    b) negative window positions
        scores = scores.masked_fill(neg_mask.unsqueeze(1), neg_inf)  # (B,H,Lq,Kw)

        # 7) Softmax + value mix
        probs = self.drop(F.softmax(scores, dim=-1))  # (B,H,Lq,Kw)
        out = (probs.unsqueeze(-1) * v_sel).sum(dim=-2)  # (B,H,Lq,Dh)

        # 8) Merge heads, project, and apply query padding mask (always applied)
        out = out.transpose(1, 2).reshape(B, Lq, self.d_attn)  # (B,Lq,D)
        out = self.o_proj(out)  # (B,Lq,q_dim)

        if q_pad_mask is None:
            qpm = torch.zeros((B, Lq), dtype=torch.bool, device=device)
        else:
            qpm = q_pad_mask.to(device=device, dtype=torch.bool)

        out = out.masked_fill(qpm.unsqueeze(-1), 0.0)
        return out

    # ----------------------------- helpers ----------------------------- #
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, H, L, Dh) with a consistent contiguous layout
        B, L, _ = x.shape
        return (
            x.reshape(B, L, self.n_heads, self.hdim)  # safer than view when non-contiguous
            .permute(0, 2, 1, 3)  # (B,H,L,Dh)
            .contiguous()  # ensure standard (row-major) stride
        )

