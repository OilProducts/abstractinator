import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks

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



# class MultiheadLatentAttention(nn.Module):
#     """
#     DeepSeek‑V3 style Multi‑Head Latent Attention with asymmetric compression
#     and optional FlexAttention acceleration + arbitrary masking.
#     -----------------------------------------------------------------------
#       latent  (K=128)  →  Q_big  (d_c' = 1536)
#                        →  Q_kv   (d_c  = 512)
#       K/V     (d_c)    →  retrieval K_r  (r = 64)
#       Q_big   (d_c')   →  retrieval Q_r  (r = 64)
#       scores  : <Q_r , K_r>  (√r scale baked into W_qr)
#       values  : V  =  K/V_compressed   (d_c)
#     """
#
#     def __init__(
#         self,
#         dim_q: int = 384,  # d_model
#         num_heads: int = 6,  # H
#         head_dim: int = 64,  # K
#         kv_comp_dim: int = 48,  # d_c/kv_c/latent dim
#         q_comp_dim: int = 96,  # d_c'/query_dim
#         retr_dim: int = 16,  # r/rope_dim/scoring_dim
#         score_mod: Callable | None = None,  # Flex mask/bias callback
#         use_flex_attention: bool = True,
#     ):
#         super().__init__()
#         self.h = num_heads
#         self.k = head_dim
#         self.d_c = kv_comp_dim
#         self.d_cq = q_comp_dim
#         self.r = retr_dim
#
#         # latent → K‑space
#         self.q_proj = nn.Linear(dim_q, self.h * self.k, bias=False)
#
#         # absorbed projections to compressed spaces
#         self.w_kc_q = nn.Parameter(torch.empty(self.h, self.k, self.d_cq))  # K→d_c'
#         self.w_kc_kv = nn.Parameter(torch.empty(self.h, self.k, self.d_c))  # K→d_c
#
#         # retrieval adapters
#         self.W_qr = nn.Parameter(torch.empty(self.h, self.d_cq, self.r))
#         self.W_kr = nn.Parameter(torch.empty(self.h, self.d_c, self.r))
#
#         # output
#         self.out_proj = nn.Linear(self.h * self.k, dim_q, bias=False)
#
#         self._init_weights()
#
#         # pick dtype/device from a real param so it follows .to() and AMP
#         param = self.q_proj.weight
#         self.rope = RoPECache(
#             max_seqlen=2048,
#             head_dim=self.r,  # <-- rotate in retrieval space
#             device=param.device,
#             dtype=param.dtype,
#         )
#
#         self.score_mod = score_mod or (lambda s, *_: s)  # identity if none
#         self.use_flex_attention = use_flex_attention
#         if self.use_flex_attention:
#             self._attn = self._flex_attention
#         else:
#             self._attn = self._fallback_attention
#
#         self._pad_for_scores = None  # [B, S] bool or None
#         self._q_seg_for_scores = None  # [B, Q] int or None
#         self._seg_id_for_scores = None  # [B, S] int or None
#
#     # ------------------------------------------------------------------ #
#     # @torch.no_grad()
#     def _init_weights(self):
#         for p in (self.w_kc_q, self.w_kc_kv, self.W_qr, self.W_kr):
#             nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#         # bake √r into W_qr  ⇒ no runtime scaling needed
#         with torch.no_grad():
#             self.W_qr.div_(math.sqrt(self.r))
#
#     # clear RoPE tables when dtype/device flip
#     def load_state_dict(self, *a, **kw):
#         ret = super().load_state_dict(*a, **kw)
#         # _RoPECache.clear()
#         return ret
#
#     def _score_mod_with_pad(self, scores, b, h, q, k):
#
#         # user-provided score_mod first, if any
#         if self.score_mod is not None and self.score_mod is not self._score_mod_with_pad:
#             scores = self.score_mod(scores, b, h, q, k)
#
#         # 1) padding (what you already had)
#         if self._pad_for_scores is not None:
#             scores = scores.masked_fill(self._pad_for_scores[b, k], float("-inf"))
#
#         # 2) segment gating (new)
#         if self._q_seg_for_scores is not None and self._seg_id_for_scores is not None:
#             # bad if key's seg != query's seg
#             bad = self._seg_id_for_scores[b, k] != self._q_seg_for_scores[b, q]
#             scores = scores.masked_fill(bad, float("-inf"))
#
#         return scores
#
#     def _flex_attention(self, q_r, k_r, v_c, *, block_mask: torch.Tensor | None = None):
#         # NOTE: we now always pass score_mod, even when block_mask is present.
#         return flex_attention(
#             q_r,
#             k_r,
#             v_c,
#             score_mod=self._score_mod_with_pad,
#             block_mask=block_mask,
#         )
#
#     def _fallback_attention(
#         self,
#         q_r: torch.Tensor,  # [B, H, S, r]
#         k_r: torch.Tensor,  # [B, H, L, r]
#         v_c: torch.Tensor,  # [B, H, L, d_c]
#         *,
#         block_mask: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Light‑weight dot‑product attention used when FlexAttention is unavailable.
#
#         Returns
#         -------
#         ctx_c : torch.Tensor, shape [B, H, d_c]
#             Per‑head context vectors in compressed space.
#         """
#         # q_r: [B,H,S,r]   k_r: [B,H,L,r]  →  scores: [B,H,S,L]
#         scores = torch.matmul(q_r, k_r.transpose(-2, -1))  # [B, H, S, L]
#
#         # -- 2. user‑supplied bias / mask operates on raw logits ---------------
#         scores = self.score_mod(scores)
#
#         mask = torch.zeros_like(scores, dtype=torch.bool)
#         if block_mask is not None:
#             if hasattr(block_mask, 'mask_mod'):
#                 B, H, _, L = scores.shape
#                 q_idx = torch.arange(L, device=scores.device)
#                 k_idx = q_idx[:, None]
#                 keep = block_mask.mask_mod(0, 0, q_idx[:, None], k_idx[:, None])
#                 scores = scores.masked_fill(keep, -float('inf'))
#                 mask = mask | keep
#             else:
#                 scores = scores.masked_fill(block_mask, -float('inf'))
#                 mask = mask | block_mask
#
#         # -- 3. softmax & value aggregation ------------------------------------
#         attn = safe_softmax(scores, mask, dim=-1)  # [B, H, S, L]
#
#         # fused gemm: (B·H·S)×L @ L×d_c  →  (B·H·S)×d_c
#         return torch.matmul(attn, v_c)  # [B, H, S, d_c]
#
#     def forward(self, hidden_q: torch.Tensor, kv_c: torch.Tensor, block_mask=None) -> torch.Tensor:
#         B, S, _ = hidden_q.shape
#         _, L, _d = kv_c.shape
#         dev, dt = kv_c.device, kv_c.dtype
#
#         q_hk = self.q_proj(hidden_q).view(B, S, self.h, self.k)  # [B,S,H,K]
#         q_big = torch.einsum('bshk,hkq->bshq', q_hk, self.w_kc_q)  # [B,S,H,d_c′]
#
#         # Project into the actual scoring space r
#         q_r = torch.einsum('bshq,hqr->bshr', q_big, self.W_qr)  # [B,S,H,r]
#         q_r = q_r.permute(0, 2, 1, 3)  # [B,H,S,r]
#
#         k_r = torch.einsum('bld,hdr->bhlr', kv_c, self.W_kr)  # [B,H,L,r]
#
#         # --- RoPE in retrieval space (the dot-product space) ---
#         cos_S, sin_S = self.rope.slice(S)  # (1,1,S,r/2)
#         q_r = apply_rope(q_r, cos_S, sin_S)  # [B,H,S,r]
#
#         cos_L, sin_L = self.rope.slice(L)  # (1,1,L,r/2)
#         k_r = apply_rope(k_r, cos_L, sin_L)  # [B,H,L,r]
#
#         v_c = kv_c.unsqueeze(1).expand(-1, self.h, -1, -1)  # [B,H,L,d_c]
#
#         # 5) attention
#         ctx_c = self._attn(q_r, k_r, v_c, block_mask=block_mask)  # [B,H,d_c]
#
#         # 6) compressed → latent
#         ctx_lat = torch.einsum('bhsd,hdK->bhsK', ctx_c, self.w_kc_kv.transpose(1, 2))  # [B,H,S,K]
#         ctx_lat = ctx_lat.permute(0, 2, 1, 3).reshape(B, S, -1)  # [B,S,H*K]
#
#         # 7) output
#         return self.out_proj(ctx_lat)  # [B,D]
#
#     def cache_append(self, kv_c_cache, k_r_cache, kv_c_new, pos):
#         # kv_c_new: (B, 1, c)
#         # 1) build k_r for JUST the new token
#         #    (B,1,c) x (H,c,r) → (B,H,1,r)
#         k_r_new = torch.einsum('bld,hdr->bhlr', kv_c_new, self.W_kr)
#
#         # 2) RoPE at absolute position "pos"
#         #    Add an API like: cos,sin = self.rope.at(pos, device=dtype=kv_c_new.dtype)
#         cos, sin = self.rope.at(pos)  # shapes: (1,1,1,r/2)
#         k_r_new = apply_rope(k_r_new, cos, sin)
#
#         # 3) append (use your own paged-ring; concat here for clarity)
#         kv_c_cache = torch.cat([kv_c_cache, kv_c_new], dim=1)  # (B,L+1,c)
#         k_r_cache = torch.cat([k_r_cache, k_r_new], dim=2)  # (B,H,L+1,r)
#         return kv_c_cache, k_r_cache