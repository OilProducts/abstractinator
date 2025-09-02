from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Optional, Tuple

import torch
from torch import nn, Tensor
import torch._dynamo as dynamo
from torch.nn.attention.flex_attention import and_masks, create_block_mask, flex_attention

from components.rope import RoPECache, apply_rope
from components.swiglu import SwiGLU
from components.utils import safe_softmax

class MultiheadLatentAttention(nn.Module):
    """
    DeepSeek‑style Multi‑Head Latent Attention with:
      • Asymmetric compression (shared latent c and retrieval r)
      • Optional FlexAttention acceleration + arbitrary masking
      • Optional fused query/output paths for inference
      • Streaming caches: cache both kv_c (latent) and k_r (RoPE'd keys)

    Naming (this class  ⇄  MLA paper / prior discussion):
      dim_q         ⇄  d_model
      num_heads     ⇄  H
      head_dim (K)  ⇄  d_head
      kv_comp_dim   ⇄  c        (shared latent, cached once per token)
      q_comp_dim    ⇄  d_c'     (query compression; not L‑scaled at decode)
      retr_dim (r)  ⇄  d_rope   (retrieval/scoring space; RoPE is applied here)

    Forward modes
    -------------
    • Training / block mode (default):
        forward(hidden_q, kv_c, block_mask=None)
      - No caches used, RoPE tables start from 0 for both Q and K (standard).

    • Prefill/decode with caches (no recompute of old keys):
        forward(hidden_q, kv_c_new, block_mask=None,
                use_cache=True, pos_start=abs_pos_of_kv_c_new_start,
                cache=(kv_c_cache, k_r_cache), return_cache=True)

      Inputs:
        - hidden_q:   [B, S, dim_q]  (S=1 typical for decode; can be >1)
        - kv_c_new:   [B, L, c]      (the NEW tokens to append; L can be 0)
        - pos_start:  int absolute position index of kv_c_new[*, 0]
        - cache:      tuple of (kv_c_cache: [B, Ltot, c], k_r_cache: [B, H, Ltot, r])
      Returns:
        - out:        [B, S, dim_q]
        - (kv_c_cache_updated, k_r_cache_updated) if return_cache=True

    Notes
    -----
    • Fusions are disabled in training to preserve gradients and perf.
    • Keys are RoPE‑rotated at WRITE‑time and stored rotated in cache.
    • Queries are RoPE‑rotated at READ‑time with absolute positions.
    • Keep r and c multiples of 16/64 for better kernels.
    """

    def __init__(
        self,
        dim_q: int = 384,
        num_heads: int = 8,
        head_dim: int = 64,     # K
        kv_comp_dim: int = 64,  # c
        q_comp_dim: int = 16,   # d_c'
        retr_dim: int = 64,     # r  (must be even for RoPE)
        score_mod: Optional[Callable] = None,  # Flex mask/bias callback
        use_flex_attention: bool = True,
        rope_max_seqlen: int = 2048,
    ):
        super().__init__()
        assert retr_dim % 2 == 0, "retr_dim (r) must be even for RoPE."

        self.h = num_heads
        self.k = head_dim
        self.d_c = kv_comp_dim
        self.d_cq = q_comp_dim
        self.r = retr_dim
        self.dim_q = dim_q

        # --------------- Projections & parameters -----------------------------
        # latent → per‑head K space (standard q-proj)
        self.q_proj = nn.Linear(dim_q, self.h * self.k, bias=False)

        # absorbed projections to compressed/query spaces  (per head)
        # shapes follow your original code
        self.w_kc_q  = nn.Parameter(torch.empty(self.h, self.k, self.d_cq))  # K→d_c'
        self.w_kc_kv = nn.Parameter(torch.empty(self.h, self.k, self.d_c))   # K→c   (value up-proj uses transpose)

        # retrieval adapters (per head)
        self.W_qr = nn.Parameter(torch.empty(self.h, self.d_cq, self.r))     # d_c'→r (√r baked in)
        self.W_kr = nn.Parameter(torch.empty(self.h, self.d_c,  self.r))     # c→r

        # output projection: concat(H×K) → dim_q
        self.out_proj = nn.Linear(self.h * self.k, dim_q, bias=False)

        self._init_weights()

        # ------------------- RoPE ---------------------------------------------
        # pick dtype/device from a real param so it follows .to() and AMP
        param = self.q_proj.weight
        self.rope = RoPECache(
            max_seqlen=rope_max_seqlen,
            head_dim=self.r,  # rotate in retrieval space
            device=param.device,
            dtype=param.dtype,
        )

        # ------------------- Attention kernels --------------------------------
        self.score_mod = score_mod or (lambda s, *_: s)  # identity if none
        self.use_flex_attention = use_flex_attention
        self._attn = self._flex_attention if use_flex_attention else self._fallback_attention

        # ------------------- Optional fusions (inference only) -----------------
        self._use_fused_query = False
        self._use_fused_output = False
        self.register_buffer("_W_qr_fused", None, persistent=False)  # (H, K, r)
        self.register_buffer("_U_fused",     None, persistent=False)  # (H, c, dim_q)

        # ------------------- Score‑mod helpers (optional) ---------------------
        self._pad_for_scores: Optional[Tensor] = None   # [B, S_keys] bool
        self._q_seg_for_scores: Optional[Tensor] = None # [B, S_queries] int
        self._seg_id_for_scores: Optional[Tensor] = None# [B, S_keys] int

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _init_weights(self):
        for p in (self.w_kc_q, self.w_kc_kv, self.W_qr, self.W_kr):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        # bake √r into W_qr  ⇒ no runtime scaling needed
        self.W_qr.div_(math.sqrt(self.r))

    # ======================= Public controls ================================ #

    @torch.no_grad()
    def enable_fused_query_path(self) -> None:
        """Enable query-path fusion w_kc_q ∘ W_qr → W_qr_fused (inference)."""
        self._ensure_fused_query_fresh()
        self._use_fused_query = True

    @torch.no_grad()
    def enable_fused_output_path(self) -> None:
        """Enable output-path fusion (w_kc_kv, out_proj) → U_fused (inference)."""
        self._ensure_fused_output_fresh()
        self._use_fused_output = True

    @torch.no_grad()
    def disable_fusions(self) -> None:
        """Disable fused paths; training-safe."""
        self._use_fused_query = False
        self._use_fused_output = False

    @torch.no_grad()
    def enable_inference_fusions(self) -> None:
        """Convenience: enable both fusions at once (call under eval())."""
        self.enable_fused_query_path()
        self.enable_fused_output_path()

    # Optional helpers to drive padding/segment gating from the outside
    def set_pad_for_scores(self, pad_mask: Optional[Tensor]) -> None:
        """pad_mask: [B, S_keys] bool; True = pad (mask out)"""
        self._pad_for_scores = pad_mask

    def set_segment_gating(self, q_seg_ids: Optional[Tensor], k_seg_ids: Optional[Tensor]) -> None:
        """q_seg_ids: [B, S_queries], k_seg_ids: [B, S_keys]; mismatched segs are masked."""
        self._q_seg_for_scores = q_seg_ids
        self._seg_id_for_scores = k_seg_ids

    # ======================= Internal fusion builders ======================= #

    @torch.no_grad()
    def _ensure_fused_query_fresh(self):
        """Build or refresh W_qr_fused on the right device/dtype."""
        if self.training:
            return  # don't build in training
        want_dtype = self.q_proj.weight.dtype
        want_dev = self.q_proj.weight.device
        need = (
            self._W_qr_fused is None
            or self._W_qr_fused.dtype != want_dtype
            or self._W_qr_fused.device != want_dev
        )
        if need:
            W_qr_fused = torch.einsum("hkq,hqr->hkr", self.w_kc_q.to(want_dtype), self.W_qr.to(want_dtype))
            self.register_buffer("_W_qr_fused", W_qr_fused.to(want_dev), persistent=False)

    @torch.no_grad()
    def _ensure_fused_output_fresh(self):
        """Build or refresh U_fused: per‑head (c → dim_q)."""
        if self.training:
            return
        want_dtype = self.q_proj.weight.dtype
        want_dev = self.q_proj.weight.device
        need = (
            self._U_fused is None
            or self._U_fused.dtype != want_dtype
            or self._U_fused.device != want_dev
        )
        if need:
            D, HK = self.out_proj.weight.shape  # (dim_q, H*K)
            assert HK == self.h * self.k
            # W_out_per_head: (H, K, D)
            W_out = self.out_proj.weight.view(D, self.h, self.k).permute(1, 2, 0).to(want_dtype)
            # w_kc_kv^T: (H, c, K)
            W_kvT = self.w_kc_kv.transpose(1, 2).to(want_dtype)
            # (H,K,D) x (H,c,K) -> (H,c,D)
            U = torch.einsum("hkd,hck->hcd", W_out, W_kvT)
            self.register_buffer("_U_fused", U.to(want_dev), persistent=False)

    # ======================= RoPE helpers (offset‑aware) ==================== #

    def _rope_slice_with_offset(self, start: int, length: int) -> Tuple[Tensor, Tensor]:
        """
        Return (cos, sin) for absolute positions [start .. start+length-1].
        Falls back to composing from .slice() if the RoPE cache doesn't expose an offset API.
        Shapes: (1,1,length,r/2)
        """
        # Prefer native methods if present
        if hasattr(self.rope, "slice_with_offset"):
            return self.rope.slice_with_offset(start, length)
        if hasattr(self.rope, "at") and length == 1:
            return self.rope.at(start)

        # Fallback: get tables for [0..start+length-1] then slice off the prefix.
        cos, sin = self.rope.slice(start + length)  # (1,1,start+length,r/2)
        if start > 0:
            cos = cos[..., start:, :]
            sin = sin[..., start:, :]
        return cos, sin

    # ======================= Attention kernels ============================= #

    def _score_mod_with_pad(self, scores: Tensor, b: int, h: int, q: int, k: int) -> Tensor:
        # 1) external score_mod (if provided)
        if self.score_mod is not None and self.score_mod is not self._score_mod_with_pad:
            scores = self.score_mod(scores, b, h, q, k)

        # 2) padding
        if self._pad_for_scores is not None:
            scores = scores.masked_fill(self._pad_for_scores[b, k], float("-inf"))

        # 3) segment gating
        if self._q_seg_for_scores is not None and self._seg_id_for_scores is not None:
            bad = self._seg_id_for_scores[b, k] != self._q_seg_for_scores[b, q]
            scores = scores.masked_fill(bad, float("-inf"))

        return scores

    def _flex_attention(self, q_r: Tensor, k_r: Tensor, v_c: Tensor, *, block_mask: Optional[Tensor] = None) -> Tensor:
        # Must pass score_mod hook even when block_mask is present.
        return flex_attention(q_r, k_r, v_c, score_mod=self._score_mod_with_pad, block_mask=block_mask)

    def _fallback_attention(
        self,
        q_r: Tensor,  # [B, H, S, r]
        k_r: Tensor,  # [B, H, L, r]
        v_c: Tensor,  # [B, H, L, c]
        *,
        block_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Lightweight dot‑product attention used when FlexAttention is unavailable.

        Returns
        -------
        ctx_c : torch.Tensor, shape [B, H, S, c]
            Per‑head context vectors in compressed space.
        """
        scores = torch.matmul(q_r, k_r.transpose(-2, -1))  # [B, H, S, L]

        # user‑supplied logits mod
        scores = self._score_mod_with_pad(scores, b=0, h=0, q=0, k=0) if self._score_mod_with_pad else scores

        mask = torch.zeros_like(scores, dtype=torch.bool)
        if block_mask is not None:
            if hasattr(block_mask, "mask_mod"):
                B, H, S, L = scores.shape
                q_idx = torch.arange(S, device=scores.device)
                k_idx = torch.arange(L, device=scores.device)
                keep = block_mask.mask_mod(0, 0, q_idx[:, None], k_idx[None, :])
                scores = scores.masked_fill(keep, -float("inf"))
                mask = mask | keep
            else:
                scores = scores.masked_fill(block_mask, -float("inf"))
                mask = mask | block_mask

        attn = safe_softmax(scores, mask, dim=-1)  # [B, H, S, L]
        return torch.matmul(attn, v_c)             # [B, H, S, c]

    # ======================= Main forward ================================== #

    def forward(
        self,
        hidden_q: Tensor,              # [B, S, dim_q]
        kv_c: Tensor,                  # [B, L, c]  (block for training; "new block" for cache mode)
        block_mask: Optional[Tensor] = None,
        *,
        use_cache: bool = False,
        pos_start: int = 0,
        cache: Optional[Tuple[Tensor, Tensor]] = None,  # (kv_c_cache: [B,Ltot,c], k_r_cache: [B,H,Ltot,r])
        return_cache: bool = False,
    ):
        """
        If use_cache=False (default): standard block attention over kv_c (training).
        If use_cache=True: treat kv_c as the NEW block to append at absolute pos_start, use 'cache' as past.
        """
        B, S, _ = hidden_q.shape
        _, L, _ = kv_c.shape
        dev, dt = hidden_q.device, hidden_q.dtype

        # Decide fusion usage: NEVER in training (keeps perf & grads intact).
        use_fused_query = (not self.training) and self._use_fused_query
        use_fused_output = (not self.training) and self._use_fused_output

        if use_fused_query:
            self._ensure_fused_query_fresh()
        if use_fused_output:
            self._ensure_fused_output_fresh()

        # --------- 1) Q path: hidden → per‑head K → retrieval r ---------------
        q_hk = self.q_proj(hidden_q).view(B, S, self.h, self.k)  # [B,S,H,K]
        if use_fused_query and self._W_qr_fused is not None:
            # (B,S,H,K) × (H,K,r) → (B,S,H,r)
            q_r = torch.einsum("bshk,hkr->bshr", q_hk, self._W_qr_fused)
        else:
            # two‑step (training/default): K→d_c'→r
            q_big = torch.einsum("bshk,hkq->bshq", q_hk, self.w_kc_q)    # [B,S,H,d_c′]
            q_r   = torch.einsum("bshq,hqr->bshr", q_big, self.W_qr)    # [B,S,H,r]
        q_r = q_r.permute(0, 2, 1, 3).contiguous()  # [B,H,S,r]

        # --------- 2) K path: build k_r for keys (new block OR full block) ---
        if use_cache:
            # Build for NEW tokens only and rotate with absolute positions [pos_start..pos_start+L-1]
            k_r_block = torch.einsum("bld,hdr->bhlr", kv_c, self.W_kr)  # [B,H,L,r]
            cos_k, sin_k = self._rope_slice_with_offset(pos_start, L)
            k_r_block = apply_rope(k_r_block, cos_k, sin_k)

            # Extend caches (or start fresh if none provided)
            if cache is None:
                kv_c_cache = kv_c
                k_r_cache  = k_r_block
            else:
                kv_c_cache, k_r_cache = cache
                kv_c_cache = torch.cat([kv_c_cache, kv_c], dim=1)  # [B,Ltot+L,c]
                k_r_cache  = torch.cat([k_r_cache,  k_r_block], dim=2)  # [B,H,Ltot+L,r]
        else:
            # Training/block mode: build full k_r for provided kv_c
            k_r_cache = torch.einsum("bld,hdr->bhlr", kv_c, self.W_kr)  # [B,H,L,r]
            cos_k, sin_k = self._rope_slice_with_offset(0, L)
            k_r_cache = apply_rope(k_r_cache, cos_k, sin_k)
            kv_c_cache = kv_c

        # --------- 3) RoPE for queries at READ‑time (absolute positions) -----
        # If you're decoding S=1 at absolute position p, pass pos_start=p for q as well.
        cos_q, sin_q = self._rope_slice_with_offset(pos_start, S) if use_cache else self._rope_slice_with_offset(0, S)
        q_r = apply_rope(q_r, cos_q, sin_q)  # [B,H,S,r]

        # --------- 4) Values: broadcast kv_c_cache across heads --------------
        v_c = kv_c_cache.unsqueeze(1).expand(-1, self.h, -1, -1).contiguous()  # [B,H,Ltot,c] or [B,H,L,c]

        # --------- 5) Attention in compressed space --------------------------
        ctx_c = self._attn(q_r, k_r_cache, v_c, block_mask=block_mask)  # [B,H,S,c]

        # --------- 6) Output path: compressed → model dimension --------------
        if use_fused_output and self._U_fused is not None:
            # (B,H,S,c) × (H,c,dim_q) → (B,S,dim_q)
            out = torch.einsum("bhsd,hdD->bsD", ctx_c, self._U_fused)
        else:
            # two‑step (training/default): c→K (per head), concat heads, proj to dim_q
            ctx_lat = torch.einsum("bhsd,hdK->bhsK", ctx_c, self.w_kc_kv.transpose(1, 2))  # [B,H,S,K]
            out = self.out_proj(ctx_lat.permute(0, 2, 1, 3).reshape(B, S, -1))            # [B,S,dim_q]

        if use_cache and return_cache:
            return out, (kv_c_cache, k_r_cache)
        return out

    # ======================= Misc / Compatibility =========================== #

    def load_state_dict(self, *a, **kw):
        """
        Standard load; clear any derived fused buffers on load so they rebuild
        on demand with the correct dtype/device.
        """
        ret = super().load_state_dict(*a, **kw)
        # Invalidate fused buffers (they'll rebuild lazily in eval)
        self.register_buffer("_W_qr_fused", None, persistent=False)
        self.register_buffer("_U_fused",     None, persistent=False)
        return ret




def _build_time_keep_fn(window: int):
    return lambda _b, _h, q, k: (k <= q) & ((q - k) <= window)


@torch._dynamo.disable()
@lru_cache(maxsize=64)
def _cached_flex_sliding_mask(seq_len: int, window: int, device):
    keep = _build_time_keep_fn(window)
    return create_block_mask(keep, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, BLOCK_SIZE=128, device=device)


class SlidingWindowMLA(nn.Module):
    """
    Wraps MultiheadLatentAttention and supplies a cached sliding block mask.
    """

    def __init__(self, window_size: int, *mla_args, **mla_kwargs):
        super().__init__()
        self.window = window_size
        self.mla = MultiheadLatentAttention(*mla_args, **mla_kwargs)
        self.kv_proj = nn.Linear(mla_kwargs.get("dim_q"), self.mla.d_c, bias=False)
        self.o_proj = nn.Linear(mla_kwargs.get("dim_q"), mla_kwargs.get("dim_q"), bias=False)

    @staticmethod
    # @dynamo.disable
    def _make_block_mask(
        seq_len: int,
        window: int,
        pad: Optional[torch.Tensor],
        device: torch.device,
        block_size: int,
        num_heads: int,
    ):
        """Eager-only BlockMask builder to avoid Dynamo/Triton miscompiles."""
        keep_time = _build_time_keep_fn(window)

        if pad is not None:
            # ensure clean indexing
            pad = pad.to(torch.bool).contiguous()

            def _keep_pad(b, h, _q, k):
                return ~pad[b, k]

            keep = and_masks(keep_time, _keep_pad)
            B = int(pad.size(0))
        else:
            keep, B = keep_time, None

        return create_block_mask(
            keep,
            B=B,
            H=num_heads,  # make H explicit
            Q_LEN=int(seq_len),
            KV_LEN=int(seq_len),
            BLOCK_SIZE=int(block_size),
            device=device,
        )

    def build_sliding_masks(
        self,
        seq_len: int,
        window: int,
        pad: Optional[torch.Tensor],
        use_flex: bool,
        device: torch.device,
        block_size: int = 128,
    ) -> Tuple[Optional[Callable], Optional[torch.Tensor]]:
        """
        Returns (flex_mask, dense_mask). Flex path uses a cached time-only mask.
        Padding is handled in score_mod, not in the BlockMask.
        """
        # time-window only; cached (no closure over pad)
        flex_mask = _cached_flex_sliding_mask(seq_len, window, device)
        return flex_mask, None

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        B, S, _ = x.shape
        dev = x.device

        kv_c = self.kv_proj(x)
        if key_padding_mask is not None:
            kv_c = kv_c.masked_fill(key_padding_mask[..., None], 0.0)

        # cache: time-only block mask
        flex_mask, dense_mask = self.build_sliding_masks(
            seq_len=S,
            window=self.window,
            pad=key_padding_mask.to(torch.bool) if key_padding_mask is not None else None,
            use_flex=self.mla.use_flex_attention,
            device=dev,
        )
        block_mask = flex_mask if flex_mask is not None else dense_mask

        # hand pad to MLA to be applied in score_mod inside the kernel
        self.mla._pad_for_scores = key_padding_mask.to(torch.bool) if key_padding_mask is not None else None

        ctx = self.mla(x, kv_c, block_mask=block_mask)

        # optional hygiene: clear the attribute so nothing stale lingers
        self.mla._pad_for_scores = None
        return ctx


    @staticmethod
    @dynamo.disable
    def _build_decode_block_mask(
        q_len: int,
        kv_len: int,
        window: int,
        pos_start: int,          # absolute position of first new query/key
        device: torch.device,
        block_size: int = 128,
    ):
        """
        FlexAttention 'keep' mask for decode with caches.
        keep iff key_abs <= query_abs and (query_abs - key_abs) <= window
        where key_abs ∈ [0..kv_len-1], query_abs ∈ [pos_start..pos_start+q_len-1].
        NOTE: use elementwise tensor ops (&), not Python 'and'.
        """
        def _keep(_b, _h, q, k):
            # q, k are *tensor* index grids provided by create_block_mask/vmap
            q_abs = q + pos_start          # tensor + int → tensor
            return (k <= q_abs) & ((q_abs - k) <= window)

        return create_block_mask(
            _keep,
            B=None, H=None,
            Q_LEN=int(q_len), KV_LEN=int(kv_len),
            BLOCK_SIZE=int(block_size),
            device=device,
        )

    def forward_cached(
        self,
        x_new: Tensor,                               # [B, S_new, D]  (pre‑LN hidden to attend with)
        *,
        pos_start: int,                              # absolute pos index of x_new[:,0]
        cache: tuple[Tensor, Tensor] | None = None,  # (kv_c_cache [B,Ltot,c], k_r_cache [B,H,Ltot,r])
        key_padding_mask_new: Tensor | None = None,  # [B, S_new] True=pad
        return_cache: bool = True,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Decode path that appends new keys and attends with caches.
        Returns (context_new, updated_cache). Context is in model space (after o_proj).
        """
        B, S_new, _ = x_new.shape
        dev = x_new.device

        # Project new tokens into compressed K/V space
        kv_c_new = self.kv_proj(x_new)
        if key_padding_mask_new is not None:
            kv_c_new = kv_c_new.masked_fill(key_padding_mask_new[..., None], 0.0)

        # KV length prior to appending
        L_old = 0 if cache is None else int(cache[0].size(1))
        # Build a time‑window mask (Flex path); for torch fallback, leave None
        block_mask = None
        if self.mla.use_flex_attention:
            block_mask = self._build_decode_block_mask(
                q_len=S_new, kv_len=L_old + S_new,
                window=self.window, pos_start=pos_start, device=dev
            )

        # Hand per‑key padding to MLA’s score_mod (cheap)
        self.mla._pad_for_scores = key_padding_mask_new.to(torch.bool) if key_padding_mask_new is not None else None

        # Hand a KV‑length pad mask to MLA’s score_mod (must be length L_old+S_new)
        if key_padding_mask_new is not None:
            B = x_new.size(0)

            if L_old > 0:
                pad_old = torch.zeros(B, L_old, dtype=torch.bool, device=dev)
                pad_all = torch.cat([pad_old, key_padding_mask_new.to(torch.bool)], dim=1)  # (B, L_old+S_new)
            else:
                pad_all = key_padding_mask_new.to(torch.bool)
            # sanity: match KV len

            assert pad_all.size(1) == (L_old + S_new), (pad_all.size(), L_old, S_new)
            self.mla._pad_for_scores = pad_all
        else:
            self.mla._pad_for_scores = None


        # MLA does: rotate new keys at write‑time, append caches, rotate queries at read‑time
        out_m = self.mla(
            x_new, kv_c_new, block_mask=block_mask,
            use_cache=True, pos_start=pos_start, cache=cache, return_cache=return_cache
        )
        if return_cache:
            out, new_cache = out_m
        else:
            out, new_cache = out_m, None

        # hygiene
        self.mla._pad_for_scores = None

        return out, new_cache




class SlidingWindowMLATransformerBlock(nn.Module):
    """
    A single Transformer block that uses the Sliding‑Window Multi‑Head Latent
    Attention (MLA) you defined above.  Structure is Pre‑LN:

        x  →  RMSNorm  →  MLA (sliding‑window)  →  +residual
           →  RMSNorm  →  SwiGLU FFN           →  +residual

    Parameters
    ----------
    dim : int
        Model / embedding dimension (D).
    num_heads : int, default 128
        Number of latent heads (H = 128 for DeepSeek‑V3‑style MLA).
    window_size : int
        Half‑window radius for the sliding attention mask.
    head_dim : int, default 128
        Latent head width (K).
    kv_comp_dim : int, default 512
        Compressed key/value width (d_c).
    q_comp_dim : int, default 1536
        “Big” compressed query width (d_c′).
    retr_dim : int, default 64
        Retrieval sub‑space width (r).
    ffn_dim_multiplier : int, default 4
        Hidden size multiplier for the SwiGLU feed‑forward block.
    use_flex_attention : bool, default True
        Enables FlexAttention kernel when available.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        *,
        head_dim: int = 128,
        kv_comp_dim: int = 512,
        q_comp_dim: int = 1536,
        retr_dim: int = 64,
        ffn_dim_multiplier: int = 4,
        use_flex_attention: bool = True,
    ):
        super().__init__()

        # Attention sub‑layer
        self.norm1 = nn.RMSNorm(dim)

        self.attn = SlidingWindowMLA(
            window_size,
            # args for MultiheadLatentAttention internally:
            dim_q=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_comp_dim=kv_comp_dim,
            q_comp_dim=q_comp_dim,
            retr_dim=retr_dim,
            use_flex_attention=use_flex_attention,
        )

        # Feed‑forward sub‑layer
        self.norm2 = nn.RMSNorm(dim)
        hidden_dim = dim * ffn_dim_multiplier
        self.ffn = SwiGLU(dim, hidden_dim)

    # --------------------------------------------------------------------- #
    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args
        ----
        x : Tensor, shape [B, S, D]
            Input sequence.
        key_padding_mask : Optional[Tensor], shape [B, S]
            True for padding positions (passed straight to MLA).

        Returns
        -------
        Tensor, shape [B, S, D]
            Output of the Transformer block.
        """
        # Attention (Pre‑LN)
        x = self.norm1(x)  # [B, S, D]
        x = x + self.attn(x, key_padding_mask=key_padding_mask)  # [B, S, D]
        # Feed‑forward (Pre‑LN)
        x = x + self.ffn(self.norm2(x))
        return x


    @torch.no_grad()
    def prefill(
        self,
        x_block: Tensor,                         # [B, S0, D]  full prefix
        *,
        pos_start: int = 0,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Build caches for the whole prefix in one go (fast prefill).
        Returns (y_block, cache) where y_block is the block output for the whole prefix.
        """
        q = self.norm1(x_block)
        ctx, cache = self.attn.forward_cached(
            q, pos_start=pos_start, cache=None,
            key_padding_mask_new=key_padding_mask, return_cache=True
        )
        y = x_block + ctx
        y = y + self.ffn(self.norm2(y))
        return y, cache

    @torch.no_grad()
    def step(
        self,
        x_new: Tensor,                            # [B, S_new, D]  NEW tokens only (usually S_new=1)
        *,
        cache: tuple[Tensor, Tensor],
        pos_start: int,                           # absolute index of x_new[:,0]
        key_padding_mask_new: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        One streaming decode step (or micro‑batch of steps).
        Returns (y_new, updated_cache) for just the new slice.
        """
        q = self.norm1(x_new)
        ctx, cache = self.attn.forward_cached(
            q, pos_start=pos_start, cache=cache,
            key_padding_mask_new=key_padding_mask_new, return_cache=True
        )
        y = x_new + ctx
        y = y + self.ffn(self.norm2(y))
        return y, cache

@torch._dynamo.disable()
@lru_cache(maxsize=64)
def _cached_causal_mask(seq_len: int, device: torch.device):
    """
    Fast‑path builder for a FlexAttention‑compatible causal mask.

    keep(b,h,q,k) == True  ➜ *allow* logit      (Flex keeps allowed positions)
    We therefore “keep” only keys k that are **not** in the future (k ≤ q).
    """

    def _keep(_b, _h, q, k):  # noqa: D401  (simple lambda clearer here)
        return k <= q

    return create_block_mask(
        _keep,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        BLOCK_SIZE=128,
        device=device,
    )


def _boolean_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Fallback mask for the pure‑PyTorch path (shape [1,1,S,S])"""
    q_idx = torch.arange(seq_len, device=device)[:, None]
    k_idx = torch.arange(seq_len, device=device)[None, :]
    return (k_idx > q_idx).unsqueeze(0).unsqueeze(0)  # True = *mask out*


# ----------------------------------------------------------------------------- #


class CausalMLA(nn.Module):
    """
    Wraps `MultiheadLatentAttention` with a standard causal mask.
    """

    def __init__(self, *mla_args, dim_q: int, **mla_kwargs):
        """
        Parameters
        ----------
        dim_q : int
            Model width (passed to the K/V projection and output projection).
        *mla_args / **mla_kwargs
            Forwarded to `MultiheadLatentAttention`.
        """
        super().__init__()
        self.mla = MultiheadLatentAttention(*mla_args, dim_q=dim_q, **mla_kwargs)

        # Project tokens into compressed K/V space once per layer
        self.kv_proj = nn.Linear(dim_q, self.mla.d_c, bias=False)

        # Final linear to mix latent heads back to model space
        self.o_proj = nn.Linear(dim_q, dim_q, bias=False)

    def _build_mask(self, seq_len, key_padding_mask, device):
        if self.mla.use_flex_attention:
            return _cached_causal_mask(seq_len, device)  # time-only causal
        block_mask = _boolean_causal_mask(seq_len, device)
        if key_padding_mask is not None:
            block_mask = block_mask | key_padding_mask[:, None, None, :]
        return block_mask

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        B, S, _ = x.shape
        kv_c = self.kv_proj(x)
        if key_padding_mask is not None:
            kv_c = kv_c.masked_fill(key_padding_mask[..., None], 0.0)

        block_mask = self._build_mask(S, key_padding_mask, x.device)

        # let score_mod handle padding inside the kernel
        self.mla._pad_for_scores = key_padding_mask.to(torch.bool) if key_padding_mask is not None else None

        ctx = self.mla(x, kv_c, block_mask=block_mask)
        self.mla._pad_for_scores = None
        return self.o_proj(ctx)


# ----------------------------------------------------------------------------- #
# Transformer Block
# ----------------------------------------------------------------------------- #
class CausalMLATransformerBlock(nn.Module):
    """
    Standard Pre‑LN causal Transformer block powered by MLA.

    Layout:
        x  →  RMSNorm →  CausalMLA  → +residual
           →  RMSNorm →  SwiGLU FFN → +residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        head_dim: int = 128,
        kv_comp_dim: int = 512,
        q_comp_dim: int = 1536,
        retr_dim: int = 64,
        ffn_dim_multiplier: int = 4,
        use_flex_attention: bool = True,
    ):
        super().__init__()

        # Attention sub‑layer --------------------------------------------------
        self.norm1 = nn.RMSNorm(dim)
        self.attn = CausalMLA(
            dim_q=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_comp_dim=kv_comp_dim,
            q_comp_dim=q_comp_dim,
            retr_dim=retr_dim,
            use_flex_attention=use_flex_attention,
        )

        # Feed‑forward sub‑layer ----------------------------------------------
        self.norm2 = nn.RMSNorm(dim)
        hidden_dim = dim * ffn_dim_multiplier
        self.ffn = SwiGLU(dim, hidden_dim)

    # --------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,  # [B,S,D]
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Causal MLA
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)

        # 2. Feed‑forward
        x = x + self.ffn(self.norm2(x))
        return x
