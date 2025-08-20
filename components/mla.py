from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Optional, Tuple

import torch
import torch._dynamo as dynamo
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import and_masks, create_block_mask, flex_attention

from .rope import RoPECache, apply_rope
from .swiglu import SwiGLU
from .utils import safe_softmax

class MultiheadLatentAttention(nn.Module):
    """
    DeepSeek‑V3 style Multi‑Head Latent Attention with asymmetric compression
    and optional FlexAttention acceleration + arbitrary masking.
    -----------------------------------------------------------------------
      latent  (K=128)  →  Q_big  (d_c' = 1536)
                       →  Q_kv   (d_c  = 512)
      K/V     (d_c)    →  retrieval K_r  (r = 64)
      Q_big   (d_c')   →  retrieval Q_r  (r = 64)
      scores  : <Q_r , K_r>  (√r scale baked into W_qr)
      values  : V  =  K/V_compressed   (d_c)
    """

    def __init__(
        self,
        dim_q: int,  # full model width (7168)
        num_heads: int = 128,
        head_dim: int = 128,  # K
        kv_comp_dim: int = 512,  # d_c
        q_comp_dim: int = 1536,  # d_c'
        retr_dim: int = 64,  # r
        score_mod: Callable | None = None,  # Flex mask/bias callback
        use_flex_attention: bool = True,
    ):
        super().__init__()
        self.h = num_heads
        self.k = head_dim
        self.d_c = kv_comp_dim
        self.d_cq = q_comp_dim
        self.r = retr_dim

        # latent → K‑space
        self.q_proj = nn.Linear(dim_q, self.h * self.k, bias=False)

        # absorbed projections to compressed spaces
        self.w_kc_q = nn.Parameter(torch.empty(self.h, self.k, self.d_cq))  # K→d_c'
        self.w_kc_kv = nn.Parameter(torch.empty(self.h, self.k, self.d_c))  # K→d_c

        # retrieval adapters
        self.W_qr = nn.Parameter(torch.empty(self.h, self.d_cq, self.r))
        self.W_kr = nn.Parameter(torch.empty(self.h, self.d_c, self.r))

        # output
        self.out_proj = nn.Linear(self.h * self.k, dim_q, bias=False)

        self._init_weights()

        # pick dtype/device from a real param so it follows .to() and AMP
        param = self.q_proj.weight
        self.rope = RoPECache(
            max_seqlen=2048,
            head_dim=self.r,  # <-- rotate in retrieval space
            device=param.device,
            dtype=param.dtype,
        )

        self.score_mod = score_mod or (lambda s, *_: s)  # identity if none
        self.use_flex_attention = use_flex_attention
        if self.use_flex_attention:
            self._attn = self._flex_attention
        else:
            self._attn = self._fallback_attention

        self._pad_for_scores = None  # [B, S] bool or None
        self._q_seg_for_scores = None  # [B, Q] int or None
        self._seg_id_for_scores = None  # [B, S] int or None

    # ------------------------------------------------------------------ #
    def _init_weights(self):
        for p in (self.w_kc_q, self.w_kc_kv, self.W_qr, self.W_kr):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        # bake √r into W_qr  ⇒ no runtime scaling needed
        with torch.no_grad():
            self.W_qr.div_(math.sqrt(self.r))

    # clear RoPE tables when dtype/device flip
    def load_state_dict(self, *a, **kw):
        ret = super().load_state_dict(*a, **kw)
        # _RoPECache.clear()
        return ret

    def _score_mod_with_pad(self, scores, b, h, q, k):

        # user-provided score_mod first, if any
        if self.score_mod is not None and self.score_mod is not self._score_mod_with_pad:
            scores = self.score_mod(scores, b, h, q, k)

        # 1) padding (what you already had)
        if self._pad_for_scores is not None:
            scores = scores.masked_fill(self._pad_for_scores[b, k], float("-inf"))

        # 2) segment gating (new)
        if self._q_seg_for_scores is not None and self._seg_id_for_scores is not None:
            # bad if key's seg != query's seg
            bad = self._seg_id_for_scores[b, k] != self._q_seg_for_scores[b, q]
            scores = scores.masked_fill(bad, float("-inf"))

        return scores

    def _flex_attention(self, q_r, k_r, v_c, *, block_mask: torch.Tensor | None = None):
        # NOTE: we now always pass score_mod, even when block_mask is present.
        return flex_attention(
            q_r,
            k_r,
            v_c,
            score_mod=self._score_mod_with_pad,
            block_mask=block_mask,
        )

    def _fallback_attention(
        self,
        q_r: torch.Tensor,  # [B, H, S, r]
        k_r: torch.Tensor,  # [B, H, L, r]
        v_c: torch.Tensor,  # [B, H, L, d_c]
        *,
        block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Light‑weight dot‑product attention used when FlexAttention is unavailable.

        Returns
        -------
        ctx_c : torch.Tensor, shape [B, H, d_c]
            Per‑head context vectors in compressed space.
        """
        # q_r: [B,H,S,r]   k_r: [B,H,L,r]  →  scores: [B,H,S,L]
        scores = torch.matmul(q_r, k_r.transpose(-2, -1))  # [B, H, S, L]

        # -- 2. user‑supplied bias / mask operates on raw logits ---------------
        scores = self.score_mod(scores)

        mask = torch.zeros_like(scores, dtype=torch.bool)
        if block_mask is not None:
            if hasattr(block_mask, 'mask_mod'):
                B, H, _, L = scores.shape
                q_idx = torch.arange(L, device=scores.device)
                k_idx = q_idx[:, None]
                keep = block_mask.mask_mod(0, 0, q_idx[:, None], k_idx[:, None])
                scores = scores.masked_fill(keep, -float('inf'))
                mask = mask | keep
            else:
                scores = scores.masked_fill(block_mask, -float('inf'))
                mask = mask | block_mask

        # -- 3. softmax & value aggregation ------------------------------------
        attn = safe_softmax(scores, mask, dim=-1)  # [B, H, S, L]

        # fused gemm: (B·H·S)×L @ L×d_c  →  (B·H·S)×d_c
        return torch.matmul(attn, v_c)  # [B, H, S, d_c]

    def forward(self, hidden_q: torch.Tensor, kv_c: torch.Tensor, block_mask=None) -> torch.Tensor:
        B, S, _ = hidden_q.shape
        _, L, _d = kv_c.shape
        dev, dt = kv_c.device, kv_c.dtype

        q_hk = self.q_proj(hidden_q).view(B, S, self.h, self.k)  # [B,S,H,K]
        q_big = torch.einsum('bshk,hkq->bshq', q_hk, self.w_kc_q)  # [B,S,H,d_c′]

        # Project into the actual scoring space r
        q_r = torch.einsum('bshq,hqr->bshr', q_big, self.W_qr)  # [B,S,H,r]
        q_r = q_r.permute(0, 2, 1, 3)  # [B,H,S,r]

        k_r = torch.einsum('bld,hdr->bhlr', kv_c, self.W_kr)  # [B,H,L,r]

        # --- RoPE in retrieval space (the dot-product space) ---
        cos_S, sin_S = self.rope.slice(S)  # (1,1,S,r/2)
        q_r = apply_rope(q_r, cos_S, sin_S)  # [B,H,S,r]

        cos_L, sin_L = self.rope.slice(L)  # (1,1,L,r/2)
        k_r = apply_rope(k_r, cos_L, sin_L)  # [B,H,L,r]

        v_c = kv_c.unsqueeze(1).expand(-1, self.h, -1, -1)  # [B,H,L,d_c]

        # 5) attention
        ctx_c = self._attn(q_r, k_r, v_c, block_mask=block_mask)  # [B,H,d_c]

        # 6) compressed → latent
        ctx_lat = torch.einsum('bhsd,hdK->bhsK', ctx_c, self.w_kc_kv.transpose(1, 2))  # [B,H,S,K]
        ctx_lat = ctx_lat.permute(0, 2, 1, 3).reshape(B, S, -1)  # [B,S,H*K]

        # 7) output
        return self.out_proj(ctx_lat)  # [B,D]


def _build_time_keep_fn(window: int):
    return lambda _b, _h, q, k: (k <= q) & ((q - k) <= window)


# @lru_cache(maxsize=64)
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
    @dynamo.disable
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
        # if use_flex:
        # time-window only; cached (no closure over pad)
        flex_mask = _cached_flex_sliding_mask(seq_len, window, device)
        return flex_mask, None

        # ---------- dense mask path (fallback) ----------
        q_idx = torch.arange(seq_len, device=device)[:, None]
        k_idx = torch.arange(seq_len, device=device)[None, :]
        dense = (k_idx > q_idx) | (k_idx < q_idx - window)  # outside window
        dense = dense.unsqueeze(0).unsqueeze(0)  # [1,1,S,S]
        if pad is not None:
            dense = dense | pad[:, None, None, :]
        return None, dense

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

        # NEW: hand pad to MLA to be applied in score_mod inside the kernel
        self.mla._pad_for_scores = key_padding_mask.to(torch.bool) if key_padding_mask is not None else None

        ctx = self.mla(x, kv_c, block_mask=block_mask)

        # optional hygiene: clear the attribute so nothing stale lingers
        self.mla._pad_for_scores = None
        return ctx


# @torch.compile
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
        # if torch.isnan(x).any():
        #     print('nan')
        x = self.norm1(x)  # [B, S, D]
        # if torch.isnan(x).any():
        #     print('nan')
        x = x + self.attn(x, key_padding_mask=key_padding_mask)  # [B, S, D]

        # x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)

        # Feed‑forward (Pre‑LN)
        x = x + self.ffn(self.norm2(x))
        return x


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


# @torch.compile
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
