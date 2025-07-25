import math
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and negate the last‑dimensional halves: (x1, x2) → (‑x2, x1)."""
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    Standard RoPE:  (x_even, x_odd) → (x_even·cos – x_odd·sin, x_even·sin + x_odd·cos)
    `sin`/`cos` must be broadcast‑compatible with `x`.
    """
    return (x * cos) + (_rotate_half(x) * sin)


class _RoPECache:
    """
    Memoises (sin,cos) lookup tables keyed on
        (seq_len, dim, device, dtype)
    so we only pay the outer‑product cost once.
    """
    _cache: dict[tuple[int, int, torch.device, torch.dtype],
    tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def get(
            cls,
            seq_len: int,
            dim: int,
            device: torch.device,
            dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, dim, device, dtype)
        if key not in cls._cache:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2,
                                                     device=device,
                                                     dtype=dtype) / dim))
            t = torch.arange(seq_len, device=device, dtype=dtype)  # [L]
            freqs = torch.einsum('l,d->ld', t, inv_freq)  # [L, dim/2]
            sin = freqs.sin()  # [L, dim/2]
            cos = freqs.cos()

            # interleave quickly:  (x0,x1,…,x_{d/2-1}) -> (x0,x0,x1,x1,…)
            sin = sin.repeat_interleave(2, dim=-1)  # [L, dim]
            cos = cos.repeat_interleave(2, dim=-1)

            cls._cache[key] = (sin, cos)
        return cls._cache[key]

    @classmethod
    def clear(cls) -> None:
        """Call in `load_state_dict()` to avoid stale dtype/device entries."""
        cls._cache.clear()


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
            score_mod: Optional[Callable] = None,  # Flex mask/bias callback
            use_flex: bool = True,
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

        self.score_mod = score_mod or (lambda s, *_: s)  # identity if none
        self.use_flex = bool(use_flex and flex_attention)

    # ------------------------------------------------------------------ #
    def _init_weights(self):
        for p in (self.w_kc_q, self.w_kc_kv, self.W_qr, self.W_kr):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        # bake √r into W_qr  ⇒ no runtime scaling needed
        with torch.no_grad():
            self.W_qr.div_(math.sqrt(self.r))

    # - q_hk = self.q_proj(hidden_q).view(B, self.h, self.k)
    # + q_hk = self.q_proj(hidden_q).reshape(B, self.h, self.k)
    #
    # # (optional) rotate both Q_kv and KV
    # - kv_c = _apply_rope(kv_c, sin[None], cos[None])
    # + kv_c = _apply_rope(kv_c, sin[None], cos[None])
    # + q_kv = torch.einsum('bhk,hkc->bhc', q_hk, self.w_kc_kv)  # [B,H,d_c]
    # + q_kv = _apply_rope(q_kv, sin[None, None], cos[None, None])
    #
    # # use pre‑transposed weight once
    # - ctx_lat = torch.einsum('bhd,hdk->bhk', ctx_c, self.w_kc_kv.transpose(1, 2))
    # + ctx_lat = torch.einsum('bhd,hdK->bhk', ctx_c, self.w_kc_kv_T)

    # clear RoPE tables when dtype/device flip
    def load_state_dict(self, *a, **kw):
        ret = super().load_state_dict(*a, **kw)
        _RoPECache.clear()
        return ret

    # ------------------------------------------------------------------ #
    # fast path when FlexAttention is available
    def _flex_attention(self, q_r, k_r, v_c):
        # shapes: q_r [B,H,1,r]   k_r [B,H,L,r]   v_c [B,H,L,d_c]
        return flex_attention(
            q_r, k_r, v_c,  # Q, K, V
            score_mod=self.score_mod  # your masking fn
        ).squeeze(2)  # -> [B,H,d_c]

    def _fallback_attention(
            self,
            q_r: torch.Tensor,  # [B, H, 1, r]
            k_r: torch.Tensor,  # [B, H, L, r]
            v_c: torch.Tensor,  # [B, H, L, d_c]
    ) -> torch.Tensor:
        """
        Light‑weight dot‑product attention used when FlexAttention is unavailable.

        Returns
        -------
        ctx_c : torch.Tensor, shape [B, H, d_c]
            Per‑head context vectors in compressed space.
        """
        # -- 1. scores ----------------------------------------------------------
        #   squeeze away the singleton query‑length → [B, H, r]
        q = q_r.squeeze(2)
        #   transpose for classic (Q @ Kᵀ) formulation → [B, H, r, L]
        k = k_r.transpose(-1, -2)

        scores = torch.matmul(q, k).squeeze(-2)  # [B, H, L]

        # -- 2. user‑supplied bias / mask operates on raw logits ---------------
        scores = self.score_mod(scores)

        # -- 3. softmax & value aggregation ------------------------------------
        attn = torch.softmax(scores, dim=-1)  # [B, H, L]

        # fused gemm: (B·H)×L @ L×d_c  →  (B·H)×d_c
        return torch.matmul(attn, v_c)  # [B, H, d_c]

    # ------------------------------------------------------------------ #
    def forward(self, hidden_q: torch.Tensor, kv_c: torch.Tensor) -> torch.Tensor:
        """
        hidden_q : [B, D]          latent query
        kv_c     : [B, L, d_c]     pre‑compressed keys/values
        """
        B, L, _ = kv_c.shape
        dev, dt = kv_c.device, kv_c.dtype

        # 1) latent -> [B,H,K]
        q_hk = self.q_proj(hidden_q).view(B, self.h, self.k)

        # 2) two compressed views: Q_big, Q_kv
        q_big = torch.einsum('bhk,hkq->bhq', q_hk, self.w_kc_q)  # [B,H,d_c']

        sin, cos = _RoPECache.get(L, self.d_c, dev, dt)  # tables [L,d_c]
        kv_c = _apply_rope(kv_c, sin.unsqueeze(0), cos.unsqueeze(0))

        # 3) retrieval Q_r  /  K_r
        q_r = torch.einsum('bhq,hqr->bhr', q_big, self.W_qr)  # [B,H,r]
        k_r = torch.einsum('bld,hdr->bhdr', kv_c, self.W_kr)  # [B,H,L,r]

        # shapes for attention helpers
        q_r = q_r.unsqueeze(2)  # [B,H,1,r]
        kv_c_h = kv_c.unsqueeze(1).expand(-1, self.h, -1, -1)  # [B,H,L,d_c]

        # 4) attention
        ctx_c = (
            self._flex_attention(q_r, k_r, kv_c_h)
            if self.use_flex else
            self._fallback_attention(q_r, k_r, kv_c_h)
        )  # [B,H,d_c]

        # 5) compressed → latent
        ctx_lat = torch.einsum(
            'bhd,hdk->bhk', ctx_c, self.w_kc_kv.transpose(1, 2)
        ).reshape(B, -1)

        return self.out_proj(ctx_lat)
