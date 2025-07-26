import math
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks
from torch import Tensor
from functools import lru_cache

from .utils import safe_softmax
from .swiglu import SwiGLU


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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.w_kc_q = nn.Parameter(torch.empty(self.h, self.k, self.d_cq, device=device))  # K→d_c'
        self.w_kc_kv = nn.Parameter(torch.empty(self.h, self.k, self.d_c, device=device))  # K→d_c

        self.register_buffer(
            "w_kc_kv_T", self.w_kc_kv.transpose(1, 2), persistent=False
        )

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


    # clear RoPE tables when dtype/device flip
    def load_state_dict(self, *a, **kw):
        ret = super().load_state_dict(*a, **kw)
        _RoPECache.clear()
        return ret

    # ------------------------------------------------------------------ #
    def _flex_attention(self, q_r, k_r, v_c, *, block_mask=None):
        # q_r : [B, H, S, r]   (S can be 1 or >1)
        # k_r : [B, H, L, r]
        # v_c : [B, H, L, d_c]
        return flex_attention(
            q_r, k_r, v_c,
            score_mod=self.score_mod if block_mask is None else None,
            block_mask=block_mask,
        )  # -> [B, H, S, d_c]

    def _fallback_attention(
            self,
            q_r: torch.Tensor,  # [B, H, 1, r]
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
        # -- 1. scores ----------------------------------------------------------
        q = q_r
        #   transpose for classic (Q @ Kᵀ) formulation → [B, H, r, L]
        k = k_r.transpose(-1, -2)

        scores = torch.matmul(q, k)

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
                scores = scores.masked_fill(block_mask, float('inf'))
                mask = mask | block_mask

        # -- 3. softmax & value aggregation ------------------------------------
        attn = safe_softmax(scores, mask, dim=-1)  # [B, H, L]

        # fused gemm: (B·H)×L @ L×d_c  →  (B·H)×d_c
        return torch.matmul(attn, v_c)  # [B, H, d_c]

    def forward(self, hidden_q: torch.Tensor, kv_c: torch.Tensor, block_mask=None) -> torch.Tensor:
        B, S, _ = hidden_q.shape
        _, L, _d = kv_c.shape
        dev, dt = kv_c.device, kv_c.dtype

        # 1) latent → head space
        q_hk = self.q_proj(hidden_q).view(B, S, self.h, self.k)  # [B,S,H,K]

        # 2) big compressed view (only one needed now)
        q_big = torch.einsum('bshk,hkq->bshq', q_hk, self.w_kc_q)  # [B,S,H,d_c′]

        # 3) rotary on KV only
        sin, cos = _RoPECache.get(L, self.d_c, dev, dt)
        kv_c = _apply_rope(kv_c, sin[None], cos[None])  # [B,L,d_c]

        # 4) retrieval space
        q_r = torch.einsum('bshq,hqr->bshr', q_big, self.W_qr)  # [B,S,H,r]
        q_r = q_r.permute(0, 2, 1, 3)  # [B,H,S,r]

        k_r = torch.einsum('bld,hdr->bhlr', kv_c, self.W_kr)  # [B,H,L,r]
        v_c = kv_c.unsqueeze(1).expand(-1, self.h, -1, -1)  # [B,H,L,d_c]

        # 5) attention
        ctx_c = (self._flex_attention(q_r, k_r, v_c, block_mask=block_mask)
                 if self.use_flex else
                 self._fallback_attention(q_r, k_r, v_c, block_mask=block_mask))  # [B,H,d_c]

        # 6) compressed → latent
        ctx_lat = torch.einsum('bhsd,hdK->bhsK', ctx_c, self.w_kc_kv_T)  # [B,H,S,K]
        ctx_lat = ctx_lat.permute(0, 2, 1, 3).reshape(B, S, -1)  # [B,S,H*K]

        # 7) output
        return self.out_proj(ctx_lat)  # [B,D]




@lru_cache(maxsize=64)
def _cached_sliding_mask(seq_len, window, device):
    def mask_mod(b, h, q, k):
        #  k can’t be before q and must be within the look‑back window
        return (k <= q) & ((q - k) <= window)  # all terms non‑negative

    return create_block_mask(
        mask_mod,
        B=None, H=None,
        Q_LEN=seq_len, KV_LEN=seq_len,
        BLOCK_SIZE=128,
        device=device,
    )

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

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        B, S, _ = x.shape
        dev = x.device

        kv_c = self.kv_proj(x)                     # [B,S,d_c]
        if key_padding_mask is not None:
            kv_c = kv_c.masked_fill(key_padding_mask[..., None], 0.)

        block_mask = None
        # If we are NOT using flex_attention, we must create a simple boolean mask tensor.
        if not self.mla.use_flex:
            # Create a standard [S, S] sliding window mask.
            q_indices = torch.arange(S, device=dev)[:, None]
            k_indices = torch.arange(S, device=dev)[None, :]

            # True for positions within the causal sliding window
            window_mask = (k_indices >= q_indices - self.window) & (k_indices <= q_indices)

            # The `masked_fill` function in fallback attention expects True for positions to MASK.
            # So, we invert the window mask.
            block_mask = ~window_mask
            block_mask = block_mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, S, S]

            # Also incorporate the key_padding_mask if it exists
            if key_padding_mask is not None:
                # key_padding_mask is True for padding. Shape [B, S].
                # Reshape to [B, 1, 1, S] to broadcast with scores and window mask.
                padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
                # Combine masks. A position is masked if it's outside the window OR it's a pad token.
                block_mask = block_mask | padding_mask_expanded
        else:
            block_mask = _cached_sliding_mask(S, self.window, dev)
            if key_padding_mask is not None:
                pad = key_padding_mask.to(torch.bool)
                def _keep(b,h,q,k): return ~pad[b,k]
                block_mask = create_block_mask(
                    and_masks(block_mask.mask_mod, _keep),
                    B=pad.size(0), H=None,
                    Q_LEN=S, KV_LEN=S,
                    BLOCK_SIZE=block_mask.BLOCK_SIZE,
                    device=dev
                )

        ctx = self.mla(x, kv_c, block_mask=block_mask)  # single MLA call
        return self.o_proj(ctx)


@torch.compile
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
            use_flex=use_flex_attention,
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
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)

        # Feed‑forward (Pre‑LN)
        x = x + self.ffn(self.norm2(x))
        return x