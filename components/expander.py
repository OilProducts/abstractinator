from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPECache, apply_rope
from .attention.cache import AttnCache
from .attention.base import SegmentContext
from .attention.sdpa.adapter import SDPASegmentCrossAttention
from .config_types import AttentionConfig
from .swiglu import SwiGLU
from .vector_quantizer import MultiStageResidualVQ, RVQEmbeddingAdapter, RVQFactorizedHead, ComposedIndexCodec, \
    LearnedCodebookAdapter
from .mla import MultiheadLatentAttention, CausalMLA


@lru_cache(maxsize=64)
def _cached_causal_mask(length: int) -> torch.Tensor:
    """Return a causal mask computed on CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
    return mask


def get_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Return the cached causal mask on ``device``."""
    return _cached_causal_mask(length).to(device)


# ------------------------- SDPA + fused QKV -------------------------

def _to_additive_mask(
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    B: int, H: int, L_q: int, L_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Convert boolean / mixed masks to an *additive* attention bias suitable for SDPA.
    Result is broadcastable to (B, H, L_q, L_k). Returns None if no masking needed.
    Using an additive mask preserves fast SDPA/Flash paths better than boolean masks.
    """
    bias = None

    # Handle attn_mask: accept (L_q, L_k), (B, L_q, L_k), or (B, H, L_q, L_k)
    if attn_mask is not None:
        m = attn_mask
        if m.dim() == 2:             # (L_q, L_k)
            m = m.unsqueeze(0).unsqueeze(0)      # (1,1,L_q,L_k)
        elif m.dim() == 3:           # (B, L_q, L_k)
            m = m.unsqueeze(1)                   # (B,1,L_q,L_k)
        # else: assume (B, H, L_q, L_k)

        if m.dtype == torch.bool:
            neg = torch.finfo(dtype).min
            bias = torch.where(m.to(device), torch.tensor(neg, dtype=dtype, device=device), torch.tensor(0, dtype=dtype, device=device))
        else:
            bias = m.to(dtype=dtype, device=device)

    # Fold key_padding_mask: (B, L_k) -> (B,1,1,L_k)
    if key_padding_mask is not None:
        kpm = key_padding_mask.to(device=device).bool().view(B, 1, 1, L_k)
        neg = torch.finfo(dtype).min
        kpm_bias = torch.where(kpm, torch.tensor(neg, dtype=dtype, device=device), torch.tensor(0, dtype=dtype, device=device))
        bias = kpm_bias if bias is None else (bias + kpm_bias)

    return bias

class MLASelfAttention(nn.Module):
    """
    Drop-in for MultiHeadAttentionFused using MultiheadLatentAttention (MLA).
    Preserves: attn_mask (bool), key_padding_mask. Causality enforced via block mask.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        head_dim: Optional[int] = None,     # default: d_model // num_heads
        kv_comp_dim: int = 32,              # c
        q_comp_dim: int = 16,               # d_c'
        retr_dim: int = 32,                 # r (even)
        rope_max_seqlen: int = 8192,
        use_flex_attention: bool = False,   # keep False unless you wired flex_attention/safe_softmax
    ):
        super().__init__()
        H = num_heads
        K = head_dim or (d_model // num_heads)
        assert retr_dim % 2 == 0, "retr_dim must be even"
        # Compress values/keys into c once per token
        self.kv_comp = nn.Linear(d_model, kv_comp_dim, bias=False)
        self.mla = MultiheadLatentAttention(
            dim_q=d_model, num_heads=H, head_dim=K,
            kv_comp_dim=kv_comp_dim, q_comp_dim=q_comp_dim, retr_dim=retr_dim,
            rope_max_seqlen=rope_max_seqlen, use_flex_attention=use_flex_attention
        )

    @staticmethod
    def _build_block_mask(
        Lq: int, Lk: int, device: torch.device,
        attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Returns boolean mask (True = mask out) shaped [B,1,Lq,Lk] or [1,1,Lq,Lk].
        Combines: causal upper‑tri, attn_mask (bool/broadcast), and key_padding_mask.
        """
        # Causal mask (Lq x Lk): True where k > q
        causal = torch.triu(torch.ones((Lq, Lk), dtype=torch.bool, device=device), diagonal=1)

        mask = causal.unsqueeze(0).unsqueeze(0)  # [1,1,Lq,Lk]

        if attn_mask is not None:
            m = attn_mask
            if m.dtype != torch.bool:
                # Treat large negative additive bias as masked
                m = (m <= torch.finfo(m.dtype).min / 2).to(torch.bool)
            if m.dim() == 2:            # (Lq, Lk)
                m = m[None, None, :, :]
            elif m.dim() == 3:          # (B, Lq, Lk)
                m = m[:, None, :, :]
            # else assume (B,H,Lq,Lk) or (B,1,Lq,Lk)
            mask = mask | m.to(device)

        if key_padding_mask is not None:
            # (B, Lk) True = PAD ⇒ mask out entire column
            kpm = key_padding_mask[:, None, None, :].to(device=device, dtype=torch.bool)
            mask = mask | kpm

        return mask

    def forward(
        self,
        query: torch.Tensor,                          # (B, L, D)
        key: Optional[torch.Tensor] = None,           # ignored; self-attn
        value: Optional[torch.Tensor] = None,         # ignored; self-attn
        attn_mask: Optional[torch.Tensor] = None,     # bool or additive
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=pad
        cache: Optional[Dict[str, torch.Tensor]] = None,   # optional in future
    ) -> torch.Tensor:
        x = query
        B, L, D = x.shape
        dev = x.device

        # Compress once per token for values/keys
        kv_c = self.kv_comp(x)  # [B, L, c]

        block_mask = self._build_block_mask(L, L, dev, attn_mask, key_padding_mask)  # causal + extras

        # Training/block mode (no cache here; you can add MLA caches later if needed)
        out = self.mla(hidden_q=x, kv_c=kv_c, block_mask=block_mask)
        return out

class MLASegmentedCrossAttention(nn.Module):
    """
    Drop-in for SegmentCausalCrossAttention using MLA.
    Enforces segment lookback with a boolean block mask:
      allow keys where (q_seg - lookback) <= k_seg <= q_seg.
    """
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        d_attn: int,         # keep = q_dim for parity with your block
        n_heads: int,
        lookback: int = 0,
        kv_comp_dim: int = 32,
        q_comp_dim: int = 16,
        retr_dim: int = 32,
        rope_max_seqlen: int = 8192,
        use_flex_attention: bool = False,
    ):
        super().__init__()
        assert d_attn == q_dim, "Set d_attn=q_dim to mirror your original block."
        self.lookback = int(lookback)

        # Compress memory to c
        self.kv_comp = nn.Linear(kv_dim, kv_comp_dim, bias=False)

        # MLA over (q_dim, n_heads)
        K = q_dim // n_heads
        self.mla = MultiheadLatentAttention(
            dim_q=q_dim, num_heads=n_heads, head_dim=K,
            kv_comp_dim=kv_comp_dim, q_comp_dim=q_comp_dim, retr_dim=retr_dim,
            rope_max_seqlen=rope_max_seqlen, use_flex_attention=use_flex_attention
        )

    @staticmethod
    def _combine_masks(
        base: Optional[torch.Tensor],    # [B,1,Lq,Lk] or None
        extra: Optional[torch.Tensor]    # [B,1,Lq,Lk] or None
    ) -> Optional[torch.Tensor]:
        if base is None:  return extra
        if extra is None: return base
        return base | extra

    def _lookback_mask(self, seg_id_q: torch.Tensor, Lk: int) -> torch.Tensor:
        """
        Build [B,1,Lq,Lk] boolean mask (True = mask out) based on segment lookback.
        k_seg is assumed to be 0..Lk-1 (one segment per memory position).
        """
        B, Lq = seg_id_q.shape
        dev = seg_id_q.device
        k_seg = torch.arange(Lk, device=dev).view(1, 1, Lk)      # [1,1,Lk]
        q_seg = seg_id_q.view(B, Lq, 1)                          # [B,Lq,1]
        good = (k_seg <= q_seg) & (k_seg >= (q_seg - self.lookback))
        mask = ~good                                             # [B,Lq,Lk]
        return mask.unsqueeze(1)                                 # [B,1,Lq,Lk]

    def forward(
        self,
        q: torch.Tensor,                   # (B, Lq, q_dim)
        kv_src: torch.Tensor,              # (B, Lk, kv_dim)
        seg_id: torch.Tensor,              # (B, Lq) int
        q_pos_ids: torch.Tensor,           # (B, Lq) or (Lq,)  (unused here; MLA handles RoPE internally)
        kv_pos_ids: torch.Tensor,          # (Lk,)             (unused; absolute write in training)
        kv_mask: Optional[torch.Tensor] = None,     # (B, Lk) bool True = pad
        q_pad_mask: Optional[torch.Tensor] = None,  # (B, Lq) bool True = pad (zeroed after)
        cache: Optional[Dict[str, torch.Tensor]] = None,  # optional in future
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        Lk = kv_src.size(1)
        dev = q.device

        kv_c = self.kv_comp(kv_src)  # [B, Lk, c]

        # Segment lookback ⇒ block mask
        block = self._lookback_mask(seg_id.to(device=dev, dtype=torch.long), Lk)

        # Padding on keys
        if kv_mask is not None:
            kpm = kv_mask[:, None, None, :].to(device=dev, dtype=torch.bool)  # [B,1,1,Lk]
            block = self._combine_masks(block, kpm)


        out = self.mla(q, kv_c, block_mask=block)  # [B,Lq,q_dim]

        if q_pad_mask is not None:
            out = out.masked_fill(q_pad_mask.to(device=dev, dtype=torch.bool).unsqueeze(-1), 0.0)
        return out




class MultiHeadAttentionFused(nn.Module):
    """
    Multi-Head Attention with:
      - fused QKV projection (single matmul)
      - RoPE applied to Q and K
      - PyTorch SDPA backend (routes to Flash/Memory-Efficient/Math as appropriate)

    API is compatible with the original class (self/cross usage).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        causal: bool = False,
        rope_max_seqlen: int = 2048,
        rope_base: float = 10000.0,
        rope_dtype: torch.dtype = torch.bfloat16,
        rope_device: torch.device | str = "cuda",
        bias: bool = True,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        # Fused QKV projection. We reuse slices of this weight even for cross-attn (no extra params).
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # RoPE cache (moves with module.to(device))
        self.rope = RoPECache(
            max_seqlen=rope_max_seqlen,
            head_dim=self.head_dim,
            base=rope_base,
            dtype=rope_dtype,
            device=rope_device,
        )

    # --- lightweight helpers ---
    def _shape(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """(B, L, D) -> (B, H, L, d)"""
        H, d = self.num_heads, self.head_dim
        return x.view(B, L, H, d).transpose(1, 2)

    def _q_from(self, x: torch.Tensor) -> torch.Tensor:
        W, b = self.in_proj.weight, self.in_proj.bias
        D = self.d_model
        return F.linear(x, W[:D, :], b[:D] if b is not None else None)

    def _kv_from(self, x: torch.Tensor) -> torch.Tensor:
        W, b = self.in_proj.weight, self.in_proj.bias
        D = self.d_model
        return F.linear(x, W[D:, :], b[D:] if b is not None else None)  # (B,L,2D)

    # --- forward ---
    def forward(
        self,
        query: torch.Tensor,                          # (B, L_q, D)
        key: Optional[torch.Tensor] = None,           # (B, L_k, D) or None -> self-attn
        value: Optional[torch.Tensor] = None,         # (B, L_k, D)
        attn_mask: Optional[torch.Tensor] = None,     # (L_q,L_k), (B,L_q,L_k), or (B,H,L_q,L_k); bool or additive
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L_k) True = pad
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key

        B, L_q, _ = query.shape
        L_k = key.size(1)

        # Self-attn path: do one fused projection. Cross-attn path: slice weights to avoid
        # computing unused 'q' for the memory (no extra parameters, minimal compute waste).
        if key is query and value is key:
            qkv = self.in_proj(query)                              # (B,L_q,3D)
            q, k, v = qkv.split(self.d_model, dim=-1)              # each (B,L_q,D)
        else:
            q = self._q_from(query)                                # (B,L_q,D)
            kv = self._kv_from(key)                                # (B,L_k,2D)
            k, v = kv.split(self.d_model, dim=-1)                  # (B,L_k,D)
            if value is not key:
                # Use the v-slice directly on value
                W, b = self.in_proj.weight, self.in_proj.bias
                D = self.d_model
                v = F.linear(value, W[2*D:3*D, :], b[2*D:3*D] if b is not None else None)

        # Reshape to (B,H,L, d)
        q = self._shape(q, B, L_q)
        k = self._shape(k, B, L_k)
        v = self._shape(v, B, L_k)

        # RoPE (use per-length slices; RoPE buffers move with module.to(...))
        cos_q, sin_q = self.rope.slice(L_q)
        cos_k, sin_k = self.rope.slice(L_k)
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)

        # Merge masks into an additive bias (prefer fast SDPA paths).
        bias = _to_additive_mask(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            B=B, H=self.num_heads, L_q=L_q, L_k=L_k,
            dtype=q.dtype, device=q.device,
        )

        # SDPA routes to Flash/Memory-Efficient/Math automatically.
        # Prefer passing `is_causal=True` (and no causal mask) for fastest path.
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias,               # additive or None
            is_causal=self.causal and (attn_mask is None),
        )  # (B,H,L_q,d)

        out = attn.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.out_proj(out)


class SegmentCausalCrossAttention(nn.Module):
    """
    Cross-attention where queries attend to a *compressed* memory of segments.
    RoPE is mandatory and applied segment-relatively (Q: per query positions;
    K: per selected memory segment positions).
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
        device: torch.device = torch.device("cuda"),
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
        self.device = device

        self.rope_cache = RoPECache(max_seqlen=8192, head_dim=self.hdim, dtype=torch.bfloat16, device=device)

        assert (self.hdim % 2) == 0, "RoPE requires even head dim per head"
        # cache must match Dh/2
        half_in_cache = self.rope_cache.cos.size(-1)
        assert half_in_cache * 2 == self.hdim, \
            f"RoPECache head_dim mismatch: cache half={half_in_cache}, Dh/2={self.hdim//2}"


        # Projections
        self.q_proj = nn.Linear(q_dim, d_attn, bias=bias)
        self.kv_proj = nn.Linear(kv_dim, 2 * d_attn, bias=bias)
        self.o_proj = nn.Linear(d_attn, q_dim, bias=bias)
        self.drop = nn.Dropout(dropout)

        # small perf: reuse offsets tensor
        self.register_buffer("offsets", torch.arange(self.lookback + 1), persistent=False)

    def forward(
        self,
        q: torch.Tensor,                   # (B, Lq, q_dim)
        kv_src: torch.Tensor,              # (B, Lkv, kv_dim)
        seg_id: torch.Tensor,              # (B, Lq) int   (used for windowing)
        q_pos_ids: torch.Tensor,           # (B, Lq) or (Lq,) int  (positions for Q)
        kv_pos_ids: torch.Tensor,          # (Lkv,) int            (positions for K/V segments)
        kv_mask: Optional[torch.Tensor] = None,   # (B, Lkv) bool – True => mask out
        q_pad_mask: Optional[torch.Tensor] = None,# (B, Lq)  bool – True => padding
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        device, dtype = q.device, q.dtype
        H, Dh = self.n_heads, self.hdim

        # --- RoPE tables (views) with dtype/device hygiene
        cos_all = self.rope_cache.cos.to(dtype=dtype, device=device)  # (1,1,Smax,Dh/2)
        sin_all = self.rope_cache.sin.to(dtype=dtype, device=device)
        Smax = cos_all.size(2)

        # --- 1) Q projection & RoPE(Q)
        qh = self._split_heads(self.q_proj(q))  # (B,H,Lq,Dh)

        # positions for Q: prefer per-batch (B,Lq); if (Lq,), broadcast
        if q_pos_ids.dim() == 1:
            q_pos = q_pos_ids.to(device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        else:
            q_pos = q_pos_ids.to(device=device, dtype=torch.long)
        q_pos = torch.clamp(q_pos, 0, Smax - 1)                          # (B,Lq)

        # gather cos/sin for Q: (B,1,Lq,Dh/2), broadcast over heads
        q_cos = torch.take_along_dim(cos_all.expand(B, 1, -1, -1), q_pos[:, None, :, None], dim=2)
        q_sin = torch.take_along_dim(sin_all.expand(B, 1, -1, -1), q_pos[:, None, :, None], dim=2)
        qh = apply_rope(qh, q_cos, q_sin)                                 # (B,H,Lq,Dh)

        # --- 2) K/V: cached vs recompute (use python bool)
        has_cache = (
            cache is not None
            and isinstance(cache.get("k", None), torch.Tensor)
            and isinstance(cache.get("v", None), torch.Tensor)
        )

        if has_cache:
            kh = cache["k"].clone(memory_format=torch.contiguous_format)  # (B,H,Lkv,Dh), *unrotated*
            vh = cache["v"].clone(memory_format=torch.contiguous_format)  # (B,H,Lkv,Dh)
        else:
            kv = self.kv_proj(kv_src)                                     # (B,Lkv,2*D)
            k, v = kv.split(self.d_attn, dim=-1)                          # (B,Lkv,D), (B,Lkv,D)
            kh = self._split_heads(k)                                     # (B,H,Lkv,Dh)
            vh = self._split_heads(v)                                     # (B,H,Lkv,Dh)

        Lkv = kh.size(2)
        Kw = self.lookback + 1

        # --- 3) Segment window indices
        offsets = self.offsets.to(device)  # (Kw,)
        gather = seg_id.to(device).unsqueeze(-1) - offsets                 # (B,Lq,Kw)
        neg_mask = gather < 0
        gather_clamped = torch.clamp(gather, 0, Lkv - 1)                   # (B,Lq,Kw)

        # --- 4) Select K/V windows
        idx = gather_clamped.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, -1, Dh)  # (B,H,Lq,Kw,Dh)
        k_sel = torch.take_along_dim(kh.unsqueeze(3), idx, dim=2)           # (B,H,Lq,Kw,Dh)
        v_sel = torch.take_along_dim(vh.unsqueeze(3), idx, dim=2)           # (B,H,Lq,Kw,Dh)

        # --- 4b) RoPE(K) *after* selection using per-window positions
        # build per-window K positions from kv_pos_ids
        kv_pos_ids = kv_pos_ids.to(device=device, dtype=torch.long)        # (Lkv,)
        # (B,1,Lq,Kw)
        k_pos = torch.take_along_dim(
            kv_pos_ids.view(1, 1, Lkv, 1).expand(B, 1, -1, Kw),
            gather_clamped.unsqueeze(1), dim=2
        )
        k_pos = torch.clamp(k_pos, 0, Smax - 1)                             # (B,1,Lq,Kw)

        # gather cos/sin for these positions: (B,1,Lq,Kw,Dh/2)
        k_cos = torch.take_along_dim(
            cos_all.unsqueeze(3).expand(B, 1, -1, Kw, cos_all.size(-1)),
            k_pos.unsqueeze(-1), dim=2
        )
        k_sin = torch.take_along_dim(
            sin_all.unsqueeze(3).expand(B, 1, -1, Kw, sin_all.size(-1)),
            k_pos.unsqueeze(-1), dim=2
        )

        k_sel = apply_rope(k_sel, k_cos, k_sin)                             # (B,H,Lq,Kw,Dh)

        # --- 5) Attention
        scores = (qh.unsqueeze(-2) * k_sel).sum(dim=-1) * self.scale       # (B,H,Lq,Kw)

        # --- 6) Masks
        if kv_mask is None:
            kvm_win = torch.zeros((B, Lq, Kw), dtype=torch.bool, device=device)
        else:
            kv_mask_b = kv_mask.to(device=device, dtype=torch.bool)        # (B,Lkv)
            b_idx = torch.arange(B, device=device)[:, None, None]
            kvm_win = kv_mask_b[b_idx, gather_clamped]                      # (B,Lq,Kw)

        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(kvm_win.unsqueeze(1), neg_inf)
        scores = scores.masked_fill(neg_mask.unsqueeze(1), neg_inf)

        # --- 7) Softmax + mix
        probs = self.drop(F.softmax(scores, dim=-1))                        # (B,H,Lq,Kw)
        out = (probs.unsqueeze(-1) * v_sel).sum(dim=-2)                     # (B,H,Lq,Dh)

        # --- 8) Merge heads, project, pad mask
        out = out.transpose(1, 2).reshape(B, Lq, self.d_attn)               # (B,Lq,D)
        out = self.o_proj(out)                                              # (B,Lq,q_dim)

        if q_pad_mask is not None:
            out = out.masked_fill(q_pad_mask.to(device=device, dtype=torch.bool).unsqueeze(-1), 0.0)
        return out

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return (
            x.reshape(B, L, self.n_heads, self.hdim)
             .permute(0, 2, 1, 3)
             .contiguous()
        )


class SlidingDecoderBlock(nn.Module):
    """Decoder block using causal self-attention and sliding-window cross attention."""

    def __init__(self, idx: int, d_model: int, num_heads: int, cross_window: int, q_dim, kv_dim, *, cross_attn_config: "AttentionConfig" | None = None, self_attn_config: "AttentionConfig" | None = None):
        super().__init__()
        self.layer_id = f'L{idx}'
        self.norm1 = nn.RMSNorm(d_model)
        from .attention.factory import make_causal_self_block
        self.self_attn = make_causal_self_block(
            dim=d_model, num_heads=num_heads, ffn_dim_multiplier=4,
            cfg=self_attn_config or AttentionConfig(variant="mla", kernel="flex", kv_comp_dim=d_model // 16, q_comp_dim=d_model // 32, retr_dim=d_model // num_heads),
        )
        self.norm2 = nn.RMSNorm(d_model)
        lookback = cross_attn_config.lookback if (cross_attn_config and cross_attn_config.lookback is not None) else cross_window
        self.cross_attn = SDPASegmentCrossAttention(
            q_dim=q_dim, kv_dim=kv_dim, d_attn=d_model, n_heads=num_heads, lookback=lookback, bias=False
        )
        self.norm3 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, 4 * d_model)

    def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            q_pos_ids: torch.Tensor | None = None,
            kv_pos_ids: torch.Tensor | None = None,
            seg_ids: torch.Tensor | None = None,
            cache: AttnCache | None = None,
            tgt_mask: torch.Tensor | None = None,
            tgt_key_padding_mask: torch.Tensor | None = None,
            memory_key_padding_mask: torch.Tensor | None = None,
            # cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), key_padding_mask=tgt_key_padding_mask) # attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)

        seg_ctx = SegmentContext(
            seg_id=seg_ids,
            q_pos_ids=q_pos_ids if q_pos_ids is not None else torch.arange(x.size(1), device=x.device),
            kv_pos_ids=kv_pos_ids if kv_pos_ids is not None else torch.arange(memory.size(1), device=memory.device),
            lookback=0,
            kv_mask=memory_key_padding_mask,
            q_pad_mask=tgt_key_padding_mask,
        )
        x = x + self.cross_attn(self.norm2(x), memory, segment=seg_ctx)
        x = x + self.ffn(self.norm3(x))
        return x


class SlidingDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, cross_window: int, q_dim: int, kv_dim: int, *, cross_attn_config: "AttentionConfig" | None = None, self_attn_config: "AttentionConfig" | None = None):
        super().__init__()
        self.layers = nn.ModuleList()
        for l_idx in range(num_layers):
            layer = SlidingDecoderBlock(
                idx=l_idx, d_model=d_model, num_heads=num_heads, cross_window=cross_window, q_dim=q_dim, kv_dim=kv_dim,
                cross_attn_config=cross_attn_config, self_attn_config=self_attn_config,
            )
            self.layers.append(layer)

        self.final_norm = nn.RMSNorm(d_model)

    def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            q_pos_ids: torch.Tensor | None = None,
            kv_pos_ids: torch.Tensor | None = None,
            cache: AttnCache | None = None,
            tgt_mask: torch.Tensor | None = None,
            tgt_key_padding_mask: torch.Tensor | None = None,
            memory_key_padding_mask: torch.Tensor | None = None,
            # cross_attn_mask: Optional[torch.Tensor] = None,
            seg_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                memory,
                q_pos_ids=q_pos_ids,
                kv_pos_ids=kv_pos_ids,
                cache=cache,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                # cross_attn_mask=cross_attn_mask,
                seg_ids=seg_ids,
            )
        return self.final_norm(x)


class DecoderOnlyExpanderRVQ(nn.Module):
    """
    Decoder-only expander that:
      - embeds high codes by reusing the *high-level* MS-RVQ codebooks (no emb_hi),
      - embeds low codes likewise (no emb_lo),
      - outputs factorized stage-wise logits (no D×K_eff softmax).
    """

    def __init__(self,
                 lo_vq: "MultiStageResidualVQ",  # MS-RVQ for the *target* code space
                 hi_vq: Optional["MultiStageResidualVQ"],  # MS-RVQ for the *memory* code space
                 K_lo: Optional[int] = None,  # if lo_vq is None
                 D: int = 256,
                 N_dec: int = 4,
                 H: int = 8,
                 cross_window: int = 128,
                 eos_id: int = 257,
                 eop_id: int = 259,
                 max_len: int = 2048,
                 lo_d_c: int = 64,  # Factorized rank for bytes
                 residual_conditioning: bool = True,
                 use_sqdist_logits: bool = False,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.lo_vq = lo_vq
        self.hi_vq = hi_vq
        self.D = D
        self.eos_id = eos_id
        self.eop_id = eop_id
        self.max_len = max_len
        self.K_lo = K_lo

        # Reuse VQ projections and codebooks
        # ---- low side (what we predict) ----
        if lo_vq is not None:
            self.lo_adapt = RVQEmbeddingAdapter(lo_vq, target_D=D,
                                                use_shared_proj=True,
                                                tie_up_down=True)
            self.lo_codec = ComposedIndexCodec(K=lo_vq.K, depth=lo_vq.depth,
                                               bos=lo_vq.bos_token_id, eos=lo_vq.eos_token_id,
                                               pad=lo_vq.padding_token_id, eop=lo_vq.eop_token_id)

        else:
            assert K_lo is not None, "Need K_lo for bottom level when lo_vq is None."
            self.lo_adapt = LearnedCodebookAdapter(K=K_lo, D=D, d_c=lo_d_c, device=device, )
            self.lo_codec = ComposedIndexCodec(K=K_lo, depth=1, bos=-1, eos=-1, pad=-1, eop=-1)

        # ---- high side (memory) ----
        self.hi_adapt = RVQEmbeddingAdapter(hi_vq, target_D=D, use_shared_proj=True) if hi_vq is not None else None

        # Your existing decoder block
        self.decoder = SlidingDecoder(N_dec, D, H, cross_window, q_dim=D, kv_dim=D, cross_attn_config=None)

        # Factorized head
        self.head = RVQFactorizedHead(self.lo_adapt,
                                      residual_conditioning=residual_conditioning,
                                      use_sqdist_logits=use_sqdist_logits)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return get_causal_mask(length, device)

    def _embed_memory(self, codes_hi: torch.Tensor) -> torch.Tensor:
        # Either high side is already continuous (D), or embed via hi_vq
        if codes_hi.is_floating_point():
            return codes_hi
        if self.hi_adapt is None:
            raise ValueError("codes_hi is discrete but hi_vq/hi_adapt not provided")
        return self.hi_adapt.embed_composed(codes_hi)

    def _embed_decoder_inputs(self, codes_lo: torch.Tensor) -> torch.Tensor:
        # Shift-right with EOS
        decoder_input_ids = F.pad(codes_lo[:, :-1], (1, 0), value=self.eos_id)
        return self.lo_adapt.embed_composed(decoder_input_ids)

    def forward(self,
                codes_hi: torch.Tensor,  # (B, S_hi) or (B, S_hi, D)
                codes_lo: torch.Tensor,  # (B, L)
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                seg_ids: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        device = codes_hi.device
        memory = self._embed_memory(codes_hi)  # (B, S_hi, D)
        dec_inp = self._embed_decoder_inputs(codes_lo)  # (B, L, D)

        L_q = codes_lo.size(1)
        L_kv = memory.size(1)
        q_pos_ids = torch.arange(L_q, device=device)
        kv_pos_ids = torch.arange(L_kv, device=device)

        tgt_mask = self._causal_mask(codes_lo.size(1), device)
        adjusted_tgt_kpm = None
        if tgt_key_padding_mask is not None:
            adjusted_tgt_kpm = F.pad(tgt_key_padding_mask[:, :-1], (1, 0), value=False)

        h = self.decoder(dec_inp, memory,
                         q_pos_ids=q_pos_ids,
                         kv_pos_ids=kv_pos_ids,
                         tgt_mask=tgt_mask,
                         tgt_key_padding_mask=adjusted_tgt_kpm,
                         memory_key_padding_mask=src_key_padding_mask,
                         seg_ids=seg_ids)  # (B, L, D)

        # Teacher-forced residual conditioning uses *true* digits at (B,L)
        teacher_digits, _ = self.lo_codec.decompose(codes_lo)
        logits = self.head(h, teacher_digits=teacher_digits)  # dict

        return logits  # {"stage_logits": [B,L,K]*depth, "special_logits": (B,L,S) or None}

    @torch.no_grad()
    def generate(
            self,
            codes_hi: torch.Tensor,  # (B, S_hi) or (B, S_hi, D)
            codes_lo: torch.Tensor,  # (B, T0)  seed(s)
            src_key_padding_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,  # <- compat
            max_len: Optional[int] = None,
            max_new_tokens: Optional[int] = None,
            seg_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = codes_hi.device
        B = codes_hi.size(0)
        memory = self._embed_memory(codes_hi)  # (B, S_hi, D)

        kv_pos_ids = torch.arange(memory.size(1), device=device)

        generated = codes_lo.clone()  # (B, T)
        kpm = tgt_key_padding_mask.clone() if tgt_key_padding_mask is not None \
            else torch.zeros_like(generated, dtype=torch.bool)

        cache = AttnCache()

        # steps budget: how many *new* tokens we’re allowed to produce
        steps_budget = int(max_new_tokens) if max_new_tokens is not None else \
            int((max_len or self.max_len) - 1)
        steps_budget = max(0, steps_budget)

        # helper: align seg_ids length to current T
        def _align_seg_ids(curr_T: int) -> torch.Tensor:
            if seg_ids is None:
                last_seg = memory.size(1) - 1
                return torch.full((B, curr_T), last_seg, dtype=torch.long, device=device)
            if seg_ids.size(1) < curr_T:
                pad = seg_ids[:, -1:].expand(-1, curr_T - seg_ids.size(1))
                return torch.cat([seg_ids, pad], dim=1)
            if seg_ids.size(1) > curr_T:
                return seg_ids[:, :curr_T]
            return seg_ids

        depth = self.lo_codec.depth  # works for both MS-RVQ and learned single-stage
        new_tokens = torch.zeros((B, 0), dtype=generated.dtype, device=device)  # (B,0)
        for step in range(steps_budget):
            T = generated.size(1)
            q_pos_ids = torch.arange(T, device=device)

            dec_inp = self.lo_adapt.embed_composed(generated)  # (B, T, D)
            tgt_mask = self._causal_mask(T, device)
            seg_ids_cur = _align_seg_ids(T)  # (B, T)

            h = self.decoder(
                dec_inp, memory,
                q_pos_ids=q_pos_ids,
                kv_pos_ids=kv_pos_ids,
                cache=cache,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=kpm,
                memory_key_padding_mask=src_key_padding_mask,
                seg_ids=seg_ids_cur,
            )  # (B, T, D)

            last_h = h[:, -1, :]  # (B,1,D)
            r_dc = self.lo_adapt.down(last_h)  # (B,1,d_c)

            # Greedy stage-by-stage with residual updates
            digits = []
            for s in range(depth):
                W = self.lo_adapt.stage_codebook(s)  # (K, d_c)
                logit_s = self.head._stage_logits(r_dc, W)  # (B,1,K)
                idx_s = logit_s.argmax(dim=-1)  # (B,1)
                digits.append(idx_s)  # keep time dim
                r_dc = r_dc - F.embedding(idx_s, W)  # (B,1,d_c)

            next_id = self.lo_codec.compose(digits)  # (B,1) or (B,)
            if next_id.ndim == 1:
                next_id = next_id.unsqueeze(1)  # (B,1)

            # Append next_id to the sequence and extend masks accordingly
            generated = torch.cat([generated, next_id], dim=1)  # (B, T+1)
            kpm = F.pad(kpm, (0, 1), value=False)  # new token is real (not PAD)
            new_tokens = torch.cat([new_tokens, next_id], dim=1)
            # Early stop on EOS/EOP (after appending)
            if ((next_id == self.eos_id) | (next_id == self.eop_id)).all():
                break


        return new_tokens
