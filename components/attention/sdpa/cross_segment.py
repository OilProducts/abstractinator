from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from components.rope import RoPECache, apply_rope


class SegmentCausalCrossAttention(nn.Module):
    """
    Cross-attention where queries attend to a compressed memory of segments.
    Uses RoPE and a fixed lookback over segment ids. Ported from
    `components/sliding_window_attention.py` (behavior preserved).
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
        device: torch.device | str = "cuda",
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

        self.rope_cache = RoPECache(max_seqlen=8192, head_dim=self.hdim, dtype=torch.bfloat16, device=device)
        assert (self.hdim % 2) == 0, "RoPE requires even head dim per head"
        half_in_cache = self.rope_cache.cos.size(-1)
        assert half_in_cache * 2 == self.hdim, "RoPECache/head_dim mismatch"

        self.q_proj = nn.Linear(q_dim, d_attn, bias=bias)
        self.kv_proj = nn.Linear(kv_dim, 2 * d_attn, bias=bias)
        self.o_proj = nn.Linear(d_attn, q_dim, bias=bias)
        self.drop = nn.Dropout(dropout)

        self.register_buffer("offsets", torch.arange(self.lookback + 1), persistent=False)

    def forward(
        self,
        q: torch.Tensor,                   # (B, Lq, q_dim)
        kv_src: torch.Tensor,              # (B, Lkv, kv_dim)
        seg_id: torch.Tensor,              # (B, Lq) int (used for lookback window)
        q_pos_ids: torch.Tensor,           # (B, Lq) or (Lq,)
        kv_pos_ids: torch.Tensor,          # (Lkv,)
        kv_mask: Optional[torch.Tensor] = None,  # (B, Lkv) bool – True => mask out
        q_pad_mask: Optional[torch.Tensor] = None,  # (B, Lq) bool – True => padding
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        device, dtype = q.device, q.dtype
        H, Dh = self.n_heads, self.hdim

        # RoPE tables (views)
        cos_all = self.rope_cache.cos.to(dtype=dtype, device=device)
        sin_all = self.rope_cache.sin.to(dtype=dtype, device=device)
        Smax = cos_all.size(2)

        # Q projection & RoPE(Q)
        qh = self._split_heads(self.q_proj(q))  # (B,H,Lq,Dh)
        if q_pos_ids.dim() == 1:
            q_pos = q_pos_ids.to(device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        else:
            q_pos = q_pos_ids.to(device=device, dtype=torch.long)
        q_pos = torch.clamp(q_pos, 0, Smax - 1)
        q_cos = torch.take_along_dim(cos_all.expand(B, 1, -1, -1), q_pos[:, None, :, None], dim=2)
        q_sin = torch.take_along_dim(sin_all.expand(B, 1, -1, -1), q_pos[:, None, :, None], dim=2)
        qh = apply_rope(qh, q_cos, q_sin)

        # K/V from memory (optionally cached)
        has_cache = (
            cache is not None
            and isinstance(cache.get("k", None), torch.Tensor)
            and isinstance(cache.get("v", None), torch.Tensor)
        )

        if has_cache:
            kh = cache["k"].clone(memory_format=torch.contiguous_format)  # (B,H,Lkv,Dh)
            vh = cache["v"].clone(memory_format=torch.contiguous_format)  # (B,H,Lkv,Dh)
        else:
            kv = self.kv_proj(kv_src)  # (B,Lkv,2*D)
            k, v = kv.split(self.d_attn, dim=-1)
            kh = self._split_heads(k)  # (B,H,Lkv,Dh)
            vh = self._split_heads(v)  # (B,H,Lkv,Dh)

        Lkv = kh.size(2)
        Kw = self.lookback + 1
        offsets = self.offsets.to(device)
        gather = seg_id.to(device).unsqueeze(-1) - offsets  # (B,Lq,Kw)
        neg_mask = gather < 0
        gather_clamped = torch.clamp(gather, 0, Lkv - 1)

        # Select K/V windows
        idx = gather_clamped.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, -1, Dh)  # (B,H,Lq,Kw,Dh)
        k_sel = torch.take_along_dim(kh.unsqueeze(3), idx, dim=2)
        v_sel = torch.take_along_dim(vh.unsqueeze(3), idx, dim=2)

        # RoPE(K) after selection using per-window positions
        kv_pos_ids = kv_pos_ids.to(device=device, dtype=torch.long)  # (Lkv,)
        k_pos = torch.take_along_dim(
            kv_pos_ids.view(1, 1, Lkv, 1).expand(B, 1, -1, Kw),
            gather_clamped.unsqueeze(1), dim=2,
        )
        k_pos = torch.clamp(k_pos, 0, Smax - 1)
        k_cos = torch.take_along_dim(
            cos_all.unsqueeze(3).expand(B, 1, -1, Kw, cos_all.size(-1)),
            k_pos.unsqueeze(-1), dim=2,
        )
        k_sin = torch.take_along_dim(
            sin_all.unsqueeze(3).expand(B, 1, -1, Kw, sin_all.size(-1)),
            k_pos.unsqueeze(-1), dim=2,
        )
        k_sel = apply_rope(k_sel, k_cos, k_sin)

        # Attention
        scores = (qh.unsqueeze(-2) * k_sel).sum(dim=-1) * self.scale  # (B,H,Lq,Kw)

        # Masks
        if kv_mask is None:
            kvm_win = torch.zeros((B, Lq, Kw), dtype=torch.bool, device=device)
        else:
            kv_mask_b = kv_mask.to(device=device, dtype=torch.bool)
            b_idx = torch.arange(B, device=device)[:, None, None]
            kvm_win = kv_mask_b[b_idx, gather_clamped]  # (B,Lq,Kw)

        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(kvm_win.unsqueeze(1), neg_inf)
        scores = scores.masked_fill(neg_mask.unsqueeze(1), neg_inf)

        probs = self.drop(F.softmax(scores, dim=-1))  # (B,H,Lq,Kw)
        out = (probs.unsqueeze(-1) * v_sel).sum(dim=-2)  # (B,H,Lq,Dh)
        out = out.transpose(1, 2).reshape(B, Lq, self.d_attn)
        out = self.o_proj(out)

        if q_pad_mask is not None:
            out = out.masked_fill(q_pad_mask.to(device=device, dtype=torch.bool).unsqueeze(-1), 0.0)
        return out

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return (
            x.reshape(B, L, self.n_heads, self.hdim).permute(0, 2, 1, 3).contiguous()
        )

