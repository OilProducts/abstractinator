from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from components.rope import RoPECache, apply_rope
from components.swiglu import SwiGLU

from ...backends.flex import run as flex_run


@lru_cache(maxsize=256)
def _cached_causal_block_mask_by_shape(B: int, H: int, Q: int, K: int, dev_type: str, dev_index: int):
    def keep(_b, _h, q, k):
        return k <= q

    device = torch.device(dev_type) if dev_index < 0 else torch.device(dev_type, dev_index)
    return create_block_mask(
        keep,
        B=B,
        H=H,
        Q_LEN=Q,
        KV_LEN=K,
        BLOCK_SIZE=128,
        device=device,
    )


@lru_cache(maxsize=256)
def _cached_sliding_block_mask_by_shape(B: int, H: int, Q: int, K: int, window: int, dev_type: str, dev_index: int):
    def keep(_b, _h, q, k):
        return (k <= q) & ((q - k) <= window)

    device = torch.device(dev_type) if dev_index < 0 else torch.device(dev_type, dev_index)
    return create_block_mask(
        keep,
        B=B,
        H=H,
        Q_LEN=Q,
        KV_LEN=K,
        BLOCK_SIZE=128,
        device=device,
    )


class CausalSelfFlexBlock(nn.Module):
    """
    Regular causal self-attention block implemented with FlexAttention.
    Mirrors SDPA TransformerBlock structure (Pre-LN + SwiGLU).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        ln_eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_ff = d_ff or (4 * d_model)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)

        self.ln1 = nn.LayerNorm(d_model, eps=ln_eps)
        self.ln2 = nn.LayerNorm(d_model, eps=ln_eps)
        self.mlp = nn.Sequential(
            SwiGLU(d_model, self.d_ff),
        )

        # RoPE cache (mandatory positional info)
        assert (self.head_dim % 2) == 0, "RoPE expects even head dim"
        self.rope_cache = RoPECache(
            max_seqlen=8192,
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            device="cpu",
        )

    def _shape(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        H, d = self.n_heads, self.head_dim
        return x.view(B, L, H, d).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = self._shape(q, B, L)
        k = self._shape(k, B, L)
        v = self._shape(v, B, L)

        # Apply RoPE to Q/K
        Smax = int(self.rope_cache.cos.size(2))
        if L > Smax:
            new_max = 1 << (L - 1).bit_length()
            self.rope_cache = RoPECache(max_seqlen=new_max, head_dim=self.head_dim, dtype=torch.bfloat16, device="cpu")
        cos = self.rope_cache.cos[..., :L, :].to(device=q.device, dtype=q.dtype)
        sin = self.rope_cache.sin[..., :L, :].to(device=q.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Cached block mask with explicit batch/head sizes
        Bn, Hn, Qn, Kn = q.size(0), q.size(1), q.size(2), k.size(2)
        dev_type = x.device.type
        dev_index = -1 if x.device.index is None else int(x.device.index)
        block = _cached_causal_block_mask_by_shape(Bn, Hn, Qn, Kn, dev_type, dev_index)

        pad = key_padding_mask.to(torch.bool) if key_padding_mask is not None else None

        def score_mod(scores, b, h, q_idx, k_idx):
            if pad is not None:
                return scores.masked_fill(pad[b, k_idx], float("-inf"))
            return scores

        attn = flex_run(q, k, v, block_mask=block, score_mod=score_mod)
        attn = attn.transpose(1, 2).contiguous().view(B, L, self.d_model)
        x = x + self.proj(attn)
        x = x + self.mlp(self.ln2(x))
        return x


class CausalLocalSelfFlexBlock(nn.Module):
    """
    Regular sliding-window causal self-attention with FlexAttention.
    Window is a half-width (radius) over the past.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        window_size: int,
        d_ff: Optional[int] = None,
        ln_eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = int(window_size)
        self.d_ff = d_ff or (4 * d_model)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.ln1 = nn.LayerNorm(d_model, eps=ln_eps)
        self.ln2 = nn.LayerNorm(d_model, eps=ln_eps)
        self.mlp = nn.Sequential(
            SwiGLU(d_model, self.d_ff),
        )

        # RoPE cache (mandatory positional info)
        assert (self.head_dim % 2) == 0, "RoPE expects even head dim"
        self.rope_cache = RoPECache(
            max_seqlen=8192,
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            device="cpu",
        )

    def _shape(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        H, d = self.n_heads, self.head_dim
        return x.view(B, L, H, d).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = self._shape(q, B, L)
        k = self._shape(k, B, L)
        v = self._shape(v, B, L)

        # Apply RoPE to Q/K
        Smax = int(self.rope_cache.cos.size(2))
        if L > Smax:
            new_max = 1 << (L - 1).bit_length()
            self.rope_cache = RoPECache(max_seqlen=new_max, head_dim=self.head_dim, dtype=torch.bfloat16, device="cpu")
        cos = self.rope_cache.cos[..., :L, :].to(device=q.device, dtype=q.dtype)
        sin = self.rope_cache.sin[..., :L, :].to(device=q.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Cached block mask with explicit batch/head sizes
        Bn, Hn, Qn, Kn = q.size(0), q.size(1), q.size(2), k.size(2)
        dev_type = x.device.type
        dev_index = -1 if x.device.index is None else int(x.device.index)
        block = _cached_sliding_block_mask_by_shape(Bn, Hn, Qn, Kn, int(self.window), dev_type, dev_index)
        pad = key_padding_mask.to(torch.bool) if key_padding_mask is not None else None

        def score_mod(scores, b, h, q_idx, k_idx):
            if pad is not None:
                return scores.masked_fill(pad[b, k_idx], float("-inf"))
            return scores

        attn = flex_run(q, k, v, block_mask=block, score_mod=score_mod)
        attn = attn.transpose(1, 2).contiguous().view(B, L, self.d_model)
        x = x + self.proj(attn)
        x = x + self.mlp(self.ln2(x))
        return x
