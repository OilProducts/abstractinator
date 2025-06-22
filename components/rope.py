import torch
from typing import Dict

class RotaryCache:
    _store: dict[tuple[int, int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def get(max_seq, dim, device, dtype):
        key = (max_seq, dim, device, dtype)
        if key not in RotaryCache._store:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            t = torch.arange(max_seq, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            cos = freqs.cos()[None, None, :, :].to(dtype)
            sin = freqs.sin()[None, None, :, :].to(dtype)
            RotaryCache._store[key] = (cos, sin)
        return RotaryCache._store[key]



def apply_rope(x: torch.Tensor, seq_pos: slice | None = None) -> torch.Tensor:
    """Apply Rotary Positional Embedding to a (B,H,S,D) tensor."""
    cos, sin = RotaryCache.get(x.size(-2), x.size(-1), x.device, x.dtype)
    if seq_pos is not None:
        cos, sin = cos[..., seq_pos, :], sin[..., seq_pos, :]

    x_even, x_odd = x[..., ::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out
