# import torch
#
#
# # rope.py
# import torch
#
# class RotaryCache:
#     _store: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
#
#     @classmethod
#     @torch.no_grad()
#     @torch._dynamo.disable()  # <-- critical: don't let compile/cudagraphs see/cache these
#     def get(cls, seqlen: int, head_dim: int, device, dtype, base: float = 10000.0):
#         """
#         Returns (cos, sin) shaped (1, 1, seqlen, head_dim//2), on the requested device/dtype.
#         Safe under torch.compile + CUDA graphs.
#         """
#         key = (int(seqlen), int(head_dim), torch.device(device).type, str(dtype))
#         if key in cls._store:
#             cos, sin = cls._store[key]
#             # If device/dtype mismatch (e.g., moved model), recast once
#             if cos.device != device or cos.dtype != dtype:
#                 cos, sin = cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)
#                 cls._store[key] = (cos, sin)
#             return cos, sin
#
#         # Build in plain eager (no capture), then clone to ensure independent storage
#         # Build at float32 then cast once to target dtype
#         t = torch.arange(seqlen, device=device, dtype=torch.float32)                         # (S,)
#         inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))  # (H/2,)
#         freqs = torch.outer(t, inv)                                                          # (S, H/2)
#
#         cos = freqs.cos().to(dtype).contiguous().clone().unsqueeze(0).unsqueeze(0)          # (1,1,S,H/2)
#         sin = freqs.sin().to(dtype).contiguous().clone().unsqueeze(0).unsqueeze(0)          # (1,1,S,H/2)
#
#         cls._store[key] = (cos, sin)
#         return cos, sin
#
# # class RotaryCache:
# #     _store: dict[tuple[int, int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
# #
# #     @staticmethod
# #     def get(max_seq, dim, device, dtype):
# #         key = (max_seq, dim, device, dtype)
# #         if key not in RotaryCache._store:
# #             inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
# #             t = torch.arange(max_seq, device=device, dtype=torch.float32)
# #             freqs = torch.outer(t, inv_freq)
# #             cos = freqs.cos()[None, None, :, :].to(dtype)
# #             sin = freqs.sin()[None, None, :, :].to(dtype)
# #             RotaryCache._store[key] = (cos, sin)
# #         return RotaryCache._store[key]
#
#
# def apply_rope(x: torch.Tensor, seq_pos: slice | None = None) -> torch.Tensor:
#     """Apply Rotary Positional Embedding to a (B,H,S,D) tensor."""
#     cos, sin = RotaryCache.get(x.size(-2), x.size(-1), x.device, x.dtype)
#     if seq_pos is not None:
#         cos, sin = cos[..., seq_pos, :], sin[..., seq_pos, :]
#
#     x_even, x_odd = x[..., ::2], x[..., 1::2]
#     out = torch.empty_like(x)
#     out[..., ::2] = x_even * cos - x_odd * sin
#     out[..., 1::2] = x_even * sin + x_odd * cos
#     return out


# rope.py
import torch, math
import torch.nn as nn

# class RoPECache(nn.Module):
#     def __init__(self, max_seqlen: int, head_dim: int,
#                  base: float = 10000.0, dtype=torch.bfloat16, device="cuda"):
#         super().__init__()
#         # build once in eager
#         t = torch.arange(max_seqlen, dtype=torch.float32, device=device)             # (S,)
#         inv = (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
#                                      device=device) / head_dim)).reciprocal()       # (H/2,)
#         freqs = torch.outer(t, inv)                                                  # (S, H/2)
#         cos = freqs.cos().to(dtype).unsqueeze(0).unsqueeze(0).contiguous()           # (1,1,S,H/2)
#         sin = freqs.sin().to(dtype).unsqueeze(0).unsqueeze(0).contiguous()
#         self.register_buffer("cos", cos, persistent=False)
#         self.register_buffer("sin", sin, persistent=False)
#
#     def slice(self, seqlen: int):
#         return self.cos[..., :seqlen, :], self.sin[..., :seqlen, :]

# components/rope.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPECache(nn.Module):
    """
    Precompute RoPE tables once; at step time we only slice views (no kernels).
    Shapes: cos/sin -> (1, 1, max_seqlen, head_dim//2)
    """
    def __init__(self, max_seqlen: int, head_dim: int,
                 base: float = 10000.0,
                 dtype: torch.dtype = torch.bfloat16,
                 device: torch.device | str = "cuda"):
        super().__init__()
        # Build once in eager; store as non-persistent buffers
        t = torch.arange(max_seqlen, dtype=torch.float32, device=device)                            # (S,)
        inv = (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)).reciprocal()  # (H/2,)
        freqs = torch.outer(t, inv)                                                                 # (S, H/2)
        cos = freqs.cos().to(dtype).unsqueeze(0).unsqueeze(0).contiguous()                          # (1,1,S,H/2)
        sin = freqs.sin().to(dtype).unsqueeze(0).unsqueeze(0).contiguous()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def slice(self, seqlen: int):
        # Views only; no kernels, fusible inside compiled graphs
        return self.cos[..., :seqlen, :], self.sin[..., :seqlen, :]

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (..., L, D_head) with D_head even (no need to know heads if they are folded).
    cos/sin: (1,1,L,D_head//2) broadcastable to x.
    """
    D = x.size(-1)
    assert D % 2 == 0, "RoPE expects even head dim"
    x_even = x[..., 0::2]     # (..., L, D/2)
    x_odd  = x[..., 1::2]
    # Broadcast cos/sin to x_even/x_odd
    x_rot_even = x_even * cos - x_odd  * sin
    x_rot_odd  = x_even * sin + x_odd  * cos
    # Interleave back
    out = torch.empty_like(x)
    out[..., 0::2] = x_rot_even
    out[..., 1::2] = x_rot_odd
    return out
