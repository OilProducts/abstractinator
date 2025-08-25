import torch
from torch import nn
from typing import Tuple

class RoPECache(nn.Module):
    """
    Precompute and cache rotary position embeddings (RoPE) tables.

    Notes
    -----
    - Precomputes cos/sin once in eager mode as non-persistent buffers.
    - At step/forward time we only take views (slices): no extra kernels.
    - Shapes: cos/sin -> (1, 1, S_max, D_head//2)
    """
    def __init__(
        self,
        max_seqlen: int,
        head_dim: int,
        base: float = 10000.0,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ):
        super().__init__()

        # Frequencies per even index in the head dimension (standard RoPE formulation).
        # inv_freq: (D_head//2,)
        idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = (base ** (idx / head_dim)).reciprocal()

        # Outer product gives per-(position, dim) phase angles.
        # freqs: (S_max, D_head//2)
        t = torch.arange(max_seqlen, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Precompute cos/sin tables once; keep in low precision to save memory traffic.
        cos = freqs.cos().to(dtype).unsqueeze(0).unsqueeze(0).contiguous()  # (1,1,S_max,D/2)
        sin = freqs.sin().to(dtype).unsqueeze(0).unsqueeze(0).contiguous()

        # Non-persistent: excluded from state_dict; rebuilt on init.
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def slice(self, seqlen: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return views for the first `seqlen` positions.
        Shapes: (1, 1, seqlen, D_head//2)
        """
        return self.cos[..., :seqlen, :], self.sin[..., :seqlen, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to the last dimension of `x`.

    Parameters
    ----------
    x   : (..., L, D)       with D even.
    cos : (1, 1, L, D//2)   broadcastable to x[..., :, 0::2]
    sin : (1, 1, L, D//2)   broadcastable to x[..., :, 0::2]

    Returns
    -------
    out : same shape as x
    """
    D = x.size(-1)
    assert (D & 1) == 0, "RoPE expects even head dim"

    # Split even/odd components along the last dimension.
    x_even = x[..., 0::2]  # (..., L, D/2)
    x_odd  = x[..., 1::2]  # (..., L, D/2)

    # Rotate (complex multiply): (a + i b) * (cos + i sin)
    x_rot_even = x_even * cos - x_odd  * sin
    x_rot_odd  = x_even * sin + x_odd  * cos

    # Interleave back to (..., L, D) without extra kernels.
    out = torch.empty_like(x)
    out[..., 0::2] = x_rot_even
    out[..., 1::2] = x_rot_odd
    return out