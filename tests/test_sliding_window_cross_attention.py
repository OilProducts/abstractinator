import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.sliding_window_attention import SlidingWindowCrossAttention


def test_cross_window_mask():
    attn = SlidingWindowCrossAttention(embed_dim=4, num_heads=1, window_size=1)
    mask = attn._cross_window_mask(4, 5, torch.device("cpu"))
    expected = torch.tensor(
        [[False, False, True, True, True],
         [False, False, False, True, True],
         [True, False, False, False, True],
         [True, True, False, False, False]],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_forward_output_shape():
    attn = SlidingWindowCrossAttention(embed_dim=8, num_heads=2, window_size=1)
    q = torch.randn(2, 3, 8)
    kv = torch.randn(2, 5, 8)
    out = attn(q, kv, kv)
    assert out.shape == (2, 3, 8)
