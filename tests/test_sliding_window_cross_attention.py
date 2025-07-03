import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.sliding_window_attention import SlidingWindowCrossAttention


def test_cross_window_mask():
    attn = SlidingWindowCrossAttention(embed_dim=4, num_heads=1, window_size=2)
    mask = attn._cross_window_mask(3, 5, device=torch.device("cpu"))
    expected = torch.tensor([
        [True, True, False],
        [True, False, False],
        [False, False, False],
    ])
    assert torch.equal(mask, expected)


def test_cross_attention_forward_shape():
    torch.manual_seed(0)
    attn = SlidingWindowCrossAttention(embed_dim=4, num_heads=2, window_size=1)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 4, 4)
    v = torch.randn(2, 4, 4)
    out = attn(q, k, v)
    assert out.shape == (2, 3, 4)


def test_cross_attention_runtime_scaling():
    torch.manual_seed(0)
    attn = SlidingWindowCrossAttention(embed_dim=16, num_heads=4, window_size=2)
    q = torch.randn(1, 64, 16)
    k_small = torch.randn(1, 64, 16)
    v_small = torch.randn(1, 64, 16)
    k_big = torch.randn(1, 1024, 16)
    v_big = torch.randn(1, 1024, 16)

    t0 = time.time()
    attn(q, k_small, v_small)
    small_time = time.time() - t0

    t0 = time.time()
    attn(q, k_big, v_big)
    big_time = time.time() - t0

    assert big_time < small_time * 20
