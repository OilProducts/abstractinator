import torch

from components.sliding_window_attention import (
    SegmentCausalCrossAttention,
)


def test_cross_attention_forward_shape():
    torch.manual_seed(0)
    attn = SegmentCausalCrossAttention(q_dim=4, kv_dim=4, d_attn=4, n_heads=2, lookback=1)
    q = torch.randn(2, 3, 4)
    kv = torch.randn(2, 5, 4)
    seg_id = torch.tensor([[0, 1, 2], [1, 2, 3]])
    out = attn(q, kv, seg_id)
    assert out.shape == (2, 3, 4)
