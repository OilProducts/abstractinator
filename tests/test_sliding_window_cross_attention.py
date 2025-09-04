import torch

from components.attention.forms.regular.cross_segment_sdpa import (
    SegmentCausalCrossAttention,
)


def test_cross_attention_forward_shape():
    torch.manual_seed(0)
    attn = SegmentCausalCrossAttention(q_dim=4, kv_dim=4, d_attn=4, n_heads=2, lookback=1)
    q = torch.randn(2, 3, 4)
    kv = torch.randn(2, 5, 4)
    seg_id = torch.tensor([[0, 1, 2], [1, 2, 3]])
    # Provide minimal positional ids for queries and memory
    q_pos = torch.arange(q.size(1))  # (Lq,)
    kv_pos = torch.arange(kv.size(1))  # (Lkv,)
    out = attn(q, kv, seg_id, q_pos, kv_pos)
    assert out.shape == (2, 3, 4)
