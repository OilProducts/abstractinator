import torch

from components.sliding_window_attention import (
    get_cross_window_mask,
    SegmentCausalCrossAttention,
)


def test_cross_window_mask():
    mask = get_cross_window_mask(3, 5, window=2, device=torch.device("cpu"))
    expected = torch.tensor(
        [
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, False, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_cross_attention_forward_shape():
    torch.manual_seed(0)
    attn = SegmentCausalCrossAttention(q_dim=4, kv_dim=4, d_attn=4, n_heads=2, lookback=1)
    q = torch.randn(2, 3, 4)
    kv = torch.randn(2, 5, 4)
    seg_id = torch.tensor([[0, 1, 2], [1, 2, 3]])
    out = attn(q, kv, seg_id)
    assert out.shape == (2, 3, 4)
