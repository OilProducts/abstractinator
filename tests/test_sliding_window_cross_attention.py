import torch


from components.sliding_window_attention import SlidingWindowCrossAttention


def test_cross_window_mask():
    attn = SlidingWindowCrossAttention(embed_dim=4, num_heads=1, window_size=2)
    mask = attn._cross_window_mask(3, 5, device=torch.device("cpu"))
    expected = torch.tensor([
        [False, True, True, True, True],
        [False, False, True, True, True],
        [False, False, False, True, True],
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
