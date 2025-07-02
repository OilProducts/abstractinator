import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.utils import safe_softmax, token_entropy, entropy_segments


def test_safe_softmax_partial_and_all_masked():
    scores = torch.tensor([[1.0, 2.0, 3.0],
                           [0.5, -1.0, 0.0]])
    mask = torch.tensor([[False, True, False],
                        [True, True, True]])
    out = safe_softmax(scores, mask, dim=1)

    expected_row0 = torch.softmax(scores[0].masked_fill(mask[0], float('-inf')), dim=0)
    expected_row0 = expected_row0.masked_fill(mask[0], 0.0)
    expected_row1 = torch.zeros(3)
    expected = torch.stack([expected_row0, expected_row1])
    assert torch.allclose(out, expected)
    assert torch.all(out[1] == 0)


def test_safe_softmax_no_nan_for_all_masked():
    scores = torch.tensor([[0.2, -0.1]])
    mask = torch.tensor([[True, True]])
    out = safe_softmax(scores, mask, dim=1)
    assert torch.all(out == 0)


def test_token_entropy_uniform_distribution():
    logits = torch.zeros(2, 3, 5)
    ent = token_entropy(logits)
    expected = torch.full((2, 3), fill_value=torch.log(torch.tensor(5.0)))
    assert torch.allclose(ent, expected)


def test_token_entropy_peaked_distribution():
    logits = torch.tensor([[[10.0, -10.0]]])
    ent = token_entropy(logits)
    assert ent.shape == (1, 1)
    assert ent.item() < 1e-3


def test_entropy_segments_basic_increase():
    ent = torch.tensor([[1.0, 1.3, 1.7, 1.65, 2.0]])
    seg = entropy_segments(ent)
    expected = torch.tensor([[0, 1, 2, 2, 3]])
    assert torch.equal(seg, expected)


def test_entropy_segments_with_threshold():
    ent = torch.tensor([[0.5, 1.1, 1.2, 0.9, 1.3]])
    seg = entropy_segments(ent, abs_threshold=1.0)
    expected = torch.tensor([[0, 1, 2, 2, 3]])
    assert torch.equal(seg, expected)


def test_entropy_segments_custom_delta():
    ent = torch.tensor([[1.0, 1.1, 1.4, 1.45]])
    seg = entropy_segments(ent, increase_delta=0.3)
    expected = torch.tensor([[0, 0, 0, 0]])
    assert torch.equal(seg, expected)


def test_entropy_segments_edge_cases():
    # No increase -> single segment
    ent = torch.tensor([[1.0, 0.9, 0.8]])
    seg = entropy_segments(ent)
    assert torch.equal(seg, torch.zeros_like(ent, dtype=torch.long))

    # Single token
    ent1 = torch.tensor([[0.5]])
    seg1 = entropy_segments(ent1)
    assert torch.equal(seg1, torch.zeros_like(ent1, dtype=torch.long))

    # Empty sequence
    ent2 = torch.empty(1, 0)
    seg2 = entropy_segments(ent2)
    assert seg2.shape == ent2.shape

    # Wrong dims
    with pytest.raises(ValueError):
        entropy_segments(torch.tensor([1.0, 2.0, 3.0]))
