import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.vector_quantizer import VectorQuantizer

def test_vector_quantizer_forward_no_ema():
    torch.manual_seed(0)
    vq = VectorQuantizer(K=8, D=4, ema=False, eop_token_id=0)
    x = torch.randn(2, 3, 4)
    z_q, vq_loss, indices, perplexity = vq(x)
    assert z_q.shape == x.shape
    assert vq_loss.requires_grad
    assert indices.shape == (2, 3)
    assert perplexity.dim() == 0
    counts = torch.bincount(indices.view(-1), minlength=vq.K).float()
    counts[vq.eop_token_id] = 0
    counts = counts.clamp(min=vq.eps)
    probs = counts / (counts.sum() + vq.eps)
    entropy = -(probs.double() * probs.double().log()).sum()
    expected = entropy.exp().float()
    assert torch.allclose(perplexity, expected)
