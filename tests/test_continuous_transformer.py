import torch

from components.code_sequence_transformer import CodeSequenceTransformer
from components.vector_quantizer import VectorQuantizer
from components.expander import CodeExpander


def test_continuous_transformer_forward():
    torch.manual_seed(0)
    vq = VectorQuantizer(K=8, D=4, ema=False, eop_token_id=0)
    model = CodeSequenceTransformer(embed_dim=4, dim=8, num_layers=1, num_heads=1, vq=vq)
    x = torch.randn(2, 3, 4)
    out = model(x)
    assert out["predictions_pre_vq"].shape == x.shape
    assert out["predictions"].shape == x.shape
    assert out["indices"].shape == (2, 3)
    assert out["vq_loss"].requires_grad


def test_expander_with_continuous_memory():
    torch.manual_seed(0)
    exp = CodeExpander(K_hi=8, K_lo=4, D=4, N_enc=1, N_dec=1, H=1)
    memory = torch.randn(2, 5, 4)
    codes_lo = torch.randint(0, 4, (2, 6))
    out = exp(memory, codes_lo)
    assert out["logits"].shape == (2, 6, 4)

