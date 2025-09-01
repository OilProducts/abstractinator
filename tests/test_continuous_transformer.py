import torch

from components.code_sequence_transformer import CodeSequenceTransformer
from components.vector_quantizer import VectorQuantizer
from components.config_types import TopTransformerConfig


def test_continuous_transformer_forward():
    torch.manual_seed(0)
    vq = VectorQuantizer(
        K=8,
        D=4,
        ema=False,
        bos_token_id=0,
        eos_token_id=0,
        padding_token_id=0,
        eop_token_id=0,
    )
    cfg = TopTransformerConfig(
        embed_dim=4,
        dim=8,
        num_layers=1,
        num_heads=1,
    )
    model = CodeSequenceTransformer(cfg, vq=vq)
    x = torch.randn(2, 3, 4)
    out = model(x)
    assert out["predictions_pre_vq"].shape == x.shape
    assert out["predictions"].shape == x.shape
    assert out["indices"].shape == (2, 3)
    assert out["vq_loss"].requires_grad
