import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.hierarchical_autoencoder import HierarchicalAutoencoder
from train import save_base_components


def build_base_model():
    comp_cfg = [{
        "dim": 4,
        "heads": 1,
        "window": 2,
        "num_encoder_layers": 1,
        "encoder_ffn_dim_multiplier": 2,
        "num_queries": 1,
        "codebook_size": 4,
        "beta": 0.25,
    }]
    model = HierarchicalAutoencoder(
        num_levels=1,
        compressor_level_configs=comp_cfg,
        initial_vocab_size=259,
        expander_dim_scale=1.0,
        expander_num_enc_layers=1,
        expander_num_dec_layers=1,
        expander_heads_scale=1.0,
        expander_eos_id=1,
        expander_max_len=8,
        propagate_key_padding_mask=True,
        aux_lm_loss_weight=0.0,
        top_transformer_config=None,
        top_lm_loss_weight=0.0,
    )
    return model


def build_top_model():
    comp_cfg = [{
        "dim": 4,
        "heads": 1,
        "window": 2,
        "num_encoder_layers": 1,
        "encoder_ffn_dim_multiplier": 2,
        "num_queries": 1,
        "codebook_size": 4,
        "beta": 0.25,
    }]
    model = HierarchicalAutoencoder(
        num_levels=1,
        compressor_level_configs=comp_cfg,
        initial_vocab_size=259,
        expander_dim_scale=1.0,
        expander_num_enc_layers=1,
        expander_num_dec_layers=1,
        expander_heads_scale=1.0,
        expander_eos_id=1,
        expander_max_len=8,
        propagate_key_padding_mask=True,
        aux_lm_loss_weight=0.0,
        top_transformer_config={"dim": 4, "num_layers": 1, "num_heads": 1, "ffn_dim_multiplier": 2},
        top_lm_loss_weight=1.0,
    )
    return model


def test_load_and_freeze_base_components(tmp_path):
    torch.manual_seed(0)
    base = build_base_model()
    save_path = tmp_path / "base.pt"
    save_base_components(base, str(save_path))

    top = build_top_model()
    ckpt = torch.load(save_path, map_location="cpu")
    top.compressors.load_state_dict(ckpt["compressors"], strict=False)
    top.expanders.load_state_dict(ckpt["expanders"], strict=False)
    top.compressors.requires_grad_(False)
    top.expanders.requires_grad_(False)
    top.compressors.eval()
    top.expanders.eval()

    for p in top.compressors.parameters():
        assert not p.requires_grad
    for p in top.expanders.parameters():
        assert not p.requires_grad
    trainable_flags = [p.requires_grad for p in top.code_sequence_transformer.parameters()]
    assert any(trainable_flags)

    opt_params = [p for p in top.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=1e-3)
    assert opt.param_groups[0]["params"] == opt_params
