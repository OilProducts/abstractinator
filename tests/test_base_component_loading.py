import torch

from components.checkpoint_utils import load_base_components, save_base_components
from components.hierarchical_autoencoder import HierarchicalAutoencoder


def build_base_model():
    comp_cfg = [
        {
            "dim": 4,
            "heads": 1,
            "window": 2,
            "num_encoder_layers": 1,
            "encoder_ffn_dim_multiplier": 2,
            "num_queries": 1,
            "codebook_size": 4,
            "beta": 0.25,
        }
    ]
    exp_cfg = [
        {
            "dim_scale": 1.0,
            "num_enc_layers": 1,
            "num_dec_layers": 1,
            "heads_scale": 1.0,
            "eos_id": 1,
            "max_len": 8,
            "use_decoder_only": True,
        }
    ]
    model = HierarchicalAutoencoder(
        num_levels=1,
        compressor_level_configs=comp_cfg,
        expander_level_configs=exp_cfg,
        initial_vocab_size=259,
        propagate_key_padding_mask=True,
        aux_lm_loss_weight=0.0,
        top_transformer_config=None,
        top_lm_loss_weight=0.0,
    )
    return model


def build_top_model():
    comp_cfg = [
        {
            "dim": 4,
            "heads": 1,
            "window": 2,
            "num_encoder_layers": 1,
            "encoder_ffn_dim_multiplier": 2,
            "num_queries": 1,
            "codebook_size": 4,
            "beta": 0.25,
        }
    ]
    exp_cfg = [
        {
            "dim_scale": 1.0,
            "num_enc_layers": 1,
            "num_dec_layers": 1,
            "heads_scale": 1.0,
            "eos_id": 1,
            "max_len": 8,
            "use_decoder_only": True,
        }
    ]
    model = HierarchicalAutoencoder(
        num_levels=1,
        compressor_level_configs=comp_cfg,
        expander_level_configs=exp_cfg,
        initial_vocab_size=259,
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
    load_base_components(top, str(save_path))

    for p in top.compressors.parameters():
        assert not p.requires_grad
    for p in top.expanders.parameters():
        assert not p.requires_grad
    trainable_flags = [p.requires_grad for p in top.code_sequence_transformer.parameters()]
    assert any(trainable_flags)

    opt_params = [p for p in top.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=1e-3)
    assert opt.param_groups[0]["params"] == opt_params


def test_load_base_components_from_full_state(tmp_path):
    torch.manual_seed(0)
    base = build_base_model()
    save_path = tmp_path / "base_full.pt"
    torch.save({"model_state": base.state_dict()}, save_path)

    top = build_top_model()
    load_base_components(top, str(save_path))

    for p in top.compressors.parameters():
        assert not p.requires_grad
    for p in top.expanders.parameters():
        assert not p.requires_grad
    trainable_flags = [p.requires_grad for p in top.code_sequence_transformer.parameters()]
    assert any(trainable_flags)

    opt_params = [p for p in top.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=1e-3)
    assert opt.param_groups[0]["params"] == opt_params


def build_two_level_model():
    comp_cfg = [
        {
            "dim": 4,
            "heads": 1,
            "window": 2,
            "num_encoder_layers": 1,
            "encoder_ffn_dim_multiplier": 2,
            "num_queries": 1,
            "codebook_size": 4,
            "beta": 0.25,
        },
        {
            "dim": 8,
            "heads": 1,
            "window": 2,
            "num_encoder_layers": 1,
            "encoder_ffn_dim_multiplier": 2,
            "num_queries": 1,
            "codebook_size": 8,
            "beta": 0.25,
        },
    ]
    exp_cfg = [
        {
            "dim_scale": 1.0,
            "num_enc_layers": 1,
            "num_dec_layers": 1,
            "heads_scale": 1.0,
            "eos_id": 1,
            "max_len": 8,
            "use_decoder_only": True,
        },
        {
            "dim_scale": 1.0,
            "num_enc_layers": 1,
            "num_dec_layers": 1,
            "heads_scale": 1.0,
            "eos_id": 1,
            "max_len": 8,
            "use_decoder_only": True,
        },
    ]
    model = HierarchicalAutoencoder(
        num_levels=2,
        compressor_level_configs=comp_cfg,
        expander_level_configs=exp_cfg,
        initial_vocab_size=259,
        propagate_key_padding_mask=True,
        aux_lm_loss_weight=0.0,
        top_transformer_config=None,
        top_lm_loss_weight=0.0,
    )
    return model


def test_load_base_components_into_larger_model(tmp_path):
    torch.manual_seed(0)
    base = build_base_model()
    save_path = tmp_path / "base.pt"
    save_base_components(base, str(save_path))

    model = build_two_level_model()
    load_base_components(model, str(save_path))

    # Bottom level (index 0 for compressors, index 1 for expanders) should match
    for k, v in base.compressors[0].state_dict().items():
        assert torch.allclose(model.compressors[0].state_dict()[k], v)
    for k, v in base.expanders[0].state_dict().items():
        assert torch.allclose(model.expanders[1].state_dict()[k], v)

    # Loaded levels are frozen
    for p in model.compressors[0].parameters():
        assert not p.requires_grad
    for p in model.expanders[1].parameters():
        assert not p.requires_grad
