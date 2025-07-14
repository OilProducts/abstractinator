import torch


from components.hierarchical_autoencoder import HierarchicalAutoencoder


def build_model(target):
    comp_cfg = [{
        "dim": 4,
        "heads": 1,
        "window": 2,
        "num_encoder_layers": 1,
        "encoder_ffn_dim_multiplier": 2,
        "num_queries": 1,
        "codebook_size": 4,
        "beta": 0.25,
        "target_compression_ratio": target,
        "compression_loss_weight": 1.0,
    }]
    return HierarchicalAutoencoder(
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


def test_forward_reports_compression_loss():
    torch.manual_seed(0)
    model = build_model(0.5)
    tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    out = model.forward(tokens)
    assert "avg_compression_loss" in out
    assert "compression_loss_L0" in out["compression_loss_details"]


def test_different_targets_change_loss():
    torch.manual_seed(0)
    tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    model_a = build_model(0.0)
    model_b = build_model(1.0)
    out_a = model_a.forward(tokens)
    out_b = model_b.forward(tokens)
    assert not torch.allclose(out_a["total_loss"], out_b["total_loss"])
