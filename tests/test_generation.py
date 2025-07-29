import torch


from components.hierarchical_autoencoder import HierarchicalAutoencoder


def build_model():
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
    exp_cfg = [{
        "dim_scale": 1.0,
        "num_enc_layers": 1,
        "num_dec_layers": 1,
        "heads_scale": 1.0,
        "eos_id": 1,
        "max_len": 8,
        "use_decoder_only": True,
    }]
    model = HierarchicalAutoencoder(
        num_levels=1,
        compressor_level_configs=comp_cfg,
        expander_level_configs=exp_cfg,
        initial_vocab_size=259,
        propagate_key_padding_mask=True,
        aux_lm_loss_weight=0.0,
        top_transformer_config={
            "dim": 4,
            "num_layers": 1,
            "num_heads": 1,
            "ffn_dim_multiplier": 2,
            "output_lm_logits": True,
        },
        top_lm_loss_weight=0.0,
    )
    model.eval()
    return model


def test_generate_bytes_longer_than_compressed():
    torch.manual_seed(0)
    model = build_model()
    tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    kpm = torch.zeros_like(tokens, dtype=torch.bool)

    comp_res = model.compress(tokens, key_padding_mask=kpm)
    compressed_len = comp_res["top_codes"].size(1)

    out = model.generate_bytes(tokens, key_padding_mask=kpm, max_len_override=5)
    assert out.size(1) > compressed_len
