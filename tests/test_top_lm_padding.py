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
            "pad_to_length": 6,
        },
        top_lm_loss_weight=1.0,
    )
    return model


def test_top_lm_pads_inputs():
    torch.manual_seed(0)
    model = build_model()
    tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    kpm = torch.zeros_like(tokens, dtype=torch.bool)

    comp = model.compress(tokens, key_padding_mask=kpm)
    emb = comp["all_pre_vq"][-1]
    mask = comp["final_key_padding_mask"]
    padded_emb, padded_mask = model._prepare_top_lm_inputs(emb, mask)

    assert padded_emb.size(1) == 6
    assert padded_mask is not None
    assert padded_mask.size(1) == 6
    assert padded_mask[0, -1]
