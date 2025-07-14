import torch


from components.hierarchical_autoencoder import HierarchicalAutoencoder


def test_vq_reset_interval_configured():
    comp_cfg = [{
        "dim": 4,
        "heads": 1,
        "window": 2,
        "num_encoder_layers": 1,
        "encoder_ffn_dim_multiplier": 2,
        "num_queries": 1,
        "codebook_size": 4,
        "beta": 0.25,
        "vq_reset_interval": 99,
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

    assert model.compressors[0].vq.reset_interval == 99
