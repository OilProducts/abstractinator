

from components.hierarchical_autoencoder import HierarchicalAutoencoder


def test_vq_reset_interval_configured():
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
            "vq_reset_interval": 99,
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

    assert model.compressors[0].vq.reset_interval == 99
