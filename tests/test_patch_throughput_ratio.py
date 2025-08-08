import torch

from components.hierarchical_autoencoder import HierarchicalAutoencoder


def build_model():
    comp_cfg = [
        {
            "dim": 4,
            "heads": 1,
            "window": 2,
            "num_encoder_layers": 1,
            "encoder_ffn_dim_multiplier": 2,
            "num_queries": 1,
            "codebook_size": 512,
            "beta": 0.25,
            "output_length": 8,
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
    model.eval()
    return model


def test_patches_per_token_matches_ratio():
    torch.manual_seed(0)
    model = build_model()
    tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    kpm = torch.zeros_like(tokens, dtype=torch.bool)

    out = model.forward(tokens, key_padding_mask=kpm)

    tokens_total = (~kpm).sum().item()
    comp = out["compression_results"]
    comp_step = comp["steps"][0]
    num_q = model.compressors[0].num_queries_per_segment
    # effective output length = valid segments * queries per segment
    output_len_total = (comp_step.valid_mask.sum().item() * num_q)

    ratio_measured = output_len_total / tokens_total
    ratio_reported = out["compression_ratios"][0]
    if isinstance(ratio_reported, torch.Tensor):
        ratio_reported = ratio_reported.item()

    assert abs(ratio_measured - ratio_reported) < 1e-5
