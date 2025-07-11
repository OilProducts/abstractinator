import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    model.eval()
    return model


def test_patches_per_token_matches_ratio():
    torch.manual_seed(0)
    model = build_model()
    tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    kpm = torch.zeros_like(tokens, dtype=torch.bool)

    out = model.forward(tokens, key_padding_mask=kpm)

    tokens_total = (~kpm).sum().item()
    output_len_total = out["output_seq_lengths_compressors"][0] * tokens.size(0)

    tokens_per_second = tokens_total  # assume 1 second elapsed
    patches_per_second = output_len_total

    ratio_measured = patches_per_second / tokens_per_second
    ratio_reported = out["compression_ratios"][0]
    if isinstance(ratio_reported, torch.Tensor):
        ratio_reported = ratio_reported.item()

    assert abs(ratio_measured - ratio_reported) < 1e-5
