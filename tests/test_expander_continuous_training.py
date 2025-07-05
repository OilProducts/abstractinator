import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
            "codebook_size": 4,
            "beta": 0.25,
        }
    ]
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
        use_continuous_expander_inputs=True,
    )
    return model


def test_continuous_inputs_to_expanders():
    torch.manual_seed(0)
    model = build_model()
    tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    kpm = torch.zeros_like(tokens, dtype=torch.bool)

    seen_inputs = []

    for exp in model.expanders:
        orig_forward = exp.forward

        def make_wrapper(fn):
            def wrapper(codes_hi, codes_lo, src_key_padding_mask=None, tgt_key_padding_mask=None):
                seen_inputs.append(codes_hi)
                return fn(codes_hi, codes_lo, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            return wrapper

        exp.forward = make_wrapper(orig_forward)

    model.forward(tokens, key_padding_mask=kpm)

    assert len(seen_inputs) == model.num_levels
    assert all(t.is_floating_point() for t in seen_inputs)
