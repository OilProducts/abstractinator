import argparse
import importlib.util
from dataclasses import asdict

import torch

from components.hierarchical_autoencoder import HierarchicalAutoencoder
from components.tokenizer import ByteLevelTokenizer


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("config_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DEVICE, mod.N_CPU, mod.exp_config


def build_model(exp_cfg, device):
    model = HierarchicalAutoencoder(
        num_levels=exp_cfg.num_levels,
        compressor_level_configs=[asdict(c) for c in exp_cfg.compressor_level_configs],
        initial_vocab_size=exp_cfg.initial_vocab_size,
        expander_level_configs=[asdict(e) for e in exp_cfg.expander_level_configs],
        propagate_key_padding_mask=exp_cfg.propagate_key_padding_mask,
        aux_lm_loss_weight=exp_cfg.aux_lm_loss_weight,
        top_transformer_config=(asdict(exp_cfg.top_transformer_config) if exp_cfg.top_transformer_config else None),
        top_lm_loss_weight=exp_cfg.top_lm_loss_weight,
        top_lm_mse_weight=exp_cfg.top_lm_mse_weight,
        top_lm_ce_weight=exp_cfg.top_lm_ce_weight,
        use_flex_attention=getattr(exp_cfg, "flex_attention", False),
    ).to(device)
    return model


def decode_from_level(model: HierarchicalAutoencoder, level: int, index: int, max_len: int = 128) -> torch.Tensor:
    device = next(model.parameters()).device
    codes = torch.tensor([[index]], dtype=torch.long, device=device)
    start = model.num_levels - 1 - level
    for i in range(start, model.num_levels):
        expander = model.expanders[i]
        codes = expander.generate(codes_hi=codes, src_key_padding_mask=None, max_len=max_len)
    return codes.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Expand VectorQuantizer entries")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--level", type=int, default=0, help="Which VQ level to inspect (0=bottom)")
    parser.add_argument("--max-len", type=int, default=128, help="Maximum expansion length")
    args = parser.parse_args()

    device, _, exp_cfg = load_config(args.config)
    tokenizer = ByteLevelTokenizer(add_bos=True, add_eos=True, expected_vocab_size=exp_cfg.initial_vocab_size)

    model = build_model(exp_cfg, device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = state.get("model_state", state)
    model.load_state_dict(sd, strict=False)
    model.eval()

    vq = model.compressors[args.level].vq
    for idx in range(vq.K):
        with torch.no_grad():
            tokens = decode_from_level(model, args.level, idx, max_len=args.max_len)
        text = tokenizer.decode(tokens.tolist(), cut_at_eos=True)
        print(f"Code {idx}: {text}")


if __name__ == "__main__":
    main()
