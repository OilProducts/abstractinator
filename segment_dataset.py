import argparse
import importlib.util
import logging
from dataclasses import asdict

import torch
from datasets import load_dataset

from components.hierarchical_autoencoder import HierarchicalAutoencoder
from components.tokenizer import ByteLevelTokenizer
from components.utils import entropy_segments, token_entropy
from data_utils import tokenize_and_process_examples


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
        expander_dim_scale=exp_cfg.expander.dim_scale,
        expander_num_enc_layers=exp_cfg.expander.num_enc_layers,
        expander_num_dec_layers=exp_cfg.expander.num_dec_layers,
        expander_heads_scale=exp_cfg.expander.heads_scale,
        expander_eos_id=exp_cfg.expander.eos_id,
        expander_max_len=exp_cfg.expander.max_len,
        use_decoder_only_expander=exp_cfg.expander.use_decoder_only,
        propagate_key_padding_mask=exp_cfg.propagate_key_padding_mask,
        aux_lm_loss_weight=exp_cfg.aux_lm_loss_weight,
        top_transformer_config=(asdict(exp_cfg.top_transformer_config) if exp_cfg.top_transformer_config else None),
        top_lm_loss_weight=exp_cfg.top_lm_loss_weight,
        use_continuous_expander_inputs=exp_cfg.expander.use_continuous_inputs,
        top_lm_mse_weight=exp_cfg.top_lm_mse_weight,
        top_lm_ce_weight=exp_cfg.top_lm_ce_weight,
        use_flex_attention=exp_cfg.flex_attention,
    ).to(device)
    return model


def decode_segments(tokens, seg_ids, tokenizer):
    segments = []
    start = 0
    current = seg_ids[0]
    for i in range(1, len(seg_ids)):
        if seg_ids[i] != current:
            segment = tokens[start:i]
            segments.append((current, tokenizer.decode(segment, cut_at_eos=False)))
            start = i
            current = seg_ids[i]
    segments.append((current, tokenizer.decode(tokens[start:], cut_at_eos=False)))
    return segments


def main():
    parser = argparse.ArgumentParser(description="Inspect compressor segments")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to segment")
    parser.add_argument("--level", type=int, default=0, help="Compressor level to inspect")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device, n_cpu, exp_cfg = load_config(args.config)

    tokenizer = ByteLevelTokenizer(add_bos=True, add_eos=True, expected_vocab_size=exp_cfg.initial_vocab_size)

    raw_dataset = load_dataset(exp_cfg.dataset_name, name=exp_cfg.dataset_config, split=exp_cfg.dataset_train_split)
    raw_dataset = raw_dataset.select(range(args.num_samples))

    tokenized = raw_dataset.map(
        tokenize_and_process_examples,
        batched=True,
        fn_kwargs={
            "sequence_length": exp_cfg.sequence_length,
            "tokenizer": tokenizer,
            "text_column": exp_cfg.text_column_name,
        },
        remove_columns=raw_dataset.column_names,
        num_proc=1,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "key_padding_mask"])

    model = build_model(exp_cfg, device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = state.get("model_state", state)
    model.load_state_dict(sd, strict=False)
    model.eval()

    compressor = model.compressors[args.level]
    compressor.eval()

    for idx in range(args.num_samples):
        sample = tokenized[idx]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        kpm = sample["key_padding_mask"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = compressor(input_ids, key_padding_mask=kpm)
            logits = out["encoder_logits"]
            entropy = token_entropy(logits)
            seg = entropy_segments(
                entropy, increase_delta=compressor.entropy_delta, abs_threshold=compressor.entropy_abs_threshold
            )
        ids = input_ids.squeeze(0).cpu().tolist()
        seg_ids = seg.squeeze(0).cpu().tolist()

        if tokenizer.pad_id in ids:
            cut = ids.index(tokenizer.pad_id)
            ids = ids[:cut]
            seg_ids = seg_ids[:cut]
        segments = decode_segments(ids, seg_ids, tokenizer)

        print(f"Sample {idx}")
        for sid, text in segments:
            print(f":{text}", end="")
        print()


if __name__ == "__main__":
    main()
