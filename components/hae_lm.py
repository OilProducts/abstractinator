from __future__ import annotations
import importlib.util
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from components import HierarchicalAutoencoder
from components.tokenizer import ByteLevelTokenizer
from configs.base_config import ExpConfig


def _load_config(config: str | dict[str, Any] | ExpConfig):
    """Return (exp_config, device) from a config path, dict or ExpConfig."""
    if isinstance(config, ExpConfig):
        exp_config = config
        device = getattr(config, "DEVICE", None)
    elif isinstance(config, dict):
        device = config.get("DEVICE")
        cfg = {k: v for k, v in config.items() if k != "DEVICE"}
        exp_config = ExpConfig(**cfg)
    elif isinstance(config, str):
        spec = importlib.util.spec_from_file_location("config_module", config)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        exp_config = module.exp_config
        device = getattr(module, "DEVICE", None)
    else:
        raise TypeError("config must be a dict, ExpConfig instance or path to a python file")

    return exp_config, device


@register_model("hier_ae")
class HierarchicalAELM(LM):
    """LM-Eval adapter for HierarchicalAutoencoder checkpoints."""

    def __init__(self, checkpoint: str, config: str | dict[str, Any], device: str | None = None) -> None:
        super().__init__()
        exp_config, cfg_device = _load_config(config)
        self.exp_config = exp_config
        self.device = device or cfg_device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = ByteLevelTokenizer(add_bos=True, add_eos=True)
        self.model = HierarchicalAutoencoder(
            num_levels=exp_config.num_levels,
            compressor_level_configs=[asdict(c) for c in exp_config.compressor_level_configs],
            initial_vocab_size=exp_config.initial_vocab_size,
            expander_dim_scale=exp_config.expander.dim_scale,
            expander_num_enc_layers=exp_config.expander.num_enc_layers,
            expander_num_dec_layers=exp_config.expander.num_dec_layers,
            expander_heads_scale=exp_config.expander.heads_scale,
            expander_eos_id=exp_config.expander.eos_id,
            expander_max_len=exp_config.expander.max_len,
            use_decoder_only_expander=exp_config.expander.use_decoder_only,
            propagate_key_padding_mask=exp_config.expander.propagate_key_padding_mask,
            aux_lm_loss_weight=exp_config.aux_lm_loss_weight,
            top_transformer_config=(
                asdict(exp_config.top_transformer_config) if exp_config.top_transformer_config else None
            ),
            top_lm_loss_weight=exp_config.top_lm_loss_weight,
            use_continuous_expander_inputs=exp_config.expander.use_continuous_inputs,
            top_lm_mse_weight=exp_config.top_lm_mse_weight,
            top_lm_ce_weight=exp_config.top_lm_ce_weight,
        ).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @classmethod
    def create_from_arg_string(cls, arg_string: str, additional_config=None):
        args = utils.simple_parse_args_string(arg_string)
        checkpoint = args.get("checkpoint")
        device = args.get("device")
        if checkpoint is None:
            raise ValueError("checkpoint path must be provided")
        config = args.get("config", "config.py")
        return cls(checkpoint=checkpoint, config=config, device=device)

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_bos=True, add_eos=False).tolist()

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, cut_at_eos=True)

    @property
    def max_length(self) -> int:
        return getattr(self.exp_config.expander, "max_len", 2048)

    def _tokens_logprobs(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(tokens)
            logits = out["final_reconstructed_logits"]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def loglikelihood(self, requests, disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        results = []
        for context, continuation in tqdm(requests, disable=disable_tqdm):
            full_text = context + continuation
            full_tokens = torch.tensor(self.tok_encode(full_text)).unsqueeze(0).to(self.device)
            context_len = len(self.tok_encode(context))
            log_probs = self._tokens_logprobs(full_tokens)
            target_ids = full_tokens[:, 1:]
            per_token = log_probs[0, :-1].gather(1, target_ids.t().unsqueeze(1)).squeeze(1)
            cont_ll = per_token[context_len - 1 :].sum().item()
            results.append((cont_ll, True))
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        results = []
        for (string,) in tqdm(requests, disable=disable_tqdm):
            tokens = torch.tensor(self.tok_encode(string)).unsqueeze(0).to(self.device)
            log_probs = self._tokens_logprobs(tokens)
            target_ids = tokens[:, 1:]
            per_token = log_probs[0, :-1].gather(1, target_ids.t().unsqueeze(1)).squeeze(1)
            results.append(per_token.sum().item())
        return results

    def generate_until(self, requests, disable_tqdm: bool = False):
        results = []
        for context, gen_kwargs in tqdm(requests, disable=disable_tqdm):
            tokens = torch.tensor(self.tok_encode(context)).unsqueeze(0).to(self.device)
            max_len = gen_kwargs.get("max_gen_toks", 32)
            out_tokens = self.model.generate_bytes(tokens, max_len_override=max_len)
            results.append(self.tok_decode(out_tokens.squeeze(0).tolist()))
        return results
