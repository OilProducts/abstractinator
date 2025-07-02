from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import utils

from config import exp_config, DEVICE
from train import ByteLevelTokenizer
from components import HierarchicalAutoencoder


@register_model("hier_ae")
class HierarchicalAELM(LM):
    """LM-Eval adapter for HierarchicalAutoencoder checkpoints."""

    def __init__(self, checkpoint: str, device: Optional[str] = None) -> None:
        super().__init__()
        self.device = device or DEVICE
        self.tokenizer = ByteLevelTokenizer(add_bos=True, add_eos=True)
        self.model = HierarchicalAutoencoder(
            num_levels=exp_config["num_levels"],
            compressor_level_configs=exp_config["compressor_level_configs"],
            initial_vocab_size=exp_config["initial_vocab_size"],
            expander_dim_scale=exp_config["expander_dim_scale"],
            expander_num_enc_layers=exp_config["expander_num_enc_layers"],
            expander_num_dec_layers=exp_config["expander_num_dec_layers"],
            expander_heads_scale=exp_config["expander_heads_scale"],
            expander_eos_id=exp_config["expander_eos_id"],
            expander_max_len=exp_config["expander_max_len"],
            use_decoder_only_expander=exp_config.get("use_decoder_only_expander", False),
            propagate_key_padding_mask=exp_config["propagate_key_padding_mask"],
            aux_lm_loss_weight=exp_config["aux_lm_loss_weight"],
            top_transformer_config=exp_config.get("top_transformer_config"),
            top_lm_loss_weight=exp_config.get("top_lm_loss_weight", 0.0),
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
        return cls(checkpoint=checkpoint, device=device)

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_bos=True, add_eos=False).tolist()

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, cut_at_eos=True)

    @property
    def max_length(self) -> int:
        return exp_config.get("expander_max_len", 2048)

    def _tokens_logprobs(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(tokens)
            logits = out["final_reconstructed_logits"]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def loglikelihood(self, requests, disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
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

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
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
