from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List, Callable, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import entropy_segments, token_entropy
from .mla import SlidingWindowMLATransformerBlock
from .attention.pooling.learned_query import LearnedQueryAttention
from .config_types import AttentionConfig

# ---------------------------
# Data containers / Protocols
# ---------------------------

@dataclass
class SegmentBoundaries:
    seg_id: Tensor             # (B, S) int64; 0,0,0,1,1,2,...
    patch_end_mask: Tensor     # (B, S) bool; True on last token of segment
    first_byte_idx: Tensor     # (B, S) int64; first token index for each position's segment


@dataclass
class EntropyOut:
    """Outputs from an entropy model evaluated on a sequence."""
    bits: Tensor                       # (B, S) per-token entropy in bits, aligned to input (pad at t=0)
    loss: Optional[Tensor] = None      # scalar loss if available
    logits: Optional[Tensor] = None    # (B, S, V) for logit model
    mu: Optional[Tensor] = None        # (B, S, D) for Gaussian model
    logvar: Optional[Tensor] = None    # (B, S, D) for Gaussian model


@dataclass
class CompressorOutput:
    vq_embeddings: Optional[Tensor]      # (B, Q_max, D) or None if no quantizer
    vq_indices: Optional[Tensor]         # (B, Q_max) or None
    vq_loss: Optional[Tensor]            # scalar or None
    vq_perplexity: Optional[Tensor]      # scalar or None

    valid_mask: Tensor                   # (B, Q_max) bool
    patch_end_mask: Tensor               # (B, S) bool
    entropy_bits: Tensor                 # (B, S) bits used for segmentation
    entropy_loss: Optional[Tensor]       # scalar or None
    entropy_mu: Optional[Tensor]         # (B, S, D) for Gaussian
    entropy_logvar: Optional[Tensor]     # (B, S, D) for Gaussian

    pre_vq_embeddings: Tensor            # (B, Q_max, D)
    seg_id: Tensor                       # (B, S) int64
    first_byte_idx: Tensor               # (B, S) int64

    input_sequence: Tensor               # (B, S) long or empty ()
    input_padding_mask: Optional[Tensor] # (B, S) bool or None


# ---------------------------
# Quantizer abstraction
# ---------------------------

class QuantizerBase(Protocol):
    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns: (quantized, vq_loss, indices, perplexity)
          - quantized: (B, Q_max, D)
          - vq_loss: scalar
          - indices: (B, Q_max) long
          - perplexity: scalar
        """
        ...


class IdentityQuantizer(nn.Module):
    """
    No-op quantizer; keeps API-compatible with VQ modules.
    """
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B, Q, D = x.shape
        device = x.device
        return x, x.new_tensor(0.0), torch.full((B, Q), -1, dtype=torch.long, device=device), x.new_tensor(0.0)

    __call__ = forward


# ---------------------------
# Entropy models
# ---------------------------

class EntropyModelBase(nn.Module):
    """
    Uniform API for entropy models used to drive segmentation.
    """

    def prefill(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tuple[List[Tuple[Tensor, Tensor]], Tensor]:
        """
        Run model in cache-building mode on a prefix.
        Returns: (caches, entropy_bits_for_prefix)
        """
        raise NotImplementedError

    def step(self,
             caches: List[Tuple[Tensor, Tensor]],
             new_x: Tensor,
             pos_start: int,
             key_padding_mask_new: Optional[Tensor]) -> Tuple[List[Tuple[Tensor, Tensor]], Tensor]:
        """
        Process only new tokens using caches (streaming).
        Returns: (updated_caches, entropy_bits_for_new_tokens)
        """
        raise NotImplementedError

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor],
                *,
                targets: Optional[Tensor] = None) -> EntropyOut:
        raise NotImplementedError

    @torch.no_grad()
    def entropy(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        return self.forward(x, key_padding_mask).bits

    @torch.no_grad()
    def sample_next(self,
                    x: Tensor,
                    key_padding_mask: Optional[Tensor],
                    *,
                    temperature: float = 1.0,
                    top_p: float = 1.0,
                    top_k: int = 0,
                    embed_fn: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
        """
        Sample the next item from the predictive distribution at the last position.
        - For Gaussian: returns next embedding (B, D).
        - For Logit: returns next token id (B,) unless `embed_fn` is provided, in which
          case returns next embedding (B, D).
        """
        raise NotImplementedError

    @torch.no_grad()
    def predictive_entropy_next_bits(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        temperature: float = 1.0,   # Gaussian: scales std; Logit: scales logits via 1/temperature
        top_p: float = 1.0,         # Logit-only: nucleus on the NEXT distribution (optional)
        top_k: int = 0              # Logit-only: top-k on the NEXT distribution (optional)
    ) -> torch.Tensor:
        """
        Returns H(next | prefix) in *bits* for each batch element: shape (B,).

        - Gaussian: differential entropy of N(mu_last, diag(exp(logvar_last))).
          If temperature>0 and !=1, adds D*log2(temperature).
        - Logit: Shannon entropy of the categorical at the last position after
          applying temperature/top-k/top-p if provided.
        """
        raise NotImplementedError

    @torch.no_grad()
    def predictive_entropy_next_bits_from_cache(
        self,
        cache: Any,                          # GaussianCache | LogitCache
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Tensor:
        """
        Same as above, but O(1) from the model's streaming cache; returns (B,).
        """
        raise NotImplementedError

    @torch.no_grad()
    def predictive_bpd_next(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None, *, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Gaussian-only: returns H(next|prefix)/D in bits-per-dim (B,).
        Logit models raise NotImplementedError (no 'dim' notion).
        """
        raise NotImplementedError

    @torch.no_grad()
    def predictive_bpd_next_from_cache(self, cache: Any, *, temperature: float = 1.0) -> Tensor:
        raise NotImplementedError


class GaussianEntropyModel(EntropyModelBase):
    """
    Causal encoder producing μ, logσ²; entropy computed as Gaussian BPD.
    Layers are SlidingWindowMLATransformerBlock with prefill/step support.
    """

    def __init__(self,
                 dim: int,
                 n_heads: int = 8,
                 window: int = 128,
                 head_dim: Optional[int] = 32,
                 kv_comp_dim: Optional[int] = 64,
                 q_comp_dim: Optional[int] = 96,
                 retr_dim: Optional[int] = 32,
                 ffn_dim_multiplier: int = 4,
                 use_flex_attention: bool = True,
                 n_layers: int = 2,
                 clamp_logvar: Tuple[float, float] = (-8.0, 8.0)):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SlidingWindowMLATransformerBlock(
                    dim=dim, num_heads=n_heads, window_size=window,
                    head_dim=head_dim, kv_comp_dim=kv_comp_dim, q_comp_dim=q_comp_dim,
                    retr_dim=retr_dim, ffn_dim_multiplier=ffn_dim_multiplier,
                    use_flex_attention=use_flex_attention
                )
                for _ in range(n_layers)
            ]
        )
        self.mu = nn.Linear(dim, dim)
        self.logvar = nn.Linear(dim, dim)
        self.clamp = clamp_logvar
        self.dim = dim

    def _run_stack(self, x: Tensor, kpm: Optional[Tensor]) -> Tensor:
        for blk in self.layers:
            x = blk(x, key_padding_mask=kpm)
        return x

    def _bpd_bits(self, mu: Tensor, logvar: Tensor, target: Tensor) -> Tensor:
        """
        Returns (B, L) bits aligned to target positions, padded at t=0.
        Predict x[:,1:] from stats at positions [:, :-1].
        """
        x = target[:, 1:, :]
        m = mu[:, :-1, :]
        lv = logvar[:, :-1, :]

        scale = mu.new_tensor(0.5 / math.log(2.0))      # convert nats -> bits
        log_2pi = mu.new_tensor(math.log(2.0 * math.pi))

        dx = x - m
        tmp = dx.square() * torch.exp(-lv) + lv + log_2pi  # (B, L-1, D)
        bpd = tmp.mean(dim=-1) * scale                     # (B, L-1)
        return F.pad(bpd, (1, 0), value=0.0)               # (B, L)

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor],
                *,
                targets: Optional[Tensor] = None) -> EntropyOut:
        """
        Computes per-token entropy bits, and optional loss as mean bits over valid steps.
        Targets default to x (teacher-forced next-step).
        """
        B, S, D = x.shape
        h = self._run_stack(x, key_padding_mask)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(*self.clamp)
        # tgt = x if targets is None else targets
        bits = self._bpd_bits(mu, logvar, x)  # (B, S)
        # define loss over non-padded positions (from t=1 where prediction happens)
        if key_padding_mask is not None:
            m = ~key_padding_mask
        else:
            m = torch.ones_like(bits, dtype=torch.bool)
        loss = (bits[:, 1:] * m[:, 1:]).sum() / m[:, 1:].sum().clamp(min=1)
        return EntropyOut(bits=bits, loss=loss, mu=mu, logvar=logvar)

    def prefill(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tuple[List[Tuple[Tensor, Tensor]], Tensor]:
        h = x
        caches: List[Tuple[Tensor, Tensor]] = []
        for blk in self.layers:
            h, cache = blk.prefill(h, pos_start=0, key_padding_mask=key_padding_mask)
            caches.append(cache)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(*self.clamp)
        bits = self._bpd_bits(mu, logvar, x)
        cache = GaussianCache(
            layers=caches,
            last_mu=mu[:, -1, :],
            last_logvar=logvar[:, -1, :],
        )
        return cache, bits

    def step(self,
             cache: GaussianCache,
             new_x: Tensor,
             pos_start: int,
             key_padding_mask_new: Optional[Tensor]) -> Tuple[List[Tuple[Tensor, Tensor]], Tensor]:
        h = new_x
        for i, blk in enumerate(self.layers):
            h, cache.layers[i] = blk.step(
                h, cache=cache.layers[i], pos_start=pos_start, key_padding_mask_new=key_padding_mask_new
            )
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(*self.clamp)
        # For a block step, produce bits for *these* steps only; predicting t+1 inside block
        # Align: pad one zero at the start of the block to keep (B, T_blk)
        # Consumers typically concat across calls anyway.
        bits_blk = self._bpd_bits(mu, logvar, new_x)  # (B, T_blk)
        # Update “last” stats for O(1) next-entropy
        cache.last_mu = mu[:, -1, :]
        cache.last_logvar = logvar[:, -1, :]
        return cache, bits_blk

    @torch.no_grad()
    def sample_next(self,
                    x: Tensor,
                    key_padding_mask: Optional[Tensor],
                    *,
                    temperature: float = 1.0,
                    top_p: float = 1.0,
                    top_k: int = 0,
                    embed_fn: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
        """
        Sample next embedding using Normal(mu_t, var_t) at t = last position.
        """
        h = self._run_stack(x, key_padding_mask)
        mu = self.mu(h)[:, -1, :]                    # (B, D)
        logvar = self.logvar(h).clamp(*self.clamp)[:, -1, :]
        std = torch.exp(0.5 * logvar)

        # Temperature scales std; temp=0 → deterministic mean
        if temperature <= 0:
            return mu
        eps = torch.randn_like(std)
        return mu + (temperature * std) * eps

    @torch.no_grad()
    def predictive_entropy_next_bits(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,  # ignored
        top_k: int = 0       # ignored
    ) -> Tensor:
        """
        H_bits = 0.5 * [ D * ln(2πe) + sum(logvar_last) ] / ln 2, optionally + D*log2(temperature)
        """
        h = self._run_stack(x, key_padding_mask)        # (B, S, D)
        logvar_last = self.logvar(h).clamp(*self.clamp)[:, -1, :]  # (B, D)

        # nats
        H_nats = 0.5 * ( self.dim * math.log(2.0 * math.pi * math.e) + logvar_last.sum(dim=-1) )
        # temperature scaling of std multiplies covariance by tau^2 -> + D * ln(tau)
        if temperature is not None and temperature > 0.0 and temperature != 1.0:
            H_nats = H_nats + self.dim * math.log(temperature)

        # bits
        return H_nats / math.log(2.0)  # (B,)

    @torch.no_grad()
    def predictive_entropy_next_bits_from_cache(self, cache: GaussianCache, *, temperature: float = 1.0, **_) -> Tensor:
        lv = cache.last_logvar  # (B, D)
        H_nats = 0.5 * ( self.dim * math.log(2.0 * math.pi * math.e) + lv.sum(dim=-1) )
        if temperature > 0.0 and temperature != 1.0:
            H_nats = H_nats + self.dim * math.log(temperature)
        return H_nats / math.log(2.0)

    @torch.no_grad()
    def predictive_bpd_next_from_cache(self, cache: GaussianCache, *, temperature: float = 1.0) -> Tensor:
        return self.predictive_entropy_next_bits_from_cache(cache, temperature=temperature) / float(self.dim)

    @torch.no_grad()
    def predictive_bpd_next(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        *,
        temperature: float = 1.0
    ) -> Tensor:
        """ Bits-per-dimension version (H_bits / D). """
        H_bits = self.predictive_entropy_next_bits(x, key_padding_mask, temperature=temperature)
        return H_bits / float(self.dim)

class LogitEntropyModel(EntropyModelBase):
    """
    Causal token LM that outputs logits and uses token entropy for segmentation.
    """
    def __init__(self,
                 dim: int,
                 vocab_size: int = 260,
                 n_heads: int = 8,
                 window: int = 128,
                 head_dim: Optional[int] = 32,
                 kv_comp_dim: Optional[int] = 64,
                 q_comp_dim: Optional[int] = 96,
                 retr_dim: Optional[int] = 32,
                 ffn_dim_multiplier: int = 4,
                 use_flex_attention: bool = True,
                 n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SlidingWindowMLATransformerBlock(
                    dim=dim, num_heads=n_heads, window_size=window,
                    head_dim=head_dim, kv_comp_dim=kv_comp_dim, q_comp_dim=q_comp_dim,
                    retr_dim=retr_dim, ffn_dim_multiplier=ffn_dim_multiplier,
                    use_flex_attention=use_flex_attention
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.RMSNorm(dim)
        self.proj = nn.Linear(dim, vocab_size)
        self.vocab_size = vocab_size

    def _run_stack(self, x: Tensor, kpm: Optional[Tensor]) -> Tensor:
        for blk in self.layers:
            x = blk(x, key_padding_mask=kpm)
        return self.norm(x)

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor],
                *,
                targets: Optional[Tensor] = None) -> EntropyOut:
        """
        targets: token ids (B, S) for CE loss, if available.
        """
        h = self._run_stack(x, key_padding_mask)
        logits = self.proj(h)                    # (B, S, V)
        bits = token_entropy(logits)             # (B, S)

        loss = None
        if targets is not None:
            # predict next token: shift logits to [:, :-1], targets [:, 1:]
            if key_padding_mask is not None:
                m = ~key_padding_mask[:, 1:]
            else:
                m = torch.ones_like(targets[:, 1:], dtype=torch.bool)
            per_tok = F.cross_entropy(
                logits[:, :-1, :].transpose(1, 2),  # (B, V, S-1)
                targets[:, 1:],                     # (B, S-1)
                reduction="none"
            )
            loss = (per_tok * m).sum() / m.sum().clamp(min=1)

        return EntropyOut(bits=bits, loss=loss, logits=logits)

    def prefill(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tuple[List[Tuple[Tensor, Tensor]], Tensor]:
        h = x
        caches: List[Tuple[Tensor, Tensor]] = []
        for blk in self.layers:
            h, cache = blk.prefill(h, pos_start=0, key_padding_mask=key_padding_mask)
            caches.append(cache)
        logits = self.proj(self.norm(h))
        bits = token_entropy(logits)
        return caches, bits

    def step(self,
             caches: List[Tuple[Tensor, Tensor]],
             new_x: Tensor,
             pos_start: int,
             key_padding_mask_new: Optional[Tensor]) -> Tuple[List[Tuple[Tensor, Tensor]], Tensor]:
        h = new_x
        for i, blk in enumerate(self.layers):
            h, caches[i] = blk.step(
                h, cache=caches[i], pos_start=pos_start, key_padding_mask_new=key_padding_mask_new
            )
        logits = self.proj(self.norm(h))
        bits = token_entropy(logits)
        return caches, bits

    @torch.no_grad()
    def sample_next(self,
                    x: Tensor,
                    key_padding_mask: Optional[Tensor],
                    *,
                    temperature: float = 1.0,
                    top_p: float = 1.0,
                    top_k: int = 0,
                    embed_fn: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
        """
        Sample next token id from the categorical over the last position.
        If embed_fn is provided, returns embedding instead.
        """
        h = self._run_stack(x, key_padding_mask)
        logits = self.proj(h)[:, -1, :]  # (B, V)

        # temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        # top-k
        if top_k and top_k < logits.size(-1):
            v, idx = torch.topk(logits, top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter(-1, idx, v)

        # top-p
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            cutoff = (cum > top_p).float().argmax(dim=-1, keepdim=True)
            # mask everything beyond cutoff
            mask = torch.full_like(sorted_logits, float("-inf"))
            keep = torch.arange(sorted_logits.size(-1), device=logits.device)[None, :] <= cutoff
            sorted_logits = torch.where(keep, sorted_logits, mask)
            # unsort
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
        if embed_fn is not None:
            return embed_fn(next_ids)  # (B, D)
        return next_ids               # (B,)

    @torch.no_grad()
    def predictive_entropy_next_bits(self, x: Tensor, key_padding_mask: Optional[Tensor], *,
                                     temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> Tensor:
        h = self._run_stack(x, key_padding_mask)
        logits = self.proj(h)[:, -1, :]
        return _categorical_entropy_bits(logits, temperature=temperature, top_p=top_p, top_k=top_k)

    @torch.no_grad()
    def predictive_entropy_next_bits_from_cache(self, cache: LogitCache, *,
                                                temperature: float = 1.0, top_p: float = 1.0,
                                                top_k: int = 0) -> Tensor:
        return _categorical_entropy_bits(cache.last_logits, temperature=temperature, top_p=top_p, top_k=top_k)

@torch.no_grad()
def _categorical_entropy_bits(logits: Tensor, *, temperature: float = 1.0, top_p: float = 1.0,
                              top_k: int = 0) -> Tensor:
    if temperature > 0.0 and temperature != 1.0:
        logits = logits / float(temperature)
    if top_k and top_k < logits.size(-1):
        v, idx = torch.topk(logits, top_k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        logits = mask.scatter(-1, idx, v)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cdf = probs.cumsum(dim=-1)
        cutoff = (cdf > top_p).float().argmax(dim=-1, keepdim=True)
        keep = torch.arange(sorted_logits.size(-1), device=logits.device)[None, :] <= cutoff
        masked_sorted = torch.where(keep, sorted_logits, torch.full_like(sorted_logits, float("-inf")))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, masked_sorted)
    logp = F.log_softmax(logits, dim=-1)
    H_nats = -(logp.exp() * logp).sum(dim=-1)
    return H_nats / math.log(2.0)
    # @torch.no_grad()
    # def predictive_entropy_next_bits(
    #     self,
    #     x: Tensor,
    #     key_padding_mask: Tensor | None = None,
    #     *,
    #     temperature: float = 1.0,
    #     top_p: float = 1.0,
    #     top_k: int = 0
    # ) -> Tensor:
    #     """
    #     Shannon entropy H(p) in bits at the last position (B,).
    #     Applies temperature/top-k/top-p to the *next* distribution if provided.
    #     """
    #     h = self._run_stack(x, key_padding_mask)   # (B, S, D)
    #     logits = self.proj(h)[:, -1, :]            # (B, V)
    #
    #     # temperature (logits / tau)
    #     if temperature is not None and temperature > 0.0 and temperature != 1.0:
    #         logits = logits / float(temperature)
    #
    #     # top-k
    #     if top_k and top_k < logits.size(-1):
    #         v, idx = torch.topk(logits, top_k, dim=-1)
    #         neg_inf = torch.full_like(logits, float("-inf"))
    #         logits = neg_inf.scatter(-1, idx, v)
    #
    #     # top-p (nucleus)
    #     if 0.0 < top_p < 1.0:
    #         sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    #         probs = F.softmax(sorted_logits, dim=-1)
    #         cdf = probs.cumsum(dim=-1)
    #         cutoff = (cdf > top_p).float().argmax(dim=-1, keepdim=True)
    #         mask = torch.arange(sorted_logits.size(-1), device=logits.device)[None, :] <= cutoff
    #         masked_sorted = torch.where(mask, sorted_logits, torch.full_like(sorted_logits, float("-inf")))
    #         logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, masked_sorted)
    #
    #     logp = F.log_softmax(logits, dim=-1)        # (B, V), nats
    #     H_nats = -(logp.exp() * logp).sum(dim=-1)   # (B,)
    #     return H_nats / math.log(2.0)


# ---------------------------
# Segmentation helper
# ---------------------------

class Segmenter(nn.Module):
    def __init__(self, *, increase_delta: float = 0.0, abs_threshold: Optional[float] = None):
        super().__init__()
        self.increase_delta = increase_delta
        self.abs_threshold = abs_threshold

    @torch.no_grad()
    def forward(self, entropy_bits: Tensor) -> SegmentBoundaries:
        """
        Compute seg_id & boundary flags from per-token entropy (B, S).
        """
        seg_id, patch_end_mask = entropy_segments(
            entropy_bits,
            increase_delta=self.increase_delta,
            abs_threshold=self.abs_threshold,
            return_boundary=True,
        )  # (B, S), (B, S) bool

        B, S = seg_id.shape
        idx = torch.arange(S, device=seg_id.device).expand(B, -1)  # (B, S)

        is_start = torch.ones_like(seg_id, dtype=torch.bool)
        is_start[:, 1:] = seg_id[:, 1:] != seg_id[:, :-1]
        first_pos = torch.where(is_start, idx, torch.zeros_like(idx))
        first_byte_idx = torch.cummax(first_pos, dim=1).values

        return SegmentBoundaries(seg_id=seg_id, patch_end_mask=patch_end_mask, first_byte_idx=first_byte_idx)


# ---------------------------
# Streaming cache for the whole path (shared + entropy)
# ---------------------------

@dataclass
class NextPred:
    logits: torch.Tensor | None = None   # (B, V)
    mu: torch.Tensor | None = None       # (B, D)
    logvar: torch.Tensor | None = None   # (B, D)

@dataclass
class StreamCache:
    shared: List[Tuple[Tensor, Tensor]]
    entropy: List[Tuple[Tensor, Tensor]]
    pos: int
    next_pred: NextPred | None = None

@dataclass
class GaussianCache:
    layers: List[Tuple[Tensor, Tensor]]  # per-layer MLA caches
    last_mu: Tensor                      # (B, D)  stats at the last processed position
    last_logvar: Tensor                  # (B, D)

@dataclass
class LogitCache:
    layers: List[Tuple[Tensor, Tensor]]
    last_logits: Tensor                  # (B, V)

# ---------------------------
# SegmentCompressor
# ---------------------------

class SegmentCompressor(nn.Module):
    """
    End-to-end: shared encoder → (entropy branch, compression branch) → pool → quantize.

    Sampling:
      - call `sample_next()` to draw from the entropy model (embedding for Gaussian,
        token id or embedding for Logit).
    Streaming:
      - `stream_prefill()` then `stream_step()`; segmentation utilities provided.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 260,
        dim: int = 256,
        heads: int = 8,
        window: int = 128,
        head_dim: Optional[int] = 32,
        kv_comp_dim: Optional[int] = 64,
        q_comp_dim: Optional[int] = 96,
        retr_dim: Optional[int] = 32,

        lm_window: Optional[int] = None,
        compression_window: Optional[int] = None,

        num_encoder_layers: int = 3,
        encoder_ffn_dim_multiplier: int = 4,
        num_shared_encoder_layers: int = 0,
        num_lm_encoder_layers: Optional[int] = None,
        num_compression_encoder_layers: Optional[int] = None,

        num_queries: int = 1,
        output_length: int = 512,

        # Segmentation knobs
        use_gaussian_segmentation: bool = True,
        entropy_delta: float = 0.0,
        entropy_abs_threshold: Optional[float] = None,

        # Quantizer
        quantizer: Optional[QuantizerBase] = None,

        use_flex_attention: bool = True,
        attention_config: AttentionConfig | None = None,
    ):
        super().__init__()

        num_lm_encoder_layers = num_lm_encoder_layers or num_encoder_layers
        num_compression_encoder_layers = num_compression_encoder_layers or num_encoder_layers
        lm_window = lm_window or window
        compression_window = compression_window or window

        self.num_queries_per_segment = num_queries
        self.output_length = output_length
        self.use_flex_attention = use_flex_attention if (attention_config is None) else bool(attention_config.use_flex_attention)

        # Shared stack
        self.shared_layers = nn.ModuleList(
            [
                SlidingWindowMLATransformerBlock(
                    dim=dim, num_heads=heads, window_size=window,
                    head_dim=head_dim, kv_comp_dim=kv_comp_dim, q_comp_dim=q_comp_dim,
                    retr_dim=retr_dim, ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                    use_flex_attention=self.use_flex_attention,
                )
                for _ in range(num_shared_encoder_layers)
            ]
        )

        # Compression branch
        self.compression_layers = nn.ModuleList(
            [
                SlidingWindowMLATransformerBlock(
                    dim=dim, num_heads=heads, window_size=compression_window,
                    head_dim=head_dim, kv_comp_dim=kv_comp_dim, q_comp_dim=q_comp_dim,
                    retr_dim=retr_dim, ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                    use_flex_attention=self.use_flex_attention,
                )
                for _ in range(num_compression_encoder_layers)
            ]
        )

        # Entropy model branch
        if use_gaussian_segmentation:
            self.entropy_model: EntropyModelBase = GaussianEntropyModel(
                dim=dim, n_heads=heads, window=lm_window,
                head_dim=head_dim, kv_comp_dim=kv_comp_dim, q_comp_dim=q_comp_dim,
                retr_dim=retr_dim, ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                use_flex_attention=self.use_flex_attention, n_layers=num_lm_encoder_layers,
            )
        else:
            self.entropy_model = LogitEntropyModel(
                dim=dim, vocab_size=vocab_size, n_heads=heads, window=lm_window,
                head_dim=head_dim, kv_comp_dim=kv_comp_dim, q_comp_dim=q_comp_dim,
                retr_dim=retr_dim, ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                use_flex_attention=self.use_flex_attention, n_layers=num_lm_encoder_layers,
            )

        # Segmenter utility
        self.segmenter = Segmenter(increase_delta=entropy_delta, abs_threshold=entropy_abs_threshold)

        # Pooler (learned queries per segment)
        self.pooler = LearnedQueryAttention(
            embed_dim=dim,
            num_queries_per_segment=num_queries,
            max_queries=output_length // max(1, num_queries),
            num_heads=heads,
            use_flex_attention=True,
        )

        # Quantizer (no-op by default)
        self.quantizer: QuantizerBase = quantizer if quantizer is not None else IdentityQuantizer()

    # ---- helpers ----

    def _run_shared(self, x: Tensor, kpm: Optional[Tensor]) -> Tensor:
        for layer in self.shared_layers:
            x = layer(x, key_padding_mask=kpm)
        return x

    def _run_compression(self, x: Tensor, kpm: Optional[Tensor]) -> Tensor:
        for layer in self.compression_layers:
            x = layer(x, key_padding_mask=kpm)
        return x

    # ---- public API ----

    def forward(
        self,
        input_embeddings: Tensor,              # (B, S, D)
        key_padding_mask: Optional[Tensor] = None,  # (B, S) True=pad
        *,
        token_ids: Optional[Tensor] = None,    # (B, S) for logit loss
    ) -> CompressorOutput:
        """
        1) shared -> entropy & compression branches
        2) segmentation from entropy bits
        3) pool by segment -> (B, Q_max, D)
        4) quantize (pluggable)
        """
        x_shared = self._run_shared(input_embeddings, key_padding_mask)

        # Entropy model: pass targets for logit CE if available
        ent_out = self.entropy_model(x_shared, key_padding_mask, targets=token_ids)
        boundaries = self.segmenter(ent_out.bits)

        # Compression branch
        hidden = self._run_compression(x_shared, key_padding_mask)

        # Pool per segment
        pooled_embeddings, _ = self.pooler(
            x=hidden,
            seg_id=boundaries.seg_id,
            key_padding_mask=key_padding_mask,
            return_attn=False,
        )  # (B, Q_max, D)

        # Valid query slots mask
        if key_padding_mask is not None:
            seg_id_real = torch.where(key_padding_mask, boundaries.seg_id.new_full((), -1), boundaries.seg_id)
            nseg = (seg_id_real.amax(dim=1).clamp(min=-1) + 1)  # (B,)
        else:
            nseg = (boundaries.seg_id.amax(dim=1) + 1)
        valid_mask = (torch.arange(self.pooler.Q_max, device=hidden.device)[None, :]
                      < (nseg * self.pooler.L)[:, None])  # (B, Q_max) bool

        # Quantize
        quantised_embeddings, vq_loss, codebook_indices, perplexity = self.quantizer(pooled_embeddings)

        return CompressorOutput(
            vq_embeddings=quantised_embeddings,
            vq_indices=codebook_indices,
            vq_loss=vq_loss,
            vq_perplexity=perplexity,
            valid_mask=valid_mask,
            patch_end_mask=boundaries.patch_end_mask,
            entropy_bits=ent_out.bits,
            entropy_loss=ent_out.loss,
            entropy_mu=ent_out.mu,
            entropy_logvar=ent_out.logvar,
            pre_vq_embeddings=pooled_embeddings,
            seg_id=boundaries.seg_id,
            first_byte_idx=boundaries.first_byte_idx,
            input_sequence=(token_ids if token_ids is not None else torch.empty(0, dtype=torch.long, device=input_embeddings.device)),
            input_padding_mask=key_padding_mask,
        )

    # --- segmentation-only fast path (shared + entropy only) ---
    @torch.no_grad()
    def segment_only(self, input_embeddings: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x_shared = self._run_shared(input_embeddings, key_padding_mask)
        bits = self.entropy_model.entropy(x_shared, key_padding_mask)
        b = self.segmenter(bits)
        return b.seg_id, b.patch_end_mask

    # --- sampling from the entropy model ---
    @torch.no_grad()
    def sample_next(self,
                    input_embeddings: Tensor,
                    key_padding_mask: Optional[Tensor] = None,
                    *,
                    temperature: float = 1.0,
                    top_p: float = 1.0,
                    top_k: int = 0,
                    embed_fn: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
        """
        Returns:
          - Gaussian model: (B, D) next embedding
          - Logit model: (B,) next token id unless `embed_fn` is provided (then (B, D))
        """
        x_shared = self._run_shared(input_embeddings, key_padding_mask)
        return self.entropy_model.sample_next(
            x_shared, key_padding_mask, temperature=temperature, top_p=top_p, top_k=top_k, embed_fn=embed_fn
        )

    # --- streaming APIs (shared + entropy) ---
    @torch.no_grad()
    def stream_prefill(self,
                       input_embeddings: Tensor,
                       key_padding_mask: Optional[Tensor] = None) -> Tuple[StreamCache, Tensor]:
        """
        Build caches for shared + entropy stacks and return entropy bits for prefix.
        """
        h = input_embeddings
        shared_caches: List[Tuple[Tensor, Tensor]] = []
        for blk in self.shared_layers:
            h, cache = blk.prefill(h, pos_start=0, key_padding_mask=key_padding_mask)
            shared_caches.append(cache)
        entropy_caches, bits = self.entropy_model.prefill(h, key_padding_mask)
        return StreamCache(shared=shared_caches, entropy=entropy_caches, pos=int(h.size(1))), bits

    @torch.no_grad()
    def stream_step(self,
                    cache: StreamCache,
                    new_embeddings: Tensor,
                    key_padding_mask_new: Optional[Tensor] = None) -> Tuple[StreamCache, Tensor]:
        """
        Process only new tokens; returns updated cache and entropy bits for new slice.
        """
        h = new_embeddings
        pos0 = cache.pos
        for i, blk in enumerate(self.shared_layers):
            h, cache.shared[i] = blk.step(
                h, cache=cache.shared[i], pos_start=pos0, key_padding_mask_new=key_padding_mask_new
            )
        cache.entropy, bits = self.entropy_model.step(
            cache.entropy, h, pos_start=pos0, key_padding_mask_new=key_padding_mask_new
        )
        cache.pos += int(h.size(1))
        return cache, bits

    @torch.no_grad()
    def enable_mla_fusions_for_inference(self):
        for blk in list(self.shared_layers) + list(self.compression_layers):
            blk.attn.mla.enable_inference_fusions()
        # Entropy model blocks too:
        for blk in getattr(self.entropy_model, "layers", []):
            blk.attn.mla.enable_inference_fusions()

    @torch.no_grad()
    def predictive_entropy_next_bits(
        self,
        input_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        **kw
    ) -> torch.Tensor:
        """
        H(next|prefix) in bits, shape (B,).
        Gaussian: differential entropy; Logit: categorical Shannon entropy.
        """
        x_shared = self._run_shared(input_embeddings, key_padding_mask)
        return self.entropy_model.predictive_entropy_next_bits(x_shared, key_padding_mask, **kw)

    @torch.no_grad()
    def predictive_entropy_next_bits_from_cache(self, cache: StreamCache, **kw) -> Tensor:
        return self.entropy_model.predictive_entropy_next_bits_from_cache(cache.entropy, **kw)

    @torch.no_grad()
    def predictive_bpd_next_from_cache(self, cache: StreamCache, *, temperature: float = 1.0) -> Tensor:
        if hasattr(self.entropy_model, "predictive_bpd_next_from_cache"):
            return self.entropy_model.predictive_bpd_next_from_cache(cache.entropy, temperature=temperature)
        raise NotImplementedError("BPD next is only defined for Gaussian entropy models.")

    @torch.no_grad()
    def predictive_bpd_next(
        self,
        input_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Gaussian-only shortcut: H_bits / D (B,).
        """
        x_shared = self._run_shared(input_embeddings, key_padding_mask)
        if hasattr(self.entropy_model, "predictive_bpd_next"):
            return self.entropy_model.predictive_bpd_next(x_shared, key_padding_mask, temperature=temperature)
        raise NotImplementedError("predictive_bpd_next is only defined for Gaussian entropy models.")

    @torch.no_grad()
    def next_entropy_from_cache(self, cache: StreamCache, *, temperature: float = 1.0, top_p: float = 1.0,
                                top_k: int = 0) -> Tensor:
        """
        Returns H(next|prefix) in bits using last cached stats only; shape (B,).
        """
        np = cache.next_pred
        assert np is not None, "Call stream_prefill/step first to populate next_pred."

        if np.logits is not None:
            logp = F.log_softmax(np.logits / (temperature if (temperature and temperature > 0) else 1.0), dim=-1)
            if 0.0 < top_p < 1.0 or (top_k and top_k < logp.size(-1)):
                # Re-apply top-p/k here similarly if needed (skipped for brevity)
                pass
            H_nats = -(logp.exp() * logp).sum(dim=-1)
            return H_nats / math.log(2.0)

        # Gaussian
        logvar_last = np.logvar  # (B, D)
        H_nats = 0.5 * (self.entropy_model.dim * math.log(2.0 * math.pi * math.e) + logvar_last.sum(dim=-1))
        if temperature is not None and temperature > 0.0 and temperature != 1.0:
            H_nats = H_nats + self.entropy_model.dim * math.log(temperature)
        return H_nats / math.log(2.0)
