from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention.factory import make_sliding_self_block
from .attention.pooling.learned_query import LearnedQueryAttention
from .config_types import AttentionConfig, EntropyModelConfig, EntropyGaussianHeadConfig, EntropyLogitHeadConfig, EntropyLossConfig, EntropySegmentationConfig
from .utils import entropy_segments, token_entropy

# ---------------------------
# Data containers / Protocols
# ---------------------------


@dataclass
class SegmentBoundaries:
    seg_id: Tensor  # (B, S) int64; 0,0,0,1,1,2,...
    patch_end_mask: Tensor  # (B, S) bool; True on last token of segment
    first_byte_idx: Tensor  # (B, S) int64; first token index for each position's segment


@dataclass
class EntropyOut:
    """Outputs from an entropy model evaluated on a sequence."""

    bits: Tensor  # (B, S) per-token entropy in bits, aligned to input (pad at t=0)
    loss: Optional[Tensor] = None  # scalar loss if available
    logits: Optional[Tensor] = None  # (B, S, V) for logit model
    mu: Optional[Tensor] = None  # (B, S, D) for Gaussian model
    logvar: Optional[Tensor] = None  # (B, S, D) for Gaussian model


@dataclass
class CompressorOutput:
    vq_embeddings: Optional[Tensor]  # (B, Q_max, D) or None if no quantizer
    vq_indices: Optional[Tensor]  # (B, Q_max) or None
    vq_loss: Optional[Tensor]  # scalar or None
    vq_perplexity: Optional[Tensor]  # scalar or None

    valid_mask: Tensor  # (B, Q_max) bool
    patch_end_mask: Tensor  # (B, S) bool
    entropy_bits: Tensor  # (B, S) bits used for segmentation
    entropy_loss: Optional[Tensor]  # scalar or None
    entropy_mu: Optional[Tensor]  # (B, S, D) for Gaussian
    entropy_logvar: Optional[Tensor]  # (B, S, D) for Gaussian

    pre_vq_embeddings: Tensor  # (B, Q_max, D)
    seg_id: Tensor  # (B, S) int64
    first_byte_idx: Tensor  # (B, S) int64

    input_sequence: Tensor  # (B, S) long or empty ()
    input_padding_mask: Optional[Tensor]  # (B, S) bool or None


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

    def step(
        self, caches: List[Tuple[Tensor, Tensor]], new_x: Tensor, pos_start: int, key_padding_mask_new: Optional[Tensor]
    ) -> Tuple[List[Tuple[Tensor, Tensor]], Tensor]:
        """
        Process only new tokens using caches (streaming).
        Returns: (updated_caches, entropy_bits_for_new_tokens)
        """
        raise NotImplementedError

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor], *, targets: Optional[Tensor] = None) -> EntropyOut:
        raise NotImplementedError

    @torch.no_grad()
    def entropy(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        return self.forward(x, key_padding_mask).bits

    @torch.no_grad()
    def sample_next(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        embed_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
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
        temperature: float = 1.0,  # Gaussian: scales std; Logit: scales logits via 1/temperature
        top_p: float = 1.0,  # Logit-only: nucleus on the NEXT distribution (optional)
        top_k: int = 0,  # Logit-only: top-k on the NEXT distribution (optional)
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
        cache: Any,  # FlexCache (FlexibleEntropyModel)
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


@dataclass
class FlexCache:
    """Streaming cache for FlexibleEntropyModel.

    Stores trunk layer caches and the last-step predictions for available heads.
    """

    layers: List[Tuple[Tensor, Tensor]]
    last_logits: Tensor | None = None  # (B, V)
    last_mu: Tensor | None = None  # (B, D)
    last_logvar: Tensor | None = None  # (B, D)


class FlexibleEntropyModel(EntropyModelBase):
    """
    Causal transformer trunk + selectable heads (logit, gaussian) with configurable losses
    and segmentation source.
    """

    def __init__(
        self,
        *,
        dim: int,
        vocab_size: int,
        trunk_heads: int,
        trunk_window: int,
        trunk_layers: int,
        attn_cfg: AttentionConfig,
        cfg: EntropyModelConfig,
        tied_embedding_weight: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        # Trunk
        self.layers = nn.ModuleList(
            [
                make_sliding_self_block(
                    dim=dim,
                    num_heads=trunk_heads,
                    window_size=trunk_window,
                    ffn_dim_multiplier=4,
                    cfg=attn_cfg,
                )
                for _ in range(trunk_layers)
            ]
        )
        self.dim = dim

        # Heads configuration
        self.cfg = cfg
        self.vocab_size = vocab_size

        # Logit head
        self.use_logit = bool(getattr(cfg.logit, "use", True))
        self.logit_norm = nn.RMSNorm(dim) if self.use_logit else None
        self.logit_tied_weight: Tensor | None = None
        if self.use_logit:
            if getattr(cfg.logit, "tie_to_embedding", False) and tied_embedding_weight is not None:
                self.logit_tied_weight = tied_embedding_weight
                self.logit_proj = None
            else:
                self.logit_proj = nn.Linear(dim, vocab_size)

        # Gaussian head (single Gaussian for now)
        self.use_gaussian = bool(getattr(cfg.gaussian, "use", False))
        if self.use_gaussian:
            self.gauss_mu = nn.Linear(dim, dim)
            self.gauss_logvar = nn.Linear(dim, dim)
            self.gauss_clamp = (
                float(getattr(cfg.gaussian, "clamp_logvar_min", -8.0)),
                float(getattr(cfg.gaussian, "clamp_logvar_max", 8.0)),
            )
            self.gauss_extra_layers = nn.ModuleList(
                [
                    make_sliding_self_block(
                        dim=dim,
                        num_heads=trunk_heads,
                        window_size=trunk_window,
                        ffn_dim_multiplier=4,
                        cfg=attn_cfg,
                    )
                    for _ in range(int(getattr(cfg.gaussian, "extra_layers", 0)))
                ]
            )
            self.gauss_detach_trunk = bool(getattr(cfg.gaussian, "detach_trunk", False))

        # Loss weights
        self.ce_weight = float(getattr(cfg.loss, "ce_weight", 1.0))
        self.nll_weight = float(getattr(cfg.loss, "nll_weight", 0.0))

        # Segmentation policy
        self.seg_source = getattr(cfg.segmentation, "source", "logit")
        self.seg_temperature = float(getattr(cfg.segmentation, "temperature", 1.0))
        self.seg_top_p = float(getattr(cfg.segmentation, "top_p", 1.0))
        self.seg_top_k = int(getattr(cfg.segmentation, "top_k", 0))

    def _run_trunk(self, x: Tensor, kpm: Optional[Tensor]) -> Tensor:
        h = x
        for blk in self.layers:
            h = blk(h, key_padding_mask=kpm)
        return h

    def _run_gauss_path(self, h: Tensor, kpm: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if not self.use_gaussian:
            raise RuntimeError("Gaussian head not enabled")
        hg = h.detach() if getattr(self, "gauss_detach_trunk", False) else h
        for blk in getattr(self, "gauss_extra_layers", []):
            hg = blk(hg, key_padding_mask=kpm)
        mu = self.gauss_mu(hg)
        logvar = self.gauss_logvar(hg).clamp(*self.gauss_clamp)
        return mu, logvar

    @staticmethod
    def _bpd_bits(mu: Tensor, logvar: Tensor, target: Tensor) -> Tensor:
        # Predict x[:,1:] from stats at positions [:, :-1]. Return (B,S) with pad at t=0.
        x = target[:, 1:, :]
        m = mu[:, :-1, :]
        lv = logvar[:, :-1, :]
        scale = mu.new_tensor(0.5 / math.log(2.0))
        log_2pi = mu.new_tensor(math.log(2.0 * math.pi))
        dx = x - m
        tmp = dx.square() * torch.exp(-lv) + lv + log_2pi  # (B, L-1, D)
        bpd = tmp.mean(dim=-1) * scale  # (B, L-1)
        return F.pad(bpd, (1, 0), value=0.0)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor], *, targets: Optional[Tensor] = None) -> EntropyOut:
        h = self._run_trunk(x, key_padding_mask)

        logits = None
        bits_logit = None
        ce_loss = x.new_zeros(())
        if self.use_logit:
            hn = self.logit_norm(h)
            if self.logit_proj is not None:
                logits = self.logit_proj(hn)
            else:
                logits = F.linear(hn, self.logit_tied_weight)  # type: ignore[arg-type]
            bits_logit = token_entropy(logits)
            if targets is not None:
                if key_padding_mask is not None:
                    m = ~key_padding_mask[:, 1:]
                else:
                    m = torch.ones_like(targets[:, 1:], dtype=torch.bool)
                per_tok = F.cross_entropy(
                    logits[:, :-1, :].transpose(1, 2),
                    targets[:, 1:],
                    reduction="none",
                )
                ce_loss = (per_tok * m).sum() / m.sum().clamp(min=1)

        mu = None
        logvar = None
        bits_gauss = None
        nll_loss = x.new_zeros(())
        if self.use_gaussian:
            mu, logvar = self._run_gauss_path(h, key_padding_mask)
            bits_gauss = self._bpd_bits(mu, logvar, x)
            # average over non-padded steps t>=1
            if key_padding_mask is not None:
                msk = ~key_padding_mask
            else:
                msk = torch.ones_like(bits_gauss, dtype=torch.bool)
            nll_loss = (bits_gauss[:, 1:] * msk[:, 1:]).sum() / msk[:, 1:].sum().clamp(min=1)

        # Choose segmentation bits
        bits = None
        if self.seg_source == "logit" and bits_logit is not None:
            bits = bits_logit
        elif self.seg_source == "gaussian" and bits_gauss is not None:
            bits = bits_gauss
        else:
            bits = bits_logit if bits_logit is not None else (bits_gauss if bits_gauss is not None else x.new_zeros(x.size()[:2]))

        total_loss = self.ce_weight * ce_loss + self.nll_weight * nll_loss

        return EntropyOut(bits=bits, loss=total_loss, logits=logits, mu=mu, logvar=logvar)

    def prefill(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tuple[FlexCache, Tensor]:
        h = x
        caches: List[Tuple[Tensor, Tensor]] = []
        for blk in self.layers:
            h, cache = blk.prefill(h, pos_start=0, key_padding_mask=key_padding_mask)
            caches.append(cache)

        last_logits = None
        bits_logit = None
        if self.use_logit:
            hn = self.logit_norm(h)
            logits = self.logit_proj(hn) if self.logit_proj is not None else F.linear(hn, self.logit_tied_weight)  # type: ignore[arg-type]
            last_logits = logits[:, -1, :]
            bits_logit = token_entropy(logits)

        last_mu = None
        last_logvar = None
        bits_gauss = None
        if self.use_gaussian:
            mu, logvar = self._run_gauss_path(h, key_padding_mask)
            last_mu = mu[:, -1, :]
            last_logvar = logvar[:, -1, :]
            bits_gauss = self._bpd_bits(mu, logvar, x)

        # Choose segmentation bits
        bits = bits_logit if (self.seg_source == "logit" and bits_logit is not None) else (
            bits_gauss if (self.seg_source == "gaussian" and bits_gauss is not None) else (bits_logit or bits_gauss)
        )
        if bits is None:
            bits = x.new_zeros(x.size()[:2])

        return FlexCache(layers=caches, last_logits=last_logits, last_mu=last_mu, last_logvar=last_logvar), bits

    def step(
        self, cache: FlexCache, new_x: Tensor, pos_start: int, key_padding_mask_new: Optional[Tensor]
    ) -> Tuple[FlexCache, Tensor]:
        h = new_x
        for i, blk in enumerate(self.layers):
            h, cache.layers[i] = blk.step(
                h, cache=cache.layers[i], pos_start=pos_start, key_padding_mask_new=key_padding_mask_new
            )

        bits_logit_blk = None
        if self.use_logit:
            hn = self.logit_norm(h)
            logits_new = self.logit_proj(hn) if self.logit_proj is not None else F.linear(hn, self.logit_tied_weight)  # type: ignore[arg-type]
            cache.last_logits = logits_new[:, -1, :]
            bits_logit_blk = token_entropy(logits_new)

        bits_gauss_blk = None
        if self.use_gaussian:
            mu_new, logvar_new = self._run_gauss_path(h, key_padding_mask_new)
            cache.last_mu = mu_new[:, -1, :]
            cache.last_logvar = logvar_new[:, -1, :]
            bits_gauss_blk = self._bpd_bits(mu_new, logvar_new, new_x)

        bits_blk = bits_logit_blk if (self.seg_source == "logit" and bits_logit_blk is not None) else (
            bits_gauss_blk if (self.seg_source == "gaussian" and bits_gauss_blk is not None) else (bits_logit_blk or bits_gauss_blk)
        )
        if bits_blk is None:
            bits_blk = new_x.new_zeros(new_x.size()[:2])

        return cache, bits_blk

    @torch.no_grad()
    def sample_next(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        embed_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
        h = self._run_trunk(x, key_padding_mask)
        if self.seg_source == "gaussian" and self.use_gaussian:
            mu, logvar = self._run_gauss_path(h, key_padding_mask)
            mu_last = mu[:, -1, :]
            logvar_last = logvar[:, -1, :]
            if temperature <= 0:
                return mu_last
            std = torch.exp(0.5 * logvar_last)
            return mu_last + (temperature * std) * torch.randn_like(std)
        # default to logit
        assert self.use_logit, "Logit head not enabled"
        hn = self.logit_norm(h)
        logits = self.logit_proj(hn) if self.logit_proj is not None else F.linear(hn, self.logit_tied_weight)  # type: ignore[arg-type]
        logits = logits[:, -1, :]
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        if top_k and top_k < logits.size(-1):
            v, idx = torch.topk(logits, top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter(-1, idx, v)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            cutoff = (cum > top_p).float().argmax(dim=-1, keepdim=True)
            mask = torch.full_like(sorted_logits, float("-inf"))
            keep = torch.arange(sorted_logits.size(-1), device=logits.device)[None, :] <= cutoff
            sorted_logits = torch.where(keep, sorted_logits, mask)
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return embed_fn(next_ids) if embed_fn is not None else next_ids

    @torch.no_grad()
    def predictive_entropy_next_bits(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Tensor:
        h = self._run_trunk(x, key_padding_mask)
        if self.seg_source == "gaussian" and self.use_gaussian:
            _, logvar = self._run_gauss_path(h, key_padding_mask)
            lv = logvar[:, -1, :]
            H_nats = 0.5 * (self.dim * math.log(2.0 * math.pi * math.e) + lv.sum(dim=-1))
            if temperature is not None and temperature > 0.0 and temperature != 1.0:
                H_nats = H_nats + self.dim * math.log(temperature)
            return H_nats / math.log(2.0)
        # logit
        assert self.use_logit
        hn = self.logit_norm(h)
        logits = self.logit_proj(hn) if self.logit_proj is not None else F.linear(hn, self.logit_tied_weight)  # type: ignore[arg-type]
        logits = logits[:, -1, :]
        return _categorical_entropy_bits(logits, temperature=temperature, top_p=top_p, top_k=top_k)

    @torch.no_grad()
    def predictive_entropy_next_bits_from_cache(
        self, cache: FlexCache, *, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0
    ) -> Tensor:
        if self.seg_source == "gaussian" and self.use_gaussian and cache.last_logvar is not None:
            lv = cache.last_logvar
            H_nats = 0.5 * (self.dim * math.log(2.0 * math.pi * math.e) + lv.sum(dim=-1))
            if temperature is not None and temperature > 0.0 and temperature != 1.0:
                H_nats = H_nats + self.dim * math.log(temperature)
            return H_nats / math.log(2.0)
        assert cache.last_logits is not None, "No last_logits in cache"
        return _categorical_entropy_bits(cache.last_logits, temperature=temperature, top_p=top_p, top_k=top_k)

    @torch.no_grad()
    def predictive_bpd_next(
        self, x: Tensor, key_padding_mask: Tensor | None = None, *, temperature: float = 1.0
    ) -> Tensor:
        if not self.use_gaussian:
            raise NotImplementedError("predictive_bpd_next only available with Gaussian head")
        H_bits = self.predictive_entropy_next_bits(x, key_padding_mask, temperature=temperature)
        return H_bits / float(self.dim)

    @torch.no_grad()
    def predictive_bpd_next_from_cache(self, cache: FlexCache, *, temperature: float = 1.0) -> Tensor:
        if not self.use_gaussian or cache.last_logvar is None:
            raise NotImplementedError("predictive_bpd_next_from_cache only available with Gaussian head")
        lv = cache.last_logvar
        H_nats = 0.5 * (self.dim * math.log(2.0 * math.pi * math.e) + lv.sum(dim=-1))
        if temperature is not None and temperature > 0.0 and temperature != 1.0:
            H_nats = H_nats + self.dim * math.log(temperature)
        return (H_nats / math.log(2.0)) / float(self.dim)


 


@torch.no_grad()
def _categorical_entropy_bits(
    logits: Tensor, *, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0
) -> Tensor:
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
    logits: torch.Tensor | None = None  # (B, V)
    mu: torch.Tensor | None = None  # (B, D)
    logvar: torch.Tensor | None = None  # (B, D)


@dataclass
class StreamCache:
    shared: List[Tuple[Tensor, Tensor]]
    entropy: Any  # FlexibleEntropyModel cache (FlexCache)
    pos: int
    next_pred: NextPred | None = None


## Legacy caches removed in favor of FlexCache


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
        output_length: int = 1024,
        # Segmentation knobs
        use_gaussian_segmentation: bool = False,
        entropy_delta: float = 0.0,
        entropy_abs_threshold: Optional[float] = None,
        # Quantizer
        quantizer: Optional[QuantizerBase] = None,
        use_flex_attention: bool = True,
        attention_config: AttentionConfig | None = None,
        # New entropy model configuration
        entropy_config: EntropyModelConfig | None = None,
        # Optional token embedding for weight tying in logit head
        token_embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__()

        num_lm_encoder_layers = num_lm_encoder_layers or num_encoder_layers
        num_compression_encoder_layers = num_compression_encoder_layers or num_encoder_layers
        lm_window = lm_window or window
        compression_window = compression_window or window

        self.num_queries_per_segment = num_queries
        self.output_length = output_length
        self.use_flex_attention = (
            use_flex_attention if (attention_config is None) else (attention_config.kernel == "flex")
        )

        # Token embedding shared by both entropy and compression branches
        self.embedding = nn.Embedding(vocab_size, dim)

        # Shared stack
        attn_cfg = attention_config or AttentionConfig(
            variant="mla",
            kernel=("flex" if self.use_flex_attention else "sdpa"),
            head_dim=head_dim,
            kv_comp_dim=kv_comp_dim,
            q_comp_dim=q_comp_dim,
            retr_dim=retr_dim,
        )

        self.shared_layers = nn.ModuleList(
            [
                make_sliding_self_block(
                    dim=dim,
                    num_heads=heads,
                    window_size=window,
                    ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                    cfg=attn_cfg,
                )
                for _ in range(num_shared_encoder_layers)
            ]
        )

        # Compression branch
        self.compression_layers = nn.ModuleList(
            [
                make_sliding_self_block(
                    dim=dim,
                    num_heads=heads,
                    window_size=compression_window,
                    ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                    cfg=attn_cfg,
                )
                for _ in range(num_compression_encoder_layers)
            ]
        )

        # Entropy model branch (Flexible)
        if entropy_config is None:
            # Build a default config from legacy args
            default_cfg = EntropyModelConfig(
                n_layers=int(num_lm_encoder_layers),
                n_heads=int(heads),
                window=int(lm_window),
                attention_config=attn_cfg,
            )
            # Default policy: use logit unless gaussian flag requested
            default_cfg.logit.use = not use_gaussian_segmentation
            default_cfg.gaussian.use = use_gaussian_segmentation
            default_cfg.loss.ce_weight = 1.0 if default_cfg.logit.use else 0.0
            default_cfg.loss.nll_weight = 1.0 if default_cfg.gaussian.use else 0.0
            default_cfg.segmentation.source = "gaussian" if use_gaussian_segmentation else "logit"
            entropy_cfg_local = default_cfg
        else:
            entropy_cfg_local = entropy_config

        tied_w = None
        if getattr(entropy_cfg_local.logit, "use", True) and getattr(
            entropy_cfg_local.logit, "tie_to_embedding", False
        ):
            tied_w = self.embedding.weight

        self.entropy_model: EntropyModelBase = FlexibleEntropyModel(
            dim=dim,
            vocab_size=vocab_size,
            trunk_heads=heads,
            trunk_window=lm_window,
            trunk_layers=num_lm_encoder_layers,
            attn_cfg=attn_cfg,
            cfg=entropy_cfg_local,
            tied_embedding_weight=tied_w,
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

    def forward_ids(
        self,
        token_ids: Tensor,  # (B, S)
        key_padding_mask: Optional[Tensor] = None,
    ) -> CompressorOutput:
        """Convenience wrapper: embed tokens then run forward path."""
        embeddings = self.embedding(token_ids)
        return self.forward(embeddings, key_padding_mask=key_padding_mask, token_ids=token_ids)

    def forward(
        self,
        input_embeddings: Tensor,  # (B, S, D)
        key_padding_mask: Optional[Tensor] = None,  # (B, S) True=pad
        *,
        token_ids: Optional[Tensor] = None,  # (B, S) for logit loss
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
            nseg = seg_id_real.amax(dim=1).clamp(min=-1) + 1  # (B,)
        else:
            nseg = boundaries.seg_id.amax(dim=1) + 1
        valid_mask = (
            torch.arange(self.pooler.Q_max, device=hidden.device)[None, :] < (nseg * self.pooler.L)[:, None]
        )  # (B, Q_max) bool

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
            input_sequence=(
                token_ids if token_ids is not None else torch.empty(0, dtype=torch.long, device=input_embeddings.device)
            ),
            input_padding_mask=key_padding_mask,
        )

    # --- segmentation-only fast path (shared + entropy only) ---
    @torch.no_grad()
    def segment_only_ids(
        self, token_ids: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self.segment_only(self.embedding(token_ids), key_padding_mask)

    @torch.no_grad()
    def segment_only(
        self, input_embeddings: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        x_shared = self._run_shared(input_embeddings, key_padding_mask)
        bits = self.entropy_model.entropy(x_shared, key_padding_mask)
        b = self.segmenter(bits)
        return b.seg_id, b.patch_end_mask

    @torch.compile()
    def entropy_loss_ids(
        self, token_ids: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> EntropyOut:
        """Compute entropy-model outputs (bits and loss) from token ids only."""
        x = self.embedding(token_ids)
        h = self._run_shared(x, key_padding_mask)
        return self.entropy_model(h, key_padding_mask, targets=token_ids)

    # --- sampling from the entropy model ---
    @torch.no_grad()
    def sample_next(
        self,
        input_embeddings: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        embed_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
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
    def stream_prefill_ids(
        self, token_ids: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tuple[StreamCache, Tensor]:
        return self.stream_prefill(self.embedding(token_ids), key_padding_mask)

    @torch.no_grad()
    def stream_prefill(
        self, input_embeddings: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tuple[StreamCache, Tensor]:
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
    def stream_step_ids(
        self, cache: StreamCache, new_token_ids: Tensor, key_padding_mask_new: Optional[Tensor] = None
    ) -> Tuple[StreamCache, Tensor]:
        return self.stream_step(cache, self.embedding(new_token_ids), key_padding_mask_new)

    @torch.no_grad()
    def stream_step(
        self, cache: StreamCache, new_embeddings: Tensor, key_padding_mask_new: Optional[Tensor] = None
    ) -> Tuple[StreamCache, Tensor]:
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
        self, input_embeddings: torch.Tensor, key_padding_mask: torch.Tensor | None = None, **kw
    ) -> torch.Tensor:
        """
        H(next|prefix) in bits, shape (B,).
        Gaussian: differential entropy; Logit: categorical Shannon entropy.
        """
        x_shared = self._run_shared(input_embeddings, key_padding_mask)
        return self.entropy_model.predictive_entropy_next_bits(x_shared, key_padding_mask, **kw)

    @torch.no_grad()
    def predictive_entropy_next_bits_ids(
        self, token_ids: torch.Tensor, key_padding_mask: torch.Tensor | None = None, **kw
    ) -> torch.Tensor:
        return self.predictive_entropy_next_bits(self.embedding(token_ids), key_padding_mask, **kw)

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
        self, input_embeddings: torch.Tensor, key_padding_mask: torch.Tensor | None = None, *, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Gaussian-only shortcut: H_bits / D (B,).
        """
        x_shared = self._run_shared(input_embeddings, key_padding_mask)
        if hasattr(self.entropy_model, "predictive_bpd_next"):
            return self.entropy_model.predictive_bpd_next(x_shared, key_padding_mask, temperature=temperature)
        raise NotImplementedError("predictive_bpd_next is only defined for Gaussian entropy models.")

    @torch.no_grad()
    def next_entropy_from_cache(
        self, cache: StreamCache, *, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0
    ) -> Tensor:
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
