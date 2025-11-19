"""Residual code bridge managing discrete/latent boundaries per level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .vector_quantizer import (
    ComposedIndexCodec,
    MultiStageResidualVQ,
    embed_rvq_indices,
    rvq_stage_logits,
    rvq_stage_logits_and_greedy,
)


@dataclass
class QuantizeOutput:
    """Quantizer return payload for a single bridge."""

    quantized: Tensor
    indices: Tensor
    digits: List[Tensor]
    special_mask: Tensor
    vq_loss: Tensor
    perplexity: Tensor


class ResidualCodeBridge(nn.Module):
    """Owns both inbound embedding and outbound residual quantization.

    Parameters
    ----------
    model_dim: int
        Latent dimensionality ``D`` processed within this level.
    inbound_K: int
        Base of the inbound codec (vocabulary size per stage).
    inbound_depth: int
        Number of residual digits used to embed the input vocabulary. ``1``
        reduces to a standard embedding table.
    outbound_K: int
        Base of the outbound residual quantizer (size per stage).
    outbound_depth: int
        Number of residual stages for the outbound quantizer.
    share_codebooks: bool, default True
        When True, the inbound embedding reuses the outbound stage codebooks
        (summing the first ``inbound_depth`` stages). When False, dedicated
        inbound codebooks are allocated.
    """

    def __init__(
        self,
        *,
        model_dim: int,
        inbound_K: int,
        inbound_depth: int,
        outbound_K: int,
        outbound_depth: int,
        share_codebooks: bool = True,
        beta: float = 0.25,
        ema: bool = True,
        decay: float = 0.999,
        eps: float = 1e-5,
        reset_codes: bool = True,
        reset_interval: int = 250,
        max_codes_to_reset_pct: float = 0.1,
        replacement_buffer_size: int = 65536,
        vectors_per_step_to_buffer: int = 1024,
        inbound_bos: int = 0,
        inbound_eos: int = 1,
        inbound_pad: int = 2,
        outbound_bos: int = 256,
        outbound_eos: int = 257,
        outbound_pad: int = 258,
    ) -> None:
        super().__init__()

        self.model_dim = int(model_dim)
        self.inbound_K = int(inbound_K)
        self.inbound_depth = int(inbound_depth)
        self.share_codebooks = bool(share_codebooks)

        # Outbound quantizer (latent -> residual codes)
        self.outbound = MultiStageResidualVQ(
            K=outbound_K,
            D=model_dim,
            depth=outbound_depth,
            beta=beta,
            ema=ema,
            decay=decay,
            eps=eps,
            reset_codes=reset_codes,
            reset_interval=reset_interval,
            max_codes_to_reset_pct=max_codes_to_reset_pct,
            replacement_buffer_size=replacement_buffer_size,
            vectors_per_step_to_buffer=vectors_per_step_to_buffer,
            bos_token_id=outbound_bos,
            eos_token_id=outbound_eos,
            padding_token_id=outbound_pad,
        )

        if share_codebooks and inbound_depth <= outbound_depth and inbound_K == outbound_K:
            self.inbound_stage_weights = None  # reuse outbound stage codebooks
            self.inbound_codec = self.outbound.codec
        else:
            # Allocate dedicated inbound codebooks
            self.inbound_stage_weights = nn.ParameterList(
                [nn.Parameter(torch.randn(inbound_K, model_dim) * 0.02) for _ in range(inbound_depth)]
            )
            self.inbound_codec = ComposedIndexCodec(
                K=inbound_K,
                depth=inbound_depth,
                bos=inbound_bos,
                eos=inbound_eos,
                pad=inbound_pad,
            )

    # ------------------------------------------------------------------
    # Inbound helpers
    # ------------------------------------------------------------------
    def embed_tokens(
        self,
        token_ids: Optional[Tensor] = None,
        *,
        digits: Optional[List[Tensor]] = None,
        special_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Convert inbound discrete symbols to latent vectors in ℝ^D."""

        if digits is None:
            if token_ids is None:
                raise ValueError("embed_tokens requires token_ids when digits are not provided")
            digits, special_mask = self.inbound_codec.decompose(token_ids)
        elif special_mask is None and token_ids is not None:
            special_mask = self.inbound_codec.is_special(token_ids)

        emb = torch.zeros(
            digits[0].size(0),
            digits[0].size(1),
            self.model_dim,
            device=digits[0].device,
            dtype=self._stage_weight(0).dtype,
        )
        for s, digit in enumerate(digits):
            emb = emb + F.embedding(digit, self._stage_weight(s))

        if special_mask is not None and special_mask.any():
            pad_vec = self.outbound.pad_vector.to(device=emb.device, dtype=emb.dtype)
            emb = torch.where(special_mask.unsqueeze(-1), pad_vec.view(1, 1, -1), emb)

        return emb

    def inbound_compose(self, digits: List[Tensor]) -> Tensor:
        return self.inbound_codec.compose(digits)

    def inbound_decompose(self, ids: Tensor) -> Tuple[List[Tensor], Tensor]:
        return self.inbound_codec.decompose(ids)

    # ------------------------------------------------------------------
    # Outbound helpers
    # ------------------------------------------------------------------
    def quantize(self, latents: Tensor) -> QuantizeOutput:
        quantized, vq_loss, indices, perplexity = self.outbound(latents)
        digits, special_mask = self.outbound.codec.decompose(indices)
        return QuantizeOutput(
            quantized=quantized,
            indices=indices,
            digits=digits,
            special_mask=special_mask,
            vq_loss=vq_loss,
            perplexity=perplexity,
        )

    def codes_to_latents(
        self,
        indices: Optional[Tensor] = None,
        *,
        digits: Optional[List[Tensor]] = None,
        special_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if indices is None and digits is None:
            raise ValueError("codes_to_latents requires indices or digits")
        if digits is None:
            assert indices is not None
            return embed_rvq_indices(self.outbound, indices)
        if indices is None:
            indices = self.outbound.codec.compose(digits)
        if special_mask is None:
            special_mask = self.outbound.codec.is_special(indices)

        emb = torch.zeros(
            digits[0].size(0),
            digits[0].size(1),
            self.model_dim,
            device=digits[0].device,
            dtype=self.outbound.stage_codebook(0).dtype,
        )
        for s, digit in enumerate(digits):
            emb = emb + F.embedding(digit, self.outbound.stage_codebook(s))

        if special_mask is not None and special_mask.any():
            pad_vec = self.outbound.pad_vector.to(device=emb.device, dtype=emb.dtype)
            emb = torch.where(special_mask.unsqueeze(-1), pad_vec.view(1, 1, -1), emb)

        return emb

    def outbound_compose(self, digits: List[Tensor]) -> Tensor:
        return self.outbound.codec.compose(digits)

    def outbound_decompose(self, ids: Tensor) -> Tuple[List[Tensor], Tensor]:
        return self.outbound.codec.decompose(ids)

    def stage_logits(
        self,
        latents: Tensor,
        *,
        teacher_digits: Optional[List[Tensor]] = None,
        use_sqdist_logits: bool = False,
        residual_conditioning: bool = True,
    ) -> List[Tensor]:
        return rvq_stage_logits(
            self.outbound,
            latents,
            residual_conditioning=residual_conditioning,
            use_sqdist_logits=use_sqdist_logits,
            teacher_digits=teacher_digits,
        )

    def greedy_digits(
        self,
        latents: Tensor,
        *,
        use_sqdist_logits: bool = False,
        residual_conditioning: bool = True,
    ) -> List[Tensor]:
        _, digits = rvq_stage_logits_and_greedy(
            self.outbound,
            latents,
            residual_conditioning=residual_conditioning,
            use_sqdist_logits=use_sqdist_logits,
        )
        assert digits is not None
        return digits

    def logits_and_greedy(
        self,
        latents: Tensor,
        *,
        use_sqdist_logits: bool = False,
        residual_conditioning: bool = True,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        logits, digits = rvq_stage_logits_and_greedy(
            self.outbound,
            latents,
            residual_conditioning=residual_conditioning,
            use_sqdist_logits=use_sqdist_logits,
        )
        assert digits is not None
        return logits, digits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _stage_weight(self, stage: int) -> Tensor:
        if self.share_codebooks:
            return self.outbound.stage_codebook(stage)
        assert self.inbound_stage_weights is not None
        return self.inbound_stage_weights[stage]
