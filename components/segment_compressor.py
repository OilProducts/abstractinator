from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .learned_query_attention import LearnedQueryAttention
from .mla import SlidingWindowMLATransformerBlock
from .utils import _compiled, build_segment_queries_mask, entropy_segments, token_entropy, build_segment_queries_qseg
from .vector_quantizer import VectorQuantizer, MultiStageResidualVQ


@dataclass
class CompressorOutput:
    vq_embeddings: torch.Tensor
    vq_indices: torch.Tensor
    vq_loss: torch.Tensor
    vq_perplexity: torch.Tensor  # Out of the VQ module
    valid_mask: torch.Tensor
    patch_end_mask: torch.Tensor  # True if this is the last token of a segment
    # entropy_model_logits: torch.Tensor  # Logits from the LM branch
    # entropy_model_bpd: torch.Tensor  # Bits-per-dim from the LM branch (if Gaussian segmentation)
    # entropy_mu: torch.Tensor  # Mu from the LM branch (if Gaussian segmentation)
    # entropy_logvar: torch.Tensor
    entropy_model_ent: torch.Tensor  # Entropy from the LM branch
    entropy_model_loss: torch.Tensor  # NLL loss from the LM branch (if Gaussian segmentation)
    pre_vq_embeddings: torch.Tensor  # Embeddings before VQ
    seg_id: torch.Tensor  # Segment IDs for each token (B, S)
    first_byte_idx: torch.Tensor  # First byte index for each segment (B, S)
    input_sequence: torch.Tensor
    input_padding_mask: torch.Tensor


@dataclass
class _LMCache:
    shared: list[tuple[Tensor, Tensor]]  # per-layer (kv_c_cache, k_r_cache)
    lm: list[tuple[Tensor, Tensor]]
    pos: int  # absolute next position (== tokens processed)


@dataclass
class _SegStream:
    lm_cache: _LMCache
    entropies: torch.Tensor  # [B, T] buffer of per-token entropies


class SegmentCompressor(nn.Module):
    """
    End-to-end module that processes a sequence of byte-level tokens.

    Tokens are embedded then passed through a stack of shared
    ``SlidingWindowTransformerBlock`` layers. The output feeds two branches:
    one continues through additional layers to produce logits for the next-token
    prediction objective, while the other goes through a separate set of layers
    to create rich representations used for compression.  Entropy of the logits
    drives segmentation. Each segment is pooled with ``LearnedQueryAttention``
    and the resulting embeddings are vector-quantized.

    The module outputs continuous segment embeddings, discrete codebook indices
    for these embeddings, the VQ loss, and a validity mask for the segments.

    Output dictionary structure:
      {
        'continuous': torch.Tensor (B, Q_total, D)  # Pooled & quantized segment vectors
        'codes'     : torch.Tensor (B, Q_total)     # Integer codebook indices
        'vq_loss'   : torch.Tensor (scalar)         # VQ loss (0 if VQ disabled/eval)
        'valid_mask': torch.Tensor (B, S_hat)       # Boolean mask for valid segments
                                                    # (S_hat is max segments in batch)
      }
    where Q_total = S_hat * num_queries_per_segment.
    """

    def __init__(
            self,
            vocab_size: int = 260,
            dim: int = 256,
            heads: int = 8,  # Num heads for both encoder and pooler
            window: int = 128,  # Window size for shared encoder layers

            # MLA params
            head_dim: int | None = 32,
            kv_comp_dim: int | None = 64,  # d_c
            q_comp_dim: int | None = 96,  # d_c`
            retr_dim: int | None = 32,

            lm_window: int | None = None,
            compression_window: int | None = None,
            num_encoder_layers: int = 3,
            encoder_ffn_dim_multiplier: int = 4,
            num_shared_encoder_layers: int = 0,
            num_lm_encoder_layers: int | None = None,
            num_compression_encoder_layers: int | None = None,
            num_queries: int = 1,  # L: Number of queries per segment for the pooler
            codebook_size: int = 512,  # K: Number of codes in VQ codebook (per stage if vq_depth>1)
            vq_depth: int = 1,  # Number of residual VQ stages; 1 means single-stage VQ
            vq_d_c: int | None = None,  # Compressed dim used inside residual VQ
            beta: float = 0.25,  # Beta for VQ commitment loss
            vq_reset_interval: int = 250,
            entropy_delta: float = 0.0,
            entropy_abs_threshold: float | None = None,
            use_flex_attention: bool = True,
            output_length: int = 512,
            use_gaussian_segmentation: bool = True,
    ):
        super().__init__()
        self.num_queries_per_segment = num_queries
        self.entropy_delta = entropy_delta
        self.entropy_abs_threshold = entropy_abs_threshold
        self.use_flex_attention = use_flex_attention
        self.output_length = output_length if output_length is not None else 512
        self.use_gaussian_segmentation = use_gaussian_segmentation

        if lm_window is None:
            lm_window = window
        if compression_window is None:
            compression_window = window

        self.lm_window = lm_window
        self.compression_window = compression_window

        if num_lm_encoder_layers is None:
            num_lm_encoder_layers = num_encoder_layers
        if num_compression_encoder_layers is None:
            num_compression_encoder_layers = num_encoder_layers

        self.shared_layers = nn.ModuleList(
            [
                # _compiled(
                SlidingWindowMLATransformerBlock(
                    dim=dim,
                    num_heads=heads,
                    window_size=window,
                    head_dim=head_dim,
                    kv_comp_dim=kv_comp_dim,
                    q_comp_dim=q_comp_dim,
                    retr_dim=retr_dim,
                    ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                    use_flex_attention=self.use_flex_attention,
                )
                # )
                for _ in range(num_shared_encoder_layers)
            ]
        )

        self.compression_layers = nn.ModuleList(
            [
                # _compiled(
                SlidingWindowMLATransformerBlock(
                    dim=dim,
                    num_heads=heads,
                    window_size=self.compression_window,
                    head_dim=head_dim,
                    kv_comp_dim=kv_comp_dim,
                    q_comp_dim=q_comp_dim,
                    retr_dim=retr_dim,
                    ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                    use_flex_attention=self.use_flex_attention,
                )
                # )
                for _ in range(num_compression_encoder_layers)
            ]
        )
        if self.use_gaussian_segmentation:
            self.entropy_model = CausalGaussianCore(
                dim=dim,
                n_heads=heads,
                window=self.lm_window,
                head_dim=head_dim,
                kv_comp_dim=kv_comp_dim,
                q_comp_dim=q_comp_dim,
                retr_dim=retr_dim,
                ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                use_flex_attention=self.use_flex_attention,
            )
        else:
            self.entropy_model = CausalLogitCore(
                dim=dim,
                n_heads=heads,
                window=self.lm_window,
                head_dim=head_dim,
                kv_comp_dim=kv_comp_dim,
                q_comp_dim=q_comp_dim,
                retr_dim=retr_dim,
                ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                use_flex_attention=self.use_flex_attention,
            )

        self.lm_final_norm = nn.RMSNorm(dim)
        # self.logit_proj = nn.Linear(dim, vocab_size)

        self.pooler = LearnedQueryAttention(
            embed_dim=dim,
            num_queries_per_segment=num_queries,
            max_queries=output_length // max(1, num_queries),
            num_heads=heads,
            use_flex_attention=True
        )

        if vq_depth is not None:  # and vq_depth >= 2:
            # Use residual multi-stage VQ with a shared compressed space
            d_c = vq_d_c if vq_d_c is not None else (kv_comp_dim if kv_comp_dim is not None else 64)
            self.vq = MultiStageResidualVQ(
                K=codebook_size,
                D=dim,
                depth=int(vq_depth),
                d_c=int(d_c),
                beta=beta,
                reset_interval=vq_reset_interval,
            )
        else:
            K = codebook_size
            self.vq = VectorQuantizer(
                K=codebook_size,
                D=dim,
                beta=beta,
                reset_interval=vq_reset_interval,
                forbidden_ids=[K - 1, K - 2, K - 3, K - 4],
                protected_ids=[K - 1, K - 2, K - 3, K - 4]

            )

    def forward(
            self,
            input_embeddings: torch.Tensor,  # [B, S, D]
            key_padding_mask: torch.Tensor | None = None,
            *,
            token_ids: torch.Tensor | None = None,  # optional, returned in CompressorOutput for losses/metrics
    ) -> CompressorOutput:
        """
        Process precomputed token embeddings to produce compressed segment representations.

        Args:
            input_embeddings (torch.Tensor):
                Input embeddings of shape (batch_size, sequence_length, embed_dim).
            key_padding_mask (Optional[torch.Tensor]):
                Boolean mask aligned with the sequence where True indicates padding.
                Shape: (batch_size, sequence_length). Passed to attention/pooling layers.
            token_ids (Optional[torch.Tensor]):
                Original token IDs of shape (batch_size, sequence_length). Used only
                for bookkeeping/aux losses and returned in the output as
                `input_sequence`. May be None.

        Returns:
            CompressorOutput: Dataclass with fields including:
                - vq_embeddings: Quantized continuous segment embeddings.
                  Shape: (batch_size, S_hat * L, embed_dim), where S_hat is the
                  max number of segments in the batch and L is num_queries_per_segment.
                - vq_indices: Discrete codebook indices for the embeddings.
                  Shape: (batch_size, S_hat * L).
                - vq_loss: Vector quantization loss (scalar tensor).
                - vq_perplexity: Perplexity of the VQ code usage (scalar tensor).
                - valid_mask: Boolean mask for valid pooled queries.
                  Shape: (batch_size, Q_max). When L==1, Q_max==S_hat; when L>1,
                  Q_max==S_hat*L.
                - patch_end_mask: Boundary flags over the input sequence (batch_size, sequence_length).
                - entropy_model_logits: Next-token logits from the LM branch (batch_size, sequence_length, vocab_size).
                - pre_vq_embeddings: Pooled embeddings pre-quantization (batch_size, S_hat * L, embed_dim).
                - seg_id: Per-token segment ids (batch_size, sequence_length).
                - first_byte_idx: First token index per token's segment (batch_size, sequence_length).
                - input_sequence: Echo of `token_ids` if provided, else an empty tensor.
                - input_padding_mask: Echo of the provided key_padding_mask.
        """
        B, S, D = input_embeddings.shape
        # 1. Shared encoder on provided embeddings
        x = input_embeddings
        for layer in self.shared_layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        entropy, loss = self.entropy_model.prediction_loss(x, key_padding_mask=key_padding_mask)

        # 2. Branch for next-token prediction/entropy
        # mu, logvar = self.entropy_model(x)
        # lm_x = x
        # for _idx, layer in enumerate(self.lm_layers):
        #     lm_x = layer(lm_x, key_padding_mask=key_padding_mask)
        #
        # if self.use_gaussian_segmentation:
        #     mu = self.mu(lm_x)
        #     logvar = self.logvar(lm_x)
        # else:
        #     logits = self.logit_proj(self.lm_final_norm(lm_x))

        # 3. Branch for compression representation
        hidden = x
        for layer in self.compression_layers:
            hidden = layer(hidden, key_padding_mask=key_padding_mask)

        # def gaussian_bpd(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        #     """
        #     Per-step bits-per-dim for diagonal Gaussian.
        #     Shapes: (B, L, D) for mu/logvar/target, returns (B, L).
        #     """
        #     const = 0.5 * math.log(2 * math.pi)
        #     nll = 0.5 * ((target - mu) ** 2 * torch.exp(-logvar) + logvar) + const  # (B,L,D) in nats?
        #     nll_bits = nll / math.log(2.0)  # (B,L,D) bits
        #     bpd = nll_bits.mean(dim=-1)  # (B,L)
        #     return bpd
        #
        # def gaussian_entropy_bits(logvar: torch.Tensor) -> torch.Tensor:
        #     """
        #     Differential entropy of diag Gaussian per step (bits).
        #     logvar: (B, L, D) -> (B, L)
        #     H = 0.5 * [ D*log(2πe) + sum_i logvar_i ]  (nats) -> /log 2
        #     """
        #     B, L, D = logvar.shape
        #     const = 0.5 * (D * math.log(2 * math.pi * math.e))
        #     H = const + 0.5 * logvar.sum(dim=-1)  # (B,L) nats
        #     return H / math.log(2.0)  # bits

        # 2. Perform Entropy-Based Segmentation

        with torch.no_grad():
            seg_id, patch_end_mask = entropy_segments(entropy,
                                                      increase_delta=self.entropy_delta,
                                                      abs_threshold=self.entropy_abs_threshold,
                                                      return_boundary=True,
                                                      )

        # else:
        #     # This part determines segment boundaries based on token prediction entropy.
        #     # It's done with no_grad as the segmentation logic itself is not learned here.
        #     # ``token_entropy`` computes entropy from the LM logits and ``entropy_segments``
        #     # converts the resulting sequence of entropies into segment identifiers.
        #     with torch.no_grad():
        #         entropy = token_entropy(logits)  # (B,S)
        #         seg_id, patch_end_mask = entropy_segments(
        #             entropy,
        #             increase_delta=self.entropy_delta,
        #             abs_threshold=self.entropy_abs_threshold,
        #             return_boundary=True,
        #         )
        # seg_id  : (B, S)   int64   0,0,0,1,1,2,2,2,…
        B, S = seg_id.shape
        idx = torch.arange(S, device=seg_id.device).expand(B, -1)  # (B,S)

        # True at the first token of each segment
        is_start = torch.ones_like(seg_id, dtype=torch.bool)
        is_start[:, 1:] = seg_id[:, 1:] != seg_id[:, :-1]

        # keep idx where a segment starts, else 0
        first_pos = torch.where(is_start, idx, torch.zeros_like(idx))

        # propagate the latest non-zero value to the right
        first_byte_idx = torch.cummax(first_pos, dim=1).values  # (B,S)

        # seg_id now maps each token position to a segment index.

        # ── 2b. Build queries & tensor mask ─────────────
        L = self.num_queries_per_segment

        # (1) Pool segment-locally. The pooler builds queries internally.
        pooled_embeddings, _ = self.pooler(
            x=hidden,
            seg_id=seg_id,
            key_padding_mask=key_padding_mask,
            return_attn=False,
        )  # (B, Q_max, D)

        # (2) Build valid mask (B, Q_max) for downstream (next-level KPM & metrics).
        #     Must be computed the same way pooler did it:
        if key_padding_mask is not None:
            seg_id_real = torch.where(key_padding_mask, seg_id.new_full((), -1), seg_id)
            nseg = (seg_id_real.amax(dim=1).clamp(min=-1) + 1)  # (B,)
        else:
            nseg = (seg_id.amax(dim=1) + 1)
        valid_segments_mask = (torch.arange(self.pooler.Q_max, device=hidden.device)[None, :]
                               < (nseg * self.pooler.L)[:, None])  # (B, Q_max)

        # (3) For unused query slots, snap to the VQ pad sentinel to avoid side effects.
        vq_pad_vec = getattr(self.vq, "pad_vector", None)
        pad_vec = (vq_pad_vec.view(1, 1, -1).detach()
                   if vq_pad_vec is not None
                   else self.vq.codebook[self.vq.padding_token_id].detach().view(1, 1, -1))
        pooled_embeddings = torch.where(
            valid_segments_mask.unsqueeze(-1), pooled_embeddings, pad_vec.expand_as(pooled_embeddings)
        )

        # 4. Vector-Quantize Pooled Embeddings
        # Apply vector quantization to the pooled segment embeddings.
        quantised_embeddings, vq_loss, codebook_indices, perplexity = None, None, None, None  # self.vq(pooled_embeddings)

        return CompressorOutput(
            vq_embeddings=quantised_embeddings,  # (B, S_hat*L, D)
            vq_indices=codebook_indices,  # (B, S_hat*L)
            vq_loss=vq_loss,  # Scalar tensor
            vq_perplexity=perplexity,  # Scalar tensor
            valid_mask=valid_segments_mask,  # (B, S)
            patch_end_mask=patch_end_mask,  # (B, S)  # True if this is the last token of a segment
            entropy_model_ent=entropy,  # (B, S)
            entropy_model_loss=loss,  # (B, S)  NLL loss if Gaussian segmentation
            # entropy_model_logits=logits if not self.use_gaussian_segmentation else None,  # (B, S, vocab_size)
            # entropy_model_bpd=bpd if self.use_gaussian_segmentation else None,  # (B, S)
            # entropy_mu=mu if self.use_gaussian_segmentation else None,  # (B, S, D)
            # entropy_logvar=logvar if self.use_gaussian_segmentation else None,  #
            pre_vq_embeddings=pooled_embeddings,  # (B, S_hat*L, D)
            seg_id=seg_id,  # (B, S)  integers 0…
            first_byte_idx=first_byte_idx,  # (B, S)
            input_sequence=(token_ids if token_ids is not None else torch.empty(0, dtype=torch.long, device=x.device)),
            input_padding_mask=key_padding_mask,
        )

    @torch.no_grad()
    def segment_only(
            self,
            input_embeddings: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fast segmentation using only the LM branch.
        Returns (seg_id, patch_end_mask), both (B, S).
        """
        x = input_embeddings
        for layer in self.shared_layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        lm_x = x
        for layer in self.lm_layers:
            lm_x = layer(lm_x, key_padding_mask=key_padding_mask)
        logits = self.logit_proj(self.lm_final_norm(lm_x))

        entropy = token_entropy(logits)
        seg_id, patch_end_mask = entropy_segments(
            entropy,
            increase_delta=self.entropy_delta,
            abs_threshold=self.entropy_abs_threshold,
            return_boundary=True,
        )
        return seg_id, patch_end_mask

    @torch.no_grad()
    def lm_stream_prefill(
            self,
            input_embeddings: torch.Tensor,  # [B, S0, D]
            key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[_LMCache, torch.Tensor]:
        """
        Run embeddings → shared layers → lm layers in cache mode to build caches.
        Returns (cache, logits_for_prefix).
        """
        x = input_embeddings
        # Shared stack
        shared_caches: list[tuple[Tensor, Tensor]] = []
        for blk in self.shared_layers:
            x, cache = blk.prefill(x, pos_start=0, key_padding_mask=key_padding_mask)
            shared_caches.append(cache)
        # LM stack
        lm_caches: list[tuple[Tensor, Tensor]] = []
        for blk in self.lm_layers:
            x, cache = blk.prefill(x, pos_start=0, key_padding_mask=key_padding_mask)
            lm_caches.append(cache)
        # Head
        logits = self.logit_proj(self.lm_final_norm(x))  # [B, S0, V]
        cache = _LMCache(shared=shared_caches, lm=lm_caches, pos=int(x.size(1)))
        return cache, logits

    # ---------- LM streaming: step on new tokens ----------
    @torch.no_grad()
    def lm_stream_step(
            self,
            cache: _LMCache,
            new_input_embeddings: torch.Tensor,  # [B, S_new, D] (S_new=1 typical)
            key_padding_mask_new: torch.Tensor | None = None,
    ) -> tuple[_LMCache, torch.Tensor]:
        """
        Process *only* the new tokens through shared+LM stacks using caches.
        Returns (updated_cache, logits_for_new_tokens).
        """
        x = new_input_embeddings  # [B, S_new, D]
        pos0 = cache.pos

        # Shared stack (new slice only)
        for i, blk in enumerate(self.shared_layers):
            x, cache.shared[i] = blk.step(
                x, cache=cache.shared[i], pos_start=pos0, key_padding_mask_new=key_padding_mask_new
            )

        # LM stack
        for i, blk in enumerate(self.lm_layers):
            x, cache.lm[i] = blk.step(
                x, cache=cache.lm[i], pos_start=pos0, key_padding_mask_new=key_padding_mask_new
            )

        # Head on the new slice
        logits_new = self.logit_proj(self.lm_final_norm(x))  # [B, S_new, V]
        cache.pos += int(x.size(1))
        return cache, logits_new

    @torch.no_grad()
    def segment_stream_init(
            self, input_embeddings: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> _SegStream:
        lm_cache, logits = self.lm_stream_prefill(input_embeddings, key_padding_mask)
        ents = token_entropy(logits)  # [B, S0]
        return _SegStream(lm_cache=lm_cache, entropies=ents)

    @torch.no_grad()
    def segment_stream_step(
            self, stream: _SegStream, new_embeddings: torch.Tensor, key_padding_mask_new: torch.Tensor | None = None
    ) -> tuple[_SegStream, bool]:
        stream.lm_cache, logits_new = self.lm_stream_step(
            stream.lm_cache, new_embeddings, key_padding_mask_new
        )
        ent_new = token_entropy(logits_new)  # [B, S_new]
        stream.entropies = torch.cat([stream.entropies, ent_new], dim=1)

        # recompute segments on the *entropy buffer* only (cheap)
        _, patch_end = entropy_segments(
            stream.entropies,
            increase_delta=self.entropy_delta,
            abs_threshold=self.entropy_abs_threshold,
            return_boundary=True,
        )
        is_boundary = bool(patch_end[0, -1].item())
        return stream, is_boundary

    # segment_compressor.py
    @torch.no_grad()
    def segment_stream_step_block(
            self, stream, new_block_embeddings: torch.Tensor, key_padding_mask_new: torch.Tensor | None = None
    ):
        """
        Advance segmentation by a block of tokens (B, T_blk).
        Returns: (stream, stop_idx) where stop_idx is the earliest index in [0..T_blk-1]
        where a boundary is detected, or None if no boundary inside the block.
        """
        # Run LM on the *block* (one cached call)
        stream.lm_cache, logits_blk = self.lm_stream_step(
            stream.lm_cache, new_block_embeddings, key_padding_mask_new
        )
        ent_blk = token_entropy(logits_blk)  # (B, T_blk)
        old_len = stream.entropies.size(1)
        stream.entropies = torch.cat([stream.entropies, ent_blk], dim=1)

        # Recompute boundary flags on the *entire* entropy buffer (cheap)
        _, patch_end = entropy_segments(
            stream.entropies,
            increase_delta=self.entropy_delta,
            abs_threshold=self.entropy_abs_threshold,
            return_boundary=True,
        )
        block_pe = patch_end[:, old_len: old_len + new_block_embeddings.size(1)]  # (B, T_blk)

        if block_pe.any():
            stop_idx = int(block_pe[0].nonzero(as_tuple=False)[0].item())  # earliest boundary inside block
            return stream, stop_idx
        return stream, None

    @torch.no_grad()
    def enable_mla_fusions_for_inference(self):
        for blk in list(self.shared_layers) + list(self.lm_layers) + list(self.compression_layers):
            # blk.attn is SlidingWindowMLA; blk.attn.mla is MultiheadLatentAttention
            blk.attn.mla.enable_inference_fusions()


class CausalLogitCore(nn.Module):
    def __init__(self,
                 dim: int,
                 n_heads: int = 8,
                 window: int = 128,
                 head_dim: int | None = 32,
                 kv_comp_dim: int | None = 64,  # d_c
                 q_comp_dim: int | None = 96,  # d_c`
                 retr_dim: int | None = 32,
                 ffn_dim_multiplier: int = 4,
                 use_flex_attention: bool = True,
                 n_layers: int = 2,
                 vocab_size: int = 260,
                 ):

        super().__init__()
        self.lm_layers = nn.ModuleList(
            [
                SlidingWindowMLATransformerBlock(
                    dim=dim,
                    num_heads=n_heads,
                    window_size=window,
                    head_dim=head_dim,
                    kv_comp_dim=kv_comp_dim,
                    q_comp_dim=q_comp_dim,
                    retr_dim=retr_dim,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    use_flex_attention=use_flex_attention,
                )
                # )
                for _ in range(n_layers)
            ]
        )
        self.logit_proj = nn.Linear(dim, vocab_size)


    def forward(self, x: torch.Tensor, key_padding_mask) -> torch.Tensor:
        """
        x: (B,S,D) embeddings
        Returns stats for positions 1..S-1 aligned to x[:,1:].
        """
        B, S, D = x.shape
        assert S >= 2, "Need S>=2"
        for _idx, layer in enumerate(self.lm_layers):
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.logit_proj(x)
        return x

    def entropy(self, x: torch.Tensor, key_padding_mask) -> Tuple[Tensor, Tensor]:
        """
        x: (B,S,D) embeddings
        Returns per-step entropy (B,S) in bits.
        """
        B, S, D = x.shape
        assert S >= 2, "Need S>=2"
        logits = self.forward(x, key_padding_mask=key_padding_mask)
        token_entropy(logits)
        return token_entropy(logits), logits

    def prediction_loss(self, x: torch.Tensor, key_padding_mask) -> Tuple[Any, Tensor]:
        entropy, logits = self.entropy = self.entropy(x, key_padding_mask)

        lm_logits = logits[:, :-1, :]  # predict next low token
        lm_target = x[:, 1:]
        lm_kpm = key_padding_mask[:, 1:]
        per_tok = F.cross_entropy(lm_logits.transpose(1, 2), lm_target, reduction="none")
        m = ~lm_kpm
        loss = (per_tok * m).sum() / m.sum().clamp(min=1)

        return entropy, loss



class CausalGaussianCore(nn.Module):
    """
    Causal encoder producing μ, logσ² for next-step prediction over D-dim embeddings.
    """

    def __init__(self,
                 dim: int,
                 n_heads: int = 8,
                 window: int = 128,
                 head_dim: int | None = 32,
                 kv_comp_dim: int | None = 64,  # d_c
                 q_comp_dim: int | None = 96,  # d_c`
                 retr_dim: int | None = 32,
                 ffn_dim_multiplier: int = 4,
                 use_flex_attention: bool = True,
                 n_layers: int = 2,
                 clamp_logvar: Tuple[float, float] = (-8.0, 8.0)
                 ):
        super().__init__()
        self.lm_layers = nn.ModuleList(
            [
                SlidingWindowMLATransformerBlock(
                    dim=dim,
                    num_heads=n_heads,
                    window_size=window,
                    head_dim=head_dim,
                    kv_comp_dim=kv_comp_dim,
                    q_comp_dim=q_comp_dim,
                    retr_dim=retr_dim,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    use_flex_attention=use_flex_attention,
                )
                # )
                for _ in range(n_layers)
            ]
        )

        self.mu = nn.Linear(dim, dim)
        self.logvar = nn.Linear(dim, dim)
        self.clamp = clamp_logvar
        self.dim = dim

    def forward(self, x: torch.Tensor, key_padding_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,S,D) embeddings
        Returns stats for positions 1..S-1 aligned to x[:,1:].
        """
        B, S, D = x.shape
        assert S >= 2, "Need S>=2"
        for _idx, layer in enumerate(self.lm_layers):
            x = layer(x, key_padding_mask=key_padding_mask)
        mu = self.mu(x)
        logvar = self.logvar(x).clamp(*self.clamp)
        return mu, logvar

    def entropy(self, x: torch.Tensor, key_padding_mask) -> torch.Tensor:
        """
        x: (B,S,D) embeddings
        Returns per-step entropy (B,S) in bits.
        """
        B, S, D = x.shape
        assert S >= 2, "Need S>=2"
        tgt = x.detach()
        mu, logvar = self.forward(x, key_padding_mask)
        bpd = self.gaussian_bpd(mu, logvar, tgt)  # (B,S)
        entropy = torch.zeros(B, S, device=x.device)
        entropy[:, 1:] = bpd
        entropy[:, 0] = bpd[:, 0] if bpd.size(1) > 0 else 0.0

        return entropy

    def prediction_loss(self, x: torch.Tensor, key_padding_mask) -> Tuple[Any, Tensor]:
        entropy = self.entropy(x, key_padding_mask)
        return entropy, entropy.mean()

    def gaussian_bpd(self, mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Diagonal-Gaussian BPD per step.
        Inputs (B,L,D) -> returns (B,L) bits-per-dim.
        """
        mu_shift = mu[:, :-1, :]
        logvar_shift = logvar[:, :-1, :]
        target_shift = target[:, 1:, :]
        const = 0.5 * math.log(2 * math.pi)
        nll = 0.5 * (
                (target_shift - mu_shift) ** 2 * torch.exp(
            -logvar_shift) + logvar_shift) + const  # (B,L,D) (nats but const is in nats too)
        bpd = (nll / math.log(2.0)).mean(dim=-1)  # bits-per-dim

        return bpd

    def gaussian_entropy_bits(self, logvar: torch.Tensor) -> torch.Tensor:
        """
        Differential entropy H[N(μ,σ^2)] per step (bits).
        logvar: (B,L,D) -> (B,L)
        """
        B, L, D = logvar.shape
        const = 0.5 * (D * math.log(2 * math.e * math.pi))
        H = const + 0.5 * logvar.sum(dim=-1)  # nats
        return H / math.log(2.0)  # bits
