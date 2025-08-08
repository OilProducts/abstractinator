from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn

from .learned_query_attention import LearnedQueryAttention
from .mla import SlidingWindowMLATransformerBlock
from .utils import _compiled, build_segment_queries_mask, entropy_segments, token_entropy
from .vector_quantizer import VectorQuantizer


@dataclass
class CompressorOutput:
    vq_embeddings: torch.Tensor
    vq_indices: torch.Tensor
    vq_loss: torch.Tensor
    vq_perplexity: torch.Tensor  # Out of the VQ module
    valid_mask: torch.Tensor
    patch_end_mask: torch.Tensor  # True if this is the last token of a segment
    entropy_model_logits: torch.Tensor  # Logits from the LM branch
    pre_vq_embeddings: torch.Tensor  # Embeddings before VQ
    seg_id: torch.Tensor  # Segment IDs for each token (B, S)
    first_byte_idx: torch.Tensor  # First byte index for each segment (B, S)
    input_sequence: torch.Tensor
    input_padding_mask: torch.Tensor


# If you remove this, or do not use dynamic=True, there may be a segfault related to the caching of the LQA query
@torch.compile(dynamic=True)
class ByteSegmentCompressor(nn.Module):
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
        vocab_size: int = 259,
        dim: int = 256,
        heads: int = 8,  # Num heads for both encoder and pooler
        window: int = 128,  # Window size for shared encoder layers
        head_dim: int | None = 32,
        kv_comp_dim: int | None = 32,
        q_comp_dim: int | None = 48,
        retr_dim: int | None = 32,
        lm_window: int | None = None,
        compression_window: int | None = None,
        num_encoder_layers: int = 3,
        encoder_ffn_dim_multiplier: int = 4,
        num_shared_encoder_layers: int = 0,
        num_lm_encoder_layers: int | None = None,
        num_compression_encoder_layers: int | None = None,
        num_queries: int = 1,  # L: Number of queries per segment for the pooler
        codebook_size: int = 512,  # K: Number of codes in VQ codebook
        beta: float = 0.25,  # Beta for VQ commitment loss
        vq_reset_interval: int = 250,
        entropy_delta: float = 0.2,
        entropy_abs_threshold: float | None = None,
        use_flex_attention: bool = True,
        output_length: int = 512,
    ):
        super().__init__()
        self.num_queries_per_segment = num_queries
        self.entropy_delta = entropy_delta
        self.entropy_abs_threshold = entropy_abs_threshold
        self.use_flex_attention = use_flex_attention
        self.output_length = output_length

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

        self.embedding = nn.Embedding(vocab_size, dim)

        self.shared_layers = nn.ModuleList(
            [
                _compiled(
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
                )
                for _ in range(num_shared_encoder_layers)
            ]
        )

        self.compression_layers = nn.ModuleList(
            [
                _compiled(
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
                )
                for _ in range(num_compression_encoder_layers)
            ]
        )

        self.lm_layers = nn.ModuleList(
            [
                _compiled(
                    SlidingWindowMLATransformerBlock(
                        dim=dim,
                        num_heads=heads,
                        window_size=self.lm_window,
                        head_dim=head_dim,
                        kv_comp_dim=kv_comp_dim,
                        q_comp_dim=q_comp_dim,
                        retr_dim=retr_dim,
                        ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                        use_flex_attention=self.use_flex_attention,
                    )
                )
                for _ in range(num_lm_encoder_layers)
            ]
        )

        self.lm_final_norm = nn.RMSNorm(dim)
        self.logit_proj = nn.Linear(dim, vocab_size)

        self.pooler = LearnedQueryAttention(
            embed_dim=dim,
            num_queries_per_segment=num_queries,
            num_heads=heads,
        )

        self.vq = VectorQuantizer(
            K=codebook_size,
            D=dim,
            beta=beta,
            reset_interval=vq_reset_interval,
        )

    def forward(self, token_ids: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> CompressorOutput:
        """
        Processes input token IDs to produce compressed segment representations.

        Args:
            token_ids (torch.Tensor): Input token IDs.
                Shape: (batch_size, sequence_length). Expected dtype: int16/int32/int64.
            key_padding_mask (Optional[torch.Tensor]): Boolean mask for `token_ids`
                where `True` indicates a padded token.
                Shape: (batch_size, sequence_length).
                Note: This mask is passed to the pooler and the attention layers.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'continuous': Quantized continuous segment embeddings.
                  Shape: (batch_size, S_hat * L, embed_dim), where S_hat is the
                  max number of segments in the batch, and L is num_queries_per_segment.
                - 'codes': Discrete codebook indices for the embeddings.
                  Shape: (batch_size, S_hat * L).
                - 'vq_loss': The vector quantization loss (scalar tensor).
                - 'valid_mask': A boolean mask indicating valid segments among the S_hat
                  potential segment slots. Shape: (batch_size, S_hat). To get a mask
                  for valid query vectors, this would need to be expanded/repeated.
        """
        # 1. Token embeddings and shared encoder
        x = self.embedding(token_ids)
        for layer in self.shared_layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        # 2. Branch for next-token prediction
        lm_x = x
        for layer in self.lm_layers:
            lm_x = layer(lm_x, key_padding_mask=key_padding_mask)
        logits = self.logit_proj(self.lm_final_norm(lm_x))

        # 3. Branch for compression representation
        hidden = x
        for layer in self.compression_layers:
            hidden = layer(hidden, key_padding_mask=key_padding_mask)

        # 2. Perform Entropy-Based Segmentation
        # This part determines segment boundaries based on token prediction entropy.
        # It's done with no_grad as the segmentation logic itself is not learned here.
        # ``token_entropy`` computes entropy from the LM logits and ``entropy_segments``
        # converts the resulting sequence of entropies into segment identifiers.
        with torch.no_grad():
            entropy = token_entropy(logits)  # (B,S)
            seg_id, patch_end_mask = entropy_segments(
                entropy,
                increase_delta=self.entropy_delta,
                abs_threshold=self.entropy_abs_threshold,
                return_boundary=True,
            )
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
        queries, seg_attn_mask, valid_segments_mask = build_segment_queries_mask(
            seg_id,
            self.pooler.query_template,
            self.pooler.num_heads,
        )
        # queries: (B, S_hat*L, D) - Tiled learned queries for each potential segment.
        # seg_attn_mask: (B*H, S_hat*L, S_original) - Masks attention outside segments.
        # valid_segments_mask: (B, S_hat) - Indicates which of S_hat segments are real.

        # ── 3. Learned-Query Pooling (Segment-Restricted) ───────────────────
        # Pool features from `hidden` states using the constructed `queries`.
        # Attention is restricted by `seg_attn_mask` and `key_padding_mask`.
        pooled_embeddings, _ = self.pooler(
            x=hidden,  # Keys and Values from encoder output (B,S,D)
            queries=queries,  # Segment-specific queries (B, S_hat*L, D)
            attn_mask=seg_attn_mask,  # Restricts attention within segments
            key_padding_mask=key_padding_mask,  # Masks padded tokens in `hidden`
        )  # pooled_embeddings: (B, S_hat*L, D)

        # ── 4. Vector-Quantize Pooled Embeddings ─────────────────────────────
        # Apply vector quantization to the pooled segment embeddings.
        quantised_embeddings, vq_loss, codebook_indices, perplexity = self.vq(pooled_embeddings)

        # Pad outputs to output_length
        pad_id = self.vq.padding_token_id
        if quantised_embeddings.size(1) < self.output_length:
            pad_size = self.output_length - quantised_embeddings.size(1)
            padding = torch.full(
                (quantised_embeddings.size(0), pad_size, quantised_embeddings.size(2)),
                pad_id,
                device=quantised_embeddings.device,
                dtype=quantised_embeddings.dtype,
            )
            quantised_embeddings = torch.cat((quantised_embeddings, padding), dim=1)
            codebook_indices = torch.cat(
                (
                    codebook_indices,
                    torch.full((codebook_indices.size(0), pad_size), pad_id, device=codebook_indices.device),
                ),
                dim=1,
            )
            valid_segments_mask = torch.cat(
                (
                    valid_segments_mask,
                    torch.zeros(
                        (valid_segments_mask.size(0), pad_size), dtype=torch.bool, device=valid_segments_mask.device
                    ),
                ),
                dim=1,
            )

        return CompressorOutput(
            vq_embeddings=quantised_embeddings,  # (B, S_hat*L, D)
            vq_indices=codebook_indices,  # (B, S_hat*L)
            vq_loss=vq_loss,  # Scalar tensor
            vq_perplexity=perplexity,  # Scalar tensor
            valid_mask=valid_segments_mask,  # (B, S)
            patch_end_mask=patch_end_mask,  # (B, S)  # True if this is the last token of a segment
            entropy_model_logits=logits,  # (B, S, vocab_size)
            pre_vq_embeddings=pooled_embeddings,  # (B, S_hat*L, D)
            seg_id=seg_id,  # (B, S)  integers 0…
            first_byte_idx=first_byte_idx,  # (B, S)
            input_sequence=token_ids,
            input_padding_mask=key_padding_mask,
        )
