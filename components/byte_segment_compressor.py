from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

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
    entropy_model_logits: torch.Tensor  # Logits from the LM branch
    pre_vq_embeddings: torch.Tensor  # Embeddings before VQ
    seg_id: torch.Tensor  # Segment IDs for each token (B, S)
    first_byte_idx: torch.Tensor  # First byte index for each segment (B, S)
    input_sequence: torch.Tensor
    input_padding_mask: torch.Tensor


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
        vocab_size: int = 260,
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
        codebook_size: int = 512,  # K: Number of codes in VQ codebook (per stage if vq_depth>1)
        vq_depth: int = 1,  # Number of residual VQ stages; 1 means single-stage VQ
        vq_d_c: int | None = None,  # Compressed dim used inside residual VQ
        beta: float = 0.25,  # Beta for VQ commitment loss
        vq_reset_interval: int = 250,
        entropy_delta: float = 0.0,
        entropy_abs_threshold: float | None = None,
        use_flex_attention: bool = True,
        output_length: int = 512,
    ):
        super().__init__()
        self.num_queries_per_segment = num_queries
        self.entropy_delta = entropy_delta
        self.entropy_abs_threshold = entropy_abs_threshold
        self.use_flex_attention = use_flex_attention
        self.output_length = output_length if output_length is not None else 512

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

        self.lm_layers = nn.ModuleList(
            [
                # _compiled(
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
                # )
                for _ in range(num_lm_encoder_layers)
            ]
        )

        self.lm_final_norm = nn.RMSNorm(dim)
        self.logit_proj = nn.Linear(dim, vocab_size)

        self.pooler = LearnedQueryAttention(
            embed_dim=dim,
            num_queries_per_segment=num_queries,
            num_heads=heads,
            use_flex_attention=False
        )

        if vq_depth is not None: # and vq_depth >= 2:
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
            self.vq = VectorQuantizer(
                K=codebook_size,
                D=dim,
                beta=beta,
                reset_interval=vq_reset_interval,
            )
    # @torch.compile(dynamic=True)
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
        if not torch._dynamo.is_compiling():
            V = self.embedding.num_embeddings
            # Cheap, friendly check—remove once stable
            if token_ids.numel():
                tmin = int(token_ids.min())
                tmax = int(token_ids.max())
                assert 0 <= tmin and tmax < V, f"OOB token id(s): [{tmin}..{tmax}] vs V={V}"

        x = self.embedding(token_ids)
        # if torch.isnan(x).any():
        #     print('nan')
        for layer in self.shared_layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        # 2. Branch for next-token prediction
        lm_x = x
        for _idx, layer in enumerate(self.lm_layers):
            # if torch.isnan(lm_x).any():
            #     print('nan')
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
        L = self.num_queries_per_segment
        Q_max = (int(self.output_length) // max(1, L)) * max(1, L)  # Ensure output length multiple of L
        queries, q_seg, valid_segments_mask = build_segment_queries_qseg(
            seg_id,
            self.pooler.query_template,
            Q_max=Q_max,
        )


        # ── 3. Learned-Query Pooling (Segment-Restricted) ───────────────────
        # Pool features from `hidden` states using the constructed `queries`.
        # Attention is restricted by `seg_attn_mask` and `key_padding_mask`.
        pooled_embeddings, _ = self.pooler(
            x=hidden,  # (B,S,D)
            queries=queries,  # (B,Q_max,D)
            q_seg=q_seg,  # (B,Q_max) int; -1 for padded queries
            seg_id=seg_id,  # (B,S)     int
            key_padding_mask=key_padding_mask,  # (B,S) bool
            attn_mask=None,  # ignored on flex path
            return_attn=False,
        )

        # Zero-out invalid queries by snapping them to the padding code vector.
        # Queries that are not assigned to any real segment (q_seg < 0) are invalid.
        q_valid = (q_seg >= 0)  # (B, Q_max) bool

        # Choose the pad vector once
        vq_pad_vec = getattr(self.vq, "pad_vector", None)
        if vq_pad_vec is not None:
            pad_vec = vq_pad_vec.view(1, 1, -1).detach()
        else:
            pad_vec = self.vq.codebook[self.vq.padding_token_id].detach().view(1, 1, -1)

        pooled_embeddings = torch.where(
            q_valid.unsqueeze(-1), pooled_embeddings, pad_vec.expand_as(pooled_embeddings)
        )

        # ── 4. Vector-Quantize Pooled Embeddings ─────────────────────────────
        # Apply vector quantization to the pooled segment embeddings.
        quantised_embeddings, vq_loss, codebook_indices, perplexity = self.vq(pooled_embeddings)

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
