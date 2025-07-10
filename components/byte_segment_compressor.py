from typing import Dict, Optional # Already imported if combined

import torch
import torch.nn as nn

from .learned_query_attention import LearnedQueryAttention
from .vector_quantizer import VectorQuantizer
from .sliding_window_attention import StackedSlidingWindowEncoder
from .utils import token_entropy, entropy_segments, build_segment_queries_mask

# @torch.compile
class ByteSegmentCompressor(nn.Module):
    """
    End-to-end module that processes a sequence of byte-level tokens.

    It first encodes the tokens using a `StackedSlidingWindowEncoder`. Then, it
    segments the encoded sequence based on token entropy. For each segment,
    a fixed number (`num_queries`) of learned query vectors are used to pool
    features from the segment via `LearnedQueryAttention`. Finally, these
    pooled segment embeddings are vector-quantized.

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
    def __init__(self,
                 vocab_size: int = 259,
                 dim: int = 256,
                 heads: int = 8, # Num heads for both encoder and pooler
                 window: int = 128, # Window size for encoder
                 num_encoder_layers: int = 3,
                 encoder_ffn_dim_multiplier: int = 4,
                 num_queries: int = 1, # L: Number of queries per segment for the pooler
                 codebook_size: int = 512,    # K: Number of codes in VQ codebook
                 beta: float = 0.25,          # Beta for VQ commitment loss
                 entropy_delta: float = 0.2,
                 entropy_abs_threshold: float | None = None):
        super().__init__()
        self.num_queries_per_segment = num_queries # Store L for convenience
        self.entropy_delta = entropy_delta
        self.entropy_abs_threshold = entropy_abs_threshold

        # Initialize the token encoder
        # Use FlexAttention only when CUDA is available
        self.encoder = StackedSlidingWindowEncoder(
            vocab_size=vocab_size,
            dim=dim,
            num_heads=heads,
            window_size=window,
            num_layers=num_encoder_layers,
            ffn_dim_multiplier=encoder_ffn_dim_multiplier,
            use_flex_attention=torch.cuda.is_available()
        )

        # Initialize the attention pooler that uses learned queries per segment
        self.pooler = LearnedQueryAttention(
            embed_dim=dim,
            num_queries_per_segment=num_queries,
            num_heads=heads
        )

        # Initialize the Vector Quantizer
        self.vq = VectorQuantizer(
            K=codebook_size,
            D=dim,
            beta=beta
        )

    def forward(self, token_ids: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None
               ) -> Dict[str, torch.Tensor]:
        """
        Processes input token IDs to produce compressed segment representations.

        Args:
            token_ids (torch.Tensor): Input token IDs.
                Shape: (batch_size, sequence_length). Expected dtype: int16/int32/int64.
            key_padding_mask (Optional[torch.Tensor]): Boolean mask for `token_ids`
                where `True` indicates a padded token.
                Shape: (batch_size, sequence_length).
                Note: This mask is passed to the pooler. Its usage by the encoder
                depends on the StackedSlidingWindowEncoder implementation.

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
        # 1. Encode Tokens
        # `hidden` are token embeddings, `logits` for prediction (e.g., for entropy calc)
        # Note: key_padding_mask is not explicitly passed to self.encoder here.
        # If StackedSlidingWindowEncoder supports it, it should be passed.
        hidden, logits = self.encoder(token_ids) # hidden: (B,S,D), logits: (B,S,Vocab)
        #TODO should encoder take key_padding_mask? does it need safe_softmax then?

        # 2. Perform Entropy-Based Segmentation
        # This part determines segment boundaries based on token prediction entropy.
        # It's done with no_grad as the segmentation logic itself is not learned here.
        with torch.no_grad():
            # token_entropy and entropy_segments are assumed to be defined elsewhere
            entropy = token_entropy(logits)  # (B,S)
            seg_id = entropy_segments(
                entropy,
                increase_delta=self.entropy_delta,
                abs_threshold=self.entropy_abs_threshold,
            )
        # seg_id now maps each token position to a segment index.
        patch_end_mask = torch.zeros_like(seg_id, dtype=torch.bool)
        if seg_id.size(1) > 0:
            patch_end_mask[:, :-1] = seg_id[:, 1:] != seg_id[:, :-1]
            patch_end_mask[:, -1] = True

        # Build queries and attention mask for segment-restricted pooling.
        # `self.pooler.query_template` are the L learned base queries (L,D).
        # `self.pooler.num_heads` is used for repeating the attention mask.
        queries, seg_attn_mask, valid_segments_mask = build_segment_queries_mask(
            seg_id,
            self.pooler.query_template,
            self.pooler.num_heads
        )
        # queries: (B, S_hat*L, D) - Tiled learned queries for each potential segment.
        # seg_attn_mask: (B*H, S_hat*L, S_original) - Masks attention outside segments.
        # valid_segments_mask: (B, S_hat) - Indicates which of S_hat segments are real.

        # ── 3. Learned-Query Pooling (Segment-Restricted) ───────────────────
        # Pool features from `hidden` states using the constructed `queries`.
        # Attention is restricted by `seg_attn_mask` and `key_padding_mask`.
        pooled_embeddings, _ = self.pooler(
            x=hidden,                         # Keys and Values from encoder output (B,S,D)
            queries=queries,                  # Segment-specific queries (B, S_hat*L, D)
            attn_mask=seg_attn_mask,          # Restricts attention within segments
            key_padding_mask=key_padding_mask # Masks padded tokens in `hidden`
        ) # pooled_embeddings: (B, S_hat*L, D)

        # ── 4. Vector-Quantize Pooled Embeddings ─────────────────────────────
        # Apply vector quantization to the pooled segment embeddings.
        quantised_embeddings, vq_loss, codebook_indices, perplexity = self.vq(pooled_embeddings)
        # quantised_embeddings: (B, S_hat*L, D)
        # vq_loss: scalar
        # codebook_indices: (B, S_hat*L)
        # Calculate codebook perplexity
        codebook_perplexity = torch.tensor(1.0, device=token_ids.device)  # Default for empty/no codes
        if codebook_indices.numel() > 0:
            flat_indices = codebook_indices.reshape(-1)
            # Ensure K is correctly accessed from your VQ instance
            # Assuming self.vq.K holds the codebook size
            num_codebook_vectors = self.vq.K

            # Clamp indices to be safe, though they should be in range [0, K-1]
            clamped_indices = torch.clamp(flat_indices, 0, num_codebook_vectors - 1)

            counts = torch.bincount(clamped_indices.long(), minlength=num_codebook_vectors)

            # Probabilities of using each code
            p_codes = counts.float() / flat_indices.numel()
            # Filter out p=0 for entropy calculation to avoid log(0)
            p_codes_nz = p_codes[p_codes > 0]

            if p_codes_nz.numel() > 0:  # If any codes were actually used
                entropy_val = -torch.sum(p_codes_nz * torch.log(p_codes_nz))  # Natural log for nats
                codebook_perplexity = torch.exp(entropy_val)  # Perplexity = e^H
            # else: perplexity remains 1.0 (e.g., if flat_indices was empty or all codes had 0 prob somehow)
        elif pooled_embeddings.numel() > 0 and pooled_embeddings.size(
                1) == 0:  # Pooled embeddings exist but have 0 query dim
            codebook_perplexity = torch.tensor(1.0, device=token_ids.device)  # No codes to choose from

        # Update the return dictionary
        return_dict = {
            'continuous': quantised_embeddings,
            'codes': codebook_indices,
            'vq_loss': vq_loss,
            'valid_mask': valid_segments_mask,  # (B, S_hat_segments)
            'patch_end_mask': patch_end_mask,
            'encoder_logits': logits,
            'current_codebook_perplexity': codebook_perplexity,
            'smoothed_codebook_perplexity': perplexity,
            'pre_vq_embeddings': pooled_embeddings,
        }
        return return_dict
