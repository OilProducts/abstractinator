from typing import Dict, Optional

import torch
import torch.nn as nn

from .learned_query_attention import LearnedQueryAttention
from .vector_quantizer import VectorQuantizer
from .sliding_window_attention import SlidingWindowTransformerBlock
from .utils import token_entropy, entropy_segments, build_segment_queries_mask

# @torch.compile
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
        lm_window: Optional[int] = None,
        compression_window: Optional[int] = None,
        num_encoder_layers: int = 3,
        encoder_ffn_dim_multiplier: int = 4,
        num_shared_encoder_layers: int = 0,
        num_lm_encoder_layers: Optional[int] = None,
        num_compression_encoder_layers: Optional[int] = None,
        num_queries: int = 1,  # L: Number of queries per segment for the pooler
        codebook_size: int = 512,  # K: Number of codes in VQ codebook
        beta: float = 0.25,  # Beta for VQ commitment loss
        vq_reset_interval: int = 250,
        entropy_delta: float = 0.2,
        entropy_abs_threshold: float | None = None,
    ):
        super().__init__()
        self.num_queries_per_segment = num_queries
        self.entropy_delta = entropy_delta
        self.entropy_abs_threshold = entropy_abs_threshold

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
                SlidingWindowTransformerBlock(
                    dim=dim,
                    num_heads=heads,
                    window_size=window,
                    ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                    use_flex_attention=torch.cuda.is_available(),
                )
                for _ in range(num_shared_encoder_layers)
            ]
        )

        self.compression_layers = nn.ModuleList(
            [
                SlidingWindowTransformerBlock(
                    dim=dim,
                    num_heads=heads,
                    window_size=self.compression_window,
                    ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                    use_flex_attention=torch.cuda.is_available(),
                )
                for _ in range(num_compression_encoder_layers)
            ]
        )

        self.lm_layers = nn.ModuleList(
            [
                SlidingWindowTransformerBlock(
                    dim=dim,
                    num_heads=heads,
                    window_size=self.lm_window,
                    ffn_dim_multiplier=encoder_ffn_dim_multiplier,
                    use_flex_attention=torch.cuda.is_available(),
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
