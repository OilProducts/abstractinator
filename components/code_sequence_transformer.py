import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

@torch.compile
class CodeSequenceTransformer(nn.Module):
    """
    A standard Transformer encoder stack to process sequences of codes.
    Can be used for language modeling on codes or extracting contextual representations.
    """

    def __init__(self,
                 code_vocab_size: int,  # Vocabulary size of the input codes
                 dim: int,  # Embedding dimension and model dimension
                 num_layers: int,  # Number of Transformer encoder layers
                 num_heads: int,  # Number of attention heads
                 ffn_dim_multiplier: int = 4,  # Multiplier for FFN hidden dimension
                 dropout: float = 0.1,
                 max_seq_len: int = 2048,  # Max sequence length for positional encoding
                 output_lm_logits: bool = True  # If True, adds an LM head to predict next code
                 ):
        super().__init__()
        self.code_vocab_size = code_vocab_size
        self.dim = dim
        self.output_lm_logits = output_lm_logits

        self.token_embedding = nn.Embedding(code_vocab_size, dim)
        # Using nn.Parameter for learned positional encoding, could also use sinusoidal
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * ffn_dim_multiplier,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Using Pre-LN for potentially better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(dim)  # Final norm after all layers
        )

        if self.output_lm_logits:
            self.lm_head = nn.Linear(dim, code_vocab_size)  # Projects to code vocabulary

    def forward(self,
                input_codes: torch.Tensor,  # Shape: (batch_size, sequence_length)
                key_padding_mask: Optional[torch.Tensor] = None  # Shape: (batch_size, sequence_length)
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_codes: A batch of code sequences.
            key_padding_mask: Mask for padded positions in input_codes (True if padded).

        Returns:
            A dictionary containing:
            - 'hidden_states': Output hidden states from the Transformer.
                               Shape: (batch_size, sequence_length, dim).
            - 'logits': Output logits for next code prediction (if output_lm_logits=True).
                        Shape: (batch_size, sequence_length, code_vocab_size).
        """
        B, S = input_codes.shape
        if S > self.positional_encoding.size(1):
            raise ValueError(
                f"Input sequence length ({S}) exceeds CodeSequenceTransformer's max_seq_len "
                f"for positional encoding ({self.positional_encoding.size(1)})."
            )

        x = self.token_embedding(input_codes)  # (B, S, D)
        x = x + self.positional_encoding[:, :S, :]  # Add positional encoding
        x = self.embed_dropout(x)

        # `key_padding_mask` should be True where tokens are padding.
        # `nn.TransformerEncoder` expects src_key_padding_mask.
        hidden_states = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        output_dict = {'hidden_states': hidden_states}

        if self.output_lm_logits:
            logits = self.lm_head(hidden_states)  # (B, S, code_vocab_size)
            output_dict['logits'] = logits

        return output_dict
