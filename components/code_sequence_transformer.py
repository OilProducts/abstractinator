import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from .expander import EncoderBlock

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
                 output_lm_logits: bool = True  # If True, adds an LM head to predict next code
                 ):
        super().__init__()
        self.code_vocab_size = code_vocab_size
        self.dim = dim
        self.output_lm_logits = output_lm_logits

        self.token_embedding = nn.Embedding(code_vocab_size, dim)
        self.encoder = nn.ModuleList([
            EncoderBlock(dim, num_heads, dim * ffn_dim_multiplier)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(dim)

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

        x = self.token_embedding(input_codes)  # (B, S, D)
        for layer in self.encoder:
            x = layer(x, key_padding_mask=key_padding_mask)
        hidden_states = self.final_norm(x)

        output_dict = {'hidden_states': hidden_states}

        if self.output_lm_logits:
            logits = self.lm_head(hidden_states)  # (B, S, code_vocab_size)
            output_dict['logits'] = logits

        return output_dict
