import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from .expander import EncoderBlock

# Disable torch.compile on this module to keep unit tests lightweight
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

    @torch.no_grad()
    def generate_codes(
        self,
        prefix: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        sample: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate a single next code given a prefix.

        Args:
            prefix: Previously generated codes ``(B, S)``.
            key_padding_mask: Optional mask for ``prefix``.
            sample: If ``True`` sample from the distribution instead of greedy
                argmax.
            temperature: Sampling temperature when ``sample`` is ``True``.

        Returns:
            Tensor of shape ``(B, 1)`` containing the next predicted code.
        """
        out = self.forward(prefix, key_padding_mask=key_padding_mask)
        if "logits" not in out:
            raise RuntimeError("CodeSequenceTransformer was initialized without lm logits")
        logits = out["logits"][:, -1, :]  # (B, V)

        if sample:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_codes = torch.multinomial(probs, num_samples=1)
        else:
            next_codes = logits.argmax(dim=-1, keepdim=True)
        return next_codes
