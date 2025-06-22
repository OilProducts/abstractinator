import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

@torch.compile
class CodeExpander(nn.Module):
    """
    A Transformer-based sequence-to-sequence model that converts a sequence of
    high-level codes (from vocabulary K_hi, length L_hi) into a longer sequence
    of low-level codes (from vocabulary K_lo, length L_lo).

    The model consists of an encoder processing the high-level codes and a
    decoder generating the low-level codes.

    Training is performed using teacher forcing with (codes_hi, codes_lo) pairs.
    Inference is done autoregressively, generating low-level codes one by one
    until an End-of-Sequence (EOS) token is produced or a maximum length is reached.

    Attributes:
        K_hi (int): Vocabulary size of the high-level codes (encoder input).
        K_lo (int): Vocabulary size of the low-level codes (decoder output).
        D (int): The internal dimension of the model (embedding size, Transformer hidden size).
        eos_id (int): The token ID used for End-of-Sequence. It's also used as the
                      Start-of-Sequence (SOS/BOS) token for the decoder input during training
                      and inference.
        max_len (int): The maximum length of the low-level sequence to generate during inference.
        emb_hi (nn.Embedding): Embedding layer for high-level codes.
        emb_lo (nn.Embedding): Embedding layer for low-level codes.
        pos_enc (nn.Parameter): Learned absolute positional encoding parameters.
        encoder (nn.TransformerEncoder): The Transformer encoder stack.
        decoder (nn.TransformerDecoder): The Transformer decoder stack.
        out_proj (nn.Linear): Linear layer to project decoder output to low-level vocab logits.

    Note on Padding:
        This implementation does not explicitly handle padding masks for the Transformer
        encoder or decoder (i.e., `src_key_padding_mask`, `memory_key_padding_mask`,
        `tgt_key_padding_mask`). If input sequences `codes_hi` or `codes_lo` (during
        training) can contain padding, this should be added for optimal performance,
        to prevent attention from being paid to padded positions.
    """

    def __init__(self,
                 K_hi: int,
                 K_lo: int,
                 D: int = 256,         # Model dimension
                 N_enc: int = 4,       # Number of encoder layers
                 N_dec: int = 4,       # Number of decoder layers
                 H: int = 8,           # Number of attention heads
                 dropout: float = 0.1,
                 eos_id: int = 1,      # EOS token ID. Also used as BOS.
                                       # (Reserve 0 for PAD if planning to add padding handling)
                 max_len: int = 2048): # Max generation length for low-level codes
        super().__init__()
        self.K_hi, self.K_lo = K_hi, K_lo
        self.D, self.eos_id, self.max_len = D, eos_id, max_len

        # --- Embeddings & Positional Encodings ---
        self.emb_hi = nn.Embedding(K_hi, D)
        self.emb_lo = nn.Embedding(K_lo, D)
        # Learned absolute positional encodings, up to max_len
        self.pos_enc = nn.Parameter(torch.randn(max_len, D) * 0.02) # Small init

        # --- Transformer Encoder & Decoder Stacks ---
        # Standard Transformer layers with batch_first=True.
        # These use Post-LayerNorm by default (norm_first=False).
        # Feed-forward network dimension is set to 4*D.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=H, dim_feedforward=4*D, dropout=dropout, batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D, nhead=H, dim_feedforward=4*D, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_enc)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=N_dec)

        # --- Output Projection to Low-Level Vocabulary ---
        self.out_proj = nn.Linear(D, K_lo)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """
        Generates a causal (look-ahead) mask for the decoder.

        Args:
            length (int): The length of the sequence for which the mask is generated.
            device (torch.device): The device to create the mask on.

        Returns:
            torch.Tensor: A boolean tensor of shape (length, length) where `True`
                          indicates positions that should be masked (preventing
                          attention to future tokens).
        """
        # Creates an upper triangular matrix of True, excluding the diagonal.
        # True values indicate positions that should be masked.
        mask = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self,
                codes_hi: torch.Tensor,
                codes_lo: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training using teacher forcing.

        Args:
            codes_hi (torch.Tensor): Batch of high-level code sequences.
                                     Shape: (batch_size, sequence_length_hi).
            codes_lo (torch.Tensor): Batch of ground truth low-level code sequences.
                                     Shape: (batch_size, sequence_length_lo).
            src_key_padding_mask (Optional[torch.Tensor]): Boolean mask for `codes_hi`.
                                                           `True` indicates padding.
                                                           Shape: (batch_size, sequence_length_hi).
            tgt_key_padding_mask (Optional[torch.Tensor]): Boolean mask for `codes_lo` (target sequence).
                                                           `True` indicates padding.
                                                           Shape: (batch_size, sequence_length_lo).
        Returns:
            Dict[str, torch.Tensor]: {'logits': (B, L_lo, K_lo)}
        """
        B, L_hi = codes_hi.shape
        _, L_lo = codes_lo.shape
        device = codes_hi.device

        # --- Encoder Pass ---
        embedded_hi = self.emb_hi(codes_hi) + self.pos_enc[:L_hi, :]
        encoder_output = self.encoder(
            embedded_hi,
            src_key_padding_mask=src_key_padding_mask
        )  # (B, L_hi, D)

        # --- Decoder Pass (Teacher Forcing) ---
        decoder_input_ids = F.pad(codes_lo[:, :-1], (1, 0), mode='constant', value=self.eos_id)
        embedded_lo_input = self.emb_lo(decoder_input_ids) + self.pos_enc[:L_lo, :]

        target_causal_mask = self._causal_mask(L_lo, device)

        # Note: tgt_key_padding_mask for F.pad'd decoder_input_ids would be shifted too.
        # If tgt_key_padding_mask is for original codes_lo, then for decoder_input_ids,
        # it should also be padded at the beginning (e.g. with False for the BOS token)
        # and truncated at the end.
        # For simplicity, if tgt_key_padding_mask refers to the *shifted* input, use it directly.
        # If it refers to the original codes_lo, careful adjustment is needed.
        # PyTorch TransformerDecoderLayer uses tgt_key_padding_mask for the 'tgt' input.
        adjusted_tgt_kpm = None
        if tgt_key_padding_mask is not None:
            # Assuming tgt_key_padding_mask corresponds to the original codes_lo.
            # We need a mask for `decoder_input_ids` which is `codes_lo` shifted right with BOS.
            # So, pad `tgt_key_padding_mask` for `codes_lo[:, :-1]` on the left with False (for BOS).
            adjusted_tgt_kpm = F.pad(tgt_key_padding_mask[:, :-1], (1, 0), mode='constant',
                                     value=False)

        decoder_output = self.decoder(
            tgt=embedded_lo_input,
            memory=encoder_output,
            tgt_mask=target_causal_mask,
            tgt_key_padding_mask=adjusted_tgt_kpm,  # KPM for the decoder input tokens
            memory_key_padding_mask=src_key_padding_mask  # KPM for the encoder output (memory)
        )  # (B, L_lo, D)

        logits = self.out_proj(decoder_output)
        return {'logits': logits}

    @torch.no_grad()
    def generate(self,
                 codes_hi: torch.Tensor,
                 src_key_padding_mask: Optional[torch.Tensor] = None,
                 max_len: Optional[int] = None
                 ) -> torch.Tensor:
        """
        Generates low-level code sequences autoregressively.

        Args:
            codes_hi (torch.Tensor): Batch of high-level code sequences.
                                     Shape: (batch_size, sequence_length_hi).
            src_key_padding_mask (Optional[torch.Tensor]): Boolean mask for `codes_hi`.
                                                           `True` indicates padding.
                                                           Shape: (batch_size, sequence_length_hi).
            max_len (Optional[int]): Maximum generation length. Uses `self.max_len` if None.

        Returns:
            torch.Tensor: Generated low-level code sequences, (B, <=max_len).
                          Initial BOS token is removed.
        """
        B, L_hi = codes_hi.shape
        device = codes_hi.device
        current_max_len = max_len if max_len is not None else self.max_len

        # --- Encoder Pass ---
        embedded_hi = self.emb_hi(codes_hi) + self.pos_enc[:L_hi, :]
        encoder_output = self.encoder(
            embedded_hi,
            src_key_padding_mask=src_key_padding_mask
        )  # (B, L_hi, D)

        # --- Autoregressive Decoding ---
        generated_ids = torch.full((B, 1), self.eos_id, device=device,
                                   dtype=torch.long)  # Start with BOS

        for _ in range(current_max_len - 1):  # Max output len, accounting for initial BOS
            current_seq_len = generated_ids.size(1)
            if current_seq_len >= current_max_len:  # Safety break if already at max_len
                break

            embedded_lo_input = self.emb_lo(generated_ids) + self.pos_enc[:current_seq_len, :]
            target_causal_mask = self._causal_mask(current_seq_len, device)

            # Decoder does not use tgt_key_padding_mask during generation as generated sequence is not padded.
            # memory_key_padding_mask comes from the encoder input.
            decoder_output = self.decoder(
                tgt=embedded_lo_input,
                memory=encoder_output,
                tgt_mask=target_causal_mask,
                memory_key_padding_mask=src_key_padding_mask
            )  # (B, current_seq_len, D)

            next_token_logits = self.out_proj(decoder_output[:, -1, :])  # (B, K_lo)
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            if (next_token_id == self.eos_id).all():  # Stop if all sequences in batch output EOS
                break

        return generated_ids[:, 1:]  # Remove initial BOS token
