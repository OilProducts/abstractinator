import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryCache, apply_rope
from .swiglu import SwiGLU
from .sliding_window_attention import SlidingWindowCrossAttention


# ---------------------------------------------------------------------------
# Multi-Head Attention with RoPE
# ---------------------------------------------------------------------------
class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, causal: bool = False):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key

        B, L_q, _ = query.shape
        L_k = key.size(1)
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, L_q, H, d).transpose(1, 2)  # (B,H,Lq,d)
        k = self.k_proj(key).view(B, L_k, H, d).transpose(1, 2)
        v = self.v_proj(value).view(B, L_k, H, d).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                scores = scores.masked_fill(attn_mask.bool()[None, None, :, :], float('-inf'))
            else:
                scores = scores.masked_fill(attn_mask.bool(), float('-inf'))
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].bool()
            scores = scores.masked_fill(mask, float('-inf'))
        if self.causal:
            causal_mask = torch.triu(torch.ones(L_q, L_k, device=query.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B,H,Lq,d)
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Transformer Blocks
# ---------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, causal: bool = False):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = MultiHeadAttentionRoPE(d_model, num_heads, causal=causal)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionRoPE(d_model, num_heads, causal=True)
        self.norm2 = nn.RMSNorm(d_model)
        self.cross_attn = MultiHeadAttentionRoPE(d_model, num_heads)
        self.norm3 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = x + self.cross_attn(self.norm2(x), key=memory, value=memory, key_padding_mask=memory_key_padding_mask)
        x = x + self.ffn(self.norm3(x))
        return x


class SimpleEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, 4 * d_model) for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.final_norm(x)


class SimpleDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, 4 * d_model) for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.final_norm(x)


class SlidingDecoderBlock(nn.Module):
    """Decoder block using causal self-attention and sliding-window cross attention."""

    def __init__(self, d_model: int, num_heads: int, cross_window: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionRoPE(d_model, num_heads, causal=True)
        self.norm2 = nn.RMSNorm(d_model)
        self.cross_attn = SlidingWindowCrossAttention(d_model, num_heads, window_size=cross_window)
        self.norm3 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, 4 * d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = x + self.cross_attn(self.norm2(x), key=memory, value=memory, key_padding_mask=memory_key_padding_mask)
        x = x + self.ffn(self.norm3(x))
        return x


class SlidingDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, cross_window: int):
        super().__init__()
        self.layers = nn.ModuleList([
            SlidingDecoderBlock(d_model, num_heads, cross_window) for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# CodeExpander: Sequence-to-Sequence model using the custom Transformer blocks
# ---------------------------------------------------------------------------

# @torch.compile
class CodeExpander(nn.Module):
    def __init__(
        self,
        K_hi: int,
        K_lo: int,
        D: int = 256,
        N_enc: int = 4,
        N_dec: int = 4,
        H: int = 8,
        eos_id: int = 1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.K_hi, self.K_lo = K_hi, K_lo
        self.D, self.eos_id, self.max_len = D, eos_id, max_len

        self.emb_hi = nn.Embedding(K_hi, D)
        self.emb_lo = nn.Embedding(K_lo, D)

        self.encoder = SimpleEncoder(N_enc, D, H)
        self.decoder = SimpleDecoder(N_dec, D, H)

        self.out_proj = nn.Linear(D, K_lo)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        codes_hi: torch.Tensor,
        codes_lo: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L_hi = codes_hi.shape
        _, L_lo = codes_lo.shape
        device = codes_hi.device

        enc_inp = self.emb_hi(codes_hi)
        memory = self.encoder(enc_inp, key_padding_mask=src_key_padding_mask)

        decoder_input_ids = F.pad(codes_lo[:, :-1], (1, 0), value=self.eos_id)
        dec_inp = self.emb_lo(decoder_input_ids)
        tgt_mask = self._causal_mask(L_lo, device)
        adjusted_tgt_kpm = None
        if tgt_key_padding_mask is not None:
            adjusted_tgt_kpm = F.pad(tgt_key_padding_mask[:, :-1], (1, 0), value=False)

        dec_out = self.decoder(
            dec_inp,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=adjusted_tgt_kpm,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.out_proj(dec_out)
        return {"logits": logits}

    @torch.no_grad()
    def generate(
        self,
        codes_hi: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        B, L_hi = codes_hi.shape
        device = codes_hi.device
        current_max_len = max_len if max_len is not None else self.max_len

        memory = self.encoder(self.emb_hi(codes_hi), key_padding_mask=src_key_padding_mask)

        generated_ids = torch.full((B, 1), self.eos_id, dtype=torch.long, device=device)

        for _ in range(current_max_len - 1):
            seq_len = generated_ids.size(1)
            dec_inp = self.emb_lo(generated_ids)
            tgt_mask = self._causal_mask(seq_len, device)
            dec_out = self.decoder(
                dec_inp,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_token_logits = self.out_proj(dec_out[:, -1, :])
            next_id = next_token_logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)
            if (next_id == self.eos_id).all():
                break
        return generated_ids[:, 1:]


class DecoderOnlyExpander(nn.Module):
    """Decoder-only variant using sliding-window cross attention for memory access."""

    def __init__(
        self,
        K_hi: int,
        K_lo: int,
        D: int = 256,
        N_dec: int = 4,
        H: int = 8,
        cross_window: int = 128,
        eos_id: int = 1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.K_hi, self.K_lo = K_hi, K_lo
        self.D = D
        self.eos_id = eos_id
        self.max_len = max_len

        self.emb_hi = nn.Embedding(K_hi, D)
        self.emb_lo = nn.Embedding(K_lo, D)

        self.decoder = SlidingDecoder(N_dec, D, H, cross_window)
        self.out_proj = nn.Linear(D, K_lo)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        codes_hi: torch.Tensor,
        codes_lo: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = codes_hi.device
        memory = self.emb_hi(codes_hi)

        decoder_input_ids = F.pad(codes_lo[:, :-1], (1, 0), value=self.eos_id)
        dec_inp = self.emb_lo(decoder_input_ids)
        tgt_mask = self._causal_mask(codes_lo.size(1), device)
        adjusted_tgt_kpm = None
        if tgt_key_padding_mask is not None:
            adjusted_tgt_kpm = F.pad(tgt_key_padding_mask[:, :-1], (1, 0), value=False)

        dec_out = self.decoder(
            dec_inp,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=adjusted_tgt_kpm,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.out_proj(dec_out)
        return {"logits": logits}

    @torch.no_grad()
    def generate(
        self,
        codes_hi: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        device = codes_hi.device
        current_max_len = max_len if max_len is not None else self.max_len

        memory = self.emb_hi(codes_hi)

        generated_ids = torch.full((codes_hi.size(0), 1), self.eos_id, dtype=torch.long, device=device)

        for _ in range(current_max_len - 1):
            seq_len = generated_ids.size(1)
            dec_inp = self.emb_lo(generated_ids)
            tgt_mask = self._causal_mask(seq_len, device)
            dec_out = self.decoder(
                dec_inp,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_logits = self.out_proj(dec_out[:, -1, :])
            next_id = next_logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)
            if (next_id == self.eos_id).all():
                break
        return generated_ids[:, 1:]
