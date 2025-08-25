# ---------------------------------------------------------------------------
# CodeExpander: Sequence-to-Sequence model using the custom Transformer blocks. Saved here for reference.
# ---------------------------------------------------------------------------
from functools import lru_cache
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from components.swiglu import SwiGLU
from components.rope import RoPECache, apply_rope
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

        self.rope = RoPECache(max_seqlen=2048, head_dim=self.head_dim, dtype=torch.bfloat16, device="cuda")

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
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

        # q = apply_rope(q)
        # k = apply_rope(k)

        L = q.size(-2)
        cos, sin = self.rope.slice(L)  # views only, no kernels
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

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


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with RoPE applied to Q and K.

    Parameters
    ----------
    d_model   : total model width
    num_heads : number of attention heads (d_model % num_heads == 0)
    causal    : if True, apply standard causal (lower-triangular) masking

    Notes
    -----
    - Accepts self- or cross-attention inputs (query vs key/value).
    - RoPE tables are cached once per module via `RoPECache`.
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            causal: bool = False,
            bias: bool = False,
            rope_max_seqlen: int = 2048,
            rope_base: float = 10000.0,
            device: Optional[torch.device] = None
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        # Cache is bf16 to reduce memory bandwidth; slices are views.
        self.rope = RoPECache(
            max_seqlen=2048,
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )

        # Standard projections (no bias change here for parity with original).
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(
            self,
            query: torch.Tensor,  # (B, L_q, d_model)
            key: Optional[torch.Tensor] = None,  # (B, L_k, d_model)
            value: Optional[torch.Tensor] = None,  # (B, L_k, d_model)
            attn_mask: Optional[torch.Tensor] = None,  # broadcastable to (B, H, L_q, L_k)
            key_padding_mask: Optional[torch.Tensor] = None,  # (B, L_k), True = mask
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key

        B, L_q, _ = query.shape
        L_k = key.size(1)
        H, d = self.num_heads, self.head_dim
        inv_sqrt_d = 1.0 / math.sqrt(d)

        # Project and reshape into (B, H, L, d).
        q = self.q_proj(query).view(B, L_q, H, d).transpose(1, 2)  # (B,H,L_q,d)
        k = self.k_proj(key).view(B, L_k, H, d).transpose(1, 2)  # (B,H,L_k,d)
        v = self.v_proj(value).view(B, L_k, H, d).transpose(1, 2)  # (B,H,L_k,d)

        # Apply RoPE with per-length slices (views only).
        cos_q, sin_q = self.rope.slice(L_q)
        cos_k, sin_k = self.rope.slice(L_k)
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)

        # Attention scores: (B, H, L_q, L_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * inv_sqrt_d

        # Optional masks -------------------------------------------------------
        # attn_mask: supports 2D (L_q, L_k), 3D (B, L_q, L_k), or 4D (B, H, L_q, L_k).
        if attn_mask is not None:
            m = attn_mask
            if m.dim() == 2:  # (L_q, L_k)
                m = m[None, None, :, :]
            elif m.dim() == 3:  # (B, L_q, L_k)
                m = m[:, None, :, :]
            # else assume already broadcastable 4D
            scores = scores.masked_fill(m.bool(), float("-inf"))

        # key_padding_mask: (B, L_k) -> broadcast to (B, H, L_q, L_k).
        if key_padding_mask is not None:
            m = key_padding_mask[:, None, None, :].bool()
            scores = scores.masked_fill(m, float("-inf"))

        # Causal mask: lower-triangular (query cannot see future keys).
        if self.causal:
            causal = torch.triu(
                torch.ones(L_q, L_k, dtype=torch.bool, device=scores.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal, float("-inf"))
        # ----------------------------------------------------------------------

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B,H,L_q,d)
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.o_proj(out)


@lru_cache(maxsize=64)
def _cached_causal_mask(length: int) -> torch.Tensor:
    """Return a causal mask computed on CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
    return mask


def get_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Return the cached causal mask on ``device``."""
    return _cached_causal_mask(length).to(device)

@torch.compile
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
        # self.out_proj = nn.Linear(D, K_lo, bias=False)
        # self.out_proj.weight = self.emb_lo.weight  # share the same Parameter

        self.encoder = SimpleEncoder(N_enc, D, H)
        self.decoder = SimpleDecoder(N_dec, D, H)

        self.out_proj = nn.Linear(D, K_lo)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return get_causal_mask(length, device)

    def forward(
        self,
        codes_hi: torch.Tensor,
        codes_lo: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if codes_hi.is_floating_point():
            B, L_hi, _ = codes_hi.shape
            memory_input = codes_hi
        else:
            B, L_hi = codes_hi.shape
            memory_input = self.emb_hi(codes_hi)
        _, L_lo = codes_lo.shape
        device = codes_hi.device

        memory = self.encoder(memory_input, key_padding_mask=src_key_padding_mask)

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
        src_key_padding_mask: torch.Tensor | None = None,
        max_len: int | None = None,
    ) -> torch.Tensor:
        if codes_hi.is_floating_point():
            B, L_hi, _ = codes_hi.shape
            memory_input = codes_hi
        else:
            B, L_hi = codes_hi.shape
            memory_input = self.emb_hi(codes_hi)
        device = codes_hi.device
        current_max_len = max_len if max_len is not None else self.max_len

        memory = self.encoder(memory_input, key_padding_mask=src_key_padding_mask)

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


class SimpleEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, 4 * d_model) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.final_norm(x)


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

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
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
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = x + self.cross_attn(
            self.norm2(x), key=memory, value=memory, attn_mask=cross_attn_mask, key_padding_mask=memory_key_padding_mask
        )
        x = x + self.ffn(self.norm3(x))
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, 4 * d_model) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        # cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                # cross_attn_mask=cross_attn_mask,
            )
        return self.final_norm(x)