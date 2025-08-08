import math
from functools import lru_cache
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope
from .sliding_window_attention import AttnCache, SegmentCausalCrossAttention
from .swiglu import SwiGLU


@lru_cache(maxsize=64)
def _cached_causal_mask(length: int) -> torch.Tensor:
    """Return a causal mask computed on CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
    return mask


def get_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Return the cached causal mask on ``device``."""
    return _cached_causal_mask(length).to(device)


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
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = x + self.cross_attn(
            self.norm2(x), key=memory, value=memory, attn_mask=cross_attn_mask, key_padding_mask=memory_key_padding_mask
        )
        x = x + self.ffn(self.norm3(x))
        return x


class SimpleEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, 4 * d_model) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.final_norm(x)


class SimpleDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, 4 * d_model) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
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


class SlidingDecoderBlock(nn.Module):
    """Decoder block using causal self-attention and sliding-window cross attention."""

    def __init__(self, idx: int, d_model: int, num_heads: int, cross_window: int, q_dim, kv_dim):
        super().__init__()
        self.layer_id = f'L{idx}'
        self.norm1 = nn.RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionRoPE(d_model, num_heads, causal=True)
        self.norm2 = nn.RMSNorm(d_model)
        self.cross_attn = SegmentCausalCrossAttention(
            q_dim=q_dim, kv_dim=kv_dim, d_attn=d_model, n_heads=num_heads, lookback=cross_window, bias=False
        )
        # self.cross_attn = SlidingWindowCrossAttention(d_model, num_heads, window_size=cross_window)
        self.norm3 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, 4 * d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        seg_ids: Optional[torch.Tensor] = None,
        cache: Optional[AttnCache] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        # cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)

        cross_cache = None if cache is None else cache.cross.setdefault(self.layer_id, {})
        x = x + self.cross_attn(
            self.norm2(x),
            kv_src=memory,
            seg_id=seg_ids,
            kv_mask=memory_key_padding_mask,
            q_pad_mask=tgt_key_padding_mask,
            cache=cross_cache,
        )
        x = x + self.ffn(self.norm3(x))
        return x


class SlidingDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, cross_window: int, q_dim: int, kv_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for l_idx in range(num_layers):
            layer = SlidingDecoderBlock(
                idx=l_idx, d_model=d_model, num_heads=num_heads, cross_window=cross_window, q_dim=q_dim, kv_dim=kv_dim
            )
            self.layers.append(layer)

        # self.layers = nn.ModuleList([
        #     SlidingDecoderBlock(d_model, num_heads, cross_window, q_dim, kv_dim) for _ in range(num_layers)
        # ])
        self.final_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        cache: Optional[AttnCache] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        # cross_attn_mask: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                memory,
                cache=cache,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                # cross_attn_mask=cross_attn_mask,
                seg_ids=seg_ids,
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
        return get_causal_mask(length, device)

    def forward(
        self,
        codes_hi: torch.Tensor,
        codes_lo: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
        src_key_padding_mask: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
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


# @torch.compile
class DecoderOnlyExpander(nn.Module):
    """Decoder-only variant using sliding-window cross attention for memory access."""

    def __init__(
        self,
        K_hi: int,
        K_lo: int,
        hi_dim: int = 256,  # kv_dim
        lo_dim: int = 256,  # q_dim
        D: int = 256,
        N_dec: int = 4,
        H: int = 8,
        cross_window: int = 128,
        eos_id: int = 257,
        eop_id: int = 259,
        max_len: int = 2048,
        cross_lookback_bytes: int = 128,
    ) -> None:
        super().__init__()
        self.K_hi, self.K_lo = K_hi, K_lo
        self.D = D
        self.H = H
        self.eos_id = eos_id
        self.eop_id = eop_id
        self.max_len = max_len
        self.cross_lookback_bytes = cross_lookback_bytes
        self.cross_window = cross_window

        self.emb_hi = nn.Embedding(K_hi, D)
        self.emb_lo = nn.Embedding(K_lo, D)

        self.decoder = SlidingDecoder(N_dec, D, H, cross_window, q_dim=lo_dim, kv_dim=hi_dim)
        self.out_proj = nn.Linear(D, K_lo)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return get_causal_mask(length, device)

    def forward(
        self,
        codes_hi: torch.Tensor,
        codes_lo: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = codes_hi.device
        if codes_hi.is_floating_point():
            memory = codes_hi
        else:
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
            seg_ids=seg_ids,
        )
        logits = self.out_proj(dec_out)
        return {"logits": logits}

    @torch.no_grad()
    def generate(
        self,
        codes_hi: torch.Tensor,
        codes_lo: torch.Tensor,  # (B, L)  – already padded to L
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressively fills the PAD slots of `codes_lo` while keeping its length
        static.  `tgt_key_padding_mask` must match `codes_lo` (True == PAD).
        """
        device = codes_hi.device
        B, L = codes_lo.shape

        # ── high-level memory ───────────────────────────────────────────────
        memory = codes_hi if codes_hi.is_floating_point() else self.emb_hi(codes_hi)

        # ensure we have a padding mask to mutate in-place
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = torch.zeros_like(codes_lo, dtype=torch.bool)

        generated = codes_lo.clone()  # (B, L)
        cache = AttnCache()  # KV-cache for all decoder layers

        # remaining capacity per row
        len_valid = (~tgt_key_padding_mask).sum(dim=1)  # (B,)
        capacity = (L - len_valid).clamp(min=0)  # (B,)
        steps_budget = int(capacity.max().item())
        if max_new_tokens is not None:
            steps_budget = min(steps_budget, int(max_new_tokens))

        for _ in range(steps_budget):
            # how many non-PAD tokens are present in each sample
            len_valid = (~tgt_key_padding_mask).sum(dim=1)  # (B,)

            # stop if every sequence is full
            if (len_valid == L).all():
                break

            # run the decoder on the *full* sequence
            dec_inp = self.emb_lo(generated)  # (B, L, D)
            tgt_mask = self._causal_mask(L, device)

            dec_out = self.decoder(
                dec_inp,
                memory,
                cache=cache,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
                seg_ids=seg_ids,
            )  # (B, L, D)

            # ── logits for the last valid token in each batch ─────
            gather_idx = (len_valid - 1).unsqueeze(1).unsqueeze(2)  # (B,1,1)
            last_hid = dec_out.gather(1, gather_idx.expand(-1, 1, dec_out.size(-1))).squeeze(1)  # (B, D)

            next_logits = self.out_proj(last_hid)  # (B, K_lo)
            next_id = next_logits.argmax(dim=-1)  # (B,)

            # ── write `next_id` into the first PAD slot, update mask ────
            insert_pos = len_valid.unsqueeze(1)  # (B,1)
            generated.scatter_(1, insert_pos, next_id.unsqueeze(1))
            tgt_key_padding_mask.scatter_(1, insert_pos, False)

            # optional early-stop on EOS/EOP
            if ((next_id == self.eos_id) | (next_id == self.eop_id)).all():
                break

        return generated
