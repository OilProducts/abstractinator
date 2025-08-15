from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPECache, apply_rope
from .sliding_window_attention import AttnCache, SegmentCausalCrossAttention
from .swiglu import SwiGLU
from .vector_quantizer import MultiStageResidualVQ, RVQEmbeddingAdapter, RVQFactorizedHead, ComposedIndexCodec, LearnedCodebookAdapter


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


class SimpleEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, 4 * d_model) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
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
        seg_ids: torch.Tensor | None = None,
        cache: AttnCache | None = None,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
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
            # cache=cross_cache,
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
        cache: AttnCache | None = None,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        # cross_attn_mask: Optional[torch.Tensor] = None,
        seg_ids: torch.Tensor | None = None,
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

        self.rope = RoPECache(max_seqlen=self.max_len, head_dim=kv_dim_or_q_dim,  # match your use
                              dtype=model_dtype, device=next(self.parameters()).device)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return get_causal_mask(length, device)

    def forward(
        self,
        codes_hi: torch.Tensor,
        codes_lo: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        seg_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = codes_hi.device
        memory = codes_hi if codes_hi.is_floating_point() else self.emb_hi(codes_hi)

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
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        seg_ids: torch.Tensor | None = None,
        max_len: int | None = None,
        max_new_tokens: int | None = None,
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


class DecoderOnlyExpanderRVQ(nn.Module):
    """
    Decoder-only expander that:
      - embeds high codes by reusing the *high-level* MS-RVQ codebooks (no emb_hi),
      - embeds low codes likewise (no emb_lo),
      - outputs factorized stage-wise logits (no D×K_eff softmax).
    """
    def __init__(self,
                 lo_vq: "MultiStageResidualVQ",       # MS-RVQ for the *target* code space
                 hi_vq: Optional["MultiStageResidualVQ"],  # MS-RVQ for the *memory* code space
                 K_lo: Optional[int] = None,          # if lo_vq is None
                 D: int = 256,
                 N_dec: int = 4,
                 H: int = 8,
                 cross_window: int = 128,
                 eos_id: int = 257,
                 eop_id: int = 259,
                 max_len: int = 2048,
                 lo_d_c: int = 64,                    # Factorized rank for bytes
                 residual_conditioning: bool = True,
                 use_sqdist_logits: bool = False,
                 predict_specials: bool = True,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.lo_vq = lo_vq
        self.hi_vq = hi_vq
        self.D = D
        self.eos_id = eos_id
        self.eop_id = eop_id
        self.max_len = max_len
        self.K_lo = K_lo

        # Reuse VQ projections and codebooks
        # ---- low side (what we predict) ----
        if lo_vq is not None:
            self.lo_adapt = RVQEmbeddingAdapter(lo_vq, target_D=D,
                                                use_shared_proj=True,
                                                # tie_up_down=True,
                                                specials_in_D=True)
            self.lo_codec = ComposedIndexCodec(K=lo_vq.K, depth=lo_vq.depth,
                                               bos=lo_vq.bos_token_id, eos=lo_vq.eos_token_id,
                                               pad=lo_vq.padding_token_id, eop=lo_vq.eop_token_id)
            # self.head = RVQFactorizedHead(self.lo_adapt,
            #                               residual_conditioning=residual_conditioning,
            #                               use_sqdist_logits=use_sqdist_logits,
            #                               predict_specials=predict_specials)
        else:
            assert K_lo is not None, "Need K_lo for bottom level when lo_vq is None."
            self.lo_adapt = LearnedCodebookAdapter(K=K_lo, D=D, d_c=lo_d_c, device=device,)
            self.lo_codec = ComposedIndexCodec(K=K_lo, depth=1, bos=-1, eos=-1, pad=-1, eop=-1)
            # self.head = RVQFactorizedHead(self.lo_adapt,
            #                               residual_conditioning=False,
            #                               use_sqdist_logits=False,
            #                               predict_specials=False)

        # ---- high side (memory) ----
        self.hi_adapt = RVQEmbeddingAdapter(hi_vq, target_D=D, use_shared_proj=True, specials_in_D=True) if hi_vq is not None else None

        # Your existing decoder block
        self.decoder = SlidingDecoder(N_dec, D, H, cross_window, q_dim=D, kv_dim=D)

        # Factorized head
        self.head = RVQFactorizedHead(self.lo_adapt,
                                      residual_conditioning=residual_conditioning,
                                      use_sqdist_logits=use_sqdist_logits,
                                      predict_specials=predict_specials)

        # Codec for (de)composition
        # self.codec = ComposedIndexCodec(K=lo_vq.K, depth=lo_vq.depth,
        #                                 bos=lo_vq.bos_token_id, eos=lo_vq.eos_token_id,
        #                                 pad=lo_vq.padding_token_id, eop=lo_vq.eop_token_id)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return get_causal_mask(length, device)

    def _embed_memory(self, codes_hi: torch.Tensor) -> torch.Tensor:
        # Either high side is already continuous (D), or embed via hi_vq
        if codes_hi.is_floating_point():
            return codes_hi
        if self.hi_adapt is None:
            raise ValueError("codes_hi is discrete but hi_vq/hi_adapt not provided")
        return self.hi_adapt.embed_composed(codes_hi)

    def _embed_decoder_inputs(self, codes_lo: torch.Tensor) -> torch.Tensor:
        # Shift-right with EOS
        decoder_input_ids = F.pad(codes_lo[:, :-1], (1, 0), value=self.eos_id)
        return self.lo_adapt.embed_composed(decoder_input_ids)

    def forward(self,
                codes_hi: torch.Tensor,                # (B, S_hi) or (B, S_hi, D)
                codes_lo: torch.Tensor,                # (B, L)
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                seg_ids: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        device = codes_hi.device
        memory = self._embed_memory(codes_hi)          # (B, S_hi, D)
        dec_inp = self._embed_decoder_inputs(codes_lo) # (B, L, D)

        tgt_mask = self._causal_mask(codes_lo.size(1), device)
        adjusted_tgt_kpm = None
        if tgt_key_padding_mask is not None:
            adjusted_tgt_kpm = F.pad(tgt_key_padding_mask[:, :-1], (1, 0), value=False)

        h = self.decoder(dec_inp, memory,
                         tgt_mask=tgt_mask,
                         tgt_key_padding_mask=adjusted_tgt_kpm,
                         memory_key_padding_mask=src_key_padding_mask,
                         seg_ids=seg_ids)              # (B, L, D)

        # Teacher-forced residual conditioning uses *true* digits at (B,L)
        teacher_digits, _ = self.lo_codec.decompose(codes_lo)
        logits = self.head(h, teacher_digits=teacher_digits)  # dict

        return logits  # {"stage_logits": [B,L,K]*depth, "special_logits": (B,L,S) or None}


    @torch.no_grad()
    def generate(
            self,
            codes_hi: torch.Tensor,  # (B, S_hi) or (B, S_hi, D)
            codes_lo: torch.Tensor,  # (B, T0)  seed(s)
            src_key_padding_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,  # <- compat
            max_len: Optional[int] = None,
            max_new_tokens: Optional[int] = None,
            seg_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = codes_hi.device
        B = codes_hi.size(0)
        memory = self._embed_memory(codes_hi)  # (B, S_hi, D)
        generated = codes_lo.clone()  # (B, T)
        kpm = tgt_key_padding_mask.clone() if tgt_key_padding_mask is not None \
            else torch.zeros_like(generated, dtype=torch.bool)

        cache = AttnCache()

        # steps budget: how many *new* tokens we’re allowed to produce
        steps_budget = int(max_new_tokens) if max_new_tokens is not None else \
            int((max_len or self.max_len) - 1)
        steps_budget = max(0, steps_budget)

        # helper: align seg_ids length to current T
        def _align_seg_ids(curr_T: int) -> torch.Tensor:
            if seg_ids is None:
                last_seg = memory.size(1) - 1
                return torch.full((B, curr_T), last_seg, dtype=torch.long, device=device)
            if seg_ids.size(1) < curr_T:
                pad = seg_ids[:, -1:].expand(-1, curr_T - seg_ids.size(1))
                return torch.cat([seg_ids, pad], dim=1)
            if seg_ids.size(1) > curr_T:
                return seg_ids[:, :curr_T]
            return seg_ids

        depth = self.lo_codec.depth  # works for both MS-RVQ and learned single-stage

        for _ in range(steps_budget):
            T = generated.size(1)
            dec_inp = self.lo_adapt.embed_composed(generated)  # (B, T, D)
            tgt_mask = self._causal_mask(T, device)
            seg_ids_cur = _align_seg_ids(T)  # (B, T)

            h = self.decoder(
                dec_inp, memory,
                cache=cache,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=kpm,
                memory_key_padding_mask=src_key_padding_mask,
                seg_ids=seg_ids_cur,
            )  # (B, T, D)

            last_h = h[:, -1:, :]  # (B,1,D)
            r_dc = self.lo_adapt.down(last_h)  # (B,1,d_c)

            # Greedy stage-by-stage with residual updates
            digits = []
            for s in range(depth):
                W = self.lo_adapt.stage_codebook(s)  # (K, d_c)
                logit_s = self.head._stage_logits(r_dc, W)  # (B,1,K)
                idx_s = logit_s.argmax(dim=-1)  # (B,1)
                digits.append(idx_s)  # keep time dim
                r_dc = r_dc - F.embedding(idx_s, W)  # (B,1,d_c)

            next_id = self.lo_codec.compose(digits)  # (B,1) or (B,)
            if next_id.ndim == 1:
                next_id = next_id.unsqueeze(1)  # (B,1)

            generated = torch.cat([generated, next_id], dim=1)
            if kpm is not None:
                kpm = F.pad(kpm, (0, 1), value=False)  # newly appended is real

            # optional early stop if you actually predict specials
            if ((next_id == self.eos_id) | (next_id == self.eop_id)).all():
                break

        return generated
    #
    # @torch.no_grad()
    # def generate(self,
    #              codes_hi: torch.Tensor,               # (B, S_hi) or (B, S_hi, D)
    #              codes_lo: Optional[torch.Tensor] = None,  # (B, L) or None
    #              src_key_padding_mask: Optional[torch.Tensor] = None,
    #              max_len: Optional[int] = None,
    #              max_new_tokens: Optional[int] = None,
    #              seg_ids: Optional[torch.Tensor] = None
    #              ) -> torch.Tensor:
    #     """
    #     Autoregressively generates low-level composed codes (B, L).
    #     Within each time step, it selects stage indices sequentially (residual RVQ style).
    #     """
    #     device = codes_hi.device
    #     memory = self._embed_memory(codes_hi)     # (B, S_hi, D)
    #
    #     if codes_lo is None:
    #         B = memory.size(0)
    #         generated = torch.full((B, 1), self.eos_id, device=device, dtype=torch.long)
    #     else:
    #         B, _ = codes_lo.shape
    #         generated = codes_lo.clone()  # (B, L)
    #
    #     hard_cap = int(max_len or self.max_len)
    #     plan_len = None
    #
    #     if seg_ids is not None and isinstance(seg_ids, (tuple, list)) and len(seg_ids == 2):
    #         _, q_seg_full = seg_ids
    #         # q_seg_full: (B, T_plan). We decode at most this many steps.
    #         plan_len = int(q_seg_full.size(1))
    #     target_len = hard_cap if plan_len is None else min(hard_cap, plan_len)
    #
    #     # Also respect max_new_tokens if set
    #     if max_new_tokens is not None:
    #         target_len = min(target_len, generated.size(1) + int(max_new_tokens))
    #         # No padding in an ever-growing decode buffer; keep a dummy mask of zeros
    #         kpm = torch.zeros_like(generated, dtype=torch.bool)
    #
    #     # # B = codes_hi.size(0)
    #     # Lmax = max_len or self.max_len
    #     # B, L = codes_lo.shape
    #     #
    #     # # generated = torch.full((B, 1), self.eos_id, device=device, dtype=torch.long)
    #     # # if tgt_key_padding_mask is None:
    #     # kpm = torch.zeros_like(codes_lo, dtype=torch.bool)
    #     # # kpm = torch.zeros_like(generated, dtype=torch.bool)
    #     # generated = codes_lo.clone()
    #
    #     cache = AttnCache()
    #
    #     # Decode until we reach the target length
    #     while generated.size(1) < target_len:
    #         dec_inp = self.lo_adapt.embed_composed(generated)  # (B, T, D)
    #         tgt_mask = self._causal_mask(dec_inp.size(1), device)
    #
    #         h = self.decoder(dec_inp, memory,
    #                          cache=cache,
    #                          tgt_mask=tgt_mask,
    #                          tgt_key_padding_mask=kpm,
    #                          memory_key_padding_mask=src_key_padding_mask,
    #                          seg_ids=seg_ids)  # (B, T, D)
    #
    #         last_h = h[:, -1:, :]  # (B,1,D)
    #         # pick stage digits sequentially and compose
    #         digits = []
    #         r_dc = self.lo_adapt.down(last_h)  # (B,1,d_c)
    #         for s in range(self.lo_vq.depth):
    #             W = self.lo_adapt.stage_codebook(s)
    #             logit_s = self.head._stage_logits(r_dc, W)  # (B,1,K)
    #             idx_s = logit_s.argmax(dim=-1)  # (B,1)
    #             digits.append(idx_s)
    #             # residual update
    #             r_dc = r_dc - F.embedding(idx_s, W)
    #         next_id = self.lo_codec.compose([d for d in digits])  # (B,1)
    #         generated = torch.cat([generated, next_id], dim=1)
    #         if next_id.ndim == 1:
    #             next_id = next_id.unsqueeze(1)
    #         # stop early if all EOS/EOP (optional)
    #         if ((next_id == self.eos_id) | (next_id == self.eop_id)).all():
    #             break
    #
    #         # keep mask all non-pad
    #         kpm = torch.zeros_like(generated, dtype=torch.bool)
    #
    #     return generated

        # for _ in range(Lmax - 1):
        #     dec_inp = self.lo_adapt.embed_composed(generated)    # (B, T, D)
        #     tgt_mask = self._causal_mask(dec_inp.size(1), device)
        #
        #     h = self.decoder(dec_inp, memory,
        #                      cache=cache,
        #                      tgt_mask=tgt_mask,
        #                      tgt_key_padding_mask=kpm,
        #                      memory_key_padding_mask=src_key_padding_mask,
        #                      seg_ids=seg_ids)                    # (B, T, D)
        #
        #     last_h = h[:, -1:, :]                                # (B,1,D)
        #     # No teacher digits in inference; the head will greedy-condition residuals
        #     logits = self.head(last_h, teacher_digits=None)
        #     stage_logits: List[torch.Tensor] = logits["stage_logits"]  # each (B,1,K)
        #
        #     # pick stage digits sequentially and compose
        #     digits = []
        #     r_dc = self.lo_adapt.down(last_h)                     # (B,1,d_c)
        #     for s in range(self.lo_vq.depth):
        #         W = self.lo_adapt.stage_codebook(s)
        #         logit_s = self.head._stage_logits(r_dc, W)        # (B,1,K)
        #         idx_s = logit_s.argmax(dim=-1)                    # (B,1)
        #         digits.append(idx_s.squeeze(1))
        #         # residual update
        #         r_dc = r_dc - F.embedding(idx_s, W)
        #     next_id = self.lo_codec.compose([d for d in digits])     # (B,1)
        #
        #     generated = torch.cat([generated, next_id], dim=1)
        #     # stop early if all EOS/EOP (optional)
        #     if ((next_id == self.eos_id) | (next_id == self.eop_id)).all():
        #         break
        #
        #     # keep mask all non-pad
        #     kpm = torch.zeros_like(generated, dtype=torch.bool)
        #
        # return generated
