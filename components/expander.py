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
from .vector_quantizer import MultiStageResidualVQ, RVQEmbeddingAdapter, RVQFactorizedHead, ComposedIndexCodec, \
    LearnedCodebookAdapter


@lru_cache(maxsize=64)
def _cached_causal_mask(length: int) -> torch.Tensor:
    """Return a causal mask computed on CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
    return mask


def get_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Return the cached causal mask on ``device``."""
    return _cached_causal_mask(length).to(device)


# ------------------------- SDPA + fused QKV -------------------------

def _to_additive_mask(
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    B: int, H: int, L_q: int, L_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Convert boolean / mixed masks to an *additive* attention bias suitable for SDPA.
    Result is broadcastable to (B, H, L_q, L_k). Returns None if no masking needed.
    Using an additive mask preserves fast SDPA/Flash paths better than boolean masks.
    """
    bias = None

    # Handle attn_mask: accept (L_q, L_k), (B, L_q, L_k), or (B, H, L_q, L_k)
    if attn_mask is not None:
        m = attn_mask
        if m.dim() == 2:             # (L_q, L_k)
            m = m.unsqueeze(0).unsqueeze(0)      # (1,1,L_q,L_k)
        elif m.dim() == 3:           # (B, L_q, L_k)
            m = m.unsqueeze(1)                   # (B,1,L_q,L_k)
        # else: assume (B, H, L_q, L_k)

        if m.dtype == torch.bool:
            neg = torch.finfo(dtype).min
            bias = torch.where(m.to(device), torch.tensor(neg, dtype=dtype, device=device), torch.tensor(0, dtype=dtype, device=device))
        else:
            bias = m.to(dtype=dtype, device=device)

    # Fold key_padding_mask: (B, L_k) -> (B,1,1,L_k)
    if key_padding_mask is not None:
        kpm = key_padding_mask.to(device=device).bool().view(B, 1, 1, L_k)
        neg = torch.finfo(dtype).min
        kpm_bias = torch.where(kpm, torch.tensor(neg, dtype=dtype, device=device), torch.tensor(0, dtype=dtype, device=device))
        bias = kpm_bias if bias is None else (bias + kpm_bias)

    return bias


class MultiHeadAttentionFused(nn.Module):
    """
    Multi-Head Attention with:
      - fused QKV projection (single matmul)
      - RoPE applied to Q and K
      - PyTorch SDPA backend (routes to Flash/Memory-Efficient/Math as appropriate)

    API is compatible with the original class (self/cross usage).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        causal: bool = False,
        rope_max_seqlen: int = 2048,
        rope_base: float = 10000.0,
        rope_dtype: torch.dtype = torch.bfloat16,
        rope_device: torch.device | str = "cuda",
        bias: bool = True,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        # Fused QKV projection. We reuse slices of this weight even for cross-attn (no extra params).
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # RoPE cache (moves with module.to(device))
        self.rope = RoPECache(
            max_seqlen=rope_max_seqlen,
            head_dim=self.head_dim,
            base=rope_base,
            dtype=rope_dtype,
            device=rope_device,
        )

    # --- lightweight helpers ---
    def _shape(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """(B, L, D) -> (B, H, L, d)"""
        H, d = self.num_heads, self.head_dim
        return x.view(B, L, H, d).transpose(1, 2)

    def _q_from(self, x: torch.Tensor) -> torch.Tensor:
        W, b = self.in_proj.weight, self.in_proj.bias
        D = self.d_model
        return F.linear(x, W[:D, :], b[:D] if b is not None else None)

    def _kv_from(self, x: torch.Tensor) -> torch.Tensor:
        W, b = self.in_proj.weight, self.in_proj.bias
        D = self.d_model
        return F.linear(x, W[D:, :], b[D:] if b is not None else None)  # (B,L,2D)

    # --- forward ---
    def forward(
        self,
        query: torch.Tensor,                          # (B, L_q, D)
        key: Optional[torch.Tensor] = None,           # (B, L_k, D) or None -> self-attn
        value: Optional[torch.Tensor] = None,         # (B, L_k, D)
        attn_mask: Optional[torch.Tensor] = None,     # (L_q,L_k), (B,L_q,L_k), or (B,H,L_q,L_k); bool or additive
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L_k) True = pad
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key

        B, L_q, _ = query.shape
        L_k = key.size(1)

        # Self-attn path: do one fused projection. Cross-attn path: slice weights to avoid
        # computing unused 'q' for the memory (no extra parameters, minimal compute waste).
        if key is query and value is key:
            qkv = self.in_proj(query)                              # (B,L_q,3D)
            q, k, v = qkv.split(self.d_model, dim=-1)              # each (B,L_q,D)
        else:
            q = self._q_from(query)                                # (B,L_q,D)
            kv = self._kv_from(key)                                # (B,L_k,2D)
            k, v = kv.split(self.d_model, dim=-1)                  # (B,L_k,D)
            if value is not key:
                # Use the v-slice directly on value
                W, b = self.in_proj.weight, self.in_proj.bias
                D = self.d_model
                v = F.linear(value, W[2*D:3*D, :], b[2*D:3*D] if b is not None else None)

        # Reshape to (B,H,L, d)
        q = self._shape(q, B, L_q)
        k = self._shape(k, B, L_k)
        v = self._shape(v, B, L_k)

        # RoPE (use per-length slices; RoPE buffers move with module.to(...))
        cos_q, sin_q = self.rope.slice(L_q)
        cos_k, sin_k = self.rope.slice(L_k)
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)

        # Merge masks into an additive bias (prefer fast SDPA paths).
        bias = _to_additive_mask(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            B=B, H=self.num_heads, L_q=L_q, L_k=L_k,
            dtype=q.dtype, device=q.device,
        )

        # SDPA routes to Flash/Memory-Efficient/Math automatically.
        # Prefer passing `is_causal=True` (and no causal mask) for fastest path.
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias,               # additive or None
            is_causal=self.causal and (attn_mask is None),
        )  # (B,H,L_q,d)

        out = attn.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.out_proj(out)


class SlidingDecoderBlock(nn.Module):
    """Decoder block using causal self-attention and sliding-window cross attention."""

    def __init__(self, idx: int, d_model: int, num_heads: int, cross_window: int, q_dim, kv_dim):
        super().__init__()
        self.layer_id = f'L{idx}'
        self.norm1 = nn.RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionFused(d_model, num_heads, causal=True)
        self.norm2 = nn.RMSNorm(d_model)
        self.cross_attn = SegmentCausalCrossAttention(
            q_dim=q_dim, kv_dim=kv_dim, d_attn=d_model, n_heads=num_heads, lookback=cross_window, bias=False
        )
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

        x = x + self.cross_attn(
            self.norm2(x),
            kv_src=memory,
            seg_id=seg_ids,
            kv_mask=memory_key_padding_mask,
            q_pad_mask=tgt_key_padding_mask,
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


class DecoderOnlyExpanderRVQ(nn.Module):
    """
    Decoder-only expander that:
      - embeds high codes by reusing the *high-level* MS-RVQ codebooks (no emb_hi),
      - embeds low codes likewise (no emb_lo),
      - outputs factorized stage-wise logits (no D×K_eff softmax).
    """

    def __init__(self,
                 lo_vq: "MultiStageResidualVQ",  # MS-RVQ for the *target* code space
                 hi_vq: Optional["MultiStageResidualVQ"],  # MS-RVQ for the *memory* code space
                 K_lo: Optional[int] = None,  # if lo_vq is None
                 D: int = 256,
                 N_dec: int = 4,
                 H: int = 8,
                 cross_window: int = 128,
                 eos_id: int = 257,
                 eop_id: int = 259,
                 max_len: int = 2048,
                 lo_d_c: int = 64,  # Factorized rank for bytes
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

        else:
            assert K_lo is not None, "Need K_lo for bottom level when lo_vq is None."
            self.lo_adapt = LearnedCodebookAdapter(K=K_lo, D=D, d_c=lo_d_c, device=device, )
            self.lo_codec = ComposedIndexCodec(K=K_lo, depth=1, bos=-1, eos=-1, pad=-1, eop=-1)

        # ---- high side (memory) ----
        self.hi_adapt = RVQEmbeddingAdapter(hi_vq, target_D=D, use_shared_proj=True,
                                            specials_in_D=True) if hi_vq is not None else None

        # Your existing decoder block
        self.decoder = SlidingDecoder(N_dec, D, H, cross_window, q_dim=D, kv_dim=D)

        # Factorized head
        self.head = RVQFactorizedHead(self.lo_adapt,
                                      residual_conditioning=residual_conditioning,
                                      use_sqdist_logits=use_sqdist_logits,
                                      predict_specials=predict_specials)

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
                codes_hi: torch.Tensor,  # (B, S_hi) or (B, S_hi, D)
                codes_lo: torch.Tensor,  # (B, L)
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                seg_ids: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        device = codes_hi.device
        memory = self._embed_memory(codes_hi)  # (B, S_hi, D)
        dec_inp = self._embed_decoder_inputs(codes_lo)  # (B, L, D)

        tgt_mask = self._causal_mask(codes_lo.size(1), device)
        adjusted_tgt_kpm = None
        if tgt_key_padding_mask is not None:
            adjusted_tgt_kpm = F.pad(tgt_key_padding_mask[:, :-1], (1, 0), value=False)

        h = self.decoder(dec_inp, memory,
                         tgt_mask=tgt_mask,
                         tgt_key_padding_mask=adjusted_tgt_kpm,
                         memory_key_padding_mask=src_key_padding_mask,
                         seg_ids=seg_ids)  # (B, L, D)

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
