import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...swiglu import SwiGLU
from ..masks import merge_masks


class TransformerBlock(nn.Module):
    """
    Standard SDPA-based Transformer block (self-attention + MLP) with residuals
    and LayerNorm. Ported from components/attentions.py without behavior changes.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        prenorm: bool = True,
        ln_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_ff = d_ff or 4 * d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop_p = attn_dropout
        self.resid_drop = nn.Dropout(resid_dropout)
        self.mlp_drop = nn.Dropout(resid_dropout)

        self.ln1 = nn.LayerNorm(d_model, eps=ln_eps)
        self.ln2 = nn.LayerNorm(d_model, eps=ln_eps)
        self.prenorm = prenorm

        self.mlp = nn.Sequential(
            SwiGLU(d_model, self.d_ff),
            self.mlp_drop,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _combine_masks(
        self,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        B: int,
        H: int,
        T_q: int,
        T_k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        return merge_masks(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            B=B,
            H=H,
            L_q=T_q,
            L_k=T_k,
            dtype=dtype,
            device=device,
        )

    def _self_attention(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        mask = self._combine_masks(attn_mask, key_padding_mask, B, self.n_heads, T, T, x.device, q.dtype)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=is_causal,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(attn)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.prenorm:
            x = x + self._self_attention(self.ln1(x), attn_mask, key_padding_mask, is_causal)
            x = x + self.mlp(self.ln2(x))
        else:
            x = self.ln1(x + self._self_attention(x, attn_mask, key_padding_mask, is_causal))
            x = self.ln2(x + self.mlp(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: Optional[int] = None,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        prenorm: bool = True,
        final_layer_norm: bool = True,
        ln_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                prenorm=prenorm,
                ln_eps=ln_eps,
                bias=bias,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model, eps=ln_eps) if final_layer_norm else nn.Identity()

    @torch.compile(fullgraph=False)
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_hidden_states: bool = False,
    ):
        hiddens: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
            )
            if return_hidden_states:
                hiddens.append(x)
        x = self.final_norm(x)
        return (x, hiddens) if return_hidden_states else x
