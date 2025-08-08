from __future__ import annotations

import torch
import torch.nn as nn

from .mla import CausalMLATransformerBlock
from .vector_quantizer import VectorQuantizer


@torch.compile(dynamic=True)
class CodeSequenceTransformer(nn.Module):
    """Causal Transformer that predicts continuous embeddings."""

    def __init__(
        self,
        embed_dim: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim_multiplier: int = 4,
        head_dim: int | None = 32,  # K
        kv_comp_dim: int | None = 64,  # d_c
        q_comp_dim: int | None = 96,  # d_c`
        retr_dim: int | None = 32,  # r
        vq: VectorQuantizer | None = None,
        use_flex_attention: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dim = dim
        self.vq = vq
        self.use_flex_attention = use_flex_attention

        self.in_proj = nn.Linear(embed_dim, dim)
        self.encoder = nn.ModuleList(
            [
                CausalMLATransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    head_dim=head_dim,
                    kv_comp_dim=kv_comp_dim,
                    q_comp_dim=q_comp_dim,
                    retr_dim=retr_dim,
                    use_flex_attention=self.use_flex_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(dim)
        self.out_proj = nn.Linear(dim, embed_dim)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process a sequence of embeddings.

        Args:
            input_embeddings: ``(B, S, embed_dim)`` tensor of embeddings.
            key_padding_mask: Optional ``(B, S)`` mask where ``True`` marks padding.

        Returns:
            Dictionary with keys:
                ``'hidden_states'`` – transformer hidden states ``(B, S, dim)``.
                ``'predictions'`` – quantized predicted embeddings ``(B, S, embed_dim)``.
                ``'indices'`` – VQ code indices if ``vq`` is provided.
                ``'vq_loss'`` – VQ regularization loss (scalar).
        """
        x = self.in_proj(input_embeddings)
        for layer in self.encoder:
            x = layer(x, key_padding_mask=key_padding_mask)
        hidden_states = self.final_norm(x)

        preds_pre_vq = self.out_proj(hidden_states)
        preds = preds_pre_vq
        vq_loss = torch.tensor(0.0, device=preds.device)
        indices = None
        if self.vq is not None:
            preds, vq_loss, indices, _ = self.vq(preds_pre_vq)

        return {
            "hidden_states": hidden_states,
            "predictions_pre_vq": preds_pre_vq,
            "predictions": preds,
            "indices": indices,
            "vq_loss": vq_loss,
        }

    @torch.no_grad()
    def generate_embeddings(
        self,
        prefix: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        max_len: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate embeddings until EOS is produced."""
        assert self.vq is not None, "generate_embeddings requires a VectorQuantizer"

        B = prefix.size(0)
        generated = prefix
        kpm = key_padding_mask
        max_steps = max_len if max_len is not None else 1024

        for _ in range(max_steps):
            out = self.forward(generated, key_padding_mask=kpm)
            next_vec = out["predictions_pre_vq"][:, -1, :]
            next_idx = out["indices"][:, -1] if out["indices"] is not None else None

            generated = torch.cat([generated, next_vec.unsqueeze(1)], dim=1)
            if kpm is not None:
                pad = torch.zeros(B, 1, dtype=kpm.dtype, device=kpm.device)
                kpm = torch.cat([kpm, pad], dim=1)

            if next_idx is not None and (next_idx == self.vq.eos_token_id).all():
                break

        return generated
