from __future__ import annotations

from components.config_types import AbstractinatorConfig, AttentionConfig, PyramidConfig, TopTransformerConfig
from experiments.exp_config import DEVICE, ExpConfig

# Attention: MLA form with SDPA-like setting maps to MLA fallback backend in implementation
MLA_FALLBACK = AttentionConfig(
    variant="mla",
    kernel="sdpa",  # interpreted as non-flex (fallback) for MLA
    head_dim=32,
    kv_comp_dim=64,
    q_comp_dim=128,
    retr_dim=32,
)


# One level, 256‑wide, smallish; entropy model up to 12 layers; top LM up to 12 layers
level = AbstractinatorConfig(
    D=256,
    c_heads=8,
    c_window=64,
    c_head_dim=32,
    c_kv_comp_dim=64,
    c_q_comp_dim=128,
    c_retr_dim=32,
    c_num_encoder_layers=6,
    c_num_entropy_encoder_layers=12,
    c_num_compression_encoder_layers=4,
    d_layers=6,
    d_heads=8,
    d_cross_window=1,
    d_max_len=512,
)
level.compressor_attention = MLA_FALLBACK
level.decoder_self_attention = MLA_FALLBACK
level.decoder_cross_attention = MLA_FALLBACK


top_lm = TopTransformerConfig(
    embed_dim=256,
    dim=256,
    num_layers=12,
    num_heads=8,
    head_dim=32,
    kv_comp_dim=64,
    q_comp_dim=128,
    retr_dim=32,
    attention_config=MLA_FALLBACK,
)


exp_config = ExpConfig(
    run_name="mla_sdpa_like_256d",
    device=DEVICE,
    pyramid_config=PyramidConfig(levels=[level], use_top_code_lm=True),
    top_transformer_config=top_lm,
    batch_size=8,
    sequence_length=2048,
    num_epochs=1,
)
