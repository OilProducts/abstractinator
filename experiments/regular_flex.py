from __future__ import annotations

from experiments.exp_config import ExpConfig, DEVICE
from components.config_types import PyramidConfig, AbstractinatorConfig, TopTransformerConfig, AttentionConfig


# Attention: Regular form on FlexAttention backend
REGULAR_FLEX = AttentionConfig(
    variant="regular",
    kernel="flex",
)


# One level, 256‑wide, smallish; entropy model up to 12 layers; top LM up to 12 layers
level = AbstractinatorConfig(
    D=256,
    c_heads=8,
    c_window=64,
    c_num_encoder_layers=6,
    c_num_shared_encoder_layers=0,
    c_num_lm_encoder_layers=12,
    c_num_compression_encoder_layers=4,
    d_layers=6,
    d_heads=8,
    d_cross_window=1,
    d_max_len=512,
)
level.compressor_attention = REGULAR_FLEX
level.decoder_self_attention = REGULAR_FLEX
level.decoder_cross_attention = REGULAR_FLEX


top_lm = TopTransformerConfig(
    embed_dim=256,
    dim=256,
    num_layers=12,
    num_heads=8,
    attention_config=REGULAR_FLEX,
)


exp_config = ExpConfig(
    run_name="regular_flex_256d",
    device=DEVICE,
    pyramid_config=PyramidConfig(levels=[level], use_top_code_lm=True),
    top_transformer_config=top_lm,
    batch_size=16,
    sequence_length=2048,
    num_epochs=1,
    generation_interval = 500
)

