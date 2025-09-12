from __future__ import annotations

from components.config_types import (
    AbstractinatorConfig,
    AttentionConfig,
    PyramidConfig,
    TopTransformerConfig,
    EntropyModelConfig,
    EntropyLogitHeadConfig,
    EntropyGaussianHeadConfig,
    EntropyLossConfig,
    EntropySegmentationConfig,
)
from experiments.exp_config import DEVICE, ExpConfig

# Attention: Regular form on FlexAttention backend
REGULAR_FLEX = AttentionConfig(
    variant="regular",
    kernel="flex",
)

# Explicit entropy model: ~50M params (D=512, 12 layers, tied logit + 1x Gaus

# One level, 256‑wide, smallish; entropy model up to 12 layers; top LM up to 12 layers
level = AbstractinatorConfig(
    D=512,
    c_heads=16,
    c_window=128,
    c_num_encoder_layers=6,
    c_num_entropy_encoder_layers=12,
    c_num_compression_encoder_layers=4,
    c_entropy_load_path="./models/entropy_stack_512.pt",
    c_entropy_freeze=True,
    c_vq_d_c=512,
    d_lo_d_c=512,
    d_layers=6,
    d_heads=16,
    d_cross_window=1,
    d_max_len=2048,
    d_use_standard_vq=True,

)
level.compressor_attention = REGULAR_FLEX
level.decoder_self_attention = REGULAR_FLEX
level.decoder_cross_attention = REGULAR_FLEX

# Configure entropy model used by the compressor's segmentation stack
level.c_entropy_config = EntropyModelConfig(
    n_layers=12,
    n_heads=16,
    window=128,
    attention_config=REGULAR_FLEX,
    logit=EntropyLogitHeadConfig(use=True, tie_to_embedding=True),
    gaussian=EntropyGaussianHeadConfig(
        use=True,
        clamp_logvar_min=-6.0,
        clamp_logvar_max=6.0,
        extra_layers=0,
        detach_trunk=False,
    ),
    loss=EntropyLossConfig(ce_weight=1.0, nll_weight=0.25),
    segmentation=EntropySegmentationConfig(source="logit", temperature=1.0, top_p=1.0, top_k=0),
)


top_lm = TopTransformerConfig(
    embed_dim=512,
    dim=512,
    num_layers=12,
    num_heads=16,
    attention_config=REGULAR_FLEX,
)


exp_config = ExpConfig(
    run_name="regular_flex_512d",
    device=DEVICE,
    pyramid_config=PyramidConfig(levels=[level], use_top_code_lm=False),
    top_transformer_config=None,
    batch_size=8,
    gradient_accumulation_steps=8,
    sequence_length=4096,
    num_epochs=1,
    generation_interval=500,
    learning_rate=1e-4,
)
