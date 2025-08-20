from copy import deepcopy

from configs import AbstractinatorConfig
from configs.base_config import (
    # CompressorLevelConfig,
    # ExpanderConfig,
    ExpConfig,
    TopTransformerConfig,
)
from configs.base_config import (
    exp_config as _base_exp_config,
)

# Use defaults from base_config without modification
exp_config = _base_exp_config #: ExpConfig = deepcopy(_base_exp_config)

exp_config.batch_size = 2 #16
exp_config.gradient_accumulation_steps = 1
exp_config.sequence_length = 2048


exp_config.num_epochs = 2
exp_config.num_levels = 2

layer_2 = AbstractinatorConfig(
    vocab_size=512**2,
    D = 256,

    # Compressor
    c_heads = 16,
    c_window = 128,
    # c_num_encoder_layers=4,
    c_num_shared_encoder_layers=0,
    c_num_lm_encoder_layers=14,
    c_num_compression_encoder_layers=4,
    c_num_queries=1,
    c_entropy_delta= 0.2,
    c_output_length=512,
    c_vq_K=512,
    c_vq_depth=2,
    c_vq_d_c=128,
    c_vq_beta=0.25,
    c_vq_reset_interval=250,

    # Expander/decoder
    d_layers=4,
    d_heads=16,
    d_cross_window=1,
    d_residual_conditioning=True,
    d_use_sqdist_logits=False,
    d_predict_specials=True,
    d_max_len=1024,  # Should be previous level's output length
    d_lo_d_c=128,

    # Loss weights
    w_code_ce= 1.0,
    w_special_ce= 1.0,
    w_byte_lm_ce= 1.0,
    w_vq=0.1,  # Weight for the VQ loss


)


exp_config.pyramid_config.levels.append(layer_2)

# exp_config.compressor_level_configs.append(
#     CompressorLevelConfig(
#         dim=256,
#         heads=16,
#         window=64,
#         head_dim=16,  # K
#         kv_comp_dim=64,  # d_c
#         q_comp_dim=96,  # d_c`
#         retr_dim=32,  # r
#         lm_window=64,
#         compression_window=8,
#         num_encoder_layers=0,
#         num_shared_encoder_layers=0,
#         num_lm_encoder_layers=14,
#         num_compression_encoder_layers=4,
#         encoder_ffn_dim_multiplier=2,
#         num_queries=1,
#         codebook_size=1024,
#         beta=1.0,
#         vq_reset_interval=250,
#         vq_depth=2,
#         entropy_delta=0.2,
#         entropy_abs_threshold=None,
#         target_compression_ratio=None,
#         compression_loss_weight=1.0,
#         output_length=384,
#     )
# )

# exp_config.expander_level_configs.append(
#     ExpanderConfig(
#         dim_scale=1.0,
#         num_enc_layers=2,
#         num_dec_layers=4,
#         heads_scale=1.0,
#         eos_id=1,
#         max_len=8192,
#         use_decoder_only=True,
#         use_continuous_inputs=True,
#         hi_dim=256,  # High-dimensional input for the expander
#         lo_dim=128,
#     )
# )

exp_config.top_transformer_config = TopTransformerConfig(
    embed_dim=256,
    dim=512,
    num_layers=12,
    num_heads=32,
    head_dim=16,  # K
    kv_comp_dim=64,  # d_c
    q_comp_dim=96,  # d_c`
    retr_dim=64,  # r
    continuous=True,  # When False, the top LM predicts discrete codes using cross-entropy
    mse_weight=1.0,  # Weight for the MSE component of the top LM loss
    ce_weight=1.0,  # Weight for the cross-entropy component of the top LM loss
    lm_window=1024,
)
