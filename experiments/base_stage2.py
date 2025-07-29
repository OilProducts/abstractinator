from copy import deepcopy
from configs.base_config import DEVICE, N_CPU, exp_config as _base_exp_config, ExpConfig, CompressorLevelConfig, \
    ExpanderConfig

# Use defaults from base_config without modification
exp_config: ExpConfig = deepcopy(_base_exp_config)

exp_config.top_transformer_config = None
exp_config.num_epochs = 2
exp_config.compressor_level_configs.append(
    CompressorLevelConfig(
        dim=256,
        heads=16,
        window=64,
        head_dim=16,  # K
        kv_comp_dim=64,  # d_c
        q_comp_dim=96,  # d_c`
        retr_dim=32,  # r
        lm_window=64,
        compression_window=8,
        num_encoder_layers=0,
        num_shared_encoder_layers=0,
        num_lm_encoder_layers=14,
        num_compression_encoder_layers=4,
        encoder_ffn_dim_multiplier=2,
        num_queries=1,
        codebook_size=262144,
        beta=1.0,
        vq_reset_interval=250,
        entropy_delta=0.2,
        entropy_abs_threshold=None,
        target_compression_ratio=None,
        compression_loss_weight=1.0,
    ))

exp_config.expander_level_configs.append(
    ExpanderConfig(
        dim_scale=1.0,
        num_enc_layers=2,
        num_dec_layers=4,
        heads_scale=1.0,
        eos_id=1,
        max_len=8192,
        use_decoder_only=True,
        use_continuous_inputs=False,
    ))
