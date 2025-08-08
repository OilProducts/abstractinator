from copy import deepcopy

from configs.base_config import (
    CompressorLevelConfig,
    ExpConfig,
    TopTransformerConfig,
)
from configs.base_config import (
    exp_config as _base_exp_config,
)

exp_config: ExpConfig = deepcopy(_base_exp_config)

# Extra-small settings for quick sanity checks
exp_config.compressor_level_configs = [
    CompressorLevelConfig(
        dim=128,
        heads=8,
        window=64,
        num_encoder_layers=1,
        encoder_ffn_dim_multiplier=4,
        max_seq_len_encoder=4096,
        num_queries=1,
        codebook_size=8192,
        beta=1.0,
        entropy_delta=0.2,
        entropy_abs_threshold=None,
        target_compression_ratio=None,
        compression_loss_weight=1.0,
    )
]
exp_config.expander_num_enc_layers = 1
exp_config.expander_num_dec_layers = 1
exp_config.aux_lm_loss_weight = 1.0
exp_config.top_lm_loss_weight = 1.0
exp_config.top_transformer_config = TopTransformerConfig(
    embed_dim=128,
    dim=256,
    num_layers=8,
    num_heads=8,
    ffn_dim_multiplier=4,
    continuous=True,
)
exp_config.learning_rate = 5e-4
exp_config.batch_size = 64
exp_config.gradient_accumulation_steps = 1
exp_config.dataset_name = "roneneldan/TinyStories"
exp_config.dataset_config = None
exp_config.sample_prompt_for_generation = "Once upon a time, in a land far, far away,"
