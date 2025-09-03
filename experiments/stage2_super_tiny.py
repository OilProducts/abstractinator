from copy import deepcopy

from components.config_types import AbstractinatorConfig, TopTransformerConfig
from experiments.exp_config import ExpConfig
from experiments.exp_config import exp_config as _base_exp_config

exp_config: ExpConfig = deepcopy(_base_exp_config)

# Extra-small settings for quick sanity checks (Stage 2: train top LM over codes)
exp_config.pyramid_config.levels = [
    AbstractinatorConfig(
        vocab_size=260,
        D=128,
        c_heads=8,
        c_window=64,
        c_num_shared_encoder_layers=0,
        c_num_lm_encoder_layers=2,
        c_num_compression_encoder_layers=1,
        c_num_queries=1,
        c_entropy_delta=0.2,
        c_output_length=512,
        c_vq_K=8192,
        c_vq_depth=1,
        c_vq_d_c=64,
        c_vq_beta=0.25,
        c_vq_reset_interval=250,
        d_layers=1,
        d_heads=8,
        d_cross_window=1,
        d_residual_conditioning=True,
        d_use_sqdist_logits=False,
        d_predict_specials=False,
        d_max_len=1024,
        d_lo_d_c=64,
    )
]

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
