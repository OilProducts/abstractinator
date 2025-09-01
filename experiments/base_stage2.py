from copy import deepcopy

from experiments.exp_config import ExpConfig, exp_config as _base_exp_config
from components.config_types import AbstractinatorConfig, TopTransformerConfig


exp_config: ExpConfig = deepcopy(_base_exp_config)
exp_config.dataset_name = "HuggingFaceFW/fineweb-edu"
exp_config.dataset_config = "sample-10BT"
exp_config.dataset_train_split = "train"
exp_config.sample_prompt_for_generation = "The purpose of education is to"

exp_config.num_epochs = 2

# Two-level example
exp_config.pyramid_config.levels = [
    AbstractinatorConfig(
        D=256,
        c_heads=16,
        c_window=64,
        c_num_shared_encoder_layers=0,
        c_num_lm_encoder_layers=14,
        c_num_compression_encoder_layers=4,
        c_entropy_delta=0.2,
        c_output_length=384,
        c_vq_K=512,
        c_vq_depth=2,
        c_vq_d_c=128,
        d_layers=4,
        d_heads=16,
        d_cross_window=1,
        d_max_len=1024,
        d_lo_d_c=128,
    ),
    AbstractinatorConfig(
        D=256,
        c_heads=16,
        c_window=64,
        c_num_shared_encoder_layers=0,
        c_num_lm_encoder_layers=14,
        c_num_compression_encoder_layers=4,
        c_entropy_delta=0.2,
        c_output_length=384,
        c_vq_K=512,
        c_vq_depth=2,
        c_vq_d_c=128,
        d_layers=4,
        d_heads=16,
        d_cross_window=1,
        d_max_len=1024,
        d_lo_d_c=128,
    ),
]

exp_config.top_transformer_config = TopTransformerConfig(
    embed_dim=256,
    dim=512,
    num_layers=12,
    num_heads=32,
    head_dim=16,
    kv_comp_dim=64,
    q_comp_dim=96,
    retr_dim=64,
    continuous=True,
)
