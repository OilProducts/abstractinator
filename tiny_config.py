from copy import deepcopy
from base_config import DEVICE, N_CPU, exp_config as _base_exp_config

exp_config = deepcopy(_base_exp_config)

# Overrides for smaller toy experiments
exp_config.update({
    "compressor_level_configs": [
        {"dim": 128, "heads": 4, "window": 128,
         "num_encoder_layers": 4,
         "encoder_ffn_dim_multiplier": 4,
         "max_seq_len_encoder": 4096,
         "num_queries": 1,
         "codebook_size": 1024,
         "beta": 1.0,
         "entropy_delta": 0.2,
         "entropy_abs_threshold": None},
    ],
    "top_transformer_config": {
        "embed_dim": 128,
        "dim": 256,
        "num_layers": 8,
        "num_heads": 8,
        "ffn_dim_multiplier": 4,
        "continuous": True,
    },
    "batch_size": 64,
    "gradient_accumulation_steps": 2,
    "dataset_name": "roneneldan/TinyStories",
    "dataset_config": None,
    "sample_prompt_for_generation": "Once upon a time, in a land far, far away,",
})
