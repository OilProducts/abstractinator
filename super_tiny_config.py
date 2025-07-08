from copy import deepcopy
from base_config import DEVICE, N_CPU, exp_config as _base_exp_config

exp_config = deepcopy(_base_exp_config)

# Extra-small settings for quick sanity checks
exp_config.update({
    "compressor_level_configs": [
        {"dim": 128, "heads": 8, "window": 64,
         "num_encoder_layers": 1,
         "encoder_ffn_dim_multiplier": 4,
         "max_seq_len_encoder": 4096,
         "num_queries": 1,
         "codebook_size": 2048,
         "beta": 1.0,
         "entropy_delta": 0.2,
         "entropy_abs_threshold": None},
    ],
    "expander_num_enc_layers": 1,
    "expander_num_dec_layers": 1,
    "aux_lm_loss_weight": 0.1,
    "top_lm_loss_weight": 1.0,
    "top_transformer_config": {
        "embed_dim": 128,
        "dim": 128,
        "num_layers": 8,
        "num_heads": 8,
        "ffn_dim_multiplier": 4,
        "continuous": True,
    },
    "learning_rate": 5e-4,
    "gradient_accumulation_steps": 2,
    "dataset_name": "roneneldan/TinyStories",
    "dataset_config": None,
    "sample_prompt_for_generation": "Once upon a time, in a land far, far away,",
})
