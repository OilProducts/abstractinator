import os
import torch

# Determine CPU count for data loading and processing
N_CPU = int(os.cpu_count()) if os.cpu_count() else 1  # Ensure at least one worker

# Determine the preferred device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available() and DEVICE == "cpu":
    DEVICE = torch.device("mps")

# Experiment configuration dictionary
exp_config = {
    "run_name": "HierarchicalAE_KPM_Run_v2",
    "project_name": "TemporalAutoencodedLanguageModelling",
    "num_levels": 1,
    "initial_vocab_size": 259,
    "compressor_level_configs": [
        {"dim": 768, "heads": 12, "window": 512,
         "num_encoder_layers": 6,
        'encoder_ffn_dim_multiplier': 4,
         'max_seq_len_encoder': 4096,
         "num_queries": 1,
        "codebook_size": 49404,
        "beta": 1.0,
        "entropy_delta": 0.2,
        "entropy_abs_threshold": None},
    ],
    "expander_dim_scale": 1.0,
    "expander_num_enc_layers": 2,
    "expander_num_dec_layers": 4,
    "expander_heads_scale": 1.0,
    "expander_eos_id": 1,
    "expander_max_len": 2048,
    "propagate_key_padding_mask": True,
    "aux_lm_loss_weight": 1.0,
    "top_lm_loss_weight": 0.2,
    "top_transformer_config": {
        "dim": 768,
        "num_layers": 16,
        "num_heads": 12,
        "ffn_dim_multiplier": 4,
        "max_seq_len": 2048,
        "output_lm_logits": True
    },
    "learning_rate": 1e-4,
    "batch_size": 16,
    "sequence_length": 1024,
    "num_epochs": 1,
    "max_steps": None,
    "log_interval": 1,
    "gradient_clip_norm": 1.0,
    "gradient_accumulation_steps": 16,
    "scheduler_type": "cosine_with_min_lr",
    "warmup_steps": 1000,
    "scheduler_specific_kwargs": {
        "min_lr": 1e-6,  # Minimum learning rate for cosine scheduler
    },
    "dataset_name": "HuggingFaceFW/fineweb-edu",
    "dataset_config": "sample-10BT",
    "dataset_train_split": "train",
    "text_column_name": "text",
    "generation_interval": 50,
    "sample_prompt_for_generation": "The purpose of education is ",
    "generation_max_len_override": 512,
    "checkpoint_interval": 1000,
    "checkpoint_dir": "./checkpoints",
    "resume_from_checkpoint": None,
}
