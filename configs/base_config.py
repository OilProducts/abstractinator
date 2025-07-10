import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch

# Determine CPU count for data loading and processing
N_CPU = int(os.cpu_count()) if os.cpu_count() else 1  # Ensure at least one worker

# Determine the preferred device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available() and DEVICE == "cpu":
    DEVICE = torch.device("mps")


@dataclass
class CompressorLevelConfig:
    dim: int = 768
    heads: int = 12
    window: int = 512
    num_encoder_layers: int = 6
    encoder_ffn_dim_multiplier: int = 4
    max_seq_len_encoder: int = 4096
    num_queries: int = 1
    codebook_size: int = 49404
    beta: float = 1.0
    entropy_delta: float = 0.2
    entropy_abs_threshold: Optional[float] = None
    target_compression_ratio: Optional[List[float]] = None
    compression_loss_weight: float = 1.0


@dataclass
class TopTransformerConfig:
    embed_dim: int = 768
    dim: int = 768
    num_layers: int = 16
    num_heads: int = 12
    ffn_dim_multiplier: int = 4
    continuous: bool = True


@dataclass
class ExpConfig:
    run_name: str = "HierarchicalAE_KPM_Run_v2"
    project_name: str = "TemporalAutoencodedLanguageModelling"
    num_levels: int = 1
    initial_vocab_size: int = 259
    compressor_level_configs: List[CompressorLevelConfig] = field(
        default_factory=lambda: [CompressorLevelConfig()]
    )
    expander_dim_scale: float = 1.0
    expander_num_enc_layers: int = 2
    expander_num_dec_layers: int = 4
    expander_heads_scale: float = 1.0
    expander_eos_id: int = 1
    expander_max_len: int = 2048
    use_decoder_only_expander: bool = True
    propagate_key_padding_mask: bool = True
    aux_lm_loss_weight: float = 1.0
    top_lm_loss_weight: float = 0.2
    top_transformer_config: Optional[TopTransformerConfig] = field(
        default_factory=TopTransformerConfig
    )
    learning_rate: float = 1e-4
    batch_size: int = 16
    sequence_length: int = 1024
    num_epochs: int = 1
    max_steps: Optional[int] = None
    log_interval: int = 1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 16
    scheduler_type: str = "cosine_with_min_lr"
    warmup_steps: int = 1000
    scheduler_specific_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"min_lr": 1e-6}
    )
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: Optional[str] = "sample-10BT"
    dataset_train_split: str = "train"
    text_column_name: str = "text"
    generation_interval: int = 50
    sample_prompt_for_generation: str = "The purpose of education is "
    generation_max_len_override: int = 512
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    save_base_components_path: Optional[str] = None
    use_continuous_expander_inputs: bool = False
    mlflow_batch_interval: int = 50

    def as_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        return asdict(self)

    # Provide minimal dict-like interface for backward compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


exp_config = ExpConfig()
