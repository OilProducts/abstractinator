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
    dim: int = 128
    heads: int = 4
    window: int = 128
    lm_window: Optional[int] = 128
    compression_window: Optional[int] = 16
    num_encoder_layers: int = 0
    num_shared_encoder_layers: int = 1
    num_lm_encoder_layers: Optional[int] = 8
    num_compression_encoder_layers: Optional[int] = 1
    encoder_ffn_dim_multiplier: int = 4
    max_seq_len_encoder: int = 4096
    num_queries: int = 1
    codebook_size: int = 8192
    beta: float = 1.0
    vq_reset_interval: int = 250
    entropy_delta: float = 0.2
    entropy_abs_threshold: Optional[float] = None
    target_compression_ratio: Optional[List[float]] = None
    compression_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.lm_window is None:
            self.lm_window = self.window
        if self.compression_window is None:
            self.compression_window = self.window


@dataclass
class TopTransformerConfig:
    embed_dim: int = 128
    dim: int = 256
    num_layers: int = 12
    num_heads: int = 8
    ffn_dim_multiplier: int = 4
    continuous: bool = False  # When False, the top LM predicts discrete codes using cross-entropy
    mse_weight: float = 1.0  # Weight for the MSE component of the top LM loss
    ce_weight: float = 1.0   # Weight for the cross-entropy component of the top LM loss


@dataclass
class ExpanderConfig:
    dim_scale: float = 1.0
    num_enc_layers: int = 2
    num_dec_layers: int = 4
    heads_scale: float = 1.0
    eos_id: int = 1
    max_len: int = 8192
    use_decoder_only: bool = True
    propagate_key_padding_mask: bool = True
    use_continuous_inputs: bool = False


@dataclass
class ExpConfig:
    run_name: str = "HierarchicalAE_Default"
    project_name: str = "TemporalAutoencodedLanguageModelling"
    num_levels: int = 1
    initial_vocab_size: int = 259
    compressor_level_configs: List[CompressorLevelConfig] = field(
        default_factory=lambda: [CompressorLevelConfig()]
    )
    expander: ExpanderConfig = field(default_factory=ExpanderConfig)
    aux_lm_loss_weight: float = 1.0
    top_lm_loss_weight: float = 1.0
    top_lm_mse_weight: float = 1.0
    top_lm_ce_weight: float = 1.0
    top_transformer_config: Optional[TopTransformerConfig] = field(
        default_factory=TopTransformerConfig
    )
    learning_rate: float = 5e-4
    batch_size: int = 4
    sequence_length: int = 4096
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
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: Optional[str] = None
    dataset_train_split: str = "train"
    text_column_name: str = "text"
    generation_interval: int = 50
    sample_prompt_for_generation: str = "In a land far away, "
    generation_max_len_override: int = 512
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    save_base_components_path: Optional[str] = None
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
