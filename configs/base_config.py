import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from torch.nn.attention.flex_attention import flex_attention

import torch

# Determine CPU count for data loading and processing
N_CPU = int(os.cpu_count()) if os.cpu_count() else 1  # Ensure at least one worker


# Determine the preferred device
def _check_device() -> str:
    """Return the preferred device for PyTorch operations."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Check if flex_attention can run on the current system
def _check_flex_attention() -> bool:
    """Return True if flex_attention can run on this system."""
    if not torch.cuda.is_available():
        return False
    try:
        q = torch.randn(1, 1, 1, 8, device="cuda")
        flex_attention(q, q, q)
        return True
    except Exception:
        return False


# Select the default floating-point dtype based on device capabilities
def _select_default_dtype() -> torch.dtype:
    """
    Choose the best‑supported floating‑point dtype for the current device.

    Order of preference
    -------------------
    1. bfloat16  – fastest on Ampere/Hopper GPUs that expose BF16 kernels.
    2. float16   – good fallback on other CUDA or MPS devices.
    3. float32   – always safe (CPU or any device without *any* reduced‑precision support).
    """
    # CUDA path
    if torch.cuda.is_available():
        # PyTorch ≥1.12 ships `torch.cuda.is_bf16_supported`; fall back to
        # capability sniffing if the symbol is missing (older builds).
        if getattr(torch.cuda, "is_bf16_supported", None):
            if torch.cuda.is_bf16_supported():  # A100, H100, etc.
                return torch.bfloat16
        else:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:  # sm80+ very likely BF16
                return torch.bfloat16
        return torch.float16  # any other CUDA card

    # Apple Silicon / Metal backend
    if torch.backends.mps.is_available():
        return torch.float16

    # Pure CPU
    return torch.float32


DEVICE = _check_device()
FLEX_ATTENTION = _check_flex_attention()
DEFAULT_DTYPE = _select_default_dtype()
torch.set_default_dtype(DEFAULT_DTYPE)


@dataclass
class CompressorLevelConfig:
    dim: int = 128
    heads: int = 8
    window: int = 64
    head_dim: Optional[int] = 16  # K
    kv_comp_dim: Optional[int] = 32  # d_c
    q_comp_dim: Optional[int] = 48  # d_c`
    retr_dim: Optional[int] = 32  # r
    lm_window: Optional[int] = 64
    compression_window: Optional[int] = 8
    num_encoder_layers: int = 0
    num_shared_encoder_layers: int = 0
    num_lm_encoder_layers: Optional[int] = 14
    num_compression_encoder_layers: Optional[int] = 4
    encoder_ffn_dim_multiplier: int = 2
    num_queries: int = 1
    codebook_size: int = 8192
    beta: float = 1.0
    vq_reset_interval: int = 250
    entropy_delta: float = 0.2
    entropy_abs_threshold: Optional[float] = None
    target_compression_ratio: Optional[List[float]] = None
    compression_loss_weight: float = 1.0
    output_length: int = 512

    def __post_init__(self) -> None:
        if self.lm_window is None:
            self.lm_window = self.window
        if self.compression_window is None:
            self.compression_window = self.window


@dataclass
class TopTransformerConfig:
    embed_dim: int = 128
    dim: int = 256
    num_layers: int = 24
    num_heads: int = 16
    ffn_dim_multiplier: int = 4
    continuous: bool = True  # When False, the top LM predicts discrete codes using cross-entropy
    mse_weight: float = 1.0  # Weight for the MSE component of the top LM loss
    ce_weight: float = 1.0  # Weight for the cross-entropy component of the top LM loss
    head_dim: Optional[int] = 32  # K
    kv_comp_dim: Optional[int] = 64  # d_c
    q_comp_dim: Optional[int] = 96  # d_c`
    retr_dim: Optional[int] = 32  # r
    lm_window: Optional[int] = 128
    lm_fixed_length: Optional[int] = 512 # Fixed length for the top LM input, will be padded if necessary
    lm_pad_id: int = 258


@dataclass
class ExpanderConfig:
    dim_scale: float = 1.0
    num_enc_layers: int = 2
    num_dec_layers: int = 4
    heads_scale: float = 1.0
    eos_id: int = 1
    max_len: int = 8192
    use_decoder_only: bool = True
    use_continuous_inputs: bool = True
    cross_window: Optional[int] = 128  # If None, use the same window as the compressor level
    hi_dim: int = 128 # Dimension of the high-level representation
    lo_dim: int = 128 # Dimension of the low-level representation



@dataclass
class ExpConfig:
    run_name: str = "HierarchicalAE_Default"
    project_name: str = "TemporalAutoencodedLanguageModelling"
    device: str = DEVICE
    flex_attention: bool = FLEX_ATTENTION
    num_levels: int = 1
    initial_vocab_size: int = 260
    compressor_level_configs: List[CompressorLevelConfig] = field(
        default_factory=lambda: [CompressorLevelConfig()]
    )
    expander_level_configs: List[ExpanderConfig] = field(default_factory=lambda: [ExpanderConfig()])
    aux_lm_loss_weight: float = 1.0
    top_lm_loss_weight: float = 1.0
    top_lm_mse_weight: float = 1.0
    top_lm_ce_weight: float = 1.0
    top_transformer_config: Optional[TopTransformerConfig] = field(
        default_factory=TopTransformerConfig
    )
    propagate_key_padding_mask: bool = True
    learning_rate: float = 5e-4
    batch_size: int = 16
    sequence_length: int = 1024
    num_epochs: int = 1
    max_steps: Optional[int] = None
    log_interval: int = 1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1 # 8
    scheduler_type: str = "cosine_with_min_lr"
    warmup_steps: int = 1000
    scheduler_specific_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"min_lr": 1e-6}
    )
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: Optional[str] = None
    dataset_train_split: str = "train"
    text_column_name: str = "text"
    generation_interval: int = 300
    sample_prompt_for_generation: str = "In a land far away, "
    generation_max_len_override: int = 128
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
