import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
from torch.nn.attention.flex_attention import flex_attention

from components.config_types import PyramidConfig, TopTransformerConfig

# ----------------- Runtime / device helpers -----------------

N_CPU = int(os.cpu_count()) if os.cpu_count() else 1


def _check_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _check_flex_attention() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        q = torch.randn(1, 1, 1, 8, device="cuda")
        flex_attention(q, q, q)
        return True
    except Exception:
        print("Disabling flex_attention: it cannot run on this system.")
        return False


def _select_default_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if getattr(torch.cuda, "is_bf16_supported", None):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        else:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return torch.bfloat16
        return torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


DEVICE = _check_device()
FLEX_ATTENTION = _check_flex_attention()
DEFAULT_DTYPE = _select_default_dtype()
torch.set_default_dtype(DEFAULT_DTYPE)


# ----------------- Experiment-level configuration -----------------


@dataclass
class ExpConfig:
    run_name: str = "AbstractinatorBaseConfig"
    project_name: str = "Abstractinator"
    device: str = DEVICE
    flex_attention: bool = FLEX_ATTENTION
    num_levels: int = 1
    initial_vocab_size: int = 260

    pyramid_config: PyramidConfig = field(default_factory=lambda: PyramidConfig())
    aux_lm_loss_weight: float = 1.0
    top_lm_loss_weight: float = 1.0
    top_lm_mse_weight: float = 1.0
    top_lm_ce_weight: float = 1.0
    top_transformer_config: Optional[TopTransformerConfig] = field(default_factory=TopTransformerConfig)
    propagate_key_padding_mask: bool = True

    learning_rate: float = 5e-4
    batch_size: int = 8
    sequence_length: int = 2048
    num_epochs: int = 1
    max_steps: Optional[int] = None
    log_interval: int = 10
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    scheduler_type: str = "cosine_with_min_lr"
    warmup_steps: int = 1000
    scheduler_specific_kwargs: Dict[str, Any] = field(default_factory=lambda: {"min_lr": 1e-6})

    # dataset_name: str = "roneneldan/TinyStories"
    # dataset_config: Optional[str] = None
    # dataset_train_split: str = "train"
    text_column_name: str = "text"

    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_train_split: str = "train"
    sample_prompt_for_generation = "The purpose of education is to"

    # sample_prompt_for_generation: str = "In a land far away, "
    generation_interval: int = 100
    generation_max_len_override: int = 32

    checkpoint_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    save_base_components_path: Optional[str] = None
    mlflow_batch_interval: int = 50

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


# Default experiment config instance
exp_config = ExpConfig()
