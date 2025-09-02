from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AttentionConfig:
    # Variant: attention formulation
    variant: str = "mla"                 # "regular" | "mla"

    # Kernel: how to compute it
    kernel: str = "flex"                 # "sdpa" | "flex"

    use_rope: bool = True
    causal: bool = True
    window: Optional[int] = None         # sliding window radius (self-attn)
    lookback: Optional[int] = None       # segment cross-attn lookback

    # MLA-specific knobs (ignored for regular)
    head_dim: Optional[int] = None
    kv_comp_dim: Optional[int] = 64
    q_comp_dim: Optional[int] = 128
    retr_dim: Optional[int] = 32

@dataclass
class AbstractinatorConfig:
    # Shared / tokens
    vocab_size: int = 260
    D: int = 512
    eos_id: int = 257
    eop_id: int = 259
    bos_id: int = 256
    pad_id: int = 258

    # Compressor (encoder) hyperparams
    c_heads: int = 8
    c_window: int = 128

    c_head_dim: Optional[int] = 64  # if None, inferred as D / c_heads
    c_kv_comp_dim: Optional[int] = 64  # d_c; if None, inferred as D / 2
    c_q_comp_dim: Optional[int] = 128  # d_c`
    c_retr_dim: Optional[int] = 32  # r; if None, inferred as D / 4

    c_num_encoder_layers: int = 6
    c_num_shared_encoder_layers: int = 0
    c_num_lm_encoder_layers: Optional[int] = 4
    c_num_compression_encoder_layers: Optional[int] = None
    c_num_queries: int = 1
    c_entropy_delta: float = 0.2
    c_entropy_abs_threshold: Optional[float] = None
    c_output_length: int = 1024
    c_vq_K: int = 8192
    c_vq_depth: int = 1
    c_vq_d_c: Optional[int] = None
    c_vq_beta: float = 0.25
    c_vq_reset_interval: int = 250

    # Expander (decoder) hyperparams
    d_layers: int = 9
    d_heads: int = 8
    d_cross_window: int = 1
    d_residual_conditioning: bool = True
    d_use_sqdist_logits: bool = False
    d_predict_specials: bool = False
    d_max_len: int = 2048
    d_lo_d_c: int = 64

    # Loss weights (component-local defaults)
    w_code_ce: float = 1.0
    w_special_ce: float = 1.0
    w_byte_lm_ce: float = 1.0
    w_vq: float = 1.0

    # Device hint (string form only; experiments resolve actual device)
    device: Optional[str] = None

    # Attention configs (optional overrides to thread through to components)
    compressor_attention: Optional["AttentionConfig"] = None
    decoder_self_attention: Optional["AttentionConfig"] = None
    decoder_cross_attention: Optional["AttentionConfig"] = None


@dataclass
class TopTransformerConfig:
    embed_dim: int = 512
    dim: int = 512
    num_layers: int = 16
    num_heads: int = 8
    ffn_dim_multiplier: int = 4
    continuous: bool = False
    mse_weight: float = 1.0
    ce_weight: float = 1.0

    # MLA params (mirror compressor settings)
    head_dim: Optional[int] = 64
    kv_comp_dim: Optional[int] = 64
    q_comp_dim: Optional[int] = 128
    retr_dim: Optional[int] = 32
    lm_window: Optional[int] = 128
    lm_fixed_length: Optional[int] = 1024
    lm_pad_id: int = 258
    # Optional attention configuration for the top transformer
    attention_config: Optional["AttentionConfig"] = None


@dataclass
class PyramidConfig:
    levels: List[AbstractinatorConfig] = field(default_factory=lambda: [AbstractinatorConfig()])
    w_vq: float = 1.0
    w_byte_lm: float = 1.0
    use_top_code_lm: bool = True
