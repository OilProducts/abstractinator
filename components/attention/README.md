Attention System Overview

This package implements attention as two orthogonal axes:

- Form axis (what the attention computes):
  - regular: standard scaled dot-product attention over Q/K/V with optional masking
  - mla: Multi-Head Latent Attention (DeepSeek-style), which projects to compressed/value spaces and a retrieval space

- Backend axis (how the attention is executed):
  - sdpa: PyTorch scaled_dot_product_attention (routes to Flash/ME/Math)
  - flex: torch.nn.attention.flex_attention (block masks + score_mod)

Directory layout

- forms/
  - regular/
    - full_self.py        – causal self-attention over full history (backend selectable)
    - sliding_self.py     – causal self-attention with a local sliding window (backend selectable)
    - segment_cross.py    – cross-attention with segment lookback (backend selectable)
    - self_sdpa.py        – SDPA implementation used by full/sliding (internal)
    - sliding_sdpa.py     – SDPA sliding implementation (internal)
    - cross_segment_sdpa.py – SDPA cross-attention implementation (internal)
    - flex_self.py        – Flex full/sliding implementations (internal)
    - cross_segment_flex.py – Flex cross-attention (internal)

  - mla/
    - full_self.py        – causal self-attention block (wraps CausalMLATransformerBlock; backend→use_flex)
    - sliding_self.py     – sliding-window self-attention block (wraps SlidingWindowMLA; backend→use_flex)
    - segment_cross.py    – segment lookback cross-attention (wraps MLASegmentCrossAttention; backend→use_flex)
    - impl.py             – MLA core and concrete blocks (internal)

- backends/
  - sdpa.py              – SDPA runner wrapper
  - flex.py              – FlexAttention runner wrapper

Factory

Use components/attention/factory.py to construct attention blocks. It composes the Form (regular|mla) with the Backend (sdpa|flex) based on AttentionConfig.

Naming and responsibilities

- Filenames encode the operation only (full_self, sliding_self, segment_cross). Backend is not in the filename.
- Each file exposes a single class with a clear constructor:
  - regular/full_self.FullSelf(d_model, n_heads, ffn_dim_multiplier=4, backend='sdpa'|'flex')
  - regular/sliding_self.SlidingSelf(d_model, n_heads, window_size, ffn_dim_multiplier=4, backend='sdpa'|'flex')
  - regular/segment_cross.SegmentCross(q_dim, kv_dim, d_attn, n_heads, lookback=0, backend='sdpa'|'flex')
  - mla/full_self.FullSelf(..., backend='flex'|'fallback')
  - mla/sliding_self.SlidingSelf(..., backend='flex'|'fallback')
  - mla/segment_cross.SegmentCross(..., backend='flex'|'fallback')

Internally, each regular operation selects its backend implementation (SDPA or Flex) and forwards the call. MLA operations map backend to use_flex_attention inside their wrapped modules.

Why this design

- Orthogonality: Form (regular vs mla) and Backend (sdpa vs flex) are independent choices.
- Clarity: Paths name the Form, filenames name the Operation; backend is a parameter.
- Flexibility: Adding a new backend or form doesn’t require renaming files or changing call sites—only adding an implementation and mapping in the factory.

