from .learned_query_attention import LearnedQueryAttention as LearnedQueryAttention
from .utils import (
    build_segment_queries_mask as build_segment_queries_mask,
)
from .utils import (
    entropy_segments as entropy_segments,
)
from .utils import (
    safe_softmax as safe_softmax,
)
from .utils import (
    token_entropy as token_entropy,
)
from .vector_quantizer import VectorQuantizer as VectorQuantizer

__all__ = [
    "LearnedQueryAttention",
    "build_segment_queries_mask",
    "entropy_segments",
    "safe_softmax",
    "token_entropy",
    "VectorQuantizer",
]

try:
    from .byte_segment_compressor import ByteSegmentCompressor as ByteSegmentCompressor
    from .checkpoint_utils import (
        load_base_components as load_base_components,
    )
    from .checkpoint_utils import (
        save_base_components as save_base_components,
    )
    from .code_sequence_transformer import CodeSequenceTransformer as CodeSequenceTransformer
    from .expander import DecoderOnlyExpander as DecoderOnlyExpander
    from .hierarchical_autoencoder import HierarchicalAutoencoder as HierarchicalAutoencoder
    from .swiglu import SwiGLU as SwiGLU
    from .tokenizer import ByteLevelTokenizer as ByteLevelTokenizer

    __all__ += [
        "ByteSegmentCompressor",
        "load_base_components",
        "save_base_components",
        "CodeSequenceTransformer",
        "DecoderOnlyExpander",
        "HierarchicalAutoencoder",
        "SwiGLU",
        "ByteLevelTokenizer",
    ]
except ImportError:  # pragma: no cover - optional components may fail to import
    # Optional heavy deps (FlexAttention/CUDA) may be unavailable at import time.
    # Import of core symbols above remains usable.
    pass
