from .learned_query_attention import LearnedQueryAttention
from .utils import build_segment_queries_mask, entropy_segments, safe_softmax, token_entropy
from .vector_quantizer import VectorQuantizer

try:
    from .byte_segment_compressor import ByteSegmentCompressor
    from .checkpoint_utils import load_base_components, save_base_components
    from .code_sequence_transformer import CodeSequenceTransformer
    from .expander import DecoderOnlyExpander
    from .hierarchical_autoencoder import HierarchicalAutoencoder
    from .swiglu import SwiGLU
    from .tokenizer import ByteLevelTokenizer
except Exception:  # pragma: no cover - optional components may fail to import
    pass
