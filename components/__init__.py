from .vector_quantizer import VectorQuantizer
from .learned_query_attention import LearnedQueryAttention
from .utils import token_entropy, entropy_segments, build_segment_queries_mask, safe_softmax

try:
    from .hierarchical_autoencoder import HierarchicalAutoencoder
    from .byte_segment_compressor import ByteSegmentCompressor
    from .swiglu import SwiGLU
    from .expander import DecoderOnlyExpander
    from .tokenizer import ByteLevelTokenizer
    from .checkpoint_utils import save_base_components, load_base_components
    from .code_sequence_transformer import CodeSequenceTransformer
except Exception:  # pragma: no cover - optional components may fail to import
    pass
