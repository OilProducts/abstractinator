from __future__ import annotations

# Thin wrappers to expose MLA self-attention blocks from the new path.

from .impl import CausalMLA as CausalMLA  # noqa: F401
from .impl import SlidingWindowMLATransformerBlock as SlidingWindowMLATransformerBlock  # noqa: F401
from .impl import CausalMLATransformerBlock as CausalMLATransformerBlock  # noqa: F401

__all__ = [
    "CausalMLA",
    "SlidingWindowMLATransformerBlock",
    "CausalMLATransformerBlock",
]
