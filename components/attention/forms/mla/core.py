from __future__ import annotations

# Thin re-export of MLA core to align with the new layout.
# The implementation remains in components/mla.py

from .impl import MultiheadLatentAttention  # noqa: F401

__all__ = ["MultiheadLatentAttention"]
