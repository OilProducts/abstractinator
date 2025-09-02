from __future__ import annotations

# Thin wrapper to expose SDPA segment cross-attention under the forms/regular path.

from ...sdpa.adapter import SDPASegmentCrossAttention as SDPASegmentCrossAttention  # noqa: F401

__all__ = ["SDPASegmentCrossAttention"]

