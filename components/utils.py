from __future__ import annotations

import functools
import math
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_softmax(
    scores: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1, eps: float = 1e-20
) -> torch.Tensor:
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Rowwise max; guard all -inf rows
    row_max = scores.max(dim=dim, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    scores = scores - row_max

    exp = torch.exp(scores)
    if mask is not None:
        exp = exp.masked_fill(mask, 0.0)

    denom = exp.sum(dim=dim, keepdim=True)
    return exp / torch.clamp_min(denom, eps)


# def safe_softmax(scores: torch.Tensor, mask: torch.Tensor | None, dim: int = -1) -> torch.Tensor:
#     # Apply provided mask if present
#     if mask is not None:
#         scores = scores.masked_fill(mask, float("-inf"))
#
#     # Always guard against fully-masked rows (all -inf), even if mask is None.
#     all_masked = torch.isneginf(scores).all(dim=dim, keepdim=True)
#
#     # Replace -inf with 0 so softmax's internal max-subtraction doesn't generate NaNs.
#     safe_scores = scores.masked_fill(all_masked, 0.0)
#
#     attn = torch.softmax(safe_scores, dim=dim)
#     return attn.masked_fill(all_masked, 0.0)


# def safe_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
#     """
#     Identical to soft‑max on `scores.masked_fill(mask, -inf)` *except*
#     rows where everything is masked yield a zero vector (no NaNs).
#     """
#     scores = scores.masked_fill(mask, float("-inf"))
#
#     # Identify rows that are completely −inf
#     all_masked = torch.isneginf(scores).all(dim=dim, keepdim=True)
#
#     # Replace −inf with 0 in such rows so exp() = 1 → softmax = 1/rowlen
#     safe_scores = scores.masked_fill(all_masked, 0.0)
#
#     attn = torch.softmax(safe_scores, dim=dim)
#
#     # Bring the fully‑masked rows back to exact zeros
#     attn = attn.masked_fill(all_masked, 0.0)
#     return attn

# def safe_softmax(scores: torch.Tensor, mask: torch.Tensor | None, dim: int = -1) -> torch.Tensor:
#     if mask is not None:
#         scores = scores.masked_fill(mask, float("-inf"))
#     return torch.softmax(scores, dim=dim) if mask is None else _safe_softmax_with_mask(scores, mask, dim)
#
# def _safe_softmax_with_mask(scores, mask, dim):
#     all_masked = torch.isneginf(scores).all(dim=dim, keepdim=True)
#     safe_scores = scores.masked_fill(all_masked, 0.0)
#     attn = torch.softmax(safe_scores, dim=dim)
#     return attn.masked_fill(all_masked, 0.0)


def short_num(n):
    n = float(n)
    millnames = ['', 'k', 'm', 'b', 't', 'q']

    if n == 0:
        return '0'
    # Determine the appropriate suffix index
    millidx = max(0, min(len(millnames) - 1, int(math.floor(math.log10(abs(n)) / 3))))
    # Scale the number down by the appropriate power of 1000
    scaled = n / 10 ** (3 * millidx)
    # Determine the number of decimal places based on the scaled value
    if scaled < 10:
        formatted = f"{scaled:.2f}"
    elif scaled < 100:
        formatted = f"{scaled:.1f}"
    else:
        formatted = f"{scaled:.0f}"
    return f"{formatted}{millnames[millidx]}"


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))

    # Compute days, hours, minutes, and seconds.
    days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
    minutes, secs = divmod(remainder, 60)  # 60 seconds in a minute

    parts = []
    # If there are days, show days, hours, and minutes (ignore seconds)
    if days > 0:
        parts.append(f"{days}d")
        parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
    # If there are hours but no days, show hours and minutes (ignore seconds)
    elif hours > 0:
        parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
    # If it's less than one hour, show minutes and seconds (or only seconds if under a minute)
    else:
        if minutes > 0:
            parts.append(f"{minutes}m")
            parts.append(f"{secs}s")
        else:
            parts.append(f"{secs}s")

    return " ".join(parts)


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Shannon entropy for each token's predictive distribution.

    Given a tensor of logits, this function first computes the normalized
    probabilities (softmax), then calculates the entropy for each token's
    distribution over the vocabulary.

    Entropy is a measure of uncertainty. A higher entropy indicates a more
    uniform (less certain) predictive distribution, while a lower entropy
    indicates a more peaked (more certain) distribution.

    Args:
        logits (torch.Tensor): The raw, unnormalized predictions (logits) from a model.
            Expected shape: (batch_size, sequence_length, vocab_size).

    Returns:
        torch.Tensor: A tensor containing the entropy for each token in the input sequences.
            Shape: (batch_size, sequence_length).
    """
    # Apply log_softmax along the vocabulary dimension (last dimension)
    # log_p will have the same shape as logits: (batch_size, sequence_length, vocab_size)
    log_p = F.log_softmax(logits, dim=-1)

    # Convert log probabilities to probabilities
    # p will also have the shape: (batch_size, sequence_length, vocab_size)
    p = log_p.exp()  # Equivalent to F.softmax(logits, dim=-1)

    # Calculate entropy: H(P) = - sum(p * log_p)
    # The product p * log_p is element-wise.
    # The sum is taken over the vocabulary dimension (the last dimension).
    # The negative sign makes the entropy non-negative.
    entropy = -(p * log_p).sum(dim=-1)  # Shape: (batch_size, sequence_length)

    return entropy


def entropy_segments(
    ent: torch.Tensor,
    *,
    increase_delta: float = 0.2,
    abs_threshold: float | None = None,
    return_boundary: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Generate segment IDs based on token entropy.

    A new segment begins when either:

    1. The current token's entropy exceeds ``abs_threshold`` (if provided).
    2. The entropy increased by more than ``increase_delta`` compared to the
       previous token.

    The first token of each sequence always starts segment ``0``.

    Args:
        ent: Tensor of per-token entropies ``(B, S)``.
        increase_delta: Minimum increase from the previous token required to
            start a new segment.
        abs_threshold: Optional absolute entropy level that also triggers a new
            segment when crossed.
        return_boundary: If ``True``, also return a boolean mask indicating the
            end of each segment (True at the last token of each segment).

    Returns:
        seg_id          : (B,S₀)  int64
        patch_end_mask? : (B,S₀)  bool  (True at last token of each segment) IDs ``(B, S)``.
    """
    # increase_delta = 0  # TODO: remove this
    if ent.ndim != 2:
        raise ValueError(
            f"Input entropy tensor `ent` must be 2D (batch_size, sequence_length), but got shape {ent.shape}"
        )
    if ent.size(1) == 0:
        seg_id = torch.empty_like(ent, dtype=torch.long)
        if return_boundary:
            return seg_id, torch.empty_like(ent, dtype=torch.bool)
        return seg_id
    if ent.size(1) == 1:
        seg_id = torch.zeros_like(ent, dtype=torch.long)  # Single token sequence is segment 0
        if return_boundary:
            return seg_id, torch.ones_like(ent, dtype=torch.bool)
        return seg_id

    # Determine where entropy increases compared to the previous token.
    # ent[:, 1:] compares ent[t] with ent[t-1] for t from 1 to S-1.
    # Shape: (batch_size, sequence_length - 1)
    entropy_increased = ent[:, 1:] > ent[:, :-1] + increase_delta
    if abs_threshold is not None:
        entropy_increased |= ent[:, 1:] > abs_threshold

    # Convert boolean to integer (True -> 1, False -> 0)
    # These are indicators for starting a new segment (incrementing segment ID).
    inc_indicators = entropy_increased.int()  # Shape: (batch_size, sequence_length - 1)

    # Pad the indicators at the beginning of the sequence (dim=1, left side).
    # The first token effectively has no preceding token to compare against,
    # so its indicator is 0. This ensures the cumsum starts the segment ID at 0
    # for the first token.
    # Pad with (1 element on left, 0 elements on right) for the last dimension (sequence dimension).
    # `inc` will have shape: (batch_size, sequence_length)
    inc = F.pad(inc_indicators, (1, 0), mode='constant', value=0)

    # Perform a cumulative sum along the sequence dimension.
    # This creates the segment IDs. Each time `inc` is 1, the segment ID increments.
    seg_id = torch.cumsum(inc, dim=1)  # Shape: (batch_size, sequence_length)

    if return_boundary:
        # seg_id differs at patch borders
        patch_end = torch.zeros_like(seg_id, dtype=torch.bool)
        if seg_id.size(1) > 0:  # guard empty seq
            patch_end[:, :-1] = seg_id[:, 1:] != seg_id[:, :-1]
            patch_end[:, -1] = True
        return seg_id, patch_end
    else:
        return seg_id


def _compiled(module: nn.Module):
    # dynamic=True handles S/L changes; instance-local cache avoids shape collisions
    return torch.compile(module, dynamic=True)


def make_seg_mask_fn(seg_id: torch.Tensor, L: int) -> Callable:
    """
    Returns a closure (b, h, q, k) → bool that implements
    segment‑restricted masking.

    seg_id : (B,S₀)  int64
    L      : queries per segment
    """
    seg_id = seg_id.contiguous()  # keep a private copy

    def mask(b: int, h: int, q_idx: torch.Tensor, k_idx: torch.Tensor) -> torch.Tensor:
        """
        q_idx, k_idx: broadcast‑compatible tensors
        Returns True where attention is **blocked**.
        """
        seg_k = seg_id[b][k_idx]  # ✓ broadcasting, same shape as k_idx
        return (q_idx // L) != seg_k  # result shape (Q,S)

    return mask


@functools.lru_cache(maxsize=32)
def _cached_tiled_template(L: int, D: int, S_hat: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Returns an **UNINITIALISED** tensor of shape (Ŝ·L, D) *without* batch dim.
    This buffer is *never* written to; it exists only to avoid
    reallocating GPU memory every forward pass.
    """
    return torch.empty((S_hat * L, D), device=device, dtype=dtype)


def get_tiled_queries(base_queries: torch.Tensor, B: int, S_hat: int) -> torch.Tensor:
    """
    Parameters
    ----------
    base_queries : Tensor, shape (L, D)
        Learnable parameter that changes every optimiser step.
    B           : int
        Current batch size.
    S_hat       : int
        Number of *segments per query* (how many times to repeat each row).

    Returns
    -------
    Tensor, shape (B, Ŝ·L, D), contiguous
        Ready to be fed into attention.
    """
    L, D = base_queries.shape
    dev, dt = base_queries.device, base_queries.dtype

    # ---- 1. Grab **template** buffer from the cache ------------------------
    template = _cached_tiled_template(L, D, S_hat, dev, dt)

    # ---- 2. Clone it → fresh storage (fast; no data copy yet) -------------
    tiled = template.clone()  # (Ŝ·L, D), version = 0

    # ---- 3. Fill with the current parameter values ------------------------
    tiled.copy_(base_queries.repeat_interleave(S_hat, 0))

    # ---- 4. Add batch dim and broadcast -----------------------------------
    return tiled.unsqueeze(0).expand(B, -1, -1).contiguous()


# utils_segment_queries.py (or wherever you keep helpers)



@torch.no_grad()
def build_segment_queries_qseg(
    seg_id: torch.Tensor,  # [B, S] int; token -> segment id
    query_embed: torch.Tensor,  # [L, D] learned template
    *,
    Q_max: int,  # constant total #queries per item
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build segment-aware queries *without* a dense (B·H,Q,S) mask.

    Returns
    -------
    queries : (B, Q_max, D)
    q_seg   : (B, Q_max) int   (segment id for each query; -1 for padded queries)
    valid_segments_mask : (B, S_hat_cap) bool where S_hat_cap = Q_max // L
    """
    B, S = seg_id.shape
    L, D = query_embed.shape
    device = seg_id.device

    # Enforce Q_max multiple of L (compile-stable shapes)
    S_hat_cap = max(1, int(Q_max // L))
    Q_max = int(S_hat_cap * L)

    # Per-sample #segments
    seg_count = seg_id.amax(dim=1) + 1  # [B], >=1

    # (1) Tile the L learned queries across S_hat_cap segment slots
    #     Shape is constant: (B, S_hat_cap, L, D) -> (B, Q_max, D)
    queries = (
        query_embed.unsqueeze(0)
        .unsqueeze(0)  # [1,1,L,D]
        .expand(B, S_hat_cap, L, D)  # [B,S_hat_cap,L,D]
        .reshape(B, Q_max, D)  # [B,Q_max,D]
        .contiguous()
    )

    # (2) q_seg: segment id per query (0..S_hat_cap-1 for real slots, -1 for invalid)
    q_pos = torch.arange(Q_max, device=device)  # [Q_max]
    seg_idx_per_q = q_pos // L  # [Q_max] 0..S_hat_cap-1
    q_seg = seg_idx_per_q.unsqueeze(0).expand(B, Q_max).clone()  # [B,Q_max]
    # mark queries that belong to non-existent segments
    invalid_q = seg_idx_per_q.unsqueeze(0) >= seg_count.unsqueeze(1)  # [B,Q_max]
    q_seg[invalid_q] = -1

    # (3) valid segments per item: [B, S_hat_cap]
    valid_segments_mask = torch.arange(S_hat_cap, device=device).unsqueeze(0) < seg_count.unsqueeze(1)

    return queries, q_seg, valid_segments_mask


def build_segment_queries_mask(
    seg_id: torch.Tensor,  # [B, S_original]   – integer segment ids
    query_embed: torch.Tensor,  # [L, D]            – learned query template
    num_heads: int,  # number of attention heads
    *,
    Q_max: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct segment-aware queries and attention masks.

    This helper expands a set of ``L`` learned query vectors for every possible
    segment in ``seg_id`` and builds an attention mask that restricts each query
    to attend only to its corresponding segment.  All computations use symbolic
    ``torch`` shapes so the function is compatible with ``torch.compile``.

    Args:
        seg_id: Integer tensor of shape ``(B, S_original)`` mapping each token
            to a segment index.
        query_embed: Base query tensor of shape ``(L, D)``.
        num_heads: Number of attention heads used by the pooler.

    Returns:
        Tuple containing:
            - ``queries``: Tiled queries of shape ``(B, S_hat * L, D)`` where
              ``S_hat`` is the maximum number of segments in the batch.
            - ``att_mask``: Boolean mask of shape
              ``(B * num_heads, S_hat * L, S_original)`` where ``True`` blocks
              attention outside a query's segment.
            - ``valid_segments``: Boolean tensor of shape ``(B, S_hat)`` marking
              which segment slots are real for each item in the batch.
    """
    B, S_original = seg_id.shape
    L, D = query_embed.shape
    device = seg_id.device

    seg_count = seg_id.amax(dim=1) + 1  # (B,)
    if Q_max is None:
        # --- dynamic (old) path ---
        S_hat = seg_count.amax()
        queries = (
            query_embed.unsqueeze(0)
            .unsqueeze(0)  # [1,1,L,D]
            .expand(B, S_hat, L, D)  # [B,Ŝ,L,D]
            .reshape(B, -1, D)  # [B,Ŝ·L,D]
        )
        q_positions = torch.arange(queries.shape[1], device=device)
        seg_for_q = (q_positions // L).expand(B, -1)  # [B, Ŝ·L]
        same_segment = seg_for_q[:, :, None].eq(seg_id[:, None, :])
        att_mask = (~same_segment).repeat_interleave(num_heads, dim=0)
        valid_segments = torch.arange(S_hat, device=device).unsqueeze(0).lt(seg_count.unsqueeze(1))
        return queries, att_mask, valid_segments

    # --- fixed-shape path ---
    S_hat_cap = max(1, int(Q_max // L))
    Q_max = int(S_hat_cap * L)  # enforce multiple of L

    # Queries: [B, S_hat_cap*L, D] with constant shape
    queries = (
        query_embed.unsqueeze(0)
        .unsqueeze(0)  # [1,1,L,D]
        .expand(B, S_hat_cap, L, D)  # [B,Ŝ_cap,L,D]
        .reshape(B, Q_max, D)  # [B,Q_max,D]
        .contiguous()
    )

    # Valid segments per item: [B, S_hat_cap]
    valid_segments = torch.arange(S_hat_cap, device=device).unsqueeze(0).lt(seg_count.unsqueeze(1))
    # Valid queries per item: [B, Q_max]
    q_valid = valid_segments.unsqueeze(-1).expand(-1, -1, L).reshape(B, Q_max)

    # Attention mask (True = block), constant shape [B*H, Q_max, S]
    seg_for_q = (torch.arange(Q_max, device=device) // L).view(1, -1, 1)  # [1,Q_max,1]
    same_segment = seg_for_q.eq(seg_id[:, None, :])  # [B,Q_max,S]
    att_mask = (~same_segment).repeat_interleave(num_heads, dim=0)  # [B*H,Q_max,S]
    # Also block *all* keys for padded queries so they become no-ops
    # att_mask |= (~q_valid)[:, :, None]
    att_mask |= (~q_valid).repeat_interleave(num_heads, dim=0)[:, :, None]

    return queries, att_mask, valid_segments
