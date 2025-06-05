import math
from typing import Tuple

import torch
import torch.nn.functional as F

def safe_softmax(scores: torch.Tensor,
                 mask: torch.Tensor,
                 dim: int = -1) -> torch.Tensor:
    """
    Identical to soft‑max on `scores.masked_fill(mask, -inf)` *except*
    rows where everything is masked yield a zero vector (no NaNs).
    """
    scores = scores.masked_fill(mask, float("-inf"))

    # Identify rows that are completely −inf
    all_masked = torch.isneginf(scores).all(dim=dim, keepdim=True)

    # Replace −inf with 0 in such rows so exp() = 1 → softmax = 1/rowlen
    safe_scores = scores.masked_fill(all_masked, 0.0)

    attn = torch.softmax(safe_scores, dim=dim)

    # Bring the fully‑masked rows back to exact zeros
    attn = attn.masked_fill(all_masked, 0.0)
    return attn


def short_num(n):
    n = float(n)
    millnames = ['', 'k', 'm', 'b', 't', 'q']

    if n == 0:
        return '0'
    # Determine the appropriate suffix index
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(math.log10(abs(n)) / 3))))
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
    p = log_p.exp() # Equivalent to F.softmax(logits, dim=-1)

    # Calculate entropy: H(P) = - sum(p * log_p)
    # The product p * log_p is element-wise.
    # The sum is taken over the vocabulary dimension (the last dimension).
    # The negative sign makes the entropy non-negative.
    entropy = -(p * log_p).sum(dim=-1)  # Shape: (batch_size, sequence_length)

    return entropy

def entropy_segments(ent: torch.Tensor) -> torch.Tensor:
    """
    Generates segment IDs for sequences based on increasing per-token entropy.

    A new segment is initiated whenever the entropy of the current token is
    strictly greater than the entropy of the immediately preceding token.
    The first token of any sequence always starts segment 0.

    Args:
        ent (torch.Tensor): A tensor of per-token entropies.
            Expected shape: (batch_size, sequence_length).

    Returns:
        torch.Tensor: A tensor of segment IDs assigned to each token.
            Segment IDs are 0-indexed and increment within each sequence in the batch.
            Shape: (batch_size, sequence_length).
    """
    if ent.ndim != 2:
        raise ValueError(f"Input entropy tensor `ent` must be 2D (batch_size, sequence_length), but got shape {ent.shape}")
    if ent.size(1) == 0:
        return torch.empty_like(ent, dtype=torch.long) # Handle empty sequence case
    if ent.size(1) == 1:
        return torch.zeros_like(ent, dtype=torch.long) # Single token sequence is segment 0

    # Determine where entropy increases compared to the previous token.
    # ent[:, 1:] compares ent[t] with ent[t-1] for t from 1 to S-1.
    # Shape: (batch_size, sequence_length - 1)
    entropy_increased = (ent[:, 1:] > ent[:, :-1] + .2)

    # Convert boolean to integer (True -> 1, False -> 0)
    # These are indicators for starting a new segment (incrementing segment ID).
    inc_indicators = entropy_increased.int() # Shape: (batch_size, sequence_length - 1)

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

    return seg_id


def build_segment_queries_mask(
    seg_id: torch.Tensor,          # [B, S_original]   – integer segment ids
    query_embed: torch.Tensor,     # [L, D]            – learned query template
    num_heads: int                 # number of attention heads
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Same contract as before, but **without any .item() calls** so it can live
    inside `torch.compile` / FX graphs without forcing a graph break.
    """
    B, S_original = seg_id.shape          # SymInts
    L, D          = query_embed.shape
    device        = seg_id.device

    # ----------------------------------------------------------------------
    # 1.  Maximum #segments in the batch (scalar SymInt, NOT Python int)
    # ----------------------------------------------------------------------
    seg_count = seg_id.amax(dim=1) + 1          # (B,)  actual segments per item
    S_hat     = seg_count.amax()                # 0‑D tensor / SymInt ✔️

    # ----------------------------------------------------------------------
    # 2.  Build queries   [B, S_hat * L, D]
    # ----------------------------------------------------------------------
    #   Expand with SymInt S_hat – no Python conversion.
    queries = (
        query_embed                 # [L, D]
          .unsqueeze(0)             # [1, L, D]
          .unsqueeze(0)             # [1, 1, L, D]
          .expand(B, S_hat, L, D)   # [B, S_hat, L, D]
          .reshape(B, -1, D)        # [B, S_hat*L, D]
    )

    # ----------------------------------------------------------------------
    # 3.  Segment‑id for every query   [B, S_hat*L]
    # ----------------------------------------------------------------------
    # Build with arithmetic on SymInts – still no .item().
    q_positions = torch.arange(queries.shape[1], device=device)   # 0 … S_hat*L‑1
    seg_for_q   = (q_positions // L).expand(B, -1)                # repeat for batch

    # ----------------------------------------------------------------------
    # 4.  Attention mask   [B*num_heads, S_hat*L, S_original]   (True ⇒ BLOCK)
    # ----------------------------------------------------------------------
    same_segment = seg_for_q[:, :, None].eq(seg_id[:, None, :])   # broadcast compare
    att_mask     = (~same_segment).repeat_interleave(num_heads, dim=0)

    # ----------------------------------------------------------------------
    # 5.  Valid segment slots   [B, S_hat]
    # ----------------------------------------------------------------------
    valid_segments = (
        torch.arange(S_hat, device=device)      # 0 … S_hat‑1   (SymInt end)
              .unsqueeze(0)
              .lt(seg_count.unsqueeze(1))       # i < seg_count[b]
    )

    return queries, att_mask, valid_segments