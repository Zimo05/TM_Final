"""Shared similarity-derived features for episodic-memory control.

The local recurrence count is intentionally independent of the number of
occupied rows in a memory bank.  Similarity below ``similarity_low`` is a
non-match, while similarity above ``similarity_high`` contributes a saturated
match.  A smoothstep transition keeps the feature continuous for controller
training.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def local_recurrence_count(
    similarity: Tensor,
    valid_mask: Optional[Tensor] = None,
    *,
    similarity_low: float = 0.35,
    similarity_high: float = 0.65,
    exponent: float = 2.0,
    topk: Optional[int] = None,
    saturation: float = 3.0,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Compute a compact-support local recurrence count.

    Args:
        similarity: Similarities with memory rows.  The final dimension is
            reduced, so both ``[R]`` and batched ``[..., R]`` inputs work.
        valid_mask: Optional boolean/0-1 mask for padded memory rows.
        similarity_low/high: Bounds of the smooth compact-support transition.
        exponent: Positive-match sharpening exponent.
        topk: Optional future sparsification limit. ``None`` intentionally
            sums every valid row, including the full 128-capacity bank.
        saturation: Count scale in ``1 - exp(-support / saturation)``.
        eps: Numerical tolerance used when validating positive scales.

    Returns:
        ``(normalized_count, local_support)`` with the final similarity
        dimension removed.
    """
    if not isinstance(similarity, Tensor):
        similarity = torch.as_tensor(similarity)
    if similarity.ndim == 0:
        raise ValueError("similarity must have a final memory-row dimension")
    if not similarity.is_floating_point():
        similarity = similarity.float()

    if not similarity_low < similarity_high:
        raise ValueError("similarity_low must be less than similarity_high")
    if not -1.0 <= similarity_low < similarity_high <= 1.0:
        raise ValueError(
            "similarity thresholds must satisfy -1 <= low < high <= 1"
        )
    if exponent < 1.0:
        raise ValueError("exponent must be at least 1")
    if saturation <= 0.0:
        raise ValueError("saturation must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if topk is not None and (
        isinstance(topk, bool)
        or not isinstance(topk, int)
        or topk <= 0
    ):
        raise ValueError("topk must be a positive integer or None")

    if valid_mask is not None:
        if not isinstance(valid_mask, Tensor):
            valid_mask = torch.as_tensor(valid_mask)
        if valid_mask.shape != similarity.shape:
            raise ValueError(
                "valid_mask must have the same shape as similarity"
            )
        valid = valid_mask.to(device=similarity.device, dtype=torch.bool)
        # Padded rows may contain arbitrary values in a future mirror. Remove
        # them before the nonlinear transform so they can never contribute.
        similarity = torch.where(valid, similarity, torch.zeros_like(similarity))
    else:
        valid = None

    x = (
        (similarity - float(similarity_low))
        / float(similarity_high - similarity_low)
    ).clamp(0.0, 1.0)
    match = x.square() * (3.0 - 2.0 * x)
    match = match.pow(float(exponent))
    if valid is not None:
        match = match * valid.to(match.dtype)

    if topk is not None and match.size(-1) > topk:
        # Kept as an opt-in switch for a later experiment. Production's
        # default is topk=None, i.e. the complete capacity is summed.
        match = torch.topk(match, k=topk, dim=-1).values

    local_support = match.sum(dim=-1)
    scale = max(float(saturation), float(eps))
    normalized_count = 1.0 - torch.exp(-local_support / scale)
    return normalized_count, local_support
