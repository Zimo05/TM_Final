"""Cross-sequence probation for persistent episodic-memory writes.

Local Write evidence creates a candidate, but a candidate remains invisible to
normal retrieval until independent sequences establish a positive lower
confidence bound on its reuse utility.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Hashable, Optional

from torch import Tensor


@dataclass
class ProbationCandidate:
    """A locally useful residual awaiting independent-sequence validation."""

    token: tuple[Hashable, int]
    source_sequence_id: Hashable
    event_index: int
    owner_id: str
    key: Tensor
    delta_theta: Tensor
    window: Any
    local_utility: float
    local_quality: float
    write_probability: float
    split_probability: float
    # q_struct = p_split. This is metadata only until promotion; it must not
    # contribute to Controller.split_queues or any Sleep evidence beforehand.
    queue_weight: float
    weight_sum: float = 0.0
    weight_square_sum: float = 0.0
    gain_mean: float = 0.0
    gain_m2: float = 0.0
    validated_sequence_ids: set[Hashable] = field(default_factory=set)

    def can_validate_on(self, sequence_id: Hashable) -> bool:
        return (
            sequence_id != self.source_sequence_id
            and sequence_id not in self.validated_sequence_ids
        )

    def record_validation(
        self,
        sequence_id: Hashable,
        *,
        gain: float,
        weight: float,
    ) -> bool:
        """Add one signed, soft-weighted independent-sequence observation."""
        if not self.can_validate_on(sequence_id):
            return False
        if not math.isfinite(gain):
            raise ValueError("cross-sequence gain must be finite")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("validation weight must be finite and non-negative")

        # A sequence is consumed even when its soft weight underflows to zero;
        # it must never be replayed as a second independent sample.
        self.validated_sequence_ids.add(sequence_id)
        if weight == 0.0:
            return True

        old_weight = self.weight_sum
        new_weight = old_weight + weight
        difference = gain - self.gain_mean
        new_mean = self.gain_mean + weight * difference / new_weight
        self.gain_m2 += weight * difference * (gain - new_mean)
        self.gain_mean = new_mean
        self.weight_sum = new_weight
        self.weight_square_sum += weight * weight
        return True

    @property
    def gain_variance(self) -> float:
        if self.weight_sum <= 0.0:
            return 0.0
        return max(self.gain_m2 / self.weight_sum, 0.0)

    @property
    def effective_sample_size(self) -> float:
        if self.weight_square_sum <= 0.0:
            return 0.0
        return self.weight_sum * self.weight_sum / self.weight_square_sum

    def lower_confidence_bound(self, kappa: float) -> float:
        if kappa < 0.0:
            raise ValueError("LCB kappa must be non-negative")
        effective = self.effective_sample_size
        if effective <= 0.0:
            return float("-inf")
        return self.gain_mean - kappa * math.sqrt(
            self.gain_variance / effective
        )

    def promotion_ready(
        self,
        *,
        minimum_effective_samples: float,
        kappa: float,
        persist_threshold: float,
    ) -> bool:
        return (
            self.effective_sample_size >= minimum_effective_samples
            and self.lower_confidence_bound(kappa) > persist_threshold
        )

    def promoted_quality(self, gain_reference: float) -> float:
        if gain_reference <= 0.0:
            raise ValueError("gain_reference must be positive")
        return 1.0 - math.exp(-max(self.gain_mean, 0.0) / gain_reference)


class WriteProbationBuffer:
    """Bounded insertion-ordered collection of probation candidates."""

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("probation capacity must be positive")
        self.capacity = int(capacity)
        self._candidates: OrderedDict[
            tuple[Hashable, int], ProbationCandidate
        ] = OrderedDict()

    def __len__(self) -> int:
        return len(self._candidates)

    def __iter__(self):
        return iter(tuple(self._candidates.values()))

    def get(
        self, token: tuple[Hashable, int]
    ) -> Optional[ProbationCandidate]:
        return self._candidates.get(token)

    def add(self, candidate: ProbationCandidate) -> None:
        if candidate.token in self._candidates:
            self._candidates[candidate.token] = candidate
            self._candidates.move_to_end(candidate.token)
            return
        if len(self._candidates) >= self.capacity:
            self._candidates.popitem(last=False)
        self._candidates[candidate.token] = candidate

    def remove(
        self, token: tuple[Hashable, int]
    ) -> Optional[ProbationCandidate]:
        return self._candidates.pop(token, None)

    def candidates_for(self, sequence_id: Hashable) -> list[ProbationCandidate]:
        return [
            candidate
            for candidate in self._candidates.values()
            if candidate.can_validate_on(sequence_id)
        ]
