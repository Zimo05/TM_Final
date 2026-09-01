"""Composition root for the modular memory-tree trainer."""

from __future__ import annotations

from Train.TrainingCheckpoint import TrainingCheckpointMixin
from Train.TrainingLifecycle import TrainingLifecycleMixin
from Train.TrainingLikelihood import TrainingLikelihoodMixin
from Train.TrainingLoop import TrainingLoopMixin
from Train.TrainingObjectives import TrainingObjectivesMixin
from Train.TrainingSleep import TrainingSleepMixin
from Train.TrainingWake import TrainingWakeMixin
from Train.TrainingWakeSupport import TrainingWakeSupportMixin


class MemoryTreeTrainer(
    TrainingLifecycleMixin,
    TrainingWakeSupportMixin,
    TrainingWakeMixin,
    TrainingLikelihoodMixin,
    TrainingObjectivesMixin,
    TrainingSleepMixin,
    TrainingLoopMixin,
    TrainingCheckpointMixin,
):
    """Optimize streaming wake updates and periodic sleep consolidation."""


__all__ = ["MemoryTreeTrainer"]
