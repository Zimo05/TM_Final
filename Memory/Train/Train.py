"""End-to-end Hawkes Memory Tree training pipeline.

This compatibility entry point re-exports the established training API while
composing the implementation from focused training modules.
"""

from __future__ import annotations

from Train.TrainingComponents import *  # noqa: F403
from Train.TrainingComponents import (
    _assert_finite_without_cuda_sync,
    _differentiable_merge_settings,
    _frontier_config_from_checkpoint,
)
from Train.TrainingCLI import (
    _leaf_spectral_radius_summary,
    _parse_args,
    _run_semantic_smoke_test,
    main,
)
from Train.TrainingTrainer import MemoryTreeTrainer


if __name__ == "__main__":
    main()
