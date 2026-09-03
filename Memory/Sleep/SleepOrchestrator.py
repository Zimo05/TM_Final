"""Single Light-to-Deep Sleep control-flow owner.

The trainer supplies the numerical phase implementations.  This module owns
their ordering and the lifecycle boundary: every completed Deep proposal
evaluation closes the inspection and resets the Deep gate, even when the
unified selector chooses Null.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Tensor


class SleepOrchestrator:
    """Thin state-machine facade for one complete Sleep cycle."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def run_full_sleep_cycle(
        self,
        responsibilities: Tensor,
        *,
        allow_topology_prune: bool = True,
        epoch: Optional[int] = None,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        epoch_label = (
            self.trainer.completed_epochs + 1
            if epoch is None else int(epoch)
        )
        result = self.trainer._run_full_sleep_cycle_impl(
            responsibilities,
            allow_topology_prune=allow_topology_prune,
            epoch=epoch_label,
            show_progress=show_progress,
        )

        evaluated = bool(result["deep_gate"]["evaluated"])
        triggered = bool(
            result["deep_gate"].get(
                "execution_requested",
                result["deep_gate"]["hard_gate"],
            )
        )
        # Light owns the frozen snapshot and the Bank-mode probe.  Keep the
        # diagnostic order aligned with the actual lifecycle rather than
        # reporting the gate before the work it gates.
        control_flow = [
            "freeze_snapshot",
            "bank_mode_probe",
            "light_or_preserve",
            "deep_gate",
        ]
        if evaluated:
            control_flow.extend((
                "build_candidates",
                "temporal_smoothing",
                "unified_selector",
            ))
            if triggered:
                control_flow.append("coordinator_commit")
            self.trainer.deep_sleep_gate.reset_after_deep()
            self.trainer.sleep_state["last_deep_epoch"] = epoch_label
            control_flow.append("reset_after_deep_evaluation")

        result["control_flow"] = control_flow
        result["deep_gate"]["reset_after_evaluation"] = evaluated
        return result


def run_full_sleep_cycle(
    trainer: Any,
    responsibilities: Tensor,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Public single entry for a complete trainer Sleep cycle."""
    return SleepOrchestrator(trainer).run_full_sleep_cycle(
        responsibilities,
        **kwargs,
    )


__all__ = ["SleepOrchestrator", "run_full_sleep_cycle"]
