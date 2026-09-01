"""Exact, state-isolated single-candidate Write counterfactuals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from ControllerIsolation import file_sha256
from Train.Inference import InferenceConfig, MemoryTreeInference


def _run(
    checkpoint: str | Path,
    sequence: Mapping[str, Any],
    *,
    allowlist: tuple[int, ...],
    device: str,
) -> dict[str, Any]:
    inference = MemoryTreeInference.from_checkpoint(
        checkpoint,
        device=device,
        inference_config=InferenceConfig(
            adapt_working_memory=True,
            allow_memory_writes=True,
            update_memory_usage=True,
            probe_write_counterfactuals=False,
            write_event_allowlist=allowlist,
        ),
    )
    inference.controller.split_enabled.fill_(False)
    return inference.run_sequence(sequence)


def paired_physical_write_utility(
    checkpoint: str | Path,
    sequence: Mapping[str, Any],
    event_index: int,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Compare no writes with exactly one normal physical write on C.

    Both branches load independently from the same checkpoint.  The treatment
    candidate follows the production F-window construction, add_memory call,
    sparse retrieval, path aggregation, Retrieve gate and Hawkes likelihood.
    """
    checkpoint = Path(checkpoint)
    before_sha = file_sha256(checkpoint)
    baseline = _run(checkpoint, sequence, allowlist=(), device=device)
    treatment = _run(
        checkpoint, sequence, allowlist=(int(event_index),), device=device
    )
    if file_sha256(checkpoint) != before_sha:
        raise AssertionError("physical Write probe modified its source checkpoint")
    h = int(
        MemoryTreeInference.from_checkpoint(checkpoint, device="cpu")
        .wake_config.write_horizon
    )
    start, end = int(event_index) + h, int(event_index) + 2 * h
    if end > len(baseline["events"]):
        raise ValueError("physical Write probe requires a complete C window")
    baseline_by_event = {
        int(event["event_index"]): event for event in baseline["events"]
    }
    treatment_by_event = {
        int(event["event_index"]): event for event in treatment["events"]
    }
    gains = [
        float(baseline_by_event[index]["nll"])
        - float(treatment_by_event[index]["nll"])
        for index in range(start, end)
    ]
    accepted = [
        event for event in treatment["events"]
        if bool(event.get("write_accepted", False))
    ]
    accepted_indices = [int(event["event_index"]) for event in accepted]
    if accepted_indices != [int(event_index)]:
        raise AssertionError(
            f"forced physical probe accepted {accepted_indices}, expected {[int(event_index)]}"
        )
    lambda_write = float(
        MemoryTreeInference.from_checkpoint(checkpoint, device="cpu")
        .wake_config.lambda_write
    )
    return {
        "event_index": int(event_index),
        "construction_window": [int(event_index), int(event_index) + h],
        "score_window": [start, end],
        "event_gains": gains,
        "mean_nll_gain": sum(gains) / h,
        "write_cost": lambda_write,
        "write_utility": sum(gains) / h - lambda_write,
        "accepted": True,
        "source_checkpoint_sha256": before_sha,
    }


def select_physical_probe_events(
    events: list[Mapping[str, Any]],
    *,
    horizon: int,
    seed: int,
    top_count: int = 16,
    random_count: int = 16,
) -> list[dict[str, Any]]:
    eligible = [
        event for event in events
        if int(event["event_index"]) + 2 * int(horizon) <= len(events)
    ]
    ranked = sorted(
        eligible,
        key=lambda event: float(
            event.get("raw_action_probabilities", event["action_probabilities"])[2]
        ),
        reverse=True,
    )
    top = ranked[:top_count]
    remaining = ranked[top_count:]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    count = min(int(random_count), len(remaining))
    chosen = torch.randperm(len(remaining), generator=generator)[:count].tolist()
    explored = [remaining[index] for index in chosen]
    return [
        {
            "event_index": int(event["event_index"]),
            "write_gate": float(
                event.get("raw_action_probabilities", event["action_probabilities"])[2]
            ),
            "top": event in top,
            "exploration": event in explored,
            "propensity": (
                1.0 if event in top else count / max(len(remaining), 1)
            ),
        }
        for event in [*top, *explored]
    ]
