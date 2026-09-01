"""Calibrate only the Write threshold with isolated validation rollouts.

The source checkpoint is immutable.  Retrieve and Adapt parameters, buffers,
and thresholds are hashed before and after derivation.  Policy selection uses
the realized full-online NLL rather than the local Write probe utility.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ControllerIsolation import (
    controller_item_sha256,
    file_sha256,
    head_policy_sha256,
    protected_write_only_items,
    tensor_items_sha256,
)
from DataSplit import load_split_manifest, select_sequences
from Evaluate import aggregate_metrics, bootstrap_ci, load_dataset
from Train.Inference import InferenceConfig, MemoryTreeInference


WRITE_THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 1.01)
MIN_ONLINE_GAIN = 1e-6


def event_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    stable = [{
        key: row[key]
        for key in (
            "source_index", "event_index", "nll", "true_type",
            "predicted_type_at_event_time", "type_probabilities",
        )
    } for row in rows]
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validation_subset(
    dataset: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    validation = select_sequences(dataset, manifest, "validation")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for sequence in validation:
        grouped[int(sequence["cluster_id"])].append(sequence)
    selected = [
        sequence
        for cluster in sorted(grouped)
        for sequence in sorted(
            grouped[cluster], key=lambda item: int(item["source_index"])
        )[:5]
    ]
    if len(grouped) != 13 or any(len(rows) < 5 for rows in grouped.values()):
        raise ValueError("Write rollout calibration requires 5 validation rows in each of 13 clusters")
    if len(selected) != 65:
        raise AssertionError(f"expected 65 validation sequences, got {len(selected)}")
    return selected


def _row(source: int, event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_index": source,
        "event_index": int(event["event_index"]),
        "nll": float(event["nll"]),
        "true_type": int(event["true_type"]),
        "predicted_type_at_event_time": int(event["predicted_type"]),
        "type_probabilities": [
            float(value) for value in event["type_probabilities_at_event_time"]
        ],
        "true_time": float(event["true_time"]),
        "predicted_time": float(event["predicted_time"]),
        "write_accepted": bool(event.get("write_accepted", False)),
    }


def run_policy(
    checkpoint: Path,
    sequences: Sequence[Mapping[str, Any]],
    *,
    write_threshold: float,
    online: bool,
    device: str,
) -> list[dict[str, Any]]:
    inference = MemoryTreeInference.from_checkpoint(
        checkpoint,
        device=device,
        inference_config=InferenceConfig(
            adapt_working_memory=True,
            allow_memory_writes=online,
            update_memory_usage=online,
            probe_write_counterfactuals=False,
        ),
    )
    existing = inference.controller.calibration_thresholds.detach().cpu().tolist()
    inference.controller.set_calibration_thresholds(
        float(existing[1]), float(existing[0]), float(write_threshold)
    )
    inference.controller.split_enabled.fill_(False)
    rows: list[dict[str, Any]] = []
    label = f"online@{write_threshold:.2f}" if online else "frozen"
    for position, sequence in enumerate(sequences, 1):
        result = inference.run_sequence(sequence)
        source = int(sequence["source_index"])
        rows.extend(_row(source, event) for event in result["events"])
        print(f"[WriteRollout] {label} {position}/{len(sequences)}", flush=True)
    return rows


def paired_rollout_metrics(
    frozen: Sequence[Mapping[str, Any]],
    online: Sequence[Mapping[str, Any]],
    *,
    num_types: int,
    sequence_count: int,
    seed: int,
) -> dict[str, Any]:
    frozen_by_key = {
        (int(row["source_index"]), int(row["event_index"])): row for row in frozen
    }
    online_by_key = {
        (int(row["source_index"]), int(row["event_index"])): row for row in online
    }
    keys = sorted(set(frozen_by_key) & set(online_by_key))
    if len(keys) != len(frozen_by_key) or len(keys) != len(online_by_key):
        raise RuntimeError("frozen and online validation events do not align")
    gains = [
        float(frozen_by_key[key]["nll"]) - float(online_by_key[key]["nll"])
        for key in keys
    ]
    frozen_metrics = aggregate_metrics(frozen, num_types, seed, 1000)
    online_metrics = aggregate_metrics(online, num_types, seed, 1000)
    interval = bootstrap_ci(gains, seed, samples=1000)
    physical_writes = sum(bool(row.get("write_accepted", False)) for row in online)
    event_count = max(len(gains), 1)
    gain = sum(gains) / event_count
    lower = float(interval[0]) if interval else float("-inf")
    harmful = sum(value < 0.0 for value in gains) / event_count
    writes_per_sequence = physical_writes / max(sequence_count, 1)
    feasible = bool(
        gain > MIN_ONLINE_GAIN
        and lower > 0.0
        and harmful < 0.45
        and writes_per_sequence <= 2.0
        and online_metrics["accuracy"]
        >= frozen_metrics["accuracy"] - 1.0 / event_count
        and online_metrics["macro_f1"]
        >= frozen_metrics["macro_f1"] - 0.001
    )
    return {
        "mean_nll_gain": gain,
        "bootstrap_95ci": interval,
        "improved_event_fraction": sum(value > 0.0 for value in gains) / event_count,
        "harmful_event_fraction": harmful,
        "physical_accepted_count": physical_writes,
        "writes_per_sequence": writes_per_sequence,
        "frozen_nll": frozen_metrics["nll_per_event"],
        "full_online_nll": online_metrics["nll_per_event"],
        "frozen_accuracy": frozen_metrics["accuracy"],
        "full_online_accuracy": online_metrics["accuracy"],
        "frozen_macro_f1": frozen_metrics["macro_f1"],
        "full_online_macro_f1": online_metrics["macro_f1"],
        "minimum_effect_size": MIN_ONLINE_GAIN,
        "feasible": feasible,
    }


def paired_policy_improvement(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Paired candidate improvement over the immutable online baseline."""
    baseline_by_key = {
        (int(row["source_index"]), int(row["event_index"])): row
        for row in baseline
    }
    candidate_by_key = {
        (int(row["source_index"]), int(row["event_index"])): row
        for row in candidate
    }
    keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    if len(keys) != len(baseline_by_key) or len(keys) != len(candidate_by_key):
        raise RuntimeError("baseline and candidate validation events do not align")
    gains = [
        float(baseline_by_key[key]["nll"])
        - float(candidate_by_key[key]["nll"])
        for key in keys
    ]
    interval = bootstrap_ci(gains, seed, samples=1000)
    mean_gain = sum(gains) / max(len(gains), 1)
    num_types = len(baseline[0].get("type_probabilities", ())) if baseline else 0
    baseline_metrics = (
        aggregate_metrics(baseline, num_types, seed, 1000) if num_types else {}
    )
    candidate_metrics = (
        aggregate_metrics(candidate, num_types, seed, 1000) if num_types else {}
    )
    classification_passed = bool(
        not num_types
        or (
            candidate_metrics["accuracy"]
            >= baseline_metrics["accuracy"] - 1.0 / max(len(gains), 1)
            and candidate_metrics["macro_f1"]
            >= baseline_metrics["macro_f1"] - 0.001
        )
    )
    return {
        "mean_nll_gain": mean_gain,
        "bootstrap_95ci": interval,
        "baseline_accuracy": baseline_metrics.get("accuracy"),
        "candidate_accuracy": candidate_metrics.get("accuracy"),
        "baseline_macro_f1": baseline_metrics.get("macro_f1"),
        "candidate_macro_f1": candidate_metrics.get("macro_f1"),
        "classification_passed": classification_passed,
        "passed": bool(
            mean_gain > MIN_ONLINE_GAIN
            and len(interval) >= 2
            and float(interval[0]) >= 0.0
            and classification_passed
        ),
    }


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    dataset, type_map = load_dataset(args.data_path)
    manifest = load_split_manifest(args.split_manifest, data_path=args.data_path)
    validation = validation_subset(dataset, manifest)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    module_state = source["controller_state"]["module_state_dict"]
    original_thresholds = torch.as_tensor(
        module_state["calibration_thresholds"]
    ).detach().cpu().clone()
    protected_before = protected_write_only_items(module_state)
    protected_sha = tensor_items_sha256(protected_before)
    retrieve_sha = head_policy_sha256(module_state, "retrieve")
    adapt_sha = head_policy_sha256(module_state, "adapt")

    frozen = run_policy(
        args.checkpoint, validation, write_threshold=1.01,
        online=False, device=args.device,
    )
    table: list[dict[str, Any]] = []
    for threshold in WRITE_THRESHOLDS:
        online = run_policy(
            args.checkpoint, validation, write_threshold=threshold,
            online=True, device=args.device,
        )
        metrics = paired_rollout_metrics(
            frozen, online, num_types=len(type_map),
            sequence_count=len(validation), seed=args.seed,
        )
        table.append({"write_threshold": threshold, **metrics})
        print(json.dumps(table[-1], indent=2), flush=True)

    feasible = [row for row in table if row["feasible"]]
    selected = (
        max(feasible, key=lambda row: (row["mean_nll_gain"], row["write_threshold"]))
        if feasible
        else next(row for row in table if row["write_threshold"] == 1.01)
    )
    output = copy.deepcopy(source)
    output_state = output["controller_state"]["module_state_dict"]
    thresholds = torch.as_tensor(output_state["calibration_thresholds"]).clone()
    thresholds[2] = float(selected["write_threshold"])
    output_state["calibration_thresholds"] = thresholds
    output_state["split_enabled"] = torch.zeros_like(
        torch.as_tensor(output_state["split_enabled"]), dtype=torch.bool
    )
    calibration = {
        **output.get("controller_calibration", {}),
        "adapt_threshold": float(original_thresholds[0]),
        "retrieve_threshold": float(original_thresholds[1]),
        "write_threshold": float(selected["write_threshold"]),
        "calibration_method": "realized_validation_rollout",
        "validation_data_sha256": manifest["data_sha256"],
        "frozen_event_sha256": event_rows_sha256(frozen),
        "frozen_source_indices": [int(row["source_index"]) for row in validation],
    }
    metadata = {
        "controller_policy_revision": 3,
        "calibration_method": "realized_validation_rollout",
        "base_checkpoint": str(args.checkpoint),
        "base_checkpoint_sha256": file_sha256(args.checkpoint),
        "validation_sequence_count": len(validation),
        "validation_data_sha256": manifest["data_sha256"],
        "threshold_search_table": table,
        "selected_threshold": float(selected["write_threshold"]),
        "selected_metrics": selected,
        "constraint_passed": bool(feasible),
        "requires_write_only_finetune": not bool(feasible),
        "frozen_controller_item_sha256": {
            name: controller_item_sha256(module_state).get(name)
            for name in controller_item_sha256(module_state)
            if name != "calibration_thresholds"
        },
        "protected_write_only_sha256": protected_sha,
        "retrieve_policy_sha256": retrieve_sha,
        "adapt_policy_sha256": adapt_sha,
        "inactive_controller_heads_verified": True,
    }
    output["controller_policy_revision"] = 3
    output["controller_calibration"] = calibration
    output["write_rollout_calibration"] = metadata
    output["controller_state"]["calibration"] = calibration
    output["controller_state"]["split_enabled"] = False
    output["controller_state"]["controller_policy_revision"] = 3
    output["checkpoint_identity"] = {
        "path": str(args.output.resolve()), "role": "write_rollout_calibrated"
    }
    protected_after = protected_write_only_items(output_state)
    if tensor_items_sha256(protected_after) != protected_sha:
        raise AssertionError("Write rollout calibration changed Retrieve/Adapt/Split state")
    if head_policy_sha256(output_state, "retrieve") != retrieve_sha:
        raise AssertionError("Retrieve policy changed during Write calibration")
    if head_policy_sha256(output_state, "adapt") != adapt_sha:
        raise AssertionError("Adapt policy changed during Write calibration")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(output, temporary)
    temporary.replace(args.output)
    report = args.output.with_suffix(".json")
    report.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Write threshold with realized validation rollouts"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(calibrate(args), indent=2))


if __name__ == "__main__":
    main()
