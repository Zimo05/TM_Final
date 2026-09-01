"""Derive a validation-calibrated Controller checkpoint without training.

The source checkpoint is never modified.  All counterfactual passes are
state-isolated and physical writes/usage updates are disabled.
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

from DataSplit import load_split_manifest, select_sequences
from Evaluate import clear_episodic_memory, load_dataset
from Train.Inference import InferenceConfig, MemoryTreeInference


ACTION_COST = 1e-4


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_digest(items: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(items):
        value = torch.as_tensor(items[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validation_subset(
    dataset: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    validation = select_sequences(dataset, manifest, "validation")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for sequence in validation:
        grouped[int(sequence["cluster_id"])].append(sequence)
    selected = [
        row
        for cluster in sorted(grouped)
        for row in sorted(grouped[cluster], key=lambda item: int(item["source_index"]))[:5]
    ]
    if len(grouped) != 13 or any(len(rows) < 5 for rows in grouped.values()):
        raise ValueError("validation calibration requires 13 clusters with at least 5 rows each")
    if len(selected) != 65:
        raise AssertionError(f"expected 65 validation sequences, got {len(selected)}")
    return selected


def _run(
    checkpoint: Path,
    sequences: Sequence[Mapping[str, Any]],
    *,
    device: str,
    episodic: bool,
    working: bool,
    thresholds: tuple[float, float, float],
    write_probe: bool = False,
    seed: int = 42,
) -> dict[tuple[int, int], dict[str, Any]]:
    inference = MemoryTreeInference.from_checkpoint(
        checkpoint,
        device=device,
        inference_config=InferenceConfig(
            adapt_working_memory=working,
            allow_memory_writes=False,
            update_memory_usage=False,
            probe_write_counterfactuals=write_probe,
            write_probe_random_count=16,
            write_probe_seed=seed,
        ),
    )
    inference.controller.set_calibration_thresholds(*thresholds)
    inference.controller.split_enabled.fill_(False)
    if not episodic:
        clear_episodic_memory(inference)
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for position, sequence in enumerate(sequences, 1):
        result = inference.run_sequence(sequence)
        source = int(sequence["source_index"])
        for event in result["events"]:
            rows[(source, int(event["event_index"]))] = event
        print(f"[Recalibrate] {position}/{len(sequences)}", flush=True)
    return rows


def _grid(start: int) -> list[float]:
    return [value / 100.0 for value in range(start, 100, 5)] + [1.01]


def _joint_ra_search(
    semantic: Mapping[tuple[int, int], Mapping[str, Any]],
    retrieve: Mapping[tuple[int, int], Mapping[str, Any]],
    adapt: Mapping[tuple[int, int], Mapping[str, Any]],
    both: Mapping[tuple[int, int], Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    keys = sorted(set(semantic) & set(retrieve) & set(adapt) & set(both))
    if not keys:
        raise RuntimeError("counterfactual validation passes have no aligned events")
    table: list[dict[str, Any]] = []
    for tau_r in _grid(10):
        for tau_a in _grid(10):
            total_gain = 0.0
            retrieve_utilities: list[float] = []
            adapt_utilities: list[float] = []
            for key in keys:
                raw = both[key].get(
                    "raw_action_probabilities", both[key]["action_probabilities"]
                )
                use_a = float(raw[0]) >= tau_a
                use_r = float(raw[1]) >= tau_r
                chosen = both if use_a and use_r else adapt if use_a else retrieve if use_r else semantic
                chosen_nll = float(chosen[key]["nll"])
                total_gain += float(semantic[key]["nll"]) - chosen_nll
                if use_r:
                    no_r = adapt if use_a else semantic
                    retrieve_utilities.append(float(no_r[key]["nll"]) - chosen_nll - ACTION_COST)
                if use_a:
                    no_a = retrieve if use_r else semantic
                    adapt_utilities.append(float(no_a[key]["nll"]) - chosen_nll - ACTION_COST)
            r_sum = sum(retrieve_utilities)
            a_sum = sum(adapt_utilities)
            r_harm = sum(value < 0.0 for value in retrieve_utilities) / max(len(retrieve_utilities), 1)
            a_harm = sum(value < 0.0 for value in adapt_utilities) / max(len(adapt_utilities), 1)
            feasible = (
                total_gain >= 0.0
                and r_sum >= 0.0
                and a_sum >= 0.0
                and r_harm <= 0.45
                and a_harm <= 0.45
            )
            table.append({
                "retrieve_threshold": tau_r,
                "adapt_threshold": tau_a,
                "memory_gain_sum": total_gain,
                "retrieve_utility_sum": r_sum,
                "adapt_utility_sum": a_sum,
                "retrieve_harmful_fraction": r_harm,
                "adapt_harmful_fraction": a_harm,
                "retrieve_selected_count": len(retrieve_utilities),
                "adapt_selected_count": len(adapt_utilities),
                "feasible": feasible,
            })
    feasible = [row for row in table if row["feasible"]]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["memory_gain_sum"],
                row["retrieve_threshold"] + row["adapt_threshold"],
            ),
        )
    else:
        selected = next(
            row for row in table
            if row["retrieve_threshold"] == 1.01 and row["adapt_threshold"] == 1.01
        )
        selected = {**selected, "fallback_actions_closed": True}
    return selected, table


def _write_search(
    rows: Mapping[tuple[int, int], Mapping[str, Any]], sequence_count: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probes = []
    for key, event in rows.items():
        if not event.get("write_probed") or event.get("write_utility") is None:
            continue
        raw = event.get("raw_action_probabilities", event["action_probabilities"])
        probes.append({
            "source_index": key[0],
            "event_index": key[1],
            "gate": float(raw[2]),
            "utility": float(event["write_utility"]),
            "propensity": max(float(event.get("write_probe_propensity", 1.0)), 1e-6),
            "top": bool(event.get("write_probe_top", False)),
        })
    positives = sum(row["utility"] > 0.0 for row in probes)
    negatives = len(probes) - positives
    table = []
    for threshold in _grid(50):
        selected = [row for row in probes if row["gate"] >= threshold]
        objective = sum(
            row["utility"] * min(10.0, 1.0 / row["propensity"])
            for row in selected
        )
        harmful = sum(row["utility"] <= 0.0 for row in selected) / max(len(selected), 1)
        eligible = [row for row in selected if row["top"] and row["utility"] > 0.0]
        by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            by_source[row["source_index"]].append(row)
        accepted = sum(min(4, len(values)) for values in by_source.values())
        writes_per_sequence = accepted / max(sequence_count, 1)
        feasible = (
            positives >= 20
            and negatives >= 20
            and objective >= 0.0
            and harmful <= 0.45
            and writes_per_sequence <= 2.0
            and bool(selected)
        )
        table.append({
            "write_threshold": threshold,
            "ipw_utility_sum": objective,
            "harmful_fraction": harmful,
            "selected_count": len(selected),
            "accepted_count": accepted,
            "writes_per_sequence": writes_per_sequence,
            "feasible": feasible,
        })
    feasible = [row for row in table if row["feasible"]]
    selected = (
        max(feasible, key=lambda row: (row["ipw_utility_sum"], row["write_threshold"]))
        if feasible
        else next(row for row in table if row["write_threshold"] == 1.01)
    )
    return {
        **selected,
        "sample_count": len(probes),
        "positive_count": positives,
        "negative_count": negatives,
        "fallback_action_closed": not bool(feasible),
    }, table


def recalibrate(args: argparse.Namespace) -> dict[str, Any]:
    dataset, _ = load_dataset(args.data_path)
    manifest = load_split_manifest(args.split_manifest, data_path=args.data_path)
    validation = _validation_subset(dataset, manifest)
    closed = (1.01, 1.01, 1.01)
    semantic = _run(args.checkpoint, validation, device=args.device, episodic=False, working=False, thresholds=closed, seed=args.seed)
    retrieve = _run(args.checkpoint, validation, device=args.device, episodic=True, working=False, thresholds=(0.0, 1.01, 1.01), seed=args.seed)
    adapt = _run(args.checkpoint, validation, device=args.device, episodic=False, working=True, thresholds=(1.01, 0.0, 1.01), seed=args.seed)
    both = _run(args.checkpoint, validation, device=args.device, episodic=True, working=True, thresholds=(0.0, 0.0, 1.01), seed=args.seed)
    ra, ra_table = _joint_ra_search(semantic, retrieve, adapt, both)
    thresholds = (float(ra["retrieve_threshold"]), float(ra["adapt_threshold"]), 0.0)
    write_rows = _run(args.checkpoint, validation, device=args.device, episodic=True, working=True, thresholds=thresholds, write_probe=True, seed=args.seed)
    write, write_table = _write_search(write_rows, len(validation))

    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    output = copy.deepcopy(source)
    module_state = output["controller_state"]["module_state_dict"]
    inference = MemoryTreeInference.from_checkpoint(args.checkpoint, device="cpu")
    parameter_names = {name for name, _ in inference.controller.named_parameters()}
    parameter_state = {name: module_state[name] for name in parameter_names}
    parameter_sha = _tensor_digest(parameter_state)
    selected = {
        "retrieve_threshold": float(ra["retrieve_threshold"]),
        "adapt_threshold": float(ra["adapt_threshold"]),
        "write_threshold": float(write["write_threshold"]),
    }
    module_state["calibration_thresholds"] = torch.tensor([
        selected["adapt_threshold"], selected["retrieve_threshold"], selected["write_threshold"]
    ], dtype=module_state["calibration_thresholds"].dtype)
    module_state["split_enabled"] = torch.tensor(False, dtype=torch.bool)
    metadata = {
        "controller_policy_revision": 1,
        "calibration_method": "joint_counterfactual",
        "base_checkpoint": str(args.checkpoint),
        "base_checkpoint_sha256": _sha256_file(args.checkpoint),
        "controller_parameter_sha256": parameter_sha,
        "selected_thresholds": selected,
        "threshold_search_table": {"retrieve_adapt": ra_table, "write": write_table},
        "validation_metrics": {"retrieve_adapt": ra, "write": write},
        "validation_data_sha256": manifest["data_sha256"],
        "validation_sequence_count": len(validation),
    }
    output["controller_policy_revision"] = 1
    output["controller_recalibration"] = metadata
    output["controller_calibration"] = {
        **output.get("controller_calibration", {}), **selected,
        "calibration_method": "joint_counterfactual",
        "validation_data_sha256": manifest["data_sha256"],
    }
    output["controller_state"]["calibration"] = output["controller_calibration"]
    output["controller_state"]["split_enabled"] = False
    output["controller_state"]["controller_policy_revision"] = 1
    output["checkpoint_identity"] = {"path": str(args.output.resolve()), "role": "recalibrated"}
    if _tensor_digest({name: module_state[name] for name in parameter_names}) != parameter_sha:
        raise AssertionError("recalibration changed Controller parameters")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(output, temporary)
    temporary.replace(args.output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Jointly recalibrate a frozen Controller checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    metadata = recalibrate(args)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
