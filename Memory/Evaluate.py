"""End-to-end evaluation for Hawkes Memory Tree checkpoints.

The evaluator separates predictive quality, routing/tree health, and the
causal contribution of episodic and working memory.  Every ablation reloads
the checkpoint so persistent state cannot leak between conditions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import torch

from Train.Inference import InferenceConfig, MemoryTreeInference
from DataSplit import load_split_manifest
from ControllerIsolation import head_policy_sha256


VARIANTS = {
    "full_frozen": dict(episodic=True, working=True, online=False),
    "no_episodic": dict(episodic=False, working=True, online=False),
    "no_working": dict(episodic=True, working=False, online=False),
    "semantic_only": dict(episodic=False, working=False, online=False),
    "full_online": dict(episodic=True, working=True, online=True),
    # Keeps online usage/working-state behavior but prohibits physical writes.
    # This avoids deriving a threshold-1.01 checkpoint merely for ablation.
    "full_online_no_write": dict(
        episodic=True, working=True, online=True, writes=False
    ),
}

SUPPORTED_ROUTER_KINDS = {
    "node_semantic_compat_v1",
    "active_frontier_v1",
    "posterior_frontier_v2",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def frozen_event_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    stable = [{
        "source_index": int(row["source_index"]),
        "event_index": int(row["event_index"]),
        "nll": float(row["nll"]),
        "true_type": int(row["true_type"]),
        "predicted_type_at_event_time": int(row["predicted_type_at_event_time"]),
        "type_probabilities": [float(value) for value in row["type_probabilities"]],
    } for row in rows]
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_evaluation_provenance(
    checkpoint_meta: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    selected_test_ids: Sequence[int],
    manifest_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Validate whether a checkpoint may be reported as strict held-out."""
    provenance = dict(checkpoint_meta.get("data_provenance") or {})
    regime = provenance.get("evaluation_regime", "transductive")
    if regime != "strict_inductive":
        return "transductive", provenance or {
            "evaluation_regime": "transductive"
        }
    if manifest is None or manifest_path is None:
        raise ValueError(
            "strict_inductive checkpoint evaluation requires --split-manifest"
        )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if provenance.get("manifest_sha256") != manifest_sha:
        raise ValueError("checkpoint and evaluation split manifest SHA-256 differ")
    if provenance.get("data_sha256") != manifest.get("data_sha256"):
        raise ValueError("checkpoint provenance and manifest data SHA-256 differ")
    train_ids = set(map(int, provenance.get("train_source_ids", ())))
    validation_ids = set(map(int, provenance.get("validation_source_ids", ())))
    upstream_test_ids = set(map(int, provenance.get("test_source_ids", ())))
    selected = set(map(int, selected_test_ids))
    overlap = selected & (train_ids | validation_ids)
    if overlap:
        raise ValueError(
            "strict held-out evaluation refused: test IDs overlap upstream "
            f"train/validation IDs ({sorted(overlap)[:10]})"
        )
    if not selected <= upstream_test_ids:
        raise ValueError(
            "strict held-out evaluation refused: selected IDs are not a subset "
            "of the upstream test split"
        )
    splits = manifest["splits"]
    if (
        train_ids != set(map(int, splits["train"]))
        or validation_ids != set(map(int, splits["validation"]))
        or upstream_test_ids != set(map(int, splits["test"]))
    ):
        raise ValueError("checkpoint provenance split IDs differ from the manifest")
    if provenance.get("node_pool") != "train_only":
        raise ValueError("strict held-out evaluation requires node_pool=train_only")
    return "strict_inductive", provenance


def _parse_list(value: Any, cast) -> list:
    text = str(value).strip().strip("[]")
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], dict[int, int]]:
    frame = pd.read_csv(path)
    missing = {"event_times", "event_types"}.difference(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    raw: list[dict[str, Any]] = []
    all_types: set[int] = set()
    for source_index, row in frame.iterrows():
        times = _parse_list(row["event_times"], float)
        types = _parse_list(row["event_types"], int)
        if not times or len(times) != len(types):
            continue
        if times[0] < 0 or any(b < a for a, b in zip(times, times[1:])):
            raise ValueError(f"invalid event times at source row {source_index}")
        all_types.update(types)
        item: dict[str, Any] = {
            "source_index": int(source_index),
            "times": times,
            "types": types,
        }
        if "cluster" in frame.columns and not pd.isna(row["cluster"]):
            item["cluster_id"] = int(row["cluster"])
        raw.append(item)
    if not raw:
        raise ValueError("dataset contains no valid sequences")
    type_map = {value: index for index, value in enumerate(sorted(all_types))}
    dataset = []
    for item in raw:
        sequence = {
            "times": torch.tensor(item["times"], dtype=torch.float32),
            "types": torch.tensor(
                [type_map[value] for value in item["types"]], dtype=torch.long
            ),
            "source_index": item["source_index"],
        }
        if "cluster_id" in item:
            sequence["cluster_id"] = item["cluster_id"]
        dataset.append(sequence)
    return dataset, type_map


def dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_split(
    size: int, seed: int, validation_ratio: float, test_ratio: float
) -> dict[str, list[int]]:
    if not 0 <= validation_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("validation ratio must be in [0,1), test ratio in (0,1)")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation and test ratios must sum to less than one")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    test_count = max(1, round(size * test_ratio))
    validation_count = round(size * validation_ratio)
    return {
        "test": sorted(indices[:test_count]),
        "validation": sorted(indices[test_count:test_count + validation_count]),
        "train": sorted(indices[test_count + validation_count:]),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(p * math.log(max(p, 1e-12)) for p in probabilities if p > 0)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left, mean_right = _mean(left), _mean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in left)
        * sum((b - mean_right) ** 2 for b in right)
    )
    return numerator / denominator if denominator > 0 else None


def bootstrap_ci(
    values: Sequence[float], seed: int, samples: int = 1000
) -> list[float | None]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return [None, None]
    if len(clean) == 1:
        return [clean[0], clean[0]]
    rng = random.Random(seed)
    means = sorted(
        _mean(clean[rng.randrange(len(clean))] for _ in clean)
        for _ in range(max(samples, 1))
    )
    return [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]


def classification_metrics(rows: Sequence[Mapping[str, Any]], num_types: int) -> dict:
    if not rows:
        return {}
    confusion = [[0 for _ in range(num_types)] for _ in range(num_types)]
    correct = 0
    top3 = 0
    cross_entropy = 0.0
    brier = 0.0
    calibration = [[] for _ in range(10)]
    for row in rows:
        truth = int(row["true_type"])
        probs = [float(v) for v in row["type_probabilities"]]
        prediction = max(range(len(probs)), key=probs.__getitem__)
        correct += prediction == truth
        top3 += truth in sorted(range(len(probs)), key=probs.__getitem__, reverse=True)[:3]
        confusion[truth][prediction] += 1
        cross_entropy -= math.log(max(probs[truth], 1e-12))
        brier += sum((p - float(index == truth)) ** 2 for index, p in enumerate(probs))
        confidence = max(probs)
        calibration[min(int(confidence * 10), 9)].append((confidence, prediction == truth))
    f1_values = []
    for label in range(num_types):
        tp = confusion[label][label]
        fp = sum(confusion[t][label] for t in range(num_types) if t != label)
        fn = sum(confusion[label][p] for p in range(num_types) if p != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1_values.append(2 * precision * recall / max(precision + recall, 1e-12))
    ece = sum(
        len(bucket) / len(rows)
        * abs(_mean(c for c, _ in bucket) - _mean(float(ok) for _, ok in bucket))
        for bucket in calibration if bucket
    )
    return {
        "accuracy": correct / len(rows),
        "error_rate": 1.0 - correct / len(rows),
        "top3_accuracy": top3 / len(rows),
        "macro_f1": _mean(f1_values),
        "micro_f1": correct / len(rows),
        "cross_entropy": cross_entropy / len(rows),
        "brier_score": brier / len(rows),
        "ece_10bin": ece,
        "confusion_matrix": confusion,
    }


def aggregate_metrics(
    rows: Sequence[Mapping[str, Any]],
    num_types: int,
    seed: int,
    bootstrap_samples: int = 1000,
) -> dict:
    event_nll = [float(row["nll"]) for row in rows]
    time_errors = [abs(float(row["predicted_time"]) - float(row["true_time"])) for row in rows]
    by_sequence: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sequence[int(row["source_index"])].append(row)
    sequence_nll = [_mean(float(row["nll"]) for row in group) for group in by_sequence.values()]
    nll = _mean(event_nll)
    total_events = max(len(rows), 1)
    raw_residual = _mean(
        float(row.get("raw_episodic_residual_norm", 0.0)) for row in rows
    )
    gated_residual = _mean(
        float(row.get("gated_episodic_residual_norm", 0.0)) for row in rows
    )
    funnel_counts = {
        "memorize_argmax_count": sum(
            bool(row.get("memorize_argmax", False)) for row in rows
        ),
        "write_gate_active_count": sum(
            bool(row.get("write_gate_active", False)) for row in rows
        ),
        "write_candidate_count": sum(
            bool(row.get("write_candidate", False)) for row in rows
        ),
        "write_gate_pass_count": sum(
            bool(row.get("write_gate_passed", False)) for row in rows
        ),
        "write_priority_pass_count": sum(
            bool(row.get("write_priority_passed", False)) for row in rows
        ),
        "write_window_complete_count": sum(
            bool(row.get("write_window_complete", False)) for row in rows
        ),
        "write_accepted_count": sum(
            int(row.get(
                "write_promotion_count",
                int(bool(row.get("write_accepted", False))),
            ))
            for row in rows
        ),
        "write_local_accepted_count": sum(
            bool(row.get("write_local_accepted", False)) for row in rows
        ),
        "write_probation_enqueued_count": sum(
            bool(row.get("write_probation_enqueued", False)) for row in rows
        ),
        "write_retrieved_later_count": sum(
            bool(row.get("write_retrieved_later", False)) for row in rows
        ),
        "write_beneficial_count": sum(
            bool(row.get("write_beneficial", False)) for row in rows
        ),
    }
    accepted = funnel_counts["write_accepted_count"]
    metrics = {
        "events": len(rows),
        "sequences": len(by_sequence),
        "nll_per_event": nll,
        "nll_bootstrap_95ci": bootstrap_ci(
            sequence_nll, seed, bootstrap_samples
        ),
        "sequence_macro_nll": _mean(sequence_nll),
        "perplexity": math.exp(min(nll, 50.0)),
        "local_time_mae": _mean(time_errors),
        "local_time_rmse": math.sqrt(_mean(error * error for error in time_errors)),
        "local_time_median_ae": statistics.median(time_errors) if time_errors else None,
        "memory_hit_fraction": _mean(
            float(float(row.get("retrieval_alpha_mass", 0.0)) > 1e-6)
            for row in rows
        ),
        "read_coverage_fraction": _mean(
            float(int(row.get("visited_bank_count", 0)) > 0) for row in rows
        ),
        "nonempty_read_coverage_fraction": _mean(
            float(int(row.get("visited_nonempty_bank_count", 0)) > 0)
            for row in rows
        ),
        "owner_path_coverage_fraction": _mean(
            float(bool(row.get("owner_on_retrieval_path", False)))
            for row in rows
        ),
        "mean_retrieval_alpha_mass": _mean(
            float(row.get("retrieval_alpha_mass", 0.0)) for row in rows
        ),
        "mean_retrieval_alpha_per_visited_node": _mean(
            float(row.get("retrieval_alpha_per_visited_node", 0.0)) for row in rows
        ),
        "mean_retrieval_effective_k": _mean(
            float(row.get("retrieval_effective_k", 0.0)) for row in rows
        ),
        "mean_retrieval_similarity": _mean(
            float(row.get("retrieval_similarity", -1.0)) for row in rows
        ),
        "mean_retrieval_null_alpha": _mean(
            float(row.get("retrieval_null_alpha", 1.0)) for row in rows
        ),
        "mean_episodic_residual_norm": _mean(
            float(row.get("episodic_residual_norm", 0.0)) for row in rows
        ),
        "mean_raw_episodic_residual_norm": raw_residual,
        "mean_gated_episodic_residual_norm": gated_residual,
        "raw_to_gated_residual_ratio": (
            gated_residual / raw_residual if raw_residual > 0.0 else 0.0
        ),
        "mean_retrieve_gate": _mean(
            float(row.get("retrieve_gate", 0.0)) for row in rows
        ),
        "write_funnel": {
            **funnel_counts,
            "gate_active_per_event": (
                funnel_counts["write_gate_active_count"] / total_events
            ),
            "candidate_per_event": (
                funnel_counts["write_candidate_count"] / total_events
            ),
            "gate_pass_per_candidate": (
                funnel_counts["write_gate_pass_count"]
                / max(funnel_counts["write_candidate_count"], 1)
            ),
            "priority_pass_per_gate_pass": (
                funnel_counts["write_priority_pass_count"]
                / max(funnel_counts["write_gate_pass_count"], 1)
            ),
            "window_complete_per_priority_pass": (
                funnel_counts["write_window_complete_count"]
                / max(funnel_counts["write_priority_pass_count"], 1)
            ),
            "accepted_per_window_complete": (
                accepted / max(funnel_counts["write_window_complete_count"], 1)
            ),
            "accepted_per_candidate": (
                accepted / max(funnel_counts["write_candidate_count"], 1)
            ),
            "accepted_reuse_fraction": (
                funnel_counts["write_retrieved_later_count"] / max(accepted, 1)
            ),
            "accepted_beneficial_fraction": (
                sum(
                    bool(row.get("write_accepted", False))
                    and bool(row.get("write_beneficial", False))
                    for row in rows
                ) / max(accepted, 1)
            ),
        },
    }
    metrics.update(classification_metrics(rows, num_types))
    return metrics


def clear_episodic_memory(inference: MemoryTreeInference) -> None:
    for bank in inference.tree.episodic_memory.banks.values():
        bank.clear()
    inference.tree.episodic_memory._packed_mirror = None
    inference.tree.episodic_memory._packed_mirror_signature = None


def run_variant(
    checkpoint: Path,
    sequences: Sequence[Mapping[str, Any]],
    variant: str,
    device: str | None,
    progress_dir: Path | None = None,
    resume_partial: bool = False,
    prototype_duplicate_threshold: float | None = None,
    prototype_mode_threshold: float | None = None,
    prototype_context_alias_capacity: int | None = None,
    verbose: bool = True,
) -> tuple[list[dict], MemoryTreeInference, float]:
    settings = VARIANTS[variant]
    inference = MemoryTreeInference.from_checkpoint(
        checkpoint,
        device=device,
        inference_config=InferenceConfig(
            adapt_working_memory=settings["working"],
            allow_memory_writes=settings.get("writes", settings["online"]),
            update_memory_usage=settings["online"],
            probe_write_counterfactuals=(
                variant in {"full_online", "full_online_no_write"}
            ),
            write_probe_seed=42,
            prototype_duplicate_threshold=prototype_duplicate_threshold,
            prototype_mode_threshold=prototype_mode_threshold,
            prototype_context_alias_capacity=prototype_context_alias_capacity,
        ),
    )
    if not settings["episodic"]:
        clear_episodic_memory(inference)
    partial_path = (
        None if progress_dir is None else progress_dir / f"{variant}.partial.json"
    )
    rows: list[dict] = []
    start_position = 0
    if (
        resume_partial
        and not settings["online"]
        and partial_path is not None
        and partial_path.is_file()
    ):
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        rows = list(partial.get("rows", ()))
        start_position = int(partial.get("completed_sequences", 0))
        print(f"[Resume] {variant} continuing at sequence {start_position + 1}")
    start = time.perf_counter()
    for sequence_position, sequence in enumerate(sequences[start_position:], start=start_position):
        result = inference.run_sequence(sequence)
        cluster_id = sequence.get("cluster_id")
        for event in result["events"]:
            posterior = [float(v) for v in event["frontier_posterior"]]
            action_probabilities = [float(v) for v in event["action_probabilities"]]
            raw_action_probabilities = [
                float(v) for v in event.get(
                    "raw_action_probabilities", event["action_probabilities"]
                )
            ]
            probs = [
                float(v)
                for v in event["type_probabilities_at_event_time"]
            ]
            visited_nodes = int(event.get(
                "visited_bank_count", len(event["frontier_node_ids"])
            ))
            rows.append({
                "variant": variant,
                "sequence_position": sequence_position,
                "source_index": int(sequence["source_index"]),
                "cluster_id": cluster_id,
                "event_index": int(event["event_index"]),
                "true_type": int(event["true_type"]),
                "predicted_type_at_event_time": int(event["predicted_type"]),
                "type_probabilities": probs,
                "prefix_type_probabilities": [
                    float(v)
                    for v in event["forecast_type_probabilities"]
                ],
                "nll": float(event["nll"]),
                "true_time": float(event["true_time"]),
                "predicted_time": float(event["predicted_time"]),
                "predicted_delta": float(event["predicted_delta"]),
                "owner_id": str(event["owner_id"]),
                "frontier_node_ids": list(event["frontier_node_ids"]),
                "frontier_posterior": posterior,
                "frontier_entropy": _entropy(posterior),
                "frontier_max_probability": max(posterior) if posterior else 0.0,
                "action": event["action"],
                "memorize_argmax": bool(event.get(
                    "memorize_argmax", event["action"] == "MEMORIZE"
                )),
                "action_probabilities": action_probabilities,
                "raw_action_probabilities": raw_action_probabilities,
                "retrieval_alpha_mass": float(event["retrieval_alpha_mass"]),
                "retrieval_alpha_per_visited_node": float(event.get(
                    "retrieval_alpha_per_visited_node",
                    float(event["retrieval_alpha_mass"]) / max(visited_nodes, 1),
                )),
                "retrieval_similarity": float(event.get(
                    "retrieval_similarity", -1.0
                )),
                "retrieval_effective_k": int(event["retrieval_effective_k"]),
                "retrieval_null_alpha": float(event["retrieval_null_alpha"]),
                "visited_bank_count": visited_nodes,
                "visited_nonempty_bank_count": int(event.get(
                    "visited_nonempty_bank_count", 0
                )),
                "raw_episodic_residual_norm": float(event.get(
                    "raw_episodic_residual_norm",
                    event.get("episodic_residual_norm", 0.0),
                )),
                "retrieve_gate": float(event.get(
                    "retrieve_gate", action_probabilities[1]
                )),
                "gated_episodic_residual_norm": float(event.get(
                    "gated_episodic_residual_norm", 0.0
                )),
                "owner_on_retrieval_path": bool(event.get(
                    "owner_on_retrieval_path", False
                )),
                "retrieval_counterfactual_gain": event.get(
                    "retrieval_counterfactual_gain"
                ),
                "retrieval_counterfactual_unavailable_reason": event.get(
                    "retrieval_counterfactual_unavailable_reason"
                ),
                "episodic_residual_norm": float(event["episodic_residual_norm"]),
                "write_token": event.get("write_token"),
                "write_candidate": bool(event.get("write_candidate", False)),
                "write_gate_active": bool(event.get("write_gate_active", False)),
                "write_probed": bool(event.get("write_probed", False)),
                "write_utility": event.get("write_utility"),
                "write_priority": event.get("write_priority"),
                "write_accepted": bool(event.get("write_accepted", False)),
                "write_local_accepted": bool(event.get(
                    "write_local_accepted", False
                )),
                "write_probation_enqueued": bool(event.get(
                    "write_probation_enqueued", False
                )),
                "write_promotion_count": int(event.get(
                    "write_promotion_count", 0
                )),
                "write_probe_propensity": event.get("write_probe_propensity"),
                "write_probe_top": bool(event.get("write_probe_top", False)),
                "write_probe_exploration": bool(
                    event.get("write_probe_exploration", False)
                ),
                "write_gate_passed": bool(event.get("write_gate_passed", False)),
                "write_priority_passed": bool(event.get(
                    "write_priority_passed", False
                )),
                "write_window_complete": bool(event.get(
                    "write_window_complete", False
                )),
                "write_retrieved_later": bool(event.get(
                    "write_retrieved_later", False
                )),
                "write_beneficial": bool(event.get(
                    "write_beneficial", False
                )),
                "write_utility_passed": bool(event.get("write_utility_passed", False)),
                "write_owner_on_score_path": bool(
                    event.get("write_owner_on_score_path", False)
                ),
                "write_virtual_candidate_alpha": event.get(
                    "write_virtual_candidate_alpha"
                ),
            })
        completed = sequence_position + 1
        elapsed_now = time.perf_counter() - start
        eta = elapsed_now / completed * (len(sequences) - completed)
        if verbose:
            print(
                f"[Evaluate] {variant} {completed}/{len(sequences)} "
                f"elapsed={elapsed_now:.1f}s eta={eta:.1f}s",
                flush=True,
            )
        if progress_dir is not None:
            progress_dir.mkdir(parents=True, exist_ok=True)
            (progress_dir / f"{variant}.progress.json").write_text(
                json.dumps({
                    "variant": variant,
                    "completed_sequences": completed,
                    "total_sequences": len(sequences),
                    "elapsed_seconds": elapsed_now,
                    "last_source_index": int(sequence["source_index"]),
                }, indent=2), encoding="utf-8"
            )
            if partial_path is not None:
                partial_path.write_text(json.dumps({
                    "completed_sequences": completed,
                    "rows": _jsonable(rows),
                }), encoding="utf-8")
    elapsed = time.perf_counter() - start
    return rows, inference, elapsed


def preflight_checkpoint(
    checkpoint: Path,
    sequence: Mapping[str, Any],
    device: str | None,
    prototype_duplicate_threshold: float | None = None,
    prototype_mode_threshold: float | None = None,
    prototype_context_alias_capacity: int | None = None,
) -> None:
    """Exercise one causal sequence before starting expensive ablations."""
    inference = MemoryTreeInference.from_checkpoint(
        checkpoint,
        device=device,
        inference_config=InferenceConfig(
            adapt_working_memory=True,
            allow_memory_writes=False,
            update_memory_usage=False,
            prototype_duplicate_threshold=prototype_duplicate_threshold,
            prototype_mode_threshold=prototype_mode_threshold,
            prototype_context_alias_capacity=prototype_context_alias_capacity,
        ),
    )
    result = inference.run_sequence(sequence)
    if not result.get("events"):
        raise RuntimeError("preflight produced no event predictions")
    required = {
        "nll",
        "type_probabilities_at_event_time",
        "forecast_type_probabilities",
        "predicted_time",
        "frontier_posterior",
        "retrieval_alpha_mass",
    }
    missing = required.difference(result["events"][0])
    if missing:
        raise RuntimeError(
            f"inference/evaluator contract is missing fields: {sorted(missing)}"
        )
    numeric = (
        float(event["nll"])
        for event in result["events"]
    )
    if not all(math.isfinite(value) for value in numeric):
        raise FloatingPointError("preflight produced non-finite event NLL")


def tree_metrics(inference: MemoryTreeInference, rows: Sequence[Mapping[str, Any]]) -> tuple[dict, list[dict]]:
    tree = inference.tree
    owner_counts = Counter(str(row["owner_id"]) for row in rows)
    frontier_counts = Counter(
        node_id for row in rows for node_id in row["frontier_node_ids"]
    )
    total_owner = sum(owner_counts.values())
    owner_probabilities = [count / total_owner for count in owner_counts.values()] if total_owner else []
    node_rows = []
    leaf_set = set(tree.leaf_ids)
    for node_id in tree.all_node_ids:
        node = tree.nodes[node_id]
        bank = tree.episodic_memory.banks.get(node_id)
        size = len(bank) if bank is not None else 0
        node_rows.append({
            "node_id": node_id,
            "parent": node.parent,
            "left": node.left,
            "right": node.right,
            "depth": int(node.depth),
            "is_leaf": node_id in leaf_set,
            "owner_count": owner_counts[node_id],
            "frontier_exposure": frontier_counts[node_id],
            "bank_size": size,
            "bank_capacity": tree.episodic_memory.capacity_per_node,
            "bank_utilization": size / tree.episodic_memory.capacity_per_node,
            "mean_usage": float(bank.usage.mean().cpu()) if size else 0.0,
            "mean_age": float(bank.effective_age(tree.episodic_memory._age_clock).mean().cpu()) if size else 0.0,
            "mean_write_quality": float(bank.write_quality.mean().cpu()) if size else 0.0,
            "mean_residual_norm": float(bank.deltas.norm(dim=-1).mean().cpu()) if size else 0.0,
        })
    leaf_depths = [int(tree.nodes[node_id].depth) for node_id in tree.leaf_ids]
    internal_memory = sum(row["bank_size"] for row in node_rows if not row["is_leaf"])
    leaf_memory = sum(row["bank_size"] for row in node_rows if row["is_leaf"])
    leaf_parameters = [
        tree.semantic_theta(node_id).detach().reshape(-1).cpu()
        for node_id in tree.leaf_ids
    ]
    pairwise_semantic_distance = [
        float((leaf_parameters[left] - leaf_parameters[right]).norm())
        for left in range(len(leaf_parameters))
        for right in range(left + 1, len(leaf_parameters))
    ]
    descendant_leaf_count: dict[str, int] = {}
    def count_descendants(node_id: str) -> int:
        node = tree.nodes[node_id]
        if node_id in leaf_set:
            descendant_leaf_count[node_id] = 1
        else:
            descendant_leaf_count[node_id] = sum(
                count_descendants(child)
                for child in (node.left, node.right)
                if child is not None
            )
        return descendant_leaf_count[node_id]
    count_descendants("root")
    branch_imbalances = []
    for node_id in tree.all_node_ids:
        node = tree.nodes[node_id]
        if node.left is not None and node.right is not None:
            left = descendant_leaf_count[node.left]
            right = descendant_leaf_count[node.right]
            branch_imbalances.append(abs(left - right) / max(left + right, 1))
    metrics = {
        "node_count": len(tree.all_node_ids),
        "leaf_count": len(tree.leaf_ids),
        "max_depth": max(leaf_depths, default=0),
        "mean_leaf_depth": _mean(leaf_depths),
        "leaf_depth_std": statistics.pstdev(leaf_depths) if len(leaf_depths) > 1 else 0.0,
        "owner_entropy": _entropy(owner_probabilities),
        "normalized_owner_entropy": (
            _entropy(owner_probabilities) / math.log(max(len(owner_probabilities), 2))
            if owner_probabilities else 0.0
        ),
        "effective_owner_count": math.exp(_entropy(owner_probabilities)),
        "owner_top1_share": max(owner_probabilities, default=0.0),
        "unvisited_leaf_fraction": _mean(float(frontier_counts[node_id] == 0) for node_id in tree.leaf_ids),
        "total_memory_rows": internal_memory + leaf_memory,
        "internal_memory_rows": internal_memory,
        "leaf_memory_rows": leaf_memory,
        "mean_branch_size_imbalance": _mean(branch_imbalances),
        "mean_leaf_semantic_distance": _mean(pairwise_semantic_distance),
        "min_leaf_semantic_distance": min(pairwise_semantic_distance, default=0.0),
    }
    return metrics, node_rows


def routing_label_metrics(
    rows: Sequence[Mapping[str, Any]],
    expected_clusters: set[int] | None = None,
) -> dict:
    labeled = [row for row in rows if row.get("cluster_id") is not None]
    if not labeled:
        return {"available": False}
    clusters = sorted({int(row["cluster_id"]) for row in labeled})
    if len(clusters) < 2 or (
        expected_clusters is not None and set(clusters) != expected_clusters
    ):
        return {
            "available": False,
            "cluster_count": len(clusters),
            "represented_clusters": clusters,
            "note": "Purity/NMI/ARI skipped because cluster coverage is incomplete.",
        }
    # Sequence-majority owner avoids weighting long sequences more heavily.
    sequence_rows: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in labeled:
        sequence_rows[int(row["source_index"])].append(row)
    pairs = []
    for group in sequence_rows.values():
        pairs.append((Counter(str(row["owner_id"]) for row in group).most_common(1)[0][0], int(group[0]["cluster_id"])))
    owner_cluster: dict[str, Counter] = defaultdict(Counter)
    for owner, cluster in pairs:
        owner_cluster[owner][cluster] += 1
    purity = sum(max(counts.values()) for counts in owner_cluster.values()) / len(pairs)
    result = {"available": True, "sequence_count": len(pairs), "owner_purity": purity}
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        owner_index = {owner: index for index, owner in enumerate(sorted(owner_cluster))}
        predicted = [owner_index[owner] for owner, _ in pairs]
        truth = [cluster for _, cluster in pairs]
        result.update({
            "nmi": float(normalized_mutual_info_score(truth, predicted)),
            "ari": float(adjusted_rand_score(truth, predicted)),
        })
    except ImportError:
        result["note"] = "scikit-learn unavailable; NMI/ARI skipped"
    return result


def ablation_metrics(all_rows: Mapping[str, Sequence[Mapping[str, Any]]], seed: int) -> list[dict]:
    keyed = {
        variant: {(int(row["source_index"]), int(row["event_index"])): row for row in rows}
        for variant, rows in all_rows.items()
    }
    comparisons = [
        ("episodic_gain", "no_episodic", "full_frozen"),
        ("working_gain", "no_working", "full_frozen"),
        ("total_memory_gain", "semantic_only", "full_frozen"),
        ("online_vs_frozen_gain", "full_frozen", "full_online"),
        ("realized_write_gain", "full_online_no_write", "full_online"),
    ]
    output = []
    for name, baseline, target in comparisons:
        if baseline not in keyed or target not in keyed:
            continue
        common = sorted(set(keyed[baseline]).intersection(keyed[target]))
        gains = [float(keyed[baseline][key]["nll"]) - float(keyed[target][key]["nll"]) for key in common]
        item = {
            "comparison": name,
            "baseline": baseline,
            "target": target,
            "events": len(gains),
            "mean_nll_gain": _mean(gains),
            "median_nll_gain": statistics.median(gains) if gains else None,
            "improved_event_fraction": _mean(float(value > 0) for value in gains),
            "negative_gain_fraction": _mean(float(value < 0) for value in gains),
            "bootstrap_95ci": bootstrap_ci(gains, seed),
        }
        if target == "full_frozen":
            diagnostic_fields = {
                "alpha_mass": "retrieval_alpha_mass",
                "alpha_per_visited_node": "retrieval_alpha_per_visited_node",
                "similarity": "retrieval_similarity",
                "raw_residual_norm": "raw_episodic_residual_norm",
                "gated_residual_norm": "gated_episodic_residual_norm",
                "retrieve_gate": "retrieve_gate",
            }
            correlations = {}
            for label, field in diagnostic_fields.items():
                values = [
                    float(keyed[target][key].get(field, 0.0))
                    for key in common
                ]
                correlations[label] = {
                    "pearson": _pearson(values, gains),
                    "spearman": _spearman(values, gains),
                }
            item["retrieval_diagnostic_correlations"] = correlations
            item["retrieval_gain_correlation"] = correlations[
                "alpha_per_visited_node"
            ]["pearson"]
        output.append(item)
    return output


def annotate_retrieval_counterfactual_gain(
    all_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Attach paired no-episodic gains without fabricating missing values."""
    target = all_rows.get("full_frozen")
    baseline = all_rows.get("no_episodic")
    if target is None or baseline is None:
        for row in target or ():
            row["retrieval_counterfactual_gain"] = None
            row["retrieval_counterfactual_unavailable_reason"] = (
                "requires both full_frozen and no_episodic variants"
            )
        return {
            "available": False,
            "reason": "requires both full_frozen and no_episodic variants",
        }
    baseline_by_key = {
        (int(row["source_index"]), int(row["event_index"])): row
        for row in baseline
    }
    paired = 0
    for row in target:
        key = (int(row["source_index"]), int(row["event_index"]))
        other = baseline_by_key.get(key)
        if other is None:
            row["retrieval_counterfactual_gain"] = None
            row["retrieval_counterfactual_unavailable_reason"] = (
                "no matching no_episodic event"
            )
            continue
        row["retrieval_counterfactual_gain"] = (
            float(other["nll"]) - float(row["nll"])
        )
        row["retrieval_counterfactual_unavailable_reason"] = None
        paired += 1
    return {
        "available": paired > 0,
        "paired_event_count": paired,
        "unpaired_event_count": len(target) - paired,
    }


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2.0 + 1.0
        for offset in range(position, end):
            ranks[order[offset]] = rank
        position = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    return _pearson(_rank(left), _rank(right))


def _roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _rank(scores)
    positive_rank = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _pr_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, index in enumerate(ordered, start=1):
        if labels[index]:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def controller_metrics(
    all_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    variant_metrics: Mapping[str, Mapping[str, Any]],
    checkpoint_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure whether each gate ranks its own observed counterfactual utility."""
    keyed = {
        variant: {
            (int(row["source_index"]), int(row["event_index"])): row
            for row in rows
        }
        for variant, rows in all_rows.items()
    }
    specifications = {
        "retrieve": (1, "no_episodic", "full_frozen", 0.5),
        "adapt": (0, "no_working", "full_frozen", 0.5),
    }
    output: dict[str, Any] = {}
    calibration = (checkpoint_meta or {}).get("controller_calibration", {})
    for action, (gate_index, baseline, target, threshold) in specifications.items():
        if baseline not in keyed or target not in keyed:
            output[action] = {"available": False, "reason": "required ablation missing"}
            continue
        common = sorted(set(keyed[baseline]) & set(keyed[target]))
        gates = [
            float(keyed[target][key].get(
                "raw_action_probabilities",
                keyed[target][key]["action_probabilities"],
            )[gate_index])
            for key in common
        ]
        utilities = [
            float(keyed[baseline][key]["nll"]) - float(keyed[target][key]["nll"])
            - 1e-4
            for key in common
        ]
        threshold = float(calibration.get(f"{action}_threshold", threshold))
        selected = [gate >= threshold for gate in gates]
        output[action] = _action_utility_metrics(gates, utilities, selected)

    if "full_online" in keyed:
        write_probe_variant = (
            "full_online_no_write"
            if "full_online_no_write" in keyed else "full_online"
        )
        rows = [
            row for row in keyed[write_probe_variant].values()
            if row.get("write_utility") is not None
        ]
        gates = [float(row.get("raw_action_probabilities", row["action_probabilities"])[2]) for row in rows]
        utilities = [float(row["write_utility"]) for row in rows]
        write_threshold = float(calibration.get("write_threshold", 0.5))
        selected = (
            [gate >= write_threshold for gate in gates]
            if write_probe_variant == "full_online_no_write"
            else [bool(row.get("write_accepted", False)) for row in rows]
        )
        output["write"] = _action_utility_metrics(gates, utilities, selected)
        propensities = [
            max(float(row.get("write_probe_propensity") or 1.0), 1e-6)
            for row in rows
        ]
        ipw = [min(10.0, 1.0 / propensity) for propensity in propensities]
        mean_weight = _mean(ipw) or 1.0
        ipw = [weight / mean_weight for weight in ipw]
        output["write"]["ipw"] = _action_utility_metrics(
            gates, [utility * weight for utility, weight in zip(utilities, ipw)], selected
        )
        positives = sum(utility > 0 for utility in utilities)
        negatives = len(utilities) - positives
        output["write"]["positive_count"] = positives
        output["write"]["negative_count"] = negatives
        output["write"]["auc_sample_sufficient"] = positives >= 20 and negatives >= 20
        if not output["write"]["auc_sample_sufficient"]:
            output["write"]["beneficial_roc_auc"] = None
            output["write"]["beneficial_pr_auc"] = None
            output["write"]["auc_reason"] = (
                "insufficient samples: at least 20 positive and 20 negative probes required"
            )
        output["write"].update({
            "candidate_count": sum(
                float(row.get("raw_action_probabilities", row["action_probabilities"])[2]) >= 0.6
                for row in keyed[write_probe_variant].values()
            ),
            "probe_count": len(rows),
            "probed_gate_selected_count": sum(selected),
            "probed_accepted_count": sum(
                bool(row.get("write_accepted", False))
                and row.get("write_utility") is not None
                for row in keyed["full_online"].values()
            ),
            "physical_accepted_count": sum(
                bool(row.get("write_accepted", False))
                for row in keyed["full_online"].values()
            ),
            "unprobed_accepted_count": sum(
                bool(row.get("write_accepted", False))
                and row.get("write_utility") is None
                for row in keyed["full_online"].values()
            ),
        })
        # Backward-compatible name now uses the physical action count so budget
        # and rollout metrics cannot silently omit late, unprobed commits.
        output["write"]["accepted_count"] = output["write"][
            "physical_accepted_count"
        ]
        output["write"]["ranking"] = _write_ranking_metrics(rows, seed=42)
        output["write"]["local_utility_method"] = (
            "state-isolated physical add_memory/read_nodes branch"
            if write_probe_variant == "full_online_no_write"
            else "committed-item exclusion compatibility path"
        )
        output["write"]["local_utility_variant"] = write_probe_variant
        output["write"]["local_utility_is_policy_selection_metric"] = False
        if "full_online_no_write" in keyed:
            common = sorted(
                set(keyed["full_online_no_write"]) & set(keyed["full_online"])
            )
            realized = [
                float(keyed["full_online_no_write"][key]["nll"])
                - float(keyed["full_online"][key]["nll"])
                for key in common
            ]
            output["write"]["realized_rollout"] = {
                "event_count": len(realized),
                "mean_nll_gain": _mean(realized),
                "total_nll_gain": sum(realized),
                "bootstrap_95ci": bootstrap_ci(realized, 42),
                "harmful_event_fraction": _mean(
                    float(value < 0.0) for value in realized
                ),
            }
    else:
        output["write"] = {"available": False, "reason": "full_online missing"}
    replay_state = (
        (checkpoint_meta or {}).get("controller_state", {})
        .get("utility_replay", {})
    )
    split_rows = []
    for store_name in ("uniform", "hard"):
        store = replay_state.get(store_name, {})
        for sign in (0, 1):
            split_rows.extend(store.get((3, sign), ()))
    if split_rows:
        gates = [float(torch.as_tensor(row["gate"])[3]) for row in split_rows]
        utilities = [float(torch.as_tensor(row["utility"])[3]) for row in split_rows]
        selected = [
            float(torch.as_tensor(row["gate"])[2]) >= 0.6 and gate >= 0.5
            for row, gate in zip(split_rows, gates)
        ]
        output["split"] = _action_utility_metrics(gates, utilities, selected)
        output["split"]["label_source"] = "Sleep-evaluated transaction replay"
    else:
        output["split"] = {
            "available": False,
            "reason": "No Sleep-evaluated Split labels in checkpoint replay.",
        }
    for action, metrics in output.items():
        if not metrics.get("available", False):
            continue
        elapsed_variant = "full_online" if action == "write" else "full_frozen"
        elapsed = float(variant_metrics.get(elapsed_variant, {}).get("elapsed_seconds", 0.0))
        metrics["gain_per_second"] = metrics["selected_total_gain"] / max(elapsed, 1e-12)
    return output


def _write_ranking_metrics(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    """Sequence-budget ranking diagnostics over state-isolated Write probes."""
    groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        gate = float(row.get(
            "raw_action_probabilities", row["action_probabilities"]
        )[2])
        groups[int(row["source_index"])].append((gate, float(row["write_utility"])))
    pair_correct = 0
    pair_count = 0
    ndcg = {1: [], 2: [], 4: []}
    top4_values = []
    random4_values = []
    regrets = []
    harmful_selected = 0
    selected_count = 0
    rng = random.Random(int(seed))
    for _, group in sorted(groups.items()):
        if not group:
            continue
        utilities = sorted(value for _, value in group)
        q25 = utilities[int(0.25 * (len(utilities) - 1))]
        q75 = utilities[int(0.75 * (len(utilities) - 1))]
        iqr = q75 - q25
        minimum_gap = max(1e-4, 0.1 * iqr)
        high = [item for item in group if item[1] >= q75]
        low = [item for item in group if item[1] <= q25]
        for high_item in high:
            for low_item in low:
                if high_item[1] - low_item[1] > minimum_gap:
                    pair_count += 1
                    pair_correct += int(high_item[0] > low_item[0])
        predicted = sorted(group, key=lambda item: item[0], reverse=True)
        ideal = sorted(group, key=lambda item: item[1], reverse=True)
        for k in (1, 2, 4):
            actual_dcg = sum(
                max(value, 0.0) / math.log2(rank + 2.0)
                for rank, (_, value) in enumerate(predicted[:k])
            )
            ideal_dcg = sum(
                max(value, 0.0) / math.log2(rank + 2.0)
                for rank, (_, value) in enumerate(ideal[:k])
            )
            ndcg[k].append(actual_dcg / ideal_dcg if ideal_dcg > 0.0 else 1.0)
        selected = predicted[:4]
        selected_utility = sum(value for _, value in selected)
        top4_values.append(selected_utility)
        sample = rng.sample(group, min(4, len(group)))
        random4_values.append(sum(value for _, value in sample))
        ideal_utility = sum(value for _, value in ideal[:4])
        regrets.append(max(0.0, ideal_utility - selected_utility))
        harmful_selected += sum(value <= 0.0 for _, value in selected)
        selected_count += len(selected)
    top4 = _mean(top4_values)
    random4 = _mean(random4_values)
    return {
        "group_count": len(groups),
        "pair_count": pair_count,
        "pairwise_accuracy": pair_correct / max(pair_count, 1),
        "ndcg_at_1": _mean(ndcg[1]),
        "ndcg_at_2": _mean(ndcg[2]),
        "ndcg_at_4": _mean(ndcg[4]),
        "top4_cumulative_utility": top4,
        "random_top4_utility": random4,
        "top4_uplift_over_random": top4 - random4,
        "sequence_regret": _mean(regrets),
        "harmful_selection_rate": harmful_selected / max(selected_count, 1),
    }


def _action_utility_metrics(
    gates: Sequence[float], utilities: Sequence[float], selected: Sequence[bool]
) -> dict[str, Any]:
    if not utilities:
        return {"available": False, "reason": "no labeled probes"}
    labels = [value > 0.0 for value in utilities]
    harmful_loss = sum(-value for value, take in zip(utilities, selected) if take and value < 0.0)
    missed_gain = sum(value for value, take in zip(utilities, selected) if not take and value > 0.0)
    selected_gain = sum(value for value, take in zip(utilities, selected) if take)
    selected_count = sum(selected)
    ordered = sorted(range(len(gates)), key=lambda index: gates[index])
    deciles = []
    for decile in range(10):
        start = decile * len(ordered) // 10
        end = (decile + 1) * len(ordered) // 10
        indices = ordered[start:end]
        if indices:
            deciles.append({
                "decile": decile + 1,
                "count": len(indices),
                "mean_gate": _mean(gates[index] for index in indices),
                "mean_utility": _mean(utilities[index] for index in indices),
                "beneficial_fraction": _mean(float(labels[index]) for index in indices),
            })
    return {
        "available": True,
        "labeled_count": len(utilities),
        "positive_utility_fraction": _mean(map(float, labels)),
        "pearson_gate_utility": _pearson(gates, utilities),
        "spearman_gate_utility": _pearson(_rank(gates), _rank(utilities)),
        "beneficial_roc_auc": _roc_auc(gates, labels),
        "beneficial_pr_auc": _pr_auc(gates, labels),
        "utility_deciles": deciles,
        "false_positive_harmful_loss": harmful_loss,
        "false_negative_missed_gain": missed_gain,
        "controller_regret": (harmful_loss + missed_gain) / len(utilities),
        "selected_count": selected_count,
        "selected_total_gain": selected_gain,
        "gain_per_action": selected_gain / max(selected_count, 1),
    }


def warnings_for(summary: Mapping[str, Any]) -> list[str]:
    warnings = []
    tree = summary.get("tree", {})
    if tree.get("owner_top1_share", 0) > 0.8:
        warnings.append("Routing collapse risk: one owner receives more than 80% of sequences/events.")
    if tree.get("unvisited_leaf_fraction", 0) > 0.25:
        warnings.append("Coverage risk: more than 25% of leaves were never exposed on the test set.")
    for comparison in summary.get("ablations", []):
        if comparison["comparison"] == "total_memory_gain" and comparison["mean_nll_gain"] < 0:
            warnings.append("Memory hurts held-out NLL on average relative to semantic-only inference.")
    return warnings


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_jsonable(row.get(key)), ensure_ascii=False) if isinstance(row.get(key), (list, dict, tuple)) else row.get(key) for key in keys})


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Hawkes Memory Tree Evaluation", "",
        f"Evaluation regime: **{summary.get('evaluation_regime', 'transductive')}**.",
        "", "## Predictive quality", "",
    ]
    for variant, metrics in summary["variants"].items():
        lines.append(
            f"- **{variant}**: NLL/event={metrics['nll_per_event']:.6f}, "
            f"ACC={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}, "
            f"local-time MAE={metrics['local_time_mae']:.4f}"
        )
    lines.extend(["", "## Memory contribution", ""])
    for item in summary["ablations"]:
        lines.append(
            f"- **{item['comparison']}**: mean ΔNLL={item['mean_nll_gain']:+.6f}, "
            f"improved events={item['improved_event_fraction']:.1%}, "
            f"95% CI={item['bootstrap_95ci']}"
        )
    lines.extend(["", "## Controller utility diagnostics", ""])
    for action, metrics in summary.get("controller", {}).items():
        if not metrics.get("available", False):
            lines.append(f"- **{action}**: unavailable ({metrics.get('reason', 'no labels')}).")
            continue
        lines.append(
            f"- **{action}**: Spearman={metrics['spearman_gate_utility']}, "
            f"ROC-AUC={metrics['beneficial_roc_auc']}, "
            f"PR-AUC={metrics['beneficial_pr_auc']}, "
            f"regret/event={metrics['controller_regret']:.6f}."
        )
        if action == "write" and metrics.get("ranking"):
            ranking = metrics["ranking"]
            lines.append(
                "  Write ranking: "
                f"pair-acc={ranking['pairwise_accuracy']:.4f}, "
                f"NDCG@4={ranking['ndcg_at_4']:.4f}, "
                f"Top-4={ranking['top4_cumulative_utility']:.6f}, "
                f"random-Top-4={ranking['random_top4_utility']:.6f}, "
                f"regret/sequence={ranking['sequence_regret']:.6f}."
            )
    acceptance = summary.get("controller_research_acceptance", {})
    if acceptance:
        lines.append(
            f"- Research thresholds overall: **{'PASS' if acceptance.get('passed') else 'FAIL'}**; "
            f"details: `{acceptance.get('checks')}`."
        )
    full = summary.get("variants", {}).get("full_frozen", {})
    if full:
        funnel = full.get("write_funnel", {})
        episodic_diagnostics = next((
            item.get("retrieval_diagnostic_correlations", {})
            for item in summary.get("ablations", ())
            if item.get("comparison") == "episodic_gain"
        ), {})
        lines.extend([
            "", "## Memory call-chain diagnostics", "",
            f"- Read/nonempty/owner-path coverage: {full.get('read_coverage_fraction', 0):.1%} / "
            f"{full.get('nonempty_read_coverage_fraction', 0):.1%} / "
            f"{full.get('owner_path_coverage_fraction', 0):.1%}.",
            f"- Raw→gated residual ratio: {full.get('raw_to_gated_residual_ratio', 0):.4f}; "
            f"mean retrieve gate: {full.get('mean_retrieve_gate', 0):.4f}.",
            f"- Write funnel: argmax={funnel.get('memorize_argmax_count', 0)}, "
            f"candidate={funnel.get('write_candidate_count', 0)}, "
            f"gate-pass={funnel.get('write_gate_pass_count', 0)}, "
            f"priority-pass={funnel.get('write_priority_pass_count', 0)}, "
            f"window-complete={funnel.get('write_window_complete_count', 0)}, "
            f"accepted={funnel.get('write_accepted_count', 0)}.",
            f"- Accepted-write reuse/beneficial rate: "
            f"{funnel.get('accepted_reuse_fraction', 0):.1%} / "
            f"{funnel.get('accepted_beneficial_fraction', 0):.1%}.",
            f"- Retrieval/NLL correlations: `{episodic_diagnostics}`.",
        ])
    tree = summary["tree"]
    lines.extend([
        "", "## Tree and memory health", "",
        f"- Nodes/leaves: {tree['node_count']}/{tree['leaf_count']}; max depth: {tree['max_depth']}.",
        f"- Owner top-1 share: {tree['owner_top1_share']:.1%}; effective owners: {tree['effective_owner_count']:.2f}.",
        f"- Memory rows: {tree['total_memory_rows']} (internal={tree['internal_memory_rows']}, leaf={tree['leaf_memory_rows']}).",
        f"- Label diagnostics: `{summary['routing_labels']}`.",
        "", "## Warnings", "",
    ])
    lines.extend(f"- {warning}" for warning in summary["warnings"])
    if not summary["warnings"]:
        lines.append("- No threshold-based health warning was triggered.")
    lines.extend(["", "> Time MAE/RMSE use the model's documented local-constant-rate approximation."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(
    output_dir: Path,
    summary: Mapping[str, Any],
    ablations: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(summary["variants"])
    nlls = [summary["variants"][name]["nll_per_event"] for name in names]
    accuracies = [summary["variants"][name]["accuracy"] for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(names, nlls)
    axes[0].set_title("NLL per event")
    axes[1].bar(names, accuracies)
    axes[1].set_title("Next-type accuracy")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_ablation.png", dpi=160)
    plt.close(figure)
    if ablations:
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.bar([row["comparison"] for row in ablations], [row["mean_nll_gain"] for row in ablations])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Positive means lower NLL")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(output_dir / "memory_gain.png", dpi=160)
        plt.close(figure)
    if history:
        x = [row.get("epoch", index + 1) for index, row in enumerate(history)]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for key in ("wake_nll", "global_nll"):
            if all(isinstance(row.get(key), (int, float)) for row in history):
                axes[0].plot(x, [row[key] for row in history], label=key)
        for key in ("leaf_count", "writes", "structural_actions"):
            if all(isinstance(row.get(key), (int, float)) for row in history):
                axes[1].plot(x, [row[key] for row in history], label=key)
        axes[0].set_title("Prediction history")
        axes[1].set_title("Tree and memory growth")
        for axis in axes:
            axis.set_xlabel("epoch")
            axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "training_growth.png", dpi=160)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Hawkes Memory Tree checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", choices=("frozen", "online", "both"), default="both")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prototype-duplicate-threshold", type=float, default=None)
    parser.add_argument("--prototype-mode-threshold", type=float, default=None)
    parser.add_argument(
        "--prototype-context-alias-capacity", type=int, default=None,
        help="Override the number of retrieval/context aliases per law prototype.",
    )
    parser.add_argument("--max-test-sequences", type=int, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--quick-per-cluster", type=int, default=None)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse completed variant row files")
    parser.add_argument(
        "--save-event-predictions", action="store_true",
        help="Write the large combined event_predictions.csv artifact.",
    )
    parser.add_argument(
        "--write-baseline-summary", type=Path, default=None,
        help="Optional immutable v6.1 summary for Write ranking comparisons.",
    )
    parser.add_argument(
        "--verify-frozen-base", action="store_true",
        help="Re-run the recorded base checkpoint and require identical frozen outputs",
    )
    parser.add_argument(
        "--reference-summary", type=Path, default=None,
        help="Optional v4 summary.json used for ACC/Macro-F1 non-regression.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, type_map = load_dataset(args.data_path)
    checkpoint_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    router_kind = checkpoint_meta.get("model_config", {}).get("router_kind")
    if router_kind not in SUPPORTED_ROUTER_KINDS:
        raise ValueError(
            f"incompatible checkpoint {args.checkpoint}: router_kind="
            f"{router_kind!r}. Evaluate.py requires one of "
            f"{sorted(SUPPORTED_ROUTER_KINDS)}. For the tree_13 data in this "
            "repository, use Memory/Checkpoints/"
            "dws_13_posterior_frontier_v2.pt. Legacy z-only checkpoints "
            "cannot be faithfully converted because they do not contain the "
            "current Compat/frontier router parameters."
        )
    expected_types = int(checkpoint_meta["model_config"]["num_event_types"])
    if len(type_map) != expected_types:
        raise ValueError(f"dataset/checkpoint type mismatch: {len(type_map)} != {expected_types}")
    manifest = None
    if args.split_manifest is not None:
        manifest = load_split_manifest(args.split_manifest, data_path=args.data_path)
        split = manifest["splits"]
        selected_sources = list(map(int, split["test"]))
        by_source = {int(row["source_index"]): row for row in dataset}
        missing = set(selected_sources).difference(by_source)
        if missing:
            raise ValueError(f"dataset is missing {len(missing)} manifest test rows")
        sequences = [by_source[index] for index in selected_sources]
    else:
        split = make_split(len(dataset), args.seed, args.val_ratio, args.test_ratio)
        sequences = [dataset[index] for index in split["test"]]
    if args.quick_per_cluster is not None:
        if args.quick_per_cluster <= 0:
            raise ValueError("--quick-per-cluster must be positive")
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sequence in sequences:
            if "cluster_id" not in sequence:
                raise ValueError("--quick-per-cluster requires cluster labels")
            grouped[int(sequence["cluster_id"])].append(sequence)
        sequences = [
            row for cluster in sorted(grouped)
            for row in grouped[cluster][:args.quick_per_cluster]
        ]
        if any(len(rows) < args.quick_per_cluster for rows in grouped.values()):
            raise ValueError("a test cluster has too few sequences for quick sampling")
    if args.max_test_sequences is not None:
        sequences = sequences[:args.max_test_sequences]
    if not sequences:
        raise ValueError("no test sequences selected")
    evaluation_regime, data_provenance = validate_evaluation_provenance(
        checkpoint_meta,
        manifest,
        [int(sequence["source_index"]) for sequence in sequences],
        args.split_manifest,
    )
    print(f"[Preflight] validating checkpoint on source sequence "
          f"{sequences[0]['source_index']}")
    preflight_checkpoint(
        args.checkpoint,
        sequences[0],
        args.device,
        args.prototype_duplicate_threshold,
        args.prototype_mode_threshold,
        args.prototype_context_alias_capacity,
    )
    print("[Preflight] passed")
    variants = args.variants
    if variants is None:
        variants = ["full_frozen", "no_episodic", "no_working", "semantic_only"]
        if args.protocol == "online":
            variants = ["full_online"]
        elif args.protocol == "both":
            variants.append("full_online")

    all_rows: dict[str, list[dict]] = {}
    variant_metrics = {}
    final_inference = None
    for variant in variants:
        print(f"[Evaluate] {variant}: {len(sequences)} sequences")
        completed_path = args.output_dir / f"{variant}.rows.json"
        # Inference is still loaded for tree diagnostics. Completed rows can
        # be reused safely because each variant starts from the same checkpoint.
        if args.resume and completed_path.is_file():
            rows = json.loads(completed_path.read_text(encoding="utf-8"))
            inference = MemoryTreeInference.from_checkpoint(
                args.checkpoint,
                device=args.device,
                inference_config=InferenceConfig(
                    prototype_duplicate_threshold=args.prototype_duplicate_threshold,
                    prototype_mode_threshold=args.prototype_mode_threshold,
                    prototype_context_alias_capacity=args.prototype_context_alias_capacity,
                ),
            )
            elapsed = 0.0
            print(f"[Resume] reused completed variant {variant}")
        else:
            rows, inference, elapsed = run_variant(
                args.checkpoint, sequences, variant, args.device, args.output_dir,
                resume_partial=args.resume,
                prototype_duplicate_threshold=args.prototype_duplicate_threshold,
                prototype_mode_threshold=args.prototype_mode_threshold,
                prototype_context_alias_capacity=args.prototype_context_alias_capacity,
            )
            completed_path.write_text(json.dumps(_jsonable(rows)), encoding="utf-8")
        all_rows[variant] = rows
        metrics = aggregate_metrics(
            rows,
            expected_types,
            args.seed,
            args.bootstrap_samples,
        )
        metrics.update({
            "elapsed_seconds": elapsed,
            "events_per_second": len(rows) / max(elapsed, 1e-12),
        })
        variant_metrics[variant] = metrics
        if final_inference is None or variant == "full_frozen":
            final_inference = inference
    assert final_inference is not None
    retrieval_counterfactual = annotate_retrieval_counterfactual_gain(all_rows)
    if "full_frozen" in all_rows:
        (args.output_dir / "full_frozen.rows.json").write_text(
            json.dumps(_jsonable(all_rows["full_frozen"])), encoding="utf-8"
        )
    diagnostic_rows = all_rows.get("full_frozen", all_rows[variants[0]])
    tree, node_rows = tree_metrics(final_inference, diagnostic_rows)
    ablations = ablation_metrics(all_rows, args.seed)
    controller = controller_metrics(all_rows, variant_metrics, checkpoint_meta)
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_path": str(args.data_path.resolve()),
        "data_sha256": dataset_fingerprint(args.data_path),
        "evaluation_regime": evaluation_regime,
        "data_provenance": data_provenance,
        "split_seed": args.seed,
        "test_source_indices": [int(sequence["source_index"]) for sequence in sequences],
        "type_map": type_map,
        "variants": variant_metrics,
        "ablations": ablations,
        "controller": controller,
        "retrieval_counterfactual": retrieval_counterfactual,
        "tree": tree,
        "routing_labels": routing_label_metrics(
            diagnostic_rows,
            {int(sequence["cluster_id"]) for sequence in dataset if "cluster_id" in sequence},
        ),
        "checkpoint_history_epochs": len(checkpoint_meta.get("history", [])),
        "note": (
            "Strict held-out: THP, Attention selection, H-tree node pooling, and Memory "
            "were validated against the same manifest."
            if evaluation_regime == "strict_inductive" else
            "Transductive upstream artifacts: this report does not claim strict held-out evaluation."
        ),
    }
    frozen_rows = all_rows.get("full_frozen", ())
    frozen_actual_sha = frozen_event_sha256(frozen_rows) if frozen_rows else None
    rollout_meta = checkpoint_meta.get("write_rollout_calibration", {})
    expected_frozen_sha = rollout_meta.get("frozen_event_sha256")
    expected_frozen_sources = rollout_meta.get("frozen_source_indices")
    actual_frozen_sources = sorted({int(row["source_index"]) for row in frozen_rows})
    comparable_frozen_set = bool(
        expected_frozen_sources is not None
        and actual_frozen_sources == sorted(map(int, expected_frozen_sources))
    )
    frozen_base_checkpoint = None
    trainable_heads = set(
        checkpoint_meta.get("controller_state", {}).get("trainable_heads", ())
    )
    write_isolated_policy = bool(rollout_meta) or trainable_heads == {"write"}
    if args.verify_frozen_base and write_isolated_policy:
        frozen_base_checkpoint = (
            rollout_meta.get("base_checkpoint")
            or checkpoint_meta.get("controller_only_invariants", {}).get(
                "base_checkpoint"
            )
        )
        if not frozen_base_checkpoint:
            raise ValueError(
                "--verify-frozen-base requires base checkpoint metadata"
            )
        frozen_base_path = Path(frozen_base_checkpoint)
        if not frozen_base_path.is_absolute():
            frozen_base_path = Path.cwd() / frozen_base_path
        if not frozen_base_path.is_file():
            raise FileNotFoundError(
                f"frozen-policy base checkpoint is missing: {frozen_base_path}"
            )
        base_rows, _, _ = run_variant(
            frozen_base_path,
            sequences,
            "full_frozen",
            args.device,
            prototype_duplicate_threshold=args.prototype_duplicate_threshold,
            prototype_mode_threshold=args.prototype_mode_threshold,
            prototype_context_alias_capacity=args.prototype_context_alias_capacity,
        )
        expected_frozen_sha = frozen_event_sha256(base_rows)
        expected_frozen_sources = actual_frozen_sources
        comparable_frozen_set = True
    module_state = checkpoint_meta.get("controller_state", {}).get(
        "module_state_dict", {}
    )
    controller_invariants = checkpoint_meta.get("controller_only_invariants", {})
    retrieve_actual_sha = (
        head_policy_sha256(module_state, "retrieve") if module_state else None
    )
    adapt_actual_sha = (
        head_policy_sha256(module_state, "adapt") if module_state else None
    )
    retrieve_expected_sha = (
        rollout_meta.get("retrieve_policy_sha256")
        or controller_invariants.get("retrieve_policy_expected_sha256")
    )
    adapt_expected_sha = (
        rollout_meta.get("adapt_policy_sha256")
        or controller_invariants.get("adapt_policy_expected_sha256")
    )
    policy_invariants = {
        "frozen_event_expected_sha256": expected_frozen_sha,
        "frozen_event_actual_sha256": frozen_actual_sha,
        "frozen_event_verified": (
            None if expected_frozen_sha is None or not comparable_frozen_set
            else frozen_actual_sha == expected_frozen_sha
        ),
        "frozen_event_comparable_dataset": comparable_frozen_set,
        "frozen_base_checkpoint": frozen_base_checkpoint,
        "retrieve_policy_sha256": retrieve_actual_sha,
        "retrieve_policy_expected_sha256": retrieve_expected_sha,
        "retrieve_policy_verified": (
            None if retrieve_expected_sha is None
            else retrieve_actual_sha == retrieve_expected_sha
        ),
        "adapt_policy_sha256": adapt_actual_sha,
        "adapt_policy_expected_sha256": adapt_expected_sha,
        "adapt_policy_verified": (
            None if adapt_expected_sha is None
            else adapt_actual_sha == adapt_expected_sha
        ),
    }
    summary["controller_policy_invariants"] = policy_invariants
    policy_revision = int(checkpoint_meta.get("controller_policy_revision", 0))
    write_baseline_comparison = None
    if args.write_baseline_summary is not None:
        baseline_summary = json.loads(
            args.write_baseline_summary.read_text(encoding="utf-8")
        )
        current_online = variant_metrics.get("full_online", {})
        baseline_online = baseline_summary.get("variants", {}).get("full_online", {})
        current_write = controller.get("write", {}).get("ranking", {})
        baseline_write = baseline_summary.get("controller", {}).get("write", {}).get(
            "ranking", {}
        )
        if not baseline_write:
            baseline_rows_path = (
                args.write_baseline_summary.parent
                / "full_online_no_write.rows.json"
            )
            if baseline_rows_path.is_file():
                baseline_probe_rows = [
                    row for row in json.loads(
                        baseline_rows_path.read_text(encoding="utf-8")
                    )
                    if row.get("write_utility") is not None
                ]
                baseline_write = _write_ranking_metrics(
                    baseline_probe_rows, seed=args.seed
                )
        write_baseline_comparison = {
            "baseline_summary": str(args.write_baseline_summary),
            "candidate_full_online_nll": current_online.get("nll_per_event"),
            "baseline_full_online_nll": baseline_online.get("nll_per_event"),
            "online_nll_not_worse": bool(
                current_online.get("nll_per_event", float("inf"))
                <= baseline_online.get("nll_per_event", float("inf"))
            ),
            "candidate_ndcg_at_4": current_write.get("ndcg_at_4"),
            "baseline_ndcg_at_4": baseline_write.get("ndcg_at_4"),
            "candidate_top4_utility": current_write.get("top4_cumulative_utility"),
            "baseline_top4_utility": baseline_write.get("top4_cumulative_utility"),
            "ndcg_at_4_improved": bool(
                current_write.get("ndcg_at_4", float("-inf"))
                > baseline_write.get("ndcg_at_4", float("inf"))
            ),
            "top4_utility_improved": bool(
                current_write.get("top4_cumulative_utility", float("-inf"))
                > baseline_write.get("top4_cumulative_utility", float("inf"))
            ),
        }
        summary["write_ranking_baseline_comparison"] = write_baseline_comparison
    ablation_by_name = {row["comparison"]: row for row in ablations}
    total_gain = ablation_by_name.get("total_memory_gain", {}).get("mean_nll_gain")
    online_harm = ablation_by_name.get("online_vs_frozen_gain", {}).get(
        "negative_gain_fraction"
    )
    online_row = ablation_by_name.get("online_vs_frozen_gain", {})
    online_gain = online_row.get(
        "mean_nll_gain"
    )
    online_interval = online_row.get("bootstrap_95ci") or ()
    online_ci_lower = (
        float(online_interval[0]) if len(online_interval) >= 2 else None
    )
    write = controller.get("write", {})
    retrieve = controller.get("retrieve", {})
    adapt = controller.get("adapt", {})
    sequence_count = max(len(sequences), 1)
    accepted_writes = int(write.get("accepted_count", 0))
    memory_target = 0.000254 if args.quick_per_cluster is not None else 0.00038
    reference_check = None
    reference_note = "ACC/Macro-F1 v4 reference summary was not supplied."
    if args.reference_summary is not None:
        reference = json.loads(args.reference_summary.read_text(encoding="utf-8"))
        current_frozen = variant_metrics.get("full_frozen", {})
        reference_frozen = reference.get("variants", {}).get("full_frozen", {})
        required = ("accuracy", "macro_f1")
        if not all(key in current_frozen and key in reference_frozen for key in required):
            raise ValueError("reference summary lacks full_frozen accuracy/macro_f1")
        reference_check = all(
            float(current_frozen[key]) >= float(reference_frozen[key]) - 1e-12
            for key in required
        )
        reference_note = f"Compared against v4 summary: {args.reference_summary}"
    ranking = write.get("ranking", {})
    research_checks = {
        "mean_writes_le_2": accepted_writes / sequence_count <= 2.0,
        "write_budget_utilization_le_0_5": accepted_writes / (4 * sequence_count) <= 0.5,
        "memory_gain_target": (
            True if policy_revision >= 4
            else total_gain is not None and total_gain > memory_target
        ),
        "adapt_spearman_ge_0_35": (
            adapt.get("spearman_gate_utility") is not None
            and adapt["spearman_gate_utility"] >= 0.35
        ),
        "retrieve_spearman_gt_0_10": (
            retrieve.get("spearman_gate_utility") is not None
            and retrieve["spearman_gate_utility"] > 0.10
        ),
        "write_roc_auc_gt_0_60": (
            True if policy_revision >= 4 else (
                write.get("auc_sample_sufficient") is True
                and write.get("beneficial_roc_auc") is not None
                and write["beneficial_roc_auc"] > 0.60
            )
        ),
        "write_ranking_spearman_gt_0_10": (
            True if policy_revision < 4 else
            write.get("spearman_gate_utility") is not None
            and write["spearman_gate_utility"] > 0.10
        ),
        "write_pairwise_accuracy_gt_0_55": (
            True if policy_revision < 4 else
            ranking.get("pairwise_accuracy", 0.0) > 0.55
        ),
        "write_top4_uplift_positive": (
            True if policy_revision < 4 else
            ranking.get("top4_uplift_over_random", 0.0) > 0.0
        ),
        "write_ranking_better_than_baseline": (
            True if policy_revision < 4 or write_baseline_comparison is None
            else write_baseline_comparison["online_nll_not_worse"]
            and write_baseline_comparison["ndcg_at_4_improved"]
            and write_baseline_comparison["top4_utility_improved"]
        ),
        "write_positive_negative_ge_20": (
            write.get("positive_count", 0) >= 20
            and write.get("negative_count", 0) >= 20
        ),
        "online_harmful_fraction_lt_0_45": (
            online_harm is not None and online_harm < 0.45
        ),
        "full_online_gain_gt_0": (
            online_gain is not None
            and online_gain > 1e-6
            and online_ci_lower is not None
            and online_ci_lower > 0.0
        ),
        "frozen_acc_macro_f1_not_below_v4": reference_check,
        "frozen_controller_output_unchanged": policy_invariants[
            "frozen_event_verified"
        ],
        "retrieve_policy_unchanged": policy_invariants[
            "retrieve_policy_verified"
        ],
        "adapt_policy_unchanged": policy_invariants[
            "adapt_policy_verified"
        ],
    }
    summary["controller_research_acceptance"] = {
        "passed": all(value is True for value in research_checks.values() if value is not None),
        "checks": research_checks,
        "mean_writes_per_sequence": accepted_writes / sequence_count,
        "write_budget_utilization": accepted_writes / (4 * sequence_count),
        "note": reference_note,
    }
    summary["warnings"] = warnings_for(summary)
    (args.output_dir / "split_indices.json").write_text(
        json.dumps({"data_sha256": summary["data_sha256"], "splits": split}, indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flat_rows = [row for rows in all_rows.values() for row in rows]
    if args.save_event_predictions:
        write_csv(args.output_dir / "event_predictions.csv", flat_rows)
    write_csv(args.output_dir / "node_metrics.csv", node_rows)
    write_csv(args.output_dir / "ablation_metrics.csv", ablations)
    controller_rows = []
    for action, metrics in controller.items():
        if metrics.get("available", False):
            controller_rows.append({
                "action": action,
                **{key: value for key, value in metrics.items() if key != "utility_deciles"},
            })
    write_csv(args.output_dir / "controller_metrics.csv", controller_rows)
    decile_rows = [
        {"action": action, **row}
        for action, metrics in controller.items()
        for row in metrics.get("utility_deciles", [])
    ]
    write_csv(args.output_dir / "controller_utility_deciles.csv", decile_rows)
    history_rows = []
    for epoch_row in checkpoint_meta.get("history", []):
        history_rows.append({
            key: value
            for key, value in epoch_row.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        })
    write_csv(args.output_dir / "training_history.csv", history_rows)
    sequence_rows = []
    for variant, rows in all_rows.items():
        groups: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            groups[int(row["source_index"])].append(row)
        for source_index, group in groups.items():
            sequence_rows.append({
                "variant": variant,
                "source_index": source_index,
                "cluster_id": group[0].get("cluster_id"),
                "events": len(group),
                "nll_per_event": _mean(float(row["nll"]) for row in group),
                "accuracy": _mean(float(max(range(expected_types), key=lambda index: row["type_probabilities"][index]) == row["true_type"]) for row in group),
                "local_time_mae": _mean(abs(float(row["predicted_time"]) - float(row["true_time"])) for row in group),
            })
    write_csv(args.output_dir / "sequence_metrics.csv", sequence_rows)
    write_report(args.output_dir / "report.md", summary)
    if not args.no_plots:
        make_plots(args.output_dir, summary, ablations, history_rows)
    print(f"[Done] report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
