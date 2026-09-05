"""Continual-learning evaluation for Hawkes Memory Tree checkpoints.

The ordinary :mod:`Evaluate` entry point evaluates one flat CSV with one
train/validation/test split.  CL has a different contract: every checkpoint
is evaluated on its current task, all previous task test sets, and the
independent frozen anchor banks.

Run from the repository root with ``PYTHONPATH`` containing both the project
root and ``Memory``::

    PYTHONPATH="$PWD:$PWD/Memory" python -u -m EvaluateCL \
      --data-root "$PWD/Data/CL/Data" \
      --checkpoint-dir "$PWD/Memory/Checkpoints/CL" \
      --output-dir "$PWD/Memory/Eval/CL" \
      --protocol both \
      --device cuda
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from Evaluate import (
        VARIANTS,
        SUPPORTED_ROUTER_KINDS,
        clear_episodic_memory,
        _jsonable,
        aggregate_metrics,
        dataset_fingerprint,
        run_variant,
        write_csv,
    )
except ModuleNotFoundError:
    from Memory.Evaluate import (
        VARIANTS,
        SUPPORTED_ROUTER_KINDS,
        clear_episodic_memory,
        _jsonable,
        aggregate_metrics,
        dataset_fingerprint,
        run_variant,
        write_csv,
    )

from Train.Inference import InferenceConfig, MemoryTreeInference


TASK_RE = re.compile(r"^task_(\d+)$")
CHECKPOINT_RE = re.compile(r"^task_(\d+)\.pt$")
SCALAR_METRICS = (
    "events",
    "sequences",
    "nll_per_event",
    "sequence_macro_nll",
    "perplexity",
    "accuracy",
    "top3_accuracy",
    "macro_f1",
    "micro_f1",
    "cross_entropy",
    "brier_score",
    "ece_10bin",
    "local_time_mae",
    "local_time_rmse",
    "local_time_median_ae",
    "memory_hit_fraction",
    "read_coverage_fraction",
    "nonempty_read_coverage_fraction",
    "owner_path_coverage_fraction",
    "mean_retrieval_alpha_mass",
    "mean_retrieval_effective_k",
    "mean_retrieval_similarity",
    "mean_retrieval_null_alpha",
    "mean_episodic_residual_norm",
    "mean_raw_episodic_residual_norm",
    "mean_gated_episodic_residual_norm",
    "raw_to_gated_residual_ratio",
    "mean_retrieve_gate",
)


@dataclass(frozen=True)
class EvaluationSet:
    """One model-facing CL evaluation CSV."""

    name: str
    kind: str
    path: Path
    task_id: int | None = None
    regime_id: str | None = None
    stage_label: str | None = None


@dataclass(frozen=True)
class GroundTruthLaw:
    """Positive Hawkes parameters used only by the CL evaluation layer."""

    regime_id: str
    mu: np.ndarray
    W: np.ndarray
    betas: np.ndarray
    kind: str
    parent_regime: str | None = None


def _parse_sequence_list(value: Any, cast: type) -> list[Any]:
    """Parse the Python/JSON list representation used by CL CSV files."""

    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"cannot parse sequence value {value!r}") from error
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            f"sequence value must be a list, got {type(parsed).__name__}"
        )
    return [cast(item) for item in parsed]


def _load_cl_dataset(
    path: Path,
    expected_types: int,
    max_sequences: int | None = None,
) -> list[dict[str, Any]]:
    """Load IDs directly; do not infer a new sparse mapping for each split."""

    frame = pd.read_csv(path)
    required = {"event_times", "event_types"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    dataset: list[dict[str, Any]] = []
    for source_index, row in frame.iterrows():
        try:
            times = _parse_sequence_list(row["event_times"], float)
            types = _parse_sequence_list(row["event_types"], int)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid sequence at {path}:{source_index + 2}"
            ) from error
        if not times or len(times) != len(types):
            raise ValueError(
                f"{path}:{source_index + 2} has mismatched or empty events"
            )
        if any(not math.isfinite(value) for value in times):
            raise ValueError(f"{path}:{source_index + 2} contains non-finite times")
        if times[0] < 0.0 or any(b < a for a, b in zip(times, times[1:])):
            raise ValueError(
                f"{path}:{source_index + 2} event_times must be non-decreasing"
            )
        if any(event_type < 0 or event_type >= expected_types for event_type in types):
            raise ValueError(
                f"{path}:{source_index + 2} contains an event type outside "
                f"[0, {expected_types - 1}]"
            )
        dataset.append({
            "times": torch.tensor(times, dtype=torch.float32),
            "types": torch.tensor(types, dtype=torch.long),
            "source_index": int(source_index),
        })

    if max_sequences is not None:
        dataset = dataset[:max_sequences]
    if not dataset:
        raise ValueError(f"{path} contains no valid sequences")
    return dataset


def _normalise_data_root(path: Path) -> Path:
    """Accept either ``Data/CL`` or the generated ``Data/CL/Data`` root."""

    path = path.expanduser()
    if (path / "task_00").is_dir():
        return path
    nested = path / "Data"
    if (nested / "task_00").is_dir():
        return nested
    return path


def _discover_task_sets(data_root: Path) -> dict[int, EvaluationSet]:
    result: dict[int, EvaluationSet] = {}
    for directory in sorted(data_root.glob("task_*")):
        match = TASK_RE.fullmatch(directory.name)
        test_path = directory / "test.csv"
        if match is None or not test_path.is_file():
            continue
        task_id = int(match.group(1))
        result[task_id] = EvaluationSet(
            name=f"task_{task_id:02d}_test",
            kind="task_test",
            path=test_path,
            task_id=task_id,
        )
    return result


def _discover_checkpoints(checkpoint_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(checkpoint_dir.glob("task_*.pt")):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match is not None and path.is_file():
            result[int(match.group(1))] = path
    return result


def _read_stage_metadata(data_root: Path) -> dict[int, dict[str, Any]]:
    """Read labels from the oracle manifest; never use its parameters."""

    path = data_root / "stream_manifest.csv"
    if not path.is_file():
        return {}
    metadata: dict[int, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_task = str(row.get("task_id", ""))
            if not raw_task.isdigit():
                continue
            task_id = int(raw_task)
            metadata.setdefault(task_id, {
                "stage_label": row.get("stage_label"),
                "regime_id": row.get("regime_id"),
                "shift_type": row.get("shift_type"),
                "recurrence_of": row.get("recurrence_of") or None,
                "regime_weights": row.get("regime_weights"),
            })
    return metadata


def _first_seen_regimes(stage_metadata: Mapping[int, Mapping[str, Any]]) -> dict[str, int]:
    """Return the first task in which each pure law is present.

    A mixture task contributes every component in ``regime_weights``.  This
    makes seen-law averages different from a naive average over task IDs:
    recurrence of A_1 must not count as a newly seen law.
    """

    first_seen: dict[str, int] = {}
    for task_id in sorted(stage_metadata):
        metadata = stage_metadata[task_id]
        regimes: list[str] = []
        raw_weights = metadata.get("regime_weights")
        if raw_weights:
            try:
                parsed = json.loads(str(raw_weights))
                if isinstance(parsed, dict):
                    regimes.extend(str(key) for key in parsed)
            except json.JSONDecodeError:
                pass
        if not regimes and metadata.get("regime_id"):
            regimes.append(str(metadata["regime_id"]))
        for regime_id in regimes:
            first_seen.setdefault(regime_id, task_id)
    return first_seen


def _discover_anchors(data_root: Path) -> list[EvaluationSet]:
    return [
        EvaluationSet(
            name=f"anchor_{path.stem}",
            kind="anchor",
            path=path,
            regime_id=path.stem,
            stage_label="frozen_anchor",
        )
        for path in sorted((data_root / "anchors").glob("*.csv"))
        if not path.name.startswith("._") and not path.name.startswith(".")
    ]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_scalar(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    return value


def _mean(values: Iterable[Any]) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return sum(clean) / len(clean) if clean else None


def _checkpoint_meta(path: Path) -> dict[str, Any]:
    metadata = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(metadata, dict):
        raise TypeError(f"checkpoint must contain a dictionary: {path}")
    router_kind = metadata.get("model_config", {}).get("router_kind")
    if router_kind not in SUPPORTED_ROUTER_KINDS:
        raise ValueError(
            f"incompatible CL checkpoint {path}: router_kind={router_kind!r}; "
            f"expected one of {sorted(SUPPORTED_ROUTER_KINDS)}"
        )
    config = metadata.get("model_config", {})
    if "num_event_types" not in config:
        raise KeyError(f"checkpoint has no model_config.num_event_types: {path}")
    return metadata


def _tree_health(checkpoint: Path) -> dict[str, Any]:
    """Collect topology facts without evaluating any data."""

    inference = MemoryTreeInference.from_checkpoint(
        checkpoint,
        device="cpu",
        inference_config=InferenceConfig(
            adapt_working_memory=False,
            allow_memory_writes=False,
            update_memory_usage=False,
        ),
    )
    tree = inference.tree
    leaf_depths = [int(tree.nodes[node_id].depth) for node_id in tree.leaf_ids]
    memory_rows = sum(len(bank) for bank in tree.episodic_memory.banks.values())
    return {
        "node_count": len(tree.all_node_ids),
        "leaf_count": len(tree.leaf_ids),
        "leaf_ids": list(tree.leaf_ids),
        "max_depth": max(leaf_depths, default=0),
        "mean_leaf_depth": _mean(leaf_depths),
        "memory_rows": memory_rows,
    }


def _select_variants(
    protocol: str,
    requested: Sequence[str] | None,
) -> list[str]:
    if requested:
        return list(requested)
    if protocol == "frozen":
        return ["full_frozen", "no_episodic", "no_working", "semantic_only"]
    if protocol == "online":
        return ["full_online", "full_online_no_write"]
    return [
        "full_frozen",
        "no_episodic",
        "no_working",
        "semantic_only",
        "full_online",
        "full_online_no_write",
    ]


def _cache_dir(
    output_dir: Path,
    checkpoint_task: int,
    evaluation_set: EvaluationSet,
    variant: str,
) -> Path:
    return (
        output_dir
        / "cache"
        / f"checkpoint_task_{checkpoint_task:02d}"
        / _safe_name(evaluation_set.name)
        / _safe_name(variant)
    )


def _load_or_run(
    *,
    checkpoint: Path,
    checkpoint_task: int,
    evaluation_set: EvaluationSet,
    variant: str,
    sequences: Sequence[Mapping[str, Any]],
    data_sha256: str,
    checkpoint_sha256: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], float, bool]:
    """Run one matrix cell, optionally reusing a completed cell."""

    cache = _cache_dir(args.output_dir, checkpoint_task, evaluation_set, variant)
    rows_path = cache / "rows.json"
    meta_path = cache / "meta.json"
    expected_meta = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "data_path": str(evaluation_set.path.resolve()),
        "data_sha256": data_sha256,
        "variant": variant,
        "sequence_count": len(sequences),
    }
    cache_valid = False
    if args.resume and rows_path.is_file() and meta_path.is_file():
        try:
            cache_valid = (
                json.loads(meta_path.read_text(encoding="utf-8"))
                == expected_meta
            )
        except (OSError, json.JSONDecodeError):
            cache_valid = False
    if cache_valid:
        return json.loads(rows_path.read_text(encoding="utf-8")), 0.0, True

    progress_dir = None
    if args.resume and not VARIANTS[variant]["online"]:
        progress_dir = cache
        cache.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(expected_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif args.resume:
        cache.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(expected_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    rows, _inference, elapsed = run_variant(
        checkpoint,
        sequences,
        variant,
        args.device,
        progress_dir=progress_dir,
        resume_partial=(
            args.resume
            and not VARIANTS[variant]["online"]
            and not rows_path.is_file()
        ),
        prototype_duplicate_threshold=args.prototype_duplicate_threshold,
        prototype_mode_threshold=args.prototype_mode_threshold,
        prototype_context_alias_capacity=args.prototype_context_alias_capacity,
        verbose=args.verbose,
    )
    if args.resume:
        rows_path.write_text(
            json.dumps(_jsonable(rows), ensure_ascii=False), encoding="utf-8"
        )
    return rows, elapsed, False


def _metric_row(
    *,
    checkpoint_task: int,
    checkpoint: Path,
    evaluation_set: EvaluationSet,
    variant: str,
    metrics: Mapping[str, Any],
    tree: Mapping[str, Any],
    elapsed: float,
    from_cache: bool,
    data_sha256: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checkpoint_task": checkpoint_task,
        "checkpoint": str(checkpoint.resolve()),
        "eval_name": evaluation_set.name,
        "eval_kind": evaluation_set.kind,
        "eval_task": evaluation_set.task_id,
        "regime_id": evaluation_set.regime_id,
        "stage_label": evaluation_set.stage_label,
        "data_path": str(evaluation_set.path.resolve()),
        "data_sha256": data_sha256,
        "variant": variant,
        "elapsed_seconds": elapsed,
        "from_cache": from_cache,
        "leaf_count": tree.get("leaf_count"),
        "node_count": tree.get("node_count"),
        "max_depth": tree.get("max_depth"),
        "mean_leaf_depth": tree.get("mean_leaf_depth"),
        "memory_rows": tree.get("memory_rows"),
    }
    for name in SCALAR_METRICS:
        if name in metrics:
            row[name] = _finite_scalar(metrics[name])
    events = metrics.get("events")
    row["events_per_second"] = (
        float(events) / elapsed if events and elapsed > 0.0 else None
    )
    return row


def _decorate_event_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fields = {
        "checkpoint_task": metric_row["checkpoint_task"],
        "checkpoint": metric_row["checkpoint"],
        "eval_name": metric_row["eval_name"],
        "eval_kind": metric_row["eval_kind"],
        "eval_task": metric_row["eval_task"],
        "regime_id": metric_row["regime_id"],
        "stage_label": metric_row["stage_label"],
    }
    return [{**fields, **dict(row)} for row in rows]


class _EventPredictionWriter:
    """Stream event predictions so a full CL matrix stays memory-bounded."""

    def __init__(self, path: Path) -> None:
        self.handle = path.open("w", newline="", encoding="utf-8-sig")
        self.writer: csv.DictWriter | None = None
        self.fieldnames: list[str] = []

    def write(
        self,
        rows: Sequence[Mapping[str, Any]],
        metric_row: Mapping[str, Any],
    ) -> None:
        decorated = _decorate_event_rows(rows, metric_row)
        if not decorated:
            return
        if self.writer is None:
            self.fieldnames = sorted({
                key for row in decorated for key in row
            })
            self.writer = csv.DictWriter(
                self.handle,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )
            self.writer.writeheader()
        for row in decorated:
            values = {}
            for key in self.fieldnames:
                value = row.get(key)
                if isinstance(value, (list, dict, tuple)):
                    value = json.dumps(
                        _jsonable(value), ensure_ascii=False
                    )
                values[key] = value
            self.writer.writerow(values)

    def close(self) -> None:
        self.handle.close()


def _continual_summary(
    metric_rows: Sequence[Mapping[str, Any]],
    checkpoint_tasks: Sequence[int],
    variants: Sequence[str],
    tree_by_checkpoint: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize current-task quality and positive NLL forgetting."""

    output: list[dict[str, Any]] = []
    task_rows = [row for row in metric_rows if row["eval_kind"] == "task_test"]
    anchor_rows = [row for row in metric_rows if row["eval_kind"] == "anchor"]
    for variant in variants:
        for checkpoint_task in checkpoint_tasks:
            current = next(
                (
                    row for row in task_rows
                    if row["variant"] == variant
                    and row["checkpoint_task"] == checkpoint_task
                    and row["eval_task"] == checkpoint_task
                ),
                None,
            )
            seen = [
                row for row in task_rows
                if row["variant"] == variant
                and row["checkpoint_task"] == checkpoint_task
                and row["eval_task"] is not None
                and int(row["eval_task"]) <= checkpoint_task
            ]
            anchors = [
                row for row in anchor_rows
                if row["variant"] == variant
                and row["checkpoint_task"] == checkpoint_task
            ]
            forgetting: list[float] = []
            seen_task_ids = sorted({
                int(row["eval_task"])
                for row in seen
                if row["eval_task"] is not None
            })
            for task_id in seen_task_ids:
                if task_id >= checkpoint_task:
                    continue
                latest = next(
                    (row for row in seen if int(row["eval_task"]) == task_id),
                    None,
                )
                history = [
                    row for row in task_rows
                    if row["variant"] == variant
                    and int(row["eval_task"]) == task_id
                    and int(row["checkpoint_task"]) < checkpoint_task
                    and row.get("nll_per_event") is not None
                ]
                if latest is not None and history:
                    forgetting.append(
                        max(
                            0.0,
                            float(latest["nll_per_event"])
                            - min(float(row["nll_per_event"]) for row in history),
                        )
                    )
            tree = tree_by_checkpoint[checkpoint_task]
            output.append({
                "variant": variant,
                "checkpoint_task": checkpoint_task,
                "current_nll_per_event": current.get("nll_per_event") if current else None,
                "current_accuracy": current.get("accuracy") if current else None,
                "current_macro_f1": current.get("macro_f1") if current else None,
                "current_local_time_mae": current.get("local_time_mae") if current else None,
                "seen_task_count": len(seen),
                "seen_mean_nll_per_event": _mean(row.get("nll_per_event") for row in seen),
                "seen_mean_accuracy": _mean(row.get("accuracy") for row in seen),
                "seen_mean_macro_f1": _mean(row.get("macro_f1") for row in seen),
                "mean_positive_task_test_nll_forgetting": _mean(forgetting),
                "anchor_count": len(anchors),
                "anchor_mean_nll_per_event": _mean(row.get("nll_per_event") for row in anchors),
                "anchor_mean_accuracy": _mean(row.get("accuracy") for row in anchors),
                "anchor_mean_macro_f1": _mean(row.get("macro_f1") for row in anchors),
                "leaf_count": tree.get("leaf_count"),
                "node_count": tree.get("node_count"),
                "memory_rows": tree.get("memory_rows"),
            })
    return output


def _law_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    checkpoint_tasks: Sequence[int],
    variants: Sequence[str],
    first_seen: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute CLNLL, forgetting, and BWT on the frozen anchor matrix."""

    anchors = [row for row in metric_rows if row["eval_kind"] == "anchor"]
    regimes = sorted({
        str(row["regime_id"])
        for row in anchors
        if row.get("regime_id") is not None
    })
    law_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        for checkpoint_task in checkpoint_tasks:
            current_rows = [
                row for row in anchors
                if row["variant"] == variant
                and int(row["checkpoint_task"]) == checkpoint_task
            ]
            seen_current = [
                row for row in current_rows
                if row.get("regime_id") is not None
                and first_seen.get(str(row["regime_id"]), math.inf) <= checkpoint_task
            ]
            for regime_id in regimes:
                seen_task = first_seen.get(regime_id)
                if seen_task is None or seen_task > checkpoint_task:
                    continue
                current = next(
                    (
                        row for row in current_rows
                        if str(row.get("regime_id")) == regime_id
                    ),
                    None,
                )
                history = [
                    row for row in anchors
                    if row["variant"] == variant
                    and str(row.get("regime_id")) == regime_id
                    and seen_task <= int(row["checkpoint_task"]) <= checkpoint_task
                    and row.get("nll_per_event") is not None
                ]
                if current is None or current.get("nll_per_event") is None or not history:
                    continue
                baseline = next(
                    (
                        row for row in history
                        if int(row["checkpoint_task"]) == seen_task
                    ),
                    min(history, key=lambda row: int(row["checkpoint_task"])),
                )
                best_nll = min(float(row["nll_per_event"]) for row in history)
                current_nll = float(current["nll_per_event"])
                baseline_nll = (
                    float(baseline["nll_per_event"])
                    if baseline.get("nll_per_event") is not None
                    else None
                )
                law_rows.append({
                    "variant": variant,
                    "checkpoint_task": checkpoint_task,
                    "regime_id": regime_id,
                    "first_seen_task": seen_task,
                    "baseline_checkpoint_task": int(baseline["checkpoint_task"]),
                    "baseline_nll_per_event": baseline_nll,
                    "current_nll_per_event": current_nll,
                    "best_nll_since_first_seen": best_nll,
                    "forgetting_nll": current_nll - best_nll,
                    "bwt_nll": (
                        baseline_nll - current_nll
                        if baseline_nll is not None else None
                    ),
                })
            summary_rows.append({
                "variant": variant,
                "checkpoint_task": checkpoint_task,
                "seen_law_count": len(seen_current),
                "clnll": _mean(
                    row.get("nll_per_event") for row in seen_current
                ),
                "average_forgetting": _mean(
                    row["forgetting_nll"] for row in law_rows
                    if row["variant"] == variant
                    and row["checkpoint_task"] == checkpoint_task
                ),
                "average_bwt": _mean(
                    row["bwt_nll"] for row in law_rows
                    if row["variant"] == variant
                    and row["checkpoint_task"] == checkpoint_task
                ),
            })
    return law_rows, summary_rows


def _stage_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    variants: Sequence[str],
) -> list[dict[str, Any]]:
    """Compute pre/post task-test adaptation gain for every available task."""

    output: list[dict[str, Any]] = []
    task_ids = sorted({
        int(row["eval_task"])
        for row in metric_rows
        if row.get("eval_task") is not None
        and row["eval_kind"] in {"task_test", "task_test_pre"}
    })
    for variant in variants:
        for task_id in task_ids:
            pre = next(
                (
                    row for row in metric_rows
                    if row["variant"] == variant
                    and row["eval_kind"] == "task_test_pre"
                    and int(row["eval_task"]) == task_id
                ),
                None,
            )
            post = next(
                (
                    row for row in metric_rows
                    if row["variant"] == variant
                    and row["eval_kind"] == "task_test"
                    and int(row["eval_task"]) == task_id
                    and int(row["checkpoint_task"]) == task_id
                ),
                None,
            )
            if pre is None or post is None:
                continue
            pre_nll = pre.get("nll_per_event") if pre else None
            post_nll = post.get("nll_per_event") if post else None
            pre_acc = pre.get("accuracy") if pre else None
            post_acc = post.get("accuracy") if post else None
            output.append({
                "task_id": task_id,
                "variant": variant,
                "stage_label": (post or pre).get("stage_label"),
                "regime_id": (post or pre).get("regime_id"),
                "pre_checkpoint_task": pre.get("checkpoint_task") if pre else None,
                "post_checkpoint_task": post.get("checkpoint_task") if post else None,
                "pre_nll_per_event": pre_nll,
                "post_nll_per_event": post_nll,
                "adaptation_gain_nll": (
                    float(pre_nll) - float(post_nll)
                    if pre_nll is not None and post_nll is not None else None
                ),
                "pre_accuracy": pre_acc,
                "post_accuracy": post_acc,
                "accuracy_delta": (
                    float(post_acc) - float(pre_acc)
                    if pre_acc is not None and post_acc is not None else None
                ),
                "pre_macro_f1": pre.get("macro_f1") if pre else None,
                "post_macro_f1": post.get("macro_f1") if post else None,
                "pre_local_time_mae": pre.get("local_time_mae") if pre else None,
                "post_local_time_mae": post.get("local_time_mae") if post else None,
            })
    return output


def _memory_decomposition(
    metric_rows: Sequence[Mapping[str, Any]],
    checkpoint_tasks: Sequence[int],
) -> list[dict[str, Any]]:
    """Compare semantic, episodic, working, and full frozen predictions."""

    variants = ("full_frozen", "no_episodic", "no_working", "semantic_only")
    lookup = {
        (
            int(row["checkpoint_task"]),
            str(row.get("regime_id")),
            row["variant"],
        ): row
        for row in metric_rows
        if row["eval_kind"] == "anchor"
    }
    output: list[dict[str, Any]] = []
    regimes = sorted({
        str(row.get("regime_id"))
        for row in metric_rows
        if row["eval_kind"] == "anchor" and row.get("regime_id") is not None
    })
    for checkpoint_task in checkpoint_tasks:
        for regime_id in regimes:
            values = {
                variant: lookup.get((checkpoint_task, regime_id, variant), {}).get(
                    "nll_per_event"
                )
                for variant in variants
            }
            if not any(value is not None for value in values.values()):
                continue
            full = values["full_frozen"]
            no_episodic = values["no_episodic"]
            no_working = values["no_working"]
            semantic = values["semantic_only"]
            output.append({
                "checkpoint_task": checkpoint_task,
                "regime_id": regime_id,
                "full_frozen_nll": full,
                "no_episodic_nll": no_episodic,
                "no_working_nll": no_working,
                "semantic_only_nll": semantic,
                "episodic_gain_nll": (
                    float(no_episodic) - float(full)
                    if no_episodic is not None and full is not None else None
                ),
                "working_gain_nll": (
                    float(no_working) - float(full)
                    if no_working is not None and full is not None else None
                ),
                "total_memory_gain_nll": (
                    float(semantic) - float(full)
                    if semantic is not None and full is not None else None
                ),
            })
    return output


def _anchor_nll_matrix(
    metric_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Make the paper-style checkpoint × regime NLL matrix."""

    anchors = [row for row in metric_rows if row["eval_kind"] == "anchor"]
    regimes = sorted({
        str(row["regime_id"])
        for row in anchors
        if row.get("regime_id") is not None
    })
    groups: dict[tuple[int, str], dict[str, Any]] = {}
    for row in anchors:
        if row.get("regime_id") is None:
            continue
        key = (int(row["checkpoint_task"]), str(row["variant"]))
        target = groups.setdefault(key, {
            "checkpoint_task": key[0],
            "variant": key[1],
        })
        target[str(row["regime_id"])] = row.get("nll_per_event")
    # Insert all columns so the CSV has a stable schema even when a particular
    # run is stopped before every anchor has been evaluated.
    for target in groups.values():
        for regime_id in regimes:
            target.setdefault(regime_id, None)
    return [groups[key] for key in sorted(groups)]


def _decomposition_summary(
    decomposition_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = (
        "episodic_gain_nll",
        "working_gain_nll",
        "total_memory_gain_nll",
    )
    groups = sorted({
        (int(row["checkpoint_task"])) for row in decomposition_rows
    })
    for checkpoint_task in groups:
        current = [
            row for row in decomposition_rows
            if int(row["checkpoint_task"]) == checkpoint_task
        ]
        output.append({
            "checkpoint_task": checkpoint_task,
            "anchor_count": len(current),
            **{
                key: _mean(row.get(key) for row in current)
                for key in keys
            },
        })
    return output


def _special_case_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    variant: str = "full_frozen",
) -> list[dict[str, Any]]:
    """Compute recurrence, near-recurrence, and long-gap diagnostic gains."""

    lookup = {
        (int(row["checkpoint_task"]), str(row.get("regime_id"))): row.get("nll_per_event")
        for row in metric_rows
        if row["eval_kind"] == "anchor" and row["variant"] == variant
    }
    task_lookup = {
        (
            int(row["checkpoint_task"]),
            int(row["eval_task"]),
            str(row["eval_kind"]),
        ): row.get("nll_per_event")
        for row in metric_rows
        if row.get("eval_task") is not None and row["variant"] == variant
    }

    definitions = (
        (
            "A1_retention_before_task3",
            "L_2,A_1 - L_0,A_1",
            (2, "A_1", 0, "A_1"),
            "near_zero_or_negative",
        ),
        (
            "A1_recovery_task3",
            "L_2,A_1 - L_3,A_1",
            (2, "A_1", 3, "A_1"),
            "positive",
        ),
        (
            "A1_long_gap_reference",
            "L_7,A_1 - L_0,A_1",
            (7, "A_1", 0, "A_1"),
            "near_zero_or_negative",
        ),
        (
            "A1_long_gap_recovery_task8",
            "L_7,A_1 - L_8,A_1",
            (7, "A_1", 8, "A_1"),
            "positive",
        ),
        (
            "B1_near_recurrence_impact",
            "L_5,B_1 - L_4,B_1",
            (5, "B_1", 4, "B_1"),
            "near_zero_or_negative",
        ),
        (
            "Bprime1_recovery_task5",
            "L_4,B_prime_1 - L_5,B_prime_1",
            (4, "B_prime_1", 5, "B_prime_1"),
            "positive",
        ),
        (
            "A2_specialization_gain_task6",
            "L_5,A_2 - L_6,A_2",
            (5, "A_2", 6, "A_2"),
            "positive",
        ),
    )
    output = []
    for name, formula, (left_task, left_regime, right_task, right_regime), expected in definitions:
        left = lookup.get((left_task, left_regime))
        right = lookup.get((right_task, right_regime))
        if left is None or right is None:
            continue
        output.append({
            "metric": name,
            "variant": variant,
            "formula": formula,
            "value": float(left) - float(right),
            "left_value": left,
            "right_value": right,
            "expected": expected,
        })
    mixture_pre = task_lookup.get((8, 9, "task_test_pre"))
    mixture_post = task_lookup.get((9, 9, "task_test"))
    if mixture_pre is not None and mixture_post is not None:
        output.append({
            "metric": "EB_mixture_adaptation_task9",
            "variant": variant,
            "formula": "P_9^pre - P_9^post",
            "value": float(mixture_pre) - float(mixture_post),
            "left_value": mixture_pre,
            "right_value": mixture_post,
            "expected": "positive",
        })
    return output


def _load_ground_truth(
    data_root: Path,
    expected_types: int,
    expected_basis: int,
) -> tuple[dict[str, GroundTruthLaw], dict[str, Any]]:
    """Load oracle Hawkes laws for diagnostics, never for model setup."""

    ground_truth_dir = data_root / "ground_truth"
    metadata_path = ground_truth_dir / "regimes.json"
    arrays_path = ground_truth_dir / "regimes.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        return {}, {
            "available": False,
            "reason": "ground_truth/regimes.json or regimes.npz is missing",
        }

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    betas = np.asarray(metadata.get("betas", ()), dtype=np.float64).reshape(-1)
    if betas.size != expected_basis or np.any(~np.isfinite(betas)) or np.any(betas <= 0):
        raise ValueError(
            f"ground-truth betas must contain {expected_basis} positive finite values; "
            f"got {betas.tolist()}"
        )
    regimes = metadata.get("regimes")
    if not isinstance(regimes, dict):
        raise ValueError(f"invalid regimes metadata in {metadata_path}")

    laws: dict[str, GroundTruthLaw] = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for raw_regime_id, raw_info in regimes.items():
            regime_id = str(raw_regime_id)
            info = raw_info if isinstance(raw_info, dict) else {}
            array_key = str(info.get("array_key", regime_id))
            mu_key = f"{array_key}__mu"
            W_key = f"{array_key}__W"
            if mu_key not in arrays or W_key not in arrays:
                raise KeyError(
                    f"ground-truth arrays for {regime_id!r} are missing "
                    f"({mu_key}, {W_key})"
                )
            mu = np.asarray(arrays[mu_key], dtype=np.float64).copy()
            W = np.asarray(arrays[W_key], dtype=np.float64).copy()
            expected_W_shape = (expected_types, expected_types, expected_basis)
            if mu.shape != (expected_types,) or W.shape != expected_W_shape:
                raise ValueError(
                    f"ground-truth shape mismatch for {regime_id!r}: "
                    f"mu={mu.shape}, W={W.shape}; expected "
                    f"{(expected_types,)}, {expected_W_shape}"
                )
            if (
                np.any(~np.isfinite(mu))
                or np.any(~np.isfinite(W))
                or np.any(mu < 0.0)
                or np.any(W < 0.0)
            ):
                raise ValueError(f"ground-truth law {regime_id!r} is not finite/non-negative")
            laws[regime_id] = GroundTruthLaw(
                regime_id=regime_id,
                mu=mu,
                W=W,
                betas=betas.copy(),
                kind=str(info.get("kind", "unknown")),
                parent_regime=(
                    str(info["parent_regime"])
                    if info.get("parent_regime") else None
                ),
            )
    return laws, {
        "available": True,
        "metadata_path": str(metadata_path.resolve()),
        "arrays_path": str(arrays_path.resolve()),
        "betas": betas.tolist(),
        "regime_count": len(laws),
    }


def _hawkes_intensity_at_time(
    event_times: np.ndarray,
    event_types: np.ndarray,
    time: float,
    mu: np.ndarray,
    W: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Evaluate the strict-causal exponential Hawkes intensity at one time."""

    intensity = np.asarray(mu, dtype=np.float64).copy()
    mask = event_times < float(time)
    if not np.any(mask):
        return np.maximum(intensity, 1e-12)
    past_times = event_times[mask]
    past_types = event_types[mask].astype(np.int64, copy=False)
    kernels = np.exp(
        -(float(time) - past_times)[:, None] * betas[None, :]
    )
    intensity += (
        W[:, past_types, :] * kernels[None, :, :]
    ).sum(axis=(1, 2))
    return np.maximum(intensity, 1e-12)


def _hawkes_intensity_curve(
    event_times: np.ndarray,
    event_types: np.ndarray,
    grid: np.ndarray,
    mu: np.ndarray,
    W: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    return np.stack([
        _hawkes_intensity_at_time(
            event_times, event_types, float(time), mu, W, betas
        )
        for time in grid
    ], axis=0)


def _decode_model_law(
    raw_theta: torch.Tensor,
    expected_types: int,
    expected_basis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the model's unconstrained semantic/effective theta to mu/W."""

    theta = raw_theta.detach().reshape(-1)
    expected_size = expected_types + expected_types * expected_types * expected_basis
    if theta.numel() != expected_size:
        raise ValueError(
            f"model theta has {theta.numel()} values; expected {expected_size}"
        )
    positive = F.softplus(theta)
    mu = positive[:expected_types].cpu().numpy().astype(np.float64, copy=True)
    W = positive[expected_types:].reshape(
        expected_types, expected_types, expected_basis
    ).cpu().numpy().astype(np.float64, copy=True)
    return mu, W


def _model_semantic_law(
    inference: MemoryTreeInference,
    events: Sequence[Mapping[str, Any]],
    expected_types: int,
    expected_basis: int,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Use the most frequently selected semantic owner for law recovery."""

    owner_counts = Counter(
        str(event.get("owner_id"))
        for event in events
        if event.get("owner_id") is not None
    )
    if owner_counts:
        owner_id, owner_count = owner_counts.most_common(1)[0]
    else:
        owner_id, owner_count = "root", 0
    if owner_id not in inference.tree.all_node_ids:
        owner_id = "root"
    raw_theta = inference.tree.semantic_theta(owner_id)
    mu, W = _decode_model_law(raw_theta, expected_types, expected_basis)
    owner_share = owner_count / max(len(events), 1)
    return mu, W, owner_id, owner_share


def _branching_matrix(W: np.ndarray, betas: np.ndarray) -> np.ndarray:
    return np.sum(
        W / np.asarray(betas, dtype=np.float64).reshape(1, 1, -1), axis=2
    )


def _relative_parameter_errors(
    model_mu: np.ndarray,
    model_W: np.ndarray,
    law: GroundTruthLaw,
    model_betas: np.ndarray,
) -> tuple[float, float, float]:
    model_K = _branching_matrix(model_W, model_betas)
    target_K = _branching_matrix(law.W, law.betas)
    epsilon = 1e-12
    mu_error = float(
        np.linalg.norm(model_mu - law.mu)
        / (np.linalg.norm(law.mu) + epsilon)
    )
    K_error = float(
        np.linalg.norm(model_K - target_K, ord="fro")
        / (np.linalg.norm(target_K, ord="fro") + epsilon)
    )
    decay_error = float(
        np.max(np.abs(model_betas - law.betas))
        if model_betas.size else 0.0
    )
    return mu_error, K_error, decay_error


def _representative_event_types(
    event_types: np.ndarray,
    expected_types: int,
    count: int = 3,
) -> list[int]:
    frequencies = Counter(int(value) for value in event_types.tolist())
    representatives = [item for item, _ in frequencies.most_common(max(count, 1))]
    return [item for item in representatives if 0 <= item < expected_types]


def _nise(
    predicted: np.ndarray,
    target: np.ndarray,
    grid: np.ndarray,
) -> tuple[float, np.ndarray]:
    integrate = getattr(np, "trapezoid", np.trapz)
    squared_error = (predicted - target) ** 2
    numerator = float(integrate(squared_error.sum(axis=1), grid))
    denominator = float(integrate((target ** 2).sum(axis=1), grid)) + 1e-12
    nise = numerator / denominator
    per_type = np.asarray([
        float(integrate(squared_error[:, index], grid))
        / (float(integrate(target[:, index] ** 2, grid)) + 1e-12)
        for index in range(target.shape[1])
    ], dtype=np.float64)
    return float(nise), per_type


def _plot_intensity_curve(
    path: Path,
    *,
    grid: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    event_times: np.ndarray,
    event_types: np.ndarray,
    regime_id: str,
    checkpoint_task: int,
    anchor_index: int,
    nise: float,
    expected_types: int,
) -> str | None:
    """Write a compact total-plus-representative-types Hawkes plot."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    representatives = _representative_event_types(
        event_types, expected_types, count=3
    )
    figure, axes = plt.subplots(
        1 + len(representatives),
        1,
        figsize=(11, 2.7 * (1 + len(representatives))),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    total_target = target.sum(axis=1)
    total_predicted = predicted.sum(axis=1)
    axes[0].plot(grid, total_target, ":", linewidth=1.8, label="ground truth")
    axes[0].plot(grid, total_predicted, "-", linewidth=1.4, label="predicted")
    axes[0].set_ylabel("total intensity")
    axes[0].set_title(
        f"checkpoint task_{checkpoint_task:02d} | {regime_id} | "
        f"anchor {anchor_index:03d} | NISE={nise:.5f}"
    )
    axes[0].legend(loc="upper right")
    for axis, event_type in zip(axes[1:], representatives):
        axis.plot(
            grid, target[:, event_type], ":", linewidth=1.8,
            label="ground truth",
        )
        axis.plot(
            grid, predicted[:, event_type], "-", linewidth=1.4,
            label="predicted",
        )
        axis.set_ylabel(f"type {event_type}")
        axis.legend(loc="upper right")
    for axis in axes:
        # Ticks are drawn in axis coordinates so they remain visible without
        # distorting the intensity y-scale.
        axis.vlines(
            event_times,
            0.0,
            1.0,
            transform=axis.get_xaxis_transform(),
            color="0.65",
            linewidth=0.35,
            alpha=0.45,
        )
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("time")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path.resolve())


def _law_inference(
    checkpoint: Path,
    variant: str,
    args: argparse.Namespace,
) -> MemoryTreeInference:
    """Build the same causal variant used by the ordinary matrix evaluator."""

    settings = VARIANTS[variant]
    inference = MemoryTreeInference.from_checkpoint(
        checkpoint,
        device=args.device,
        inference_config=InferenceConfig(
            adapt_working_memory=settings["working"],
            allow_memory_writes=settings.get("writes", settings["online"]),
            update_memory_usage=settings["online"],
            probe_write_counterfactuals=(
                variant in {"full_online", "full_online_no_write"}
            ),
            write_probe_seed=42,
            prototype_duplicate_threshold=args.prototype_duplicate_threshold,
            prototype_mode_threshold=args.prototype_mode_threshold,
            prototype_context_alias_capacity=args.prototype_context_alias_capacity,
        ),
    )
    if not settings["episodic"]:
        clear_episodic_memory(inference)
    return inference


def _hawkes_law_evaluation(
    *,
    checkpoint_paths: Mapping[int, Path],
    checkpoint_tasks: Sequence[int],
    anchors: Sequence[EvaluationSet],
    evaluation_cache: Mapping[Path, Sequence[Mapping[str, Any]]],
    ground_truth: Mapping[str, GroundTruthLaw],
    regime_first_seen: Mapping[str, int],
    args: argparse.Namespace,
    expected_types: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate causal intensity curves and semantic Hawkes law recovery."""

    if not anchors or not ground_truth:
        return [], [], [], []

    variant = args.intensity_variant
    expected_basis = len(next(iter(ground_truth.values())).betas)
    intensity_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for checkpoint_task in checkpoint_tasks:
        checkpoint = checkpoint_paths[checkpoint_task]
        inference = _law_inference(checkpoint, variant, args)
        model_betas = inference.hawkes.decays.detach().cpu().numpy().astype(
            np.float64, copy=True
        )
        if model_betas.size != expected_basis:
            raise ValueError(
                f"checkpoint task_{checkpoint_task:02d} has {model_betas.size} "
                f"decay bases, expected {expected_basis}"
            )
        for anchor in anchors:
            regime_id = str(anchor.regime_id)
            law = ground_truth.get(regime_id)
            if law is None:
                continue
            sequences = evaluation_cache.get(anchor.path, ())
            for anchor_index, sequence in enumerate(sequences):
                result = inference.run_sequence(sequence)
                events = result.get("events", ())
                snapshots = [
                    event.get("prediction_theta")
                    for event in events
                ]
                if not events or any(snapshot is None for snapshot in snapshots):
                    raise RuntimeError(
                        "inference did not expose causal prediction_theta; "
                        "please use the matching Memory/Train/Inference.py"
                    )
                event_times = sequence["times"].detach().cpu().numpy().astype(
                    np.float64, copy=False
                )
                event_types = sequence["types"].detach().cpu().numpy().astype(
                    np.int64, copy=False
                )
                horizon = max(float(event_times[-1]), 1e-6)
                grid = np.linspace(
                    0.0,
                    horizon,
                    max(int(args.intensity_samples), 2),
                    dtype=np.float64,
                )
                target_curve = _hawkes_intensity_curve(
                    event_times,
                    event_types,
                    grid,
                    law.mu,
                    law.W,
                    law.betas,
                )
                model_mu_by_event = []
                model_W_by_event = []
                for snapshot in snapshots:
                    snapshot_tensor = (
                        snapshot
                        if isinstance(snapshot, torch.Tensor)
                        else torch.as_tensor(snapshot)
                    )
                    model_mu, model_W = _decode_model_law(
                        snapshot_tensor, expected_types, expected_basis
                    )
                    model_mu_by_event.append(model_mu)
                    model_W_by_event.append(model_W)
                model_mu_by_event = np.stack(model_mu_by_event, axis=0)
                model_W_by_event = np.stack(model_W_by_event, axis=0)
                snapshot_indices = np.searchsorted(
                    event_times, grid, side="left"
                ).clip(max=len(events) - 1)
                predicted_curve = np.stack([
                    _hawkes_intensity_at_time(
                        event_times,
                        event_types,
                        float(time),
                        model_mu_by_event[index],
                        model_W_by_event[index],
                        model_betas,
                    )
                    for time, index in zip(grid, snapshot_indices)
                ], axis=0)
                nise, nise_by_type = _nise(
                    predicted_curve, target_curve, grid
                )
                plot_path = None
                if anchor_index < args.intensity_plot_anchors:
                    plot_path = _plot_intensity_curve(
                        args.output_dir
                        / "intensity_curves"
                        / f"checkpoint_task_{checkpoint_task:02d}"
                        / f"{_safe_name(regime_id)}_{anchor_index:03d}.png",
                        grid=grid,
                        target=target_curve,
                        predicted=predicted_curve,
                        event_times=event_times,
                        event_types=event_types,
                        regime_id=regime_id,
                        checkpoint_task=checkpoint_task,
                        anchor_index=anchor_index,
                        nise=nise,
                        expected_types=expected_types,
                    )
                scope = (
                    "ood_unseen"
                    if law.kind == "transient"
                    or regime_id not in regime_first_seen
                    else "seen_law"
                )
                intensity_row = {
                    "checkpoint_task": checkpoint_task,
                    "checkpoint": str(checkpoint.resolve()),
                    "variant": variant,
                    "regime_id": regime_id,
                    "anchor_index": anchor_index,
                    "events": int(len(event_times)),
                    "evaluation_scope": scope,
                    "first_seen_task": regime_first_seen.get(regime_id),
                    "nise": nise,
                    "plot_path": plot_path,
                }
                for event_type, value in enumerate(nise_by_type):
                    intensity_row[f"nise_type_{event_type}"] = float(value)
                intensity_rows.append(intensity_row)

                model_mu, model_W, owner_id, owner_share = _model_semantic_law(
                    inference, events, expected_types, expected_basis
                )
                E_mu, E_K, decay_error = _relative_parameter_errors(
                    model_mu, model_W, law, model_betas
                )
                parameter_rows.append({
                    "checkpoint_task": checkpoint_task,
                    "checkpoint": str(checkpoint.resolve()),
                    "variant": variant,
                    "regime_id": regime_id,
                    "anchor_index": anchor_index,
                    "events": int(len(event_times)),
                    "evaluation_scope": scope,
                    "first_seen_task": regime_first_seen.get(regime_id),
                    "parameter_source": "semantic_owner_mode",
                    "model_owner_id": owner_id,
                    "owner_share": owner_share,
                    "E_mu": E_mu,
                    "E_K": E_K,
                    "decay_max_abs_error": decay_error,
                })

    def grouped_summary(
        rows: Sequence[Mapping[str, Any]],
        value_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[
                (
                    int(row["checkpoint_task"]),
                    str(row["variant"]),
                    str(row["regime_id"]),
                    str(row["evaluation_scope"]),
                )
            ].append(row)
        output: list[dict[str, Any]] = []
        for key in sorted(groups):
            checkpoint_task, row_variant, regime_id, scope = key
            current = groups[key]
            item: dict[str, Any] = {
                "checkpoint_task": checkpoint_task,
                "variant": row_variant,
                "regime_id": regime_id,
                "evaluation_scope": scope,
                "first_seen_task": regime_first_seen.get(regime_id),
                "sequence_count": len(current),
            }
            for value_name in value_names:
                values = [row.get(value_name) for row in current]
                item[f"{value_name}_mean"] = _mean(values)
                clean = [
                    float(value) for value in values
                    if value is not None and math.isfinite(float(value))
                ]
                item[f"{value_name}_median"] = (
                    float(np.median(clean)) if clean else None
                )
            output.append(item)
        return output

    intensity_summary = grouped_summary(intensity_rows, ("nise",))
    parameter_summary = grouped_summary(parameter_rows, ("E_mu", "E_K"))
    return intensity_rows, intensity_summary, parameter_rows, parameter_summary


def _ood_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, GroundTruthLaw],
    regime_first_seen: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Keep transient/unseen anchors out of CL averages and report them here."""

    output = []
    for row in metric_rows:
        if row.get("eval_kind") != "anchor":
            continue
        regime_id = str(row.get("regime_id"))
        law = ground_truth.get(regime_id)
        if law is None:
            continue
        if law.kind != "transient" and regime_id in regime_first_seen:
            continue
        output.append({
            "checkpoint_task": row.get("checkpoint_task"),
            "variant": row.get("variant"),
            "regime_id": regime_id,
            "evaluation_scope": "ood_unseen",
            "nll_per_event": row.get("nll_per_event"),
            "accuracy": row.get("accuracy"),
            "macro_f1": row.get("macro_f1"),
            "memory_hit_fraction": row.get("memory_hit_fraction"),
            "read_coverage_fraction": row.get("read_coverage_fraction"),
            "mean_retrieval_alpha_mass": row.get("mean_retrieval_alpha_mass"),
            "mean_retrieval_effective_k": row.get("mean_retrieval_effective_k"),
            "mean_retrieval_similarity": row.get("mean_retrieval_similarity"),
            "mean_retrieval_null_alpha": row.get("mean_retrieval_null_alpha"),
            "mean_episodic_residual_norm": row.get("mean_episodic_residual_norm"),
            "owner_path_coverage_fraction": row.get("owner_path_coverage_fraction"),
        })
    return output


def _plot_summary_figures(
    output_dir: Path,
    *,
    continual_rows: Sequence[Mapping[str, Any]],
    anchor_matrix_rows: Sequence[Mapping[str, Any]],
    decomposition_rows: Sequence[Mapping[str, Any]],
    stage_rows: Sequence[Mapping[str, Any]],
    special_rows: Sequence[Mapping[str, Any]],
    intensity_rows: Sequence[Mapping[str, Any]],
    parameter_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Plot the compact CL figures most useful for diagnosis and a paper."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(
            f"[CL Plot] summary plots skipped: matplotlib unavailable ({error})",
            flush=True,
        )
        return []

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def finite(value: Any) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def save(figure: Any, filename: str) -> None:
        path = plot_dir / filename
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        written.append(str(path.resolve()))

    def plot_by_variant(
        axis: Any,
        rows: Sequence[Mapping[str, Any]],
        metric: str,
    ) -> bool:
        variants = sorted({str(row.get("variant")) for row in rows})
        plotted = False
        task_ticks: set[int] = set()
        for variant in variants:
            points = []
            for row in rows:
                if str(row.get("variant")) != variant:
                    continue
                task = finite(row.get("checkpoint_task"))
                value = finite(row.get(metric))
                if task is not None and value is not None:
                    points.append((int(task), value))
            points.sort()
            if not points:
                continue
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=1.8,
                label=variant,
            )
            task_ticks.update(point[0] for point in points)
            plotted = True
        axis.set_xlabel("checkpoint task")
        if task_ticks:
            axis.set_xticks(sorted(task_ticks))
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(fontsize=8)
        return plotted

    # Current-task prediction quality separates immediate fit from retention.
    quality_metrics = (
        ("current_nll_per_event", "Current-task NLL/event", "lower is better"),
        ("current_accuracy", "Current-task accuracy", "higher is better"),
        ("current_macro_f1", "Current-task macro-F1", "higher is better"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False)
    quality_plotted = False
    for axis, (metric, title, direction) in zip(axes[0], quality_metrics):
        quality_plotted |= plot_by_variant(axis, continual_rows, metric)
        axis.set_title(f"{title}\n({direction})")
    if quality_plotted:
        save(figure, "current_task_quality.png")
    else:
        plt.close(figure)

    # Stability metrics on laws seen by each checkpoint.
    continual_metrics = (
        ("clnll", "Seen-law CLNLL", "lower is better"),
        ("average_forgetting", "Average forgetting", "near zero is best"),
        ("average_bwt", "Backward transfer", "higher is better"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False)
    continual_plotted = False
    for axis, (metric, title, direction) in zip(axes[0], continual_metrics):
        continual_plotted |= plot_by_variant(axis, continual_rows, metric)
        axis.set_title(f"{title}\n({direction})")
        if metric in {"average_forgetting", "average_bwt"}:
            axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
    if continual_plotted:
        save(figure, "continual_learning.png")
    else:
        plt.close(figure)

    # Frozen-anchor checkpoint x law matrix. Prefer the paper's main variant.
    matrix_variants = sorted({str(row.get("variant")) for row in anchor_matrix_rows})
    matrix_variant = (
        "full_frozen" if "full_frozen" in matrix_variants
        else (matrix_variants[0] if matrix_variants else None)
    )
    matrix_rows = sorted(
        (
            row for row in anchor_matrix_rows
            if str(row.get("variant")) == matrix_variant
        ),
        key=lambda row: int(row["checkpoint_task"]),
    )
    metadata_columns = {"checkpoint_task", "variant"}
    regime_columns = sorted({
        key
        for row in matrix_rows
        for key in row
        if key not in metadata_columns
        and finite(row.get(key)) is not None
    })
    if matrix_rows and regime_columns:
        matrix = np.full(
            (len(matrix_rows), len(regime_columns)), np.nan, dtype=np.float64
        )
        for row_index, row in enumerate(matrix_rows):
            for column_index, regime_id in enumerate(regime_columns):
                value = finite(row.get(regime_id))
                if value is not None:
                    matrix[row_index, column_index] = value
        figure, axis = plt.subplots(
            figsize=(max(8.0, 1.05 * len(regime_columns)),
                     max(4.0, 0.58 * len(matrix_rows) + 1.8))
        )
        masked = np.ma.masked_invalid(matrix)
        colour_map = plt.get_cmap("viridis").copy()
        colour_map.set_bad("white")
        image = axis.imshow(masked, aspect="auto", cmap=colour_map)
        axis.set_xticks(range(len(regime_columns)), labels=regime_columns)
        axis.set_yticks(
            range(len(matrix_rows)),
            labels=[f"C{int(row['checkpoint_task'])}" for row in matrix_rows],
        )
        axis.set_xlabel("frozen anchor law")
        axis.set_ylabel("checkpoint")
        axis.set_title(f"Anchor NLL/event matrix — {matrix_variant} (lower is better)")
        figure.colorbar(image, ax=axis, label="NLL/event")
        if matrix.size <= 140:
            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    value = matrix[row_index, column_index]
                    if not math.isfinite(float(value)):
                        continue
                    red, green, blue, _ = colour_map(image.norm(value))
                    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if luminance > 0.55 else "white",
                    )
        save(figure, "anchor_nll_heatmap.png")

    # Plasticity of each stage before versus after learning the current task.
    stage_variants = sorted({str(row.get("variant")) for row in stage_rows})
    stage_variant = (
        "full_frozen" if "full_frozen" in stage_variants
        else (stage_variants[0] if stage_variants else None)
    )
    stage_points = sorted(
        (
            (int(row["task_id"]), value)
            for row in stage_rows
            if str(row.get("variant")) == stage_variant
            and (value := finite(row.get("adaptation_gain_nll"))) is not None
        ),
        key=lambda item: item[0],
    )
    if stage_points:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        values = [point[1] for point in stage_points]
        axis.bar(
            [point[0] for point in stage_points],
            values,
            color=["#2a9d8f" if value >= 0.0 else "#e76f51" for value in values],
        )
        axis.axhline(0.0, color="0.25", linewidth=0.9)
        axis.set_xlabel("task")
        axis.set_ylabel("pre NLL - post NLL")
        axis.set_title(f"Stage adaptation gain — {stage_variant} (positive is better)")
        axis.grid(axis="y", alpha=0.25)
        save(figure, "stage_adaptation_gain.png")

    # Contribution of episodic and working memory to anchor NLL.
    decomposition_metrics = (
        ("episodic_gain_nll", "episodic gain"),
        ("working_gain_nll", "working gain"),
        ("total_memory_gain_nll", "total memory gain"),
    )
    figure, axis = plt.subplots(figsize=(9, 4.8))
    decomposition_plotted = False
    for metric, label in decomposition_metrics:
        points = sorted(
            (
                (int(row["checkpoint_task"]), value)
                for row in decomposition_rows
                if (value := finite(row.get(metric))) is not None
            ),
            key=lambda item: item[0],
        )
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=1.8,
                label=label,
            )
            decomposition_plotted = True
    if decomposition_plotted:
        axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        axis.set_xlabel("checkpoint task")
        axis.set_ylabel("NLL gain")
        axis.set_title("Memory decomposition on frozen anchors (positive is better)")
        axis.grid(alpha=0.25)
        axis.legend()
        save(figure, "memory_decomposition.png")
    else:
        plt.close(figure)

    # Underlying Hawkes-law recovery, averaged over seen anchor laws.
    law_variants = sorted({
        str(row.get("variant")) for row in (*intensity_rows, *parameter_rows)
    })
    law_variant = (
        "full_frozen" if "full_frozen" in law_variants
        else (law_variants[0] if law_variants else None)
    )
    seen_intensity = [
        row for row in intensity_rows
        if str(row.get("variant")) == law_variant
        and row.get("evaluation_scope") == "seen_law"
    ]
    seen_parameters = [
        row for row in parameter_rows
        if str(row.get("variant")) == law_variant
        and row.get("evaluation_scope") == "seen_law"
    ]
    law_tasks = sorted({
        int(row["checkpoint_task"])
        for row in (*seen_intensity, *seen_parameters)
    })
    if law_tasks:
        law_metrics = (
            (seen_intensity, "nise_mean", "Intensity NISE"),
            (seen_parameters, "E_mu_mean", "Baseline error E_mu"),
            (seen_parameters, "E_K_mean", "Branching error E_K"),
        )
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), squeeze=False)
        law_plotted = False
        for axis, (rows, metric, title) in zip(axes[0], law_metrics):
            points = []
            for task_id in law_tasks:
                value = _mean(
                    row.get(metric)
                    for row in rows
                    if int(row["checkpoint_task"]) == task_id
                )
                value = finite(value)
                if value is not None:
                    points.append((task_id, value))
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    linewidth=1.8,
                    color="#264653",
                )
                law_plotted = True
            axis.set_xlabel("checkpoint task")
            axis.set_title(f"{title}\n(lower is better)")
            axis.grid(alpha=0.25)
        if law_plotted:
            save(figure, "hawkes_law_recovery.png")
        else:
            plt.close(figure)

    # Structural growth is separated from memory-row growth because the scales differ.
    topology_points = sorted(
        checkpoint_rows, key=lambda row: int(row["checkpoint_task"])
    )
    if topology_points:
        tasks = [int(row["checkpoint_task"]) for row in topology_points]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), squeeze=False)
        structural_plotted = False
        for metric, label in (("node_count", "nodes"), ("leaf_count", "leaves")):
            values = [finite(row.get(metric)) for row in topology_points]
            valid = [(task, value) for task, value in zip(tasks, values) if value is not None]
            if valid:
                axes[0, 0].plot(
                    [item[0] for item in valid],
                    [item[1] for item in valid],
                    marker="o",
                    linewidth=1.8,
                    label=label,
                )
                structural_plotted = True
        memory_values = [finite(row.get("memory_rows")) for row in topology_points]
        valid_memory = [
            (task, value) for task, value in zip(tasks, memory_values)
            if value is not None
        ]
        if valid_memory:
            axes[0, 1].plot(
                [item[0] for item in valid_memory],
                [item[1] for item in valid_memory],
                marker="o",
                linewidth=1.8,
                color="#e76f51",
            )
        axes[0, 0].set_title("Tree size")
        axes[0, 0].set_ylabel("count")
        axes[0, 1].set_title("Persistent episodic memory")
        axes[0, 1].set_ylabel("memory rows")
        for axis in axes[0]:
            axis.set_xlabel("checkpoint task")
            axis.grid(alpha=0.25)
        if structural_plotted:
            axes[0, 0].legend()
        if structural_plotted or valid_memory:
            save(figure, "topology_and_memory_growth.png")
        else:
            plt.close(figure)

    # Dataset-specific recurrence/specialization diagnostics become available gradually.
    special_points = [
        (str(row.get("metric")), value)
        for row in special_rows
        if (value := finite(row.get("value"))) is not None
    ]
    if special_points:
        figure, axis = plt.subplots(
            figsize=(max(9.0, 0.85 * len(special_points)), 5.0)
        )
        values = [point[1] for point in special_points]
        axis.bar(
            range(len(special_points)),
            values,
            color=["#2a9d8f" if value >= 0.0 else "#e76f51" for value in values],
        )
        axis.axhline(0.0, color="0.25", linewidth=0.9)
        axis.set_xticks(
            range(len(special_points)),
            labels=[point[0] for point in special_points],
            rotation=28,
            ha="right",
        )
        axis.set_ylabel("NLL difference / gain")
        axis.set_title("Recurrence, near-recurrence, specialization, and mixture cases")
        axis.grid(axis="y", alpha=0.25)
        save(figure, "special_case_metrics.png")

    print(f"[CL Plot] wrote {len(written)} figures to {plot_dir}", flush=True)
    return written


def _write_report(
    path: Path,
    *,
    data_root: Path,
    checkpoint_dir: Path,
    checkpoint_tasks: Sequence[int],
    variants: Sequence[str],
    metric_rows: Sequence[Mapping[str, Any]],
    continual_rows: Sequence[Mapping[str, Any]],
    law_rows: Sequence[Mapping[str, Any]],
    stage_rows: Sequence[Mapping[str, Any]],
    decomposition_summary_rows: Sequence[Mapping[str, Any]],
    special_rows: Sequence[Mapping[str, Any]],
    intensity_summary_rows: Sequence[Mapping[str, Any]],
    parameter_summary_rows: Sequence[Mapping[str, Any]],
    ood_rows: Sequence[Mapping[str, Any]],
    summary_plot_paths: Sequence[str],
    tree_by_checkpoint: Mapping[int, Mapping[str, Any]],
    skipped_tasks: Sequence[int],
    anchors_enabled: bool,
) -> None:
    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float) and not math.isfinite(value):
            return "NA"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{float(value):.{digits}f}"
        return str(value)

    lines = [
        "# Hawkes Memory Tree CL Evaluation",
        "",
        f"- Data root: `{data_root.resolve()}`",
        f"- Checkpoints: `{checkpoint_dir.resolve()}`",
        f"- Checkpoint tasks: `{list(checkpoint_tasks)}`",
        f"- Variants: `{list(variants)}`",
        "- Task test protocol: checkpoint `task_k` is evaluated on `task_0..task_k` test sets.",
        f"- Frozen anchors: `{'enabled' if anchors_enabled else 'disabled'}`.",
        "",
        "## Checkpoint topology",
        "",
        "| checkpoint | nodes | leaves | max depth | memory rows |",
        "|---:|---:|---:|---:|---:|",
    ]
    for task_id in checkpoint_tasks:
        tree = tree_by_checkpoint[task_id]
        lines.append(
            f"| task_{task_id:02d} | {tree.get('node_count')} | "
            f"{tree.get('leaf_count')} | {tree.get('max_depth')} | "
            f"{tree.get('memory_rows')} |"
        )

    lines.extend([
        "",
        "## Current-task test quality",
        "",
        "| checkpoint | variant | NLL/event | accuracy | macro-F1 | time MAE |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for row in metric_rows:
        if row["eval_kind"] != "task_test" or row["eval_task"] != row["checkpoint_task"]:
            continue
        lines.append(
            f"| task_{int(row['checkpoint_task']):02d} | {row['variant']} | "
            f"{fmt(row.get('nll_per_event'), 6)} | {fmt(row.get('accuracy'))} | "
            f"{fmt(row.get('macro_f1'))} | {fmt(row.get('local_time_mae'))} |"
        )

    lines.extend([
        "",
        "## Continual retention and anchors",
        "",
        "CLNLL averages only anchor laws whose first occurrence is no later than the checkpoint. "
        "Forgetting is current NLL minus the best NLL since that law was first seen; BWT is start NLL minus current NLL.",
        "",
        "| checkpoint | variant | CLNLL | avg forgetting | avg BWT | seen tasks | anchor mean NLL |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in continual_rows:
        lines.append(
            f"| task_{int(row['checkpoint_task']):02d} | {row['variant']} | "
            f"{fmt(row.get('clnll'), 6)} | "
            f"{fmt(row.get('average_forgetting'), 6)} | "
            f"{fmt(row.get('average_bwt'), 6)} | "
            f"{fmt(row.get('seen_task_count'), 0)} | "
            f"{fmt(row.get('anchor_mean_nll_per_event'), 6)} |"
        )

    lines.extend([
        "",
        "## Stage-level plasticity",
        "",
        "`adaptation_gain_nll = pre_nll - post_nll`; positive means the current task improved after training.",
        "",
        "| task | variant | pre NLL | post NLL | adaptation gain | pre ACC | post ACC |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in stage_rows:
        lines.append(
            f"| task_{int(row['task_id']):02d} | {row['variant']} | "
            f"{fmt(row.get('pre_nll_per_event'), 6)} | "
            f"{fmt(row.get('post_nll_per_event'), 6)} | "
            f"{fmt(row.get('adaptation_gain_nll'), 6)} | "
            f"{fmt(row.get('pre_accuracy'))} | {fmt(row.get('post_accuracy'))} |"
        )

    lines.extend([
        "",
        "## Memory decomposition",
        "",
        "Positive gains mean the corresponding memory component lowered frozen-anchor NLL.",
        "",
        "| checkpoint | anchors | episodic gain | working gain | total memory gain |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in decomposition_summary_rows:
        lines.append(
            f"| task_{int(row['checkpoint_task']):02d} | {row.get('anchor_count')} | "
            f"{fmt(row.get('episodic_gain_nll'), 6)} | "
            f"{fmt(row.get('working_gain_nll'), 6)} | "
            f"{fmt(row.get('total_memory_gain_nll'), 6)} |"
        )

    lines.extend([
        "",
        "## Special recurrence diagnostics",
        "",
        "For NLL differences, positive values mean the right-hand condition has lower NLL.",
        "",
        "| metric | formula | value | expected |",
        "|---|---|---:|---|",
    ])
    for row in special_rows:
        lines.append(
            f"| {row['metric']} | `{row['formula']}` | "
            f"{fmt(row.get('value'), 6)} | {row.get('expected')} |"
        )

    seen_intensity = [
        row for row in intensity_summary_rows
        if row.get("evaluation_scope") == "seen_law"
    ]
    seen_parameters = [
        row for row in parameter_summary_rows
        if row.get("evaluation_scope") == "seen_law"
    ]
    if seen_intensity or seen_parameters:
        parameter_lookup = {
            (
                int(row["checkpoint_task"]),
                str(row["variant"]),
                str(row["regime_id"]),
            ): row
            for row in seen_parameters
        }
        lines.extend([
            "",
            "## Hawkes law recovery",
            "",
            "NISE compares causal intensity curves against `ground_truth/regimes.npz`. "
            "E_mu and E_K compare the semantic owner law after applying softplus; "
            "E_K uses K[d,d'] = sum_m W[d,d',m] / beta_m.",
            "",
            "| checkpoint | variant | regime | NISE | E_mu | E_K | sequences |",
            "|---:|---|---|---:|---:|---:|---:|",
        ])
        for row in seen_intensity:
            parameter = parameter_lookup.get((
                int(row["checkpoint_task"]),
                str(row["variant"]),
                str(row["regime_id"]),
            ), {})
            lines.append(
                f"| task_{int(row['checkpoint_task']):02d} | {row['variant']} | "
                f"{row['regime_id']} | {fmt(row.get('nise_mean'), 6)} | "
                f"{fmt(parameter.get('E_mu_mean'), 6)} | "
                f"{fmt(parameter.get('E_K_mean'), 6)} | "
                f"{fmt(row.get('sequence_count'), 0)} |"
            )

    if ood_rows:
        lines.extend([
            "",
            "## Unseen/OOD novelty control",
            "",
            "Transient/unseen anchors are reported separately and never enter CLNLL, "
            "average forgetting, or average seen-task NLL.",
            "",
            "| checkpoint | variant | regime | NLL/event | retrieval similarity | memory hit | read coverage | residual norm |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in ood_rows:
            lines.append(
                f"| task_{int(row['checkpoint_task']):02d} | {row['variant']} | "
                f"{row['regime_id']} | {fmt(row.get('nll_per_event'), 6)} | "
                f"{fmt(row.get('mean_retrieval_similarity'))} | "
                f"{fmt(row.get('memory_hit_fraction'))} | "
                f"{fmt(row.get('read_coverage_fraction'))} | "
                f"{fmt(row.get('mean_episodic_residual_norm'))} |"
            )

    if law_rows:
        lines.extend([
            "",
            f"The full per-law anchor table contains `{len(law_rows)}` rows; "
            "see `law_metrics.csv` for start/current/best NLL, forgetting, and BWT.",
        ])

    if skipped_tasks:
        lines.extend([
            "",
            "## Skipped task IDs",
            "",
            f"No matching task test/checkpoint pair was available for: `{list(skipped_tasks)}`.",
        ])
    if summary_plot_paths:
        lines.extend(["", "## Summary plots", ""])
        for plot_value in summary_plot_paths:
            plot_path = Path(plot_value)
            try:
                relative_path = plot_path.relative_to(path.parent)
            except ValueError:
                relative_path = plot_path
            title = plot_path.stem.replace("_", " ").title()
            lines.extend([
                f"### {title}",
                "",
                f"![{title}]({relative_path.as_posix()})",
                "",
            ])
    lines.extend([
        "",
        "## Output files",
        "",
        "- `task_metrics.csv`: checkpoint × task-test × variant metrics.",
        "- `anchor_metrics.csv`: checkpoint × frozen-anchor × variant metrics.",
        "- `continual_summary.csv`: current quality, retention, forgetting, and anchor averages.",
        "- `law_metrics.csv`: per-law CLNLL support, forgetting, and BWT terms.",
        "- `stage_metrics.csv`: pre/post task-test adaptation gains.",
        "- `memory_decomposition.csv`: episodic/working/total memory NLL gains.",
        "- `memory_decomposition_summary.csv`: average decomposition gains per checkpoint.",
        "- `anchor_nll_matrix.csv`: paper-style wide checkpoint × regime NLL matrix.",
        "- `special_case_metrics.csv`: A_1 recurrence, B'_1 near-recurrence, and long-gap diagnostics.",
        "- `intensity_metrics.csv` / `intensity_summary.csv`: causal intensity-curve NISE and checkpoint summaries.",
        "- `parameter_recovery.csv` / `parameter_recovery_summary.csv`: E_mu and branching-matrix E_K.",
        "- `ood_metrics.csv`: transient/unseen-anchor novelty control, excluded from CL averages.",
        "- `intensity_curves/`: optional total-plus-representative-type GT/prediction plots.",
        "- `plots/`: summary CL curves, anchor heatmap, memory decomposition, law recovery, and topology growth.",
        "- `checkpoint_tree.csv`: leaf/node counts and checkpoint memory sizes.",
        "- `summary.json`: machine-readable copy of the complete evaluation manifest.",
        "- `event_predictions.csv`: written only when `--save-event-predictions` is supplied.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Hawkes Memory Tree CL checkpoints on task tests and frozen anchors"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", choices=("frozen", "online", "both"), default="frozen",
        help="frozen is the comparable CL protocol; online adapts memory during evaluation",
    )
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=None)
    parser.add_argument("--task-start", type=int, default=None)
    parser.add_argument("--task-end", type=int, default=None)
    parser.add_argument(
        "--current-only", action="store_true",
        help="evaluate each checkpoint only on its own task test set",
    )
    parser.add_argument(
        "--no-anchors", action="store_true",
        help="skip the independent frozen anchor banks",
    )
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true", help="reuse completed matrix cells")
    parser.add_argument(
        "--save-event-predictions", action="store_true",
        help="write the large combined event_predictions.csv artifact",
    )
    parser.add_argument(
        "--intensity-variant",
        choices=tuple(VARIANTS),
        default="full_frozen",
        help="variant used for causal intensity curves and parameter recovery",
    )
    parser.add_argument(
        "--intensity-samples",
        type=int,
        default=256,
        help="number of time-grid samples used by the NISE integral",
    )
    parser.add_argument(
        "--intensity-plot-anchors",
        type=int,
        default=2,
        help="number of anchor sequences per regime/checkpoint to plot",
    )
    parser.add_argument(
        "--no-hawkes-law-evaluation",
        action="store_true",
        help="skip ground-truth intensity NISE and parameter-law recovery",
    )
    parser.add_argument(
        "--no-summary-plots",
        action="store_true",
        help="skip automatic plots derived from the aggregate CL metrics",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print per-sequence progress from the underlying evaluator",
    )
    parser.add_argument("--prototype-duplicate-threshold", type=float, default=None)
    parser.add_argument("--prototype-mode-threshold", type=float, default=None)
    parser.add_argument("--prototype-context-alias-capacity", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_sequences is not None and args.max_sequences <= 0:
        raise ValueError("--max-sequences must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.intensity_samples < 2:
        raise ValueError("--intensity-samples must be at least 2")
    if args.intensity_plot_anchors < 0:
        raise ValueError("--intensity-plot-anchors cannot be negative")
    if (
        args.task_start is not None
        and args.task_end is not None
        and args.task_start > args.task_end
    ):
        raise ValueError("--task-start cannot be greater than --task-end")

    args.data_root = args.data_root.expanduser()
    args.checkpoint_dir = args.checkpoint_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    data_root = _normalise_data_root(args.data_root)
    task_sets = _discover_task_sets(data_root)
    checkpoint_paths = _discover_checkpoints(args.checkpoint_dir)
    if not task_sets:
        raise FileNotFoundError(f"no task_XX/test.csv files found below {data_root}")
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"no task_XX.pt checkpoints found below {args.checkpoint_dir}"
        )

    available_ids = sorted(set(task_sets).intersection(checkpoint_paths))
    selected_ids = [
        task_id for task_id in available_ids
        if (args.task_start is None or task_id >= args.task_start)
        and (args.task_end is None or task_id <= args.task_end)
    ]
    if not selected_ids:
        raise ValueError(
            "no task has both a test.csv and checkpoint after applying the task range; "
            f"data={sorted(task_sets)}, checkpoints={sorted(checkpoint_paths)}"
        )
    skipped_ids = sorted(set(task_sets).symmetric_difference(checkpoint_paths))

    stage_metadata = _read_stage_metadata(data_root)
    for task_id, evaluation_set in list(task_sets.items()):
        metadata = stage_metadata.get(task_id, {})
        task_sets[task_id] = EvaluationSet(
            name=evaluation_set.name,
            kind=evaluation_set.kind,
            path=evaluation_set.path,
            task_id=task_id,
            regime_id=metadata.get("regime_id"),
            stage_label=metadata.get("stage_label"),
        )
    regime_first_seen = _first_seen_regimes(stage_metadata)
    if not regime_first_seen:
        for task_id in sorted(task_sets):
            regime_id = task_sets[task_id].regime_id
            if regime_id:
                regime_first_seen.setdefault(str(regime_id), task_id)

    variants = _select_variants(args.protocol, args.variants)
    anchors = [] if args.no_anchors else _discover_anchors(data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[CL Eval] data={data_root} checkpoints={args.checkpoint_dir} "
        f"tasks={selected_ids} variants={variants} anchors={len(anchors)}",
        flush=True,
    )

    checkpoint_meta: dict[int, dict[str, Any]] = {}
    checkpoint_sha: dict[int, str] = {}
    expected_types: int | None = None
    expected_basis: int | None = None
    tree_by_checkpoint: dict[int, dict[str, Any]] = {}
    for task_id in selected_ids:
        checkpoint = checkpoint_paths[task_id]
        metadata = _checkpoint_meta(checkpoint)
        checkpoint_meta[task_id] = metadata
        checkpoint_sha[task_id] = _sha256(checkpoint)
        current_types = int(metadata["model_config"]["num_event_types"])
        if expected_types is None:
            expected_types = current_types
        elif current_types != expected_types:
            raise ValueError(
                f"checkpoint type mismatch at task_{task_id:02d}: "
                f"{current_types} != {expected_types}"
            )
        current_basis = int(metadata["model_config"]["num_basis"])
        if expected_basis is None:
            expected_basis = current_basis
        elif current_basis != expected_basis:
            raise ValueError(
                f"checkpoint basis mismatch at task_{task_id:02d}: "
                f"{current_basis} != {expected_basis}"
            )
        print(
            f"[CL Eval] loading topology for checkpoint task_{task_id:02d}",
            flush=True,
        )
        tree_by_checkpoint[task_id] = _tree_health(checkpoint)
    assert expected_types is not None
    assert expected_basis is not None
    ground_truth: dict[str, GroundTruthLaw] = {}
    ground_truth_meta: dict[str, Any] = {"available": False, "disabled": True}
    if not args.no_hawkes_law_evaluation and anchors:
        ground_truth, ground_truth_meta = _load_ground_truth(
            data_root, expected_types, expected_basis
        )
        if ground_truth:
            print(
                f"[CL Eval] Hawkes law layer: regimes={len(ground_truth)} "
                f"variant={args.intensity_variant} "
                f"grid={args.intensity_samples}",
                flush=True,
            )
        else:
            print(
                "[CL Eval] Hawkes law layer skipped: "
                f"{ground_truth_meta.get('reason', 'no ground truth')}",
                flush=True,
            )

    evaluation_cache: dict[Path, list[dict[str, Any]]] = {}
    data_sha_cache: dict[Path, str] = {}
    metric_rows: list[dict[str, Any]] = []
    task_matrix_rows: list[dict[str, Any]] = []
    anchor_matrix_rows: list[dict[str, Any]] = []
    event_writer = (
        _EventPredictionWriter(args.output_dir / "event_predictions.csv")
        if args.save_event_predictions
        else None
    )
    all_data_task_ids = sorted(task_sets)

    for checkpoint_task in selected_ids:
        checkpoint = checkpoint_paths[checkpoint_task]
        evaluation_sets: list[EvaluationSet] = []
        if args.current_only:
            evaluation_sets.append(task_sets[checkpoint_task])
        else:
            evaluation_sets.extend(
                task_sets[task_id]
                for task_id in all_data_task_ids
                if task_id <= checkpoint_task
            )
            next_task = checkpoint_task + 1
            if next_task in selected_ids and next_task in task_sets:
                next_set = task_sets[next_task]
                evaluation_sets.append(EvaluationSet(
                    name=f"{next_set.name}_pre",
                    kind="task_test_pre",
                    path=next_set.path,
                    task_id=next_set.task_id,
                    regime_id=next_set.regime_id,
                    stage_label=next_set.stage_label,
                ))
        evaluation_sets.extend(anchors)
        for evaluation_set in evaluation_sets:
            if evaluation_set.path not in evaluation_cache:
                evaluation_cache[evaluation_set.path] = _load_cl_dataset(
                    evaluation_set.path,
                    expected_types,
                    args.max_sequences,
                )
                data_sha_cache[evaluation_set.path] = dataset_fingerprint(
                    evaluation_set.path
                )
            sequences = evaluation_cache[evaluation_set.path]
            for variant in variants:
                print(
                    f"[CL Eval] checkpoint=task_{checkpoint_task:02d} "
                    f"dataset={evaluation_set.name} variant={variant} "
                    f"sequences={len(sequences)}",
                    flush=True,
                )
                rows, elapsed, from_cache = _load_or_run(
                    checkpoint=checkpoint,
                    checkpoint_task=checkpoint_task,
                    evaluation_set=evaluation_set,
                    variant=variant,
                    sequences=sequences,
                    data_sha256=data_sha_cache[evaluation_set.path],
                    checkpoint_sha256=checkpoint_sha[checkpoint_task],
                    args=args,
                )
                metrics = aggregate_metrics(
                    rows,
                    expected_types,
                    args.seed,
                    args.bootstrap_samples,
                )
                current_metric_row = _metric_row(
                    checkpoint_task=checkpoint_task,
                    checkpoint=checkpoint,
                    evaluation_set=evaluation_set,
                    variant=variant,
                    metrics=metrics,
                    tree=tree_by_checkpoint[checkpoint_task],
                    elapsed=elapsed,
                    from_cache=from_cache,
                    data_sha256=data_sha_cache[evaluation_set.path],
                )
                metric_rows.append(current_metric_row)
                if evaluation_set.kind in {"task_test", "task_test_pre"}:
                    task_matrix_rows.append(current_metric_row)
                else:
                    anchor_matrix_rows.append(current_metric_row)
                if event_writer is not None:
                    event_writer.write(rows, current_metric_row)

    continual_rows = _continual_summary(
        metric_rows, selected_ids, variants, tree_by_checkpoint
    )
    law_rows, law_summary_rows = _law_metrics(
        metric_rows, selected_ids, variants, regime_first_seen
    )
    law_summary_by_key = {
        (row["variant"], row["checkpoint_task"]): row
        for row in law_summary_rows
    }
    for row in continual_rows:
        row.update(law_summary_by_key.get(
            (row["variant"], row["checkpoint_task"]),
            {},
        ))
    stage_rows = _stage_metrics(metric_rows, variants)
    decomposition_rows = _memory_decomposition(metric_rows, selected_ids)
    decomposition_summary_rows = _decomposition_summary(decomposition_rows)
    anchor_matrix_rows_wide = _anchor_nll_matrix(metric_rows)
    special_rows = _special_case_metrics(metric_rows)
    checkpoint_rows = []
    for task_id in selected_ids:
        checkpoint_rows.append({
            "checkpoint_task": task_id,
            "checkpoint": str(checkpoint_paths[task_id].resolve()),
            "checkpoint_sha256": checkpoint_sha[task_id],
            **tree_by_checkpoint[task_id],
            "history_epochs": len(checkpoint_meta[task_id].get("history", [])),
        })

    write_csv(args.output_dir / "all_metrics.csv", metric_rows)
    write_csv(args.output_dir / "task_metrics.csv", task_matrix_rows)
    write_csv(args.output_dir / "anchor_metrics.csv", anchor_matrix_rows)
    write_csv(args.output_dir / "continual_summary.csv", continual_rows)
    write_csv(args.output_dir / "law_metrics.csv", law_rows)
    write_csv(args.output_dir / "stage_metrics.csv", stage_rows)
    write_csv(args.output_dir / "memory_decomposition.csv", decomposition_rows)
    write_csv(
        args.output_dir / "memory_decomposition_summary.csv",
        decomposition_summary_rows,
    )
    write_csv(args.output_dir / "anchor_nll_matrix.csv", anchor_matrix_rows_wide)
    write_csv(args.output_dir / "special_case_metrics.csv", special_rows)
    write_csv(args.output_dir / "checkpoint_tree.csv", checkpoint_rows)
    if event_writer is not None:
        event_writer.close()

    intensity_rows, intensity_summary_rows, parameter_rows, parameter_summary_rows = (
        _hawkes_law_evaluation(
            checkpoint_paths=checkpoint_paths,
            checkpoint_tasks=selected_ids,
            anchors=anchors,
            evaluation_cache=evaluation_cache,
            ground_truth=ground_truth,
            regime_first_seen=regime_first_seen,
            args=args,
            expected_types=expected_types,
        )
    )
    ood_rows = _ood_metrics(
        metric_rows, ground_truth, regime_first_seen
    )
    write_csv(args.output_dir / "intensity_metrics.csv", intensity_rows)
    write_csv(args.output_dir / "intensity_summary.csv", intensity_summary_rows)
    write_csv(args.output_dir / "parameter_recovery.csv", parameter_rows)
    write_csv(
        args.output_dir / "parameter_recovery_summary.csv",
        parameter_summary_rows,
    )
    write_csv(args.output_dir / "ood_metrics.csv", ood_rows)

    summary_plot_paths = (
        []
        if args.no_summary_plots
        else _plot_summary_figures(
            args.output_dir,
            continual_rows=continual_rows,
            anchor_matrix_rows=anchor_matrix_rows_wide,
            decomposition_rows=decomposition_summary_rows,
            stage_rows=stage_rows,
            special_rows=special_rows,
            intensity_rows=intensity_summary_rows,
            parameter_rows=parameter_summary_rows,
            checkpoint_rows=checkpoint_rows,
        )
    )

    summary = {
        "data_root": str(data_root.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "protocol": args.protocol,
        "variants": variants,
        "task_ids": selected_ids,
        "available_data_task_ids": sorted(task_sets),
        "available_checkpoint_task_ids": sorted(checkpoint_paths),
        "skipped_task_ids": skipped_ids,
        "current_only": bool(args.current_only),
        "anchors_enabled": not args.no_anchors,
        "anchor_files": [str(item.path.resolve()) for item in anchors],
        "tasks": [
            {
                "task_id": task_id,
                "test_path": str(task_sets[task_id].path.resolve()),
                "stage_label": task_sets[task_id].stage_label,
                "regime_id": task_sets[task_id].regime_id,
                "metadata": stage_metadata.get(task_id, {}),
            }
            for task_id in selected_ids
        ],
        "tree": tree_by_checkpoint,
        "regime_first_seen": regime_first_seen,
        "ground_truth": ground_truth_meta,
        "hawkes_law_evaluation": {
            "enabled": not args.no_hawkes_law_evaluation and bool(anchors),
            "variant": args.intensity_variant,
            "intensity_samples": args.intensity_samples,
            "intensity_plot_anchors": args.intensity_plot_anchors,
        },
        "metrics": metric_rows,
        "continual_summary": continual_rows,
        "law_metrics": law_rows,
        "stage_metrics": stage_rows,
        "memory_decomposition": decomposition_rows,
        "memory_decomposition_summary": decomposition_summary_rows,
        "anchor_nll_matrix": anchor_matrix_rows_wide,
        "special_case_metrics": special_rows,
        "intensity_metrics": intensity_rows,
        "intensity_summary": intensity_summary_rows,
        "parameter_recovery": parameter_rows,
        "parameter_recovery_summary": parameter_summary_rows,
        "ood_metrics": ood_rows,
        "summary_plots": summary_plot_paths,
        "note": (
            "CL evaluation uses model-facing task CSVs and independent frozen anchors. "
            "Oracle manifests are used only for task labels."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(
        args.output_dir / "report.md",
        data_root=data_root,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_tasks=selected_ids,
        variants=variants,
        metric_rows=metric_rows,
        continual_rows=continual_rows,
        law_rows=law_rows,
        stage_rows=stage_rows,
        decomposition_summary_rows=decomposition_summary_rows,
        special_rows=special_rows,
        intensity_summary_rows=intensity_summary_rows,
        parameter_summary_rows=parameter_summary_rows,
        ood_rows=ood_rows,
        summary_plot_paths=summary_plot_paths,
        tree_by_checkpoint=tree_by_checkpoint,
        skipped_tasks=skipped_ids,
        anchors_enabled=not args.no_anchors,
    )
    print(f"[Done] CL report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
