"""Persist compact epoch metrics and render post-training diagnostics."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def _nested(
    source: Optional[Mapping[str, Any]],
    *keys: str,
    default: float = 0.0,
) -> float:
    value: Any = source
    for key in keys:
        if not isinstance(value, Mapping):
            return float(default)
        value = value.get(key)
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


def _top_share(counts: Any) -> float:
    if not isinstance(counts, Mapping) or not counts:
        return 0.0
    values = [max(float(value), 0.0) for value in counts.values()]
    total = sum(values)
    return max(values, default=0.0) / total if total > 0.0 else 0.0


def compact_training_metrics(
    history: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int]]:
    """Extract stable scalar fields from checkpoint training history."""
    rows: list[dict[str, float | int]] = []
    for fallback_epoch, item in enumerate(history, start=1):
        global_update = item.get("global_update")
        sleep = item.get("sleep")
        phase = item.get("phase_seconds")
        leaf_ids = item.get("leaf_ids")
        transaction = sleep.get("transaction") if isinstance(sleep, Mapping) else None
        merge = sleep.get("merge") if isinstance(sleep, Mapping) else None
        topology_prune = (
            sleep.get("topology_prune")
            if isinstance(sleep, Mapping) else None
        )
        unified = (
            sleep.get("unified_topology")
            if isinstance(sleep, Mapping) else None
        )
        actions = (
            transaction.get("actions", [])
            if isinstance(transaction, Mapping)
            else []
        )
        rows.append({
            "epoch": int(item.get("epoch", fallback_epoch)),
            "wake_nll": float(item.get("wake_loss_per_event", 0.0)),
            "global_nll": _nested(global_update, "prediction_nll"),
            "mixture_diagnostic": _nested(
                global_update, "likelihood_mixture"
            ),
            "full_objective": float(item.get("full_objective", 0.0)),
            "conditional_entropy": _nested(
                global_update, "conditional_entropy"
            ),
            "marginal_entropy": _nested(
                global_update, "marginal_entropy"
            ),
            "mutual_information": _nested(
                global_update, "mutual_information"
            ),
            "branch_distill": _nested(global_update, "branch_distill"),
            "teacher_confidence": _nested(
                global_update, "teacher_confidence"
            ),
            "teacher_student_js": _nested(
                global_update, "teacher_student_js"
            ),
            "prior_kl": _nested(global_update, "prior_kl"),
            "posterior_kl_diagnostic": _nested(
                global_update, "posterior_kl"
            ),
            "encoder_route_gate": _nested(
                global_update, "encoder_route_gate"
            ),
            "encoder_route_grad_scale": _nested(
                global_update, "encoder_route_grad_scale"
            ),
            "owner_top1_share": _top_share(
                item.get("hard_assignment_counts")
            ),
            "memory_top1_share": _top_share(
                item.get("memory_assignment_counts")
            ),
            "max_leaf_mass": float(item.get("max_leaf_mass", 0.0)),
            "writes": int(item.get("writes", 0)),
            "write_decisions": int(item.get("write_decisions", 0)),
            "memorizes": int(item.get("memorizes", 0)),
            "queue_splits": int(item.get("queue_splits", 0)),
            "frontier_size": float(item.get("mean_frontier_size", 0.0)),
            "frontier_visited": float(
                item.get("mean_frontier_visited_nodes", 0.0)
            ),
            "frontier_branches": float(
                item.get("mean_frontier_branches", 0.0)
            ),
            "leaf_count": len(leaf_ids) if isinstance(leaf_ids, Sequence) else 0,
            "deep_pressure": _nested(sleep, "deep_pressure", "value"),
            "light_absorbed": int(
                _nested(sleep, "light", "absorbed_leaves")
            ),
            "structural_actions": len(actions),
            "unified_selected_gain": _nested(unified, "selected_gain"),
            "unified_null_probability": _nested(
                unified, "probabilities", "null", default=1.0
            ),
            "unified_candidate_count": _nested(
                unified, "candidate_count"
            ),
            "merge_mean_gain": _nested(merge, "mean_gain"),
            "merge_lambda": _nested(merge, "lambda_T"),
            "merge_candidate_count": _nested(merge, "candidate_count"),
            "merge_positive_gain_count": _nested(
                merge, "positive_gain_count"
            ),
            "topology_prune_candidate_count": _nested(
                topology_prune, "candidate_count"
            ),
            "topology_prune_ready_count": _nested(
                topology_prune, "ready_count"
            ),
            "topology_complexity": _nested(
                topology_prune, "current_complexity"
            ),
            "topology_budget": _nested(topology_prune, "budget_KT"),
            "wake_seconds": _nested(phase, "wake"),
            "global_seconds": _nested(phase, "global"),
            "sleep_seconds": _nested(phase, "sleep"),
            "cuda_peak_memory_mb": float(
                item.get("cuda_peak_memory_mb", 0.0)
            ),
        })
    return rows


def training_artifact_paths(
    checkpoint_path: str | Path,
    *,
    metrics_path: Optional[str | Path] = None,
    plot_path: Optional[str | Path] = None,
) -> tuple[Path, Path]:
    checkpoint = Path(checkpoint_path)
    stem = checkpoint.stem
    metrics = (
        Path(metrics_path)
        if metrics_path is not None
        else checkpoint.with_name(f"{stem}_training_metrics.json")
    )
    plot = (
        Path(plot_path)
        if plot_path is not None
        else checkpoint.with_name(f"{stem}_training_curves.png")
    )
    return metrics, plot


def save_metrics_log(
    rows: Sequence[Mapping[str, float | int]],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(list(rows), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path.resolve()


def _series(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0)) for row in rows]


def plot_metrics_log(
    metrics_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Render one 6x2 diagnostics figure from the saved compact JSON log."""
    cache_root = Path(tempfile.gettempdir()) / "hawkes-memory-plot-cache"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    rows = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("training metrics log contains no epochs")
    epochs = _series(rows, "epoch")
    figure, axes = plt.subplots(6, 2, figsize=(16, 26), sharex=True)
    figure.suptitle("Hawkes Memory Training Diagnostics", fontsize=18)

    panels = [
        (
            "Prediction and objective",
            [
                ("Wake NLL", "wake_nll"),
                ("Global NLL", "global_nll"),
                ("Mixture (diag)", "mixture_diagnostic"),
            ],
            "NLL / event",
        ),
        (
            "Routing specialization",
            [
                ("H conditional", "conditional_entropy"),
                ("H marginal", "marginal_entropy"),
                ("Mutual information", "mutual_information"),
            ],
            "nats",
        ),
        (
            "Teacher and Router supervision",
            [
                ("Branch distill", "branch_distill"),
                ("Teacher confidence", "teacher_confidence"),
                ("Teacher-student JS", "teacher_student_js"),
            ],
            "value",
        ),
        (
            "Routing coupling and regularization",
            [
                ("Encoder gate", "encoder_route_gate"),
                ("Encoder grad scale", "encoder_route_grad_scale"),
                ("Prior KL", "prior_kl"),
                ("Posterior KL (diag)", "posterior_kl_diagnostic"),
            ],
            "value",
        ),
        (
            "Assignment concentration",
            [
                ("Owner top-1 share", "owner_top1_share"),
                ("Memory top-1 share", "memory_top1_share"),
                ("Max leaf mass", "max_leaf_mass"),
            ],
            "fraction",
        ),
        (
            "Memory decisions",
            [
                ("Writes", "writes"),
                ("Write decisions", "write_decisions"),
                ("Memorize", "memorizes"),
                ("Queue split", "queue_splits"),
            ],
            "count / epoch",
        ),
        (
            "Frontier and structure",
            [
                ("Frontier size", "frontier_size"),
                ("Visited nodes", "frontier_visited"),
                ("Branches", "frontier_branches"),
                ("Leaves", "leaf_count"),
                ("Deep gate probability", "deep_pressure"),
            ],
            "count / value",
        ),
        (
            "Unified topology arbitration",
            [
                ("Selected conservative gain", "unified_selected_gain"),
                ("Null probability", "unified_null_probability"),
                ("Candidates", "unified_candidate_count"),
                ("Dual lambda", "merge_lambda"),
            ],
            "value",
        ),
        (
            "Merge and TopologyPrune evidence",
            [
                ("Merge mean gain", "merge_mean_gain"),
                ("Merge candidates", "merge_candidate_count"),
                ("Positive Merge", "merge_positive_gain_count"),
                ("Prune candidates", "topology_prune_candidate_count"),
                ("Prune ready", "topology_prune_ready_count"),
            ],
            "gain / count",
        ),
        (
            "Topology complexity budget",
            [
                ("Current branches", "topology_complexity"),
                ("Budget K_T", "topology_budget"),
            ],
            "branch decisions",
        ),
        (
            "Phase runtime",
            [
                ("Wake", "wake_seconds"),
                ("Global", "global_seconds"),
                ("Sleep", "sleep_seconds"),
            ],
            "seconds",
        ),
    ]

    for axis, (title, definitions, ylabel) in zip(axes.flat, panels):
        for label, key in definitions:
            axis.plot(
                epochs,
                _series(rows, key),
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=label,
            )
        axis.set_title(title, fontsize=13)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, ncol=2, frameon=False)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in axes[-1, :]:
        axis.set_xlabel("Epoch")

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)
    return path.resolve()


def save_training_diagnostics(
    history: Sequence[Mapping[str, Any]],
    checkpoint_path: str | Path,
    *,
    metrics_path: Optional[str | Path] = None,
    plot_path: Optional[str | Path] = None,
) -> dict[str, Path]:
    """Save a compact log, then generate the final multi-panel plot from it."""
    metrics, plot = training_artifact_paths(
        checkpoint_path,
        metrics_path=metrics_path,
        plot_path=plot_path,
    )
    saved_metrics = save_metrics_log(
        compact_training_metrics(history), metrics
    )
    saved_plot = plot_metrics_log(saved_metrics, plot)
    return {"metrics": saved_metrics, "plot": saved_plot}


__all__ = [
    "compact_training_metrics",
    "plot_metrics_log",
    "save_metrics_log",
    "save_training_diagnostics",
    "training_artifact_paths",
]
