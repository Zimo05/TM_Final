#!/usr/bin/env python3
"""Generate continual-learning benchmarks for a multivariate Hawkes TPP.

The generated stream follows the protocol described in the shared strategy:

* fixed event vocabulary and exponential bases (D=8, M=2, beta=(0.5, 1.5));
* structured latent regimes with recurrence, near-recurrence and drift;
* Ogata thinning for independent event sequences;
* a training stream separated from frozen anchor/evaluation banks;
* oracle regime metadata kept outside the model-facing CSV files.

Example:
    python generate_continual_hawkes.py \
        --output Data/continual_hawkes \
        --benchmark recurrence \
        --seed 7 --train-per-stage 128 --val-per-stage 32 \
        --test-per-stage 32 --seq-len 64

The CSV schema is exactly two columns: ``event_times`` and ``event_types``.
Each cell contains a JSON list, e.g. ``[0.1, 1.4, 2.0]`` and ``[3, 0, 3]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_BETAS = (0.5, 1.5)
EVENT_DIM = 8


@dataclass
class Regime:
    """Ground-truth parameters for one latent Hawkes law."""

    regime_id: str
    mu: np.ndarray  # [D]
    W: np.ndarray  # [D, D, M], target <- source <- basis
    parent_regime: str = ""
    kind: str = "base"
    parameter_distance: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)

    def copy(self, regime_id: Optional[str] = None, **updates: object) -> "Regime":
        values = {
            "regime_id": regime_id or self.regime_id,
            "mu": self.mu.copy(),
            "W": self.W.copy(),
            "parent_regime": self.parent_regime,
            "kind": self.kind,
            "parameter_distance": self.parameter_distance,
            "metadata": dict(self.metadata),
        }
        values.update(updates)
        return Regime(**values)


@dataclass(frozen=True)
class StageSpec:
    task_id: int
    label: str
    mixture: Mapping[str, float]
    shift_type: str
    recurrence_of: str = ""


def _json_dumps(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _stable_offset(value: str, modulus: int) -> int:
    """A process-independent string offset for reproducible RNG substreams."""

    return sum((index + 1) * ord(char) for index, char in enumerate(value)) % modulus


def spectral_radius(W: np.ndarray, betas: Sequence[float]) -> float:
    """Return rho(K), where K[i,j] = sum_m W[i,j,m] / beta_m."""

    K = np.sum(W / np.asarray(betas, dtype=np.float64)[None, None, :], axis=2)
    eigenvalues = np.linalg.eigvals(K)
    return float(np.max(np.abs(eigenvalues)))


def stabilize(W: np.ndarray, betas: Sequence[float], target: float) -> np.ndarray:
    """Rescale non-negative excitation weights to a prescribed spectral radius."""

    if np.any(W < 0) or not np.all(np.isfinite(W)):
        raise ValueError("Hawkes excitation weights must be finite and non-negative")
    rho = spectral_radius(W, betas)
    if rho <= 1e-12:
        return W.copy()
    return W * (float(target) / rho)


def parameter_distance(a: Regime, b: Regime) -> float:
    """Relative L2 distance over mu and W, useful for manifest diagnostics."""

    numerator = float(np.linalg.norm(a.mu - b.mu) ** 2 + np.linalg.norm(a.W - b.W) ** 2) ** 0.5
    denominator = float(np.linalg.norm(a.mu) ** 2 + np.linalg.norm(a.W) ** 2) ** 0.5
    return numerator / max(denominator, 1e-12)


def _edge(W: np.ndarray, target: int, source: int, strength: float) -> None:
    """Add a causal edge to both exponential basis components."""

    W[target, source, 0] += strength * 0.72
    W[target, source, 1] += strength * 0.28


def make_shared_backbone(rng: np.random.Generator, dim: int, n_basis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a sparse shared law used by all regimes."""

    mu = rng.uniform(0.025, 0.055, size=dim)
    W = np.zeros((dim, dim, n_basis), dtype=np.float64)
    # A weak sparse backbone prevents regimes from becoming unrelated random matrices.
    for target in range(dim):
        for source in range(dim):
            if rng.random() < 0.13:
                base = float(rng.uniform(0.004, 0.016))
                W[target, source, 0] = base * 0.65
                W[target, source, 1] = base * 0.35
    return mu, W


MOTIFS: Dict[str, Tuple[Tuple[int, int, float], ...]] = {
    # (target, source, strength) causal motifs; each regime has a visibly different structure.
    "A": ((1, 0, 0.13), (0, 1, 0.12), (3, 2, 0.105), (2, 3, 0.055)),
    "B": ((5, 4, 0.14), (4, 5, 0.115), (6, 5, 0.125), (7, 6, 0.045)),
    "C": ((4, 0, 0.115), (5, 1, 0.11), (0, 4, 0.08), (1, 5, 0.08)),
    "D": ((6, 2, 0.14), (2, 6, 0.12), (7, 3, 0.13), (3, 7, 0.1)),
    "E": ((7, 0, 0.13), (0, 7, 0.1), (6, 1, 0.12), (1, 6, 0.085), (5, 2, 0.1)),
}


def make_motif_regime(
    regime_id: str,
    motif_name: str,
    shared_mu: np.ndarray,
    shared_W: np.ndarray,
    rng: np.random.Generator,
    betas: Sequence[float],
    *,
    target_rho: Optional[float] = None,
    parent_regime: str = "",
    kind: str = "base",
) -> Regime:
    mu = np.clip(shared_mu + rng.normal(0.0, 0.0025, size=shared_mu.shape), 0.012, 0.09)
    W = shared_W.copy()
    for target, source, strength in MOTIFS[motif_name]:
        _edge(W, target, source, strength * float(rng.uniform(0.88, 1.12)))
    # Small regime-specific rate signature makes the laws distinguishable even when
    # a short sequence does not activate every motif edge.
    active_nodes = sorted({x for edge in MOTIFS[motif_name] for x in edge[:2]})
    mu[active_nodes] *= rng.uniform(1.02, 1.12)
    rho_target = target_rho if target_rho is not None else float(rng.uniform(0.66, 0.78))
    W = stabilize(W, betas, rho_target)
    regime = Regime(regime_id, mu, W, parent_regime=parent_regime, kind=kind)
    regime.metadata.update({"motif": motif_name, "spectral_radius": spectral_radius(W, betas)})
    return regime


def perturb_regime(
    base: Regime,
    regime_id: str,
    rng: np.random.Generator,
    betas: Sequence[float],
    relative_scale: float,
    *,
    kind: str,
    target_rho: Optional[float] = None,
) -> Regime:
    """Make a near recurrence/specialization while retaining the parent law."""

    mu = np.clip(base.mu * (1.0 + rng.normal(0.0, relative_scale, size=base.mu.shape)), 0.008, 0.12)
    W = np.clip(base.W * (1.0 + rng.normal(0.0, relative_scale, size=base.W.shape)), 0.0, None)
    # Retain a small amount of the original structure but make the specialization
    # observable in short sequences.
    if kind == "specialization":
        node = int(rng.integers(0, base.W.shape[0]))
        _edge(W, (node + 1) % base.W.shape[0], node, 0.035 * relative_scale / 0.12)
    target = target_rho if target_rho is not None else float(rng.uniform(0.66, 0.78))
    W = stabilize(W, betas, target)
    result = Regime(
        regime_id,
        mu,
        W,
        parent_regime=base.regime_id,
        kind=kind,
        parameter_distance=0.0,
        metadata=dict(base.metadata),
    )
    result.parameter_distance = parameter_distance(base, result)
    result.metadata["spectral_radius"] = spectral_radius(W, betas)
    result.metadata["parent"] = base.regime_id
    return result


def interpolate_regime(a: Regime, b: Regime, alpha: float, regime_id: str, betas: Sequence[float]) -> Regime:
    """Linear parameter drift between two laws."""

    alpha = float(alpha)
    mu = (1.0 - alpha) * a.mu + alpha * b.mu
    W = (1.0 - alpha) * a.W + alpha * b.W
    result = Regime(
        regime_id,
        mu,
        W,
        parent_regime=a.regime_id,
        kind="drift",
        parameter_distance=parameter_distance(a, Regime("tmp", mu, W)),
        metadata={"from": a.regime_id, "to": b.regime_id, "alpha": alpha},
    )
    result.metadata["spectral_radius"] = spectral_radius(W, betas)
    return result


def simulate_hawkes(
    regime: Regime,
    rng: np.random.Generator,
    betas: Sequence[float],
    n_events: int,
    *,
    max_time: float = 100000.0,
) -> Tuple[List[float], List[int]]:
    """Sample one sequence with Ogata thinning for exponential Hawkes kernels."""

    if n_events < 1:
        raise ValueError("n_events must be positive")
    dim, _, n_basis = regime.W.shape
    beta_arr = np.asarray(betas, dtype=np.float64)
    mu = np.asarray(regime.mu, dtype=np.float64)
    W = np.asarray(regime.W, dtype=np.float64)
    if W.shape != (dim, dim, n_basis) or beta_arr.shape != (n_basis,):
        raise ValueError("inconsistent Hawkes parameter shapes")

    # state[source, basis] stores the decayed contribution of prior events.
    state = np.zeros((dim, n_basis), dtype=np.float64)
    times: List[float] = []
    types: List[int] = []
    time = 0.0
    intensity = mu.copy()
    upper = float(np.sum(intensity))
    if upper <= 0.0:
        raise ValueError("background intensity must be positive")

    attempts = 0
    max_attempts = max(10000, n_events * 10000)
    while len(times) < n_events:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Ogata thinning exceeded safety limit; check Hawkes stability")
        wait = float(rng.exponential(1.0 / max(upper, 1e-12)))
        candidate_time = time + wait
        if candidate_time > max_time:
            raise RuntimeError("sequence exceeded max_time; increase max_time or background intensity")
        decay = np.exp(-beta_arr * wait)
        candidate_state = state * decay[None, :]
        candidate_intensity = mu + np.einsum("ijm,jm->i", W, candidate_state)
        candidate_total = float(np.sum(candidate_intensity))
        if candidate_total > 0.0 and rng.random() <= min(1.0, candidate_total / max(upper, 1e-12)):
            type_prob = candidate_intensity / candidate_total
            event_type = int(rng.choice(dim, p=type_prob))
            times.append(candidate_time)
            types.append(event_type)
            candidate_state[event_type, :] += 1.0
            state = candidate_state
            time = candidate_time
            intensity = mu + np.einsum("ijm,jm->i", W, state)
            upper = float(np.sum(intensity))
        else:
            # No accepted event: only the decayed state is retained.
            state = candidate_state
            time = candidate_time
            intensity = candidate_intensity
            upper = max(candidate_total, float(np.sum(mu)), 1e-12)
    return times, types


def _write_sequences(path: Path, sequences: Iterable[Tuple[Sequence[float], Sequence[int]]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_times", "event_types"])
        writer.writeheader()
        for times, types in sequences:
            if len(times) != len(types) or len(times) == 0:
                raise ValueError("each sequence must contain equally sized non-empty times/types")
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError("event_times must be strictly increasing")
            writer.writerow({"event_times": _json_dumps([round(float(x), 8) for x in times]), "event_types": _json_dumps([int(x) for x in types])})
            count += 1
    return count


def _sequence_count(rng: np.random.Generator, min_events: int, max_events: int) -> int:
    if min_events == max_events:
        return min_events
    return int(rng.integers(min_events, max_events + 1))


def _sample_stage(
    stage: StageSpec,
    regimes: Mapping[str, Regime],
    rng: np.random.Generator,
    betas: Sequence[float],
    count: int,
    min_events: int,
    max_events: int,
) -> Tuple[List[Tuple[List[float], List[int]]], List[Dict[str, object]]]:
    names = list(stage.mixture.keys())
    weights = np.asarray([stage.mixture[name] for name in names], dtype=np.float64)
    weights /= np.sum(weights)
    output: List[Tuple[List[float], List[int]]] = []
    rows: List[Dict[str, object]] = []
    for index in range(count):
        selected = str(rng.choice(names, p=weights))
        n_events = _sequence_count(rng, min_events, max_events)
        times, types = simulate_hawkes(regimes[selected], rng, betas, n_events)
        output.append((times, types))
        rows.append({
            "task_id": stage.task_id,
            "stage_label": stage.label,
            "split_index": index,
            "regime_id": selected,
            "regime_weights": _json_dumps(dict(stage.mixture)),
            "shift_type": stage.shift_type,
            "recurrence_of": stage.recurrence_of,
            "parameter_distance": regimes[selected].parameter_distance,
            "num_events": n_events,
        })
    return output, rows


def _build_regimes(seed: int, betas: Sequence[float]) -> Dict[str, Regime]:
    rng = np.random.default_rng(seed)
    shared_mu, shared_W = make_shared_backbone(rng, EVENT_DIM, len(betas))
    regimes: Dict[str, Regime] = {}
    for name in ("A", "B", "C", "D", "E"):
        regimes[f"{name}_1"] = make_motif_regime(f"{name}_1", name, shared_mu, shared_W, rng, betas)
    regimes["B_prime_1"] = perturb_regime(regimes["B_1"], "B_prime_1", rng, betas, 0.07, kind="near_recurrence")
    regimes["A_2"] = perturb_regime(regimes["A_1"], "A_2", rng, betas, 0.14, kind="specialization")
    # Transient X is deliberately unique and appears only in HM-Transient's one-off stage.
    regimes["X_transient"] = make_motif_regime("X_transient", "E", shared_mu, shared_W, rng, betas, target_rho=0.68, kind="transient")
    regimes["X_transient"].W = np.roll(regimes["X_transient"].W, shift=2, axis=0)
    regimes["X_transient"].W = stabilize(regimes["X_transient"].W, betas, 0.68)
    regimes["X_transient"].metadata["spectral_radius"] = spectral_radius(regimes["X_transient"].W, betas)
    regimes["X_transient"].metadata.update({"transient": True, "never_reappears": True})
    return regimes


def _recurrence_schedule() -> List[StageSpec]:
    return [
        StageSpec(0, "A_1_initial", {"A_1": 1.0}, "initial"),
        StageSpec(1, "B_1_novel", {"B_1": 1.0}, "novel"),
        StageSpec(2, "C_1_novel", {"C_1": 1.0}, "novel"),
        StageSpec(3, "A_1_exact_recurrence", {"A_1": 1.0}, "exact_recurrence", "A_1"),
        StageSpec(4, "D_1_novel", {"D_1": 1.0}, "novel"),
        StageSpec(5, "B_prime_1_near_recurrence", {"B_prime_1": 1.0}, "near_recurrence", "B_1"),
        StageSpec(6, "A_2_specialization", {"A_2": 1.0}, "specialization", "A_1"),
        StageSpec(7, "E_1_novel", {"E_1": 1.0}, "novel"),
        StageSpec(8, "A_1_long_gap_recurrence", {"A_1": 1.0}, "long_gap_recurrence", "A_1"),
        StageSpec(9, "E_B_mixture", {"E_1": 0.7, "B_1": 0.3}, "mixture"),
    ]


def _drift_regimes(regimes: MutableMapping[str, Regime], betas: Sequence[float]) -> None:
    a = regimes["A_1"]
    b = regimes["B_1"]
    for alpha in (0.2, 0.4, 0.6, 0.8):
        rid = f"A_drift_{alpha:.1f}"
        regimes[rid] = interpolate_regime(a, b, alpha, rid, betas)
    b2 = perturb_regime(b, "B_2", np.random.default_rng(9001), betas, 0.10, kind="specialization")
    regimes["B_2"] = b2


def _hierarchy_regimes(seed: int, betas: Sequence[float]) -> Dict[str, Regime]:
    rng = np.random.default_rng(seed + 17)
    shared_mu, shared_W = make_shared_backbone(rng, EVENT_DIM, len(betas))
    # Global -> group -> leaf hierarchy: leaf deltas are intentionally smaller.
    global_base = make_motif_regime("global", "A", shared_mu, shared_W, rng, betas, target_rho=0.68, kind="global")
    regimes: Dict[str, Regime] = {"Global": global_base}
    group_a = make_motif_regime("Group_A", "A", global_base.mu, global_base.W, rng, betas, target_rho=0.72, parent_regime="Global", kind="group")
    group_b = make_motif_regime("Group_B", "B", global_base.mu, global_base.W, rng, betas, target_rho=0.72, parent_regime="Global", kind="group")
    regimes.update({"Group_A": group_a, "Group_B": group_b})
    regimes["A_1"] = perturb_regime(group_a, "A_1", rng, betas, 0.035, kind="leaf", target_rho=0.73)
    regimes["A_2"] = perturb_regime(group_a, "A_2", rng, betas, 0.045, kind="leaf", target_rho=0.73)
    regimes["B_1"] = perturb_regime(group_b, "B_1", rng, betas, 0.04, kind="leaf", target_rho=0.74)
    regimes["B_2"] = perturb_regime(group_b, "B_2", rng, betas, 0.045, kind="leaf", target_rho=0.74)
    regimes["A_merge"] = interpolate_regime(regimes["A_1"], regimes["A_2"], 0.5, "A_merge", betas)
    return regimes


def build_benchmark(name: str, seed: int, betas: Sequence[float]) -> Tuple[Dict[str, Regime], List[StageSpec]]:
    if name == "hierarchy":
        regimes = _hierarchy_regimes(seed, betas)
        stages = [
            StageSpec(0, "A_1_initial", {"A_1": 1.0}, "initial"),
            StageSpec(1, "A_2_new_leaf", {"A_2": 1.0}, "new_specialization", "A_1"),
            StageSpec(2, "B_1_novel_group", {"B_1": 1.0}, "novel"),
            StageSpec(3, "A_1_recurrence", {"A_1": 1.0}, "exact_recurrence", "A_1"),
            StageSpec(4, "A_merge", {"A_merge": 1.0}, "merge", "A_1|A_2"),
            StageSpec(5, "A_2_recurrence", {"A_2": 1.0}, "exact_recurrence", "A_2"),
        ]
        return regimes, stages
    regimes = _build_regimes(seed, betas)
    if name == "recurrence":
        return regimes, _recurrence_schedule()
    if name == "drift":
        _drift_regimes(regimes, betas)
        stages = [
            StageSpec(0, "A_initial", {"A_1": 1.0}, "initial"),
            StageSpec(1, "A_drift_0.2", {"A_drift_0.2": 1.0}, "gradual_drift", "A_1"),
            StageSpec(2, "A_drift_0.4", {"A_drift_0.4": 1.0}, "gradual_drift", "A_1"),
            StageSpec(3, "B_novel", {"B_1": 1.0}, "abrupt_shift"),
            StageSpec(4, "B_drift_0.3", {"B_drift_0.3": 1.0}, "gradual_drift", "B_1"),
            StageSpec(5, "A_return", {"A_1": 1.0}, "exact_recurrence", "A_1"),
        ]
        regimes["B_drift_0.3"] = interpolate_regime(regimes["B_1"], regimes["B_2"], 0.3, "B_drift_0.3", betas)
        return regimes, stages
    if name == "transient":
        stages = [
            StageSpec(0, "A_persistent", {"A_1": 1.0}, "initial"),
            StageSpec(1, "B_persistent", {"B_1": 1.0}, "novel"),
            StageSpec(2, "C_with_transient_X", {"C_1": 0.9, "X_transient": 0.1}, "transient_anomaly"),
            StageSpec(3, "A_return", {"A_1": 1.0}, "exact_recurrence", "A_1"),
            StageSpec(4, "B_return", {"B_1": 1.0}, "exact_recurrence", "B_1"),
        ]
        return regimes, stages
    raise ValueError(f"unknown benchmark: {name}")


def _write_manifest(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_regime_artifacts(root: Path, regimes: Mapping[str, Regime], betas: Sequence[float], seed: int) -> None:
    gt = root / "ground_truth"
    gt.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, object] = {
        "seed": seed,
        "event_dim": EVENT_DIM,
        "num_basis": len(betas),
        "betas": list(map(float, betas)),
        "regimes": {},
    }
    arrays: Dict[str, np.ndarray] = {}
    for rid, regime in regimes.items():
        key = rid.replace(".", "p").replace("-", "_")
        rho = spectral_radius(regime.W, betas)
        metadata["regimes"][rid] = {
            "array_key": key,
            "parent_regime": regime.parent_regime,
            "kind": regime.kind,
            "parameter_distance": float(regime.parameter_distance),
            "spectral_radius": rho,
            "metadata": regime.metadata,
        }
        arrays[f"{key}__mu"] = regime.mu
        arrays[f"{key}__W"] = regime.W
    with (gt / "regimes.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    np.savez_compressed(gt / "regimes.npz", **arrays)
    # A torch artifact is convenient for the existing PyTorch pipeline but remains
    # optional so the generator itself only requires numpy.
    try:
        import torch  # type: ignore

        torch.save({rid: {"mu": torch.as_tensor(r.mu), "W": torch.as_tensor(r.W)} for rid, r in regimes.items()}, gt / "regimes.pt")
    except Exception:
        pass


def _write_readme(root: Path, benchmark: str, stages: Sequence[StageSpec], betas: Sequence[float]) -> None:
    lines = [
        "# Continual Hawkes benchmark",
        "",
        f"Benchmark: `{benchmark}`",
        f"Event dimension: `{EVENT_DIM}`; exponential decay bases: `{list(betas)}`",
        "",
        "Model-facing files contain only `event_times,event_types`; do not use the oracle manifest for initialization.",
        "Initialize the model on `task_00`, train each later task while retaining model/tree/memory/optimizer state, and evaluate all frozen anchors after every task.",
        "",
        "## Stage schedule",
        "",
        "| task_id | stage | regime mixture | shift | recurrence_of |",
        "|---:|---|---|---|---|",
    ]
    for stage in stages:
        mix = ", ".join(f"{k}:{v:g}" for k, v in stage.mixture.items())
        lines.append(f"| {stage.task_id} | {stage.label} | {mix} | {stage.shift_type} | {stage.recurrence_of} |")
    lines.extend([
        "",
        "`stream_manifest.csv` and `ground_truth/regimes.*` are oracle-only artifacts for evaluation and plotting.",
        "Frozen anchor banks under `anchors/` are newly sampled from the same law, not copies of stream sequences.",
    ])
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_benchmark(
    output: Path,
    benchmark: str,
    seed: int,
    betas: Sequence[float],
    train_per_stage: int,
    val_per_stage: int,
    test_per_stage: int,
    anchor_count: int,
    min_events: int,
    max_events: int,
) -> None:
    if train_per_stage < 1 or val_per_stage < 1 or test_per_stage < 1:
        raise ValueError("per-stage counts must be positive")
    if min_events < 1 or max_events < min_events:
        raise ValueError("event count range is invalid")
    regimes, stages = build_benchmark(benchmark, seed, betas)
    for regime in regimes.values():
        rho = spectral_radius(regime.W, betas)
        if not math.isfinite(rho) or rho >= 1.0:
            raise RuntimeError(f"unstable generated regime {regime.regime_id}: spectral radius={rho}")
    output.mkdir(parents=True, exist_ok=True)
    _write_regime_artifacts(output, regimes, betas, seed)

    manifest_rows: List[Dict[str, object]] = []
    split_counts = {"train": train_per_stage, "val": val_per_stage, "test": test_per_stage}
    for stage in stages:
        for split, count in split_counts.items():
            split_rng = np.random.default_rng(seed + 100003 * (stage.task_id + 1) + _stable_offset(split, 1000))
            sequences, rows = _sample_stage(stage, regimes, split_rng, betas, count, min_events, max_events)
            task_dir = output / f"task_{stage.task_id:02d}"
            _write_sequences(task_dir / f"{split}.csv", sequences)
            for row_index, row in enumerate(rows):
                row.update({"split": split, "sequence_id": f"task_{stage.task_id:02d}_{split}_{row_index:05d}", "source": "stream"})
                manifest_rows.append(row)

    # Frozen evaluation bank: each law receives independent observations and is
    # never reused in any task CSV.
    for rid, regime in regimes.items():
        if regime.kind in {"global", "group"}:
            continue
        anchor_rng = np.random.default_rng(seed + 700001 + _stable_offset(rid, 100000))
        anchor_sequences = [simulate_hawkes(regime, anchor_rng, betas, _sequence_count(anchor_rng, min_events, max_events)) for _ in range(anchor_count)]
        safe_name = rid.replace("/", "_").replace(".", "p")
        _write_sequences(output / "anchors" / f"{safe_name}.csv", anchor_sequences)
        for index, (times, types) in enumerate(anchor_sequences):
            manifest_rows.append({
                "task_id": "anchor",
                "stage_label": "frozen_anchor",
                "split_index": index,
                "regime_id": rid,
                "regime_weights": _json_dumps({rid: 1.0}),
                "shift_type": "frozen_evaluation",
                "recurrence_of": "",
                "parameter_distance": regime.parameter_distance,
                "num_events": len(times),
                "split": "anchor",
                "sequence_id": f"anchor_{safe_name}_{index:05d}",
                "source": "anchor",
            })
    _write_manifest(output / "stream_manifest.csv", manifest_rows)
    # Alias kept for evaluation scripts that expect the oracle file to be named
    # ground_truth_manifest.csv rather than stream_manifest.csv.
    _write_manifest(output / "ground_truth_manifest.csv", manifest_rows)
    _write_readme(output, benchmark, stages, betas)


def _parse_betas(value: str) -> Tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("betas must be a comma-separated list of positive numbers")
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("Data/continual_hawkes"), help="output directory")
    parser.add_argument("--benchmark", choices=("recurrence", "drift", "hierarchy", "transient", "all"), default="recurrence")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--betas", type=_parse_betas, default=DEFAULT_BETAS, help="exponential decay constants")
    parser.add_argument("--train-per-stage", type=int, default=128)
    parser.add_argument("--val-per-stage", type=int, default=32)
    parser.add_argument("--test-per-stage", type=int, default=32)
    parser.add_argument("--anchor-count", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=None, help="fixed events per sequence; overrides min/max")
    parser.add_argument("--min-events", type=int, default=48)
    parser.add_argument("--max-events", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    min_events = args.seq_len if args.seq_len is not None else args.min_events
    max_events = args.seq_len if args.seq_len is not None else args.max_events
    benchmarks = ("recurrence", "drift", "hierarchy", "transient") if args.benchmark == "all" else (args.benchmark,)
    for benchmark in benchmarks:
        output = args.output / (f"HM-{benchmark.title()}" if len(benchmarks) > 1 else "")
        generate_benchmark(
            output=output,
            benchmark=benchmark,
            seed=args.seed + benchmarks.index(benchmark),
            betas=args.betas,
            train_per_stage=args.train_per_stage,
            val_per_stage=args.val_per_stage,
            test_per_stage=args.test_per_stage,
            anchor_count=args.anchor_count,
            min_events=min_events,
            max_events=max_events,
        )
        print(f"Generated {benchmark} benchmark at {output}")


if __name__ == "__main__":
    main()
