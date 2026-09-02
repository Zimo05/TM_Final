"""Replay-based compression of sibling leaves in :class:`HawkesTree`.

The implementation follows the merge criterion from the paper, but uses the
actual model interfaces in this repository: topology lives in ``tree.nodes``,
episodic replay lives in ``tree.episodic_memory.banks``, and semantic Hawkes
parameters are generated from path-additive node embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from HawkesBackbone import (
    EVENT_TIME_FEATURES_KEY,
    HAWKES_CACHE_SIGNATURE_KEY,
    HAWKES_HISTORY_STATS_KEY,
    HAWKES_INTERVAL_STATS_KEY,
    HawkesFamily,
    inv_softplus,
)
from MemoryResiduals.MemoryBank import EventWindow, MemoryBank
from MemoryResiduals.Replay import replay_log_likelihood
from Sleep.Collapse import rebase_memory_to_new_leaf


MergeResult = Dict[str, object]


@dataclass(frozen=True)
class PromotionRecord:
    """Accepted shared-memory candidate and its two-branch evidence."""

    source_node: str
    source_index: int
    support_a: float
    support_b: float
    gain_a: float
    gain_b: float
    balance: float


def _validate_leaf_siblings(tree, node_a: str, node_b: str) -> str:
    if node_a == node_b:
        raise ValueError("node_a and node_b must be different nodes.")
    if node_a not in tree.nodes or node_b not in tree.nodes:
        missing = [node_id for node_id in (node_a, node_b) if node_id not in tree.nodes]
        raise KeyError(f"Unknown node(s): {missing}")

    parent = tree.nodes[node_a].parent
    if parent is None or tree.nodes[node_b].parent != parent:
        raise ValueError("node_a and node_b must be siblings.")
    if not tree.nodes[node_a].is_leaf or not tree.nodes[node_b].is_leaf:
        raise ValueError("Only leaf-sibling merge is implemented.")

    parent_node = tree.nodes[parent]
    if {parent_node.left, parent_node.right} != {node_a, node_b}:
        raise RuntimeError("Tree topology is inconsistent.")
    return parent


def _node_embedding(tree, node_id: str) -> Tensor:
    """Return the path-additive embedding for a leaf or internal node."""
    if node_id not in tree.nodes:
        raise KeyError(f"Unknown node: {node_id}")

    path = []
    current: Optional[str] = node_id
    while current is not None:
        path.append(current)
        current = tree.nodes[current].parent
    return torch.stack([tree.node_emb[item] for item in reversed(path)], dim=0).sum(dim=0)


def _semantic_theta(tree, node_id: str) -> Tensor:
    if hasattr(tree, "semantic_theta"):
        return tree.semantic_theta(node_id)
    params = tree.hyper(_node_embedding(tree, node_id))
    return torch.cat(
        [params.mu_tilde.reshape(-1), params.W_tilde.reshape(-1)],
        dim=0,
    )


def _normalized_child_weights(
    tree,
    node_a: str,
    node_b: str,
    *,
    reference: Tensor,
    child_weights: Optional[Tensor | Sequence[float]] = None,
) -> Tensor:
    """Return routing-mass weights shared by proposal and commit.

    Episodic-bank cardinality is a write-policy artifact, not an estimate of
    regime probability.  The maintained leaf ``mass_ema`` is the production
    routing statistic and is therefore the only automatic Merge weight.
    """
    if child_weights is None:
        child_weights = reference.new_tensor([
            max(float(tree.mass_ema.get(node_a, 0.0)), 0.0),
            max(float(tree.mass_ema.get(node_b, 0.0)), 0.0),
        ])
    else:
        child_weights = torch.as_tensor(
            child_weights,
            device=reference.device,
            dtype=reference.dtype,
        )
    child_weights = child_weights.reshape(-1)
    if child_weights.shape != (2,):
        raise ValueError("child_weights must contain exactly two values")
    if not bool(torch.isfinite(child_weights).all()) or bool(
        (child_weights < 0.0).any()
    ):
        raise ValueError("child_weights must be finite and non-negative")
    total = child_weights.sum()
    if float(total.detach().cpu()) <= 0.0:
        return child_weights.new_full((2,), 0.5)
    return child_weights / total


def fused_parent_theta(
    tree,
    node_a: str,
    node_b: str,
    *,
    child_weights: Optional[Tensor | Sequence[float]] = None,
) -> tuple[Tensor, Tensor]:
    """Fuse child Hawkes laws in physical, rather than raw, coordinates.

    Semantic tensors store unconstrained pre-softplus parameters.  Averaging
    those raw coordinates does not average the represented base intensities
    and excitation kernels.  Merge therefore links both child tensors into
    physical Hawkes space, forms the routing-mass barycenter there, and maps
    the result back with inverse-softplus for storage in the tree.
    """
    parent_id = _validate_leaf_siblings(tree, node_a, node_b)
    parent_theta = _semantic_theta(tree, parent_id)
    weights = _normalized_child_weights(
        tree,
        node_a,
        node_b,
        reference=parent_theta,
        child_weights=child_weights,
    )
    child_theta = torch.stack((
        _semantic_theta(tree, node_a),
        _semantic_theta(tree, node_b),
    ))
    physical_theta = F.softplus(child_theta)
    physical_target = (
        weights[:, None] * physical_theta
    ).sum(dim=0)
    target = inv_softplus(physical_target)
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError("physical-space Merge fusion is non-finite")
    return target, weights


def _ranked_bank_indices(
    bank: MemoryBank,
    *,
    limit: Optional[int] = None,
    exclude_index: Optional[int] = None,
) -> list[int]:
    valid = [
        index
        for index, window in enumerate(bank.windows)
        if window is not None and index != exclude_index
    ]
    if limit is None or len(valid) <= limit:
        return valid
    if limit <= 0:
        return []
    indices = torch.tensor(valid, device=bank.device, dtype=torch.long)
    score = (
        bank.write_quality.index_select(0, indices).clamp_min(0.0)
        * (1.0 + torch.log1p(
            bank.usage.index_select(0, indices).clamp_min(0.0)
        ))
        * torch.exp(
            -0.2
            * bank.stale_cycles.index_select(0, indices).clamp_min(0.0)
        )
    )
    selected = torch.topk(score, k=int(limit)).indices
    return indices.index_select(0, selected).detach().cpu().tolist()


def _replay_windows(
    tree,
    node_ids: Iterable[str],
    *,
    max_windows: Optional[int] = None,
) -> list[EventWindow]:
    windows: list[EventWindow] = []
    score_parts: list[Tensor] = []
    for node_id in node_ids:
        bank = tree.episodic_memory.banks.get(node_id)
        if bank is not None:
            valid = _ranked_bank_indices(bank)
            if not valid:
                continue
            indices = torch.tensor(
                valid, device=bank.device, dtype=torch.long
            )
            score_parts.append(
                bank.write_quality.index_select(0, indices).clamp_min(0.0)
                * (
                    1.0
                    + torch.log1p(
                        bank.usage.index_select(0, indices).clamp_min(0.0)
                    )
                )
                * torch.exp(
                    -0.2
                    * bank.stale_cycles.index_select(
                        0, indices
                    ).clamp_min(0.0)
                )
            )
            windows.extend(bank.windows[index] for index in valid)
    if not windows:
        return []
    if max_windows is None or len(windows) <= max_windows:
        return windows
    if max_windows <= 0:
        return []
    scores = torch.cat(score_parts, dim=0)
    selected = torch.topk(scores, k=int(max_windows)).indices
    return [
        windows[index]
        for index in selected.detach().cpu().tolist()
    ]


def _resolve_hawkes_family(
    tree,
    decays: Tensor,
    hawkes_ll: Optional[HawkesFamily],
) -> HawkesFamily:
    if hawkes_ll is not None:
        if hawkes_ll.num_types != tree.hyper.D or hawkes_ll.num_basis != tree.hyper.M:
            raise ValueError("hawkes_ll dimensions do not match the Hawkes tree.")
        return hawkes_ll

    # HawkesFamily's likelihood methods operate on the supplied parameters;
    # its own trainable raw parameters are not used here.
    return HawkesFamily(
        num_types=tree.hyper.D,
        num_basis=tree.hyper.M,
        decays=decays.detach().clone(),
    )


def _window_log_likelihood(
    window: EventWindow,
    theta: Tensor,
    decays: Tensor,
    hawkes_ll: HawkesFamily,
    normalize_by_events: bool,
) -> Tensor:
    return replay_log_likelihood(
        window=window,
        theta=theta,
        hawkes_ll=hawkes_ll,
        decays=decays,
        normalize_by_events=normalize_by_events,
    )


def branch_support_and_gain(
    candidate_key: Tensor,
    candidate_delta: Tensor,
    branch_bank: MemoryBank,
    branch_theta: Tensor,
    decays: Tensor,
    num_types: int,
    num_basis: int,
    sim_threshold: float = 0.7,
    temperature: float = 0.1,
    hawkes_ll: Optional[HawkesFamily] = None,
    exclude_index: Optional[int] = None,
    max_references: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Measure a residual candidate's soft coverage and utility on one branch."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if branch_theta.ndim != 1:
        raise ValueError("branch_theta must be one-dimensional")
    expected_dim = num_types + num_types * num_types * num_basis
    if branch_theta.numel() != expected_dim:
        raise ValueError(f"branch_theta must contain {expected_dim} values")
    if candidate_delta.numel() != expected_dim:
        raise ValueError(f"candidate_delta must contain {expected_dim} values")
    if candidate_key.numel() != branch_bank.key_dim:
        raise ValueError("candidate key dimension does not match branch bank")

    device = branch_theta.device
    dtype = branch_theta.dtype
    valid_indices = _ranked_bank_indices(
        branch_bank,
        limit=max_references,
        exclude_index=exclude_index,
    )
    if not valid_indices:
        return branch_theta.new_zeros(()), branch_theta.new_full((), -torch.inf)

    if hawkes_ll is None:
        hawkes_ll = HawkesFamily(
            num_types=num_types,
            num_basis=num_basis,
            decays=torch.as_tensor(decays).detach().clone(),
        )
    elif hawkes_ll.num_types != num_types or hawkes_ll.num_basis != num_basis:
        raise ValueError("hawkes_ll dimensions do not match promotion parameters")

    candidate_key = torch.nn.functional.normalize(
        candidate_key.to(device=device, dtype=dtype).reshape(-1), dim=0
    )
    corrected_theta = branch_theta + candidate_delta.to(
        device=device, dtype=dtype
    ).reshape(-1)

    index_tensor = torch.tensor(
        valid_indices,
        device=branch_bank.device,
        dtype=torch.long,
    )
    replay_keys = torch.nn.functional.normalize(
        branch_bank.keys.index_select(0, index_tensor).to(
            device=device,
            dtype=dtype,
        ),
        dim=-1,
    )
    similarities = replay_keys @ candidate_key
    weights_tensor = torch.sigmoid(
        (similarities - sim_threshold) / temperature
    )
    windows = [branch_bank.windows[index] for index in valid_indices]
    if any(window is None for window in windows):
        raise RuntimeError("ranked promotion references contain an empty window")
    # Reuse the event-flattened Merge likelihood kernel. The third model is a
    # harmless duplicate needed by that kernel's [left, right, parent] shape.
    theta_models = torch.stack((
        branch_theta,
        corrected_theta,
        branch_theta,
    )).unsqueeze(0)
    replay_nll = _batched_replay_nll(
        windows,
        [0] * len(windows),
        theta_models,
        hawkes_ll,
        normalize_by_events=True,
    )
    gains_tensor = replay_nll[:, 0] - replay_nll[:, 1]
    support = weights_tensor.sum()
    mean_gain = (weights_tensor * gains_tensor).sum() / (support + 1e-8)
    return support, mean_gain


@torch.no_grad()
def find_shared_memory_promotions(
    tree,
    parent_id: str,
    decays: Tensor,
    min_support: float = 2.0,
    min_gain: float = 0.0,
    min_balance: float = 0.5,
    sim_threshold: float = 0.7,
    temperature: float = 0.1,
    hawkes_ll: Optional[HawkesFamily] = None,
    max_candidates: Optional[int] = None,
    max_references: Optional[int] = None,
) -> list[PromotionRecord]:
    """Evaluate promotion candidates without mutating tree or memory state."""
    if parent_id not in tree.nodes:
        raise KeyError(f"Unknown parent node: {parent_id}")
    parent_node = tree.nodes[parent_id]
    if parent_node.is_leaf:
        return []
    if parent_node.left is None or parent_node.right is None:
        raise RuntimeError("Internal parent must have two children")
    node_a, node_b = parent_node.left, parent_node.right
    _validate_leaf_siblings(tree, node_a, node_b)
    if min_support < 0.0:
        raise ValueError("min_support must be non-negative")
    if not 0.0 <= min_balance <= 1.0:
        raise ValueError("min_balance must be in [0, 1]")

    theta_a = _semantic_theta(tree, node_a)
    theta_b = _semantic_theta(tree, node_b)
    bank_a = tree.episodic_memory.get_bank(node_a)
    bank_b = tree.episodic_memory.get_bank(node_b)
    hawkes_ll = _resolve_hawkes_family(tree, decays, hawkes_ll)
    records: list[PromotionRecord] = []

    for source_node, source_bank in ((node_a, bank_a), (node_b, bank_b)):
        candidate_indices = _ranked_bank_indices(
            source_bank,
            limit=max_candidates,
        )
        for source_index in candidate_indices:
            candidate_key = source_bank.keys[source_index]
            candidate_delta = source_bank.deltas[source_index]
            support_a, gain_a = branch_support_and_gain(
                candidate_key, candidate_delta, bank_a, theta_a, decays,
                tree.hyper.D, tree.hyper.M, sim_threshold, temperature, hawkes_ll,
                source_index if source_node == node_a else None,
                max_references,
            )
            support_b, gain_b = branch_support_and_gain(
                candidate_key, candidate_delta, bank_b, theta_b, decays,
                tree.hyper.D, tree.hyper.M, sim_threshold, temperature, hawkes_ll,
                source_index if source_node == node_b else None,
                max_references,
            )
            balance = 2.0 * torch.minimum(support_a, support_b) / (
                support_a + support_b + 1e-8
            )
            hard_state = torch.stack((
                support_a,
                support_b,
                gain_a,
                gain_b,
                balance,
            )).detach().cpu().tolist()
            support_a_value, support_b_value, gain_a_value, gain_b_value, balance_value = (
                hard_state
            )
            if not (
                support_a_value >= min_support
                and support_b_value >= min_support
                and gain_a_value >= min_gain
                and gain_b_value >= min_gain
                and balance_value >= min_balance
            ):
                continue
            records.append(
                PromotionRecord(
                    source_node=source_node,
                    source_index=source_index,
                    support_a=float(support_a_value),
                    support_b=float(support_b_value),
                    gain_a=float(gain_a_value),
                    gain_b=float(gain_b_value),
                    balance=float(balance_value),
                )
            )
    return records


def _append_bank_rows(
    target: MemoryBank,
    source: MemoryBank,
    indices: Tensor,
    target_node_id: str,
) -> None:
    target.append_from(source, indices, node_id=target_node_id)


@torch.no_grad()
def promote_shared_memories(
    tree,
    parent_id: str,
    decays: Tensor,
    min_support: float = 2.0,
    min_gain: float = 0.0,
    min_balance: float = 0.5,
    sim_threshold: float = 0.7,
    temperature: float = 0.1,
    hawkes_ll: Optional[HawkesFamily] = None,
    max_candidates: Optional[int] = None,
    max_references: Optional[int] = None,
    records: Optional[Sequence[PromotionRecord]] = None,
) -> list[PromotionRecord]:
    """Move accepted child residuals to their shared parent memory bank."""
    if records is None:
        records = find_shared_memory_promotions(
            tree=tree,
            parent_id=parent_id,
            decays=decays,
            min_support=min_support,
            min_gain=min_gain,
            min_balance=min_balance,
            sim_threshold=sim_threshold,
            temperature=temperature,
            hawkes_ll=hawkes_ll,
            max_candidates=max_candidates,
            max_references=max_references,
        )
    else:
        records = list(records)
    if not records:
        return []

    parent_bank = tree.episodic_memory.get_bank(parent_id)
    by_source: Dict[str, list[int]] = {}
    for record in records:
        by_source.setdefault(record.source_node, []).append(record.source_index)

    for source_node, source_indices in by_source.items():
        source_bank = tree.episodic_memory.get_bank(source_node)
        promoted_idx = torch.tensor(source_indices, device=source_bank.device)
        size_before = len(source_bank)
        _append_bank_rows(
            target=parent_bank,
            source=source_bank,
            indices=promoted_idx,
            target_node_id=parent_id,
        )
        # Promotion keeps the raw residual unchanged. Since parent is on both
        # child paths, this preserves the source correction and shares it.
        keep_mask = torch.ones(size_before, dtype=torch.bool, device=source_bank.device)
        keep_mask[promoted_idx] = False
        source_bank.keep(torch.nonzero(keep_mask, as_tuple=False).flatten())

    if len(parent_bank) > parent_bank.capacity:
        parent_bank.prune()
    return records


def leaf_sibling_pairs(
    tree,
    leaf_snapshot: Optional[Iterable[str]] = None,
) -> list[tuple[str, str]]:
    """Build non-overlapping leaf-sibling candidates from one topology snapshot."""
    snapshot = list(tree.leaf_ids if leaf_snapshot is None else leaf_snapshot)
    leaves = set(snapshot)
    pairs: list[tuple[str, str]] = []
    seen_parents: set[str] = set()
    for leaf_id in snapshot:
        if leaf_id not in tree.nodes:
            continue
        parent_id = tree.nodes[leaf_id].parent
        if parent_id is None or parent_id in seen_parents:
            continue
        parent = tree.nodes[parent_id]
        if parent.left in leaves and parent.right in leaves:
            pairs.append((parent.left, parent.right))
            seen_parents.add(parent_id)
    return pairs


def _window_event_bounds(window: EventWindow) -> tuple[int, int]:
    if window.has_full_history:
        start = int(window.start_idx)
        end = int(window.end_idx)
        if start < 0 or end < start or end > window.times.numel():
            raise ValueError("full-history EventWindow has an invalid event range")
        return start, end
    return 0, int(window.times.numel())


def _window_target_tensors(
    window: EventWindow,
    hawkes_ll: HawkesFamily,
    reference: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return cached target-event tensors without evaluating a model."""
    times = window.times.to(device=reference.device, dtype=reference.dtype)
    types = window.types.to(device=reference.device).long()
    if times.ndim != 1 or types.ndim != 1 or times.numel() != types.numel():
        raise ValueError("window times/types must be aligned one-dimensional tensors")
    start, end = _window_event_bounds(window)
    if not window.has_full_history and times.numel() > 0:
        times = times - times[0]

    sequence: Dict[str, Any] = {"times": times, "types": types}
    if window.has_full_history:
        cached_fields = {
            EVENT_TIME_FEATURES_KEY: getattr(window, "event_time_features", None),
            HAWKES_HISTORY_STATS_KEY: getattr(window, "hawkes_history_stats", None),
            HAWKES_INTERVAL_STATS_KEY: getattr(window, "hawkes_interval_stats", None),
            HAWKES_CACHE_SIGNATURE_KEY: getattr(
                window, "hawkes_cache_signature", None
            ),
        }
        sequence.update({
            key: value for key, value in cached_fields.items()
            if value is not None
        })
    sequence = hawkes_ll.prepare_sequence_cache(sequence, inplace=True)
    if window.has_full_history:
        window.event_time_features = sequence[EVENT_TIME_FEATURES_KEY]
        window.hawkes_history_stats = sequence[HAWKES_HISTORY_STATS_KEY]
        window.hawkes_interval_stats = sequence[HAWKES_INTERVAL_STATS_KEY]
        window.hawkes_cache_signature = sequence[HAWKES_CACHE_SIGNATURE_KEY]

    history = sequence[HAWKES_HISTORY_STATS_KEY][start:end].to(reference)
    interval = sequence[HAWKES_INTERVAL_STATS_KEY][start:end].to(reference)
    previous = torch.cat([times.new_zeros(1), times[:-1]])
    duration = (times - previous).clamp_min(0.0)[start:end]
    return history, interval, duration, types[start:end]


def _batched_replay_nll(
    windows: Sequence[EventWindow],
    window_pair_indices: Sequence[int],
    theta_models: Tensor,
    hawkes_ll: HawkesFamily,
    *,
    normalize_by_events: bool,
) -> Tensor:
    """Evaluate every ``(replay, child-a/child-b/parent)`` in one tensor graph.

    Parameter-independent replay caches are assembled in Python, but all
    model-dependent Hawkes algebra is flattened across candidates and target
    events. On CUDA this avoids one kernel-launch chain per replay window.
    """
    if theta_models.ndim != 3 or theta_models.shape[1] != 3:
        raise ValueError("theta_models must have shape [pairs, 3, parameters]")
    if len(windows) != len(window_pair_indices):
        raise ValueError("window_pair_indices must align with windows")

    histories: list[Tensor] = []
    intervals: list[Tensor] = []
    durations: list[Tensor] = []
    target_types: list[Tensor] = []
    event_pair_indices: list[Tensor] = []
    event_window_indices: list[Tensor] = []
    event_counts: list[int] = []
    for window_index, (window, pair_index) in enumerate(zip(
        windows, window_pair_indices
    )):
        history, interval, duration, types = _window_target_tensors(
            window, hawkes_ll, theta_models
        )
        count = int(types.numel())
        if count <= 0:
            raise ValueError("differentiable merge replay windows must contain events")
        histories.append(history)
        intervals.append(interval)
        durations.append(duration)
        target_types.append(types)
        event_pair_indices.append(torch.full(
            (count,), pair_index, device=theta_models.device, dtype=torch.long
        ))
        event_window_indices.append(torch.full(
            (count,), window_index, device=theta_models.device, dtype=torch.long
        ))
        event_counts.append(count)

    history = torch.cat(histories, dim=0)
    interval = torch.cat(intervals, dim=0)
    duration = torch.cat(durations, dim=0)
    types = torch.cat(target_types, dim=0)
    event_pair = torch.cat(event_pair_indices, dim=0)
    event_window = torch.cat(event_window_indices, dim=0)

    selected_theta = theta_models.index_select(0, event_pair)
    D = hawkes_ll.num_types
    M = hawkes_ll.num_basis
    raw_mu = selected_theta[..., :D]
    raw_W = selected_theta[..., D:].reshape(-1, 3, D, D, M)
    mu = F.softplus(raw_mu)
    W = F.softplus(raw_W)
    intensity = (
        mu
        + torch.einsum("eqdsm,esm->eqd", W, history)
    ).clamp_min(1e-8)
    gathered = intensity.gather(
        2,
        types[:, None, None].expand(-1, 3, 1),
    ).squeeze(-1)
    log_term = -gathered.log()
    integral = (
        mu.sum(dim=-1) * duration.unsqueeze(-1)
        + torch.einsum("eqdsm,esm->eq", W, interval)
    )
    event_nll = log_term + integral
    window_nll = theta_models.new_zeros((len(windows), 3))
    window_nll.index_add_(0, event_window, event_nll)
    if normalize_by_events:
        counts = theta_models.new_tensor(event_counts).clamp_min(1.0)
        window_nll = window_nll / counts.unsqueeze(-1)
    return window_nll


def _compute_child_mixture_merge_objective_legacy(
    tree,
    pairs: Sequence[tuple[str, str]],
    decays: Tensor,
    *,
    lambda_T: float | Tensor,
    budget_KT: Optional[float | Tensor] = None,
    gate_temperature: float = 1.0,
    min_replay: int = 8,
    normalize_by_events: bool = True,
    hawkes_ll: Optional[HawkesFamily] = None,
    max_replay: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the replay-mixture Merge objective for all sibling pairs.

    Child likelihood is a memory-mass mixture, never an oracle max. The
    returned loss remains differentiable through semantic Hawkes parameters,
    node embeddings, and HyperNet. No topology mutation occurs here.
    """
    if gate_temperature <= 0.0:
        raise ValueError("gate_temperature must be positive")
    if min_replay < 0:
        raise ValueError("min_replay must be non-negative")
    if max_replay is not None and max_replay <= 0:
        raise ValueError("max_replay must be positive when provided")
    hawkes_ll = _resolve_hawkes_family(tree, decays, hawkes_ll)

    eligible_pairs: list[tuple[str, str]] = []
    parents: list[str] = []
    replay_windows: list[EventWindow] = []
    replay_pair_indices: list[int] = []
    replay_counts: list[int] = []
    child_masses: list[tuple[int, int]] = []
    branch_support: list[tuple[int, int]] = []
    skipped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for pair in pairs:
        node_a, node_b = pair
        parent = _validate_leaf_siblings(tree, node_a, node_b)
        windows = [
            window for window in _replay_windows(
                tree, pair, max_windows=max_replay
            )
            if _window_event_bounds(window)[1]
            > _window_event_bounds(window)[0]
        ]
        if not windows or len(windows) < min_replay:
            skipped[pair] = {
                "merge": False,
                "reason": "insufficient_replay",
                "replay_size": len(windows),
            }
            continue
        pair_index = len(eligible_pairs)
        eligible_pairs.append(pair)
        parents.append(parent)
        replay_windows.extend(windows)
        replay_pair_indices.extend([pair_index] * len(windows))
        replay_counts.append(len(windows))
        branch_support.append((
            sum(window.node_id == node_a for window in windows),
            sum(window.node_id == node_b for window in windows),
        ))
        child_masses.append((
            len(tree.episodic_memory.get_bank(node_a)),
            len(tree.episodic_memory.get_bank(node_b)),
        ))

    root_id = next(
        node_id for node_id, node in tree.nodes.items()
        if node.parent is None
    )
    reference = _semantic_theta(tree, root_id)
    if not eligible_pairs:
        zero = reference.sum() * 0.0
        return {
            "loss": zero,
            "likelihood_loss": zero,
            "dual_term": zero,
            "expected_complexity": zero,
            "full_keep_complexity": zero,
            "pairs": [],
            "parents": [],
            "replay_counts": [],
            "effective_sample_size": zero.new_empty((0,)),
            "branch_support": zero.new_empty((0, 2)),
            "skipped": skipped,
            "delta_keep": zero.new_empty((0,)),
            "delta_keep_variance": zero.new_empty((0,)),
            "complexity_saving": zero.new_empty((0,)),
            "pair_loss": zero.new_empty((0,)),
            "keep_probability": zero.new_empty((0,)),
            "child_mix_nll": zero.new_empty((0,)),
            "parent_nll": zero.new_empty((0,)),
            "merge_parent_theta": zero.new_empty((0, reference.numel())),
            "child_weights": zero.new_empty((0, 2)),
            "embedding_distance": zero.new_empty((0,)),
            "semantic_distance": zero.new_empty((0,)),
        }

    leaf_theta = {
        leaf_id: _semantic_theta(tree, leaf_id)
        for leaf_id in tree.leaf_ids
    }
    child_mass_tensor = reference.new_tensor(child_masses)
    child_weights = child_mass_tensor / child_mass_tensor.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    zero_mass = child_mass_tensor.sum(dim=-1).eq(0.0)
    if bool(zero_mass.any()):
        child_weights[zero_mass] = 0.5
    merge_parent_theta = torch.stack([
        _semantic_theta(tree, parent)
        + (
            child_weights[index, :, None]
            * (
                torch.stack((leaf_theta[node_a], leaf_theta[node_b]))
                - _semantic_theta(tree, parent)
            )
        ).sum(dim=0)
        for index, ((node_a, node_b), parent) in enumerate(zip(
            eligible_pairs, parents
        ))
    ])
    theta_models = torch.stack([
        torch.stack((
            leaf_theta[node_a],
            leaf_theta[node_b],
            merge_parent_theta[index],
        ))
        for index, (node_a, node_b) in enumerate(eligible_pairs)
    ])
    replay_pair = torch.tensor(
        replay_pair_indices, device=theta_models.device, dtype=torch.long
    )
    replay_nll = _batched_replay_nll(
        replay_windows,
        replay_pair_indices,
        theta_models,
        hawkes_ll,
        normalize_by_events=normalize_by_events,
    )

    branch_support_tensor = theta_models.new_tensor(branch_support)
    alpha = child_weights.to(theta_models)
    log_alpha = torch.where(
        alpha > 0.0,
        alpha.clamp_min(torch.finfo(alpha.dtype).tiny).log(),
        torch.full_like(alpha, -torch.inf),
    )
    replay_log_alpha = log_alpha.index_select(0, replay_pair)
    child_mix_nll_each = -torch.logsumexp(
        replay_log_alpha - replay_nll[:, :2], dim=-1
    )
    parent_nll_each = replay_nll[:, 2]
    delta_each = parent_nll_each - child_mix_nll_each

    pair_count = len(eligible_pairs)
    count_tensor = theta_models.new_tensor(replay_counts).clamp_min(1.0)
    # Replay windows enter the objective with equal weight. Therefore the
    # weighted ESS, (sum w)^2 / sum w^2, is exactly the number of windows.
    effective_sample_size = count_tensor
    delta_keep = theta_models.new_zeros(pair_count)
    delta_keep.index_add_(0, replay_pair, delta_each)
    delta_keep = delta_keep / count_tensor
    delta_second_moment = theta_models.new_zeros(pair_count)
    delta_second_moment.index_add_(0, replay_pair, delta_each.square())
    delta_second_moment = delta_second_moment / count_tensor
    delta_keep_variance = (
        delta_second_moment - delta_keep.square()
    ).clamp_min(0.0)
    child_mix_nll = theta_models.new_zeros(pair_count)
    child_mix_nll.index_add_(0, replay_pair, child_mix_nll_each)
    child_mix_nll = child_mix_nll / count_tensor
    parent_nll = theta_models.new_zeros(pair_count)
    parent_nll.index_add_(0, replay_pair, parent_nll_each)
    parent_nll = parent_nll / count_tensor

    # Every complete binary refinement is one common topology-complexity
    # unit.  Parameter norms are not mixed into the action currency.
    complexity_saving = theta_models.new_ones(pair_count)
    lambda_tensor = torch.as_tensor(
        lambda_T, device=theta_models.device, dtype=theta_models.dtype
    ).detach()
    logits = (
        delta_keep - lambda_tensor * complexity_saving
    ) / gate_temperature
    keep_probability = torch.sigmoid(logits)
    replay_logits = logits.index_select(0, replay_pair)
    pair_nll_each = -torch.logsumexp(torch.stack((
        F.logsigmoid(replay_logits) - child_mix_nll_each,
        F.logsigmoid(-replay_logits) - parent_nll_each,
    ), dim=-1), dim=-1)
    pair_loss = theta_models.new_zeros(pair_count)
    pair_loss.index_add_(0, replay_pair, pair_nll_each)
    pair_loss = pair_loss / count_tensor
    likelihood_loss = pair_loss.sum()

    full_keep_complexity = theta_models.new_tensor(
        float(len(tree.internal_ids))
    )
    expected_complexity = full_keep_complexity - (
        1.0 - keep_probability
    ).sum()
    if budget_KT is None:
        budget = full_keep_complexity.detach()
    else:
        budget = torch.as_tensor(
            budget_KT,
            device=theta_models.device,
            dtype=theta_models.dtype,
        ).detach()
    dual_term = lambda_tensor * (expected_complexity - budget)
    loss = likelihood_loss + dual_term

    embedding_distance = torch.stack([
        torch.linalg.vector_norm(tree.node_emb[node_a] - tree.node_emb[node_b])
        for node_a, node_b in eligible_pairs
    ])
    semantic_distance = torch.linalg.vector_norm(
        theta_models[:, 0] - theta_models[:, 1], dim=-1
    )
    return {
        "loss": loss,
        "likelihood_loss": likelihood_loss,
        "dual_term": dual_term,
        "expected_complexity": expected_complexity,
        "full_keep_complexity": full_keep_complexity,
        "pairs": eligible_pairs,
        "parents": parents,
        "replay_counts": replay_counts,
        "effective_sample_size": effective_sample_size,
        "branch_support": branch_support_tensor,
        "skipped": skipped,
        "delta_keep": delta_keep,
        "delta_keep_variance": delta_keep_variance,
        "complexity_saving": complexity_saving,
        "pair_loss": pair_loss,
        "keep_probability": keep_probability,
        "child_mix_nll": child_mix_nll,
        "parent_nll": parent_nll,
        "merge_parent_theta": merge_parent_theta,
        "child_weights": child_weights,
        "embedding_distance": embedding_distance,
        "semantic_distance": semantic_distance,
    }


@torch.no_grad()
def compute_differentiable_merge_objective(
    tree,
    pairs: Sequence[tuple[str, str]],
    decays: Tensor,
    *,
    lambda_T: float | Tensor,
    budget_KT: Optional[float | Tensor] = None,
    gate_temperature: float = 1.0,
    min_replay: int = 8,
    normalize_by_events: bool = True,
    hawkes_ll: Optional[HawkesFamily] = None,
    max_replay: Optional[int] = None,
    embedding_fn: Optional[Any] = None,
    embedding_batch_fn: Optional[Any] = None,
    stale_weight: float = 0.2,
    uncertainty_kappa: float = 1.0,
    dynamics_weight: float = 0.1,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Evaluate frozen production KEEP versus a virtual merged contraction.

    KEEP is the actual current model output, including routing, frontier mass,
    ancestor contributions and episodic retrieval.  MERGE replaces only the
    candidate region with a physical-space fused parent and a leave-one-out
    virtual bank whose rows have been exactly rebased.  No model or topology
    state is mutated and replay rows use the same quality/usage/staleness
    importance estimator as Topology Prune.

    The historical function name is retained for checkpoint/API compatibility;
    this estimator is intentionally frozen and does not expose a training
    gradient.
    """
    if gate_temperature <= 0.0:
        raise ValueError("gate_temperature must be positive")
    if min_replay < 0:
        raise ValueError("min_replay must be non-negative")
    if max_replay is not None and max_replay <= 0:
        raise ValueError("max_replay must be positive when provided")
    if stale_weight < 0.0 or uncertainty_kappa < 0.0 or dynamics_weight < 0.0:
        raise ValueError("Merge counterfactual weights must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    hawkes_ll = _resolve_hawkes_family(tree, decays, hawkes_ll)

    # Local imports avoid a module cycle: TopologyPrune itself reuses Merge's
    # replay/cache helpers.  These are the shared frozen-counterfactual kernels.
    from Sleep.TopologyPrune import (
        _batched_topology_prune_terms,
        _build_virtual_bank,
        _collect_replay,
        _fallback_keep_state,
        _normalized_weights,
        _path_delta_batch,
        _production_region_terms,
        _region_terms_from_frontier,
        _retrieve_virtual_loo,
    )

    root_id = next(
        node_id for node_id, node in tree.nodes.items()
        if node.parent is None
    )
    reference = _semantic_theta(tree, root_id).detach()
    eligible_pairs: list[tuple[str, str]] = []
    parents: list[str] = []
    rows_by_pair: list[list[Any]] = []
    replay_counts: list[int] = []
    branch_support: list[tuple[int, int]] = []
    targets: list[Tensor] = []
    child_weight_rows: list[Tensor] = []
    skipped: Dict[tuple[str, str], Dict[str, Any]] = {}

    for pair in pairs:
        node_a, node_b = pair
        parent_id = _validate_leaf_siblings(tree, node_a, node_b)
        rows = _collect_replay(
            tree,
            parent_id,
            max_replay=max_replay,
            stale_weight=stale_weight,
        )
        if len(rows) < min_replay:
            skipped[pair] = {
                "merge": False,
                "reason": "insufficient_replay",
                "replay_size": len(rows),
            }
            continue
        target, child_weights = fused_parent_theta(tree, node_a, node_b)
        eligible_pairs.append(pair)
        parents.append(parent_id)
        rows_by_pair.append(rows)
        replay_counts.append(len(rows))
        branch_support.append((
            sum(row.source_node == node_a for row in rows),
            sum(row.source_node == node_b for row in rows),
        ))
        targets.append(target.detach())
        child_weight_rows.append(child_weights.detach())

    if not eligible_pairs:
        zero = reference.new_zeros(())
        empty = zero.new_empty((0,))
        return {
            "loss": zero,
            "likelihood_loss": zero,
            "dual_term": zero,
            "expected_complexity": zero,
            "full_keep_complexity": zero,
            "pairs": [],
            "parents": [],
            "replay_counts": [],
            "effective_sample_size": empty,
            "branch_support": zero.new_empty((0, 2)),
            "skipped": skipped,
            "delta_keep": empty,
            "delta_keep_variance": empty,
            "predictive_damage": empty,
            "uncertainty_margin": empty,
            "tpp_divergence": empty,
            "retention_cost": empty,
            "complexity_saving": empty,
            "pair_loss": empty,
            "keep_probability": empty,
            "child_mix_nll": empty,
            "parent_nll": empty,
            "keep_nll": empty,
            "merge_nll": empty,
            "merge_parent_theta": zero.new_empty((0, reference.numel())),
            "child_weights": zero.new_empty((0, 2)),
            "embedding_distance": empty,
            "semantic_distance": empty,
            "keep_mode": [],
        }

    # Compute the frozen production state for all candidates with one encoder
    # batch and one tree forward whenever the batched callback is available.
    production_states: Dict[
        str, tuple[Tensor, Tensor, Tensor, Tensor, str]
    ] = {}
    if embedding_batch_fn is not None:
        flat_rows = [row for rows in rows_by_pair for row in rows]
        embeddings = embedding_batch_fn([
            row.window for row in flat_rows
        ]).detach()
        if embeddings.shape != (len(flat_rows), tree.z_dim):
            raise ValueError(
                "batched Merge replay embeddings must have shape "
                "[total_replay, tree.z_dim]"
            )
        output = tree(
            embeddings,
            working_delta=embeddings.new_zeros(tree.param_dim),
            decays=hawkes_ll.decays,
            update_memory_state=False,
            detach_routing=True,
            update_search_state=False,
            materialize_diagnostics=True,
        )
        start = 0
        for parent_id, rows in zip(parents, rows_by_pair):
            end = start + len(rows)
            production_states[parent_id] = _region_terms_from_frontier(
                tree,
                parent_id,
                output["effective_params"].theta.detach()[start:end],
                output["memory_query"].detach()[start:end],
                output["frontier_mass"].detach()[start:end],
                output["frontier_theta"].detach()[start:end],
                output["frontier_node_ids"][start:end],
            )
            start = end
    elif embedding_fn is not None:
        for parent_id, rows in zip(parents, rows_by_pair):
            production_states[parent_id] = _production_region_terms(
                tree,
                parent_id,
                rows,
                hawkes_ll.decays,
                embedding_fn,
            )

    keep_nll_rows: list[Tensor] = []
    merge_nll_rows: list[Tensor] = []
    predictive_rows: list[Tensor] = []
    variance_rows: list[Tensor] = []
    uncertainty_rows: list[Tensor] = []
    tpp_rows: list[Tensor] = []
    retention_rows: list[Tensor] = []
    effective_rows: list[Tensor] = []
    keep_modes: list[str] = []

    for parent_id, rows, target in zip(parents, rows_by_pair, targets):
        if parent_id in production_states:
            keep_theta, queries, other_theta, region_mass, keep_mode = (
                production_states[parent_id]
            )
        else:
            keep_theta, queries, other_theta, region_mass, keep_mode = (
                _fallback_keep_state(tree, parent_id, rows)
            )
        weights = _normalized_weights(rows, keep_theta)
        effective = 1.0 / weights.square().sum().clamp_min(eps)

        virtual = _build_virtual_bank(
            tree,
            parent_id,
            target_parent_theta=target,
        )
        virtual_delta, _ = _retrieve_virtual_loo(
            tree, queries, rows, virtual
        )
        ancestor_delta = _path_delta_batch(
            tree, queries, parent_id, include_node=False
        )
        merged_region_theta = (
            target.unsqueeze(0) + ancestor_delta + virtual_delta
        )
        merge_theta = (
            other_theta
            + region_mass.unsqueeze(-1) * merged_region_theta
        )
        keep_loss, merge_loss, tpp_each = _batched_topology_prune_terms(
            [row.window for row in rows],
            keep_theta,
            merge_theta,
            hawkes_ll,
            eps,
            normalize_by_events=normalize_by_events,
        )
        damage = merge_loss - keep_loss
        predictive_damage = (weights * damage).sum()
        denominator = (1.0 - weights.square().sum()).clamp_min(eps)
        variance = (
            weights * (damage - predictive_damage).square()
        ).sum() / denominator
        uncertainty = (
            float(uncertainty_kappa)
            * variance.clamp_min(0.0).sqrt()
            / effective.clamp_min(eps).sqrt()
        )
        tpp_divergence = (weights * tpp_each.clamp_min(0.0)).sum()
        retention = (
            predictive_damage
            + uncertainty
            + float(dynamics_weight) * tpp_divergence
        )
        keep_nll_rows.append((weights * keep_loss).sum())
        merge_nll_rows.append((weights * merge_loss).sum())
        predictive_rows.append(predictive_damage)
        variance_rows.append(variance.clamp_min(0.0))
        uncertainty_rows.append(uncertainty)
        tpp_rows.append(tpp_divergence)
        retention_rows.append(retention)
        effective_rows.append(effective)
        keep_modes.append(keep_mode)

    keep_nll = torch.stack(keep_nll_rows)
    merge_nll = torch.stack(merge_nll_rows)
    predictive_damage = torch.stack(predictive_rows)
    damage_variance = torch.stack(variance_rows)
    uncertainty_margin = torch.stack(uncertainty_rows)
    tpp_divergence = torch.stack(tpp_rows)
    retention_cost = torch.stack(retention_rows)
    effective_sample_size = torch.stack(effective_rows)
    merge_parent_theta = torch.stack(targets)
    child_weights = torch.stack(child_weight_rows)
    complexity_saving = reference.new_ones(len(eligible_pairs))
    lambda_tensor = torch.as_tensor(
        lambda_T, device=reference.device, dtype=reference.dtype
    ).detach()
    logits = (
        retention_cost - lambda_tensor * complexity_saving
    ) / gate_temperature
    keep_probability = torch.sigmoid(logits)
    pair_loss = retention_cost
    likelihood_loss = retention_cost.sum()
    full_keep_complexity = reference.new_tensor(
        float(len(tree.internal_ids))
    )
    expected_complexity = full_keep_complexity - (
        1.0 - keep_probability
    ).sum()
    budget = (
        full_keep_complexity
        if budget_KT is None
        else torch.as_tensor(
            budget_KT, device=reference.device, dtype=reference.dtype
        ).detach()
    )
    dual_term = lambda_tensor * (expected_complexity - budget)
    loss = likelihood_loss + dual_term
    embedding_distance = torch.stack([
        torch.linalg.vector_norm(tree.node_emb[node_a] - tree.node_emb[node_b])
        for node_a, node_b in eligible_pairs
    ])
    semantic_distance = torch.stack([
        torch.linalg.vector_norm(
            _semantic_theta(tree, node_a) - _semantic_theta(tree, node_b)
        )
        for node_a, node_b in eligible_pairs
    ])
    return {
        "loss": loss,
        "likelihood_loss": likelihood_loss,
        "dual_term": dual_term,
        "expected_complexity": expected_complexity,
        "full_keep_complexity": full_keep_complexity,
        "pairs": eligible_pairs,
        "parents": parents,
        "replay_counts": replay_counts,
        "effective_sample_size": effective_sample_size,
        "branch_support": reference.new_tensor(branch_support),
        "skipped": skipped,
        # Compatibility aliases: delta_keep is now the actual frozen
        # production-to-Merge predictive damage, not a child mixture gap.
        "delta_keep": predictive_damage,
        "delta_keep_variance": damage_variance,
        "predictive_damage": predictive_damage,
        "uncertainty_margin": uncertainty_margin,
        "tpp_divergence": tpp_divergence,
        "retention_cost": retention_cost,
        "complexity_saving": complexity_saving,
        "pair_loss": pair_loss,
        "keep_probability": keep_probability,
        "child_mix_nll": keep_nll,
        "parent_nll": merge_nll,
        "keep_nll": keep_nll,
        "merge_nll": merge_nll,
        "merge_parent_theta": merge_parent_theta,
        "child_weights": child_weights,
        "embedding_distance": embedding_distance,
        "semantic_distance": semantic_distance,
        "keep_mode": keep_modes,
    }


@torch.no_grad()
def make_merge_decisions(
    objective: Mapping[str, Any],
    *,
    lambda_T: float,
    gate_temperature: float,
) -> Dict[tuple[str, str], MergeResult]:
    """MAP-decode detached gates after optimization; never mutate topology."""
    if gate_temperature <= 0.0:
        raise ValueError("gate_temperature must be positive")
    decisions: Dict[tuple[str, str], MergeResult] = {
        pair: dict(value)
        for pair, value in objective.get("skipped", {}).items()
    }
    delta_keep = objective["delta_keep"].detach()
    retention_cost = objective.get("retention_cost", delta_keep).detach()
    complexity_saving = objective["complexity_saving"].detach()
    logits = (
        retention_cost - float(lambda_T) * complexity_saving
    ) / gate_temperature
    keep_probability = torch.sigmoid(logits)
    for index, pair in enumerate(objective["pairs"]):
        keep = bool(keep_probability[index].item() > 0.5)
        decisions[pair] = {
            "merge": not keep,
            "reason": "map_keep" if keep else "map_merge",
            "keep_probability": float(keep_probability[index].item()),
            "delta_keep": float(delta_keep[index].item()),
            "retention_cost": float(retention_cost[index].item()),
            "tpp_divergence": float(
                objective.get(
                    "tpp_divergence",
                    torch.zeros_like(delta_keep),
                )[index].item()
            ),
            "complexity_saving": float(complexity_saving[index].item()),
            "pair_loss": float(objective["pair_loss"][index].item()),
            "child_mix_nll": float(objective["child_mix_nll"][index].item()),
            "parent_nll": float(objective["parent_nll"][index].item()),
            "embedding_distance": float(
                objective["embedding_distance"][index].item()
            ),
            "semantic_distance": float(
                objective["semantic_distance"][index].item()
            ),
            "replay_size": int(objective["replay_counts"][index]),
        }
    return decisions


def compute_merge_score(
    tree,
    node_a: str,
    node_b: str,
    decays: Tensor,
    normalize_by_events: bool = True,
    hawkes_ll: Optional[HawkesFamily] = None,
    max_replay: Optional[int] = None,
) -> Tensor:
    """Compatibility diagnostic: virtual-Merge minus frozen-KEEP NLL."""
    result = compute_differentiable_merge_objective(
        tree,
        [(node_a, node_b)],
        decays,
        lambda_T=0.0,
        gate_temperature=1.0,
        min_replay=1,
        normalize_by_events=normalize_by_events,
        hawkes_ll=hawkes_ll,
        max_replay=max_replay,
    )
    if not result["pairs"]:
        return _semantic_theta(tree, node_a).new_tensor(float("inf"))
    return result["delta_keep"][0]


def _remove_parameters_from_optimizer(
    optimizer: torch.optim.Optimizer,
    removed_parameters: Iterable[torch.nn.Parameter],
) -> None:
    removed_ids = {id(parameter) for parameter in removed_parameters}
    for group in optimizer.param_groups:
        group["params"] = [
            parameter for parameter in group["params"] if id(parameter) not in removed_ids
        ]
    for parameter in list(optimizer.state):
        if id(parameter) in removed_ids:
            del optimizer.state[parameter]


@torch.no_grad()
def _commit_merge_legacy(
    tree,
    node_a: str,
    node_b: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    decays: Optional[Tensor] = None,
    hawkes_ll: Optional[HawkesFamily] = None,
    memory_hard_threshold: float = 0.0,
) -> str:
    """Merge sibling banks, then keep replay explained well by the parent.

    The parent's semantic parameters are never changed. Child residuals are
    rebased before all three banks are combined. For each replay window, the
    hard filter compares the parent's normalized log-likelihood with the
    memory-specialized effective parameters. The item is retained when the
    specialized advantage is at most ``memory_hard_threshold``.

    Legacy items without a replay window are retained conservatively because
    their parent fit cannot be evaluated.
    """
    if not torch.isfinite(torch.tensor(memory_hard_threshold)):
        raise ValueError("memory_hard_threshold must be finite")
    if memory_hard_threshold < 0.0:
        raise ValueError("memory_hard_threshold must be non-negative")
    parent = _validate_leaf_siblings(tree, node_a, node_b)

    theta_parent = _semantic_theta(tree, parent).detach()
    child_thetas = {
        node_a: _semantic_theta(tree, node_a).detach(),
        node_b: _semantic_theta(tree, node_b).detach(),
    }
    parent_bank: MemoryBank = tree.episodic_memory.get_bank(parent)

    # Preserve each replay correction in unconstrained Hawkes space:
    # theta_child + delta_old == theta_parent + delta_new.
    for child_id in (node_a, node_b):
        child_bank = tree.episodic_memory.banks.get(child_id)
        if child_bank is None or len(child_bank) == 0:
            continue
        rebased_deltas = rebase_memory_to_new_leaf(
            delta_theta=child_bank.deltas,
            old_theta=child_thetas[child_id],
            new_theta=theta_parent,
        )
        parent_bank.append_from(
            child_bank,
            torch.arange(len(child_bank), device=child_bank.device),
            deltas=rebased_deltas,
            node_id=parent,
        )

    # Filter only after parent and both child banks have been combined. A
    # memory-specific residual is the strongest local reference available for
    # deciding whether the unchanged parent semantic parameters explain that
    # sequence sufficiently well.
    if any(window is not None for window in parent_bank.windows):
        if hawkes_ll is None and decays is None:
            raise ValueError(
                "decays or hawkes_ll is required to filter merged replay memories"
            )
        resolved_decays = hawkes_ll.decays if decays is None else decays
        resolved_hawkes = _resolve_hawkes_family(
            tree, resolved_decays, hawkes_ll
        )
        keep_mask = torch.ones(
            len(parent_bank), dtype=torch.bool, device=parent_bank.device
        )
        for index, window in enumerate(parent_bank.windows):
            if window is None:
                continue
            parent_logp = _window_log_likelihood(
                window=window,
                theta=theta_parent,
                decays=resolved_decays,
                hawkes_ll=resolved_hawkes,
                normalize_by_events=True,
            )
            effective_logp = _window_log_likelihood(
                window=window,
                theta=theta_parent + parent_bank.deltas[index],
                decays=resolved_decays,
                hawkes_ll=resolved_hawkes,
                normalize_by_events=True,
            )
            finite_parent = bool(torch.isfinite(parent_logp).item())
            finite_effective = bool(torch.isfinite(effective_logp).item())
            keep_mask[index] = finite_parent and (
                not finite_effective
                or effective_logp.item() - parent_logp.item()
                <= memory_hard_threshold
            )
        parent_bank.keep(torch.nonzero(keep_mask, as_tuple=False).flatten())

    if len(parent_bank) > parent_bank.capacity:
        parent_bank.prune()

    parent_node = tree.nodes[parent]
    parent_node.left = None
    parent_node.right = None
    parent_node.split_queue.clear()
    child_mass = 0.0
    if isinstance(getattr(tree, "mass_ema", None), dict):
        child_mass = sum(tree.mass_ema.get(child_id, 0.0) for child_id in (node_a, node_b))

    removed_parameters = [tree.node_emb[node_a], tree.node_emb[node_b]]
    if hasattr(tree, "semantic_offset"):
        removed_parameters.extend(
            [tree.semantic_offset[node_a], tree.semantic_offset[node_b]]
        )
    for child_id in (node_a, node_b):
        del tree.nodes[child_id]
        del tree.node_emb[child_id]
        if hasattr(tree, "semantic_offset"):
            del tree.semantic_offset[child_id]
        tree.episodic_memory.banks.pop(child_id, None)
    # Compatibility with optional pruning statistics used by some sleep loops.
    for attribute in ("mass_ema", "low_mass_streak"):
        state = getattr(tree, attribute, None)
        if isinstance(state, dict):
            state.pop(node_a, None)
            state.pop(node_b, None)
    low_mass_streak = getattr(tree, "low_mass_streak", None)
    if isinstance(low_mass_streak, dict):
        low_mass_streak[parent] = 0
    mass_ema = getattr(tree, "mass_ema", None)
    if isinstance(mass_ema, dict):
        mass_ema[parent] = child_mass

    if optimizer is not None:
        _remove_parameters_from_optimizer(optimizer, removed_parameters)

    tree.refresh_structure_buffers()
    return parent


@torch.no_grad()
def commit_merge(
    tree,
    node_a: str,
    node_b: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    decays: Optional[Tensor] = None,
    hawkes_ll: Optional[HawkesFamily] = None,
    memory_hard_threshold: float = 0.0,
    target_parent_theta: Optional[Tensor] = None,
    child_weights: Optional[Tensor | Sequence[float]] = None,
    snapshot_signature: Optional[tuple] = None,
    return_result: bool = False,
) -> Any:
    """Commit child-semantic fusion followed by atomic contraction.

    Stable child semantics are compressed upward into the parent. Parent and
    child episodic rows are then rebased against that fused target before the
    children disappear. Sleep never deletes replay rows during topology
    commit; the historical replay-filter argument remains validation-only.
    """
    del decays, hawkes_ll
    if not torch.isfinite(torch.tensor(memory_hard_threshold)):
        raise ValueError("memory_hard_threshold must be finite")
    if memory_hard_threshold < 0.0:
        raise ValueError("memory_hard_threshold must be non-negative")
    parent = _validate_leaf_siblings(tree, node_a, node_b)
    if target_parent_theta is None:
        target_parent_theta, normalized_weights = fused_parent_theta(
            tree,
            node_a,
            node_b,
            child_weights=child_weights,
        )
    else:
        target_parent_theta = torch.as_tensor(
            target_parent_theta,
            device=_semantic_theta(tree, parent).device,
            dtype=_semantic_theta(tree, parent).dtype,
        ).detach()
        normalized_weights = _normalized_child_weights(
            tree,
            node_a,
            node_b,
            reference=target_parent_theta,
            child_weights=child_weights,
        )
    del normalized_weights
    from Sleep.Collapse import contract_leaf_pair

    result = contract_leaf_pair(
        tree,
        parent,
        target_parent_theta=target_parent_theta,
        snapshot_signature=snapshot_signature,
        optimizer=optimizer,
        preserve_memory=True,
        reconciliation_policy="merge",
    )
    return result if return_result else result.parent_id
