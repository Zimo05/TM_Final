"""Prediction-compression pruning for complete leaf refinements."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from HawkesBackbone import HawkesFamily
from MemoryResiduals.MemoryBank import EventWindow
from Sleep.Collapse import (
    collapse_snapshot_signature,
    contract_leaf_pair,
    rebase_memory_to_new_leaf,
)
from Sleep.Merge import (
    _ranked_bank_indices,
    _window_target_tensors,
)


ReplayEmbeddingFn = Callable[[EventWindow], Tensor]
ReplayEmbeddingBatchFn = Callable[[Sequence[EventWindow]], Tensor]


def _responsibility_by_leaf(
    tree,
    responsibilities: Mapping[str, Tensor] | Tensor,
) -> Dict[str, Tensor]:
    if torch.is_tensor(responsibilities):
        if responsibilities.ndim == 0:
            raise ValueError("responsibilities tensor must include a leaf dimension")
        if responsibilities.shape[-1] != len(tree.leaf_ids):
            raise ValueError(
                "responsibilities final dimension must match tree.leaf_ids"
            )
        return {
            leaf_id: responsibilities[..., index]
            for index, leaf_id in enumerate(tree.leaf_ids)
        }
    if isinstance(responsibilities, Mapping):
        return dict(responsibilities)
    raise TypeError("responsibilities must be a tensor or leaf-to-tensor mapping")


@torch.no_grad()
def update_leaf_mass(
    tree,
    responsibilities: Mapping[str, Tensor] | Tensor,
    ema_decay: float = 0.95,
) -> Dict[str, float]:
    """Maintain routing-mass diagnostics without authorizing deletion."""
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    by_leaf = _responsibility_by_leaf(tree, responsibilities)
    active_leaves = set(tree.get_leaf_ids())
    prepared = []
    for leaf_id, responsibility in by_leaf.items():
        if leaf_id not in active_leaves:
            continue
        if not torch.is_tensor(responsibility) or responsibility.numel() == 0:
            raise ValueError(
                f"responsibility for {leaf_id!r} must be a non-empty tensor"
            )
        prepared.append((leaf_id, responsibility.detach().float()))

    aligned = bool(prepared) and len({
        (tuple(value.shape), value.device)
        for _, value in prepared
    }) == 1
    if aligned:
        packed = torch.stack([
            value.reshape(-1) for _, value in prepared
        ])
        means = packed.mean(dim=-1)
        finite = torch.isfinite(packed).all(dim=-1)
        in_range = ((packed >= 0.0) & (packed <= 1.0)).all(dim=-1)
        summary = torch.cat((
            means,
            finite.to(means.dtype),
            in_range.to(means.dtype),
        )).cpu().tolist()
        count = len(prepared)
        current_values = summary[:count]
        finite_values = summary[count:2 * count]
        range_values = summary[2 * count:]
    else:
        current_values = []
        finite_values = []
        range_values = []
        for _, value in prepared:
            row = torch.stack((
                value.mean(),
                torch.isfinite(value).all().to(value.dtype),
                ((value >= 0.0) & (value <= 1.0)).all().to(value.dtype),
            )).cpu().tolist()
            current_values.append(row[0])
            finite_values.append(row[1])
            range_values.append(row[2])

    for (leaf_id, _), current_mass, finite, in_range in zip(
        prepared,
        current_values,
        finite_values,
        range_values,
    ):
        if not bool(finite):
            raise ValueError(f"responsibility for {leaf_id!r} must be finite")
        if not bool(in_range):
            raise ValueError(f"responsibility for {leaf_id!r} must lie in [0, 1]")
        old_mass = tree.mass_ema.get(leaf_id, current_mass)
        tree.mass_ema[leaf_id] = (
            ema_decay * old_mass + (1.0 - ema_decay) * current_mass
        )
    for state in (tree.mass_ema, tree.low_mass_streak):
        for node_id in set(state).difference(active_leaves):
            state.pop(node_id, None)
    return {
        leaf_id: tree.mass_ema[leaf_id]
        for leaf_id in active_leaves
        if leaf_id in tree.mass_ema
    }


@dataclass(frozen=True)
class TopologyPruneProposal:
    parent_id: str
    child_ids: tuple[str, str]
    snapshot_signature: tuple
    eligible: bool
    reason: str
    keep_mode: str
    replay_size: int
    effective_replay_size: float
    branch_balance: float
    keep_nll: float
    prune_nll: float
    predictive_damage: float
    near_zero_damage: bool
    uncertainty_margin: float
    tpp_divergence: float
    retention_cost: float
    complexity_saving: float
    prior_probability: float
    prune_gain: float
    prune_probability: float
    posterior_gain: float = float("-inf")
    locally_positive: bool = False
    persistence_ok: bool = False
    persistence: int = 0


@dataclass(frozen=True)
class _ReplayRow:
    window: EventWindow
    query: Tensor
    source_node: str
    source_index: int
    importance: float


@dataclass(frozen=True)
class _VirtualBank:
    keys: Tensor
    context_keys: Tensor
    context_valid: Tensor
    deltas: Tensor
    write_quality: Tensor
    usage: Tensor
    age: Tensor
    source_positions: Mapping[tuple[str, int], int]


def tree_complexity(tree) -> float:
    """Count active binary branch decisions."""
    return float(len(tree.internal_ids))


def candidate_prune_parents(tree) -> list[str]:
    """Return internal parents whose two children are current leaves."""
    candidates = []
    for parent_id in tree.internal_ids:
        parent = tree.nodes[parent_id]
        if parent.left is None or parent.right is None:
            continue
        if tree.nodes[parent.left].is_leaf and tree.nodes[parent.right].is_leaf:
            candidates.append(parent_id)
    return candidates


def _importance_batch(bank, indices: Tensor, stale_weight: float) -> Tensor:
    """Score aligned replay rows in one device kernel."""
    indices = indices.to(device=bank.device, dtype=torch.long)
    return (
        bank.write_quality.index_select(0, indices).clamp_min(0.0)
        * (
            1.0
            + torch.log1p(
                bank.usage.index_select(0, indices).clamp_min(0.0)
            )
        )
        * torch.exp(
            -float(stale_weight)
            * bank.stale_cycles.index_select(0, indices).clamp_min(0.0)
        )
    )


def _collect_replay(
    tree,
    parent_id: str,
    *,
    max_replay: Optional[int],
    stale_weight: float,
) -> list[_ReplayRow]:
    parent = tree.nodes[parent_id]
    node_ids = (parent_id, parent.left, parent.right)
    rows: list[_ReplayRow] = []
    for node_id in node_ids:
        bank = tree.episodic_memory.banks.get(node_id)
        if bank is None:
            continue
        valid_indices = []
        for index in _ranked_bank_indices(bank):
            window = bank.windows[index]
            if (
                window is None
                or window.end_idx <= window.start_idx
                or window.end_idx > int(window.times.numel())
            ):
                continue
            valid_indices.append(index)
        if not valid_indices:
            continue
        index_tensor = torch.tensor(
            valid_indices,
            device=bank.device,
            dtype=torch.long,
        )
        # One synchronization per bank replaces one synchronization per row.
        importance_values = _importance_batch(
            bank,
            index_tensor,
            stale_weight,
        ).detach().cpu().tolist()
        for index, importance in zip(valid_indices, importance_values):
            window = bank.windows[index]
            rows.append(_ReplayRow(
                window=window,
                query=bank.keys[index].detach(),
                source_node=node_id,
                source_index=index,
                importance=float(importance),
            ))
    rows.sort(key=lambda row: row.importance, reverse=True)
    if max_replay is not None:
        rows = rows[: int(max_replay)]
    return rows


def _normalized_weights(rows: Sequence[_ReplayRow], reference: Tensor) -> Tensor:
    values = reference.new_tensor([row.importance for row in rows]).clamp_min(0.0)
    total = values.sum()
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("topology-prune replay weights are non-finite")
    if float(total.detach().cpu()) <= 0.0:
        return torch.full_like(values, 1.0 / max(len(rows), 1))
    return values / total


def _build_virtual_bank(
    tree,
    parent_id: str,
    target_parent_theta: Optional[Tensor] = None,
) -> _VirtualBank:
    """Build the non-mutating collapsed bank for Prune or Merge.

    Prune retains the existing parent semantic.  Merge supplies its fused
    parent semantic.  In both cases every source row is exactly rebased to the
    counterfactual reference before leave-one-out retrieval.
    """
    parent = tree.nodes[parent_id]
    child_ids = (parent.left, parent.right)
    parent_theta = tree.semantic_theta(parent_id).detach()
    target_theta = (
        parent_theta
        if target_parent_theta is None
        else torch.as_tensor(
            target_parent_theta,
            device=parent_theta.device,
            dtype=parent_theta.dtype,
        ).detach().reshape(-1)
    )
    if target_theta.shape != parent_theta.shape:
        raise ValueError("target_parent_theta must match parent semantics")
    memory = tree.episodic_memory
    source_positions: Dict[tuple[str, int], int] = {}
    keys = []
    deltas = []
    qualities = []
    usage = []
    ages = []
    context_keys = []
    context_valid = []
    alias_capacity = 3
    position = 0
    for node_id in (parent_id, *child_ids):
        bank = memory.banks.get(node_id)
        if bank is None or not len(bank):
            continue
        node_delta = rebase_memory_to_new_leaf(
            bank.deltas,
            tree.semantic_theta(node_id).detach(),
            target_theta,
        )
        keys.append(bank.keys)
        bank._ensure_prototype_state()
        alias_capacity = max(alias_capacity, int(bank.context_alias_capacity))
        context_keys.append(bank.context_keys)
        context_valid.append(bank.context_valid)
        deltas.append(node_delta)
        qualities.append(bank.write_quality)
        usage.append(bank.usage)
        ages.append(bank.effective_age(memory._age_clock))
        for index in range(len(bank)):
            source_positions[(node_id, index)] = position
            position += 1
    reference = target_theta
    if not keys:
        return _VirtualBank(
            keys=reference.new_empty((0, memory.key_dim)),
            context_keys=reference.new_empty((0, alias_capacity, memory.key_dim)),
            context_valid=torch.empty(
                (0, alias_capacity), dtype=torch.bool, device=reference.device
            ),
            deltas=reference.new_empty((0, memory.param_dim)),
            write_quality=reference.new_empty(0),
            usage=reference.new_empty(0),
            age=reference.new_empty(0),
            source_positions=source_positions,
        )
    padded_context_keys = [
        F.pad(value, (0, 0, 0, alias_capacity - value.size(1)))
        for value in context_keys
    ]
    padded_context_valid = [
        F.pad(value, (0, alias_capacity - value.size(1)))
        for value in context_valid
    ]
    return _VirtualBank(
        keys=torch.cat(keys, dim=0),
        context_keys=torch.cat(padded_context_keys, dim=0),
        context_valid=torch.cat(padded_context_valid, dim=0),
        deltas=torch.cat(deltas, dim=0),
        write_quality=torch.cat(qualities, dim=0),
        usage=torch.cat(usage, dim=0),
        age=torch.cat(ages, dim=0),
        source_positions=source_positions,
    )


def _retrieve_virtual_loo(
    tree,
    queries: Tensor,
    rows: Sequence[_ReplayRow],
    virtual: _VirtualBank,
) -> tuple[Tensor, Tensor]:
    replay_count = queries.size(0)
    if virtual.keys.size(0) == 0:
        return (
            queries.new_zeros((replay_count, tree.param_dim)),
            queries.new_zeros((replay_count, 0)),
        )
    row_count = virtual.keys.size(0)
    valid = torch.ones(
        replay_count,
        row_count,
        dtype=torch.bool,
        device=queries.device,
    )
    positions = torch.tensor(
        [
            virtual.source_positions.get(
                (row.source_node, row.source_index),
                -1,
            )
            for row in rows
        ],
        device=queries.device,
        dtype=torch.long,
    )
    present = positions >= 0
    replay_indices = torch.arange(
        replay_count,
        device=queries.device,
        dtype=torch.long,
    )
    valid[replay_indices[present], positions[present]] = False
    active = valid.any(dim=-1)
    delta = queries.new_zeros((replay_count, tree.param_dim))
    alpha = queries.new_zeros((replay_count, row_count))
    if bool(active.any().item()):
        active_delta, info = tree.episodic_memory.retriever.forward_batched(
            query=queries[active],
            keys=virtual.context_keys.unsqueeze(0).expand(
                replay_count, -1, -1, -1
            )[active],
            deltas=virtual.deltas.unsqueeze(0).expand(replay_count, -1, -1)[active],
            usage=virtual.usage.unsqueeze(0).expand(replay_count, -1)[active],
            age=virtual.age.unsqueeze(0).expand(replay_count, -1)[active],
            valid_mask=valid[active],
            write_quality=virtual.write_quality.unsqueeze(0).expand(
                replay_count, -1
            )[active],
            context_valid=virtual.context_valid.unsqueeze(0).expand(
                replay_count, -1, -1
            ).clone()[active].masked_fill(
                ~valid[active].unsqueeze(-1), False
            ),
        )
        delta[active] = active_delta
        alpha[active] = info["alpha"]
    return delta, alpha


def _path_delta_batch(tree, queries: Tensor, node_id: str, *, include_node: bool) -> Tensor:
    path = list(tree.node_paths[node_id])
    if not include_node:
        path = path[:-1]
    if not path:
        return queries.new_zeros((queries.size(0), tree.param_dim))
    node_index = {value: index for index, value in enumerate(tree.all_node_ids)}
    indices = torch.tensor(
        [node_index[value] for value in path],
        device=queries.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(queries.size(0), -1)
    mask = torch.ones_like(indices, dtype=torch.bool)
    delta, _ = tree.episodic_memory.read_packed(
        queries,
        indices,
        mask,
        tree.all_node_ids,
        update_state=False,
    )
    return delta.sum(dim=1)


def _production_region_terms(
    tree,
    parent_id: str,
    rows: Sequence[_ReplayRow],
    decays: Tensor,
    embedding_fn: ReplayEmbeddingFn,
    embedding_batch_fn: Optional[ReplayEmbeddingBatchFn] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, str]:
    windows = [row.window for row in rows]
    embeddings = (
        embedding_batch_fn(windows).detach()
        if embedding_batch_fn is not None
        else torch.stack([
            embedding_fn(window).detach().reshape(-1)
            for window in windows
        ])
    )
    if embeddings.shape != (len(rows), tree.z_dim):
        raise ValueError("replay embeddings must have shape [R, tree.z_dim]")
    output = tree(
        embeddings,
        working_delta=embeddings.new_zeros(tree.param_dim),
        decays=decays,
        update_memory_state=False,
        detach_routing=True,
        update_search_state=False,
        materialize_diagnostics=True,
    )
    keep_theta = output["effective_params"].theta.detach()
    queries = output["memory_query"].detach()
    return _region_terms_from_frontier(
        tree,
        parent_id,
        keep_theta,
        queries,
        output["frontier_mass"].detach(),
        output["frontier_theta"].detach(),
        output["frontier_node_ids"],
    )


def _region_terms_from_frontier(
    tree,
    parent_id: str,
    keep_theta: Tensor,
    queries: Tensor,
    frontier_mass: Tensor,
    frontier_theta: Tensor,
    frontier_node_ids: Sequence[Sequence[str]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, str]:
    """Remove one candidate region from an already batched tree output."""
    parent = tree.nodes[parent_id]
    region_ids = {parent_id, parent.left, parent.right}
    region_mask = torch.zeros_like(frontier_mass, dtype=torch.bool)
    mask_rows = []
    mask_slots = []
    for replay_index, frontier_ids in enumerate(frontier_node_ids):
        for slot, node_id in enumerate(frontier_ids):
            if node_id in region_ids:
                mask_rows.append(replay_index)
                mask_slots.append(slot)
    if mask_rows:
        region_mask[
            torch.tensor(mask_rows, device=frontier_mass.device),
            torch.tensor(mask_slots, device=frontier_mass.device),
        ] = True
    region_weight = frontier_mass * region_mask.to(frontier_mass.dtype)
    region_mass = region_weight.sum(dim=-1)
    other_theta = keep_theta - (
        region_weight.unsqueeze(-1) * frontier_theta
    ).sum(dim=1)
    return (
        keep_theta,
        queries,
        other_theta,
        region_mass,
        "production_parameter_mix",
    )


def _fallback_keep_state(
    tree,
    parent_id: str,
    rows: Sequence[_ReplayRow],
) -> tuple[Tensor, Tensor, Tensor, Tensor, str]:
    queries = torch.stack([row.query for row in rows], dim=0)
    parent = tree.nodes[parent_id]
    child_ids = (parent.left, parent.right)
    child_theta = []
    for child_id in child_ids:
        child_theta.append(
            tree.semantic_theta(child_id).detach().unsqueeze(0)
            + _path_delta_batch(tree, queries, child_id, include_node=True)
        )
    masses = queries.new_tensor([
        max(float(tree.mass_ema.get(child_id, 0.0)), 0.0)
        for child_id in child_ids
    ])
    if float(masses.sum().detach().cpu()) <= 0.0:
        alpha = masses.new_full((2,), 0.5)
    else:
        alpha = masses / masses.sum()
    keep_theta = alpha[0] * child_theta[0] + alpha[1] * child_theta[1]
    return (
        keep_theta,
        queries,
        keep_theta.new_zeros(keep_theta.shape),
        keep_theta.new_ones(len(rows)),
        "mass_mixture_fallback",
    )


def _event_local_tpp_divergence(
    window: EventWindow,
    keep_theta: Tensor,
    prune_theta: Tensor,
    hawkes_ll: HawkesFamily,
    eps: float,
) -> Tensor:
    history, _, duration, types = _window_target_tensors(
        window, hawkes_ll, keep_theta
    )
    count = int(types.numel())
    if count <= 0:
        return keep_theta.new_zeros(())
    D = hawkes_ll.num_types
    M = hawkes_ll.num_basis

    def intensity(theta: Tensor) -> Tensor:
        mu = F.softplus(theta[:D])
        W = F.softplus(theta[D:].reshape(D, D, M))
        return (
            mu.unsqueeze(0)
            + torch.einsum("dsm,esm->ed", W, history)
        ).clamp_min(eps)

    keep_intensity = intensity(keep_theta).detach()
    prune_intensity = intensity(prune_theta)
    divergence = duration.unsqueeze(-1) * (
        keep_intensity
        * (keep_intensity.log() - prune_intensity.log())
        - keep_intensity
        + prune_intensity
    )
    return divergence.sum() / max(count, 1)


def _batched_topology_prune_terms(
    windows: Sequence[EventWindow],
    keep_theta: Tensor,
    prune_theta: Tensor,
    hawkes_ll: HawkesFamily,
    eps: float,
    *,
    normalize_by_events: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate KEEP/PRUNE NLL and TPP divergence in one event batch.

    Window caches are prepared on shallow copies so proposal evaluation stays
    observationally pure. Variable-length target intervals are flattened and
    reduced back to their replay windows with ``index_add_``; consequently all
    parameter-dependent Hawkes algebra runs as one GPU tensor graph.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if keep_theta.ndim != 2 or prune_theta.shape != keep_theta.shape:
        raise ValueError(
            "keep_theta and prune_theta must be aligned [replay, parameter] tensors"
        )
    replay_count = len(windows)
    if keep_theta.size(0) != replay_count:
        raise ValueError("theta rows must align with replay windows")
    D = hawkes_ll.num_types
    M = hawkes_ll.num_basis
    expected_parameters = D + D * D * M
    if keep_theta.size(1) != expected_parameters:
        raise ValueError(
            f"theta rows must contain {expected_parameters} Hawkes parameters"
        )
    if replay_count == 0:
        empty = keep_theta.new_empty((0,))
        return empty, empty.clone(), empty.clone()

    histories: list[Tensor] = []
    intervals: list[Tensor] = []
    durations: list[Tensor] = []
    target_types: list[Tensor] = []
    event_windows: list[Tensor] = []
    event_counts: list[int] = []
    for window_index, window in enumerate(windows):
        history, interval, duration, types = _window_target_tensors(
            replace(window), hawkes_ll, keep_theta
        )
        count = int(types.numel())
        if count <= 0:
            raise ValueError("topology-prune replay windows must contain events")
        histories.append(history)
        intervals.append(interval)
        durations.append(duration)
        target_types.append(types)
        event_windows.append(torch.full(
            (count,),
            window_index,
            device=keep_theta.device,
            dtype=torch.long,
        ))
        event_counts.append(count)

    history = torch.cat(histories, dim=0)
    interval = torch.cat(intervals, dim=0)
    duration = torch.cat(durations, dim=0)
    types = torch.cat(target_types, dim=0)
    event_window = torch.cat(event_windows, dim=0)
    counts = keep_theta.new_tensor(event_counts).clamp_min(1.0)

    theta_models = torch.stack((keep_theta, prune_theta), dim=1)
    selected_theta = theta_models.index_select(0, event_window)
    raw_mu = selected_theta[..., :D]
    raw_W = selected_theta[..., D:].reshape(-1, 2, D, D, M)
    mu = F.softplus(raw_mu)
    W = F.softplus(raw_W)
    raw_intensity = mu + torch.einsum(
        "eqdsm,esm->eqd", W, history
    )
    # Preserve the two safeguards used by the original scalar helpers:
    # replay NLL always clamps at 1e-8, while TPP divergence uses caller eps.
    nll_intensity = raw_intensity.clamp_min(1e-8)
    observed = nll_intensity.gather(
        2,
        types[:, None, None].expand(-1, 2, 1),
    ).squeeze(-1)
    event_nll = (
        -observed.log()
        + mu.sum(dim=-1) * duration.unsqueeze(-1)
        + torch.einsum("eqdsm,esm->eq", W, interval)
    )
    window_nll = keep_theta.new_zeros((replay_count, 2))
    window_nll.index_add_(0, event_window, event_nll)
    if normalize_by_events:
        window_nll = window_nll / counts.unsqueeze(-1)

    tpp_intensity = raw_intensity.clamp_min(eps)
    keep_intensity = tpp_intensity[:, 0].detach()
    prune_intensity = tpp_intensity[:, 1]
    event_divergence = duration.unsqueeze(-1) * (
        keep_intensity
        * (keep_intensity.log() - prune_intensity.log())
        - keep_intensity
        + prune_intensity
    )
    window_divergence = keep_theta.new_zeros(replay_count)
    window_divergence.index_add_(
        0,
        event_window,
        event_divergence.sum(dim=-1),
    )
    if normalize_by_events:
        window_divergence = window_divergence / counts
    return window_nll[:, 0], window_nll[:, 1], window_divergence


def _deferred_proposal(
    tree,
    parent_id: str,
    reason: str,
    replay_size: int,
) -> TopologyPruneProposal:
    parent = tree.nodes[parent_id]
    return TopologyPruneProposal(
        parent_id=parent_id,
        child_ids=(parent.left, parent.right),
        snapshot_signature=collapse_snapshot_signature(tree, parent_id),
        eligible=False,
        reason=reason,
        keep_mode="unavailable",
        replay_size=replay_size,
        effective_replay_size=0.0,
        branch_balance=0.0,
        keep_nll=0.0,
        prune_nll=0.0,
        predictive_damage=0.0,
        near_zero_damage=False,
        uncertainty_margin=0.0,
        tpp_divergence=0.0,
        retention_cost=float("inf"),
        complexity_saving=1.0,
        prior_probability=0.0,
        prune_gain=float("-inf"),
        prune_probability=0.0,
    )


@torch.no_grad()
def evaluate_topology_prune_candidate(
    tree,
    parent_id: str,
    hawkes_ll: HawkesFamily,
    *,
    lambda_T: float,
    embedding_fn: Optional[ReplayEmbeddingFn] = None,
    embedding_batch_fn: Optional[ReplayEmbeddingBatchFn] = None,
    production_state: Optional[
        tuple[Tensor, Tensor, Tensor, Tensor, str]
    ] = None,
    min_replay: int = 8,
    min_effective_replay: float = 4.0,
    min_branch_replay: int = 1,
    max_replay: Optional[int] = 32,
    stale_weight: float = 0.2,
    uncertainty_kappa: float = 1.0,
    dynamics_weight: float = 0.1,
    prior_bias: float = 0.0,
    prior_semantic_weight: float = 1.0,
    prior_balance_weight: float = 1.0,
    prior_evidence_scale: float = 8.0,
    semantic_scale: float = 1.0,
    gate_beta: float = 0.1,
    commit_probability: float = 0.5,
    min_gain: float = 0.0,
    near_zero_damage_threshold: float = 1e-3,
    eps: float = 1e-8,
) -> TopologyPruneProposal:
    """Evaluate one frozen KEEP/PRUNE counterfactual without mutation."""
    if (
        lambda_T < 0.0
        or stale_weight < 0.0
        or dynamics_weight < 0.0
        or uncertainty_kappa < 0.0
    ):
        raise ValueError("topology-prune weights must be non-negative")
    if min_replay <= 0 or min_effective_replay <= 0.0:
        raise ValueError("topology-prune replay thresholds must be positive")
    if min_branch_replay < 0:
        raise ValueError("min_branch_replay must be non-negative")
    if max_replay is not None and max_replay <= 0:
        raise ValueError("max_replay must be positive when provided")
    if (
        gate_beta <= 0.0
        or semantic_scale <= 0.0
        or prior_evidence_scale <= 0.0
    ):
        raise ValueError(
            "gate_beta, semantic_scale, and prior_evidence_scale must be positive"
        )
    if prior_balance_weight < 0.0:
        raise ValueError("prior_balance_weight must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if near_zero_damage_threshold < 0.0:
        raise ValueError("near_zero_damage_threshold must be non-negative")
    if not 0.0 < commit_probability < 1.0:
        raise ValueError("commit_probability must lie in (0, 1)")
    scalar_controls = (
        lambda_T,
        stale_weight,
        uncertainty_kappa,
        dynamics_weight,
        prior_bias,
        prior_semantic_weight,
        prior_balance_weight,
        prior_evidence_scale,
        semantic_scale,
        gate_beta,
        commit_probability,
        min_gain,
        near_zero_damage_threshold,
        eps,
    )
    if not all(math.isfinite(float(value)) for value in scalar_controls):
        raise ValueError("topology-prune controls must be finite")
    if parent_id not in candidate_prune_parents(tree):
        raise ValueError("parent is not a complete leaf-refinement candidate")

    rows = _collect_replay(
        tree,
        parent_id,
        max_replay=max_replay,
        stale_weight=stale_weight,
    )
    if len(rows) < min_replay:
        return _deferred_proposal(tree, parent_id, "insufficient_replay", len(rows))
    parent = tree.nodes[parent_id]
    branch_counts = [
        sum(row.source_node == child_id for row in rows)
        for child_id in (parent.left, parent.right)
    ]
    if min(branch_counts) < min_branch_replay:
        return _deferred_proposal(
            tree, parent_id, "insufficient_branch_support", len(rows)
        )

    if production_state is not None:
        keep_theta, queries, other_theta, region_mass, keep_mode = (
            production_state
        )
        if keep_theta.size(0) != len(rows):
            raise ValueError(
                "precomputed topology-prune state does not align with replay"
            )
    elif embedding_fn is None:
        keep_theta, queries, other_theta, region_mass, keep_mode = (
            _fallback_keep_state(tree, parent_id, rows)
        )
    else:
        keep_theta, queries, other_theta, region_mass, keep_mode = (
            _production_region_terms(
                tree,
                parent_id,
                rows,
                hawkes_ll.decays,
                embedding_fn,
                embedding_batch_fn,
            )
        )

    weights = _normalized_weights(rows, keep_theta)
    effective_replay = float(
        (1.0 / weights.square().sum().clamp_min(eps)).detach().cpu()
    )
    if effective_replay < min_effective_replay:
        return _deferred_proposal(
            tree, parent_id, "insufficient_effective_replay", len(rows)
        )

    virtual = _build_virtual_bank(tree, parent_id)
    virtual_delta, _ = _retrieve_virtual_loo(tree, queries, rows, virtual)
    ancestor_delta = _path_delta_batch(
        tree, queries, parent_id, include_node=False
    )
    collapsed_region_theta = (
        tree.semantic_theta(parent_id).detach().unsqueeze(0)
        + ancestor_delta
        + virtual_delta
    )
    prune_theta = other_theta + region_mass.unsqueeze(-1) * collapsed_region_theta

    keep_loss, prune_loss, tpp_terms = _batched_topology_prune_terms(
        [row.window for row in rows],
        keep_theta,
        prune_theta,
        hawkes_ll,
        eps,
    )
    damage = prune_loss - keep_loss
    predictive_damage = (weights * damage).sum()
    near_zero_tensor = predictive_damage.abs().le(
        float(near_zero_damage_threshold)
    )
    denominator = (1.0 - weights.square().sum()).clamp_min(eps)
    variance = (
        weights * (damage - predictive_damage).square()
    ).sum() / denominator
    uncertainty = (
        float(uncertainty_kappa)
        * variance.clamp_min(0.0).sqrt()
        / math.sqrt(max(effective_replay, eps))
    )
    tpp_divergence = (weights * tpp_terms.clamp_min(0.0)).sum()
    retention = predictive_damage + uncertainty + float(dynamics_weight) * tpp_divergence

    left_theta = tree.semantic_theta(parent.left).detach()
    right_theta = tree.semantic_theta(parent.right).detach()
    parent_theta = tree.semantic_theta(parent_id).detach()
    specialization_distance = torch.linalg.vector_norm(
        (left_theta - parent_theta) - (right_theta - parent_theta)
    )
    similarity = torch.exp(
        -specialization_distance.square() / (2.0 * semantic_scale ** 2)
    )
    branch_total = sum(branch_counts)
    branch_balance = (
        4.0 * branch_counts[0] * branch_counts[1]
        / max(float(branch_total * branch_total), 1.0)
    )
    evidence_confidence = (
        1.0 - math.exp(-effective_replay / float(prior_evidence_scale))
    ) * branch_balance
    prior = torch.sigmoid(
        similarity * float(prior_semantic_weight)
        + float(prior_balance_weight) * evidence_confidence
        + float(prior_bias)
    ).clamp(eps, 1.0 - eps)
    complexity_saving = 1.0
    gain = keep_theta.new_tensor(float(lambda_T) * complexity_saving) - retention
    probability = torch.sigmoid(
        torch.logit(prior) + gain / float(gate_beta)
    )
    posterior_gain = gain + float(gate_beta) * torch.logit(prior)
    eligible_tensor = (
        torch.isfinite(retention) & torch.isfinite(probability)
    )
    # Historical hard thresholds are diagnostics only. The prior-adjusted
    # posterior gain enters the unified selector, whose explicit Null action
    # remains the final commit boundary.
    del min_gain, commit_probability
    locally_positive_tensor = eligible_tensor & gain.gt(0.0)
    scalar_values = torch.stack((
        (weights * keep_loss).sum(),
        (weights * prune_loss).sum(),
        predictive_damage,
        uncertainty,
        tpp_divergence,
        retention,
        prior,
        gain,
        probability,
        posterior_gain,
        near_zero_tensor.to(keep_theta.dtype),
        eligible_tensor.to(keep_theta.dtype),
        locally_positive_tensor.to(keep_theta.dtype),
    )).detach().cpu().tolist()
    (
        keep_nll_value,
        prune_nll_value,
        predictive_damage_value,
        uncertainty_value,
        tpp_divergence_value,
        retention_value,
        prior_value,
        gain_value,
        probability_value,
        posterior_gain_value,
        near_zero_value,
        eligible_value,
        locally_positive_value,
    ) = scalar_values
    near_zero_damage = bool(near_zero_value)
    eligible = bool(eligible_value)
    locally_positive = bool(locally_positive_value)
    return TopologyPruneProposal(
        parent_id=parent_id,
        child_ids=(parent.left, parent.right),
        snapshot_signature=collapse_snapshot_signature(tree, parent_id),
        eligible=eligible,
        reason="ready" if eligible else "non_finite_score",
        keep_mode=keep_mode,
        replay_size=len(rows),
        effective_replay_size=effective_replay,
        branch_balance=branch_balance,
        keep_nll=float(keep_nll_value),
        prune_nll=float(prune_nll_value),
        predictive_damage=float(predictive_damage_value),
        near_zero_damage=near_zero_damage,
        uncertainty_margin=float(uncertainty_value),
        tpp_divergence=float(tpp_divergence_value),
        retention_cost=float(retention_value),
        complexity_saving=complexity_saving,
        prior_probability=float(prior_value),
        prune_gain=float(gain_value),
        prune_probability=float(probability_value),
        posterior_gain=float(posterior_gain_value),
        locally_positive=locally_positive,
    )


@torch.no_grad()
def commit_topology_prune(
    tree,
    parent_id: str,
    *,
    snapshot_signature: Optional[tuple] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """Demote both child specializations while keeping parent semantics fixed."""
    target_parent_theta = tree.semantic_theta(parent_id).detach()
    return contract_leaf_pair(
        tree,
        parent_id,
        target_parent_theta=target_parent_theta,
        snapshot_signature=snapshot_signature,
        optimizer=optimizer,
        preserve_memory=True,
        reconciliation_policy="prune",
    )


@torch.no_grad()
def evaluate_topology_prune(
    tree,
    hawkes_ll: HawkesFamily,
    *,
    lambda_T: float,
    embedding_fn: Optional[ReplayEmbeddingFn] = None,
    **kwargs: Any,
) -> tuple[Dict[str, TopologyPruneProposal], Dict[str, float]]:
    parent_ids = candidate_prune_parents(tree)
    production_states: Dict[
        str, tuple[Tensor, Tensor, Tensor, Tensor, str]
    ] = {}
    embedding_fn = kwargs.get("embedding_fn", embedding_fn)
    embedding_batch_fn = kwargs.get("embedding_batch_fn")
    if embedding_fn is not None and embedding_batch_fn is not None:
        rows_by_parent: Dict[str, list[_ReplayRow]] = {}
        flat_rows: list[_ReplayRow] = []
        offsets: Dict[str, tuple[int, int]] = {}
        min_replay = int(kwargs.get("min_replay", 8))
        min_branch_replay = int(kwargs.get("min_branch_replay", 1))
        for parent_id in parent_ids:
            rows = _collect_replay(
                tree,
                parent_id,
                max_replay=kwargs.get("max_replay", 32),
                stale_weight=float(kwargs.get("stale_weight", 0.2)),
            )
            parent = tree.nodes[parent_id]
            branch_counts = [
                sum(row.source_node == child_id for row in rows)
                for child_id in (parent.left, parent.right)
            ]
            if (
                len(rows) < min_replay
                or min(branch_counts) < min_branch_replay
            ):
                continue
            start = len(flat_rows)
            flat_rows.extend(rows)
            offsets[parent_id] = (start, len(flat_rows))
            rows_by_parent[parent_id] = rows

        if flat_rows:
            embeddings = embedding_batch_fn([
                row.window for row in flat_rows
            ]).detach()
            if embeddings.shape != (len(flat_rows), tree.z_dim):
                raise ValueError(
                    "batched replay embeddings must have shape "
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
            all_keep_theta = output["effective_params"].theta.detach()
            all_queries = output["memory_query"].detach()
            all_frontier_mass = output["frontier_mass"].detach()
            all_frontier_theta = output["frontier_theta"].detach()
            all_frontier_ids = output["frontier_node_ids"]
            for parent_id, (start, end) in offsets.items():
                production_states[parent_id] = _region_terms_from_frontier(
                    tree,
                    parent_id,
                    all_keep_theta[start:end],
                    all_queries[start:end],
                    all_frontier_mass[start:end],
                    all_frontier_theta[start:end],
                    all_frontier_ids[start:end],
                )

    proposals = {}
    for parent_id in parent_ids:
        proposals[parent_id] = evaluate_topology_prune_candidate(
            tree,
            parent_id,
            hawkes_ll,
            lambda_T=lambda_T,
            embedding_fn=embedding_fn,
            production_state=production_states.get(parent_id),
            **kwargs,
        )
    eligible = [proposal for proposal in proposals.values() if proposal.eligible]
    expected_prunes = sum(proposal.prune_probability for proposal in eligible)
    current_complexity = tree_complexity(tree)
    expected_complexity = current_complexity - sum(
        proposal.prune_probability * proposal.complexity_saving
        for proposal in eligible
    )
    metrics = {
        "candidate_count": float(len(proposals)),
        "eligible_count": float(len(eligible)),
        "ready_count": float(sum(proposal.persistence_ok for proposal in eligible)),
        "near_zero_count": float(sum(
            proposal.near_zero_damage for proposal in eligible
        )),
        "replay_windows": float(sum(proposal.replay_size for proposal in proposals.values())),
        "expected_prunes": float(expected_prunes),
        "current_complexity": float(current_complexity),
        "expected_complexity": float(expected_complexity),
        "mean_prune_probability": (
            float(expected_prunes / len(eligible)) if eligible else 0.0
        ),
        "mean_retention_cost": (
            float(sum(proposal.retention_cost for proposal in eligible) / len(eligible))
            if eligible else 0.0
        ),
    }
    return proposals, metrics


@torch.no_grad()
def apply_prune_persistence(
    tree,
    proposals: Mapping[str, TopologyPruneProposal],
    *,
    patience: int,
    allow_candidate: bool,
    near_zero_confirmations: int = 2,
) -> Dict[str, TopologyPruneProposal]:
    if patience <= 0:
        raise ValueError("topology-prune patience must be positive")
    if near_zero_confirmations <= 0:
        raise ValueError("near_zero_confirmations must be positive")
    if not hasattr(tree, "topology_prune_near_zero_streak"):
        tree.topology_prune_near_zero_streak = {}
    active = set(candidate_prune_parents(tree))
    for parent_id in set(tree.topology_prune_streak).difference(active):
        tree.topology_prune_streak.pop(parent_id, None)
    for parent_id in set(
        tree.topology_prune_near_zero_streak
    ).difference(active):
        tree.topology_prune_near_zero_streak.pop(parent_id, None)
    result = {}
    for parent_id, proposal in proposals.items():
        near_zero = proposal.eligible and proposal.near_zero_damage
        evidence_ready = proposal.eligible
        if near_zero:
            # First near-zero observation is deliberately interpreted as
            # unresolved replay scarcity. A second consecutive observation
            # confirms that the refinement has no detectable predictive
            # value and may override the ordinary probability threshold.
            near_zero_streak = (
                tree.topology_prune_near_zero_streak.get(parent_id, 0) + 1
            )
            tree.topology_prune_near_zero_streak[parent_id] = near_zero_streak
            tree.topology_prune_streak[parent_id] = 0
            confirmed = near_zero_streak >= near_zero_confirmations
            persistence_ok = bool(allow_candidate and confirmed)
            reason = (
                "near_zero_confirmed"
                if persistence_ok
                else "near_zero_confirmed_shadow"
                if confirmed
                else "near_zero_collect_replay"
            )
            persistence = near_zero_streak
        else:
            tree.topology_prune_near_zero_streak[parent_id] = 0
            streak = (
                tree.topology_prune_streak.get(parent_id, 0) + 1
                if evidence_ready
                else 0
            )
            tree.topology_prune_streak[parent_id] = streak
            persistence_ok = bool(
                allow_candidate and evidence_ready and streak >= patience
            )
            reason = (
                "ready_to_compare"
                if persistence_ok
                else "warmup_shadow"
                if not allow_candidate and evidence_ready
                else "persistence"
                if evidence_ready
                else proposal.reason
            )
            persistence = streak
        result[parent_id] = replace(
            proposal,
            persistence_ok=persistence_ok,
            persistence=persistence,
            reason=reason,
        )
    return result


__all__ = [
    "TopologyPruneProposal",
    "apply_prune_persistence",
    "candidate_prune_parents",
    "commit_topology_prune",
    "evaluate_topology_prune",
    "evaluate_topology_prune_candidate",
    "tree_complexity",
    "update_leaf_mass",
]
