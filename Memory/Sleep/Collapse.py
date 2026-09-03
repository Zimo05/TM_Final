"""Atomic contraction machinery for a complete leaf refinement.

The transaction is intentionally agnostic to the action policy.  Merge may
replace the parent semantic parameters with a fused child target, while
Topology Prune keeps the existing parent target.  In both cases every replay
row is rebased so its effective Hawkes parameters are preserved exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import Tensor

from MemoryResiduals.MemoryBank import MemoryBank


# Keep topology-created banks on the same adaptive two-radius and Dual
# Identity configuration as the source bank.  Collapse can run on banks loaded
# from older checkpoints, so every field has a conservative constructor
# fallback rather than assuming the attribute was serialized.
_PROTOTYPE_POLICY_DEFAULTS = {
    "duplicate_threshold": 0.98,
    "mode_threshold": 0.90,
    "mode_capacity": 12,
    "ema_beta_min": 0.01,
    "ema_beta_max": 0.25,
    "retention_support_weight": 1.0,
    "retention_usage_weight": 0.5,
    "retention_stale_weight": 1.0,
    "retention_age_weight": 0.1,
    "adaptive_history_size": 64,
    "adaptive_min_samples": 16,
    "duplicate_quantile": 0.85,
    "mode_quantile": 0.975,
    "radius_margin": 1e-3,
    "gain_quantile": 0.95,
    "gain_ema_decay": 0.8,
    "gain_confirmation_min_count": 2,
    "gain_floor": 0.0,
    "context_alias_capacity": 3,
}


def _prototype_policy_from_bank(bank: MemoryBank) -> dict:
    return {
        name: getattr(bank, name, default)
        for name, default in _PROTOTYPE_POLICY_DEFAULTS.items()
    }


@dataclass(frozen=True)
class CollapseCommitResult:
    parent_id: str
    child_ids: tuple[str, str]
    rebased_rows: int
    total_rows: int
    overflow_rows: int
    snapshot_valid: bool


def _tensor_signature(tensor: Tensor) -> tuple[int, int, tuple[int, ...]]:
    return id(tensor), int(tensor._version), tuple(tensor.shape)


def _optional_tensor_signature(tensor: Optional[Tensor]) -> tuple | None:
    return None if tensor is None else _tensor_signature(tensor)


def _window_signature(window) -> tuple | None:
    if window is None:
        return None
    return (
        id(window),
        window.node_id,
        int(window.start_idx),
        int(window.end_idx),
        bool(window.has_full_history),
        _tensor_signature(window.times),
        _tensor_signature(window.types),
        _optional_tensor_signature(getattr(window, "T", None)),
        _optional_tensor_signature(
            getattr(window, "event_time_features", None)
        ),
        _optional_tensor_signature(
            getattr(window, "hawkes_history_stats", None)
        ),
        _optional_tensor_signature(
            getattr(window, "hawkes_interval_stats", None)
        ),
        getattr(window, "hawkes_cache_signature", None),
    )


def collapse_snapshot_signature(tree, parent_id: str) -> tuple:
    """Return an in-process signature that detects stale collapse proposals."""
    if parent_id not in tree.nodes:
        raise KeyError(f"Unknown parent node: {parent_id}")
    parent = tree.nodes[parent_id]
    if parent.left is None or parent.right is None:
        raise ValueError("collapse parent must have exactly two children")
    child_ids = (parent.left, parent.right)
    if any(child_id not in tree.nodes for child_id in child_ids):
        raise RuntimeError("collapse topology references a missing child")

    node_rows = []
    for node_id in (parent_id, *child_ids):
        node = tree.nodes[node_id]
        bank = tree.episodic_memory.banks.get(node_id)
        bank_signature = None
        if bank is not None:
            bank._ensure_prototype_state()
            bank_signature = (
                id(bank),
                len(bank),
                int(bank._age_reference_clock),
                _tensor_signature(bank.keys),
                _tensor_signature(bank.context_keys),
                _tensor_signature(bank.context_valid),
                _tensor_signature(bank.context_support),
                _tensor_signature(bank.deltas),
                _tensor_signature(bank.write_quality),
                _tensor_signature(bank.queue_weight),
                _tensor_signature(bank.usage),
                _tensor_signature(bank.cycle_usage),
                _tensor_signature(bank.stale_cycles),
                _tensor_signature(bank.age),
                tuple(_window_signature(window) for window in bank.windows),
            )
        node_rows.append((
            node_id,
            node.parent,
            node.left,
            node.right,
            node.depth,
            _tensor_signature(tree.node_emb[node_id]),
            _tensor_signature(tree.semantic_offset[node_id]),
            bank_signature,
        ))
    # Include shared tree-owned parameters that affect the frozen KEEP or
    # PRUNE counterfactual (hypernetwork, routing, and retriever). Candidate-
    # local dynamic parameters are already represented in ``node_rows``.
    # Excluding unrelated node parameters lets disjoint proposals from the
    # same frozen snapshot commit in one transaction without making each
    # other spuriously stale.
    shared_parameter_rows = tuple(
        (name, _tensor_signature(parameter))
        for name, parameter in tree.named_parameters()
        if not name.startswith(("node_emb.", "semantic_offset."))
    )
    return (
        parent_id,
        child_ids,
        int(tree.episodic_memory._age_clock),
        tuple(node_rows),
        shared_parameter_rows,
    )


def _remove_parameters_from_optimizer(
    optimizer: torch.optim.Optimizer,
    removed_parameters: Iterable[torch.nn.Parameter],
) -> None:
    removed_ids = {id(parameter) for parameter in removed_parameters}
    for group in optimizer.param_groups:
        group["params"] = [
            parameter
            for parameter in group["params"]
            if id(parameter) not in removed_ids
        ]
    for parameter in list(optimizer.state):
        if id(parameter) in removed_ids:
            del optimizer.state[parameter]


def _empty_replacement_bank(
    parent_bank: MemoryBank,
    capacity: int,
    *,
    age_reference_clock: int,
) -> MemoryBank:
    replacement = MemoryBank(
        device=str(parent_bank.device),
        key_dim=parent_bank.key_dim,
        param_dim=parent_bank.param_dim,
        capacity=max(int(capacity), 1),
        law_dim=int(parent_bank.key_dim),
    )
    # Preserve storage dtype without adding a synthetic row.
    parent_bank._ensure_prototype_state()
    replacement.configure_prototype_policy(
        **_prototype_policy_from_bank(parent_bank)
    )
    replacement.keys = parent_bank.keys[:0]
    replacement.context_keys = parent_bank.context_keys[:0]
    replacement.context_valid = parent_bank.context_valid[:0]
    replacement.context_support = parent_bank.context_support[:0]
    replacement.deltas = parent_bank.deltas[:0]
    replacement.write_quality = parent_bank.write_quality[:0]
    replacement.queue_weight = parent_bank.queue_weight[:0]
    replacement.usage = parent_bank.usage[:0]
    replacement.cycle_usage = parent_bank.cycle_usage[:0]
    replacement.stale_cycles = parent_bank.stale_cycles[:0]
    replacement.age = parent_bank.age[:0]
    replacement._age_reference_clock = int(age_reference_clock)
    return replacement


def _transient_empty_bank(memory) -> MemoryBank:
    """Construct an unstored empty bank for rollback-safe prebuilding."""
    bank = MemoryBank(
        device=str(memory.device),
        key_dim=memory.key_dim,
        param_dim=memory.param_dim,
        capacity=memory.capacity_per_node,
        law_dim=int(memory.key_dim),
    )
    # A transient bank participates in the same collapse transaction as the
    # tree's persistent banks.  Carry the tree-level policy into it so a
    # missing/empty parent or child does not silently fall back to global
    # thresholds or the default alias width.
    memory_policy = getattr(memory, "_prototype_policy", None)
    policy = _prototype_policy_from_bank(bank)
    if isinstance(memory_policy, dict):
        policy.update({
            name: memory_policy[name]
            for name in _PROTOTYPE_POLICY_DEFAULTS
            if name in memory_policy
        })
    bank.configure_prototype_policy(**policy)
    bank._age_reference_clock = memory._age_clock
    return bank


@torch.no_grad()
def rebase_memory_to_new_leaf(
    delta_theta: Tensor,
    old_theta: Tensor,
    new_theta: Tensor,
) -> Tensor:
    """Preserve ``old_theta + delta`` under a new semantic reference."""
    if old_theta.ndim != 1 or new_theta.ndim != 1:
        raise ValueError("old_theta and new_theta must be one-dimensional")
    if old_theta.shape != new_theta.shape:
        raise ValueError("old_theta and new_theta must have the same shape")
    if delta_theta.ndim == 0 or delta_theta.shape[-1] != old_theta.numel():
        raise ValueError(
            "delta_theta final dimension must match the semantic parameter size"
        )
    old_theta = old_theta.to(device=delta_theta.device, dtype=delta_theta.dtype)
    new_theta = new_theta.to(device=delta_theta.device, dtype=delta_theta.dtype)
    return (old_theta + delta_theta - new_theta).detach()


@torch.no_grad()
def contract_leaf_pair(
    tree,
    parent_id: str,
    *,
    target_parent_theta: Tensor,
    snapshot_signature: Optional[tuple] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    preserve_memory: bool = True,
    reconciliation_policy: str = "prune",
) -> CollapseCommitResult:
    """Contract ``parent -> (left, right)`` to ``target_parent_theta``.

    The full replacement bank is built before topology mutation. Child rows
    and pre-existing parent rows are rebased so their effective raw Hawkes
    parameters are exactly preserved. Capacity overflow is retained for later
    memory reconciliation.
    """
    if not preserve_memory:
        raise ValueError(
            "contract_leaf_pair only implements memory-preserving contraction"
        )
    if reconciliation_policy not in {"merge", "prune"}:
        raise ValueError(
            "reconciliation_policy must be 'merge' or 'prune'"
        )
    current_signature = collapse_snapshot_signature(tree, parent_id)
    if snapshot_signature is not None and current_signature != snapshot_signature:
        raise RuntimeError("collapse proposal snapshot is stale")

    parent = tree.nodes[parent_id]
    child_ids = (parent.left, parent.right)
    if any(child_id is None for child_id in child_ids):
        raise ValueError("collapse parent must have two children")
    left_id, right_id = child_ids
    if not tree.nodes[left_id].is_leaf or not tree.nodes[right_id].is_leaf:
        raise ValueError("collapse currently requires two leaf children")

    memory = tree.episodic_memory
    parent_bank = memory.banks.get(parent_id)
    if parent_bank is None:
        parent_bank = _transient_empty_bank(memory)
    child_banks = {}
    for child_id in child_ids:
        child_bank = memory.banks.get(child_id)
        child_banks[child_id] = (
            child_bank
            if child_bank is not None
            else _transient_empty_bank(memory)
        )
    parent_theta = tree.semantic_theta(parent_id).detach()
    target_parent_theta = torch.as_tensor(
        target_parent_theta,
        device=parent_theta.device,
        dtype=parent_theta.dtype,
    ).detach().reshape(-1)
    if target_parent_theta.shape != parent_theta.shape:
        raise ValueError(
            "target_parent_theta must match the parent semantic shape"
        )
    if not bool(torch.isfinite(target_parent_theta).all()):
        raise FloatingPointError("target_parent_theta contains NaN or Inf")
    child_theta = {
        child_id: tree.semantic_theta(child_id).detach()
        for child_id in child_ids
    }
    total_rows = len(parent_bank) + sum(
        len(bank) for bank in child_banks.values()
    )
    base_capacity = int(memory.capacity_per_node)
    replacement = _empty_replacement_bank(
        parent_bank,
        capacity=max(base_capacity, total_rows),
        age_reference_clock=memory._age_clock,
    )

    if len(parent_bank):
        parent_indices = torch.arange(
            len(parent_bank), device=parent_bank.device, dtype=torch.long
        )
        parent_rebased = rebase_memory_to_new_leaf(
            parent_bank.deltas,
            parent_theta,
            target_parent_theta,
        )
        replacement.append_from(
            parent_bank,
            parent_indices,
            deltas=parent_rebased,
            node_id=parent_id,
        )
        replacement.age[:] = parent_bank.effective_age(memory._age_clock)

    rebased_rows = 0
    for child_id in child_ids:
        bank = child_banks[child_id]
        if not len(bank):
            continue
        indices = torch.arange(len(bank), device=bank.device, dtype=torch.long)
        rebased = rebase_memory_to_new_leaf(
            bank.deltas,
            child_theta[child_id],
            target_parent_theta,
        )
        replacement.append_from(
            bank,
            indices,
            deltas=rebased,
            node_id=parent_id,
        )
        replacement.age[-len(bank):] = bank.effective_age(
            memory._age_clock
        )
        rebased_rows += len(bank)

    if len(replacement) != total_rows:
        raise RuntimeError("collapse replacement bank lost aligned rows")
    if len(replacement) and not bool(torch.isfinite(replacement.deltas).all()):
        raise FloatingPointError("collapse produced non-finite rebased residuals")

    # Commit begins only after the complete replacement has been validated.
    child_mass = sum(tree.mass_ema.get(child_id, 0.0) for child_id in child_ids)
    tree.set_semantic_theta(parent_id, target_parent_theta)
    if optimizer is not None and not torch.equal(
        target_parent_theta, parent_theta
    ):
        # Merge writes a new semantic reference outside optimizer.step().
        # Old momentum for the parent offset belongs to the pre-merge state.
        optimizer.state.pop(tree.semantic_offset[parent_id], None)
    memory.banks[parent_id] = replacement
    parent.left = None
    parent.right = None
    parent.split_queue.clear()

    removed_parameters = []
    for child_id in child_ids:
        removed_parameters.extend(
            [tree.node_emb[child_id], tree.semantic_offset[child_id]]
        )
        del tree.nodes[child_id]
        del tree.node_emb[child_id]
        del tree.semantic_offset[child_id]
        memory.banks.pop(child_id, None)
        tree.mass_ema.pop(child_id, None)
        tree.low_mass_streak.pop(child_id, None)
        tree.topology_prune_streak.pop(child_id, None)
        tree.topology_prune_near_zero_streak.pop(child_id, None)
        tree.memory_reconciliation.pop(child_id, None)

    if optimizer is not None:
        _remove_parameters_from_optimizer(optimizer, removed_parameters)

    if child_mass == 0.0:
        child_mass = tree.mass_ema.get(parent_id, 0.0)
    tree.mass_ema[parent_id] = child_mass
    tree.low_mass_streak[parent_id] = 0
    tree.topology_prune_streak.pop(parent_id, None)
    tree.topology_prune_near_zero_streak.pop(parent_id, None)
    overflow_rows = max(0, total_rows - base_capacity)
    if overflow_rows:
        tree.memory_reconciliation[parent_id] = {
            "policy": reconciliation_policy,
            # Merge has already promoted shared semantics and can reconcile
            # on the next Light cycle. Prune deliberately demotes child
            # specialization into episodic memory, so preserve it for one
            # additional Light cycle before compression.
            "delay_cycles": 0 if reconciliation_policy == "merge" else 1,
        }
    else:
        tree.memory_reconciliation.pop(parent_id, None)

    memory._packed_mirror_signature = None
    memory._packed_mirror = None
    tree.refresh_structure_buffers()
    return CollapseCommitResult(
        parent_id=parent_id,
        child_ids=(left_id, right_id),
        rebased_rows=rebased_rows,
        total_rows=total_rows,
        overflow_rows=overflow_rows,
        snapshot_valid=True,
    )


@torch.no_grad()
def collapse_leaf_pair(
    tree,
    parent_id: str,
    *,
    snapshot_signature: Optional[tuple] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    preserve_memory: bool = True,
) -> CollapseCommitResult:
    """Compatibility wrapper for parent-preserving Topology Prune semantics."""
    return contract_leaf_pair(
        tree,
        parent_id,
        target_parent_theta=tree.semantic_theta(parent_id).detach(),
        snapshot_signature=snapshot_signature,
        optimizer=optimizer,
        preserve_memory=preserve_memory,
        reconciliation_policy="prune",
    )


__all__ = [
    "CollapseCommitResult",
    "collapse_leaf_pair",
    "collapse_snapshot_signature",
    "contract_leaf_pair",
    "rebase_memory_to_new_leaf",
]
