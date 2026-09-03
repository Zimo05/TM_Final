"""Transactional commit of a pre-selected Deep-Sleep topology action."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor

from HawkesBackbone import HawkesFamily
from Sleep.Merge import commit_merge, leaf_sibling_pairs, promote_shared_memories
from Sleep.Split import commit_split
from Sleep.TopologyPrune import (
    TopologyPruneProposal,
    commit_topology_prune,
    update_leaf_mass,
)
from Sleep.UnifiedTopology import (
    TopologyActionKind,
    UnifiedTopologySelection,
)


def _logged_scalar(value: Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().cpu())


def _cleanup_external_state(tree, controllers: Iterable[Any]) -> None:
    active_leaves = set(tree.get_leaf_ids())
    for controller in controllers:
        split_queues = getattr(controller, "split_queues", None)
        if isinstance(split_queues, dict):
            for node_id in list(split_queues):
                if node_id not in active_leaves:
                    del split_queues[node_id]


@torch.no_grad()
def run_sleep_cycle(
    tree,
    responsibilities: Mapping[str, Tensor] | Tensor,
    decays: Tensor,
    *,
    hawkes_ll: Optional[HawkesFamily] = None,
    selection: UnifiedTopologySelection,
    optimizer: Optional[torch.optim.Optimizer] = None,
    controllers: Iterable[Any] = (),
    usage_decay: float = 0.95,
    effective_usage_threshold: float = 0.0,
    leaf_mass_ema_decay: float = 0.95,
    promotion_kwargs: Optional[Mapping[str, Any]] = None,
    promote_when_structure_unchanged: bool = True,
    split_memory_hard_threshold: float = 0.0,
    split_init_steps: int = 30,
    split_init_lr: float = 1e-2,
    allow_topology_prune: bool = True,
    statistics_prepared: bool = False,
    protected_leaf_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    """Commit at most one topology edit selected on a frozen snapshot.

    Action preference is fully resolved before this function is called. The
    coordinator only validates and commits the selected Split, Merge,
    complete Topology Prune, or Null action. Sleep-time memory deletion and
    asymmetric low-mass leaf pruning have been removed.
    """
    if selection is None:
        raise ValueError(
            "Deep Sleep requires UnifiedTopologySelection"
        )
    promotion_kwargs = {} if promotion_kwargs is None else dict(promotion_kwargs)
    protected = set(protected_leaf_ids)

    leaf_snapshot = list(tree.leaf_ids)
    pair_snapshot = leaf_sibling_pairs(tree, leaf_snapshot)
    if statistics_prepared:
        mass_ema = {
            leaf_id: tree.mass_ema[leaf_id]
            for leaf_id in leaf_snapshot
            if leaf_id in tree.mass_ema
        }
    else:
        tree.episodic_memory.materialize_all_ages()
        tree.episodic_memory.consolidate_sleep_cycle(
            usage_decay=usage_decay,
            effective_usage_threshold=effective_usage_threshold,
        )
        mass_ema = update_leaf_mass(
            tree,
            responsibilities,
            ema_decay=leaf_mass_ema_decay,
        )

    selected = None if selection.is_null else selection.selected
    physical_rejection: Optional[Dict[str, Any]] = None
    if selected is not None:
        conservative_gain = _logged_scalar(selected.conservative_gain)
        physical_valid = bool(
            selected.eligible
            and selected.ready
            and math.isfinite(conservative_gain)
            and conservative_gain > 0.0
        )
        if not physical_valid:
            # Defense in depth: even a stale checkpoint or externally forged
            # selection cannot bypass the non-learnable structural safety
            # boundary enforced by UnifiedTopologySelector.
            physical_rejection = {
                "action_id": selected.action_id,
                "kind": selected.kind.value,
                "conservative_gain": conservative_gain,
                "eligible": bool(selected.eligible),
                "ready": bool(selected.ready),
                "reason": "physical_gain_must_be_strictly_positive",
            }
            selected = None
    actions: list[Dict[str, Any]] = []
    mutated_memory_nodes: set[str] = set()
    claimed: set[str] = set()

    if selected is not None:
        if not selected.eligible or not selected.ready:
            raise ValueError("selected topology action is not eligible and ready")
        if selected.kind is TopologyActionKind.SPLIT:
            leaf_id = selected.target
            if leaf_id is None or leaf_id not in tree.nodes or not tree.nodes[leaf_id].is_leaf:
                raise RuntimeError("selected Split target is stale")
            split_module, split_output = selected.payload
            children = commit_split(
                tree=tree,
                leaf_id=leaf_id,
                split_module=split_module,
                split_output=split_output,
                optimizer=optimizer,
                memory_hard_threshold=split_memory_hard_threshold,
                init_steps=split_init_steps,
                init_lr=split_init_lr,
                authorized=True,
            )
            claimed.add(leaf_id)
            mutated_memory_nodes.update((leaf_id, *children))
            actions.append({
                "action": "split",
                "node": leaf_id,
                "children": children,
                "action_id": selected.action_id,
                "conservative_gain": _logged_scalar(
                    selected.conservative_gain
                ),
            })
        elif selected.kind is TopologyActionKind.MERGE:
            payload = selected.payload
            if not isinstance(payload, Mapping):
                raise TypeError("Merge payload has the wrong type")
            parent_id = payload["parent_id"]
            node_a, node_b = payload["child_ids"]
            commit = commit_merge(
                tree,
                node_a,
                node_b,
                target_parent_theta=payload.get("target_parent_theta"),
                child_weights=payload.get("child_weights"),
                snapshot_signature=payload["snapshot_signature"],
                optimizer=optimizer,
                return_result=True,
            )
            claimed.update((parent_id, node_a, node_b))
            mutated_memory_nodes.update((parent_id, node_a, node_b))
            actions.append({
                "action": "merge",
                "parent": parent_id,
                "nodes": (node_a, node_b),
                "action_id": selected.action_id,
                "conservative_gain": _logged_scalar(
                    selected.conservative_gain
                ),
                "rebased_rows": commit.rebased_rows,
                "overflow_rows": commit.overflow_rows,
            })
        elif selected.kind is TopologyActionKind.TOPOLOGY_PRUNE:
            if not allow_topology_prune:
                raise ValueError("Topology Prune is disabled for this cycle")
            proposal = selected.payload
            if not isinstance(proposal, TopologyPruneProposal):
                raise TypeError("Topology Prune payload has the wrong type")
            parent_id = proposal.parent_id
            node_a, node_b = proposal.child_ids
            commit = commit_topology_prune(
                tree,
                parent_id,
                snapshot_signature=proposal.snapshot_signature,
                optimizer=optimizer,
            )
            claimed.update((parent_id, node_a, node_b))
            mutated_memory_nodes.update((parent_id, node_a, node_b))
            actions.append({
                "action": "topology_prune",
                "parent": parent_id,
                "nodes": (node_a, node_b),
                "action_id": selected.action_id,
                "conservative_gain": _logged_scalar(
                    selected.conservative_gain
                ),
                "decision_reason": proposal.reason,
                "rebased_rows": commit.rebased_rows,
                "overflow_rows": commit.overflow_rows,
            })
        else:
            raise ValueError(f"unsupported topology action: {selected.kind}")

    # Promotion is reconciliation, not a competing topology action. Only
    # sibling regions untouched by the selected action are considered.
    if promote_when_structure_unchanged:
        for node_a, node_b in pair_snapshot:
            if node_a in protected or node_b in protected:
                continue
            if node_a in claimed or node_b in claimed:
                continue
            if node_a not in tree.nodes or node_b not in tree.nodes:
                continue
            parent_id = tree.nodes[node_a].parent
            if parent_id is None:
                continue
            records = promote_shared_memories(
                tree=tree,
                parent_id=parent_id,
                decays=decays,
                hawkes_ll=hawkes_ll,
                **promotion_kwargs,
            )
            if records:
                mutated_memory_nodes.add(parent_id)
                mutated_memory_nodes.update(record.source_node for record in records)
                actions.append({
                    "action": "promotion",
                    "parent": parent_id,
                    "count": len(records),
                })

    _cleanup_external_state(tree, controllers)
    return {
        "leaf_snapshot": leaf_snapshot,
        "leaf_mass_ema": mass_ema,
        "actions": actions,
        "memory": {},
        "mutated_memory_nodes": sorted(mutated_memory_nodes),
        "leaf_ids": list(tree.leaf_ids),
        "selected_action_id": (
            TopologyActionKind.NULL.value
            if selected is None else selected.action_id
        ),
        "physical_rejection": physical_rejection,
        "topology_prune_enabled": bool(allow_topology_prune),
    }


__all__ = ["run_sleep_cycle"]
