"""Active-frontier routing, retrieval, and Hawkes parameter composition."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .configuration import FrontierRoutingConfig
from .prototype_store import NodePrototypeStore


@dataclass
class BranchDecision:
    """Diagnostics for one locally evaluated internal node."""

    node_id: str
    child_ids: tuple[str, str]
    semantic_score: Tensor
    data_score: Tensor
    log_prior: Tensor
    total_score: Tensor
    probability: Tensor
    entropy: Tensor
    priority: Optional[Tensor] = None


@dataclass
class FrontierSample:
    """One sample's ragged active frontier."""

    node_ids: tuple[str, ...]
    mass: Tensor
    visited_node_ids: tuple[str, ...]
    expanded_node_ids: tuple[str, ...]
    decisions: Dict[str, BranchDecision] = field(default_factory=dict)


@dataclass
class PackedFrontierBatch:
    """Fixed-width active-frontier state for a batch of causal prefixes.

    Posterior and training responsibility live on these slots. They are never
    projected onto leaves that did not participate in the computation.
    """

    node_indices: Tensor
    mass: Tensor
    mask: Tensor
    visited_indices: Tensor
    visited_mask: Tensor
    path_incidence: Tensor
    expanded_node_indices: Tensor
    expanded_child_indices: Tensor
    expanded_probability: Tensor
    expanded_semantic_score: Tensor
    expanded_mask: Tensor
    expansion_utility: Tensor

    def slice(self, start: int, end: int) -> "PackedFrontierBatch":
        """Return a view over a contiguous prefix-row interval."""
        return PackedFrontierBatch(**{
            field.name: getattr(self, field.name)[start:end]
            for field in fields(self)
        })

    def index_select(self, indices: Tensor) -> "PackedFrontierBatch":
        """Select arbitrary prefix rows without materializing Python samples."""
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("frontier row indices must be one-dimensional long")
        return PackedFrontierBatch(**{
            field.name: getattr(self, field.name).index_select(0, indices)
            for field in fields(self)
        })


@dataclass
class FrontierEffectiveParameters:
    """Batched effective Hawkes parameters in raw and constrained spaces."""

    theta: Tensor
    raw_mu: Tensor
    raw_W: Tensor
    mu: Tensor
    W: Tensor
    decays: Optional[Tensor] = None

    def select(self, index: int) -> "FrontierEffectiveParameters":
        return FrontierEffectiveParameters(
            theta=self.theta[index],
            raw_mu=self.raw_mu[index],
            raw_W=self.raw_W[index],
            mu=self.mu[index],
            W=self.W[index],
            decays=self.decays,
        )


@dataclass
class FrontierBatchOutput:
    """Complete result of one frontier-routed prediction."""

    samples: tuple[FrontierSample, ...]
    effective_params: FrontierEffectiveParameters
    query: Tensor
    semantic_theta: tuple[Tensor, ...]
    episodic_delta: tuple[Tensor, ...]
    frontier_theta: tuple[Tensor, ...]
    expanded_child_theta: Tensor
    memory_info: tuple[Mapping[str, Dict[str, Tensor]], ...]
    working_delta: Tensor
    frontier: PackedFrontierBatch
    frontier_mass: Tensor
    frontier_mask: Tensor
    frontier_node_indices: Tensor
    visited_node_indices: Tensor
    visited_node_mask: Tensor
    semantic_theta_packed: Tensor
    episodic_delta_packed: Tensor
    frontier_theta_packed: Tensor
    packed_memory_info: Mapping[str, Tensor]


@dataclass(frozen=True)
class FrontierStaticCache:
    """Parameter-dependent tables reusable while tree parameters are frozen."""

    topology_signature: tuple[str, ...]
    node_embedding_table: Tensor
    normalized_node_table: Tensor
    semantic_theta_table: Tensor


class FrontierRoutingRetrieval(nn.Module):
    """Budgeted coarse-to-fine adapter around the current ``HawkesTree``.

    The wrapped tree remains the owner of topology, semantic parameters,
    memory banks, query network, and sparse retriever. This module changes
    only the computational construction used by one prediction:

    1. start from ``{root}``;
    2. score only children of active internal frontier nodes;
    3. expand best-first until the frontier budget is reached;
    4. retrieve the union of the selected frontier paths once;
    5. mix semantic + episodic frontier experts and add working memory.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep the wrapped tree out of PyTorch's child-module registry.

        ``HawkesTree`` owns this adapter as ``tree.frontier_routing``.  A
        regular ``self.tree = tree`` assignment would therefore register the
        owner as a child of its own child and create the recursive module graph

            HawkesTree -> FrontierRoutingRetrieval -> HawkesTree.

        Such a graph recurses forever in ``Module.to()``, ``state_dict()``,
        ``train()``, and related PyTorch traversals.  Handle this attribute
        explicitly so later refactors cannot accidentally reintroduce that
        cycle.
        """
        if name == "tree":
            modules = self.__dict__.get("_modules")
            if modules is not None:
                modules.pop("tree", None)
            object.__setattr__(self, "_tree_owner", value)
            return
        super().__setattr__(name, value)

    @property
    def tree(self) -> nn.Module:
        tree = self.__dict__.get("_tree_owner")
        if tree is None:
            raise RuntimeError("frontier routing is detached from its HawkesTree")
        return tree

    def __init__(
        self,
        tree: nn.Module,
        config: Optional[FrontierRoutingConfig] = None,
    ) -> None:
        super().__init__()
        # The adapter is registered as a child of HawkesTree in the integrated
        # model. Registering the tree back as a child would create a recursive
        # module graph/state_dict, so keep only a non-module reference.
        self.tree = tree
        self.config = (
            FrontierRoutingConfig() if config is None else config
        )
        self.config.validate()
        self.prototypes = NodePrototypeStore(
            feature_dim=int(tree.z_dim),
            node_ids=tuple(tree.all_node_ids),
            device=tree._device_anchor.device,
        )
        self._topology_signature = tuple(tree.all_node_ids)
        self._validated_frontiers: set[tuple[str, ...]] = set()
        self.expansion_gain: Dict[str, float] = {}
        self.expansion_visits: Dict[str, int] = {}
        # Training-only, p_expand-independent coverage state for Regional
        # Probe. Counts are keyed by actual leaves and survive checkpoints.
        self.probe_leaf_visits: Dict[str, int] = {}
        self._topology_tensors: Dict[str, Tensor] = {}
        self._target_leaf_mass_by_id: Dict[str, float] = {}
        self._pending_target_leaf_mass: Optional[tuple[float, ...]] = None
        self._reset_target_leaf_mass_from_config()

    @staticmethod
    def _is_descendant(node_id: str, ancestor_id: str) -> bool:
        return node_id == ancestor_id or node_id.startswith(
            f"{ancestor_id}_"
        )

    def _store_target_leaf_mass(
        self,
        mass_by_id: Mapping[str, float],
    ) -> None:
        """Normalize and store the prior in current leaf-ID order."""
        leaf_ids = tuple(self.tree.leaf_ids)
        if set(mass_by_id) != set(leaf_ids):
            raise ValueError(
                "target leaf mass IDs must match the current tree leaves"
            )
        values = torch.tensor(
            [mass_by_id[leaf_id] for leaf_id in leaf_ids],
            dtype=torch.float64,
        )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("target_leaf_mass must contain finite values")
        if bool((values < 0).any()) or float(values.sum()) <= 0.0:
            raise ValueError(
                "target_leaf_mass must be non-negative with positive sum"
            )
        values = values / values.sum()
        self._target_leaf_mass_by_id = {
            leaf_id: float(values[index])
            for index, leaf_id in enumerate(leaf_ids)
        }
        # Keep the public configuration synchronized for diagnostics and for
        # callers that serialize the configuration separately from the model.
        self.config.target_leaf_mass = tuple(
            self._target_leaf_mass_by_id[leaf_id]
            for leaf_id in leaf_ids
        )

    def set_target_leaf_mass(
        self,
        target_leaf_mass: Sequence[float] | Tensor,
        *,
        leaf_ids: Optional[Sequence[str]] = None,
    ) -> None:
        """Install an empirical routing prior for the current active leaves."""
        self._sync_topology()
        self._pending_target_leaf_mass = None
        resolved_leaf_ids = (
            tuple(self.tree.leaf_ids)
            if leaf_ids is None
            else tuple(str(leaf_id) for leaf_id in leaf_ids)
        )
        values = torch.as_tensor(
            target_leaf_mass,
            dtype=torch.float64,
        ).detach().cpu().reshape(-1)
        if len(resolved_leaf_ids) != values.numel():
            raise ValueError(
                "target_leaf_mass must match the supplied number of leaf IDs"
            )
        if len(set(resolved_leaf_ids)) != len(resolved_leaf_ids):
            raise ValueError("target leaf IDs must be unique")
        self._store_target_leaf_mass({
            leaf_id: float(values[index])
            for index, leaf_id in enumerate(resolved_leaf_ids)
        })
        self._rebuild_topology_tensors()

    def _reset_target_leaf_mass_from_config(self) -> None:
        leaf_ids = tuple(self.tree.leaf_ids)
        configured = self.config.target_leaf_mass
        values: Sequence[float] | Tensor
        if configured is None:
            values = torch.ones(len(leaf_ids), dtype=torch.float64)
        else:
            values = configured
            if len(configured) != len(leaf_ids):
                # Checkpoint restoration configures routing before the saved
                # dynamic topology is installed. Defer positional binding
                # until that topology arrives; new checkpoints will instead
                # restore the unambiguous node-ID mapping from extra state.
                self._pending_target_leaf_mass = tuple(configured)
                if not self._topology_tensors:
                    self._store_target_leaf_mass({
                        leaf_id: 1.0 for leaf_id in leaf_ids
                    })
                    self.config.target_leaf_mass = tuple(configured)
                    self._rebuild_topology_tensors()
                return
        self._pending_target_leaf_mass = None
        self.set_target_leaf_mass(values, leaf_ids=leaf_ids)

    def _apply_pending_target_leaf_mass(self) -> bool:
        pending = self._pending_target_leaf_mass
        leaf_ids = tuple(self.tree.leaf_ids)
        if pending is None or len(pending) != len(leaf_ids):
            return False
        self._pending_target_leaf_mass = None
        self._store_target_leaf_mass({
            leaf_id: pending[index]
            for index, leaf_id in enumerate(leaf_ids)
        })
        return True

    def _reconcile_target_leaf_mass(self) -> None:
        """Conserve prior mass when leaves are split or merged."""
        current_leaf_ids = tuple(self.tree.leaf_ids)
        old_mass = dict(self._target_leaf_mass_by_id)
        if not old_mass:
            self._reset_target_leaf_mass_from_config()
            return
        if set(old_mass) == set(current_leaf_ids):
            self._store_target_leaf_mass(old_mass)
            return

        reconciled = {leaf_id: 0.0 for leaf_id in current_leaf_ids}
        for old_leaf_id, mass in old_mass.items():
            descendants = [
                leaf_id
                for leaf_id in current_leaf_ids
                if self._is_descendant(leaf_id, old_leaf_id)
            ]
            if descendants:
                # A former leaf was split. With no child-level evidence yet,
                # retain the parent's total prior and divide it symmetrically.
                share = mass / len(descendants)
                for leaf_id in descendants:
                    reconciled[leaf_id] += share
                continue

            ancestors = [
                leaf_id
                for leaf_id in current_leaf_ids
                if self._is_descendant(old_leaf_id, leaf_id)
            ]
            if not ancestors:
                raise RuntimeError(
                    f"cannot migrate target mass for removed leaf {old_leaf_id!r}"
                )
            # A set of old sibling leaves was merged. Assign each contribution
            # to the nearest surviving ancestor, so their masses add exactly.
            nearest = max(ancestors, key=lambda leaf_id: leaf_id.count("_"))
            reconciled[nearest] += mass
        self._store_target_leaf_mass(reconciled)

    def _sync_topology(self) -> None:
        node_ids = tuple(self.tree.all_node_ids)
        if node_ids == self._topology_signature:
            return
        self.prototypes.sync_nodes(node_ids)
        active = set(node_ids)
        self.expansion_gain = {
            node_id: gain
            for node_id, gain in self.expansion_gain.items()
            if node_id in active
        }
        self.expansion_visits = {
            node_id: visits
            for node_id, visits in self.expansion_visits.items()
            if node_id in active
        }
        active_leaves = set(self.tree.leaf_ids)
        self.probe_leaf_visits = {
            leaf_id: visits
            for leaf_id, visits in self.probe_leaf_visits.items()
            if leaf_id in active_leaves
        }
        self._topology_signature = node_ids
        self._validated_frontiers.clear()
        if not self._apply_pending_target_leaf_mass():
            self._reconcile_target_leaf_mass()
        self._rebuild_topology_tensors()

    def _rebuild_topology_tensors(self) -> None:
        """Cache integer topology and neutral-prior tensors on the tree device."""
        node_ids = tuple(self.tree.all_node_ids)
        if not node_ids:
            self._topology_tensors = {}
            return
        device = self.tree._device_anchor.device
        index = {node_id: i for i, node_id in enumerate(node_ids)}
        child_index = torch.full(
            (len(node_ids), 2),
            -1,
            dtype=torch.long,
            device=device,
        )
        internal = torch.zeros(len(node_ids), dtype=torch.bool, device=device)
        for node_id, node_index in index.items():
            node = self.tree.nodes[node_id]
            if not node.is_leaf:
                internal[node_index] = True
                child_index[node_index, 0] = index[node.left]
                child_index[node_index, 1] = index[node.right]

        leaf_ids = tuple(self.tree.leaf_ids)
        if set(self._target_leaf_mass_by_id) != set(leaf_ids):
            raise RuntimeError(
                "target leaf mass is stale; synchronize topology before routing"
            )
        leaf_mass = torch.tensor(
            [
                self._target_leaf_mass_by_id[leaf_id]
                for leaf_id in leaf_ids
            ],
            dtype=torch.float32,
            device=device,
        )
        node_mass = torch.zeros(len(node_ids), device=device)
        for leaf_index, path in enumerate(self.tree.leaf_paths):
            for node_id in path:
                node_mass[index[node_id]] += leaf_mass[leaf_index]
        child_prior = torch.zeros(len(node_ids), 2, device=device)
        valid_internal = internal.nonzero(as_tuple=False).reshape(-1)
        if valid_internal.numel():
            children = child_index.index_select(0, valid_internal)
            mass = node_mass.index_select(0, children.reshape(-1)).reshape(-1, 2)
            mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            child_prior.index_copy_(0, valid_internal, mass)

        path_mask = self.tree.node_path_mask.to(device=device, dtype=torch.bool)
        self._topology_tensors = {
            "child_index": child_index,
            "internal": internal,
            "leaf_mass": leaf_mass,
            "node_mass": node_mass,
            "child_prior": child_prior,
            "path_mask": path_mask,
        }

    def build_static_cache(
        self,
        *,
        detach: bool = False,
    ) -> FrontierStaticCache:
        """Build the node/semantic tables shared by a frozen Wake sequence."""
        self._sync_topology()
        node_embedding_table = self.tree._node_embedding_table()
        # Router losses may train the compatibility network, but must not move
        # the semantic node geometry.  Prediction still consumes the live
        # table below through ``semantic_theta_table``.
        normalized_node_table = self.tree.router_compat.normalize_nodes(
            node_embedding_table.detach()
        )
        semantic_theta_table = self.tree.semantic_theta_table(
            node_embedding_table
        )
        if detach:
            node_embedding_table = node_embedding_table.detach()
            normalized_node_table = normalized_node_table.detach()
            semantic_theta_table = semantic_theta_table.detach()
        return FrontierStaticCache(
            topology_signature=self._topology_signature,
            node_embedding_table=node_embedding_table,
            normalized_node_table=normalized_node_table,
            semantic_theta_table=semantic_theta_table,
        )

    def _validate_static_cache(
        self,
        cache: FrontierStaticCache,
        reference: Tensor,
    ) -> None:
        if cache.topology_signature != tuple(self.tree.all_node_ids):
            raise ValueError(
                "frontier static cache is stale after a topology change"
            )
        expected_node_shape = (
            len(self.tree.all_node_ids),
            int(self.tree.node_dim),
        )
        expected_semantic_shape = (
            len(self.tree.all_node_ids),
            int(self.tree.param_dim),
        )
        if cache.node_embedding_table.shape != expected_node_shape:
            raise ValueError("cached node embedding table has the wrong shape")
        if cache.normalized_node_table.shape != expected_node_shape:
            raise ValueError("cached normalized node table has the wrong shape")
        if cache.semantic_theta_table.shape != expected_semantic_shape:
            raise ValueError("cached semantic table has the wrong shape")
        if (
            cache.node_embedding_table.device != reference.device
            or cache.normalized_node_table.device != reference.device
            or cache.semantic_theta_table.device != reference.device
        ):
            raise ValueError("frontier static cache is on the wrong device")

    @torch.no_grad()
    def set_expansion_gain(self, node_id: str, gain: float) -> None:
        if node_id not in self.tree.nodes:
            raise KeyError(f"unknown tree node: {node_id}")
        if gain < 0.0:
            raise ValueError("expansion gain must be non-negative")
        self.expansion_gain[node_id] = float(gain)

    @torch.no_grad()
    def update_expansion_gain(
        self,
        node_indices: Tensor,
        observed_gain: Tensor,
        mask: Tensor,
    ) -> None:
        """EMA-update historical refinement value after targets are observed."""
        if (
            node_indices.shape != observed_gain.shape
            or mask.shape != node_indices.shape
        ):
            raise ValueError("expansion gain tensors must have the same shape")
        decay = self.config.expansion_gain_decay
        for node_index in node_indices[mask].unique().cpu().tolist():
            selected = mask & (node_indices == int(node_index))
            value = float(
                observed_gain.masked_select(selected).mean().cpu()
            )
            node_id = self.tree.all_node_ids[int(node_index)]
            previous = self.expansion_gain.get(
                node_id, self.config.default_expansion_gain
            )
            self.expansion_gain[node_id] = (
                decay * previous + (1.0 - decay) * max(value, 0.0)
            )

    def _child_ids(self, node_id: str) -> tuple[str, str]:
        node = self.tree.nodes[node_id]
        if node.left is None or node.right is None:
            raise ValueError(f"node is not expandable: {node_id}")
        return node.left, node.right

    def _tempered_child_prior(
        self,
        child_ids: Sequence[str],
        reference: Tensor,
    ) -> Tensor:
        node_index = {
            node_id: index
            for index, node_id in enumerate(self.tree.all_node_ids)
        }
        parent_id = self.tree.nodes[child_ids[0]].parent
        if parent_id is None:
            raise RuntimeError("child prior requested for root")
        return self._topology_tensors["child_prior"][
            node_index[parent_id]
        ].to(reference)

    def _branch_distribution(
        self,
        node_id: str,
        z: Tensor,
        projected_z: Tensor,
        normalized_node_table: Tensor,
        node_index: Mapping[str, int],
    ) -> BranchDecision:
        child_ids = self._child_ids(node_id)
        child_indices = torch.tensor(
            [node_index[child_id] for child_id in child_ids],
            device=normalized_node_table.device,
            dtype=torch.long,
        )
        child_normalized = normalized_node_table.index_select(
            0,
            child_indices,
        )
        semantic_score = self.tree.router_compat.score_normalized(
            projected_z.unsqueeze(0),
            child_normalized,
        ).squeeze(0)
        # Data-dependent evidence is introduced only after observing the
        # target through the frontier posterior. Search itself remains a
        # strictly causal cheap semantic router.
        data_score = torch.zeros_like(semantic_score)
        prior = self._tempered_child_prior(
            child_ids,
            semantic_score,
        )
        log_prior = prior.clamp_min(1e-12).log()

        total_score = (
            self.config.semantic_weight
            * semantic_score
            / self.config.routing_temperature
            + log_prior
        )
        probability = F.softmax(total_score, dim=-1)
        entropy = -(
            probability * probability.clamp_min(1e-12).log()
        ).sum()
        return BranchDecision(
            node_id=node_id,
            child_ids=child_ids,
            semantic_score=semantic_score,
            data_score=data_score,
            log_prior=log_prior,
            total_score=total_score,
            probability=probability,
            entropy=entropy,
        )

    def _priority(
        self,
        node_id: str,
        mass: Tensor,
        decision: BranchDecision,
    ) -> Tensor:
        gain = self.expansion_gain.get(
            node_id,
            self.config.default_expansion_gain,
        )
        confidence = (
            1.0
            - decision.entropy.detach()
            / math.log(2.0)
        ).clamp(0.0, 1.0)
        priority = (
            mass.detach()
            * (
                gain
                + self.config.confidence_weight * confidence
            )
            - self.config.expansion_compute_cost
        )
        return priority

    def _route_one(
        self,
        z: Tensor,
        projected_z: Tensor,
        normalized_node_table: Tensor,
        node_index: Mapping[str, int],
        *,
        update_search_state: bool,
    ) -> FrontierSample:
        frontier_ids = ["root"]
        frontier_mass = [z.new_ones(())]
        decisions: Dict[str, BranchDecision] = {}
        expanded: list[str] = []

        while len(frontier_ids) < self.config.frontier_budget:
            candidates: list[tuple[Tensor, int, str]] = []
            for index, (node_id, mass) in enumerate(
                zip(frontier_ids, frontier_mass)
            ):
                if self.tree.nodes[node_id].is_leaf:
                    continue
                if node_id not in decisions:
                    decisions[node_id] = self._branch_distribution(
                        node_id,
                        z,
                        projected_z,
                        normalized_node_table,
                        node_index,
                    )
                priority = self._priority(
                    node_id,
                    mass,
                    decisions[node_id],
                )
                candidates.append((priority, index, node_id))
            if not candidates:
                break

            # One device-to-host synchronization per expansion. The previous
            # Python ``max(float(priority.cpu()))`` synchronized once for
            # every candidate and left the GPU idle between tiny kernels.
            candidate_priority = torch.stack(
                [item[0] for item in candidates]
            )
            selected = int(candidate_priority.argmax().item())
            _, frontier_index, node_id = candidates[selected]
            parent_mass = frontier_mass[frontier_index]
            decision = decisions[node_id]
            decision.priority = candidate_priority[selected].detach()
            child_mass = parent_mass * decision.probability
            left_id, right_id = decision.child_ids
            frontier_ids[frontier_index:frontier_index + 1] = [
                left_id,
                right_id,
            ]
            frontier_mass[frontier_index:frontier_index + 1] = [
                child_mass[0],
                child_mass[1],
            ]
            expanded.append(node_id)
            if update_search_state:
                self.expansion_visits[node_id] = (
                    self.expansion_visits.get(node_id, 0) + 1
                )

        mass = torch.stack(frontier_mass)
        # Numerical drift is tiny, but this keeps the partition invariant
        # exact enough for long trees and mixed precision.
        mass = mass / mass.sum().clamp_min(1e-12)
        visited = tuple(dict.fromkeys(
            path_node
            for frontier_node in frontier_ids
            for path_node in self.tree.path_to_node(frontier_node)
        ))
        result = FrontierSample(
            node_ids=tuple(frontier_ids),
            mass=mass,
            visited_node_ids=visited,
            expanded_node_ids=tuple(expanded),
            decisions=decisions,
        )
        self._validate_frontier(result)
        return result

    def _validate_frontier(self, sample: FrontierSample) -> None:
        signature = sample.node_ids
        if signature in self._validated_frontiers:
            return
        if len(sample.node_ids) > self.config.frontier_budget:
            raise RuntimeError("frontier exceeds configured budget")
        if not torch.allclose(
            sample.mass.sum(),
            sample.mass.new_ones(()),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError("frontier mass does not sum to one")
        paths = {
            node_id: set(self.tree.node_paths[node_id])
            for node_id in sample.node_ids
        }
        for left_index, left_id in enumerate(sample.node_ids):
            for right_id in sample.node_ids[left_index + 1:]:
                if left_id in paths[right_id] or right_id in paths[left_id]:
                    raise RuntimeError(
                        "frontier is not an ancestor-free antichain"
                    )
        for leaf_path in self.tree.leaf_paths:
            coverage = sum(
                node_id in leaf_path
                for node_id in sample.node_ids
            )
            if coverage != 1:
                raise RuntimeError(
                    "frontier does not partition all stored leaves"
                )
        self._validated_frontiers.add(signature)

    def route_packed(
        self,
        z_t: Tensor,
        *,
        update_search_state: bool = True,
        node_embedding_table: Optional[Tensor] = None,
        normalized_node_table: Optional[Tensor] = None,
        projected_z: Optional[Tensor] = None,
    ) -> PackedFrontierBatch:
        """Run at most ``K_max-1`` masked expansion rounds on fixed tensors."""
        if z_t.ndim != 2 or z_t.size(-1) != self.tree.z_dim:
            raise ValueError(
                f"z_t must have shape [B, {self.tree.z_dim}]"
            )
        self._sync_topology()
        if node_embedding_table is None:
            node_embedding_table = self.tree._node_embedding_table()
        if normalized_node_table is None:
            normalized_node_table = self.tree.router_compat.normalize_nodes(
                node_embedding_table
            )
        if projected_z is None:
            projected_z = self.tree.router_compat.project_z(z_t)

        batch = z_t.size(0)
        width = self.config.frontier_budget
        rounds = max(width - 1, 0)
        device = z_t.device
        node_indices = torch.full(
            (batch, width), -1, dtype=torch.long, device=device
        )
        mass = z_t.new_zeros(batch, width)
        mask = torch.zeros(batch, width, dtype=torch.bool, device=device)
        node_indices[:, 0] = 0
        mass[:, 0] = 1.0
        mask[:, 0] = True

        expanded_node = torch.full(
            (batch, rounds), -1, dtype=torch.long, device=device
        )
        expanded_children = torch.full(
            (batch, rounds, 2), -1, dtype=torch.long, device=device
        )
        expanded_probability = z_t.new_zeros(batch, rounds, 2)
        expanded_semantic = z_t.new_zeros(batch, rounds, 2)
        expanded_mask = torch.zeros(
            batch, rounds, dtype=torch.bool, device=device
        )
        expansion_utility = z_t.new_full((batch, rounds), -torch.inf)

        topology = self._topology_tensors
        child_table = topology["child_index"]
        internal_table = topology["internal"]
        prior_table = topology["child_prior"].to(z_t)
        gain_table = z_t.new_tensor([
            self.expansion_gain.get(
                node_id, self.config.default_expansion_gain
            )
            for node_id in self.tree.all_node_ids
        ])

        for round_index in range(rounds):
            safe_nodes = node_indices.clamp_min(0)
            candidate_internal = (
                mask & internal_table.index_select(
                    0, safe_nodes.reshape(-1)
                ).reshape(batch, width)
            )
            children = child_table.index_select(
                0, safe_nodes.reshape(-1)
            ).reshape(batch, width, 2)
            safe_children = children.clamp_min(0)
            child_embedding = normalized_node_table.index_select(
                0, safe_children.reshape(-1)
            ).reshape(batch, width, 2, -1)
            z_for_children = projected_z[:, None, None, :].expand(
                -1, width, 2, -1
            )
            semantic_score = self.tree.router_compat.score_normalized(
                z_for_children.reshape(batch, width * 2, -1),
                child_embedding.reshape(batch, width * 2, -1),
            ).reshape(batch, width, 2)
            prior = prior_table.index_select(
                0, safe_nodes.reshape(-1)
            ).reshape(batch, width, 2)
            total_score = (
                self.config.semantic_weight
                * semantic_score
                / self.config.routing_temperature
                + prior.clamp_min(1e-12).log()
            )
            probability = F.softmax(total_score, dim=-1)
            entropy = -(
                probability * probability.clamp_min(1e-12).log()
            ).sum(dim=-1)
            confidence = (
                1.0 - entropy / math.log(2.0)
            ).clamp(0.0, 1.0)
            gain = gain_table.index_select(
                0, safe_nodes.reshape(-1)
            ).reshape(batch, width)
            utility = (
                mass.detach()
                * (gain + self.config.confidence_weight * confidence.detach())
                - self.config.expansion_compute_cost
            ).masked_fill(~candidate_internal, -torch.inf)
            selected_utility, selected_slot = utility.max(dim=-1)
            count = mask.sum(dim=-1)
            must_expand = count < self.config.frontier_min_experts
            active = candidate_internal.any(dim=-1) & (
                must_expand | (selected_utility > 0.0)
            )
            if round_index == 0:
                # The root is forced whenever it is expandable.
                active = candidate_internal[:, 0]
                selected_slot = torch.zeros_like(selected_slot)
                selected_utility = utility[:, 0]

            selected_node = node_indices.gather(
                1, selected_slot.unsqueeze(1)
            ).squeeze(1)
            selected_children = children.gather(
                1,
                selected_slot[:, None, None].expand(-1, 1, 2),
            ).squeeze(1)
            selected_probability = probability.gather(
                1,
                selected_slot[:, None, None].expand(-1, 1, 2),
            ).squeeze(1)
            selected_semantic = semantic_score.gather(
                1,
                selected_slot[:, None, None].expand(-1, 1, 2),
            ).squeeze(1)
            parent_mass = mass.gather(
                1, selected_slot.unsqueeze(1)
            ).squeeze(1)
            new_mass = parent_mass.unsqueeze(-1) * selected_probability

            expanded_mask[:, round_index] = active
            expanded_node[:, round_index] = torch.where(
                active, selected_node, expanded_node[:, round_index]
            )
            expanded_children[:, round_index] = torch.where(
                active[:, None],
                selected_children,
                expanded_children[:, round_index],
            )
            expanded_probability[:, round_index] = torch.where(
                active[:, None],
                selected_probability,
                expanded_probability[:, round_index],
            )
            expanded_semantic[:, round_index] = torch.where(
                active[:, None],
                selected_semantic,
                expanded_semantic[:, round_index],
            )
            expansion_utility[:, round_index] = torch.where(
                active,
                selected_utility,
                expansion_utility[:, round_index],
            )

            # Replace the selected parent by its left child and append the
            # right child into the first free slot. Keep the mass update
            # functional: later rounds need earlier mass values for gradient
            # computation, so mutating the same tensor would invalidate
            # autograd's saved versions.
            free_slot = (~mask).to(torch.int64).argmax(dim=-1)
            selected_one_hot = F.one_hot(
                selected_slot, num_classes=width
            ).bool() & active.unsqueeze(-1)
            free_one_hot = F.one_hot(
                free_slot, num_classes=width
            ).bool() & active.unsqueeze(-1)
            node_indices = torch.where(
                selected_one_hot,
                selected_children[:, 0].unsqueeze(-1),
                node_indices,
            )
            node_indices = torch.where(
                free_one_hot,
                selected_children[:, 1].unsqueeze(-1),
                node_indices,
            )
            mass = torch.where(
                selected_one_hot,
                new_mass[:, 0].unsqueeze(-1),
                mass,
            )
            mass = torch.where(
                free_one_hot,
                new_mass[:, 1].unsqueeze(-1),
                mass,
            )
            mask = mask | free_one_hot

        mass = mass * mask.to(mass.dtype)
        mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        safe_frontier = node_indices.clamp_min(0)
        frontier_paths = topology["path_mask"].index_select(
            0, safe_frontier.reshape(-1)
        ).reshape(batch, width, -1)
        frontier_paths = frontier_paths & mask.unsqueeze(-1)
        visited_all = frontier_paths.any(dim=1)
        max_visited = 2 * width - 1
        all_index = torch.arange(
            len(self.tree.all_node_ids), device=device
        ).expand(batch, -1)
        sentinel = len(self.tree.all_node_ids)
        ordered = torch.where(
            visited_all, all_index, all_index.new_full((), sentinel)
        ).sort(dim=-1).values[:, :max_visited]
        visited_mask = ordered < sentinel
        visited_indices = ordered.masked_fill(~visited_mask, -1)
        safe_visited = visited_indices.clamp_min(0)
        path_incidence = frontier_paths.gather(
            2,
            safe_visited[:, None, :].expand(-1, width, -1),
        )
        path_incidence = (
            path_incidence
            & mask.unsqueeze(-1)
            & visited_mask.unsqueeze(1)
        )

        if update_search_state:
            with torch.no_grad():
                flat = expanded_node[expanded_mask]
                if flat.numel():
                    counts = torch.bincount(
                        flat,
                        minlength=len(self.tree.all_node_ids),
                    ).cpu().tolist()
                    for node_id, value in zip(self.tree.all_node_ids, counts):
                        if value:
                            self.expansion_visits[node_id] = (
                                self.expansion_visits.get(node_id, 0)
                                + int(value)
                            )

        return PackedFrontierBatch(
            node_indices=node_indices,
            mass=mass,
            mask=mask,
            visited_indices=visited_indices,
            visited_mask=visited_mask,
            path_incidence=path_incidence,
            expanded_node_indices=expanded_node,
            expanded_child_indices=expanded_children,
            expanded_probability=expanded_probability,
            expanded_semantic_score=expanded_semantic,
            expanded_mask=expanded_mask,
            expansion_utility=expansion_utility,
        )

    def route(
        self,
        z_t: Tensor,
        *,
        update_search_state: bool = True,
        node_embedding_table: Optional[Tensor] = None,
        normalized_node_table: Optional[Tensor] = None,
        projected_z: Optional[Tensor] = None,
    ) -> tuple[FrontierSample, ...]:
        """Compatibility view over :meth:`route_packed`.

        Training and prediction consume packed tensors directly. Materializing
        Python samples is retained only for diagnostics and older callers.
        """
        packed = self.route_packed(
            z_t,
            update_search_state=update_search_state,
            node_embedding_table=node_embedding_table,
            normalized_node_table=normalized_node_table,
            projected_z=projected_z,
        )
        result = []
        node_ids = tuple(self.tree.all_node_ids)
        for batch_index in range(z_t.size(0)):
            active = packed.mask[batch_index]
            indices = packed.node_indices[batch_index, active].tolist()
            frontier_ids = tuple(node_ids[index] for index in indices)
            expanded = packed.expanded_node_indices[
                batch_index, packed.expanded_mask[batch_index]
            ].tolist()
            visited = packed.visited_indices[
                batch_index, packed.visited_mask[batch_index]
            ].tolist()
            result.append(FrontierSample(
                node_ids=frontier_ids,
                mass=packed.mass[batch_index, active],
                visited_node_ids=tuple(node_ids[index] for index in visited),
                expanded_node_ids=tuple(node_ids[index] for index in expanded),
                decisions={},
            ))
        return tuple(result)

    def leaf_responsibility(
        self,
        samples: Sequence[FrontierSample],
    ) -> Tensor:
        """Retired: v2 never invents responsibility for unseen leaves."""
        raise RuntimeError(
            "full-leaf projection is not part of posterior_frontier_v2; "
            "consume PackedFrontierBatch.node_indices/mass/mask directly"
        )

    @torch.no_grad()
    def _credit_frontier_retrieval(
        self,
        sample: FrontierSample,
        info_by_node: Mapping[str, Dict[str, Tensor]],
    ) -> None:
        for node_id, info in info_by_node.items():
            if "alpha" not in info:
                continue
            node_mass = sample.mass.new_zeros(())
            for frontier_index, frontier_id in enumerate(sample.node_ids):
                if node_id in self.tree.path_to_node(frontier_id):
                    node_mass = node_mass + sample.mass[frontier_index]
            bank = self.tree.episodic_memory.banks.get(node_id)
            if bank is not None and len(bank) > 0:
                bank.cycle_usage.add_(
                    info["alpha"].to(bank.device)
                    * node_mass.to(bank.device)
                )

    def _retrieve_one(
        self,
        query: Tensor,
        sample: FrontierSample,
        *,
        update_memory_state: bool,
    ) -> tuple[Tensor, Mapping[str, Dict[str, Tensor]]]:
        memory = self.tree.episodic_memory
        delta_by_node, info_by_node = memory.read_nodes(
            query=query,
            node_ids=sample.visited_node_ids,
            update_state=False,
        )
        frontier_delta = torch.stack(
            [
                memory.aggregate_path(
                    self.tree.path_to_node(node_id),
                    delta_by_node,
                )
                for node_id in sample.node_ids
            ],
            dim=0,
        )
        if update_memory_state:
            self._credit_frontier_retrieval(sample, info_by_node)
        return frontier_delta, info_by_node

    def _working_delta(
        self,
        z_t: Tensor,
        working_delta: Optional[Tensor],
    ) -> Tensor:
        if working_delta is None:
            working_delta = (
                self.tree.working_memory.make_trainable_delta()
            )
        if working_delta.ndim == 1:
            return working_delta.unsqueeze(0).expand(
                z_t.size(0), -1
            )
        if working_delta.shape != (
            z_t.size(0),
            self.tree.param_dim,
        ):
            raise ValueError(
                "working_delta must have shape [P] or [B, P]"
            )
        return working_delta

    def forward(
        self,
        z_t: Tensor,
        working_delta: Optional[Tensor] = None,
        decays: Optional[Tensor] = None,
        static_cache: Optional[FrontierStaticCache] = None,
        projected_z: Optional[Tensor] = None,
        precomputed_query: Optional[Tensor] = None,
        *,
        update_memory_state: bool = True,
        update_search_state: bool = True,
        detach_routing: bool = False,
        materialize_diagnostics: bool = True,
        precomputed_frontier: Optional[PackedFrontierBatch] = None,
        precomputed_node_delta: Optional[Tensor] = None,
        precomputed_episodic_delta: Optional[Tensor] = None,
        precomputed_memory_info: Optional[Mapping[str, Tensor]] = None,
    ) -> FrontierBatchOutput:
        if static_cache is None:
            static_cache = self.build_static_cache()
        else:
            self._validate_static_cache(static_cache, z_t)
        node_embedding_table = static_cache.node_embedding_table
        if precomputed_frontier is None:
            frontier = self.route_packed(
                z_t,
                update_search_state=update_search_state,
                node_embedding_table=node_embedding_table,
                normalized_node_table=static_cache.normalized_node_table,
                projected_z=projected_z,
            )
        else:
            frontier = precomputed_frontier
            if (
                frontier.mass.size(0) != z_t.size(0)
                or frontier.node_indices.device != z_t.device
            ):
                raise ValueError(
                    "precomputed frontier must align with z_t batch/device"
                )
        memory = self.tree.episodic_memory
        if memory.query_net is None:
            raise RuntimeError(
                "the wrapped tree must construct memory.query_net"
            )
        query = (
            memory.query_net(z_t)
            if precomputed_query is None
            else precomputed_query
        )
        expected_query_shape = (z_t.size(0), memory.key_dim)
        if query.shape != expected_query_shape:
            raise ValueError(
                "precomputed query must have shape "
                f"{expected_query_shape}, got {tuple(query.shape)}"
            )
        if query.device != z_t.device:
            raise ValueError("precomputed query is on the wrong device")
        working = self._working_delta(z_t, working_delta)
        semantic_table = static_cache.semantic_theta_table
        safe_frontier = frontier.node_indices.clamp_min(0)
        semantic_theta = semantic_table.index_select(
            0, safe_frontier.reshape(-1)
        ).reshape(
            z_t.size(0),
            self.config.frontier_budget,
            self.tree.param_dim,
        )
        semantic_theta = (
            semantic_theta * frontier.mask.unsqueeze(-1)
        )
        if precomputed_node_delta is None:
            if precomputed_memory_info is not None:
                raise ValueError(
                    "precomputed_memory_info requires precomputed_node_delta"
                )
            node_delta, packed_memory_info = memory.read_packed(
                query=query,
                node_indices=frontier.visited_indices,
                node_mask=frontier.visited_mask,
                node_ids=self.tree.all_node_ids,
                update_state=update_memory_state,
            )
        else:
            expected_delta_shape = (
                *frontier.visited_indices.shape,
                self.tree.param_dim,
            )
            if precomputed_node_delta.shape != expected_delta_shape:
                raise ValueError(
                    "precomputed_node_delta must have shape "
                    f"{expected_delta_shape}, got "
                    f"{tuple(precomputed_node_delta.shape)}"
                )
            if precomputed_node_delta.device != z_t.device:
                raise ValueError(
                    "precomputed_node_delta is on the wrong device"
                )
            if precomputed_memory_info is None:
                raise ValueError(
                    "precomputed_memory_info is required with "
                    "precomputed_node_delta"
                )
            for key in (
                "alpha",
                "similarity",
                "effective_k",
                "null_alpha",
                "valid_mask",
            ):
                value = precomputed_memory_info.get(key)
                if value is None:
                    raise ValueError(
                        "precomputed_memory_info is missing "
                        f"{key!r}"
                    )
                if value.shape[:2] != frontier.visited_indices.shape:
                    raise ValueError(
                        "precomputed memory info must align with "
                        "frontier.visited_indices"
                    )
                if value.device != z_t.device:
                    raise ValueError(
                        "precomputed memory info is on the wrong device"
                    )
            node_delta = precomputed_node_delta
            packed_memory_info = precomputed_memory_info
        if precomputed_episodic_delta is None:
            episodic_delta = torch.einsum(
                "nkv,nvp->nkp",
                frontier.path_incidence.to(node_delta.dtype),
                node_delta,
            )
        else:
            expected_episodic_shape = (
                z_t.size(0),
                frontier.node_indices.size(1),
                self.tree.param_dim,
            )
            if precomputed_episodic_delta.shape != expected_episodic_shape:
                raise ValueError(
                    "precomputed_episodic_delta must have shape "
                    f"{expected_episodic_shape}, got "
                    f"{tuple(precomputed_episodic_delta.shape)}"
                )
            if precomputed_episodic_delta.device != z_t.device:
                raise ValueError(
                    "precomputed_episodic_delta is on the wrong device"
                )
            episodic_delta = precomputed_episodic_delta
        frontier_theta = semantic_theta + episodic_delta

        # Materialize immediate-child semantic + episodic dynamics for every
        # actually expanded node. This is the prior-independent child-energy
        # teacher support: it contains neither Router mixture mass nor working
        # memory. A child may have been expanded again and no longer appear on
        # the final frontier, so deriving this table from final slots would be
        # incomplete.
        safe_children = frontier.expanded_child_indices.clamp_min(0)
        expanded_child_semantic = semantic_table.index_select(
            0,
            safe_children.reshape(-1),
        ).reshape(
            z_t.size(0),
            frontier.expanded_child_indices.size(1),
            2,
            self.tree.param_dim,
        )
        child_paths = self._topology_tensors["path_mask"].index_select(
            0,
            safe_children.reshape(-1),
        ).reshape(
            z_t.size(0),
            frontier.expanded_child_indices.size(1),
            2,
            len(self.tree.all_node_ids),
        )
        safe_visited = frontier.visited_indices.clamp_min(0)
        child_path_incidence = child_paths.gather(
            3,
            safe_visited[:, None, None, :].expand(
                -1,
                frontier.expanded_child_indices.size(1),
                2,
                -1,
            ),
        )
        child_path_incidence = (
            child_path_incidence
            & frontier.expanded_mask[:, :, None, None]
            & frontier.visited_mask[:, None, None, :]
        )
        expanded_child_episodic = torch.einsum(
            "nrcv,nvp->nrcp",
            child_path_incidence.to(node_delta.dtype),
            node_delta,
        )
        expanded_child_theta = (
            expanded_child_semantic + expanded_child_episodic
        ).masked_fill(
            ~frontier.expanded_mask[:, :, None, None],
            0.0,
        )
        routing_mass = (
            frontier.mass.detach()
            if detach_routing
            else frontier.mass
        )
        theta = (
            routing_mass.unsqueeze(-1) * frontier_theta
        ).sum(dim=1) + working

        # Compatibility diagnostics are materialized after the packed hot
        # path. They never define responsibility or retrieval computation.
        # Global training disables them: converting every prefix to Python
        # tuples and CPU lists would otherwise serialize an already-packed
        # GPU batch.
        samples = []
        memory_info = []
        if materialize_diagnostics:
            all_node_ids = tuple(self.tree.all_node_ids)
            for batch_index in range(z_t.size(0)):
                active = frontier.mask[batch_index]
                active_indices = frontier.node_indices[
                    batch_index, active
                ].detach().cpu().tolist()
                visited = frontier.visited_indices[
                    batch_index, frontier.visited_mask[batch_index]
                ].detach().cpu().tolist()
                expanded = frontier.expanded_node_indices[
                    batch_index, frontier.expanded_mask[batch_index]
                ].detach().cpu().tolist()
                samples.append(FrontierSample(
                    node_ids=tuple(
                        all_node_ids[index] for index in active_indices
                    ),
                    mass=frontier.mass[batch_index, active],
                    visited_node_ids=tuple(
                        all_node_ids[index] for index in visited
                    ),
                    expanded_node_ids=tuple(
                        all_node_ids[index] for index in expanded
                    ),
                    decisions={},
                ))
                memory_info.append({
                    all_node_ids[node_index]: {
                        "alpha": packed_memory_info["alpha"][
                            batch_index, visit_index
                        ],
                        "sim": packed_memory_info["similarity"][
                            batch_index, visit_index
                        ],
                        "effective_k": packed_memory_info["effective_k"][
                            batch_index, visit_index
                        ],
                    }
                    for visit_index, node_index in enumerate(visited)
                })

        updater = memory.parameter_update
        raw_mu = theta[..., :updater.D]
        raw_W = theta[..., updater.D:].reshape(
            z_t.size(0),
            updater.D,
            updater.D,
            updater.M,
        )
        effective = FrontierEffectiveParameters(
            theta=theta,
            raw_mu=raw_mu,
            raw_W=raw_W,
            mu=F.softplus(raw_mu),
            W=F.softplus(raw_W),
            decays=decays,
        )
        return FrontierBatchOutput(
            samples=tuple(samples),
            effective_params=effective,
            query=query,
            semantic_theta=(
                tuple(
                    semantic_theta[index, frontier.mask[index]]
                    for index in range(z_t.size(0))
                )
                if materialize_diagnostics
                else ()
            ),
            episodic_delta=(
                tuple(
                    episodic_delta[index, frontier.mask[index]]
                    for index in range(z_t.size(0))
                )
                if materialize_diagnostics
                else ()
            ),
            frontier_theta=(
                tuple(
                    frontier_theta[index, frontier.mask[index]]
                    for index in range(z_t.size(0))
                )
                if materialize_diagnostics
                else ()
            ),
            expanded_child_theta=expanded_child_theta,
            memory_info=tuple(memory_info),
            working_delta=working,
            frontier=frontier,
            frontier_mass=frontier.mass,
            frontier_mask=frontier.mask,
            frontier_node_indices=frontier.node_indices,
            visited_node_indices=frontier.visited_indices,
            visited_node_mask=frontier.visited_mask,
            semantic_theta_packed=semantic_theta,
            episodic_delta_packed=episodic_delta,
            frontier_theta_packed=frontier_theta,
            packed_memory_info=packed_memory_info,
        )

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "expansion_gain": dict(self.expansion_gain),
            "expansion_visits": dict(self.expansion_visits),
            "probe_leaf_visits": dict(self.probe_leaf_visits),
            "target_leaf_mass_by_id": dict(
                self._target_leaf_mass_by_id
            ),
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        self.expansion_gain = {
            str(key): float(value)
            for key, value in state.get(
                "expansion_gain", {}
            ).items()
        }
        self.expansion_visits = {
            str(key): int(value)
            for key, value in state.get(
                "expansion_visits", {}
            ).items()
        }
        self.probe_leaf_visits = {
            str(key): int(value)
            for key, value in state.get(
                "probe_leaf_visits", {}
            ).items()
        }
        stored_mass = state.get("target_leaf_mass_by_id")
        if stored_mass is not None:
            self._pending_target_leaf_mass = None
            self._target_leaf_mass_by_id = {
                str(key): float(value)
                for key, value in stored_mass.items()
            }
            self._reconcile_target_leaf_mass()
            self._rebuild_topology_tensors()
        elif self._apply_pending_target_leaf_mass():
            self._rebuild_topology_tensors()
