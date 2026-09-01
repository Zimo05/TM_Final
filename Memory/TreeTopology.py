"""Dynamic topology, path buffers, and serialization for HawkesTree."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from Config import TreeNode


class TreeTopologyMixin:
    """Topology methods mixed into :class:`LatentHawkesTree.HawkesTree`."""

    def get_extra_state(self):
        return {
            "topology": {
                node_id: {
                    "parent": node.parent,
                    "left": node.left,
                    "right": node.right,
                    "depth": node.depth,
                }
                for node_id, node in self.nodes.items()
            },
            "mass_ema": dict(self.mass_ema),
            "low_mass_streak": dict(self.low_mass_streak),
            "topology_prune_streak": dict(self.topology_prune_streak),
            "topology_prune_near_zero_streak": dict(
                self.topology_prune_near_zero_streak
            ),
            "memory_reconciliation": dict(self.memory_reconciliation),
            "residual_probe": {
                "leaf_ids": list(self.residual_probe_leaf_ids),
                "prototypes": self.residual_probe_prototypes.detach().cpu(),
                "target_mass": self.residual_probe_target_mass.detach().cpu(),
            },
        }

    def _restore_topology_modules(self, state) -> None:
        topology = (
            state.get("topology")
            if isinstance(state, dict)
            else None
        )
        if not topology:
            return
        device = self._device_anchor.device
        dtype = next(self.hyper.parameters()).dtype
        self.nodes = {
            node_id: TreeNode(
                node_id=node_id,
                parent=node_state["parent"],
                left=node_state["left"],
                right=node_state["right"],
                depth=node_state["depth"],
            )
            for node_id, node_state in topology.items()
        }
        self.node_emb = nn.ParameterDict({
            node_id: nn.Parameter(
                torch.empty(
                    self.node_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            for node_id in self.nodes
        })
        self.semantic_offset = nn.ParameterDict({
            node_id: nn.Parameter(
                torch.empty(
                    self.param_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            for node_id in self.nodes
        })
        self.refresh_structure_buffers()

    def _prepare_topology_for_load(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        extra_state = state_dict.get(prefix + "_extra_state")
        if extra_state is not None:
            self._restore_topology_modules(extra_state)
        # Old checkpoints stored one dense left/right bias module per
        # internal node. Ignore those tensors during the one-way migration to
        # active-frontier routing.
        for key in tuple(state_dict):
            if key.startswith(prefix + "routers."):
                state_dict.pop(key)

    def set_extra_state(self, state) -> None:
        self.mass_ema = dict(state.get("mass_ema", {}))
        self.low_mass_streak = dict(
            state.get("low_mass_streak", {})
        )
        self.topology_prune_streak = dict(
            state.get("topology_prune_streak", {})
        )
        self.topology_prune_near_zero_streak = dict(
            state.get("topology_prune_near_zero_streak", {})
        )
        self.memory_reconciliation = dict(
            state.get("memory_reconciliation", {})
        )
        residual_probe = state.get("residual_probe", {})
        leaf_ids = tuple(
            str(value) for value in residual_probe.get("leaf_ids", ())
        )
        prototypes = residual_probe.get("prototypes")
        target_mass = residual_probe.get("target_mass")
        if leaf_ids and prototypes is not None and target_mass is not None:
            self.set_residual_probe_prototypes(
                leaf_ids,
                prototypes,
                target_mass,
            )
        else:
            self.residual_probe_leaf_ids = ()
            self.residual_probe_prototypes = torch.empty(
                0,
                self.param_dim,
                device=self._device_anchor.device,
            )
            self.residual_probe_target_mass = torch.empty(
                0,
                device=self._device_anchor.device,
            )

    def _add_node(
        self,
        node_id: str,
        parent: Optional[str],
        depth: int,
    ) -> None:
        if node_id in self.nodes:
            raise ValueError(f"Node already exists: {node_id}")
        self.nodes[node_id] = TreeNode(
            node_id=node_id,
            parent=parent,
            depth=depth,
        )
        self.node_emb[node_id] = nn.Parameter(
            torch.empty(
                self.node_dim,
                device=self._device_anchor.device,
            )
        )
        nn.init.normal_(
            self.node_emb[node_id],
            mean=0.0,
            std=0.02,
        )
        self.semantic_offset[node_id] = nn.Parameter(
            torch.zeros(
                self.param_dim,
                device=self._device_anchor.device,
            )
        )

    @staticmethod
    def _add_parameters_to_optimizer(
        optimizer: torch.optim.Optimizer,
        parameters: List[nn.Parameter],
    ) -> None:
        optimized_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        new_parameters = [
            parameter
            for parameter in parameters
            if id(parameter) not in optimized_parameter_ids
        ]
        if new_parameters:
            optimizer.add_param_group({"params": new_parameters})

    def split_leaf(
        self,
        leaf_id: str,
        refresh: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> List[nn.Parameter]:
        if leaf_id not in self.nodes:
            raise KeyError(f"Unknown node: {leaf_id}")
        node = self.nodes[leaf_id]
        if not node.is_leaf:
            raise ValueError(f"Node is already internal: {leaf_id}")
        parameter_ids_before = {
            id(parameter)
            for parameter in self.parameters()
        }
        left_id = f"{leaf_id}_L"
        right_id = f"{leaf_id}_R"
        node.left = left_id
        node.right = right_id
        self._add_node(
            left_id,
            parent=leaf_id,
            depth=node.depth + 1,
        )
        self._add_node(
            right_id,
            parent=leaf_id,
            depth=node.depth + 1,
        )
        if refresh:
            self.refresh_structure_buffers()
        new_parameters = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in parameter_ids_before
        ]
        if optimizer is not None:
            self._add_parameters_to_optimizer(
                optimizer,
                new_parameters,
            )
        return new_parameters

    def _dfs(
        self,
        node_id: str,
        internal_ids: List[str],
        leaf_ids: List[str],
    ) -> None:
        node = self.nodes[node_id]
        if node.is_leaf:
            leaf_ids.append(node_id)
            return
        if node.left is None or node.right is None:
            raise RuntimeError(
                f"Internal node {node_id!r} must have two children."
            )
        internal_ids.append(node_id)
        self._dfs(node.left, internal_ids, leaf_ids)
        self._dfs(node.right, internal_ids, leaf_ids)

    def refresh_structure_buffers(self) -> None:
        """Rebuild all immutable path structures after topology changes."""
        internal_ids: List[str] = []
        leaf_ids: List[str] = []
        self._dfs("root", internal_ids, leaf_ids)
        self.internal_ids = internal_ids
        self.leaf_ids = leaf_ids
        self.all_node_ids = list(self.nodes)

        # Paths are topology-only. They used to be rebuilt by walking parent
        # pointers in every routed event, retrieval aggregation, and semantic
        # lookup. Materialize them once per structural change instead.
        self.node_paths = {}
        for target_id in self.all_node_ids:
            path: List[str] = []
            current: Optional[str] = target_id
            while current is not None:
                path.append(current)
                current = self.nodes[current].parent
            self.node_paths[target_id] = tuple(reversed(path))
        self.leaf_paths = tuple(
            self.node_paths[leaf_id]
            for leaf_id in self.leaf_ids
        )
        self.path_node_ids = tuple(dict.fromkeys(
            node_id
            for path in self.leaf_paths
            for node_id in path
        ))
        leaf_index = {
            leaf_id: index
            for index, leaf_id in enumerate(self.leaf_ids)
        }
        self.descendant_leaf_indices = {
            node_id: tuple(
                leaf_index[leaf_id]
                for leaf_id, path in zip(self.leaf_ids, self.leaf_paths)
                if node_id in path
            )
            for node_id in self.all_node_ids
        }

        node_index = {
            node_id: index
            for index, node_id in enumerate(self.all_node_ids)
        }
        path_node_mask = torch.zeros(
            len(leaf_ids),
            len(self.all_node_ids),
            device=self._device_anchor.device,
        )
        node_path_mask = torch.zeros(
            len(self.all_node_ids),
            len(self.all_node_ids),
            device=self._device_anchor.device,
        )
        for target_index, target_id in enumerate(self.all_node_ids):
            for ancestor_id in self.node_paths[target_id]:
                node_path_mask[
                    target_index,
                    node_index[ancestor_id],
                ] = 1.0
        for leaf_index, leaf_id in enumerate(leaf_ids):
            for node_id in self.leaf_paths[leaf_index]:
                path_node_mask[
                    leaf_index,
                    node_index[node_id],
                ] = 1.0
        self.path_node_mask = path_node_mask
        self.node_path_mask = node_path_mask
        if hasattr(self, "episodic_memory"):
            self.episodic_memory.sync_nodes(self.nodes)
        if getattr(self, "frontier_routing", None) is not None:
            self.frontier_routing._sync_topology()

    def get_leaf_ids(self) -> List[str]:
        return [
            node_id
            for node_id, node in self.nodes.items()
            if node.is_leaf
        ]

    def path_to_leaf(self, leaf_id: str) -> List[str]:
        if (
            leaf_id not in self.nodes
            or not self.nodes[leaf_id].is_leaf
        ):
            raise KeyError(f"Unknown leaf: {leaf_id}")
        return self.path_to_node(leaf_id)

    def path_to_node(self, node_id: str) -> List[str]:
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node: {node_id}")
        cached_paths = getattr(self, "node_paths", None)
        if cached_paths is not None and node_id in cached_paths:
            return list(cached_paths[node_id])
        path: List[str] = []
        current: Optional[str] = node_id
        while current is not None:
            path.append(current)
            current = self.nodes[current].parent
        return list(reversed(path))

    def best_leaf(
        self,
        responsibility: torch.Tensor,
    ) -> List[str]:
        if responsibility.ndim != 2:
            raise ValueError(
                "responsibility must have shape [B, L]."
            )
        if responsibility.size(1) != len(self.leaf_ids):
            raise ValueError(
                "responsibility leaf dimension does not match "
                "the current tree."
            )
        indices = (
            responsibility.argmax(dim=-1)
            .detach()
            .cpu()
            .tolist()
        )
        return [
            self.leaf_ids[index]
            for index in indices
        ]
