"""Composed latent Hawkes tree.

Responsibilities are intentionally split across:

* ``TreeTopology``: dynamic nodes, paths, masks, and checkpoint topology;
* ``TreeRouting``: local active-frontier child compatibility;
* ``TreeSemantics``: HyperNet parameters and semantic offsets;
* this module: component construction, Encoder alignment, Memory, and forward.

The mixin design keeps every trainable module directly on ``HawkesTree`` so
existing checkpoint keys and external imports remain compatible.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from Config import TreeNode
from MemoryResiduals import TreeEpisodicMemory, WorkingMemoryAdapter
from TreeRouting import (
    ExpansionEvidencePredictor,
    FrontierRoutingOutput,
    NodeSemanticCompatibility,
    TreeRoutingMixin,
)
from TreeSemantics import (
    HawkesHyperNet,
    HawkesParamPack,
    TreeSemanticsMixin,
)
from TreeTopology import TreeTopologyMixin
try:
    from Routing_Retrieval_Investigation.routing_retrieval_investigation import (
        FrontierRoutingConfig,
        FrontierRoutingRetrieval,
        FrontierStaticCache,
        PackedFrontierBatch,
    )
except ModuleNotFoundError as error:
    # The server checkout uses the shorter top-level directory name
    # ``Routing_Retrieval`` for the same package. Keep both layouts valid so
    # checkpoints and training commands do not depend on a local folder name.
    if error.name != "Routing_Retrieval_Investigation":
        raise
    from Routing_Retrieval.routing_retrieval_investigation import (
        FrontierRoutingConfig,
        FrontierRoutingRetrieval,
        FrontierStaticCache,
        PackedFrontierBatch,
    )


class HawkesTree(
    TreeTopologyMixin,
    TreeRoutingMixin,
    TreeSemanticsMixin,
    nn.Module,
):
    """Dynamic semantic Hawkes tree with sparse episodic Memory."""

    path_node_mask: torch.Tensor

    def __init__(
        self,
        z_dim: int,
        node_dim: int,
        num_event_types: int,
        num_basis: int,
        init_depth: int = 1,
        temperature: float = 1.0,
        hyper_hidden_dim: int = 256,
        memory_key_dim: Optional[int] = None,
        memory_capacity_per_node: int = 128,
        working_rho: float = 0.8,
        working_eta: float = 1e-2,
    ) -> None:
        super().__init__()
        if z_dim <= 0:
            raise ValueError("z_dim must be positive.")
        if node_dim <= 0:
            raise ValueError("node_dim must be positive.")
        if init_depth < 0:
            raise ValueError("init_depth must be non-negative.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.z_dim = z_dim
        self.node_dim = node_dim
        self.num_event_types = num_event_types
        self.num_basis = num_basis
        self.param_dim = (
            num_event_types
            + num_event_types * num_event_types * num_basis
        )
        self.temperature = float(temperature)

        # Direct registration is required for historical state-dict paths.
        self.node_emb = nn.ParameterDict()
        self.semantic_offset = nn.ParameterDict()
        self.nodes: Dict[str, TreeNode] = {}
        self.router_compat = NodeSemanticCompatibility(
            z_dim,
            node_dim,
        )
        self.expansion_predictor = ExpansionEvidencePredictor(
            z_dim,
            node_dim,
        )
        self.semantic_blend = 0.0
        self.initialization_metadata: Dict[str, object] = {}
        self.mass_ema: Dict[str, float] = {}
        self.low_mass_streak: Dict[str, int] = {}
        # Structural-prune persistence is based on prediction-compression
        # evidence, not routing mass. Memory reconciliation marks collapsed
        # banks that temporarily exceed the ordinary per-node capacity.
        self.topology_prune_streak: Dict[str, int] = {}
        # A near-zero predictive-damage estimate is ambiguous on its first
        # observation: the refinement may be useless, or replay may still be
        # too sparse to reveal its value. Track that two-hit confirmation
        # independently from the ordinary prune-score persistence.
        self.topology_prune_near_zero_streak: Dict[str, int] = {}
        self.memory_reconciliation: Dict[str, object] = {}
        # Cold-start residual prototypes live in the unscaled Hawkes-gradient
        # space.  They are a fixed teacher for the training-only regional
        # probe and must not be reconstructed from scaled semantic offsets.
        self.residual_probe_leaf_ids: tuple[str, ...] = ()
        self.residual_probe_prototypes = torch.empty(0, self.param_dim)
        self.residual_probe_target_mass = torch.empty(0)

        self.register_buffer(
            "_device_anchor",
            torch.empty(0),
            persistent=False,
        )
        for buffer_name in (
            "path_node_mask",
            "node_path_mask",
        ):
            self.register_buffer(
                buffer_name,
                torch.empty(0, 0),
                persistent=False,
            )
        self.frontier_routing = None

        self._add_node("root", parent=None, depth=0)
        for _ in range(init_depth):
            for leaf_id in list(self.get_leaf_ids()):
                self.split_leaf(leaf_id, refresh=False)
        self.refresh_structure_buffers()

        self.hyper = HawkesHyperNet(
            embed_dim=node_dim,
            num_event_types=num_event_types,
            num_basis=num_basis,
            hidden_dim=hyper_hidden_dim,
        )
        memory_key_dim = (
            z_dim
            if memory_key_dim is None
            else memory_key_dim
        )
        self.episodic_memory = TreeEpisodicMemory(
            key_dim=memory_key_dim,
            query_input_dim=z_dim,
            num_event_types=num_event_types,
            num_basis=num_basis,
            capacity_per_node=memory_capacity_per_node,
            device=str(self._device_anchor.device),
        )
        self.episodic_memory.sync_nodes(self.nodes)
        self.working_memory = WorkingMemoryAdapter(
            param_dim=self.episodic_memory.param_dim,
            rho=working_rho,
            eta=working_eta,
            device=str(self._device_anchor.device),
        )
        self.frontier_routing = FrontierRoutingRetrieval(
            self,
            FrontierRoutingConfig(
                routing_temperature=float(temperature),
            ),
        )
        register_pre_hook = getattr(
            self,
            "register_load_state_dict_pre_hook",
            None,
        )
        if register_pre_hook is not None:
            register_pre_hook(self._prepare_topology_for_load)
        else:
            # Compatibility with older PyTorch releases that expose only the
            # private registration API. ``with_module=True`` preserves the
            # callback signature used by the public API above.
            try:
                self._register_load_state_dict_pre_hook(
                    self._prepare_topology_for_load,
                    with_module=True,
                )
            except TypeError:
                # Very old releases do not support ``with_module``. Adapt
                # their state_dict-first callback without changing the actual
                # topology restoration implementation.
                def legacy_topology_pre_hook(
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                ):
                    return self._prepare_topology_for_load(
                        self,
                        state_dict,
                        prefix,
                        local_metadata,
                        strict,
                        missing_keys,
                        unexpected_keys,
                        error_msgs,
                    )

                self._register_load_state_dict_pre_hook(
                    legacy_topology_pre_hook
                )

    def configure_frontier_routing(
        self,
        *,
        config: Optional[FrontierRoutingConfig] = None,
    ) -> None:
        """Configure the model's only routing/retrieval construction."""
        config = FrontierRoutingConfig() if config is None else config
        config.validate()
        self.frontier_routing._sync_topology()
        self.frontier_routing.config = config
        self.frontier_routing._reset_target_leaf_mass_from_config()

    def _frontier_router_logits(
        self,
        samples,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []
        masks = []
        for sample in samples:
            values = []
            evaluated = []
            for node_id in self.internal_ids:
                decision = sample.decisions.get(node_id)
                if decision is None:
                    values.append(reference.new_zeros(()))
                    evaluated.append(False)
                else:
                    values.append(
                        decision.total_score[0]
                        - decision.total_score[1]
                    )
                    evaluated.append(True)
            rows.append(
                torch.stack(values)
                if values
                else reference.new_empty(0)
            )
            masks.append(torch.tensor(
                evaluated,
                device=reference.device,
                dtype=torch.bool,
            ))
        return torch.stack(rows), torch.stack(masks)

    def frontier_route(
        self,
        z_t: torch.Tensor,
        *,
        update_search_state: bool = False,
    ) -> FrontierRoutingOutput:
        frontier = self.frontier_routing.route_packed(
            z_t,
            update_search_state=update_search_state,
        )
        responsibility = (
            frontier.mass * frontier.mask.to(frontier.mass.dtype)
        )
        probabilities = frontier.expanded_probability.clamp_min(1e-12)
        logits = (
            probabilities[..., 0].log()
            - probabilities[..., 1].log()
        ).masked_fill(~frontier.expanded_mask, 0.0)
        return FrontierRoutingOutput(
            responsibility=responsibility,
            log_responsibility=responsibility.clamp_min(1e-12).log(),
            router_logits=logits,
            frontier_mask=frontier.mask,
            frontier_node_indices=frontier.node_indices,
            expanded_node_indices=frontier.expanded_node_indices,
            expanded_probability=frontier.expanded_probability,
            expanded_mask=frontier.expanded_mask,
        )

    # Checkpoint-era compatibility name. The return value is an actual
    # frontier and is never projected to leaves.
    frontier_route_as_leaves = frontier_route

    def _forward_frontier(
        self,
        z_t: torch.Tensor,
        working_delta: Optional[torch.Tensor],
        decays: Optional[torch.Tensor],
        frontier_static_cache: Optional[FrontierStaticCache],
        frontier_projected_z: Optional[torch.Tensor],
        frontier_query: Optional[torch.Tensor],
        *,
        update_memory_state: bool,
        update_search_state: bool,
        detach_routing: bool,
        materialize_diagnostics: bool,
        precomputed_frontier: Optional[PackedFrontierBatch],
        precomputed_node_delta: Optional[torch.Tensor],
        precomputed_episodic_delta: Optional[torch.Tensor],
        precomputed_memory_info: Optional[Mapping[str, torch.Tensor]],
    ):
        output = self.frontier_routing(
            z_t,
            working_delta=working_delta,
            decays=decays,
            static_cache=frontier_static_cache,
            projected_z=frontier_projected_z,
            precomputed_query=frontier_query,
            update_memory_state=update_memory_state,
            update_search_state=update_search_state,
            detach_routing=detach_routing,
            materialize_diagnostics=materialize_diagnostics,
            precomputed_frontier=precomputed_frontier,
            precomputed_node_delta=precomputed_node_delta,
            precomputed_episodic_delta=precomputed_episodic_delta,
            precomputed_memory_info=precomputed_memory_info,
        )
        batch_size = z_t.size(0)
        frontier_mass = output.frontier_mass
        frontier_mask = output.frontier_mask
        width = frontier_mass.size(1)
        semantic = output.semantic_theta_packed
        episodic = output.episodic_delta_packed
        frontier_theta = output.frontier_theta_packed
        frontier_ids = []
        if materialize_diagnostics:
            frontier_ids = [sample.node_ids for sample in output.samples]

        responsibility = frontier_mass
        routing_weights = (
            responsibility.detach()
            if detach_routing
            else responsibility
        )
        expanded_probability = (
            output.frontier.expanded_probability.clamp_min(1e-12)
        )
        logits = (
            expanded_probability[..., 0].log()
            - expanded_probability[..., 1].log()
        ).masked_fill(~output.frontier.expanded_mask, 0.0)
        evaluated_mask = output.frontier.expanded_mask
        semantic_mix_theta = (
            frontier_mass.unsqueeze(-1) * semantic
        ).sum(dim=1)
        D = self.num_event_types
        theta_sem_mix = HawkesParamPack(
            mu_tilde=semantic_mix_theta[:, :D],
            W_tilde=semantic_mix_theta[:, D:].reshape(
                batch_size,
                D,
                D,
                self.num_basis,
            ),
        )
        return {
            "r": routing_weights,
            "log_r": routing_weights.clamp_min(1e-12).log(),
            "router_logits": logits,
            "router_evaluated_mask": evaluated_mask,
            "theta_sem_mix": theta_sem_mix,
            "theta_sem_leaf": None,
            "effective_params": output.effective_params,
            "episodic_delta": episodic,
            "frontier_semantic_theta": semantic,
            "frontier_episodic_delta": episodic,
            "frontier_theta": frontier_theta,
            "frontier_mass": frontier_mass,
            "frontier_mask": frontier_mask,
            "frontier_node_ids": tuple(frontier_ids),
            "frontier_node_indices": output.frontier_node_indices,
            "visited_node_indices": output.visited_node_indices,
            "visited_node_mask": output.visited_node_mask,
            "expanded_node_indices": output.frontier.expanded_node_indices,
            "expanded_child_indices": output.frontier.expanded_child_indices,
            "expanded_probability": output.frontier.expanded_probability,
            "expanded_semantic_score": (
                output.frontier.expanded_semantic_score
            ),
            "expanded_child_theta": output.expanded_child_theta,
            "expanded_mask": output.frontier.expanded_mask,
            "expansion_utility": output.frontier.expansion_utility,
            "frontier_samples": output.samples,
            "memory_query": output.query,
            "memory_node_ids": list(self.all_node_ids),
            "memory_info": output.memory_info,
            "packed_memory_info": output.packed_memory_info,
            "working_delta": output.working_delta,
            "leaf_ids": list(self.leaf_ids),
            "all_node_ids": list(self.all_node_ids),
        }

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "working_memory"):
            self.working_memory.delta = fn(
                self.working_memory.delta
            )
            self.working_memory.device = str(
                self.working_memory.delta.device
            )
        return self

    def reset_working_memory(self) -> None:
        self.working_memory.reset()

    def forward(
        self,
        z_t: torch.Tensor,
        working_delta: Optional[torch.Tensor] = None,
        decays: Optional[torch.Tensor] = None,
        frontier_static_cache: Optional[FrontierStaticCache] = None,
        frontier_projected_z: Optional[torch.Tensor] = None,
        frontier_query: Optional[torch.Tensor] = None,
        update_memory_state: bool = True,
        detach_routing: bool = False,
        update_search_state: bool = True,
        materialize_diagnostics: bool = True,
        precomputed_frontier: Optional[PackedFrontierBatch] = None,
        precomputed_node_delta: Optional[torch.Tensor] = None,
        precomputed_episodic_delta: Optional[torch.Tensor] = None,
        precomputed_memory_info: Optional[Mapping[str, torch.Tensor]] = None,
    ):
        return self._forward_frontier(
            z_t,
            working_delta,
            decays,
            frontier_static_cache,
            frontier_projected_z,
            frontier_query,
            update_memory_state=update_memory_state,
            update_search_state=update_search_state,
            detach_routing=detach_routing,
            materialize_diagnostics=materialize_diagnostics,
            precomputed_frontier=precomputed_frontier,
            precomputed_node_delta=precomputed_node_delta,
            precomputed_episodic_delta=precomputed_episodic_delta,
            precomputed_memory_info=precomputed_memory_info,
        )


__all__ = [
    "FrontierRoutingOutput",
    "HawkesHyperNet",
    "HawkesParamPack",
    "HawkesTree",
    "NodeSemanticCompatibility",
    "TreeNode",
]
