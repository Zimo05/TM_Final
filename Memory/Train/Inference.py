"""Wake-only online inference for a trained Hawkes Memory Tree."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from HawkesBackbone import (
    EVENT_TIME_FEATURES_KEY,
    HAWKES_CACHE_SIGNATURE_KEY,
    HAWKES_HISTORY_STATS_KEY,
    HAWKES_INTERVAL_STATS_KEY,
    HawkesFamily,
)
from LatentHawkesTree import HawkesTree
from Train.Train import (
    CausalPrefixEncoder,
    WakeObjectiveConfig,
    _frontier_config_from_checkpoint,
)
from Wake.HawkesParams import HawkesParams
from Wake.SequentialController import Action, Controller


@dataclass
class InferenceConfig:
    adapt_working_memory: bool = True
    allow_memory_writes: bool = True
    update_memory_usage: bool = True
    probe_write_counterfactuals: bool = False
    write_probe_random_count: int = 16
    write_probe_seed: int = 42
    # Diagnostic-only causal replay.  When set, no event outside the allowlist
    # may write; listed events bypass the learned Write threshold but still use
    # the normal candidate construction, add_memory and retrieval path.
    write_event_allowlist: Optional[tuple[int, ...]] = None


class MemoryTreeInference:
    def _controller_effective_parameters(
        self,
        memory_output: Mapping[str, Any],
        working_delta: Tensor,
        retrieval_gate: Tensor,
    ):
        semantic = memory_output["frontier_semantic_theta"]
        episodic = memory_output["frontier_episodic_delta"]
        gate = torch.as_tensor(retrieval_gate).to(episodic)
        while gate.ndim < episodic.ndim - 1:
            gate = gate.unsqueeze(-1)
        D = self.hawkes.num_types
        return self.tree.episodic_memory.parameter_update.compose_effective_parameters(
            semantic_mu=semantic[..., :D],
            semantic_W=semantic[..., D:].reshape(
                *semantic.shape[:-1], D, D, self.hawkes.num_basis
            ),
            episodic_delta=episodic * gate.unsqueeze(-1),
            routing_weights=memory_output["r"],
            working_delta=working_delta,
            decays=self.hawkes.decays,
        )

    """Route, retrieve, adapt working memory, and predict without sleep updates."""

    def __init__(
        self,
        tree: HawkesTree,
        hawkes: HawkesFamily,
        encoder: nn.Module,
        *,
        wake_config: Optional[WakeObjectiveConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.wake_config = (
            WakeObjectiveConfig() if wake_config is None else wake_config
        )
        self.config = InferenceConfig() if inference_config is None else inference_config
        if device is None:
            device = next(tree.parameters()).device
        self.device = torch.device(device)
        self.tree = tree.to(self.device).eval()
        self.hawkes = hawkes.to(self.device).eval()
        self.encoder = encoder.to(self.device).eval()
        self.controller = Controller(
            nll_fn=self.hawkes,
            tau_s=self.wake_config.tau_surprise,
            tau_n=self.wake_config.tau_novelty,
            tau_c=self.wake_config.tau_count,
            tau_sim=self.wake_config.tau_similarity,
            eta_mem=self.wake_config.eta_memory_write,
            memory_write_grad_clip=self.wake_config.memory_write_grad_clip,
            write_horizon=self.wake_config.write_horizon,
            episodic_memory=self.tree.episodic_memory,
            working_memory=self.tree.working_memory,
            action_temperature=self.wake_config.action_temperature,
            novelty_temperature=self.wake_config.novelty_temperature,
            count_exponent=self.wake_config.count_exponent,
            count_similarity_low=self.wake_config.count_similarity_low,
            count_similarity_high=self.wake_config.count_similarity_high,
            count_topk=self.wake_config.count_topk,
            count_saturation=self.wake_config.count_saturation,
            surprise_ema_decay=self.wake_config.surprise_ema_decay,
            write_candidate_threshold=(
                self.wake_config.controller_write_admission_threshold
            ),
            # Inference is deterministic and never creates exploration writes.
            exploration_rate=0.0,
            utility_topc_multiplier=(
                self.wake_config.controller_utility_topc_multiplier
            ),
            utility_stage_enabled=(
                self.wake_config.controller_utility_stage_enabled
            ),
            utility_temperature=self.wake_config.controller_utility_temperature,
            utility_cost_margin=self.wake_config.controller_utility_cost_margin,
        ).to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: Optional[torch.device | str] = None,
        inference_config: Optional[InferenceConfig] = None,
        encoder: Optional[nn.Module] = None,
    ) -> "MemoryTreeInference":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        config = checkpoint["model_config"]
        if config.get("router_kind") not in {
            "node_semantic_compat_v1",
            "active_frontier_v1",
            "posterior_frontier_v2",
        }:
            raise ValueError(
                "checkpoint uses the retired z-only Linear Router; the current "
                "model requires Compat(z_t, u_n). Retrain before inference."
            )
        decays = torch.as_tensor(config["decays"], dtype=torch.float32)
        hawkes = HawkesFamily(
            num_types=config["num_event_types"],
            num_basis=config["num_basis"],
            decays=decays,
        ).to(device)
        tree = HawkesTree(
            z_dim=config["z_dim"],
            node_dim=config["node_dim"],
            num_event_types=config["num_event_types"],
            num_basis=config["num_basis"],
            init_depth=0,
            temperature=config.get("tree_temperature", 1.0),
            hyper_hidden_dim=config.get("hyper_hidden_dim", 256),
            memory_key_dim=config["memory_key_dim"],
            memory_capacity_per_node=config.get(
                "memory_capacity_per_node", 128
            ),
            working_rho=config.get("working_rho", 0.8),
            working_eta=config.get("working_eta", 1e-2),
        ).to(device)
        tree.configure_frontier_routing(
            config=_frontier_config_from_checkpoint(config)
        )
        if encoder is None:
            encoder_config = config.get("encoder_config", {"kind": "causal_prefix"})
            kind = encoder_config.get("kind", "causal_prefix")
            if kind == "attention_memory":
                raise ValueError(
                    "checkpoint encoder_config.kind='attention_memory' is no longer "
                    "supported; retrain with CausalPrefixEncoder and optional "
                    "--h-tree semantic initialization"
                )
            type_dim = config.get("encoder_type_dim") or 32
            hidden_dim = config.get("encoder_hidden_dim") or 128
            encoder = CausalPrefixEncoder(
                num_event_types=config["num_event_types"],
                z_dim=config["z_dim"],
                type_dim=type_dim,
                hidden_dim=hidden_dim,
            )
        encoder = encoder.to(device)
        hawkes.load_state_dict(checkpoint["hawkes_state_dict"])
        incompatible = tree.load_state_dict(
            checkpoint["tree_state_dict"],
            strict=False,
        )
        legacy_missing = {
            key
            for key in incompatible.missing_keys
            if key.startswith("frontier_routing.prototypes.")
        }
        if set(incompatible.missing_keys).difference(legacy_missing):
            raise RuntimeError(
                "checkpoint is missing model tensors: "
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "checkpoint contains unexpected model tensors: "
                f"{incompatible.unexpected_keys}"
            )
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        wake_config = WakeObjectiveConfig(**checkpoint.get("wake_config", {}))
        inference = cls(
            tree=tree,
            hawkes=hawkes,
            encoder=encoder,
            wake_config=wake_config,
            inference_config=inference_config,
            device=device,
        )
        controller_state = checkpoint.get("controller_state", {})
        module_state = controller_state.get("module_state_dict")
        if module_state is not None:
            incompatible_controller = inference.controller.load_state_dict(
                module_state, strict=False
            )
            allowed_missing = {
                "controller_version",
                "context_gate.weight",
                "bias_assimilate",
                "utility_mean",
                "utility_variance",
                "utility_observations",
                "utility_temperatures",
                "calibration_thresholds",
                "split_enabled",
            }
            if set(incompatible_controller.missing_keys).difference(
                allowed_missing
            ) or incompatible_controller.unexpected_keys:
                raise RuntimeError(
                    "checkpoint controller state is incompatible: "
                    f"missing={incompatible_controller.missing_keys}, "
                    f"unexpected={incompatible_controller.unexpected_keys}"
                )
            if controller_state.get("controller_version", 1) < 2:
                inference.controller.migrate_legacy_policy()
            if controller_state.get("controller_version", 1) < 5:
                # v2-v4 checkpoints predate the conservative v5 Split policy.
                inference.controller.split_enabled.fill_(True)
        inference.controller.utility_stage_enabled = bool(
            controller_state.get(
                "utility_stage_enabled",
                inference.controller.utility_stage_enabled,
            )
        )
        inference.controller.split_queues.update(
            controller_state.get("split_queues", {})
        )
        return inference

    def _move_sequence(self, sequence: Mapping[str, Tensor]) -> Dict[str, Any]:
        cached = self.hawkes.prepare_sequence_cache(sequence, inplace=True)
        result = {
            "times": cached["times"].to(self.device),
            "types": cached["types"].to(self.device).long(),
            EVENT_TIME_FEATURES_KEY: cached[EVENT_TIME_FEATURES_KEY].to(
                self.device
            ),
            HAWKES_HISTORY_STATS_KEY: cached[HAWKES_HISTORY_STATS_KEY].to(
                self.device
            ),
            HAWKES_INTERVAL_STATS_KEY: cached[HAWKES_INTERVAL_STATS_KEY].to(
                self.device
            ),
            HAWKES_CACHE_SIGNATURE_KEY: cached[HAWKES_CACHE_SIGNATURE_KEY],
        }
        if "T" in sequence:
            result["T"] = sequence["T"].to(self.device)
        return result

    def _action(
        self,
        memory_output: Mapping[str, Any],
        nll: Tensor,
        frontier_energy: Tensor,
    ) -> tuple[Action, str, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        mask = memory_output["frontier_mask"][0]
        prior = memory_output["frontier_mass"][0]
        temperature = (
            self.tree.frontier_routing.config.posterior_temperature
        )
        posterior = torch.softmax(
            (
                prior.clamp_min(1e-12).log()
                - frontier_energy / temperature
            ).masked_fill(~mask, -torch.inf),
            dim=-1,
        ).masked_fill(~mask, 0.0)
        frontier_ids = memory_output["frontier_node_ids"][0]
        owner_id = self._posterior_owner(frontier_ids, posterior)
        query = memory_output["memory_query"][0].detach()
        novelty, count, retrieval_similarity = (
            self.controller.leaf_novelty_count(
            query, owner_id
            )
        )
        controller_output = self.controller.action_distribution(
            surprise=nll.detach(),
            novelty=novelty,
            count=count,
            update_statistics=False,
            owner_confidence=posterior.max(),
            retrieval_similarity=retrieval_similarity,
            retrieval_residual_norm=memory_output[
                "frontier_episodic_delta"
            ][0].norm(dim=-1).mean(),
        )
        probabilities = controller_output["probabilities"]
        raw_probabilities = controller_output.get("raw_probabilities", probabilities)
        action = tuple(Action)[
            int(probabilities.detach().argmax().item())
        ]
        return (
            action,
            owner_id,
            query,
            probabilities,
            raw_probabilities,
            posterior,
            novelty.detach(),
            retrieval_similarity.detach(),
        )

    def _lowest_common_ancestor(self, node_ids: Sequence[str]) -> str:
        paths = [self.tree.path_to_node(node_id) for node_id in node_ids]
        common = "root"
        for values in zip(*paths):
            if len(set(values)) != 1:
                break
            common = values[0]
        return common

    def _posterior_owner(
        self,
        frontier_ids: Sequence[str],
        posterior: Tensor,
    ) -> str:
        config = self.tree.frontier_routing.config
        order = posterior[: len(frontier_ids)].argsort(descending=True)
        cumulative = posterior.index_select(0, order).cumsum(dim=0)
        count = min(
            int((cumulative < config.credible_mass).sum().item()) + 1,
            len(frontier_ids),
        )
        top = int(order[0].item())
        if (
            count == 1
            and float(posterior[top]) >= config.owner_confidence_threshold
        ):
            return frontier_ids[top]
        return self._lowest_common_ancestor([
            frontier_ids[index]
            for index in order[:count].cpu().tolist()
        ])

    def _frontier_event_energy(
        self,
        sequence: Mapping[str, Tensor],
        memory_output: Mapping[str, Any],
        event_index: int,
    ) -> Tensor:
        theta = memory_output["frontier_theta"][0]
        D = self.hawkes.num_types
        energies = []
        for row in theta:
            params = HawkesParams(
                row[:D],
                row[D:].reshape(D, D, self.hawkes.num_basis),
            )
            energies.append(
                self.hawkes.event_NLL(sequence, params, event_index)
            )
        result = torch.stack(energies)
        return result.masked_fill(
            ~memory_output["frontier_mask"][0], torch.inf
        )

    def _delayed_write_evidence(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if int(self.controller.controller_version.detach().cpu()) >= 6:
            return self._delayed_write_evidence_v6(sequence, request)
        D = self.hawkes.num_types
        start = int(request["event_index"])
        end = min(
            int(sequence["times"].numel()),
            start + self.wake_config.write_horizon + 1,
        )
        energy = []
        for theta in request["frontier_theta"]:
            params = HawkesParams(
                theta[:D],
                theta[D:].reshape(D, D, self.hawkes.num_basis),
            )
            value = theta.new_zeros(())
            for event_index in range(start, end):
                value += self.hawkes.event_NLL(
                    sequence, params, event_index
                )
            energy.append(value / max(end - start, 1))
        energy_tensor = torch.stack(energy)
        prior = request["frontier_mass"].clamp_min(1e-12)
        temperature = (
            self.tree.frontier_routing.config.posterior_temperature
        )
        posterior = torch.softmax(
            prior.log() - energy_tensor / temperature,
            dim=-1,
        )
        owner_id = self._posterior_owner(
            request["frontier_node_ids"], posterior
        )
        mixture_energy = -torch.logsumexp(
            prior.log() - energy_tensor / temperature,
            dim=0,
        )
        theta_owner = self.tree.semantic_theta(owner_id).detach()
        owner_params = HawkesParams(
            theta_owner[:D].clone(),
            theta_owner[D:].reshape(D, D, self.hawkes.num_basis).clone(),
        )
        candidate_item = self.controller.write_residual_memory(
            q_t=request["query"],
            theta_sem_leaf=owner_params,
            times=sequence["times"],
            types=sequence["types"],
            k=start,
            node_id=owner_id,
            cached_sequence=sequence,
            write_quality=request["write_gate"],
            queue_weight=request["queue_weight"],
        )
        delta = candidate_item.delta_theta
        if int(self.controller.controller_version.detach().cpu()) < 5:
            before = energy_tensor.new_zeros(())
            after = energy_tensor.new_zeros(())
            candidate_theta = theta_owner + delta
            candidate_params = HawkesParams(
                candidate_theta[:D],
                candidate_theta[D:].reshape(D, D, self.hawkes.num_basis),
            )
            for future_index in range(start, end):
                before += self.hawkes.event_NLL(sequence, owner_params, future_index)
                after += self.hawkes.event_NLL(sequence, candidate_params, future_index)
            raw_write_gain = before - after
            write_utility = raw_write_gain / max(end - start, 1) - self.wake_config.lambda_write
            write_gain = raw_write_gain.clamp_min(0.0)
            bounded_gain = -torch.expm1(
                -write_gain / self.wake_config.controller_gain_reference
            )
            return {
                "owner_id": owner_id,
                "posterior": posterior,
                "write_gain": write_gain,
                "write_utility": write_utility,
                "owner_on_score_path": True,
                "virtual_candidate_alpha": 0.0,
                "bounded_gain": bounded_gain,
                "candidate_item": candidate_item,
                "priority": (
                    request["write_gate"] * posterior.max()
                    * write_utility.clamp_min(0.0)
                    * request["novelty"].clamp(0.0, 1.0)
                ),
            }
        score_start = start + self.wake_config.write_horizon + 1
        score_end = start + 2 * self.wake_config.write_horizon + 1
        contexts = request.get("future_contexts") or []
        if score_end > len(contexts):
            raise RuntimeError("Write probe score window is incomplete")
        before = energy_tensor.new_zeros(())
        after = energy_tensor.new_zeros(())
        virtual_usage = 1.0
        virtual_alpha = []
        owner_on_path = False
        for age, context in enumerate(contexts[score_start:score_end]):
            query = context["query"]
            base_delta, _ = self.tree.episodic_memory.read_nodes(
                query, [owner_id], update_state=False
            )
            virtual_delta, virtual_info = self.tree.episodic_memory.read_node_with_virtual_item(
                query, owner_id, key=candidate_item.key, delta=delta,
                write_quality=request["write_gate"], virtual_usage=virtual_usage,
                virtual_age=float(age),
            )
            candidate_alpha = float(virtual_info["alpha"][-1].detach().cpu())
            virtual_alpha.append(candidate_alpha)
            virtual_usage += candidate_alpha
            episodic = context["frontier_episodic_delta"].clone()
            difference = virtual_delta - base_delta[owner_id]
            for slot, frontier_id in enumerate(context["frontier_node_ids"]):
                if owner_id in self.tree.path_to_node(frontier_id):
                    episodic[slot] = episodic[slot] + difference
                    owner_on_path = True
            virtual_output = {
                "frontier_semantic_theta": context["frontier_semantic_theta"].unsqueeze(0),
                "frontier_episodic_delta": episodic.unsqueeze(0),
                "r": context["posterior"].unsqueeze(0),
            }
            with_params = self._controller_effective_parameters(
                virtual_output, context["working_delta"], context["retrieve_gate"]
            ).select(0)
            theta = context["no_write_theta"]
            without_params = HawkesParams(
                theta[:D], theta[D:].reshape(D, D, self.hawkes.num_basis)
            )
            event_index = int(context["event_index"])
            before += self.hawkes.event_NLL(sequence, without_params, event_index)
            after += self.hawkes.event_NLL(sequence, with_params, event_index)
        raw_write_gain = before - after
        if not owner_on_path:
            raw_write_gain = raw_write_gain * 0.0
        write_utility = (
            raw_write_gain / self.wake_config.write_horizon
            - self.wake_config.lambda_write
        )
        write_gain = raw_write_gain.clamp_min(0.0)
        bounded_gain = -torch.expm1(
            -write_gain / self.wake_config.controller_gain_reference
        )
        priority = (
            request["write_gate"]
            * posterior.max()
            * write_utility.clamp_min(0.0)
            * request["novelty"].clamp(0.0, 1.0)
        )
        return {
            "owner_id": owner_id,
            "posterior": posterior,
            "write_gain": write_gain,
            "write_utility": write_utility,
            "owner_on_score_path": owner_on_path,
            "virtual_candidate_alpha": sum(virtual_alpha) / max(len(virtual_alpha), 1),
            "bounded_gain": bounded_gain,
            "candidate_item": candidate_item,
            "priority": priority,
        }

    def _causal_write_evidence_v6(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Build a candidate from exactly F=[t,t+h), without future labels."""
        D = self.hawkes.num_types
        h = self.wake_config.write_horizon
        start = int(request["event_index"])
        build_end = start + h
        if build_end > int(sequence["times"].numel()):
            raise RuntimeError("v6 construction window is incomplete")
        energy = []
        for theta in request["frontier_theta"]:
            params = HawkesParams(
                theta[:D], theta[D:].reshape(D, D, self.hawkes.num_basis)
            )
            value = theta.new_zeros(())
            for event_index in range(start, build_end):
                value = value + self.hawkes.event_NLL(sequence, params, event_index)
            energy.append(value / h)
        energy_tensor = torch.stack(energy)
        prior = request["frontier_mass"].clamp_min(1e-12)
        posterior = torch.softmax(
            prior.log() - energy_tensor / self.tree.frontier_routing.config.posterior_temperature,
            dim=-1,
        )
        owner_id = self._posterior_owner(request["frontier_node_ids"], posterior)
        theta_owner = self.tree.semantic_theta(owner_id).detach()
        owner_params = HawkesParams(
            theta_owner[:D].clone(),
            theta_owner[D:].reshape(D, D, self.hawkes.num_basis).clone(),
        )
        candidate_item = self.controller.write_residual_memory(
            q_t=request["query"], theta_sem_leaf=owner_params,
            times=sequence["times"], types=sequence["types"], k=start,
            node_id=owner_id, cached_sequence=sequence,
            write_quality=request["write_gate"], queue_weight=request["queue_weight"],
            window_events=h,
        )
        threshold = self.controller.calibration_thresholds[2].to(request["write_gate"])
        priority = (
            (request["write_gate"] - threshold).clamp_min(0.0)
            * posterior.max() * request["novelty"].clamp(0.0, 1.0)
        )
        return {
            "owner_id": owner_id,
            "posterior": posterior,
            "candidate_item": candidate_item,
            "priority": priority,
            "bounded_gain": request["write_gate"].clamp(0.0, 1.0),
            "construction_window": [start, build_end],
        }

    def _delayed_write_evidence_v6(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Score v6 candidate on C=[t+h,t+2h), after causal admission."""
        base = dict(request.get("admission_evidence") or self._causal_write_evidence_v6(sequence, request))
        D = self.hawkes.num_types
        h = self.wake_config.write_horizon
        start = int(request["event_index"])
        score_start, score_end = start + h, start + 2 * h
        contexts = request.get("future_contexts") or []
        if score_end > len(contexts):
            raise RuntimeError("v6 score window C is incomplete")
        owner_id = base["owner_id"]
        item = base["candidate_item"]
        before = request["frontier_theta"].new_zeros(())
        after = before.clone()
        virtual_usage = 1.0
        virtual_alpha: list[float] = []
        owner_on_path = False
        committed = bool(request.get("committed", False))
        for age, context in enumerate(contexts[score_start:score_end]):
            query = context["query"]
            stored_episodic = context["frontier_episodic_delta"].clone()
            if committed:
                with_delta, with_info = self.tree.episodic_memory.read_nodes(
                    query, [owner_id], update_state=False
                )
                without_delta, without_info = self.tree.episodic_memory.read_node_without_item(
                    query, owner_id, key=item.key, delta=item.delta_theta
                )
                difference = with_delta[owner_id] - without_delta
                excluded = without_info.get("excluded_index")
                alpha = 0.0
                if excluded is not None and with_info[owner_id].get("alpha") is not None:
                    index = int(excluded.detach().cpu())
                    values = with_info[owner_id]["alpha"]
                    if 0 <= index < values.numel():
                        alpha = float(values[index].detach().cpu())
                virtual_alpha.append(alpha)
                baseline_episodic = stored_episodic.clone()
                for slot, frontier_id in enumerate(context["frontier_node_ids"]):
                    if owner_id in self.tree.path_to_node(frontier_id):
                        baseline_episodic[slot] = baseline_episodic[slot] - difference
                        owner_on_path = True
                baseline_output = {
                    "frontier_semantic_theta": context["frontier_semantic_theta"].unsqueeze(0),
                    "frontier_episodic_delta": baseline_episodic.unsqueeze(0),
                    "r": context["posterior"].unsqueeze(0),
                }
                before_params = self._controller_effective_parameters(
                    baseline_output, context["working_delta"], context["retrieve_gate"]
                ).select(0)
                theta = context["no_write_theta"]
                after_params = HawkesParams(
                    theta[:D], theta[D:].reshape(D, D, self.hawkes.num_basis)
                )
            else:
                # Training/read-only probes use the production add_memory and
                # sparse read path on an isolated bank snapshot.  This avoids
                # a second, approximate "virtual retriever" implementation.
                memory = self.tree.episodic_memory
                if age == 0:
                    baseline_memory = copy.deepcopy(memory)
                    treatment_memory = copy.deepcopy(memory)
                    base_by_age = []
                    original_clock = baseline_memory._age_clock
                    for base_age, base_context in enumerate(
                        contexts[score_start:score_end]
                    ):
                        baseline_memory._age_clock = original_clock + base_age
                        values, _ = baseline_memory.read_nodes(
                            base_context["query"], [owner_id], update_state=False
                        )
                        base_by_age.append(values[owner_id].detach().clone())
                    treatment_memory.add_memory(
                        owner_id, item.key, item.delta_theta,
                        write_quality=base["bounded_gain"],
                        queue_weight=request["queue_weight"],
                    )
                    request["_physical_probe_memory"] = treatment_memory
                    request["_physical_probe_clock"] = original_clock
                    request["_physical_probe_base_delta"] = base_by_age
                probe_memory = request["_physical_probe_memory"]
                probe_memory._age_clock = request["_physical_probe_clock"] + age
                with_delta, info_by_node = probe_memory.read_nodes(
                    query, [owner_id], update_state=False
                )
                info = info_by_node[owner_id]
                alpha_values = info.get("alpha")
                alpha = 0.0
                if alpha_values is not None and alpha_values.numel():
                    alpha = float(alpha_values[-1].detach().cpu())
                virtual_alpha.append(alpha)
                difference = (
                    with_delta[owner_id]
                    - request["_physical_probe_base_delta"][age]
                )
                virtual_episodic = stored_episodic.clone()
                for slot, frontier_id in enumerate(context["frontier_node_ids"]):
                    if owner_id in self.tree.path_to_node(frontier_id):
                        virtual_episodic[slot] = virtual_episodic[slot] + difference
                        owner_on_path = True
                virtual_output = {
                    "frontier_semantic_theta": context["frontier_semantic_theta"].unsqueeze(0),
                    "frontier_episodic_delta": virtual_episodic.unsqueeze(0),
                    "r": context["posterior"].unsqueeze(0),
                }
                after_params = self._controller_effective_parameters(
                    virtual_output, context["working_delta"], context["retrieve_gate"]
                ).select(0)
                theta = context["no_write_theta"]
                before_params = HawkesParams(
                    theta[:D], theta[D:].reshape(D, D, self.hawkes.num_basis)
                )
            event_index = int(context["event_index"])
            before = before + self.hawkes.event_NLL(sequence, before_params, event_index)
            after = after + self.hawkes.event_NLL(sequence, after_params, event_index)
        if not committed and "_physical_probe_memory" in request:
            request.pop("_physical_probe_memory")
            request.pop("_physical_probe_clock", None)
            request.pop("_physical_probe_base_delta", None)
        raw_gain = before - after
        if not owner_on_path:
            raw_gain = raw_gain * 0.0
        utility = raw_gain / h - self.wake_config.lambda_write
        return {
            **base,
            "write_gain": raw_gain,
            "write_utility": utility,
            "owner_on_score_path": owner_on_path,
            "virtual_candidate_alpha": sum(virtual_alpha) / max(len(virtual_alpha), 1),
            "score_window": [score_start, score_end],
        }

    def _commit_delayed_write(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
    ) -> None:
        evidence = request.get("window_evidence")
        if evidence is None:
            evidence = self._delayed_write_evidence(sequence, request)
        owner_id = evidence["owner_id"]
        item = evidence.get("candidate_item")
        if item is None:
            raise RuntimeError("write evidence is missing its residual candidate")
        item.write_quality = float(evidence["bounded_gain"].detach().cpu())
        item.queue_weight = float(request["queue_weight"])
        self.tree.episodic_memory.add_memory(
            owner_id,
            item.key,
            item.delta_theta,
            item.window,
            write_quality=item.write_quality,
            queue_weight=item.queue_weight,
        )
        self.controller.split_queues[
            owner_id
        ] += item.queue_weight

    def run_sequence(
        self,
        cpu_sequence: Mapping[str, Tensor],
    ) -> Dict[str, Any]:
        """Process observed events causally using only the wake mechanism.

        Memory writes are delayed until their future write horizon has actually
        been observed, preventing offline look-ahead leakage during inference.
        """
        source_value = cpu_sequence.get("source_index", -1)
        source_index = int(
            source_value.item() if hasattr(source_value, "item") else source_value
        )
        sequence = self._move_sequence(cpu_sequence)
        self.tree.reset_working_memory()
        pending_writes: list[Dict[str, Any]] = []
        write_probe_contexts: list[Dict[str, Any]] = []
        outputs = []
        total_nll = 0.0
        accepted_write_count = 0
        accepted_write_requests: list[Dict[str, Any]] = []

        for event_index in range(sequence["times"].numel()):
            with torch.no_grad():
                if isinstance(self.encoder, CausalPrefixEncoder):
                    z_t = self.encoder(
                        sequence["times"],
                        sequence["types"],
                        event_index,
                        time_features=sequence.get(
                            EVENT_TIME_FEATURES_KEY
                        ),
                    ).reshape(1, -1)
                else:
                    z_t = self.encoder(
                        sequence["times"],
                        sequence["types"],
                        event_index,
                    ).reshape(1, -1)
            working_delta = self.tree.working_memory.make_trainable_delta()
            if not self.config.adapt_working_memory:
                working_delta = working_delta.detach()

            with torch.set_grad_enabled(self.config.adapt_working_memory):
                memory_output = self.tree(
                    z_t=z_t,
                    working_delta=working_delta,
                    decays=self.hawkes.decays,
                    update_memory_state=False,
                )
                pre_action_params = self._controller_effective_parameters(
                    memory_output,
                    working_delta,
                    working_delta.new_zeros(()),
                ).select(0)
                nll = self.hawkes.event_NLL(
                    sequence, pre_action_params, event_index
                )
                # The working-memory gradient is evaluated after the retrieval
                # gate has recomposed the final effective parameters below.
            if self.config.update_memory_usage:
                # Retrieval uses age in its differentiable scores; mutate it
                # only after the event's working-memory gradient is complete.
                self.tree.episodic_memory.step_age()

            with torch.no_grad():
                # A causal local-rate forecast made at the end of the prefix.
                # This is distinct from ``intensity_at_cached_event`` below,
                # which conditions mark prediction on the observed event time.
                forecast_origin = (
                    sequence["times"].new_tensor(0.0)
                    if event_index == 0
                    else sequence["times"][event_index - 1]
                )
                frontier_energy = self._frontier_event_energy(
                    sequence, memory_output, event_index
                )
                (
                    action,
                    owner_id,
                    query,
                    action_probabilities,
                    raw_action_probabilities,
                    posterior,
                    novelty,
                    retrieval_similarity,
                ) = self._action(
                    memory_output, nll, frontier_energy
                )
                with torch.set_grad_enabled(self.config.adapt_working_memory):
                    params = self._controller_effective_parameters(
                        memory_output,
                        working_delta,
                        action_probabilities[1],
                    ).select(0)
                    nll = self.hawkes.event_NLL(
                        sequence, params, event_index
                    )
                    if self.config.adapt_working_memory:
                        working_grad = torch.autograd.grad(
                            nll, working_delta
                        )[0]
                forecast_intensity = self.hawkes.intensity_at_event(
                    {
                        "times": sequence["times"][:event_index],
                        "types": sequence["types"][:event_index],
                    },
                    forecast_origin + forecast_origin.new_tensor(1e-6),
                    params,
                )
                forecast_rate = forecast_intensity.sum().clamp_min(1e-8)
                forecast_type_probabilities = forecast_intensity / forecast_rate
                predicted_delta = forecast_rate.reciprocal()
                predicted_time = forecast_origin + predicted_delta
                intensity = self.hawkes.intensity_at_cached_event(
                    sequence, event_index, params
                )
                type_probabilities_at_event_time = (
                    intensity / intensity.sum().clamp_min(1e-8)
                )
                predicted_type = int(intensity.argmax().item())
                if self.config.adapt_working_memory:
                    self.tree.working_memory.update_from_gradient(
                        working_grad,
                        adaptation_probability=action_probabilities[0],
                    )
                if self.config.update_memory_usage:
                    self.tree.episodic_memory.credit_retrieval(
                        info_by_batch=memory_output["memory_info"],
                        leaf_paths=[
                            self.tree.path_to_node(node_id)
                            for node_id in memory_output[
                                "frontier_node_ids"
                            ][0]
                        ],
                        routing_weights=posterior.unsqueeze(0),
                        retrieval_probability=action_probabilities[1],
                    )
                selected_memory_info = (
                    memory_output["memory_info"][0]
                    if memory_output.get("memory_info")
                    else {}
                )
                for accepted_request in accepted_write_requests:
                    if accepted_request.get("retrieved_later", False):
                        continue
                    accepted_event = int(accepted_request["event_index"])
                    if event_index <= accepted_event:
                        continue
                    evidence = accepted_request.get("admission_evidence") or {}
                    owner = evidence.get("owner_id")
                    item = evidence.get("candidate_item")
                    info = selected_memory_info.get(owner, {})
                    bank = self.tree.episodic_memory.banks.get(owner)
                    alpha = info.get("alpha")
                    if item is None or bank is None or alpha is None or not len(bank):
                        continue
                    key = item.key.to(bank.keys)
                    delta = item.delta_theta.to(bank.deltas)
                    matches = torch.isclose(
                        bank.keys, key.unsqueeze(0), rtol=1e-5, atol=1e-7
                    ).all(dim=-1) & torch.isclose(
                        bank.deltas, delta.unsqueeze(0), rtol=1e-5, atol=1e-7
                    ).all(dim=-1)
                    indices = torch.nonzero(matches, as_tuple=False).flatten()
                    if indices.numel() and bool(
                        (alpha.index_select(0, indices.to(alpha.device)) > 1e-6)
                        .any().detach().cpu()
                    ):
                        accepted_request["retrieved_later"] = True
                        outputs[accepted_event]["write_retrieved_later"] = True
                packed_memory_info = memory_output.get(
                    "packed_memory_info", {}
                )
                retrieval_alpha_mass = sum(
                    float(info["alpha"].sum().detach().cpu())
                    for info in selected_memory_info.values()
                    if "alpha" in info
                )
                retrieval_effective_k = sum(
                    int(info["effective_k"].sum().detach().cpu())
                    for info in selected_memory_info.values()
                    if "effective_k" in info
                )
                packed_null_alpha = packed_memory_info.get("null_alpha")
                if packed_null_alpha is not None:
                    packed_null_alpha = packed_null_alpha[0]
                    visited_mask = memory_output["visited_node_mask"][0]
                    null_values = packed_null_alpha[visited_mask]
                    retrieval_null_alpha = (
                        float(null_values.mean().detach().cpu())
                        if null_values.numel() else 1.0
                    )
                else:
                    retrieval_null_alpha = 1.0

                visited_mask = memory_output["visited_node_mask"][0]
                visited_indices = memory_output["visited_node_indices"][0][
                    visited_mask
                ]
                visited_node_ids = tuple(
                    memory_output["memory_node_ids"][int(index)]
                    for index in visited_indices.detach().cpu().tolist()
                )
                visited_bank_count = len(visited_node_ids)
                visited_nonempty_bank_count = sum(
                    bool(
                        node_id in self.tree.episodic_memory.banks
                        and len(self.tree.episodic_memory.banks[node_id]) > 0
                    )
                    for node_id in visited_node_ids
                )
                raw_episodic_residual_norm = float(
                    memory_output["frontier_episodic_delta"][0]
                    .detach().norm(dim=-1).mean().cpu()
                )
                retrieve_gate = float(action_probabilities[1].detach().cpu())
                gated_episodic_residual_norm = (
                    raw_episodic_residual_norm * abs(retrieve_gate)
                )
                owner_on_retrieval_path = owner_id in set(visited_node_ids)

            frontier_ids = tuple(memory_output["frontier_node_ids"][0])
            write_probe_contexts.append({
                "event_index": int(event_index),
                "query": query.detach().clone(),
                "frontier_node_ids": frontier_ids,
                "frontier_semantic_theta": memory_output["frontier_semantic_theta"][
                    0, :len(frontier_ids)
                ].detach().clone(),
                "frontier_episodic_delta": memory_output["frontier_episodic_delta"][
                    0, :len(frontier_ids)
                ].detach().clone(),
                "posterior": posterior[:len(frontier_ids)].detach().clone(),
                "working_delta": working_delta.detach().clone(),
                "retrieve_gate": action_probabilities[1].detach().clone(),
                "no_write_theta": params.theta.detach().clone(),
            })

            if (
                self.config.allow_memory_writes
                or self.config.probe_write_counterfactuals
            ):
                pending_writes.append({
                    "event_index": event_index,
                    "ready_index": (
                        event_index + 2 * self.wake_config.write_horizon - 1
                        if int(self.controller.controller_version.detach().cpu()) >= 6
                        else event_index + (
                            2 if int(self.controller.controller_version.detach().cpu()) >= 5 else 1
                        ) * self.wake_config.write_horizon
                    ),
                    "admission_index": event_index + self.wake_config.write_horizon - 1,
                    "query": query,
                    "frontier_node_ids": frontier_ids,
                    "frontier_mass": memory_output["frontier_mass"][
                        0, : len(frontier_ids)
                    ].detach().clone(),
                    "frontier_theta": memory_output["frontier_theta"][
                        0, : len(frontier_ids)
                    ].detach().clone(),
                    "action": action,
                    "write_gate": raw_action_probabilities[2].detach(),
                    "exploration": False,
                    "novelty": novelty,
                    "queue_weight": float(
                        self.controller.queue_weight(
                            raw_action_probabilities
                        ).detach().cpu()
                    ),
                    "future_contexts": write_probe_contexts,
                })

            total_nll += float(nll.detach().cpu())
            outputs.append({
                "event_index": event_index,
                "nll": float(nll.detach().cpu()),
                "predicted_type": predicted_type,
                "true_type": int(sequence["types"][event_index].item()),
                "intensity": intensity.detach().cpu(),
                "type_probabilities_at_event_time": (
                    type_probabilities_at_event_time.detach().cpu()
                ),
                "forecast_type_probabilities": (
                    forecast_type_probabilities.detach().cpu()
                ),
                "predicted_delta": float(predicted_delta.detach().cpu()),
                "predicted_time": float(predicted_time.detach().cpu()),
                "true_time": float(
                    sequence["times"][event_index].detach().cpu()
                ),
                "responsibility": memory_output["r"][0].detach().cpu(),
                "owner_id": owner_id,
                "frontier_node_ids": tuple(
                    memory_output["frontier_node_ids"][0]
                ),
                "frontier_posterior": posterior.detach().cpu(),
                "action": str(action),
                "memorize_argmax": action == Action.MEMORIZE,
                "action_probabilities": (
                    action_probabilities.detach().cpu()
                ),
                "raw_action_probabilities": (
                    raw_action_probabilities.detach().cpu()
                ),
                "retrieval_alpha_mass": retrieval_alpha_mass,
                "retrieval_alpha_per_visited_node": (
                    retrieval_alpha_mass / visited_bank_count
                    if visited_bank_count else 0.0
                ),
                "retrieval_similarity": float(
                    retrieval_similarity.detach().cpu()
                ),
                "retrieval_effective_k": retrieval_effective_k,
                "retrieval_null_alpha": retrieval_null_alpha,
                "visited_bank_count": visited_bank_count,
                "visited_nonempty_bank_count": visited_nonempty_bank_count,
                "raw_episodic_residual_norm": raw_episodic_residual_norm,
                "retrieve_gate": retrieve_gate,
                "gated_episodic_residual_norm": (
                    gated_episodic_residual_norm
                ),
                "owner_on_retrieval_path": owner_on_retrieval_path,
                "retrieval_counterfactual_gain": None,
                "retrieval_counterfactual_unavailable_reason": (
                    "requires paired no_episodic evaluation"
                ),
                "episodic_residual_norm": float(
                    memory_output["episodic_delta"][0]
                    .detach().norm().cpu()
                ),
                "write_token": (
                    f"{source_index}:{int(event_index)}"
                ),
                "write_candidate": bool(
                    float(raw_action_probabilities[2].detach().cpu())
                    >= self.controller.write_candidate_threshold
                ),
                "write_gate_active": bool(
                    float(action_probabilities[2].detach().cpu()) > 0.0
                ),
                "write_gate_passed": bool(
                    float(raw_action_probabilities[2].detach().cpu())
                    >= float(self.controller.calibration_thresholds[2].detach().cpu())
                ),
                "write_priority_passed": False,
                "write_window_complete": False,
                "write_accepted": False,
                "write_retrieved_later": False,
                "write_beneficial": False,
            })

            # v6 commits immediately after F=[t,t+h) becomes observable.  The
            # later C-window utility remains diagnostic/training-only.
            if (
                int(self.controller.controller_version.detach().cpu()) >= 6
                and self.config.allow_memory_writes
                and accepted_write_count < min(
                    4, self.tree.frontier_routing.config.max_writes_per_sequence
                )
            ):
                due = [
                    request for request in pending_writes
                    if request["admission_index"] == event_index
                    and not request.get("admission_evaluated", False)
                ]
                for request in due:
                    evidence = self._causal_write_evidence_v6(sequence, request)
                    allowlist = self.config.write_event_allowlist
                    if (
                        allowlist is not None
                        and int(request["event_index"]) in allowlist
                        and float(evidence["priority"].detach().cpu()) <= 0.0
                    ):
                        evidence["priority"] = (
                            request["write_gate"].clamp_min(1e-6)
                            * evidence["posterior"].max().clamp_min(1e-6)
                            * request["novelty"].clamp_min(1e-6)
                        )
                    request["admission_evidence"] = evidence
                    request["admission_evaluated"] = True
                    origin = outputs[request["event_index"]]
                    origin["write_window_complete"] = True
                    origin["write_priority_passed"] = bool(
                        float(evidence["priority"].detach().cpu())
                        > self.wake_config.controller_priority_threshold
                    )
                due = [
                    request for request in due
                    if (
                        (
                            self.config.write_event_allowlist is not None
                            and int(request["event_index"])
                            in self.config.write_event_allowlist
                        )
                        or (
                            self.config.write_event_allowlist is None
                            and self.controller.write_admissible(
                                request["write_gate"], 0.0,
                                request["admission_evidence"]["priority"],
                                future_window_complete=True,
                                priority_threshold=self.wake_config.controller_priority_threshold,
                            )
                        )
                    )
                ]
                due.sort(
                    key=lambda request: float(
                        request["admission_evidence"]["priority"].detach().cpu()
                    ), reverse=True,
                )
                remaining_budget = min(
                    4, self.tree.frontier_routing.config.max_writes_per_sequence
                ) - accepted_write_count
                for request in due[:remaining_budget]:
                    request["window_evidence"] = request["admission_evidence"]
                    self._commit_delayed_write(sequence, request)
                    request["committed"] = True
                    accepted_write_requests.append(request)
                    accepted_write_count += 1
                    outputs[request["event_index"]].update({
                        "write_accepted": True,
                        "write_gate_passed": True,
                        "write_priority_passed": True,
                        "write_window_complete": True,
                        "write_priority": float(
                            request["admission_evidence"]["priority"].detach().cpu()
                        ),
                    })

        event_count = int(sequence["times"].numel())
        eligible = [
            request
            for request in pending_writes
            if request["ready_index"] < event_count
        ]
        top_c = self.wake_config.controller_write_probe_topc
        ranked = sorted(
            eligible,
            key=lambda request: float(
                request["write_gate"].detach().cpu()
            ),
            reverse=True,
        )
        top = ranked[:top_c]
        remaining = ranked[top_c:]
        explored = []
        if self.config.probe_write_counterfactuals and remaining:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.config.write_probe_seed + source_index)
            count = min(self.config.write_probe_random_count, len(remaining))
            indices = torch.randperm(len(remaining), generator=generator)[:count].tolist()
            explored = [remaining[index] for index in indices]
        eligible = [*top, *explored]
        top_ids = {id(request) for request in top}
        explored_ids = {id(request) for request in explored}
        for request in eligible:
            outputs[request["event_index"]]["write_probed"] = True
            outputs[request["event_index"]]["write_probe_top"] = id(request) in top_ids
            outputs[request["event_index"]]["write_probe_exploration"] = id(request) in explored_ids
            request["write_probe_exploration"] = id(request) in explored_ids
            outputs[request["event_index"]]["write_probe_propensity"] = (
                1.0 if id(request) in top_ids else len(explored) / max(len(remaining), 1)
            )
        for request in eligible:
            request["window_evidence"] = self._delayed_write_evidence(
                sequence, request
            )
            outputs[request["event_index"]].update({
                "write_utility": float(
                    request["window_evidence"]["write_utility"].detach().cpu()
                ),
                "write_priority": float(
                    request["window_evidence"]["priority"].detach().cpu()
                ),
                "write_owner_on_score_path": bool(
                    request["window_evidence"].get("owner_on_score_path", False)
                ),
                "write_virtual_candidate_alpha": float(
                    request["window_evidence"].get("virtual_candidate_alpha", 0.0)
                ),
                "write_gate_passed": bool(
                    float(request["write_gate"])
                    >= float(self.controller.calibration_thresholds[2])
                ),
                "write_utility_passed": bool(
                    float(request["window_evidence"]["write_utility"]) > 0.0
                ),
                "write_beneficial": bool(
                    float(request["window_evidence"]["write_utility"]) > 0.0
                ),
                "write_priority_passed": bool(
                    float(request["window_evidence"]["priority"])
                    > self.wake_config.controller_priority_threshold
                ),
                "write_window_complete": True,
                "write_accepted": bool(request.get("committed", False)),
            })
        if int(self.controller.controller_version.detach().cpu()) < 6:
            eligible = [
                request for request in eligible
                if not request.get("write_probe_exploration", False)
                and self.controller.write_admissible(
                    request["write_gate"],
                    request["window_evidence"].get("write_utility", 0.0),
                    request["window_evidence"]["priority"],
                    future_window_complete=True,
                    priority_threshold=self.wake_config.controller_priority_threshold,
                )
            ]
            if eligible:
                priorities = torch.stack([
                    request["window_evidence"]["priority"]
                    for request in eligible
                ])
                selected = priorities.topk(
                    min(
                        len(eligible),
                        min(4, self.tree.frontier_routing.config.max_writes_per_sequence),
                    )
                ).indices.cpu().tolist()
            else:
                selected = []
            for index in selected:
                request = eligible[index]
                if self.config.allow_memory_writes:
                    self._commit_delayed_write(sequence, request)
                    request["committed"] = True
                    accepted_write_requests.append(request)
                    outputs[request["event_index"]]["write_accepted"] = True
            if not self.config.allow_memory_writes:
                selected = []
        pending_writes = [
            request
            for request in pending_writes
            if request["ready_index"] >= event_count
        ]
        return {
            "events": outputs,
            "total_nll": total_nll,
            "nll_per_event": total_nll / max(len(outputs), 1),
            "pending_write_count": len(pending_writes),
            "accepted_write_count": (
                accepted_write_count
                if int(self.controller.controller_version.detach().cpu()) >= 6
                else len(selected)
            ),
            "write_probe_count": sum(
                bool(event.get("write_probed", False)) for event in outputs
            ),
            "leaf_ids": list(self.tree.leaf_ids),
        }

    @torch.no_grad()
    def predict_next_event(
        self,
        cpu_prefix: Mapping[str, Tensor],
    ) -> Dict[str, Any]:
        """Return a local-rate next-event forecast from the observed prefix.

        The event type distribution is the normalized current Hawkes intensity.
        The reported time uses the standard locally constant-rate expectation
        ``1 / sum_d lambda_d``; it is deterministic and is not an exact Hawkes
        sample. Exact simulation can be added with Ogata thinning if required.
        """
        prefix = self._move_sequence(cpu_prefix)
        event_index = int(prefix["times"].numel())
        if isinstance(self.encoder, CausalPrefixEncoder):
            z_t = self.encoder(
                prefix["times"],
                prefix["types"],
                event_index,
                time_features=prefix.get(EVENT_TIME_FEATURES_KEY),
            ).reshape(1, -1)
        else:
            z_t = self.encoder(
                prefix["times"], prefix["types"], event_index
            ).reshape(1, -1)
        memory_output = self.tree(
            z_t=z_t,
            working_delta=self.tree.working_memory.delta,
            decays=self.hawkes.decays,
            update_memory_state=False,
        )
        params = memory_output["effective_params"].select(0)
        if event_index == 0:
            current_time = prefix["times"].new_tensor(0.0)
        else:
            current_time = prefix["times"][-1]
        evaluation_time = current_time + current_time.new_tensor(1e-6)
        intensity = self.hawkes.intensity_at_event(
            {"times": prefix["times"], "types": prefix["types"]},
            evaluation_time,
            params,
        )
        total_rate = intensity.sum().clamp_min(1e-8)
        probabilities = intensity / total_rate
        expected_delta = total_rate.reciprocal()
        return {
            "predicted_time": float((current_time + expected_delta).cpu()),
            "expected_delta": float(expected_delta.cpu()),
            "predicted_type": int(probabilities.argmax().item()),
            "type_probabilities": probabilities.cpu(),
            "intensity": intensity.cpu(),
            "responsibility": memory_output["r"][0].cpu(),
        }


def _parse_args():
    parser = argparse.ArgumentParser(description="Wake-only Memory Tree inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--types", type=int, nargs="+", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inference = MemoryTreeInference.from_checkpoint(
        args.checkpoint,
        device=args.device,
        inference_config=InferenceConfig(allow_memory_writes=not args.no_write),
    )
    sequence = {
        "times": torch.tensor(args.times, dtype=torch.float32),
        "types": torch.tensor(args.types, dtype=torch.long),
    }
    result = inference.run_sequence(sequence)
    forecast = inference.predict_next_event(sequence)
    print(result)
    print(forecast)


if __name__ == "__main__":
    main()
