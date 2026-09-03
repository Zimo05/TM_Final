"""Event-scale Wake training operations."""

from __future__ import annotations

from Train.TrainingComponents import *  # noqa: F403
from Train.TrainingComponents import _assert_finite_without_cuda_sync


class TrainingWakeMixin:
    def train_wake_sequence(
        self,
        cpu_sequence: Mapping[str, Tensor],
        *,
        sequence_index: Optional[int] = None,
        precomputed_z: Optional[Tensor] = None,
        precomputed_projected_z: Optional[Tensor] = None,
        precomputed_memory_query: Optional[Tensor] = None,
        frontier_static_cache: Optional[Any] = None,
        precomputed_frontier: Optional[Any] = None,
        precomputed_frontier_rows: Optional[
            Sequence[tuple[tuple[str, ...], int, int]]
        ] = None,
    ) -> Dict[str, Any]:
        """Run one streaming sequence and update only working memory online.

        Encoder, Router, semantic/hypernetwork, and retrieval parameters are
        frozen for the entire sequence.  Their prediction objective is
        recomputed later by :meth:`train_global_batch_epoch`.
        """
        sequence = self._move_sequence(cpu_sequence)
        self.tree.reset_working_memory()
        prediction_total = torch.zeros((), device=self.device)
        wm_penalty_total = torch.zeros((), device=self.device)
        write_penalty_total = torch.zeros((), device=self.device)
        accepted_write_count = 0
        append_count = 0
        refresh_count = 0
        write_decision_count = 0
        memorize_count = 0
        queue_split_count = 0
        responsibilities = []
        action_indices: list[Tensor] = []
        action_probability_total = torch.zeros(4, device=self.device)
        gate_activation_total = torch.zeros(4, device=self.device)
        novelty_total = torch.zeros((), device=self.device)
        max_similarity_total = torch.zeros((), device=self.device)
        similarity_count_total = torch.zeros((), device=self.device)
        pending_writes: list[Dict[str, Any]] = []
        adapt_probes: list[Dict[str, Any]] = []
        write_probe_contexts: list[Dict[str, Any]] = []
        max_gradient_norm = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float64,
        )
        assignment_counts: Counter[str] = Counter()
        owner_depth_total = 0.0
        owner_lca_count = 0
        posterior_entropy_total = torch.zeros((), device=self.device)
        prior_posterior_kl_total = torch.zeros((), device=self.device)
        write_candidates = 0
        frontier_size_total = 0
        frontier_visited_total = 0
        frontier_branch_total = 0
        frontier_node_counts: Counter[str] = Counter()
        expansion_utility_total = torch.zeros((), device=self.device)
        expansion_utility_count = 0
        structural_mass_total = torch.zeros((), device=self.device)

        self.tree.train()
        self.encoder.train()
        # Remove stale gradients from the preceding global/sleep step. Wake
        # must leave all persistent parameters with ``grad is None``.
        self.optimizer.zero_grad(set_to_none=True)
        with self._freeze_global_parameters():
            if precomputed_z is not None:
                expected_rows = int(sequence["times"].numel())
                if precomputed_z.shape != (
                    expected_rows,
                    self.tree.z_dim,
                ):
                    raise ValueError(
                        "precomputed Wake states must have shape [events, z]"
                    )
            elif isinstance(self.encoder, CausalPrefixEncoder):
                with torch.no_grad():
                    precomputed_z = self._encode_memory_sequence(
                        sequence
                    )
            # Tree parameters are frozen for the complete Wake sequence.
            # Reuse its cumulative node embeddings and semantic HyperNet
            # outputs across all events; retrieval/query/working memory remain
            # event-dependent and are intentionally not cached.
            if frontier_static_cache is None:
                frontier_static_cache = (
                    self.tree.frontier_routing.build_static_cache(detach=True)
                )
            semantic_theta_cache = {
                node_id: frontier_static_cache.semantic_theta_table[
                    node_index
                ]
                for node_index, node_id in enumerate(
                    self.tree.all_node_ids
                )
            }
            if precomputed_z is None:
                projected_z_cache = None
                memory_query_cache = None
            else:
                with torch.no_grad():
                    projected_z_cache = (
                        self.tree.router_compat.project_z(precomputed_z)
                        if precomputed_projected_z is None
                        else precomputed_projected_z
                    )
                    memory_query_cache = (
                        self.tree.episodic_memory.query_net(precomputed_z)
                        if precomputed_memory_query is None
                        else precomputed_memory_query
                    )
            for event_index in range(sequence["times"].numel()):
                if precomputed_z is None:
                    z_t = self._encode_memory_event(
                        sequence,
                        event_index,
                    )
                else:
                    z_t = precomputed_z[event_index : event_index + 1]
                working_delta = (
                    self.tree.working_memory.make_trainable_delta()
                )
                memory_output = self.tree(
                    z_t=z_t,
                    working_delta=working_delta,
                    decays=self.hawkes.decays,
                    frontier_static_cache=frontier_static_cache,
                    frontier_projected_z=(
                        None
                        if projected_z_cache is None
                        else projected_z_cache[
                            event_index : event_index + 1
                        ]
                    ),
                    frontier_query=(
                        None
                        if memory_query_cache is None
                        else memory_query_cache[
                            event_index : event_index + 1
                        ]
                    ),
                    precomputed_frontier=(
                        None
                        if precomputed_frontier is None
                        else precomputed_frontier.slice(
                            event_index, event_index + 1
                        )
                    ),
                    update_memory_state=False,
                    update_search_state=(
                        not self.training_config.controller_only_finetune
                    ),
                    materialize_diagnostics=(
                        precomputed_frontier_rows is None
                    ),
                )
                if precomputed_frontier_rows is not None:
                    (
                        frontier_node_ids,
                        visited_count,
                        branch_count,
                    ) = precomputed_frontier_rows[event_index]
                    frontier_size_total += len(frontier_node_ids)
                    frontier_visited_total += visited_count
                    frontier_branch_total += branch_count
                    frontier_node_counts.update(frontier_node_ids)
                else:
                    frontier_samples = memory_output.get(
                        "frontier_samples"
                    )
                    if not frontier_samples:
                        raise RuntimeError(
                            "Wake requires frontier IDs for hard ownership"
                        )
                    frontier_sample = frontier_samples[0]
                    frontier_node_ids = frontier_sample.node_ids
                    frontier_size_total += len(
                        frontier_node_ids
                    )
                    frontier_visited_total += len(
                        frontier_sample.visited_node_ids
                    )
                    frontier_branch_total += len(
                        frontier_sample.expanded_node_ids
                    )
                    frontier_node_counts.update(
                        frontier_node_ids
                    )
                utility = memory_output["expansion_utility"][
                    0, memory_output["expanded_mask"][0]
                ]
                if utility.numel():
                    expansion_utility_total = (
                        expansion_utility_total
                        + utility.detach().sum()
                    )
                    expansion_utility_count += int(utility.numel())
                pre_action_effective = self._controller_effective_parameters(
                    memory_output,
                    working_delta,
                    working_delta.new_zeros(()),
                ).select(0)
                pre_action_nll = self.hawkes.event_NLL(
                    sequence=sequence,
                    params=pre_action_effective,
                    k=event_index,
                )

                frontier_energy = self._frontier_event_nll(
                    sequence,
                    memory_output,
                    event_index,
                )
                (
                    action_index,
                    owner_id,
                    owner_is_lca,
                    query,
                    novelty,
                    similarity_count,
                    max_similarity,
                    action_probabilities,
                    raw_action_probabilities,
                    posterior,
                ) = self._decide_action(
                    memory_output,
                    pre_action_nll,
                    frontier_energy,
                    frontier_node_ids,
                )
                structural_mass_total = (
                    structural_mass_total
                    + self.controller.queue_weight(
                        raw_action_probabilities
                    ).detach()
                )
                effective = self._controller_effective_parameters(
                    memory_output,
                    working_delta,
                    action_probabilities[1],
                ).select(0)
                prediction_nll = self.hawkes.event_NLL(
                    sequence=sequence,
                    params=effective,
                    k=event_index,
                )
                assignment_counts[owner_id] += 1
                owner_depth_total += self.tree.nodes[owner_id].depth
                owner_lca_count += int(owner_is_lca)
                frontier_prior = memory_output["frontier_mass"][0]
                frontier_mask = memory_output["frontier_mask"][0]
                posterior_entropy_total = (
                    posterior_entropy_total
                    - (
                        posterior.clamp_min(1e-12)
                        * posterior.clamp_min(1e-12).log()
                    ).sum()
                )
                prior_posterior_kl_total = (
                    prior_posterior_kl_total
                    + (
                        posterior
                        * (
                            posterior.clamp_min(1e-12).log()
                            - frontier_prior.clamp_min(1e-12).log()
                        )
                    ).masked_fill(~frontier_mask, 0.0).sum()
                )
                action = tuple(Action)[
                    int(action_index.detach().cpu())
                ]
                write_probe_contexts.append(self._write_probe_context(
                    event_index=event_index, memory_output=memory_output,
                    batch_index=0, frontier_node_ids=frontier_node_ids,
                    query=query, posterior=posterior,
                    working_delta=working_delta,
                    retrieve_gate=action_probabilities[1],
                    no_write_theta=effective.theta,
                ))
                adapt_request = self._make_write_request(
                    event_index, owner_id, query, novelty, memory_output,
                    action_probabilities, action_index, posterior,
                    semantic_theta_cache, frontier_node_ids=frontier_node_ids,
                    hard_action=action,
                    controller_inputs={
                        "surprise": pre_action_nll,
                        "novelty": novelty,
                        "count": similarity_count,
                        "owner_confidence": pre_action_nll.new_tensor(float(posterior.max().detach())),
                        "retrieval_similarity": max_similarity,
                        "retrieval_residual_norm": memory_output["frontier_episodic_delta"][0].norm(dim=-1).mean(),
                        "working_memory_norm": working_delta.norm(),
                        "pending_write_ratio": pre_action_nll.new_tensor(
                            len(pending_writes) / max(self.tree.frontier_routing.config.max_writes_per_sequence, 1)
                        ).clamp_max(1.0),
                    },
                    assimilation_theta=effective.theta,
                    assimilation_grad=torch.autograd.grad(
                        prediction_nll, working_delta, retain_graph=True
                    )[0],
                    future_contexts=write_probe_contexts,
                    raw_action_probabilities=raw_action_probabilities,
                )
                adapt_request["exploration"] = bool(
                    torch.rand((), device=pre_action_nll.device)
                    < self.controller.exploration_rate
                )
                adapt_probes.append(adapt_request)
                # Every event is a potential Write probe. Expensive delayed
                # evidence remains bounded later by Top-C plus exploration.
                write_candidates += 1
                write_request = self._make_write_request(
                        event_index,
                        owner_id,
                        query,
                        novelty,
                        memory_output,
                        action_probabilities,
                        action_index,
                        posterior,
                        semantic_theta_cache,
                        frontier_node_ids=frontier_node_ids,
                        hard_action=action,
                        controller_inputs={
                            "surprise": pre_action_nll,
                            "novelty": novelty,
                            "count": similarity_count,
                            "owner_confidence": pre_action_nll.new_tensor(float(posterior.max().detach())),
                            "retrieval_similarity": max_similarity,
                            "retrieval_residual_norm": memory_output["frontier_episodic_delta"][0].norm(dim=-1).mean(),
                            "working_memory_norm": working_delta.norm(),
                            "pending_write_ratio": pre_action_nll.new_zeros(()),
                        },
                        future_contexts=write_probe_contexts,
                        raw_action_probabilities=raw_action_probabilities,
                )
                write_request["exploration"] = bool(
                    torch.rand((), device=pre_action_nll.device)
                    < self.controller.exploration_rate
                )
                wm_penalty = (
                    self.wake_config.lambda_wm
                    * working_delta.pow(2).sum()
                )
                write_penalty = (
                    self.wake_config.lambda_write
                    * action_probabilities[2]
                    + self.wake_config.controller_split_cost
                    * self.controller.queue_weight(action_probabilities)
                )
                wake_loss = prediction_nll + wm_penalty + write_penalty
                _assert_finite_without_cuda_sync(
                    wake_loss,
                    "wake objective became non-finite",
                )

                # This is the only event-level gradient/update path.
                working_grad = torch.autograd.grad(
                    prediction_nll,
                    working_delta,
                    retain_graph=False,
                )[0]
                gradient_norm = working_grad.detach().double().norm()
                max_gradient_norm = torch.maximum(
                    max_gradient_norm,
                    gradient_norm,
                )
                self.tree.working_memory.update_from_gradient(
                    working_grad,
                    adaptation_probability=action_probabilities[0],
                )
                if not self.training_config.controller_only_finetune:
                    if precomputed_frontier_rows is not None:
                        packed_info = memory_output["packed_memory_info"]
                        frontier = precomputed_frontier.slice(
                            event_index, event_index + 1
                        )
                        self.tree.episodic_memory.credit_retrieval_packed(
                            alpha=packed_info["alpha"],
                            visited_node_indices=frontier.visited_indices,
                            visited_node_mask=frontier.visited_mask,
                            path_incidence=frontier.path_incidence,
                            routing_weights=posterior.unsqueeze(0),
                            retrieval_probability=(
                                action_probabilities[1].detach()
                            ),
                            node_ids=self.tree.all_node_ids,
                        )
                    else:
                        self.tree.episodic_memory.credit_retrieval(
                            info_by_batch=memory_output["memory_info"],
                            leaf_paths=[
                                self.tree.path_to_node(node_id)
                                for node_id in frontier_node_ids
                            ],
                            routing_weights=posterior.unsqueeze(0),
                            retrieval_probability=(
                                action_probabilities[1].detach()
                            ),
                        )
                # Age only after the working-memory graph that read bank.age
                # has been released.
                if not self.training_config.controller_only_finetune:
                    self.tree.episodic_memory.step_age()

                if write_request is not None:
                    write_request["assimilation_theta"] = (
                        effective.theta.detach().clone()
                    )
                    write_request["assimilation_grad"] = (
                        working_grad.detach().clone()
                    )
                    pending_writes.append(write_request)

                prediction_total = (
                    prediction_total + prediction_nll.detach()
                )
                wm_penalty_total = wm_penalty_total + wm_penalty.detach()
                write_penalty_total = (
                    write_penalty_total + write_penalty.detach()
                )
                # Sleep receives only mass that was actually evaluated at a
                # leaf frontier expert. Coarse-node posterior is not projected
                # to unexpanded descendants.
                leaf_mass = posterior.new_zeros(len(self.tree.leaf_ids))
                leaf_index = {
                    leaf_id: index
                    for index, leaf_id in enumerate(self.tree.leaf_ids)
                }
                for slot, node_id in enumerate(
                    frontier_node_ids
                ):
                    if node_id in leaf_index:
                        leaf_mass[leaf_index[node_id]] = posterior[slot]
                responsibilities.append(leaf_mass)
                action_indices.append(action_index)
                action_probability_total = (
                    action_probability_total
                    + action_probabilities.detach()
                )
                gate_activation_total = gate_activation_total + (
                    action_probabilities.detach() >= 0.5
                ).to(gate_activation_total)
                novelty_total = novelty_total + novelty.detach()
                max_similarity_total = (
                    max_similarity_total + max_similarity
                )
                similarity_count_total = (
                    similarity_count_total + similarity_count
                )

        event_count = int(sequence["times"].numel())
        # Rank only requests whose complete future horizon is now observable.
        # This enforces a true per-sequence top-B physical-write budget instead
        # of accepting the first B controller actions.
        eligible_writes = [
            request
            for request in pending_writes
            if request["ready_index"] < event_count
        ]
        incomplete_writes = [
            request
            for request in pending_writes
            if request["ready_index"] >= event_count
        ]
        eligible_writes = self._preselect_write_requests(eligible_writes)
        probed_writes = list(eligible_writes)
        write_probe_count = len(eligible_writes)
        write_gate_pass_count = 0
        write_utility_pass_count = 0
        if eligible_writes:
            for request in eligible_writes:
                request["window_evidence"] = (
                    self._window_write_evidence(sequence, request)
                )
            write_gate_pass_count = sum(
                float(request["write_gate"].detach().cpu())
                >= float(self.controller.calibration_thresholds[2].detach().cpu())
                for request in eligible_writes
            )
            write_utility_pass_count = sum(
                float(request["window_evidence"]["write_utility"].detach().cpu()) > 0.0
                for request in eligible_writes
            )
            eligible_writes = [
                request for request in eligible_writes
                if (
                    not self.training_config.controller_only_finetune
                    or not request.get("exploration", False)
                )
                and self.controller.write_admissible(
                    request["write_gate"],
                    request["window_evidence"]["write_utility"],
                    request["window_evidence"]["priority"],
                    future_window_complete=True,
                    priority_threshold=self.wake_config.controller_priority_threshold,
                )
            ]
        if eligible_writes:
            priorities = torch.stack([
                request["window_evidence"]["priority"]
                for request in eligible_writes
            ])
            selected_indices = priorities.topk(
                k=min(
                    len(eligible_writes),
                    min(4, self.tree.frontier_routing.config.max_writes_per_sequence),
                )
            ).indices.cpu().tolist()
            selected_writes = [
                eligible_writes[index] for index in selected_indices
            ]
        else:
            selected_writes = []
        # Physical memory mutation follows causal sequence order.  Ranking
        # decides *which* rows survive the per-sequence budget; it does not
        # reorder the committed transaction.
        persistent_writes = []
        selected_writes.sort(
            key=lambda request: int(request["event_index"])
        )
        if not self.training_config.controller_only_finetune:
            admission_actions = self._commit_write_requests_batch(
                sequence,
                selected_writes,
                return_actions=True,
            )
            persistent_writes = [
                request
                for request, action in zip(
                    selected_writes, admission_actions
                )
                if action != "queue"
            ]
            append_count = sum(
                action == "append" for action in admission_actions
            )
            refresh_count = sum(
                action == "refresh" for action in admission_actions
            )
            source_id = int(torch.as_tensor(
                sequence.get(
                    "source_index",
                    -1 if sequence_index is None else sequence_index,
                )
            ).detach().cpu())
            accepted_tokens = [
                (source_id, int(request["event_index"]))
                for request in persistent_writes
            ]
            selected_ids = {id(request) for request in selected_writes}
            shadow_candidates = []
            for request in probed_writes:
                if id(request) in selected_ids:
                    continue
                evidence = request["window_evidence"]
                priority = (
                    request["queue_weight"]
                    * evidence["confidence"]
                    * evidence["bounded_gain"]
                )
                if float(priority.detach().cpu()) <= 0.0:
                    continue
                shadow_candidates.append((priority, request))
            shadow_candidates.sort(
                key=lambda pair: float(pair[0].detach().cpu()),
                reverse=True,
            )
            max_shadow = min(
                4,
                self.tree.frontier_routing.config.max_writes_per_sequence,
            )
            shadow_records = []
            for priority, request in shadow_candidates[:max_shadow]:
                evidence = request["window_evidence"]
                item = evidence["candidate_item"]
                item.write_quality = float(
                    evidence["bounded_gain"].detach().cpu()
                )
                item.queue_weight = float((
                    request["queue_weight"] * evidence["confidence"]
                ).detach().cpu())
                shadow_records.append({
                    "token": (source_id, int(request["event_index"])),
                    "owner_id": evidence["owner_id"],
                    "priority": float(priority.detach().cpu()),
                    "item": item,
                })
            self._update_structural_evidence_buffer(
                shadow_records,
                accepted_tokens=accepted_tokens,
            )
        for request in self._select_adapt_probes(adapt_probes):
            self._record_adapt_utility(sequence, request)
        if self.training_config.controller_write_ranking:
            self.controller_utility_replay.finalize_write_group(
                int(torch.as_tensor(sequence.get("source_index", -1)).detach().cpu())
            )
        # Selection/probe counts remain diagnostic.  An accepted write is a
        # persistent admission transaction and may either append a new law
        # prototype or refresh an existing one.
        write_decision_count = len(selected_writes)
        accepted_write_count = append_count + refresh_count
        # ``write_count`` is retained as a compatibility alias for callers
        # written before the evidence/prototype distinction was exposed.
        write_count = accepted_write_count
        prototype_count, evidence_mass = self._persistent_memory_stats()
        pending_writes = incomplete_writes
        action_values = tuple(Action)
        action_index_values = (
            torch.stack(action_indices).cpu().tolist()
            if action_indices
            else []
        )
        actions = [
            str(action_values[index])
            for index in action_index_values
        ]
        action_counts: Counter[str] = Counter(actions)
        memorize_count = action_counts[Action.MEMORIZE.value]
        queue_split_count = action_counts[Action.QUEUE_SPLIT.value]
        sequence_responsibility = (
            torch.stack(responsibilities).mean(dim=0)
            if responsibilities
            else torch.zeros(len(self.tree.leaf_ids))
        )
        metric_values = torch.cat(
            [
                torch.stack(
                    [
                        prediction_total.float(),
                        wm_penalty_total.float(),
                        write_penalty_total.float(),
                        novelty_total.float(),
                        max_similarity_total.float(),
                        similarity_count_total.float(),
                        max_gradient_norm.float(),
                        expansion_utility_total.float(),
                    ]
                ),
                (
                    action_probability_total
                    / max(event_count, 1)
                ).float(),
            ]
        ).cpu().tolist()
        (
            prediction_total_value,
            wm_penalty_total_value,
            write_penalty_total_value,
            novelty_total_value,
            max_similarity_total_value,
            similarity_count_total_value,
            max_gradient_norm_value,
            expansion_utility_total_value,
            *mean_action_probability_values,
        ) = metric_values
        return {
            "prediction_nll": prediction_total_value,
            "wm_penalty": wm_penalty_total_value,
            "write_penalty": write_penalty_total_value,
            "accepted_write_count": accepted_write_count,
            "append_count": append_count,
            "refresh_count": refresh_count,
            "prototype_count": prototype_count,
            "evidence_mass": evidence_mass,
            "write_count": write_count,
            "write_decision_count": write_decision_count,
            "memorize_count": memorize_count,
            "memorize_argmax_count": memorize_count,
            "write_gate_active_count": int(
                gate_activation_total[2].detach().cpu()
            ),
            "write_candidate_count": write_candidates,
            "write_priority_pass_count": write_decision_count,
            "write_window_complete_count": write_probe_count,
            "write_accepted_count": accepted_write_count,
            "write_retrieved_later_count": 0,
            "write_beneficial_count": max(
                accepted_write_count - sum(
                    float(request["window_evidence"]["write_utility"].detach().cpu()) <= 0.0
                    for request in persistent_writes
                ),
                0,
            ),
            "queue_split_count": queue_split_count,
            "event_count": event_count,
            "raw_structural_mass": float(
                structural_mass_total.detach().cpu()
            ),
            "structural_observations": event_count,
            "responsibilities": responsibilities,
            "sequence_responsibility": sequence_responsibility,
            "actions": actions,
            "action_counts": dict(action_counts),
            "mean_action_probabilities": {
                action.value: mean_action_probability_values[index]
                for index, action in enumerate(Action)
            },
            "mean_gates": {
                action.value: mean_action_probability_values[index]
                for index, action in enumerate(Action)
            },
            "gate_activation_rates": {
                action.value: float(
                    gate_activation_total[index].cpu()
                ) / max(event_count, 1)
                for index, action in enumerate(Action)
            },
            "memory_assignment_counts": dict(assignment_counts),
            "sequence_owner_id": (
                assignment_counts.most_common(1)[0][0]
                if assignment_counts
                else "root"
            ),
            "posterior_entropy": (
                float(posterior_entropy_total.detach().cpu())
                / max(event_count, 1)
            ),
            "prior_posterior_kl": (
                float(prior_posterior_kl_total.detach().cpu())
                / max(event_count, 1)
            ),
            "memory_owner_depth_mean": (
                owner_depth_total / max(event_count, 1)
            ),
            "owner_lca_rate": owner_lca_count / max(event_count, 1),
            "write_candidates": write_candidates,
            "write_probe_count": write_probe_count,
            "write_gate_pass_count": write_gate_pass_count,
            "write_utility_pass_count": write_utility_pass_count,
            "accepted_write_utility_sum": sum(
                float(request["window_evidence"]["write_utility"].detach().cpu())
                for request in persistent_writes
            ),
            "harmful_write_count": sum(
                float(request["window_evidence"]["write_utility"].detach().cpu()) <= 0.0
                for request in persistent_writes
            ),
            "mean_novelty": novelty_total_value / max(event_count, 1),
            "mean_weighted_similarity": (
                max_similarity_total_value / max(event_count, 1)
            ),
            "mean_soft_count": (
                similarity_count_total_value / max(event_count, 1)
            ),
            # Compatibility aliases for pre-controller diagnostics.
            "mean_max_similarity": (
                max_similarity_total_value / max(event_count, 1)
            ),
            "mean_similarity_count": (
                similarity_count_total_value / max(event_count, 1)
            ),
            "mean_frontier_size": (
                frontier_size_total / max(event_count, 1)
            ),
            "mean_frontier_visited_nodes": (
                frontier_visited_total / max(event_count, 1)
            ),
            "mean_frontier_branches": (
                frontier_branch_total / max(event_count, 1)
            ),
            "frontier_node_counts": dict(frontier_node_counts),
            "mean_expansion_utility": (
                expansion_utility_total_value
                / max(expansion_utility_count, 1)
            ),
            "pending_write_count": len(pending_writes),
            "max_gradient_norm": max_gradient_norm_value,
        }

    def _read_episodic_flat_batch(
        self,
        query_flat: Tensor,
        frontier_flat: Any,
        *,
        microbatch: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Read the immutable episodic snapshot for all flat Wake rows.

        The Wake time loop is causal only through working memory. Episodic
        writes and retrieval-usage credit are deferred until the transaction
        ends, so all flat rows can share one read snapshot. Chunking keeps the
        ``[rows, visited_nodes, capacity]`` retrieval tensors bounded without
        changing any row's result.
        """
        if query_flat.ndim != 2:
            raise ValueError("flat Wake query must have shape [N, key_dim]")
        row_count = query_flat.size(0)
        if row_count <= 0:
            raise ValueError("flat Wake retrieval requires at least one row")
        if (
            frontier_flat.visited_indices.size(0) != row_count
            or frontier_flat.visited_mask.size(0) != row_count
            or frontier_flat.path_incidence.size(0) != row_count
        ):
            raise ValueError("flat retrieval tensors do not align")
        if microbatch is None:
            microbatch = self.wake_config.retrieval_microbatch
        microbatch = int(microbatch)
        if microbatch <= 0:
            raise ValueError("retrieval_microbatch must be positive")

        memory = self.tree.episodic_memory
        node_ids = tuple(self.tree.all_node_ids)
        node_delta_chunks: list[Tensor] = []
        episodic_delta_chunks: list[Tensor] = []
        info_chunks: Dict[str, list[Tensor]] = {}
        for start in range(0, row_count, microbatch):
            end = min(start + microbatch, row_count)
            node_delta, packed_info = memory.read_packed(
                query=query_flat[start:end],
                node_indices=frontier_flat.visited_indices[start:end],
                node_mask=frontier_flat.visited_mask[start:end],
                node_ids=node_ids,
                update_state=False,
            )
            node_delta_chunks.append(node_delta)
            episodic_delta_chunks.append(torch.einsum(
                "nkv,nvp->nkp",
                frontier_flat.path_incidence[start:end].to(
                    node_delta.dtype
                ),
                node_delta,
            ))
            for key, value in packed_info.items():
                info_chunks.setdefault(key, []).append(value)

        return (
            torch.cat(node_delta_chunks, dim=0),
            torch.cat(episodic_delta_chunks, dim=0),
            {
                key: torch.cat(values, dim=0)
                for key, values in info_chunks.items()
            },
        )

    def train_wake_batch(
        self,
        *,
        sequences: Sequence[Mapping[str, Tensor]],
        sequence_indices: Sequence[int],
        z_flat: Tensor,
        projected_flat: Tensor,
        query_flat: Tensor,
        frontier_static_cache: Any,
        frontier_flat: Any,
        frontier_rows: Sequence[tuple[tuple[str, ...], int, int]],
        flat: Mapping[str, Tensor],
    ) -> list[Dict[str, Any]]:
        """Run a masked time-position Wake wavefront over several sequences.

        Working memory remains strictly recurrent within each row. Shared
        episodic banks form a minibatch transaction: retrieval sees the bank
        snapshot present at batch entry, usage credit is accumulated during
        the wavefront, and physical top-B writes commit in sequence order only
        after every row has finished.
        """
        batch_size = len(sequences)
        if batch_size == 0 or len(sequence_indices) != batch_size:
            raise ValueError("Wake wavefront batch metadata does not align")
        lengths = [
            int(value)
            for value in flat["sequence_lengths"].detach().cpu().tolist()
        ]
        if len(lengths) != batch_size or any(length <= 0 for length in lengths):
            raise ValueError("Wake wavefront requires non-empty sequences")
        offsets = []
        cursor = 0
        for length in lengths:
            offsets.append(cursor)
            cursor += length
        if (
            cursor != z_flat.size(0)
            or projected_flat.size(0) != cursor
            or query_flat.size(0) != cursor
            or len(frontier_rows) != cursor
        ):
            raise ValueError("flattened Wake tensors do not align")

        offsets_tensor = torch.as_tensor(
            offsets,
            device=self.device,
            dtype=torch.long,
        )
        sequence_index_flat = flat["sequence_index"].to(
            device=self.device,
            dtype=torch.long,
        )
        active_rows_by_time = tuple(
            torch.as_tensor(
                [
                    row
                    for row, length in enumerate(lengths)
                    if event_index < length
                ],
                device=self.device,
                dtype=torch.long,
            )
            for event_index in range(max(lengths))
        )

        leaf_count = len(self.tree.leaf_ids)
        node_ids = tuple(self.tree.all_node_ids)
        node_count = len(node_ids)
        param_dim = self.tree.param_dim
        controller_input_names = (
            "surprise",
            "novelty",
            "count",
            "owner_confidence",
            "retrieval_similarity",
            "retrieval_residual_norm",
            "working_memory_norm",
            "pending_write_ratio",
        )
        working_state = self.tree.working_memory.new_batch_state(batch_size)
        prediction_total = z_flat.new_zeros(batch_size)
        wm_penalty_total = z_flat.new_zeros(batch_size)
        write_penalty_total = z_flat.new_zeros(batch_size)
        action_probability_total = z_flat.new_zeros(batch_size, 4)
        gate_activation_total = z_flat.new_zeros(batch_size, 4)
        novelty_total = z_flat.new_zeros(batch_size)
        max_similarity_total = z_flat.new_zeros(batch_size)
        similarity_count_total = z_flat.new_zeros(batch_size)
        max_gradient_norm = torch.zeros(
            batch_size,
            device=self.device,
            dtype=torch.float64,
        )
        posterior_entropy_total = z_flat.new_zeros(batch_size)
        prior_posterior_kl_total = z_flat.new_zeros(batch_size)
        expansion_utility_total = z_flat.new_zeros(batch_size)
        responsibility_sum = z_flat.new_zeros(batch_size, leaf_count)
        assignment_counts_tensor = torch.zeros(
            batch_size,
            node_count,
            device=self.device,
            dtype=torch.long,
        )
        action_counts_tensor = torch.zeros(
            batch_size,
            len(Action),
            device=self.device,
            dtype=torch.long,
        )
        frontier_node_counts_tensor = torch.zeros(
            batch_size,
            node_count,
            device=self.device,
            dtype=torch.long,
        )
        owner_depth_total_tensor = z_flat.new_zeros(batch_size)
        owner_lca_count_tensor = z_flat.new_zeros(batch_size)
        frontier_size_total_tensor = z_flat.new_zeros(batch_size)
        frontier_visited_total_tensor = z_flat.new_zeros(batch_size)
        frontier_branch_total_tensor = z_flat.new_zeros(batch_size)
        expansion_utility_count_tensor = torch.zeros(
            batch_size,
            device=self.device,
            dtype=torch.long,
        )

        # Per-flat-row snapshots are filled by the recurrent time loop and
        # converted to Python request objects only after the whole transaction
        # has completed. This keeps strings, Counters, dicts and tensor clones
        # out of the GPU hot path.
        action_probability_flat = z_flat.new_zeros(cursor, len(Action))
        raw_action_probability_flat = z_flat.new_zeros(cursor, len(Action))
        action_index_flat = torch.zeros(
            cursor,
            device=self.device,
            dtype=torch.long,
        )
        working_gradient_flat = z_flat.new_zeros(cursor, param_dim)
        assimilation_theta_flat = z_flat.new_zeros(cursor, param_dim)
        working_delta_snapshot_flat = z_flat.new_zeros(cursor, param_dim)
        controller_inputs_flat = z_flat.new_zeros(
            cursor,
            len(controller_input_names),
        )
        adapt_exploration_flat = torch.zeros(
            cursor,
            device=self.device,
            dtype=torch.bool,
        )
        write_exploration_flat = torch.zeros(
            cursor,
            device=self.device,
            dtype=torch.bool,
        )
        gradient_norm_flat = z_flat.new_zeros(cursor, dtype=torch.float64)

        pending_writes: list[list[Dict[str, Any]]] = [
            [] for _ in range(batch_size)
        ]
        adapt_probes: list[list[Dict[str, Any]]] = [
            [] for _ in range(batch_size)
        ]
        write_probe_contexts: list[list[Dict[str, Any]]] = [
            [] for _ in range(batch_size)
        ]
        write_candidates = [0 for _ in range(batch_size)]
        action_values = tuple(Action)
        # Read the immutable controller mode once for the whole transaction.
        # The v4/v5 path below uses a padded [B, T, ...] cache for delayed
        # probe evidence; controller-only/v6 keeps its existing scalar
        # fallback until its causal sampler is migrated as well.
        controller_version = int(
            self.controller.controller_version.detach().cpu()
        )
        batched_write_path = (
            not self.training_config.controller_only_finetune
            and controller_version < 6
        )
        padded_wake = (
            self._padded_wake_sequences(sequences)
            if batched_write_path
            else None
        )

        self.tree.train()
        self.encoder.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Retrieval credit is additive within one Wake transaction. Keep its
        # node/capacity reduction on device and apply it once after the causal
        # wavefront, so cycle_usage does not launch a bank update at every
        # time position (and cannot invalidate the packed mirror).
        cycle_usage_credit: Optional[Tensor] = None
        cycle_usage_node_ids = tuple(self.tree.all_node_ids)
        with self._freeze_global_parameters():
            (
                node_delta_flat,
                frontier_episodic_delta_flat,
                packed_memory_info_flat,
            ) = self._read_episodic_flat_batch(
                query_flat,
                frontier_flat,
            )
            # Everything below this point is independent of recurrent
            # working memory. Build it once for all flat rows so the time loop
            # only performs the causal working-memory/controller transition.
            with torch.no_grad():
                memory_output_flat = self.tree(
                    z_t=z_flat,
                    working_delta=z_flat.new_zeros(cursor, param_dim),
                    decays=self.hawkes.decays,
                    frontier_static_cache=frontier_static_cache,
                    frontier_projected_z=projected_flat,
                    frontier_query=query_flat,
                    precomputed_frontier=frontier_flat,
                    precomputed_node_delta=node_delta_flat,
                    precomputed_episodic_delta=frontier_episodic_delta_flat,
                    precomputed_memory_info=packed_memory_info_flat,
                    update_memory_state=False,
                    update_search_state=False,
                    materialize_diagnostics=False,
                )
                frontier_energy_flat = self._batched_frontier_event_nll(
                    flat,
                    memory_output_flat,
                )
                posterior_flat = self._frontier_posterior(
                    memory_output_flat["frontier_mass"],
                    frontier_energy_flat,
                    memory_output_flat["frontier_mask"],
                ).detach()
                (
                    owner_indices_flat,
                    owner_is_lca_flat,
                    owner_confidence_flat,
                ) = self._posterior_owner_indices_batch(
                    memory_output_flat["frontier_node_indices"],
                    posterior_flat,
                )
                novelty_flat, similarity_count_flat, max_similarity_flat = (
                    self.tree.episodic_memory.novelty_count_packed(
                        query_flat,
                        owner_indices_flat,
                        node_ids,
                        temperature=self.controller.novelty_temperature,
                        count_exponent=self.controller.count_exponent,
                        eps=self.controller.controller_eps,
                        count_similarity_low=(
                            self.controller.count_similarity_low
                        ),
                        count_similarity_high=(
                            self.controller.count_similarity_high
                        ),
                        count_topk=self.controller.count_topk,
                        count_saturation=self.controller.count_saturation,
                    )
                )
                retrieval_residual_norm_flat = (
                    frontier_episodic_delta_flat.norm(dim=-1).mean(dim=-1)
                )

                frontier_size_flat = frontier_flat.mask.sum(dim=-1)
                frontier_visited_flat = (
                    frontier_flat.visited_mask.sum(dim=-1)
                )
                frontier_branch_flat = frontier_flat.expanded_mask.sum(
                    dim=-1
                )
                frontier_size_total_tensor.index_add_(
                    0,
                    sequence_index_flat,
                    frontier_size_flat.to(z_flat.dtype),
                )
                frontier_visited_total_tensor.index_add_(
                    0,
                    sequence_index_flat,
                    frontier_visited_flat.to(z_flat.dtype),
                )
                frontier_branch_total_tensor.index_add_(
                    0,
                    sequence_index_flat,
                    frontier_branch_flat.to(z_flat.dtype),
                )
                expansion_utility_count_tensor.index_add_(
                    0,
                    sequence_index_flat,
                    frontier_branch_flat.to(torch.long),
                )

                frontier_node_counts_flat = torch.zeros(
                    cursor,
                    node_count,
                    device=self.device,
                    dtype=torch.long,
                )
                frontier_node_counts_flat.scatter_add_(
                    1,
                    frontier_flat.node_indices.clamp_min(0),
                    frontier_flat.mask.to(torch.long),
                )
                frontier_node_counts_tensor.index_add_(
                    0,
                    sequence_index_flat,
                    frontier_node_counts_flat,
                )

                node_index_by_id = {
                    node_id: index for index, node_id in enumerate(node_ids)
                }
                node_depth_table = z_flat.new_tensor([
                    self.tree.nodes[node_id].depth
                    for node_id in node_ids
                ])
                owner_depth_total_tensor.index_add_(
                    0,
                    sequence_index_flat,
                    node_depth_table.index_select(
                        0,
                        owner_indices_flat,
                    ),
                )
                owner_lca_count_tensor.index_add_(
                    0,
                    sequence_index_flat,
                    owner_is_lca_flat.to(z_flat.dtype),
                )
                assignment_counts_tensor.view(-1).index_add_(
                    0,
                    sequence_index_flat * node_count
                    + owner_indices_flat,
                    torch.ones(
                        cursor,
                        device=self.device,
                        dtype=torch.long,
                    ),
                )

                prior_flat = memory_output_flat["frontier_mass"]
                frontier_mask_flat = memory_output_flat["frontier_mask"]
                posterior_entropy_flat = -(
                    posterior_flat.clamp_min(1e-12)
                    * posterior_flat.clamp_min(1e-12).log()
                ).sum(dim=-1)
                prior_posterior_kl_flat = (
                    posterior_flat
                    * (
                        posterior_flat.clamp_min(1e-12).log()
                        - prior_flat.clamp_min(1e-12).log()
                    )
                ).masked_fill(~frontier_mask_flat, 0.0).sum(dim=-1)
                utility_flat = memory_output_flat[
                    "expansion_utility"
                ].masked_fill(
                    ~memory_output_flat["expanded_mask"],
                    0.0,
                ).sum(dim=-1)
                posterior_entropy_total.index_add_(
                    0,
                    sequence_index_flat,
                    posterior_entropy_flat,
                )
                prior_posterior_kl_total.index_add_(
                    0,
                    sequence_index_flat,
                    prior_posterior_kl_flat,
                )
                expansion_utility_total.index_add_(
                    0,
                    sequence_index_flat,
                    utility_flat,
                )
                novelty_total.index_add_(
                    0,
                    sequence_index_flat,
                    novelty_flat,
                )
                max_similarity_total.index_add_(
                    0,
                    sequence_index_flat,
                    max_similarity_flat,
                )
                similarity_count_total.index_add_(
                    0,
                    sequence_index_flat,
                    similarity_count_flat,
                )

                # Routing composition is affine in unconstrained parameter
                # space.  Reduce the semantic and episodic expert axes once
                # for the complete flat wavefront; the recurrent loop then
                # only adds its causal working-memory row and retrieval gate.
                semantic_base_flat = torch.einsum(
                    "nkp,nk->np",
                    memory_output_flat["frontier_semantic_theta"],
                    memory_output_flat["r"],
                )
                episodic_base_flat = torch.einsum(
                    "nkp,nk->np",
                    memory_output_flat["frontier_episodic_delta"],
                    memory_output_flat["r"],
                )

                node_to_leaf = torch.full(
                    (node_count,),
                    -1,
                    device=self.device,
                    dtype=torch.long,
                )
                for leaf_slot, leaf_id in enumerate(self.tree.leaf_ids):
                    node_to_leaf[node_index_by_id[leaf_id]] = leaf_slot
                leaf_slots = node_to_leaf[
                    frontier_flat.node_indices.clamp_min(0)
                ]
                leaf_valid = frontier_flat.mask & leaf_slots.ge(0)
                leaf_mass_flat = z_flat.new_zeros(cursor, leaf_count)
                leaf_mass_flat.scatter_add_(
                    1,
                    leaf_slots.clamp_min(0),
                    posterior_flat * leaf_valid.to(posterior_flat.dtype),
                )
                responsibility_sum.index_add_(
                    0,
                    sequence_index_flat,
                    leaf_mass_flat,
                )
                # Keep only fields needed by the recurrent recomposition and
                # the delayed request snapshot. Large diagnostic tensors such
                # as expanded-child parameters do not need to live through
                # the whole Wake transaction.
                memory_output_flat = {
                    key: memory_output_flat[key]
                    for key in (
                        "r",
                        "frontier_semantic_theta",
                        "frontier_episodic_delta",
                        "frontier_mass",
                        "frontier_mask",
                        "frontier_node_indices",
                        "frontier_theta",
                    )
                }

            for event_index, active in enumerate(active_rows_by_time):
                if active.numel() == 0:
                    continue
                flat_rows = offsets_tensor.index_select(0, active) + event_index
                working_delta = self.tree.working_memory.make_trainable_rows(
                    working_state,
                    active,
                )
                step_flat = {
                    key: flat[key].index_select(0, flat_rows)
                    for key in (
                        "types",
                        "duration",
                        HAWKES_HISTORY_STATS_KEY,
                        HAWKES_INTERVAL_STATS_KEY,
                    )
                }
                owner_indices = owner_indices_flat.index_select(0, flat_rows)
                owner_confidence = owner_confidence_flat.index_select(
                    0,
                    flat_rows,
                )
                novelty = novelty_flat.index_select(0, flat_rows)
                similarity_count = similarity_count_flat.index_select(
                    0,
                    flat_rows,
                )
                max_similarity = max_similarity_flat.index_select(
                    0,
                    flat_rows,
                )
                # The controller consumes the no-retrieval pre-action NLL.
                # Do not evaluate the tree output's full-retrieval NLL here:
                # the final Wake objective below uses only the gated result.
                semantic_base = semantic_base_flat.index_select(0, flat_rows)
                episodic_base = episodic_base_flat.index_select(0, flat_rows)
                pre_action_theta = semantic_base + working_delta
                pre_action_effective = self._effective_parameters_from_theta(
                    pre_action_theta,
                    detach=True,
                )
                pre_action_nll = self._batched_sequence_event_nll(
                    step_flat,
                    {"effective_params": pre_action_effective},
                )
                pending_write_ratio = pre_action_nll.new_full(
                    (active.numel(),),
                    float(event_index)
                    / max(
                        self.tree.frontier_routing.config
                        .max_writes_per_sequence,
                        1,
                    ),
                ).clamp_max(1.0)
                controller_output = self.controller.action_distribution_batch(
                    pre_action_nll,
                    novelty,
                    similarity_count,
                    owner_confidence=owner_confidence,
                    retrieval_similarity=max_similarity,
                    retrieval_residual_norm=(
                        retrieval_residual_norm_flat.index_select(
                            0,
                            flat_rows,
                        )
                    ),
                    working_memory_norm=working_delta.norm(dim=-1),
                    pending_write_ratio=pending_write_ratio,
                )
                action_probabilities = controller_output["probabilities"]
                raw_action_probabilities = controller_output.get(
                    "raw_probabilities",
                    action_probabilities,
                )
                gated_theta = (
                    pre_action_theta
                    + action_probabilities[:, 1, None] * episodic_base
                )
                gated_effective = self._effective_parameters_from_theta(
                    gated_theta,
                    detach=True,
                )
                # This is the only prediction NLL used for Wake loss and the
                # recurrent working-memory gradient at this time position.
                prediction_nll, working_grad = self._batched_sequence_event_nll_and_grad(
                    step_flat,
                    gated_effective,
                )
                action_index = action_probabilities.detach().argmax(dim=-1)
                wm_penalty = (
                    self.wake_config.lambda_wm
                    * working_delta.square().sum(dim=-1)
                )
                write_penalty = (
                    self.wake_config.lambda_write
                    * action_probabilities[:, 2]
                    + self.wake_config.controller_split_cost
                    * self.controller.queue_weight(action_probabilities)
                )
                wake_loss = prediction_nll + wm_penalty + write_penalty
                _assert_finite_without_cuda_sync(
                    wake_loss,
                    "batched wake objective became non-finite",
                )
                gradient_norm = working_grad.detach().double().norm(dim=-1)
                self.tree.working_memory.update_batch_rows(
                    working_state,
                    active,
                    working_grad,
                    adaptation_probability=action_probabilities[:, 0],
                )

                prediction_total.index_add_(
                    0,
                    active,
                    prediction_nll.detach(),
                )
                wm_penalty_total.index_add_(
                    0,
                    active,
                    wm_penalty.detach(),
                )
                write_penalty_total.index_add_(
                    0,
                    active,
                    write_penalty.detach(),
                )
                action_probability_flat.index_copy_(
                    0,
                    flat_rows,
                    action_probabilities.detach(),
                )
                raw_action_probability_flat.index_copy_(
                    0,
                    flat_rows,
                    raw_action_probabilities.detach(),
                )
                action_index_flat.index_copy_(0, flat_rows, action_index)
                working_gradient_flat.index_copy_(
                    0,
                    flat_rows,
                    working_grad.detach(),
                )
                assimilation_theta_flat.index_copy_(
                    0,
                    flat_rows,
                    gated_effective.theta.detach(),
                )
                working_delta_snapshot_flat.index_copy_(
                    0,
                    flat_rows,
                    working_delta.detach(),
                )
                controller_inputs_flat.index_copy_(
                    0,
                    flat_rows,
                    torch.stack(
                        [
                            pre_action_nll,
                            novelty,
                            similarity_count,
                            owner_confidence,
                            max_similarity,
                            retrieval_residual_norm_flat.index_select(
                                0,
                                flat_rows,
                            ),
                            working_delta.norm(dim=-1),
                            pending_write_ratio,
                        ],
                        dim=-1,
                    ).detach(),
                )
                gradient_norm_flat.index_copy_(
                    0,
                    flat_rows,
                    gradient_norm,
                )
                adapt_exploration_flat.index_copy_(
                    0,
                    flat_rows,
                    torch.rand(
                        active.numel(),
                        device=pre_action_nll.device,
                    ) < self.controller.exploration_rate,
                )
                write_exploration_flat.index_copy_(
                    0,
                    flat_rows,
                    torch.rand(
                        active.numel(),
                        device=pre_action_nll.device,
                    ) < self.controller.exploration_rate,
                )

            # Retrieval credit is additive and independent of the causal
            # working-memory transition.  Reduce all flat rows once after the
            # wavefront instead of launching one packed contraction per time
            # position.  The age clock follows the same event-count update.
            if not self.training_config.controller_only_finetune:
                cycle_usage_credit = packed_memory_info_flat["alpha"].new_zeros(
                    len(cycle_usage_node_ids),
                    packed_memory_info_flat["alpha"].size(-1),
                )
                self.tree.episodic_memory.credit_retrieval_packed(
                    alpha=packed_memory_info_flat["alpha"],
                    visited_node_indices=frontier_flat.visited_indices,
                    visited_node_mask=frontier_flat.visited_mask,
                    path_incidence=frontier_flat.path_incidence,
                    routing_weights=posterior_flat,
                    retrieval_probability=action_probability_flat[:, 1],
                    node_ids=cycle_usage_node_ids,
                    cycle_usage_accumulator=cycle_usage_credit,
                )
                self.tree.episodic_memory.step_age(cursor)

            action_probability_total.index_add_(
                0,
                sequence_index_flat,
                action_probability_flat,
            )
            gate_activation_total.index_add_(
                0,
                sequence_index_flat,
                (action_probability_flat >= 0.5).to(z_flat),
            )
            max_gradient_norm.scatter_reduce_(
                0,
                sequence_index_flat,
                gradient_norm_flat,
                reduce="amax",
                include_self=True,
            )
            action_counts_tensor.view(-1).index_add_(
                0,
                sequence_index_flat * len(Action) + action_index_flat,
                torch.ones(
                    cursor,
                    device=self.device,
                    dtype=torch.long,
                ),
            )

        # The recurrent GPU phase is complete.  The optimized v4/v5 path keeps
        # delayed Adapt/Write candidates as a tensor-backed probe buffer.  No
        # per-event Python request is created; only top-C probes cross the
        # host boundary below.  The controller-only/v6 fallback retains its
        # historical request objects because its disjoint future sampler is
        # not yet packed.
        if batched_write_path:
            flat_sequence_rows = ()
            flat_event_indices = ()
        else:
            flat_sequence_rows = [
                sequence_row
                for sequence_row, length in enumerate(lengths)
                for _ in range(length)
            ]
            flat_event_indices = [
                event_index
                for length in lengths
                for event_index in range(length)
            ]
        owner_indices_cpu = owner_indices_flat.detach().cpu().tolist()
        action_indices_cpu = action_index_flat.detach().cpu().tolist()
        adapt_exploration_cpu = (
            adapt_exploration_flat.detach().cpu().tolist()
        )
        write_exploration_cpu = (
            write_exploration_flat.detach().cpu().tolist()
        )
        assignment_counts_cpu = assignment_counts_tensor.detach().cpu().tolist()
        action_counts_cpu = action_counts_tensor.detach().cpu().tolist()
        frontier_node_counts_cpu = (
            frontier_node_counts_tensor.detach().cpu().tolist()
        )
        probe_buffer: Optional[Dict[str, Tensor]] = None
        if batched_write_path:
            sequence_rows_tensor = sequence_index_flat
            event_indices_tensor = (
                torch.arange(cursor, device=self.device, dtype=torch.long)
                - offsets_tensor.index_select(0, sequence_rows_tensor)
            )
            probe_buffer = {
                "sequence_rows": sequence_rows_tensor,
                "event_indices": event_indices_tensor,
                "ready_indices": event_indices_tensor
                + int(self.wake_config.write_horizon),
                "queries": query_flat.detach(),
                "frontier_node_indices": memory_output_flat[
                    "frontier_node_indices"
                ].detach(),
                "frontier_mask": memory_output_flat["frontier_mask"].detach(),
                "frontier_mass": memory_output_flat["frontier_mass"].detach(),
                "frontier_theta": memory_output_flat["frontier_theta"].detach(),
                # Delayed packing uses the raw policy probabilities, matching
                # the scalar request builder's write/queue metadata.
                "action_probabilities": raw_action_probability_flat.detach(),
                "write_gate": raw_action_probability_flat[:, 2].detach(),
                "novelty": novelty_flat.detach(),
                "queue_weight": self.controller.queue_weight(
                    raw_action_probability_flat
                ).detach(),
                "adapt_exploration": adapt_exploration_flat.detach(),
                "write_exploration": write_exploration_flat.detach(),
                "controller_inputs": controller_inputs_flat.detach(),
                "assimilation_theta": assimilation_theta_flat.detach(),
                "assimilation_grad": working_gradient_flat.detach(),
                "owner_indices": owner_indices_flat.detach(),
            }
            write_candidates = [int(length) for length in lengths]
        else:
            for flat_row, (sequence_row, event_index) in enumerate(zip(
                flat_sequence_rows,
                flat_event_indices,
            )):
                frontier_node_ids = frontier_rows[flat_row][0]
                owner_id = node_ids[owner_indices_cpu[flat_row]]
                action = action_values[action_indices_cpu[flat_row]]
                write_probe_contexts[sequence_row].append(
                    self._write_probe_context(
                        event_index=event_index,
                        memory_output=memory_output_flat,
                        batch_index=flat_row,
                        frontier_node_ids=frontier_node_ids,
                        query=query_flat[flat_row],
                        posterior=posterior_flat[flat_row],
                        working_delta=working_delta_snapshot_flat[flat_row],
                        retrieve_gate=action_probability_flat[flat_row, 1],
                        no_write_theta=assimilation_theta_flat[flat_row],
                    )
                )
                controller_inputs = {
                    name: controller_inputs_flat[flat_row, index]
                    for index, name in enumerate(controller_input_names)
                }
                request = self._make_write_request(
                    event_index,
                    owner_id,
                    query_flat[flat_row],
                    novelty_flat[flat_row],
                    memory_output_flat,
                    action_probability_flat[flat_row],
                    action_index_flat[flat_row],
                    posterior_flat[flat_row],
                    frontier_node_ids=frontier_node_ids,
                    hard_action=action,
                    batch_index=flat_row,
                    controller_inputs=controller_inputs,
                    assimilation_theta=assimilation_theta_flat[flat_row],
                    assimilation_grad=working_gradient_flat[flat_row],
                    future_contexts=write_probe_contexts[sequence_row],
                    raw_action_probabilities=raw_action_probability_flat[
                        flat_row
                    ],
                    exploration=bool(write_exploration_cpu[flat_row]),
                    controller_version=controller_version,
                )
                request["_sequence_row"] = sequence_row
                adapt_request = dict(request)
                adapt_request["exploration"] = bool(
                    adapt_exploration_cpu[flat_row]
                )
                adapt_probes[sequence_row].append(adapt_request)
                pending_writes[sequence_row].append(request)
                write_candidates[sequence_row] += 1

        assignment_counts = []
        action_counts = []
        actions = []
        frontier_node_counts = []
        for row, length in enumerate(lengths):
            assignment_counts.append(Counter({
                node_ids[index]: int(count)
                for index, count in enumerate(assignment_counts_cpu[row])
                if count
            }))
            action_counts.append(Counter({
                action.value: int(count)
                for action, count in zip(
                    action_values,
                    action_counts_cpu[row],
                )
                if count
            }))
            start = offsets[row]
            end = start + length
            actions.append([
                str(action_values[action_index])
                for action_index in action_indices_cpu[start:end]
            ])
            frontier_node_counts.append(Counter({
                node_ids[index]: int(count)
                for index, count in enumerate(frontier_node_counts_cpu[row])
                if count
            }))
        owner_depth_total = owner_depth_total_tensor.detach().cpu().tolist()
        owner_lca_count = owner_lca_count_tensor.detach().cpu().tolist()
        frontier_size_total = (
            frontier_size_total_tensor.detach().cpu().tolist()
        )
        frontier_visited_total = (
            frontier_visited_total_tensor.detach().cpu().tolist()
        )
        frontier_branch_total = (
            frontier_branch_total_tensor.detach().cpu().tolist()
        )
        expansion_utility_count = (
            expansion_utility_count_tensor.detach().cpu().tolist()
        )

        if cycle_usage_credit is not None:
            self.tree.episodic_memory.apply_cycle_usage_credit(
                cycle_usage_credit,
                cycle_usage_node_ids,
            )

        # Commit persistent writes only after the read-only wavefront
        # transaction is complete. Sequence order is deterministic here.
        accepted_write_counts = [0 for _ in range(batch_size)]
        append_counts = [0 for _ in range(batch_size)]
        refresh_counts = [0 for _ in range(batch_size)]
        write_decision_counts = [0 for _ in range(batch_size)]
        write_probe_counts = [0 for _ in range(batch_size)]
        write_gate_pass_counts = [0 for _ in range(batch_size)]
        write_utility_pass_counts = [0 for _ in range(batch_size)]
        accepted_write_utility_sums = [0.0 for _ in range(batch_size)]
        harmful_write_counts = [0 for _ in range(batch_size)]
        pending_counts = [0 for _ in range(batch_size)]
        if batched_write_path:
            if padded_wake is None:
                raise RuntimeError("batched Wake cache was not prepared")
            if probe_buffer is None:
                raise RuntimeError("batched Wake probe buffer was not prepared")
            # Adapt utility has the same candidate packing as Write; perform
            # selection and all future-window evaluations on device.  Python
            # metadata is materialized only for the selected top-C rows.
            selected_adapt_indices, adapt_packed = self._select_probe_buffer_batch(
                probe_buffer,
                topc=self.wake_config.controller_adapt_probe_topc,
                score="adapt",
                sequence_count=batch_size,
                exploration_key="adapt",
            )
            selected_adapt = self._materialize_probe_requests_batch(
                probe_buffer,
                selected_adapt_indices,
                exploration_key="adapt",
            )
            if selected_adapt_indices.numel():
                self._record_adapt_utility_batch(
                    sequences,
                    selected_adapt,
                    adapt_packed,
                    padded_wake,
                )
            write_summary = self._finalize_write_probe_batch(
                sequences,
                probe_buffer,
                lengths,
                padded_wake,
                frontier_static_cache.semantic_theta_table,
                controller_version=controller_version,
            )
            accepted_write_counts = write_summary.get(
                "accepted_write_counts",
                write_summary["write_counts"],
            )
            append_counts = write_summary.get(
                "append_counts",
                [0 for _ in range(batch_size)],
            )
            refresh_counts = write_summary.get(
                "refresh_counts",
                [0 for _ in range(batch_size)],
            )
            write_decision_counts = write_summary.get(
                "write_decision_counts", list(accepted_write_counts)
            )
            write_probe_counts = write_summary["write_probe_counts"]
            write_gate_pass_counts = write_summary[
                "write_gate_pass_counts"
            ]
            write_utility_pass_counts = write_summary[
                "write_utility_pass_counts"
            ]
            accepted_write_utility_sums = write_summary[
                "accepted_write_utility_sums"
            ]
            harmful_write_counts = write_summary["harmful_write_counts"]
            pending_counts = write_summary["pending_counts"]
        else:
            # Controller-only/v6 still has version-specific delayed sampling
            # semantics. Keep that path isolated until its sampler is batched.
            for row, sequence in enumerate(sequences):
                for request in self._select_adapt_probes(adapt_probes[row]):
                    self._record_adapt_utility(sequence, request)
                eligible = [
                    request
                    for request in pending_writes[row]
                    if request["ready_index"] < lengths[row]
                ]
                incomplete = [
                    request
                    for request in pending_writes[row]
                    if request["ready_index"] >= lengths[row]
                ]
                eligible = self._preselect_write_requests(eligible)
                write_probe_counts[row] = len(eligible)
                if eligible:
                    for request in eligible:
                        request["window_evidence"] = (
                            self._window_write_evidence(sequence, request)
                        )
                    probed = list(eligible)
                    write_gate_pass_counts[row] = sum(
                        float(request["write_gate"].detach().cpu())
                        >= float(
                            self.controller.calibration_thresholds[2]
                            .detach()
                            .cpu()
                        )
                        for request in eligible
                    )
                    write_utility_pass_counts[row] = sum(
                        float(
                            request["window_evidence"]["write_utility"]
                            .detach()
                            .cpu()
                        )
                        > 0.0
                        for request in eligible
                    )
                    eligible = [
                        request
                        for request in eligible
                        if (
                            not self.training_config.controller_only_finetune
                            or not request.get("exploration", False)
                        )
                        and self.controller.write_admissible(
                            request["write_gate"],
                            request["window_evidence"]["write_utility"],
                            request["window_evidence"]["priority"],
                            future_window_complete=True,
                            priority_threshold=(
                                self.wake_config.controller_priority_threshold
                            ),
                        )
                    ]
                else:
                    probed = []
                if eligible:
                    priorities = torch.stack([
                        request["window_evidence"]["priority"]
                        for request in eligible
                    ])
                    selected_indices = priorities.topk(
                        k=min(
                            len(eligible),
                            min(
                                4,
                                self.tree.frontier_routing.config.max_writes_per_sequence,
                            ),
                        )
                    ).indices.cpu().tolist()
                    selected = [
                        eligible[index] for index in selected_indices
                    ]
                else:
                    selected = []
                persistent_selected = []
                if not self.training_config.controller_only_finetune:
                    admission_actions = self._commit_write_requests_batch(
                        sequence,
                        selected,
                        return_actions=True,
                    )
                    persistent_selected = [
                        request
                        for request, action in zip(selected, admission_actions)
                        if action != "queue"
                    ]
                    append_counts[row] = sum(
                        action == "append" for action in admission_actions
                    )
                    refresh_counts[row] = sum(
                        action == "refresh" for action in admission_actions
                    )
                    source_id = int(torch.as_tensor(
                        sequence.get("source_index", row)
                    ).detach().cpu())
                    accepted_tokens = [
                        (source_id, int(request["event_index"]))
                        for request in persistent_selected
                    ]
                    selected_ids = {id(request) for request in selected}
                    shadow_candidates = []
                    for request in probed:
                        if id(request) in selected_ids:
                            continue
                        evidence_row = request["window_evidence"]
                        priority = (
                            request["queue_weight"]
                            * evidence_row["confidence"]
                            * evidence_row["bounded_gain"]
                        )
                        if float(priority.detach().cpu()) > 0.0:
                            shadow_candidates.append((priority, request))
                    shadow_candidates.sort(
                        key=lambda pair: float(
                            pair[0].detach().cpu()
                        ),
                        reverse=True,
                    )
                    shadow_records = []
                    for priority, request in shadow_candidates[:4]:
                        evidence_row = request["window_evidence"]
                        item = evidence_row["candidate_item"]
                        item.write_quality = float(
                            evidence_row["bounded_gain"].detach().cpu()
                        )
                        item.queue_weight = float((
                            request["queue_weight"]
                            * evidence_row["confidence"]
                        ).detach().cpu())
                        shadow_records.append({
                            "token": (
                                source_id,
                                int(request["event_index"]),
                            ),
                            "owner_id": evidence_row["owner_id"],
                            "priority": float(priority.detach().cpu()),
                            "item": item,
                        })
                    self._update_structural_evidence_buffer(
                        shadow_records,
                        accepted_tokens=accepted_tokens,
                    )
                if self.training_config.controller_write_ranking:
                    self.controller_utility_replay.finalize_write_group(
                        int(
                            torch.as_tensor(
                                sequence.get("source_index", -1)
                            )
                            .detach()
                            .cpu()
                        )
                    )
                # A selected row that the bank quarantines as ``queue`` is
                # not a persistent write yet. Keep decision/probe/gate
                # diagnostics on their pre-admission masks, but report
                # accepted/write metrics from the actual admission result.
                accepted_write_counts[row] = (
                    append_counts[row] + refresh_counts[row]
                )
                write_decision_counts[row] = len(selected)
                accepted_write_utility_sums[row] = sum(
                    float(
                        request["window_evidence"]["write_utility"]
                        .detach()
                        .cpu()
                    )
                    for request in persistent_selected
                )
                harmful_write_counts[row] = sum(
                    float(
                        request["window_evidence"]["write_utility"]
                        .detach()
                        .cpu()
                    )
                    <= 0.0
                    for request in persistent_selected
                )
                pending_counts[row] = len(incomplete)

        flat_sequence_row_tensor = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device),
            torch.tensor(lengths, device=self.device),
        )
        structural_mass_by_sequence = z_flat.new_zeros(batch_size)
        structural_mass_by_sequence.index_add_(
            0,
            flat_sequence_row_tensor,
            self.controller.queue_weight(raw_action_probability_flat),
        )
        structural_mass_cpu = (
            structural_mass_by_sequence.detach().cpu().tolist()
        )

        metric_matrix = torch.cat(
            [
                prediction_total[:, None].float(),
                wm_penalty_total[:, None].float(),
                write_penalty_total[:, None].float(),
                novelty_total[:, None].float(),
                max_similarity_total[:, None].float(),
                similarity_count_total[:, None].float(),
                max_gradient_norm[:, None].float(),
                expansion_utility_total[:, None].float(),
                posterior_entropy_total[:, None].float(),
                prior_posterior_kl_total[:, None].float(),
                (
                    action_probability_total
                    / torch.tensor(
                        lengths,
                        device=self.device,
                        dtype=action_probability_total.dtype,
                    )[:, None]
                ).float(),
            ],
            dim=-1,
        ).cpu().tolist()
        gate_activation_total_cpu = gate_activation_total.detach().cpu().tolist()

        prototype_count, evidence_mass = self._persistent_memory_stats()
        results = []
        for row, values in enumerate(metric_matrix):
            (
                prediction_value,
                wm_value,
                write_penalty_value,
                novelty_value,
                similarity_value,
                count_value,
                max_grad_value,
                utility_value,
                entropy_value,
                prior_kl_value,
                *mean_action_values,
            ) = values
            sequence_responsibility = (
                responsibility_sum[row] / lengths[row]
            )
            assignments = assignment_counts[row]
            results.append({
                "prediction_nll": prediction_value,
                "wm_penalty": wm_value,
                "write_penalty": write_penalty_value,
                "accepted_write_count": accepted_write_counts[row],
                "append_count": append_counts[row],
                "refresh_count": refresh_counts[row],
                "prototype_count": prototype_count,
                "evidence_mass": evidence_mass,
                # Compatibility alias for the pre-split metric name.
                "write_count": accepted_write_counts[row],
                "write_decision_count": write_decision_counts[row],
                "memorize_count": action_counts[row][
                    Action.MEMORIZE.value
                ],
                "memorize_argmax_count": action_counts[row][
                    Action.MEMORIZE.value
                ],
                "write_gate_active_count": int(
                    gate_activation_total_cpu[row][2]
                ),
                "write_candidate_count": write_candidates[row],
                "write_priority_pass_count": write_decision_counts[row],
                "write_window_complete_count": write_probe_counts[row],
                "write_accepted_count": accepted_write_counts[row],
                "write_retrieved_later_count": 0,
                "write_beneficial_count": max(
                    accepted_write_counts[row] - harmful_write_counts[row], 0
                ),
                "queue_split_count": action_counts[row][
                    Action.QUEUE_SPLIT.value
                ],
                "event_count": lengths[row],
                "raw_structural_mass": float(structural_mass_cpu[row]),
                "structural_observations": lengths[row],
                "responsibilities": [],
                "sequence_responsibility": sequence_responsibility,
                "actions": actions[row],
                "action_counts": dict(action_counts[row]),
                "mean_action_probabilities": {
                    action.value: mean_action_values[index]
                    for index, action in enumerate(Action)
                },
                "mean_gates": {
                    action.value: mean_action_values[index]
                    for index, action in enumerate(Action)
                },
                "gate_activation_rates": {
                    action.value: float(
                        gate_activation_total_cpu[row][index]
                    ) / lengths[row]
                    for index, action in enumerate(Action)
                },
                "memory_assignment_counts": dict(assignments),
                "sequence_owner_id": (
                    assignments.most_common(1)[0][0]
                    if assignments
                    else "root"
                ),
                "posterior_entropy": entropy_value / lengths[row],
                "prior_posterior_kl": prior_kl_value / lengths[row],
                "memory_owner_depth_mean": (
                    owner_depth_total[row] / lengths[row]
                ),
                "owner_lca_rate": owner_lca_count[row] / lengths[row],
                "write_candidates": write_candidates[row],
                "write_probe_count": write_probe_counts[row],
                "write_gate_pass_count": write_gate_pass_counts[row],
                "write_utility_pass_count": write_utility_pass_counts[row],
                "accepted_write_utility_sum": accepted_write_utility_sums[row],
                "harmful_write_count": harmful_write_counts[row],
                "mean_novelty": novelty_value / lengths[row],
                "mean_weighted_similarity": (
                    similarity_value / lengths[row]
                ),
                "mean_soft_count": count_value / lengths[row],
                "mean_max_similarity": similarity_value / lengths[row],
                "mean_similarity_count": count_value / lengths[row],
                "mean_frontier_size": (
                    frontier_size_total[row] / lengths[row]
                ),
                "mean_frontier_visited_nodes": (
                    frontier_visited_total[row] / lengths[row]
                ),
                "mean_frontier_branches": (
                    frontier_branch_total[row] / lengths[row]
                ),
                "frontier_node_counts": dict(
                    frontier_node_counts[row]
                ),
                "mean_expansion_utility": (
                    utility_value
                    / max(expansion_utility_count[row], 1)
                ),
                "pending_write_count": pending_counts[row],
                "max_gradient_norm": max_grad_value,
            })
        return results
