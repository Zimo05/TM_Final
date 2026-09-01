"""Light and Deep Sleep structural-update operations."""

from __future__ import annotations

import inspect

from Train.TrainingComponents import *  # noqa: F403
from Train.TrainingComponents import (
    _differentiable_merge_settings,
    _topology_prune_settings,
)


class TrainingSleepMixin:
    def _sync_split_modules(self) -> None:
        active_leaves = set(self.tree.leaf_ids)
        for leaf_id in set(self.split_modules).difference(active_leaves):
            del self.split_modules[leaf_id]
        for leaf_id in active_leaves:
            if leaf_id not in self.split_modules:
                self.split_modules[leaf_id] = SplitModule(
                    P=self.tree.param_dim,
                    z_dim=self.tree.episodic_memory.key_dim,
                    m_min=self.sleep_config.split_min_mass,
                    nll_fn=self.hawkes,
                ).to(self.device)

    def build_split_proposals(
        self,
        *,
        progress: Optional[tqdm] = None,
        max_evidence: Optional[int] = None,
    ) -> Dict[str, tuple[SplitModule, Dict[str, Any]]]:
        self._sync_split_modules()
        proposals = {}
        # Legacy checkpoints may contain rejected/shadow probes here. They are
        # deliberately ignored: Sleep structural evidence starts at the
        # persistent EpisodicMemory boundary.
        self.sleep_state["structural_evidence_buffer"] = {}
        for leaf_id in list(self.tree.leaf_ids):
            if progress is not None:
                progress.set_postfix(
                    phase="split",
                    leaf=leaf_id,
                    refresh=True,
                )
            if (
                self.sleep_config.require_split_trigger
                and self.controller.split_queues.get(leaf_id, 0) <= 0
            ):
                if progress is not None:
                    progress.update(1)
                continue
            module = self.split_modules[leaf_id]
            episodic_batch = module.build_split_batch_from_memory_bank(
                self.tree.episodic_memory.get_bank(leaf_id),
                max_items=max_evidence,
            )
            batch = episodic_batch
            if batch is None or batch.residuals.shape[0] < 2:
                if progress is not None:
                    progress.update(1)
                continue
            output = module.optimize_leaf_split(
                theta_sem=self.tree.semantic_theta(leaf_id),
                batch=batch,
                hawkes_ll=self.hawkes,
                num_steps=self.sleep_config.split_steps,
                lr=self.sleep_config.split_lr,
                m_min=self.sleep_config.split_min_mass,
                min_structural_strength=(
                    self.sleep_config.split_min_structural_strength
                ),
                min_effective_sample_size=(
                    self.sleep_config.split_min_effective_sample_size
                ),
                lambda_T=float(self.merge_lambda_T),
                # Splitting one leaf creates exactly one internal node.
                delta_complexity=1.0,
            )
            output["replay_weights"] = batch.weights.detach()
            proposals[leaf_id] = (module, output)
            if progress is not None:
                progress.update(1)
        return proposals

    def _light_sleep_settings(self) -> LightSleepSettings:
        return LightSleepSettings(
            replay_budget=self.sleep_config.light_replay_budget,
            scan_budget_multiplier=(
                self.sleep_config.light_scan_budget_multiplier
            ),
            min_per_leaf=self.sleep_config.light_min_per_leaf,
            routing_mass_mix=self.sleep_config.light_mass_mix,
            max_directions=self.sleep_config.light_max_directions,
            direction_similarity=(
                self.sleep_config.light_direction_similarity
            ),
            gain_evaluations_per_direction=(
                self.sleep_config.light_gain_evaluations_per_direction
            ),
            min_direction_support=(
                self.sleep_config.light_min_direction_support
            ),
            min_gain=self.sleep_config.light_min_gain,
            coherence_threshold=(
                self.sleep_config.light_coherence_threshold
            ),
            alpha_max=self.sleep_config.light_alpha_max,
            trust_radius=self.sleep_config.light_trust_radius,
            gain_reference=self.sleep_config.light_gain_reference,
        )

    @torch.no_grad()
    def _cheap_merge_candidate_count(self) -> int:
        min_replay = _differentiable_merge_settings(
            self.structure_config.merge_kwargs
        )["min_replay"]
        count = 0
        for node_a, node_b in leaf_sibling_pairs(self.tree):
            replay_count = 0
            for node_id in (node_a, node_b):
                bank = self.tree.episodic_memory.banks.get(node_id)
                if bank is None:
                    continue
                replay_count += sum(
                    window is not None and window.end_idx > window.start_idx
                    for window in bank.windows
                )
            if replay_count >= min_replay:
                count += 1
        return count

    @torch.no_grad()
    def _continuous_split_demand(
        self,
        *,
        accepted_writes: int = 0,
    ) -> Dict[str, float]:
        """Estimate structural pressure from persistent-memory votes only."""
        if accepted_writes < 0:
            raise ValueError("accepted_writes must be non-negative")
        previous_snapshot = dict(
            self.sleep_state.get("split_queue_snapshot", {})
        )
        next_snapshot: Dict[str, float] = {}
        queue_increment = 0.0
        active_increment_count = 0
        queue_total = 0.0
        for leaf_id in self.tree.leaf_ids:
            current = max(
                float(self.controller.split_queues.get(leaf_id, 0.0)),
                0.0,
            )
            previous = max(
                float(previous_snapshot.get(leaf_id, 0.0)),
                0.0,
            )
            increment = max(current - previous, 0.0)
            queue_increment += increment
            queue_total += current
            active_increment_count += int(increment > 0.0)
            next_snapshot[leaf_id] = current
        # Raw controller pressure is retained only as a discarded compatibility
        # diagnostic. It cannot drive topology before memory persistence.
        discarded_structural_mass = max(float(
            self.sleep_state.get("structural_mass_since_sleep", 0.0)
        ), 0.0)
        discarded_structural_observations = max(int(
            self.sleep_state.get(
                "structural_observations_since_sleep", 0
            )
        ), 0)
        observation = 1.0 - math.exp(
            -queue_increment / self.sleep_config.deep_split_queue_scale
        )
        previous_ema = float(
            self.sleep_state.get("structural_demand_ema", 0.0)
        )
        decay = self.sleep_config.deep_split_demand_decay
        demand = decay * previous_ema + (1.0 - decay) * observation
        self.sleep_state["split_queue_snapshot"] = next_snapshot
        self.sleep_state["structural_demand_ema"] = float(demand)
        self.sleep_state["structural_mass_since_sleep"] = 0.0
        self.sleep_state["structural_observations_since_sleep"] = 0
        return {
            "value": float(min(max(demand, 0.0), 1.0)),
            "observation": float(observation),
            "queue_increment": float(queue_increment),
            "queue_total": float(queue_total),
            "active_increment_count": float(active_increment_count),
            "accepted_writes": float(accepted_writes),
            "structural_mass": 0.0,
            "structural_observations": 0.0,
            "discarded_raw_structural_mass": float(
                discarded_structural_mass
            ),
            "discarded_raw_structural_observations": float(
                discarded_structural_observations
            ),
        }

    @torch.no_grad()
    def _deep_sleep_features(
        self,
        low_mass: Sequence[str],
        *,
        epoch_label: int,
        predictive_residual_utility: Optional[float] = None,
        raw_residual_energy: float = 0.0,
        residual_energy: Optional[float] = None,
        accepted_writes: int = 0,
    ) -> Dict[str, Any]:
        """Build detached post-Light sufficient statistics for the gate.

        ``residual_energy`` is a checkpoint/test compatibility alias.  The
        gate's residual feature is exclusively predictive residual utility;
        raw parameter norm is returned only as a debug diagnostic.
        """
        del low_mass
        if predictive_residual_utility is None:
            if residual_energy is None:
                raise ValueError("predictive residual utility is required")
            predictive_residual_utility = float(residual_energy)
        predictive_residual_utility = float(predictive_residual_utility)
        memory_count = sum(
            len(bank)
            for bank in self.tree.episodic_memory.banks.values()
        )
        effective_memory_count = sum(
            float(bank.write_quality.clamp_min(0.0).sum().detach().cpu())
            for bank in self.tree.episodic_memory.banks.values()
            if len(bank) > 0
        )
        memory_budget = (
            self.tree.episodic_memory.capacity_per_node
            * max(len(self.tree.all_node_ids), 1)
            * self.sleep_config.deep_memory_budget_multiplier
        )
        split_demand = self._continuous_split_demand(
            accepted_writes=accepted_writes,
        )
        prune_settings = _topology_prune_settings(
            self.structure_config.topology_prune_kwargs
        )
        current_complexity = tree_complexity(self.tree)
        if self.merge_budget_KT is None and current_complexity > 0.0:
            # Freeze the first meaningful target.  It must not be recomputed
            # as 0.95 * current complexity on every scheduler evaluation.
            self.merge_budget_KT = (
                prune_settings["budget_ratio"] * current_complexity
            )
        target_complexity = (
            float(self.merge_budget_KT)
            if self.merge_budget_KT is not None
            else 1.0
        )
        residual_reference = max(
            self.sleep_config.deep_residual_energy_budget, 1e-8
        )
        residual_ratio = (
            max(predictive_residual_utility, 0.0) / residual_reference
        )
        memory_ratio = effective_memory_count / max(memory_budget, 1e-8)
        topology_excess = max(current_complexity - target_complexity, 0.0)
        topology_ratio = topology_excess / max(target_complexity, 1e-8)
        # The learned monotone gate receives comparable bounded pressures.
        # Raw ratios remain below as diagnostics only.
        residual_pressure = (
            max(predictive_residual_utility, 0.0)
            / (
                max(predictive_residual_utility, 0.0)
                + residual_reference
            )
        )
        memory_pressure = min(max(memory_ratio, 0.0), 1.0)
        topology_pressure = topology_ratio / (1.0 + topology_ratio)
        elapsed = max(
            epoch_label
            - int(self.sleep_state.get("last_deep_epoch", 0)),
            0,
        )
        deep_availability = 1.0 - math.exp(
            -float(elapsed) / self.sleep_config.deep_availability_tau
        )
        values = [
            residual_pressure,
            memory_pressure,
            topology_pressure,
            split_demand["value"],
        ]
        prune_candidates = len(candidate_prune_parents(self.tree))
        return {
            "tensor": torch.tensor(
                values,
                device=self.device,
                dtype=self.deep_sleep_gate.bias.dtype,
            ),
            "value": 0.0,
            "residual": float(values[0]),
            "memory": float(values[1]),
            "topology": float(values[2]),
            "structural": float(values[3]),
            "deep_availability": float(deep_availability),
            "residual_ratio": float(residual_ratio),
            "memory_ratio": float(memory_ratio),
            "topology_ratio": float(topology_ratio),
            "topology_excess": float(topology_excess),
            "residual_energy": predictive_residual_utility,
            "predictive_residual_utility": predictive_residual_utility,
            "raw_residual_energy": float(raw_residual_energy),
            "memory_count": float(memory_count),
            "effective_memory_count": float(effective_memory_count),
            "memory_budget": float(memory_budget),
            "split_candidates": split_demand["active_increment_count"],
            "split_queue_increment": split_demand["queue_increment"],
            "split_queue_total": split_demand["queue_total"],
            "split_observation": split_demand["observation"],
            "split_accepted_writes": split_demand["accepted_writes"],
            "merge_candidates": 0.0,
            "topology_prune_candidates": float(prune_candidates),
            "topology_complexity": float(current_complexity),
            "topology_budget": float(target_complexity),
            "low_mass_candidates": 0.0,
        }

    @torch.no_grad()
    def _estimate_deep_gain(
        self,
        candidates,
    ) -> Dict[str, Any]:
        """Return the common null-relative gain used to train DeepGate."""
        component_gains = {
            candidate.action_id: float(torch.as_tensor(
                candidate.conservative_gain
            ).detach().cpu())
            for candidate in candidates
            if candidate.eligible
        }
        gain = max(0.0, max(component_gains.values(), default=0.0))
        return {
            "value": float(gain),
            "components": component_gains,
        }

    def _select_unified_topology_action(
        self,
        split_proposals,
        merge_objective,
        topology_prune_proposals,
    ):
        revision = int(self.sleep_state.get("topology_revision", 0))
        candidates = []
        for leaf_id, (module, output) in split_proposals.items():
            candidates.append(build_split_candidate(
                leaf_id,
                module,
                output,
                topology_revision=revision,
                lambda_T=float(self.merge_lambda_T),
                uncertainty_kappa=self.sleep_config.action_uncertainty_kappa,
                min_child_effective_mass=self.sleep_config.split_min_mass,
                min_structural_strength=(
                    self.sleep_config.split_min_structural_strength
                ),
                min_effective_sample_size=(
                    self.sleep_config.split_min_effective_sample_size
                ),
            ))
        merge_pairs = tuple(merge_objective.get("pairs", ()))
        current_cycle = int(self.sleep_state.get("deep_cycle_count", 0))
        merge_settings = _differentiable_merge_settings(
            self.structure_config.merge_kwargs
        )
        merge_statistics = (
            torch.stack((
                merge_objective["delta_keep"].detach(),
                merge_objective["delta_keep_variance"].detach(),
                merge_objective["effective_sample_size"].detach(),
            ), dim=-1).cpu().tolist()
            if merge_pairs else ()
        )
        for index, (node_a, node_b) in enumerate(merge_pairs):
            parent_id = merge_objective["parents"][index]
            delta_keep, delta_variance, effective_sample_size = (
                merge_statistics[index]
            )
            branch_support = (
                merge_objective["branch_support"][index]
                .detach().cpu().tolist()
            )
            child_weights = (
                merge_objective["child_weights"][index]
                .detach().cpu().tolist()
            )
            candidates.append(build_merge_candidate(
                node_a,
                node_b,
                parent_id,
                collapse_snapshot_signature(self.tree, parent_id),
                delta_keep=float(delta_keep),
                delta_keep_variance=float(delta_variance),
                replay_size=int(merge_objective["replay_counts"][index]),
                effective_sample_size=float(effective_sample_size),
                branch_support=branch_support,
                topology_revision=revision,
                lambda_T=float(self.merge_lambda_T),
                uncertainty_kappa=self.sleep_config.action_uncertainty_kappa,
                min_replay_size=merge_settings["min_replay"],
                min_effective_sample_size=merge_settings[
                    "min_effective_sample_size"
                ],
                min_branch_support=merge_settings["min_branch_support"],
                target_parent_theta=(
                    merge_objective["merge_parent_theta"][index]
                ),
                child_weights=child_weights,
                tpp_divergence=float(
                    merge_objective["tpp_divergence"][index]
                    .detach().cpu()
                ),
                dynamics_weight=merge_settings["dynamics_weight"],
            ))
        for proposal in topology_prune_proposals.values():
            candidates.append(build_topology_prune_candidate(
                proposal,
                topology_revision=revision,
            ))
        candidates = smooth_candidate_gains(
            candidates,
            self.sleep_state.setdefault("unified_action_evidence", {}),
            decay=self.sleep_config.action_gain_ema_decay,
            temporal_uncertainty_kappa=(
                self.sleep_config.action_uncertainty_kappa
            ),
            min_observations=self.sleep_config.action_min_observations,
        )
        # Smooth the counterfactual evidence itself, then subtract the
        # instantaneous local edit cost exactly as G_tilde = G - C_edit.
        candidates = apply_structural_inertia(
            candidates,
            self.sleep_state.setdefault("topology_last_edit_cycle", {}),
            current_cycle=current_cycle,
            strength=self.sleep_config.topology_inertia_strength,
            tau=self.sleep_config.topology_inertia_tau,
        )
        # TopologyPrune already applies its replay uncertainty. The common
        # selector owns action-type/target arbitration and the Null boundary.
        selection = self.topology_selector(
            candidates,
        )
        return candidates, selection

    def _optimize_topology_selector(
        self,
        selection,
    ) -> Dict[str, float]:
        objective = self.topology_selector.objective(
            selection,
            entropy_weight=(
                self.sleep_config.action_selector_entropy_weight
            ),
        )
        self.topology_selector_optimizer.zero_grad(set_to_none=True)
        objective["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            self.topology_selector.parameters(),
            self.sleep_config.action_selector_grad_clip,
        )
        self.topology_selector_optimizer.step()
        self.topology_selector_optimizer.zero_grad(set_to_none=True)
        return {
            key: float(value.detach().cpu())
            for key, value in objective.items()
        }

    def _optimize_deep_sleep_gate(
        self,
        gate_output: Mapping[str, Any],
        *,
        estimated_gain: float,
    ) -> Dict[str, float]:
        objective = self.deep_sleep_gate.objective(
            gate_output,
            estimated_gain=estimated_gain,
            computation_cost=self.sleep_config.deep_computation_cost,
            prior_probability=self.sleep_config.deep_prior_probability,
            prior_weight=self.sleep_config.deep_prior_weight,
        )
        self.deep_gate_optimizer.zero_grad(set_to_none=True)
        objective["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            self.deep_sleep_gate.parameters(),
            self.sleep_config.deep_gate_grad_clip,
        )
        self.deep_gate_optimizer.step()
        self.deep_gate_optimizer.zero_grad(set_to_none=True)
        return {
            key: float(value.detach().cpu())
            for key, value in objective.items()
        }

    def _call_deep_sleep_gate(
        self,
        pressure: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Call either side of the structural-feature gate migration.

        The current gate learns structural pressure as its fourth feature.
        Older deployments consume three learned features and require the same
        value separately as ``split_demand``.  Dispatch from the gate's own
        contract so partially synchronized workers do not fail mid-Sleep.
        """
        features = torch.as_tensor(pressure["tensor"])
        expected_feature_count = len(self.deep_sleep_gate.feature_names)
        if features.numel() != expected_feature_count:
            legacy_transition = (
                features.numel() == 4
                and expected_feature_count == 3
            )
            if not legacy_transition:
                raise ValueError(
                    "Deep Sleep pressure/gate feature mismatch: "
                    f"pressure has {features.numel()} values but gate expects "
                    f"{expected_feature_count}"
                )
            features = features[:expected_feature_count]

        gate_parameters = inspect.signature(
            self.deep_sleep_gate.forward
        ).parameters
        gate_kwargs = {
            "deep_availability": pressure["deep_availability"],
        }
        if "split_demand" in gate_parameters:
            gate_kwargs["split_demand"] = pressure["structural"]
        return self.deep_sleep_gate(features, **gate_kwargs)

    @torch.no_grad()
    def _encode_topology_prune_window(self, window) -> Tensor:
        """Re-encode a stored full prefix under the frozen post-Light model."""
        sequence = {
            "times": window.times,
            "types": window.types,
        }
        cached = {
            EVENT_TIME_FEATURES_KEY: window.event_time_features,
            HAWKES_HISTORY_STATS_KEY: window.hawkes_history_stats,
            HAWKES_INTERVAL_STATS_KEY: window.hawkes_interval_stats,
            HAWKES_CACHE_SIGNATURE_KEY: window.hawkes_cache_signature,
        }
        sequence.update({key: value for key, value in cached.items() if value is not None})
        return self._encode_memory_event(
            sequence,
            int(window.start_idx),
        ).reshape(-1)

    @torch.no_grad()
    def _encode_topology_prune_windows(self, windows) -> Tensor:
        """Encode variable-length replay prefixes in one GPU GRU batch."""
        if not windows:
            return torch.empty(
                0,
                self.tree.z_dim,
                device=self.device,
            )
        if not isinstance(self.encoder, CausalPrefixEncoder):
            return torch.stack([
                self._encode_topology_prune_window(window)
                for window in windows
            ])

        lengths = torch.tensor(
            [int(window.times.numel()) for window in windows],
            device=self.device,
            dtype=torch.long,
        )
        times = nn.utils.rnn.pad_sequence(
            [window.times for window in windows],
            batch_first=True,
        )
        types = nn.utils.rnn.pad_sequence(
            [window.types for window in windows],
            batch_first=True,
        )
        feature_rows = []
        for window in windows:
            features = window.event_time_features
            if features is None:
                previous = torch.cat([
                    window.times.new_zeros(1),
                    window.times[:-1],
                ])
                features = torch.stack([
                    torch.log1p(window.times),
                    torch.log1p(
                        (window.times - previous).clamp_min(0.0)
                    ),
                ], dim=-1)
            feature_rows.append(features)
        time_features = nn.utils.rnn.pad_sequence(
            feature_rows,
            batch_first=True,
        )
        valid = (
            torch.arange(times.size(1), device=self.device)[None, :]
            < lengths[:, None]
        )
        padded, _ = self.encoder.forward_padded_prefix(
            times,
            types,
            valid,
            time_features=time_features,
        )
        start_indices = torch.tensor(
            [int(window.start_idx) for window in windows],
            device=self.device,
            dtype=torch.long,
        )
        return padded[
            torch.arange(len(windows), device=self.device),
            start_indices,
        ]

    def _evaluate_topology_prune(
        self,
        *,
        max_replay: int,
        allow_candidate: bool,
        update_persistence: bool = True,
    ):
        settings = _topology_prune_settings(
            self.structure_config.topology_prune_kwargs
        )
        current_complexity = tree_complexity(self.tree)
        # Do not freeze a zero topology budget while the tree is still a
        # single root leaf.  A forced first Split evaluates Deep Sleep before
        # committing that Split; the useful initial budget must therefore be
        # established on the first later snapshot that actually has a branch.
        if self.merge_budget_KT is None and current_complexity > 0.0:
            self.merge_budget_KT = (
                settings["budget_ratio"] * current_complexity
            )
        budget_KT = (
            0.0
            if self.merge_budget_KT is None
            else float(self.merge_budget_KT)
        )
        evaluation_kwargs = {
            key: value
            for key, value in settings.items()
            if key not in {
                "patience",
                "budget_ratio",
                "dual_lr",
                "dual_initial",
                "near_zero_confirmations",
            }
        }
        evaluation_kwargs["max_replay"] = min(
            int(max_replay),
            int(settings["max_replay"]),
        )
        proposals, metrics = evaluate_topology_prune(
            self.tree,
            self.hawkes,
            lambda_T=self.merge_lambda_T,
            embedding_fn=self._encode_topology_prune_window,
            embedding_batch_fn=self._encode_topology_prune_windows,
            **evaluation_kwargs,
        )
        if update_persistence:
            proposals = apply_prune_persistence(
                self.tree,
                proposals,
                patience=settings["patience"],
                allow_candidate=allow_candidate,
                near_zero_confirmations=settings[
                    "near_zero_confirmations"
                ],
            )
        metrics["ready_count"] = float(sum(
            proposal.persistence_ok for proposal in proposals.values()
        ))
        metrics["budget_KT"] = budget_KT
        metrics["constraint"] = (
            metrics["expected_complexity"] - budget_KT
        )
        metrics["lambda_T"] = float(self.merge_lambda_T)
        return proposals, metrics

    @torch.no_grad()
    def _evaluate_merge_candidates(
        self,
        *,
        max_replay: int,
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Evaluate frozen sibling Merge counterfactuals without mutation."""
        settings = _differentiable_merge_settings(
            self.structure_config.merge_kwargs
        )
        objective = compute_differentiable_merge_objective(
            self.tree,
            leaf_sibling_pairs(self.tree),
            self.hawkes.decays,
            lambda_T=self.merge_lambda_T,
            budget_KT=self.merge_budget_KT,
            gate_temperature=settings["gate_temperature"],
            min_replay=settings["min_replay"],
            normalize_by_events=settings["normalize_by_events"],
            hawkes_ll=self.hawkes,
            max_replay=max_replay,
            embedding_fn=self._encode_topology_prune_window,
            embedding_batch_fn=self._encode_topology_prune_windows,
            stale_weight=settings["stale_weight"],
            uncertainty_kappa=(
                self.sleep_config.action_uncertainty_kappa
            ),
            dynamics_weight=settings["dynamics_weight"],
        )
        gains = (
            self.merge_lambda_T - objective["retention_cost"].detach()
        )
        return objective, {
            "lambda_T": float(self.merge_lambda_T),
            "candidate_count": float(len(objective["pairs"])),
            "replay_windows": float(sum(objective["replay_counts"])),
            "mean_gain": (
                float(gains.mean().cpu()) if gains.numel() else 0.0
            ),
            "positive_gain_count": float(
                (gains > 0.0).sum().cpu() if gains.numel() else 0.0
            ),
        }

    def _run_full_sleep_cycle_impl(
        self,
        responsibilities: Tensor,
        *,
        allow_topology_prune: bool = True,
        epoch: Optional[int] = None,
        show_progress: bool = False,
        accepted_writes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run Light Sleep, then route through the differentiable Deep gate."""
        self.tree.train()
        self.deep_sleep_gate.train()
        self.topology_selector.train()
        self.optimizer.zero_grad(set_to_none=True)
        leaf_count = len(self.tree.leaf_ids)
        epoch_label = (
            self.completed_epochs + 1 if epoch is None else int(epoch)
        )
        memory_before = sum(
            len(bank)
            for bank in self.tree.episodic_memory.banks.values()
        )
        new_memories = max(
            0,
            memory_before
            - int(self.sleep_state.get("last_memory_count", 0)),
        )
        accepted_writes = (
            int(new_memories)
            if accepted_writes is None else int(accepted_writes)
        )
        if accepted_writes < 0:
            raise ValueError("accepted_writes must be non-negative")
        with tqdm(
            total=leaf_count + 4,
            desc=f"[Epoch {epoch_label:03d}] Sleep",
            unit="step",
            ascii=True,
            dynamic_ncols=False,
            ncols=110,
            mininterval=2.0,
            maxinterval=10.0,
            smoothing=0.1,
            leave=True,
            disable=not show_progress,
            file=sys.stdout,
        ) as sleep_progress:
            # Statistics are consolidated once per Light cycle, independently
            # of whether the rarer Deep transaction runs.
            self.tree.episodic_memory.materialize_all_ages()
            self.tree.episodic_memory.consolidate_sleep_cycle(
                usage_decay=self.structure_config.usage_decay,
                effective_usage_threshold=(
                    self.structure_config.effective_usage_threshold
                ),
            )
            observed_leaf_rows = responsibilities.sum(dim=-1) > 0.0
            if bool(observed_leaf_rows.any()):
                update_leaf_mass(
                    self.tree,
                    responsibilities[observed_leaf_rows],
                    ema_decay=self.structure_config.leaf_mass_ema_decay,
                )
            # Routing mass remains diagnostic evidence only. It no longer
            # accumulates a streak that can delete topology.
            for leaf_id in self.tree.leaf_ids:
                self.tree.low_mass_streak[leaf_id] = 0
            low_mass = []
            sleep_progress.update(1)

            light_result = run_light_sleep(
                self.tree,
                self.hawkes,
                optimizer=self.optimizer,
                settings=self._light_sleep_settings(),
                state=self.sleep_state.setdefault("light_index", {}),
            )
            self.sleep_state["last_light_epoch"] = epoch_label
            sleep_progress.set_postfix(
                phase="light",
                replay=light_result["replay_windows"],
                absorbed=light_result["absorbed_leaves"],
                refresh=True,
            )
            sleep_progress.update(1)

            cycle_count = (
                int(self.sleep_state.get("deep_cycle_count", 0)) + 1
            )
            self.sleep_state["deep_cycle_count"] = cycle_count
            pressure = self._deep_sleep_features(
                low_mass,
                epoch_label=epoch_label,
                predictive_residual_utility=float(
                    light_result["predictive_residual_utility"]
                ),
                raw_residual_energy=float(
                    light_result["raw_residual_energy"]
                ),
                accepted_writes=accepted_writes,
            )
            gate_output = self._call_deep_sleep_gate(pressure)
            deep_triggered = bool(gate_output["hard_gate"])
            probe_due = (
                cycle_count % self.sleep_config.deep_probe_interval == 0
            )
            evaluation_needed = bool(deep_triggered or probe_due)
            pressure.pop("tensor")
            pressure["value"] = float(
                torch.as_tensor(
                    gate_output["probability"]
                ).detach().cpu()
            )
            pressure["base_probability"] = float(
                torch.as_tensor(
                    gate_output["base_probability"]
                ).detach().cpu()
            )
            pressure["joint_pressure"] = float(
                torch.as_tensor(
                    gate_output["joint_pressure"]
                ).detach().cpu()
            )
            pressure["gate"] = float(
                torch.as_tensor(gate_output["gate"]).detach().cpu()
            )
            pressure["activation_probability"] = float(
                torch.as_tensor(
                    gate_output["activation_probability"]
                ).detach().cpu()
            )
            pressure["accumulator"] = float(
                torch.as_tensor(
                    gate_output["accumulator"]
                ).detach().cpu()
            )
            pressure["evidence"] = float(
                torch.as_tensor(
                    gate_output["evidence"]
                ).detach().cpu()
            )

            split_proposals = {}
            merge_objective = {"pairs": []}
            merge_result = {
                "lambda_T": float(self.merge_lambda_T),
                "candidate_count": 0.0,
                "replay_windows": 0.0,
                "mean_gain": 0.0,
                "positive_gain_count": 0.0,
            }
            topology_prune_proposals = {}
            unified_candidates = ()
            selection = None
            selection_log = {
                "selected_action_id": "null",
                "selected_kind": "null",
                "selected_action": "null",
                "selected_gain": 0.0,
                "candidate_count": 0,
                "probabilities": {"null": 1.0},
                "candidates": [],
                "objective": {
                    "loss": 0.0,
                    "expected_gain": 0.0,
                    "entropy": 0.0,
                },
            }
            deep_gain = {"value": 0.0, "components": {}}
            gate_objective = {
                "loss": 0.0,
                "reward_term": 0.0,
                "cost_term": 0.0,
                "prior_term": 0.0,
                "kl": 0.0,
                "estimated_gain": 0.0,
            }
            topology_prune_result = {
                "candidate_count": 0.0,
                "eligible_count": 0.0,
                "ready_count": 0.0,
                "near_zero_count": 0.0,
                "replay_windows": 0.0,
                "expected_prunes": 0.0,
                "current_complexity": tree_complexity(self.tree),
                "expected_complexity": tree_complexity(self.tree),
                "mean_prune_probability": 0.0,
                "mean_retention_cost": 0.0,
                "budget_KT": float(
                    0.0 if self.merge_budget_KT is None
                    else self.merge_budget_KT
                ),
                "constraint": 0.0,
                "lambda_T": float(self.merge_lambda_T),
            }
            if evaluation_needed:
                topology_prune_proposals, topology_prune_result = (
                    self._evaluate_topology_prune(
                        max_replay=self.sleep_config.deep_evidence_budget,
                        allow_candidate=(
                            deep_triggered and allow_topology_prune
                        ),
                        update_persistence=deep_triggered,
                    )
                )
                merge_objective, merge_result = (
                    self._evaluate_merge_candidates(
                        max_replay=self.sleep_config.deep_evidence_budget,
                    )
                )
                split_proposals = self.build_split_proposals(
                    progress=(
                        sleep_progress if show_progress else None
                    ),
                    max_evidence=self.sleep_config.deep_evidence_budget,
                )
                if not show_progress:
                    sleep_progress.update(leaf_count)
                unified_candidates, selection = (
                    self._select_unified_topology_action(
                        split_proposals,
                        merge_objective,
                        topology_prune_proposals,
                    )
                )
                selection_log = selection.as_log_dict()
                selection_log["objective"] = (
                    self._optimize_topology_selector(selection)
                )
                deep_gain = self._estimate_deep_gain(
                    unified_candidates,
                )
                gate_objective = self._optimize_deep_sleep_gate(
                    gate_output,
                    estimated_gain=deep_gain["value"],
                )
                # The shadow buffer is single-use structural evidence. It is
                # consumed by this frozen Deep evaluation and never promoted
                # into the online retrieval bank.
                self.sleep_state["structural_evidence_buffer"] = {}
            else:
                sleep_progress.update(leaf_count)

            if deep_triggered:
                promotion_kwargs = dict(
                    self.structure_config.promotion_kwargs
                )
                promotion_kwargs.setdefault(
                    "max_candidates",
                    self.sleep_config.deep_evidence_budget,
                )
                promotion_kwargs.setdefault(
                    "max_references",
                    self.sleep_config.deep_evidence_budget,
                )
                transaction = run_sleep_cycle(
                    tree=self.tree,
                    responsibilities=responsibilities,
                    decays=self.hawkes.decays,
                    hawkes_ll=self.hawkes,
                    selection=selection,
                    optimizer=self.optimizer,
                    controllers=(self.controller,),
                    usage_decay=self.structure_config.usage_decay,
                    effective_usage_threshold=(
                        self.structure_config.effective_usage_threshold
                    ),
                    leaf_mass_ema_decay=(
                        self.structure_config.leaf_mass_ema_decay
                    ),
                    split_memory_hard_threshold=(
                        self.structure_config.split_memory_hard_threshold
                    ),
                    split_init_steps=self.sleep_config.split_init_steps,
                    split_init_lr=self.sleep_config.split_init_lr,
                    allow_topology_prune=allow_topology_prune,
                    promotion_kwargs=promotion_kwargs,
                    statistics_prepared=True,
                )
                topology_settings = _topology_prune_settings(
                    self.structure_config.topology_prune_kwargs
                )
                post_complexity = tree_complexity(self.tree)
                budget_KT = float(
                    0.0 if self.merge_budget_KT is None
                    else self.merge_budget_KT
                )
                topology_prune_result["current_complexity"] = post_complexity
                topology_prune_result["constraint"] = post_complexity - budget_KT
                self.merge_lambda_T = max(
                    0.0,
                    self.merge_lambda_T
                    + topology_settings["dual_lr"]
                    * topology_prune_result["constraint"],
                )
                topology_prune_result["lambda_T"] = float(
                    self.merge_lambda_T
                )
                topology_actions = [
                    action
                    for action in transaction["actions"]
                    if action["action"] in {
                        "split", "merge", "topology_prune"
                    }
                ]
                last_edit = self.sleep_state.setdefault(
                    "topology_last_edit_cycle", {}
                )
                active_nodes = set(self.tree.nodes)
                for node_id in tuple(last_edit):
                    if node_id not in active_nodes:
                        last_edit.pop(node_id, None)
                for action in topology_actions:
                    region_id = (
                        action["node"]
                        if action["action"] == "split"
                        else action["parent"]
                    )
                    if region_id not in self.tree.nodes:
                        continue
                    last_edit[region_id] = cycle_count
                    parent_id = self.tree.nodes[region_id].parent
                    if parent_id is not None:
                        last_edit[parent_id] = cycle_count
                if topology_actions:
                    self.sleep_state["topology_revision"] = (
                        int(self.sleep_state.get("topology_revision", 0)) + 1
                    )
                    self.sleep_state["unified_action_evidence"] = {}
                mode = "deep"
            else:
                transaction = {
                    "leaf_snapshot": list(self.tree.leaf_ids),
                    "leaf_mass_ema": dict(self.tree.mass_ema),
                    "actions": [],
                    "memory": {},
                    "leaf_ids": list(self.tree.leaf_ids),
                    "topology_prune_enabled": bool(allow_topology_prune),
                }
                mode = (
                    "light"
                    if light_result["active_leaves"] > 0
                    else "none"
                )
            sleep_progress.set_postfix(
                phase=mode,
                pressure=f"{pressure['value']:.3f}",
                actions=len(transaction["actions"]),
                refresh=True,
            )
            sleep_progress.update(1)

            residual_energy = float(light_result["residual_energy"])
            memory_count = sum(
                len(bank)
                for bank in self.tree.episodic_memory.banks.values()
            )
            sleep_progress.update(1)

        self._sync_split_modules()
        self._reconcile_optimizer_parameters()
        self.sleep_state["last_memory_count"] = sum(
            len(bank)
            for bank in self.tree.episodic_memory.banks.values()
        )
        return {
            "mode": mode,
            "memory_count": float(memory_count),
            "residual_energy": float(residual_energy),
            "predictive_residual_utility": float(residual_energy),
            "new_memories": int(new_memories),
            "light": light_result,
            "deep_pressure": pressure,
            "deep_gate": {
                "probability": pressure["value"],
                "sample": pressure["gate"],
                "activation_probability": pressure[
                    "activation_probability"
                ],
                "hard_gate": bool(deep_triggered),
                "probe": bool(probe_due and not deep_triggered),
                "evaluated": bool(evaluation_needed),
                "gain": deep_gain,
                "objective": gate_objective,
            },
            "unified_topology": selection_log,
            "merge": merge_result,
            "topology_prune": topology_prune_result,
            "transaction": transaction,
        }

    def train_sleep(
        self,
        responsibilities: Tensor,
        *,
        allow_topology_prune: bool = True,
        epoch: Optional[int] = None,
        show_progress: bool = False,
        accepted_writes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the sole production Sleep state machine."""
        from Sleep.SleepOrchestrator import (
            SleepOrchestrator,
            run_full_sleep_cycle,
        )

        sleep_kwargs = {
            "allow_topology_prune": allow_topology_prune,
            "epoch": epoch,
            "show_progress": show_progress,
        }
        # Accept mixed deployments where TrainingSleep has the exact write
        # count API but SleepOrchestrator still uses the older signature.
        if "accepted_writes" in inspect.signature(
            SleepOrchestrator.run_full_sleep_cycle
        ).parameters:
            sleep_kwargs["accepted_writes"] = accepted_writes
        return run_full_sleep_cycle(
            self,
            responsibilities,
            **sleep_kwargs,
        )
