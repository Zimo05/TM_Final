"""Light and Deep Sleep structural-update operations."""

from __future__ import annotations

from Train.TrainingComponents import *  # noqa: F403
from Train.TrainingComponents import (
    _differentiable_merge_settings,
    _topology_prune_settings,
    _variational_prune_settings,
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
            batch = module.build_split_batch_from_memory_bank(
                self.tree.episodic_memory.get_bank(leaf_id),
                max_items=max_evidence,
            )
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
                gate_threshold=self.sleep_config.split_gate_threshold,
                m_min=self.sleep_config.split_min_mass,
                min_structural_strength=(
                    self.sleep_config.split_min_structural_strength
                ),
                min_effective_sample_size=(
                    self.sleep_config.split_min_effective_sample_size
                ),
                B_sleep=self.sleep_config.split_patience,
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
    def _continuous_split_demand(self) -> Dict[str, float]:
        """Convert cumulative queue mass into a decaying per-cycle signal."""
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
        return {
            "value": float(min(max(demand, 0.0), 1.0)),
            "observation": float(observation),
            "queue_increment": float(queue_increment),
            "queue_total": float(queue_total),
            "active_increment_count": float(active_increment_count),
        }

    @torch.no_grad()
    def _deep_sleep_features(
        self,
        low_mass: Sequence[str],
        *,
        residual_energy: float,
        epoch_label: int,
    ) -> Dict[str, Any]:
        """Build detached post-Light sufficient statistics for the gate."""
        del low_mass
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
        split_demand = self._continuous_split_demand()
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
        residual_ratio = (
            residual_energy
            / self.sleep_config.deep_residual_energy_budget
        )
        memory_ratio = effective_memory_count / max(memory_budget, 1e-8)
        topology_ratio = (
            current_complexity / max(target_complexity, 1e-8)
            if current_complexity > 0.0 else 0.0
        )
        elapsed = max(
            epoch_label
            - int(self.sleep_state.get("last_deep_epoch", 0)),
            0,
        )
        cooldown = 1.0 - math.exp(
            -float(elapsed) / self.sleep_config.deep_cooldown_tau
        )
        values = [
            math.log1p(max(residual_ratio, 0.0)),
            math.log1p(max(memory_ratio, 0.0)),
            math.log1p(max(topology_ratio, 0.0)),
            split_demand["value"],
            cooldown,
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
            "structural": float(split_demand["value"]),
            "cooldown": float(cooldown),
            "residual_ratio": float(residual_ratio),
            "memory_ratio": float(memory_ratio),
            "topology_ratio": float(topology_ratio),
            "residual_energy": float(residual_energy),
            "memory_count": float(memory_count),
            "effective_memory_count": float(effective_memory_count),
            "memory_budget": float(memory_budget),
            "split_candidates": split_demand["active_increment_count"],
            "split_queue_increment": split_demand["queue_increment"],
            "split_queue_total": split_demand["queue_total"],
            "split_observation": split_demand["observation"],
            "merge_candidates": 0.0,
            "topology_prune_candidates": float(prune_candidates),
            "topology_complexity": float(current_complexity),
            "topology_budget": float(target_complexity),
            "low_mass_candidates": 0.0,
        }

    @torch.no_grad()
    def _estimate_deep_gain(
        self,
        split_proposals,
        topology_prune_proposals,
        prune_result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Estimate J_light - J_virtual_deep from bounded shadow proposals."""
        component_gains: Dict[str, float] = {}
        for leaf_id, (_, output) in split_proposals.items():
            parent_log_likelihood = torch.as_tensor(output["logp0"])
            replay_weights = torch.as_tensor(
                output.get(
                    "replay_weights",
                    torch.ones_like(parent_log_likelihood),
                ),
                device=parent_log_likelihood.device,
                dtype=parent_log_likelihood.dtype,
            )
            parent_nll = -(
                replay_weights * parent_log_likelihood
            ).sum() / replay_weights.sum().clamp_min(1e-8)
            virtual_objective = torch.as_tensor(output["loss"])
            component_gains[f"split:{leaf_id}"] = float(
                (parent_nll - virtual_objective).detach().cpu()
            )
        for parent_id, proposal in topology_prune_proposals.items():
            if not proposal.eligible or not math.isfinite(
                proposal.retention_cost
            ):
                continue
            component_gains[f"topology_prune:{parent_id}"] = float(
                self.merge_lambda_T * proposal.complexity_saving
                - proposal.retention_cost
            )
        eligible_memories = float(
            prune_result.get("eligible_memories", 0.0)
        )
        if eligible_memories > 0.0:
            compression_saving = max(
                eligible_memories
                - float(prune_result.get("expected_keep", 0.0)),
                0.0,
            ) / eligible_memories
            component_gains["memory_prune"] = float(
                prune_result.get("reference_nll", 0.0)
                - prune_result.get("prediction_nll", 0.0)
                - prune_result.get("dynamics_kl", 0.0)
                - prune_result.get("prior_kl", 0.0)
                + self.memory_prune_lambda_M * compression_saving
            )
        gain = max(component_gains.values(), default=0.0)
        return {
            "value": float(gain),
            "components": component_gains,
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
        self.optimizer.zero_grad(set_to_none=True)
        objective["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            self.deep_sleep_gate.parameters(),
            self.sleep_config.deep_gate_grad_clip,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {
            key: float(value.detach().cpu())
            for key, value in objective.items()
        }

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

    def _evaluate_topology_prune(
        self,
        *,
        max_replay: int,
        allow_commit: bool,
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
            **evaluation_kwargs,
        )
        if update_persistence:
            proposals = apply_prune_persistence(
                self.tree,
                proposals,
                patience=settings["patience"],
                allow_commit=allow_commit,
                near_zero_confirmations=settings[
                    "near_zero_confirmations"
                ],
            )
        metrics["ready_count"] = float(sum(
            proposal.should_prune for proposal in proposals.values()
        ))
        metrics["budget_KT"] = budget_KT
        metrics["constraint"] = (
            metrics["expected_complexity"] - budget_KT
        )
        metrics["lambda_T"] = float(self.merge_lambda_T)
        return proposals, metrics

    def _optimize_differentiable_merge(
        self,
        *,
        max_replay: int,
    ) -> tuple[Dict[tuple[str, str], Dict[str, Any]], Dict[str, float]]:
        """Optimize Merge gates, update the dual, then MAP-decode decisions.

        Replay caches are gathered on the host, while all candidate pairs and
        their child/parent Hawkes likelihoods share one flattened tensor graph
        on the model device.
        """
        settings = _differentiable_merge_settings(
            self.structure_config.merge_kwargs
        )
        pairs = leaf_sibling_pairs(self.tree)
        objective = compute_differentiable_merge_objective(
            self.tree,
            pairs,
            self.hawkes.decays,
            lambda_T=self.merge_lambda_T,
            budget_KT=self.merge_budget_KT,
            gate_temperature=settings["gate_temperature"],
            min_replay=settings["min_replay"],
            normalize_by_events=settings["normalize_by_events"],
            hawkes_ll=self.hawkes,
            max_replay=max_replay,
        )
        if not objective["pairs"]:
            decisions = make_merge_decisions(
                objective,
                lambda_T=self.merge_lambda_T,
                gate_temperature=settings["gate_temperature"],
            )
            return decisions, {
                "loss": 0.0,
                "likelihood_loss": 0.0,
                "dual_term": 0.0,
                "expected_complexity": 0.0,
                "full_keep_complexity": 0.0,
                "budget_KT": float(
                    0.0 if self.merge_budget_KT is None
                    else self.merge_budget_KT
                ),
                "constraint": 0.0,
                "lambda_T": float(self.merge_lambda_T),
                "mean_keep_probability": 0.0,
                "eligible_pairs": 0.0,
                "merge_count": 0.0,
                "replay_windows": 0.0,
            }

        if self.merge_budget_KT is None:
            self.merge_budget_KT = float(
                settings["budget_ratio"]
                * objective["full_keep_complexity"].detach().item()
            )
            # Rebuild only the scalar dual term against the initialized
            # global budget; model-dependent NLL tensors stay shared.
            lambda_tensor = objective["loss"].new_tensor(
                self.merge_lambda_T
            )
            dual_term = lambda_tensor * (
                objective["expected_complexity"]
                - self.merge_budget_KT
            )
            objective["dual_term"] = dual_term
            objective["loss"] = objective["likelihood_loss"] + dual_term

        weighted_loss = settings["loss_weight"] * objective["loss"]
        self.optimizer.zero_grad(set_to_none=True)
        if settings["loss_weight"] > 0.0:
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._named_optimized_parameters().values(),
                self.training_config.grad_clip,
            )
            self.optimizer.step()

        # Re-evaluate once after the parameter update. The dual uses the
        # current policy's expected complexity; the new lambda is then used
        # only for detached MAP decoding and the next Deep cycle.
        with torch.no_grad():
            post = compute_differentiable_merge_objective(
                self.tree,
                pairs,
                self.hawkes.decays,
                lambda_T=self.merge_lambda_T,
                budget_KT=self.merge_budget_KT,
                gate_temperature=settings["gate_temperature"],
                min_replay=settings["min_replay"],
                normalize_by_events=settings["normalize_by_events"],
                hawkes_ll=self.hawkes,
                max_replay=max_replay,
            )
            expected_complexity = float(
                post["expected_complexity"].item()
            )
            constraint = expected_complexity - float(
                self.merge_budget_KT
            )
            self.merge_lambda_T = max(
                0.0,
                self.merge_lambda_T
                + settings["dual_lr"] * constraint,
            )
            decisions = make_merge_decisions(
                post,
                lambda_T=self.merge_lambda_T,
                gate_temperature=settings["gate_temperature"],
            )

        keep_probabilities = [
            float(decision["keep_probability"])
            for decision in decisions.values()
            if "keep_probability" in decision
        ]
        return decisions, {
            "loss": float(weighted_loss.detach().item()),
            "likelihood_loss": float(
                objective["likelihood_loss"].detach().item()
            ),
            "dual_term": float(objective["dual_term"].detach().item()),
            "expected_complexity": expected_complexity,
            "full_keep_complexity": float(
                post["full_keep_complexity"].item()
            ),
            "budget_KT": float(self.merge_budget_KT),
            "constraint": float(constraint),
            "lambda_T": float(self.merge_lambda_T),
            "mean_keep_probability": (
                sum(keep_probabilities) / len(keep_probabilities)
                if keep_probabilities else 0.0
            ),
            "eligible_pairs": float(len(post["pairs"])),
            "merge_count": float(sum(
                bool(decision.get("merge", False))
                for decision in decisions.values()
            )),
            "replay_windows": float(sum(post["replay_counts"])),
        }

    def _optimize_variational_prune(
        self,
        *,
        max_replay_per_node: int,
        update_dual: bool = True,
    ):
        """Optimize temporary memory gates and advance projected dual state."""
        settings = _variational_prune_settings(
            self.structure_config.memory_prune_kwargs
        )
        proposal, metrics = optimize_variational_memory_prune(
            self.tree,
            self.hawkes,
            lambda_M=self.memory_prune_lambda_M,
            budget_ratio=settings["budget_ratio"],
            num_steps=settings["num_steps"],
            learning_rate=settings["learning_rate"],
            temperature_start=settings["temperature_start"],
            temperature_end=settings["temperature_end"],
            dynamics_weight=settings["dynamics_weight"],
            prior_kl_weight=settings["prior_kl_weight"],
            null_logit=settings["null_logit"],
            min_replay=settings["min_replay"],
            min_memories_per_node=settings["min_memories_per_node"],
            max_replay_per_node=max_replay_per_node,
        )
        if update_dual and metrics["eligible_memories"] > 0.0:
            self.memory_prune_lambda_M = max(
                0.0,
                self.memory_prune_lambda_M
                + settings["dual_lr"] * metrics["constraint"],
            )
        metrics["lambda_M"] = float(self.memory_prune_lambda_M)
        return proposal, metrics

    def train_sleep(
        self,
        responsibilities: Tensor,
        *,
        allow_leaf_prune: bool = True,
        epoch: Optional[int] = None,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """Run Light Sleep, then route through the differentiable Deep gate."""
        self.tree.train()
        self.deep_sleep_gate.train()
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
                residual_energy=float(light_result["residual_energy"]),
                epoch_label=epoch_label,
            )
            gate_output = self.deep_sleep_gate(
                pressure["tensor"],
                split_demand=pressure["structural"],
                cooldown=pressure["cooldown"],
            )
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

            memory_prune_proposal = None
            split_proposals = {}
            topology_prune_proposals = {}
            deep_gain = {"value": 0.0, "components": {}}
            gate_objective = {
                "loss": 0.0,
                "reward_term": 0.0,
                "cost_term": 0.0,
                "prior_term": 0.0,
                "kl": 0.0,
                "estimated_gain": 0.0,
            }
            merge_decisions = {}
            merge_result = {
                "loss": 0.0,
                "likelihood_loss": 0.0,
                "dual_term": 0.0,
                "expected_complexity": 0.0,
                "full_keep_complexity": 0.0,
                "budget_KT": float(
                    0.0 if self.merge_budget_KT is None
                    else self.merge_budget_KT
                ),
                "constraint": 0.0,
                "lambda_T": float(self.merge_lambda_T),
                "mean_keep_probability": 0.0,
                "eligible_pairs": 0.0,
                "merge_count": 0.0,
                "replay_windows": 0.0,
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
            prune_result = {
                "loss": 0.0,
                "prediction_nll": 0.0,
                "reference_nll": 0.0,
                "dynamics_kl": 0.0,
                "prior_kl": 0.0,
                "expected_keep": 0.0,
                "budget_KM": 0.0,
                "constraint": 0.0,
                "lambda_M": float(self.memory_prune_lambda_M),
                "mean_keep_probability": 0.0,
                "eligible_memories": 0.0,
                "prune_count": 0.0,
                "committed_prune_count": 0.0,
                "skipped_nodes": 0.0,
                "replay_windows": 0.0,
                "mean_null_probability": 0.0,
            }

            if evaluation_needed:
                memory_prune_proposal, prune_result = (
                    self._optimize_variational_prune(
                        max_replay_per_node=(
                            self.sleep_config.deep_evidence_budget
                        ),
                        update_dual=deep_triggered,
                    )
                )
                topology_prune_proposals, topology_prune_result = (
                    self._evaluate_topology_prune(
                        max_replay=self.sleep_config.deep_evidence_budget,
                        allow_commit=(
                            deep_triggered and allow_leaf_prune
                        ),
                        update_persistence=deep_triggered,
                    )
                )
                # The former Merge objective was already a vertical
                # parent-vs-refinement test and duplicated Topology Prune.
                # Keep sibling pairs available for memory promotion, but do
                # not submit a second collapse proposal for the same region.
                merge_decisions = {
                    pair: {
                        "merge": False,
                        "reason": "superseded_by_topology_prune",
                    }
                    for pair in leaf_sibling_pairs(self.tree)
                }
                merge_result = {
                    "loss": 0.0,
                    "likelihood_loss": 0.0,
                    "dual_term": 0.0,
                    "expected_complexity": topology_prune_result[
                        "expected_complexity"
                    ],
                    "full_keep_complexity": topology_prune_result[
                        "current_complexity"
                    ],
                    "budget_KT": topology_prune_result["budget_KT"],
                    "constraint": topology_prune_result["constraint"],
                    "lambda_T": float(self.merge_lambda_T),
                    "mean_keep_probability": 1.0 - topology_prune_result[
                        "mean_prune_probability"
                    ],
                    "eligible_pairs": topology_prune_result["eligible_count"],
                    "merge_count": 0.0,
                    "replay_windows": topology_prune_result["replay_windows"],
                }
                split_proposals = self.build_split_proposals(
                    progress=(
                        sleep_progress if show_progress else None
                    ),
                    max_evidence=self.sleep_config.deep_evidence_budget,
                )
                if not show_progress:
                    sleep_progress.update(leaf_count)
                deep_gain = self._estimate_deep_gain(
                    split_proposals,
                    topology_prune_proposals,
                    prune_result,
                )
                gate_objective = self._optimize_deep_sleep_gate(
                    gate_output,
                    estimated_gain=deep_gain["value"],
                )
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
                    split_proposals=split_proposals,
                    merge_decisions=merge_decisions,
                    topology_prune_proposals=topology_prune_proposals,
                    optimizer=self.optimizer,
                    controllers=(self.controller,),
                    usage_decay=self.structure_config.usage_decay,
                    effective_usage_threshold=(
                        self.structure_config.effective_usage_threshold
                    ),
                    leaf_mass_ema_decay=(
                        self.structure_config.leaf_mass_ema_decay
                    ),
                    mass_threshold=self.structure_config.mass_threshold,
                    mass_patience=self.structure_config.mass_patience,
                    split_memory_hard_threshold=(
                        self.structure_config.split_memory_hard_threshold
                    ),
                    split_init_steps=self.sleep_config.split_init_steps,
                    split_init_lr=self.sleep_config.split_init_lr,
                    merge_memory_hard_threshold=(
                        self.structure_config.merge_memory_hard_threshold
                    ),
                    prune_memory_hard_threshold=(
                        self.structure_config.prune_memory_hard_threshold
                    ),
                    allow_leaf_prune=allow_leaf_prune,
                    allow_legacy_leaf_prune=False,
                    promotion_kwargs=promotion_kwargs,
                    memory_prune_proposal=memory_prune_proposal,
                    statistics_prepared=True,
                    precomputed_low_mass=low_mass,
                )
                topology_settings = _topology_prune_settings(
                    self.structure_config.topology_prune_kwargs
                )
                self.merge_lambda_T = max(
                    0.0,
                    self.merge_lambda_T
                    + topology_settings["dual_lr"]
                    * topology_prune_result["constraint"],
                )
                topology_prune_result["lambda_T"] = float(
                    self.merge_lambda_T
                )
                merge_result["lambda_T"] = float(self.merge_lambda_T)
                prune_result["committed_prune_count"] = float(sum(
                    int(item.get("pruned", 0))
                    for item in transaction["memory"].values()
                ))
                prune_result["skipped_nodes"] = float(sum(
                    item.get("reason") != "variational_topk_commit"
                    for item in transaction["memory"].values()
                ))
                self.sleep_state["last_deep_epoch"] = epoch_label
                self.deep_sleep_gate.reset_after_deep()
                mode = "deep"
            else:
                transaction = {
                    "leaf_snapshot": list(self.tree.leaf_ids),
                    "leaf_mass_ema": dict(self.tree.mass_ema),
                    "merge_decisions": {},
                    "actions": [],
                    # Stale-memory pruning scans every bank and therefore
                    # belongs to bounded, pressure-triggered Deep Sleep.
                    "memory": {},
                    "leaf_ids": list(self.tree.leaf_ids),
                    "leaf_prune_enabled": bool(allow_leaf_prune),
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
            "merge": merge_result,
            "topology_prune": topology_prune_result,
            "prune": prune_result,
            "transaction": transaction,
        }
