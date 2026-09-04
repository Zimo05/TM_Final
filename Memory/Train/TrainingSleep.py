"""Light and Deep Sleep structural-update operations."""

from __future__ import annotations

import inspect
from dataclasses import replace

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
                # Split modules are now materialized before Light so their
                # Bank probe can protect a leaf.  Keep that bookkeeping
                # reorder from consuming the global RNG stream used by the
                # stochastic Deep gate (and by the rest of training).
                with torch.random.fork_rng(devices=[]):
                    split_module = SplitModule(
                        P=self.tree.param_dim,
                        z_dim=self.tree.episodic_memory.key_dim,
                        m_min=0.0,
                        lambda_mass=0.0,
                        nll_fn=self.hawkes,
                    )
                self.split_modules[leaf_id] = split_module.to(self.device)
            # Migrate old checkpoints away from the retired mass threshold
            # and its soft surrogate as soon as the module becomes active.
            self.split_modules[leaf_id].m_min = 0.0
            self.split_modules[leaf_id].lambda_mass = 0.0

    @torch.no_grad()
    def _split_bank_diagnostics(
        self,
        leaf_id: str,
        bank: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Return the persistent evidence owned by one leaf bank.

        ``split_queues`` records controller decisions and is intentionally
        not part of the structural-evidence source of truth.  The bank keeps
        the accumulated mass aligned with prototype support, including after
        refresh and mode compression, so Split/Deep Sleep must read these
        values directly from here.
        """
        if bank is None:
            bank = self.tree.episodic_memory.get_bank(leaf_id)
        bank._ensure_prototype_state()
        count = len(bank)
        if count == 0:
            evidence_mass = 0.0
            persistent_count = 0.0
            law_mode_count = 0
            effective_mode_count = 0
        else:
            evidence_mass = float(
                bank.split_mass[:count].detach().clamp_min(0.0).sum().cpu()
            )
            # ``support`` is the number of persistent observations represented
            # by the bank's compact prototype rows, rather than the number of
            # controller decisions that happened to precede admission.
            persistent_count = float(
                bank.support[:count].detach().clamp_min(0.0).sum().cpu()
            )
            law_mode_count = int(
                torch.unique(bank.mode_ids[:count].detach()).numel()
            )
            # K_effective_mode is a structural Bank quantity.  Count modes
            # from persistent support rather than from the current bounded
            # replay mask; replay coverage is reported separately by Split's
            # h1_defined evaluability invariant.
            _, mode_inverse = torch.unique(
                bank.mode_ids[:count].detach().to(dtype=torch.long),
                sorted=True,
                return_inverse=True,
            )
            mode_support = bank.support[:count].new_zeros(
                int(mode_inverse.max().item()) + 1
            )
            mode_support.index_add_(0, mode_inverse, bank.support[:count])
            effective_mode_count = int(
                (mode_support > 1e-8).sum().item()
            )
        return {
            "Q_decision": float(
                max(float(self.controller.split_queues.get(leaf_id, 0.0)), 0.0)
            ),
            "E_bank_struct": max(evidence_mass, 0.0),
            "N_persistent": max(persistent_count, 0.0),
            "K_law_mode": float(max(law_mode_count, 0)),
            "K_effective_mode": float(max(effective_mode_count, 0)),
        }

    def build_split_proposals(
        self,
        *,
        progress: Optional[tqdm] = None,
        max_evidence: Optional[int] = None,
        bank_mode_probes: Optional[Mapping[str, Dict[str, Any]]] = None,
    ) -> Dict[str, tuple[SplitModule, Dict[str, Any]]]:
        self._sync_split_modules()
        proposals = {}
        # Legacy checkpoints may contain rejected/shadow probes here. They are
        # deliberately ignored: Sleep structural evidence starts at the
        # persistent EpisodicMemory boundary.
        self.sleep_state["structural_evidence_buffer"] = {}
        # Split search domain must always be all currently active leaves.
        # ``bank_mode_probes`` provides frozen Pre-Light hypotheses for
        # selected leaves; it must never restrict which leaves are evaluated.
        leaf_ids = list(self.tree.leaf_ids)
        for leaf_id in leaf_ids:
            if progress is not None:
                progress.set_postfix(
                    phase="split",
                    leaf=leaf_id,
                    refresh=True,
                )
            bank = self.tree.episodic_memory.get_bank(leaf_id)
            bank_diagnostics = self._split_bank_diagnostics(
                leaf_id,
                bank,
            )
            module = self.split_modules[leaf_id]
            probe = (
                None
                if bank_mode_probes is None
                else bank_mode_probes.get(leaf_id)
            )
            hypothesis = None
            if probe is None:
                # Keep checkpoints/config objects created before this option
                # was added runnable; the historical replay coverage default
                # is two samples per Bank-mode group.
                min_replay_per_group = getattr(
                    self.sleep_config,
                    "split_min_replay_per_group",
                    2,
                )
                batch = module.build_split_batch_from_memory_bank(
                    bank,
                    max_items=max_evidence,
                    min_replay_per_group=min_replay_per_group,
                )
                theta_snapshot = self.tree.semantic_theta(leaf_id)
            else:
                # Deep reuses the exact frozen hypothesis produced by
                # Pre-Light: rows, q_bank, delta_bar, H0, and Bank child
                # initialization are all the same object.  There is no
                # second batch normalization or child-law reconstruction.
                hypothesis = probe.get("hypothesis")
                if hypothesis is None:
                    # Compatibility for probes written before the frozen
                    # hypothesis object was introduced.
                    batch = replace(
                        probe["batch"],
                        weights=probe["replay_weights"],
                        effective_sample_size=torch.as_tensor(
                            probe["replay_ess"],
                            device=probe["replay_weights"].device,
                            dtype=probe["replay_weights"].dtype,
                        ),
                    )
                    theta_snapshot = probe["theta_sem_snapshot"]
                else:
                    batch = hypothesis.batch
                    theta_snapshot = hypothesis.theta_h0
            if batch is None:
                if progress is not None:
                    progress.update(1)
                continue
            output = module.optimize_leaf_split(
                theta_sem=theta_snapshot,
                batch=batch,
                hawkes_ll=self.hawkes,
                num_steps=self.sleep_config.split_steps,
                lr=self.sleep_config.split_lr,
                m_min=0.0,
                min_structural_strength=0.0,
                min_effective_sample_size=0.0,
                lambda_T=float(self.merge_lambda_T),
                # Splitting one leaf creates exactly one internal node.
                delta_complexity=1.0,
                # Router learning is a separate distillation objective; it
                # does not contribute to the Bank-based structural gain.
                lambda_anchor=getattr(
                    self.sleep_config,
                    "split_anchor_weight",
                    1e-2,
                ),
                lambda_route=getattr(
                    self.sleep_config,
                    "split_route_loss_weight",
                    1.0,
                ),
                hypothesis=hypothesis,
            )
            output["replay_weights"] = (
                hypothesis.replay_weights.detach()
                if hypothesis is not None
                else batch.weights.detach()
            )
            if probe is not None:
                output["replay_weights"] = probe["replay_weights"].detach()
                output["probe_advantage"] = float(probe["advantage"])
                output["probe_loss_h0"] = float(probe["loss_h0"])
                output["probe_loss_h1"] = float(probe["loss_h1"])
                output["probe_mode_ids"] = (
                    hypothesis.mode_ids.detach()
                    if hypothesis is not None
                    else probe["mode_ids"].detach()
                )
                output["probe_mode_group_ids"] = (
                    hypothesis.mode_group_ids.detach()
                    if hypothesis is not None
                    else probe["mode_group_ids"].detach()
                )
            # Preserve evidence-size provenance for the diagnostic logger.
            # The current production path has no shadow buffer, so make that
            # explicit instead of conflating it with bounded replay rows.
            output["N_bank"] = int(len(bank))
            output["N_shadow"] = 0
            output["N_replay"] = int(batch.residuals.shape[0])
            output.update(bank_diagnostics)
            proposals[leaf_id] = (module, output)
            if progress is not None:
                progress.update(1)
        return proposals

    @torch.no_grad()
    def _probe_bank_modes_for_split(
        self,
        leaf_id: str,
        bank: Any,
        *,
        max_evidence: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build and run the independent pre-Light structural probe.

        The probe is intentionally sourced from the persistent Bank rather
        than from a Light proposal.  Its result can therefore decide whether
        Light, Merge, Prune, and Promotion must be vetoed for this leaf before
        any fixed-topology consolidation is attempted.
        """
        self._sync_split_modules()
        bank._ensure_prototype_state()
        count = len(bank)
        if count == 0 or int(
            torch.unique(bank.mode_ids[:count].detach()).numel()
        ) < 2:
            return None
        module = self.split_modules[leaf_id]
        min_replay_per_group = getattr(
            self.sleep_config,
            "split_min_replay_per_group",
            2,
        )
        batch = module.build_split_batch_from_memory_bank(
            bank,
            max_items=max_evidence,
            min_replay_per_group=min_replay_per_group,
        )
        if batch is None:
            return None
        probe = module.bank_structural_split_probe(
            theta_sem=self.tree.semantic_theta(leaf_id),
            batch=batch,
            hawkes_ll=self.hawkes,
        )
        if probe is None:
            return None
        probe["leaf_id"] = leaf_id
        return probe

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
        """Estimate structural pressure from persistent Bank ``split_mass``."""
        if accepted_writes < 0:
            raise ValueError("accepted_writes must be non-negative")
        previous_queue_snapshot = dict(
            self.sleep_state.get("split_queue_snapshot", {})
        )
        previous_mass_snapshot = dict(
            self.sleep_state.get("split_mass_snapshot", {})
        )
        next_queue_snapshot: Dict[str, float] = {}
        next_mass_snapshot: Dict[str, float] = {}
        queue_increment = 0.0
        queue_total = 0.0
        structural_mass_total = 0.0
        structural_mass_increment = 0.0
        active_increment_count = 0
        persistent_count_total = 0.0
        law_mode_count_total = 0
        effective_mode_count_total = 0
        for leaf_id in self.tree.leaf_ids:
            bank = self.tree.episodic_memory.get_bank(leaf_id)
            bank_diagnostics = self._split_bank_diagnostics(
                leaf_id,
                bank,
            )
            current_queue = bank_diagnostics["Q_decision"]
            current_mass = bank_diagnostics["E_bank_struct"]
            previous_queue = max(
                float(previous_queue_snapshot.get(leaf_id, 0.0)),
                0.0,
            )
            previous_mass = max(
                float(previous_mass_snapshot.get(leaf_id, 0.0)),
                0.0,
            )
            queue_increment += max(current_queue - previous_queue, 0.0)
            queue_total += current_queue
            structural_mass_total += current_mass
            mass_increment = max(current_mass - previous_mass, 0.0)
            structural_mass_increment += mass_increment
            active_increment_count += int(mass_increment > 0.0)
            persistent_count_total += bank_diagnostics["N_persistent"]
            law_mode_count_total += int(bank_diagnostics["K_law_mode"])
            effective_mode_count_total += int(
                bank_diagnostics["K_effective_mode"]
            )
            next_queue_snapshot[leaf_id] = current_queue
            next_mass_snapshot[leaf_id] = current_mass
        # Legacy raw counters are retained only as discarded diagnostics. They
        # cannot drive topology before evidence is present in a persistent bank.
        discarded_structural_mass = max(float(
            self.sleep_state.get("structural_mass_since_sleep", 0.0)
        ), 0.0)
        discarded_structural_observations = max(int(
            self.sleep_state.get(
                "structural_observations_since_sleep", 0
            )
        ), 0)
        # ``deep_split_queue_scale`` is a checkpoint-compatible name; the
        # quantity being scaled is now newly observed persistent Bank
        # evidence, not Q_decision.  The current total remains available as
        # E_bank_struct for eligibility and diagnostics.
        observation = 1.0 - math.exp(
            -structural_mass_increment
            / self.sleep_config.deep_split_queue_scale
        )
        previous_ema = float(
            self.sleep_state.get("structural_demand_ema", 0.0)
        )
        decay = self.sleep_config.deep_split_demand_decay
        demand = decay * previous_ema + (1.0 - decay) * observation
        # Keep Q_decision history for diagnostics/backward-compatible
        # checkpoints, but use the separate Bank snapshot for structural
        # evidence bookkeeping.
        self.sleep_state["split_queue_snapshot"] = next_queue_snapshot
        self.sleep_state["split_mass_snapshot"] = next_mass_snapshot
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
            "structural_mass": float(structural_mass_total),
            "structural_mass_increment": float(structural_mass_increment),
            "structural_observations": float(persistent_count_total),
            "persistent_count": float(persistent_count_total),
            "law_mode_count": float(law_mode_count_total),
            "Q_decision": float(queue_total),
            "E_bank_struct": float(structural_mass_total),
            "N_persistent": float(persistent_count_total),
            "K_law_mode": float(law_mode_count_total),
            "K_effective_mode": float(effective_mode_count_total),
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
            "Q_decision": split_demand["Q_decision"],
            "E_bank_struct": split_demand["E_bank_struct"],
            "N_persistent": split_demand["N_persistent"],
            "K_law_mode": split_demand["K_law_mode"],
            "K_effective_mode": split_demand["K_effective_mode"],
            "split_mass_increment": split_demand[
                "structural_mass_increment"
            ],
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
        *,
        topology_log_lines: Optional[list[str]] = None,
        protected_leaf_ids: Sequence[str] = (),
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
                # Retained by the public builder signature for checkpoint/API
                # compatibility; production Split no longer gates on them.
                min_child_effective_mass=0.0,
                min_structural_strength=0.0,
                min_effective_sample_size=0.0,
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
        protected = set(protected_leaf_ids)
        if protected:
            candidates = [
                candidate
                for candidate in candidates
                if (
                    candidate.kind.value == "split"
                    or candidate.claims.isdisjoint(protected)
                )
            ]
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
        # Mark Split candidates only after arbitration so the persisted line
        # distinguishes a physical acceptance from a positive-but-deferred
        # candidate. Router diagnostics remain visible beside the structural
        # gain but do not affect this status.
        selected_action_id = (
            None
            if selection.is_null or selection.selected is None
            else selection.selected.action_id
        )
        annotated_candidates = []
        for candidate in candidates:
            if candidate.kind.value != "split":
                annotated_candidates.append(candidate)
                continue
            if not candidate.eligible or not candidate.ready:
                status = "rejected"
            elif candidate.action_id == selected_action_id:
                status = "accepted"
            elif float(torch.as_tensor(
                candidate.conservative_gain
            ).detach().cpu()) > 0.0:
                status = "deferred"
            else:
                status = "rejected"
            diagnostics = dict(candidate.diagnostics)
            diagnostics["split_status"] = status
            annotated_candidates.append(replace(candidate, diagnostics=diagnostics))
        candidates = tuple(annotated_candidates)
        selected_candidate = None
        if selection.selected is not None:
            selected_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.action_id == selection.selected.action_id
                ),
                selection.selected,
            )
        selection = replace(
            selection,
            selected=selected_candidate,
            candidates=candidates,
        )
        # This method is called only when evaluation_needed=True. Capture the
        # final Split evidence after smoothing/inertia so it matches the
        # candidate passed to the unified selector. The training loop writes
        # these lines under an epoch header; nothing is printed here.
        if topology_log_lines is not None:
            topology_log_lines.extend(split_candidate_log_lines(candidates))
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

            # A rejected preservation probe gets one ordinary Light cycle.
            # Structural evidence is otherwise tested before Light creates a
            # proposal, so the result can veto Light and every later
            # topology/memory reconciliation stage for the same leaf.
            released_once = set(
                self.sleep_state.pop(
                    "bank_mode_probe_release_once", ()
                )
            )
            self._sync_split_modules()

            all_structural_probes: Dict[str, Dict[str, Any]] = {}
            for leaf_id in self.tree.leaf_ids:
                if leaf_id in released_once:
                    continue
                bank = self.tree.episodic_memory.get_bank(leaf_id)
                probe = self._probe_bank_modes_for_split(
                    leaf_id,
                    bank,
                    max_evidence=self.sleep_config.deep_evidence_budget,
                )
                if probe is not None:
                    all_structural_probes[leaf_id] = probe
            bank_mode_probes = all_structural_probes
            protected_probes = {
                leaf_id: probe
                for leaf_id, probe in bank_mode_probes.items()
                if bool(probe.get("protect", False))
            }

            light_result = run_light_sleep(
                self.tree,
                self.hawkes,
                optimizer=self.optimizer,
                settings=self._light_sleep_settings(),
                state=self.sleep_state.setdefault("light_index", {}),
                protected_leaf_ids=protected_probes.keys(),
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
            evaluation_needed = bool(
                deep_triggered or probe_due or protected_probes
            )
            deep_execution_requested = bool(
                deep_triggered or protected_probes
            )
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
            topology_log_lines: list[str] = []
            evaluated_split_nodes: tuple[str, ...] = ()
            split_eval_leaf_count = 0
            split_proposal_count = 0
            split_protected_count = 0
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
                # Every active leaf enters Split evaluation.  A leaf may still
                # produce no proposal when its current Bank has insufficient
                # replay evidence, which is intentionally tracked separately.
                evaluated_split_nodes = tuple(self.tree.leaf_ids)
                split_eval_leaf_count = len(evaluated_split_nodes)
                split_protected_count = len(protected_probes)
                split_proposals = self.build_split_proposals(
                    progress=(
                        sleep_progress if show_progress else None
                    ),
                    max_evidence=self.sleep_config.deep_evidence_budget,
                    # Every valid Pre-Light hypothesis is reused by Deep;
                    # ``protected_probes`` remains only the Light veto set.
                    bank_mode_probes=bank_mode_probes,
                )
                split_proposal_count = len(split_proposals)
                topology_log_lines.append(
                    "split_eval={} split_proposal={} split_protected={}".format(
                        split_eval_leaf_count,
                        split_proposal_count,
                        split_protected_count,
                    )
                )
                if not show_progress:
                    sleep_progress.update(leaf_count)
                unified_candidates, selection = (
                    self._select_unified_topology_action(
                        split_proposals,
                        merge_objective,
                        topology_prune_proposals,
                        topology_log_lines=topology_log_lines,
                        protected_leaf_ids=protected_probes,
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

            # A protected probe has three possible outcomes after the final
            # unified-selector arbitration:
            #
            #   accepted  -> this node's Split is selected;
            #   deferred  -> its final G_cons is positive, but another action
            #                wins the one-action physical commit;
            #   rejected  -> its final G_cons is non-positive.
            #
            # ``deferred`` is still protected on the next cycle.  In
            # particular, selector arbitration must not be mistaken for a
            # negative hypothesis test and release confirmed Bank evidence.
            selected_protected_split = set()
            if (
                selection is not None
                and not selection.is_null
                and selection.selected is not None
                and selection.selected.kind.value == "split"
                and selection.selected.target in protected_probes
            ):
                selected_protected_split.add(selection.selected.target)

            split_candidates_by_target = {
                candidate.target: candidate
                for candidate in unified_candidates
                if (
                    candidate.kind.value == "split"
                    and candidate.target in protected_probes
                )
            }
            accepted_probes = set(selected_protected_split)
            deferred_probes = set()
            rejected_probes = set()
            unresolved_probes = set()
            probe_states = {}
            for leaf_id in protected_probes:
                candidate = split_candidates_by_target.get(leaf_id)
                if leaf_id in accepted_probes:
                    probe_states[leaf_id] = "accepted"
                    continue
                if candidate is None:
                    # No candidate means that this cycle did not produce a
                    # selector-visible G_cons.  Preserve the node until a
                    # later cycle supplies an explicit non-positive result.
                    unresolved_probes.add(leaf_id)
                    deferred_probes.add(leaf_id)
                    probe_states[leaf_id] = "deferred"
                    continue
                conservative_gain = torch.as_tensor(
                    candidate.conservative_gain
                ).detach().reshape(())
                if bool(conservative_gain.le(0.0).cpu()):
                    rejected_probes.add(leaf_id)
                    probe_states[leaf_id] = "rejected"
                else:
                    # Positive and non-finite gains are not release evidence;
                    # keep the Bank evidence protected until arbitration or a
                    # later explicit rejection resolves it.
                    deferred_probes.add(leaf_id)
                    probe_states[leaf_id] = "deferred"
            if rejected_probes:
                # Only an explicit G_cons <= 0 rejection gets exactly one
                # unprotected Light cycle before Bank evidence is probed
                # again.  Positive-but-unselected probes remain protected.
                self.sleep_state["bank_mode_probe_release_once"] = sorted(
                    rejected_probes
                )

            if deep_execution_requested:
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
                    # Keep Promotion implementation available for isolated
                    # callers, but disable it in the current Bank-backed
                    # training path so it cannot move Split evidence.
                    promote_when_structure_unchanged=False,
                    statistics_prepared=True,
                    protected_leaf_ids=protected_probes,
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
                "execution_requested": bool(deep_execution_requested),
                "probe": bool(probe_due and not deep_triggered),
                "evaluated": bool(evaluation_needed),
                "gain": deep_gain,
                "objective": gate_objective,
            },
            "unified_topology": selection_log,
            "evaluated_split_nodes": list(evaluated_split_nodes),
            "split_evaluation": {
                "split_eval": int(split_eval_leaf_count),
                "split_proposal": int(split_proposal_count),
                "split_protected": int(split_protected_count),
            },
            "bank_mode_probe": {
                "evaluated_nodes": len(bank_mode_probes),
                "protected_nodes": sorted(protected_probes),
                "accepted_nodes": sorted(accepted_probes),
                "deferred_nodes": sorted(deferred_probes),
                "rejected_nodes": sorted(rejected_probes),
                "unresolved_nodes": sorted(unresolved_probes),
                "released_once": sorted(rejected_probes),
                "nodes": {
                    leaf_id: {
                        **{
                            key: value
                            for key, value in probe.items()
                            if key not in {
                                "hypothesis",
                                "batch",
                                "theta_sem_snapshot",
                                "bank_group_weights",
                                "q_bank",
                                "shared_delta",
                                "group_delta",
                                "theta_h0",
                                "theta_h1",
                                "replay_weights",
                                "mode_ids",
                                "mode_group_ids",
                                "selected_direction_indices",
                                "light_replay_indices",
                                "light_replay_weights",
                            }
                        },
                        **({"state": probe_states[leaf_id]}
                           if leaf_id in probe_states else {}),
                    }
                    for leaf_id, probe in bank_mode_probes.items()
                },
            },
            "unified_topology_log_lines": topology_log_lines,
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
