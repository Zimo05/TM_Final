"""Trainer construction, optimizer lifecycle, and checkpoint restoration."""

from __future__ import annotations

from Train.TrainingComponents import *  # noqa: F403
from Train.TrainingComponents import (
    _differentiable_merge_settings,
    _frontier_config_from_checkpoint,
    _topology_prune_settings,
)


class TrainingLifecycleMixin:
    def __init__(
        self,
        tree: HawkesTree,
        hawkes: HawkesFamily,
        encoder: nn.Module,
        *,
        wake: Optional[WakeObjectiveConfig] = None,
        sleep: Optional[SleepConfig] = None,
        structure: Optional[StructureConfig] = None,
        training: Optional[TrainingConfig] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.wake_config = WakeObjectiveConfig() if wake is None else wake
        self.sleep_config = SleepConfig() if sleep is None else sleep
        self.structure_config = StructureConfig() if structure is None else structure
        self.training_config = TrainingConfig() if training is None else training
        legacy_cooldown_tau = self.sleep_config.deep_cooldown_tau
        if legacy_cooldown_tau is not None:
            self.sleep_config.deep_availability_tau = float(
                legacy_cooldown_tau
            )
            self.sleep_config.deep_cooldown_tau = None
        if device is None:
            device = next(tree.parameters()).device
        self.device = torch.device(device)
        self.tree = tree.to(self.device)
        self.hawkes = hawkes.to(self.device)
        self.encoder = encoder.to(self.device)
        self.tree.episodic_memory.configure_prototype_memory(
            duplicate_threshold=self.wake_config.prototype_duplicate_threshold,
            mode_threshold=self.wake_config.prototype_mode_threshold,
            duplicate_quantile=self.wake_config.prototype_duplicate_quantile,
            mode_capacity=self.wake_config.prototype_mode_capacity,
            context_alias_capacity=(
                self.wake_config.prototype_context_alias_capacity
            ),
        )
        self.tree.episodic_memory.rebuild_law_keys(
            self.tree.semantic_theta,
            self.hawkes.decays,
        )

        if self.tree.hyper.D != self.hawkes.num_types:
            raise ValueError("tree and Hawkes backbone event dimensions differ")
        if self.tree.hyper.M != self.hawkes.num_basis:
            raise ValueError("tree and Hawkes backbone basis dimensions differ")
        if getattr(self.encoder, "z_dim", self.tree.z_dim) != self.tree.z_dim:
            raise ValueError("encoder output dimension must match tree.z_dim")
        if self.wake_config.lambda_route_mi < 0.0:
            raise ValueError("lambda_route_mi must be non-negative")
        if self.wake_config.lambda_route_posterior < 0.0:
            raise ValueError(
                "lambda_route_posterior must be non-negative"
            )
        if self.wake_config.lambda_route_distill < 0.0:
            raise ValueError(
                "lambda_route_distill must be non-negative"
            )
        if self.wake_config.lambda_route_balance < 0.0:
            raise ValueError("lambda_route_balance must be non-negative")
        if self.wake_config.lambda_route_mix < 0.0:
            raise ValueError("lambda_route_mix must be non-negative")
        if self.wake_config.route_energy_temperature <= 0.0:
            raise ValueError("route_energy_temperature must be positive")
        if self.wake_config.route_encoder_warmup_epochs < 0:
            raise ValueError(
                "route_encoder_warmup_epochs must be non-negative"
            )
        if not 0.0 <= self.wake_config.route_encoder_grad_scale <= 1.0:
            raise ValueError(
                "route_encoder_grad_scale must lie in [0, 1]"
            )
        if not 0.0 <= self.wake_config.route_encoder_reliability_decay < 1.0:
            raise ValueError(
                "route_encoder_reliability_decay must lie in [0, 1)"
            )
        if self.wake_config.route_teacher_temperature <= 0.0:
            raise ValueError("route_teacher_temperature must be positive")
        if self.wake_config.lambda_route_probe < 0.0:
            raise ValueError("lambda_route_probe must be non-negative")
        if not 0.0 <= self.wake_config.route_probe_leaf_smoothing < 1.0:
            raise ValueError(
                "route_probe_leaf_smoothing must lie in [0, 1)"
            )
        if min(
            self.wake_config.route_probe_router_weight,
            self.wake_config.route_probe_expand_weight,
            self.wake_config.route_probe_leaf_weight,
        ) < 0.0:
            raise ValueError("route probe objective weights must be non-negative")
        if self.wake_config.route_probe_residual_temperature <= 0.0:
            raise ValueError(
                "route_probe_residual_temperature must be positive"
            )
        if self.wake_config.route_probe_gain_temperature <= 0.0:
            raise ValueError("route_probe_gain_temperature must be positive")
        if self.wake_config.route_probe_complexity_weight < 0.0:
            raise ValueError(
                "route_probe_complexity_weight must be non-negative"
            )
        if self.wake_config.route_probe_residual_rank < 0:
            raise ValueError("route_probe_residual_rank must be non-negative")
        if self.wake_config.route_probe_residual_grad_clip < 0.0:
            raise ValueError(
                "route_probe_residual_grad_clip must be non-negative"
            )
        if self.wake_config.route_balance_batch_size < 2:
            raise ValueError("route_balance_batch_size must be at least 2")
        if self.wake_config.wake_wavefront_batch_size <= 0:
            raise ValueError("wake_wavefront_batch_size must be positive")
        if self.wake_config.retrieval_microbatch <= 0:
            raise ValueError("retrieval_microbatch must be positive")
        if self.wake_config.route_balance_max_steps <= 0:
            raise ValueError("route_balance_max_steps must be positive")
        if self.wake_config.route_balance_target_kl < 0.0:
            raise ValueError("route_balance_target_kl must be non-negative")
        if self.wake_config.memory_write_grad_clip <= 0.0:
            raise ValueError("memory_write_grad_clip must be positive")
        if self.wake_config.action_temperature <= 0.0:
            raise ValueError("action_temperature must be positive")
        if self.wake_config.novelty_temperature <= 0.0:
            raise ValueError("novelty_temperature must be positive")
        if self.wake_config.count_exponent < 1.0:
            raise ValueError("count_exponent must be at least 1")
        if not (
            -1.0
            <= self.wake_config.count_similarity_low
            < self.wake_config.count_similarity_high
            <= 1.0
        ):
            raise ValueError(
                "count similarity thresholds must satisfy "
                "-1 <= low < high <= 1"
            )
        if (
            self.wake_config.count_topk is not None
            and (
                isinstance(self.wake_config.count_topk, bool)
                or not isinstance(self.wake_config.count_topk, int)
                or self.wake_config.count_topk <= 0
            )
        ):
            raise ValueError("count_topk must be a positive integer or None")
        if self.wake_config.count_saturation <= 0.0:
            raise ValueError("count_saturation must be positive")
        if not 0.0 <= self.wake_config.surprise_ema_decay < 1.0:
            raise ValueError("surprise_ema_decay must lie in [0, 1)")
        if self.training_config.router_lr_scale <= 0.0:
            raise ValueError("router_lr_scale must be positive")
        if self.structure_config.prune_warmup_epochs < 0:
            raise ValueError("prune_warmup_epochs must be non-negative")
        if self.sleep_config.light_replay_budget <= 0:
            raise ValueError("light_replay_budget must be positive")
        if self.sleep_config.light_scan_budget_multiplier <= 0:
            raise ValueError(
                "light_scan_budget_multiplier must be positive"
            )
        if self.sleep_config.deep_residual_energy_budget <= 0.0:
            raise ValueError(
                "deep_residual_energy_budget must be positive"
            )
        if self.sleep_config.deep_memory_budget_multiplier <= 0.0:
            raise ValueError(
                "deep_memory_budget_multiplier must be positive"
            )
        if self.sleep_config.deep_evidence_budget <= 0:
            raise ValueError("deep_evidence_budget must be positive")
        # Older SleepConfig objects may not carry this newer replay-coverage
        # option.  Preserve their historical default during validation.
        split_min_replay_per_group = getattr(
            self.sleep_config,
            "split_min_replay_per_group",
            2,
        )
        if split_min_replay_per_group < 0:
            raise ValueError(
                "split_min_replay_per_group must be non-negative"
            )
        if self.sleep_config.deep_availability_tau <= 0.0:
            raise ValueError("deep_availability_tau must be positive")
        if not 0.0 <= self.sleep_config.deep_accumulator_decay < 1.0:
            raise ValueError(
                "deep_accumulator_decay must lie in [0, 1)"
            )
        if not 0.0 <= self.sleep_config.deep_split_demand_decay < 1.0:
            raise ValueError(
                "deep_split_demand_decay must lie in [0, 1)"
            )
        if self.sleep_config.deep_split_queue_scale <= 0.0:
            raise ValueError("deep_split_queue_scale must be positive")
        if self.sleep_config.deep_evidence_temperature <= 0.0:
            raise ValueError("deep_evidence_temperature must be positive")
        if self.sleep_config.deep_hard_concrete_temperature <= 0.0:
            raise ValueError(
                "deep_hard_concrete_temperature must be positive"
            )
        if self.sleep_config.deep_hard_concrete_gamma >= 0.0:
            raise ValueError("deep_hard_concrete_gamma must be negative")
        if self.sleep_config.deep_hard_concrete_zeta <= 1.0:
            raise ValueError("deep_hard_concrete_zeta must exceed one")
        if not 0.0 < self.sleep_config.deep_execution_threshold < 1.0:
            raise ValueError(
                "deep_execution_threshold must lie in (0, 1)"
            )
        if self.sleep_config.deep_gate_weight_initial <= 0.0:
            raise ValueError("deep_gate_weight_initial must be positive")
        if self.sleep_config.deep_probe_interval <= 0:
            raise ValueError("deep_probe_interval must be positive")
        if self.sleep_config.deep_computation_cost < 0.0:
            raise ValueError("deep_computation_cost must be non-negative")
        if not 0.0 < self.sleep_config.deep_prior_probability < 1.0:
            raise ValueError(
                "deep_prior_probability must lie in (0, 1)"
            )
        if self.sleep_config.deep_prior_weight < 0.0:
            raise ValueError("deep_prior_weight must be non-negative")
        if self.sleep_config.deep_gate_grad_clip <= 0.0:
            raise ValueError("deep_gate_grad_clip must be positive")
        if self.sleep_config.deep_gate_learning_rate <= 0.0:
            raise ValueError("deep_gate_learning_rate must be positive")
        if self.sleep_config.action_temperature <= 0.0:
            raise ValueError("action_temperature must be positive")
        if not 0.0 <= self.sleep_config.action_gain_ema_decay < 1.0:
            raise ValueError("action_gain_ema_decay must lie in [0, 1)")
        if self.sleep_config.action_uncertainty_kappa < 0.0:
            raise ValueError("action_uncertainty_kappa must be non-negative")
        if self.sleep_config.action_min_observations <= 0:
            raise ValueError("action_min_observations must be positive")
        if self.sleep_config.action_selector_learning_rate <= 0.0:
            raise ValueError(
                "action_selector_learning_rate must be positive"
            )
        if self.sleep_config.action_selector_entropy_weight < 0.0:
            raise ValueError(
                "action_selector_entropy_weight must be non-negative"
            )
        if self.sleep_config.action_selector_grad_clip <= 0.0:
            raise ValueError("action_selector_grad_clip must be positive")
        if self.sleep_config.topology_inertia_strength < 0.0:
            raise ValueError("topology_inertia_strength must be non-negative")
        if self.sleep_config.topology_inertia_tau <= 0.0:
            raise ValueError("topology_inertia_tau must be positive")
        merge_settings = _differentiable_merge_settings(
            self.structure_config.merge_kwargs
        )
        topology_prune_settings = _topology_prune_settings(
            self.structure_config.topology_prune_kwargs
        )
        self.structure_config.topology_prune_kwargs = dict(
            topology_prune_settings
        )

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
            exploration_rate=self.wake_config.controller_exploration_rate,
            utility_topc_multiplier=(
                self.wake_config.controller_utility_topc_multiplier
            ),
            utility_stage_enabled=(
                self.wake_config.controller_utility_stage_enabled
            ),
            utility_temperature=self.wake_config.controller_utility_temperature,
            utility_cost_margin=self.wake_config.controller_utility_cost_margin,
        ).to(self.device)
        self.deep_sleep_gate = DeepSleepGate(
            accumulator_decay=self.sleep_config.deep_accumulator_decay,
            evidence_temperature=(
                self.sleep_config.deep_evidence_temperature
            ),
            hard_concrete_temperature=(
                self.sleep_config.deep_hard_concrete_temperature
            ),
            hard_concrete_gamma=(
                self.sleep_config.deep_hard_concrete_gamma
            ),
            hard_concrete_zeta=(
                self.sleep_config.deep_hard_concrete_zeta
            ),
            execution_threshold=(
                self.sleep_config.deep_execution_threshold
            ),
            bias_initial=self.sleep_config.deep_gate_bias_initial,
            weight_initial=self.sleep_config.deep_gate_weight_initial,
        ).to(self.device)
        self.deep_gate_optimizer = torch.optim.Adam(
            self.deep_sleep_gate.parameters(),
            lr=self.sleep_config.deep_gate_learning_rate,
        )
        self.topology_selector = UnifiedTopologySelector(
            temperature=self.sleep_config.action_temperature,
        ).to(self.device)
        self.topology_selector_optimizer = torch.optim.Adam(
            self.topology_selector.parameters(),
            lr=self.sleep_config.action_selector_learning_rate,
        )
        named_parameters = self._named_optimized_parameters()
        router_parameters = [
            parameter
            for name, parameter in named_parameters.items()
            if self._is_router_parameter(name)
        ]
        router_ids = {id(parameter) for parameter in router_parameters}
        base_parameters = [
            parameter
            for parameter in named_parameters.values()
            if id(parameter) not in router_ids
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": base_parameters,
                    "lr": self.training_config.learning_rate,
                    "group_name": "base",
                },
                {
                    "params": router_parameters,
                    "lr": (
                        self.training_config.learning_rate
                        * self.training_config.router_lr_scale
                    ),
                    "group_name": "router",
                },
            ],
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        self.split_modules: Dict[str, SplitModule] = {}
        self.controller_utility_replay = ControllerUtilityReplay(
            self.wake_config.controller_replay_capacities,
            seed=self.training_config.seed,
        )
        self.controller_utility_replay.write_ranking_enabled = bool(
            self.training_config.controller_write_ranking
        )
        self.sleep_state: Dict[str, Any] = {
            "last_light_epoch": 0,
            "last_deep_epoch": 0,
            "deep_cycle_count": 0,
            "structural_demand_ema": 0.0,
            # Persistent Bank structural evidence has its own snapshot.  The
            # legacy split_queue_snapshot remains a controller diagnostic.
            "split_mass_snapshot": {},
            "split_queue_snapshot": {},
            "accepted_writes_since_sleep": 0,
            "structural_mass_since_sleep": 0.0,
            "structural_observations_since_sleep": 0,
            # Deprecated checkpoint field. Non-persistent probes are never
            # consumed by Sleep/Split and this mapping remains empty.
            "structural_evidence_buffer": {},
            "last_memory_count": sum(
                len(bank)
                for bank in self.tree.episodic_memory.banks.values()
            ),
            "light_index": {},
            "topology_revision": 0,
            "unified_action_evidence": {},
            "topology_last_edit_cycle": {},
        }
        # lambda_T is a projected dual variable, not an Adam parameter.
        self.merge_lambda_T = float(
            topology_prune_settings.get(
                "dual_initial", merge_settings["dual_initial"]
            )
        )
        self.merge_budget_KT: Optional[float] = None
        self.history: list[Dict[str, Any]] = []
        self.completed_epochs = 0
        self.validation_history: list[Dict[str, Any]] = []
        self.best_validation: Optional[Dict[str, Any]] = None
        self.controller_calibration: Dict[str, Any] = {}
        self.encoder_routing_reliability = 0.0
        self.last_teacher_confidence = 0.0
        self.last_teacher_student_js = math.log(2.0)
        self.last_teacher_student_alignment = 0.0
        self._resident_cache_hits = 0
        self._resident_cache_misses = 0
        torch.manual_seed(self.training_config.seed)

    def _named_optimized_parameters(self) -> Dict[str, nn.Parameter]:
        named = {
            f"tree.{name}": parameter
            for name, parameter in self.tree.named_parameters()
        }
        named.update({
            f"encoder.{name}": parameter
            for name, parameter in self.encoder.named_parameters()
        })
        named.update({
            f"controller.{name}": parameter
            for name, parameter in self.controller.named_parameters()
        })
        return named

    @staticmethod
    def _is_router_parameter(name: str) -> bool:
        return name.startswith((
            "tree.router_compat.",
            "tree.expansion_predictor.",
        ))

    @contextmanager
    def _freeze_global_parameters(self):
        """Temporarily stop gradients into every persistent model parameter.

        The working-memory delta is not an ``nn.Parameter`` and is created
        after entering this context, so event NLL remains differentiable with
        respect to that fast sequence-local state only.
        """
        parameters = list(dict.fromkeys(
            self._named_optimized_parameters().values()
        ))
        original = [parameter.requires_grad for parameter in parameters]
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            yield
        finally:
            for parameter, requires_grad in zip(parameters, original):
                parameter.requires_grad_(requires_grad)

    def _reconcile_optimizer_parameters(self) -> None:
        """Rebuild base/router groups after dynamic topology changes.

        Structural commits may temporarily register mixed parameter groups.
        Repartitioning by stable names keeps the shared frontier compatibility
        scorer on ``router_lr_scale`` and new node parameters on the base rate.
        """
        named = self._named_optimized_parameters()
        if self.training_config.controller_only_finetune:
            named = {
                name: parameter
                for name, parameter in named.items()
                if name.startswith("controller.") and parameter.requires_grad
            }
        current_ids = {id(parameter) for parameter in named.values()}
        for parameter in list(self.optimizer.state):
            if id(parameter) not in current_ids:
                self.optimizer.state.pop(parameter, None)

        router_parameters = [
            parameter
            for name, parameter in named.items()
            if self._is_router_parameter(name)
        ]
        router_ids = {id(parameter) for parameter in router_parameters}
        base_parameters = [
            parameter
            for parameter in named.values()
            if id(parameter) not in router_ids
        ]
        groups_by_name = {
            group.get("group_name"): group
            for group in self.optimizer.param_groups
            if group.get("group_name") in {"base", "router"}
        }
        base_group = groups_by_name.get("base")
        if base_group is None:
            base_group = {
                key: value
                for key, value in self.optimizer.param_groups[0].items()
                if key != "params"
            }
            base_group["group_name"] = "base"
        base_group["params"] = base_parameters
        base_group["lr"] = self.training_config.learning_rate

        router_group = groups_by_name.get("router")
        if router_group is None:
            router_group = {
                key: value for key, value in base_group.items() if key != "params"
            }
            router_group["group_name"] = "router"
        router_group["params"] = router_parameters
        router_group["lr"] = (
            self.training_config.learning_rate
            * self.training_config.router_lr_scale
        )
        self.optimizer.param_groups = [base_group, router_group]

    def _restore_optimizer(
        self,
        state_dict: Mapping[str, Any],
        parameter_groups: Optional[Sequence[Sequence[str]]],
    ) -> None:
        """Restore optimizer state against dynamic parameters by stable names."""
        if parameter_groups is None:
            if len(state_dict.get("param_groups", [])) != len(
                self.optimizer.param_groups
            ):
                raise RuntimeError(
                    "legacy checkpoint has dynamic optimizer groups without "
                    "parameter-name metadata; resume requires a format_version >= 3 checkpoint"
                )
            self.optimizer.load_state_dict(state_dict)
            return

        named = self._named_optimized_parameters()
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(parameter_groups):
            raise ValueError("optimizer group-name metadata is inconsistent")
        migrated_groups = []
        migrated_names = []
        for saved_group, names in zip(saved_groups, parameter_groups):
            if len(saved_group["params"]) != len(names):
                raise ValueError(
                    "optimizer parameter IDs and names are inconsistent"
                )
            kept = [
                (parameter_id, name)
                for parameter_id, name in zip(
                    saved_group["params"],
                    names,
                )
                if name in named
            ]
            removed = set(names).difference(name for _, name in kept)
            unsupported = {
                name
                for name in removed
                if not name.startswith((
                    "tree.routers.",
                    "deep_sleep_gate.",
                ))
            }
            if unsupported:
                raise KeyError(
                    "checkpoint optimizer parameters are missing: "
                    f"{sorted(unsupported)}"
                )
            migrated_group = dict(saved_group)
            migrated_group["params"] = [
                parameter_id for parameter_id, _ in kept
            ]
            migrated_groups.append(migrated_group)
            migrated_names.append([name for _, name in kept])
        retained_ids = {
            parameter_id
            for group in migrated_groups
            for parameter_id in group["params"]
        }
        migrated_state = {
            **state_dict,
            "param_groups": migrated_groups,
            "state": {
                parameter_id: value
                for parameter_id, value in state_dict["state"].items()
                if parameter_id in retained_ids
            },
        }
        saved_groups = migrated_groups
        parameter_groups = migrated_names
        groups = []
        seen: set[str] = set()
        for saved_group, names in zip(saved_groups, parameter_groups):
            duplicate = seen.intersection(names)
            if duplicate:
                raise ValueError(f"optimizer parameter names are duplicated: {duplicate}")
            seen.update(names)
            group = {
                key: value
                for key, value in saved_group.items()
                if key != "params"
            }
            group["params"] = [named[name] for name in names]
            groups.append(group)
        untracked = set(named).difference(seen)
        if untracked:
            # Newer model revisions may add a trainable head while resuming a
            # name-aware older checkpoint. Add such parameters without Adam
            # moments; all saved parameters retain their exact state.
            next_parameter_id = 1 + max(
                (
                    parameter_id
                    for group in migrated_state["param_groups"]
                    for parameter_id in group["params"]
                ),
                default=-1,
            )
            for name in sorted(untracked):
                desired_group = (
                    "router" if self._is_router_parameter(name) else "base"
                )
                target_index = next(
                    (
                        index
                        for index, group in enumerate(groups)
                        if group.get("group_name") == desired_group
                    ),
                    None,
                )
                if target_index is None:
                    raise KeyError(
                        f"optimizer has no {desired_group!r} group for {name!r}"
                    )
                groups[target_index]["params"].append(named[name])
                migrated_state["param_groups"][target_index]["params"].append(
                    next_parameter_id
                )
                next_parameter_id += 1
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=self.training_config.learning_rate,
        )
        self.optimizer.load_state_dict(migrated_state)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: Optional[torch.device | str] = None,
        encoder: Optional[nn.Module] = None,
        training: Optional[TrainingConfig] = None,
    ) -> "MemoryTreeTrainer":
        """Restore a complete trainer for additional wake/sleep epochs."""
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
                "model requires Compat(z_t, u_n). Retrain from H_tree/Hawkes "
                "initialization instead of resuming this checkpoint."
            )
        hawkes = HawkesFamily(
            num_types=config["num_event_types"],
            num_basis=config["num_basis"],
            decays=torch.as_tensor(config["decays"], dtype=torch.float32),
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
            memory_capacity_per_node=config.get("memory_capacity_per_node", 128),
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
            encoder = CausalPrefixEncoder(
                num_event_types=config["num_event_types"],
                z_dim=config["z_dim"],
                type_dim=config.get("encoder_type_dim") or 32,
                hidden_dim=config.get("encoder_hidden_dim") or 128,
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
            if key.startswith((
                "frontier_routing.prototypes.",
                "expansion_predictor.",
            ))
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
        tree.semantic_blend = float(config.get("semantic_blend", 0.0))
        tree.initialization_metadata = dict(
            config.get("initialization_metadata", {})
        )
        encoder.load_state_dict(checkpoint["encoder_state_dict"])

        loaded_training = TrainingConfig(**checkpoint.get("training_config", {}))
        sleep_payload = dict(checkpoint.get("sleep_config", {}))
        if (
            "deep_availability_tau" not in sleep_payload
            and "deep_cooldown_tau" in sleep_payload
        ):
            sleep_payload["deep_availability_tau"] = sleep_payload.pop(
                "deep_cooldown_tau"
            )
        sleep_field_names = {
            config_field.name for config_field in fields(SleepConfig)
        }
        # Checkpoints written before format 7 may contain fields from the
        # removed differentiable full-bank Sleep objective.
        sleep_payload = {
            key: value
            for key, value in sleep_payload.items()
            if key in sleep_field_names
        }
        structure_payload = dict(checkpoint.get("structure_config", {}))
        structure_field_names = {
            config_field.name for config_field in fields(StructureConfig)
        }
        # Controller v4-v6 checkpoints used low-mass/variational-prune fields
        # that have no structural meaning in the unified topology selector.
        structure_payload = {
            key: value
            for key, value in structure_payload.items()
            if key in structure_field_names
        }
        trainer = cls(
            tree=tree,
            hawkes=hawkes,
            encoder=encoder,
            wake=WakeObjectiveConfig(**checkpoint.get("wake_config", {})),
            sleep=SleepConfig(**sleep_payload),
            structure=StructureConfig(**structure_payload),
            training=loaded_training if training is None else training,
            device=device,
        )
        deep_gate_state = checkpoint.get("deep_sleep_gate_state_dict")
        deep_gate_was_migrated = False
        if deep_gate_state is not None:
            deep_gate_state = dict(deep_gate_state)
            raw_weights = deep_gate_state.get("raw_weights")
            expected_weights = trainer.deep_sleep_gate.raw_weights.detach()
            if raw_weights is not None:
                raw_weights = raw_weights.reshape(-1)
                expected_count = expected_weights.numel()
                if raw_weights.numel() < expected_count:
                    padding = expected_weights[raw_weights.numel():].to(
                        device=raw_weights.device,
                        dtype=raw_weights.dtype,
                    )
                    raw_weights = torch.cat((raw_weights, padding))
                    deep_gate_was_migrated = True
                elif raw_weights.numel() > expected_count:
                    raw_weights = raw_weights[:expected_count]
                    deep_gate_was_migrated = True
                deep_gate_state["raw_weights"] = raw_weights
            trainer.deep_sleep_gate.load_state_dict(deep_gate_state)
        trainer._restore_optimizer(
            checkpoint["optimizer_state_dict"],
            checkpoint.get("optimizer_param_groups"),
        )
        deep_gate_optimizer_state = checkpoint.get(
            "deep_gate_optimizer_state_dict"
        )
        if (
            deep_gate_optimizer_state is not None
            and not deep_gate_was_migrated
        ):
            trainer.deep_gate_optimizer.load_state_dict(
                deep_gate_optimizer_state
            )
        # A legacy three-feature checkpoint has three-element Adam moments.
        # Reusing them for the four-feature parameter would fail on the first
        # optimizer step, so migrated gates intentionally restart only this
        # small, disjoint optimizer.
        topology_selector_state = checkpoint.get(
            "topology_selector_state_dict"
        )
        if topology_selector_state is not None:
            trainer.topology_selector.load_state_dict(
                topology_selector_state
            )
        topology_selector_optimizer_state = checkpoint.get(
            "topology_selector_optimizer_state_dict"
        )
        if topology_selector_optimizer_state is not None:
            trainer.topology_selector_optimizer.load_state_dict(
                topology_selector_optimizer_state
            )
        trainer._sync_split_modules()
        split_states = checkpoint.get("split_module_state_dicts", {})
        for leaf_id, state in split_states.items():
            if leaf_id in trainer.split_modules:
                trainer.split_modules[leaf_id].load_state_dict(state)
        trainer.controller.split_queues.update(
            checkpoint.get("controller_state", {}).get("split_queues", {})
        )
        trainer.sleep_state.update(
            checkpoint.get("sleep_state", {})
        )
        merge_dual_state = checkpoint.get(
            "topology_prune_dual_state",
            checkpoint.get("merge_dual_state", {}),
        )
        trainer.merge_lambda_T = float(
            merge_dual_state.get(
                "lambda_T",
                trainer.merge_lambda_T,
            )
        )
        budget_KT = merge_dual_state.get("budget_KT")
        trainer.merge_budget_KT = (
            None if budget_KT is None else float(budget_KT)
        )
        reliability_state = checkpoint.get(
            "routing_reliability_state",
            {},
        )
        trainer.encoder_routing_reliability = float(
            reliability_state.get("encoder_gate", 0.0)
        )
        trainer.last_teacher_confidence = float(
            reliability_state.get("teacher_confidence", 0.0)
        )
        trainer.last_teacher_student_js = float(
            reliability_state.get(
                "teacher_student_js",
                math.log(2.0),
            )
        )
        trainer.last_teacher_student_alignment = float(
            reliability_state.get("teacher_student_alignment", 0.0)
        )
        controller_module_state = checkpoint.get(
            "controller_state",
            {},
        ).get("module_state_dict")
        if controller_module_state is not None:
            incompatible_controller = trainer.controller.load_state_dict(
                controller_module_state, strict=False
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
            ):
                raise RuntimeError(
                    "checkpoint is missing controller tensors: "
                    f"{incompatible_controller.missing_keys}"
                )
            controller_version = checkpoint.get("controller_state", {}).get(
                "controller_version", 1
            )
            if controller_version < 2:
                trainer.controller.migrate_legacy_policy()
                for parameter in trainer.controller.parameters():
                    trainer.optimizer.state.pop(parameter, None)
            if controller_version < 5:
                trainer.controller.split_enabled.fill_(True)
            if incompatible_controller.unexpected_keys:
                raise RuntimeError(
                    "checkpoint contains unexpected controller tensors: "
                    f"{incompatible_controller.unexpected_keys}"
                )
        trainer.controller.utility_stage_enabled = bool(
            checkpoint.get("controller_state", {}).get(
                "utility_stage_enabled",
                trainer.controller.utility_stage_enabled,
            )
        )
        trainer.controller_calibration = dict(
            checkpoint.get("controller_calibration", {})
        )
        replay_state = checkpoint.get("controller_state", {}).get(
            "utility_replay"
        )
        if replay_state is not None:
            trainer.controller_utility_replay.load_state_dict(replay_state)
        trainer._reconcile_optimizer_parameters()
        trainer.history = list(checkpoint.get("history", []))
        trainer.completed_epochs = int(checkpoint.get("epoch", 0))
        rng_state = checkpoint.get("rng_state", {})
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"].cpu())
        if torch.cuda.is_available() and "cuda" in rng_state:
            cuda_states = normalize_cuda_rng_states(
                rng_state["cuda"],
                max_devices=torch.cuda.device_count(),
            )
            for device_index, state in enumerate(cuda_states):
                torch.cuda.set_rng_state(state, device=device_index)
        return trainer

    def prepare_controller_only_finetune(
        self,
        *,
        base_checkpoint: str | Path,
        target_version: int = 5,
        train_heads: Sequence[str] = ("adapt", "retrieve", "write"),
    ) -> None:
        """Reset state and expose only explicitly selected Controller heads."""
        allowed = {"adapt", "retrieve", "write"}
        selected = set(train_heads)
        if target_version not in (5, 6):
            raise ValueError("target_version must be 5 or 6")
        if not selected or selected.difference(allowed):
            raise ValueError("train_heads must contain only adapt,retrieve,write")
        for parameter in self.hawkes.parameters():
            parameter.requires_grad_(False)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.tree.parameters():
            parameter.requires_grad_(False)
        for parameter in self.controller.parameters():
            parameter.requires_grad_(False)
        parameters_by_head = {
            "adapt": ("bias_assimilate", "raw_assimilate_surprise"),
            "retrieve": (
                "bias_retrieve", "raw_retrieve_surprise", "raw_retrieve_novelty",
            ),
            "write": (
                "bias_memorize", "raw_memorize_surprise",
                "raw_memorize_novelty", "raw_memorize_count",
            ),
        }
        for head in selected:
            for name in parameters_by_head[head]:
                getattr(self.controller, name).requires_grad_(True)
        context_weight = self.controller.context_gate.weight
        context_weight.requires_grad_(True)
        row_by_head = {"adapt": 0, "retrieve": 1, "write": 2}
        gradient_mask = torch.zeros_like(context_weight)
        for head in selected:
            gradient_mask[row_by_head[head]].fill_(1.0)
        previous_hook = getattr(self, "_controller_context_gradient_hook", None)
        if previous_hook is not None:
            previous_hook.remove()
        self._controller_context_gradient_hook = context_weight.register_hook(
            lambda gradient, mask=gradient_mask: gradient * mask
        )
        self.controller.controller_version.fill_(target_version)
        self.controller.split_enabled.fill_(False)
        if target_version < 6:
            self.controller.set_calibration_thresholds(0.0, 0.0, 0.6)
        row_by_head = {"adapt": 0, "retrieve": 1, "write": 2, "split": 3}
        # A Controller-only run resets statistics only for trainable heads.
        # Inactive statistics are part of the frozen policy contract.
        for head in selected:
            index = row_by_head[head]
            self.controller.utility_mean[index].zero_()
            self.controller.utility_variance[index].fill_(1.0)
            self.controller.utility_observations[index].zero_()
            self.controller.utility_temperatures[index].fill_(
                self.wake_config.controller_utility_temperature
            )
        self.controller.split_queues.clear()
        self.controller_utility_replay = ControllerUtilityReplay(
            self.wake_config.controller_replay_capacities,
            seed=self.training_config.seed,
        )
        self.controller_utility_replay.write_ranking_enabled = bool(
            self.training_config.controller_write_ranking
        )
        self.history = []
        self.completed_epochs = 0
        self.validation_history = []
        self.best_validation = None
        self.training_config.controller_only_finetune = True
        self.training_config.controller_base_checkpoint = str(base_checkpoint)
        self.training_config.controller_target_version = int(target_version)
        self.training_config.controller_train_heads = tuple(sorted(selected))
        def parameter_sha256(value: Tensor) -> str:
            tensor = value.detach().cpu().contiguous()
            digest = hashlib.sha256()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
            return digest.hexdigest()

        self._frozen_controller_parameter_sha256 = {
            name: parameter_sha256(parameter)
            for name, parameter in self.controller.named_parameters()
            if not parameter.requires_grad
        }
        # Context rows belonging to inactive heads are protected by a gradient
        # mask and audited separately at checkpoint time.
        self._frozen_controller_context_rows = {
            row: parameter_sha256(context_weight[row])
            for row in range(context_weight.size(0))
            if row not in {row_by_head[head] for head in selected}
        }
        from ControllerIsolation import (
            head_policy_sha256,
            protected_write_only_items,
            tensor_sha256 as isolation_tensor_sha256,
        )
        controller_state = self.controller.state_dict()
        self._base_retrieve_policy_sha256 = head_policy_sha256(
            controller_state, "retrieve"
        )
        self._base_adapt_policy_sha256 = head_policy_sha256(
            controller_state, "adapt"
        )
        base_path = Path(base_checkpoint)
        base_payload = (
            torch.load(base_path, map_location="cpu", weights_only=False)
            if base_path.is_file() else {}
        )
        self._base_frozen_event_sha256 = base_payload.get(
            "write_rollout_calibration", {}
        ).get("frozen_event_sha256")
        self._frozen_controller_item_sha256 = {
            name: isolation_tensor_sha256(value)
            for name, value in protected_write_only_items(controller_state).items()
        }
        scalar_parameters = [
            parameter for parameter in self.controller.parameters()
            if parameter.requires_grad and parameter is not context_weight
        ]
        optimizer_groups = []
        if scalar_parameters:
            optimizer_groups.append({
                "params": scalar_parameters,
                "weight_decay": self.training_config.weight_decay,
            })
        optimizer_groups.append({"params": [context_weight], "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.training_config.learning_rate,
        )
        self.tree.reset_working_memory()
        self._frozen_state_item_hashes = (
            self.non_controller_state_item_sha256()
        )
        self.training_config.frozen_state_sha256 = (
            self.non_controller_state_sha256()
        )

    def non_controller_state_item_sha256(self) -> Dict[str, str]:
        """Per-item hashes used to identify accidental frozen-state writes."""
        result: Dict[str, str] = {}

        def visit(value: Any, prefix: str) -> None:
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                digest = hashlib.sha256()
                digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.numpy().tobytes())
                result[prefix] = digest.hexdigest()
            elif isinstance(value, Mapping):
                for key in sorted(value, key=str):
                    visit(value[key], f"{prefix}.{key}")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    visit(item, f"{prefix}.{index}")
            else:
                result[prefix] = hashlib.sha256(repr(value).encode()).hexdigest()

        visit(self.hawkes.state_dict(), "hawkes")
        visit(self.encoder.state_dict(), "encoder")
        visit({
            key: value for key, value in self.tree.state_dict().items()
            if not key.startswith("working_memory.")
        }, "tree")
        visit(tuple(self.tree.all_node_ids), "topology.nodes")
        visit(tuple(self.tree.leaf_ids), "topology.leaves")
        visit(self.deep_sleep_gate.state_dict(), "sleep.deep_gate")
        visit(self.topology_selector.state_dict(), "sleep.topology_selector")
        visit(self.sleep_state, "sleep.scheduler")
        return result

    def non_controller_state_sha256(self) -> str:
        """Stable digest of every state item that v5 promises to freeze."""
        digest = hashlib.sha256()
        for name, value in sorted(self.non_controller_state_item_sha256().items()):
            digest.update(name.encode())
            digest.update(value.encode())
        return digest.hexdigest()

