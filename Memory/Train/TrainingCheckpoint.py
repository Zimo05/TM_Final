"""Training checkpoint persistence."""

from __future__ import annotations

from Train.TrainingComponents import *  # noqa: F403


class TrainingCheckpointMixin:
    def save_checkpoint(self, path: str | Path, *, epoch: int) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        split_state = {
            leaf_id: module.state_dict()
            for leaf_id, module in self.split_modules.items()
        }
        self._reconcile_optimizer_parameters()
        name_by_id = {
            id(parameter): name
            for name, parameter in self._named_optimized_parameters().items()
        }
        optimizer_param_groups = []
        for group in self.optimizer.param_groups:
            names = []
            for parameter in group["params"]:
                name = name_by_id.get(id(parameter))
                if name is None:
                    raise RuntimeError(
                        "optimizer contains a parameter that is not owned by tree/encoder"
                    )
                names.append(name)
            optimizer_param_groups.append(names)
        def tensor_sha256(value: Tensor) -> str:
            tensor = value.detach().cpu().contiguous()
            digest = hashlib.sha256()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
            return digest.hexdigest()

        frozen_controller_expected = getattr(
            self, "_frozen_controller_parameter_sha256", {}
        )
        frozen_controller_actual = {
            name: tensor_sha256(parameter)
            for name, parameter in self.controller.named_parameters()
            if name in frozen_controller_expected
        }
        frozen_rows_expected = getattr(
            self, "_frozen_controller_context_rows", {}
        )
        frozen_rows_actual = {
            row: tensor_sha256(self.controller.context_gate.weight[row])
            for row in frozen_rows_expected
        }
        controller_heads_verified = (
            frozen_controller_actual == frozen_controller_expected
            and frozen_rows_actual == frozen_rows_expected
        )
        from ControllerIsolation import (
            file_sha256,
            head_policy_sha256,
            protected_write_only_items,
            tensor_sha256 as isolation_tensor_sha256,
        )
        current_controller_state = self.controller.state_dict()
        write_only = set(self.training_config.controller_train_heads) == {"write"}
        frozen_items_expected = getattr(
            self, "_frozen_controller_item_sha256", {}
        )
        frozen_items_actual = {
            name: isolation_tensor_sha256(value)
            for name, value in protected_write_only_items(
                current_controller_state
            ).items()
        } if write_only else {}
        retrieve_expected = getattr(self, "_base_retrieve_policy_sha256", None)
        adapt_expected = getattr(self, "_base_adapt_policy_sha256", None)
        retrieve_actual = head_policy_sha256(current_controller_state, "retrieve")
        adapt_actual = head_policy_sha256(current_controller_state, "adapt")
        write_isolation_verified = bool(
            not write_only
            or (
                frozen_items_actual == frozen_items_expected
                and retrieve_actual == retrieve_expected
                and adapt_actual == adapt_expected
            )
        )
        controller_heads_verified = (
            controller_heads_verified and write_isolation_verified
        )
        base_path = Path(self.training_config.controller_base_checkpoint or "")
        base_checkpoint_sha256 = (
            file_sha256(base_path) if base_path.is_file() else None
        )
        checkpoint = {
            "format_version": 18,
            "epoch": epoch,
            "controller_policy_revision": (
                4 if self.training_config.controller_write_ranking else 3
            ),
            "write_supervision": (
                "sequence_relative_ranking"
                if self.training_config.controller_write_ranking
                else "signed_utility"
            ),
            "checkpoint_identity": {
                "path": str(output_path.resolve()),
                "role": (
                    "best"
                    if self.training_config.best_checkpoint_path
                    and output_path.resolve() == Path(
                        self.training_config.best_checkpoint_path
                    ).resolve()
                    else "last"
                ),
            },
            "tree_state_dict": self.tree.state_dict(),
            "hawkes_state_dict": self.hawkes.state_dict(),
            "encoder_state_dict": self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer_param_groups": optimizer_param_groups,
            "split_module_state_dicts": split_state,
            "controller_state": {
                "controller_version": int(
                    self.controller.controller_version.detach().cpu().item()
                ),
                "split_queues": dict(self.controller.split_queues),
                "module_state_dict": self.controller.state_dict(),
                "utility_stage_enabled": bool(
                    self.controller.utility_stage_enabled
                ),
                "utility_statistics": {
                    "mean": self.controller.utility_mean.detach().cpu(),
                    "variance": self.controller.utility_variance.detach().cpu(),
                    "observations": self.controller.utility_observations.detach().cpu(),
                },
                "utility_temperatures": (
                    self.controller.utility_temperatures.detach().cpu()
                ),
                "replay_sign_counts": self.controller_utility_replay.sign_counts(),
                "calibration": {
                    **self.controller.calibration_dict(),
                    **getattr(self, "controller_calibration", {}),
                },
                "split_enabled": bool(self.controller.split_enabled),
                "utility_replay": self.controller_utility_replay.state_dict(),
                "migration": (
                    "causal_post_admission_controller_v6"
                    if int(self.controller.controller_version.detach().cpu()) >= 6
                    else "retrieval_mediated_controller_v5"
                ),
                "trainable_heads": list(
                    self.training_config.controller_train_heads
                ),
                "write_ranking_enabled": bool(
                    self.training_config.controller_write_ranking
                ),
            },
            "controller_calibration": {
                **self.controller.calibration_dict(),
                **getattr(self, "controller_calibration", {}),
            },
            "controller_only_invariants": {
                "enabled": bool(
                    self.training_config.controller_only_finetune
                ),
                "base_checkpoint": self.training_config.controller_base_checkpoint,
                "expected_sha256": self.training_config.frozen_state_sha256,
                "actual_sha256": (
                    self.non_controller_state_sha256()
                    if self.training_config.controller_only_finetune
                    else None
                ),
                "verified": bool(
                    self.training_config.controller_only_finetune
                    and self.training_config.frozen_state_sha256
                    == self.non_controller_state_sha256()
                    and controller_heads_verified
                ),
                "controller_target_version": int(
                    self.training_config.controller_target_version
                ),
                "trainable_heads": list(
                    self.training_config.controller_train_heads
                ),
                "inactive_controller_heads_verified": controller_heads_verified,
                "inactive_parameter_expected_sha256": frozen_controller_expected,
                "inactive_parameter_actual_sha256": frozen_controller_actual,
                "inactive_context_rows_expected_sha256": frozen_rows_expected,
                "inactive_context_rows_actual_sha256": frozen_rows_actual,
                "frozen_controller_item_sha256": frozen_items_expected,
                "frozen_controller_item_actual_sha256": frozen_items_actual,
                "retrieve_policy_sha256": retrieve_actual,
                "retrieve_policy_expected_sha256": retrieve_expected,
                "adapt_policy_sha256": adapt_actual,
                "adapt_policy_expected_sha256": adapt_expected,
                "base_checkpoint_sha256": base_checkpoint_sha256,
                "write_isolation_verified": write_isolation_verified,
            },
            "deep_sleep_gate_state_dict": self.deep_sleep_gate.state_dict(),
            "deep_gate_optimizer_state_dict": (
                self.deep_gate_optimizer.state_dict()
            ),
            "topology_selector_state_dict": self.topology_selector.state_dict(),
            "topology_selector_optimizer_state_dict": (
                self.topology_selector_optimizer.state_dict()
            ),
            "sleep_state": dict(self.sleep_state),
            "merge_dual_state": {
                "lambda_T": float(self.merge_lambda_T),
                "budget_KT": self.merge_budget_KT,
            },
            "topology_prune_dual_state": {
                "lambda_T": float(self.merge_lambda_T),
                "budget_KT": self.merge_budget_KT,
            },
            "routing_reliability_state": {
                "encoder_gate": float(
                    self.encoder_routing_reliability
                ),
                "teacher_confidence": float(
                    self.last_teacher_confidence
                ),
                "teacher_student_js": float(
                    self.last_teacher_student_js
                ),
                "teacher_student_alignment": float(
                    self.last_teacher_student_alignment
                ),
            },
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    [
                        state.detach().cpu()
                        for state in torch.cuda.get_rng_state_all()
                    ]
                    if torch.cuda.is_available()
                    else []
                ),
            },
            "model_config": {
                "z_dim": self.tree.z_dim,
                "node_dim": self.tree.node_dim,
                "num_event_types": self.hawkes.num_types,
                "num_basis": self.hawkes.num_basis,
                "memory_key_dim": self.tree.episodic_memory.key_dim,
                "memory_capacity_per_node": (
                    self.tree.episodic_memory.capacity_per_node
                ),
                "tree_temperature": self.tree.temperature,
                "router_kind": "posterior_frontier_v2",
                "frontier_routing_config": asdict(
                    self.tree.frontier_routing.config
                ),
                "semantic_blend": self.tree.semantic_blend,
                "initialization_metadata": dict(
                    getattr(self.tree, "initialization_metadata", {})
                ),
                "hyper_hidden_dim": self.tree.hyper.hidden_dim,
                "working_rho": self.tree.working_memory.rho,
                "working_eta": self.tree.working_memory.eta,
                "decays": self.hawkes.decays.detach().cpu(),
                "encoder_type_dim": getattr(self.encoder, "type_dim", None),
                "encoder_hidden_dim": getattr(self.encoder, "hidden_dim", None),
                "encoder_config": (
                    self.encoder.get_config()
                    if hasattr(self.encoder, "get_config")
                    else {"kind": "causal_prefix"}
                ),
            },
            "data_provenance": dict(
                getattr(
                    self.tree,
                    "data_provenance",
                    {"evaluation_regime": "transductive"},
                )
            ),
            "wake_config": asdict(self.wake_config),
            "sleep_config": asdict(self.sleep_config),
            "structure_config": asdict(self.structure_config),
            "training_config": asdict(self.training_config),
            "history": self.history,
            "validation_selection": {
                "history": list(self.validation_history),
                "best": self.best_validation,
                "constraint_passed": bool(
                    self.best_validation
                    and self.best_validation.get("constraint_passed", False)
                ),
                "selection_reason": (
                    (
                        "lowest constrained full_online NLL over immutable baseline"
                        if self.training_config.controller_write_ranking
                        else "lowest constrained full_frozen NLL"
                    )
                    if self.best_validation and self.best_validation.get(
                        "constraint_passed", False
                    ) else (
                        "no ranking checkpoint improved realized rollout"
                        if self.training_config.controller_write_ranking
                        else "fallback actions-closed full_frozen NLL; constraints not met"
                    )
                ),
            },
        }
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(output_path)
        return output_path.resolve()
