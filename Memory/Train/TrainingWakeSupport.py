"""Wake-phase routing, memory-write, and encoding helpers."""

from __future__ import annotations

from Train.TrainingComponents import *  # noqa: F403


class TrainingWakeSupportMixin:
    def _controller_effective_parameters(
        self,
        memory_output: Mapping[str, Any],
        working_delta: Tensor,
        retrieval_gate: Tensor,
        *,
        row_indices: Optional[Tensor] = None,
    ):
        """Recompose Hawkes parameters while leaving routing weights untouched."""
        semantic = memory_output["frontier_semantic_theta"]
        episodic = memory_output["frontier_episodic_delta"]
        routing = memory_output["r"]
        if row_indices is not None:
            semantic = semantic.index_select(0, row_indices)
            episodic = episodic.index_select(0, row_indices)
            routing = routing.index_select(0, row_indices)
        gate = torch.as_tensor(retrieval_gate).to(episodic)
        while gate.ndim < episodic.ndim - 1:
            gate = gate.unsqueeze(-1)
        gated_episodic = episodic * gate.unsqueeze(-1)
        D = self.hawkes.num_types
        return self.tree.episodic_memory.parameter_update.compose_effective_parameters(
            semantic_mu=semantic[..., :D],
            semantic_W=semantic[..., D:].reshape(
                *semantic.shape[:-1], D, D, self.hawkes.num_basis
            ),
            episodic_delta=gated_episodic,
            routing_weights=routing,
            working_delta=working_delta,
            decays=self.hawkes.decays,
        )

    def _move_sequence(self, sequence: Mapping[str, Tensor]) -> Dict[str, Any]:
        if "times" not in sequence or "types" not in sequence:
            raise ValueError("each sequence requires times and types")

        # ``train()`` makes the complete dataset GPU-resident and builds all
        # parameter-independent Hawkes features once. Revalidating a resident
        # sequence used to perform several ``.item()`` calls here and inside
        # ``prepare_sequence_cache`` on every Wake and Global pass. On CUDA
        # those scalar checks force device synchronization and leave the GPU
        # idle. Shape/device/signature checks are host-side and are sufficient
        # for this immutable resident cache.
        times = sequence["times"]
        types = sequence["types"]
        history = sequence.get(HAWKES_HISTORY_STATS_KEY)
        interval = sequence.get(HAWKES_INTERVAL_STATS_KEY)
        time_features = sequence.get(EVENT_TIME_FEATURES_KEY)
        event_count = int(times.numel()) if torch.is_tensor(times) else -1
        expected_hawkes_shape = (
            event_count,
            self.hawkes.num_types,
            self.hawkes.num_basis,
        )
        required_tensors = (
            times,
            types,
            history,
            interval,
            time_features,
        )
        resident_ready = (
            event_count >= 0
            and all(torch.is_tensor(value) for value in required_tensors)
            and all(value.device == self.device for value in required_tensors)
            and sequence.get(HAWKES_CACHE_SIGNATURE_KEY)
            == self.hawkes.cache_signature
            and times.ndim == 1
            and types.shape == times.shape
            and types.dtype == torch.long
            and history.shape == expected_hawkes_shape
            and interval.shape == expected_hawkes_shape
            and time_features.shape == (event_count, 2)
            and (
                "T" not in sequence
                or (
                    torch.is_tensor(sequence["T"])
                    and sequence["T"].device == self.device
                )
            )
        )
        if resident_ready:
            self._resident_cache_hits += 1
            return dict(sequence)

        self._resident_cache_misses += 1
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
        if result["times"].ndim != 1 or result["types"].shape != result["times"].shape:
            raise ValueError("times/types must have the same one-dimensional shape")
        if result["times"].numel() and bool(
            (result["times"][1:] < result["times"][:-1]).any().item()
        ):
            raise ValueError("event times must be non-decreasing")
        if result["types"].numel() and (
            int(result["types"].min()) < 0
            or int(result["types"].max()) >= self.hawkes.num_types
        ):
            raise ValueError("event types are outside the model vocabulary")
        if "T" in sequence:
            result["T"] = sequence["T"].to(self.device)
        return result

    def _decide_action(
        self,
        memory_output: Mapping[str, Any],
        prediction_nll: Tensor,
        frontier_energy: Tensor,
        frontier_node_ids: Sequence[str],
    ) -> tuple[
        Tensor,
        str,
        bool,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Choose a controller action on the observed frontier posterior."""
        posterior = self._frontier_posterior(
            memory_output["frontier_mass"][0],
            frontier_energy,
            memory_output["frontier_mask"][0],
        )
        owner_id, owner_is_lca, owner_confidence = self._posterior_owner(
            frontier_node_ids,
            posterior,
        )
        query = memory_output["memory_query"][0]
        novelty, count, max_similarity = (
            self.controller.leaf_novelty_count(query, owner_id)
        )
        controller_output = self.controller.action_distribution(
            surprise=prediction_nll,
            novelty=novelty,
            count=count,
            owner_confidence=prediction_nll.new_tensor(owner_confidence),
            retrieval_similarity=max_similarity,
            retrieval_residual_norm=memory_output[
                "frontier_episodic_delta"
            ][0].norm(dim=-1).mean(),
            working_memory_norm=prediction_nll.new_zeros(()),
            pending_write_ratio=prediction_nll.new_zeros(()),
        )
        action_probabilities = controller_output["probabilities"]
        raw_action_probabilities = controller_output.get(
            "raw_probabilities", action_probabilities
        )
        # Controller probabilities drive the actual differentiable updates.
        # The argmax is diagnostic only, so keep it on-device and materialize
        # all action labels once at sequence end.
        action_index = action_probabilities.detach().argmax()
        return (
            action_index,
            owner_id,
            owner_is_lca,
            query,
            novelty,
            count.detach(),
            max_similarity.detach(),
            action_probabilities,
            raw_action_probabilities,
            posterior.detach(),
        )

    def _frontier_posterior(
        self,
        prior_mass: Tensor,
        energy: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Return q+ ∝ m- exp(-E/tau) on actual frontier support."""
        temperature = self.tree.frontier_routing.config.posterior_temperature
        logits = (
            prior_mass.clamp_min(1e-12).log()
            - energy / temperature
        ).masked_fill(~mask, -torch.inf)
        posterior = F.softmax(logits, dim=-1)
        return posterior.masked_fill(~mask, 0.0)

    def _lowest_common_ancestor(self, node_ids: Sequence[str]) -> str:
        if not node_ids:
            raise ValueError("node_ids cannot be empty")
        paths = [self.tree.path_to_node(node_id) for node_id in node_ids]
        common = "root"
        for values in zip(*paths):
            if len(set(values)) != 1:
                break
            common = values[0]
        return common

    def _posterior_owner(
        self,
        frontier_node_ids: Sequence[str],
        posterior: Tensor,
    ) -> tuple[str, bool, float]:
        """Credible-set/LCA owner from posterior over computed experts."""
        config = self.tree.frontier_routing.config
        if posterior.numel() != len(frontier_node_ids):
            posterior = posterior[: len(frontier_node_ids)]
        order = posterior.argsort(descending=True, stable=True)
        cumulative = posterior.index_select(0, order).cumsum(dim=0)
        credible_count_tensor = (
            cumulative < config.credible_mass
        ).sum() + 1
        # One boundary synchronization carries the complete hard-decision
        # state. Previously four separate .item()/.cpu() calls serialized the
        # CUDA stream for every event.
        hard_state = torch.cat([
            credible_count_tensor.reshape(1).to(posterior),
            order.to(posterior),
            posterior.index_select(0, order[:1]),
        ]).detach().cpu().tolist()
        credible_count = int(hard_state[0])
        credible_count = min(credible_count, len(frontier_node_ids))
        selected = [int(value) for value in hard_state[1:1 + credible_count]]
        top_index = int(hard_state[1])
        top_confidence = float(hard_state[-1])
        if (
            credible_count == 1
            and top_confidence >= config.owner_confidence_threshold
        ):
            return frontier_node_ids[top_index], False, top_confidence
        owner = self._lowest_common_ancestor(
            [frontier_node_ids[index] for index in selected]
        )
        return owner, True, top_confidence

    def _lca_index_table(self, reference: Tensor) -> Tensor:
        """Return a topology-versioned device table for pairwise LCA folds."""
        signature = tuple(
            (
                node_id,
                self.tree.nodes[node_id].parent,
                self.tree.nodes[node_id].left,
                self.tree.nodes[node_id].right,
            )
            for node_id in self.tree.all_node_ids
        )
        cached = getattr(self, "_wake_lca_table_cache", None)
        if (
            cached is not None
            and cached[0] == signature
            and cached[1].device == reference.device
        ):
            return cached[1]
        node_ids = tuple(self.tree.all_node_ids)
        node_index = {
            node_id: index for index, node_id in enumerate(node_ids)
        }
        values = [
            [
                node_index[self._lowest_common_ancestor((left, right))]
                for right in node_ids
            ]
            for left in node_ids
        ]
        table = torch.tensor(
            values,
            device=reference.device,
            dtype=torch.long,
        )
        self._wake_lca_table_cache = (signature, table)
        return table

    def _posterior_owner_indices_batch(
        self,
        frontier_node_indices: Tensor,
        posterior: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """GPU credible-set/LCA owner construction for a whole wavefront."""
        if (
            posterior.ndim != 2
            or frontier_node_indices.shape != posterior.shape
        ):
            raise ValueError(
                "frontier indices and posterior must align as [B, K]"
            )
        config = self.tree.frontier_routing.config
        order = posterior.argsort(
            dim=-1,
            descending=True,
            stable=True,
        )
        sorted_mass = posterior.gather(1, order)
        credible_count = (
            sorted_mass.cumsum(dim=-1) < config.credible_mass
        ).sum(dim=-1) + 1
        sorted_nodes = frontier_node_indices.clamp_min(0).gather(1, order)
        owner = sorted_nodes[:, 0]
        lca_table = self._lca_index_table(posterior)
        for slot in range(1, posterior.size(1)):
            combined = lca_table[owner, sorted_nodes[:, slot]]
            owner = torch.where(slot < credible_count, combined, owner)
        top_confidence = sorted_mass[:, 0]
        is_lca = ~(
            (credible_count == 1)
            & (
                top_confidence
                >= config.owner_confidence_threshold
            )
        )
        return owner, is_lca, top_confidence

    def _make_write_request(
        self,
        event_index: int,
        provisional_owner_id: str,
        query: Tensor,
        novelty: Tensor,
        memory_output: Mapping[str, Any],
        action_probabilities: Tensor,
        action_index: Tensor,
        posterior: Tensor,
        semantic_theta_cache: Optional[Mapping[str, Tensor]] = None,
        frontier_node_ids: Optional[Sequence[str]] = None,
        hard_action: Optional[Action] = None,
        batch_index: int = 0,
        controller_inputs: Optional[Mapping[str, Tensor]] = None,
        assimilation_theta: Optional[Tensor] = None,
        assimilation_grad: Optional[Tensor] = None,
        future_contexts: Optional[list[Mapping[str, Any]]] = None,
        raw_action_probabilities: Optional[Tensor] = None,
        exploration: Optional[Any] = None,
        controller_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Snapshot a causal write request without inspecting future events."""
        frontier_ids = (
            tuple(memory_output["frontier_node_ids"][batch_index])
            if frontier_node_ids is None
            else tuple(frontier_node_ids)
        )
        count = len(frontier_ids)
        confidence = posterior[:count].max()
        version = (
            int(controller_version)
            if controller_version is not None
            else int(self.controller.controller_version.detach().cpu())
        )
        action = (
            tuple(Action)[int(action_index.detach().cpu())]
            if hard_action is None
            else hard_action
        )
        frontier_node_indices = memory_output.get("frontier_node_indices")
        if frontier_node_indices is not None:
            frontier_node_indices = frontier_node_indices[
                batch_index, :count
            ].detach().clone()
        if exploration is None:
            exploration_value: Any = bool(
                action_probabilities[2].detach().cpu()
                < self.controller.write_candidate_threshold
            )
        elif torch.is_tensor(exploration):
            exploration_value = exploration.detach().clone()
        else:
            exploration_value = bool(exploration)
        request_probabilities = (
            action_probabilities
            if raw_action_probabilities is None
            else raw_action_probabilities
        )
        structural_weight = self.controller.queue_weight(
            request_probabilities
        ).detach()
        return {
            "event_index": event_index,
            "ready_index": (
                event_index + 2 * self.wake_config.write_horizon - 1
                if version >= 6
                else event_index + (
                    2 if self.training_config.controller_only_finetune else 1
                ) * self.wake_config.write_horizon
            ),
            "admission_index": (
                event_index + self.wake_config.write_horizon - 1
            ),
            "provisional_owner_id": provisional_owner_id,
            "query": query.detach().clone(),
            "frontier_node_ids": frontier_ids,
            "frontier_node_indices": frontier_node_indices,
            "frontier_mass": memory_output["frontier_mass"][
                batch_index, :count
            ].detach().clone(),
            "frontier_theta": memory_output["frontier_theta"][
                batch_index, :count
            ].detach().clone(),
            "action": action,
            "novelty": novelty.detach().clone(),
            "write_gate": request_probabilities[2].detach(),
            "action_probabilities": action_probabilities.detach().clone(),
            "raw_action_probabilities": (
                request_probabilities.detach().clone()
            ),
            "exploration": exploration_value,
            "structural_weight": structural_weight,
            # Compatibility alias for MemoryItem/MemoryBank checkpoints.
            "queue_weight": structural_weight,
            "controller_inputs": {
                key: value.detach().clone()
                for key, value in (controller_inputs or {}).items()
            },
            "assimilation_theta": (
                None if assimilation_theta is None
                else assimilation_theta.detach().clone()
            ),
            "assimilation_grad": (
                None if assimilation_grad is None
                else assimilation_grad.detach().clone()
            ),
            "future_contexts": future_contexts,
        }

    def _write_probe_context(
        self,
        *,
        event_index: int,
        memory_output: Mapping[str, Any],
        batch_index: int,
        frontier_node_ids: Sequence[str],
        query: Tensor,
        posterior: Tensor,
        working_delta: Tensor,
        retrieve_gate: Tensor,
        no_write_theta: Tensor,
    ) -> Dict[str, Any]:
        count = len(frontier_node_ids)
        return {
            "event_index": int(event_index),
            "query": query.detach().clone(),
            "frontier_node_ids": tuple(frontier_node_ids),
            "frontier_semantic_theta": memory_output["frontier_semantic_theta"][
                batch_index, :count
            ].detach().clone(),
            "frontier_episodic_delta": memory_output["frontier_episodic_delta"][
                batch_index, :count
            ].detach().clone(),
            "posterior": posterior[:count].detach().clone(),
            "working_delta": working_delta.detach().clone(),
            "retrieve_gate": retrieve_gate.detach().clone(),
            "no_write_theta": no_write_theta.detach().clone(),
        }

    @staticmethod
    def _controller_input_tensor(inputs: Mapping[str, Tensor]) -> Tensor:
        names = (
            "surprise", "novelty", "count", "owner_confidence",
            "retrieval_similarity", "retrieval_residual_norm",
            "working_memory_norm", "pending_write_ratio",
        )
        reference = inputs["surprise"]
        return torch.stack([
            torch.as_tensor(inputs[name]).to(reference).reshape(())
            for name in names
        ])

    @staticmethod
    def _host_int(value: Any, default: int = -1) -> int:
        """Materialize one metadata scalar without doing it per replay field."""
        if value is None:
            return int(default)
        if torch.is_tensor(value):
            if value.numel() == 0:
                return int(default)
            return int(value.detach().cpu().reshape(-1)[0].item())
        return int(value)

    def _add_controller_utility(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
        *,
        action_index: int,
        utility: Tensor,
        propensity: float,
    ) -> None:
        if not self.controller.utility_stage_enabled:
            return
        action_name = {0: "adapt", 1: "retrieve", 2: "write", 3: "split"}[
            int(action_index)
        ]
        if (
            self.training_config.controller_only_finetune
            and action_name not in self.training_config.controller_train_heads
        ):
            return
        values = utility.new_zeros(4)
        targets = utility.new_zeros(4)
        mask = torch.zeros(4, device=utility.device, dtype=torch.bool)
        propensities = utility.new_ones(4)
        values[action_index] = utility.detach()
        targets[action_index] = self.controller.utility_target(
            utility.detach(), action_index=action_index, cost_margin=0.0,
            update_statistics=True,
        )
        mask[action_index] = True
        propensities[action_index] = float(propensity)
        source = sequence.get("source_index", -1)
        cluster = sequence.get("cluster_id", -1)
        row = {
            "inputs": self._controller_input_tensor(request["controller_inputs"]),
            "utility": values,
            "target": targets,
            "label_mask": mask,
            "propensity": propensities,
            "gate": request["action_probabilities"].detach(),
            "cluster_id": int(torch.as_tensor(cluster).detach().cpu()),
            "source_index": int(torch.as_tensor(source).detach().cpu()),
            "event_index": int(request["event_index"]),
            "owner_id": str(request.get("provisional_owner_id", "root")),
        }
        if int(action_index) == 2 and self.training_config.controller_write_ranking:
            row.update({
                "group_id": int(torch.as_tensor(source).detach().cpu()),
                "raw_write_utility": float(utility.detach().cpu()),
                "probe_propensity": float(propensity),
                "probe_top": not bool(request.get("exploration", False)),
            })
        self.controller_utility_replay.add(row, action_index)

    def _add_controller_utility_batch(
        self,
        sequences: Sequence[Mapping[str, Tensor]],
        requests: Sequence[Mapping[str, Any]],
        packed: Mapping[str, Tensor],
        *,
        action_index: int,
        utility: Tensor,
        propensities: Optional[Tensor] = None,
    ) -> None:
        """Record one delayed-utility segment with one device-to-host copy.

        The utility target and all replay fields are formed on the device. A
        single packed payload crosses to CPU; reservoir/hard-example updates
        then run in request order so their RNG and replacement semantics stay
        identical to the scalar path. Running utility moments are updated in
        that same order on CPU and copied back once at the end.
        """
        if not self.controller.utility_stage_enabled:
            return
        action_index = int(action_index)
        action_name = {0: "adapt", 1: "retrieve", 2: "write", 3: "split"}[
            action_index
        ]
        if (
            self.training_config.controller_only_finetune
            and action_name not in self.training_config.controller_train_heads
        ):
            return
        if utility.ndim != 1 or utility.numel() != len(requests):
            raise ValueError("batched utility must align with requests as [Q]")
        if not requests:
            return

        reference = utility.detach()
        device = reference.device
        batch_size = len(requests)
        controller_inputs = packed["controller_inputs"]
        gates = packed["action_probabilities"]
        if controller_inputs.shape != (batch_size, 8):
            raise ValueError("packed controller inputs must have shape [Q, 8]")
        if gates.shape != (batch_size, 4):
            raise ValueError("packed action probabilities must have shape [Q, 4]")

        utility_values = reference
        # ``utility_target`` is already vectorized and the scalar path passes
        # cost_margin=0.0. Do not update moments here: that is replayed below
        # in the original request order on CPU.
        target_values = self.controller.utility_target(
            utility_values,
            action_index=action_index,
            cost_margin=0.0,
            update_statistics=False,
        ).detach()
        values = utility_values.new_zeros(batch_size, 4)
        values[:, action_index] = utility_values
        targets = utility_values.new_zeros(batch_size, 4)
        targets[:, action_index] = target_values
        label_mask = torch.zeros(
            batch_size, 4, device=device, dtype=torch.bool
        )
        label_mask[:, action_index] = True
        packed_exploration = packed.get("exploration")
        if torch.is_tensor(packed_exploration):
            exploration_values = packed_exploration.to(
                device=device, dtype=torch.bool
            ).reshape(-1)
        else:
            exploration_values = torch.tensor(
                [bool(request.get("exploration", False)) for request in requests],
                device=device,
                dtype=torch.bool,
            )
        if exploration_values.numel() != batch_size:
            raise ValueError("exploration flags must align with requests")
        if propensities is None:
            propensity_values = torch.where(
                exploration_values,
                utility_values.new_full((), self.controller.exploration_rate),
                utility_values.new_ones(()),
            )
        else:
            propensity_values = torch.as_tensor(
                propensities, device=device, dtype=utility_values.dtype
            ).reshape(-1)
            if propensity_values.numel() == 1 and batch_size != 1:
                propensity_values = propensity_values.expand(batch_size)
            if propensity_values.numel() != batch_size:
                raise ValueError("propensities must align with requests")
        all_propensities = utility_values.new_ones(batch_size, 4)
        all_propensities[:, action_index] = propensity_values

        # Preserve the highest-precision replay field while ensuring that the
        # packed observation-count snapshot remains exact at ordinary sizes.
        cpu_dtype = self.controller.utility_mean.dtype
        payload_dtype = torch.promote_types(
            torch.promote_types(utility_values.dtype, controller_inputs.dtype),
            gates.dtype,
        )
        payload_dtype = torch.promote_types(payload_dtype, cpu_dtype)
        payload = torch.cat(
            (
                controller_inputs.detach(),
                values,
                targets,
                label_mask.to(values.dtype),
                all_propensities,
                gates.detach(),
                exploration_values[:, None].to(values.dtype),
            ),
            dim=-1,
        ).to(dtype=payload_dtype)
        moment_snapshot = torch.stack((
            self.controller.utility_mean.detach().to(
                device=device, dtype=payload_dtype
            ),
            self.controller.utility_variance.detach().to(
                device=device, dtype=payload_dtype
            ),
            self.controller.utility_observations.detach().to(
                device=device, dtype=payload_dtype
            ),
        ))
        packed_cpu = torch.cat(
            (payload.reshape(-1), moment_snapshot.reshape(-1))
        ).cpu()
        payload_cpu = packed_cpu[: payload.numel()].reshape(payload.shape)
        moments_cpu = packed_cpu[payload.numel():].reshape(3, 4).to(cpu_dtype)

        input_start = 0
        utility_start = 8
        target_start = 12
        mask_start = 16
        propensity_start = 20
        gate_start = 24
        exploration_start = 28
        sequence_rows = [
            int(request.get("_sequence_row", 0)) for request in requests
        ]
        exploration_cpu = payload_cpu[:, exploration_start].to(
            torch.bool
        ).tolist()
        metadata = {}
        for sequence_row in set(sequence_rows):
            sequence = sequences[sequence_row]
            metadata[sequence_row] = (
                self._host_int(sequence.get("cluster_id", -1)),
                self._host_int(sequence.get("source_index", -1)),
            )

        for row_index, request in enumerate(requests):
            sequence_row = sequence_rows[row_index]
            cluster_id, source_index = metadata[sequence_row]

            # Match utility_target(update_statistics=True): online moments are
            # intentionally sequential even though target calculation was
            # batched above.
            observed = payload_cpu[row_index, utility_start + action_index]
            count = int(moments_cpu[2, action_index].item())
            delta = observed - moments_cpu[0, action_index]
            rate = 1.0 / float(count + 1)
            moments_cpu[0, action_index].add_(delta * rate)
            moments_cpu[1, action_index].add_(
                delta.square() - moments_cpu[1, action_index],
                alpha=rate,
            )
            moments_cpu[2, action_index].add_(1.0)

            request["exploration"] = bool(exploration_cpu[row_index])
            row = {
                "inputs": payload_cpu[
                    row_index, input_start:utility_start
                ],
                "utility": payload_cpu[
                    row_index, utility_start:target_start
                ],
                "target": payload_cpu[
                    row_index, target_start:mask_start
                ],
                "label_mask": payload_cpu[
                    row_index, mask_start:propensity_start
                ].to(torch.bool),
                "propensity": payload_cpu[
                    row_index, propensity_start:gate_start
                ],
                "gate": payload_cpu[row_index, gate_start:gate_start + 4],
                "cluster_id": cluster_id,
                "source_index": source_index,
                "event_index": int(request["event_index"]),
                "owner_id": str(request.get("provisional_owner_id", "root")),
            }
            if action_index == 2 and self.training_config.controller_write_ranking:
                row.update({
                    "group_id": source_index,
                    "raw_write_utility": float(
                        payload_cpu[row_index, utility_start + action_index]
                    ),
                    "probe_propensity": float(
                        payload_cpu[row_index, propensity_start + action_index]
                    ),
                    "probe_top": not request["exploration"],
                })
            self.controller_utility_replay.add(row, action_index)

        # The online moments are state, not graph values. One write-back keeps
        # the GPU buffers authoritative for the next Wake transaction.
        with torch.no_grad():
            self.controller.utility_mean.copy_(
                moments_cpu[0].to(
                    device=self.controller.utility_mean.device,
                    dtype=self.controller.utility_mean.dtype,
                )
            )
            self.controller.utility_variance.copy_(
                moments_cpu[1].to(
                    device=self.controller.utility_variance.device,
                    dtype=self.controller.utility_variance.dtype,
                )
            )
            self.controller.utility_observations.copy_(
                moments_cpu[2].round().to(
                    device=self.controller.utility_observations.device,
                    dtype=self.controller.utility_observations.dtype,
                )
            )

    def _record_adapt_utility(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
    ) -> Tensor:
        """Delayed Adapt probe independent of Write candidacy."""
        start = int(request["event_index"])
        end = min(
            int(sequence["times"].numel()),
            start + self.wake_config.write_horizon + 1,
        )
        theta_before = request["assimilation_theta"]
        theta_after = theta_before - (
            self.tree.working_memory.eta * request["assimilation_grad"]
        )
        D = self.hawkes.num_types
        before_params = HawkesParams(
            theta_before[:D],
            theta_before[D:].reshape(D, D, self.hawkes.num_basis),
        )
        after_params = HawkesParams(
            theta_after[:D],
            theta_after[D:].reshape(D, D, self.hawkes.num_basis),
        )
        no_adapt = theta_before.new_zeros(())
        full_adapt = theta_before.new_zeros(())
        for future_index in range(start + 1, end):
            no_adapt += self.hawkes.event_NLL(sequence, before_params, future_index)
            full_adapt += self.hawkes.event_NLL(sequence, after_params, future_index)
        utility = (
            (no_adapt - full_adapt) / max(end - start - 1, 1)
            - self.wake_config.controller_adapt_cost
        )
        self._add_controller_utility(
            sequence, request, action_index=0, utility=utility,
            propensity=(
                self.controller.exploration_rate
                if request.get("exploration", False) else 1.0
            ),
        )
        return utility

    def _select_adapt_probes(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        if self.training_config.controller_only_finetune:
            requests = [
                row for row in requests
                if int(row["event_index"]) + self.wake_config.write_horizon
                < len(row.get("future_contexts") or ())
            ]
            ranked = sorted(
                requests,
                key=lambda row: float(row["action_probabilities"][0]),
                reverse=True,
            )
            top = ranked[: self.wake_config.controller_adapt_probe_topc]
            for row in top:
                row["exploration"] = False
            exploration = [
                row for row in ranked[self.wake_config.controller_adapt_probe_topc :]
                if row.get("exploration", False)
            ]
            return [*top, *exploration]
        exploitation = sorted(
            (row for row in requests if not row.get("exploration", False)),
            key=lambda row: float(row["action_probabilities"][0]),
            reverse=True,
        )[: self.wake_config.controller_adapt_probe_topc]
        exploration = [row for row in requests if row.get("exploration", False)]
        return [*exploitation, *exploration]

    def _preselect_write_requests(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        """Cheap top-C selection plus unbiased exploration candidates."""
        if not requests:
            return []
        top_c = self.wake_config.controller_write_probe_topc
        if self.training_config.controller_only_finetune:
            ranked = sorted(
                requests,
                key=lambda request: float(
                    request["write_gate"].detach().cpu()
                ),
                reverse=True,
            )
            top = ranked[:top_c]
            for request in top:
                request["exploration"] = False
                request["probe_propensity"] = 1.0
            remaining = ranked[top_c:]
            if int(self.controller.controller_version.detach().cpu()) >= 6:
                generator = torch.Generator(device="cpu")
                signature = sum(int(row["event_index"]) for row in requests)
                generator.manual_seed(self.training_config.seed + signature)
                count = min(16, len(remaining))
                indices = (
                    torch.randperm(len(remaining), generator=generator)[:count].tolist()
                    if count else []
                )
                explored = [remaining[index] for index in indices]
                propensity = count / max(len(remaining), 1)
                for request in explored:
                    request["exploration"] = True
                    request["probe_propensity"] = propensity
            else:
                explored = [
                    request for request in remaining
                    if request.get("exploration", False)
                ]
            return [*top, *explored]
        exploitation = [r for r in requests if not r.get("exploration", False)]
        exploration = [r for r in requests if r.get("exploration", False)]
        exploitation = sorted(
            exploitation,
            key=lambda request: float(
                (request["write_gate"] * request["novelty"]).detach().cpu()
            ),
            reverse=True,
        )[:top_c]
        return [*exploitation, *exploration]

    def _padded_wake_sequences(
        self,
        sequences: Sequence[Mapping[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Pack sequence-local Hawkes caches as ``[B, T, ...]`` tensors."""
        if not sequences:
            raise ValueError("Wake sequence batches cannot be empty")
        lengths = torch.tensor(
            [int(sequence["times"].numel()) for sequence in sequences],
            device=self.device,
            dtype=torch.long,
        )
        if torch.any(lengths <= 0):
            raise ValueError("Wake sequence batches cannot contain empty rows")
        times = nn.utils.rnn.pad_sequence(
            [sequence["times"] for sequence in sequences],
            batch_first=True,
        )
        types = nn.utils.rnn.pad_sequence(
            [sequence["types"].long() for sequence in sequences],
            batch_first=True,
        )
        history = nn.utils.rnn.pad_sequence(
            [sequence[HAWKES_HISTORY_STATS_KEY] for sequence in sequences],
            batch_first=True,
        )
        interval = nn.utils.rnn.pad_sequence(
            [sequence[HAWKES_INTERVAL_STATS_KEY] for sequence in sequences],
            batch_first=True,
        )
        features = nn.utils.rnn.pad_sequence(
            [sequence[EVENT_TIME_FEATURES_KEY] for sequence in sequences],
            batch_first=True,
        )
        valid = (
            torch.arange(times.size(1), device=self.device)[None, :]
            < lengths[:, None]
        )
        previous = torch.cat(
            [times.new_zeros(times.size(0), 1), times[:, :-1]], dim=1
        )
        duration = (times - previous).clamp_min(0.0)
        return {
            "times": times,
            "types": types,
            EVENT_TIME_FEATURES_KEY: features,
            HAWKES_HISTORY_STATS_KEY: history,
            HAWKES_INTERVAL_STATS_KEY: interval,
            HAWKES_CACHE_SIGNATURE_KEY: self.hawkes.cache_signature,
            "duration": duration,
            "valid": valid,
            "lengths": lengths,
        }

    def _pack_probe_requests(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        frontier_width: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Pack request snapshots without materializing future probe objects."""
        if not requests:
            return {}
        reference = requests[0]["query"]
        device = reference.device
        counts = [len(request["frontier_node_ids"]) for request in requests]
        width = max(counts) if frontier_width is None else int(frontier_width)
        width = max(width, 1)
        param_dim = int(requests[0]["frontier_theta"].size(-1))

        def padded_values(name: str, tail: tuple[int, ...]) -> Tensor:
            result = reference.new_zeros((len(requests), width, *tail))
            for row, request in enumerate(requests):
                value = request.get(name)
                if value is None:
                    continue
                value = torch.as_tensor(value, device=device, dtype=reference.dtype)
                result[row, : counts[row]] = value.reshape(counts[row], *tail)
            return result

        frontier_mass = padded_values("frontier_mass", ())
        frontier_theta = padded_values("frontier_theta", (param_dim,))
        node_indices = torch.full(
            (len(requests), width), -1, device=device, dtype=torch.long
        )
        all_node_ids = tuple(self.tree.all_node_ids)
        node_lookup = {node_id: index for index, node_id in enumerate(all_node_ids)}
        for row, request in enumerate(requests):
            values = request.get("frontier_node_indices")
            if values is None:
                values = torch.tensor(
                    [node_lookup[node_id] for node_id in request["frontier_node_ids"]],
                    device=device,
                    dtype=torch.long,
                )
            else:
                values = torch.as_tensor(values, device=device, dtype=torch.long)
            node_indices[row, : counts[row]] = values.reshape(-1)

        def scalar_field(name: str, *, dtype: Optional[torch.dtype] = None) -> Tensor:
            values = [
                torch.as_tensor(
                    request[name],
                    device=device,
                    dtype=dtype,
                ).reshape(())
                for request in requests
            ]
            return torch.stack(values)

        def optional_matrix(name: str) -> Optional[Tensor]:
            values = [request.get(name) for request in requests]
            if any(value is None for value in values):
                return None
            return torch.stack([
                torch.as_tensor(value, device=device, dtype=reference.dtype)
                for value in values
            ])

        controller_inputs = torch.stack([
            self._controller_input_tensor(request["controller_inputs"])
            for request in requests
        ])
        exploration = torch.stack([
            torch.as_tensor(
                request.get("exploration", False),
                device=device,
                dtype=torch.bool,
            ).reshape(())
            for request in requests
        ])
        action_probabilities = torch.stack([
            torch.as_tensor(
                request.get(
                    "raw_action_probabilities",
                    request["action_probabilities"],
                ),
                device=device,
                dtype=reference.dtype,
            ).reshape(4)
            for request in requests
        ])
        return {
            "sequence_rows": torch.tensor(
                [int(request.get("_sequence_row", 0)) for request in requests],
                device=device,
                dtype=torch.long,
            ),
            "event_indices": torch.tensor(
                [int(request["event_index"]) for request in requests],
                device=device,
                dtype=torch.long,
            ),
            "queries": torch.stack([
                torch.as_tensor(request["query"], device=device, dtype=reference.dtype)
                for request in requests
            ]),
            "frontier_node_indices": node_indices,
            "frontier_mask": (
                torch.arange(width, device=device)[None, :]
                < torch.tensor(counts, device=device)[:, None]
            ),
            "frontier_mass": frontier_mass,
            "frontier_theta": frontier_theta,
            "action_probabilities": action_probabilities,
            "write_gate": scalar_field("write_gate"),
            "novelty": scalar_field("novelty"),
            "queue_weight": scalar_field("queue_weight"),
            "exploration": exploration,
            "controller_inputs": controller_inputs,
            "assimilation_theta": optional_matrix("assimilation_theta"),
            "assimilation_grad": optional_matrix("assimilation_grad"),
        }

    def _select_probe_requests_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        topc: int,
        score: str,
        sequence_count: Optional[int] = None,
    ) -> tuple[list[Mapping[str, Any]], Dict[str, Tensor]]:
        """Select top-C plus exploration rows with device-side segmented sort."""
        if not requests:
            return [], {}
        packed = self._pack_probe_requests(requests)
        values = (
            packed["action_probabilities"][:, 0]
            if score == "adapt"
            else packed["write_gate"] * packed["novelty"]
        )
        selected = packed["exploration"].clone()
        sequence_rows = packed["sequence_rows"]
        row_count = (
            int(sequence_count)
            if sequence_count is not None
            else int(sequence_rows.max().detach().cpu()) + 1
        )
        for sequence_row in range(row_count):
            candidates = torch.nonzero(
                (sequence_rows == sequence_row) & ~packed["exploration"],
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                continue
            order = torch.argsort(
                values.index_select(0, candidates),
                descending=True,
                stable=True,
            )
            selected[candidates.index_select(0, order[: int(topc)])] = True
        selected_indices = torch.nonzero(selected, as_tuple=False).flatten()
        selected_cpu = selected_indices.detach().cpu().tolist()
        selected_requests = [requests[index] for index in selected_cpu]
        if not selected_requests:
            return [], {}
        return selected_requests, self._pack_probe_requests(selected_requests)

    def _batched_window_event_nll(
        self,
        theta: Tensor,
        starts: Tensor,
        sequence_rows: Tensor,
        padded: Mapping[str, Tensor],
        offsets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate independent raw Hawkes parameters on padded future windows."""
        D = self.hawkes.num_types
        M = self.hawkes.num_basis
        times = padded["times"]
        types = padded["types"]
        history_cache = padded[HAWKES_HISTORY_STATS_KEY]
        interval_cache = padded[HAWKES_INTERVAL_STATS_KEY]
        lengths = padded["lengths"]
        grid = starts[:, None] + offsets[None, :]
        row_lengths = lengths.index_select(0, sequence_rows)
        valid = grid < row_lengths[:, None]
        safe = grid.clamp_max(times.size(1) - 1)
        rows = sequence_rows[:, None].expand(-1, offsets.numel())
        history = history_cache[rows, safe]
        interval = interval_cache[rows, safe]
        event_types = types[rows, safe]
        durations = padded["duration"][rows, safe]
        mu = F.softplus(theta[:, :D])
        W = F.softplus(theta[:, D:].reshape(-1, D, D, M))
        intensity = (
            mu[:, None, :]
            + torch.einsum("qdem,qhem->qhd", W, history)
        ).clamp_min(1e-8)
        selected = intensity.gather(2, event_types.unsqueeze(-1)).squeeze(-1)
        loss = (
            -selected.log()
            + mu.sum(dim=-1, keepdim=True) * durations
            + torch.einsum("qdem,qhem->qh", W, interval)
        ).masked_fill(~valid, 0.0)
        return loss, valid

    def _record_adapt_utility_batch(
        self,
        sequences: Sequence[Mapping[str, Tensor]],
        requests: Sequence[Mapping[str, Any]],
        packed: Mapping[str, Tensor],
        padded: Mapping[str, Tensor],
    ) -> Tensor:
        """Evaluate all selected Adapt windows in one GPU operation."""
        if not requests:
            return padded["times"].new_empty(0)
        theta_before = packed["assimilation_theta"]
        theta_after = theta_before - (
            self.tree.working_memory.eta * packed["assimilation_grad"]
        )
        offsets = torch.arange(
            1,
            self.wake_config.write_horizon + 1,
            device=padded["times"].device,
            dtype=torch.long,
        )
        before, valid = self._batched_window_event_nll(
            theta_before,
            packed["event_indices"],
            packed["sequence_rows"],
            padded,
            offsets,
        )
        after, _ = self._batched_window_event_nll(
            theta_after,
            packed["event_indices"],
            packed["sequence_rows"],
            padded,
            offsets,
        )
        count = valid.sum(dim=-1).clamp_min(1).to(before)
        utility = (before - after).sum(dim=-1) / count
        utility = utility - self.wake_config.controller_adapt_cost
        propensities = torch.where(
            packed["exploration"],
            utility.new_full((), self.controller.exploration_rate),
            utility.new_ones(()),
        )
        self._add_controller_utility_batch(
            sequences,
            requests,
            packed,
            action_index=0,
            utility=utility,
            propensities=propensities,
        )
        return utility

    def _window_write_evidence_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
        packed: Mapping[str, Tensor],
        padded: Mapping[str, Tensor],
        semantic_theta_table: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute v4 Write posterior, residuals, utility and priority for Q rows."""
        if not requests:
            return {}
        D = self.hawkes.num_types
        M = self.hawkes.num_basis
        Q, K = packed["frontier_mass"].shape
        horizon = self.wake_config.write_horizon + 1
        offsets = torch.arange(
            horizon,
            device=padded["times"].device,
            dtype=torch.long,
        )
        frontier_theta = packed["frontier_theta"].reshape(Q * K, -1)
        candidate_rows = packed["sequence_rows"][:, None].expand(-1, K).reshape(-1)
        candidate_starts = packed["event_indices"][:, None].expand(-1, K).reshape(-1)
        frontier_losses, valid = self._batched_window_event_nll(
            frontier_theta,
            candidate_starts,
            candidate_rows,
            padded,
            offsets,
        )
        frontier_losses = frontier_losses.reshape(Q, K, horizon)
        valid_count = valid.reshape(Q, K, horizon)[:, 0].sum(dim=-1).clamp_min(1)
        energy = frontier_losses.sum(dim=-1) / valid_count[:, None].to(frontier_losses)
        posterior = self._frontier_posterior(
            packed["frontier_mass"],
            energy,
            packed["frontier_mask"],
        )
        owner_indices, owner_is_lca, confidence = self._posterior_owner_indices_batch(
            packed["frontier_node_indices"], posterior
        )
        owner_theta = semantic_theta_table.index_select(
            0, owner_indices.clamp_min(0)
        )
        owner_params = HawkesParams(
            owner_theta[:, :D],
            owner_theta[:, D:].reshape(Q, D, D, M),
        )
        candidate_delta, _ = self.controller._residual_delta_batch(
            owner_params,
            padded["times"],
            padded["types"],
            packed["event_indices"],
            cached_sequence=padded,
            sequence_rows=packed["sequence_rows"],
            sequence_lengths=padded["lengths"],
            window_events=horizon,
        )
        before, valid = self._batched_window_event_nll(
            owner_theta,
            packed["event_indices"],
            packed["sequence_rows"],
            padded,
            offsets,
        )
        after, _ = self._batched_window_event_nll(
            owner_theta + candidate_delta,
            packed["event_indices"],
            packed["sequence_rows"],
            padded,
            offsets,
        )
        valid_count = valid.sum(dim=-1).clamp_min(1).to(before)
        raw_improvement = (before - after).sum(dim=-1)
        utility = (
            raw_improvement / valid_count
            - self.wake_config.lambda_write
        )
        bounded_gain = -torch.expm1(
            -raw_improvement.clamp_min(0.0)
            / self.wake_config.controller_gain_reference
        )
        priority = (
            packed["write_gate"]
            * confidence
            * utility.clamp_min(0.0)
            * packed["novelty"].clamp(0.0, 1.0)
        )
        return {
            "posterior": posterior,
            "owner_indices": owner_indices,
            "owner_is_lca": owner_is_lca,
            "confidence": confidence,
            "candidate_delta": candidate_delta,
            "write_gain": raw_improvement,
            "write_utility": utility,
            "bounded_gain": bounded_gain,
            "priority": priority,
        }

    def _update_structural_evidence_buffer(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        accepted_tokens: Sequence[tuple[int, int]] = (),
    ) -> None:
        """Discard non-persistent structural probes.

        Kept as a checkpoint/call-site compatibility boundary.  Probation or
        rejected candidates must never become Sleep/Split evidence; only rows
        promoted into ``EpisodicMemory`` may carry structural weight.
        """
        del records, accepted_tokens
        self.sleep_state["structural_evidence_buffer"] = {}

    def _finalize_write_probe_batch(
        self,
        sequences: Sequence[Mapping[str, Tensor]],
        pending_writes: Sequence[Sequence[Mapping[str, Any]]],
        lengths: Sequence[int],
        padded: Mapping[str, Tensor],
        semantic_theta_table: Tensor,
        controller_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Batch v4 Write evidence and commit only the segmented top-4 rows."""
        batch_size = len(sequences)
        version = (
            int(controller_version)
            if controller_version is not None
            else int(self.controller.controller_version.detach().cpu())
        )

        def finalize_write_groups() -> None:
            if not self.training_config.controller_write_ranking:
                return
            for sequence in sequences:
                self.controller_utility_replay.finalize_write_group(
                    int(
                        torch.as_tensor(sequence.get("source_index", -1))
                        .detach()
                        .cpu()
                    )
                )

        eligible = [
            request
            for row, requests in enumerate(pending_writes)
            for request in requests
            if int(request["ready_index"]) < int(lengths[row])
        ]
        incomplete_counts = [
            sum(
                int(request["ready_index"]) >= int(lengths[row])
                for request in requests
            )
            for row, requests in enumerate(pending_writes)
        ]
        if not eligible:
            finalize_write_groups()
            return {
                "write_counts": [0] * batch_size,
                "write_probe_counts": [0] * batch_size,
                "write_gate_pass_counts": [0] * batch_size,
                "write_utility_pass_counts": [0] * batch_size,
                "accepted_write_utility_sums": [0.0] * batch_size,
                "harmful_write_counts": [0] * batch_size,
                "pending_counts": incomplete_counts,
            }
        probe_requests, packed = self._select_probe_requests_batch(
            eligible,
            topc=self.wake_config.controller_write_probe_topc,
            score="write",
            sequence_count=batch_size,
        )
        if not probe_requests:
            finalize_write_groups()
            return {
                "write_counts": [0] * batch_size,
                "write_probe_counts": [0] * batch_size,
                "write_gate_pass_counts": [0] * batch_size,
                "write_utility_pass_counts": [0] * batch_size,
                "accepted_write_utility_sums": [0.0] * batch_size,
                "harmful_write_counts": [0] * batch_size,
                "pending_counts": incomplete_counts,
            }
        evidence = self._window_write_evidence_batch(
            probe_requests,
            packed,
            padded,
            semantic_theta_table,
        )
        sequence_rows = packed["sequence_rows"]
        gate_threshold = self.controller.calibration_thresholds[2].to(
            packed["write_gate"]
        )
        if version < 5:
            gate_threshold = torch.maximum(
                gate_threshold,
                packed["write_gate"].new_tensor(
                    self.controller.write_candidate_threshold
                ),
            )
        gate_pass = packed["write_gate"] >= gate_threshold
        utility_pass = evidence["write_utility"] > 0.0
        admissible = (
            gate_pass
            & utility_pass
            & (
                evidence["priority"]
                > float(self.wake_config.controller_priority_threshold)
            )
        )
        selected = torch.zeros_like(admissible)
        max_writes = min(
            4,
            int(self.tree.frontier_routing.config.max_writes_per_sequence),
        )
        for sequence_row in range(batch_size):
            candidates = torch.nonzero(
                admissible & (sequence_rows == sequence_row),
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                continue
            order = torch.argsort(
                evidence["priority"].index_select(0, candidates),
                descending=True,
                stable=True,
            )
            selected[candidates.index_select(0, order[:max_writes])] = True

        shadow_priority = (
            packed["queue_weight"]
            * evidence["confidence"]
            * evidence["bounded_gain"]
        )
        shadow_selected = torch.zeros_like(selected)
        for sequence_row in range(batch_size):
            candidates = torch.nonzero(
                (~selected)
                & (sequence_rows == sequence_row)
                & (shadow_priority > 0.0),
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                continue
            order = torch.argsort(
                shadow_priority.index_select(0, candidates),
                descending=True,
                stable=True,
            )
            shadow_selected[
                candidates.index_select(0, order[:max_writes])
            ] = True

        # Physical v4/v5 Write evidence is not a utility-replay label in the
        # existing full-training contract; only Adapt is replayed on this
        # path. Keep the host boundary below for final ownership/commit
        # metadata, without introducing a new Write training signal.
        finalize_write_groups()

        selected_indices = torch.nonzero(selected, as_tuple=False).flatten()
        selected_cpu = selected_indices.detach().cpu().tolist()
        owner_indices_cpu = evidence["owner_indices"].index_select(
            0, selected_indices
        ).detach().cpu().tolist()
        owner_ids = [self.tree.all_node_ids[int(index)] for index in owner_indices_cpu]
        selected_items = []
        if selected_cpu:
            selected_items = self.controller.materialize_residual_memory_items_batch(
                queries=packed["queries"].index_select(0, selected_indices),
                delta_theta=evidence["candidate_delta"].index_select(
                    0, selected_indices
                ),
                times=padded["times"],
                types=padded["types"],
                event_indices=packed["event_indices"].index_select(
                    0, selected_indices
                ),
                node_ids=owner_ids,
                cached_sequence=padded,
                write_quality=packed["write_gate"].index_select(
                    0, selected_indices
                ),
                queue_weight=packed["queue_weight"].index_select(
                    0, selected_indices
                ),
                sequence_rows=packed["sequence_rows"].index_select(
                    0, selected_indices
                ),
                sequence_lengths=padded["lengths"],
                window_events=self.wake_config.write_horizon + 1,
            )
        shadow_indices = torch.nonzero(
            shadow_selected, as_tuple=False
        ).flatten()
        shadow_cpu = shadow_indices.detach().cpu().tolist()
        shadow_owner_indices = evidence["owner_indices"].index_select(
            0, shadow_indices
        ).detach().cpu().tolist()
        shadow_owner_ids = [
            self.tree.all_node_ids[int(index)]
            for index in shadow_owner_indices
        ]
        shadow_items = []
        if shadow_cpu:
            shadow_items = (
                self.controller.materialize_residual_memory_items_batch(
                    queries=packed["queries"].index_select(
                        0, shadow_indices
                    ),
                    delta_theta=evidence["candidate_delta"].index_select(
                        0, shadow_indices
                    ),
                    times=padded["times"],
                    types=padded["types"],
                    event_indices=packed["event_indices"].index_select(
                        0, shadow_indices
                    ),
                    node_ids=shadow_owner_ids,
                    cached_sequence=padded,
                    write_quality=evidence["bounded_gain"].index_select(
                        0, shadow_indices
                    ),
                    queue_weight=(
                        packed["queue_weight"] * evidence["confidence"]
                    ).index_select(0, shadow_indices),
                    sequence_rows=packed["sequence_rows"].index_select(
                        0, shadow_indices
                    ),
                    sequence_lengths=padded["lengths"],
                    window_events=self.wake_config.write_horizon + 1,
                )
            )
        # ``selected`` is already sequence-major because requests were packed
        # in sequence/event order. Commit in that same deterministic order.
        for position, (probe_index, item, owner_id) in enumerate(
            zip(selected_cpu, selected_items, owner_ids)
        ):
            item.write_quality = float(
                evidence["bounded_gain"][probe_index].detach().cpu()
            )
            item.queue_weight = float(
                packed["queue_weight"][probe_index].detach().cpu()
            )
            self.tree.episodic_memory.add_memory(
                node_id=owner_id,
                key=item.key,
                delta_theta=item.delta_theta,
                window=item.window,
                write_quality=item.write_quality,
                queue_weight=item.queue_weight,
            )
            self.controller.split_queues[owner_id] += item.queue_weight

        def request_token(probe_index: int) -> tuple[int, int]:
            sequence_row = int(sequence_rows[probe_index].detach().cpu())
            source_index = int(torch.as_tensor(
                sequences[sequence_row].get("source_index", sequence_row)
            ).detach().cpu())
            return (
                source_index,
                int(packed["event_indices"][probe_index].detach().cpu()),
            )

        accepted_tokens = [request_token(index) for index in selected_cpu]
        shadow_records = []
        for probe_index, item, owner_id in zip(
            shadow_cpu, shadow_items, shadow_owner_ids
        ):
            item.write_quality = float(
                evidence["bounded_gain"][probe_index].detach().cpu()
            )
            item.queue_weight = float((
                packed["queue_weight"][probe_index]
                * evidence["confidence"][probe_index]
            ).detach().cpu())
            shadow_records.append({
                "token": request_token(probe_index),
                "owner_id": owner_id,
                "priority": float(
                    shadow_priority[probe_index].detach().cpu()
                ),
                "item": item,
            })
        self._update_structural_evidence_buffer(
            shadow_records,
            accepted_tokens=accepted_tokens,
        )

        def per_sequence_count(mask: Tensor) -> list[int]:
            return torch.bincount(
                sequence_rows,
                weights=mask.to(packed["write_gate"].dtype),
                minlength=batch_size,
            ).detach().cpu().to(torch.long).tolist()

        write_probe_counts = torch.bincount(
            sequence_rows, minlength=batch_size
        ).detach().cpu().tolist()
        accepted_counts = per_sequence_count(selected)
        accepted_utility = torch.zeros(
            batch_size,
            device=evidence["write_utility"].device,
            dtype=evidence["write_utility"].dtype,
        )
        accepted_utility.index_add_(
            0,
            sequence_rows,
            evidence["write_utility"] * selected.to(evidence["write_utility"]),
        )
        harmful = per_sequence_count(
            selected & (evidence["write_utility"] <= 0.0)
        )
        return {
            "write_counts": accepted_counts,
            "write_probe_counts": [int(value) for value in write_probe_counts],
            "write_gate_pass_counts": per_sequence_count(gate_pass),
            "write_utility_pass_counts": per_sequence_count(utility_pass),
            "accepted_write_utility_sums": accepted_utility.detach().cpu().tolist(),
            "harmful_write_counts": harmful,
            "pending_counts": incomplete_counts,
        }

    def _window_write_evidence(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Compute delayed posterior and signed, event-normalized utility."""
        if int(self.controller.controller_version.detach().cpu()) >= 6:
            return self._window_write_evidence_v6(sequence, request)
        frontier_ids = tuple(request["frontier_node_ids"])
        frontier_theta = request["frontier_theta"]
        D = self.hawkes.num_types
        start = int(request["event_index"])
        end = min(
            int(sequence["times"].numel()),
            start + self.wake_config.write_horizon + 1,
        )
        energies = []
        for theta in frontier_theta:
            params = HawkesParams(
                mu_tilde=theta[:D],
                W_tilde=theta[D:].reshape(
                    D, D, self.hawkes.num_basis
                ),
            )
            energy = theta.new_zeros(())
            for event_index in range(start, end):
                energy = energy + self.hawkes.event_NLL(
                    sequence=sequence,
                    params=params,
                    k=event_index,
                )
            energies.append(energy / max(end - start, 1))
        posterior = self._frontier_posterior(
            request["frontier_mass"],
            torch.stack(energies),
            torch.ones(
                len(frontier_ids),
                dtype=torch.bool,
                device=frontier_theta.device,
            ),
        )
        owner_id, _, confidence = self._posterior_owner(
            frontier_ids, posterior
        )
        energy = torch.stack(energies)
        prior = request["frontier_mass"].clamp_min(1e-12)
        mixture_energy = -torch.logsumexp(
            prior.log()
            - energy
            / self.tree.frontier_routing.config.posterior_temperature,
            dim=0,
        )
        theta_owner = self.tree.semantic_theta(owner_id).detach()
        owner_params = HawkesParams(
            mu_tilde=theta_owner[:D].clone(),
            W_tilde=theta_owner[D:].reshape(
                D, D, self.hawkes.num_basis
            ).clone(),
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
        if not self.training_config.controller_only_finetune:
            before = frontier_theta.new_zeros(())
            after = frontier_theta.new_zeros(())
            candidate_theta = theta_owner + delta
            candidate_params = HawkesParams(
                candidate_theta[:D],
                candidate_theta[D:].reshape(D, D, self.hawkes.num_basis),
            )
            for future_index in range(start, end):
                before = before + self.hawkes.event_NLL(
                    sequence, owner_params, future_index
                )
                after = after + self.hawkes.event_NLL(
                    sequence, candidate_params, future_index
                )
            raw_improvement = before - after
            improvement = raw_improvement / max(end - start, 1) - self.wake_config.lambda_write
            bounded_gain = -torch.expm1(
                -raw_improvement.clamp_min(0.0)
                / self.wake_config.controller_gain_reference
            )
            priority = (
                request["write_gate"] * confidence
                * improvement.clamp_min(0.0)
                * request["novelty"].clamp(0.0, 1.0)
            )
            return {
                "posterior": posterior,
                "owner_id": owner_id,
                "confidence": confidence,
                "bounded_gain": bounded_gain,
                "write_gain": raw_improvement,
                "write_utility": improvement,
                "owner_on_score_path": True,
                "virtual_candidate_alpha": 0.0,
                "candidate_item": candidate_item,
                "priority": priority,
            }
        score_start = start + self.wake_config.write_horizon + 1
        score_end = start + 2 * self.wake_config.write_horizon + 1
        contexts = request.get("future_contexts") or []
        if score_end > len(contexts):
            raise RuntimeError("Write utility requested before its disjoint score window arrived")
        before = frontier_theta.new_zeros(())
        after = frontier_theta.new_zeros(())
        virtual_usage = 1.0
        virtual_alpha = []
        owner_on_path = False
        for age, context in enumerate(contexts[score_start:score_end]):
            query = context["query"]
            base_delta, _ = self.tree.episodic_memory.read_nodes(
                query, [owner_id], update_state=False
            )
            virtual_delta, virtual_info = (
                self.tree.episodic_memory.read_node_with_virtual_item(
                    query, owner_id, key=candidate_item.key,
                    delta=delta, write_quality=request["write_gate"],
                    virtual_usage=virtual_usage, virtual_age=float(age),
                )
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
                virtual_output, context["working_delta"],
                context["retrieve_gate"],
            ).select(0)
            theta = context["no_write_theta"]
            without_params = HawkesParams(
                theta[:D], theta[D:].reshape(D, D, self.hawkes.num_basis)
            )
            event_index = int(context["event_index"])
            before = before + self.hawkes.event_NLL(sequence, without_params, event_index)
            after = after + self.hawkes.event_NLL(sequence, with_params, event_index)
        raw_improvement = before - after
        if not owner_on_path:
            raw_improvement = raw_improvement * 0.0
        improvement = (
            raw_improvement / self.wake_config.write_horizon
            - self.wake_config.lambda_write
        )
        positive_improvement = raw_improvement.clamp_min(0.0)
        bounded_gain = -torch.expm1(
            -positive_improvement / self.wake_config.controller_gain_reference
        )
        priority = (
            request["write_gate"]
            * confidence
            * improvement.clamp_min(0.0)
            * request["novelty"].clamp(0.0, 1.0)
        )
        if (
            self.controller.utility_stage_enabled
            and request.get("controller_inputs")
            and not request.get("utility_recorded", False)
        ):
            assimilation_gain = improvement.new_zeros(())
            if (
                request.get("assimilation_theta") is not None
                and request.get("assimilation_grad") is not None
            ):
                theta_before = request["assimilation_theta"]
                theta_after = theta_before - (
                    self.tree.working_memory.eta
                    * request["assimilation_grad"]
                )
                before_params = HawkesParams(
                    theta_before[:D],
                    theta_before[D:].reshape(D, D, self.hawkes.num_basis),
                )
                after_params = HawkesParams(
                    theta_after[:D],
                    theta_after[D:].reshape(D, D, self.hawkes.num_basis),
                )
                no_adapt = improvement.new_zeros(())
                full_adapt = improvement.new_zeros(())
                for future_index in range(start + 1, end):
                    no_adapt += self.hawkes.event_NLL(
                        sequence, before_params, future_index
                    )
                    full_adapt += self.hawkes.event_NLL(
                        sequence, after_params, future_index
                    )
                assimilation_gain = (no_adapt - full_adapt) / max(end - start - 1, 1)
            self._add_controller_utility(
                sequence,
                request,
                action_index=2,
                utility=improvement,
                propensity=(
                    self.controller.exploration_rate
                    if request.get("exploration", False) else 1.0
                ),
            )
            request["utility_recorded"] = True
        return {
            "posterior": posterior,
            "owner_id": owner_id,
            "confidence": confidence,
            "bounded_gain": bounded_gain,
            # Keep the physical acceptance path backward compatible; signed
            # cost-adjusted utility is training-only.
            "write_gain": raw_improvement,
            "write_utility": improvement,
            "owner_on_score_path": owner_on_path,
            "virtual_candidate_alpha": (
                sum(virtual_alpha) / max(len(virtual_alpha), 1)
            ),
            "candidate_item": candidate_item,
            "priority": priority,
        }

    def _window_write_evidence_v6(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Causal v6 label: build on F=[t,t+h), score on C=[t+h,t+2h)."""
        D = self.hawkes.num_types
        h = self.wake_config.write_horizon
        start = int(request["event_index"])
        build_end = start + h
        score_start, score_end = build_end, build_end + h
        contexts = request.get("future_contexts") or []
        if score_end > len(contexts):
            raise RuntimeError("v6 Write utility requested before C window arrived")
        frontier_ids = tuple(request["frontier_node_ids"])
        energies = []
        for theta in request["frontier_theta"]:
            params = HawkesParams(
                theta[:D], theta[D:].reshape(D, D, self.hawkes.num_basis)
            )
            loss = theta.new_zeros(())
            for event_index in range(start, build_end):
                loss = loss + self.hawkes.event_NLL(sequence, params, event_index)
            energies.append(loss / h)
        posterior = self._frontier_posterior(
            request["frontier_mass"],
            torch.stack(energies),
            torch.ones(
                len(frontier_ids), dtype=torch.bool,
                device=request["frontier_theta"].device,
            ),
        )
        owner_id, _, confidence = self._posterior_owner(frontier_ids, posterior)
        theta_owner = self.tree.semantic_theta(owner_id).detach()
        owner_params = HawkesParams(
            theta_owner[:D].clone(),
            theta_owner[D:].reshape(D, D, self.hawkes.num_basis).clone(),
        )
        candidate_item = self.controller.write_residual_memory(
            q_t=request["query"], theta_sem_leaf=owner_params,
            times=sequence["times"], types=sequence["types"], k=start,
            node_id=owner_id, cached_sequence=sequence,
            write_quality=request["write_gate"],
            queue_weight=request["queue_weight"], window_events=h,
        )
        before = request["frontier_theta"].new_zeros(())
        after = before.clone()
        virtual_usage = 1.0
        virtual_alpha: list[float] = []
        owner_on_path = False
        for age, context in enumerate(contexts[score_start:score_end]):
            query = context["query"]
            base_delta, _ = self.tree.episodic_memory.read_nodes(
                query, [owner_id], update_state=False
            )
            virtual_delta, info = self.tree.episodic_memory.read_node_with_virtual_item(
                query, owner_id, key=candidate_item.key,
                delta=candidate_item.delta_theta,
                write_quality=request["write_gate"],
                virtual_usage=virtual_usage, virtual_age=float(age),
            )
            alpha = float(info["alpha"][-1].detach().cpu())
            virtual_alpha.append(alpha)
            virtual_usage += alpha
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
            before = before + self.hawkes.event_NLL(sequence, without_params, event_index)
            after = after + self.hawkes.event_NLL(sequence, with_params, event_index)
        raw_gain = before - after
        if not owner_on_path:
            raw_gain = raw_gain * 0.0
        utility = raw_gain / h - self.wake_config.lambda_write
        bounded_gain = -torch.expm1(
            -raw_gain.clamp_min(0.0) / self.wake_config.controller_gain_reference
        )
        threshold = self.controller.calibration_thresholds[2].to(request["write_gate"])
        priority = (
            (request["write_gate"] - threshold).clamp_min(0.0)
            * confidence * request["novelty"].clamp(0.0, 1.0)
        )
        if (
            self.controller.utility_stage_enabled
            and request.get("controller_inputs")
            and not request.get("utility_recorded", False)
        ):
            self._add_controller_utility(
                sequence, request, action_index=2, utility=utility,
                propensity=float(request.get(
                    "probe_propensity",
                    self.controller.exploration_rate
                    if request.get("exploration", False) else 1.0,
                )),
            )
            request["utility_recorded"] = True
        return {
            "posterior": posterior,
            "owner_id": owner_id,
            "confidence": confidence,
            "bounded_gain": bounded_gain,
            "write_gain": raw_gain,
            "write_utility": utility,
            "owner_on_score_path": owner_on_path,
            "virtual_candidate_alpha": sum(virtual_alpha) / max(len(virtual_alpha), 1),
            "candidate_item": candidate_item,
            "priority": priority,
            "construction_window": [start, build_end],
            "score_window": [score_start, score_end],
        }

    def _commit_write_request(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
    ) -> None:
        """Commit a write only after its complete future horizon is observed."""
        evidence = request.get("window_evidence")
        if evidence is None:
            evidence = self._window_write_evidence(sequence, request)
        owner_id = evidence["owner_id"]
        write_quality = float(evidence["bounded_gain"].detach().cpu())
        queue_weight = float(request["queue_weight"].detach().cpu())
        item = evidence.get("candidate_item")
        if item is None:
            raise RuntimeError("write evidence is missing its residual candidate")
        item.write_quality = write_quality
        item.queue_weight = queue_weight
        self.tree.episodic_memory.add_memory(
            node_id=owner_id,
            key=item.key,
            delta_theta=item.delta_theta,
            window=item.window,
            write_quality=item.write_quality,
            queue_weight=item.queue_weight,
        )
        self.controller.split_queues[
            owner_id
        ] += item.queue_weight

    def _commit_write_requests_batch(
        self,
        sequence: Mapping[str, Tensor],
        requests: Sequence[Mapping[str, Any]],
    ) -> None:
        """Commit selected writes with one residual-gradient evaluation."""
        if not requests:
            return
        evidence_rows = []
        owner_ids = []
        for request in requests:
            evidence = request.get("window_evidence")
            if evidence is None:
                evidence = self._window_write_evidence(sequence, request)
            evidence_rows.append(evidence)
            owner_id = evidence["owner_id"]
            owner_ids.append(owner_id)
        items = []
        for request, evidence in zip(requests, evidence_rows):
            item = evidence.get("candidate_item")
            if item is None:
                raise RuntimeError("write evidence is missing its residual candidate")
            item.write_quality = float(evidence["bounded_gain"].detach().cpu())
            item.queue_weight = float(request["queue_weight"].detach().cpu())
            items.append(item)
        for owner_id, item in zip(owner_ids, items):
            self.tree.episodic_memory.add_memory(
                node_id=owner_id,
                key=item.key,
                delta_theta=item.delta_theta,
                window=item.window,
                write_quality=item.write_quality,
                queue_weight=item.queue_weight,
            )
            self.controller.split_queues[owner_id] += item.queue_weight

    def _encode_memory_event(
        self,
        sequence: Mapping[str, Tensor],
        event_index: int,
    ) -> Tensor:
        """Encode one strict prefix for Wake and global training."""
        if isinstance(self.encoder, CausalPrefixEncoder):
            z_t = self.encoder(
                sequence["times"],
                sequence["types"],
                event_index,
                time_features=sequence.get(EVENT_TIME_FEATURES_KEY),
            ).reshape(1, -1)
        else:
            z_t = self.encoder(
                sequence["times"], sequence["types"], event_index
            ).reshape(1, -1)
        return z_t

    def _encode_memory_sequence(
        self,
        sequence: Mapping[str, Tensor],
    ) -> Tensor:
        """Encode all strict prefixes."""
        if isinstance(self.encoder, CausalPrefixEncoder):
            return self.encoder.forward_all_prefix(
                sequence["times"],
                sequence["types"],
                time_features=sequence.get(EVENT_TIME_FEATURES_KEY),
            )

        event_states = [
            self._encode_memory_event(sequence, event_index)
            for event_index in range(sequence["times"].numel())
        ]
        if not event_states:
            return torch.empty(
                0,
                self.tree.z_dim,
                device=self.device,
            )
        return torch.cat(event_states, dim=0)

    def _encode_global_sequence_batch(
        self,
        sequences: Sequence[Mapping[str, Tensor]],
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """Encode and flatten a variable-length cross-sequence minibatch.

        Flattening follows padded row-major order, which is exactly the former
        ``for sequence: for event:`` order. Sequence-local durations are
        materialized before concatenation so no integral crosses a sequence
        boundary.
        """
        if not sequences:
            raise ValueError("global sequence batches cannot be empty")
        lengths = torch.tensor(
            [int(sequence["times"].numel()) for sequence in sequences],
            device=self.device,
            dtype=torch.long,
        )
        if bool((lengths <= 0).any()):
            raise ValueError("global sequence batches cannot contain empties")

        if isinstance(self.encoder, CausalPrefixEncoder):
            times = nn.utils.rnn.pad_sequence(
                [sequence["times"] for sequence in sequences],
                batch_first=True,
            )
            types = nn.utils.rnn.pad_sequence(
                [sequence["types"] for sequence in sequences],
                batch_first=True,
            )
            time_features = nn.utils.rnn.pad_sequence(
                [
                    sequence[EVENT_TIME_FEATURES_KEY]
                    for sequence in sequences
                ],
                batch_first=True,
            )
            valid = (
                torch.arange(times.size(1), device=self.device)[None, :]
                < lengths[:, None]
            )
            padded_z, _ = self.encoder.forward_padded_prefix(
                times,
                types,
                valid,
                time_features=time_features,
            )
            z_flat = padded_z[valid]
        else:
            z_flat = torch.cat(
                [
                    self._encode_memory_sequence(sequence)
                    for sequence in sequences
                ],
                dim=0,
            )

        duration_rows = []
        for sequence in sequences:
            times = sequence["times"]
            previous = torch.cat([times.new_zeros(1), times[:-1]])
            duration_rows.append((times - previous).clamp_min(0.0))
        sequence_index = torch.repeat_interleave(
            torch.arange(
                len(sequences),
                device=self.device,
                dtype=torch.long,
            ),
            lengths,
        )
        flat = {
            "types": torch.cat(
                [sequence["types"].long() for sequence in sequences],
                dim=0,
            ),
            "duration": torch.cat(duration_rows, dim=0),
            HAWKES_HISTORY_STATS_KEY: torch.cat(
                [
                    sequence[HAWKES_HISTORY_STATS_KEY]
                    for sequence in sequences
                ],
                dim=0,
            ),
            HAWKES_INTERVAL_STATS_KEY: torch.cat(
                [
                    sequence[HAWKES_INTERVAL_STATS_KEY]
                    for sequence in sequences
                ],
                dim=0,
            ),
            "sequence_index": sequence_index,
            "sequence_lengths": lengths,
        }
        if z_flat.size(0) != sequence_index.numel():
            raise RuntimeError(
                "flattened Encoder rows do not align with sequence events"
            )
        return z_flat, flat
