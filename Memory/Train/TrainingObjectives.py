"""Cross-sequence routing and regional-probe objectives."""

from __future__ import annotations

from Train.TrainingComponents import *  # noqa: F403


class TrainingObjectivesMixin:
    @staticmethod
    def _segment_sum(
        values: Tensor,
        segment_index: Tensor,
        segment_count: int,
    ) -> Tensor:
        """Differentiable GPU segment sum implemented with ``index_add_``."""
        shape = (segment_count, *values.shape[1:])
        result = values.new_zeros(shape)
        if values.numel():
            result.index_add_(0, segment_index, values)
        return result

    @classmethod
    def _segment_mean(
        cls,
        values: Tensor,
        segment_index: Tensor,
        segment_count: int,
    ) -> tuple[Tensor, Tensor]:
        sums = cls._segment_sum(values, segment_index, segment_count)
        counts = cls._segment_sum(
            values.new_ones(values.size(0)),
            segment_index,
            segment_count,
        )
        count_shape = (segment_count,) + (1,) * (values.ndim - 1)
        means = sums / counts.clamp_min(1.0).reshape(count_shape)
        return means, counts

    def _probe_leaf_local_theta(self, leaf_id: str) -> Tensor:
        """Leaf parameters whose only trainable term is its local offset."""
        fixed = self.tree.base_semantic_theta(leaf_id).detach()
        for ancestor in self.tree.node_paths[leaf_id][:-1]:
            fixed = fixed + self.tree.semantic_offset[ancestor].detach()
        return fixed + self.tree.semantic_offset[leaf_id]

    def _probe_sequence_energy(
        self,
        flat: Mapping[str, Tensor],
        sequence_index: Tensor,
        sequence_count: int,
        raw_theta: Tensor,
    ) -> Tensor:
        """Mean event NLL for ``[sequence, candidate, theta]`` in one pass."""
        if (
            raw_theta.ndim != 3
            or raw_theta.shape[0] != sequence_count
            or raw_theta.shape[-1] != self.tree.param_dim
        ):
            raise ValueError("probe theta must have shape [S, C, P]")
        event_theta = raw_theta.index_select(0, sequence_index)
        D = self.hawkes.num_types
        M = self.hawkes.num_basis
        mu = F.softplus(event_theta[..., :D])
        W = F.softplus(
            event_theta[..., D:].reshape(
                event_theta.size(0), event_theta.size(1), D, D, M
            )
        )
        history = flat[HAWKES_HISTORY_STATS_KEY]
        interval = flat[HAWKES_INTERVAL_STATS_KEY]
        intensity = (
            mu + torch.einsum("ncdem,nem->ncd", W, history)
        ).clamp_min(1e-8)
        selected = intensity.gather(
            2,
            flat["types"][:, None, None].expand(
                -1, raw_theta.size(1), 1
            ),
        ).squeeze(2)
        event_energy = (
            -selected.log()
            + mu.sum(dim=-1) * flat["duration"][:, None]
            + torch.einsum("ncdem,nem->nc", W, interval)
        )
        energy_sum = self._segment_sum(
            event_energy, sequence_index, sequence_count
        )
        event_count = self._segment_sum(
            event_energy.new_ones(event_energy.size(0)),
            sequence_index,
            sequence_count,
        )
        return energy_sum / event_count.clamp_min(1.0)[:, None]

    @staticmethod
    def _regional_probe_leaf_count(descendant_count: int) -> int:
        if descendant_count <= 0:
            raise ValueError("descendant_count must be positive")
        return (descendant_count + 1) // 2

    def _least_probed_leaves(
        self,
        descendant_leaves: Sequence[str],
    ) -> list[str]:
        """Probe half the region, with two-round coverage for odd counts."""
        visits = self.tree.frontier_routing.probe_leaf_visits
        order = {leaf_id: index for index, leaf_id in enumerate(descendant_leaves)}
        probe_count = self._regional_probe_leaf_count(
            len(descendant_leaves)
        )
        selected = sorted(
            descendant_leaves,
            key=lambda leaf_id: (visits.get(leaf_id, 0), order[leaf_id]),
        )[:probe_count]
        # Coverage counters are Router training state. Controller-only v5 may
        # read them to preserve the fixed routing policy, but must never advance
        # them (they are serialized in frontier_routing._extra_state).
        if not self.training_config.controller_only_finetune:
            for leaf_id in selected:
                visits[leaf_id] = visits.get(leaf_id, 0) + 1
        return selected

    def _regional_probe_topology(self) -> dict[str, Any]:
        """Cache topology-only tensors used by the regional probe.

        Probe coverage is deliberately not cached: its visit counters are
        mutable training state.  Paths, node indices, and padded descendant
        slots depend only on topology and are therefore safe to reuse across
        global batches until Sleep changes the tree.
        """
        signature = (
            tuple(self.tree.all_node_ids),
            tuple(self.tree.internal_ids),
            tuple(self.tree.leaf_ids),
            tuple(
                (leaf_id, tuple(self.tree.node_paths[leaf_id]))
                for leaf_id in self.tree.leaf_ids
            ),
        )
        cached = getattr(self, "_regional_probe_topology_cache", None)
        if cached is not None and cached["signature"] == signature:
            return cached

        node_index = {
            node_id: index
            for index, node_id in enumerate(self.tree.all_node_ids)
        }
        region_ids: list[str] = []
        descendants_by_region: list[tuple[str, ...]] = []
        probe_counts: list[int] = []
        for coarse_id in self.tree.internal_ids:
            descendants = tuple(
                leaf_id
                for leaf_id in self.tree.leaf_ids
                if coarse_id in self.tree.node_paths[leaf_id]
            )
            if len(descendants) < 2:
                continue
            region_ids.append(coarse_id)
            descendants_by_region.append(descendants)
            probe_counts.append(self._regional_probe_leaf_count(len(descendants)))

        region_count = len(region_ids)
        max_descendants = max(
            (len(value) for value in descendants_by_region),
            default=0,
        )
        max_probe_count = max(probe_counts, default=0)
        max_path_length = 0
        for coarse_id, descendants in zip(region_ids, descendants_by_region):
            for leaf_id in descendants:
                path = self.tree.node_paths[leaf_id]
                max_path_length = max(
                    max_path_length,
                    len(path) - path.index(coarse_id) - 1,
                )

        descendant_nodes = torch.full(
            (region_count, max_descendants),
            -1,
            dtype=torch.long,
        )
        router_nodes = torch.full(
            (region_count, max_descendants, max_path_length),
            -1,
            dtype=torch.long,
        )
        router_targets = torch.full_like(router_nodes, -1)
        path_valid = torch.zeros_like(router_nodes, dtype=torch.bool)
        for region_index, (coarse_id, descendants) in enumerate(
            zip(region_ids, descendants_by_region)
        ):
            for leaf_index, leaf_id in enumerate(descendants):
                descendant_nodes[region_index, leaf_index] = node_index[leaf_id]
                path = self.tree.node_paths[leaf_id]
                start = path.index(coarse_id)
                for path_index in range(start, len(path) - 1):
                    router_id = path[path_index]
                    next_id = path[path_index + 1]
                    router = self.tree.nodes[router_id]
                    if router.left == next_id:
                        target = 0
                    elif router.right == next_id:
                        target = 1
                    else:
                        raise RuntimeError("invalid descendant routing path")
                    offset = path_index - start
                    router_nodes[region_index, leaf_index, offset] = (
                        node_index[router_id]
                    )
                    router_targets[region_index, leaf_index, offset] = target
                    path_valid[region_index, leaf_index, offset] = True

        cached = {
            "signature": signature,
            "region_ids": tuple(region_ids),
            "region_node_indices": torch.tensor(
                [node_index[node_id] for node_id in region_ids],
                dtype=torch.long,
            ),
            "descendants_by_region": tuple(descendants_by_region),
            "probe_counts": tuple(probe_counts),
            "max_probe_count": max_probe_count,
            "descendant_nodes": descendant_nodes,
            "router_nodes": router_nodes,
            "router_targets": router_targets,
            "path_valid": path_valid,
        }
        self._regional_probe_topology_cache = cached
        return cached

    def _regional_probe_objective(
        self,
        memory_output: Mapping[str, Any],
        frontier_posterior: Tensor,
        sequence_index: Tensor,
        sequence_count: int,
        sequence_event_embeddings: Tensor,
        flat: Mapping[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Counterfactually compare stop vs. a covered subset of deep leaves.

        Wake routing is unchanged.  For every coarse final-frontier region the
        probe evaluates its own Hawkes energy plus ``Kp`` least-probed leaves.
        The detached energy teacher independently supervises expansion,
        dormant paths, and leaf-local predictive calibration.
        """
        zero = sequence_event_embeddings.sum() * 0.0
        empty_long = torch.empty(0, dtype=torch.long, device=self.device)
        empty_float = sequence_event_embeddings.new_empty(0)

        def empty_result() -> Dict[str, Tensor]:
            return {
                "loss": zero,
                "router_loss": zero,
                "expand_loss": zero,
                "leaf_loss": zero,
                "node_indices": empty_long,
                "refinement_gain": empty_float,
                "expand_probability": empty_float,
                "expand_target": empty_float,
                "stop_distortion": empty_float,
                "leaf_distortion": empty_float,
                "assignment_confidence": empty_float,
                "regions": zero.detach(),
                "router_rows": zero.detach(),
                "probe_leaves": zero.detach(),
            }

        if self.wake_config.lambda_route_probe == 0.0:
            return empty_result()
        frontier_nodes = memory_output["frontier_node_indices"]
        frontier_mask = memory_output["frontier_mask"]
        if frontier_posterior.shape != frontier_nodes.shape:
            raise ValueError("frontier posterior and node slots must align")

        node_count = len(self.tree.all_node_ids)
        event_node_mass = frontier_posterior.detach().new_zeros(
            frontier_nodes.size(0), node_count
        )
        event_node_mass.scatter_add_(
            1,
            frontier_nodes.clamp_min(0),
            frontier_posterior.detach().masked_fill(~frontier_mask, 0.0),
        )
        sequence_node_mass, _ = self._segment_mean(
            event_node_mass, sequence_index, sequence_count
        )
        sequence_embedding, _ = self._segment_mean(
            sequence_event_embeddings, sequence_index, sequence_count
        )
        topology = self._regional_probe_topology()
        region_count = len(topology["region_ids"])
        if region_count == 0:
            return empty_result()

        region_nodes = topology["region_node_indices"].to(self.device)
        coarse_weight = sequence_node_mass.index_select(
            1, region_nodes
        ).transpose(0, 1).detach()
        total_weight = coarse_weight.sum(dim=1)
        active_region = total_weight > 1e-12
        active_count = int(active_region.sum().item())
        if active_count == 0:
            return empty_result()

        # The only host-side work left here is the small mutable coverage
        # policy.  No event rows or per-region energies cross the device
        # boundary; all active regions share the tensor path below.
        active_host = active_region.detach().cpu().tolist()
        selected_by_region: list[list[str]] = []
        selected_positions = torch.full(
            (region_count, topology["max_probe_count"]),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        for region_index, descendants in enumerate(
            topology["descendants_by_region"]
        ):
            selected = (
                self._least_probed_leaves(descendants)
                if active_host[region_index]
                else []
            )
            selected_by_region.append(selected)
            position_by_leaf = {
                leaf_id: position
                for position, leaf_id in enumerate(descendants)
            }
            if selected:
                selected_positions[region_index, :len(selected)] = (
                    selected_positions.new_tensor(
                        [position_by_leaf[leaf_id] for leaf_id in selected]
                    )
                )

        cumulative_node = self.tree._node_embedding_table()
        coarse_theta = torch.stack([
            self.tree.semantic_theta(region_id).detach()
            for region_id in topology["region_ids"]
        ])
        leaf_theta_rows: list[Tensor] = []
        for region_index, selected in enumerate(selected_by_region):
            rows = [
                self._probe_leaf_local_theta(leaf_id)
                for leaf_id in selected
            ]
            while len(rows) < topology["max_probe_count"]:
                rows.append(coarse_theta[region_index].detach())
            leaf_theta_rows.append(torch.stack(rows))
        leaf_theta = torch.stack(leaf_theta_rows)
        candidate_theta = torch.cat(
            (coarse_theta[:, None, :], leaf_theta),
            dim=1,
        )

        # Flatten [region, sequence] into the existing segmented energy kernel.
        region_offsets = torch.arange(
            region_count, device=self.device, dtype=sequence_index.dtype
        ).repeat_interleave(sequence_index.numel()) * sequence_count
        regional_sequence_index = sequence_index.repeat(region_count) + region_offsets
        regional_flat = {
            key: value.repeat(
                (region_count, *([1] * (value.ndim - 1)))
            )
            if value.ndim > 0 else value
            for key, value in flat.items()
        }
        regional_theta = candidate_theta[:, None, :, :].expand(
            region_count,
            sequence_count,
            candidate_theta.size(1),
            candidate_theta.size(2),
        ).reshape(region_count * sequence_count, candidate_theta.size(1), -1)
        energy = self._probe_sequence_energy(
            regional_flat,
            regional_sequence_index,
            region_count * sequence_count,
            regional_theta,
        ).reshape(region_count, sequence_count, -1)
        coarse_energy = energy[:, :, 0]
        selected_energy = energy[:, :, 1:]
        candidate_valid = selected_positions.ge(0)
        selected_energy = selected_energy.masked_fill(
            ~candidate_valid[:, None, :],
            1e6,
        )

        target_mass = self.tree.frontier_routing._target_leaf_mass_by_id
        prior = selected_energy.new_ones(
            region_count, topology["max_probe_count"]
        )
        for region_index, selected in enumerate(selected_by_region):
            if selected:
                prior[region_index, :len(selected)] = selected_energy.new_tensor([
                    target_mass.get(leaf_id, 1.0)
                    for leaf_id in selected
                ])
        probe = counterfactual_energy_probe(
            coarse_energy.detach(),
            selected_energy.detach(),
            coarse_weight,
            prior,
            teacher_temperature=(
                self.wake_config.route_probe_residual_temperature
            ),
            gain_temperature=(
                self.wake_config.route_probe_gain_temperature
            ),
            leaf_smoothing=self.wake_config.route_probe_leaf_smoothing,
        )

        sequence_embedding_batch = sequence_embedding[None, :, :].expand(
            region_count, -1, -1
        )
        region_embedding = cumulative_node.index_select(0, region_nodes).detach()
        expansion_logit = self.tree.expansion_predictor(
            sequence_embedding_batch.reshape(-1, sequence_embedding.size(-1)),
            region_embedding[:, None, :].expand(
                region_count, sequence_count, -1
            ).reshape(-1, region_embedding.size(-1)),
        ).reshape(region_count, sequence_count)
        safe_total = total_weight.clamp_min(1e-12)
        expansion_loss_by_region = (
            coarse_weight
            * F.binary_cross_entropy_with_logits(
                expansion_logit,
                probe.expand_target,
                reduction="none",
            )
        ).sum(dim=1) / safe_total
        leaf_loss_by_region = (
            coarse_weight
            * (probe.smoothed_leaf_credit * selected_energy).sum(dim=-1)
        ).sum(dim=1) / safe_total

        pooling_weight = coarse_weight[:, :, None] * probe.teacher[:, :, 1:]
        leaf_mass = pooling_weight.sum(dim=1)
        pooled_query = torch.einsum(
            "rsk,sz->rkz", pooling_weight, sequence_embedding
        ) / leaf_mass.clamp_min(1e-12)[:, :, None]
        leaf_fraction = leaf_mass / leaf_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        path_router = topology["router_nodes"].to(self.device).gather(
            1,
            selected_positions.clamp_min(0)[:, :, None].expand(
                -1, -1, topology["router_nodes"].size(-1)
            ),
        )
        path_targets = topology["router_targets"].to(self.device).gather(
            1,
            selected_positions.clamp_min(0)[:, :, None].expand(
                -1, -1, topology["router_targets"].size(-1)
            ),
        )
        path_valid = topology["path_valid"].to(self.device).gather(
            1,
            selected_positions.clamp_min(0)[:, :, None].expand(
                -1, -1, topology["path_valid"].size(-1)
            ),
        ) & candidate_valid[:, :, None]
        valid_rows = path_valid.reshape(-1)
        query = pooled_query[:, :, None, :].expand(
            region_count,
            topology["max_probe_count"],
            path_valid.size(-1),
            pooled_query.size(-1),
        ).reshape(-1, pooled_query.size(-1))[valid_rows]
        router_nodes = path_router.reshape(-1)[valid_rows]
        targets = path_targets.reshape(-1)[valid_rows]
        weights = leaf_fraction[:, :, None].expand_as(path_valid).reshape(-1)[
            valid_rows
        ].detach()
        child_index = self.tree.frontier_routing._topology_tensors[
            "child_index"
        ].index_select(0, router_nodes)
        normalized_node = self.tree.router_compat.normalize_nodes(
            cumulative_node.detach()
        )
        child_embedding = normalized_node.index_select(
            0, child_index.reshape(-1)
        ).reshape(query.size(0), 2, -1)
        semantic_score = self.tree.router_compat.score_normalized(
            self.tree.router_compat.project_z(query), child_embedding
        )
        child_prior = self.tree.frontier_routing._topology_tensors[
            "child_prior"
        ].to(semantic_score).index_select(0, router_nodes)
        logits = (
            self.tree.frontier_routing.config.semantic_weight
            * semantic_score
            / self.tree.frontier_routing.config.routing_temperature
            + child_prior.clamp_min(1e-12).log()
        )
        router_loss = (
            weights * F.cross_entropy(logits, targets, reduction="none")
        ).sum() / active_region.sum().to(weights).clamp_min(1.0)
        expand_loss = expansion_loss_by_region[active_region].mean()
        leaf_loss = leaf_loss_by_region[active_region].mean()
        loss = (
            self.wake_config.route_probe_router_weight * router_loss
            + self.wake_config.route_probe_expand_weight * expand_loss
            + self.wake_config.route_probe_leaf_weight * leaf_loss
        )
        return {
            "loss": loss,
            "router_loss": router_loss,
            "expand_loss": expand_loss,
            "leaf_loss": leaf_loss,
            "node_indices": region_nodes[active_region],
            "refinement_gain": probe.observed_gain[active_region],
            "expand_probability": (
                coarse_weight * expansion_logit.sigmoid().detach()
            ).sum(dim=1).div(safe_total)[active_region],
            "expand_target": (
                coarse_weight * probe.expand_target
            ).sum(dim=1).div(safe_total)[active_region],
            "stop_distortion": (
                coarse_weight * coarse_energy.detach()
            ).sum(dim=1).div(safe_total)[active_region],
            "leaf_distortion": (
                coarse_weight * probe.fine_energy
            ).sum(dim=1).div(safe_total)[active_region],
            "assignment_confidence": (
                coarse_weight * probe.assignment_confidence
            ).sum(dim=1).div(safe_total)[active_region],
            "regions": active_region.sum().detach().to(loss),
            "router_rows": path_valid.sum().detach().to(loss),
            "probe_leaves": candidate_valid.sum().detach().to(loss),
        }

    def _batched_local_frontier_objective(
        self,
        memory_output: Mapping[str, Any],
        child_energy: Tensor,
        sequence_index: Tensor,
        sequence_count: int,
    ) -> Dict[str, Tensor]:
        """Local teacher/MI loss with exact sequence and node normalization.

        The old implementation first averaged rows within each node, averaged
        nodes within each sequence, then averaged sequences. Two-level segment
        reductions reproduce that weighting without a Python node loop.
        """
        probability = memory_output["expanded_probability"]
        expanded_mask = memory_output["expanded_mask"]
        expanded_nodes = memory_output["expanded_node_indices"]
        zero = probability.sum() * 0.0
        if not bool(expanded_mask.any()):
            return {
                "distill": zero,
                "mutual_information": zero,
                "balance_kl": zero,
                "conditional_entropy": zero,
                "marginal_entropy": zero,
                "observed_gain": probability.new_zeros(
                    expanded_mask.shape
                ),
                "teacher": probability.new_zeros(probability.shape),
                "energy_teacher": probability.new_zeros(probability.shape),
                "student": probability.new_zeros(probability.shape),
                "reliability": probability.new_zeros(expanded_mask.shape),
            }

        topology = self.tree.frontier_routing._topology_tensors
        node_count = len(self.tree.all_node_ids)
        safe_expanded = expanded_nodes.clamp_min(0)
        child_prior = topology["child_prior"].to(probability)
        fixed_prior = child_prior.index_select(
            0, safe_expanded.reshape(-1)
        ).reshape_as(probability)
        safe_energy = child_energy.detach().masked_fill(
            ~expanded_mask.unsqueeze(-1), 0.0
        )
        teacher_logits = (
            fixed_prior.clamp_min(1e-12).log()
            - safe_energy / self.wake_config.route_teacher_temperature
        )
        branch_target = F.softmax(teacher_logits, dim=-1).masked_fill(
            ~expanded_mask.unsqueeze(-1), 0.0
        )
        energy_teacher = F.softmax(
            -safe_energy / self.wake_config.route_teacher_temperature,
            dim=-1,
        ).masked_fill(~expanded_mask.unsqueeze(-1), 0.0)
        reliability_rows = (
            1.0
            + (
                energy_teacher.clamp_min(1e-12)
                * energy_teacher.clamp_min(1e-12).log()
            ).sum(dim=-1)
            / math.log(2.0)
        ).clamp(0.0, 1.0).masked_fill(~expanded_mask, 0.0)
        # Distill the distribution used by the actual search.  It already
        # contains the same fixed topology prior as ``branch_target``.
        route_student = probability.masked_fill(
            ~expanded_mask.unsqueeze(-1), 0.0
        )
        distill_rows = (
            branch_target.detach()
            * (
                branch_target.detach().clamp_min(1e-12).log()
                - route_student.clamp_min(1e-12).log()
            )
        ).sum(dim=-1)

        expanded_sequence = sequence_index[:, None].expand_as(expanded_mask)
        selected_sequence = expanded_sequence[expanded_mask]
        selected_node = expanded_nodes[expanded_mask]
        selected_probability = probability[expanded_mask]
        selected_distill = distill_rows[expanded_mask]
        selected_reliability = reliability_rows[expanded_mask]

        # L_router = sum rho KL / (sum rho + eps). Equal child energies give
        # rho=0, so topology prior alone cannot manufacture supervision.
        distill = (
            selected_reliability * selected_distill
        ).sum() / selected_reliability.sum().clamp_min(1e-12)

        combined_segment = (
            selected_sequence * node_count + selected_node
        )
        combined_count = sequence_count * node_count
        marginal, row_counts = self._segment_mean(
            selected_probability, combined_segment, combined_count
        )
        row_entropy = -(
            selected_probability.clamp_min(1e-12)
            * selected_probability.clamp_min(1e-12).log()
        ).sum(dim=-1)
        conditional, _ = self._segment_mean(
            row_entropy, combined_segment, combined_count
        )
        marginal_entropy = -(
            marginal.clamp_min(1e-12)
            * marginal.clamp_min(1e-12).log()
        ).sum(dim=-1)
        segment_prior = child_prior.repeat(sequence_count, 1)
        balance = (
            marginal.clamp_min(1e-12)
            * (
                marginal.clamp_min(1e-12).log()
                - segment_prior.clamp_min(1e-12).log()
            )
        ).sum(dim=-1)
        observed = row_counts > 0
        marginal_entropy = marginal_entropy.masked_fill(~observed, 0.0)
        conditional = conditional.masked_fill(~observed, 0.0)
        balance = balance.masked_fill(~observed, 0.0)
        mutual_information = marginal_entropy - conditional

        observed_2d = observed.reshape(sequence_count, node_count)
        node_denominator = observed_2d.sum(dim=-1).clamp_min(1)

        def mean_nodes_then_sequences(values: Tensor) -> Tensor:
            return (
                values.reshape(sequence_count, node_count).sum(dim=-1)
                / node_denominator
            ).mean()

        return {
            "distill": distill,
            "mutual_information": mean_nodes_then_sequences(
                mutual_information
            ),
            "balance_kl": mean_nodes_then_sequences(balance),
            "conditional_entropy": mean_nodes_then_sequences(conditional),
            "marginal_entropy": mean_nodes_then_sequences(
                marginal_entropy
            ),
            "observed_gain": (
                reliability_rows * distill_rows
            ).detach().masked_fill(~expanded_mask, 0.0),
            "teacher": branch_target.detach(),
            "energy_teacher": energy_teacher.detach(),
            "student": route_student.detach(),
            "reliability": reliability_rows.detach(),
        }

    def _local_frontier_objective(
        self,
        memory_output: Mapping[str, Any],
        child_energy: Tensor,
    ) -> Dict[str, Tensor]:
        """Fixed-prior child-energy distillation and local regularizers."""
        probability = memory_output["expanded_probability"]
        expanded_mask = memory_output["expanded_mask"]
        expanded_nodes = memory_output["expanded_node_indices"]
        zero = probability.sum() * 0.0
        if not bool(expanded_mask.any()):
            return {
                "distill": zero,
                "mutual_information": zero,
                "balance_kl": zero,
                "conditional_entropy": zero,
                "marginal_entropy": zero,
                "observed_gain": probability.new_zeros(
                    expanded_mask.shape
                ),
                "teacher": probability.new_zeros(probability.shape),
                "energy_teacher": probability.new_zeros(probability.shape),
                "student": probability.new_zeros(probability.shape),
                "reliability": probability.new_zeros(expanded_mask.shape),
            }

        topology = self.tree.frontier_routing._topology_tensors
        safe_expanded = expanded_nodes.clamp_min(0)
        fixed_prior = topology["child_prior"].to(probability).index_select(
            0,
            safe_expanded.reshape(-1),
        ).reshape_as(probability)
        safe_energy = child_energy.detach().masked_fill(
            ~expanded_mask.unsqueeze(-1),
            0.0,
        )
        teacher_logits = (
            fixed_prior.clamp_min(1e-12).log()
            - safe_energy / self.wake_config.route_teacher_temperature
        )
        branch_target = F.softmax(teacher_logits, dim=-1).masked_fill(
            ~expanded_mask.unsqueeze(-1),
            0.0,
        )
        energy_teacher = F.softmax(
            -safe_energy / self.wake_config.route_teacher_temperature,
            dim=-1,
        ).masked_fill(~expanded_mask.unsqueeze(-1), 0.0)
        reliability_rows = (
            1.0
            + (
                energy_teacher.clamp_min(1e-12)
                * energy_teacher.clamp_min(1e-12).log()
            ).sum(dim=-1)
            / math.log(2.0)
        ).clamp(0.0, 1.0).masked_fill(~expanded_mask, 0.0)
        route_student = probability.masked_fill(
            ~expanded_mask.unsqueeze(-1), 0.0
        )
        distill_rows = (
            branch_target.detach()
            * (
                branch_target.detach().clamp_min(1e-12).log()
                - route_student.clamp_min(1e-12).log()
            )
        ).sum(dim=-1)
        selected_reliability = reliability_rows.masked_select(expanded_mask)
        distill = (
            selected_reliability
            * distill_rows.masked_select(expanded_mask)
        ).sum() / selected_reliability.sum().clamp_min(1e-12)

        node_mi = []
        node_balance = []
        node_conditional = []
        node_marginal = []
        child_prior = topology["child_prior"].to(probability)
        for node_index in expanded_nodes[expanded_mask].unique():
            selected = expanded_mask & (expanded_nodes == node_index)
            rows = probability[selected]
            if not rows.numel():
                continue
            marginal = rows.mean(dim=0)
            conditional_entropy = -(
                rows.clamp_min(1e-12)
                * rows.clamp_min(1e-12).log()
            ).sum(dim=-1).mean()
            marginal_entropy = -(
                marginal.clamp_min(1e-12)
                * marginal.clamp_min(1e-12).log()
            ).sum()
            prior = child_prior[int(node_index.item())]
            balance = (
                marginal.clamp_min(1e-12)
                * (
                    marginal.clamp_min(1e-12).log()
                    - prior.clamp_min(1e-12).log()
                )
            ).sum()
            node_conditional.append(conditional_entropy)
            node_marginal.append(marginal_entropy)
            node_mi.append(marginal_entropy - conditional_entropy)
            node_balance.append(balance)

        def mean_or_zero(values: Sequence[Tensor]) -> Tensor:
            return torch.stack(list(values)).mean() if values else zero

        return {
            "distill": distill,
            "mutual_information": mean_or_zero(node_mi),
            "balance_kl": mean_or_zero(node_balance),
            "conditional_entropy": mean_or_zero(node_conditional),
            "marginal_entropy": mean_or_zero(node_marginal),
            "observed_gain": (
                reliability_rows * distill_rows
            ).detach().masked_fill(~expanded_mask, 0.0),
            "teacher": branch_target.detach(),
            "energy_teacher": energy_teacher.detach(),
            "student": route_student.detach(),
            "reliability": reliability_rows.detach(),
        }

    def _sequence_route_mean_for_router(
        self,
        sequence: Mapping[str, Tensor],
        *,
        encoder_grad_scale: float = 0.0,
    ) -> Tensor:
        """Compatibility diagnostic: dense mass over actual tree nodes."""
        if not 0.0 <= encoder_grad_scale <= 1.0:
            raise ValueError("encoder_grad_scale must lie in [0, 1]")
        event_routes = []
        for event_index in range(sequence["times"].numel()):
            if encoder_grad_scale == 0.0:
                # During warm-up MI is a Router-only objective.
                with torch.no_grad():
                    z_t = self._encode_memory_event(
                        sequence,
                        event_index,
                    )
                routed_z = z_t.detach()
            else:
                z_t = self._encode_memory_event(
                    sequence,
                    event_index,
                )
                # Forward value is unchanged. Only the gradient entering the
                # Encoder is scaled; Router gradients retain full strength.
                routed_z = (
                    z_t.detach()
                    + encoder_grad_scale * (z_t - z_t.detach())
                )
            route = self.tree.route(routed_z)
            dense = routed_z.new_zeros(len(self.tree.all_node_ids))
            dense.scatter_add_(
                0,
                route.frontier_node_indices[
                    0, route.frontier_mask[0]
                ],
                route.responsibility[0, route.frontier_mask[0]],
            )
            event_routes.append(dense)
        if not event_routes:
            raise ValueError("global batches cannot contain empty sequences")
        return torch.stack(event_routes, dim=0).mean(dim=0)

    def train_global_batch_epoch(
        self,
        dataset: Sequence[Mapping[str, Tensor]],
        generator: torch.Generator,
        *,
        epoch: Optional[int] = None,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """Train experts and Router on the actual computed frontier.

        Prediction NLL updates experts with detached routing mass. Router
        updates come from the fixed-prior local child-energy teacher, the
        training-only Regional Probe under unexpanded coarse regions, and weak
        MI/balance regularizers. Frontier posterior and likelihood mixture
        remain detached diagnostics/ownership signals.
        """
        if not dataset:
            raise ValueError("global training requires at least one sequence")
        effective_epoch = (
            self.completed_epochs + 1 if epoch is None else int(epoch)
        )
        if effective_epoch <= 0:
            raise ValueError("global training epoch must be positive")

        order = torch.randperm(len(dataset), generator=generator).tolist()
        batch_size = self.wake_config.route_balance_batch_size
        batches = [
            order[start : start + batch_size]
            for start in range(0, len(order), batch_size)
        ]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())

        total_loss = 0.0
        total_prediction = 0.0
        total_likelihood_mixture = 0.0
        total_prior_kl = 0.0
        total_posterior_kl = 0.0
        total_distill = 0.0
        total_mi = 0.0
        total_conditional_entropy = 0.0
        total_marginal_entropy = 0.0
        total_probe_loss = 0.0
        total_probe_router_loss = 0.0
        total_probe_expand_loss = 0.0
        total_probe_leaf_loss = 0.0
        total_probe_expand_probability = 0.0
        total_probe_expand_target = 0.0
        total_probe_refinement_gain = 0.0
        total_probe_assignment_confidence = 0.0
        total_probe_regions = 0.0
        total_sequences = 0
        total_events = 0
        optimizer_steps = 0
        max_gradient_norm = 0.0
        total_encoder_grad_scale = 0.0
        reliability_updates = 0
        total_teacher_confidence = 0.0
        total_teacher_student_js = 0.0
        total_teacher_student_alignment = 0.0
        total_controller_loss = 0.0
        min_controller_head_grad_norm = math.inf
        max_controller_grad_norm = 0.0
        max_controller_head_grad_norms = torch.zeros(4, dtype=torch.float64)
        self.tree.train()
        self.encoder.train()

        global_progress = tqdm(
            total=len(dataset),
            desc=f"[Epoch {effective_epoch:03d}] Global",
            unit="seq-pass",
            ascii=True,
            dynamic_ncols=False,
            ncols=110,
            mininterval=2.0,
            maxinterval=10.0,
            smoothing=0.1,
            leave=True,
            disable=not show_progress,
            file=sys.stdout,
        )
        for batch_index, batch_indices in enumerate(batches, start=1):
            # Reliability is updated from the preceding observed batch. This
            # one-step lag keeps the gate causal and prevents its own current
            # gradients from changing the scale used in the same graph.
            encoder_grad_scale = (
                self.wake_config.route_encoder_grad_scale
                * self.encoder_routing_reliability
            )
            self.optimizer.zero_grad(set_to_none=True)
            moved_sequences = [
                self._move_sequence(dataset[index])
                for index in batch_indices
            ]
            batch_event_count = sum(
                int(sequence["times"].numel())
                for sequence in moved_sequences
            )
            if batch_event_count <= 0:
                continue
            sequence_count = len(moved_sequences)
            global_progress.set_postfix(
                phase="train",
                batch=f"{batch_index}/{len(batches)}",
                refresh=False,
            )
            z_all, flat = self._encode_global_sequence_batch(
                moved_sequences
            )
            routed_z = reliability_gated_route_state(
                z_all,
                reliability=self.encoder_routing_reliability,
                alpha_max=self.wake_config.route_encoder_grad_scale,
            )
            projected_routed_z = self.tree.router_compat.project_z(
                routed_z
            )
            memory_query = self.tree.episodic_memory.query_net(z_all)
            memory_output = self.tree(
                z_t=z_all,
                working_delta=torch.zeros(
                    self.tree.param_dim,
                    device=self.device,
                    dtype=z_all.dtype,
                ),
                decays=self.hawkes.decays,
                frontier_projected_z=projected_routed_z,
                frontier_query=memory_query,
                update_memory_state=False,
                update_search_state=False,
                detach_routing=True,
                materialize_diagnostics=False,
            )
            sequence_index = flat["sequence_index"]
            event_terms = self._batched_sequence_event_nll(
                flat, memory_output
            )
            full_retrieval_event_terms = event_terms
            batch_prediction_sum = event_terms.sum()
            batch_prediction = batch_prediction_sum / batch_event_count

            # Posterior/mix remain useful for ownership, memory assignment,
            # prototype credit, and diagnostics, but are outside autograd.
            with torch.no_grad():
                frontier_terms = self._batched_frontier_event_nll(
                    flat, memory_output
                )
                frontier_mask = memory_output["frontier_mask"]
                prior = memory_output["frontier_mass"].detach().masked_fill(
                    ~frontier_mask, 0.0
                )
                component = (
                    prior.clamp_min(1e-12).log()
                    - frontier_terms
                    / self.wake_config.route_energy_temperature
                ).masked_fill(~frontier_mask, -torch.inf)
                mixture_rows = -torch.logsumexp(component, dim=-1)
                mixture_by_sequence, _ = self._segment_mean(
                    mixture_rows, sequence_index, sequence_count
                )
                batch_likelihood_mixture = mixture_by_sequence.mean()
                posterior = F.softmax(component, dim=-1).masked_fill(
                    ~frontier_mask, 0.0
                )
                posterior_kl_rows = (
                    posterior
                    * (
                        posterior.clamp_min(1e-12).log()
                        - prior.clamp_min(1e-12).log()
                    )
                ).masked_fill(~frontier_mask, 0.0).sum(dim=-1)
                posterior_kl_by_sequence, _ = self._segment_mean(
                    posterior_kl_rows, sequence_index, sequence_count
                )
                batch_posterior_kl = posterior_kl_by_sequence.mean()
                child_energy = self._batched_expanded_child_event_nll(
                    flat, memory_output
                )
            owner_indices, _, owner_confidence = (
                self._posterior_owner_indices_batch(
                    memory_output["frontier_node_indices"], posterior
                )
            )
            novelty, soft_count, retrieval_similarity = (
                self.tree.episodic_memory.novelty_count_packed(
                    memory_query,
                    owner_indices,
                    self.tree.all_node_ids,
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
            zero_working = z_all.new_zeros(self.tree.param_dim)
            pre_action_effective = self._controller_effective_parameters(
                memory_output,
                zero_working,
                z_all.new_zeros(z_all.size(0)),
            )
            pre_action_terms = self._batched_sequence_event_nll(
                flat, {"effective_params": pre_action_effective}
            )
            retrieval_norm = memory_output[
                "frontier_episodic_delta"
            ].detach().norm(dim=-1)
            retrieval_norm = (
                retrieval_norm * memory_output["r"].detach()
            ).sum(dim=-1)
            controller_output = self.controller.action_distribution_batch(
                pre_action_terms.detach(),
                novelty.detach(),
                soft_count.detach(),
                update_statistics=False,
                owner_confidence=owner_confidence.detach(),
                retrieval_similarity=retrieval_similarity.detach(),
                retrieval_residual_norm=retrieval_norm.detach(),
                working_memory_norm=z_all.new_zeros(z_all.size(0)),
                pending_write_ratio=z_all.new_zeros(z_all.size(0)),
            )
            controller_output["logits"].retain_grad()
            gated_global_effective = self._controller_effective_parameters(
                memory_output,
                zero_working,
                controller_output["probabilities"][:, 1],
            )
            event_terms = self._batched_sequence_event_nll(
                flat, {"effective_params": gated_global_effective}
            )
            batch_prediction_sum = event_terms.sum()
            batch_prediction = batch_prediction_sum / batch_event_count
            # Epochs 1/2/3 use 0.1, 0.0667, 0.0333; epoch 4+ is utility-only.
            # Real counterfactuals never leak across heads: every label is masked.
            bootstrap_weight = max(0.0, 0.1 * (4 - effective_epoch) / 3.0)
            if self.training_config.controller_only_finetune:
                bootstrap_weight = 0.0
            controller_loss = batch_prediction * 0.0
            if bootstrap_weight > 0.0:
                controller_loss = self.controller.supervision_loss(
                    controller_output,
                    utility_targets=None,
                    bootstrap_weight=bootstrap_weight,
                    write_cost=self.wake_config.lambda_write,
                    split_cost=self.wake_config.controller_split_cost,
                    entropy_weight=self.wake_config.controller_entropy_weight,
                )

            if self.controller.utility_stage_enabled:
                train_retrieve = bool(
                    not self.training_config.controller_only_finetune
                    or "retrieve" in self.training_config.controller_train_heads
                )
                retrieval_utility = (
                    pre_action_terms - full_retrieval_event_terms
                    - self.wake_config.controller_retrieve_cost
                ).detach()
                retrieval_targets = controller_output["probabilities"].new_zeros(
                    controller_output["probabilities"].shape
                )
                retrieval_mask = torch.zeros_like(
                    retrieval_targets, dtype=torch.bool
                )
                retrieval_values = retrieval_targets.clone()
                retrieval_values[:, 1] = retrieval_utility
                retrieval_targets[:, 1] = self.controller.utility_target(
                    retrieval_utility, action_index=1, cost_margin=0.0,
                    update_statistics=train_retrieve,
                )
                retrieval_mask[:, 1] = train_retrieve
                controller_loss = controller_loss + self.controller.masked_utility_loss(
                    controller_output, retrieval_targets, retrieval_mask,
                    retrieval_values,
                    false_positive_weight=self.wake_config.controller_false_positive_weight,
                )

                inputs = torch.stack((
                    pre_action_terms.detach(), novelty.detach(), soft_count.detach(),
                    owner_confidence.detach(), retrieval_similarity.detach(),
                    retrieval_norm.detach(), z_all.new_zeros(z_all.size(0)),
                    z_all.new_zeros(z_all.size(0)),
                ), dim=-1)
                event_offsets = torch.cat([
                    torch.arange(int(sequence["times"].numel()), device=self.device)
                    for sequence in moved_sequences
                ])
                owner_values = owner_indices.detach().cpu().tolist()
                sequence_values = sequence_index.detach().cpu().tolist()
                if train_retrieve and inputs.size(0):
                    # Keep the reservoir's event order and Random stream, but
                    # move the differentiable payload to CPU exactly once.
                    cluster_by_sequence = [
                        int(torch.as_tensor(sequence.get("cluster_id", -1)))
                        for sequence in moved_sequences
                    ]
                    source_by_sequence = [
                        int(torch.as_tensor(sequence.get("source_index", -1)))
                        for sequence in moved_sequences
                    ]
                    metadata = {
                        "cluster_id": [
                            cluster_by_sequence[index]
                            for index in sequence_values
                        ],
                        "source_index": [
                            source_by_sequence[index]
                            for index in sequence_values
                        ],
                        "event_index": event_offsets.detach().cpu().tolist(),
                        "owner_id": [
                            self.tree.all_node_ids[index]
                            for index in owner_values
                        ],
                    }
                    self.controller_utility_replay.add_batch(
                        inputs,
                        retrieval_values,
                        retrieval_targets,
                        retrieval_mask,
                        retrieval_values.new_ones(4).expand_as(retrieval_values),
                        controller_output["probabilities"].detach(),
                        1,
                        metadata=metadata,
                    )

                replay_batch = self.controller_utility_replay.sample_batch(
                    self.wake_config.controller_replay_batch_sizes,
                    pin_memory=self.device.type == "cuda",
                )
                if replay_batch is not None:
                    # A pinned columnar payload avoids the old event-level
                    # ``.to``/``torch.stack`` pipeline.  Checkpoint rows and
                    # freshly collected rows share the same CPU representation.
                    replay_batch = replay_batch.to(
                        self.device,
                        non_blocking=self.device.type == "cuda",
                    )
                    replay_inputs = replay_batch.inputs
                    replay_output = self.controller.action_distribution_batch(
                        replay_inputs[:, 0], replay_inputs[:, 1], replay_inputs[:, 2],
                        update_statistics=False,
                        owner_confidence=replay_inputs[:, 3],
                        retrieval_similarity=replay_inputs[:, 4],
                        retrieval_residual_norm=replay_inputs[:, 5],
                        working_memory_norm=replay_inputs[:, 6],
                        pending_write_ratio=replay_inputs[:, 7],
                    )
                    targets = replay_batch.target
                    masks = replay_batch.mask
                    utilities = replay_batch.utility
                    propensities = replay_batch.propensity
                    weights = self.controller.normalized_inverse_propensity(
                        propensities, masks
                    )
                    controller_loss = controller_loss + self.controller.masked_utility_loss(
                        replay_output, targets, masks, utilities,
                        importance_weight=weights,
                        false_positive_weight=self.wake_config.controller_false_positive_weight,
                    )
                if self.training_config.controller_write_ranking:
                    ranking = self.controller_utility_replay.sample_write_ranking(
                        max_rows=96, max_pairs=192
                    )
                    ranking_rows = ranking["rows"]
                    if ranking_rows:
                        ranking_inputs = torch.stack([
                            row["inputs"].to(self.device) for row in ranking_rows
                        ])
                        ranking_output = self.controller.action_distribution_batch(
                            ranking_inputs[:, 0], ranking_inputs[:, 1], ranking_inputs[:, 2],
                            update_statistics=False,
                            owner_confidence=ranking_inputs[:, 3],
                            retrieval_similarity=ranking_inputs[:, 4],
                            retrieval_residual_norm=ranking_inputs[:, 5],
                            working_memory_norm=ranking_inputs[:, 6],
                            pending_write_ratio=ranking_inputs[:, 7],
                        )
                        ranking_loss, ranking_metrics = self.controller.write_ranking_loss(
                            ranking_output, ranking_rows, ranking["pairs"]
                        )
                        controller_loss = controller_loss + ranking_loss
                        self._last_write_ranking_metrics = ranking_metrics
            local = self._batched_local_frontier_objective(
                memory_output,
                child_energy,
                sequence_index,
                sequence_count,
            )
            regional = self._regional_probe_objective(
                memory_output,
                posterior,
                sequence_index,
                sequence_count,
                routed_z,
                flat,
            )
            objective = (
                batch_prediction
                + self.wake_config.lambda_route_distill
                * local["distill"]
                - self.wake_config.lambda_route_mi
                * local["mutual_information"]
                + self.wake_config.lambda_route_balance
                * local["balance_kl"]
                + self.wake_config.lambda_route_probe
                * regional["loss"]
                + controller_loss
            )
            if not torch.isfinite(objective):
                raise FloatingPointError(
                    "global frontier objective became non-finite"
                )
            objective.backward()
            logit_gradient = controller_output["logits"].grad
            if logit_gradient is None:
                raise RuntimeError("controller logits received no gradient")
            head_norms = logit_gradient.detach().double().norm(dim=0)
            max_controller_head_grad_norms = torch.maximum(
                max_controller_head_grad_norms, head_norms.cpu()
            )
            min_controller_head_grad_norm = min(
                min_controller_head_grad_norm,
                float(head_norms.min().cpu()),
            )
            controller_gradient = torch.stack([
                parameter.grad.detach().double().norm()
                for parameter in self.controller.parameters()
                if parameter.grad is not None
            ]).norm()
            max_controller_grad_norm = max(
                max_controller_grad_norm,
                float(controller_gradient.cpu()),
            )
            global_progress.update(sequence_count)
            gradient_norm = clip_grad_norm_finite(
                self._named_optimized_parameters(),
                self.training_config.grad_clip,
                context="cross-sequence global update",
            )
            max_gradient_norm = max(max_gradient_norm, gradient_norm)
            # Global optimization moves semantic references. Snapshot the
            # old coordinate system immediately before the atomic update, then
            # translate every existing residual before the next batch can
            # retrieve it. This preserves each effective Hawkes law exactly.
            old_refs = self.tree.episodic_memory.snapshot_semantic_references(
                self.tree.semantic_theta
            )
            self.optimizer.step()
            self.tree.episodic_memory.rebase_semantic_references(
                old_refs,
                self.tree.semantic_theta,
            )
            reliability = child_teacher_reliability(
                local["energy_teacher"],
                local["student"],
                memory_output["expanded_node_indices"].detach(),
                memory_output["expanded_mask"].detach(),
                node_count=len(self.tree.all_node_ids),
            )
            decay = self.wake_config.route_encoder_reliability_decay
            observed_reliability = float(
                reliability["reliability"].cpu()
            )
            observed_teacher_confidence = float(
                reliability["teacher_confidence"].cpu()
            )
            observed_teacher_student_js = float(
                reliability["teacher_student_js"].cpu()
            )
            observed_teacher_student_alignment = float(
                reliability["teacher_student_alignment"].cpu()
            )
            if not self.training_config.controller_only_finetune:
                self.encoder_routing_reliability = (
                    decay * self.encoder_routing_reliability
                    + (1.0 - decay) * observed_reliability
                )
                self.last_teacher_confidence = observed_teacher_confidence
                self.last_teacher_student_js = observed_teacher_student_js
                self.last_teacher_student_alignment = (
                    observed_teacher_student_alignment
                )
            total_teacher_confidence += observed_teacher_confidence
            total_teacher_student_js += observed_teacher_student_js
            total_teacher_student_alignment += (
                observed_teacher_student_alignment
            )
            reliability_updates += 1
            if not self.training_config.controller_only_finetune:
                self.tree.frontier_routing.prototypes.update_frontier_responsibility(
                    z_all.detach(),
                    memory_output["frontier_node_indices"].detach(),
                    posterior.detach(),
                    frontier_mask.detach(),
                )
                self.tree.frontier_routing.update_expansion_gain(
                    memory_output["expanded_node_indices"].detach(),
                    local["observed_gain"],
                    memory_output["expanded_mask"].detach(),
                )
                if regional["node_indices"].numel():
                    regional_mask = torch.ones_like(
                        regional["node_indices"], dtype=torch.bool
                    )
                    self.tree.frontier_routing.update_expansion_gain(
                        regional["node_indices"].detach(),
                        regional["refinement_gain"].clamp_min(0.0).detach(),
                        regional_mask,
                    )
            optimizer_steps += 1
            total_encoder_grad_scale += encoder_grad_scale

            batch_values = torch.stack(
                [
                    batch_prediction_sum.detach(),
                    batch_likelihood_mixture.detach(),
                    batch_posterior_kl.detach(),
                    local["distill"].detach(),
                    local["balance_kl"].detach(),
                    local["mutual_information"].detach(),
                    local["conditional_entropy"].detach(),
                    local["marginal_entropy"].detach(),
                    regional["loss"].detach(),
                    regional["router_loss"].detach(),
                    regional["expand_loss"].detach(),
                    regional["leaf_loss"].detach(),
                    regional["regions"].detach(),
                ]
            ).cpu().tolist()
            (
                batch_prediction_sum_value,
                batch_likelihood_mixture_value,
                posterior_kl_value,
                distill_value,
                prior_kl_value,
                mutual_information_value,
                conditional_entropy_value,
                marginal_entropy_value,
                probe_loss_value,
                probe_router_loss_value,
                probe_expand_loss_value,
                probe_leaf_loss_value,
                probe_regions_value,
            ) = batch_values
            batch_prediction_value = (
                batch_prediction_sum_value / batch_event_count
            )
            batch_loss = (
                batch_prediction_value
                + self.wake_config.lambda_route_distill
                * distill_value
                - self.wake_config.lambda_route_mi
                * mutual_information_value
                + self.wake_config.lambda_route_balance
                * prior_kl_value
                + self.wake_config.lambda_route_probe
                * probe_loss_value
                + float(controller_loss.detach().cpu())
            )
            total_sequences += sequence_count
            total_events += batch_event_count
            total_loss += batch_loss * sequence_count
            total_prediction += batch_prediction_sum_value
            total_likelihood_mixture += (
                batch_likelihood_mixture_value * sequence_count
            )
            total_posterior_kl += posterior_kl_value * sequence_count
            total_distill += distill_value * sequence_count
            total_prior_kl += prior_kl_value * sequence_count
            total_mi += mutual_information_value * sequence_count
            total_conditional_entropy += (
                conditional_entropy_value * sequence_count
            )
            total_marginal_entropy += (
                marginal_entropy_value * sequence_count
            )
            total_probe_loss += probe_loss_value * sequence_count
            total_probe_router_loss += (
                probe_router_loss_value * sequence_count
            )
            total_probe_expand_loss += (
                probe_expand_loss_value * sequence_count
            )
            total_probe_leaf_loss += probe_leaf_loss_value * sequence_count
            total_probe_regions += probe_regions_value
            total_controller_loss += (
                float(controller_loss.detach().cpu()) * sequence_count
            )
            if regional["expand_probability"].numel():
                total_probe_expand_probability += float(
                    regional["expand_probability"].mean().cpu()
                ) * probe_regions_value
                total_probe_expand_target += float(
                    regional["expand_target"].mean().cpu()
                ) * probe_regions_value
                total_probe_refinement_gain += float(
                    regional["refinement_gain"].mean().cpu()
                ) * probe_regions_value
                total_probe_assignment_confidence += float(
                    regional["assignment_confidence"].mean().cpu()
                ) * probe_regions_value
        global_progress.close()

        sequence_denominator = max(total_sequences, 1)
        event_denominator = max(total_events, 1)
        return {
            "loss": total_loss / sequence_denominator,
            "prediction_nll": total_prediction / event_denominator,
            "likelihood_mixture": (
                total_likelihood_mixture / sequence_denominator
            ),
            "balance_kl": total_prior_kl / sequence_denominator,
            "prior_kl": total_prior_kl / sequence_denominator,
            "posterior_kl": total_posterior_kl / sequence_denominator,
            "branch_distill": total_distill / sequence_denominator,
            "mutual_information": total_mi / sequence_denominator,
            "conditional_entropy": (
                total_conditional_entropy / sequence_denominator
            ),
            "marginal_entropy": (
                total_marginal_entropy / sequence_denominator
            ),
            "regional_probe_loss": (
                total_probe_loss / sequence_denominator
            ),
            "regional_probe_router_loss": (
                total_probe_router_loss / sequence_denominator
            ),
            "regional_probe_expand_loss": (
                total_probe_expand_loss / sequence_denominator
            ),
            "regional_probe_leaf_loss": (
                total_probe_leaf_loss / sequence_denominator
            ),
            "regional_probe_regions": total_probe_regions,
            "controller_loss": (
                total_controller_loss / sequence_denominator
            ),
            "controller_grad_norm": max_controller_grad_norm,
            "controller_head_grad_norms": {
                action.value: float(max_controller_head_grad_norms[index])
                for index, action in enumerate(Action)
            },
            "controller_min_head_grad_norm": (
                0.0 if math.isinf(min_controller_head_grad_norm)
                else min_controller_head_grad_norm
            ),
            "controller_utility_stage_enabled": bool(
                self.controller.utility_stage_enabled
            ),
            "regional_probe_expand_probability": (
                total_probe_expand_probability
                / max(total_probe_regions, 1.0)
            ),
            "regional_probe_expand_target": (
                total_probe_expand_target / max(total_probe_regions, 1.0)
            ),
            "regional_probe_refinement_gain": (
                total_probe_refinement_gain
                / max(total_probe_regions, 1.0)
            ),
            "regional_probe_assignment_confidence": (
                total_probe_assignment_confidence
                / max(total_probe_regions, 1.0)
            ),
            "samples": total_sequences,
            "sequences": total_sequences,
            "events": total_events,
            "batches": optimizer_steps,
            "optimizer_steps": optimizer_steps,
            "max_gradient_norm": max_gradient_norm,
            "encoder_route_grad_scale": (
                total_encoder_grad_scale / max(optimizer_steps, 1)
            ),
            # Backward-compatible diagnostics key.
            "mi_encoder_grad_scale": (
                total_encoder_grad_scale / max(optimizer_steps, 1)
            ),
            "encoder_route_gate": self.encoder_routing_reliability,
            "teacher_confidence": (
                total_teacher_confidence / max(reliability_updates, 1)
            ),
            "teacher_student_js": (
                total_teacher_student_js / max(reliability_updates, 1)
            ),
            "teacher_student_alignment": (
                total_teacher_student_alignment
                / max(reliability_updates, 1)
            ),
            "reliability_updates": reliability_updates,
            "steps": optimizer_steps,
        }
