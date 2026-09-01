"""Epoch-level Wake scheduling and end-to-end training loop."""

from __future__ import annotations

import inspect
from collections import defaultdict

from Train.TrainingComponents import *  # noqa: F403


class TrainingLoopMixin:
    @staticmethod
    def _rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
        if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
            return 0.0
        def ranks(values):
            order = sorted(range(len(values)), key=lambda index: values[index])
            result = [0.0] * len(values)
            for rank, index in enumerate(order):
                result[index] = float(rank)
            return result
        rx, ry = ranks(x), ranks(y)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        denominator = math.sqrt(
            sum((a - mx) ** 2 for a in rx)
            * sum((b - my) ** 2 for b in ry)
        )
        return numerator / denominator if denominator > 0 else 0.0

    def _calibrate_controller_checkpoint(
        self,
        checkpoint_path: str | Path,
        dataset: Sequence[Mapping[str, Tensor]],
    ) -> Dict[str, Any]:
        """Calibrate v5 action thresholds on isolated validation probes."""
        from Train.Inference import InferenceConfig, MemoryTreeInference

        trained_heads = set(self.training_config.controller_train_heads)
        preserved_thresholds = self.controller.calibration_dict()
        if trained_heads == {"write"}:
            from CalibrateWriteRollout import (
                WRITE_THRESHOLDS,
                paired_policy_improvement,
                paired_rollout_metrics,
                run_policy,
            )
            checkpoint = Path(checkpoint_path)
            frozen_rows = run_policy(
                checkpoint, dataset, write_threshold=1.01,
                online=False, device=str(self.device),
            )
            ranking_mode = bool(self.training_config.controller_write_ranking)
            baseline_rows = None
            baseline_threshold = None
            if ranking_mode:
                cache = getattr(self, "_write_ranking_baseline_cache", None)
                if cache is None:
                    base_path = Path(self.training_config.controller_base_checkpoint)
                    base_payload = torch.load(
                        base_path, map_location="cpu", weights_only=False
                    )
                    baseline_threshold = float(
                        base_payload.get("controller_calibration", {}).get(
                            "write_threshold", 0.95
                        )
                    )
                    baseline_rows = run_policy(
                        base_path, dataset, write_threshold=baseline_threshold,
                        online=True, device=str(self.device),
                    )
                    self._write_ranking_baseline_cache = {
                        "rows": baseline_rows,
                        "threshold": baseline_threshold,
                    }
                else:
                    baseline_rows = cache["rows"]
                    baseline_threshold = cache["threshold"]
            rollout_table = []
            for threshold in WRITE_THRESHOLDS:
                online_rows = run_policy(
                    checkpoint, dataset, write_threshold=threshold,
                    online=True, device=str(self.device),
                )
                row = {
                    "write_threshold": threshold,
                    **paired_rollout_metrics(
                        frozen_rows, online_rows,
                        num_types=self.hawkes.num_types,
                        sequence_count=len(dataset),
                        seed=self.training_config.seed,
                    ),
                }
                if ranking_mode:
                    comparison = paired_policy_improvement(
                        baseline_rows, online_rows,
                        seed=self.training_config.seed,
                    )
                    row["baseline_comparison"] = comparison
                    row["feasible"] = bool(
                        row["feasible"] and comparison["passed"]
                    )
                rollout_table.append(row)
            feasible = [row for row in rollout_table if row["feasible"]]
            selected = (
                max(feasible, key=lambda row: (
                    row["mean_nll_gain"], row["write_threshold"]
                ))
                if feasible else next(
                    row for row in rollout_table
                    if row["write_threshold"] == 1.01
                )
            )
            self.controller.set_calibration_thresholds(
                preserved_thresholds["retrieve_threshold"],
                preserved_thresholds["adapt_threshold"],
                selected["write_threshold"],
            )
            calibration = {
                "calibration_method": "realized_validation_rollout",
                "retrieve_threshold": preserved_thresholds["retrieve_threshold"],
                "adapt_threshold": preserved_thresholds["adapt_threshold"],
                "write_threshold": selected["write_threshold"],
                "selected_utility": {
                    "retrieve": 0.0, "adapt": 0.0,
                    "write": selected["mean_nll_gain"],
                },
                "harmful_fraction": {
                    "write": selected["harmful_event_fraction"]
                },
                "writes_per_sequence": selected["writes_per_sequence"],
                "threshold_search_table": {"write_rollout": rollout_table},
                "rollout_constraint_passed": bool(feasible),
                "fallback_to_baseline": bool(ranking_mode and not feasible),
                "baseline_write_threshold": baseline_threshold,
                "controller_policy_revision": 4 if ranking_mode else 3,
                "write_supervision": (
                    "sequence_relative_ranking" if ranking_mode
                    else "signed_utility"
                ),
            }
            self.controller_calibration = calibration
            return calibration

        def evaluate(*, episodic=True, working=False, write_probe=False):
            inference = MemoryTreeInference.from_checkpoint(
                checkpoint_path,
                device=self.device,
                inference_config=InferenceConfig(
                    adapt_working_memory=working,
                    allow_memory_writes=False,
                    update_memory_usage=False,
                    probe_write_counterfactuals=write_probe,
                    write_probe_seed=self.training_config.seed,
                ),
            )
            inference.controller.set_calibration_thresholds(0.0, 0.0, 0.0)
            if not episodic:
                for bank in inference.tree.episodic_memory.banks.values():
                    bank.clear()
                inference.tree.episodic_memory._packed_mirror = None
                inference.tree.episodic_memory._packed_mirror_signature = None
            rows = {}
            for sequence in dataset:
                source = int(torch.as_tensor(sequence["source_index"]).item())
                result = inference.run_sequence(sequence)
                for event in result["events"]:
                    rows[(source, int(event["event_index"]))] = event
            return rows

        full = evaluate(episodic=True, working=True)
        no_episodic = evaluate(episodic=False, working=True)
        no_working = evaluate(episodic=True, working=False)
        semantic = evaluate(episodic=False, working=False)
        write_rows = evaluate(episodic=True, working=True, write_probe=True)
        action_rows = {"retrieve": [], "adapt": [], "write": []}
        for key, event in full.items():
            raw = event.get("raw_action_probabilities", event["action_probabilities"])
            action_rows["retrieve"].append((
                float(raw[1]),
                float(no_episodic[key]["nll"]) - float(event["nll"]) - 1e-4,
                1.0,
                key[0],
            ))
            action_rows["adapt"].append((
                float(raw[0]),
                float(no_working[key]["nll"]) - float(event["nll"]) - 1e-4,
                1.0,
                key[0],
            ))
        for key, event in write_rows.items():
            if event.get("write_probed") and event.get("write_utility") is not None:
                raw = event.get("raw_action_probabilities", event["action_probabilities"])
                action_rows["write"].append((
                    float(raw[2]),
                    float(event["write_utility"]),
                    max(float(event.get("write_probe_propensity", 1.0)), 1e-6),
                    key[0],
                    bool(event.get("write_probe_top", False)),
                ))

        def select(name, thresholds, *, min_spearman=None):
            rows = action_rows[name]
            gates = [row[0] for row in rows]
            utilities = [row[1] for row in rows]
            positives = sum(row[1] > 0.0 for row in rows)
            negatives = sum(row[1] <= 0.0 for row in rows)
            spearman = self._rank_correlation(gates, utilities)
            candidates = []
            for threshold in thresholds:
                selected = [row for row in rows if row[0] >= threshold]
                objective_rows = selected
                if name == "write":
                    if int(self.controller.controller_version.detach().cpu()) < 6:
                        objective_rows = [row for row in selected if row[1] > 0.0]
                        selected = [
                            row for row in selected if row[1] > 0.0 and row[4]
                        ]
                    else:
                        # v6 admission cannot inspect the future signed label.
                        # Calibration therefore scores every gate-selected row.
                        selected = [row for row in selected if row[4]]
                    by_source = defaultdict(list)
                    for row in selected:
                        by_source[row[3]].append(row)
                    selected = [
                        row
                        for source_rows in by_source.values()
                        for row in sorted(
                            source_rows, key=lambda value: value[0],
                            reverse=True,
                        )[:4]
                    ]
                objective = sum(
                    row[1] * min(10.0, 1.0 / row[2])
                    for row in objective_rows
                )
                harmful = (
                    sum(row[1] < 0.0 for row in selected) / len(selected)
                    if selected else 0.0
                )
                writes_per_sequence = (
                    len(selected) / max(len(dataset), 1) if name == "write" else 0.0
                )
                feasible = objective >= 0.0 and harmful <= 0.45
                if min_spearman is not None:
                    feasible = feasible and spearman >= min_spearman
                if name == "write":
                    feasible = (
                        feasible and writes_per_sequence <= 2.0
                        and positives >= 20 and negatives >= 20
                    )
                candidates.append((feasible, objective, threshold, harmful, writes_per_sequence, len(selected)))
            feasible_rows = [row for row in candidates if row[0] and row[5] > 0]
            if feasible_rows:
                chosen = max(feasible_rows, key=lambda row: (row[1], row[2]))
            else:
                chosen = (True, 0.0, 1.01, 0.0, 0.0, 0)
            return {
                "threshold": float(chosen[2]),
                "selected_utility": float(chosen[1]),
                "harmful_fraction": float(chosen[3]),
                "writes_per_sequence": float(chosen[4]),
                "selected_count": int(chosen[5]),
                "sample_count": len(rows),
                "positive_count": positives,
                "negative_count": negatives,
                "spearman": float(spearman),
                "search_table": [
                    {
                        "feasible": bool(row[0]),
                        "objective": float(row[1]),
                        "threshold": float(row[2]),
                        "harmful_fraction": float(row[3]),
                        "writes_per_sequence": float(row[4]),
                        "selected_count": int(row[5]),
                    }
                    for row in candidates
                ],
            }

        if int(self.controller.controller_version.detach().cpu()) >= 6:
            from RecalibrateController import _joint_ra_search
            joint, joint_table = _joint_ra_search(
                semantic, no_working, no_episodic, full
            )
            def joint_result(name, threshold_key, utility_key, harmful_key, count_key):
                rows = action_rows[name]
                return {
                    "threshold": float(joint[threshold_key]),
                    "selected_utility": float(joint[utility_key]),
                    "harmful_fraction": float(joint[harmful_key]),
                    "writes_per_sequence": 0.0,
                    "selected_count": int(joint[count_key]),
                    "sample_count": len(rows),
                    "positive_count": sum(row[1] > 0.0 for row in rows),
                    "negative_count": sum(row[1] <= 0.0 for row in rows),
                    "spearman": float(self._rank_correlation(
                        [row[0] for row in rows], [row[1] for row in rows]
                    )),
                }
            retrieve = joint_result(
                "retrieve", "retrieve_threshold", "retrieve_utility_sum",
                "retrieve_harmful_fraction", "retrieve_selected_count",
            )
            adapt = joint_result(
                "adapt", "adapt_threshold", "adapt_utility_sum",
                "adapt_harmful_fraction", "adapt_selected_count",
            )
        else:
            joint_table = None
            retrieve = select("retrieve", [i / 100 for i in range(10, 100, 5)] + [1.01])
            adapt = select(
                "adapt", [i / 100 for i in range(10, 100, 5)] + [1.01],
                min_spearman=0.30,
            )
        write = select("write", [i / 100 for i in range(50, 100, 5)] + [1.01])
        # Calibration is head-isolated just like gradient updates.  Inactive
        # thresholds are policy state and must remain byte-for-byte stable.
        if "retrieve" not in trained_heads:
            retrieve["threshold"] = preserved_thresholds["retrieve_threshold"]
        if "adapt" not in trained_heads:
            adapt["threshold"] = preserved_thresholds["adapt_threshold"]
        if "write" not in trained_heads:
            write["threshold"] = preserved_thresholds["write_threshold"]
        self.controller.set_calibration_thresholds(
            retrieve["threshold"], adapt["threshold"], write["threshold"]
        )
        data_digest = hashlib.sha256()
        for sequence in sorted(
            dataset,
            key=lambda row: int(torch.as_tensor(row["source_index"]).item()),
        ):
            source = int(torch.as_tensor(sequence["source_index"]).item())
            data_digest.update(str(source).encode())
            for key in ("times", "types"):
                tensor = torch.as_tensor(sequence[key]).detach().cpu().contiguous()
                data_digest.update(str(tensor.dtype).encode())
                data_digest.update(tensor.numpy().tobytes())
        data_sha = data_digest.hexdigest()
        calibration = {
            "calibration_method": (
                "joint_counterfactual" if joint_table is not None else "independent"
            ),
            "retrieve_threshold": retrieve["threshold"],
            "adapt_threshold": adapt["threshold"],
            "write_threshold": write["threshold"],
            "validation_data_sha256": data_sha,
            "utility_counts": {
                name: {
                    key: value for key, value in result.items()
                    if key in {"sample_count", "positive_count", "negative_count"}
                }
                for name, result in (("retrieve", retrieve), ("adapt", adapt), ("write", write))
            },
            "selected_utility": {
                "retrieve": retrieve["selected_utility"],
                "adapt": adapt["selected_utility"],
                "write": write["selected_utility"],
            },
            "harmful_fraction": {
                "retrieve": retrieve["harmful_fraction"],
                "adapt": adapt["harmful_fraction"],
                "write": write["harmful_fraction"],
            },
            "writes_per_sequence": write["writes_per_sequence"],
            "spearman": {
                "retrieve": retrieve["spearman"],
                "adapt": adapt["spearman"],
                "write": write["spearman"],
            },
            "threshold_search_table": (
                {
                    "retrieve_adapt": joint_table,
                    "write": write.get("search_table", []),
                }
                if joint_table is not None else None
            ),
        }
        self.controller_calibration = calibration
        return calibration

    def _validate_controller_checkpoint(
        self,
        checkpoint_path: str | Path,
        dataset: Sequence[Mapping[str, Tensor]],
    ) -> Dict[str, float]:
        """State-isolated validation; Write-only runs include real online rollout."""
        from Train.Inference import InferenceConfig, MemoryTreeInference
        from Evaluate import bootstrap_ci, classification_metrics

        totals = {}
        rows_by_name = {}
        modes = [("semantic_only", False, False), ("full_frozen", True, False)]
        write_only = set(self.training_config.controller_train_heads) == {"write"}
        if write_only:
            modes.append(("full_online", True, True))
        for name, episodic, online in modes:
            inference = MemoryTreeInference.from_checkpoint(
                checkpoint_path,
                device=self.device,
                inference_config=InferenceConfig(
                    adapt_working_memory=episodic,
                    allow_memory_writes=online,
                    update_memory_usage=online,
                ),
            )
            if not episodic:
                for bank in inference.tree.episodic_memory.banks.values():
                    bank.clear()
                inference.tree.episodic_memory._packed_mirror = None
                inference.tree.episodic_memory._packed_mirror_signature = None
            nll, events = 0.0, 0
            rows = []
            accepted = 0
            for sequence in dataset:
                result = inference.run_sequence(sequence)
                nll += float(result["total_nll"])
                events += len(result["events"])
                accepted += int(result.get("accepted_write_count", 0))
                for event in result["events"]:
                    rows.append({
                        "source_index": int(torch.as_tensor(sequence["source_index"]).item()),
                        "event_index": int(event["event_index"]),
                        "nll": float(event["nll"]),
                        "true_type": int(event["true_type"]),
                        "predicted_type_at_event_time": int(event["predicted_type"]),
                        "type_probabilities": [
                            float(value) for value in event["type_probabilities_at_event_time"]
                        ],
                    })
            totals[name] = nll / max(events, 1)
            rows_by_name[name] = rows
            if name == "full_frozen":
                payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
                totals["full_frozen_event_sha256"] = hashlib.sha256(
                    payload.encode("utf-8")
                ).hexdigest()
            if online:
                totals["physical_accepted_count"] = accepted
                totals["writes_per_sequence"] = accepted / max(len(dataset), 1)
        totals["memory_gain"] = totals["semantic_only"] - totals["full_frozen"]
        if write_only:
            frozen = rows_by_name["full_frozen"]
            online_rows = rows_by_name["full_online"]
            gains = [
                float(base["nll"]) - float(target["nll"])
                for base, target in zip(frozen, online_rows)
            ]
            interval = bootstrap_ci(gains, self.training_config.seed, samples=1000)
            frozen_cls = classification_metrics(frozen, self.hawkes.num_types)
            online_cls = classification_metrics(online_rows, self.hawkes.num_types)
            totals.update({
                "online_gain": sum(gains) / max(len(gains), 1),
                "online_gain_bootstrap_95ci": interval,
                "online_harmful_fraction": sum(value < 0.0 for value in gains) / max(len(gains), 1),
                "full_frozen_accuracy": frozen_cls["accuracy"],
                "full_online_accuracy": online_cls["accuracy"],
                "full_frozen_macro_f1": frozen_cls["macro_f1"],
                "full_online_macro_f1": online_cls["macro_f1"],
            })
        return totals

    def _iter_batched_wake_encodings(
        self,
        dataset: Sequence[Mapping[str, Tensor]],
        order: Sequence[int],
    ):
        """Compatibility iterator over prepared masked-wavefront batches."""
        for batch in self._iter_masked_wavefront_batches(dataset, order):
            lengths = batch["lengths"]
            offset = 0
            for local_row, (sequence_index, length) in enumerate(zip(
                batch["sequence_indices"],
                lengths,
            )):
                end = offset + int(length)
                yield (
                    sequence_index,
                    batch["z_flat"][offset:end],
                    batch["projected_flat"][offset:end],
                    batch["query_flat"][offset:end],
                    batch["frontier_static_cache"],
                    batch["frontier_flat"].slice(offset, end),
                    batch["frontier_rows"][offset:end],
                )
                offset = end

    def _iter_masked_wavefront_batches(
        self,
        dataset: Sequence[Mapping[str, Tensor]],
        order: Sequence[int],
    ):
        """Precompute stateless prefix data and yield one Wake transaction."""
        chunk_size = max(
            1, int(self.wake_config.wake_wavefront_batch_size)
        )
        static_cache = self.tree.frontier_routing.build_static_cache(
            detach=True
        )
        for start in range(0, len(order), chunk_size):
            chunk_indices = order[start : start + chunk_size]
            sequences = [dataset[index] for index in chunk_indices]
            with torch.no_grad():
                z_flat, flat = self._encode_global_sequence_batch(sequences)
                projected_flat = self.tree.router_compat.project_z(z_flat)
                query_flat = self.tree.episodic_memory.query_net(z_flat)
                frontier_flat = self.tree.frontier_routing.route_packed(
                    z_flat,
                    update_search_state=(
                        not self.training_config.controller_only_finetune
                    ),
                    node_embedding_table=static_cache.node_embedding_table,
                    normalized_node_table=(
                        static_cache.normalized_node_table
                    ),
                    projected_z=projected_flat,
                )
            # Materialize the hard-routing identity view once per chunk.
            # Counts and slot indices are already final and independent of
            # retrieval/working-memory state.
            frontier_count = frontier_flat.mask.sum(dim=-1)
            packed_counts = torch.stack(
                [
                    frontier_count,
                    frontier_flat.visited_mask.sum(dim=-1),
                    frontier_flat.expanded_mask.sum(dim=-1),
                ],
                dim=-1,
            ).detach().cpu()
            frontier_indices_cpu = (
                frontier_flat.node_indices.detach().cpu()
            )
            all_node_ids = tuple(self.tree.all_node_ids)
            frontier_rows = []
            for row_index in range(frontier_indices_cpu.size(0)):
                size, visited, branches = packed_counts[row_index].tolist()
                node_ids = tuple(
                    all_node_ids[int(node_index)]
                    for node_index in frontier_indices_cpu[
                        row_index, : int(size)
                    ].tolist()
                )
                frontier_rows.append(
                    (node_ids, int(visited), int(branches))
                )
            lengths = flat["sequence_lengths"].detach().cpu().tolist()
            yield {
                "sequences": sequences,
                "sequence_indices": tuple(chunk_indices),
                "lengths": lengths,
                "z_flat": z_flat,
                "projected_flat": projected_flat,
                "query_flat": query_flat,
                "frontier_static_cache": static_cache,
                "frontier_flat": frontier_flat,
                "frontier_rows": frontier_rows,
                "flat": flat,
            }

    def train(
        self,
        dataset: Sequence[Mapping[str, Tensor]],
        *,
        verbose: bool = True,
        validation_dataset: Optional[Sequence[Mapping[str, Tensor]]] = None,
    ) -> list[Dict[str, Any]]:
        if not dataset:
            raise ValueError("training requires at least one sequence")
        if self.training_config.epochs <= 0:
            raise ValueError("training epochs must be positive")
        if self.wake_config.route_balance_max_steps <= 0:
            raise ValueError("route_balance_max_steps must be positive")
        if self.wake_config.route_balance_target_kl < 0.0:
            raise ValueError("route_balance_target_kl must be non-negative")
        cache_started = time.perf_counter()
        caches_built = 0
        cached_events = 0
        resident_dataset = []
        cache_progress = tqdm(
            dataset,
            total=len(dataset),
            desc="[Init] Cache",
            unit="seq",
            ascii=True,
            dynamic_ncols=False,
            ncols=110,
            mininterval=2.0,
            maxinterval=10.0,
            smoothing=0.1,
            leave=True,
            disable=not verbose,
            file=sys.stdout,
        )
        for sequence in cache_progress:
            cached_events += int(sequence["times"].numel())
            if (
                sequence.get(HAWKES_CACHE_SIGNATURE_KEY)
                != self.hawkes.cache_signature
            ):
                caches_built += 1
            resident_sequence = {
                key: (
                    value.to(self.device, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in sequence.items()
            }
            self.hawkes.prepare_sequence_cache(
                resident_sequence,
                inplace=True,
            )
            resident_dataset.append(resident_sequence)
            cache_progress.set_postfix(
                events=cached_events,
                built=caches_built,
                refresh=False,
            )
        cache_progress.close()
        dataset = resident_dataset
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if verbose:
            print(
                f"[Cache] sequences={len(dataset)} events={cached_events} "
                f"built={caches_built} "
                f"device={self.device.type} "
                f"time={time.perf_counter() - cache_started:.3f}s"
            )
        generator = torch.Generator(device="cpu")
        final_epoch = self.completed_epochs + self.training_config.epochs

        for epoch in range(self.completed_epochs + 1, final_epoch + 1):
            # Epoch-indexed shuffling makes an interrupted/resumed run use the
            # same data order as an uninterrupted run.
            generator.manual_seed(self.training_config.seed + epoch - 1)
            order = torch.randperm(len(dataset), generator=generator).tolist()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            epoch_started = time.perf_counter()
            wake_started = epoch_started
            self._resident_cache_hits = 0
            self._resident_cache_misses = 0
            wake_prediction = 0.0
            wake_wm = 0.0
            writes = 0
            write_decisions = 0
            memorizes = 0
            queue_splits = 0
            event_count = 0
            max_gradient_norm = 0.0
            wake_action_counts: Counter[str] = Counter()
            gate_probability_totals = {action.value: 0.0 for action in Action}
            gate_activation_totals = {action.value: 0.0 for action in Action}
            novelty_total = 0.0
            max_similarity_total = 0.0
            similarity_count_total = 0.0
            frontier_size_total = 0.0
            frontier_visited_total = 0.0
            frontier_branch_total = 0.0
            expansion_utility_total = 0.0
            frontier_node_counts: Counter[str] = Counter()
            sequence_responsibility_rows = []
            memory_assignment_counts: Counter[str] = Counter()
            sequence_owner_counts: Counter[str] = Counter()
            posterior_entropy_total = 0.0
            prior_posterior_kl_total = 0.0
            owner_depth_total = 0.0
            owner_lca_total = 0.0
            write_candidates = 0
            write_probes = 0
            write_gate_passes = 0
            write_utility_passes = 0
            accepted_write_utility_sum = 0.0
            harmful_writes = 0
            raw_structural_mass = 0.0
            structural_observations = 0
            wake_progress = tqdm(
                total=len(order),
                desc=f"[Epoch {epoch:03d}] Wake",
                unit="seq",
                ascii=True,
                dynamic_ncols=False,
                ncols=110,
                mininterval=2.0,
                maxinterval=10.0,
                smoothing=0.1,
                leave=True,
                disable=not verbose,
                file=sys.stdout,
            )
            for wake_batch in self._iter_masked_wavefront_batches(
                dataset,
                order,
            ):
                batch_results = self.train_wake_batch(
                    sequences=wake_batch["sequences"],
                    sequence_indices=wake_batch["sequence_indices"],
                    z_flat=wake_batch["z_flat"],
                    projected_flat=wake_batch["projected_flat"],
                    query_flat=wake_batch["query_flat"],
                    frontier_static_cache=(
                        wake_batch["frontier_static_cache"]
                    ),
                    frontier_flat=wake_batch["frontier_flat"],
                    frontier_rows=wake_batch["frontier_rows"],
                    flat=wake_batch["flat"],
                )
                for result in batch_results:
                    wake_prediction += result["prediction_nll"]
                    wake_wm += result["wm_penalty"]
                    writes += result["write_count"]
                    write_decisions += result["write_decision_count"]
                    memorizes += result["memorize_count"]
                    queue_splits += result["queue_split_count"]
                    event_count += result["event_count"]
                    wake_action_counts.update(result["action_counts"])
                    for action in Action:
                        gate_probability_totals[action.value] += (
                            result["mean_gates"][action.value]
                            * result["event_count"]
                        )
                        gate_activation_totals[action.value] += (
                            result["gate_activation_rates"][action.value]
                            * result["event_count"]
                        )
                    novelty_total += (
                        result["mean_novelty"] * result["event_count"]
                    )
                    max_similarity_total += (
                        result["mean_max_similarity"]
                        * result["event_count"]
                    )
                    similarity_count_total += (
                        result["mean_similarity_count"]
                        * result["event_count"]
                    )
                    frontier_size_total += (
                        result["mean_frontier_size"]
                        * result["event_count"]
                    )
                    frontier_visited_total += (
                        result["mean_frontier_visited_nodes"]
                        * result["event_count"]
                    )
                    frontier_branch_total += (
                        result["mean_frontier_branches"]
                        * result["event_count"]
                    )
                    expansion_utility_total += (
                        result["mean_expansion_utility"]
                        * result["event_count"]
                    )
                    frontier_node_counts.update(
                        result["frontier_node_counts"]
                    )
                    max_gradient_norm = max(
                        max_gradient_norm,
                        result["max_gradient_norm"],
                    )
                    sequence_responsibility_rows.append(
                        result["sequence_responsibility"]
                    )
                    memory_assignment_counts.update(
                        result["memory_assignment_counts"]
                    )
                    sequence_owner_counts[
                        result["sequence_owner_id"]
                    ] += 1
                    posterior_entropy_total += (
                        result["posterior_entropy"]
                        * result["event_count"]
                    )
                    prior_posterior_kl_total += (
                        result["prior_posterior_kl"]
                        * result["event_count"]
                    )
                    owner_depth_total += (
                        result["memory_owner_depth_mean"]
                        * result["event_count"]
                    )
                    owner_lca_total += (
                        result["owner_lca_rate"]
                        * result["event_count"]
                    )
                    write_candidates += result["write_candidates"]
                    write_probes += result["write_probe_count"]
                    write_gate_passes += result["write_gate_pass_count"]
                    write_utility_passes += result["write_utility_pass_count"]
                    accepted_write_utility_sum += result[
                        "accepted_write_utility_sum"
                    ]
                    harmful_writes += result["harmful_write_count"]
                    raw_structural_mass += float(
                        result.get("raw_structural_mass", 0.0)
                    )
                    structural_observations += int(
                        result.get(
                            "structural_observations",
                            result["event_count"],
                        )
                    )
                    wake_progress.set_postfix(
                        events=event_count,
                        writes=writes,
                        loss=(
                            f"{wake_prediction / max(event_count, 1):.4f}"
                        ),
                        refresh=False,
                    )
                    wake_progress.update(1)
            wake_progress.close()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            wake_seconds = time.perf_counter() - wake_started

            wake_loss = (
                wake_prediction
                + wake_wm
                + self.wake_config.lambda_write * write_decisions
            ) / max(event_count, 1)
            responsibility_matrix = torch.stack(
                sequence_responsibility_rows,
                dim=0,
            )
            # Leaf tensors are retained only for Sleep mass bookkeeping and
            # contain posterior credit for actually evaluated leaf experts.
            # Routing diagnostics/ownership live on actual frontier nodes.
            leaf_observed_mass = responsibility_matrix.mean(dim=0)
            owner_mass = leaf_observed_mass.new_tensor([
                sequence_owner_counts[node_id]
                for node_id in self.tree.all_node_ids
            ])
            owner_mass = owner_mass / owner_mass.sum().clamp_min(1.0)
            route_marginal_entropy = float(
                -(
                    owner_mass.clamp_min(1e-12)
                    * owner_mass.clamp_min(1e-12).log()
                ).sum().detach().cpu()
            )
            route_conditional_entropy = (
                posterior_entropy_total / max(event_count, 1)
            )
            route_entropy = route_marginal_entropy
            route_mutual_information = max(
                route_marginal_entropy - route_conditional_entropy,
                0.0,
            )
            max_leaf_mass = float(
                leaf_observed_mass.max().detach().cpu()
            )
            hard_assignment_counts = {
                node_id: int(sequence_owner_counts[node_id])
                for node_id in self.tree.all_node_ids
            }
            # This is the only ordinary optimizer phase in the epoch. With
            # batch_size >= number of sequences it performs exactly one step.
            global_started = time.perf_counter()
            global_update = self.train_global_batch_epoch(
                dataset,
                generator,
                epoch=epoch,
                show_progress=verbose,
            )
            if (
                self.wake_config.controller_auto_enable_utility_stage
                and not self.controller.utility_stage_enabled
                and epoch >= self.wake_config.controller_utility_min_epoch
                and global_update.get(
                    "controller_min_head_grad_norm", 0.0
                ) > self.wake_config.controller_min_head_grad_norm
                and writes <= max(write_candidates, 1)
            ):
                self.controller.utility_stage_enabled = True
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            global_seconds = time.perf_counter() - global_started
            router_calibration = global_update
            sleep_result = None
            accepted_writes_since_sleep = (
                int(self.sleep_state.get("accepted_writes_since_sleep", 0))
            )
            if not self.training_config.controller_only_finetune:
                accepted_writes_since_sleep += int(writes)
                self.sleep_state["accepted_writes_since_sleep"] = (
                    accepted_writes_since_sleep
                )
                self.sleep_state["structural_mass_since_sleep"] = (
                    float(self.sleep_state.get(
                        "structural_mass_since_sleep", 0.0
                    ))
                    + raw_structural_mass
                )
                self.sleep_state[
                    "structural_observations_since_sleep"
                ] = (
                    int(self.sleep_state.get(
                        "structural_observations_since_sleep", 0
                    ))
                    + structural_observations
                )
            topology_prune_enabled = (
                epoch > self.structure_config.prune_warmup_epochs
            )
            sleep_started = time.perf_counter()
            if (
                not self.training_config.controller_only_finetune
                and
                epoch % self.training_config.sleep_every == 0
                and sequence_responsibility_rows
            ):
                responsibilities = torch.stack(
                    sequence_responsibility_rows,
                    dim=0,
                )
                sleep_kwargs = {
                    "allow_topology_prune": topology_prune_enabled,
                    "epoch": epoch,
                    "show_progress": verbose,
                }
                # Keep the training loop compatible with checkpoints/code
                # deployments whose Sleep mixin predates the exact accepted-
                # write counter.  Newer implementations consume the value;
                # older ones retain their memory-count fallback.
                if "accepted_writes" in inspect.signature(
                    self.train_sleep
                ).parameters:
                    sleep_kwargs["accepted_writes"] = (
                        accepted_writes_since_sleep
                    )
                sleep_result = self.train_sleep(
                    responsibilities,
                    **sleep_kwargs,
                )
                self.sleep_state["accepted_writes_since_sleep"] = 0
                transaction = sleep_result.get("transaction") or {}
                split_nodes = {
                    action.get("node")
                    for action in transaction.get("actions", [])
                    if action.get("action") == "split"
                }
                evaluated_nodes = set(sleep_result.get("evaluated_split_nodes", ()))
                for row in self.controller_utility_replay.rows(action=2):
                    owner_id = row.get("owner_id")
                    if owner_id in evaluated_nodes:
                        label = float(owner_id in split_nodes)
                        split_row = dict(row)
                        split_row["utility"] = row["utility"].clone()
                        split_row["target"] = row["target"].clone()
                        split_row["label_mask"] = row["label_mask"].clone()
                        split_row["propensity"] = row["propensity"].clone()
                        split_row["utility"][3] = 1.0 if label else -1.0
                        split_row["target"][3] = label
                        split_row["label_mask"][3] = True
                        split_row["propensity"][3] = 1.0
                        self.controller_utility_replay.add(split_row, 3)
            utility_rows = self.controller_utility_replay.rows()
            utilities_by_action = []
            for action_index in range(4):
                # A resumed checkpoint can contain CUDA replay rows while
                # rows collected in this run are stored on CPU.  Normalize
                # each scalar before stacking for the temperature update.
                values = [
                    row["utility"][action_index].to(self.device)
                    for row in utility_rows
                    if bool(row["label_mask"][action_index])
                ]
                utilities_by_action.append(
                    torch.stack(values)
                    if values
                    else torch.empty(0, device=self.device)
                )
            self.controller.update_utility_temperatures(utilities_by_action)
            if self.training_config.controller_only_finetune:
                self.tree.reset_working_memory()
                actual_hash = self.non_controller_state_sha256()
                expected_hash = self.training_config.frozen_state_sha256
                if actual_hash != expected_hash:
                    expected_items = getattr(
                        self, "_frozen_state_item_hashes", {}
                    )
                    actual_items = self.non_controller_state_item_sha256()
                    changed = sorted(
                        name for name in set(expected_items) | set(actual_items)
                        if expected_items.get(name) != actual_items.get(name)
                    )
                    raise RuntimeError(
                        "Controller-only invariant failed: non-Controller state changed "
                        f"({expected_hash} != {actual_hash}); changed_items={changed[:32]}"
                    )
            replay_action_statistics = {}
            for action_index, action_name in enumerate(
                ("adapt", "retrieve", "write", "split")
            ):
                labeled = [
                    row for row in utility_rows
                    if bool(row["label_mask"][action_index])
                ]
                values = [
                    float(row["utility"][action_index]) for row in labeled
                ]
                targets = [
                    float(row["target"][action_index]) for row in labeled
                ]
                replay_action_statistics[action_name] = {
                    "labeled": len(labeled),
                    "mean_utility": sum(values) / max(len(values), 1),
                    "mean_target": sum(targets) / max(len(targets), 1),
                    "positive_utility_fraction": sum(value > 0 for value in values)
                    / max(len(values), 1),
                    "negative_utility_fraction": sum(value < 0 for value in values)
                    / max(len(values), 1),
                }
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            sleep_seconds = time.perf_counter() - sleep_started
            epoch_seconds = time.perf_counter() - epoch_started
            cuda_peak_memory_mb = (
                float(torch.cuda.max_memory_allocated(self.device))
                / (1024.0 * 1024.0)
                if self.device.type == "cuda"
                else 0.0
            )
            epoch_result = {
                "epoch": epoch,
                "wake_loss_per_event": wake_loss,
                "wake_prediction_nll_per_event": wake_prediction / max(event_count, 1),
                "writes": writes,
                "write_decisions": write_decisions,
                "memorizes": memorizes,
                "queue_splits": queue_splits,
                "wake_action_counts": {
                    action.value: int(wake_action_counts[action.value])
                    for action in Action
                },
                "controller_gate_means": {
                    action.value: gate_probability_totals[action.value]
                    / max(event_count, 1)
                    for action in Action
                },
                "controller_gate_over_0_5_fraction": {
                    action.value: gate_activation_totals[action.value]
                    / max(event_count, 1)
                    for action in Action
                },
                "controller_gate_gradient_norms": global_update.get(
                    "controller_head_grad_norms", {}
                ),
                "memory_assignment_counts": {
                    node_id: int(memory_assignment_counts[node_id])
                    for node_id in self.tree.all_node_ids
                },
                "mean_novelty": novelty_total / max(event_count, 1),
                "mean_weighted_similarity": (
                    max_similarity_total / max(event_count, 1)
                ),
                "mean_soft_count": (
                    similarity_count_total / max(event_count, 1)
                ),
                "mean_max_similarity": (
                    max_similarity_total / max(event_count, 1)
                ),
                "mean_similarity_count": (
                    similarity_count_total / max(event_count, 1)
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
                "mean_expansion_utility": (
                    expansion_utility_total / max(event_count, 1)
                ),
                "frontier_node_counts": {
                    node_id: int(frontier_node_counts[node_id])
                    for node_id in self.tree.all_node_ids
                },
                "events": event_count,
                "max_wake_gradient_norm": max_gradient_norm,
                "route_entropy": route_entropy,
                "route_conditional_entropy": route_conditional_entropy,
                "route_marginal_entropy": route_marginal_entropy,
                "route_mutual_information": route_mutual_information,
                "mean_sequence_route_std": 0.0,
                "max_sequence_route_std": 0.0,
                "max_leaf_mass": max_leaf_mass,
                "marginal_leaf_mass": {
                    leaf_id: float(mass)
                    for leaf_id, mass in zip(
                        self.tree.leaf_ids,
                        leaf_observed_mass.detach().cpu().tolist(),
                    )
                },
                "hard_assignment_counts": hard_assignment_counts,
                "posterior_entropy": route_conditional_entropy,
                "prior_posterior_kl": (
                    prior_posterior_kl_total / max(event_count, 1)
                ),
                "memory_owner_depth_mean": (
                    owner_depth_total / max(event_count, 1)
                ),
                "owner_lca_rate": (
                    owner_lca_total / max(event_count, 1)
                ),
                "write_candidates": write_candidates,
                "write_probes": write_probes,
                "write_gate_passes": write_gate_passes,
                "write_utility_passes": write_utility_passes,
                "accepted_writes": writes,
                "raw_structural_mass": raw_structural_mass,
                "mean_raw_structural_pressure": (
                    raw_structural_mass
                    / max(structural_observations, 1)
                ),
                "accepted_write_mean_utility": (
                    accepted_write_utility_sum / max(writes, 1)
                ),
                "harmful_write_fraction": harmful_writes / max(writes, 1),
                "utility_evaluated": len(self.controller_utility_replay),
                "controller_utility_stage_enabled": bool(self.controller.utility_stage_enabled),
                "controller_utility_statistics": {
                    "mean": self.controller.utility_mean.detach().cpu().tolist(),
                    "variance": self.controller.utility_variance.detach().cpu().tolist(),
                    "observations": self.controller.utility_observations.detach().cpu().tolist(),
                    "temperatures": self.controller.utility_temperatures.detach().cpu().tolist(),
                },
                "controller_replay_sign_counts": self.controller_utility_replay.sign_counts(),
                "controller_replay_action_statistics": replay_action_statistics,
                "write_ranking": dict(getattr(
                    self, "_last_write_ranking_metrics", {}
                )),
                "writes_per_sequence": writes / max(len(dataset), 1),
                "write_budget_utilization": writes / max(4 * len(dataset), 1),
                "global_update": global_update,
                "router_calibration": router_calibration,
                "topology_prune_enabled": topology_prune_enabled,
                "sleep": sleep_result,
                "phase_seconds": {
                    "wake": wake_seconds,
                    "global": global_seconds,
                    "sleep": sleep_seconds,
                    "epoch": epoch_seconds,
                },
                "resident_cache_hits": self._resident_cache_hits,
                "resident_cache_misses": self._resident_cache_misses,
                "cuda_peak_memory_mb": cuda_peak_memory_mb,
                "full_objective": (
                    wake_loss
                    + global_update["loss"]
                ),
                "leaf_ids": list(self.tree.leaf_ids),
            }
            self.history.append(epoch_result)
            self.completed_epochs = epoch
            self.save_checkpoint(self.training_config.checkpoint_path, epoch=epoch)
            if validation_dataset:
                calibration = None
                if self.training_config.controller_only_finetune:
                    calibration = self._calibrate_controller_checkpoint(
                        self.training_config.checkpoint_path,
                        validation_dataset,
                    )
                    # Persist calibrated thresholds before measuring frozen NLL.
                    self.save_checkpoint(
                        self.training_config.checkpoint_path, epoch=epoch
                    )
                validation = self._validate_controller_checkpoint(
                    self.training_config.checkpoint_path,
                    validation_dataset,
                )
                selected_utility = (
                    {} if calibration is None
                    else calibration.get("selected_utility", {})
                )
                calibrated_writes = (
                    epoch_result["writes_per_sequence"]
                    if calibration is None
                    else calibration.get("writes_per_sequence", 0.0)
                )
                trained_heads = set(self.training_config.controller_train_heads)
                write_only = trained_heads == {"write"}
                online_interval = validation.get("online_gain_bootstrap_95ci") or ()
                write_rollout_passed = bool(
                    not write_only
                    or (
                        validation.get("online_gain", 0.0) > 1e-6
                        and len(online_interval) >= 2
                        and float(online_interval[0]) > 0.0
                        and validation.get("online_harmful_fraction", 1.0) < 0.45
                        and validation.get("writes_per_sequence", float("inf")) <= 2.0
                        and validation.get("full_online_accuracy", 0.0)
                        >= validation.get("full_frozen_accuracy", 0.0)
                        - 1.0 / max(sum(len(row["times"]) for row in validation_dataset), 1)
                        and validation.get("full_online_macro_f1", 0.0)
                        >= validation.get("full_frozen_macro_f1", 0.0) - 0.001
                    )
                )
                if self.training_config.controller_write_ranking:
                    write_rollout_passed = bool(
                        write_rollout_passed
                        and calibration is not None
                        and not calibration.get("fallback_to_baseline", False)
                    )
                expected_frozen_sha = getattr(
                    self, "_base_frozen_event_sha256", None
                )
                frozen_output_verified = bool(
                    expected_frozen_sha is None
                    or validation.get("full_frozen_event_sha256")
                    == expected_frozen_sha
                )
                validation.update({
                    "epoch": epoch,
                    "writes_per_sequence": calibrated_writes,
                    "controller_calibration": calibration,
                    "constraint_passed": bool(
                        validation["memory_gain"] >= 0.0
                        and calibrated_writes <= (
                            2.0 if self.training_config.controller_only_finetune else 4.0
                        )
                        and (
                            not self.training_config.controller_only_finetune
                            or all(
                                selected_utility.get(action, -1.0) >= 0.0
                                for action in trained_heads
                                if not (write_only and action == "write")
                            )
                        )
                        and write_rollout_passed
                        and frozen_output_verified
                    ),
                    "frozen_output_verified": frozen_output_verified,
                    "expected_frozen_event_sha256": expected_frozen_sha,
                })
                calibrated_thresholds = self.controller.calibration_thresholds.clone()
                calibrated_calibration = dict(self.controller_calibration)
                if (
                    self.training_config.controller_only_finetune
                    and not validation["constraint_passed"]
                ):
                    fallback = self.controller.calibration_dict()
                    for action in trained_heads:
                        fallback[f"{action}_threshold"] = 1.01
                    self.controller.set_calibration_thresholds(
                        fallback["retrieve_threshold"],
                        fallback["adapt_threshold"],
                        fallback["write_threshold"],
                    )
                    self.save_checkpoint(
                        self.training_config.checkpoint_path, epoch=epoch
                    )
                    closed = self._validate_controller_checkpoint(
                        self.training_config.checkpoint_path,
                        validation_dataset,
                    )
                    validation["fallback_actions_closed"] = closed
                    self.controller.calibration_thresholds.copy_(
                        calibrated_thresholds
                    )
                    self.save_checkpoint(
                        self.training_config.checkpoint_path, epoch=epoch
                    )
                self.validation_history.append(validation)
                current = self.best_validation
                selection_metric = "full_online" if write_only else "full_frozen"
                candidate_key = (
                    0 if validation["constraint_passed"] else 1,
                    (
                        validation[selection_metric]
                        if validation["constraint_passed"]
                        else validation.get("fallback_actions_closed", {}).get(
                            selection_metric, validation[selection_metric]
                        )
                    ),
                )
                current_key = (
                    0 if current and current["constraint_passed"] else 1,
                    (
                        current[selection_metric]
                        if current and current["constraint_passed"]
                        else (
                            current.get("fallback_actions_closed", {}).get(
                                selection_metric, current[selection_metric]
                            ) if current else float("inf")
                        )
                    ),
                )
                if current is None or candidate_key < current_key:
                    self.best_validation = dict(validation)
                    best_path = self.training_config.best_checkpoint_path
                    if best_path:
                        if validation["constraint_passed"]:
                            self.save_checkpoint(best_path, epoch=epoch)
                        elif not self.training_config.controller_write_ranking:
                            fallback = self.controller.calibration_dict()
                            for action in trained_heads:
                                fallback[f"{action}_threshold"] = 1.01
                            self.controller.set_calibration_thresholds(
                                fallback["retrieve_threshold"],
                                fallback["adapt_threshold"],
                                fallback["write_threshold"],
                            )
                            self.controller_calibration = {
                                **self.controller_calibration,
                                **fallback,
                                "fallback_actions_closed": True,
                            }
                            self.save_checkpoint(best_path, epoch=epoch)
                            self.controller.calibration_thresholds.copy_(
                                calibrated_thresholds
                            )
                            self.controller_calibration = calibrated_calibration
                epoch_result["validation"] = validation
                epoch_result["best_validation_epoch"] = self.best_validation["epoch"]
                if self.training_config.validation_history_path:
                    path = Path(self.training_config.validation_history_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({
                        "selection_rule": (
                            "lowest realized full_online NLL for Write-only training; "
                            "otherwise lowest calibrated full_frozen NLL; inactive "
                            "Controller heads and thresholds remain frozen"
                        ),
                        "best": self.best_validation,
                        "epochs": self.validation_history,
                    }, indent=2), encoding="utf-8")
                if self.training_config.controller_diagnostics_path:
                    path = Path(self.training_config.controller_diagnostics_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({
                        "controller_version": int(
                            self.controller.controller_version.detach().cpu().item()
                        ),
                        "best_epoch": self.best_validation["epoch"],
                        "replay_sign_counts": self.controller_utility_replay.sign_counts(),
                        "utility_temperatures": self.controller.utility_temperatures.detach().cpu().tolist(),
                        "epochs": [{
                            "epoch": row["epoch"],
                            "writes_per_sequence": row["writes_per_sequence"],
                            "write_budget_utilization": row["write_budget_utilization"],
                            "write_candidates": row["write_candidates"],
                            "write_probes": row["write_probes"],
                            "write_gate_passes": row["write_gate_passes"],
                            "write_utility_passes": row["write_utility_passes"],
                            "accepted_writes": row["accepted_writes"],
                            "accepted_write_mean_utility": row["accepted_write_mean_utility"],
                            "harmful_write_fraction": row["harmful_write_fraction"],
                            "gate_means": row["controller_gate_means"],
                            "gate_over_0_5_fraction": row[
                                "controller_gate_over_0_5_fraction"
                            ],
                            "gate_gradient_norms": row[
                                "controller_gate_gradient_norms"
                            ],
                            "utility_evaluated": row["utility_evaluated"],
                            "controller_replay_sign_counts": row["controller_replay_sign_counts"],
                            "controller_replay_action_statistics": row[
                                "controller_replay_action_statistics"
                            ],
                            "write_ranking": row.get("write_ranking", {}),
                        } for row in self.history],
                    }, indent=2), encoding="utf-8")
                # The first snapshot is what validation evaluates. Re-save the
                # last identity after selection so it also carries best metadata.
                self.save_checkpoint(
                    self.training_config.checkpoint_path, epoch=epoch
                )
            if verbose:
                sleep_actions = (
                    []
                    if sleep_result is None
                    else [
                        action["action"]
                        for action in sleep_result["transaction"]["actions"]
                    ]
                )
                sleep_mode = (
                    "none"
                    if sleep_result is None
                    else sleep_result.get("mode", "none")
                )
                sleep_residual = (
                    0.0
                    if sleep_result is None
                    else sleep_result.get("residual_energy", 0.0)
                )
                light_absorbed = (
                    0
                    if sleep_result is None
                    else sleep_result.get("light", {}).get(
                        "absorbed_leaves", 0
                    )
                )
                deep_probability = (
                    0.0
                    if sleep_result is None
                    else sleep_result.get("deep_gate", {}).get(
                        "probability", 0.0
                    )
                )
                unified_stats = (
                    {} if sleep_result is None
                    else sleep_result.get("unified_topology", {})
                )
                topology_prune_stats = (
                    {} if sleep_result is None
                    else sleep_result.get("topology_prune", {})
                )
                print(
                    f"[MemoryTree][{epoch:03d}/{final_epoch:03d}] "
                    f"wake/event={wake_loss:.6f} "
                    f"sleep_residual_utility={sleep_residual:.6f} "
                    f"writes={writes} memorize={memorizes} "
                    f"queue_split={queue_splits} "
                    f"wake_actions="
                    f"A{wake_action_counts[Action.ASSIMILATE.value]}/"
                    f"R{wake_action_counts[Action.RETRIEVE.value]}/"
                    f"M{wake_action_counts[Action.MEMORIZE.value]}/"
                    f"Q{wake_action_counts[Action.QUEUE_SPLIT.value]} "
                    f"novelty={novelty_total / max(event_count, 1):.4f} "
                    f"weighted_sim={max_similarity_total / max(event_count, 1):.4f} "
                    f"wm_grad_max={max_gradient_norm:.3e} "
                    f"global/event="
                    f"{global_update['prediction_nll']:.6f} "
                    f"mix_diag={global_update['likelihood_mixture']:.4f} "
                    f"global_steps={global_update['optimizer_steps']} "
                    f"prior_KL={global_update['prior_kl']:.3e} "
                    f"seq_H_cond="
                    f"{global_update['conditional_entropy']:.4f} "
                    f"seq_H_marg="
                    f"{global_update['marginal_entropy']:.4f} "
                    f"seq_MI={global_update['mutual_information']:.3e} "
                    f"post_KL_diag="
                    f"{global_update['posterior_kl']:.3e} "
                    f"branch_distill="
                    f"{global_update['branch_distill']:.3e} "
                    f"probe="
                    f"{global_update['regional_probe_loss']:.3e}/"
                    f"{global_update['regional_probe_expand_probability']:.3f}"
                    f"->{global_update['regional_probe_expand_target']:.3f}/"
                    f"{global_update['regional_probe_refinement_gain']:.3f}/"
                    f"{global_update['regional_probe_assignment_confidence']:.3f} "
                    f"frontier="
                    f"{frontier_size_total / max(event_count, 1):.2f}/"
                    f"{frontier_visited_total / max(event_count, 1):.2f}/"
                    f"{frontier_branch_total / max(event_count, 1):.2f} "
                    f"enc_gate="
                    f"{global_update['encoder_route_gate']:.3f} "
                    f"enc_scale="
                    f"{global_update['encoder_route_grad_scale']:.3f} "
                    f"teacher_conf="
                    f"{global_update['teacher_confidence']:.3f} "
                    f"teacher_JS="
                    f"{global_update['teacher_student_js']:.3e} "
                    f"max_mass={max_leaf_mass:.4f} "
                    f"owner_hard="
                    f"{[sequence_owner_counts[node_id] for node_id in self.tree.all_node_ids]} "
                    f"mem_assign="
                    f"{[memory_assignment_counts[node_id] for node_id in self.tree.all_node_ids]} "
                    f"topology_prune={'on' if topology_prune_enabled else 'off'} "
                    f"leaves={len(self.tree.leaf_ids)} "
                    f"sleep_mode={sleep_mode} "
                    f"light_absorb={light_absorbed} "
                    f"deep_probability={deep_probability:.3f} "
                    f"topology_action="
                    f"{unified_stats.get('selected_action', 'null')} "
                    f"topology_gain="
                    f"{unified_stats.get('selected_gain', 0.0):.3e} "
                    f"topology_candidates="
                    f"{int(unified_stats.get('candidate_count', 0))} "
                    f"collapsed="
                    f"{int(topology_prune_stats.get('committed_count', 0))} "
                    f"topology_lambda={self.merge_lambda_T:.3e} "
                    f"time="
                    f"W{wake_seconds:.1f}/"
                    f"G{global_seconds:.1f}/"
                    f"S{sleep_seconds:.1f}s "
                    f"cache="
                    f"{self._resident_cache_hits}/"
                    f"{self._resident_cache_misses} "
                    f"cuda_peak={cuda_peak_memory_mb:.0f}MiB "
                    f"sleep_actions={sleep_actions}"
                )
        if (
            self.training_config.controller_write_ranking
            and self.training_config.best_checkpoint_path
            and not any(row.get("constraint_passed", False) for row in self.validation_history)
        ):
            base_path = Path(self.training_config.controller_base_checkpoint)
            best_path = Path(self.training_config.best_checkpoint_path)
            fallback = torch.load(base_path, map_location="cpu", weights_only=False)
            fallback["checkpoint_identity"] = {
                "path": str(best_path.resolve()), "role": "best"
            }
            fallback["controller_policy_revision"] = 4
            fallback["write_supervision"] = "sequence_relative_ranking"
            fallback["validation_selection"] = {
                "history": list(self.validation_history),
                "best": self.best_validation,
                "constraint_passed": False,
                "fallback_to_baseline": True,
                "selection_reason": "no ranking checkpoint improved realized rollout",
            }
            temporary = best_path.with_suffix(best_path.suffix + ".tmp")
            torch.save(fallback, temporary)
            temporary.replace(best_path)
        if self.training_config.plot_after_training and self.history:
            try:
                from Train.PlotTraining import save_training_diagnostics

                artifacts = save_training_diagnostics(
                    self.history,
                    self.training_config.checkpoint_path,
                    metrics_path=(
                        self.training_config.training_metrics_path
                    ),
                    plot_path=self.training_config.training_plot_path,
                )
                if verbose:
                    print(
                        "[Training Plot] "
                        f"metrics={artifacts['metrics']} "
                        f"figure={artifacts['plot']}"
                    )
            except Exception as error:
                # A completed model checkpoint remains valid even if an
                # optional plotting dependency or output path is unavailable.
                print(
                    "[Training Plot] failed after training completed: "
                    f"{type(error).__name__}: {error}"
                )
        return self.history
