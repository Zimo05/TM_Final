from collections.abc import Mapping
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .MemoryBank import (
    EventWindow,
    HawkesMemoryUpdate,
    MemoryBank,
    MemoryQueryNet,
    SmoothSparseRetriever,
    TreeMemoryRead,
    UpdateHawkesParameter,
)
from .SimilarityFeatures import local_recurrence_count


class TreeEpisodicMemory(nn.Module):
    """
    Tree-level adapter around ``MemoryBank``.

    Every tree node owns one local bank. All node banks share one trainable
    entmax retriever. A leaf-path correction is

        delta_path = sum_{n in path} sum_{r in M_n} alpha_(n,r) delta_(n,r).
    """

    def __init__(
        self,
        key_dim: int,
        num_event_types: int,
        num_basis: int = 1,
        capacity_per_node: int = 128,
        device: str = "cuda",
        init_gamma: float = 10.0,
        init_tau: float = 1.0,
        init_lambda_usage: float = 0.1,
        init_lambda_age: float = 0.01,
        query_input_dim: Optional[int] = None,
    ):
        super().__init__()
        if key_dim <= 0:
            raise ValueError("key_dim must be positive")

        self.key_dim = key_dim
        self.capacity_per_node = capacity_per_node
        self.parameter_update = UpdateHawkesParameter(
            num_event_types=num_event_types,
            num_basis=num_basis,
        )
        self.param_dim = self.parameter_update.param_dim
        self.query_net = (
            MemoryQueryNet(input_dim=query_input_dim, key_dim=key_dim)
            if query_input_dim is not None
            else None
        )
        self.retriever = SmoothSparseRetriever(
            init_gamma=init_gamma,
            init_tau=init_tau,
            init_lambda_usage=init_lambda_usage,
            init_lambda_age=init_lambda_age,
        )
        self.banks: Dict[str, MemoryBank] = {}
        # Lazily rebuilt GPU mirror for packed retrieval. Bank dictionaries
        # remain authoritative; tensor identity/version signatures invalidate
        # this cache after writes, pruning, sleep consolidation, or topology
        # replacement. Volatile retrieval-credit fields are deliberately not
        # part of the signature: packed retrieval only consumes keys, deltas,
        # quality, usage, and effective age. Merely advancing the logical age
        # clock does not rebuild it because the age offset is applied as one
        # broadcast below.
        self._packed_mirror_signature = None
        self._packed_mirror = None
        self._packed_mirror_rebuilds = 0
        # Logical chronological event counter. Advancing age is O(1); each
        # bank stores the clock at which its age tensor was last materialized.
        self._age_clock = 0
        self.register_buffer(
            "_device_anchor",
            torch.empty(0, device=torch.device(device)),
            persistent=False,
        )
        self.retriever.to(self.device)
        if self.query_net is not None:
            self.query_net.to(self.device)

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def _apply(self, fn):
        """Make module.to(...) move dynamic memory tensors as well."""
        super()._apply(fn)
        for bank in self.banks.values():
            bank.keys = fn(bank.keys)
            bank.deltas = fn(bank.deltas)
            bank.write_quality = fn(bank.write_quality)
            bank.queue_weight = fn(bank.queue_weight)
            bank.usage = fn(bank.usage)
            bank.cycle_usage = fn(bank.cycle_usage)
            bank.stale_cycles = fn(bank.stale_cycles)
            bank.age = fn(bank.age)
            bank.device = bank.keys.device
            for window in bank.windows:
                if window is not None:
                    window.times = fn(window.times)
                    window.types = fn(window.types)
                    for cache_name in (
                        "event_time_features",
                        "hawkes_history_stats",
                        "hawkes_interval_stats",
                    ):
                        cached = getattr(window, cache_name, None)
                    if cached is not None:
                        setattr(window, cache_name, fn(cached))
        self._packed_mirror_signature = None
        self._packed_mirror = None
        return self

    @staticmethod
    def _tensor_state_signature(tensor: Tensor) -> tuple[int, int]:
        return id(tensor), int(tensor._version)

    def _bank_state_signature(
        self,
        node_ids: Sequence[str],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple:
        rows = []
        for node_id in node_ids:
            bank = self.banks.get(node_id)
            if bank is None:
                rows.append((node_id, None))
                continue
            rows.append((
                node_id,
                id(bank),
                len(bank),
                bank._age_reference_clock,
                self._tensor_state_signature(bank.keys),
                self._tensor_state_signature(bank.deltas),
                self._tensor_state_signature(bank.write_quality),
                self._tensor_state_signature(bank.usage),
                self._tensor_state_signature(bank.age),
                bank.capacity,
            ))
        return (
            tuple(rows),
            str(device),
            str(dtype),
            self.capacity_per_node,
        )

    def _packed_bank_mirror(
        self,
        node_ids: Sequence[str],
        reference: Tensor,
    ) -> Dict[str, Tensor]:
        """Return a persistent padded mirror of all requested node banks.

        A structural collapse may temporarily create an over-capacity parent
        bank. Keep those rows retrievable until the independent memory-prune
        transaction reconciles the bank back to its ordinary capacity.
        """
        signature = self._bank_state_signature(
            node_ids,
            dtype=reference.dtype,
            device=reference.device,
        )
        if (
            signature == self._packed_mirror_signature
            and self._packed_mirror is not None
        ):
            return self._packed_mirror

        node_count = len(node_ids)
        capacity = max(
            self.capacity_per_node,
            max(
                (len(self.banks[node_id]) for node_id in node_ids
                 if node_id in self.banks),
                default=0,
            ),
        )
        keys = reference.new_zeros(node_count, capacity, self.key_dim)
        deltas = reference.new_zeros(node_count, capacity, self.param_dim)
        quality = reference.new_zeros(node_count, capacity)
        usage = reference.new_zeros(node_count, capacity)
        base_age = reference.new_zeros(node_count, capacity)
        age_reference = torch.zeros(
            node_count,
            dtype=torch.long,
            device=reference.device,
        )
        valid = torch.zeros(
            node_count,
            capacity,
            dtype=torch.bool,
            device=reference.device,
        )
        for node_index, node_id in enumerate(node_ids):
            bank = self.banks.get(node_id)
            if bank is None or len(bank) == 0:
                continue
            width = len(bank)
            keys[node_index, :width] = bank.keys[:width]
            deltas[node_index, :width] = bank.deltas[:width]
            quality[node_index, :width] = bank.write_quality[:width]
            usage[node_index, :width] = bank.usage[:width]
            base_age[node_index, :width] = bank.age[:width]
            age_reference[node_index] = bank._age_reference_clock
            valid[node_index, :width] = True
        self._packed_mirror = {
            "keys": keys,
            "deltas": deltas,
            "quality": quality,
            "usage": usage,
            "base_age": base_age,
            "age_reference": age_reference,
            "valid": valid,
        }
        self._packed_mirror_signature = signature
        self._packed_mirror_rebuilds += 1
        return self._packed_mirror

    def get_bank(self, node_id: str) -> MemoryBank:
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if node_id not in self.banks:
            bank = MemoryBank(
                device=str(self.device),
                key_dim=self.key_dim,
                param_dim=self.param_dim,
                capacity=self.capacity_per_node,
            )
            bank._age_reference_clock = self._age_clock
            self.banks[node_id] = bank
        return self.banks[node_id]

    def sync_nodes(self, node_ids: Iterable[str], remove_stale: bool = False) -> None:
        active_ids = set(node_ids)
        for node_id in active_ids:
            self.get_bank(node_id)
        if remove_stale:
            for node_id in set(self.banks).difference(active_ids):
                del self.banks[node_id]

    def add_memory(
        self,
        node_id: str,
        key: Tensor,
        delta_theta: Tensor,
        window: Optional[EventWindow] = None,
        write_quality: float | Tensor = 1.0,
        queue_weight: float | Tensor = 0.0,
    ) -> None:
        bank = self.get_bank(node_id)
        # A newly written item must have age zero at the current event clock,
        # while existing rows retain their accumulated effective age.
        bank.materialize_age(self._age_clock)
        bank.add(
            key=key,
            delta_theta=delta_theta,
            window=window,
            write_quality=write_quality,
            queue_weight=queue_weight,
        )

    def read_nodes(
        self,
        query: Tensor,
        node_ids: Iterable[str],
        update_state: bool = True,
        *,
        keep_gate_by_node: Optional[Mapping[str, Tensor]] = None,
        null_logit: Optional[float | Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Dict[str, Tensor]]]:
        """
        Read every requested node once.

        When evaluating several candidate paths, pass the union of their node
        IDs here once, then call ``aggregate_path`` for each candidate. This
        avoids updating usage repeatedly for shared ancestors.
        """
        delta_by_node: Dict[str, Tensor] = {}
        info_by_node: Dict[str, Dict[str, Tensor]] = {}
        ordered_node_ids = list(dict.fromkeys(node_ids))
        active_node_ids = []
        active_banks = []
        for node_id in ordered_node_ids:
            bank = self.banks.get(node_id)
            if bank is None or len(bank) == 0:
                delta_by_node[node_id] = query.new_zeros(self.param_dim)
                info_by_node[node_id] = {}
                continue
            if query.device != bank.device:
                raise ValueError(
                    f"query is on {query.device}, memory bank is on {bank.device}"
                )
            active_node_ids.append(node_id)
            active_banks.append(bank)

        if not active_banks:
            return delta_by_node, info_by_node

        # Each bank already owns aligned contiguous tensors. Pad only the
        # frontier path-union so all bank similarities and residual reductions
        # run in one batched GPU call. The mask keeps rows independent.
        row_count = len(active_banks)
        lengths = [len(bank) for bank in active_banks]
        max_width = max(lengths)
        widths = torch.tensor(
            lengths,
            device=query.device,
            dtype=torch.long,
        )

        def pad_matrix(attribute: str) -> Tensor:
            return torch.stack(
                [
                    torch.nn.functional.pad(
                        getattr(bank, attribute),
                        (0, 0, 0, max_width - len(bank)),
                    )
                    for bank in active_banks
                ],
                dim=0,
            )

        def pad_vector(attribute: str) -> Tensor:
            return torch.stack(
                [
                    torch.nn.functional.pad(
                        getattr(bank, attribute),
                        (0, max_width - len(bank)),
                    )
                    for bank in active_banks
                ],
                dim=0,
            )

        age = torch.stack(
            [
                torch.nn.functional.pad(
                    bank.effective_age(self._age_clock),
                    (0, max_width - len(bank)),
                )
                for bank in active_banks
            ],
            dim=0,
        )
        valid_mask = (
            torch.arange(max_width, device=query.device).unsqueeze(0)
            < widths.unsqueeze(1)
        )
        batched_delta, batched_info = self.retriever.forward_batched(
            query=query.reshape(1, -1).expand(row_count, -1),
            keys=pad_matrix("keys"),
            deltas=pad_matrix("deltas"),
            usage=pad_vector("usage"),
            age=age,
            valid_mask=valid_mask,
            write_quality=pad_vector("write_quality"),
            keep_gate=(
                None
                if keep_gate_by_node is None
                else torch.stack([
                    torch.nn.functional.pad(
                        keep_gate_by_node.get(
                            node_id,
                            bank.keys.new_ones(len(bank)),
                        ).to(device=query.device, dtype=query.dtype),
                        (0, max_width - len(bank)),
                    )
                    for node_id, bank in zip(active_node_ids, active_banks)
                ])
            ),
            null_logit=null_logit,
        )

        vector_fields = (
            "sim",
            "rho",
            "sparse_rho",
            "dense_rho",
            "alpha",
        )
        shared_fields = (
            "gamma",
            "tau",
            "lambda_usage",
            "lambda_age",
        )
        for row_index, (node_id, bank) in enumerate(
            zip(active_node_ids, active_banks)
        ):
            width = len(bank)
            info = {
                name: batched_info[name][row_index, :width]
                for name in vector_fields
            }
            info.update({
                name: batched_info[name]
                for name in shared_fields
            })
            info["effective_k"] = batched_info["effective_k"][row_index]
            info["null_alpha"] = batched_info["null_alpha"][row_index]
            delta_by_node[node_id] = batched_delta[row_index]
            info_by_node[node_id] = info
            if update_state:
                bank.cycle_usage.add_(info["alpha"].to(bank.device))

        # Preserve caller-visible node order, including empty banks.
        return (
            {
                node_id: delta_by_node[node_id]
                for node_id in ordered_node_ids
            },
            {
                node_id: info_by_node[node_id]
                for node_id in ordered_node_ids
            },
        )

    def read_packed(
        self,
        query: Tensor,
        node_indices: Tensor,
        node_mask: Tensor,
        node_ids: Sequence[str],
        *,
        update_state: bool = True,
        keep_gate: Optional[Tensor] = None,
        null_logit: Optional[float | Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Retrieve all ``prefix × visited-node × bank-row`` entries at once.

        Persistent dictionaries remain the source of truth. This method builds
        the fixed ``[node, capacity, *]`` mirror used by Wake, gathers the
        visited rows, and applies the existing batched retriever semantics.
        """
        if query.ndim != 2 or query.size(-1) != self.key_dim:
            raise ValueError("query must have shape [N, key_dim]")
        if node_indices.ndim != 2 or node_mask.shape != node_indices.shape:
            raise ValueError("node_indices/node_mask must have shape [N, V]")
        if node_mask.dtype != torch.bool:
            raise ValueError("node_mask must be boolean")
        if node_indices.size(0) != query.size(0):
            raise ValueError("query and visited-node rows must align")
        if len(node_ids) == 0:
            raise ValueError("node_ids cannot be empty")

        node_count = len(node_ids)
        device = query.device
        mirror = self._packed_bank_mirror(node_ids, query)
        capacity = int(mirror["keys"].size(1))
        if keep_gate is not None:
            if keep_gate.shape != (node_count, capacity):
                raise ValueError(
                    "keep_gate must have shape [len(node_ids), capacity]"
                )
            keep_gate = keep_gate.to(device=device, dtype=query.dtype)
            if null_logit is None:
                raise ValueError("null_logit is required with keep_gate")
        keys = mirror["keys"]
        deltas = mirror["deltas"]
        quality = mirror["quality"]
        usage = mirror["usage"]
        valid = mirror["valid"]
        age_offset = (
            self._age_clock - mirror["age_reference"]
        ).to(query.dtype)
        age = (
            mirror["base_age"]
            + age_offset[:, None] * valid.to(query.dtype)
        )

        safe_nodes = node_indices.clamp_min(0)
        flat_nodes = safe_nodes.reshape(-1)
        gathered_valid = (
            valid.index_select(0, flat_nodes)
            & node_mask.reshape(-1, 1)
        )
        flat_query = query[:, None, :].expand(
            -1, node_indices.size(1), -1
        ).reshape(-1, self.key_dim)
        active = gathered_valid.any(dim=-1)
        active_rows = torch.nonzero(active, as_tuple=False).flatten()
        active_nodes = flat_nodes.index_select(0, active_rows)
        active_valid = gathered_valid.index_select(0, active_rows)
        flat_delta = query.new_zeros(
            flat_query.size(0), self.param_dim
        )
        alpha = query.new_zeros(flat_query.size(0), capacity)
        similarity = query.new_zeros(flat_query.size(0), capacity)
        effective_k = torch.zeros(
            flat_query.size(0), dtype=torch.long, device=device
        )
        null_alpha = query.new_zeros(flat_query.size(0))
        # Gather large key/delta rows only after removing inactive visits.
        # The former gather-then-mask flow held two expanded [R, M, P] delta
        # tensors at the same time.  Indexing by active node once preserves
        # the zero-row, synchronization-free path while halving that peak.
        retrieved, retrieval_info = self.retriever.forward_batched(
            query=flat_query.index_select(0, active_rows),
            keys=keys.index_select(0, active_nodes),
            # Keep the large [node, capacity, param] residual mirror shared.
            # ``active_nodes`` tells the retriever which bank each visit uses,
            # avoiding a multi-GiB [visit, capacity, param] materialization.
            deltas=deltas,
            row_bank_indices=active_nodes,
            usage=usage.index_select(0, active_nodes),
            age=age.index_select(0, active_nodes),
            valid_mask=active_valid,
            write_quality=quality.index_select(0, active_nodes),
            keep_gate=(
                None
                if keep_gate is None
                else keep_gate.index_select(0, active_nodes)
            ),
            null_logit=null_logit,
        )
        flat_delta = flat_delta.index_copy(0, active_rows, retrieved)
        alpha.index_copy_(0, active_rows, retrieval_info["alpha"])
        similarity.index_copy_(0, active_rows, retrieval_info["sim"])
        effective_k.index_copy_(
            0, active_rows, retrieval_info["effective_k"]
        )
        null_alpha.index_copy_(
            0, active_rows, retrieval_info["null_alpha"]
        )

        if update_state:
            with torch.no_grad():
                credit = query.new_zeros(node_count, capacity)
                credit.index_add_(
                    0,
                    active_nodes,
                    retrieval_info["alpha"],
                )
                for node_index, node_id in enumerate(node_ids):
                    bank = self.banks.get(node_id)
                    if bank is not None and len(bank):
                        bank.cycle_usage.add_(
                            credit[node_index, : len(bank)].to(bank.device)
                        )

        shape = (*node_indices.shape, self.param_dim)
        info = {
            "alpha": alpha.reshape(*node_indices.shape, capacity).detach(),
            "similarity": similarity.reshape(
                *node_indices.shape, capacity
            ).detach(),
            "effective_k": effective_k.reshape(node_indices.shape).detach(),
            "null_alpha": null_alpha.reshape(node_indices.shape).detach(),
            "valid_mask": gathered_valid.reshape(
                *node_indices.shape, capacity
            ).detach(),
        }
        return flat_delta.reshape(shape), info

    def novelty_count_packed(
        self,
        query: Tensor,
        node_indices: Tensor,
        node_ids: Sequence[str],
        *,
        temperature: float,
        count_exponent: float,
        eps: float,
        count_similarity_low: float = 0.35,
        count_similarity_high: float = 0.65,
        count_topk: Optional[int] = None,
        count_saturation: float = 3.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Vectorized novelty and local recurrence count for owner nodes."""
        if query.ndim != 2 or query.size(-1) != self.key_dim:
            raise ValueError("query must have shape [B, key_dim]")
        if node_indices.shape != (query.size(0),):
            raise ValueError("node_indices must have shape [B]")
        if temperature <= 0.0 or count_exponent < 1.0 or eps <= 0.0:
            raise ValueError("invalid novelty/count hyperparameters")

        mirror = self._packed_bank_mirror(node_ids, query)
        keys = mirror["keys"].index_select(0, node_indices)
        valid = mirror["valid"].index_select(0, node_indices)
        normalized_query = F.normalize(query, dim=-1)
        normalized_keys = F.normalize(keys, dim=-1)
        similarity = torch.einsum(
            "bcd,bd->bc",
            normalized_keys,
            normalized_query,
        )
        logits = temperature * similarity
        row_has_memory = valid.any(dim=-1)
        safe_logits = logits.masked_fill(~valid, -torch.inf)
        safe_logits = torch.where(
            row_has_memory[:, None],
            safe_logits,
            torch.zeros_like(safe_logits),
        )
        beta = F.softmax(safe_logits, dim=-1) * valid.to(logits.dtype)
        beta = beta / beta.sum(dim=-1, keepdim=True).clamp_min(eps)
        weighted_similarity = (beta * similarity).sum(dim=-1)
        weighted_similarity = torch.where(
            row_has_memory,
            weighted_similarity,
            weighted_similarity.new_full((), -1.0),
        )
        novelty = (1.0 - weighted_similarity) / 2.0

        normalized_count, _ = local_recurrence_count(
            similarity,
            valid_mask=valid,
            similarity_low=count_similarity_low,
            similarity_high=count_similarity_high,
            exponent=count_exponent,
            topk=count_topk,
            saturation=count_saturation,
            eps=eps,
        )
        normalized_count = torch.where(
            row_has_memory,
            normalized_count,
            torch.zeros_like(normalized_count),
        )
        return novelty, normalized_count, weighted_similarity

    def aggregate_path(
        self,
        path_node_ids: Sequence[str],
        delta_by_node: Dict[str, Tensor],
    ) -> Tensor:
        if not path_node_ids:
            raise ValueError("path_node_ids must contain at least one node")

        missing = [
            node_id
            for node_id in path_node_ids
            if node_id not in delta_by_node
        ]
        if missing:
            raise KeyError(f"path nodes were not read: {missing}")

        total_delta = delta_by_node[path_node_ids[0]].new_zeros(self.param_dim)
        for node_id in path_node_ids:
            total_delta = total_delta + delta_by_node[node_id]
        return total_delta

    @torch.no_grad()
    def credit_retrieval(
        self,
        info_by_batch: Sequence[Mapping[str, Dict[str, Tensor]]],
        leaf_paths: Sequence[Sequence[str]],
        routing_weights: Tensor,
        retrieval_probability: Tensor,
    ) -> None:
        """Assign post-event usage credit scaled by ``p_t(RETRIEVE)``.

        Retrieval for the current prediction has already happened before the
        event surprise is available. This method implements Eq. (21) without
        changing that prediction retrospectively.
        """
        if routing_weights.ndim == 1:
            routing_weights = routing_weights.unsqueeze(0)
        probabilities = torch.as_tensor(
            retrieval_probability,
            device=routing_weights.device,
            dtype=routing_weights.dtype,
        ).reshape(-1)
        if probabilities.numel() == 1 and routing_weights.size(0) != 1:
            probabilities = probabilities.expand(routing_weights.size(0))
        if probabilities.shape != (routing_weights.size(0),):
            raise ValueError(
                "retrieval_probability must contain one scalar per batch row"
            )
        if len(info_by_batch) != routing_weights.size(0):
            raise ValueError("info_by_batch and routing_weights must align")

        for batch_index, selected_info in enumerate(info_by_batch):
            credit = probabilities[batch_index].clamp(0.0, 1.0)
            for node_id in selected_info:
                info = selected_info[node_id]
                if "alpha" not in info:
                    continue
                node_mass = routing_weights.new_zeros(())
                for leaf_index, path in enumerate(leaf_paths):
                    if node_id in path:
                        node_mass = (
                            node_mass
                            + routing_weights[batch_index, leaf_index]
                        )
                bank = self.banks.get(node_id)
                if bank is not None and len(bank) > 0:
                    bank.cycle_usage.add_(
                        info["alpha"][: len(bank)].to(bank.device)
                        * node_mass.to(bank.device)
                        * credit.to(bank.device)
                    )

    @torch.no_grad()
    def credit_retrieval_packed(
        self,
        *,
        alpha: Tensor,
        visited_node_indices: Tensor,
        visited_node_mask: Tensor,
        path_incidence: Tensor,
        routing_weights: Tensor,
        retrieval_probability: Tensor,
        node_ids: Sequence[str],
        cycle_usage_accumulator: Optional[Tensor] = None,
    ) -> None:
        """Packed equivalent of :meth:`credit_retrieval`.

        ``path_incidence[b,k,v]`` says whether visited node ``v`` belongs to
        frontier path ``k``. This replaces per-event Python dictionaries and
        path membership loops with one GPU contraction and one node scatter.
        """
        if routing_weights.ndim == 1:
            routing_weights = routing_weights.unsqueeze(0)
        if (
            alpha.ndim != 3
            or visited_node_indices.ndim != 2
            or visited_node_mask.shape != visited_node_indices.shape
            or path_incidence.shape[:1] != routing_weights.shape[:1]
            or path_incidence.shape[1] != routing_weights.shape[1]
            or path_incidence.shape[2] != visited_node_indices.shape[1]
            or alpha.shape[:2] != visited_node_indices.shape
        ):
            raise ValueError("packed retrieval-credit tensors are misaligned")
        probabilities = torch.as_tensor(
            retrieval_probability,
            device=routing_weights.device,
            dtype=routing_weights.dtype,
        ).reshape(-1)
        if probabilities.numel() == 1 and routing_weights.size(0) != 1:
            probabilities = probabilities.expand(routing_weights.size(0))
        if probabilities.shape != (routing_weights.size(0),):
            raise ValueError(
                "retrieval_probability must contain one scalar per batch row"
            )

        node_mass = torch.einsum(
            "bk,bkv->bv",
            routing_weights,
            path_incidence.to(routing_weights.dtype),
        )
        row_credit = (
            alpha
            * node_mass.unsqueeze(-1).to(alpha)
            * probabilities[:, None, None].clamp(0.0, 1.0).to(alpha)
        )
        capacity = alpha.size(-1)
        node_credit = alpha.new_zeros(len(node_ids), capacity)
        valid = visited_node_mask.reshape(-1)
        safe_nodes = visited_node_indices.clamp_min(0).reshape(-1)
        node_credit.index_add_(
            0,
            safe_nodes[valid],
            row_credit.reshape(-1, capacity)[valid],
        )
        if cycle_usage_accumulator is not None:
            if cycle_usage_accumulator.shape != node_credit.shape:
                raise ValueError(
                    "cycle_usage_accumulator must have shape "
                    "[len(node_ids), capacity]"
                )
            cycle_usage_accumulator.add_(node_credit)
            return
        self.apply_cycle_usage_credit(node_credit, node_ids)

    @torch.no_grad()
    def apply_cycle_usage_credit(
        self,
        node_credit: Tensor,
        node_ids: Sequence[str],
    ) -> None:
        """Apply one accumulated packed-retrieval credit per memory bank.

        Wake can read the same bank at many causal time positions. Keeping the
        reduction on device and applying it once after the wavefront preserves
        the original additive state update while avoiding one ``add_`` launch
        per position and leaving the packed mirror reusable throughout the
        transaction.
        """
        if node_credit.ndim != 2 or node_credit.size(0) != len(node_ids):
            raise ValueError(
                "node_credit must have shape [len(node_ids), capacity]"
            )
        for node_index, node_id in enumerate(node_ids):
            bank = self.banks.get(node_id)
            if bank is not None and len(bank) > 0:
                bank.cycle_usage.add_(
                    node_credit[node_index, : len(bank)].to(bank.device)
                )

    def retrieve_path(
        self,
        query: Tensor,
        path_node_ids: Sequence[str],
        update_state: bool = True,
    ) -> TreeMemoryRead:
        delta_by_node, info_by_node = self.read_nodes(
            query=query,
            node_ids=path_node_ids,
            update_state=update_state,
        )
        return TreeMemoryRead(
            delta_theta=self.aggregate_path(path_node_ids, delta_by_node),
            delta_by_node=delta_by_node,
            info_by_node=info_by_node,
        )

    def apply_update(
        self,
        raw_mu: Tensor,
        raw_W: Tensor,
        delta_theta: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return self.parameter_update.apply_delta(
            raw_mu=raw_mu,
            raw_W=raw_W,
            delta_theta=delta_theta,
        )

    def retrieve_and_update(
        self,
        query: Tensor,
        path_node_ids: Sequence[str],
        raw_mu: Tensor,
        raw_W: Tensor,
        update_state: bool = True,
    ) -> HawkesMemoryUpdate:
        read = self.retrieve_path(
            query=query,
            path_node_ids=path_node_ids,
            update_state=update_state,
        )
        mu, W = self.apply_update(
            raw_mu=raw_mu,
            raw_W=raw_W,
            delta_theta=read.delta_theta,
        )
        return HawkesMemoryUpdate(
            mu=mu,
            W=W,
            delta_theta=read.delta_theta,
            read=read,
        )

    @torch.no_grad()
    def step_age(self, steps: int = 1) -> None:
        """Advance chronological age once without launching per-bank kernels."""
        if steps < 0:
            raise ValueError("age steps must be non-negative")
        self._age_clock += int(steps)

    @torch.no_grad()
    def materialize_all_ages(self) -> None:
        """Expose current ages to sleep/topology code that reads ``bank.age``."""
        for bank in self.banks.values():
            bank.materialize_age(self._age_clock)

    @torch.no_grad()
    def consolidate_sleep_cycle(
        self,
        usage_decay: float = 0.95,
        effective_usage_threshold: float = 0.0,
    ) -> None:
        """Consolidate usage/staleness exactly once for every sleep cycle."""
        for bank in self.banks.values():
            bank.consolidate_sleep_cycle(
                usage_decay=usage_decay,
                effective_usage_threshold=effective_usage_threshold,
            )

    def read_node_with_virtual_item(
        self,
        query: Tensor,
        node_id: str,
        *,
        key: Tensor,
        delta: Tensor,
        write_quality: float | Tensor,
        virtual_usage: float | Tensor = 1.0,
        virtual_age: float | Tensor = 0.0,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Read one bank with an appended ephemeral row and no state mutation."""
        bank = self.banks.get(node_id)
        if bank is None or len(bank) == 0:
            keys = key.reshape(1, -1).to(query)
            deltas = delta.reshape(1, -1).to(query)
            usage = query.new_tensor([float(torch.as_tensor(virtual_usage))])
            age = query.new_tensor([float(torch.as_tensor(virtual_age))])
            qualities = query.new_tensor([float(torch.as_tensor(write_quality))])
        else:
            keys = torch.cat([bank.keys.to(query), key.reshape(1, -1).to(query)], 0)
            deltas = torch.cat([bank.deltas.to(query), delta.reshape(1, -1).to(query)], 0)
            usage = torch.cat([
                bank.usage.to(query), query.new_tensor([float(torch.as_tensor(virtual_usage))])
            ])
            age = torch.cat([
                bank.effective_age(self._age_clock).to(query),
                query.new_tensor([float(torch.as_tensor(virtual_age))]),
            ])
            qualities = torch.cat([
                bank.write_quality.to(query),
                query.new_tensor([float(torch.as_tensor(write_quality))]),
            ])
        return self.retriever(
            query=query, keys=keys, deltas=deltas, usage=usage, age=age,
            write_quality=qualities,
        )

    def read_node_without_item(
        self,
        query: Tensor,
        node_id: str,
        *,
        key: Tensor,
        delta: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Read a bank while excluding one exact candidate, without mutation.

        This is used only for post-admission v6 counterfactual diagnostics.
        The most recently appended exact match is removed so older duplicate
        memories, if any, remain part of the baseline.
        """
        bank = self.banks.get(node_id)
        if bank is None or len(bank) == 0:
            return query.new_zeros(self.param_dim), {}
        normalized_key = F.normalize(key.detach().reshape(1, -1).to(bank.keys), dim=-1)
        candidate_delta = delta.detach().reshape(1, -1).to(bank.deltas)
        matches = torch.isclose(
            bank.keys, normalized_key, rtol=1e-5, atol=1e-7
        ).all(dim=-1) & torch.isclose(
            bank.deltas, candidate_delta, rtol=1e-5, atol=1e-7
        ).all(dim=-1)
        indices = torch.nonzero(matches, as_tuple=False).flatten()
        if indices.numel() == 0:
            result, info = self.read_nodes(query, [node_id], update_state=False)
            return result[node_id], info[node_id]
        removed = int(indices[-1].detach().cpu())
        keep = torch.ones(len(bank), device=bank.device, dtype=torch.bool)
        keep[removed] = False
        if not bool(keep.any()):
            return query.new_zeros(self.param_dim), {
                "excluded_index": query.new_tensor(removed, dtype=torch.long)
            }
        result, info = self.retriever(
            query=query,
            keys=bank.keys[keep].to(query),
            deltas=bank.deltas[keep].to(query),
            usage=bank.usage[keep].to(query),
            age=bank.effective_age(self._age_clock)[keep].to(query),
            write_quality=bank.write_quality[keep].to(query),
        )
        info = dict(info)
        info["excluded_index"] = query.new_tensor(removed, dtype=torch.long)
        return result, info

    def get_extra_state(self):
        """Save effective ages so checkpoints remain format-compatible."""
        return {
            node_id: {
                "keys": bank.keys.detach().cpu(),
                "deltas": bank.deltas.detach().cpu(),
                "write_quality": bank.write_quality.detach().cpu(),
                "queue_weight": bank.queue_weight.detach().cpu(),
                "usage": bank.usage.detach().cpu(),
                "cycle_usage": bank.cycle_usage.detach().cpu(),
                "stale_cycles": bank.stale_cycles.detach().cpu(),
                "age": bank.effective_age(self._age_clock).detach().cpu(),
                "windows": bank.windows,
                # A topology collapse may deliberately retain more rows than
                # the ordinary per-node budget until the next independent
                # memory-prune transaction.  Preserve that temporary capacity
                # across checkpoints so the restored bank is not immediately
                # inconsistent with its own capacity before reconciliation.
                "capacity": int(bank.capacity),
            }
            for node_id, bank in self.banks.items()
        }

    def set_extra_state(self, state) -> None:
        self.banks = {}
        # Stored ages are already effective at checkpoint time; rebasing the
        # logical clock to zero preserves all future age differences.
        self._age_clock = 0
        for node_id, bank_state in state.items():
            bank = self.get_bank(node_id)
            bank.keys = bank_state["keys"].to(self.device)
            bank.deltas = bank_state["deltas"].to(self.device)
            memory_count = bank_state["keys"].shape[0]
            bank.write_quality = bank_state.get(
                "write_quality",
                torch.ones(memory_count),
            ).to(self.device)
            bank.queue_weight = bank_state.get(
                "queue_weight",
                torch.zeros(memory_count),
            ).to(self.device)
            bank.usage = bank_state["usage"].to(self.device)
            # Backward-compatible defaults for checkpoints written before
            # cycle-level pruning state was introduced.
            bank.cycle_usage = bank_state.get(
                "cycle_usage", torch.zeros_like(bank_state["usage"])
            ).to(self.device)
            bank.stale_cycles = bank_state.get(
                "stale_cycles", torch.zeros_like(bank_state["usage"])
            ).to(self.device)
            bank.age = bank_state["age"].to(self.device)
            bank._age_reference_clock = self._age_clock
            bank.windows = list(bank_state["windows"])
            bank.capacity = max(
                int(bank_state.get("capacity", self.capacity_per_node)),
                int(memory_count),
            )
