from collections.abc import Mapping
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

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
    effective_hawkes_law_key,
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
        # Law identities use the compact signed-hash representation, while
        # parameter storage remains the full physical Hawkes width.
        self.law_dim = int(key_dim)
        self._prototype_policy = {
            "duplicate_threshold": 0.98,
            "mode_threshold": 0.90,
            "mode_capacity": 12,
            "ema_beta_min": 0.01,
            "ema_beta_max": 0.25,
            "retention_support_weight": 1.0,
            "retention_usage_weight": 0.5,
            "retention_stale_weight": 1.0,
            "retention_age_weight": 0.1,
            "adaptive_history_size": 64,
            "adaptive_min_samples": 8,
            "duplicate_quantile": 0.85,
            "mode_quantile": 0.975,
            "radius_margin": 1e-3,
            "gain_quantile": 0.95,
            "gain_ema_decay": 0.8,
            "gain_confirmation_min_count": 2,
            "gain_floor": 0.0,
            "context_alias_capacity": 3,
        }
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
            bank._ensure_prototype_state()
            bank.keys = fn(bank.keys)
            bank.context_keys = fn(bank.context_keys)
            bank.context_valid = fn(bank.context_valid)
            bank.context_support = fn(bank.context_support)
            bank.deltas = fn(bank.deltas)
            bank.write_quality = fn(bank.write_quality)
            bank.queue_weight = fn(bank.queue_weight)
            bank.law_keys = fn(bank.law_keys)
            bank.support = fn(bank.support)
            bank.quality_mass = fn(bank.quality_mass)
            bank.split_mass = fn(bank.split_mass)
            bank.mode_ids = fn(bank.mode_ids)
            bank.mode_compressed = fn(bank.mode_compressed)
            bank.usage = fn(bank.usage)
            bank.cycle_usage = fn(bank.cycle_usage)
            bank.stale_cycles = fn(bank.stale_cycles)
            bank.age = fn(bank.age)
            # The fixed-capacity append store is an optimization cache, not
            # serialized state. Rebuild it lazily after a device/dtype move so
            # no stale pre-move storage can be written into later.
            bank._invalidate_fixed_storage()
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
            bank._ensure_prototype_state()
            rows.append((
                node_id,
                id(bank),
                len(bank),
                bank._age_reference_clock,
                self._tensor_state_signature(bank.keys),
                self._tensor_state_signature(bank.context_keys),
                self._tensor_state_signature(bank.context_valid),
                self._tensor_state_signature(bank.context_support),
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
        max_aliases = max(
            (int(getattr(self.banks[node_id], "context_alias_capacity", 3))
             for node_id in node_ids if node_id in self.banks),
            default=3,
        )
        context_keys = reference.new_zeros(
            node_count, capacity, max_aliases, self.key_dim
        )
        context_valid = torch.zeros(
            node_count, capacity, max_aliases,
            dtype=torch.bool, device=reference.device,
        )
        context_support = reference.new_zeros(
            node_count, capacity, max_aliases
        )
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
            alias_width = int(bank.context_alias_capacity)
            context_keys[node_index, :width, :alias_width] = bank.context_keys[:width]
            context_valid[node_index, :width, :alias_width] = bank.context_valid[:width]
            context_support[node_index, :width, :alias_width] = bank.context_support[:width]
            deltas[node_index, :width] = bank.deltas[:width]
            quality[node_index, :width] = bank.write_quality[:width]
            usage[node_index, :width] = bank.usage[:width]
            base_age[node_index, :width] = bank.age[:width]
            age_reference[node_index] = bank._age_reference_clock
            valid[node_index, :width] = True
        self._packed_mirror = {
            "keys": keys,
            "context_keys": context_keys,
            "context_valid": context_valid,
            "context_support": context_support,
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
                law_dim=self.law_dim,
            )
            bank.configure_prototype_policy(**self._prototype_policy)
            bank._age_reference_clock = self._age_clock
            self.banks[node_id] = bank
        return self.banks[node_id]

    def configure_prototype_memory(self, **settings) -> None:
        """Configure two-level dynamics matching for existing and future banks."""
        merged = dict(self._prototype_policy)
        merged.update(settings)
        # Validate through a real bank when one exists, otherwise use a small
        # temporary instance so bad CLI settings fail at construction time.
        validator = next(iter(self.banks.values()), None)
        if validator is None:
            validator = MemoryBank(
                device=str(self.device),
                key_dim=self.key_dim,
                param_dim=self.param_dim,
                capacity=self.capacity_per_node,
                law_dim=self.law_dim,
            )
        validator.configure_prototype_policy(**merged)
        self._prototype_policy = merged
        for bank in self.banks.values():
            bank.configure_prototype_policy(**merged)

    @torch.no_grad()
    def rebuild_law_keys(
        self,
        semantic_theta_for_node: Callable[[str], Tensor],
        decays: Tensor,
    ) -> None:
        """Migrate existing/legacy rows to effective-Hawkes-law identity."""
        for node_id, bank in self.banks.items():
            bank._ensure_prototype_state()
            if len(bank) == 0:
                continue
            try:
                semantic_theta = semantic_theta_for_node(node_id).detach()
            except KeyError:
                # A delayed topology reconciliation can leave an inaccessible
                # bank in a checkpoint; it is never read and will be removed
                # by the normal node synchronization transaction.
                continue
            bank.law_keys = effective_hawkes_law_key(
                semantic_theta=semantic_theta,
                delta_theta=bank.deltas[: len(bank)],
                decays=decays,
                num_event_types=self.parameter_update.D,
                num_basis=self.parameter_update.M,
                key_dim=self.key_dim,
            ).to(bank.keys)
            bank._invalidate_fixed_storage()

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
        semantic_theta: Optional[Tensor] = None,
        decays: Optional[Tensor] = None,
        law_key: Optional[Tensor] = None,
        prediction_gain: Optional[float | Tensor] = None,
        force_new_mode_confirmation: bool = False,
    ) -> Dict[str, int | float | str]:
        bank = self.get_bank(node_id)
        # A newly written item must have age zero at the current event clock,
        # while existing rows retain their accumulated effective age.
        bank.materialize_age(self._age_clock)
        law_key_builder = None
        if semantic_theta is not None or decays is not None:
            if semantic_theta is None or decays is None:
                raise ValueError("semantic_theta and decays must be provided together")

            def law_key_builder(candidate_delta: Tensor) -> Tensor:
                return effective_hawkes_law_key(
                    semantic_theta=semantic_theta,
                    delta_theta=candidate_delta,
                    decays=decays,
                    num_event_types=self.parameter_update.D,
                    num_basis=self.parameter_update.M,
                    key_dim=self.key_dim,
                )

            law_key = law_key_builder(delta_theta)
        return bank.add(
            key=key,
            delta_theta=delta_theta,
            window=window,
            write_quality=write_quality,
            queue_weight=queue_weight,
            prediction_gain=prediction_gain,
            force_new_mode_confirmation=force_new_mode_confirmation,
            law_key=law_key,
            law_key_builder=law_key_builder,
        )

    def add_memory_batch(
        self,
        node_id: str,
        keys: Tensor,
        delta_theta: Tensor,
        windows: Optional[Sequence[Optional[EventWindow]]] = None,
        write_quality: float | Tensor = 1.0,
        queue_weight: float | Tensor = 0.0,
        semantic_theta: Optional[Tensor] = None,
        decays: Optional[Tensor] = None,
        law_keys: Optional[Tensor] = None,
        prediction_gain: Optional[float | Tensor] = None,
    ) -> list[Dict[str, int | float | str]]:
        """Batch node-local memory writes before touching the persistent bank."""
        bank = self.get_bank(node_id)
        bank.materialize_age(self._age_clock)
        law_key_builder = None
        if semantic_theta is not None or decays is not None:
            if semantic_theta is None or decays is None:
                raise ValueError("semantic_theta and decays must be provided together")

            def law_key_builder(candidate_delta: Tensor) -> Tensor:
                return effective_hawkes_law_key(
                    semantic_theta=semantic_theta,
                    delta_theta=candidate_delta,
                    decays=decays,
                    num_event_types=self.parameter_update.D,
                    num_basis=self.parameter_update.M,
                    key_dim=self.key_dim,
                )

            if law_keys is None:
                law_keys = law_key_builder(delta_theta)
        return bank.add_batch(
            keys=keys,
            delta_theta=delta_theta,
            windows=windows,
            write_quality=write_quality,
            queue_weight=queue_weight,
            prediction_gain=prediction_gain,
            law_keys=law_keys,
            law_key_builder=law_key_builder,
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
            bank._ensure_prototype_state()
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

        max_aliases = max(
            int(bank.context_alias_capacity) for bank in active_banks
        )

        def pad_context_keys() -> Tensor:
            return torch.stack(
                [
                    torch.nn.functional.pad(
                        bank.context_keys,
                        (0, 0, 0, max_aliases - bank.context_alias_capacity,
                         0, max_width - len(bank)),
                    )
                    for bank in active_banks
                ],
                dim=0,
            )

        def pad_context_valid() -> Tensor:
            return torch.stack(
                [
                    torch.nn.functional.pad(
                        bank.context_valid,
                        (0, max_aliases - bank.context_alias_capacity,
                         0, max_width - len(bank)),
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
            keys=pad_context_keys(),
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
            context_valid=pad_context_valid(),
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
        context_keys = mirror["context_keys"]
        context_valid = mirror["context_valid"]
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
        gathered_context_valid = context_valid.index_select(0, flat_nodes)
        gathered_context_valid = gathered_context_valid & node_mask.reshape(-1, 1, 1)
        flat_query = query[:, None, :].expand(
            -1, node_indices.size(1), -1
        ).reshape(-1, self.key_dim)
        active = gathered_valid.any(dim=-1)
        active_rows = torch.nonzero(active, as_tuple=False).flatten()
        active_nodes = flat_nodes.index_select(0, active_rows)
        active_valid = gathered_valid.index_select(0, active_rows)
        active_context_valid = gathered_context_valid.index_select(0, active_rows)
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
            keys=context_keys.index_select(0, active_nodes),
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
            context_valid=active_context_valid,
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
        keys = mirror["context_keys"].index_select(0, node_indices)
        context_valid = mirror["context_valid"].index_select(0, node_indices)
        valid = mirror["valid"].index_select(0, node_indices)
        normalized_query = F.normalize(query, dim=-1)
        normalized_keys = F.normalize(keys, dim=-1)
        alias_similarity = torch.einsum(
            "bckd,bd->bck",
            normalized_keys,
            normalized_query,
        )
        similarity = alias_similarity.masked_fill(
            ~context_valid, -torch.inf
        ).max(dim=-1).values
        similarity_clean = torch.where(
            valid, similarity, torch.zeros_like(similarity)
        )
        logits = temperature * similarity_clean
        row_has_memory = valid.any(dim=-1)
        safe_logits = logits.masked_fill(~valid, -torch.inf)
        safe_logits = torch.where(
            row_has_memory[:, None],
            safe_logits,
            torch.zeros_like(safe_logits),
        )
        beta = F.softmax(safe_logits, dim=-1) * valid.to(logits.dtype)
        beta = beta / beta.sum(dim=-1, keepdim=True).clamp_min(eps)
        weighted_similarity = (beta * similarity_clean).sum(dim=-1)
        weighted_similarity = torch.where(
            row_has_memory,
            weighted_similarity,
            weighted_similarity.new_full((), -1.0),
        )
        novelty = (1.0 - weighted_similarity) / 2.0

        normalized_count, _ = local_recurrence_count(
            similarity_clean,
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
            alias_capacity = int(self._prototype_policy.get("context_alias_capacity", 3))
            context_keys = query.new_zeros(1, alias_capacity, self.key_dim)
            context_keys[0, 0] = F.normalize(key.reshape(1, -1).to(query), dim=-1)
            context_valid = torch.zeros(
                1, alias_capacity, dtype=torch.bool, device=query.device
            )
            context_valid[0, 0] = True
            deltas = delta.reshape(1, -1).to(query)
            usage = query.new_tensor([float(torch.as_tensor(virtual_usage))])
            age = query.new_tensor([float(torch.as_tensor(virtual_age))])
            qualities = query.new_tensor([float(torch.as_tensor(write_quality))])
        else:
            bank._ensure_prototype_state()
            context_keys = torch.cat(
                [
                    bank.context_keys.to(query),
                    query.new_zeros(1, bank.context_alias_capacity, self.key_dim),
                ],
                0,
            )
            context_keys[-1, 0] = F.normalize(key.reshape(1, -1).to(query), dim=-1)
            context_valid = torch.cat(
                [
                    bank.context_valid.to(query.device),
                    torch.zeros(
                        1, bank.context_alias_capacity,
                        dtype=torch.bool, device=query.device,
                    ),
                ],
                0,
            )
            context_valid[-1, 0] = True
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
            query=query, keys=context_keys, context_valid=context_valid,
            deltas=deltas, usage=usage, age=age,
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
        bank._ensure_prototype_state()
        normalized_key = F.normalize(key.detach().reshape(1, -1).to(bank.keys), dim=-1)
        candidate_delta = delta.detach().reshape(1, -1).to(bank.deltas)
        alias_matches = torch.isclose(
            bank.context_keys,
            normalized_key.unsqueeze(1),
            rtol=1e-5,
            atol=1e-7,
        ).all(dim=-1) & bank.context_valid
        matches = alias_matches.any(dim=-1) & torch.isclose(
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
            keys=bank.context_keys[keep].to(query),
            context_valid=bank.context_valid[keep].to(query.device),
            deltas=bank.deltas[keep].to(query),
            usage=bank.usage[keep].to(query),
            age=bank.effective_age(self._age_clock)[keep].to(query),
            write_quality=bank.write_quality[keep].to(query),
        )
        info = dict(info)
        info["excluded_index"] = query.new_tensor(removed, dtype=torch.long)
        return result, info

    def get_extra_state(self):
        """Save effective ages and adaptive admission state."""
        for bank in self.banks.values():
            bank._ensure_prototype_state()
        return {
            node_id: {
                "keys": bank.keys.detach().cpu(),
                "context_keys": bank.context_keys.detach().cpu(),
                "context_valid": bank.context_valid.detach().cpu(),
                "context_support": bank.context_support.detach().cpu(),
                "deltas": bank.deltas.detach().cpu(),
                "write_quality": bank.write_quality.detach().cpu(),
                "queue_weight": bank.queue_weight.detach().cpu(),
                "law_keys": bank.law_keys.detach().cpu(),
                "support": bank.support.detach().cpu(),
                "quality_mass": bank.quality_mass.detach().cpu(),
                "split_mass": bank.split_mass.detach().cpu(),
                "mode_ids": bank.mode_ids.detach().cpu(),
                "mode_compressed": bank.mode_compressed.detach().cpu(),
                "next_mode_id": int(bank._next_mode_id),
                "adaptive_state": {
                    "duplicate_distances": {
                        int(mode): list(values)
                        for mode, values in bank._mode_duplicate_distances.items()
                    },
                    "mode_distances": {
                        int(mode): list(values)
                        for mode, values in bank._mode_distances.items()
                    },
                    "normal_gains": {
                        int(mode): list(values)
                        for mode, values in bank._mode_normal_gains.items()
                    },
                    "pending_distances": {
                        int(mode): list(values)
                        for mode, values in bank._mode_pending_distances.items()
                    },
                    "pending_gain_ema": dict(bank._mode_pending_gain_ema),
                    "pending_gain_count": dict(bank._mode_pending_gain_count),
                    "pending_candidates": {
                        int(mode): [
                            {
                                "law_key": candidate["law_key"].detach().cpu()
                                if torch.is_tensor(candidate.get("law_key"))
                                else None,
                                "gain_ema": float(candidate.get("gain_ema", 0.0)),
                                "count": int(candidate.get("count", 0)),
                                "distance_history": [
                                    float(value)
                                    for value in candidate.get(
                                        "distance_history", []
                                    )
                                ],
                            }
                            for candidate in candidates
                        ]
                        for mode, candidates in bank._mode_pending_candidates.items()
                    },
                },
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
                "prototype_policy": dict(self._prototype_policy),
            }
            for node_id, bank in self.banks.items()
        }

    def set_extra_state(self, state) -> None:
        self.banks = {}
        # Stored ages are already effective at checkpoint time; rebasing the
        # logical clock to zero preserves all future age differences.
        self._age_clock = 0
        # A short-lived intermediate checkpoint format stored context aliases
        # but not the tree-level prototype policy.  Recover a consistent
        # alias width for future banks when every serialized bank agrees.
        has_stored_policy = any(
            isinstance(bank_state, Mapping)
            and "prototype_policy" in bank_state
            for bank_state in state.values()
        )
        if not has_stored_policy:
            stored_widths = {
                int(bank_state["context_keys"].size(1))
                for bank_state in state.values()
                if isinstance(bank_state, Mapping)
                and torch.is_tensor(bank_state.get("context_keys"))
                and bank_state["context_keys"].ndim == 3
                and bank_state["context_keys"].size(-1) == self.key_dim
            }
            if len(stored_widths) == 1:
                self._prototype_policy["context_alias_capacity"] = stored_widths.pop()
        for node_id, bank_state in state.items():
            bank = self.get_bank(node_id)
            if "prototype_policy" in bank_state:
                self._prototype_policy.update(bank_state["prototype_policy"])
                # The bank is empty at this point, so a checkpoint can
                # legitimately restore a different alias capacity.
                bank.configure_prototype_policy(**self._prototype_policy)
            bank.keys = bank_state["keys"].to(self.device)
            bank.deltas = bank_state["deltas"].to(self.device)
            memory_count = bank_state["keys"].shape[0]
            stored_context_keys = bank_state.get("context_keys")
            if stored_context_keys is not None:
                stored_context_keys = stored_context_keys.to(self.device)
                if stored_context_keys.ndim == 3 and stored_context_keys.size(-1) == self.key_dim:
                    bank.context_alias_capacity = int(stored_context_keys.size(1))
                    bank.context_keys = stored_context_keys
            stored_context_valid = bank_state.get("context_valid")
            if stored_context_valid is not None:
                bank.context_valid = stored_context_valid.to(self.device)
            stored_context_support = bank_state.get("context_support")
            if stored_context_support is not None:
                bank.context_support = stored_context_support.to(self.device)
            bank.write_quality = bank_state.get(
                "write_quality",
                torch.ones(memory_count),
            ).to(self.device)
            bank.queue_weight = bank_state.get(
                "queue_weight",
                torch.zeros(memory_count),
            ).to(self.device)
            bank.law_keys = bank_state.get(
                "law_keys", bank.keys.clone()
            ).to(self.device)
            bank.support = bank_state.get(
                "support", torch.ones(memory_count)
            ).to(self.device)
            bank.quality_mass = bank_state.get(
                "quality_mass", bank.write_quality.cpu() * bank.support.cpu()
            ).to(self.device)
            bank.split_mass = bank_state.get(
                "split_mass", bank.queue_weight.cpu() * bank.support.cpu()
            ).to(self.device)
            bank.mode_ids = bank_state.get(
                "mode_ids", torch.arange(memory_count, dtype=torch.long)
            ).to(self.device)
            bank.mode_compressed = bank_state.get(
                "mode_compressed", torch.zeros(memory_count, dtype=torch.bool)
            ).to(self.device)
            bank._next_mode_id = int(bank_state.get("next_mode_id", memory_count))
            adaptive_state = bank_state.get("adaptive_state", {})
            bank._mode_duplicate_distances = {
                int(mode): [float(value) for value in values]
                for mode, values in adaptive_state.get(
                    "duplicate_distances", {}
                ).items()
            }
            bank._mode_distances = {
                int(mode): [float(value) for value in values]
                for mode, values in adaptive_state.get("mode_distances", {}).items()
            }
            bank._mode_normal_gains = {
                int(mode): [float(value) for value in values]
                for mode, values in adaptive_state.get("normal_gains", {}).items()
            }
            bank._mode_pending_distances = {
                int(mode): [float(value) for value in values]
                for mode, values in adaptive_state.get("pending_distances", {}).items()
            }
            bank._mode_pending_gain_ema = {
                int(mode): float(value)
                for mode, value in adaptive_state.get("pending_gain_ema", {}).items()
            }
            bank._mode_pending_gain_count = {
                int(mode): int(value)
                for mode, value in adaptive_state.get("pending_gain_count", {}).items()
            }
            bank._mode_pending_candidates = {}
            for mode, candidates in adaptive_state.get(
                "pending_candidates", {}
            ).items():
                restored_candidates = []
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    law_key = candidate.get("law_key")
                    if not torch.is_tensor(law_key):
                        continue
                    restored_candidates.append(
                        {
                            "law_key": law_key.to(self.device).reshape(-1).clone(),
                            "gain_ema": float(candidate.get("gain_ema", 0.0)),
                            "count": int(candidate.get("count", 0)),
                            "distance_history": [
                                float(value)
                                for value in candidate.get(
                                    "distance_history", []
                                )
                            ],
                        }
                    )
                if restored_candidates:
                    bank._mode_pending_candidates[int(mode)] = restored_candidates
                    bank._sync_legacy_pending_state(int(mode))
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
            bank._ensure_prototype_state()
            bank._invalidate_fixed_storage()
