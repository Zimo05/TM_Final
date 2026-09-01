"""Likelihood calculations used by global training objectives."""

from __future__ import annotations

from Train.TrainingComponents import *  # noqa: F403


class TrainingLikelihoodMixin:
    def _sequence_event_nll(
        self,
        sequence: Mapping[str, Tensor],
        memory_output: Mapping[str, Any],
    ) -> Tensor:
        """Vectorized per-event NLL for batched effective parameters.

        This is the cached algebra from ``hawkes.event_NLL`` applied to all
        events at once; it does not include a post-sequence tail integral,
        matching the original global event loop exactly.
        """
        effective = memory_output["effective_params"]
        times = sequence["times"]
        types = sequence["types"].long()
        event_count = int(times.numel())
        expected_mu_shape = (event_count, self.hawkes.num_types)
        if effective.mu.shape != expected_mu_shape:
            raise ValueError(
                "effective parameter batch must align with sequence events"
            )
        history_stats = sequence[HAWKES_HISTORY_STATS_KEY]
        interval_stats = sequence[HAWKES_INTERVAL_STATS_KEY]
        intensity = (
            effective.mu
            + torch.einsum(
                "kdem,kem->kd",
                effective.W,
                history_stats,
            )
        ).clamp_min(1e-8)
        log_term = -intensity.gather(
            1,
            types.reshape(-1, 1),
        ).squeeze(1).log()
        previous = torch.cat([times.new_zeros(1), times[:-1]])
        duration = (times - previous).clamp_min(0.0)
        integral = (
            effective.mu.sum(dim=-1) * duration
            + torch.einsum(
                "kdem,kem->k",
                effective.W,
                interval_stats,
            )
        )
        return log_term + integral

    def _frontier_sequence_event_nll(
        self,
        sequence: Mapping[str, Tensor],
        memory_output: Mapping[str, Any],
    ) -> Tensor:
        """Vectorized event NLL for at most K active frontier experts."""
        raw_theta = memory_output["frontier_theta"]
        mask = memory_output["frontier_mask"]
        if raw_theta.ndim != 3 or mask.shape != raw_theta.shape[:2]:
            raise ValueError(
                "frontier tensors must align as [events, K, P]"
            )
        D = self.hawkes.num_types
        M = self.hawkes.num_basis
        raw_mu = raw_theta[..., :D]
        raw_W = raw_theta[..., D:].reshape(
            raw_theta.size(0),
            raw_theta.size(1),
            D,
            D,
            M,
        )
        mu = F.softplus(raw_mu)
        W = F.softplus(raw_W)
        types = sequence["types"].long()
        times = sequence["times"]
        history_stats = sequence[HAWKES_HISTORY_STATS_KEY]
        interval_stats = sequence[HAWKES_INTERVAL_STATS_KEY]
        intensity = (
            mu
            + torch.einsum(
                "kfdem,kem->kfd",
                W,
                history_stats,
            )
        ).clamp_min(1e-8)
        selected = intensity.gather(
            2,
            types.reshape(-1, 1, 1).expand(
                -1,
                raw_theta.size(1),
                1,
            ),
        ).squeeze(2)
        log_term = -selected.log()
        previous = torch.cat([times.new_zeros(1), times[:-1]])
        duration = (times - previous).clamp_min(0.0)
        integral = (
            mu.sum(dim=-1) * duration.unsqueeze(-1)
            + torch.einsum(
                "kfdem,kem->kf",
                W,
                interval_stats,
            )
        )
        return (log_term + integral).masked_fill(~mask, torch.inf)

    def _frontier_event_nll(
        self,
        sequence: Mapping[str, Tensor],
        memory_output: Mapping[str, Any],
        event_index: int,
    ) -> Tensor:
        """Evaluate only the computed frontier experts for one Wake event."""
        raw_theta = memory_output["frontier_theta"][0]
        mask = memory_output["frontier_mask"][0]
        D = self.hawkes.num_types
        M = self.hawkes.num_basis
        mu = F.softplus(raw_theta[:, :D])
        W = F.softplus(raw_theta[:, D:].reshape(-1, D, D, M))
        event_type = sequence["types"][event_index].long()
        history = sequence[HAWKES_HISTORY_STATS_KEY][event_index]
        interval = sequence[HAWKES_INTERVAL_STATS_KEY][event_index]
        intensity = (
            mu + torch.einsum("fdem,em->fd", W, history)
        ).clamp_min(1e-8)
        log_term = -intensity[:, event_type].log()
        previous_time = (
            sequence["times"].new_zeros(())
            if event_index == 0
            else sequence["times"][event_index - 1]
        )
        duration = (
            sequence["times"][event_index] - previous_time
        ).clamp_min(0.0)
        integral = (
            mu.sum(dim=-1) * duration
            + torch.einsum("fdem,em->f", W, interval)
        )
        return (log_term + integral).masked_fill(~mask, torch.inf)

    def _expanded_child_sequence_event_nll(
        self,
        sequence: Mapping[str, Tensor],
        memory_output: Mapping[str, Any],
    ) -> Tensor:
        """Evaluate immediate-child dynamics without Router/working mixtures."""
        raw_theta = memory_output["expanded_child_theta"]
        expanded_mask = memory_output["expanded_mask"]
        if (
            raw_theta.ndim != 4
            or raw_theta.shape[:2] != expanded_mask.shape
            or raw_theta.size(2) != 2
            or raw_theta.size(-1) != self.tree.param_dim
        ):
            raise ValueError(
                "expanded child theta must have shape [events, rounds, 2, P]"
            )
        D = self.hawkes.num_types
        M = self.hawkes.num_basis
        raw_mu = raw_theta[..., :D]
        raw_W = raw_theta[..., D:].reshape(
            raw_theta.size(0),
            raw_theta.size(1),
            2,
            D,
            D,
            M,
        )
        mu = F.softplus(raw_mu)
        W = F.softplus(raw_W)
        history_stats = sequence[HAWKES_HISTORY_STATS_KEY]
        interval_stats = sequence[HAWKES_INTERVAL_STATS_KEY]
        types = sequence["types"].long()
        times = sequence["times"]
        intensity = (
            mu
            + torch.einsum(
                "trcdem,tem->trcd",
                W,
                history_stats,
            )
        ).clamp_min(1e-8)
        selected = intensity.gather(
            3,
            types.reshape(-1, 1, 1, 1).expand(
                -1,
                raw_theta.size(1),
                2,
                1,
            ),
        ).squeeze(3)
        log_term = -selected.log()
        previous = torch.cat([times.new_zeros(1), times[:-1]])
        duration = (times - previous).clamp_min(0.0)
        integral = (
            mu.sum(dim=-1) * duration[:, None, None]
            + torch.einsum(
                "trcdem,tem->trc",
                W,
                interval_stats,
            )
        )
        return (log_term + integral).masked_fill(
            ~expanded_mask.unsqueeze(-1),
            torch.inf,
        )

    def _batched_sequence_event_nll(
        self,
        flat: Mapping[str, Tensor],
        memory_output: Mapping[str, Any],
    ) -> Tensor:
        """Per-event NLL for a flattened variable-length sequence batch."""
        effective = memory_output["effective_params"]
        types = flat["types"]
        history = flat[HAWKES_HISTORY_STATS_KEY]
        interval = flat[HAWKES_INTERVAL_STATS_KEY]
        duration = flat["duration"]
        if effective.mu.size(0) != types.numel():
            raise ValueError("effective parameters must align with flat events")
        intensity = (
            effective.mu
            + torch.einsum("ndem,nem->nd", effective.W, history)
        ).clamp_min(1e-8)
        log_term = -intensity.gather(
            1, types.reshape(-1, 1)
        ).squeeze(1).log()
        integral = (
            effective.mu.sum(dim=-1) * duration
            + torch.einsum("ndem,nem->n", effective.W, interval)
        )
        return log_term + integral

    def _batched_frontier_event_nll(
        self,
        flat: Mapping[str, Tensor],
        memory_output: Mapping[str, Any],
    ) -> Tensor:
        """Frontier expert NLL for all flattened events in one call."""
        raw_theta = memory_output["frontier_theta"]
        mask = memory_output["frontier_mask"]
        if raw_theta.ndim != 3 or mask.shape != raw_theta.shape[:2]:
            raise ValueError("frontier tensors must align as [N, K, P]")
        D = self.hawkes.num_types
        M = self.hawkes.num_basis
        mu = F.softplus(raw_theta[..., :D])
        W = F.softplus(
            raw_theta[..., D:].reshape(
                raw_theta.size(0), raw_theta.size(1), D, D, M
            )
        )
        history = flat[HAWKES_HISTORY_STATS_KEY]
        interval = flat[HAWKES_INTERVAL_STATS_KEY]
        types = flat["types"]
        duration = flat["duration"]
        intensity = (
            mu + torch.einsum("nkdem,nem->nkd", W, history)
        ).clamp_min(1e-8)
        selected = intensity.gather(
            2,
            types[:, None, None].expand(-1, raw_theta.size(1), 1),
        ).squeeze(2)
        integral = (
            mu.sum(dim=-1) * duration[:, None]
            + torch.einsum("nkdem,nem->nk", W, interval)
        )
        return (-selected.log() + integral).masked_fill(~mask, torch.inf)

    def _batched_expanded_child_event_nll(
        self,
        flat: Mapping[str, Tensor],
        memory_output: Mapping[str, Any],
    ) -> Tensor:
        """Child-energy tensor ``[N, rounds, 2]`` for one packed batch."""
        raw_theta = memory_output["expanded_child_theta"]
        expanded_mask = memory_output["expanded_mask"]
        if (
            raw_theta.ndim != 4
            or raw_theta.shape[:2] != expanded_mask.shape
            or raw_theta.size(2) != 2
            or raw_theta.size(-1) != self.tree.param_dim
        ):
            raise ValueError(
                "expanded child theta must have shape [N, rounds, 2, P]"
            )
        D = self.hawkes.num_types
        M = self.hawkes.num_basis
        mu = F.softplus(raw_theta[..., :D])
        W = F.softplus(
            raw_theta[..., D:].reshape(
                raw_theta.size(0),
                raw_theta.size(1),
                2,
                D,
                D,
                M,
            )
        )
        history = flat[HAWKES_HISTORY_STATS_KEY]
        interval = flat[HAWKES_INTERVAL_STATS_KEY]
        types = flat["types"]
        duration = flat["duration"]
        intensity = (
            mu + torch.einsum("nrcdem,nem->nrcd", W, history)
        ).clamp_min(1e-8)
        selected = intensity.gather(
            3,
            types[:, None, None, None].expand(
                -1, raw_theta.size(1), 2, 1
            ),
        ).squeeze(3)
        integral = (
            mu.sum(dim=-1) * duration[:, None, None]
            + torch.einsum("nrcdem,nem->nrc", W, interval)
        )
        return (-selected.log() + integral).masked_fill(
            ~expanded_mask.unsqueeze(-1),
            torch.inf,
        )
