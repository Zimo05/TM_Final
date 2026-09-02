"""Shared conditional replay likelihood for sleep-time algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from HawkesBackbone import (
    EVENT_TIME_FEATURES_KEY,
    HAWKES_CACHE_SIGNATURE_KEY,
    HAWKES_HISTORY_STATS_KEY,
    HAWKES_INTERVAL_STATS_KEY,
    HawkesFamily,
)
from .MemoryBank import EffectiveHawkesParameters, EventWindow


@dataclass
class ReplayBatchCache:
    """Flattened, parameter-independent Hawkes inputs for replay windows.

    ``history`` and ``interval`` contain one row per target event.  The
    ``window_indices`` vector is the flattened ``s(e)`` mapping back to the
    original replay window, while ``event_counts`` preserves each window's
    event normalization.  None of these tensors depend on a candidate
    Hawkes parameter vector, so they can be built once outside an optimizer
    loop.
    """

    history: Tensor
    interval: Tensor
    duration: Tensor
    target_types: Tensor
    window_indices: Tensor
    event_counts: Tensor

    @property
    def num_windows(self) -> int:
        return int(self.event_counts.numel())


def _window_event_bounds(window: EventWindow) -> tuple[int, int]:
    if window.has_full_history:
        start = int(window.start_idx)
        end = int(window.end_idx)
        if start < 0 or end < start or end > window.times.numel():
            raise ValueError("full-history EventWindow has an invalid event range")
        return start, end
    return 0, int(window.times.numel())


def _window_target_tensors(
    window: EventWindow,
    hawkes_ll: HawkesFamily,
    reference: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Prepare one window's target-event tensors without model evaluation."""
    times = window.times.to(device=reference.device, dtype=reference.dtype)
    types = window.types.to(device=reference.device).long()
    if times.ndim != 1 or types.ndim != 1 or times.numel() != types.numel():
        raise ValueError("window times/types must be aligned one-dimensional tensors")
    start, end = _window_event_bounds(window)
    if not window.has_full_history and times.numel() > 0:
        times = times - times[0]

    event_count = end - start
    if event_count == 0:
        shape = (0, hawkes_ll.num_types, hawkes_ll.num_basis)
        return (
            reference.new_empty(shape),
            reference.new_empty(shape),
            reference.new_empty((0,)),
            types.new_empty((0,)),
        )

    sequence = {"times": times, "types": types}
    if window.has_full_history:
        cached_fields = {
            EVENT_TIME_FEATURES_KEY: getattr(
                window, "event_time_features", None
            ),
            HAWKES_HISTORY_STATS_KEY: getattr(
                window, "hawkes_history_stats", None
            ),
            HAWKES_INTERVAL_STATS_KEY: getattr(
                window, "hawkes_interval_stats", None
            ),
            HAWKES_CACHE_SIGNATURE_KEY: getattr(
                window, "hawkes_cache_signature", None
            ),
        }
        sequence.update({
            key: value for key, value in cached_fields.items()
            if value is not None
        })
    sequence = hawkes_ll.prepare_sequence_cache(sequence, inplace=True)
    if window.has_full_history:
        window.event_time_features = sequence[EVENT_TIME_FEATURES_KEY]
        window.hawkes_history_stats = sequence[HAWKES_HISTORY_STATS_KEY]
        window.hawkes_interval_stats = sequence[HAWKES_INTERVAL_STATS_KEY]
        window.hawkes_cache_signature = sequence[HAWKES_CACHE_SIGNATURE_KEY]

    history = sequence[HAWKES_HISTORY_STATS_KEY][start:end].to(reference)
    interval = sequence[HAWKES_INTERVAL_STATS_KEY][start:end].to(reference)
    previous = torch.cat([times.new_zeros(1), times[:-1]])
    duration = (times - previous).clamp_min(0.0)[start:end]
    return history, interval, duration, types[start:end]


def build_replay_batch_cache(
    windows: Sequence[EventWindow],
    hawkes_ll: HawkesFamily,
    reference: Tensor,
) -> ReplayBatchCache:
    """Flatten replay-window Hawkes statistics once for repeated evaluation."""
    if not torch.is_tensor(reference):
        raise TypeError("reference must be a tensor")

    histories: list[Tensor] = []
    intervals: list[Tensor] = []
    durations: list[Tensor] = []
    target_types: list[Tensor] = []
    event_counts: list[int] = []
    for window in windows:
        history, interval, duration, types = _window_target_tensors(
            window,
            hawkes_ll,
            reference,
        )
        count = int(types.numel())
        histories.append(history)
        intervals.append(interval)
        durations.append(duration)
        target_types.append(types)
        event_counts.append(count)

    device = reference.device
    dtype = reference.dtype
    shape = (0, hawkes_ll.num_types, hawkes_ll.num_basis)
    history = torch.cat(histories, dim=0) if histories else reference.new_empty(shape)
    interval = torch.cat(intervals, dim=0) if intervals else reference.new_empty(shape)
    duration = torch.cat(durations, dim=0) if durations else reference.new_empty((0,))
    flattened_types = (
        torch.cat(target_types, dim=0)
        if target_types
        else torch.empty(0, device=device, dtype=torch.long)
    )
    event_counts_tensor = torch.tensor(
        event_counts,
        device=device,
        dtype=torch.long,
    )
    window_indices = torch.repeat_interleave(
        torch.arange(len(windows), device=device, dtype=torch.long),
        event_counts_tensor,
    )
    if history.device != device or history.dtype != dtype:
        history = history.to(device=device, dtype=dtype)
        interval = interval.to(device=device, dtype=dtype)
        duration = duration.to(device=device, dtype=dtype)
    return ReplayBatchCache(
        history=history,
        interval=interval,
        duration=duration,
        target_types=flattened_types.to(device=device),
        window_indices=window_indices,
        event_counts=event_counts_tensor,
    )


def batched_replay_log_likelihood(
    cache: ReplayBatchCache,
    theta_models: Tensor,
    hawkes_ll: HawkesFamily,
    *,
    normalize_by_events: bool = True,
) -> Tensor:
    """Evaluate three Hawkes models over all cached replay events at once."""
    if theta_models.ndim != 2 or theta_models.size(0) != 3:
        raise ValueError("theta_models must have shape [3, parameters]")
    D = hawkes_ll.num_types
    M = hawkes_ll.num_basis
    expected = D + D * D * M
    if theta_models.size(1) != expected:
        raise ValueError(
            f"theta_models must contain {expected} values per model"
        )
    if cache.history.shape != (cache.duration.numel(), D, M):
        raise ValueError("replay history cache has an invalid shape")
    if cache.interval.shape != cache.history.shape:
        raise ValueError("replay interval cache must align with history")
    if cache.target_types.numel() != cache.duration.numel():
        raise ValueError("replay target types must align with duration")

    raw_mu = theta_models[:, :D]
    raw_W = theta_models[:, D:].reshape(3, D, D, M)
    mu = torch.nn.functional.softplus(raw_mu)
    W = torch.nn.functional.softplus(raw_W)
    history = cache.history.to(
        device=theta_models.device,
        dtype=theta_models.dtype,
    )
    interval = cache.interval.to(
        device=theta_models.device,
        dtype=theta_models.dtype,
    )
    duration = cache.duration.to(
        device=theta_models.device,
        dtype=theta_models.dtype,
    )
    target_types = cache.target_types.to(device=theta_models.device)
    event_window = cache.window_indices.to(device=theta_models.device)
    intensity = (
        mu.unsqueeze(0)
        + torch.einsum("q d s m, e s m -> e q d", W, history)
    ).clamp_min(1e-8)
    gathered = intensity.gather(
        2,
        target_types[:, None, None].expand(-1, 3, 1),
    ).squeeze(-1)
    log_term = -gathered.log()
    integral = (
        mu.sum(dim=-1).unsqueeze(0) * duration.unsqueeze(-1)
        + torch.einsum("q d s m, e s m -> e q", W, interval)
    )
    event_nll = log_term + integral
    window_nll = (
        theta_models.sum() * 0.0
        + theta_models.new_zeros((cache.num_windows, 3))
    )
    if event_nll.numel():
        window_nll = window_nll.index_add(0, event_window, event_nll)
    if normalize_by_events:
        counts = cache.event_counts.to(
            device=theta_models.device,
            dtype=theta_models.dtype,
        ).clamp_min(1.0)
        window_nll = window_nll / counts.unsqueeze(-1)
    return window_nll


def _effective_parameters(
    theta: Tensor,
    hawkes_ll: HawkesFamily,
    decays: Optional[Tensor],
) -> EffectiveHawkesParameters:
    D = hawkes_ll.num_types
    M = hawkes_ll.num_basis
    expected = D + D * D * M
    if theta.ndim != 1 or theta.numel() != expected:
        raise ValueError(f"theta must have shape [{expected}]")
    if decays is None:
        decays = hawkes_ll.decays
    decays = torch.as_tensor(
        decays, device=theta.device, dtype=theta.dtype
    ).reshape(-1)
    if decays.numel() != M or bool((decays <= 0).any().item()):
        raise ValueError(f"decays must contain {M} positive values")

    raw_mu = theta[:D]
    raw_W = theta[D:].reshape(D, D, M)
    return EffectiveHawkesParameters(
        theta=theta,
        raw_mu=raw_mu,
        raw_W=raw_W,
        mu=torch.nn.functional.softplus(raw_mu),
        W=torch.nn.functional.softplus(raw_W),
        decays=decays,
    )


def replay_log_likelihood(
    window: EventWindow,
    theta: Tensor,
    hawkes_ll: HawkesFamily,
    decays: Optional[Tensor] = None,
    normalize_by_events: bool = True,
) -> Tensor:
    """Evaluate only the target events while retaining their prefix history.

    New EventWindow objects contain a full prefix and use their original
    ``start_idx:end_idx`` range. Legacy sliced windows lack that history; they
    are shifted to local time zero and evaluated as an explicitly truncated
    history approximation instead of introducing a false [0, absolute_time]
    no-event interval.
    """
    parameters = _effective_parameters(theta, hawkes_ll, decays)
    times = window.times.to(device=theta.device, dtype=theta.dtype)
    types = window.types.to(device=theta.device).long()
    if times.ndim != 1 or types.ndim != 1 or times.numel() != types.numel():
        raise ValueError("window times/types must be aligned one-dimensional tensors")

    if window.has_full_history:
        start = int(window.start_idx)
        end = int(window.end_idx)
        if start < 0 or end < start or end > times.numel():
            raise ValueError("full-history EventWindow has an invalid event range")
    else:
        # Backward-compatible handling for checkpoints written with sliced
        # absolute-time windows and global start/end indices.
        if times.numel() > 0:
            times = times - times[0]
        start = 0
        end = int(times.numel())

    event_count = end - start
    if event_count == 0:
        return theta.sum() * 0.0

    sequence = {"times": times, "types": types}
    if window.has_full_history:
        cached_fields = {
            EVENT_TIME_FEATURES_KEY: getattr(
                window, "event_time_features", None
            ),
            HAWKES_HISTORY_STATS_KEY: getattr(
                window, "hawkes_history_stats", None
            ),
            HAWKES_INTERVAL_STATS_KEY: getattr(
                window, "hawkes_interval_stats", None
            ),
            HAWKES_CACHE_SIGNATURE_KEY: getattr(
                window, "hawkes_cache_signature", None
            ),
        }
        sequence.update({
            key: value for key, value in cached_fields.items()
            if value is not None
        })
    sequence = hawkes_ll.prepare_sequence_cache(sequence, inplace=True)
    if window.has_full_history:
        window.event_time_features = sequence[EVENT_TIME_FEATURES_KEY]
        window.hawkes_history_stats = sequence[HAWKES_HISTORY_STATS_KEY]
        window.hawkes_interval_stats = sequence[HAWKES_INTERVAL_STATS_KEY]
        window.hawkes_cache_signature = sequence[HAWKES_CACHE_SIGNATURE_KEY]
    # Evaluate the complete target interval as one tensor. This is the same
    # cached Hawkes algebra as event_NLL, with no change to which prefix events
    # condition each target.
    history = sequence[HAWKES_HISTORY_STATS_KEY][start:end].to(theta)
    interval = sequence[HAWKES_INTERVAL_STATS_KEY][start:end].to(theta)
    target_types = types[start:end]
    mu = parameters.mu
    W = parameters.W
    intensity = (
        mu.unsqueeze(0)
        + torch.einsum("dem,kem->kd", W, history)
    ).clamp_min(1e-8)
    log_term = -intensity.gather(
        1, target_types.reshape(-1, 1)
    ).squeeze(1).log()
    previous = torch.cat([times.new_zeros(1), times[:-1]])
    duration = (times - previous).clamp_min(0.0)[start:end]
    integral = (
        mu.sum() * duration
        + torch.einsum("dem,kem->k", W, interval)
    )
    log_likelihood = -(log_term + integral).sum()
    if normalize_by_events:
        log_likelihood = log_likelihood / event_count
    return log_likelihood
