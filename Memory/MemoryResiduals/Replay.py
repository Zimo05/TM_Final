"""Shared conditional replay likelihood for sleep-time algorithms."""

from __future__ import annotations

from typing import Optional

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
