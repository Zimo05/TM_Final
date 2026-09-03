"""Budgeted fixed-topology semantic consolidation for Light Sleep.

Light Sleep treats an episodic residual as a parameter-space knowledge gap.
Only repeated, beneficial, directionally coherent residuals are absorbed into
the owning leaf's local semantic offset.  Every residual in that leaf bank is
then rebased so its effective Hawkes parameters remain exactly unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from HawkesBackbone import HawkesFamily
from MemoryResiduals.MemoryBank import MemoryBank
from MemoryResiduals.Replay import replay_log_likelihood


@dataclass(frozen=True)
class LightSleepSettings:
    replay_budget: int = 32
    scan_budget_multiplier: int = 4
    min_per_leaf: int = 2
    routing_mass_mix: float = 0.5
    max_directions: int = 3
    direction_similarity: float = 0.70
    gain_evaluations_per_direction: int = 2
    min_direction_support: int = 2
    min_gain: float = 0.0
    coherence_threshold: float = 0.60
    alpha_max: float = 0.25
    trust_radius: float = 0.10
    gain_reference: float = 0.05
    usage_power: float = 0.5
    stale_decay: float = 0.20
    eps: float = 1e-8

    def validate(self) -> None:
        if self.replay_budget <= 0:
            raise ValueError("Light replay_budget must be positive")
        if self.scan_budget_multiplier <= 0:
            raise ValueError("Light scan_budget_multiplier must be positive")
        if self.min_per_leaf <= 0:
            raise ValueError("Light min_per_leaf must be positive")
        if not 0.0 <= self.routing_mass_mix <= 1.0:
            raise ValueError("routing_mass_mix must lie in [0, 1]")
        if self.max_directions <= 0:
            raise ValueError("max_directions must be positive")
        if not -1.0 <= self.direction_similarity <= 1.0:
            raise ValueError("direction_similarity must lie in [-1, 1]")
        if self.gain_evaluations_per_direction <= 0:
            raise ValueError(
                "gain_evaluations_per_direction must be positive"
            )
        if self.min_direction_support <= 0:
            raise ValueError("min_direction_support must be positive")
        if not 0.0 <= self.coherence_threshold <= 1.0:
            raise ValueError("coherence_threshold must lie in [0, 1]")
        if not 0.0 <= self.alpha_max <= 1.0:
            raise ValueError("alpha_max must lie in [0, 1]")
        if self.trust_radius <= 0.0:
            raise ValueError("trust_radius must be positive")
        if self.gain_reference <= 0.0:
            raise ValueError("gain_reference must be positive")
        if self.usage_power < 0.0 or self.stale_decay < 0.0:
            raise ValueError(
                "usage_power and stale_decay must be non-negative"
            )
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


@dataclass
class DirectionEvidence:
    indices: Tensor
    weight: Tensor
    mean: Tensor
    coherence: Tensor
    gain: Tensor
    replay_indices: Tensor


@dataclass
class LightAbsorptionProposal:
    """The complete non-mutating proposal made by Light for one leaf."""

    theta_old: Tensor
    theta_h0: Tensor
    shared_delta: Tensor
    alpha: float
    selected_direction: Optional[DirectionEvidence]
    replay_indices: Tensor
    replay_weights: Tensor
    candidate_indices: Tensor
    clusters: list[Tensor]
    evidence: list[DirectionEvidence]
    updated_summary: Dict[str, Tensor]
    replay_used: int

    @property
    def theta_candidate(self) -> Tensor:
        """Compatibility name used by the Light application path."""
        return self.theta_h0


def _bounded_valid_window_indices(
    bank: MemoryBank,
    *,
    start: int,
    scan_budget: int,
) -> tuple[list[int], int, int]:
    """Inspect a cyclic, bounded region of a bank.

    The cursor makes successive Light cycles cover different regions without
    sorting or scanning the complete bank.
    """
    bank_size = len(bank)
    if bank_size == 0 or scan_budget <= 0:
        return [], 0, 0
    inspected = min(bank_size, scan_budget)
    start = int(start) % bank_size
    positions = [
        (start + offset) % bank_size
        for offset in range(inspected)
    ]
    valid = [
        index
        for index in positions
        if bank.windows[index] is not None
    ]
    return valid, (start + inspected) % bank_size, inspected


def _cheap_memory_weight(
    bank: MemoryBank,
    indices: Tensor,
    settings: LightSleepSettings,
) -> Tensor:
    quality = bank.write_quality.index_select(0, indices).clamp_min(0.0)
    usage = (
        1.0
        + torch.log1p(bank.usage.index_select(0, indices).clamp_min(0.0))
    ).pow(settings.usage_power)
    stale = torch.exp(
        -settings.stale_decay
        * bank.stale_cycles.index_select(0, indices).clamp_min(0.0)
    )
    return (quality * usage * stale).clamp_min(settings.eps)


def _cluster_residual_directions(
    bank: MemoryBank,
    valid_indices: Sequence[int],
    settings: LightSleepSettings,
    summary: Optional[Dict[str, Tensor]] = None,
) -> tuple[list[Tensor], Dict[str, Tensor]]:
    """Update a capped persistent direction index from bounded candidates."""
    if not valid_indices:
        return [], {} if summary is None else summary
    indices = torch.tensor(
        valid_indices,
        device=bank.device,
        dtype=torch.long,
    )
    cheap_weight = _cheap_memory_weight(bank, indices, settings)
    residuals = bank.deltas.index_select(0, indices)
    normalized = F.normalize(residuals, dim=-1, eps=settings.eps)
    order = torch.argsort(cheap_weight, descending=True)

    centroids = None if summary is None else summary.get("centroids")
    support = None if summary is None else summary.get("support")
    if (
        centroids is None
        or support is None
        or centroids.ndim != 2
        or centroids.shape[-1] != bank.param_dim
        or support.shape != centroids.shape[:1]
    ):
        centroids = residuals[:0]
        support = residuals.new_zeros(0)
    else:
        centroids = centroids.to(
            device=bank.device,
            dtype=residuals.dtype,
        )[: settings.max_directions]
        support = support.to(
            device=bank.device,
            dtype=residuals.dtype,
        )[: settings.max_directions].clamp_min(1.0)

    clusters: list[list[int]] = [
        [] for _ in range(int(centroids.shape[0]))
    ]
    centroid_sum: list[Tensor] = [
        centroids[index] * support[index]
        for index in range(int(centroids.shape[0]))
    ]
    support_values = support.detach().cpu().tolist()
    for local_index in order.detach().cpu().tolist():
        vector = normalized[local_index]
        if not clusters:
            clusters.append([local_index])
            centroid_sum.append(vector.clone())
            support_values.append(1.0)
            continue
        centroids = F.normalize(
            torch.stack(centroid_sum),
            dim=-1,
            eps=settings.eps,
        )
        similarity = centroids @ vector
        best = int(similarity.argmax().item())
        if (
            float(similarity[best].item())
            >= settings.direction_similarity
            or len(clusters) >= settings.max_directions
        ):
            clusters[best].append(local_index)
            centroid_sum[best] = centroid_sum[best] + vector
            support_values[best] += 1.0
        else:
            clusters.append([local_index])
            centroid_sum.append(vector.clone())
            support_values.append(1.0)

    current_clusters = [
        indices.index_select(
            0,
            torch.tensor(
                members,
                device=indices.device,
                dtype=torch.long,
            ),
        )
        for members in clusters
        if members
    ]
    updated_summary = {
        "centroids": F.normalize(
            torch.stack(centroid_sum),
            dim=-1,
            eps=settings.eps,
        ).detach(),
        "support": residuals.new_tensor(support_values).detach(),
    }
    return current_clusters, updated_summary


def _leaf_budgets(
    leaf_ids: Sequence[str],
    mass_ema: Dict[str, float],
    total_budget: int,
    min_per_leaf: int,
    mass_mix: float,
) -> Dict[str, int]:
    if not leaf_ids or total_budget <= 0:
        return {}
    leaf_count = len(leaf_ids)
    masses = torch.tensor(
        [max(0.0, float(mass_ema.get(leaf_id, 0.0))) for leaf_id in leaf_ids],
        dtype=torch.float64,
    )
    if float(masses.sum()) <= 0.0:
        masses.fill_(1.0)
    quota = (
        (1.0 - mass_mix) / leaf_count
        + mass_mix * masses / masses.sum()
    )
    raw = quota * total_budget
    budget = torch.floor(raw).long()
    if total_budget >= leaf_count * min_per_leaf:
        budget.clamp_(min=min_per_leaf)
    else:
        budget.zero_()
        ranked = torch.argsort(quota, descending=True)
        budget[ranked[:total_budget]] = 1

    while int(budget.sum()) > total_budget:
        removable = torch.nonzero(
            budget > (min_per_leaf if total_budget >= leaf_count * min_per_leaf else 0),
            as_tuple=False,
        ).flatten()
        if removable.numel() == 0:
            break
        score = raw[removable] - budget[removable].to(raw.dtype)
        target = removable[int(score.argmin())]
        budget[target] -= 1
    while int(budget.sum()) < total_budget:
        target = int((raw - budget.to(raw.dtype)).argmax())
        budget[target] += 1

    return {
        leaf_id: int(budget[index])
        for index, leaf_id in enumerate(leaf_ids)
        if int(budget[index]) > 0
    }


@torch.no_grad()
def _direction_gain(
    bank: MemoryBank,
    indices: Tensor,
    theta_leaf: Tensor,
    hawkes_ll: HawkesFamily,
    settings: LightSleepSettings,
) -> tuple[Tensor, Tensor]:
    cheap_weight = _cheap_memory_weight(bank, indices, settings)
    evaluation_count = min(
        settings.gain_evaluations_per_direction,
        int(indices.numel()),
    )
    selected_local = torch.topk(
        cheap_weight,
        k=evaluation_count,
    ).indices
    selected = indices.index_select(0, selected_local)
    gains = []
    used = []
    for memory_index in selected.detach().cpu().tolist():
        window = bank.windows[memory_index]
        if window is None:
            continue
        base = replay_log_likelihood(
            window=window,
            theta=theta_leaf,
            hawkes_ll=hawkes_ll,
            decays=hawkes_ll.decays,
            normalize_by_events=True,
        )
        corrected = replay_log_likelihood(
            window=window,
            theta=theta_leaf + bank.deltas[memory_index],
            hawkes_ll=hawkes_ll,
            decays=hawkes_ll.decays,
            normalize_by_events=True,
        )
        gains.append(corrected - base)
        used.append(memory_index)
    if not gains:
        return theta_leaf.new_full((), -torch.inf), indices[:0]
    return (
        torch.stack(gains).mean(),
        torch.tensor(used, device=bank.device, dtype=torch.long),
    )


@torch.no_grad()
def _predictive_residual_utility(
    bank: MemoryBank,
    indices: Tensor,
    theta_leaf: Tensor,
    hawkes_ll: HawkesFamily,
) -> Tensor:
    """Return per-row NLL reduction supplied by each episodic residual."""
    utilities = []
    for memory_index in indices.detach().cpu().tolist():
        window = bank.windows[memory_index]
        if window is None:
            utilities.append(theta_leaf.new_zeros(()))
            continue
        semantic_ll = replay_log_likelihood(
            window=window,
            theta=theta_leaf,
            hawkes_ll=hawkes_ll,
            decays=hawkes_ll.decays,
            normalize_by_events=True,
        )
        residual_ll = replay_log_likelihood(
            window=window,
            theta=theta_leaf + bank.deltas[memory_index],
            hawkes_ll=hawkes_ll,
            decays=hawkes_ll.decays,
            normalize_by_events=True,
        )
        # replay_log_likelihood is larger when prediction is better, so this
        # equals NLL(semantic) - NLL(semantic + residual).
        utilities.append((residual_ll - semantic_ll).clamp_min(0.0))
    if not utilities:
        return theta_leaf.new_empty(0)
    return torch.stack(utilities)


@torch.no_grad()
def _reconcile_contracted_banks(
    tree,
    hawkes_ll: HawkesFamily,
    settings: LightSleepSettings,
    protected_leaf_ids: Collection[str] = (),
    replay_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Consume pending post-contraction overflow in a bounded Light step.

    Collapse remains atomic and lossless.  This later transaction ranks a
    bounded replay sample by predictive residual utility, quality, usage and
    age, then retains at most one high-utility prototype per residual
    direction up to the ordinary per-node capacity.  Merge reconciles on the
    next Light cycle; Prune waits one additional cycle because its child
    specialization was intentionally demoted into episodic memory.
    """
    pending = tree.memory_reconciliation
    base_capacity = int(tree.episodic_memory.capacity_per_node)
    remaining_replay = (
        int(settings.replay_budget)
        if replay_budget is None
        else max(0, int(replay_budget))
    )
    replay_windows = 0
    replay_events = 0
    removed_rows = 0
    deferred = 0
    reconciled_nodes: list[str] = []
    policy_counts = {"merge": 0, "prune": 0}
    utility_numerator = 0.0
    utility_denominator = 0.0
    raw_energy_numerator = 0.0
    raw_energy_denominator = 0.0

    protected = set(protected_leaf_ids)
    for node_id in list(pending):
        if node_id in protected:
            deferred += 1
            continue
        bank = tree.episodic_memory.banks.get(node_id)
        if bank is None:
            pending.pop(node_id, None)
            continue
        raw_record = pending[node_id]
        if isinstance(raw_record, Mapping):
            policy = str(raw_record.get("policy", "prune"))
            delay_cycles = int(raw_record.get("delay_cycles", 0))
        else:
            # Old checkpoints stored only ``1``.  Treat unknown historical
            # contractions conservatively as delayed Prune reconciliation.
            policy = "prune"
            delay_cycles = 1
        if policy not in policy_counts:
            policy = "prune"
        if delay_cycles > 0:
            pending[node_id] = {
                "policy": policy,
                "delay_cycles": delay_cycles - 1,
            }
            deferred += 1
            continue
        if len(bank) <= base_capacity:
            bank.capacity = base_capacity
            pending.pop(node_id, None)
            reconciled_nodes.append(node_id)
            policy_counts[policy] += 1
            continue
        if remaining_replay <= 0:
            deferred += 1
            continue

        valid_indices = [
            index for index, window in enumerate(bank.windows)
            if window is not None
        ]
        if valid_indices:
            valid = torch.tensor(
                valid_indices, device=bank.device, dtype=torch.long
            )
            age = bank.effective_age(
                tree.episodic_memory._age_clock
            ).index_select(0, valid).clamp_min(0.0)
            cheap_priority = (
                bank.write_quality.index_select(0, valid).clamp_min(0.0)
                * (
                    1.0
                    + torch.log1p(
                        bank.usage.index_select(0, valid).clamp_min(0.0)
                    )
                )
                * torch.exp(-settings.stale_decay * age)
            )
            evaluation_count = min(
                int(valid.numel()), remaining_replay
            )
            selected_local = torch.topk(
                cheap_priority, k=evaluation_count
            ).indices
            evaluated = valid.index_select(0, selected_local)
            evaluated_priority = cheap_priority.index_select(
                0, selected_local
            )
            utility = _predictive_residual_utility(
                bank,
                evaluated,
                tree.semantic_theta(node_id).detach(),
                hawkes_ll,
            )
            scores = utility * evaluated_priority
            utility_numerator += float(
                (utility * evaluated_priority).sum().detach().cpu()
            )
            utility_denominator += float(
                evaluated_priority.sum().detach().cpu()
            )
            raw_energy_numerator += float((
                bank.deltas.index_select(0, evaluated).square().sum(dim=-1)
                * evaluated_priority
            ).sum().detach().cpu())
            raw_energy_denominator += float(
                evaluated_priority.sum().detach().cpu()
            )
            order = torch.argsort(scores, descending=True)
            kept: list[int] = []
            prototypes: list[Tensor] = []
            residuals = F.normalize(
                bank.deltas.index_select(0, evaluated),
                dim=-1,
                eps=settings.eps,
            )
            for local_index in order.detach().cpu().tolist():
                if len(kept) >= base_capacity:
                    break
                if float(scores[local_index].detach().cpu()) <= settings.eps:
                    continue
                prototype = residuals[local_index]
                if prototypes and bool(
                    (
                        torch.stack(prototypes) @ prototype
                    ).max().ge(0.95)
                ):
                    continue
                kept.append(int(evaluated[local_index].detach().cpu()))
                prototypes.append(prototype)
            keep_idx = torch.tensor(
                kept, device=bank.device, dtype=torch.long
            )
            replay_windows += evaluation_count
            replay_events += sum(
                max(
                    0,
                    int(bank.windows[index].end_idx)
                    - int(bank.windows[index].start_idx),
                )
                for index in evaluated.detach().cpu().tolist()
                if bank.windows[index] is not None
            )
            remaining_replay -= evaluation_count
        else:
            keep_idx = torch.empty(
                0, device=bank.device, dtype=torch.long
            )

        size_before = len(bank)
        bank.keep(keep_idx)
        bank.capacity = base_capacity
        removed_rows += size_before - len(bank)
        pending.pop(node_id, None)
        reconciled_nodes.append(node_id)
        policy_counts[policy] += 1

    if reconciled_nodes:
        tree.episodic_memory._packed_mirror_signature = None
        tree.episodic_memory._packed_mirror = None
    return {
        "reconciled_nodes": reconciled_nodes,
        "reconciled_banks": len(reconciled_nodes),
        "removed_rows": removed_rows,
        "deferred_banks": deferred,
        "replay_windows": replay_windows,
        "replay_events": replay_events,
        "merge_banks": policy_counts["merge"],
        "prune_banks": policy_counts["prune"],
        "utility_numerator": utility_numerator,
        "utility_denominator": utility_denominator,
        "raw_energy_numerator": raw_energy_numerator,
        "raw_energy_denominator": raw_energy_denominator,
    }


@torch.no_grad()
def _direction_evidence(
    bank: MemoryBank,
    indices: Tensor,
    theta_leaf: Tensor,
    hawkes_ll: HawkesFamily,
    settings: LightSleepSettings,
) -> DirectionEvidence:
    weights = _cheap_memory_weight(bank, indices, settings)
    residuals = bank.deltas.index_select(0, indices)
    weighted_sum = (weights.unsqueeze(-1) * residuals).sum(dim=0)
    total_weight = weights.sum()
    mean = weighted_sum / total_weight.clamp_min(settings.eps)
    denominator = (
        total_weight
        * (
            weights
            * residuals.square().sum(dim=-1)
        ).sum()
    )
    coherence = (
        weighted_sum.square().sum()
        / denominator.clamp_min(settings.eps)
    ).clamp(0.0, 1.0)
    gain, replay_indices = _direction_gain(
        bank,
        indices,
        theta_leaf,
        hawkes_ll,
        settings,
    )
    return DirectionEvidence(
        indices=indices,
        weight=total_weight,
        mean=mean,
        coherence=coherence,
        gain=gain,
        replay_indices=replay_indices,
    )


@torch.no_grad()
def propose_light_absorption(
    bank: MemoryBank,
    theta_leaf: Tensor,
    hawkes_ll: HawkesFamily,
    settings: LightSleepSettings,
    candidate_indices: Sequence[int] | Tensor,
    *,
    direction_summary: Optional[Dict[str, Tensor]] = None,
    replay_budget: Optional[int] = None,
) -> LightAbsorptionProposal:
    """Build exactly the proposal that ``run_light_sleep`` would apply.

    This helper has no tree or Bank mutation.  It is the single source of
    truth for direction clustering, eligibility, selected direction, replay
    rows, trust-region alpha, and the candidate semantic parameter.  Probe H0
    calls this helper's result, so H0 cannot silently drift from real Light.
    """
    settings.validate()
    if torch.is_tensor(candidate_indices):
        candidate_tensor = candidate_indices.to(
            device=bank.device, dtype=torch.long
        ).reshape(-1)
        candidate_list = candidate_tensor.detach().cpu().tolist()
    else:
        candidate_list = [int(index) for index in candidate_indices]
        candidate_tensor = torch.tensor(
            candidate_list, device=bank.device, dtype=torch.long
        )
    budget = (
        len(candidate_list)
        if replay_budget is None
        else max(0, int(replay_budget))
    )
    clusters, updated_summary = _cluster_residual_directions(
        bank,
        candidate_list,
        settings,
        direction_summary,
    )
    evidence: list[DirectionEvidence] = []
    replay_used = 0
    for cluster in clusters:
        if replay_used >= budget:
            break
        local_settings = LightSleepSettings(
            **{
                **settings.__dict__,
                "gain_evaluations_per_direction": min(
                    settings.gain_evaluations_per_direction,
                    budget - replay_used,
                ),
            }
        )
        item = _direction_evidence(
            bank,
            cluster,
            theta_leaf,
            hawkes_ll,
            local_settings,
        )
        evidence.append(item)
        replay_used += int(item.replay_indices.numel())

    eligible = [
        item
        for item in evidence
        if (
            int(item.indices.numel()) >= settings.min_direction_support
            and bool(item.gain > settings.min_gain)
            and bool(item.coherence >= settings.coherence_threshold)
        )
    ]
    selected_direction = None
    alpha = 0.0
    theta_candidate = theta_leaf.detach().clone()
    if eligible:
        selected_direction = max(
            eligible,
            key=lambda item: float(
                (
                    item.weight
                    * item.coherence
                    * item.gain.clamp_min(0.0)
                ).item()
            ),
        )
        trust = min(
            1.0,
            settings.trust_radius
            / (float(selected_direction.mean.norm().item()) + settings.eps),
        )
        gain_scale = min(
            1.0,
            max(0.0, float(selected_direction.gain.item()))
            / settings.gain_reference,
        )
        alpha = (
            settings.alpha_max
            * float(selected_direction.coherence.item())
            * trust
            * gain_scale
        )
        theta_candidate = theta_leaf + alpha * selected_direction.mean

    if selected_direction is None:
        shared_delta = theta_leaf.new_zeros(theta_leaf.shape)
        replay_indices = candidate_tensor[:0]
        replay_weights = theta_leaf.new_empty(0)
    else:
        shared_delta = selected_direction.mean.detach().clone()
        replay_indices = selected_direction.replay_indices.detach().clone()
        replay_weights = _cheap_memory_weight(
            bank, replay_indices, settings
        ).detach().clone()
    return LightAbsorptionProposal(
        theta_old=theta_leaf.detach().clone(),
        theta_h0=theta_candidate.detach().clone(),
        shared_delta=shared_delta,
        alpha=float(alpha),
        selected_direction=selected_direction,
        replay_indices=replay_indices,
        replay_weights=replay_weights,
        candidate_indices=candidate_tensor.detach().clone(),
        clusters=clusters,
        evidence=evidence,
        updated_summary=updated_summary,
        replay_used=replay_used,
    )


def _clear_optimizer_state(
    optimizer: Optional[torch.optim.Optimizer],
    parameter: torch.nn.Parameter,
) -> None:
    if optimizer is not None:
        optimizer.state.pop(parameter, None)


@torch.no_grad()
def run_light_sleep(
    tree,
    hawkes_ll: HawkesFamily,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    settings: Optional[LightSleepSettings] = None,
    state: Optional[Dict[str, object]] = None,
    protected_leaf_ids: Collection[str] = (),
    structural_probe: Optional[
        Callable[
            [str, MemoryBank, LightAbsorptionProposal],
            Optional[Dict[str, Any]],
        ]
    ] = None,
) -> Dict[str, object]:
    """Run one fixed-topology, globally budgeted consolidation cycle."""
    settings = LightSleepSettings() if settings is None else settings
    settings.validate()
    state = {} if state is None else state
    scan_cursors = state.setdefault("scan_cursors", {})
    direction_summaries = state.setdefault("direction_summaries", {})
    protected = set(protected_leaf_ids)
    defer_reconciliation = (
        structural_probe is not None and bool(tree.memory_reconciliation)
    )
    if defer_reconciliation:
        # A structural probe must see the exact post-consolidation Bank before
        # any pending collapse overflow is deleted or compressed.  The
        # reconciliation transaction is therefore postponed until after the
        # probe/Light decision and is skipped for a protected node.
        reconciliation = {
            "reconciled_nodes": [],
            "reconciled_banks": 0,
            "removed_rows": 0,
            "deferred_banks": 0,
            "replay_windows": 0,
            "replay_events": 0,
            "merge_banks": 0,
            "prune_banks": 0,
            "utility_numerator": 0.0,
            "utility_denominator": 0.0,
            "raw_energy_numerator": 0.0,
            "raw_energy_denominator": 0.0,
        }
    else:
        reconciliation = _reconcile_contracted_banks(
            tree, hawkes_ll, settings, protected
        )
    for node_id in reconciliation["reconciled_nodes"]:
        scan_cursors.pop(node_id, None)
        direction_summaries.pop(node_id, None)
    active_leaf_set = set(tree.leaf_ids)
    for stale_leaf in set(scan_cursors).difference(active_leaf_set):
        scan_cursors.pop(stale_leaf, None)
    for stale_leaf in set(direction_summaries).difference(active_leaf_set):
        direction_summaries.pop(stale_leaf, None)
    active_leaves = [
        leaf_id
        for leaf_id in tree.leaf_ids
        if (
            leaf_id not in protected
            and
            leaf_id in tree.episodic_memory.banks
            and len(tree.episodic_memory.banks[leaf_id]) > 0
        )
    ]
    budgets = _leaf_budgets(
        active_leaves,
        tree.mass_ema,
        settings.replay_budget,
        settings.min_per_leaf,
        settings.routing_mass_mix,
    )
    remaining_budget = max(
        0,
        settings.replay_budget - reconciliation["replay_windows"],
    )
    remaining_scan_budget = (
        settings.replay_budget * settings.scan_budget_multiplier
    )
    leaf_records: Dict[str, Dict[str, object]] = {}
    total_replay_windows = int(reconciliation["replay_windows"])
    total_replay_events = int(reconciliation["replay_events"])
    total_scanned_memories = 0
    sampled_utility_numerator = float(reconciliation["utility_numerator"])
    sampled_utility_denominator = float(reconciliation["utility_denominator"])
    sampled_raw_energy_numerator = float(
        reconciliation["raw_energy_numerator"]
    )
    sampled_raw_energy_denominator = float(
        reconciliation["raw_energy_denominator"]
    )
    absorbed = 0
    probe_results: Dict[str, Dict[str, Any]] = {}
    probe_protected: set[str] = set()
    probed_leaves: set[str] = set()

    for leaf_id in sorted(protected.intersection(tree.leaf_ids)):
        bank = tree.episodic_memory.banks.get(leaf_id)
        leaf_records[leaf_id] = {
            "bank_size": 0 if bank is None else len(bank),
            "scanned_memories": 0,
            "directions": 0,
            "indexed_directions": 0,
            "evaluated_directions": 0,
            "replay_windows": 0,
            "absorbed": False,
            "protected_by_bank_mode_probe": True,
        }

    for leaf_id in active_leaves:
        leaf_budget = min(
            budgets.get(leaf_id, 0),
            remaining_budget,
        )
        if leaf_budget <= 0:
            continue
        probed_leaves.add(leaf_id)
        bank = tree.episodic_memory.banks[leaf_id]
        leaf_scan_budget = min(
            remaining_scan_budget,
            max(
                leaf_budget,
                leaf_budget * settings.scan_budget_multiplier,
            ),
        )
        candidate_indices, next_cursor, inspected = (
            _bounded_valid_window_indices(
                bank,
                start=int(scan_cursors.get(leaf_id, 0)),
                scan_budget=leaf_scan_budget,
            )
        )
        scan_cursors[leaf_id] = next_cursor
        remaining_scan_budget -= inspected
        total_scanned_memories += inspected
        theta_old = tree.semantic_theta(leaf_id).detach().clone()
        proposal = propose_light_absorption(
            bank,
            theta_old,
            hawkes_ll,
            settings,
            candidate_indices,
            direction_summary=direction_summaries.get(leaf_id),
            replay_budget=leaf_budget,
        )
        clusters = proposal.clusters
        updated_summary = proposal.updated_summary
        direction_summaries[leaf_id] = updated_summary
        evidence = proposal.evidence
        replay_used = proposal.replay_used
        for item in evidence:
            for memory_index in item.replay_indices.detach().cpu().tolist():
                window = bank.windows[memory_index]
                if window is not None:
                    total_replay_events += max(
                        0,
                        int(window.end_idx) - int(window.start_idx),
                    )
        total_replay_windows += replay_used
        remaining_budget -= replay_used

        record: Dict[str, object] = {
            "bank_size": len(bank),
            "scanned_memories": inspected,
            "directions": len(clusters),
            "indexed_directions": int(
                updated_summary.get(
                    "support",
                    bank.deltas.new_zeros(0),
                ).numel()
            ),
            "evaluated_directions": len(evidence),
            "replay_windows": replay_used,
            "absorbed": False,
        }
        probe_result = None
        if structural_probe is not None:
            probe_result = structural_probe(leaf_id, bank, proposal)
            if probe_result is not None:
                probe_results[leaf_id] = probe_result
        protected_by_probe = bool(
            probe_result is not None and probe_result.get("protect", False)
        )
        if protected_by_probe:
            probe_protected.add(leaf_id)
        if proposal.selected_direction is not None and proposal.alpha > 0.0:
            chosen = proposal.selected_direction
            if not protected_by_probe:
                theta_new = proposal.theta_h0
                tree.set_semantic_theta(leaf_id, theta_new)
                # Exact rebasing:
                # theta_new + delta_new == theta_old + delta_old.
                bank.deltas.add_(theta_old - theta_new)
                _clear_optimizer_state(
                    optimizer,
                    tree.semantic_offset[leaf_id],
                )
                # Rebasing translates every residual, so the normalized
                # direction index must be rebuilt from future bounded scans.
                direction_summaries.pop(leaf_id, None)
                absorbed += 1
                record.update({
                    "absorbed": True,
                    "support": int(chosen.indices.numel()),
                    "gain": float(chosen.gain.item()),
                    "coherence": float(chosen.coherence.item()),
                    "alpha": float(proposal.alpha),
                    "semantic_shift_norm": float(
                        (theta_new - theta_old).norm().item()
                    ),
                })
            elif probe_result is not None:
                record.update({
                    "protected_by_bank_mode_probe": True,
                    "probe_advantage": float(
                        probe_result.get("advantage", 0.0)
                    ),
                })
        if candidate_indices:
            sampled_indices = torch.tensor(
                candidate_indices,
                device=bank.device,
                dtype=torch.long,
            )
            sampled_weight = _cheap_memory_weight(
                bank,
                sampled_indices,
                settings,
            )
            sampled_utility = _predictive_residual_utility(
                bank,
                sampled_indices,
                tree.semantic_theta(leaf_id).detach(),
                hawkes_ll,
            )
            sampled_utility_numerator += float(
                (sampled_weight * sampled_utility).sum().item()
            )
            sampled_utility_denominator += float(
                sampled_weight.sum().item()
            )
            sampled_raw_energy_numerator += float((
                sampled_weight
                * bank.deltas.index_select(
                    0, sampled_indices
                ).square().sum(dim=-1)
            ).sum().item())
            sampled_raw_energy_denominator += float(
                sampled_weight.sum().item()
            )
        leaf_records[leaf_id] = record
        if remaining_budget <= 0 or remaining_scan_budget <= 0:
            break

    if defer_reconciliation:
        unprobed_pending = {
            node_id
            for node_id in set(tree.memory_reconciliation)
            if (
                node_id not in probed_leaves
                and node_id in tree.leaf_ids
                and node_id in tree.episodic_memory.banks
            )
        }
        reconciliation = _reconcile_contracted_banks(
            tree,
            hawkes_ll,
            settings,
            protected
            | probe_protected
            | unprobed_pending,
            replay_budget=max(
                0,
                settings.replay_budget - total_replay_windows,
            ),
        )
        for node_id in reconciliation["reconciled_nodes"]:
            scan_cursors.pop(node_id, None)
            direction_summaries.pop(node_id, None)
        total_replay_windows += int(reconciliation["replay_windows"])
        total_replay_events += int(reconciliation["replay_events"])
        sampled_utility_numerator += float(
            reconciliation["utility_numerator"]
        )
        sampled_utility_denominator += float(
            reconciliation["utility_denominator"]
        )
        sampled_raw_energy_numerator += float(
            reconciliation["raw_energy_numerator"]
        )
        sampled_raw_energy_denominator += float(
            reconciliation["raw_energy_denominator"]
        )

    return {
        "ran": True,
        "budget": settings.replay_budget,
        "replay_windows": total_replay_windows,
        "replay_events": total_replay_events,
        "scanned_memories": total_scanned_memories,
        "scan_budget": (
            settings.replay_budget * settings.scan_budget_multiplier
        ),
        "predictive_residual_utility": (
            sampled_utility_numerator
            / max(sampled_utility_denominator, settings.eps)
        ),
        "raw_residual_energy": (
            sampled_raw_energy_numerator
            / max(sampled_raw_energy_denominator, settings.eps)
        ),
        # Compatibility alias for old checkpoints/consumers.  The value is
        # predictive utility now; it is no longer raw ||delta||^2 energy.
        "residual_energy": (
            sampled_utility_numerator
            / max(sampled_utility_denominator, settings.eps)
        ),
        "active_leaves": len(active_leaves) + len(
            protected.intersection(tree.leaf_ids)
        ),
        "protected_leaves": len(
            protected.intersection(tree.leaf_ids)
            | probe_protected.intersection(tree.leaf_ids)
        ),
        "bank_mode_probes": probe_results,
        "absorbed_leaves": absorbed,
        "memory_reconciliation": reconciliation,
        "leaf_records": leaf_records,
    }

__all__ = [
    "DirectionEvidence",
    "LightAbsorptionProposal",
    "LightSleepSettings",
    "propose_light_absorption",
    "run_light_sleep",
]
