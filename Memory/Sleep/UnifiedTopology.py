"""Unified counterfactual selection for Deep-Sleep topology actions.

The selector relaxes action probabilities, never the tree itself. Split,
Merge, and complete-refinement Topology Prune proposals are converted to the
same prediction-compression gain and compared against an explicit Null
action. Physical topology mutation is always a later deterministic
transaction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from Sleep.TopologyPrune import TopologyPruneProposal


class TopologyActionKind(StrEnum):
    NULL = "null"
    SPLIT = "split"
    MERGE = "merge"
    TOPOLOGY_PRUNE = "topology_prune"


@dataclass(frozen=True)
class UnifiedTopologyCandidate:
    """One hard action-target counterfactual in the common gain space."""

    action_id: str
    kind: TopologyActionKind
    target: Optional[str]
    claims: frozenset[str]
    payload: Any
    eligible: bool
    ready: bool
    raw_gain: torch.Tensor | float
    uncertainty: torch.Tensor | float
    conservative_gain: torch.Tensor | float
    effective_sample_size: float
    replay_size: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_log_dict(self) -> Dict[str, Any]:
        def scalar(value: torch.Tensor | float) -> float:
            return float(torch.as_tensor(value).detach().cpu())

        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "target": self.target,
            "claims": sorted(self.claims),
            "eligible": bool(self.eligible),
            "ready": bool(self.ready),
            "raw_gain": scalar(self.raw_gain),
            "uncertainty": scalar(self.uncertainty),
            "conservative_gain": scalar(self.conservative_gain),
            "effective_sample_size": float(self.effective_sample_size),
            "replay_size": int(self.replay_size),
            "diagnostics": dict(self.diagnostics),
        }


def format_split_candidate_log(
    candidate: UnifiedTopologyCandidate,
) -> Optional[str]:
    """Format one compact, node-first Split evidence line.

    The training loop owns the destination of these diagnostics.  Keep this
    formatter free of logging prefixes so a persisted epoch section can start
    directly with the node identifier (for example ``root_L``).
    """
    if candidate.kind is not TopologyActionKind.SPLIT:
        return None

    diagnostics = candidate.diagnostics
    # ``N_L_eff``/``N_R_eff`` are child ESS diagnostics.  Keep the old
    # ``child_effective_mass`` key as a compatibility fallback for logs
    # written before the two quantities were separated.
    child_effective_mass = diagnostics.get(
        "child_ess",
        diagnostics.get("child_effective_mass", (float("nan"), float("nan"))),
    )
    if not isinstance(child_effective_mass, Sequence):
        child_effective_mass = (float("nan"), float("nan"))
    child_left = (
        child_effective_mass[0]
        if len(child_effective_mass) > 0
        else float("nan")
    )
    child_right = (
        child_effective_mass[1]
        if len(child_effective_mass) > 1
        else float("nan")
    )

    def scalar(value: Any, default: float = float("nan")) -> float:
        try:
            return float(torch.as_tensor(value).detach().cpu())
        except (TypeError, ValueError, RuntimeError):
            return float(default)

    def integer(value: Any, default: int = 0) -> int:
        try:
            return int(torch.as_tensor(value).detach().cpu())
        except (TypeError, ValueError, RuntimeError):
            return int(default)

    reason = str(diagnostics.get("reason", "unknown"))
    target = candidate.target if candidate.target is not None else "<unknown>"
    route_kl = scalar(
        diagnostics.get("route_KL", diagnostics.get("route_kl", 0.0))
    )
    route_accuracy = scalar(
        diagnostics.get(
            "route_acc", diagnostics.get("route_accuracy", 0.0)
        )
    )
    structural_gain = scalar(
        diagnostics.get("G_struct", diagnostics.get("prediction_gain"))
    )
    return (
        f"{target} "
        f"Q_decision={scalar(diagnostics.get('Q_decision', 0.0)):.6g} "
        f"E_bank_struct={scalar(diagnostics.get('E_bank_struct', 0.0)):.6g} "
        f"N_persistent={scalar(diagnostics.get('N_persistent', 0.0)):.6g} "
        f"K_mode={integer(diagnostics.get('K_mode', diagnostics.get('K_law_mode', 0)))} "
        f"K_law_mode={integer(diagnostics.get('K_law_mode', 0))} "
        f"K_effective_mode={integer(diagnostics.get('K_effective_mode', 0))} "
        f"N_bank={integer(diagnostics.get('N_bank', candidate.replay_size))} "
        f"N_shadow={integer(diagnostics.get('N_shadow', 0))} "
        f"N_replay={integer(diagnostics.get('N_replay', candidate.replay_size))} "
        f"A_probe={scalar(diagnostics.get('probe_advantage', 0.0)):+.6g} "
        f"route_KL={route_kl:.6g} "
        f"route_acc={route_accuracy:.6g} "
        f"N_eff={scalar(candidate.effective_sample_size):.6g} "
        f"N_L_eff={scalar(child_left):.6g} "
        f"N_R_eff={scalar(child_right):.6g} "
        f"S_structural={scalar(diagnostics.get('structural_strength')):.6g} "
        f"G_struct={structural_gain:+.6g} "
        f"G_pred={scalar(diagnostics.get('prediction_gain')):+.6g} "
        f"G_sigma={scalar(candidate.uncertainty):.6g} "
        f"G_raw={scalar(candidate.raw_gain):+.6g} "
        f"G_conservative={scalar(candidate.conservative_gain):+.6g} "
        f"split={str(diagnostics.get('split_status', 'candidate'))} "
        f"eligible={bool(candidate.eligible)} "
        f"ready={bool(candidate.ready)} "
        f"reason={reason}"
    )


def split_candidate_log_lines(
    candidates: Sequence[UnifiedTopologyCandidate],
) -> list[str]:
    """Return one node-first diagnostic line for every Split candidate."""
    lines = []
    for candidate in candidates:
        line = format_split_candidate_log(candidate)
        if line is not None:
            lines.append(line)
    return lines


def print_split_candidate_logs(
    candidates: Sequence[UnifiedTopologyCandidate],
) -> list[str]:
    """Compatibility alias returning lines without writing to stdout.

    Older callers imported this helper by name.  Diagnostics are now routed
    to the epoch log by the training loop, so the compatibility surface is
    deliberately side-effect free.
    """
    return split_candidate_log_lines(candidates)


@dataclass(frozen=True)
class UnifiedTopologySelection:
    """Calibrated training policy plus gain-gated physical commit decode."""

    selected: Optional[UnifiedTopologyCandidate]
    selected_action_id: str
    probability_tensor: torch.Tensor
    # Learnable calibrated scores used only by softmax policy learning.
    score_tensor: torch.Tensor
    # Physical decode scores. Unsafe/non-ready/non-positive-gain actions are
    # hard-masked independently of learnable action bias and scale.
    commit_score_tensor: torch.Tensor
    gain_tensor: torch.Tensor
    action_ids: tuple[str, ...]
    candidates: tuple[UnifiedTopologyCandidate, ...]

    @property
    def is_null(self) -> bool:
        return self.selected is None

    def as_log_dict(self) -> Dict[str, Any]:
        selected_kind = (
            TopologyActionKind.NULL.value
            if self.selected is None
            else self.selected.kind.value
        )
        probability_values = self.probability_tensor.detach().cpu().tolist()
        policy_scores = self.score_tensor.detach().cpu().tolist()
        commit_scores = self.commit_score_tensor.detach().cpu().tolist()
        return {
            "selected_action_id": self.selected_action_id,
            "selected_kind": selected_kind,
            "selected_action": selected_kind,
            "selected_gain": (
                0.0
                if self.selected is None
                else float(torch.as_tensor(
                    self.selected.conservative_gain
                ).detach().cpu())
            ),
            "candidate_count": len(self.candidates),
            "probabilities": {
                action_id: float(value)
                for action_id, value in zip(
                    self.action_ids, probability_values
                )
            },
            "policy_scores": {
                action_id: float(value)
                for action_id, value in zip(self.action_ids, policy_scores)
            },
            "commit_scores": {
                action_id: float(value)
                for action_id, value in zip(self.action_ids, commit_scores)
            },
            "candidates": [item.as_log_dict() for item in self.candidates],
        }


def _invalid_candidate(
    *,
    action_id: str,
    kind: TopologyActionKind,
    target: str,
    claims: frozenset[str],
    payload: Any,
    reason: str,
) -> UnifiedTopologyCandidate:
    return UnifiedTopologyCandidate(
        action_id=action_id,
        kind=kind,
        target=target,
        claims=claims,
        payload=payload,
        eligible=False,
        ready=False,
        raw_gain=torch.tensor(float("-inf")),
        uncertainty=torch.tensor(float("inf")),
        conservative_gain=torch.tensor(float("-inf")),
        effective_sample_size=0.0,
        replay_size=0,
        diagnostics={"reason": reason},
    )


def build_split_candidate(
    leaf_id: str,
    split_module: nn.Module,
    output: Mapping[str, Any],
    *,
    topology_revision: int,
    lambda_T: float,
    uncertainty_kappa: float,
    min_child_effective_mass: float,
    min_structural_strength: float,
    min_effective_sample_size: float,
    eps: float = 1e-8,
) -> UnifiedTopologyCandidate:
    """Re-evaluate the Bank-conditioned structural H1 against the parent.

    Router likelihood is retained in the diagnostics only.  It is trained by
    Split's separate distillation term and never contributes to ``G_pred``.
    """
    action_id = f"split:{topology_revision}:{leaf_id}"
    payload = (split_module, output)
    if lambda_T < 0.0 or uncertainty_kappa < 0.0:
        raise ValueError("split topology price and uncertainty must be non-negative")
    try:
        parent = torch.as_tensor(output["logp0"])
        # New Split outputs distinguish the persistent Bank H1 from the
        # implementation router.  The old key is accepted only for payloads
        # written before that distinction existed.
        child_likelihood = output.get(
            "logp_child_bank",
            output.get("logp_child_mix"),
        )
        if child_likelihood is None:
            raise KeyError("logp_child_bank")
        children = torch.as_tensor(child_likelihood).to(parent)
        weights = torch.as_tensor(
            output.get("replay_weights", torch.ones_like(parent))
        ).to(parent)
    except (KeyError, TypeError, ValueError) as error:
        return _invalid_candidate(
            action_id=action_id,
            kind=TopologyActionKind.SPLIT,
            target=leaf_id,
            claims=frozenset((leaf_id,)),
            payload=payload,
            reason=f"missing_hard_metrics:{type(error).__name__}",
        )
    if (
        parent.ndim != 1
        or children.shape != parent.shape
        or weights.shape != parent.shape
        or parent.numel() == 0
    ):
        return _invalid_candidate(
            action_id=action_id,
            kind=TopologyActionKind.SPLIT,
            target=leaf_id,
            claims=frozenset((leaf_id,)),
            payload=payload,
            reason="misaligned_hard_metrics",
        )
    weights = weights.detach().clamp_min(0.0)
    total = weights.sum()
    total_value = float(total.detach().cpu())
    if not math.isfinite(total_value) or total_value <= 0.0:
        return _invalid_candidate(
            action_id=action_id,
            kind=TopologyActionKind.SPLIT,
            target=leaf_id,
            claims=frozenset((leaf_id,)),
            payload=payload,
            reason="zero_or_nonfinite_replay_weight",
        )
    probability = weights / total
    per_replay_gain = (children - parent).detach()
    mean_gain = (probability * per_replay_gain).sum()
    effective_sample_size = 1.0 / probability.square().sum().clamp_min(eps)
    denominator = (1.0 - probability.square().sum()).clamp_min(eps)
    variance = (
        probability * (per_replay_gain - mean_gain).square()
    ).sum() / denominator
    uncertainty = (
        float(uncertainty_kappa)
        * variance.clamp_min(0.0).sqrt()
        / effective_sample_size.clamp_min(eps).sqrt()
    )
    raw_gain = mean_gain - float(lambda_T)
    conservative_gain = raw_gain - uncertainty

    child_ess = torch.as_tensor(
        output.get("N_eff", ()),
        device=parent.device,
        dtype=parent.dtype,
    ).detach().reshape(-1)
    structural_strength_tensor = torch.as_tensor(
        output.get("structural_strength", 0.0),
        device=parent.device,
        dtype=parent.dtype,
    ).detach().reshape(())
    bank_structural_mass_tensor = torch.as_tensor(
        output.get("E_bank_struct", 0.0),
        device=parent.device,
        dtype=parent.dtype,
    ).detach().reshape(())
    proposal_ess_tensor = torch.as_tensor(
        output.get("effective_sample_size", effective_sample_size),
        device=parent.device,
        dtype=parent.dtype,
    ).detach().reshape(())
    two_sided_tensor = torch.as_tensor(
        output.get("two_sided_support", True),
        device=parent.device,
    ).detach()
    two_sided_valid = (
        two_sided_tensor.numel() == 1
        and bool(torch.isfinite(two_sided_tensor).all())
        and bool(two_sided_tensor.bool().all())
    )
    invalid_tensor = torch.as_tensor(
        output.get("invalid_proposal", False),
        device=parent.device,
    ).detach()
    invalid_flag_is_valid = (
        invalid_tensor.numel() == 1
        and bool(torch.isfinite(invalid_tensor).all())
    )
    invalid_proposal = bool(
        invalid_flag_is_valid and bool(invalid_tensor.bool().item())
    )
    if child_ess.shape == (2,):
        child_ess_values = child_ess
        child_metrics_finite = torch.isfinite(child_ess).all()
    else:
        child_ess_values = parent.new_zeros(2)
        child_metrics_finite = torch.zeros(
            (), device=parent.device, dtype=torch.bool
        )
    child_mass = torch.as_tensor(
        output.get("N_mass", output.get("N", ())),
        device=parent.device,
        dtype=parent.dtype,
    ).detach().reshape(-1)
    if child_mass.shape != (2,):
        child_mass_values = [float("nan"), float("nan")]
    else:
        child_mass_values = child_mass.cpu().tolist()
    eligible_tensor = (
        child_metrics_finite
        & torch.isfinite(bank_structural_mass_tensor)
        & torch.isfinite(structural_strength_tensor)
        & torch.isfinite(proposal_ess_tensor)
        & torch.isfinite(raw_gain)
        & torch.isfinite(uncertainty)
        & torch.as_tensor(two_sided_valid, device=parent.device)
        & torch.as_tensor(invalid_flag_is_valid, device=parent.device)
        & torch.as_tensor(not invalid_proposal, device=parent.device)
    )
    relaxed_gate = torch.as_tensor(
        output.get("g_split", 0.0),
        device=parent.device,
        dtype=parent.dtype,
    ).detach().reshape(())
    summary = torch.cat((
        torch.stack((
            raw_gain,
            uncertainty,
            conservative_gain,
            effective_sample_size,
            mean_gain,
            structural_strength_tensor,
            proposal_ess_tensor,
            relaxed_gate,
            eligible_tensor.to(parent.dtype),
        )),
        child_ess_values,
    )).detach().cpu().tolist()
    (
        raw_gain_value,
        uncertainty_value,
        conservative_gain_value,
        effective_sample_size_value,
        mean_gain_value,
        structural_strength,
        proposal_ess,
        relaxed_gate_value,
        eligible_value,
        child_ess_left,
        child_ess_right,
    ) = summary
    eligible = bool(eligible_value)
    n_bank = output.get("N_bank", parent.numel())
    n_shadow = output.get("N_shadow", 0)
    n_replay = output.get("N_replay", parent.numel())
    q_decision = float(torch.as_tensor(
        output.get("Q_decision", 0.0)
    ).detach().cpu())
    e_bank_struct = float(bank_structural_mass_tensor.cpu())
    n_persistent = float(torch.as_tensor(
        output.get("N_persistent", 0.0)
    ).detach().cpu())
    k_law_mode = int(torch.as_tensor(
        output.get("K_mode", output.get("K_law_mode", 0))
    ).detach().cpu())
    k_effective_mode = int(torch.as_tensor(
        output.get(
            "K_effective_mode",
            output.get("K_law_effective_mode", 0),
        )
    ).detach().cpu())
    return UnifiedTopologyCandidate(
        action_id=action_id,
        kind=TopologyActionKind.SPLIT,
        target=leaf_id,
        claims=frozenset((leaf_id,)),
        payload=payload,
        eligible=eligible,
        ready=eligible,
        raw_gain=raw_gain.detach(),
        uncertainty=uncertainty.detach(),
        conservative_gain=conservative_gain.detach(),
        effective_sample_size=float(effective_sample_size_value),
        replay_size=int(parent.numel()),
        diagnostics={
            "reason": "objective_competition" if eligible else "nonfinite_split_metrics",
            "N_bank": int(n_bank),
            "N_shadow": int(n_shadow),
            "N_replay": int(n_replay),
            "Q_decision": q_decision,
            "E_bank_struct": e_bank_struct,
            "N_persistent": n_persistent,
            "K_mode": k_law_mode,
            "K_law_mode": k_law_mode,
            "K_effective_mode": k_effective_mode,
            "two_sided_support": bool(two_sided_valid),
            "invalid_proposal": bool(invalid_proposal),
            "probe_advantage": float(torch.as_tensor(
                output.get("probe_advantage", 0.0)
            ).detach().cpu()),
            "probe_loss_h0": float(torch.as_tensor(
                output.get("probe_loss_h0", 0.0)
            ).detach().cpu()),
            "probe_loss_h1": float(torch.as_tensor(
                output.get("probe_loss_h1", 0.0)
            ).detach().cpu()),
            "route_kl": float(torch.as_tensor(
                output.get("route_kl", 0.0)
            ).detach().cpu()),
            "route_KL": float(torch.as_tensor(
                output.get("route_KL", output.get("route_kl", 0.0))
            ).detach().cpu()),
            "route_accuracy": float(torch.as_tensor(
                output.get("route_accuracy", 0.0)
            ).detach().cpu()),
            "route_acc": float(torch.as_tensor(
                output.get("route_acc", output.get("route_accuracy", 0.0))
            ).detach().cpu()),
            "G_struct": float(mean_gain_value),
            "prediction_gain": float(mean_gain_value),
            "complexity_delta": 1.0,
            "lambda_T": float(lambda_T),
            # New diagnostics keep physical child mass and child ESS
            # distinct.  The formatter uses child_ess for N_L_eff/N_R_eff.
            "child_effective_mass": child_mass_values,
            "child_ess": [child_ess_left, child_ess_right],
            "structural_strength": float(structural_strength),
            "proposal_effective_sample_size": float(proposal_ess),
            "relaxed_gate": float(relaxed_gate_value),
        },
    )


def build_topology_prune_candidate(
    proposal: TopologyPruneProposal,
    *,
    topology_revision: int,
) -> UnifiedTopologyCandidate:
    """Adapt an existing memory-preserving collapse to the common score."""
    action_id = f"topology_prune:{topology_revision}:{proposal.parent_id}"
    # Unified arbitration uses the same retention-cost currency as Merge:
    # prediction damage + uncertainty + dynamics cost versus complexity.
    # The learned/heuristic Prune prior remains a diagnostic calibration and
    # must not add a Prune-only preference term to the shared action score.
    conservative_gain_value = float(proposal.prune_gain)
    raw_gain_value = (
        conservative_gain_value + proposal.uncertainty_margin
    )
    eligible = bool(
        proposal.eligible
        and math.isfinite(raw_gain_value)
        and math.isfinite(proposal.prune_gain)
    )
    return UnifiedTopologyCandidate(
        action_id=action_id,
        kind=TopologyActionKind.TOPOLOGY_PRUNE,
        target=proposal.parent_id,
        claims=frozenset((proposal.parent_id, *proposal.child_ids)),
        payload=proposal,
        eligible=eligible,
        ready=bool(eligible and proposal.persistence_ok),
        raw_gain=torch.tensor(float(raw_gain_value)),
        uncertainty=torch.tensor(float(proposal.uncertainty_margin)),
        conservative_gain=torch.tensor(conservative_gain_value),
        effective_sample_size=float(proposal.effective_replay_size),
        replay_size=int(proposal.replay_size),
        diagnostics={
            "reason": proposal.reason,
            "prediction_damage": float(proposal.predictive_damage),
            "counterfactual_gain": float(proposal.prune_gain),
            "posterior_gain": float(proposal.posterior_gain),
            "retention_cost": float(proposal.retention_cost),
            "tpp_divergence": float(proposal.tpp_divergence),
            "complexity_delta": -float(proposal.complexity_saving),
            "persistence": int(proposal.persistence),
            "persistence_ok": bool(proposal.persistence_ok),
            "locally_positive": bool(proposal.locally_positive),
            "prune_probability": float(proposal.prune_probability),
        },
    )


def build_merge_candidate(
    node_a: str,
    node_b: str,
    parent_id: str,
    snapshot_signature: tuple,
    *,
    delta_keep: float,
    delta_keep_variance: float,
    replay_size: int,
    effective_sample_size: float,
    branch_support: Sequence[float],
    topology_revision: int,
    lambda_T: float,
    uncertainty_kappa: float,
    min_replay_size: int,
    min_effective_sample_size: float,
    min_branch_support: int,
    target_parent_theta: Optional[torch.Tensor] = None,
    child_weights: Optional[Sequence[float]] = None,
    tpp_divergence: float = 0.0,
    dynamics_weight: float = 0.0,
    cooldown_remaining: int = 0,
) -> UnifiedTopologyCandidate:
    """Adapt a frozen sibling Merge counterfactual to the common gain space.

    ``delta_keep`` is the importance-weighted prediction damage of the virtual
    contraction relative to current production KEEP.  The selector receives
    the same prediction + uncertainty + TPP dynamics - complexity currency as
    Topology Prune. A binary refinement is one topology-complexity unit.
    """
    if replay_size < 0:
        raise ValueError("Merge replay_size must be non-negative")
    if effective_sample_size < 0.0:
        raise ValueError("Merge effective sample size must be non-negative")
    if len(branch_support) != 2:
        raise ValueError("Merge branch_support must contain two child values")
    if (
        lambda_T < 0.0
        or uncertainty_kappa < 0.0
        or dynamics_weight < 0.0
        or tpp_divergence < 0.0
    ):
        raise ValueError("Merge topology price and uncertainty must be non-negative")
    if (
        min_replay_size < 0
        or min_effective_sample_size < 0.0
        or min_branch_support < 0
        or cooldown_remaining < 0
    ):
        raise ValueError("Merge evidence thresholds must be non-negative")
    support = tuple(float(value) for value in branch_support)
    uncertainty = (
        float(uncertainty_kappa)
        * math.sqrt(
            max(float(delta_keep_variance), 0.0)
            / max(float(effective_sample_size), 1e-8)
        )
    )
    dynamics_cost = float(dynamics_weight) * float(tpp_divergence)
    raw_gain = float(lambda_T) - float(delta_keep) - dynamics_cost
    finite = bool(
        math.isfinite(raw_gain)
        and math.isfinite(uncertainty)
        and math.isfinite(float(delta_keep))
        and math.isfinite(float(effective_sample_size))
        and math.isfinite(float(tpp_divergence))
        and math.isfinite(float(dynamics_weight))
        and all(math.isfinite(value) for value in support)
    )
    replay_sufficient = replay_size >= min_replay_size
    effective_sample_sufficient = (
        effective_sample_size >= min_effective_sample_size
    )
    branch_support_sufficient = all(
        value >= min_branch_support for value in support
    )
    eligible = bool(
        finite
        and replay_sufficient
        and effective_sample_sufficient
        and branch_support_sufficient
    )
    reason = "ready"
    if not finite:
        reason = "non_finite_evidence"
    elif not replay_sufficient:
        reason = "insufficient_replay"
    elif not effective_sample_sufficient:
        reason = "insufficient_effective_sample_size"
    elif not branch_support_sufficient:
        reason = "insufficient_branch_support"
    action_id = f"merge:{topology_revision}:{parent_id}"
    return UnifiedTopologyCandidate(
        action_id=action_id,
        kind=TopologyActionKind.MERGE,
        target=parent_id,
        claims=frozenset((parent_id, node_a, node_b)),
        payload={
            "parent_id": parent_id,
            "child_ids": (node_a, node_b),
            "snapshot_signature": snapshot_signature,
            "target_parent_theta": (
                None
                if target_parent_theta is None
                else target_parent_theta.detach().clone()
            ),
            "child_weights": (
                None
                if child_weights is None
                else tuple(float(value) for value in child_weights)
            ),
        },
        eligible=eligible,
        ready=eligible,
        raw_gain=torch.tensor(raw_gain),
        uncertainty=torch.tensor(uncertainty),
        conservative_gain=torch.tensor(raw_gain - uncertainty),
        effective_sample_size=float(effective_sample_size),
        replay_size=int(replay_size),
        diagnostics={
            "reason": reason,
            "prediction_advantage_of_children": float(delta_keep),
            "prediction_damage": float(delta_keep),
            "tpp_divergence": float(tpp_divergence),
            "dynamics_weight": float(dynamics_weight),
            "dynamics_cost": dynamics_cost,
            "retention_cost": float(delta_keep) + uncertainty + dynamics_cost,
            "complexity_delta": -1.0,
            "lambda_T": float(lambda_T),
            "branch_support": support,
            "min_replay_size": int(min_replay_size),
            "min_effective_sample_size": float(
                min_effective_sample_size
            ),
            "min_branch_support": int(min_branch_support),
            # Retained as a compatibility diagnostic only.  Local structural
            # inertia is applied uniformly to Split/Merge/Prune below.
            "cooldown_remaining": 0,
        },
    )


def apply_structural_inertia(
    candidates: Sequence[UnifiedTopologyCandidate],
    last_edit_cycle: Mapping[str, int],
    *,
    current_cycle: int,
    strength: float,
    tau: float,
) -> tuple[UnifiedTopologyCandidate, ...]:
    """Subtract a smooth, target-local edit cost from every action gain.

    ``last_edit_cycle`` is updated only after a physical transaction.  Thus
    shadow proposals and rejected actions cannot create hysteresis, and an
    edit in one branch never blocks a proposal in an unrelated branch.
    """
    if current_cycle < 0:
        raise ValueError("current_cycle must be non-negative")
    if strength < 0.0:
        raise ValueError("structural inertia strength must be non-negative")
    if tau <= 0.0:
        raise ValueError("structural inertia tau must be positive")
    result = []
    for candidate in candidates:
        last_edit = (
            None
            if candidate.target is None
            else last_edit_cycle.get(candidate.target)
        )
        if last_edit is None:
            age = math.inf
            penalty = 0.0
        else:
            age = max(int(current_cycle) - int(last_edit), 0)
            penalty = float(strength) * math.exp(-float(age) / float(tau))
        diagnostics = dict(candidate.diagnostics)
        diagnostics.update({
            "topology_age": age,
            "structural_inertia_penalty": penalty,
        })
        if not candidate.eligible or penalty == 0.0:
            result.append(replace(candidate, diagnostics=diagnostics))
            continue
        raw_gain = torch.as_tensor(candidate.raw_gain)
        conservative_gain = torch.as_tensor(candidate.conservative_gain)
        result.append(replace(
            candidate,
            raw_gain=raw_gain - raw_gain.new_tensor(penalty),
            conservative_gain=(
                conservative_gain - conservative_gain.new_tensor(penalty)
            ),
            diagnostics=diagnostics,
        ))
    return tuple(result)


def smooth_candidate_gains(
    candidates: Sequence[UnifiedTopologyCandidate],
    state: MutableMapping[str, Dict[str, float]],
    *,
    decay: float,
    temporal_uncertainty_kappa: float,
    min_observations: int,
    eps: float = 1e-8,
) -> tuple[UnifiedTopologyCandidate, ...]:
    """Apply action-keyed temporal evidence smoothing and confidence."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("action gain decay must lie in [0, 1)")
    if temporal_uncertainty_kappa < 0.0:
        raise ValueError("temporal uncertainty must be non-negative")
    if min_observations <= 0:
        raise ValueError("min_observations must be positive")
    result = []
    active = {candidate.action_id for candidate in candidates}
    for action_id in set(state).difference(active):
        state.pop(action_id, None)
    for candidate in candidates:
        if not candidate.eligible:
            result.append(candidate)
            continue
        if candidate.kind is TopologyActionKind.SPLIT:
            # Bank admission already established persistence. A Split probe
            # competes on its current frozen objective and never waits for a
            # second hand-designed consecutive-ready count.
            state.pop(candidate.action_id, None)
            diagnostics = dict(candidate.diagnostics)
            diagnostics.update({
                "instantaneous_raw_gain": float(torch.as_tensor(
                    candidate.raw_gain
                ).detach().cpu()),
                "gain_observations": 1,
                "temporal_uncertainty": 0.0,
            })
            result.append(replace(
                candidate,
                ready=True,
                diagnostics=diagnostics,
            ))
            continue
        previous = state.get(candidate.action_id)
        reference = torch.as_tensor(candidate.raw_gain).detach().reshape(())
        value = float(reference.cpu())
        if previous is None:
            mean = value
            second = value * value
            observations = 1
        else:
            mean = decay * previous["mean"] + (1.0 - decay) * value
            second = decay * previous["second"] + (1.0 - decay) * value * value
            observations = int(previous["observations"]) + 1
        state[candidate.action_id] = {
            "mean": float(mean),
            "second": float(second),
            "observations": float(observations),
        }
        temporal_variance = max(second - mean * mean, 0.0)
        temporal_uncertainty = (
            float(temporal_uncertainty_kappa)
            * math.sqrt(temporal_variance / max(observations, 1) + eps)
        )
        structural_uncertainty = float(torch.as_tensor(
            candidate.uncertainty
        ).detach().cpu())
        total_uncertainty = structural_uncertainty + temporal_uncertainty
        diagnostics = dict(candidate.diagnostics)
        diagnostics.update({
            "instantaneous_raw_gain": value,
            "gain_observations": observations,
            "temporal_uncertainty": temporal_uncertainty,
        })
        result.append(replace(
            candidate,
            ready=bool(candidate.ready and observations >= min_observations),
            raw_gain=reference.new_tensor(mean),
            uncertainty=reference.new_tensor(total_uncertainty),
            conservative_gain=reference.new_tensor(
                mean - total_uncertainty
            ),
            diagnostics=diagnostics,
        ))
    return tuple(result)


class UnifiedTopologySelector(nn.Module):
    """Learnable action calibration with deterministic physical decode."""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("action temperature must be positive")
        self.temperature = float(temperature)
        scale_initial = math.log(math.expm1(1.0))
        self.raw_action_scale = nn.Parameter(torch.full((3,), scale_initial))
        self.action_bias = nn.Parameter(torch.zeros(3))

    @property
    def positive_action_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_action_scale)

    @staticmethod
    def _action_index(kind: TopologyActionKind) -> int:
        return {
            TopologyActionKind.SPLIT: 0,
            TopologyActionKind.MERGE: 1,
            TopologyActionKind.TOPOLOGY_PRUNE: 2,
        }[kind]

    def forward(
        self,
        candidates: Sequence[UnifiedTopologyCandidate],
    ) -> UnifiedTopologySelection:
        ordered = tuple(candidates)
        reference = self.raw_action_scale
        gains = [reference.new_zeros(())]
        scores = [reference.sum() * 0.0]
        commit_scores = [reference.sum() * 0.0]
        for candidate in ordered:
            gain = torch.as_tensor(
                candidate.conservative_gain,
                device=reference.device,
                dtype=reference.dtype,
            ).reshape(()).detach()
            gains.append(gain)
            action_index = self._action_index(candidate.kind)
            calibrated = (
                self.positive_action_scale[action_index] * gain
                + self.action_bias[action_index]
            )
            scores.append(
                calibrated
                if candidate.eligible and candidate.ready
                else calibrated.new_tensor(float("-inf"))
            )
            positive_finite_gain = bool(
                (torch.isfinite(gain) & gain.gt(0.0)).detach().cpu()
            )
            structurally_safe = bool(
                candidate.eligible
                and candidate.ready
                and positive_finite_gain
            )
            commit_scores.append(
                calibrated.detach()
                if structurally_safe
                else calibrated.new_tensor(float("-inf")).detach()
            )
        gain_tensor = torch.stack(gains)
        score_tensor = torch.stack(scores)
        commit_score_tensor = torch.stack(commit_scores)
        probabilities = F.softmax(
            score_tensor / self.temperature,
            dim=0,
        )
        # Physical mutation has a non-learnable safety boundary: calibration
        # can arbitrate only among eligible, ready, strictly positive-gain
        # actions. Null is first with fixed score zero and wins exact ties.
        selected_index = int(
            torch.argmax(commit_score_tensor).detach().cpu()
        )
        selected = None if selected_index == 0 else ordered[selected_index - 1]
        ids = (TopologyActionKind.NULL.value,) + tuple(
            item.action_id for item in ordered
        )
        return UnifiedTopologySelection(
            selected=selected,
            selected_action_id=ids[selected_index],
            probability_tensor=probabilities,
            score_tensor=score_tensor,
            commit_score_tensor=commit_score_tensor,
            gain_tensor=gain_tensor,
            action_ids=ids,
            candidates=ordered,
        )

    def objective(
        self,
        selection: UnifiedTopologySelection,
        *,
        entropy_weight: float,
    ) -> Dict[str, torch.Tensor]:
        """Full-information policy loss over detached structural gains."""
        if entropy_weight < 0.0:
            raise ValueError("selector entropy weight must be non-negative")
        probability = selection.probability_tensor
        gain = selection.gain_tensor.detach()
        gain = torch.where(
            torch.isfinite(selection.score_tensor.detach()),
            gain,
            torch.zeros_like(gain),
        )
        expected_gain = (probability * gain).sum()
        entropy_regularizer = (
            probability * probability.clamp_min(1e-8).log()
        ).sum()
        loss = -expected_gain + float(entropy_weight) * entropy_regularizer
        return {
            "loss": loss,
            "expected_gain": expected_gain,
            "entropy": -entropy_regularizer,
        }


__all__ = [
    "TopologyActionKind",
    "UnifiedTopologyCandidate",
    "UnifiedTopologySelection",
    "UnifiedTopologySelector",
    "apply_structural_inertia",
    "build_merge_candidate",
    "build_split_candidate",
    "build_topology_prune_candidate",
    "format_split_candidate_log",
    "print_split_candidate_logs",
    "split_candidate_log_lines",
    "smooth_candidate_gains",
]
