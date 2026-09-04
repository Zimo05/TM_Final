"""End-to-end wake/sleep training for the Hawkes Memory Tree.

The implementation follows the paper's three-timescale separation:

* event scale: update only the sequence-local working-memory delta;
* cross-sequence scale: update shared parameters from a global batch;
* sleep scale: run bounded semantic consolidation and structural transactions.

Wake never steps the global optimizer.  The global prediction graph is
recomputed after several sequences, so correlated events from one sequence
cannot modify Encoder/Router/semantic parameters dozens of times before the
next sequence is observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from HawkesBackbone import (
    EVENT_TIME_FEATURES_KEY,
    HAWKES_CACHE_SIGNATURE_KEY,
    HAWKES_HISTORY_STATS_KEY,
    HAWKES_INTERVAL_STATS_KEY,
    HawkesFamily,
)
from LatentHawkesTree import HawkesTree
from Sleep.Coordinator import run_sleep_cycle
from Sleep.Collapse import collapse_snapshot_signature
from Sleep.DeepGate import DeepSleepGate
from Sleep.Merge import (
    compute_differentiable_merge_objective,
    leaf_sibling_pairs,
)
from Sleep.Light import (
    LightSleepSettings,
    run_light_sleep,
)
from Sleep.TopologyPrune import (
    apply_prune_persistence,
    candidate_prune_parents,
    evaluate_topology_prune,
    tree_complexity,
    update_leaf_mass,
)
from Sleep.Split import SplitModule
from Sleep.UnifiedTopology import (
    UnifiedTopologySelector,
    apply_structural_inertia,
    build_merge_candidate,
    build_split_candidate,
    build_topology_prune_candidate,
    split_candidate_log_lines,
    smooth_candidate_gains,
)
from Train.ConstructTree import ConstructMemoryTree
from Train.AlignmentInitialization import run_membership_alignment
from Train.ResidualInitialization import (
    RESIDUAL_SIGNATURE_KEY,
    compute_sequence_residual_signatures,
    initialize_tree_from_residual_signatures,
    load_h_tree_leaf_membership,
)
from Train.RegionalProbe import (
    counterfactual_energy_probe,
)
from Wake.HawkesParams import HawkesParams
from Wake.SequentialController import Action, Controller
from Wake.ControllerUtilityReplay import ControllerUtilityReplay
try:
    from Routing_Retrieval_Investigation.routing_retrieval_investigation import (
        FrontierRoutingConfig,
    )
except ModuleNotFoundError as error:
    if error.name != "Routing_Retrieval_Investigation":
        raise
    from Routing_Retrieval.routing_retrieval_investigation import (
        FrontierRoutingConfig,
    )


def _assert_finite_without_cuda_sync(value: Tensor, message: str) -> None:
    """Check finite values without a per-call CUDA-to-host barrier."""
    finite = torch.isfinite(value).all()
    if value.device.type == "cuda":
        torch._assert_async(finite, message)
    elif not bool(finite):
        raise FloatingPointError(message)


def _frontier_config_from_checkpoint(
    model_config: Mapping[str, Any],
) -> FrontierRoutingConfig:
    values = dict(model_config.get("frontier_routing_config") or {})
    if model_config.get("router_kind") != "posterior_frontier_v2":
        # v1's 0.1 tempered-count prior recreates shallow-leaf bias under the
        # v2 equation. Migrate old compatible checkpoints to the exact neutral
        # descendant-mass prior.
        values["prior_weight"] = 1.0
        values["data_weight"] = 0.0
        values["entropy_bonus"] = 0.0
        values["visit_bonus"] = 0.0
    return FrontierRoutingConfig(**values)


@dataclass
class WakeObjectiveConfig:
    lambda_wm: float = 1e-3
    lambda_write: float = 1e-3
    # Persistent-memory identity is matched in effective Hawkes-law space.
    # duplicate > mode gives: refresh / append-within-mode / new-mode.
    prototype_duplicate_threshold: float = 0.98
    prototype_mode_threshold: float = 0.90
    prototype_mode_capacity: int = 12
    # Number of independent context/retrieval aliases retained per physical
    # law prototype.  Three is the small default used by Dual Identity.
    prototype_context_alias_capacity: int = 3
    # Deprecated threshold-controller fields retained so old YAML/checkpoints
    # still deserialize. The sequential controller no longer branches on them.
    write_surrogate_temperature: float = 0.25
    tau_surprise: float = 2.0
    tau_novelty: float = 0.3
    tau_count: int = 3
    tau_similarity: float = 0.5
    action_temperature: float = 1.0
    novelty_temperature: float = 10.0
    count_exponent: float = 2.0
    surprise_ema_decay: float = 0.99
    controller_write_candidate_threshold: float = 0.5
    controller_exploration_rate: float = 0.02
    controller_utility_topc_multiplier: int = 4
    controller_utility_stage_enabled: bool = True
    controller_auto_enable_utility_stage: bool = True
    controller_utility_min_epoch: int = 1
    controller_min_head_grad_norm: float = 1e-10
    controller_bootstrap_weight: float = 0.1
    controller_bootstrap_decay: float = 0.98
    controller_bootstrap_floor: float = 0.01
    controller_split_cost: float = 1e-4
    controller_entropy_weight: float = 1e-4
    controller_write_gain_threshold: float = 0.0
    controller_priority_threshold: float = 0.0
    controller_gain_reference: float = 1.0
    controller_utility_temperature: float = 0.25
    controller_utility_cost_margin: float = 0.01
    controller_retrieve_cost: float = 1e-4
    controller_adapt_cost: float = 1e-4
    controller_adapt_probe_topc: int = 16
    controller_write_probe_topc: int = 16
    controller_replay_capacities: tuple[int, int, int, int] = (
        1024, 1024, 1536, 512
    )
    controller_replay_batch_sizes: tuple[int, int, int, int] = (
        64, 64, 96, 32
    )
    controller_false_positive_weight: float = 2.0
    controller_write_admission_threshold: float = 0.6
    eta_memory_write: float = 1e-2
    # Episodic residuals are persistent and can later be promoted to a shared
    # ancestor. Bound the source gradient before the low-rank projection so a
    # single surprising replay window cannot inject an arbitrarily large
    # Hawkes-parameter residual into the tree.
    memory_write_grad_clip: float = 5.0
    write_horizon: int = 5
    # Deprecated diagnostic-only weights retained for old checkpoints and
    # command lines. Frontier posterior and likelihood-mixture values are
    # still logged/used for ownership, but never enter the backward objective.
    lambda_route_posterior: float = 0.0
    lambda_route_distill: float = 1.0
    lambda_route_mi: float = 0.2
    # Weak local branch-load regularizer against the neutral structural prior.
    lambda_route_balance: float = 0.05
    lambda_route_mix: float = 0.0
    route_energy_temperature: float = 1.0
    # Deprecated fixed-epoch gate retained for checkpoint/CLI compatibility.
    # Encoder routing gradients are now opened continuously by online
    # child-energy teacher reliability.
    route_encoder_warmup_epochs: int = 2
    # alpha_max in z_route = sg(z) + alpha_max*g_enc*(z-sg(z)).
    route_encoder_grad_scale: float = 0.1
    route_encoder_reliability_decay: float = 0.9
    route_teacher_temperature: float = 1.0
    # Counterfactual training-only probe for final-frontier internal nodes.
    # Wake's active frontier stays fixed; a small round-robin leaf subset is
    # evaluated with the true Hawkes objective to supervise refinement.
    lambda_route_probe: float = 0.1
    # Deprecated compatibility field. Kp is now computed independently for
    # each region as ceil(number of descendant leaves / 2).
    route_probe_leaves: int = 2
    route_probe_leaf_smoothing: float = 0.05
    route_probe_router_weight: float = 1.0
    route_probe_expand_weight: float = 1.0
    route_probe_leaf_weight: float = 1.0
    # Kept under the historical name for CLI/checkpoint compatibility. It is
    # now tau_p for the stop+leaf Hawkes-energy teacher.
    route_probe_residual_temperature: float = 1.0
    # tau_G for the predictive soft-min used by expansion_gain.
    route_probe_gain_temperature: float = 0.1
    # Deprecated: predictive gain no longer subtracts residual-space cost.
    route_probe_complexity_weight: float = 0.01
    route_probe_residual_rank: int = 4
    route_probe_residual_grad_clip: float = 0.0
    route_balance_batch_size: int = 64
    # Number of independent sequence rows advanced at each Wake time
    # position. This is separate from the persistent-parameter batch so the
    # recurrent activation footprint can be tuned independently.
    wake_wavefront_batch_size: int = 64
    # Number of flat event rows read by one episodic retrieval call at Wake
    # batch entry. This bounds retrieval workspace without changing causality.
    retrieval_microbatch: int = 1024
    # Deprecated compatibility fields.  A global batch now performs exactly
    # one optimizer step; repeating calibration would violate the intended
    # update timescale.
    route_balance_max_steps: int = 8
    route_balance_target_kl: float = 0.1
    # Compact-support local recurrence count. ``count_topk=None`` deliberately
    # sums all valid rows in the bank (the current 128-capacity default).
    # Appended to preserve positional construction of older configs.
    count_similarity_low: float = 0.35
    count_similarity_high: float = 0.65
    count_topk: Optional[int] = None
    count_saturation: float = 3.0
    # Appended to preserve positional construction of older configs.
    # Accepted-sample calibrated duplicate radius quantile.
    prototype_duplicate_quantile: float = 0.85


@dataclass
class SleepConfig:
    """Configuration for budgeted Light and structural Deep Sleep."""

    split_steps: int = 30
    split_lr: float = 1e-3
    # Deprecated compatibility knobs. Bank admission owns persistence and
    # Split is decided by counterfactual predictive competition, so these no
    # longer gate proposals or add a child-mass penalty.
    split_min_mass: float = 0.0
    split_min_structural_strength: float = 0.0
    split_min_effective_sample_size: float = 0.0
    # Bounded replay reserves physical windows from both Bank-mode sides;
    # this is sampling coverage, not an eligibility threshold.
    split_min_replay_per_group: int = 2
    split_init_steps: int = 30
    split_init_lr: float = 1e-2
    require_split_trigger: bool = False

    # Light Sleep: fixed topology and fixed replay-evidence budget.
    light_replay_budget: int = 32
    light_scan_budget_multiplier: int = 4
    light_min_per_leaf: int = 2
    light_mass_mix: float = 0.5
    light_max_directions: int = 3
    light_direction_similarity: float = 0.70
    light_gain_evaluations_per_direction: int = 2
    light_min_direction_support: int = 2
    light_min_gain: float = 0.0
    light_coherence_threshold: float = 0.60
    light_alpha_max: float = 0.25
    light_trust_radius: float = 0.10
    light_gain_reference: float = 0.05

    # Deep Sleep: a differentiable prediction-compression hazard gate decides
    # whether the bounded structural proposal transaction is worth its cost.
    deep_residual_energy_budget: float = 0.05
    deep_memory_budget_multiplier: float = 1.0
    deep_evidence_budget: int = 32
    # Availability rises smoothly from 0 (refractory) to 1 (available).
    deep_availability_tau: float = 3.0
    # Deprecated constructor alias retained for older TrainingCLI.py files.
    # TrainingLifecycle normalizes it into ``deep_availability_tau``.
    deep_cooldown_tau: Optional[float] = None
    deep_accumulator_decay: float = 0.8
    deep_split_demand_decay: float = 0.8
    # Deprecated name retained for checkpoint/CLI compatibility. It scales
    # persistent Bank ``split_mass`` when forming the bounded Deep pressure;
    # controller queue mass is never used for that pressure.
    deep_split_queue_scale: float = 1.0
    deep_evidence_temperature: float = 1.0
    deep_hard_concrete_temperature: float = 2.0 / 3.0
    deep_hard_concrete_gamma: float = -0.1
    deep_hard_concrete_zeta: float = 1.1
    deep_execution_threshold: float = 0.5
    deep_gate_bias_initial: float = -2.0
    deep_gate_weight_initial: float = 1.0
    # A closed gate periodically evaluates shadow proposals so the gate can
    # learn their detached prediction-compression value without committing.
    deep_probe_interval: int = 5
    deep_computation_cost: float = 0.05
    deep_prior_probability: float = 0.15
    deep_prior_weight: float = 0.01
    deep_gate_grad_clip: float = 5.0
    deep_gate_learning_rate: float = 1e-3

    # Unified topology arbitration. A learnable full-information policy is
    # trained over detached gains; physical commit uses deterministic MAP
    # with an explicit fixed-score Null action.
    action_temperature: float = 1.0
    action_gain_ema_decay: float = 0.8
    action_uncertainty_kappa: float = 1.0
    action_min_observations: int = 1
    action_selector_learning_rate: float = 1e-3
    action_selector_entropy_weight: float = 1e-2
    action_selector_grad_clip: float = 5.0
    # Smooth local topology trust region.  A recently edited region pays a
    # decaying gain penalty instead of entering a global/hard cooldown.
    topology_inertia_strength: float = 0.01
    topology_inertia_tau: float = 3.0
    # Deprecated checkpoint/CLI compatibility field.  Hard Merge cooldown is
    # no longer applied; ``topology_inertia_*`` owns edit hysteresis.
    merge_cooldown_cycles: int = 3
    # Weight for the standalone q_bank -> router distillation objective used
    # while fitting Split. It is deliberately excluded from structural gain.
    # Appended to preserve positional construction of older SleepConfig data.
    split_route_loss_weight: float = 1.0
    # Anchor for Deep's local refinement around the frozen Bank child laws.
    # This is a fitting regularizer, not a topology-selection price.
    split_anchor_weight: float = 1e-2


@dataclass
class StructureConfig:
    usage_decay: float = 0.95
    effective_usage_threshold: float = 1e-6
    leaf_mass_ema_decay: float = 0.95
    # Complete-refinement topology collapse is delayed until routing has had
    # time to stabilize. No low-mass leaf deletion exists in Sleep.
    prune_warmup_epochs: int = 10
    # Deprecated Split-commit compatibility margin. Bank-backed Split commits
    # partition the complete source Bank from its persistent mode identity;
    # direct legacy payloads may still use this parent-retention fallback.
    split_memory_hard_threshold: float = 0.0
    # Deprecated compatibility field. Merge topology commits preserve every
    # replay row after rebasing it to the fused parent semantic reference.
    merge_memory_hard_threshold: float = 0.0
    # Frozen production-vs-virtual-contraction Merge. Legacy checkpoint keys
    # (eps_merge/delta_merge/promotion thresholds) are ignored when loading.
    merge_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "min_replay": 8,
        "min_effective_sample_size": 4.0,
        "min_branch_support": 1,
        "stale_weight": 0.2,
        "dynamics_weight": 0.1,
        "gate_temperature": 1.0,
        "loss_weight": 0.1,
        "budget_ratio": 0.95,
        "dual_lr": 1e-6,
        "dual_initial": 0.0,
        "normalize_by_events": True,
    })
    promotion_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "min_support": 2.0,
        "min_gain": 0.0,
        "min_balance": 0.5,
    })
    # Complete-refinement topology pruning. Routing mass is evidence quality,
    # never the structural decision itself.
    topology_prune_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "min_replay": 8,
        "min_effective_replay": 4.0,
        "min_branch_replay": 1,
        "max_replay": 32,
        "stale_weight": 0.2,
        "uncertainty_kappa": 1.0,
        "dynamics_weight": 0.1,
        "prior_bias": 0.0,
        "prior_semantic_weight": 1.0,
        "prior_balance_weight": 1.0,
        "prior_evidence_scale": 8.0,
        "semantic_scale": 1.0,
        "gate_beta": 0.1,
        "commit_probability": 0.5,
        "min_gain": 0.0,
        # A near-zero D_hat is ambiguous once. Wait for one more Sleep
        # observation before treating it as confirmed refinement redundancy.
        "near_zero_damage_threshold": 1e-3,
        "near_zero_confirmations": 2,
        "patience": 2,
        "budget_ratio": 0.95,
        "dual_lr": 1e-3,
        "dual_initial": 0.0,
    })


@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    sleep_every: int = 1
    seed: int = 0
    checkpoint_path: str = "checkpoints/memory_tree.pt"
    best_checkpoint_path: Optional[str] = None
    validation_history_path: Optional[str] = None
    controller_diagnostics_path: Optional[str] = None
    # Plain-text per-epoch Split candidate diagnostics.  When omitted, the
    # path is derived from ``checkpoint_path`` with a descriptive suffix.
    unified_topology_log_path: Optional[str] = None
    # At the end of each train() call, persist a compact structured metrics
    # log and render one multi-panel PNG beside the checkpoint by default.
    plot_after_training: bool = True
    training_metrics_path: Optional[str] = None
    training_plot_path: Optional[str] = None
    # Router is updated only at the cross-sequence timescale, so it no longer
    # needs the old 0.1 event-wise damage-control multiplier.
    router_lr_scale: float = 1.0
    controller_only_finetune: bool = False
    controller_base_checkpoint: Optional[str] = None
    controller_target_version: int = 5
    controller_train_heads: tuple[str, ...] = ("adapt", "retrieve", "write")
    controller_write_ranking: bool = False
    frozen_state_sha256: Optional[str] = None


def _differentiable_merge_settings(
    values: Mapping[str, Any],
) -> Dict[str, Any]:
    """Normalize frozen Merge controls while ignoring legacy hard thresholds."""
    settings = {
        "min_replay": int(values.get(
            "min_replay_size", values.get("min_replay", 8)
        )),
        "min_effective_sample_size": float(
            values.get("min_effective_sample_size", 4.0)
        ),
        "min_branch_support": int(values.get("min_branch_support", 1)),
        "stale_weight": float(values.get("stale_weight", 0.2)),
        "dynamics_weight": float(values.get("dynamics_weight", 0.1)),
        "gate_temperature": float(values.get("gate_temperature", 1.0)),
        "loss_weight": float(values.get("loss_weight", 0.1)),
        "budget_ratio": float(values.get("budget_ratio", 0.95)),
        "dual_lr": float(values.get("dual_lr", 1e-6)),
        "dual_initial": float(values.get("dual_initial", 0.0)),
        "normalize_by_events": bool(
            values.get("normalize_by_events", True)
        ),
    }
    if settings["min_replay"] < 0:
        raise ValueError("merge min_replay must be non-negative")
    if settings["min_effective_sample_size"] < 0.0:
        raise ValueError(
            "merge min_effective_sample_size must be non-negative"
        )
    if settings["min_branch_support"] < 0:
        raise ValueError("merge min_branch_support must be non-negative")
    if settings["stale_weight"] < 0.0:
        raise ValueError("merge stale_weight must be non-negative")
    if settings["dynamics_weight"] < 0.0:
        raise ValueError("merge dynamics_weight must be non-negative")
    if settings["gate_temperature"] <= 0.0:
        raise ValueError("merge gate_temperature must be positive")
    if settings["loss_weight"] < 0.0:
        raise ValueError("merge loss_weight must be non-negative")
    if not 0.0 < settings["budget_ratio"] <= 1.0:
        raise ValueError("merge budget_ratio must lie in (0, 1]")
    if settings["dual_lr"] < 0.0:
        raise ValueError("merge dual_lr must be non-negative")
    if settings["dual_initial"] < 0.0:
        raise ValueError("merge dual_initial must be non-negative")
    return settings


def _topology_prune_settings(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize prediction-compression topology-prune controls."""
    settings = {
        "min_replay": int(values.get("min_replay", 8)),
        "min_effective_replay": float(values.get("min_effective_replay", 4.0)),
        "min_branch_replay": int(values.get("min_branch_replay", 1)),
        "max_replay": int(values.get("max_replay", 32)),
        "stale_weight": float(values.get("stale_weight", 0.2)),
        "uncertainty_kappa": float(values.get("uncertainty_kappa", 1.0)),
        "dynamics_weight": float(values.get("dynamics_weight", 0.1)),
        "prior_bias": float(values.get("prior_bias", 0.0)),
        "prior_semantic_weight": float(values.get("prior_semantic_weight", 1.0)),
        "prior_balance_weight": float(values.get("prior_balance_weight", 1.0)),
        "prior_evidence_scale": float(values.get("prior_evidence_scale", 8.0)),
        "semantic_scale": float(values.get("semantic_scale", 1.0)),
        "gate_beta": float(values.get("gate_beta", 0.1)),
        "commit_probability": float(values.get("commit_probability", 0.5)),
        "min_gain": float(values.get("min_gain", 0.0)),
        "near_zero_damage_threshold": float(
            values.get("near_zero_damage_threshold", 1e-3)
        ),
        "near_zero_confirmations": int(
            values.get("near_zero_confirmations", 2)
        ),
        "patience": int(values.get("patience", 2)),
        "budget_ratio": float(values.get("budget_ratio", 0.95)),
        "dual_lr": float(values.get("dual_lr", 1e-3)),
        "dual_initial": float(values.get("dual_initial", 0.0)),
    }
    if settings["min_replay"] <= 0:
        raise ValueError("topology prune min_replay must be positive")
    if settings["min_effective_replay"] <= 0.0:
        raise ValueError("topology prune min_effective_replay must be positive")
    if settings["min_branch_replay"] < 0:
        raise ValueError("topology prune min_branch_replay cannot be negative")
    if settings["max_replay"] <= 0:
        raise ValueError("topology prune max_replay must be positive")
    for name in (
        "stale_weight",
        "uncertainty_kappa",
        "dynamics_weight",
        "prior_balance_weight",
        "dual_lr",
        "dual_initial",
    ):
        if settings[name] < 0.0:
            raise ValueError(f"topology prune {name} must be non-negative")
    if (
        settings["semantic_scale"] <= 0.0
        or settings["gate_beta"] <= 0.0
        or settings["prior_evidence_scale"] <= 0.0
    ):
        raise ValueError(
            "topology prune semantic_scale/gate_beta/prior_evidence_scale "
            "must be positive"
        )
    if not 0.0 < settings["commit_probability"] < 1.0:
        raise ValueError("topology prune commit_probability must lie in (0, 1)")
    if (
        not math.isfinite(settings["near_zero_damage_threshold"])
        or settings["near_zero_damage_threshold"] < 0.0
    ):
        raise ValueError(
            "topology prune near_zero_damage_threshold must be finite and non-negative"
        )
    if settings["near_zero_confirmations"] <= 0:
        raise ValueError(
            "topology prune near_zero_confirmations must be positive"
        )
    if settings["patience"] <= 0:
        raise ValueError("topology prune patience must be positive")
    if not 0.0 < settings["budget_ratio"] <= 1.0:
        raise ValueError("topology prune budget_ratio must lie in (0, 1]")
    return settings


def marginal_route_balance_kl(responsibilities: Tensor) -> Tensor:
    """KL of batch-marginal routing against a uniform current-leaf prior.

    The mean is taken before KL, so confident complementary assignments such
    as ``[0.9, 0.1]`` and ``[0.1, 0.9]`` have zero balance penalty. This is
    intentionally different from forcing every individual route to be uniform.
    """
    if responsibilities.ndim != 2 or responsibilities.size(0) == 0:
        raise ValueError("responsibilities must have shape [B, L] with B > 0")
    if responsibilities.size(1) == 0:
        raise ValueError("responsibilities must contain at least one leaf")
    marginal = responsibilities.mean(dim=0).clamp_min(1e-12)
    return (
        marginal * (marginal.log() + responsibilities.new_tensor(
            float(responsibilities.size(1))
        ).log())
    ).sum()


def sequence_route_information(
    sequence_responsibilities: Tensor,
) -> Dict[str, Tensor]:
    """Sequence-level routing entropy, MI, and uniform-prior KL.

    Each row must already be the event-average responsibility of one sequence,

        r_bar_s = (1 / T_s) * sum_t r_{s,t}.

    Maximizing ``H(mean_s r_bar_s) - mean_s H(r_bar_s)`` encourages confident
    sequence assignments while retaining diverse load across a batch.  This is
    fundamentally different from maximizing event-level entropy, which can
    make every sequence uniformly ambiguous.
    """
    if (
        sequence_responsibilities.ndim != 2
        or sequence_responsibilities.size(0) == 0
        or sequence_responsibilities.size(1) == 0
    ):
        raise ValueError(
            "sequence_responsibilities must have shape [B, L] with B,L > 0"
        )
    probabilities = sequence_responsibilities.clamp_min(1e-12)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    marginal = probabilities.mean(dim=0)
    conditional_entropy = -(
        probabilities * probabilities.log()
    ).sum(dim=-1).mean()
    marginal_entropy = -(marginal * marginal.log()).sum()
    mutual_information = marginal_entropy - conditional_entropy
    prior_kl = (
        marginal
        * (
            marginal.log()
            + marginal.new_tensor(float(marginal.numel())).log()
        )
    ).sum()
    return {
        "conditional_entropy": conditional_entropy,
        "marginal_entropy": marginal_entropy,
        "mutual_information": mutual_information,
        "prior_kl": prior_kl,
        "marginal": marginal,
    }


def child_teacher_reliability(
    teacher: Tensor,
    student: Tensor,
    node_indices: Tensor,
    mask: Tensor,
    *,
    node_count: int,
) -> Dict[str, Tensor]:
    """Aggregate energy-only confidence over expanded local branches.

    ``teacher`` must be ``softmax(-stopgrad(E) / tau_T)`` without the fixed
    topology prior.  Reliability is its normalized binary confidence only;
    student alignment remains a diagnostic and cannot close the Encoder gate.
    Statistics are first aggregated per expanded tree node so frequently
    visited nodes cannot silently dominate the global gate.
    """
    if teacher.shape != student.shape or teacher.shape[-1] != 2:
        raise ValueError("teacher/student must align as [..., 2]")
    if node_indices.shape != mask.shape or teacher.shape[:-1] != mask.shape:
        raise ValueError("teacher/student/node mask tensors are misaligned")
    if node_indices.dtype != torch.long or mask.dtype != torch.bool:
        raise ValueError("node_indices must be long and mask must be bool")
    if node_count <= 0:
        raise ValueError("node_count must be positive")

    zero = teacher.sum() * 0.0
    if not bool(mask.any()):
        return {
            "reliability": zero,
            "teacher_confidence": zero,
            "teacher_student_js": zero,
            "teacher_student_alignment": zero,
            "observed_nodes": zero,
        }

    q = teacher[mask].detach().clamp_min(1e-12)
    p = student[mask].detach().clamp_min(1e-12)
    q = q / q.sum(dim=-1, keepdim=True)
    p = p / p.sum(dim=-1, keepdim=True)
    indices = node_indices[mask]
    if bool((indices < 0).any()) or bool((indices >= node_count).any()):
        raise ValueError("expanded node index is outside the tree")

    counts = q.new_zeros(node_count)
    counts.index_add_(0, indices, torch.ones_like(indices, dtype=q.dtype))
    confidence_rows = (
        1.0
        + (q * q.log()).sum(dim=-1) / math.log(2.0)
    ).clamp(0.0, 1.0)
    confidence_sum = q.new_zeros(node_count)
    confidence_sum.index_add_(0, indices, confidence_rows)
    mixture = 0.5 * (q + p)
    log_mixture = mixture.clamp_min(1e-12).log()
    teacher_kl = (
        q * (q.log() - log_mixture)
    ).sum(dim=-1)
    student_kl = (
        p * (p.log() - log_mixture)
    ).sum(dim=-1)
    js_rows = (
        0.5 * (teacher_kl + student_kl)
    ).clamp(0.0, math.log(2.0))
    js_sum = q.new_zeros(node_count)
    js_sum.index_add_(0, indices, js_rows)

    observed = counts > 0
    js = js_sum / counts.clamp_min(1.0)
    node_confidence = confidence_sum / counts.clamp_min(1.0)
    node_alignment = (1.0 - js / math.log(2.0)).clamp(0.0, 1.0)
    reliability = node_confidence[observed].mean()
    return {
        "reliability": reliability,
        "teacher_confidence": node_confidence[observed].mean(),
        "teacher_student_js": js[observed].mean(),
        "teacher_student_alignment": node_alignment[observed].mean(),
        "observed_nodes": observed.sum().to(q.dtype),
    }


def reliability_gated_route_state(
    z: Tensor,
    *,
    reliability: float,
    alpha_max: float,
) -> Tensor:
    """Preserve routing values while scaling only gradients into Encoder."""
    if not 0.0 <= reliability <= 1.0:
        raise ValueError("reliability must lie in [0, 1]")
    if not 0.0 <= alpha_max <= 1.0:
        raise ValueError("alpha_max must lie in [0, 1]")
    detached = z.detach()
    return detached + float(alpha_max * reliability) * (z - detached)


@torch.no_grad()
def clip_grad_norm_finite(
    named_parameters: Mapping[str, nn.Parameter],
    max_norm: float,
    *,
    context: str,
) -> float:
    """Clip a global gradient norm without float32 norm overflow.

    PyTorch's ordinary global norm can become ``inf`` while every individual
    gradient is finite: squaring a large float32 value overflows before the
    clipping scale is computed.  This implementation first checks every
    element, scales values by the largest absolute gradient, and accumulates
    only numbers in [0, 1].  The final scalar multiplication is performed in
    float64. Actual NaN/Inf gradients still fail immediately with parameter
    names, rather than being silently scaled.
    """
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")

    gradients: list[tuple[str, Tensor]] = []
    maxima: list[Tensor] = []
    for name, parameter in named_parameters.items():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        gradients.append((name, grad))
        maxima.append(grad.abs().max())

    if not gradients:
        return 0.0

    global_max = torch.stack(maxima).max()
    if not bool(torch.isfinite(global_max)):
        invalid = []
        for name, grad in gradients:
            finite = torch.isfinite(grad)
            if not bool(finite.all()):
                invalid.append(
                    f"{name}(nan={int(torch.isnan(grad).sum().item())},"
                    f" inf={int(torch.isinf(grad).sum().item())})"
                )
        raise FloatingPointError(
            f"{context}: non-finite gradient values in "
            + ", ".join(invalid[:12])
        )

    if float(global_max) == 0.0:
        return 0.0

    scaled_square_sum = global_max.new_zeros(())
    for _, grad in gradients:
        scaled = grad / global_max
        scaled_square_sum.add_(scaled.square().sum())
    total_norm = (
        global_max.double() * scaled_square_sum.double().sqrt()
    )
    if not bool(torch.isfinite(total_norm)):
        raise FloatingPointError(
            f"{context}: finite gradients produced an unrepresentable "
            "float64 global norm"
        )

    clip_scale = torch.clamp(
        total_norm.new_tensor(max_norm) / total_norm.clamp_min(1e-300),
        max=1.0,
    )
    if float(clip_scale) < 1.0:
        for parameter in named_parameters.values():
            if parameter.grad is not None:
                parameter.grad.mul_(
                    clip_scale.to(
                        device=parameter.grad.device,
                        dtype=parameter.grad.dtype,
                    )
                )
    return float(total_norm.cpu())


def normalize_cuda_rng_states(
    saved_states: Any,
    *,
    max_devices: Optional[int] = None,
) -> list[Tensor]:
    """Return checkpoint CUDA RNG states as CPU contiguous ByteTensors.

    ``torch.load(..., map_location="cuda")`` also maps RNG-state tensors to
    CUDA, but ``torch.cuda.set_rng_state`` requires CPU ByteTensors. Limiting
    the list to currently visible devices also makes a checkpoint portable
    between different ``CUDA_VISIBLE_DEVICES`` configurations.
    """
    if isinstance(saved_states, Tensor):
        candidates = [saved_states]
    elif isinstance(saved_states, (list, tuple)):
        candidates = list(saved_states)
    else:
        raise TypeError("CUDA RNG state must be a Tensor or a sequence of Tensors")

    normalized = []
    for index, state in enumerate(candidates):
        if not isinstance(state, Tensor):
            raise TypeError(f"CUDA RNG state {index} is not a Tensor")
        normalized.append(
            state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        )
    if max_devices is not None:
        if max_devices < 0:
            raise ValueError("max_devices must be non-negative")
        normalized = normalized[:max_devices]
    return normalized


@torch.no_grad()
def routing_diagnostics(responsibilities: Tensor) -> Dict[str, Any]:
    """Return conditional/marginal entropy, mutual information, and loads."""
    if responsibilities.ndim != 2 or responsibilities.size(0) == 0:
        raise ValueError("responsibilities must have shape [B, L] with B > 0")
    probabilities = responsibilities.detach().float().clamp_min(1e-12)
    marginal = probabilities.mean(dim=0)
    conditional_entropy = float(
        -(probabilities * probabilities.log()).sum(dim=-1).mean().cpu()
    )
    marginal_entropy = float(
        -(marginal * marginal.log()).sum().cpu()
    )
    mutual_information = max(0.0, marginal_entropy - conditional_entropy)
    hard_counts = torch.bincount(
        probabilities.argmax(dim=-1),
        minlength=probabilities.size(1),
    ).cpu().tolist()
    route_std = probabilities.std(dim=0, unbiased=False)
    return {
        "conditional_entropy": conditional_entropy,
        "marginal_entropy": marginal_entropy,
        "mutual_information": mutual_information,
        "max_leaf_mass": float(marginal.max().cpu()),
        "marginal_mass": marginal.cpu().tolist(),
        "hard_counts": hard_counts,
        "mean_sequence_route_std": float(route_std.mean().cpu()),
        "max_sequence_route_std": float(route_std.max().cpu()),
    }


class CausalPrefixEncoder(nn.Module):
    """Encode only events observed before the event currently predicted."""

    def __init__(
        self,
        num_event_types: int,
        z_dim: int,
        type_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if num_event_types <= 0 or z_dim <= 0:
            raise ValueError("num_event_types and z_dim must be positive")
        self.num_event_types = num_event_types
        self.z_dim = z_dim
        self.type_dim = type_dim
        self.hidden_dim = hidden_dim
        self.type_embedding = nn.Embedding(num_event_types, type_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(2, type_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(2 * type_dim, hidden_dim, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.Tanh(),
        )
        self.empty_prefix = nn.Parameter(torch.zeros(z_dim))

    def forward(
        self,
        times: Tensor,
        types: Tensor,
        event_index: int,
        time_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Return z_t from the strict prefix ``events[:event_index]``."""
        if event_index < 0 or event_index > times.numel():
            raise IndexError("event_index is outside the sequence")
        if times.ndim != 1 or types.ndim != 1 or times.numel() != types.numel():
            raise ValueError("times/types must be aligned one-dimensional tensors")
        if event_index == 0:
            return self.empty_prefix

        prefix_types = types[:event_index].long()
        if time_features is None:
            prefix_times = times[:event_index]
            previous = torch.cat(
                [prefix_times.new_zeros(1), prefix_times[:-1]]
            )
            prefix_time_features = torch.stack(
                [
                    torch.log1p(prefix_times),
                    torch.log1p(
                        (prefix_times - previous).clamp_min(0.0)
                    ),
                ],
                dim=-1,
            )
        else:
            if time_features.shape != (times.numel(), 2):
                raise ValueError(
                    "cached time_features must have shape [events, 2]"
                )
            prefix_time_features = time_features[:event_index]
        inputs = torch.cat(
            [
                self.type_embedding(prefix_types),
                self.time_projection(prefix_time_features),
            ],
            dim=-1,
        )
        encoded, _ = self.gru(inputs.unsqueeze(0))
        return self.output(encoded[0, -1])

    def forward_all_prefix(
        self,
        times: Tensor,
        types: Tensor,
        *,
        time_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Return every strict-prefix state with one recurrent pass.

        Row ``k`` is exactly the state for ``events[:k]``. The first row is
        ``empty_prefix``; subsequent rows are the projected GRU outputs after
        events ``0 .. k-1``. No current or future event enters row ``k``.
        """
        if times.ndim != 1 or types.ndim != 1 or times.numel() != types.numel():
            raise ValueError("times/types must be aligned one-dimensional tensors")
        event_count = int(times.numel())
        if event_count == 0:
            return self.empty_prefix.new_empty((0, self.z_dim))
        if time_features is None:
            previous = torch.cat([times.new_zeros(1), times[:-1]])
            time_features = torch.stack(
                [
                    torch.log1p(times),
                    torch.log1p((times - previous).clamp_min(0.0)),
                ],
                dim=-1,
            )
        elif time_features.shape != (event_count, 2):
            raise ValueError(
                "cached time_features must have shape [events, 2]"
            )

        inputs = torch.cat(
            [
                self.type_embedding(types.long()),
                self.time_projection(time_features),
            ],
            dim=-1,
        )
        encoded, _ = self.gru(inputs.unsqueeze(0))
        projected = self.output(encoded[0])
        return torch.cat(
            [
                self.empty_prefix.unsqueeze(0),
                projected[:-1],
            ],
            dim=0,
        )

    def forward_padded_prefix(
        self,
        times: Tensor,
        types: Tensor,
        valid_mask: Tensor,
        *,
        time_features: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode a padded minibatch of strict prefixes in one GRU pass.

        Returns ``(z, prefix_mask)`` with ``z[B,T,D]``. Row ``(b,k)`` sees
        exactly events ``[:k]`` from sequence ``b``; padded positions are
        zeroed and never become routing observations.
        """
        if (
            times.ndim != 2
            or types.shape != times.shape
            or valid_mask.shape != times.shape
            or valid_mask.dtype != torch.bool
        ):
            raise ValueError(
                "times/types/valid_mask must align as padded [B, T]"
            )
        lengths = valid_mask.sum(dim=-1)
        if bool((lengths <= 0).any()):
            raise ValueError("padded prefix batches cannot contain empty rows")
        if time_features is None:
            previous = torch.cat(
                [times.new_zeros(times.size(0), 1), times[:, :-1]],
                dim=1,
            )
            time_features = torch.stack(
                [
                    torch.log1p(times),
                    torch.log1p((times - previous).clamp_min(0.0)),
                ],
                dim=-1,
            )
        elif time_features.shape != (*times.shape, 2):
            raise ValueError(
                "time_features must have shape [B, T, 2]"
            )
        inputs = torch.cat(
            [
                self.type_embedding(types.long()),
                self.time_projection(time_features),
            ],
            dim=-1,
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.gru(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=times.size(1),
        )
        projected = self.output(encoded)
        strict = torch.cat(
            [
                self.empty_prefix.reshape(1, 1, -1).expand(
                    times.size(0), 1, -1
                ),
                projected[:, :-1],
            ],
            dim=1,
        )
        return strict.masked_fill(~valid_mask.unsqueeze(-1), 0.0), valid_mask
