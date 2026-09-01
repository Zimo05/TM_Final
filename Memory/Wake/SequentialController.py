from __future__ import annotations

import torch
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from dataclasses import dataclass
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from Wake.HawkesParams import (
    HawkesParams,
    lowrank_project_hawkes_residual,
)
from HawkesBackbone import (
    EVENT_TIME_FEATURES_KEY,
    HAWKES_CACHE_SIGNATURE_KEY,
    HAWKES_HISTORY_STATS_KEY,
    HAWKES_INTERVAL_STATS_KEY,
    HawkesFamily,
)
from MemoryResiduals.EpisodicMemory import TreeEpisodicMemory
from MemoryResiduals.MemoryBank import EventWindow, MemoryItem, TreeMemoryRead
from MemoryResiduals.SimilarityFeatures import local_recurrence_count
from MemoryResiduals.WorkingMemory import WorkingMemoryAdapter
try:
    from enum import StrEnum
except ImportError:  # Python 3.10 compatibility for local diagnostics.
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)

class Action(StrEnum):
    ASSIMILATE = "ASSIMILATE"
    RETRIEVE = "RETRIEVE"
    MEMORIZE = "MEMORIZE"
    QUEUE_SPLIT = "QUEUE-SPLIT"


@dataclass(frozen=True)
class ControllerFeatures:
    normalized_pre_action_surprise: torch.Tensor
    novelty: torch.Tensor
    soft_count: torch.Tensor
    owner_confidence: torch.Tensor
    retrieval_similarity: torch.Tensor
    retrieval_residual_norm: torch.Tensor
    working_memory_norm: torch.Tensor
    pending_write_ratio: torch.Tensor


@dataclass(frozen=True)
class ControllerOutput:
    assimilate_gate: torch.Tensor
    retrieve_gate: torch.Tensor
    write_gate: torch.Tensor
    split_gate: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.stack(
            (
                self.assimilate_gate,
                self.retrieve_gate,
                self.write_gate,
                self.split_gate,
            ),
            dim=-1,
        )

class Controller(nn.Module):
    def __init__(
        self,
        nll_fn: HawkesFamily,
        tau_s: Optional[float] = None,
        tau_n: Optional[float] = None,
        tau_c: Optional[int] = None,
        tau_sim: Optional[float] = None,
        eta_fast: float = 1e-2,
        eta_mem: float = 1e-2,
        memory_write_grad_clip: float = 5.0,
        rho_wm: float = 0.9,
        topk: int = 5,
        gamma: float = 10.0,
        write_horizon: int = 5,
        projection: str = "low-rank",
        sparse_budget: int = 256,
        lowrank_rank: int = 2,
        episodic_memory: Optional[TreeEpisodicMemory] = None,
        working_memory: Optional[WorkingMemoryAdapter] = None,
        action_temperature: float = 1.0,
        novelty_temperature: float = 10.0,
        count_exponent: float = 2.0,
        surprise_ema_decay: float = 0.99,
        controller_eps: float = 1e-8,
        write_candidate_threshold: float = 0.5,
        exploration_rate: float = 0.02,
        utility_topc_multiplier: int = 4,
        utility_stage_enabled: bool = False,
        utility_temperature: float = 0.25,
        utility_cost_margin: float = 0.01,
        count_similarity_low: float = 0.35,
        count_similarity_high: float = 0.65,
        count_topk: Optional[int] = None,
        count_saturation: float = 3.0,
    ):
        super().__init__()
        if memory_write_grad_clip <= 0.0:
            raise ValueError("memory_write_grad_clip must be positive")
        if action_temperature <= 0.0:
            raise ValueError("action_temperature must be positive")
        if novelty_temperature <= 0.0:
            raise ValueError("novelty_temperature must be positive")
        if count_exponent < 1.0:
            raise ValueError("count_exponent must be at least 1")
        if not -1.0 <= count_similarity_low < count_similarity_high <= 1.0:
            raise ValueError(
                "count similarity thresholds must satisfy "
                "-1 <= low < high <= 1"
            )
        if count_topk is not None and (
            isinstance(count_topk, bool)
            or not isinstance(count_topk, int)
            or count_topk <= 0
        ):
            raise ValueError("count_topk must be a positive integer or None")
        if count_saturation <= 0.0:
            raise ValueError("count_saturation must be positive")
        if not 0.0 <= surprise_ema_decay < 1.0:
            raise ValueError("surprise_ema_decay must lie in [0, 1)")
        if controller_eps <= 0.0:
            raise ValueError("controller_eps must be positive")
        # These are collaborating modules owned by HawkesTree/Trainer, not
        # Controller submodules. Avoid duplicate parameter registration.
        object.__setattr__(self, "nll_fn", nll_fn)

        # Retained as read-only compatibility metadata for old configs and
        # checkpoints. The differentiable controller no longer branches on
        # any of these thresholds.
        self.tau_s = tau_s
        self.tau_n = tau_n
        self.tau_c = tau_c
        self.tau_sim = tau_sim
        self.action_temperature = float(action_temperature)
        self.novelty_temperature = float(novelty_temperature)
        self.count_exponent = float(count_exponent)
        self.count_similarity_low = float(count_similarity_low)
        self.count_similarity_high = float(count_similarity_high)
        self.count_topk = None if count_topk is None else int(count_topk)
        self.count_saturation = float(count_saturation)
        self.surprise_ema_decay = float(surprise_ema_decay)
        self.controller_eps = float(controller_eps)
        if not 0.0 <= write_candidate_threshold <= 1.0:
            raise ValueError("write_candidate_threshold must lie in [0, 1]")
        if not 0.0 <= exploration_rate < 1.0:
            raise ValueError("exploration_rate must lie in [0, 1)")
        if utility_topc_multiplier < 1:
            raise ValueError("utility_topc_multiplier must be positive")
        self.write_candidate_threshold = float(write_candidate_threshold)
        self.exploration_rate = float(exploration_rate)
        self.utility_topc_multiplier = int(utility_topc_multiplier)
        self.utility_stage_enabled = bool(utility_stage_enabled)
        if utility_temperature <= 0.0:
            raise ValueError("utility_temperature must be positive")
        if utility_cost_margin < 0.0:
            raise ValueError("utility_cost_margin must be non-negative")
        self.utility_temperature = float(utility_temperature)
        self.utility_cost_margin = float(utility_cost_margin)

        self.eta_fast = eta_fast
        self.eta_mem = eta_mem
        self.memory_write_grad_clip = memory_write_grad_clip
        self.rho_wm = rho_wm

        self.topk = topk
        self.gamma = gamma
        self.write_horizon = write_horizon

        self.projection = projection
        self.sparse_budget = sparse_budget
        self.lowrank_rank = lowrank_rank
        object.__setattr__(self, "episodic_memory", episodic_memory)
        object.__setattr__(self, "working_memory", working_memory)

        # Eq. (18): b_A is fixed at zero. All directional coefficients are
        # softplus-constrained so optimization can change strength but cannot
        # reverse the intended monotone semantics.
        self.bias_assimilate = nn.Parameter(torch.tensor(0.0))
        self.bias_retrieve = nn.Parameter(torch.tensor(0.0))
        self.bias_memorize = nn.Parameter(torch.tensor(-0.5))
        self.bias_queue_split = nn.Parameter(torch.tensor(-1.5))
        raw_one = self._inverse_softplus(1.0)
        self.raw_assimilate_surprise = nn.Parameter(raw_one.clone())
        self.raw_retrieve_surprise = nn.Parameter(raw_one.clone())
        self.raw_retrieve_novelty = nn.Parameter(raw_one.clone())
        self.raw_memorize_surprise = nn.Parameter(raw_one.clone())
        self.raw_memorize_novelty = nn.Parameter(raw_one.clone())
        self.raw_memorize_count = nn.Parameter(raw_one.clone())
        self.raw_queue_surprise = nn.Parameter(raw_one.clone())
        self.raw_queue_novelty = nn.Parameter(raw_one.clone())
        self.raw_queue_count = nn.Parameter(raw_one.clone())

        # The first three inputs preserve the interpretable monotone bootstrap
        # above.  The remaining state features learn corrections without
        # coupling the four functional gates through a categorical softmax.
        self.context_gate = nn.Linear(5, 4, bias=False)
        nn.init.zeros_(self.context_gate.weight)

        self.register_buffer(
            # Generic construction remains v4-compatible; the dedicated
            # Controller-only finetune entry upgrades this buffer to v5.
            "controller_version", torch.tensor(4, dtype=torch.long)
        )
        self.register_buffer("calibration_thresholds", torch.tensor([0.0, 0.0, 0.0]))
        self.register_buffer("split_enabled", torch.tensor(True, dtype=torch.bool))
        self.register_buffer(
            "utility_temperatures", torch.full((4,), float(utility_temperature))
        )
        # Persisted online moments make delayed utility labels auditable and
        # keep their scale stable across checkpoint boundaries.
        self.register_buffer("utility_mean", torch.zeros(4))
        self.register_buffer("utility_variance", torch.ones(4))
        self.register_buffer("utility_observations", torch.zeros(4, dtype=torch.long))

        self.register_buffer("surprise_mean", torch.tensor(0.0))
        self.register_buffer("surprise_variance", torch.tensor(1.0))
        self.register_buffer(
            "surprise_observations",
            torch.tensor(0, dtype=torch.long),
        )

        # Episodic contents live only in TreeEpisodicMemory. This counter is a
        # continuous structural-evidence accumulator, not a second copy of
        # MemoryItem objects.
        self.split_queues: Dict[str, float] = defaultdict(float)

    @staticmethod
    def _inverse_softplus(value: float) -> torch.Tensor:
        target = torch.tensor(float(value))
        return target + torch.log(-torch.expm1(-target))

    def _positive(self, raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw) + self.controller_eps
        
    def attach_episodic_memory(
        self,
        episodic_memory: TreeEpisodicMemory,
    ) -> None:
        """Connect the controller to the tree-level episodic memory module."""
        object.__setattr__(self, "episodic_memory", episodic_memory)

    def _require_episodic_memory(self) -> TreeEpisodicMemory:
        if self.episodic_memory is None:
            raise RuntimeError(
                "episodic_memory is not attached. Pass it to Controller(...) "
                "or call attach_episodic_memory(...) before retrieval."
            )
        return self.episodic_memory

    def attach_working_memory(
        self,
        working_memory: WorkingMemoryAdapter,
    ) -> None:
        """Connect the controller to the sequence-level working memory."""
        object.__setattr__(self, "working_memory", working_memory)

    def _require_working_memory(self) -> WorkingMemoryAdapter:
        if self.working_memory is None:
            raise RuntimeError(
                "working_memory is not attached. Pass it to Controller(...) "
                "or call attach_working_memory(...) before updating."
            )
        return self.working_memory

    def retrieve_from_node(
        self,
        query: torch.Tensor,
        node_id: str,
        update_state: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Delegate node-level retrieval to TreeEpisodicMemory.read_nodes()."""
        episodic_memory = self._require_episodic_memory()
        delta_by_node, info_by_node = episodic_memory.read_nodes(
            query=query,
            node_ids=[node_id],
            update_state=update_state,
        )
        return delta_by_node[node_id], info_by_node[node_id]

    def retrieve_from_path(
        self,
        query: torch.Tensor,
        path_node_ids: Sequence[str],
        update_state: bool = True,
    ) -> TreeMemoryRead:
        """Delegate path-level retrieval to TreeEpisodicMemory.retrieve_path()."""
        return self._require_episodic_memory().retrieve_path(
            query=query,
            path_node_ids=path_node_ids,
            update_state=update_state,
        )

    def lowrank_project(self, delta: HawkesParams, rank: int) -> HawkesParams:
        """
        Project W_tilde[:, :, m] to rank-r for each basis m.
        mu_tilde is kept unchanged.
        """
        return lowrank_project_hawkes_residual(delta, rank)
    
    @staticmethod
    def cosine_sims(q: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        q:    [d_k]
        keys: [R, d_k]
        return:
            sims: [R]
        """
        q_norm = F.normalize(q, dim=0)
        k_norm = F.normalize(keys, dim=-1)
        return k_norm @ q_norm
    
    def leaf_novelty_count(
        self,
        q_t: torch.Tensor,
        leaf_id: str,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute smooth novelty and compact-support local recurrence count.

        Returns:
            novelty, soft_count, attention_weighted_similarity
        """
        bank = self._require_episodic_memory().get_bank(leaf_id)
        if len(bank) == 0:
            novelty = q_t.new_tensor(1.0)
            count = q_t.new_tensor(0.0)
            weighted_similarity = q_t.new_tensor(-1.0)
            return novelty, count, weighted_similarity

        sims = self.cosine_sims(q_t, bank.keys.to(q_t.device))
        beta = F.softmax(self.novelty_temperature * sims, dim=0)
        weighted_similarity = (beta * sims).sum()
        novelty = (1.0 - weighted_similarity) / 2.0

        count, _ = local_recurrence_count(
            sims,
            similarity_low=self.count_similarity_low,
            similarity_high=self.count_similarity_high,
            exponent=self.count_exponent,
            topk=self.count_topk,
            saturation=self.count_saturation,
            eps=self.controller_eps,
        )

        return novelty, count, weighted_similarity

    def normalize_surprise(
        self,
        surprise: torch.Tensor,
        *,
        update_statistics: bool = True,
    ) -> torch.Tensor:
        """Normalize surprise with stop-gradient running statistics (Eq. 14)."""
        mean = self.surprise_mean.detach().to(surprise)
        variance = self.surprise_variance.detach().to(surprise)
        normalized = (surprise - mean) / torch.sqrt(
            variance + self.controller_eps
        )

        if update_statistics:
            with torch.no_grad():
                value = surprise.detach().to(self.surprise_mean)
                decay = self.surprise_ema_decay
                difference = value - self.surprise_mean
                self.surprise_mean.mul_(decay).add_(
                    value,
                    alpha=1.0 - decay,
                )
                self.surprise_variance.mul_(decay).add_(
                    difference.square(),
                    alpha=1.0 - decay,
                )
                self.surprise_observations.add_(1)
        return normalized

    def build_features(
        self,
        normalized_surprise: torch.Tensor,
        novelty: torch.Tensor,
        count: torch.Tensor,
        *,
        owner_confidence: Optional[torch.Tensor] = None,
        retrieval_similarity: Optional[torch.Tensor] = None,
        retrieval_residual_norm: Optional[torch.Tensor] = None,
        working_memory_norm: Optional[torch.Tensor] = None,
        pending_write_ratio: Optional[torch.Tensor] = None,
    ) -> ControllerFeatures:
        """Build the single canonical Controller feature representation.

        ``normalized_surprise`` may be scalar or ``[B]``.  Every optional
        feature is converted to the same shape, so scalar inference and the
        batched Wake path cannot silently use different defaults.
        """
        reference = normalized_surprise

        def aligned(value: Optional[torch.Tensor]) -> torch.Tensor:
            if value is None:
                return reference.new_zeros(reference.shape)
            return torch.as_tensor(value).to(reference).reshape_as(reference)

        return ControllerFeatures(
            normalized_pre_action_surprise=reference,
            novelty=torch.as_tensor(novelty).to(reference).reshape_as(reference),
            soft_count=torch.as_tensor(count).to(reference).reshape_as(reference),
            owner_confidence=aligned(owner_confidence),
            retrieval_similarity=aligned(retrieval_similarity),
            retrieval_residual_norm=aligned(retrieval_residual_norm),
            working_memory_norm=aligned(working_memory_norm),
            pending_write_ratio=aligned(pending_write_ratio),
        )

    def _feature_logits(self, features: ControllerFeatures) -> torch.Tensor:
        """Evaluate the four Controller heads from canonical features."""
        surprise = features.normalized_pre_action_surprise
        novelty = features.novelty
        count = features.soft_count
        base_logits = torch.stack(
            [
                self.bias_assimilate
                - self._positive(self.raw_assimilate_surprise) * surprise,
                self.bias_retrieve
                + self._positive(self.raw_retrieve_surprise) * surprise
                - self._positive(self.raw_retrieve_novelty) * novelty,
                self.bias_memorize
                + self._positive(self.raw_memorize_surprise) * surprise
                + self._positive(self.raw_memorize_novelty) * novelty
                - self._positive(self.raw_memorize_count) * count,
                self.bias_queue_split
                + self._positive(self.raw_queue_surprise) * surprise
                + self._positive(self.raw_queue_novelty) * novelty
                + self._positive(self.raw_queue_count) * count,
            ],
            dim=-1,
        )
        context = torch.stack(
            (
                features.owner_confidence,
                features.retrieval_similarity,
                features.retrieval_residual_norm,
                features.working_memory_norm,
                features.pending_write_ratio,
            ),
            dim=-1,
        )
        return base_logits + self.context_gate(context)

    def action_distribution(
        self,
        surprise: torch.Tensor,
        novelty: torch.Tensor,
        count: torch.Tensor,
        *,
        update_statistics: bool = True,
        owner_confidence: Optional[torch.Tensor] = None,
        retrieval_similarity: Optional[torch.Tensor] = None,
        retrieval_residual_norm: Optional[torch.Tensor] = None,
        working_memory_norm: Optional[torch.Tensor] = None,
        pending_write_ratio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return the four monotone logits and soft controller probabilities."""
        normalized_surprise = self.normalize_surprise(
            surprise,
            update_statistics=update_statistics,
        )
        features = self.build_features(
            normalized_surprise, novelty, count,
            owner_confidence=owner_confidence,
            retrieval_similarity=retrieval_similarity,
            retrieval_residual_norm=retrieval_residual_norm,
            working_memory_norm=working_memory_norm,
            pending_write_ratio=pending_write_ratio,
        )
        logits = self._feature_logits(features)
        raw_probabilities = torch.sigmoid(logits / self.action_temperature)
        thresholds = self.calibration_thresholds.to(raw_probabilities)
        probabilities = raw_probabilities.clone()
        for action_index in range(3):
            probabilities[action_index] = torch.where(
                raw_probabilities[action_index] >= thresholds[action_index],
                raw_probabilities[action_index],
                raw_probabilities[action_index] * 0.0,
            )
        if not bool(self.split_enabled):
            probabilities[3] = raw_probabilities[3] * 0.0
        gates = ControllerOutput(*probabilities.unbind(dim=-1))
        return {
            "normalized_surprise": normalized_surprise,
            "features": features,
            "logits": logits,
            "raw_probabilities": raw_probabilities,
            "probabilities": probabilities,
            "gates": gates,
        }

    def action_distribution_batch(
        self,
        surprise: torch.Tensor,
        novelty: torch.Tensor,
        count: torch.Tensor,
        *,
        update_statistics: bool = True,
        owner_confidence: Optional[torch.Tensor] = None,
        retrieval_similarity: Optional[torch.Tensor] = None,
        retrieval_residual_norm: Optional[torch.Tensor] = None,
        working_memory_norm: Optional[torch.Tensor] = None,
        pending_write_ratio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Wavefront controller evaluation with output shape ``[B, 4]``.

        Every active sequence at one time position observes the same detached
        EMA snapshot. The shared statistic is updated once from that
        wavefront's moments, matching the minibatch transaction semantics.
        """
        if (
            surprise.ndim != 1
            or novelty.shape != surprise.shape
            or count.shape != surprise.shape
        ):
            raise ValueError(
                "batched surprise/novelty/count must align as [B_active]"
            )
        mean = self.surprise_mean.detach().to(surprise)
        variance = self.surprise_variance.detach().to(surprise)
        normalized = (surprise - mean) / torch.sqrt(
            variance + self.controller_eps
        )
        if update_statistics:
            with torch.no_grad():
                values = surprise.detach().to(self.surprise_mean)
                decay = self.surprise_ema_decay
                difference = values - self.surprise_mean
                # Preserve the event-count timescale when a wavefront contains
                # many simultaneous rows. Using ``decay`` only once would make
                # a batch of 64 update the EMA 64x more slowly than streaming.
                batch_decay = decay ** values.numel()
                self.surprise_mean.mul_(batch_decay).add_(
                    values.mean(),
                    alpha=1.0 - batch_decay,
                )
                self.surprise_variance.mul_(batch_decay).add_(
                    difference.square().mean(),
                    alpha=1.0 - batch_decay,
                )
                self.surprise_observations.add_(values.numel())

        features = self.build_features(
            normalized, novelty, count,
            owner_confidence=owner_confidence,
            retrieval_similarity=retrieval_similarity,
            retrieval_residual_norm=retrieval_residual_norm,
            working_memory_norm=working_memory_norm,
            pending_write_ratio=pending_write_ratio,
        )
        logits = self._feature_logits(features)
        raw_probabilities = torch.sigmoid(logits / self.action_temperature)
        thresholds = self.calibration_thresholds.to(raw_probabilities)
        probabilities = raw_probabilities.clone()
        for action_index in range(3):
            probabilities[..., action_index] = torch.where(
                raw_probabilities[..., action_index] >= thresholds[action_index],
                raw_probabilities[..., action_index],
                raw_probabilities[..., action_index] * 0.0,
            )
        if not bool(self.split_enabled):
            probabilities[..., 3] = raw_probabilities[..., 3] * 0.0
        return {
            "normalized_surprise": normalized,
            "features": features,
            "logits": logits,
            "raw_probabilities": raw_probabilities,
            "probabilities": probabilities,
        }

    @staticmethod
    def queue_weight(probabilities: torch.Tensor) -> torch.Tensor:
        """Conditional split evidence: a split vote requires a write vote."""
        return probabilities[..., 2] * probabilities[..., 3]

    def write_candidate_mask(
        self,
        probabilities: torch.Tensor,
        *,
        training: bool,
    ) -> torch.Tensor:
        selected = probabilities[..., 2] >= self.write_candidate_threshold
        if training and self.exploration_rate > 0.0:
            selected = selected | (
                torch.rand_like(probabilities[..., 2]) < self.exploration_rate
            )
        return selected

    def write_admissible(
        self,
        write_gate: float | torch.Tensor,
        write_utility: float | torch.Tensor,
        priority: float | torch.Tensor,
        *,
        future_window_complete: bool,
        priority_threshold: float = 0.0,
    ) -> bool:
        """Single admission contract shared by training and online inference."""
        gate_passed = (
            float(torch.as_tensor(write_gate).detach().cpu())
            >= (
                float(self.calibration_thresholds[2].detach().cpu())
                if int(self.controller_version.detach().cpu()) >= 5
                else max(
                    float(self.calibration_thresholds[2].detach().cpu()),
                    float(self.write_candidate_threshold),
                )
            )
        )
        priority_passed = (
            float(torch.as_tensor(priority).detach().cpu())
            > float(priority_threshold)
        )
        if int(self.controller_version.detach().cpu()) >= 6:
            # v6 is deployable: admission occurs after the construction
            # window and cannot inspect the later realized-utility label.
            return bool(future_window_complete and gate_passed and priority_passed)
        return bool(
            future_window_complete
            and gate_passed
            and float(torch.as_tensor(write_utility).detach().cpu()) > 0.0
            and priority_passed
        )

    @torch.no_grad()
    def set_calibration_thresholds(
        self, retrieve: float, adapt: float, write: float
    ) -> None:
        values = (adapt, retrieve, write)
        if any(not 0.0 <= float(value) <= 1.01 for value in values):
            raise ValueError("controller thresholds must lie in [0, 1.01]")
        self.calibration_thresholds.copy_(
            self.calibration_thresholds.new_tensor(values)
        )

    def calibration_dict(self) -> Dict[str, float]:
        values = self.calibration_thresholds.detach().cpu().tolist()
        return {
            "adapt_threshold": float(values[0]),
            "retrieve_threshold": float(values[1]),
            "write_threshold": float(values[2]),
        }

    @staticmethod
    def normalized_inverse_propensity(
        propensities: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        observed = (
            propensities * label_mask.to(propensities)
        ).sum(dim=-1).clamp_min(1e-6)
        weights = observed.reciprocal().clamp(1.0, 10.0)
        return weights / weights.mean().clamp_min(1e-6)

    def bootstrap_targets(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Map a legacy categorical policy to independent functional targets."""
        denominator = probabilities[..., 2:].sum(dim=-1).clamp_min(
            self.controller_eps
        )
        return torch.stack(
            (
                probabilities[..., 0],
                probabilities[..., 1],
                denominator.clamp_max(1.0),
                probabilities[..., 3] / denominator,
            ),
            dim=-1,
        )

    def utility_target(
        self,
        gain: torch.Tensor,
        *,
        action_index: int,
        cost_margin: Optional[float] = None,
        update_statistics: bool = False,
    ) -> torch.Tensor:
        """Convert a signed per-event counterfactual gain into a gate target.

        A positive margin is subtracted before the sigmoid, so a zero-gain
        action is discouraged instead of receiving the ambiguous target 0.5.
        """
        if not 0 <= int(action_index) < 4:
            raise ValueError("action_index must lie in [0, 3]")
        value = torch.as_tensor(gain)
        margin = self.utility_cost_margin if cost_margin is None else float(cost_margin)
        temperature = self.utility_temperatures[int(action_index)].to(value)
        target = torch.sigmoid((value - margin) / temperature)
        if update_statistics:
            with torch.no_grad():
                index = int(action_index)
                observed = value.detach().to(self.utility_mean).mean()
                count = int(self.utility_observations[index])
                delta = observed - self.utility_mean[index]
                rate = 1.0 / float(count + 1)
                self.utility_mean[index].add_(delta * rate)
                self.utility_variance[index].add_(
                    (delta.square() - self.utility_variance[index]) * rate
                )
                self.utility_observations[index].add_(1)
        return target

    @torch.no_grad()
    def update_utility_temperatures(
        self,
        utilities_by_action: Sequence[torch.Tensor],
        *,
        ema: float = 0.9,
        minimum: float = 1e-4,
        maximum: float = 1.0,
    ) -> None:
        """Robust IQR temperatures for four incomparable utility scales."""
        for index, values in enumerate(utilities_by_action):
            values = torch.as_tensor(values, dtype=self.utility_temperatures.dtype)
            if values.numel() < 4:
                continue
            quantiles = torch.quantile(values.reshape(-1), values.new_tensor([0.25, 0.75]))
            scale = ((quantiles[1] - quantiles[0]) / 1.349).clamp(minimum, maximum)
            self.utility_temperatures[index].mul_(ema).add_(scale, alpha=1.0 - ema)

    def masked_utility_loss(
        self,
        output: Mapping[str, torch.Tensor],
        targets: torch.Tensor,
        label_mask: torch.Tensor,
        utilities: torch.Tensor,
        *,
        importance_weight: Optional[torch.Tensor] = None,
        false_positive_weight: float = 2.0,
    ) -> torch.Tensor:
        """Action-specific BCE; unobserved counterfactuals have zero loss."""
        gates = output.get("raw_probabilities", output["probabilities"])
        targets = targets.to(gates)
        mask = label_mask.to(device=gates.device, dtype=gates.dtype)
        rows = F.binary_cross_entropy(
            gates.clamp(self.controller_eps, 1.0 - self.controller_eps),
            targets,
            reduction="none",
        )
        false_positive = (gates.detach() >= 0.5) & (utilities.to(gates) <= 0.0)
        rows = rows * torch.where(
            false_positive,
            rows.new_tensor(float(false_positive_weight)),
            rows.new_tensor(1.0),
        )
        # Queue-Split remains meaningful only conditional on Write.
        rows[..., 3] = rows[..., 3] * gates[..., 2].detach()
        if importance_weight is not None:
            rows = rows * importance_weight.to(gates).unsqueeze(-1)
        denominator = mask.sum().clamp_min(1.0)
        return (rows * mask).sum() / denominator

    def write_ranking_loss(
        self,
        output: Mapping[str, torch.Tensor],
        rows: Sequence[Mapping[str, object]],
        pairs: Sequence[tuple[int, int, float, float]],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Sequence-relative ranking objective for the existing Write logit."""
        scores = output["logits"][..., 2]
        zero = scores.sum() * 0.0
        pair_loss = zero
        pair_accuracy = 0.0
        if pairs:
            high = torch.as_tensor([row[0] for row in pairs], device=scores.device)
            low = torch.as_tensor([row[1] for row in pairs], device=scores.device)
            gap_weight = scores.new_tensor([row[2] for row in pairs])
            ipw = scores.new_tensor([row[3] for row in pairs])
            ipw = ipw / ipw.mean().clamp_min(self.controller_eps)
            pair_loss = (
                F.softplus(-(scores[high] - scores[low])) * gap_weight * ipw
            ).sum() / (gap_weight * ipw).sum().clamp_min(self.controller_eps)
            pair_accuracy = float(
                (scores[high].detach() > scores[low].detach()).float().mean().cpu()
            )
        targets = scores.new_tensor([float(row["relative_target"]) for row in rows])
        propensities = scores.new_tensor([
            max(float(row["probe_propensity"]), 1e-6) for row in rows
        ])
        row_ipw = propensities.reciprocal().clamp(1.0, 10.0)
        row_ipw = row_ipw / row_ipw.mean().clamp_min(self.controller_eps)
        relative = (
            F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
            * row_ipw
        ).mean() if rows else zero
        utilities = scores.new_tensor([
            float(row["raw_write_utility"]) for row in rows
        ])
        harmful_mask = utilities <= 0.0
        harmful = (
            F.binary_cross_entropy_with_logits(
                scores[harmful_mask], torch.zeros_like(scores[harmful_mask])
            )
            if bool(harmful_mask.any()) else zero
        )
        total = pair_loss + 0.25 * relative + 2.0 * harmful
        return total, {
            "pair_count": float(len(pairs)),
            "pairwise_accuracy": pair_accuracy,
            "pair_loss": float(pair_loss.detach().cpu()),
            "relative_loss": float(relative.detach().cpu()),
            "harmful_loss": float(harmful.detach().cpu()),
        }

    def supervision_loss(
        self,
        output: Mapping[str, torch.Tensor],
        *,
        utility_targets: Optional[torch.Tensor] = None,
        bootstrap_weight: float = 0.1,
        write_cost: float = 0.0,
        split_cost: float = 0.0,
        entropy_weight: float = 0.0,
        importance_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Train independent gates from stopped utility or legacy targets."""
        gates = output.get("raw_probabilities", output["probabilities"])
        legacy = F.softmax(
            output["logits"].detach() / self.action_temperature, dim=-1
        )
        bootstrap = self.bootstrap_targets(legacy).detach()
        target = bootstrap if utility_targets is None else utility_targets.detach()
        loss_rows = F.binary_cross_entropy(
            gates.clamp(self.controller_eps, 1.0 - self.controller_eps),
            target.to(gates),
            reduction="none",
        )
        # Split is meaningful only conditional on a write opportunity.
        loss_rows[..., 3] = loss_rows[..., 3] * gates[..., 2].detach()
        if importance_weight is not None:
            loss_rows = loss_rows * importance_weight.to(gates).unsqueeze(-1)
        supervised = loss_rows.mean()
        if utility_targets is None:
            supervised = supervised * float(bootstrap_weight)
        entropy = -(
            gates.clamp_min(self.controller_eps).log() * gates
            + (1.0 - gates).clamp_min(self.controller_eps).log()
            * (1.0 - gates)
        ).mean()
        return (
            supervised
            + float(write_cost) * gates[..., 2].mean()
            + float(split_cost)
            * self.queue_weight(gates).mean()
            - float(entropy_weight) * entropy
        )

    def migrate_legacy_policy(self, steps: int = 150) -> float:
        """Fit v2 independent gates to the loaded v1 categorical policy."""
        device = self.bias_retrieve.device
        dtype = self.bias_retrieve.dtype
        surprise = torch.linspace(-2.0, 2.0, 9, device=device, dtype=dtype)
        novelty = torch.linspace(0.0, 1.0, 5, device=device, dtype=dtype)
        count = torch.linspace(0.0, 1.0, 5, device=device, dtype=dtype)
        grid = torch.cartesian_prod(surprise, novelty, count)
        with torch.no_grad():
            initial = self.action_distribution_batch(
                grid[:, 0], grid[:, 1], grid[:, 2], update_statistics=False
            )
            legacy_probabilities = F.softmax(
                initial["logits"] / self.action_temperature, dim=-1
            )
            targets = self.bootstrap_targets(legacy_probabilities)
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-2)
        final_loss = 0.0
        for _ in range(max(int(steps), 1)):
            optimizer.zero_grad(set_to_none=True)
            output = self.action_distribution_batch(
                grid[:, 0], grid[:, 1], grid[:, 2], update_statistics=False
            )
            loss = F.binary_cross_entropy(output["probabilities"], targets)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        return final_loss
    
    def choose_action(
        self,
        surprise: torch.Tensor,
        novelty: torch.Tensor,
        count: torch.Tensor,
    ) -> Action:
        """Return argmax only for logging; state updates use all probabilities."""
        output = self.action_distribution(
            surprise,
            novelty,
            count,
        )
        action_index = int(output["probabilities"].detach().argmax().item())
        return tuple(Action)[action_index]

    def soft_write_probability(
        self,
        surprise: torch.Tensor,
        novelty: torch.Tensor,
        temperature: float = 0.25,
    ) -> torch.Tensor:
        """Compatibility helper returning the independent write gate."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        count = novelty.new_zeros(())
        output = self.action_distribution(
            surprise,
            novelty,
            count,
            update_statistics=False,
        )
        return output["probabilities"][2]
    
    def update_working_memory(
        self,
        loss: torch.Tensor,
        delta_used: torch.Tensor,
        adaptation_probability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Delegate the online update to WorkingMemoryAdapter.

        ``delta_used`` must be the exact trainable working-memory tensor used
        to compute ``loss``; in the current tree flow this is the
        ``memory_output["working_delta"]`` returned by HawkesTree.forward().
        """
        working_memory = self._require_working_memory()
        working_memory.update(
            loss=loss,
            delta_used=delta_used,
            adaptation_probability=adaptation_probability,
        )
        return working_memory.delta
    
    def project_delta(self, delta: HawkesParams) -> HawkesParams:
        return self.lowrank_project(delta, self.lowrank_rank)

    @staticmethod
    def pack_hawkes_params(params: HawkesParams) -> torch.Tensor:
        """Pack scalar or leading-batched Hawkes parameters."""
        if params.mu_tilde.ndim < 1 or params.W_tilde.ndim < 3:
            raise ValueError("invalid Hawkes parameter shapes")
        return torch.cat(
            [
                params.mu_tilde.reshape(*params.mu_tilde.shape[:-1], -1),
                params.W_tilde.reshape(*params.W_tilde.shape[:-3], -1),
            ],
            dim=-1,
        )
    
    def write_residual_memory(
        self,
        q_t: torch.Tensor,
        theta_sem_leaf: HawkesParams,
        times: torch.Tensor,
        types: torch.Tensor,
        k: int,
        node_id: str = "",
        cached_sequence: Optional[Mapping[str, Any]] = None,
        write_quality: float | torch.Tensor = 1.0,
        queue_weight: float | torch.Tensor = 0.0,
        window_events: Optional[int] = None,
    ) -> MemoryItem:
        """
        Implements Eq. 17:

            Delta theta_write =
                P_r(
                    - eta_mem * grad_theta
                    sum_{tau=t}^{t+h} loss_tau(theta_sem_leaf)
                )

        The caller must delay this operation until the complete future window
        has actually arrived. Both MemoryTreeTrainer and MemoryTreeInference
        queue requests and commit them only at ``k + write_horizon``.
        """
        K = times.shape[0]
        horizon_events = (
            self.write_horizon + 1
            if window_events is None
            else int(window_events)
        )
        if horizon_events <= 0:
            raise ValueError("window_events must be positive")
        end = min(K, k + horizon_events)

        theta_local = theta_sem_leaf.clone_detached(requires_grad=True)
        sequence = self.nll_fn.prepare_sequence_cache(
            (
                {"times": times, "types": types}
                if cached_sequence is None
                else cached_sequence
            ),
            inplace=cached_sequence is not None,
        )

        window_loss = times.new_tensor(0.0)

        for j in range(k, end):
            window_loss = window_loss + self.nll_fn.event_NLL(
                sequence=sequence,
                params=theta_local,
                k=j,
            )

        grad_mu, grad_W = torch.autograd.grad(
            window_loss,
            [theta_local.mu_tilde, theta_local.W_tilde],
            retain_graph=False,
            create_graph=False,
        )
        packed_grad = torch.cat(
            [grad_mu.detach().reshape(-1), grad_W.detach().reshape(-1)]
        )
        if not bool(torch.isfinite(packed_grad).all()):
            raise FloatingPointError(
                f"episodic write gradient became non-finite at "
                f"node={node_id} event={k}"
            )
        # The parameter vector is small, so float64 norm evaluation is cheap
        # and avoids float32 square overflow before persistent state is built.
        grad_norm = packed_grad.double().norm()
        grad_scale = torch.clamp(
            grad_norm.new_tensor(self.memory_write_grad_clip)
            / grad_norm.clamp_min(1e-300),
            max=1.0,
        ).to(device=grad_mu.device, dtype=grad_mu.dtype)

        raw_delta = HawkesParams(
            -self.eta_mem * grad_mu.detach() * grad_scale,
            -self.eta_mem * grad_W.detach() * grad_scale,
        )

        delta_write = self.project_delta(raw_delta)
        packed_delta = self.pack_hawkes_params(delta_write)
        if not bool(torch.isfinite(packed_delta).all()):
            raise FloatingPointError(
                f"episodic residual became non-finite at "
                f"node={node_id} event={k}"
            )
        window = EventWindow(
            # Keep the full prefix so sleep replay can condition the target
            # events on exactly the same Hawkes history used above.
            times=times[:end].detach().clone(),
            types=types[:end].detach().clone(),
            node_id=node_id,
            start_idx=k,
            end_idx=end,
            has_full_history=True,
            event_time_features=sequence[EVENT_TIME_FEATURES_KEY][
                :end
            ].detach().clone(),
            hawkes_history_stats=sequence[HAWKES_HISTORY_STATS_KEY][
                :end
            ].detach().clone(),
            hawkes_interval_stats=sequence[HAWKES_INTERVAL_STATS_KEY][
                :end
            ].detach().clone(),
            hawkes_cache_signature=sequence[HAWKES_CACHE_SIGNATURE_KEY],
        )

        item = MemoryItem(
            key=q_t.detach().clone(),
            delta_theta=packed_delta.detach(),
            window=window,
            usage=1,
            age=0,
            write_quality=float(torch.as_tensor(write_quality).detach().cpu()),
            queue_weight=float(torch.as_tensor(queue_weight).detach().cpu()),
        )

        return item

    def _residual_delta_batch(
        self,
        theta_semantic: HawkesParams,
        times: torch.Tensor,
        types: torch.Tensor,
        event_indices: Sequence[int] | torch.Tensor,
        *,
        cached_sequence: Mapping[str, Any],
        sequence_rows: Optional[Sequence[int] | torch.Tensor] = None,
        sequence_lengths: Optional[Sequence[int] | torch.Tensor] = None,
        window_events: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute independent residuals for a scalar or padded sequence batch.

        ``times``/``types`` may be one-dimensional for the backwards-compatible
        single-sequence API or padded as ``[B, T]`` for Wake.  ``sequence_rows``
        maps each candidate to its padded sequence row.  The parameter rows are
        independent, so one ``autograd.grad`` call still returns one gradient
        per candidate row.
        """
        if times.ndim not in (1, 2) or types.shape != times.shape:
            raise ValueError("times and types must have matching [T] or [B, T] shapes")
        if types.dtype != torch.long:
            types = types.long()
        batch_size = int(theta_semantic.mu_tilde.size(0))
        if theta_semantic.mu_tilde.ndim != 2 or batch_size == 0:
            raise ValueError("batched semantic mu must have shape [Q, D]")
        D = self.nll_fn.num_types
        M = self.nll_fn.num_basis
        if theta_semantic.mu_tilde.shape != (batch_size, D):
            raise ValueError("batched semantic mu has the wrong shape")
        if theta_semantic.W_tilde.shape != (batch_size, D, D, M):
            raise ValueError("batched semantic W has the wrong shape")

        event_indices = torch.as_tensor(
            event_indices,
            device=times.device,
            dtype=torch.long,
        ).reshape(-1)
        if event_indices.shape != (batch_size,):
            raise ValueError("event_indices must contain one index per candidate")

        if times.ndim == 1:
            sequence_count = 1
            padded_times = times.reshape(1, -1)
            padded_types = types.reshape(1, -1)
            cache_history = cached_sequence[HAWKES_HISTORY_STATS_KEY].reshape(
                1, times.numel(), D, M
            )
            cache_interval = cached_sequence[HAWKES_INTERVAL_STATS_KEY].reshape(
                1, times.numel(), D, M
            )
            candidate_rows = torch.zeros(
                batch_size, device=times.device, dtype=torch.long
            )
            lengths = torch.full(
                (1,), times.numel(), device=times.device, dtype=torch.long
            )
        else:
            sequence_count = int(times.size(0))
            padded_times = times
            padded_types = types
            cache_history = cached_sequence[HAWKES_HISTORY_STATS_KEY]
            cache_interval = cached_sequence[HAWKES_INTERVAL_STATS_KEY]
            if cache_history.shape[:2] != times.shape:
                raise ValueError("cached history does not align with padded times")
            if cache_interval.shape != cache_history.shape:
                raise ValueError("cached interval does not align with history")
            if sequence_rows is None:
                if batch_size != sequence_count:
                    raise ValueError(
                        "sequence_rows is required when candidate and sequence counts differ"
                    )
                candidate_rows = torch.arange(
                    batch_size, device=times.device, dtype=torch.long
                )
            else:
                candidate_rows = torch.as_tensor(
                    sequence_rows, device=times.device, dtype=torch.long
                ).reshape(-1)
                if candidate_rows.shape != (batch_size,):
                    raise ValueError("sequence_rows must contain one row per candidate")
            lengths = (
                torch.as_tensor(
                    sequence_lengths, device=times.device, dtype=torch.long
                ).reshape(-1)
                if sequence_lengths is not None
                else torch.full(
                    (sequence_count,), times.size(1),
                    device=times.device, dtype=torch.long,
                )
            )
            if lengths.shape != (sequence_count,):
                raise ValueError("sequence_lengths must contain one value per sequence")
        if torch.any(candidate_rows < 0) or torch.any(candidate_rows >= sequence_count):
            raise ValueError("sequence_rows contain an out-of-range sequence")
        if torch.any(lengths <= 0) or torch.any(lengths > padded_times.size(1)):
            raise ValueError("sequence lengths must lie in [1, T]")

        horizon = (
            self.write_horizon + 1
            if window_events is None
            else int(window_events)
        )
        if horizon <= 0:
            raise ValueError("window_events must be positive")
        offsets = torch.arange(horizon, device=times.device, dtype=torch.long)
        event_grid = event_indices[:, None] + offsets[None, :]
        row_lengths = lengths.index_select(0, candidate_rows)
        valid = event_grid < row_lengths[:, None]
        safe_event = event_grid.clamp_max(padded_times.size(1) - 1)
        candidate_rows_grid = candidate_rows[:, None].expand(-1, horizon)

        history = cache_history[candidate_rows_grid, safe_event]
        interval = cache_interval[candidate_rows_grid, safe_event]
        event_types = padded_types[candidate_rows_grid, safe_event]
        previous = torch.cat(
            [
                padded_times.new_zeros(padded_times.size(0), 1),
                padded_times[:, :-1],
            ],
            dim=1,
        )
        durations = (padded_times - previous).clamp_min(0.0)
        durations = durations[candidate_rows_grid, safe_event]

        mu_raw = theta_semantic.mu_tilde.detach().clone().requires_grad_(True)
        W_raw = theta_semantic.W_tilde.detach().clone().requires_grad_(True)
        mu = F.softplus(mu_raw)
        W = F.softplus(W_raw)
        intensity = (
            mu[:, None, :]
            + torch.einsum("qdem,qhem->qhd", W, history)
        ).clamp_min(1e-8)
        selected = intensity.gather(2, event_types.unsqueeze(-1)).squeeze(-1)
        event_loss = (
            -selected.log()
            + mu.sum(dim=-1, keepdim=True) * durations
            + torch.einsum("qdem,qhem->qh", W, interval)
        ).masked_fill(~valid, 0.0)
        grad_mu, grad_W = torch.autograd.grad(
            event_loss.sum(),
            [mu_raw, W_raw],
            retain_graph=False,
            create_graph=False,
        )
        packed_grad = torch.cat(
            [grad_mu.flatten(1), grad_W.flatten(1)], dim=-1
        )
        finite = torch.isfinite(packed_grad).all()
        if packed_grad.device.type == "cuda":
            torch._assert_async(
                finite,
                "batched episodic-write gradient became non-finite",
            )
        elif not bool(finite):
            raise FloatingPointError(
                "batched episodic-write gradient became non-finite"
            )
        grad_norm = packed_grad.double().norm(dim=-1)
        grad_scale = torch.clamp(
            grad_norm.new_full(grad_norm.shape, self.memory_write_grad_clip)
            / grad_norm.clamp_min(1e-300),
            max=1.0,
        ).to(device=grad_mu.device, dtype=grad_mu.dtype)
        raw_delta = HawkesParams(
            -self.eta_mem * grad_mu.detach() * grad_scale[:, None],
            -self.eta_mem * grad_W.detach() * grad_scale[:, None, None, None],
        )
        delta_write = self.project_delta(raw_delta)
        packed_delta = self.pack_hawkes_params(delta_write).detach()
        finite = torch.isfinite(packed_delta).all()
        if packed_delta.device.type == "cuda":
            torch._assert_async(
                finite,
                "batched episodic residual became non-finite",
            )
        elif not bool(finite):
            raise FloatingPointError(
                "batched episodic residual became non-finite"
            )
        ends = torch.minimum(
            event_indices + horizon,
            row_lengths,
        )
        return packed_delta, ends

    def materialize_residual_memory_items_batch(
        self,
        *,
        queries: torch.Tensor,
        delta_theta: torch.Tensor,
        times: torch.Tensor,
        types: torch.Tensor,
        event_indices: Sequence[int] | torch.Tensor,
        node_ids: Sequence[str],
        cached_sequence: Mapping[str, Any],
        write_quality: torch.Tensor,
        queue_weight: torch.Tensor,
        sequence_rows: Optional[Sequence[int] | torch.Tensor] = None,
        sequence_lengths: Optional[Sequence[int] | torch.Tensor] = None,
        window_events: Optional[int] = None,
    ) -> List[MemoryItem]:
        """Create ``MemoryItem`` objects for already-computed residuals.

        Wake uses this only after segmented top-4 admission.  Rejected probe
        rows therefore stay as device tensors and do not allocate/copy an
        ``EventWindow`` or Python object.
        """
        event_indices = torch.as_tensor(event_indices, dtype=torch.long)
        count = int(event_indices.numel())
        if (
            queries.ndim != 2
            or delta_theta.shape[0] != count
            or queries.size(0) != count
            or len(node_ids) != count
        ):
            raise ValueError("materialized residual inputs do not align")
        if times.ndim == 1:
            row_values = [0] * count
            lengths = [int(times.numel())] * count
        elif times.ndim == 2:
            row_tensor = (
                torch.as_tensor(sequence_rows, dtype=torch.long).reshape(-1)
                if sequence_rows is not None
                else torch.arange(count, dtype=torch.long)
            )
            if row_tensor.numel() != count:
                raise ValueError("sequence_rows must contain one row per item")
            length_tensor = (
                torch.as_tensor(sequence_lengths, dtype=torch.long).reshape(-1)
                if sequence_lengths is not None
                else torch.full((times.size(0),), times.size(1), dtype=torch.long)
            )
            row_values = [int(value) for value in row_tensor.tolist()]
            lengths = [int(value) for value in length_tensor.tolist()]
        else:
            raise ValueError("times must have shape [T] or [B, T]")
        horizon = (
            self.write_horizon + 1
            if window_events is None
            else int(window_events)
        )
        starts = [int(value) for value in event_indices.tolist()]
        qualities = torch.as_tensor(write_quality).reshape(count)
        queues = torch.as_tensor(queue_weight).reshape(count)
        items: List[MemoryItem] = []
        for row, (start, node_id) in enumerate(zip(starts, node_ids)):
            sequence_row = row_values[row]
            end = min(start + horizon, lengths[sequence_row])
            if times.ndim == 1:
                row_times = times[:end]
                row_types = types[:end]
                row_history = cached_sequence[HAWKES_HISTORY_STATS_KEY][:end]
                row_interval = cached_sequence[HAWKES_INTERVAL_STATS_KEY][:end]
                row_features = cached_sequence[EVENT_TIME_FEATURES_KEY][:end]
            else:
                row_times = times[sequence_row, :end]
                row_types = types[sequence_row, :end]
                row_history = cached_sequence[HAWKES_HISTORY_STATS_KEY][
                    sequence_row, :end
                ]
                row_interval = cached_sequence[HAWKES_INTERVAL_STATS_KEY][
                    sequence_row, :end
                ]
                row_features = cached_sequence[EVENT_TIME_FEATURES_KEY][
                    sequence_row, :end
                ]
            window = EventWindow(
                times=row_times.detach().clone(),
                types=row_types.detach().clone(),
                node_id=node_id,
                start_idx=start,
                end_idx=end,
                has_full_history=True,
                event_time_features=row_features.detach().clone(),
                hawkes_history_stats=row_history.detach().clone(),
                hawkes_interval_stats=row_interval.detach().clone(),
                hawkes_cache_signature=cached_sequence.get(
                    HAWKES_CACHE_SIGNATURE_KEY
                ),
            )
            items.append(MemoryItem(
                key=queries[row].detach().clone(),
                delta_theta=delta_theta[row].detach().clone(),
                window=window,
                usage=1,
                age=0,
                write_quality=float(qualities[row].detach().cpu()),
                queue_weight=float(queues[row].detach().cpu()),
            ))
        return items

    def write_residual_memory_batch(
        self,
        queries: torch.Tensor,
        theta_semantic: HawkesParams,
        times: torch.Tensor,
        types: torch.Tensor,
        event_indices: Sequence[int],
        node_ids: Sequence[str],
        *,
        cached_sequence: Mapping[str, Any],
        write_quality: torch.Tensor,
        queue_weight: torch.Tensor,
        sequence_rows: Optional[Sequence[int] | torch.Tensor] = None,
        sequence_lengths: Optional[Sequence[int] | torch.Tensor] = None,
        window_events: Optional[int] = None,
    ) -> List[MemoryItem]:
        """Compute independent residual writes with one batched gradient call."""
        batch_size = len(event_indices)
        if batch_size == 0:
            return []
        if (
            queries.ndim != 2
            or queries.size(0) != batch_size
            or theta_semantic.mu_tilde.shape
            != (batch_size, self.nll_fn.num_types)
            or theta_semantic.W_tilde.shape
            != (
                batch_size,
                self.nll_fn.num_types,
                self.nll_fn.num_types,
                self.nll_fn.num_basis,
            )
            or len(node_ids) != batch_size
        ):
            raise ValueError("batched residual-write inputs do not align")
        packed_delta, _ = self._residual_delta_batch(
            theta_semantic,
            times,
            types,
            event_indices,
            cached_sequence=cached_sequence,
            sequence_rows=sequence_rows,
            sequence_lengths=sequence_lengths,
            window_events=window_events,
        )
        return self.materialize_residual_memory_items_batch(
            queries=queries,
            delta_theta=packed_delta,
            times=times,
            types=types,
            event_indices=event_indices,
            node_ids=node_ids,
            cached_sequence=cached_sequence,
            write_quality=write_quality,
            queue_weight=queue_weight,
            sequence_rows=sequence_rows,
            sequence_lengths=sequence_lengths,
            window_events=window_events,
        )
    
    def step(
        self,
        times: torch.Tensor,
        types: torch.Tensor,
        k: int,
        q_t: torch.Tensor,
        leaf_ids: List[str],
        r_t: torch.Tensor,
        sem_params: Dict[str, HawkesParams],
        paths: Dict[str, List[str]],
        wm_delta: HawkesParams,
    ) -> Dict:
        """
        One online wake-phase event step.

        Returns:
            {
                "loss": loss_t,
                "surprise": s_t,
                "novelty": n_t,
                "count": c_t,
                "action": action,
                "hat_leaf": hat_leaf,
                "theta_eff": theta_eff,
                "wm_next": wm_next,
            }
        """
        # 1. Form theta_eff from the same TreeEpisodicMemory used by sleep.
        episodic_memory = self._require_episodic_memory()
        routing = r_t.reshape(-1)
        if routing.numel() != len(leaf_ids):
            raise ValueError("r_t must contain one responsibility per leaf")
        unique_nodes = list(dict.fromkeys(
            node_id for leaf_id in leaf_ids for node_id in paths[leaf_id]
        ))
        delta_by_node, info_by_node = episodic_memory.read_nodes(
            query=q_t,
            node_ids=unique_nodes,
            update_state=False,
        )
        episodic_delta = torch.stack(
            [episodic_memory.aggregate_path(paths[leaf_id], delta_by_node) for leaf_id in leaf_ids],
            dim=0,
        )
        semantic_mu = torch.stack(
            [sem_params[leaf_id].mu_tilde for leaf_id in leaf_ids], dim=0
        )
        semantic_W = torch.stack(
            [sem_params[leaf_id].W_tilde for leaf_id in leaf_ids], dim=0
        )
        working_flat = self.pack_hawkes_params(wm_delta)
        if not working_flat.requires_grad:
            working_flat = working_flat.detach().clone().requires_grad_(True)
        pre_action_theta = episodic_memory.parameter_update.compose_effective_parameters(
            semantic_mu=semantic_mu,
            semantic_W=semantic_W,
            episodic_delta=torch.zeros_like(episodic_delta),
            routing_weights=routing,
            working_delta=working_flat,
            decays=self.nll_fn.decays,
        )

        # 2. The gate observes a pre-retrieval surprise.
        pre_action_loss = self.nll_fn.event_NLL(
            sequence={"times": times, "types": types},
            params=pre_action_theta,
            k=k,
        )
        surprise = pre_action_loss.detach()

        # 3. most responsible leaf
        hat_idx = int(torch.argmax(r_t).item())
        hat_leaf = leaf_ids[hat_idx]

        # 4. novelty and count from the memory bank of hat_leaf
        novelty, count, max_sim = self.leaf_novelty_count(
            q_t=q_t,
            leaf_id=hat_leaf,
        )

        # 5. Compute all four controller probabilities. Argmax is logging only.
        controller_output = self.action_distribution(
            surprise=pre_action_loss,
            novelty=novelty,
            count=count,
        )
        action_probabilities = controller_output["probabilities"]
        action = tuple(Action)[
            int(action_probabilities.detach().argmax().item())
        ]

        theta_eff = episodic_memory.parameter_update.compose_effective_parameters(
            semantic_mu=semantic_mu,
            semantic_W=semantic_W,
            episodic_delta=(
                episodic_delta * action_probabilities[1]
            ),
            routing_weights=routing,
            working_delta=working_flat,
            decays=self.nll_fn.decays,
        )
        loss_t = self.nll_fn.event_NLL(
            sequence={"times": times, "types": types},
            params=theta_eff,
            k=k,
        )

        # 6. p(A) is the effective fast-adaptation rate.
        wm_next = self.update_working_memory(
            loss=loss_t,
            delta_used=working_flat,
            adaptation_probability=action_probabilities[0],
        )
        episodic_memory.credit_retrieval(
            info_by_batch=[info_by_node],
            leaf_paths=[[hat_leaf]],
            routing_weights=routing.new_ones(1, 1),
            retrieval_probability=action_probabilities[1],
        )
        # Advance existing memories once per wake event. A memory written below
        # starts at age zero rather than being aged immediately.
        episodic_memory.step_age()

        # 7. write memory if needed
        written_item = None

        if bool(self.write_candidate_mask(
            action_probabilities.detach(), training=self.training
        ).cpu()):
            write_quality = action_probabilities[2]
            queue_weight = self.queue_weight(
                controller_output.get(
                    "raw_probabilities", action_probabilities
                )
            )
            written_item = self.write_residual_memory(
                q_t=q_t,
                theta_sem_leaf=sem_params[hat_leaf],
                times=times,
                types=types,
                k=k,
                node_id=hat_leaf,
                write_quality=write_quality,
                queue_weight=queue_weight,
            )

            self._require_episodic_memory().add_memory(
                node_id=hat_leaf,
                key=written_item.key,
                delta_theta=written_item.delta_theta,
                window=written_item.window,
                write_quality=written_item.write_quality,
                queue_weight=written_item.queue_weight,
            )
            self.split_queues[hat_leaf] += written_item.queue_weight

        return {
            "loss": loss_t,
            "surprise": surprise,
            "novelty": novelty.detach(),
            "count": count.detach(),
            "max_sim": max_sim.detach(),
            "action": action,
            "action_logits": controller_output["logits"].detach(),
            "action_probabilities": action_probabilities.detach(),
            "raw_action_probabilities": controller_output.get(
                "raw_probabilities", action_probabilities
            ).detach(),
            "hat_leaf": hat_leaf,
            "theta_eff": theta_eff,
            "wm_next": wm_next,
            "written_item": written_item,
        }
