import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from HawkesBackbone import HawkesFamily
from MemoryResiduals.MemoryBank import EventWindow, MemoryBank, MemoryItem
from MemoryResiduals.Replay import (
    ReplayBatchCache,
    batched_replay_log_likelihood,
    build_replay_batch_cache,
    replay_log_likelihood,
)
from Wake.HawkesParams import HawkesParams
from Wake.SequentialController import Controller


@dataclass
class SplitBatch:
    """
    Batch for one leaf's split optimization.

    residuals: [R, P]
    contexts:  [R, z_dim]
    weights:   [R]
    windows:   list of EventWindow, length R
    """
    residuals: torch.Tensor
    contexts: torch.Tensor
    weights: torch.Tensor
    windows: List[EventWindow]
    # ``weights`` are normalized to replay-count scale before Split uses
    # them.  Keep the unnormalized factors and their scale-free safeguards so
    # normalization cannot manufacture structural evidence from tiny
    # QUEUE_SPLIT probabilities or one dominant memory.
    base_weights: Optional[torch.Tensor] = None
    structural_weights: Optional[torch.Tensor] = None
    structural_strength: Optional[torch.Tensor] = None
    effective_sample_size: Optional[torch.Tensor] = None
    # Independent observation count represented by each physical row.
    sample_support: Optional[torch.Tensor] = None


class SplitCommitState:
    def __init__(self):
        self.consecutive_ready = 0


class DiagonalResidualDistance(nn.Module):
    """
    Learned diagonal Mahalanobis distance.

    d_phi(x, c) = sum_p softplus(beta_p) * (x_p - c_p)^2
    """

    def __init__(self, P: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(P))

    def forward(self, residuals: torch.Tensor, centers: torch.Tensor):
        """
        residuals: [R, P]
        centers:   [2, P]

        returns:
            dist: [R, 2]
        """
        scale = F.softplus(self.log_scale) + 1e-6     # [P]
        diff = residuals[:, None, :] - centers[None, :, :]
        dist = (scale[None, None, :] * diff.pow(2)).sum(dim=-1)
        return dist
    

class SplitModule(nn.Module):
    def __init__(
        self,
        P: int,
        z_dim: int,
        hidden_dim: int = 128,
        tau_c: float = 0.5,
        tau_g: float = 1.0,
        tau_route: float = 1.0,
        tau_m: float = 1.0,
        m_min: float = 10.0,
        lambda_tree: float = 1e-3,
        lambda_mass: float = 1.0,
        eps: float = 1e-8,
        controller: Optional[Controller] = None,
        nll_fn: Optional[HawkesFamily] = None,
    ):
        super().__init__()
        self.P = P
        self.z_dim = z_dim

        self.tau_c = tau_c
        self.tau_g = tau_g
        self.tau_route = tau_route
        self.tau_m = tau_m
        self.m_min = m_min
        # Legacy standalone fallback. Production Sleep passes the cycle's
        # frozen shared lambda_T explicitly so all topology proposals are
        # evaluated under the same complexity price.
        self.lambda_tree = lambda_tree
        self.lambda_mass = lambda_mass
        self.eps = eps

        # Dormant residual centers c_{ell,1}, c_{ell,2}
        self.centers = nn.Parameter(0.01 * torch.randn(2, P))

        # Split gate logit zeta_ell. Start negative so the split is initially closed.
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

        # Parent shared-residual absorption ratio alpha_ell in (0, 1).
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        # Child separation scale rho_ell^sp > 0.
        self.rho_raw = nn.Parameter(torch.tensor(0.0))

        # Learned residual distance d_phi.
        self.distance = DiagonalResidualDistance(P)

        # Context-conditioned candidate-child router pi_{ell,j}(z_r).
        self.router = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.controller = controller
        self.nll_fn = nll_fn if nll_fn is not None else (
            controller.nll_fn if controller is not None else None
        )
        self.commit_state = SplitCommitState()

    def get_extra_state(self):
        return {"consecutive_ready": self.commit_state.consecutive_ready}

    def set_extra_state(self, state) -> None:
        self.commit_state.consecutive_ready = int(
            state.get("consecutive_ready", 0)
        )

    def attach_controller(self, controller: Controller) -> None:
        self.controller = controller
        if self.nll_fn is None:
            self.nll_fn = controller.nll_fn

    def _require_controller(self) -> Controller:
        if self.controller is None:
            raise RuntimeError(
                "controller is not attached. Pass it to SplitModule(...) "
                "or call attach_controller(...) before residual writing."
            )
        return self.controller

    def make_residual_write(
        self,
        q_t: torch.Tensor,
        theta_sem_leaf: HawkesParams,
        window: EventWindow,
        k: int = 0,
    ) -> MemoryItem:
        """
        Build one written memory residual through SequentialController.

        Split receives EventWindow objects, while SequentialController owns the
        residual-write rule and projection policy.
        """
        controller = self._require_controller()
        return controller.write_residual_memory(
            q_t=q_t,
            theta_sem_leaf=theta_sem_leaf,
            times=window.times,
            types=window.types,
            k=k,
            node_id=window.node_id,
        )

    def compute_soft_residual_structure(
        self,
        residuals: torch.Tensor,
        weights: torch.Tensor,
        theta_sem: torch.Tensor,
        sample_support: Optional[torch.Tensor] = None,
    ):
        """
        Implements equations (20)-(23).

        residuals: [R, P]
        weights:   [R]
        theta_sem: [P]

        returns dictionary containing:
            q: [R, 2]
            N_mass: [2]
            N_eff: [2]
            pi_bar: [2]
            delta_bar: [2, P]
            delta_shared: [P]
            delta_child: [2, P]
            theta_plus: [P]
            theta_cand: [2, P]
        """
        eps = self.eps

        # Eq. (20): soft residual clustering.
        dist = self.distance(residuals, self.centers)          # [R, 2]
        logits = -dist / self.tau_c
        q = F.softmax(logits, dim=-1)                          # [R, 2]

        # Eq. (21): child mass is used for prototypes, while child-wise ESS
        # measures whether each candidate is supported by independent replay
        # evidence rather than one dominant weighted memory.
        support = (
            torch.ones_like(weights)
            if sample_support is None
            else sample_support.to(weights).clamp_min(0.0)
        )
        if support.shape != weights.shape:
            raise ValueError("sample_support and weights must be aligned vectors")
        weighted_q = weights[:, None] * q                      # [R, 2]
        N_mass = (support[:, None] * weighted_q).sum(dim=0)    # [2]
        N_eff = N_mass.square() / (
            (support[:, None] * weighted_q.square()).sum(dim=0) + eps
        )                                                      # [2]

        # Eq. (21): normalized candidate mass.
        pi_bar = N_mass / (N_mass.sum() + eps)                 # [2]

        # Eq. (21): candidate residual prototypes.
        aggregate_weighted_q = support[:, None] * weighted_q
        delta_bar = aggregate_weighted_q.T @ residuals         # [2, P]
        delta_bar = delta_bar / (N_mass[:, None] + eps)        # [2, P]

        # Eq. (22): parent-shared residual and child-specific deviations.
        delta_shared = (pi_bar[:, None] * delta_bar).sum(dim=0) # [P]
        delta_child = delta_bar - delta_shared[None, :]         # [2, P]

        # Eq. (23): parent absorbs shared residual.
        alpha = torch.sigmoid(self.alpha_logit)
        theta_plus = theta_sem + alpha * delta_shared          # [P]

        # Eq. (23): dormant candidates inherit only residual contrast.
        rho_sp = F.softplus(self.rho_raw) + eps
        theta_cand = theta_plus[None, :] + rho_sp * delta_child # [2, P]

        return {
            "dist": dist,
            "q": q,
            "N_mass": N_mass,
            "N_eff": N_eff,
            # Compatibility alias. New support decisions must use N_eff.
            "N": N_mass,
            "pi_bar": pi_bar,
            "delta_bar": delta_bar,
            "delta_shared": delta_shared,
            "delta_child": delta_child,
            "alpha": alpha,
            "rho_sp": rho_sp,
            "theta_plus": theta_plus,
            "theta_cand": theta_cand,
        }

    def compute_split_training_objective(
        self,
        *,
        weights: torch.Tensor,
        ell_split: torch.Tensor,
        g_split: torch.Tensor,
        N_eff: torch.Tensor,
        lambda_T: float | torch.Tensor,
        delta_complexity: float | torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Implement the prediction + complexity + child-support objective."""
        reference = ell_split
        lambda_tensor = torch.as_tensor(
            lambda_T,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(())
        delta_tensor = torch.as_tensor(
            delta_complexity,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(())
        if not bool(torch.isfinite(lambda_tensor)):
            raise FloatingPointError("lambda_T must be finite")
        if not bool(torch.isfinite(delta_tensor)):
            raise FloatingPointError("delta_complexity must be finite")
        if bool(lambda_tensor < 0.0):
            raise ValueError("lambda_T must be non-negative")
        if bool(delta_tensor <= 0.0):
            raise ValueError("delta_complexity must be positive")

        prediction_loss = -(
            weights * ell_split
        ).sum() / (weights.sum() + self.eps)
        expected_complexity_delta = g_split * delta_tensor
        complexity_penalty = lambda_tensor * expected_complexity_delta
        mass_penalty = self.lambda_mass * F.softplus(
            (self.m_min - N_eff) / self.tau_m
        ).sum()
        loss = prediction_loss + complexity_penalty + mass_penalty
        return {
            "loss": loss,
            "prediction_loss": prediction_loss,
            "complexity_penalty": complexity_penalty,
            "mass_penalty": mass_penalty,
            "lambda_T": lambda_tensor,
            "delta_tree_complexity": delta_tensor,
            "expected_delta_tree_complexity": (
                expected_complexity_delta
            ),
        }

    def compute_replay_loglikelihood(
        self,
        batch: SplitBatch,
        theta_plus: torch.Tensor,
        theta_cand: torch.Tensor,
        hawkes_ll: Optional[HawkesFamily] = None,
        replay_cache: Optional[ReplayBatchCache] = None,
    ):
        """
        Implements equations (24)-(26).

        theta_plus: [P]
        theta_cand: [2, P]

        returns:
            ell_split: [R]
            logp0: [R]
            logp_child_mix: [R]
            logp_child_each: [R, 2]
            route_prob: [R, 2]
            g_split: scalar
        """
        hawkes_ll = self._resolve_hawkes_ll(hawkes_ll)
        eps = self.eps

        # Eq. (24): differentiable split gate.
        g_split = torch.sigmoid(self.gate_logit / self.tau_g)

        # Eq. (25): context-conditioned child routing.
        route_logits = self.router(batch.contexts) / self.tau_route    # [R, 2]
        log_route_prob = F.log_softmax(route_logits, dim=-1)           # [R, 2]
        route_prob = log_route_prob.exp()

        if replay_cache is None:
            replay_cache = build_replay_batch_cache(
                batch.windows,
                hawkes_ll,
                theta_plus,
            )
        elif replay_cache.num_windows != len(batch.windows):
            raise ValueError("replay cache and batch windows must align")
        theta_models = torch.cat(
            [theta_plus.reshape(1, -1), theta_cand],
            dim=0,
        )                                                               # [3, P]
        window_nll = batched_replay_log_likelihood(
            replay_cache,
            theta_models,
            hawkes_ll,
            normalize_by_events=True,
        )                                                               # [R, 3]
        # The batched kernel returns negative log likelihoods; preserve the
        # original log-probability convention used by the relaxed objective.
        logp0 = -window_nll[:, 0]                                      # [R]
        logp_child_each = -window_nll[:, 1:3]                          # [R, 2]

        # Eq. (25): log child mixture likelihood.
        logp_child_mix = torch.logsumexp(
            log_route_prob + logp_child_each,
            dim=-1,
        )                                                             # [R]

        # Eq. (26): relaxed split likelihood.
        log_g = torch.log(g_split + eps)
        log_1_minus_g = torch.log(1.0 - g_split + eps)

        ell_split = torch.logaddexp(
            log_1_minus_g + logp0,
            log_g + logp_child_mix,
        )                                                             # [R]

        return {
            "g_split": g_split,
            "route_prob": route_prob,
            "logp0": logp0,
            "logp_child_each": logp_child_each,
            "logp_child_mix": logp_child_mix,
            "ell_split": ell_split,
        }

    def _resolve_hawkes_ll(
        self,
        hawkes_ll: Optional[HawkesFamily],
    ) -> HawkesFamily:
        if hawkes_ll is not None:
            return hawkes_ll
        if self.nll_fn is not None:
            return self.nll_fn
        if self.controller is not None:
            return self.controller.nll_fn
        raise RuntimeError(
            "hawkes_ll is not provided. Pass a HawkesFamily to forward(...), "
            "initialize SplitModule(nll_fn=...), or attach a controller."
        )

    def _window_log_prob(
        self,
        window: EventWindow,
        theta_flat: torch.Tensor,
        hawkes_ll: HawkesFamily,
    ) -> torch.Tensor:
        return replay_log_likelihood(
            window=window,
            theta=theta_flat,
            hawkes_ll=hawkes_ll,
            decays=hawkes_ll.decays,
            normalize_by_events=True,
        )

    def _theta_sem_to_flat(
        self,
        theta_sem: torch.Tensor | HawkesParams | Any,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(theta_sem, HawkesParams):
            theta_flat = torch.cat(
                [
                    theta_sem.mu_tilde.reshape(-1),
                    theta_sem.W_tilde.reshape(-1),
                ],
                dim=0,
            )
        elif torch.is_tensor(theta_sem):
            theta_flat = theta_sem
        elif hasattr(theta_sem, "raw_mu") and hasattr(theta_sem, "raw_W"):
            raw_mu = theta_sem.raw_mu() if callable(theta_sem.raw_mu) else theta_sem.raw_mu
            raw_W = theta_sem.raw_W() if callable(theta_sem.raw_W) else theta_sem.raw_W
            theta_flat = torch.cat([raw_mu.reshape(-1), raw_W.reshape(-1)], dim=0)
        else:
            raise TypeError(
                "theta_sem must be a flat tensor, HawkesParams, or an object "
                "with raw_mu/raw_W fields."
            )

        theta_flat = theta_flat.to(device=reference.device, dtype=reference.dtype)
        if theta_flat.ndim != 1:
            raise ValueError(f"theta_sem must be one-dimensional, got {tuple(theta_flat.shape)}")
        if theta_flat.numel() != self.P:
            raise ValueError(f"theta_sem must contain {self.P} values, got {theta_flat.numel()}")
        return theta_flat

    def _batch_to_device(
        self,
        batch: SplitBatch,
        reference: torch.Tensor,
    ) -> SplitBatch:
        return SplitBatch(
            residuals=batch.residuals.to(device=reference.device, dtype=reference.dtype),
            contexts=batch.contexts.to(device=reference.device, dtype=reference.dtype),
            weights=batch.weights.to(device=reference.device, dtype=reference.dtype),
            windows=batch.windows,
            base_weights=(
                None
                if batch.base_weights is None
                else batch.base_weights.to(
                    device=reference.device, dtype=reference.dtype
                )
            ),
            structural_weights=(
                None
                if batch.structural_weights is None
                else batch.structural_weights.to(
                    device=reference.device, dtype=reference.dtype
                )
            ),
            structural_strength=(
                None
                if batch.structural_strength is None
                else batch.structural_strength.to(
                    device=reference.device, dtype=reference.dtype
                )
            ),
            effective_sample_size=(
                None
                if batch.effective_sample_size is None
                else batch.effective_sample_size.to(
                    device=reference.device, dtype=reference.dtype
                )
            ),
            sample_support=(
                None
                if batch.sample_support is None
                else batch.sample_support.to(
                    device=reference.device, dtype=reference.dtype
                )
            ),
        )

    def _module_reference(self, fallback: torch.Tensor) -> torch.Tensor:
        return next(self.parameters(), fallback)

    def validate_batch(self, batch: SplitBatch) -> None:
        if batch.residuals.ndim != 2 or batch.residuals.shape[1] != self.P:
            raise ValueError(
                f"batch.residuals must have shape [R, {self.P}], "
                f"got {tuple(batch.residuals.shape)}"
            )
        if batch.contexts.ndim != 2 or batch.contexts.shape[1] != self.z_dim:
            raise ValueError(
                f"batch.contexts must have shape [R, {self.z_dim}], "
                f"got {tuple(batch.contexts.shape)}"
            )
        R = batch.residuals.shape[0]
        if batch.contexts.shape[0] != R:
            raise ValueError("batch.contexts and batch.residuals must have the same R")
        if batch.weights.shape != (R,):
            raise ValueError(f"batch.weights must have shape [{R}], got {tuple(batch.weights.shape)}")
        for name in ("base_weights", "structural_weights", "sample_support"):
            value = getattr(batch, name)
            if value is not None and value.shape != (R,):
                raise ValueError(
                    f"batch.{name} must have shape [{R}], "
                    f"got {tuple(value.shape)}"
                )
        for name in ("structural_strength", "effective_sample_size"):
            value = getattr(batch, name)
            if value is not None and value.numel() != 1:
                raise ValueError(f"batch.{name} must be scalar")
        if len(batch.windows) != R:
            raise ValueError(f"batch.windows must contain {R} windows, got {len(batch.windows)}")

    def forward(
        self,
        batch: SplitBatch,
        theta_sem: torch.Tensor | HawkesParams | Any,
        hawkes_ll: Optional[HawkesFamily] = None,
        *,
        lambda_T: Optional[float | torch.Tensor] = None,
        delta_complexity: float | torch.Tensor = 1.0,
        replay_cache: Optional[ReplayBatchCache] = None,
    ):
        """
        Full differentiable split loss for one leaf.
        """
        hawkes_ll = self._resolve_hawkes_ll(hawkes_ll)
        self.validate_batch(batch)

        reference = self._module_reference(batch.residuals)
        theta_sem = self._theta_sem_to_flat(theta_sem, reference)
        batch = self._batch_to_device(batch, theta_sem)
        residuals = batch.residuals
        weights = batch.weights
        sample_support = (
            torch.ones_like(weights)
            if batch.sample_support is None
            else batch.sample_support
        )

        struct = self.compute_soft_residual_structure(
            residuals=residuals,
            weights=weights,
            theta_sem=theta_sem,
            sample_support=sample_support,
        )

        replay = self.compute_replay_loglikelihood(
            batch=batch,
            theta_plus=struct["theta_plus"],
            theta_cand=struct["theta_cand"],
            hawkes_ll=hawkes_ll,
            replay_cache=replay_cache,
        )

        objective = self.compute_split_training_objective(
            weights=weights * sample_support,
            ell_split=replay["ell_split"],
            g_split=replay["g_split"],
            N_eff=struct["N_eff"],
            lambda_T=(
                self.lambda_tree if lambda_T is None else lambda_T
            ),
            delta_complexity=delta_complexity,
        )

        out = {}
        out.update(struct)
        out.update(replay)
        out.update(objective)
        out["weighted_nll"] = objective["prediction_loss"]
        out["tree_penalty"] = objective["complexity_penalty"]
        # Directly constructed legacy SplitBatch objects have no separate
        # structural factors.  Treat their supplied weights as trusted
        # evidence while production batches expose the real safeguards.
        out["structural_strength"] = (
            weights.new_ones(())
            if batch.structural_strength is None
            else batch.structural_strength.reshape(())
        )
        out["effective_sample_size"] = (
            weights.sum().square()
            / (weights.square().sum() + self.eps)
            if batch.effective_sample_size is None
            else batch.effective_sample_size.reshape(())
        )
        return out

    @staticmethod
    def compute_memory_weight(
        residual: torch.Tensor,
        age: float,
        usage_count: float,
        reliability: float = 1.0,
        tau_age: float = 20.0,
        residual_norm_ema: float = 1.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        recency = torch.exp(
            torch.tensor(
                -age / tau_age,
                device=residual.device,
                dtype=residual.dtype,
            )
        )
        usage = 1.0 + torch.log1p(
            torch.tensor(
                usage_count,
                device=residual.device,
                dtype=residual.dtype,
            )
        )
        strength = residual.norm() / (residual_norm_ema + eps)

        weight = recency * usage * strength * reliability
        return weight.detach()

    @staticmethod
    def normalize_split_evidence(
        base_weights: torch.Tensor,
        structural_weights: torch.Tensor,
        sample_support: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Put Split mass on replay-count scale without hiding weak evidence.

        ``normalized`` is used by the differentiable Split objective, while
        ``structural_strength`` and ``effective_sample_size`` remain on their
        original scale and are checked separately before a split can commit.
        """
        if base_weights.ndim != 1 or structural_weights.shape != base_weights.shape:
            raise ValueError(
                "base_weights and structural_weights must be aligned vectors"
            )
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        base = base_weights.clamp_min(0.0)
        structural = structural_weights.clamp(0.0, 1.0)
        support = (
            torch.ones_like(base)
            if sample_support is None
            else sample_support.to(base).clamp_min(0.0)
        )
        if support.shape != base.shape:
            raise ValueError("sample_support and base_weights must be aligned vectors")
        evidence = base * structural
        evidence_sum = (support * evidence).sum()
        support_sum = support.sum()
        normalized = (
            support_sum * evidence / (evidence_sum + eps)
        )
        structural_strength = evidence_sum / ((support * base).sum() + eps)
        effective_sample_size = evidence_sum.square() / (
            (support * evidence.square()).sum() + eps
        )
        return (
            normalized.detach(),
            structural_strength.detach(),
            effective_sample_size.detach(),
        )

    @staticmethod
    def combine_split_batches(
        batches: Sequence[Optional[SplitBatch]],
        *,
        max_items: Optional[int] = None,
    ) -> Optional[SplitBatch]:
        """Union disjoint episodic and Sleep-only structural evidence."""
        present = [batch for batch in batches if batch is not None]
        if not present:
            return None
        reference = present[0].residuals
        residuals = torch.cat([
            batch.residuals.to(reference) for batch in present
        ], dim=0)
        contexts = torch.cat([
            batch.contexts.to(
                device=reference.device,
                dtype=present[0].contexts.dtype,
            )
            for batch in present
        ], dim=0)
        base_weights = torch.cat([
            (
                batch.weights
                if batch.base_weights is None
                else batch.base_weights
            ).to(reference)
            for batch in present
        ])
        structural_weights = torch.cat([
            (
                torch.ones_like(batch.weights)
                if batch.structural_weights is None
                else batch.structural_weights
            ).to(reference)
            for batch in present
        ])
        sample_support = torch.cat([
            (
                torch.ones_like(batch.weights)
                if batch.sample_support is None
                else batch.sample_support
            ).to(reference)
            for batch in present
        ])
        windows = [
            window for batch in present for window in batch.windows
        ]
        if max_items is not None and residuals.size(0) > int(max_items):
            selected = torch.topk(
                base_weights.clamp_min(0.0)
                * structural_weights.clamp(0.0, 1.0)
                * sample_support.clamp_min(0.0),
                # Prefer rows representing more independent observations when
                # a bounded physical-row replay budget is applied.
                k=int(max_items),
            ).indices
            selected_cpu = selected.detach().cpu().tolist()
            residuals = residuals.index_select(0, selected)
            contexts = contexts.index_select(0, selected)
            base_weights = base_weights.index_select(0, selected)
            structural_weights = structural_weights.index_select(
                0, selected
            )
            sample_support = sample_support.index_select(0, selected)
            windows = [windows[index] for index in selected_cpu]
        weights, structural_strength, effective_sample_size = (
            SplitModule.normalize_split_evidence(
                base_weights,
                structural_weights,
                sample_support,
            )
        )
        return SplitBatch(
            residuals=residuals,
            contexts=contexts,
            weights=weights,
            windows=windows,
            base_weights=base_weights.detach(),
            structural_weights=structural_weights.detach(),
            structural_strength=structural_strength,
            effective_sample_size=effective_sample_size,
            sample_support=sample_support.detach(),
        )

    @staticmethod
    def build_split_batch_from_memory_items(
        items: Sequence[MemoryItem],
        tau_age: float = 20.0,
        residual_norm_ema: float = 1.0,
    ) -> Optional[SplitBatch]:
        if not all(isinstance(item, MemoryItem) for item in items):
            raise TypeError("all items must be MemoryItem instances")

        valid_items = [item for item in items if item.window is not None]
        if not valid_items:
            return None

        residuals = torch.stack(
            [
                float(item.write_quality) * item.delta_theta.reshape(-1)
                for item in valid_items
            ],
            dim=0,
        )
        contexts = torch.stack([item.key.reshape(-1) for item in valid_items], dim=0)
        base_weights = torch.stack(
            [
                SplitModule.compute_memory_weight(
                    residual=residuals[index],
                    age=float(item.age),
                    usage_count=float(item.usage),
                    reliability=float(getattr(item, "reliability", 1.0)),
                    tau_age=tau_age,
                    residual_norm_ema=residual_norm_ema,
                )
                for index, item in enumerate(valid_items)
            ],
            dim=0,
        )
        structural_weights = base_weights.new_tensor(
            [float(item.queue_weight) for item in valid_items]
        )
        sample_support = base_weights.new_tensor([
            float(getattr(item, "support", 1.0)) for item in valid_items
        ]).clamp_min(0.0)
        weights, structural_strength, effective_sample_size = (
            SplitModule.normalize_split_evidence(
                base_weights,
                structural_weights,
                sample_support,
            )
        )
        windows = [item.window for item in valid_items]

        return SplitBatch(
            residuals=residuals,
            contexts=contexts,
            weights=weights,
            windows=windows,
            base_weights=base_weights.detach(),
            structural_weights=structural_weights.detach(),
            structural_strength=structural_strength,
            effective_sample_size=effective_sample_size,
            sample_support=sample_support.detach(),
        )

    @staticmethod
    def build_split_batch_from_memory_bank(
        bank: MemoryBank,
        tau_age: float = 20.0,
        residual_norm_ema: float = 1.0,
        max_items: Optional[int] = None,
    ) -> Optional[SplitBatch]:
        """Build a bounded evidence batch from one leaf memory bank."""
        if max_items is not None and max_items <= 0:
            raise ValueError("max_items must be positive when provided")
        valid_indices = [
            index for index, window in enumerate(bank.windows) if window is not None
        ]
        if not valid_indices:
            return None
        bank._ensure_prototype_state()
        indices = torch.tensor(valid_indices, device=bank.device, dtype=torch.long)
        residuals = (
            bank.write_quality[indices].unsqueeze(-1)
            * bank.deltas[indices]
        )
        contexts = bank.keys[indices]
        recency = torch.exp(-bank.age[indices] / tau_age)
        usage = 1.0 + torch.log1p(bank.usage[indices])
        strength = residuals.norm(dim=-1) / (residual_norm_ema + 1e-8)
        base_weights = (recency * usage * strength).detach()
        structural_weights = bank.queue_weight[indices].detach()
        sample_support = bank.support[indices].detach().clamp_min(0.0)
        evidence_weights = base_weights * structural_weights * sample_support
        if max_items is not None and indices.numel() > max_items:
            selected = torch.topk(
                evidence_weights, k=int(max_items)
            ).indices
            indices = indices.index_select(0, selected)
            residuals = residuals.index_select(0, selected)
            contexts = contexts.index_select(0, selected)
            base_weights = base_weights.index_select(0, selected)
            structural_weights = structural_weights.index_select(0, selected)
            sample_support = sample_support.index_select(0, selected)
        weights, structural_strength, effective_sample_size = (
            SplitModule.normalize_split_evidence(
                base_weights,
                structural_weights,
                sample_support,
            )
        )
        selected_indices = indices.detach().cpu().tolist()
        return SplitBatch(
            residuals=residuals,
            contexts=contexts,
            weights=weights,
            windows=[bank.windows[index] for index in selected_indices],
            base_weights=base_weights,
            structural_weights=structural_weights,
            structural_strength=structural_strength,
            effective_sample_size=effective_sample_size,
            sample_support=sample_support,
        )

    @staticmethod
    def coerce_split_batch(
        batch: SplitBatch | MemoryBank | Sequence[MemoryItem],
        tau_age: float = 20.0,
        residual_norm_ema: float = 1.0,
    ) -> Optional[SplitBatch]:
        if isinstance(batch, SplitBatch):
            return batch
        if isinstance(batch, MemoryBank):
            return SplitModule.build_split_batch_from_memory_bank(
                batch,
                tau_age=tau_age,
                residual_norm_ema=residual_norm_ema,
            )
        if isinstance(batch, Sequence):
            return SplitModule.build_split_batch_from_memory_items(
                batch,
                tau_age=tau_age,
                residual_norm_ema=residual_norm_ema,
            )
        raise TypeError(
            "leaf_to_queue_batch values must be SplitBatch instances or sequences "
            "of MemoryItem objects."
        )

    @staticmethod
    def detach_split_output(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, dict):
            return {
                key: SplitModule.detach_split_output(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [SplitModule.detach_split_output(item) for item in value]
        if isinstance(value, tuple):
            return tuple(SplitModule.detach_split_output(item) for item in value)
        return value

    @staticmethod
    def evaluate_split_eligibility(
        out: Mapping[str, Any],
        commit_state: SplitCommitState,
        m_min: float = 10.0,
        min_structural_strength: float = 0.0,
        min_effective_sample_size: float = 0.0,
    ) -> bool:
        """Check statistical support without deciding action preference."""
        N_mass = torch.as_tensor(out["N"]).detach()
        N_eff = torch.as_tensor(
            out.get("N_eff", N_mass)
        ).detach()
        structural_strength = float(
            torch.as_tensor(
                out.get("structural_strength", 1.0)
            ).detach().item()
        )
        effective_sample_size = float(
            torch.as_tensor(
                out.get("effective_sample_size", N_mass.sum())
            ).detach().item()
        )

        support_ok = bool((N_eff >= m_min).all().item())
        structural_ok = structural_strength > min_structural_strength
        effective_sample_ok = (
            effective_sample_size >= min_effective_sample_size
        )

        eligible = support_ok and structural_ok and effective_sample_ok
        commit_state.consecutive_ready = int(eligible)
        return eligible

    @staticmethod
    def attach_split_eligibility_output(
        out: Dict[str, Any],
        eligible: bool,
        commit_state: SplitCommitState,
    ) -> Dict[str, Any]:
        out["eligible"] = eligible
        out["eligibility_observations"] = commit_state.consecutive_ready

        if eligible:
            # delta_child is the two child-specific residual parameters
            # delta_{ell,j}; theta_cand is also exposed for tree write-back.
            out["child_delta"] = out["delta_child"]
            out["child_theta"] = out["theta_cand"]
        else:
            out["child_delta"] = None
            out["child_theta"] = None

        return out

    def optimize_leaf_split(
        self,
        theta_sem: torch.Tensor | HawkesParams | Any,
        batch: SplitBatch,
        hawkes_ll: Optional[HawkesFamily] = None,
        num_steps: int = 200,
        lr: float = 1e-3,
        commit_state: Optional[SplitCommitState] = None,
        m_min: Optional[float] = None,
        min_structural_strength: float = 0.0,
        min_effective_sample_size: float = 0.0,
        lambda_T: Optional[float | torch.Tensor] = None,
        delta_complexity: float | torch.Tensor = 1.0,
    ) -> Dict[str, Any]:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")

        hawkes_ll = self._resolve_hawkes_ll(hawkes_ll)
        self.validate_batch(batch)
        reference = self._module_reference(batch.residuals)
        theta_sem = self._theta_sem_to_flat(theta_sem, reference).detach()

        was_training = self.training
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        replay_cache = build_replay_batch_cache(
            batch.windows,
            hawkes_ll,
            theta_sem,
        )

        last_loss = None
        try:
            for _ in range(num_steps):
                optimizer.zero_grad(set_to_none=True)
                out = self(
                    batch=batch,
                    theta_sem=theta_sem,
                    hawkes_ll=hawkes_ll,
                    lambda_T=lambda_T,
                    delta_complexity=delta_complexity,
                    replay_cache=replay_cache,
                )
                loss = out["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "split optimization produced a non-finite loss"
                    )
                loss.backward()
                optimizer.step()
                last_loss = loss.detach()

            with torch.no_grad():
                final_out = self(
                    batch=batch,
                    theta_sem=theta_sem,
                    hawkes_ll=hawkes_ll,
                    lambda_T=lambda_T,
                    delta_complexity=delta_complexity,
                    replay_cache=replay_cache,
                )
        finally:
            if not was_training:
                self.eval()

        final_out = self.detach_split_output(final_out)
        commit_state = self.commit_state if commit_state is None else commit_state
        eligible = self.evaluate_split_eligibility(
            out=final_out,
            commit_state=commit_state,
            m_min=self.m_min if m_min is None else m_min,
            min_structural_strength=min_structural_strength,
            min_effective_sample_size=min_effective_sample_size,
        )
        final_out = self.attach_split_eligibility_output(
            out=final_out,
            eligible=eligible,
            commit_state=commit_state,
        )
        final_out["num_steps"] = num_steps
        final_out["lr"] = lr
        final_out["last_train_loss"] = last_loss
        return final_out

    @staticmethod
    def sleep_phase_split_update(
        leaves: Iterable[Any],
        leaf_to_split_module: Mapping[Any, "SplitModule"],
        leaf_to_theta_sem: Mapping[Any, torch.Tensor | HawkesParams | Any],
        leaf_to_queue_batch: Mapping[
            Any, SplitBatch | MemoryBank | Sequence[MemoryItem]
        ],
        hawkes_ll: Optional[HawkesFamily] = None,
        num_steps: int = 200,
        lr: float = 1e-3,
        leaf_to_commit_state: Optional[Mapping[Any, SplitCommitState]] = None,
        m_min: float = 10.0,
        lambda_T: Optional[float | torch.Tensor] = None,
        delta_complexity: float | torch.Tensor = 1.0,
        tau_age: float = 20.0,
        residual_norm_ema: float = 1.0,
    ) -> Dict[Any, Dict[str, Any]]:
        split_outputs = {}

        for leaf_id in leaves:
            if leaf_id not in leaf_to_queue_batch:
                continue

            batch = SplitModule.coerce_split_batch(
                leaf_to_queue_batch[leaf_id],
                tau_age=tau_age,
                residual_norm_ema=residual_norm_ema,
            )
            if batch is None or batch.residuals.shape[0] < 2:
                continue

            if leaf_id not in leaf_to_split_module:
                raise KeyError(f"missing SplitModule for leaf {leaf_id!r}")
            if leaf_id not in leaf_to_theta_sem:
                raise KeyError(
                    f"missing semantic Hawkes parameters for leaf {leaf_id!r}"
                )

            split_module = leaf_to_split_module[leaf_id]
            theta_sem = leaf_to_theta_sem[leaf_id]
            commit_state = (
                leaf_to_commit_state[leaf_id]
                if leaf_to_commit_state is not None and leaf_id in leaf_to_commit_state
                else None
            )

            split_outputs[leaf_id] = split_module.optimize_leaf_split(
                theta_sem=theta_sem,
                batch=batch,
                hawkes_ll=hawkes_ll,
                num_steps=num_steps,
                lr=lr,
                commit_state=commit_state,
                m_min=m_min,
                lambda_T=lambda_T,
                delta_complexity=delta_complexity,
            )

        return split_outputs


@torch.no_grad()
def commit_split(
    tree,
    leaf_id: str,
    split_module: SplitModule,
    split_output: Mapping[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    memory_hard_threshold: float = 0.0,
    init_steps: int = 30,
    init_lr: float = 1e-2,
    authorized: bool = False,
) -> tuple[str, str]:
    """Commit an optimized split without discarding semantic or replay state.

    The old leaf bank initially remains attached to the newly internal parent.
    A replay item moves to the child with the largest routing posterior
    ``log pi_j(z_r) + log p(w_r | theta_j)`` unless the parent likelihood is
    greater than that child's likelihood by ``memory_hard_threshold``.
    Consequently, the parent retains only replay evidence that it explains
    uniquely well. Items without a replay window cannot be scored and remain
    at the parent conservatively.
    """
    if leaf_id not in tree.nodes or not tree.nodes[leaf_id].is_leaf:
        raise KeyError(f"Unknown leaf: {leaf_id}")
    if not authorized:
        raise PermissionError(
            "Split commit requires unified-selector authorization"
        )
    theta_plus = split_output.get("theta_plus")
    theta_cand = split_output.get("theta_cand")
    if not torch.is_tensor(theta_plus) or not torch.is_tensor(theta_cand):
        raise ValueError("split_output must contain theta_plus and theta_cand")
    if theta_plus.shape != (tree.param_dim,) or theta_cand.shape != (2, tree.param_dim):
        raise ValueError("split semantic targets do not match the tree parameter size")
    if not torch.isfinite(torch.tensor(memory_hard_threshold)):
        raise ValueError("memory_hard_threshold must be finite")
    if memory_hard_threshold < 0.0:
        raise ValueError("memory_hard_threshold must be non-negative")

    old_theta = tree.semantic_theta(leaf_id).detach()
    source_bank = tree.episodic_memory.get_bank(leaf_id)
    valid_memory_indices = [
        index for index, window in enumerate(source_bank.windows) if window is not None
    ]
    parent_memory_mask = torch.ones(
        len(source_bank), dtype=torch.bool, device=source_bank.device
    )
    child_assignments = torch.full(
        (len(source_bank),), -1, dtype=torch.long, device=source_bank.device
    )

    if valid_memory_indices:
        logp_parent = split_output.get("logp0")
        logp_children = split_output.get("logp_child_each")
        route_prob = split_output.get("route_prob")
        valid_count = len(valid_memory_indices)
        if not torch.is_tensor(logp_parent) or not torch.is_tensor(logp_children):
            raise ValueError(
                "split_output must contain logp0 and logp_child_each to "
                "partition replay memories"
            )
        if logp_parent.shape != (valid_count,) or logp_children.shape != (
            valid_count,
            2,
        ):
            raise ValueError(
                "split replay scores must align with the source bank's valid windows"
            )

        logp_parent = logp_parent.to(source_bank.device)
        logp_children = logp_children.to(source_bank.device)
        if route_prob is None:
            # Backward-compatible payloads predate stored routing scores and
            # therefore imply a uniform prior over the two children.
            route_prob = torch.full_like(logp_children, 0.5)
        if not torch.is_tensor(route_prob) or route_prob.shape != (
            valid_count,
            2,
        ):
            raise ValueError(
                "split_output route_prob must align with replay and children"
            )
        route_prob = route_prob.to(
            device=source_bank.device,
            dtype=logp_children.dtype,
        )
        posterior_logit = (
            route_prob.clamp_min(torch.finfo(route_prob.dtype).tiny).log()
            + logp_children
        )
        best_child = posterior_logit.argmax(dim=-1)
        best_child_logp = logp_children.gather(
            1, best_child[:, None]
        ).squeeze(1)
        finite_scores = torch.isfinite(logp_parent) & torch.isfinite(
            logp_children
        ).all(dim=-1) & torch.isfinite(route_prob).all(dim=-1)
        parent_is_uniquely_better = (
            logp_parent - best_child_logp > memory_hard_threshold
        )
        move_to_child = finite_scores & ~parent_is_uniquely_better
        valid_indices = torch.tensor(
            valid_memory_indices, dtype=torch.long, device=source_bank.device
        )
        moving_indices = valid_indices[move_to_child]
        parent_memory_mask[moving_indices] = False
        child_assignments[moving_indices] = best_child[move_to_child]

    parameter_ids_before = {id(parameter) for parameter in tree.parameters()}

    tree.split_leaf(leaf_id, refresh=True, optimizer=None)
    parent_node = tree.nodes[leaf_id]
    if parent_node.left is None or parent_node.right is None:
        raise RuntimeError("split did not create two children")
    left_id, right_id = parent_node.left, parent_node.right

    tree.set_semantic_theta(leaf_id, theta_plus)
    init_metrics = tree.fit_new_node_semantics(
        (left_id, right_id),
        theta_cand,
        num_steps=init_steps,
        learning_rate=init_lr,
    )
    tree.set_semantic_theta(left_id, theta_cand[0])
    tree.set_semantic_theta(right_id, theta_cand[1])
    if isinstance(split_output, dict):
        split_output["init_loss_before"] = init_metrics["loss_before"]
        split_output["init_loss_after"] = init_metrics["loss_after"]

    # Child-explainable memories move to the best child. Parent-only memories
    # stay in the inherited source bank. Rebase both groups because the split
    # writes new semantic parameters at every affected node.
    if len(source_bank) > 0:
        from Sleep.Merge import rebase_memory_to_new_leaf

        for child_index, child_id in enumerate((left_id, right_id)):
            indices = torch.nonzero(
                child_assignments == child_index,
                as_tuple=False,
            ).flatten()
            if indices.numel() == 0:
                continue
            child_theta = tree.semantic_theta(child_id).detach()
            rebased = rebase_memory_to_new_leaf(
                delta_theta=source_bank.deltas[indices],
                old_theta=old_theta,
                new_theta=child_theta,
            )
            tree.episodic_memory.get_bank(child_id).append_from(
                source=source_bank,
                indices=indices,
                deltas=rebased,
                node_id=child_id,
            )

        parent_indices = torch.nonzero(
            parent_memory_mask, as_tuple=False
        ).flatten()
        parent_rebased = rebase_memory_to_new_leaf(
            delta_theta=source_bank.deltas[parent_indices],
            old_theta=old_theta,
            new_theta=tree.semantic_theta(leaf_id).detach(),
        )
        source_bank.keep(parent_indices)
        source_bank.deltas = parent_rebased.to(source_bank.device)

    parent_node.split_queue.clear()
    old_mass = tree.mass_ema.pop(leaf_id, None)
    tree.low_mass_streak.pop(leaf_id, None)
    if old_mass is not None:
        pi_bar = split_output.get("pi_bar")
        if torch.is_tensor(pi_bar) and pi_bar.shape == (2,):
            proportions = pi_bar.detach().float().clamp_min(0.0)
            proportions = proportions / proportions.sum().clamp_min(1e-8)
            left_fraction, right_fraction = proportions.tolist()
        else:
            left_fraction, right_fraction = 0.5, 0.5
        tree.mass_ema[left_id] = old_mass * left_fraction
        tree.mass_ema[right_id] = old_mass * right_fraction
    tree.low_mass_streak[left_id] = 0
    tree.low_mass_streak[right_id] = 0
    split_module.commit_state.consecutive_ready = 0

    if optimizer is not None:
        new_parameters = [
            parameter
            for parameter in tree.parameters()
            if id(parameter) not in parameter_ids_before
        ]
        tree._add_parameters_to_optimizer(optimizer, new_parameters)
    return left_id, right_id
