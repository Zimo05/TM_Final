import math

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

    Bank-backed batches additionally carry row-wise ``mode_ids`` and
    effective-law ``law_keys``.  ``bank_group_weights`` is the resulting
    binary proposal consumed by Split's unchanged likelihood path.
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
    # Persistent observation support represented by each physical Bank row.
    # This is evidence provenance, not a replay sample multiplier.
    sample_support: Optional[torch.Tensor] = None
    # Number of independently replayable windows represented by each row.
    # A compressed prototype with support=100 and one retained window has
    # replay_support=1 and therefore contributes at most one unit of ESS.
    replay_support: Optional[torch.Tensor] = None
    # Bank provenance for mode-conditioned proposal formation.  These fields
    # are optional so legacy, directly-constructed batches keep the original
    # residual-clustering behavior.
    mode_ids: Optional[torch.Tensor] = None
    law_keys: Optional[torch.Tensor] = None
    # Row-wise binary proposal q_bank generated from Bank mode summaries.
    bank_group_weights: Optional[torch.Tensor] = None
    # Full-bank mode summaries are retained for diagnostics and provenance;
    # Split's likelihood/counterfactual path only consumes bank_group_weights.
    mode_summary: Optional[Dict[str, torch.Tensor]] = None


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
        m_min: float = 0.0,
        lambda_tree: float = 1e-3,
        lambda_mass: float = 0.0,
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
        replay_support: Optional[torch.Tensor] = None,
        bank_group_weights: Optional[torch.Tensor] = None,
    ):
        """
        Implements equations (20)-(23).

        residuals: [R, P]
        weights:   [R]
        theta_sem: [P]
        bank_group_weights: optional [R, 2] proposal from Bank modes

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

        # Eq. (20): proposal formation.  Bank-backed batches already carry a
        # binary proposal derived from persistent dynamics modes.  In that
        # path, do not cluster q_write * delta_theta (or delta_theta) again:
        # the Bank's effective Hawkes law identity is the source of truth.
        if bank_group_weights is None:
            dist = self.distance(residuals, self.centers)      # [R, 2]
            logits = -dist / self.tau_c
            q = F.softmax(logits, dim=-1)                      # [R, 2]
        else:
            if bank_group_weights.shape != (residuals.size(0), 2):
                raise ValueError(
                    "bank_group_weights must have shape [R, 2]"
                )
            if not bool(torch.isfinite(bank_group_weights).all()):
                raise FloatingPointError(
                    "bank_group_weights must contain only finite values"
                )
            if bool((bank_group_weights < 0.0).any()):
                raise ValueError("bank_group_weights must be non-negative")
            q = bank_group_weights.to(
                device=residuals.device,
                dtype=residuals.dtype,
            )
            row_mass = q.sum(dim=-1, keepdim=True)
            if bool((row_mass <= eps).any()):
                raise ValueError(
                    "each bank_group_weights row must have positive mass"
                )
            q = q / row_mass
            # Preserve the diagnostic shape expected by callers while making
            # it explicit that no residual-distance computation was used.
            dist = residuals.new_zeros(residuals.size(0), 2)

        # Eq. (21): child mass is used for prototypes, while child-wise ESS
        # measures whether each candidate is supported by independent replay
        # evidence rather than one dominant weighted memory.
        support = (
            torch.ones_like(weights)
            if replay_support is None
            else replay_support.to(weights).clamp_min(0.0)
        )
        if support.shape != weights.shape:
            raise ValueError("replay_support and weights must be aligned vectors")
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
            replay_support=(
                None
                if batch.replay_support is None
                else batch.replay_support.to(
                    device=reference.device, dtype=reference.dtype
                )
            ),
            mode_ids=(
                None
                if batch.mode_ids is None
                else batch.mode_ids.to(device=reference.device, dtype=torch.long)
            ),
            law_keys=(
                None
                if batch.law_keys is None
                else batch.law_keys.to(
                    device=reference.device, dtype=reference.dtype
                )
            ),
            bank_group_weights=(
                None
                if batch.bank_group_weights is None
                else batch.bank_group_weights.to(
                    device=reference.device, dtype=reference.dtype
                )
            ),
            mode_summary=(
                None
                if batch.mode_summary is None
                else {
                    name: value.to(
                        device=reference.device,
                        dtype=(
                            torch.long
                            if name in {"mode_ids", "group_ids"}
                            else reference.dtype
                        ),
                    )
                    for name, value in batch.mode_summary.items()
                }
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
        for name in (
            "base_weights",
            "structural_weights",
            "sample_support",
            "replay_support",
        ):
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
        if batch.mode_ids is not None:
            if batch.mode_ids.ndim != 1 or batch.mode_ids.shape != (R,):
                raise ValueError(
                    f"batch.mode_ids must have shape [{R}], "
                    f"got {tuple(batch.mode_ids.shape)}"
                )
            if batch.mode_ids.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise ValueError("batch.mode_ids must be an integer tensor")
        if batch.law_keys is not None:
            if batch.law_keys.ndim != 2 or batch.law_keys.size(0) != R:
                raise ValueError(
                    "batch.law_keys must have shape [R, law_dim]"
                )
        if batch.bank_group_weights is not None:
            if batch.bank_group_weights.shape != (R, 2):
                raise ValueError(
                    f"batch.bank_group_weights must have shape [{R}, 2], "
                    f"got {tuple(batch.bank_group_weights.shape)}"
                )
        if batch.mode_summary is not None:
            if not isinstance(batch.mode_summary, Mapping):
                raise ValueError("batch.mode_summary must be a mapping")
            required = {
                "mode_ids",
                "law_keys",
                "support",
                "split_mass",
                "delta_means",
                "mode_weights",
                "group_ids",
            }
            missing = required.difference(batch.mode_summary)
            if missing:
                raise ValueError(
                    "batch.mode_summary is missing "
                    + ", ".join(sorted(missing))
                )

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
        replay_support = (
            torch.ones_like(weights)
            if batch.replay_support is None
            else batch.replay_support
        )

        struct = self.compute_soft_residual_structure(
            residuals=residuals,
            weights=weights,
            theta_sem=theta_sem,
            sample_support=sample_support,
            replay_support=replay_support,
            bank_group_weights=batch.bank_group_weights,
        )

        replay = self.compute_replay_loglikelihood(
            batch=batch,
            theta_plus=struct["theta_plus"],
            theta_cand=struct["theta_cand"],
            hawkes_ll=hawkes_ll,
            replay_cache=replay_cache,
        )

        objective = self.compute_split_training_objective(
            weights=weights * replay_support,
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
        replay_support: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Put Split mass on replay-count scale without hiding weak evidence.

        Persistent support scales evidence confidence, but replay ESS is
        computed only from physically retained windows.  This prevents a
        compressed prototype from impersonating many independent replays.
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
        replay = (
            torch.ones_like(base)
            if replay_support is None
            else replay_support.to(base).clamp_min(0.0)
        )
        if replay.shape != base.shape:
            raise ValueError("replay_support and base_weights must be aligned vectors")
        evidence = support * base * structural
        evidence_sum = evidence.sum()
        replay_count = replay.sum()
        normalized = (
            replay_count * evidence / (evidence_sum + eps)
        )
        structural_strength = evidence_sum / ((support * base).sum() + eps)
        replay_evidence = replay * evidence
        effective_sample_size = replay_evidence.sum().square() / (
            replay_evidence.square().sum() + eps
        )
        return (
            normalized.detach(),
            structural_strength.detach(),
            effective_sample_size.detach(),
        )

    @staticmethod
    @torch.no_grad()
    def _build_bank_mode_groups(
        mode_ids: torch.Tensor,
        law_keys: torch.Tensor,
        deltas: torch.Tensor,
        support: torch.Tensor,
        split_mass: torch.Tensor,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Turn persistent Bank modes into one binary Split proposal.

        The reduction is intentionally performed before replay-row truncation,
        so compressed rows still contribute their historical support and
        structural mass.  Only the final row-wise ``q_bank`` is passed to the
        existing Split objective.
        """
        if mode_ids.ndim != 1:
            raise ValueError("mode_ids must have shape [R]")
        if law_keys.ndim != 2 or law_keys.size(0) != mode_ids.numel():
            raise ValueError("law_keys must have shape [R, law_dim]")
        if deltas.ndim != 2 or deltas.size(0) != mode_ids.numel():
            raise ValueError("deltas must have shape [R, P]")
        if support.shape != mode_ids.shape or split_mass.shape != mode_ids.shape:
            raise ValueError(
                "support and split_mass must have shape [R]"
            )
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        law_keys = F.normalize(law_keys, dim=-1)
        mode_ids = mode_ids.to(device=law_keys.device, dtype=torch.long)
        support = support.to(device=law_keys.device, dtype=law_keys.dtype).clamp_min(0.0)
        split_mass = split_mass.to(
            device=law_keys.device, dtype=law_keys.dtype
        ).clamp_min(0.0)
        deltas = deltas.to(device=law_keys.device)

        unique_modes, inverse = torch.unique(
            mode_ids,
            sorted=True,
            return_inverse=True,
        )
        mode_count = int(unique_modes.numel())
        mode_support = law_keys.new_zeros(mode_count)
        mode_support.index_add_(0, inverse, support)
        mode_split_mass = law_keys.new_zeros(mode_count)
        mode_split_mass.index_add_(0, inverse, split_mass)

        mode_law_keys = law_keys.new_zeros(
            mode_count, law_keys.size(-1)
        )
        mode_law_keys.index_add_(
            0,
            inverse,
            support[:, None] * law_keys,
        )
        mode_law_keys = F.normalize(
            mode_law_keys / (mode_support[:, None] + eps),
            dim=-1,
        )

        mode_delta_means = deltas.new_zeros(mode_count, deltas.size(-1))
        mode_delta_means.index_add_(
            0,
            inverse,
            support.to(deltas)[:, None] * deltas,
        )
        mode_delta_means = mode_delta_means / (
            mode_support.to(deltas)[:, None] + eps
        )

        # This is the structural mode weight from the proposal:
        # omega_m = S_m * E_m^struct / (S_m + eps).
        mode_weights = mode_support * mode_split_mass / (
            mode_support + eps
        )

        if mode_count == 1:
            # A single Bank mode has no preferred binary side.  Keep the
            # batch usable for the existing relaxed objective while making the
            # proposal independent of residual geometry.
            mode_group_weights = law_keys.new_full((1, 2), 0.5)
            mode_group_ids = torch.zeros(
                1, device=law_keys.device, dtype=torch.long
            )
        elif mode_count == 2:
            mode_group_weights = torch.eye(
                2, device=law_keys.device, dtype=law_keys.dtype
            )
            mode_group_ids = torch.arange(
                2, device=law_keys.device, dtype=torch.long
            )
        else:
            # Weighted two-center clustering runs only on mode-level law
            # identities z_m^law.  Row residuals never enter this proposal.
            points = F.normalize(mode_law_keys, dim=-1)
            pairwise_distance = 1.0 - points @ points.transpose(0, 1)
            pairwise_distance = pairwise_distance.masked_fill(
                torch.eye(
                    mode_count,
                    device=points.device,
                    dtype=torch.bool,
                ),
                -torch.inf,
            )
            farthest_pair = int(torch.argmax(pairwise_distance).item())
            first = farthest_pair // mode_count
            second = farthest_pair % mode_count
            centers = points[[first, second]].clone()
            cluster_weights = mode_weights.clamp_min(eps)
            assignments = torch.zeros(
                mode_count,
                device=points.device,
                dtype=torch.long,
            )

            for _ in range(12):
                distances = 1.0 - points @ centers.transpose(0, 1)
                assignments = distances.argmin(dim=-1)
                present = [
                    bool((assignments == child).any()) for child in range(2)
                ]
                if not all(present):
                    occupied = 1 if present[1] else 0
                    missing = 1 - occupied
                    candidate = torch.argmax(
                        distances[:, occupied]
                    )
                    assignments[candidate] = missing

                next_centers = []
                for child in range(2):
                    child_weight = cluster_weights * (
                        assignments == child
                    ).to(cluster_weights.dtype)
                    child_mass = child_weight.sum()
                    if bool(child_mass <= eps):
                        next_centers.append(centers[child])
                    else:
                        next_centers.append(
                            F.normalize(
                                (child_weight[:, None] * points).sum(0)
                                / child_mass,
                                dim=-1,
                            )
                        )
                centers = torch.stack(next_centers, dim=0)

            mode_group_weights = F.one_hot(
                assignments,
                num_classes=2,
            ).to(dtype=law_keys.dtype)
            mode_group_ids = assignments

        row_group_weights = mode_group_weights.index_select(0, inverse)
        summary = {
            "mode_ids": unique_modes.detach(),
            "law_keys": mode_law_keys.detach(),
            "support": mode_support.detach(),
            "split_mass": mode_split_mass.detach(),
            "delta_means": mode_delta_means.detach(),
            "mode_weights": mode_weights.detach(),
            "group_ids": mode_group_ids.detach(),
        }
        return row_group_weights.detach(), summary

    @staticmethod
    def _mode_stratified_topk(
        scores: torch.Tensor,
        bank_group_weights: Optional[torch.Tensor],
        max_items: int,
        min_per_group: int = 2,
    ) -> torch.Tensor:
        """Reserve physical replay rows on both candidate sides, then Top-K.

        Persistent support affects ``scores`` but never creates extra replay
        rows.  The reservation is best-effort when a side has fewer retained
        windows or the total budget is smaller than the requested reserve.
        """
        if scores.ndim != 1:
            raise ValueError("scores must be one-dimensional")
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        if min_per_group < 0:
            raise ValueError("min_per_group must be non-negative")
        count = int(scores.numel())
        if count <= max_items:
            return torch.arange(count, device=scores.device)
        if bank_group_weights is None:
            return torch.topk(scores, k=max_items).indices
        if bank_group_weights.shape != (count, 2):
            raise ValueError("bank_group_weights must have shape [R, 2]")

        assignments = bank_group_weights.argmax(dim=-1)
        ranked_by_group = []
        for group in range(2):
            group_indices = torch.nonzero(
                assignments == group, as_tuple=False
            ).flatten()
            if group_indices.numel() == 0:
                ranked_by_group.append(group_indices)
                continue
            order = torch.argsort(
                scores.index_select(0, group_indices), descending=True
            )
            ranked_by_group.append(group_indices.index_select(0, order))

        reserved: list[int] = []
        # Round-robin reservation prevents the first side from consuming a
        # small budget before the second side receives one physical replay.
        for rank in range(min_per_group):
            for group in range(2):
                if len(reserved) >= max_items:
                    break
                ranked = ranked_by_group[group]
                if rank < int(ranked.numel()):
                    reserved.append(int(ranked[rank].item()))
            if len(reserved) >= max_items:
                break

        selected_mask = torch.zeros(
            count, device=scores.device, dtype=torch.bool
        )
        if reserved:
            reserved_tensor = torch.tensor(
                reserved, device=scores.device, dtype=torch.long
            )
            selected_mask[reserved_tensor] = True
        remaining = max_items - len(reserved)
        if remaining > 0:
            available = torch.nonzero(
                ~selected_mask, as_tuple=False
            ).flatten()
            fill_order = torch.topk(
                scores.index_select(0, available), k=remaining
            ).indices
            fill = available.index_select(0, fill_order)
            selected_mask[fill] = True
        return torch.nonzero(selected_mask, as_tuple=False).flatten()

    @staticmethod
    def combine_split_batches(
        batches: Sequence[Optional[SplitBatch]],
        *,
        max_items: Optional[int] = None,
        min_replay_per_group: int = 2,
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
        replay_support = torch.cat([
            (
                torch.ones_like(batch.weights)
                if batch.replay_support is None
                else batch.replay_support
            ).to(reference)
            for batch in present
        ])
        mode_ids = None
        if all(batch.mode_ids is not None for batch in present):
            mode_ids = torch.cat([
                batch.mode_ids.to(device=reference.device, dtype=torch.long)
                for batch in present
            ])
        law_keys = None
        if all(batch.law_keys is not None for batch in present):
            law_keys = torch.cat([
                batch.law_keys.to(device=reference.device)
                for batch in present
            ], dim=0)
        bank_group_weights = None
        if all(batch.bank_group_weights is not None for batch in present):
            bank_group_weights = torch.cat([
                batch.bank_group_weights.to(reference)
                for batch in present
            ], dim=0)
        mode_summary = (
            present[0].mode_summary
            if len(present) == 1
            else None
        )
        windows = [
            window for batch in present for window in batch.windows
        ]
        if max_items is not None and residuals.size(0) > int(max_items):
            selected = SplitModule._mode_stratified_topk(
                base_weights.clamp_min(0.0)
                * structural_weights.clamp(0.0, 1.0)
                * sample_support.clamp_min(0.0),
                bank_group_weights,
                int(max_items),
                min_per_group=min_replay_per_group,
            )
            selected_cpu = selected.detach().cpu().tolist()
            residuals = residuals.index_select(0, selected)
            contexts = contexts.index_select(0, selected)
            base_weights = base_weights.index_select(0, selected)
            structural_weights = structural_weights.index_select(
                0, selected
            )
            sample_support = sample_support.index_select(0, selected)
            replay_support = replay_support.index_select(0, selected)
            if mode_ids is not None:
                mode_ids = mode_ids.index_select(0, selected)
            if law_keys is not None:
                law_keys = law_keys.index_select(0, selected)
            if bank_group_weights is not None:
                bank_group_weights = bank_group_weights.index_select(
                    0, selected
                )
            windows = [windows[index] for index in selected_cpu]
        weights, structural_strength, effective_sample_size = (
            SplitModule.normalize_split_evidence(
                base_weights,
                structural_weights,
                sample_support,
                replay_support,
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
            replay_support=replay_support.detach(),
            mode_ids=mode_ids,
            law_keys=law_keys,
            bank_group_weights=bank_group_weights,
            mode_summary=mode_summary,
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

        # Keep residual geometry equal to the learned dynamics update.  Write
        # quality is confidence and belongs in evidence weights, not in the
        # law vector used to form Split candidates.
        residuals = torch.stack(
            [item.delta_theta.reshape(-1) for item in valid_items],
            dim=0,
        )
        contexts = torch.stack([item.key.reshape(-1) for item in valid_items], dim=0)
        base_weights = torch.stack(
            [
                SplitModule.compute_memory_weight(
                    residual=item.delta_theta.reshape(-1),
                    age=float(item.age),
                    usage_count=float(item.usage),
                    reliability=float(getattr(item, "reliability", 1.0)),
                    tau_age=tau_age,
                    residual_norm_ema=residual_norm_ema,
                )
                for item in valid_items
            ],
            dim=0,
        )
        write_quality = base_weights.new_tensor([
            float(item.write_quality) for item in valid_items
        ]).clamp(0.0, 1.0)
        base_weights = (base_weights * write_quality).detach()
        structural_weights = base_weights.new_tensor(
            [float(item.queue_weight) for item in valid_items]
        )
        sample_support = base_weights.new_tensor([
            float(getattr(item, "support", 1.0)) for item in valid_items
        ]).clamp_min(0.0)
        replay_support = torch.ones_like(base_weights)
        weights, structural_strength, effective_sample_size = (
            SplitModule.normalize_split_evidence(
                base_weights,
                structural_weights,
                sample_support,
                replay_support,
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
            replay_support=replay_support.detach(),
        )

    @staticmethod
    def build_split_batch_from_memory_bank(
        bank: MemoryBank,
        tau_age: float = 20.0,
        residual_norm_ema: float = 1.0,
        max_items: Optional[int] = None,
        min_replay_per_group: int = 2,
    ) -> Optional[SplitBatch]:
        """Build a bounded evidence batch from one leaf memory bank."""
        if max_items is not None and max_items <= 0:
            raise ValueError("max_items must be positive when provided")
        bank._ensure_prototype_state()
        count = len(bank)
        if count == 0:
            return None
        mode_group_weights_all, mode_summary = (
            SplitModule._build_bank_mode_groups(
                mode_ids=bank.mode_ids[:count],
                law_keys=bank.law_keys[:count],
                deltas=bank.deltas[:count],
                support=bank.support[:count],
                split_mass=bank.split_mass[:count],
            )
        )
        valid_indices = [
            index for index, window in enumerate(bank.windows) if window is not None
        ]
        if not valid_indices:
            return None
        indices = torch.tensor(valid_indices, device=bank.device, dtype=torch.long)
        # Raw delta_theta determines the dynamics identity; quality is applied
        # only to base evidence weights below.
        residuals = bank.deltas[indices]
        contexts = bank.keys[indices]
        mode_ids = bank.mode_ids[indices].detach().to(dtype=torch.long)
        law_keys = bank.law_keys[indices].detach()
        bank_group_weights = mode_group_weights_all.index_select(0, indices)
        recency = torch.exp(-bank.age[indices] / tau_age)
        usage = 1.0 + torch.log1p(bank.usage[indices])
        strength = residuals.norm(dim=-1) / (residual_norm_ema + 1e-8)
        quality = bank.write_quality[indices].detach().clamp(0.0, 1.0)
        base_weights = (recency * usage * strength * quality).detach()
        structural_weights = bank.queue_weight[indices].detach()
        sample_support = bank.support[indices].detach().clamp_min(0.0)
        replay_support = torch.ones_like(base_weights)
        evidence_weights = base_weights * structural_weights * sample_support
        if max_items is not None and indices.numel() > max_items:
            selected = SplitModule._mode_stratified_topk(
                evidence_weights,
                bank_group_weights,
                int(max_items),
                min_per_group=min_replay_per_group,
            )
            indices = indices.index_select(0, selected)
            residuals = residuals.index_select(0, selected)
            contexts = contexts.index_select(0, selected)
            base_weights = base_weights.index_select(0, selected)
            structural_weights = structural_weights.index_select(0, selected)
            sample_support = sample_support.index_select(0, selected)
            replay_support = replay_support.index_select(0, selected)
            mode_ids = mode_ids.index_select(0, selected)
            law_keys = law_keys.index_select(0, selected)
            bank_group_weights = bank_group_weights.index_select(0, selected)
        weights, structural_strength, effective_sample_size = (
            SplitModule.normalize_split_evidence(
                base_weights,
                structural_weights,
                sample_support,
                replay_support,
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
            replay_support=replay_support,
            mode_ids=mode_ids,
            law_keys=law_keys,
            bank_group_weights=bank_group_weights,
            mode_summary=mode_summary,
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

    @torch.no_grad()
    def bank_mode_counterfactual_probe(
        self,
        *,
        theta_sem: torch.Tensor | HawkesParams | Any,
        batch: SplitBatch,
        hawkes_ll: Optional[HawkesFamily] = None,
        alpha_max: float = 0.25,
        trust_radius: float = 0.10,
        gain_reference: float = 0.05,
    ) -> Optional[Dict[str, Any]]:
        """Compare shared Light absorption with Bank-mode specialization.

        The method is deliberately non-mutating.  Both hypotheses are
        evaluated on the same frozen semantic parameter and the same bounded
        physical replay rows.  Bank mode assignment forms H1; residuals are
        never re-clustered and write quality only affects reliability weights.
        """
        if alpha_max < 0.0 or trust_radius < 0.0 or gain_reference <= 0.0:
            raise ValueError("invalid Light probe step-size settings")
        if batch.bank_group_weights is None or batch.mode_summary is None:
            return None
        if int(batch.mode_summary["mode_ids"].numel()) < 2:
            return None

        hawkes_ll = self._resolve_hawkes_ll(hawkes_ll)
        self.validate_batch(batch)
        reference = self._module_reference(batch.residuals)
        theta = self._theta_sem_to_flat(theta_sem, reference).detach().clone()
        frozen = self._batch_to_device(batch, theta)
        q = frozen.bank_group_weights
        if q is None:
            return None
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        persistent_support = (
            torch.ones_like(frozen.weights)
            if frozen.sample_support is None
            else frozen.sample_support.clamp_min(0.0)
        )
        replay_support = (
            torch.ones_like(frozen.weights)
            if frozen.replay_support is None
            else frozen.replay_support.clamp_min(0.0)
        )
        base = (
            frozen.weights
            if frozen.base_weights is None
            else frozen.base_weights.clamp_min(0.0)
        )
        reliability = base * persistent_support * replay_support
        total_weight = reliability.sum()
        if not bool(torch.isfinite(total_weight)) or bool(total_weight <= self.eps):
            return None

        weighted_sum = (reliability[:, None] * frozen.residuals).sum(0)
        shared_delta = weighted_sum / total_weight
        energy = (
            reliability
            * frozen.residuals.square().sum(dim=-1)
        ).sum()
        coherence = (
            weighted_sum.square().sum()
            / (total_weight * energy).clamp_min(self.eps)
        ).clamp(0.0, 1.0)

        replay_cache = build_replay_batch_cache(
            frozen.windows, hawkes_ll, theta
        )
        # The optimized replay kernel evaluates the parent plus exactly two
        # alternatives. Duplicate the shared alternative for this H0
        # step-size calibration pass.
        shared_models = torch.stack(
            (theta, theta + shared_delta, theta + shared_delta), dim=0
        )
        shared_nll = batched_replay_log_likelihood(
            replay_cache,
            shared_models,
            hawkes_ll,
            normalize_by_events=True,
        )
        full_step_gain = (
            reliability * (shared_nll[:, 0] - shared_nll[:, 1])
        ).sum() / total_weight
        trust = min(
            1.0,
            float(trust_radius)
            / (float(shared_delta.norm().item()) + self.eps),
        )
        gain_scale = min(
            1.0,
            max(0.0, float(full_step_gain.item())) / float(gain_reference),
        )
        alpha_light = (
            float(alpha_max)
            * float(coherence.item())
            * trust
            * gain_scale
        )
        theta_h0 = theta + alpha_light * shared_delta

        grouped_weight = reliability[:, None] * q
        group_mass = grouped_weight.sum(dim=0)
        # This is availability, not a persistence threshold: H1 is undefined
        # when the frozen replay contains no physical window for one side.
        if bool((group_mass <= self.eps).any()):
            return None
        group_delta = grouped_weight.transpose(0, 1) @ frozen.residuals
        group_delta = group_delta / group_mass[:, None]
        theta_h1 = theta[None, :] + group_delta

        probe_models = torch.cat((theta_h0[None, :], theta_h1), dim=0)
        probe_nll = batched_replay_log_likelihood(
            replay_cache,
            probe_models,
            hawkes_ll,
            normalize_by_events=True,
        )
        loss_h0 = (reliability * probe_nll[:, 0]).sum() / total_weight
        row_h1_nll = (q * probe_nll[:, 1:3]).sum(dim=-1)
        loss_h1 = (reliability * row_h1_nll).sum() / total_weight
        advantage = loss_h0 - loss_h1
        probability = reliability / total_weight
        replay_ess = 1.0 / probability.square().sum().clamp_min(self.eps)

        return {
            "theta_sem_snapshot": theta.detach(),
            "batch": batch,
            "mode_ids": batch.mode_summary["mode_ids"].detach().clone(),
            "mode_group_ids": batch.mode_summary["group_ids"].detach().clone(),
            "bank_group_weights": q.detach(),
            "shared_delta": shared_delta.detach(),
            "group_delta": group_delta.detach(),
            "theta_h0": theta_h0.detach(),
            "theta_h1": theta_h1.detach(),
            "alpha_light": float(alpha_light),
            "coherence": float(coherence.item()),
            "full_step_gain": float(full_step_gain.item()),
            "loss_h0": float(loss_h0.item()),
            "loss_h1": float(loss_h1.item()),
            "advantage": float(advantage.item()),
            "protect": bool(advantage > 0.0),
            "replay_ess": float(replay_ess.item()),
            "replay_rows": int(frozen.residuals.size(0)),
            "replay_weights": (
                reliability / total_weight * reliability.numel()
            ).detach(),
        }

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
        m_min: float = 0.0,
        min_structural_strength: float = 0.0,
        min_effective_sample_size: float = 0.0,
    ) -> bool:
        """Validate a proposal without imposing hand-designed thresholds."""
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

        # m_min, structural strength and proposal ESS are retained only as
        # diagnostics/backward-compatible arguments.  Bank admission owns
        # persistence; the predictive objective owns the decision boundary.
        eligible = bool(
            bool(torch.isfinite(N_mass).all())
            and bool(torch.isfinite(N_eff).all())
            and math.isfinite(structural_strength)
            and math.isfinite(effective_sample_size)
        )
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
        m_min: float = 0.0,
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
