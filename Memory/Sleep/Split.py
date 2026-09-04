import math

import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass, replace
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
    binary Bank prior used by structural H1 and router distillation.
    """
    residuals: torch.Tensor
    contexts: torch.Tensor
    weights: torch.Tensor
    windows: List[EventWindow]
    # ``weights`` are normalized to replay-count scale before Split uses
    # them.  Keep the unnormalized factors and their scale-free safeguards so
    # normalization cannot manufacture structural evidence from one dominant
    # memory.  ``structural_weights`` is retained for provenance/compatibility
    # but is not applied to Split evidence.
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
    # Row-wise binary Bank prior q_bank generated from Bank mode summaries.
    bank_group_weights: Optional[torch.Tensor] = None
    # Full-bank mode summaries are retained for diagnostics and provenance;
    # structural Split likelihood and counterfactual H1 consume
    # bank_group_weights, while the router is trained separately toward it.
    mode_summary: Optional[Dict[str, torch.Tensor]] = None


@dataclass(frozen=True)
class FrozenBankSplitHypothesis:
    """One immutable Bank-backed Split hypothesis shared by Light and Deep.

    The tensors are detached snapshots.  ``batch`` contains the exact replay
    rows used to form the hypothesis; Deep fitting may refine child
    parameters, but it must not reconstruct ``q_bank`` or the initial child
    laws from a second evidence pass.
    """

    theta_h0: torch.Tensor
    theta_child_init: torch.Tensor
    q_bank: torch.Tensor
    delta_bar: torch.Tensor
    replay_weights: torch.Tensor
    windows: List[EventWindow]
    mode_ids: torch.Tensor
    mode_group_ids: torch.Tensor
    mode_summary: Dict[str, torch.Tensor]
    # The batch is an internal convenience for the shared Pre-Light → Deep
    # path.  Keep it optional so the public nine-field hypothesis record can
    # also be constructed by older callers that already hold the fields
    # separately.
    batch: Optional[SplitBatch] = None
    h1_defined: bool = False
    h1_representative_counts: Optional[torch.Tensor] = None


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

        # Legacy residual-clustering gate. Bank-backed fitting never reads or
        # optimizes this parameter; retain it only for old checkpoints/API.
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

        # Legacy residual-clustering controls; Bank-backed fitting retains
        # them for checkpoint compatibility but does not read them.
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

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

        # ``weights`` is already the row-level structural evidence e_r.  In
        # particular, Bank batches include persistent support, quality,
        # recency, and usage before normalization.  Do not multiply it by
        # residual magnitude or by ``sample_support`` a second time.
        # ``replay_support`` remains an API/provenance field, but child mass,
        # prototypes, and ESS all use this one definition of u_{rj}=e_r q_{rj}.
        if sample_support is not None and sample_support.shape != weights.shape:
            raise ValueError("sample_support and weights must be aligned vectors")
        if replay_support is not None and replay_support.shape != weights.shape:
            raise ValueError("replay_support and weights must be aligned vectors")
        child_evidence = self.compute_child_evidence(
            weights=weights,
            bank_group_weights=q,
            residuals=residuals,
            eps=eps,
        )
        q = child_evidence["q"]
        N_mass = child_evidence["child_mass"]
        N_eff = child_evidence["child_ess"]
        pi_bar = child_evidence["pi_bar"]
        delta_bar = child_evidence["delta_bar"]

        if bank_group_weights is not None:
            # Canonical Bank hypothesis.  Persistent Bank modes already are
            # the law identity, so Split must not rebase the parent or
            # rescale the mode contrast with the legacy alpha/rho controls:
            #
            #   H0 = theta_sem
            #   H1_j = theta_sem + delta_bar_j
            #
            # Keep the retired fields in the output for old checkpoints and
            # diagnostics, but do not read their learned parameters on this
            # path.  The Bank prior and its mode-conditioned residual means
            # are the sole source of the child laws.
            delta_shared = torch.zeros_like(theta_sem)
            delta_child = delta_bar
            alpha = theta_sem.new_ones(())
            rho_sp = theta_sem.new_ones(())
            theta_plus = theta_sem
            theta_cand = theta_sem[None, :] + delta_bar
        else:
            # Legacy residual-clustering path.  This remains available for
            # directly constructed SplitBatch objects and old callers.
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
            "row_evidence": child_evidence["row_evidence"],
            "child_evidence": child_evidence["child_evidence"],
            "child_effective_mass": N_mass,
            "child_ess": N_eff,
            "two_sided_support": child_evidence["two_sided_support"],
            # One-sided row mass is a diagnostic.  Bank H1 evaluability is
            # determined later from persistent groups plus representative
            # windows, so it must not be encoded as an invalid proposal here.
            "invalid_proposal": False,
            "pi_bar": pi_bar,
            "delta_bar": delta_bar,
            "delta_shared": delta_shared,
            "delta_child": delta_child,
            "alpha": alpha,
            "rho_sp": rho_sp,
            "canonical_bank_hypothesis": bank_group_weights is not None,
            "theta_plus": theta_plus,
            "theta_cand": theta_cand,
        }

    @staticmethod
    def _bank_effective_mode_count(batch: SplitBatch) -> Optional[int]:
        """Read the effective Bank-mode count carried by a production batch."""
        if batch.mode_summary is None:
            return None
        value = batch.mode_summary.get("effective_mode_count")
        if value is None:
            return None
        value = torch.as_tensor(value).detach().reshape(-1)
        if value.numel() != 1 or not bool(torch.isfinite(value).all()):
            return None
        return int(value.item())

    @staticmethod
    def _bank_row_group_ids(batch: SplitBatch) -> Optional[torch.Tensor]:
        """Resolve each replay row to its persistent Bank child group.

        ``q_bank=[.5,.5]`` is intentionally neutral for an unresolved mode.
        Taking ``argmax(q_bank)`` alone would incorrectly turn that neutral
        row into a left-child representative, so production batches use the
        full mode summary and preserve its ``-1`` unresolved state here.
        """
        if batch.bank_group_weights is None:
            return None
        q_bank = batch.bank_group_weights
        if batch.mode_ids is None or batch.mode_summary is None:
            return q_bank.argmax(dim=-1)
        summary_mode_ids = batch.mode_summary.get("mode_ids")
        summary_group_ids = batch.mode_summary.get("group_ids")
        if summary_mode_ids is None or summary_group_ids is None:
            return q_bank.argmax(dim=-1)
        summary_mode_ids = torch.as_tensor(summary_mode_ids).reshape(-1)
        summary_group_ids = torch.as_tensor(summary_group_ids).reshape(-1)
        if summary_mode_ids.numel() != summary_group_ids.numel():
            return q_bank.argmax(dim=-1)
        mode_to_group = {
            int(mode): int(group)
            for mode, group in zip(
                summary_mode_ids.detach().cpu().tolist(),
                summary_group_ids.detach().cpu().tolist(),
            )
        }
        return torch.tensor(
            [
                mode_to_group.get(int(mode), -1)
                for mode in batch.mode_ids.detach().cpu().tolist()
            ],
            device=q_bank.device,
            dtype=torch.long,
        )

    @staticmethod
    def _bank_h1_evaluability(batch: SplitBatch) -> Dict[str, Any]:
        """Check only whether the Bank H1 can be numerically evaluated.

        Structural existence is decided by persistent Bank modes.  The only
        replay-side prerequisite is one retained representative window for
        each of the two persistent groups; ESS, mass, and current replay
        weight thresholds are deliberately not used here.
        """
        if batch.bank_group_weights is None:
            return {
                "h1_defined": True,
                "h1_representative_counts": torch.zeros(
                    2, dtype=torch.long, device=batch.residuals.device
                ),
            }
        group_ids = SplitModule._bank_row_group_ids(batch)
        if group_ids is None:
            return {
                "h1_defined": False,
                "h1_representative_counts": torch.zeros(
                    2, dtype=torch.long, device=batch.residuals.device
                ),
            }
        replay_present = torch.tensor(
            [window is not None for window in batch.windows],
            device=group_ids.device,
            dtype=torch.bool,
        )
        counts = torch.stack([
            ((group_ids == group) & replay_present).sum()
            for group in range(2)
        ]).to(dtype=torch.long)
        return {
            "h1_defined": bool((counts > 0).all()),
            "h1_representative_counts": counts,
        }

    @staticmethod
    def _assert_bank_mode_support_invariant(
        batch: SplitBatch,
        two_sided_support: bool,
        *,
        stage: str,
    ) -> Optional[int]:
        """Ensure Bank mode provenance agrees with binary replay support."""
        if batch.bank_group_weights is None:
            return None
        effective_mode_count = SplitModule._bank_effective_mode_count(batch)
        if effective_mode_count is not None and effective_mode_count < 2:
            assert not two_sided_support, (
                "Bank q_bank provenance mismatch during "
                f"{stage}: K_effective_mode={effective_mode_count} "
                "but two_sided_support=True"
            )
        return effective_mode_count

    def build_bank_split_hypothesis(
        self,
        batch: SplitBatch,
        theta_h0: torch.Tensor | HawkesParams | Any,
    ) -> FrozenBankSplitHypothesis:
        """Freeze the canonical Bank H0/H1 construction for Light and Deep.

        ``q_bank`` and ``delta_bar`` are computed exactly once at this
        boundary.  Consumers may evaluate or locally refine the child laws,
        but they must not derive a different assignment from replay coverage,
        router probabilities, or a second residual decomposition.
        """
        if not isinstance(batch, SplitBatch):
            raise TypeError("batch must be a SplitBatch")
        self.validate_batch(batch)
        if batch.bank_group_weights is None:
            raise ValueError(
                "Bank Split hypotheses require batch.bank_group_weights"
            )
        reference = self._module_reference(batch.residuals)
        theta = self._theta_sem_to_flat(theta_h0, reference).detach().clone()
        frozen_batch = self._batch_to_device(batch, theta)
        self.validate_batch(frozen_batch)

        q_bank = frozen_batch.bank_group_weights.detach().clone()
        q_bank = q_bank / q_bank.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        replay_weights = frozen_batch.weights.detach().clamp_min(0.0)
        child_evidence = self.compute_child_evidence(
            weights=replay_weights,
            bank_group_weights=q_bank,
            residuals=frozen_batch.residuals,
            eps=self.eps,
        )
        q_bank = child_evidence["q"].detach().clone()
        frozen_batch = replace(
            frozen_batch,
            bank_group_weights=q_bank,
        )
        h1_diagnostics = self._bank_h1_evaluability(frozen_batch)

        summary = {
            key: value.detach().clone()
            for key, value in (frozen_batch.mode_summary or {}).items()
            if torch.is_tensor(value)
        }
        summary_mode_ids = summary.get("mode_ids")
        summary_group_ids = summary.get("group_ids")
        if (
            torch.is_tensor(summary_mode_ids)
            and torch.is_tensor(summary_group_ids)
            and summary_mode_ids.ndim == 1
            and summary_group_ids.shape == summary_mode_ids.shape
        ):
            mode_ids = summary_mode_ids.detach().clone().to(dtype=torch.long)
            mode_group_ids = summary_group_ids.detach().clone().to(dtype=torch.long)
        elif frozen_batch.mode_ids is not None:
            mode_ids = torch.unique(
                frozen_batch.mode_ids.detach().to(dtype=torch.long),
                sorted=True,
            )
            row_groups = q_bank.argmax(dim=-1)
            mode_group_ids = torch.stack([
                row_groups[frozen_batch.mode_ids == mode].reshape(-1)[0]
                for mode in mode_ids
            ]).to(dtype=torch.long)
        else:
            mode_ids = torch.arange(
                frozen_batch.residuals.size(0),
                device=frozen_batch.residuals.device,
                dtype=torch.long,
            )
            mode_group_ids = q_bank.argmax(dim=-1).to(dtype=torch.long)

        delta_bar = child_evidence["delta_bar"].detach().clone()
        theta_child_init = theta[None, :] + delta_bar
        return FrozenBankSplitHypothesis(
            theta_h0=theta,
            theta_child_init=theta_child_init.detach().clone(),
            q_bank=q_bank.detach().clone(),
            delta_bar=delta_bar,
            replay_weights=replay_weights.detach().clone(),
            windows=list(frozen_batch.windows),
            mode_ids=mode_ids.detach().clone(),
            mode_group_ids=mode_group_ids.detach().clone(),
            mode_summary=summary,
            batch=frozen_batch,
            h1_defined=bool(h1_diagnostics["h1_defined"]),
            h1_representative_counts=(
                h1_diagnostics["h1_representative_counts"].detach().clone()
            ),
        )

    # Explicit alias for callers that use the terminology from the design
    # doc; both names return the same canonical frozen object shape.
    freeze_bank_split_hypothesis = build_bank_split_hypothesis

    def compute_split_training_objective(
        self,
        *,
        weights: torch.Tensor,
        ell_split: torch.Tensor,
        g_split: torch.Tensor,
        N_eff: torch.Tensor,
        lambda_T: float | torch.Tensor,
        delta_complexity: float | torch.Tensor,
        route_loss: Optional[torch.Tensor] = None,
        lambda_route: float | torch.Tensor = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Combine structural Split training with router distillation.

        ``ell_split`` is the Bank-conditioned structural likelihood.  The
        router term is additive and is never included in the structural gain
        used by ``UnifiedTopology``.
        """
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
        route_lambda_tensor = torch.as_tensor(
            lambda_route,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(())
        if not bool(torch.isfinite(lambda_tensor)):
            raise FloatingPointError("lambda_T must be finite")
        if not bool(torch.isfinite(delta_tensor)):
            raise FloatingPointError("delta_complexity must be finite")
        if not bool(torch.isfinite(route_lambda_tensor)):
            raise FloatingPointError("lambda_route must be finite")
        if bool(lambda_tensor < 0.0):
            raise ValueError("lambda_T must be non-negative")
        if bool(delta_tensor <= 0.0):
            raise ValueError("delta_complexity must be positive")
        if bool(route_lambda_tensor < 0.0):
            raise ValueError("lambda_route must be non-negative")

        prediction_loss = -(
            weights * ell_split
        ).sum() / (weights.sum() + self.eps)
        expected_complexity_delta = g_split * delta_tensor
        complexity_penalty = lambda_tensor * expected_complexity_delta
        mass_penalty = self.lambda_mass * F.softplus(
            (self.m_min - N_eff) / self.tau_m
        ).sum()
        if route_loss is None:
            route_loss_tensor = reference.new_zeros(())
        else:
            route_loss_tensor = torch.as_tensor(
                route_loss,
                device=reference.device,
                dtype=reference.dtype,
            ).reshape(())
            if not bool(torch.isfinite(route_loss_tensor)):
                raise FloatingPointError("route_loss must be finite")
        route_penalty = route_lambda_tensor * route_loss_tensor
        structural_loss = (
            prediction_loss
            + complexity_penalty
            + mass_penalty
        )
        loss = structural_loss + route_penalty
        return {
            "loss": loss,
            "structural_loss": structural_loss,
            "prediction_loss": prediction_loss,
            "complexity_penalty": complexity_penalty,
            "mass_penalty": mass_penalty,
            "route_loss": route_loss_tensor,
            "route_penalty": route_penalty,
            "lambda_T": lambda_tensor,
            "lambda_route": route_lambda_tensor,
            "delta_tree_complexity": delta_tensor,
            "expected_delta_tree_complexity": (
                expected_complexity_delta
            ),
        }

    def compute_bank_split_training_objective(
        self,
        *,
        weights: torch.Tensor,
        ell_h1: torch.Tensor,
        anchor_loss: Optional[torch.Tensor] = None,
        lambda_anchor: float | torch.Tensor = 0.0,
        route_loss: Optional[torch.Tensor] = None,
        lambda_route: float | torch.Tensor = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Fit the frozen Bank H1 directly.

        Bank mode grouping is structural identity, not a differentiable
        split gate.  Consequently this objective contains only the H1 replay
        likelihood, an optional child-initialization anchor, and an optional
        q_bank-to-router distillation term.  It deliberately has no
        ``g_split``, complexity relaxation, or mass threshold term.
        """
        reference = ell_h1
        anchor_lambda = torch.as_tensor(
            lambda_anchor,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(())
        route_lambda = torch.as_tensor(
            lambda_route,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(())
        if not bool(torch.isfinite(anchor_lambda)):
            raise FloatingPointError("lambda_anchor must be finite")
        if not bool(torch.isfinite(route_lambda)):
            raise FloatingPointError("lambda_route must be finite")
        if bool(anchor_lambda < 0.0):
            raise ValueError("lambda_anchor must be non-negative")
        if bool(route_lambda < 0.0):
            raise ValueError("lambda_route must be non-negative")

        prediction_loss = -(
            weights * ell_h1
        ).sum() / (weights.sum() + self.eps)
        if anchor_loss is None:
            anchor_loss_tensor = reference.new_zeros(())
        else:
            anchor_loss_tensor = torch.as_tensor(
                anchor_loss,
                device=reference.device,
                dtype=reference.dtype,
            ).reshape(())
        if not bool(torch.isfinite(anchor_loss_tensor)):
            raise FloatingPointError("anchor_loss must be finite")
        if route_loss is None:
            route_loss_tensor = reference.new_zeros(())
        else:
            route_loss_tensor = torch.as_tensor(
                route_loss,
                device=reference.device,
                dtype=reference.dtype,
            ).reshape(())
        if not bool(torch.isfinite(route_loss_tensor)):
            raise FloatingPointError("route_loss must be finite")

        anchor_penalty = anchor_lambda * anchor_loss_tensor
        route_penalty = route_lambda * route_loss_tensor
        structural_loss = prediction_loss + anchor_penalty
        loss = structural_loss + route_penalty
        zero = reference.new_zeros(())
        return {
            "loss": loss,
            "structural_loss": structural_loss,
            "prediction_loss": prediction_loss,
            "h1_loss": prediction_loss,
            "anchor_loss": anchor_loss_tensor,
            "anchor_penalty": anchor_penalty,
            "complexity_penalty": zero,
            "mass_penalty": zero,
            "route_loss": route_loss_tensor,
            "route_penalty": route_penalty,
            "lambda_anchor": anchor_lambda,
            "lambda_route": route_lambda,
            "lambda_T": zero,
            "delta_tree_complexity": zero,
            "expected_delta_tree_complexity": zero,
        }

    @staticmethod
    def evaluate_bank_h1(
        q_bank: torch.Tensor,
        logp_child_each: torch.Tensor,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the common Bank-conditioned child hypothesis ``H1``.

        ``q_bank`` is the persistent Bank assignment prior for each replay
        row and ``logp_child_each`` contains the two child log likelihoods.
        Both the actual Split test and the Light-vs-Bank counterfactual probe
        use this same reduction:

            log p_H1^bank(w_r) = log sum_j q^bank_{rj} p(w_r | theta_j).

        The Bank prior is detached because it is evidence supplied by the
        persistent memory, not a parameter learned by the Split optimizer.
        """
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if q_bank.ndim != 2 or q_bank.shape != logp_child_each.shape:
            raise ValueError(
                "q_bank and logp_child_each must both have shape [R, 2]"
            )
        q = q_bank.detach().to(
            device=logp_child_each.device,
            dtype=logp_child_each.dtype,
        )
        if not bool(torch.isfinite(q).all()):
            raise FloatingPointError("q_bank must contain only finite values")
        if bool((q < 0.0).any()):
            raise ValueError("q_bank must be non-negative")
        row_mass = q.sum(dim=-1, keepdim=True)
        if bool((row_mass <= eps).any()):
            raise ValueError("each q_bank row must have positive mass")
        q = q / row_mass.clamp_min(eps)
        log_q = torch.where(
            q > 0.0,
            q.clamp_min(eps).log(),
            torch.full_like(q, -torch.inf),
        )
        logp_child_bank = torch.logsumexp(
            log_q + logp_child_each,
            dim=-1,
        )
        return {
            "q_bank": q,
            "log_q_bank": log_q,
            "logp_child_bank": logp_child_bank,
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
            ell_struct: [R]
            logp0: [R]
            logp_child_bank: [R]
            logp_child_router: [R]
            logp_child_each: [R, 2]
            route_prob: [R, 2]
            q_bank: [R, 2]

        Bank-backed batches return the direct H1 likelihood in both
        ``ell_struct`` and ``ell_split``.  ``g_split`` is emitted only for
        the legacy residual-clustering fallback.
        """
        hawkes_ll = self._resolve_hawkes_ll(hawkes_ll)
        eps = self.eps
        has_bank_prior = batch.bank_group_weights is not None

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

        # Structural H1 is conditioned on the persistent Bank assignment.
        # Directly constructed legacy batches have no Bank prior; retaining
        # the router as their fallback preserves the old standalone API while
        # all production MemoryBank batches use q_bank here.
        q_prior = (
            batch.bank_group_weights
            if has_bank_prior
            else route_prob.detach()
        )
        bank_h1 = self.evaluate_bank_h1(
            q_prior,
            logp_child_each,
            eps=eps,
        )
        q_bank = bank_h1["q_bank"]
        log_q_bank = bank_h1["log_q_bank"]
        logp_child_bank = bank_h1["logp_child_bank"]

        # Router likelihood is retained as an implementation diagnostic and
        # as a separate distillation target.  It never enters ell_struct.
        logp_child_router = torch.logsumexp(
            log_route_prob + logp_child_each,
            dim=-1,
        )                                                             # [R]
        replay_weights = batch.weights.to(
            device=log_route_prob.device,
            dtype=log_route_prob.dtype,
        ).detach().clamp_min(0.0)
        total_replay_weight = replay_weights.sum().clamp_min(eps)
        route_loss_per_row = -(
            q_bank * log_route_prob
        ).sum(dim=-1)
        finite_log_q = torch.where(
            q_bank > 0.0,
            log_q_bank,
            torch.zeros_like(log_q_bank),
        )
        route_kl_per_row = (
            q_bank * (finite_log_q - log_route_prob)
        ).sum(dim=-1)
        route_match = (
            q_bank.argmax(dim=-1) == route_prob.argmax(dim=-1)
        ).to(log_route_prob.dtype)
        if has_bank_prior:
            # Keep the router term in the strategy's evidence currency:
            # L_route = -sum_r e_r sum_j q_bank[r,j] log pi_j(z_r).
            # H1 is normalized by total replay evidence; this auxiliary
            # distillation term is intentionally retained as a weighted sum.
            route_loss = (
                replay_weights * route_loss_per_row
            ).sum()
            route_kl = (
                replay_weights * route_kl_per_row
            ).sum() / total_replay_weight
            route_accuracy = (
                replay_weights * route_match
            ).sum() / total_replay_weight
        else:
            # There is no persistent Bank target in the legacy API path, so
            # there is no router-distillation penalty to optimize.
            route_loss = log_route_prob.new_zeros(())
            route_kl = log_route_prob.new_zeros(())
            route_accuracy = route_match.mean()

        # Bank law identity is already established by persistent mode
        # grouping.  Deep/Pre-Light therefore evaluate H1 directly; no
        # differentiable gate or H0/H1 log-mixture is allowed on this path.
        if has_bank_prior:
            ell_struct = logp_child_bank
        else:
            # Compatibility path for directly constructed legacy batches.
            g_split = torch.sigmoid(self.gate_logit / self.tau_g)
            log_g = torch.log(g_split + eps)
            log_1_minus_g = torch.log(1.0 - g_split + eps)
            ell_struct = torch.logaddexp(
                log_1_minus_g + logp0,
                log_g + logp_child_bank,
            )                                                     # [R]

        replay_output = {
            "route_prob": route_prob,
            "has_bank_prior": has_bank_prior,
            "logp0": logp0,
            "logp_child_each": logp_child_each,
            "q_bank": q_bank,
            "log_q_bank": log_q_bank,
            "logp_child_bank": logp_child_bank,
            "logp_child_router": logp_child_router,
            # Deprecated structural alias; new code must use the explicit
            # ``logp_child_bank``/``logp_child_router`` names above.
            "logp_child_mix": logp_child_bank,
            "route_loss": route_loss,
            "route_kl": route_kl,
            "route_accuracy": route_accuracy,
            "route_loss_per_row": route_loss_per_row,
            "route_kl_per_row": route_kl_per_row,
            "ell_struct": ell_struct,
            "ell_split": ell_struct,
        }
        if not has_bank_prior:
            replay_output["g_split"] = g_split
        return replay_output

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
                            else value.dtype
                            if value.dtype == torch.bool
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
        lambda_anchor: float | torch.Tensor = 0.0,
        lambda_route: float | torch.Tensor = 0.0,
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
        effective_mode_count = self._assert_bank_mode_support_invariant(
            batch,
            bool(struct["two_sided_support"]),
            stage="forward",
        )
        h1_diagnostics = self._bank_h1_evaluability(batch)

        replay = self.compute_replay_loglikelihood(
            batch=batch,
            theta_plus=struct["theta_plus"],
            theta_cand=struct["theta_cand"],
            hawkes_ll=hawkes_ll,
            replay_cache=replay_cache,
        )

        if batch.bank_group_weights is not None:
            # ``theta_cand`` is the frozen Bank initialization in the normal
            # forward path, so its anchor is exactly zero.  The separate
            # ``fit_bank_split`` path supplies a refined child tensor and a
            # non-zero anchor when Deep fitting is requested.
            objective = self.compute_bank_split_training_objective(
                # ``weights`` already is the normalized row-evidence measure
                # used for H1; replay support is never multiplied a second
                # time here.
                weights=weights,
                ell_h1=replay["logp_child_bank"],
                anchor_loss=theta_sem.new_zeros(()),
                lambda_anchor=lambda_anchor,
                route_loss=replay["route_loss"],
                lambda_route=lambda_route,
            )
        else:
            objective = self.compute_split_training_objective(
                # Legacy directly-constructed batches retain the historical
                # relaxed gate objective for checkpoint/API compatibility.
                weights=weights,
                ell_split=replay["ell_split"],
                g_split=replay["g_split"],
                N_eff=struct["N_eff"],
                lambda_T=(
                    self.lambda_tree if lambda_T is None else lambda_T
                ),
                delta_complexity=delta_complexity,
                route_loss=replay["route_loss"],
                lambda_route=lambda_route,
            )

        out = {}
        out.update(struct)
        out.update(replay)
        out.update(objective)
        out["has_bank_prior"] = batch.bank_group_weights is not None
        out["h1_defined"] = bool(h1_diagnostics["h1_defined"])
        out["h1_representative_counts"] = h1_diagnostics[
            "h1_representative_counts"
        ].detach().clone()
        if batch.bank_group_weights is not None:
            q_bank_mass = struct["N_mass"].detach().clone()
            out["q_bank_mass"] = q_bank_mass
            out["q_bank_mass_L"] = q_bank_mass[0]
            out["q_bank_mass_R"] = q_bank_mass[1]
            # ``two_sided_support`` remains a row-evidence diagnostic.  The
            # Bank-backed structural gate is only the independent H1
            # evaluability invariant, and must not be reintroduced through
            # ``invalid_proposal``.
            out["invalid_proposal"] = not bool(h1_diagnostics["h1_defined"])
            if effective_mode_count is not None:
                out["K_effective_mode"] = torch.as_tensor(
                    effective_mode_count,
                    device=q_bank_mass.device,
                    dtype=q_bank_mass.dtype,
                )
        if batch.mode_summary is not None:
            out["bank_mode_summary"] = {
                key: value.detach().clone()
                for key, value in batch.mode_summary.items()
            }
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
        # ``residual_norm_ema`` is retained for checkpoint/API compatibility.
        # Residual magnitude is a law value, not evidence reliability: Light
        # semantic rebasing can make a well-supported law's residual nearly
        # zero without removing its Bank evidence.
        del residual_norm_ema, eps
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
        weight = recency * usage * reliability
        return weight.detach()

    @staticmethod
    @torch.no_grad()
    def compute_bank_row_evidence(
        bank: MemoryBank,
        *,
        tau_age: float = 20.0,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        """Return the single row-evidence definition used by Bank-backed Split.

        ``e_r = recency_r * usage_r * quality_r * support_r``.
        ``queue_weight`` is retained as Bank provenance, but does not scale
        current Split evidence.
        A retained replay window is the physical observation; ``support`` is
        only the persistent observation mass represented by that row.  Rows
        without a replay window remain in the Bank for persistence/commit, but
        contribute no current Split evidence.
        """
        if tau_age <= 0.0:
            raise ValueError("tau_age must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        bank._ensure_prototype_state()
        count = len(bank)
        if count == 0:
            empty = bank.keys.new_empty(0)
            return {
                "base_weights": empty,
                "structural_weights": empty,
                "sample_support": empty,
                "replay_support": empty,
                "evidence_weights": empty,
                "valid_mask": torch.empty(
                    0, device=bank.device, dtype=torch.bool
                ),
            }

        age = bank.age[:count].clamp_min(0.0)
        usage_count = bank.usage[:count].clamp_min(0.0)
        recency = torch.exp(-age / tau_age)
        usage = 1.0 + torch.log1p(usage_count)
        quality = bank.write_quality[:count].clamp(0.0, 1.0)
        base_weights = (recency * usage * quality).detach()
        structural_weights = bank.queue_weight[:count].clamp(0.0, 1.0).detach()
        sample_support = bank.support[:count].clamp_min(0.0).detach()
        valid_mask = torch.tensor(
            [window is not None for window in bank.windows[:count]],
            device=bank.device,
            dtype=torch.bool,
        )
        evidence_weights = (
            base_weights
            * sample_support
            * valid_mask.to(dtype=base_weights.dtype)
        ).detach()
        return {
            "base_weights": base_weights,
            "structural_weights": structural_weights,
            "sample_support": sample_support,
            # One retained window is one independent replay observation even
            # when a compact row represents many historical observations.
            "replay_support": torch.ones_like(base_weights),
            "evidence_weights": evidence_weights,
            "valid_mask": valid_mask,
        }

    @staticmethod
    def compute_child_evidence(
        *,
        weights: torch.Tensor,
        bank_group_weights: torch.Tensor,
        residuals: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> Dict[str, Any]:
        """Compute child mass, prototypes, and ESS from one evidence measure.

        The returned ``child_evidence`` is exactly
        ``u[r, j] = weights[r] * q[r, j]``.  All Bank-backed Split consumers
        use this helper so effective child mass cannot silently diverge from
        the weights used for prototypes or uncertainty diagnostics.
        """
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if bank_group_weights.ndim != 2 or bank_group_weights.shape != (
            weights.numel(),
            2,
        ):
            raise ValueError("bank_group_weights must have shape [R, 2]")
        evidence_dtype = (
            weights.dtype
            if weights.is_floating_point()
            else (
                bank_group_weights.dtype
                if bank_group_weights.is_floating_point()
                else torch.get_default_dtype()
            )
        )
        q = bank_group_weights.to(
            device=weights.device,
            dtype=evidence_dtype,
        )
        if not bool(torch.isfinite(q).all()):
            raise FloatingPointError("bank_group_weights must be finite")
        if bool((q < 0.0).any()):
            raise ValueError("bank_group_weights must be non-negative")
        row_mass = q.sum(dim=-1, keepdim=True)
        if bool((row_mass <= eps).any()):
            raise ValueError("each bank_group_weights row must have positive mass")
        q = q / row_mass.clamp_min(eps)
        row_evidence = weights.to(q).clamp_min(0.0)
        child_evidence = row_evidence[:, None] * q
        child_mass = child_evidence.sum(dim=0)
        child_ess = child_mass.square() / (
            child_evidence.square().sum(dim=0) + eps
        )
        pi_bar = child_mass / (child_mass.sum() + eps)
        if residuals is None:
            delta_bar = None
        else:
            if residuals.ndim != 2 or residuals.shape[0] != weights.numel():
                raise ValueError("residuals must have shape [R, P]")
            delta_bar = child_evidence.transpose(0, 1) @ residuals.to(q)
            delta_bar = delta_bar / (child_mass[:, None] + eps)
        return {
            "q": q,
            "row_evidence": row_evidence,
            "child_evidence": child_evidence,
            "child_mass": child_mass,
            "child_ess": child_ess,
            "pi_bar": pi_bar,
            "delta_bar": delta_bar,
            "two_sided_support": bool((child_mass > eps).all()),
        }

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
        # ``structural_weights`` is a compatibility/provenance field only.
        # Controller queue confidence must not dilute Split evidence.
        evidence = support * base
        evidence_sum = evidence.sum()
        replay_count = replay.sum()
        # A retained replay window is sufficient for numerical evaluation.
        # If all optional confidence factors happen to be zero, do not turn
        # that diagnostic into another structural/mass gate: use the replay
        # observation measure itself as a finite fallback.  Bank H1 existence
        # is decided separately by persistent groups plus representative
        # windows.
        replay_evidence = torch.where(
            evidence_sum > eps,
            evidence,
            replay,
        )
        replay_evidence_sum = replay_evidence.sum()
        normalized = (
            replay_count * replay_evidence / (replay_evidence_sum + eps)
        )
        structural_strength = evidence_sum / ((support * base).sum() + eps)
        # ``normalized`` is the same row-evidence currency consumed by Split
        # and sums to the retained replay count.  Derive the row ESS from it,
        # rather than introducing a second replay-support-weighted definition.
        effective_sample_size = normalized.sum().square() / (
            normalized.square().sum() + eps
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
        effective_evidence: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Turn persistent Bank modes into one binary Split proposal.

        ``mode_ids``/``support`` describe persistent Bank discovery.  When
        ``effective_evidence`` is supplied, it is the mode-identity evidence
        used for binary grouping.  Production passes persistent support here,
        not bounded replay coverage, so a mode cannot disappear from H1 merely
        because its current replay window was not selected.  The optional
        argument remains for direct legacy callers.
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
        if effective_evidence is not None and effective_evidence.shape != mode_ids.shape:
            raise ValueError("effective_evidence must have shape [R]")
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

        # Direct callers without an explicit identity-evidence vector use
        # persistent support.  ``split_mass`` is retained as provenance and
        # demand telemetry, but never controls Split mode existence.
        structural_mode_weights = mode_support

        if effective_evidence is None:
            mode_evidence = structural_mode_weights
        else:
            effective_evidence = effective_evidence.to(
                device=law_keys.device,
                dtype=law_keys.dtype,
            )
            if not bool(torch.isfinite(effective_evidence).all()):
                raise FloatingPointError("effective_evidence must be finite")
            mode_evidence = law_keys.new_zeros(mode_count)
            mode_evidence.index_add_(
                0,
                inverse,
                effective_evidence.clamp_min(0.0),
            )
        effective_mode_mask = mode_evidence > eps
        effective_mode_count = effective_mode_mask.sum().to(dtype=torch.long)
        # Mode weights are persistent identity evidence.  Current replay
        # coverage is checked separately by ``_bank_h1_evaluability``.
        mode_weights = mode_evidence

        # Weighted two-center clustering runs only on mode-level law identities
        # with E_m > eps.  Residual rows never enter this proposal.
        active_modes = torch.nonzero(
            effective_mode_mask, as_tuple=False
        ).flatten()
        active_count = int(active_modes.numel())
        # ``-1`` is a real third state: the mode is not supported by the
        # current physical replay evidence and therefore has no justified
        # child assignment.  Only effective modes are assigned to child 0/1.
        mode_group_ids = torch.full(
            (mode_count,),
            -1,
            device=law_keys.device,
            dtype=torch.long,
        )
        if active_count == 1:
            # A single persistent mode has a valid one-sided q_bank row
            # ``[1, 0]``.  It must not become ``[.5, .5]`` and fabricate a
            # second law in diagnostics or replay likelihoods.
            mode_group_ids[active_modes[0]] = 0
        elif active_count == 2:
            mode_group_ids[active_modes[0]] = 0
            mode_group_ids[active_modes[1]] = 1
        elif active_count > 2:
            points = F.normalize(mode_law_keys.index_select(0, active_modes), dim=-1)
            pairwise_distance = 1.0 - points @ points.transpose(0, 1)
            pairwise_distance = pairwise_distance.masked_fill(
                torch.eye(
                    active_count,
                    device=points.device,
                    dtype=torch.bool,
                ),
                -torch.inf,
            )
            farthest_pair = int(torch.argmax(pairwise_distance).item())
            first = farthest_pair // active_count
            second = farthest_pair % active_count
            centers = points[[first, second]].clone()
            cluster_weights = mode_evidence.index_select(0, active_modes).clamp_min(eps)
            assignments = torch.zeros(
                active_count,
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
                    candidate = torch.argmax(distances[:, occupied])
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
            mode_group_ids[active_modes] = assignments

        # q_bank still needs a valid binary probability row for likelihood
        # evaluation.  Inactive modes receive a neutral row, but their
        # tri-state assignment remains -1 in ``group_ids`` and is respected
        # by commit_split().
        mode_group_weights = law_keys.new_full((mode_count, 2), 0.5)
        active_group_ids = mode_group_ids >= 0
        if bool(active_group_ids.any()):
            mode_group_weights[active_group_ids] = F.one_hot(
                mode_group_ids[active_group_ids],
                num_classes=2,
            ).to(dtype=law_keys.dtype)

        row_group_weights = mode_group_weights.index_select(0, inverse)
        summary = {
            "mode_ids": unique_modes.detach(),
            "law_keys": mode_law_keys.detach(),
            "support": mode_support.detach(),
            "split_mass": mode_split_mass.detach(),
            "delta_means": mode_delta_means.detach(),
            "mode_weights": mode_weights.detach(),
            "group_ids": mode_group_ids.detach(),
            "mode_evidence": mode_evidence.detach(),
            "effective_evidence": mode_evidence.detach(),
            "effective_mode_mask": effective_mode_mask.detach(),
            "effective_mode_count": effective_mode_count.detach(),
        }
        return row_group_weights.detach(), summary

    @staticmethod
    def _mode_stratified_topk(
        scores: torch.Tensor,
        bank_group_weights: Optional[torch.Tensor],
        max_items: int,
        min_per_group: int = 2,
        eps: float = 1e-8,
        row_group_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Reserve replay representatives from both Bank groups, then Top-K.

        Persistent mode identity decides whether both groups exist.  A
        representative row does not need positive/large structural weight:
        weight and ESS are numerical diagnostics, not a second structural
        gate.  ``None`` means the bounded replay cannot retain one row for
        each group.
        """
        if scores.ndim != 1:
            raise ValueError("scores must be one-dimensional")
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        if min_per_group < 0:
            raise ValueError("min_per_group must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("scores must be finite")
        count = int(scores.numel())
        if bank_group_weights is None:
            if count <= max_items:
                return torch.arange(count, device=scores.device)
            return torch.topk(scores, k=max_items).indices
        if bank_group_weights.shape != (count, 2):
            raise ValueError("bank_group_weights must have shape [R, 2]")
        if not bool(torch.isfinite(bank_group_weights).all()):
            raise FloatingPointError("bank_group_weights must be finite")
        if bool((bank_group_weights < 0.0).any()):
            raise ValueError("bank_group_weights must be non-negative")

        if row_group_ids is None:
            # This fallback is retained for direct callers.  Production Bank
            # batches pass explicit tri-state mode assignments below so a
            # neutral unresolved q row cannot masquerade as group 0.
            assignments = bank_group_weights.argmax(dim=-1)
        else:
            if row_group_ids.shape != (count,):
                raise ValueError("row_group_ids must have shape [R]")
            assignments = row_group_ids.to(
                device=scores.device,
                dtype=torch.long,
            )
            if bool(((assignments < -1) | (assignments > 1)).any()):
                raise ValueError("row_group_ids must contain only -1, 0, or 1")
        rows_by_group = [
            assignments == group for group in range(2)
        ]
        if any(not bool(mask.any()) for mask in rows_by_group):
            return None
        if count <= max_items:
            return torch.arange(count, device=scores.device)

        ranked_by_group = []
        for group in range(2):
            group_indices = torch.nonzero(
                rows_by_group[group], as_tuple=False
            ).flatten()
            order = torch.argsort(
                scores.index_select(0, group_indices), descending=True
            )
            ranked_by_group.append(group_indices.index_select(0, order))

        if max_items < 2:
            return None
        reserved: list[int] = []
        # Round-robin reservation prevents the first side from consuming a
        # small budget before the second side receives a representative.
        # At least one row per side is mandatory even when callers set
        # min_per_group=0.
        reserve_per_group = max(1, min_per_group)
        for rank in range(reserve_per_group):
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
        selected = torch.nonzero(selected_mask, as_tuple=False).flatten()
        selected_assignments = assignments.index_select(0, selected)
        if any(
            not bool((selected_assignments == group).any())
            for group in range(2)
        ):
            return None
        return selected

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
                * sample_support.clamp_min(0.0),
                bank_group_weights,
                int(max_items),
                min_per_group=min_replay_per_group,
            )
            if selected is None:
                return None
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
        indices: Optional[Sequence[int] | torch.Tensor] = None,
    ) -> Optional[SplitBatch]:
        """Build a bounded evidence batch from one leaf memory bank."""
        if max_items is not None and max_items <= 0:
            raise ValueError("max_items must be positive when provided")
        if indices is not None and max_items is not None:
            raise ValueError(
                "max_items cannot be combined with exact replay indices"
            )
        bank._ensure_prototype_state()
        count = len(bank)
        if count == 0:
            return None
        bank_evidence = SplitModule.compute_bank_row_evidence(
            bank,
            tau_age=tau_age,
        )
        evidence_weights_all = bank_evidence["evidence_weights"]
        if not bool(torch.isfinite(evidence_weights_all).all()):
            raise FloatingPointError("Bank evidence weights must be finite")
        valid_indices = [
            index
            for index, window in enumerate(bank.windows[:count])
            if window is not None
        ]
        if not valid_indices:
            return None
        # Structural mode identity comes from persistent Bank support.  Do
        # not pass ``evidence_weights_all`` here: that vector is masked by
        # retained replay windows and would make a persistent law disappear
        # from H1 when bounded replay happens not to cover it this cycle.
        persistent_mode_evidence = bank.support[:count].clamp_min(0.0).detach()
        mode_group_weights_all, mode_summary = (
            SplitModule._build_bank_mode_groups(
                mode_ids=bank.mode_ids[:count],
                law_keys=bank.law_keys[:count],
                deltas=bank.deltas[:count],
                support=bank.support[:count],
                split_mass=bank.split_mass[:count],
                effective_evidence=persistent_mode_evidence,
            )
        )
        if indices is None:
            selected_indices = valid_indices
        elif torch.is_tensor(indices):
            selected_indices = [
                int(index)
                for index in indices.detach().cpu().reshape(-1).tolist()
            ]
        else:
            selected_indices = [int(index) for index in indices]
        if not selected_indices:
            return None
        valid_set = set(valid_indices)
        invalid = sorted(set(selected_indices).difference(valid_set))
        if invalid:
            raise ValueError(
                "exact replay indices must refer to retained physical windows: "
                f"{invalid}"
            )
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("exact replay indices must be unique")
        indices = torch.tensor(
            selected_indices,
            device=bank.device,
            dtype=torch.long,
        )
        # Raw delta_theta determines the dynamics identity.  The Bank row
        # evidence is independent of residual magnitude so semantic rebasing
        # cannot erase a law that still has replay support.
        residuals = bank.deltas[indices]
        contexts = bank.keys[indices]
        mode_ids = bank.mode_ids[indices].detach().to(dtype=torch.long)
        law_keys = bank.law_keys[indices].detach()
        bank_group_weights = mode_group_weights_all.index_select(0, indices)
        base_weights = bank_evidence["base_weights"].index_select(0, indices)
        structural_weights = bank_evidence[
            "structural_weights"
        ].index_select(0, indices)
        sample_support = bank_evidence["sample_support"].index_select(0, indices)
        replay_support = bank_evidence["replay_support"].index_select(0, indices)
        evidence_weights = evidence_weights_all.index_select(0, indices)
        if max_items is not None and indices.numel() > max_items:
            # A bounded binary proposal must reserve a replay representative
            # from both persistent sides.  This is the minimal numerical
            # requirement for evaluating H1; row evidence magnitude is not a
            # structural gate.
            mode_to_group = {
                int(mode): int(group)
                for mode, group in zip(
                    mode_summary["mode_ids"].detach().cpu().tolist(),
                    mode_summary["group_ids"].detach().cpu().tolist(),
                )
            }
            selected_row_groups = torch.tensor(
                [
                    mode_to_group.get(int(mode), -1)
                    for mode in mode_ids.detach().cpu().tolist()
                ],
                device=mode_ids.device,
                dtype=torch.long,
            )
            selected = SplitModule._mode_stratified_topk(
                evidence_weights,
                bank_group_weights,
                int(max_items),
                min_per_group=min_replay_per_group,
                row_group_ids=selected_row_groups,
            )
            if selected is None:
                return None
            indices = indices.index_select(0, selected)
            residuals = residuals.index_select(0, selected)
            contexts = contexts.index_select(0, selected)
            base_weights = base_weights.index_select(0, selected)
            structural_weights = structural_weights.index_select(0, selected)
            sample_support = sample_support.index_select(0, selected)
            replay_support = replay_support.index_select(0, selected)
            evidence_weights = evidence_weights.index_select(0, selected)
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
    def bank_structural_split_probe(
        self,
        *,
        theta_sem: torch.Tensor | HawkesParams | Any,
        batch: SplitBatch,
        hawkes_ll: Optional[HawkesFamily] = None,
    ) -> Optional[Dict[str, Any]]:
        """Test a Bank-mode Split before Light Sleep is allowed to run.

        The null hypothesis is the current leaf semantic parameter
        ``H0: theta_ell``.  The alternative forms two child parameters from
        the same frozen replay rows by taking the Bank-prior-weighted residual
        means, then evaluates the Bank-conditioned child mixture.  No Light
        proposal, direction, or reliability score participates in this test.

        This is deliberately the same Bank H1 reduction used by
        :meth:`compute_replay_loglikelihood` and the production Split
        counterfactual, so a positive pre-Light result is a reusable frozen
        structural hypothesis rather than a second, Light-specific gate.
        """
        if batch.bank_group_weights is None or batch.mode_summary is None:
            return None
        persistent_mode_count = self._bank_effective_mode_count(batch)
        if persistent_mode_count is not None and persistent_mode_count < 2:
            return None
        if persistent_mode_count is None and int(
            batch.mode_summary["mode_ids"].numel()
        ) < 2:
            return None

        hawkes_ll = self._resolve_hawkes_ll(hawkes_ll)
        self.validate_batch(batch)
        reference = self._module_reference(batch.residuals)
        theta = self._theta_sem_to_flat(theta_sem, reference).detach().clone()
        hypothesis = self.build_bank_split_hypothesis(
            batch=batch,
            theta_h0=theta,
        )
        frozen = hypothesis.batch
        q = hypothesis.q_bank
        h1_diagnostics = self._bank_h1_evaluability(frozen)
        if not h1_diagnostics["h1_defined"]:
            return None

        replay_weights = hypothesis.replay_weights
        total_weight = replay_weights.sum()
        if not bool(torch.isfinite(total_weight)) or bool(
            total_weight <= self.eps
        ):
            return None

        replay_cache = build_replay_batch_cache(
            frozen.windows,
            hawkes_ll,
            theta,
        )

        child_evidence = self.compute_child_evidence(
            weights=replay_weights,
            bank_group_weights=q,
            residuals=frozen.residuals,
            eps=self.eps,
        )
        self._assert_bank_mode_support_invariant(
            frozen,
            bool(child_evidence["two_sided_support"]),
            stage="bank_structural_split_probe",
        )
        # Reuse the exact prior stored in the hypothesis; child_evidence is
        # used here only for mass/ESS diagnostics.
        q = hypothesis.q_bank
        group_mass = child_evidence["child_mass"]
        group_delta = hypothesis.delta_bar
        theta_h1 = hypothesis.theta_child_init

        probe_models = torch.cat((theta[None, :], theta_h1), dim=0)
        probe_nll = batched_replay_log_likelihood(
            replay_cache,
            probe_models,
            hawkes_ll,
            normalize_by_events=True,
        )
        loss_h0 = (
            replay_weights * probe_nll[:, 0]
        ).sum() / total_weight
        bank_h1 = self.evaluate_bank_h1(
            q,
            -probe_nll[:, 1:3],
            eps=self.eps,
        )
        loss_h1 = (
            replay_weights * -bank_h1["logp_child_bank"]
        ).sum() / total_weight
        advantage = loss_h0 - loss_h1
        replay_probability = replay_weights / total_weight
        replay_ess = 1.0 / replay_probability.square().sum().clamp_min(
            self.eps
        )

        return {
            "hypothesis": hypothesis,
            "theta_sem_snapshot": hypothesis.theta_h0.detach().clone(),
            "theta_h0": hypothesis.theta_h0.detach().clone(),
            "theta_h1": theta_h1.detach().clone(),
            "group_delta": group_delta.detach().clone(),
            "bank_group_weights": q.detach().clone(),
            "q_bank": hypothesis.q_bank.detach().clone(),
            "batch": hypothesis.batch,
            "replay_weights": replay_weights.detach().clone(),
            "replay_ess": float(replay_ess.item()),
            "replay_rows": int(frozen.residuals.size(0)),
            "child_effective_mass": group_mass.detach().clone(),
            "child_ess": child_evidence["child_ess"].detach().clone(),
            "two_sided_support": bool(child_evidence["two_sided_support"]),
            "h1_defined": True,
            "h1_representative_counts": h1_diagnostics[
                "h1_representative_counts"
            ].detach().clone(),
            "has_bank_prior": True,
            "q_bank_mass_L": group_mass[0].detach().clone(),
            "q_bank_mass_R": group_mass[1].detach().clone(),
            "invalid_proposal": False,
            "K_effective_mode": hypothesis.mode_summary.get(
                "effective_mode_count",
                torch.tensor(0, device=group_mass.device),
            ).detach().clone(),
            "mode_ids": hypothesis.mode_ids.detach().clone(),
            "mode_group_ids": hypothesis.mode_group_ids.detach().clone(),
            "loss_h0": float(loss_h0.item()),
            "loss_h1": float(loss_h1.item()),
            "advantage": float(advantage.item()),
            "protect": bool(advantage > 0.0),
        }

    @torch.no_grad()
    def bank_mode_counterfactual_probe(
        self,
        *,
        theta_sem: torch.Tensor | HawkesParams | Any,
        batch: SplitBatch,
        hawkes_ll: Optional[HawkesFamily] = None,
        light_proposal: Any,
    ) -> Optional[Dict[str, Any]]:
        """Compare shared Light absorption with Bank-mode specialization.

        ``light_proposal`` must be produced by
        :func:`Sleep.Light.propose_light_absorption`.  Split owns only H1 and
        the comparison statistic; it never reconstructs H0 or a second Light
        weighting rule.  Both hypotheses are evaluated on the same frozen
        semantic parameter and the same physical replay rows.
        """
        if batch.bank_group_weights is None or batch.mode_summary is None:
            return None
        persistent_mode_count = self._bank_effective_mode_count(batch)
        if persistent_mode_count is not None and persistent_mode_count < 2:
            return None
        if persistent_mode_count is None and int(
            batch.mode_summary["mode_ids"].numel()
        ) < 2:
            return None
        selected_direction = getattr(light_proposal, "selected_direction", None)
        if selected_direction is None:
            return None

        hawkes_ll = self._resolve_hawkes_ll(hawkes_ll)
        self.validate_batch(batch)
        reference = self._module_reference(batch.residuals)
        theta = self._theta_sem_to_flat(theta_sem, reference).detach().clone()
        hypothesis = self.build_bank_split_hypothesis(
            batch=batch,
            theta_h0=theta,
        )
        frozen = hypothesis.batch
        q = hypothesis.q_bank
        h1_diagnostics = self._bank_h1_evaluability(frozen)
        if not h1_diagnostics["h1_defined"]:
            return None

        light_replay_indices = getattr(light_proposal, "replay_indices", None)
        if light_replay_indices is None:
            return None
        if int(light_replay_indices.numel()) != int(frozen.residuals.size(0)):
            raise ValueError(
                "probe batch must contain exactly Light proposal replay rows"
            )

        # H0 is the semantic candidate that real Light would commit.  It is
        # intentionally not recomputed from the probe weights.
        theta_h0 = torch.as_tensor(
            light_proposal.theta_h0,
            device=theta.device,
            dtype=theta.dtype,
        ).detach().clone()
        shared_delta = torch.as_tensor(
            light_proposal.shared_delta,
            device=theta.device,
            dtype=theta.dtype,
        ).detach().clone()
        alpha_light = float(light_proposal.alpha)
        light_replay_weights = torch.as_tensor(
            light_proposal.replay_weights,
            device=theta.device,
            dtype=theta.dtype,
        ).reshape(-1)
        if light_replay_weights.shape != frozen.weights.shape:
            raise ValueError(
                "Light proposal replay weights must align with probe rows"
            )

        # Compatibility path for callers that still compare against a Light
        # proposal.  Use the frozen Split row evidence directly; do not
        # reconstruct the retired residual-strength/reliability product.
        reliability = hypothesis.replay_weights
        total_weight = reliability.sum()
        if not bool(torch.isfinite(total_weight)) or bool(total_weight <= self.eps):
            return None

        replay_cache = build_replay_batch_cache(
            frozen.windows, hawkes_ll, theta
        )

        child_evidence = self.compute_child_evidence(
            weights=reliability,
            bank_group_weights=q,
            residuals=frozen.residuals,
            eps=self.eps,
        )
        self._assert_bank_mode_support_invariant(
            frozen,
            bool(child_evidence["two_sided_support"]),
            stage="bank_mode_counterfactual_probe",
        )
        group_mass = child_evidence["child_mass"]
        group_delta = hypothesis.delta_bar
        theta_h1 = hypothesis.theta_child_init

        probe_models = torch.cat((theta_h0[None, :], theta_h1), dim=0)
        probe_nll = batched_replay_log_likelihood(
            replay_cache,
            probe_models,
            hawkes_ll,
            normalize_by_events=True,
        )
        loss_h0 = (reliability * probe_nll[:, 0]).sum() / total_weight
        bank_h1 = self.evaluate_bank_h1(
            q,
            -probe_nll[:, 1:3],
            eps=self.eps,
        )
        loss_h1 = (
            reliability * -bank_h1["logp_child_bank"]
        ).sum() / total_weight
        advantage = loss_h0 - loss_h1
        probability = reliability / total_weight
        replay_ess = 1.0 / probability.square().sum().clamp_min(self.eps)

        return {
            "hypothesis": hypothesis,
            "theta_sem_snapshot": theta.detach(),
            "batch": hypothesis.batch,
            "mode_ids": hypothesis.mode_ids.detach().clone(),
            "mode_group_ids": hypothesis.mode_group_ids.detach().clone(),
            "bank_group_weights": q.detach(),
            "shared_delta": shared_delta.detach(),
            "group_delta": group_delta.detach(),
            "theta_h0": theta_h0.detach(),
            "theta_h1": theta_h1.detach(),
            "alpha_light": float(alpha_light),
            "coherence": float(selected_direction.coherence.item()),
            "direction_gain": float(selected_direction.gain.item()),
            "full_step_gain": float(selected_direction.gain.item()),
            "selected_direction_indices": selected_direction.indices.detach().clone(),
            "light_replay_indices": light_replay_indices.detach().clone(),
            "light_replay_weights": light_replay_weights.detach().clone(),
            "loss_h0": float(loss_h0.item()),
            "loss_h1": float(loss_h1.item()),
            "advantage": float(advantage.item()),
            "protect": bool(advantage > 0.0),
            "replay_ess": float(replay_ess.item()),
            "replay_rows": int(frozen.residuals.size(0)),
            "child_effective_mass": group_mass.detach().clone(),
            "child_ess": child_evidence["child_ess"].detach().clone(),
            "two_sided_support": bool(child_evidence["two_sided_support"]),
            "h1_defined": True,
            "h1_representative_counts": h1_diagnostics[
                "h1_representative_counts"
            ].detach().clone(),
            "has_bank_prior": True,
            "q_bank_mass_L": group_mass[0].detach().clone(),
            "q_bank_mass_R": group_mass[1].detach().clone(),
            "invalid_proposal": False,
            "K_effective_mode": hypothesis.mode_summary.get(
                "effective_mode_count",
                torch.tensor(0, device=group_mass.device),
            ).detach().clone(),
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
        two_sided_tensor = torch.as_tensor(
            out.get("two_sided_support", True)
        ).detach()
        two_sided_support = bool(
            two_sided_tensor.numel() == 1
            and bool(torch.isfinite(two_sided_tensor).all())
            and bool(two_sided_tensor.bool().all())
        )
        invalid_tensor = torch.as_tensor(
            out.get("invalid_proposal", False)
        ).detach()
        invalid_flag_is_valid = bool(
            invalid_tensor.numel() == 1
            and bool(torch.isfinite(invalid_tensor).all())
        )
        invalid_proposal = bool(
            invalid_flag_is_valid and bool(invalid_tensor.bool().item())
        )

        has_bank_prior = bool(
            out.get("has_bank_prior", out.get("q_bank") is not None)
        )
        h1_default = two_sided_support
        h1_tensor = torch.as_tensor(
            out.get("h1_defined", h1_default)
        ).detach()
        h1_flag_is_valid = bool(
            h1_tensor.numel() == 1
            and bool(torch.isfinite(h1_tensor).all())
        )
        h1_defined = bool(
            h1_flag_is_valid and bool(h1_tensor.bool().item())
        )

        # m_min, structural strength and proposal ESS are retained only as
        # diagnostics/backward-compatible arguments.  Bank admission owns
        # persistence; the predictive objective owns the decision boundary.
        # Bank persistence decides whether the two laws exist.  The only
        # structural/evaluation gate here is ``h1_defined`` (one retained
        # representative replay window per persistent group).  ESS and mass
        # remain finite-value diagnostics, not admission thresholds.
        child_ess_finite = (
            True
            if has_bank_prior
            else bool(torch.isfinite(N_eff).all())
        )
        eligible = bool(
            bool(torch.isfinite(N_mass).all())
            and child_ess_finite
            and math.isfinite(structural_strength)
            and math.isfinite(effective_sample_size)
            and h1_flag_is_valid
            and h1_defined
            and invalid_flag_is_valid
            and not invalid_proposal
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
        lambda_route: float | torch.Tensor = 0.0,
        hypothesis: Optional[FrozenBankSplitHypothesis] = None,
        lambda_anchor: float | torch.Tensor = 0.0,
    ) -> Dict[str, Any]:
        if hypothesis is not None:
            if batch.bank_group_weights is None:
                raise ValueError(
                    "a frozen Bank hypothesis requires a Bank-backed batch"
                )
            return self.fit_bank_split(
                hypothesis=hypothesis,
                hawkes_ll=hawkes_ll,
                num_steps=num_steps,
                lr=lr,
                commit_state=commit_state,
                m_min=m_min,
                min_structural_strength=min_structural_strength,
                min_effective_sample_size=min_effective_sample_size,
                lambda_T=lambda_T,
                delta_complexity=delta_complexity,
                lambda_anchor=lambda_anchor,
                lambda_route=lambda_route,
            )
        if batch.bank_group_weights is not None:
            # Fallback callers that have a Bank batch but did not run the
            # explicit Pre-Light probe still use the same canonical path.
            hypothesis = self.build_bank_split_hypothesis(
                batch=batch,
                theta_h0=theta_sem,
            )
            return self.fit_bank_split(
                hypothesis=hypothesis,
                hawkes_ll=hawkes_ll,
                num_steps=num_steps,
                lr=lr,
                commit_state=commit_state,
                m_min=m_min,
                min_structural_strength=min_structural_strength,
                min_effective_sample_size=min_effective_sample_size,
                lambda_T=lambda_T,
                delta_complexity=delta_complexity,
                lambda_anchor=lambda_anchor,
                lambda_route=lambda_route,
            )
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
                    lambda_route=lambda_route,
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
                    lambda_route=lambda_route,
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

    def fit_bank_split(
        self,
        hypothesis: FrozenBankSplitHypothesis,
        hawkes_ll: Optional[HawkesFamily] = None,
        num_steps: int = 200,
        lr: float = 1e-3,
        commit_state: Optional[SplitCommitState] = None,
        m_min: Optional[float] = None,
        min_structural_strength: float = 0.0,
        min_effective_sample_size: float = 0.0,
        lambda_T: Optional[float | torch.Tensor] = None,
        delta_complexity: float | torch.Tensor = 1.0,
        lambda_anchor: float | torch.Tensor = 1e-2,
        lambda_route: float | torch.Tensor = 0.0,
    ) -> Dict[str, Any]:
        """Refine one frozen Bank hypothesis without changing its identity.

        The only fitted law parameters are

        ``theta_child = theta_child_init + epsilon``.

        ``theta_h0``, ``q_bank``, ``delta_bar`` and the replay windows all come
        from the frozen hypothesis.  ``lambda_T`` and ``delta_complexity`` are
        accepted for the shared Split API but intentionally do not enter this
        differentiable Bank fit; topology pricing belongs to the unified
        selector after H1 evaluation.
        """
        del lambda_T, delta_complexity, m_min
        if not isinstance(hypothesis, FrozenBankSplitHypothesis):
            raise TypeError(
                "hypothesis must be a FrozenBankSplitHypothesis"
            )
        if hypothesis.batch is None:
            raise ValueError(
                "hypothesis.batch is required for Deep Bank fitting"
            )
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")

        hawkes_ll = self._resolve_hawkes_ll(hawkes_ll)
        module_reference = self._module_reference(hypothesis.batch.residuals)
        theta_h0 = hypothesis.theta_h0.detach().to(
            device=module_reference.device,
            dtype=module_reference.dtype,
        ).clone()
        theta_child_init = hypothesis.theta_child_init.detach().to(
            device=module_reference.device,
            dtype=module_reference.dtype,
        ).clone()
        if theta_h0.shape != (self.P,):
            raise ValueError("hypothesis.theta_h0 has the wrong shape")
        if theta_child_init.shape != (2, self.P):
            raise ValueError(
                "hypothesis.theta_child_init must have shape [2, P]"
            )
        if hypothesis.q_bank.shape != (
            hypothesis.batch.residuals.size(0),
            2,
        ):
            raise ValueError("hypothesis.q_bank must align with hypothesis.batch")
        if len(hypothesis.windows) != len(hypothesis.batch.windows):
            raise ValueError(
                "hypothesis.windows must align with hypothesis.batch.windows"
            )

        batch = self._batch_to_device(hypothesis.batch, theta_h0)
        q_bank = hypothesis.q_bank.to(
            device=theta_h0.device,
            dtype=theta_h0.dtype,
        ).detach().clone()
        q_bank = q_bank / q_bank.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        replay_weights = hypothesis.replay_weights.to(
            device=theta_h0.device,
            dtype=theta_h0.dtype,
        ).detach().clamp_min(0.0)
        if replay_weights.shape != batch.weights.shape:
            raise ValueError(
                "hypothesis.replay_weights must align with hypothesis.batch"
            )
        # The frozen object, rather than a fresh batch normalization, owns
        # both the q_bank identity and the replay measure used by fitting.
        batch = replace(
            batch,
            weights=replay_weights,
            bank_group_weights=q_bank,
        )
        self.validate_batch(batch)

        child_theta = nn.Parameter(theta_child_init.clone())
        route_lambda_value = float(
            torch.as_tensor(lambda_route).detach().cpu().item()
        )
        if route_lambda_value < 0.0 or not math.isfinite(route_lambda_value):
            raise ValueError("lambda_route must be finite and non-negative")
        trainables: list[nn.Parameter] = [child_theta]
        if route_lambda_value > 0.0:
            trainables.extend(list(self.router.parameters()))
        optimizer = torch.optim.Adam(trainables, lr=lr)
        replay_cache = build_replay_batch_cache(
            hypothesis.windows,
            hawkes_ll,
            theta_h0,
        )

        was_training = self.training
        self.train()
        last_loss = None
        try:
            for _ in range(num_steps):
                optimizer.zero_grad(set_to_none=True)
                replay = self.compute_replay_loglikelihood(
                    batch=batch,
                    theta_plus=theta_h0,
                    theta_cand=child_theta,
                    hawkes_ll=hawkes_ll,
                    replay_cache=replay_cache,
                )
                anchor_loss = (
                    child_theta - theta_child_init
                ).square().sum()
                objective = self.compute_bank_split_training_objective(
                    weights=replay_weights,
                    ell_h1=replay["logp_child_bank"],
                    anchor_loss=anchor_loss,
                    lambda_anchor=lambda_anchor,
                    route_loss=replay["route_loss"],
                    lambda_route=lambda_route,
                )
                loss = objective["loss"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        "Bank Split optimization produced a non-finite loss"
                    )
                loss.backward()
                optimizer.step()
                last_loss = loss.detach()

            with torch.no_grad():
                optimized_replay = self.compute_replay_loglikelihood(
                    batch=batch,
                    theta_plus=theta_h0,
                    theta_cand=child_theta,
                    hawkes_ll=hawkes_ll,
                    replay_cache=replay_cache,
                )
                init_replay = self.compute_replay_loglikelihood(
                    batch=batch,
                    theta_plus=theta_h0,
                    theta_cand=theta_child_init,
                    hawkes_ll=hawkes_ll,
                    replay_cache=replay_cache,
                )
        finally:
            if not was_training:
                self.eval()

        total_weight = replay_weights.sum().clamp_min(self.eps)
        init_h1_loss = -(
            replay_weights * init_replay["logp_child_bank"]
        ).sum() / total_weight
        optimized_h1_loss = -(
            replay_weights * optimized_replay["logp_child_bank"]
        ).sum() / total_weight
        if not bool(torch.isfinite(init_h1_loss)):
            raise FloatingPointError(
                "frozen Bank H1 evaluation produced a non-finite loss"
            )
        use_bank_init = (
            not bool(torch.isfinite(optimized_h1_loss))
            or bool(optimized_h1_loss > init_h1_loss)
        )
        chosen_replay = init_replay if use_bank_init else optimized_replay
        chosen_theta = (
            theta_child_init if use_bank_init else child_theta.detach()
        )
        final_h1_loss = (
            init_h1_loss if use_bank_init else optimized_h1_loss
        )

        child_evidence = self.compute_child_evidence(
            weights=replay_weights,
            bank_group_weights=q_bank,
            residuals=batch.residuals,
            eps=self.eps,
        )
        self._assert_bank_mode_support_invariant(
            batch,
            bool(child_evidence["two_sided_support"]),
            stage="fit_bank_split",
        )
        h1_diagnostics = self._bank_h1_evaluability(batch)
        delta_bar = hypothesis.delta_bar.to(
            device=theta_h0.device,
            dtype=theta_h0.dtype,
        ).detach().clone()
        zero = theta_h0.new_zeros(())
        structural = {
            "dist": theta_h0.new_zeros(batch.residuals.size(0), 2),
            "q": q_bank,
            "N_mass": child_evidence["child_mass"],
            "N_eff": child_evidence["child_ess"],
            "N": child_evidence["child_mass"],
            "row_evidence": child_evidence["row_evidence"],
            "child_evidence": child_evidence["child_evidence"],
            "child_effective_mass": child_evidence["child_mass"],
            "child_ess": child_evidence["child_ess"],
            "two_sided_support": child_evidence["two_sided_support"],
            "invalid_proposal": False,
            "pi_bar": child_evidence["pi_bar"],
            "delta_bar": delta_bar,
            "delta_shared": zero.new_zeros(self.P),
            "delta_child": delta_bar,
            "alpha": zero + 1.0,
            "rho_sp": zero + 1.0,
            "canonical_bank_hypothesis": True,
            "theta_plus": theta_h0,
            "theta_cand": chosen_theta,
        }
        anchor_loss = (
            chosen_theta - theta_child_init
        ).square().sum()
        objective = self.compute_bank_split_training_objective(
            weights=replay_weights,
            ell_h1=chosen_replay["logp_child_bank"],
            anchor_loss=anchor_loss,
            lambda_anchor=lambda_anchor,
            route_loss=chosen_replay["route_loss"],
            lambda_route=lambda_route,
        )

        out: Dict[str, Any] = {}
        out.update(structural)
        out.update(chosen_replay)
        out.update(objective)
        out.update({
            "hypothesis": hypothesis,
            "has_bank_prior": True,
            "theta_bank": theta_child_init,
            "theta_child_init": theta_child_init,
            "h1_defined": bool(h1_diagnostics["h1_defined"]),
            "h1_representative_counts": h1_diagnostics[
                "h1_representative_counts"
            ].detach().clone(),
            "q_bank_mass": child_evidence["child_mass"].detach().clone(),
            "q_bank_mass_L": child_evidence["child_mass"][0],
            "q_bank_mass_R": child_evidence["child_mass"][1],
            "invalid_proposal": not bool(h1_diagnostics["h1_defined"]),
            "deep_fit_rolled_back": use_bank_init,
            "bank_h1_loss_init": init_h1_loss.detach(),
            "bank_h1_loss_optimized": optimized_h1_loss.detach(),
            "bank_h1_loss_final": final_h1_loss.detach(),
            "lambda_T": zero,
            "delta_tree_complexity": zero,
            "expected_delta_tree_complexity": zero,
        })
        effective_mode_count = self._bank_effective_mode_count(batch)
        if effective_mode_count is not None:
            out["K_effective_mode"] = torch.as_tensor(
                effective_mode_count,
                device=theta_h0.device,
                dtype=theta_h0.dtype,
            )
        if batch.mode_summary is not None:
            out["bank_mode_summary"] = {
                key: value.detach().clone()
                for key, value in batch.mode_summary.items()
            }
        out["weighted_nll"] = objective["prediction_loss"]
        out["tree_penalty"] = objective["complexity_penalty"]
        out["structural_strength"] = (
            replay_weights.new_ones(())
            if batch.structural_strength is None
            else batch.structural_strength.reshape(())
        )
        out["effective_sample_size"] = (
            replay_weights.sum().square()
            / (replay_weights.square().sum() + self.eps)
            if batch.effective_sample_size is None
            else batch.effective_sample_size.reshape(())
        )

        final_out = self.detach_split_output(out)
        commit_state = self.commit_state if commit_state is None else commit_state
        eligible = self.evaluate_split_eligibility(
            out=final_out,
            commit_state=commit_state,
            m_min=0.0,
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
        lambda_route: float | torch.Tensor = 0.0,
        lambda_anchor: float | torch.Tensor = 1e-2,
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
            if batch is None:
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
                lambda_route=lambda_route,
                lambda_anchor=lambda_anchor,
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
    For Bank-backed proposals, effective source-bank modes receive a child
    assignment.  Inactive/unresolved modes carry the explicit assignment
    ``-1`` and remain in the parent Bank.  Replay likelihood and the context
    router are intentionally not part of this topology write: Bank identity
    is determined by persistent mode grouping.  Payloads without a Bank prior
    retain a conservative legacy fallback for old checkpoints and direct
    callers.
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
    source_bank._ensure_prototype_state()
    source_mode_ids = source_bank.mode_ids[: len(source_bank)].detach().to(
        dtype=torch.long
    )
    # Reconstruct the structural partition from persistent support when a
    # caller supplies an older payload without the frozen mode summary.  The
    # current replay-window mask is for likelihood refinement only and must
    # not decide whether a Bank law exists.
    persistent_mode_evidence = source_bank.support[: len(source_bank)].clamp_min(
        0.0
    ).detach()
    bank_group_weights_all, bank_mode_summary = (
        SplitModule._build_bank_mode_groups(
            mode_ids=source_bank.mode_ids[: len(source_bank)],
            law_keys=source_bank.law_keys[: len(source_bank)],
            deltas=source_bank.deltas[: len(source_bank)],
            support=source_bank.support[: len(source_bank)],
            split_mass=source_bank.split_mass[: len(source_bank)],
            effective_evidence=persistent_mode_evidence,
        )
        if len(source_bank) > 0
        else (
            source_bank.keys.new_empty((0, 2)),
            {
                "mode_ids": source_bank.mode_ids.new_empty(0),
                "group_ids": source_bank.mode_ids.new_empty(0),
            },
        )
    )
    has_bank_prior_value = split_output.get("has_bank_prior")
    if has_bank_prior_value is None:
        # ``q_bank`` is present on all new Split outputs, including a legacy
        # direct SplitBatch fallback.  The explicit flag above is preferred,
        # while this inference keeps hand-built new payloads usable.
        has_bank_prior = any(
            torch.is_tensor(split_output.get(key))
            for key in ("q_bank", "logp_child_bank")
        ) or isinstance(split_output.get("bank_mode_summary"), Mapping)
    else:
        has_bank_prior = bool(has_bank_prior_value)

    summary_partition_used = False
    stored_mode_summary = split_output.get("bank_mode_summary")
    if (
        has_bank_prior
        and len(source_bank) > 0
        and isinstance(stored_mode_summary, Mapping)
    ):
        summary_mode_ids = stored_mode_summary.get("mode_ids")
        summary_group_ids = stored_mode_summary.get("group_ids")
        if (
            torch.is_tensor(summary_mode_ids)
            and torch.is_tensor(summary_group_ids)
            and summary_mode_ids.ndim == 1
            and summary_group_ids.shape == summary_mode_ids.shape
            and summary_mode_ids.numel() > 0
        ):
            summary_modes = summary_mode_ids.detach().cpu().tolist()
            summary_groups = summary_group_ids.detach().cpu().tolist()
            mode_to_group = {
                int(mode): int(group)
                for mode, group in zip(summary_modes, summary_groups)
            }
            source_mode_values = source_mode_ids.cpu().tolist()
            summary_is_complete = (
                len(mode_to_group) == len(summary_modes)
                and all(group in (-1, 0, 1) for group in mode_to_group.values())
                and all(mode in mode_to_group for mode in source_mode_values)
            )
            if summary_is_complete:
                summary_assignments = torch.tensor(
                    [mode_to_group[mode] for mode in source_mode_values],
                    dtype=torch.long,
                    device=source_bank.device,
                )
                # Keep inactive modes neutral for q_bank likelihood shape,
                # while preserving their explicit -1 assignment for commit.
                bank_group_weights_all = source_bank.keys.new_full(
                    (len(source_bank), 2),
                    0.5,
                )
                active_summary = summary_assignments >= 0
                if bool(active_summary.any()):
                    bank_group_weights_all[active_summary] = F.one_hot(
                        summary_assignments[active_summary],
                        num_classes=2,
                    ).to(dtype=source_bank.keys.dtype)
                bank_mode_summary = {
                    key: value.detach().clone()
                    for key, value in stored_mode_summary.items()
                    if torch.is_tensor(value)
                }
                summary_partition_used = True
    if has_bank_prior and not summary_partition_used:
        full_q_bank = split_output.get("q_bank")
        if torch.is_tensor(full_q_bank) and full_q_bank.shape == (
            len(source_bank),
            2,
        ):
            full_q_bank = full_q_bank.detach().to(
                device=source_bank.device,
                dtype=source_bank.keys.dtype,
            )
            full_q_mass = full_q_bank.sum(dim=-1, keepdim=True)
            if (
                bool(torch.isfinite(full_q_bank).all())
                and bool((full_q_bank >= 0.0).all())
                and bool((full_q_mass > 1e-8).all())
            ):
                bank_group_weights_all = full_q_bank / full_q_mass
    valid_memory_indices = [
        index
        for index, window in enumerate(source_bank.windows[: len(source_bank)])
        if window is not None
    ]

    if has_bank_prior:
        # This assignment is formed from the complete source Bank before any
        # replay/proposal budget is applied.  It is therefore also available
        # for rows without windows and for proposal batches that were
        # truncated.  Unlike q_bank.argmax(), this preserves the third
        # ``-1`` state for modes with no effective replay evidence.
        summary_mode_ids = bank_mode_summary.get("mode_ids")
        summary_group_ids = bank_mode_summary.get("group_ids")
        if (
            torch.is_tensor(summary_mode_ids)
            and torch.is_tensor(summary_group_ids)
            and summary_mode_ids.ndim == 1
            and summary_group_ids.shape == summary_mode_ids.shape
            and summary_mode_ids.numel() > 0
        ):
            mode_to_group = {
                int(mode): int(group)
                for mode, group in zip(
                    summary_mode_ids.detach().cpu().tolist(),
                    summary_group_ids.detach().cpu().tolist(),
                )
                if int(group) in (-1, 0, 1)
            }
            if all(int(mode) in mode_to_group for mode in source_mode_ids.cpu().tolist()):
                child_assignments = torch.tensor(
                    [mode_to_group[int(mode)] for mode in source_mode_ids.cpu().tolist()],
                    dtype=torch.long,
                    device=source_bank.device,
                )
            else:
                child_assignments = bank_group_weights_all.argmax(dim=-1).to(
                    dtype=torch.long
                ).clone()
                effective_rows = persistent_mode_evidence > 1e-8
                child_assignments[~effective_rows] = -1
        else:
            child_assignments = bank_group_weights_all.argmax(dim=-1).to(
                dtype=torch.long
            ).clone()
            effective_rows = persistent_mode_evidence > 1e-8
            child_assignments[~effective_rows] = -1
        parent_memory_mask = child_assignments < 0
    else:
        # Old direct commit payloads did not carry the persistent Bank prior.
        # Keep their previous conservative parent-retention behavior without
        # reintroducing the context router into the new Bank-backed path.
        child_assignments = torch.full(
            (len(source_bank),), -1, dtype=torch.long, device=source_bank.device
        )
        parent_memory_mask = torch.ones(
            len(source_bank), dtype=torch.bool, device=source_bank.device
        )

    if valid_memory_indices and not has_bank_prior:
        valid_count = len(valid_memory_indices)
        # This branch is unreachable for Bank-backed assignments: their
        # structural output is already fixed by persistent mode grouping.
        logp_children = None
        # Legacy payloads score only the rows that were explicitly supplied.
        # The router is deliberately not consulted here; a missing/truncated
        # score leaves that row inherited by the parent.
        logp_parent = split_output.get("logp0")
        logp_children = split_output.get("logp_child_each")
        if (
            torch.is_tensor(logp_parent)
            and torch.is_tensor(logp_children)
            and logp_parent.shape == (valid_count,)
            and logp_children.shape == (valid_count, 2)
        ):
            logp_parent = logp_parent.to(source_bank.device)
            logp_children = logp_children.to(source_bank.device)
            best_child = logp_children.argmax(dim=-1)
            best_child_logp = logp_children.gather(
                1, best_child[:, None]
            ).squeeze(1)
            finite_scores = torch.isfinite(logp_parent) & torch.isfinite(
                logp_children
            ).all(dim=-1)
            parent_is_uniquely_better = (
                logp_parent - best_child_logp > memory_hard_threshold
            )
            move_to_child = finite_scores & ~parent_is_uniquely_better
            valid_indices = torch.tensor(
                valid_memory_indices,
                dtype=torch.long,
                device=source_bank.device,
            )
            moving_indices = valid_indices[move_to_child]
            parent_memory_mask[moving_indices] = False
            child_assignments[moving_indices] = best_child[move_to_child]

    if isinstance(split_output, dict):
        split_output["bank_mode_summary"] = {
            key: value.detach().clone()
            for key, value in bank_mode_summary.items()
        }
        split_output["bank_child_assignments"] = child_assignments.detach().clone()

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

    # The complete source Bank has already been assigned above. Rebase every
    # child group because the split writes new semantic parameters at every
    # affected node; the inherited parent bank is cleared by this partition.
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
