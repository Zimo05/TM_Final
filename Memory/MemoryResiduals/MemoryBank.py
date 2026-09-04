import math
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

@dataclass
class EventWindow:
    times: Tensor        # [L]
    types: Tensor        # [L]
    node_id: str
    start_idx: int
    end_idx: int
    # New writes store the full prefix through ``end_idx`` so replay can
    # condition events [start_idx:end_idx] on their true Hawkes history.
    has_full_history: bool = False
    T: Optional[Tensor] = None
    # Parameter-independent caches. Optional fields preserve compatibility
    # with checkpoints written before sequence caching was introduced.
    event_time_features: Optional[Tensor] = None
    hawkes_history_stats: Optional[Tensor] = None
    hawkes_interval_stats: Optional[Tensor] = None
    hawkes_cache_signature: Optional[Tuple] = None


@dataclass
class MemoryItem:
    key: Tensor              # [d_k]
    delta_theta: Tensor      # [param_dim]
    window: Optional[EventWindow]
    usage: float = 0.0
    age: int = 0
    write_quality: float = 1.0
    queue_weight: float = 0.0
    # Number of independent persistent observations represented by this
    # physical prototype row. Retrieval usage remains a separate statistic.
    support: float = 1.0
    # Signed local prediction improvement used only to confirm a genuinely
    # new dynamics mode. It is deliberately independent of bounded quality.
    prediction_gain: float = 0.0


def _signed_hash_projection(feature: Tensor, output_dim: int) -> Tensor:
    """Deterministically project a physical-law feature without parameters."""
    if feature.ndim == 0:
        raise ValueError("feature must have at least one dimension")
    feature_dim = feature.size(-1)
    if feature_dim <= 0:
        return feature.new_empty(*feature.shape[:-1], output_dim)
    flat = feature.reshape(-1, feature_dim)
    positions = torch.arange(feature_dim, device=feature.device, dtype=torch.long)
    buckets = torch.remainder(positions * 2654435761, output_dim)
    signs = torch.where(
        torch.remainder(positions * 2246822519 + 3266489917, 2) == 0,
        flat.new_ones(()),
        -flat.new_ones(()),
    )
    projected = flat.new_zeros(flat.size(0), output_dim)
    projected.scatter_add_(
        1,
        buckets.reshape(1, -1).expand(flat.size(0), -1),
        signs.reshape(1, -1) * flat,
    )
    projected = F.normalize(projected, dim=-1)
    return projected.reshape(*feature.shape[:-1], output_dim)


def effective_hawkes_law_key(
    semantic_theta: Tensor,
    delta_theta: Tensor,
    decays: Tensor,
    *,
    num_event_types: int,
    num_basis: int,
    key_dim: int,
) -> Tensor:
    """Return a compact identity of the effective physical Hawkes law.

    The key is derived from ``semantic_theta + delta_theta`` after mapping the
    unconstrained baseline and kernels to physical parameters.  Consequently
    a Light-Sleep semantic rebase that preserves the effective law also
    preserves this identity.  Basis components are integrated before the
    physical-law feature is projected to ``key_dim``.
    """
    expected = num_event_types + num_event_types * num_event_types * num_basis
    semantic = semantic_theta.detach()
    delta = delta_theta.detach()
    if semantic.ndim == 0 or delta.ndim == 0:
        raise ValueError("semantic_theta and delta_theta need a final parameter dimension")
    if semantic.shape[-1] != expected or delta.shape[-1] != expected:
        raise ValueError(f"effective Hawkes theta must contain {expected} values")
    try:
        batch_shape = torch.broadcast_shapes(
            semantic.shape[:-1], delta.shape[:-1]
        )
    except RuntimeError as error:
        raise ValueError(
            "semantic_theta and delta_theta batch dimensions are incompatible"
        ) from error
    semantic = semantic.expand(*batch_shape, expected)
    delta = delta.expand(*batch_shape, expected)
    effective = semantic + delta
    decay = decays.detach().reshape(-1).to(effective)
    if decay.numel() != num_basis or bool((decay <= 0).any()):
        raise ValueError("decays must contain one positive value per basis")
    raw_mu = effective[..., :num_event_types]
    raw_w = effective[..., num_event_types:].reshape(
        *batch_shape, num_event_types, num_event_types, num_basis
    )
    mu = F.softplus(raw_mu)
    integrated_excitation = (
        F.softplus(raw_w)
        / decay.reshape(*((1,) * len(batch_shape)), 1, 1, -1)
    ).sum(-1)
    law_feature = torch.cat(
        [mu, integrated_excitation.reshape(*batch_shape, -1)], dim=-1
    )
    return _signed_hash_projection(law_feature, key_dim)


@dataclass
class TreeMemoryRead:
    delta_theta: Tensor
    delta_by_node: Dict[str, Tensor]
    info_by_node: Dict[str, Dict[str, Tensor]]


@dataclass
class HawkesMemoryUpdate:
    mu: Tensor
    W: Tensor
    delta_theta: Tensor
    read: TreeMemoryRead


@dataclass
class EffectiveHawkesParameters:
    """Hawkes parameters after semantic, episodic, and working-memory fusion."""

    theta: Tensor
    raw_mu: Tensor
    raw_W: Tensor
    mu: Tensor
    W: Tensor
    decays: Optional[Tensor] = None

    def select(self, index: int) -> "EffectiveHawkesParameters":
        """Select one sample from a batched effective-parameter result."""
        if self.theta.ndim == 1:
            if index not in (0, -1):
                raise IndexError("unbatched effective parameters only contain one sample")
            return self
        return EffectiveHawkesParameters(
            theta=self.theta[index],
            raw_mu=self.raw_mu[index],
            raw_W=self.raw_W[index],
            mu=self.mu[index],
            W=self.W[index],
            decays=self.decays,
        )


class UpdateHawkesParameter:
    """
    theta = (mu, W)
    mu: [D]
    W:  [D, D, M]
    delta_theta: [D + D * D * M]

    ``num_basis=1`` is backward-compatible with a two-dimensional excitation
    matrix [D, D].
    """
    def __init__(self, num_event_types: int, num_basis: int = 1):
        if num_event_types <= 0:
            raise ValueError("num_event_types must be positive")
        if num_basis <= 0:
            raise ValueError("num_basis must be positive")

        self.D = num_event_types
        self.M = num_basis
        self.param_dim = self.D + self.D * self.D * self.M

    def unpack_delta(self, delta_theta: Tensor) -> Tuple[Tensor, Tensor]:
        if delta_theta.numel() != self.param_dim:
            raise ValueError(
                f"delta_theta must contain {self.param_dim} values, "
                f"got {delta_theta.numel()}"
            )

        delta_mu = delta_theta[:self.D]
        delta_W = delta_theta[self.D:].reshape(self.D, self.D, self.M)
        return delta_mu, delta_W

    def pack_leaf_parameters(
        self,
        semantic_mu: Tensor,
        semantic_W: Tensor,
    ) -> Tensor:
        """
        Pack unconstrained semantic parameters along their final dimensions.

        Supported shapes:
            semantic_mu: [..., L, D]
            semantic_W:  [..., L, D, D, M]
        """
        if semantic_mu.ndim < 2 or semantic_mu.shape[-1] != self.D:
            raise ValueError(
                f"semantic_mu must end with [L, {self.D}], "
                f"got {tuple(semantic_mu.shape)}"
            )
        expected_w_tail = (self.D, self.D, self.M)
        if semantic_W.ndim < 4 or semantic_W.shape[-3:] != expected_w_tail:
            raise ValueError(
                f"semantic_W must end with [L, {self.D}, {self.D}, {self.M}], "
                f"got {tuple(semantic_W.shape)}"
            )
        if semantic_mu.shape[:-1] != semantic_W.shape[:-3]:
            raise ValueError(
                "semantic_mu and semantic_W must have identical batch/leaf dimensions"
            )

        flat_W = semantic_W.reshape(*semantic_W.shape[:-3], -1)
        return torch.cat([semantic_mu, flat_W], dim=-1)

    def compose_effective_theta(
        self,
        semantic_theta: Tensor,
        episodic_delta: Tensor,
        routing_weights: Tensor,
        working_delta: Tensor,
    ) -> Tensor:
        r"""
        Compose the paper's final unconstrained parameters:

            theta_eff = sum_l r_l (theta_sem_l + delta_epi_l) + delta_wm.

        ``routing_weights`` may have shape [L] or [B, L]. Semantic and
        episodic tensors may have shape [L, P] or [B, L, P], where
        P = D + D * D * M. A shared [L, P] tensor broadcasts across a batch.
        ``working_delta`` may have shape [P] or [B, P].
        """
        if routing_weights.ndim not in (1, 2):
            raise ValueError(
                "routing_weights must have shape [L] or [B, L], "
                f"got {tuple(routing_weights.shape)}"
            )
        if semantic_theta.shape[-1:] != (self.param_dim,):
            raise ValueError(
                f"semantic_theta must have final dimension {self.param_dim}, "
                f"got {tuple(semantic_theta.shape)}"
            )
        if episodic_delta.shape[-1:] != (self.param_dim,):
            raise ValueError(
                f"episodic_delta must have final dimension {self.param_dim}, "
                f"got {tuple(episodic_delta.shape)}"
            )

        try:
            leaf_theta = semantic_theta + episodic_delta
        except RuntimeError as error:
            raise ValueError(
                "semantic_theta and episodic_delta are not broadcast-compatible"
            ) from error

        expected_leaf_shape = (*routing_weights.shape, self.param_dim)
        if leaf_theta.shape != expected_leaf_shape:
            raise ValueError(
                "semantic/episodic leaf dimensions must match routing_weights: "
                f"expected {expected_leaf_shape}, got {tuple(leaf_theta.shape)}"
            )

        theta_eff = (
            routing_weights.unsqueeze(-1) * leaf_theta
        ).sum(dim=-2)

        expected_working_shapes = {
            (self.param_dim,),
            tuple(theta_eff.shape),
        }
        if tuple(working_delta.shape) not in expected_working_shapes:
            raise ValueError(
                "working_delta must be shared [P] or match the effective batch "
                f"shape; got {tuple(working_delta.shape)}"
            )

        return theta_eff + working_delta

    def compose_effective_parameters(
        self,
        semantic_mu: Tensor,
        semantic_W: Tensor,
        episodic_delta: Tensor,
        routing_weights: Tensor,
        working_delta: Tensor,
        decays: Optional[Tensor] = None,
    ) -> EffectiveHawkesParameters:
        """
        Fuse all three memory timescales in unconstrained space, then constrain.

        ``semantic_mu`` and ``semantic_W`` are the per-leaf unconstrained
        parameters produced by the semantic tree.
        """
        semantic_theta = self.pack_leaf_parameters(semantic_mu, semantic_W)
        theta_eff = self.compose_effective_theta(
            semantic_theta=semantic_theta,
            episodic_delta=episodic_delta,
            routing_weights=routing_weights,
            working_delta=working_delta,
        )

        raw_mu = theta_eff[..., :self.D]
        raw_W = theta_eff[..., self.D:].reshape(
            *theta_eff.shape[:-1],
            self.D,
            self.D,
            self.M,
        )
        return EffectiveHawkesParameters(
            theta=theta_eff,
            raw_mu=raw_mu,
            raw_W=raw_W,
            mu=F.softplus(raw_mu),
            W=F.softplus(raw_W),
            decays=decays,
        )

    def apply_delta(
        self,
        raw_mu: Tensor,
        raw_W: Tensor,
        delta_theta: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply residual in unconstrained space, then use softplus to keep
        Hawkes intensity parameters positive.
        """
        expected_w_shape = (self.D, self.D, self.M)
        if self.M == 1 and raw_W.shape == (self.D, self.D):
            delta_W_shape = (self.D, self.D)
        elif raw_W.shape != expected_w_shape:
            raise ValueError(
                f"raw_W must have shape {expected_w_shape}, got {tuple(raw_W.shape)}"
            )
        else:
            delta_W_shape = expected_w_shape
        if raw_mu.shape != (self.D,):
            raise ValueError(
                f"raw_mu must have shape {(self.D,)}, got {tuple(raw_mu.shape)}"
            )

        delta_mu, delta_W = self.unpack_delta(delta_theta)
        delta_W = delta_W.reshape(delta_W_shape)
        mu_eff = F.softplus(raw_mu + delta_mu)
        W_eff = F.softplus(raw_W + delta_W)

        return mu_eff, W_eff

class MemoryQueryNet(nn.Module):
    def __init__(self, input_dim: int, key_dim: int, output_dim: Optional[int] = None):
        super().__init__()
        output_dim = key_dim if output_dim is None else output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, key_dim),
            nn.SiLU(),
            nn.Linear(key_dim, output_dim)
        )

    def forward(self, z_t: Tensor) -> Tensor:
        q_t = self.net(z_t)
        q_t = F.normalize(q_t, dim=-1)

        return q_t
    

class _Entmax15Function(torch.autograd.Function):
    """Exact 1.5-entmax forward with its closed-form stable Jacobian.

    Differentiating the threshold-search implementation directly is unsafe:
    inactive support candidates commonly have ``delta == 0``. Autograd then
    evaluates the derivative of ``sqrt(0)`` even though those candidates
    receive zero upstream gradient, producing ``0 * inf == NaN``. The
    closed-form Jacobian depends only on the final probabilities and avoids
    differentiating through sort, support selection, clamp, or sqrt.
    """

    @staticmethod
    def forward(ctx, logits: Tensor, eps: float) -> Tensor:
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("entmax15 logits contain NaN or Inf")

        # Entmax is translation invariant. Shifting limits intermediate
        # magnitudes without changing the output distribution.
        x = (logits - logits.max()) / 2.0
        xs = torch.sort(x, descending=True).values
        rho = torch.arange(
            1,
            xs.numel() + 1,
            device=xs.device,
            dtype=xs.dtype,
        )
        mean = torch.cumsum(xs, dim=0) / rho
        mean_sq = torch.cumsum(xs.square(), dim=0) / rho
        ss = rho * (mean_sq - mean.square())
        delta = ((1.0 - ss) / rho).clamp_min(0.0)
        taus = mean - delta.sqrt()
        support_size = (taus <= xs).sum().clamp_min(1)
        tau_star = taus[support_size - 1]
        probabilities = (x - tau_star).clamp_min(0.0).square()
        probabilities = probabilities / probabilities.sum().clamp_min(eps)
        if not bool(torch.isfinite(probabilities).all()):
            raise FloatingPointError("entmax15 forward produced NaN or Inf")

        ctx.save_for_backward(probabilities)
        ctx.eps = float(eps)
        return probabilities

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if not bool(torch.isfinite(grad_output).all()):
            raise FloatingPointError(
                "entmax15 backward received a NaN/Inf upstream gradient"
            )
        (probabilities,) = ctx.saved_tensors
        # For alpha=1.5, p^(2-alpha) = sqrt(p). The vector-Jacobian
        # product is g * (dL/dp - weighted_mean(dL/dp)).
        gppr = probabilities.clamp_min(0.0).sqrt()
        weighted = grad_output * gppr
        correction = weighted.sum() / gppr.sum().clamp_min(ctx.eps)
        grad_logits = weighted - correction * gppr
        if not bool(torch.isfinite(grad_logits).all()):
            raise FloatingPointError("entmax15 backward produced NaN or Inf")
        return grad_logits, None


class _MaskedEntmax15Function(torch.autograd.Function):
    """Batched exact 1.5-entmax with independently masked rows.

    Padding is excluded from threshold search, normalization, and the
    Jacobian. Each row is therefore equivalent to calling ``entmax15_1d`` on
    only that row's valid entries.
    """

    @staticmethod
    def forward(ctx, logits: Tensor, valid_mask: Tensor, eps: float) -> Tensor:
        if logits.ndim != 2:
            raise ValueError(
                f"masked entmax expects [R, M], got {tuple(logits.shape)}"
            )
        if valid_mask.shape != logits.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape [R, M]")
        # CUDA value checks force a device-to-host synchronization. Batched
        # retrieval constructs one guaranteed-valid row for every slot, so
        # retain eager validation only for CPU/debug execution.
        if (
            logits.device.type == "cpu"
            and not bool(valid_mask.any(dim=-1).all())
        ):
            raise ValueError("every masked-entmax row needs one valid entry")
        if (
            logits.device.type == "cpu"
            and not bool(torch.isfinite(logits[valid_mask]).all())
        ):
            raise FloatingPointError("entmax15 logits contain NaN or Inf")

        masked_logits = logits.masked_fill(~valid_mask, -torch.inf)
        row_max = masked_logits.max(dim=-1, keepdim=True).values
        x = (logits - row_max) / 2.0

        # Put padding strictly below every valid value without introducing
        # infinities into the cumulative sums used by the threshold search.
        valid_min = x.masked_fill(~valid_mask, torch.inf).min(
            dim=-1, keepdim=True
        ).values
        sortable_x = torch.where(valid_mask, x, valid_min - 1.0)
        xs = torch.sort(sortable_x, dim=-1, descending=True).values

        width = logits.size(-1)
        rho = torch.arange(
            1,
            width + 1,
            device=logits.device,
            dtype=logits.dtype,
        ).unsqueeze(0)
        mean = torch.cumsum(xs, dim=-1) / rho
        mean_sq = torch.cumsum(xs.square(), dim=-1) / rho
        ss = rho * (mean_sq - mean.square())
        delta = ((1.0 - ss) / rho).clamp_min(0.0)
        taus = mean - delta.sqrt()

        positions = torch.arange(
            1, width + 1, device=logits.device
        ).unsqueeze(0)
        valid_count = valid_mask.sum(dim=-1, keepdim=True)
        in_valid_prefix = positions <= valid_count
        support_size = ((taus <= xs) & in_valid_prefix).sum(
            dim=-1
        ).clamp_min(1)
        tau_star = taus.gather(
            1, (support_size - 1).unsqueeze(-1)
        )
        probabilities = (
            (x - tau_star).clamp_min(0.0).square()
            * valid_mask.to(logits.dtype)
        )
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
        if (
            probabilities.device.type == "cpu"
            and not bool(torch.isfinite(probabilities).all())
        ):
            raise FloatingPointError("entmax15 forward produced NaN or Inf")

        ctx.save_for_backward(probabilities)
        ctx.eps = float(eps)
        return probabilities

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if (
            grad_output.device.type == "cpu"
            and not bool(torch.isfinite(grad_output).all())
        ):
            raise FloatingPointError(
                "entmax15 backward received a NaN/Inf upstream gradient"
            )
        (probabilities,) = ctx.saved_tensors
        gppr = probabilities.clamp_min(0.0).sqrt()
        weighted = grad_output * gppr
        correction = weighted.sum(dim=-1, keepdim=True) / gppr.sum(
            dim=-1, keepdim=True
        ).clamp_min(ctx.eps)
        grad_logits = weighted - correction * gppr
        if (
            grad_logits.device.type == "cpu"
            and not bool(torch.isfinite(grad_logits).all())
        ):
            raise FloatingPointError("entmax15 backward produced NaN or Inf")
        return grad_logits, None, None


def entmax15_1d(logits: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Entmax with alpha = 1.5 for a 1D tensor.

    Input:
        logits: [M]

    Output:
        probs: [M], sparse probability distribution.
    """
    if logits.ndim != 1:
        raise ValueError(f"entmax15_1d expects [M], got {tuple(logits.shape)}")
    if logits.numel() == 0:
        return logits
    return _Entmax15Function.apply(logits, eps)


def entmax15_masked(
    logits: Tensor,
    valid_mask: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """Apply independent exact 1.5-entmax distributions to padded rows."""
    return _MaskedEntmax15Function.apply(logits, valid_mask, eps)


class SmoothSparseRetriever(nn.Module):
    """
    Differentiable Sparse Episodic Retrieval.

    Replaces hard TopK with entmax-based sparse gating.
    """

    def __init__(
        self,
        init_gamma: float = 10.0,
        init_tau: float = 1.0,
        init_lambda_usage: float = 0.1,
        init_lambda_age: float = 0.01,
        dense_gradient_mass: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()

        if not 0.0 < dense_gradient_mass < 1.0:
            raise ValueError("dense_gradient_mass must be in (0, 1)")
        self.eps = eps
        self.dense_gradient_mass = float(dense_gradient_mass)
        self.raw_gamma = nn.Parameter(self._inverse_softplus(init_gamma))
        self.raw_tau = nn.Parameter(self._inverse_softplus(init_tau))
        self.raw_lambda_usage = nn.Parameter(
            self._inverse_softplus(init_lambda_usage)
        )
        self.raw_lambda_age = nn.Parameter(self._inverse_softplus(init_lambda_age))

    def _inverse_softplus(self, value: float) -> Tensor:
        target = torch.tensor(float(value) - self.eps)
        if target.item() <= 0:
            raise ValueError(f"positive parameter initial value must exceed {self.eps}")
        return target + torch.log(-torch.expm1(-target))

    def positive(self, raw: Tensor) -> Tensor:
        return F.softplus(raw) + self.eps

    def forward(
        self,
        query: Tensor,       # [d_k]
        keys: Tensor,        # [M, d_k] or [M, K_ctx, d_k]
        deltas: Tensor,      # [M, param_dim]
        usage: Tensor,       # [M]
        age: Tensor,         # [M]
        write_quality: Optional[Tensor] = None,  # [M]
        keep_gate: Optional[Tensor] = None,      # [M]
        null_logit: Optional[float | Tensor] = None,
        context_valid: Optional[Tensor] = None,  # [M, K_ctx]
    ):
        if keys.ndim not in (2, 3):
            raise ValueError("keys must have shape [M, d_k] or [M, K_ctx, d_k]")
        if keys.shape[0] == 0:
            param_dim = deltas.shape[-1]
            return query.new_zeros(param_dim), {}

        if keys.ndim == 2:
            if context_valid is not None:
                raise ValueError(
                    "context_valid is only supported with alias keys [M, K_ctx, d_k]"
                )
        else:
            if context_valid is None:
                context_valid = torch.ones(
                    keys.shape[:2], device=keys.device, dtype=torch.bool
                )
            if context_valid.shape != keys.shape[:2]:
                raise ValueError("context_valid must have shape [M, K_ctx]")
            if context_valid.dtype != torch.bool:
                raise ValueError("context_valid must be boolean")
            context_valid = context_valid.to(device=keys.device)
            if not bool(context_valid.any(dim=-1).all()):
                raise ValueError("each memory row needs at least one valid context alias")

        if query.numel() != keys.shape[-1]:
            raise ValueError(
                f"query/key dimensions differ: {query.numel()} != {keys.shape[-1]}"
            )
        memory_size = keys.shape[0]
        if deltas.shape[0] != memory_size:
            raise ValueError("keys and deltas must have the same number of entries")
        if usage.shape != (memory_size,) or age.shape != (memory_size,):
            raise ValueError("usage and age must both have shape [M]")
        if write_quality is None:
            write_quality = deltas.new_ones(memory_size)
        if write_quality.shape != (memory_size,):
            raise ValueError("write_quality must have shape [M]")
        if keep_gate is not None:
            if keep_gate.shape != (memory_size,):
                raise ValueError("keep_gate must have shape [M]")
            keep_gate = keep_gate.to(device=deltas.device, dtype=deltas.dtype)
            if null_logit is None:
                raise ValueError("null_logit is required with keep_gate")

        gamma = self.positive(self.raw_gamma)
        tau = self.positive(self.raw_tau)
        lambda_usage = self.positive(self.raw_lambda_usage)
        lambda_age = self.positive(self.raw_lambda_age)

        query = F.normalize(query.reshape(-1), dim=0)
        keys = F.normalize(keys, dim=-1)

        if keys.ndim == 3:
            alias_sim = torch.einsum("mkd,d->mk", keys, query)
            alias_any = context_valid.any(dim=-1)
            sim = alias_sim.masked_fill(~context_valid, -torch.inf).max(dim=-1).values
            # Keep padded/malformed rows finite before multiplying by the
            # trainable gamma.  The row mask still excludes them from all
            # probability mass; finiteness avoids 0 * (-inf) NaN gradients.
            sim = torch.where(alias_any, sim, torch.zeros_like(sim))
        else:
            sim = keys @ query                      # [M]
        attn_logits = gamma * sim               # [M]

        sparse_scores = (
            attn_logits
            - lambda_usage * torch.log1p(usage)
            - lambda_age * age
        )

        scaled_scores = sparse_scores / tau
        sparse_rho = entmax15_1d(scaled_scores)  # [M], sparse forward gate
        # Entmax may collapse to a single exact support.  In that state its
        # Jacobian, and the normalized one-item attention Jacobian, are both
        # zero, permanently starving the query/retriever of learning signal.
        # A small dense path preserves the sparse inductive bias while keeping
        # every retrieval decision differentiable from the prediction loss.
        dense_rho = F.softmax(scaled_scores, dim=0)
        rho = (
            (1.0 - self.dense_gradient_mass) * sparse_rho
            + self.dense_gradient_mass * dense_rho
        )

        # Stable implementation of the gated attention in the formulation:
        # alpha_r = rho_r exp(gamma sim_r)
        #           / (sum_j rho_j exp(gamma sim_j) + epsilon)
        if keep_gate is None:
            shifted_logits = attn_logits - attn_logits.max().detach()
            numerator = rho * torch.exp(shifted_logits)
            denominator = numerator.sum() + self.eps
            null_alpha = denominator.new_zeros(())
        else:
            null = torch.as_tensor(
                null_logit,
                device=attn_logits.device,
                dtype=attn_logits.dtype,
            ).reshape(())
            shift = torch.maximum(attn_logits.max(), null).detach()
            numerator = (
                rho
                * torch.exp(attn_logits - shift)
                * keep_gate.clamp(0.0, 1.0)
            )
            null_weight = torch.exp(null - shift)
            denominator = numerator.sum() + null_weight + self.eps
            null_alpha = null_weight / denominator

        alpha = numerator / denominator

        # Fold scalar row quality into the attention weights first.  Multiplying
        # ``deltas`` directly would materialize another [M, param_dim] tensor.
        delta_epi = (alpha * write_quality) @ deltas  # [param_dim]

        info = {
            "sim": sim.detach(),
            "rho": rho.detach(),
            "sparse_rho": sparse_rho.detach(),
            "dense_rho": dense_rho.detach(),
            "alpha": alpha.detach(),
            "null_alpha": null_alpha.detach(),
            "gamma": gamma.detach(),
            "tau": tau.detach(),
            "lambda_usage": lambda_usage.detach(),
            "lambda_age": lambda_age.detach(),
            "effective_k": (sparse_rho > 1e-6).sum().detach(),
        }

        return delta_epi, info

    def forward_batched(
        self,
        query: Tensor,       # [R, d_k]
        keys: Tensor,        # [R, M, d_k] or [R, M, K_ctx, d_k]
        deltas: Tensor,      # [R, M, P] or shared [B, M, P]
        usage: Tensor,       # [R, M]
        age: Tensor,         # [R, M]
        valid_mask: Tensor,  # [R, M]
        write_quality: Optional[Tensor] = None,  # [R, M]
        keep_gate: Optional[Tensor] = None,      # [R, M]
        null_logit: Optional[float | Tensor] = None,
        row_bank_indices: Optional[Tensor] = None,  # [R] into shared B
        context_valid: Optional[Tensor] = None,  # [R, M, K_ctx]
    ):
        """Retrieve from padded banks with one independently normalized row.

        No probability mass can enter padded rows. Apart from parallel
        execution, row ``r`` is mathematically identical to ``forward`` on
        ``keys[r, valid_mask[r]]`` and the corresponding bank tensors.
        """
        if query.ndim != 2:
            raise ValueError("query must have shape [R, d_k]")
        if keys.ndim not in (3, 4) or keys.shape[:2] != valid_mask.shape:
            raise ValueError("keys and valid_mask must align as [R, M, ...]")
        if keys.ndim == 3:
            if context_valid is not None:
                raise ValueError(
                    "context_valid is only supported with alias keys [R, M, K_ctx, d_k]"
                )
        else:
            if context_valid is None:
                context_valid = torch.ones(
                    keys.shape[:3], device=keys.device, dtype=torch.bool
                )
            if context_valid.shape != keys.shape[:3]:
                raise ValueError("context_valid must have shape [R, M, K_ctx]")
            if context_valid.dtype != torch.bool:
                raise ValueError("context_valid must be boolean")
            context_valid = context_valid.to(device=keys.device)
        if row_bank_indices is None:
            if deltas.ndim != 3 or deltas.shape[:2] != valid_mask.shape:
                raise ValueError(
                    "deltas and valid_mask must align as [R, M, ...]"
                )
        else:
            if row_bank_indices.shape != query.shape[:1]:
                raise ValueError("row_bank_indices must have shape [R]")
            if deltas.ndim != 3 or deltas.size(1) != valid_mask.size(1):
                raise ValueError(
                    "shared deltas must have shape [B, M, param_dim]"
                )
            row_bank_indices = row_bank_indices.to(
                device=deltas.device, dtype=torch.long
            )
        if usage.shape != valid_mask.shape or age.shape != valid_mask.shape:
            raise ValueError("usage and age must have shape [R, M]")
        if write_quality is None:
            write_quality = deltas.new_ones(valid_mask.shape)
        if write_quality.shape != valid_mask.shape:
            raise ValueError("write_quality must have shape [R, M]")
        if keep_gate is not None:
            if keep_gate.shape != valid_mask.shape:
                raise ValueError("keep_gate must have shape [R, M]")
            keep_gate = keep_gate.to(device=deltas.device, dtype=deltas.dtype)
            if null_logit is None:
                raise ValueError("null_logit is required with keep_gate")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean")
        if query.shape[0] != keys.shape[0]:
            raise ValueError("query and bank batch dimensions must match")
        if query.shape[-1] != keys.shape[-1]:
            raise ValueError("query/key dimensions must match")
        gamma = self.positive(self.raw_gamma)
        tau = self.positive(self.raw_tau)
        lambda_usage = self.positive(self.raw_lambda_usage)
        lambda_age = self.positive(self.raw_lambda_age)

        query = F.normalize(query, dim=-1)
        keys = F.normalize(keys, dim=-1)
        if keys.ndim == 4:
            alias_sim = torch.einsum("rmkd,rd->rmk", keys, query)
            alias_any = context_valid.any(dim=-1)
            sim = alias_sim.masked_fill(~context_valid, -torch.inf).max(dim=-1).values
            sim = torch.where(alias_any, sim, torch.zeros_like(sim))
        else:
            sim = torch.einsum("rmd,rd->rm", keys, query)
        attn_logits = gamma * sim
        sparse_scores = (
            attn_logits
            - lambda_usage * torch.log1p(usage)
            - lambda_age * age
        )
        scaled_scores = sparse_scores / tau
        sparse_rho = entmax15_masked(
            scaled_scores,
            valid_mask,
        )
        dense_rho = F.softmax(
            scaled_scores.masked_fill(~valid_mask, -torch.inf),
            dim=-1,
        )
        rho = (
            (1.0 - self.dense_gradient_mass) * sparse_rho
            + self.dense_gradient_mass * dense_rho
        )

        row_max = attn_logits.masked_fill(
            ~valid_mask, -torch.inf
        ).max(dim=-1, keepdim=True).values
        if keep_gate is None:
            shift = row_max.detach()
            numerator = (
                rho
                * torch.exp(attn_logits - shift)
                * valid_mask.to(rho.dtype)
            )
            denominator = numerator.sum(dim=-1, keepdim=True) + self.eps
            null_alpha = denominator.new_zeros(denominator.shape[:-1])
        else:
            null = torch.as_tensor(
                null_logit,
                device=attn_logits.device,
                dtype=attn_logits.dtype,
            )
            if null.numel() == 1:
                null = null.reshape(1, 1).expand(attn_logits.size(0), 1)
            elif null.shape == (attn_logits.size(0),):
                null = null.unsqueeze(-1)
            elif null.shape != (attn_logits.size(0), 1):
                raise ValueError(
                    "null_logit must be scalar or contain one value per row"
                )
            shift = torch.maximum(row_max, null).detach()
            numerator = (
                rho
                * torch.exp(attn_logits - shift)
                * keep_gate.clamp(0.0, 1.0)
                * valid_mask.to(rho.dtype)
            )
            null_weight = torch.exp(null - shift)
            denominator = (
                numerator.sum(dim=-1, keepdim=True)
                + null_weight
                + self.eps
            )
            null_alpha = (null_weight / denominator).squeeze(-1)
        alpha = numerator / denominator
        # Do not form ``write_quality[..., None] * deltas`` here.  In packed
        # frontier training [R, M, P] can be several GiB, so that expression
        # doubles the dominant retrieval allocation.  Combining the two
        # [R, M] weights first is algebraically identical and keeps the only
        # new intermediate independent of P.
        weighted_alpha = alpha * write_quality
        if row_bank_indices is None:
            delta_epi = torch.einsum(
                "rm,rmp->rp", weighted_alpha, deltas
            )
        else:
            # Frontier rows repeatedly visit the same small set of tree
            # nodes.  Expanding shared [node, M, P] residual banks to
            # [visit, M, P] can require several GiB.  Aggregate visits by
            # source node instead, so the dominant residual tensor remains
            # shared and peak temporary memory is independent of R * M * P.
            delta_epi = deltas.new_zeros(
                query.size(0), deltas.size(-1)
            )
            for bank_index in torch.unique(
                row_bank_indices
            ).detach().cpu().tolist():
                rows = torch.nonzero(
                    row_bank_indices == int(bank_index),
                    as_tuple=False,
                ).flatten()
                # Chunk the output-side matmul as well: a heavily visited
                # root should not create another full [R, P] temporary.
                for start in range(0, int(rows.numel()), 1024):
                    row_chunk = rows[start : start + 1024]
                    node_delta = (
                        weighted_alpha.index_select(0, row_chunk)
                        @ deltas[int(bank_index)]
                    )
                    delta_epi.index_copy_(0, row_chunk, node_delta)

        info = {
            "sim": sim.detach(),
            "rho": rho.detach(),
            "sparse_rho": sparse_rho.detach(),
            "dense_rho": dense_rho.detach(),
            "alpha": alpha.detach(),
            "null_alpha": null_alpha.detach(),
            "gamma": gamma.detach(),
            "tau": tau.detach(),
            "lambda_usage": lambda_usage.detach(),
            "lambda_age": lambda_age.detach(),
            "effective_k": (sparse_rho > 1e-6).sum(dim=-1).detach(),
            "valid_mask": valid_mask.detach(),
        }
        return delta_epi, info


class MemoryBank:
    def __init__(
        self,
        device: str,
        key_dim: int,
        param_dim: int,
        capacity: int = 128,
        law_dim: Optional[int] = None,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if key_dim <= 0:
            raise ValueError("key_dim must be positive")
        if param_dim <= 0:
            raise ValueError("param_dim must be positive")

        self.key_dim = key_dim
        self.param_dim = param_dim
        # Law identities use the compact signed-hash representation by
        # default.  Keep an explicit ``law_dim`` override for legacy and
        # generic callers that need a different storage width.
        self.law_dim = key_dim if law_dim is None else int(law_dim)
        if self.law_dim <= 0:
            raise ValueError("law_dim must be positive")
        self.capacity = capacity
        self.device = torch.device(device)

        # A physical law prototype can expose several retrieval/context
        # aliases.  ``keys`` is retained as a legacy first-alias view for
        # callers that still inspect the old field; retrieval itself uses the
        # explicit alias tensors below.
        self.context_alias_capacity = 3

        self.keys = torch.empty(0, key_dim, device=self.device)
        self.context_keys = torch.empty(
            0, self.context_alias_capacity, key_dim, device=self.device
        )
        self.context_valid = torch.empty(
            0, self.context_alias_capacity, dtype=torch.bool, device=self.device
        )
        self.context_support = torch.empty(
            0, self.context_alias_capacity, device=self.device
        )
        self.deltas = torch.empty(0, param_dim, device=self.device)
        self.write_quality = torch.empty(0, device=self.device)
        self.queue_weight = torch.empty(0, device=self.device)
        # Prototype/mode state.  ``keys`` remain the retrieval (context)
        # identity; ``law_keys`` are used only for duplicate and dynamics-mode
        # matching.
        self.law_keys = torch.empty(0, self.law_dim, device=self.device)
        self.support = torch.empty(0, device=self.device)
        self.quality_mass = torch.empty(0, device=self.device)
        self.split_mass = torch.empty(0, device=self.device)
        self.mode_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.mode_compressed = torch.empty(0, dtype=torch.bool, device=self.device)
        self._next_mode_id = 0

        self.duplicate_threshold = 0.98
        self.mode_threshold = 0.90
        self.mode_capacity = 12
        self.ema_beta_min = 0.01
        self.ema_beta_max = 0.25
        self.retention_support_weight = 1.0
        self.retention_usage_weight = 0.5
        self.retention_stale_weight = 1.0
        self.retention_age_weight = 0.1
        # Adaptive two-radius matching.  The legacy similarity thresholds are
        # retained only as cold-start priors; once a mode has accumulated
        # enough accepted observations, its own rolling distance statistics
        # determine both radii.
        self.adaptive_history_size = 64
        self.adaptive_min_samples = 18
        # Accepted-sample calibrated duplicate radius.  The lower quantile
        # keeps the duplicate gate from inheriting the upper-tail bias caused
        # by its own admission censoring.
        self.duplicate_quantile = 0.85
        self.mode_quantile = 0.95
        self.radius_margin = 1e-3
        self.gain_quantile = 0.95
        self.gain_ema_decay = 0.8
        self.gain_confirmation_min_count = 2
        self.gain_floor = 0.0
        self._mode_duplicate_distances: Dict[int, List[float]] = {}
        self._mode_distances: Dict[int, List[float]] = {}
        self._mode_normal_gains: Dict[int, List[float]] = {}
        # Law distances for outliers that are temporarily queued while a
        # signed prediction-gain confirmation is collected.  These samples
        # are deliberately kept out of the adaptive radius until a later
        # recurrence confirms that they belong to the current mode.
        self._mode_pending_distances: Dict[int, List[float]] = {}
        self._mode_pending_gain_ema: Dict[int, float] = {}
        self._mode_pending_gain_count: Dict[int, int] = {}
        # Pending new-dynamics confirmation is keyed by both the nearest
        # existing mode and the incoming law identity.  A single mode can
        # therefore hold several unrelated temporary candidates (for example
        # two different outliers that happen to be nearest to the same mode).
        # ``_mode_pending_distances`` and the aggregate gain maps above remain
        # as compatibility summaries for older checkpoints/callers; candidate
        # records are the source of truth for new admissions.
        self._mode_pending_candidates: Dict[int, List[Dict[str, Any]]] = {}

        self.windows: List[Optional[EventWindow]] = []
        self.usage = torch.empty(0, device=self.device)
        # Retrieval mass accumulated during the current wake/sleep cycle.
        # ``usage`` is consolidated as an EMA only once at sleep time.
        self.cycle_usage = torch.empty(0, device=self.device)
        # Number of completed sleep cycles since the last effective retrieval.
        # This is intentionally distinct from chronological ``age``.
        self.stale_cycles = torch.empty(0, device=self.device)
        self.age = torch.empty(0, device=self.device)
        # ``age`` is stored relative to this logical event clock. The owning
        # TreeEpisodicMemory advances one scalar clock per event and only
        # materializes this offset when a bank is mutated or inspected by a
        # sleep-time operation.
        self._age_reference_clock = 0

        # The public tensors below remain compact views so old checkpoints and
        # retrieval code keep their existing shape.  Once the first write is
        # admitted, the views are backed by these fixed-capacity tensors.  A
        # write therefore fills one free row instead of allocating and copying
        # every previous row with ``torch.cat``.  ``_storage_capacity`` may be
        # temporarily larger than ``capacity`` for topology merge operations
        # that intentionally defer pruning.
        self._storage_capacity = 0
        for field in self._tensor_state_fields():
            setattr(self, f"_storage_{field}", None)

    def __len__(self) -> int:
        return self.keys.shape[0]

    @staticmethod
    def _tensor_state_fields() -> Tuple[str, ...]:
        return (
            "keys",
            "context_keys",
            "context_valid",
            "context_support",
            "deltas",
            "write_quality",
            "queue_weight",
            "law_keys",
            "support",
            "quality_mass",
            "split_mass",
            "mode_ids",
            "mode_compressed",
            "usage",
            "cycle_usage",
            "stale_cycles",
            "age",
        )

    def _ensure_storage_metadata(self) -> None:
        """Initialize append-cache fields for pre-batch checkpoints."""
        if not hasattr(self, "_storage_capacity"):
            self._storage_capacity = 0
        for field in self._tensor_state_fields():
            name = f"_storage_{field}"
            if not hasattr(self, name):
                setattr(self, name, None)

    def __getstate__(self):
        # Fixed storage is a rebuildable append cache. Excluding it keeps
        # checkpoints compatible with the compact representation and avoids
        # serializing a second copy of every bank row.
        state = self.__dict__.copy()
        for field in self._tensor_state_fields():
            value = state.get(field)
            if torch.is_tensor(value):
                state[field] = value.clone()
        state.pop("_storage_capacity", None)
        for field in self._tensor_state_fields():
            state.pop(f"_storage_{field}", None)
        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "law_dim"):
            # Banks pickled before law/context identities were separated used
            # ``key_dim`` for both tensors.
            self.law_dim = int(getattr(self, "key_dim", 0))
        self._ensure_storage_metadata()
        # Directly pickled banks bypass ``TreeEpisodicMemory.set_extra_state``.
        # Migrate those legacy ``keys``-only payloads at load time as well, so
        # callers can inspect/use context aliases immediately after
        # ``torch.load`` instead of relying on a later lazy retrieval path.
        self._ensure_prototype_state()

    def _invalidate_fixed_storage(self) -> None:
        """Drop the append backing store after an external tensor replacement."""
        self._ensure_storage_metadata()
        self._storage_capacity = 0
        for field in self._tensor_state_fields():
            setattr(self, f"_storage_{field}", None)

    def _storage_is_bound(self, count: int) -> bool:
        self._ensure_storage_metadata()
        if self._storage_capacity < max(self.capacity, count):
            return False
        for field in self._tensor_state_fields():
            value = getattr(self, field)
            storage = getattr(self, f"_storage_{field}")
            if storage is None:
                return False
            if value.device != storage.device or value.dtype != storage.dtype:
                return False
            if value.shape[1:] != storage.shape[1:]:
                return False
            if count and value.data_ptr() != storage.data_ptr():
                return False
        return True

    def _bind_fixed_storage_views(self, count: Optional[int] = None) -> None:
        if self._storage_capacity <= 0:
            return
        count = len(self) if count is None else int(count)
        for field in self._tensor_state_fields():
            storage = getattr(self, f"_storage_{field}")
            setattr(self, field, storage[:count])

    @torch.no_grad()
    def _ensure_fixed_storage(self) -> None:
        """Make compact public rows addressable from a fixed-capacity store."""
        self._ensure_storage_metadata()
        count = len(self)
        required_capacity = max(self.capacity, count)
        if self._storage_is_bound(count):
            return

        old_values = {
            field: getattr(self, field)
            for field in self._tensor_state_fields()
        }
        for field, value in old_values.items():
            storage_shape = (required_capacity, *value.shape[1:])
            storage = torch.empty(
                storage_shape,
                device=value.device,
                dtype=value.dtype,
            )
            if count:
                storage[:count].copy_(value)
            setattr(self, f"_storage_{field}", storage)
        self._storage_capacity = required_capacity
        self._bind_fixed_storage_views(count)

    @torch.no_grad()
    def _sync_fixed_storage(self) -> None:
        """Copy compact results of keep/merge operations back into the store."""
        self._ensure_storage_metadata()
        if self._storage_capacity <= 0:
            return
        count = len(self)
        if self._storage_capacity < max(self.capacity, count):
            self._invalidate_fixed_storage()
            self._ensure_fixed_storage()
            return
        for field in self._tensor_state_fields():
            storage = getattr(self, f"_storage_{field}")
            storage[:count].copy_(getattr(self, field))
        self._bind_fixed_storage_views(count)

    @torch.no_grad()
    def _append_batch_rows(
        self,
        keys: Tensor,
        deltas: Tensor,
        law_keys: Tensor,
        quality: Tensor,
        queue: Tensor,
        mode_ids: Tensor,
        windows: Sequence[Optional[EventWindow]],
    ) -> Tensor:
        """Append already-classified rows into free fixed-capacity slots."""
        row_count = int(keys.size(0))
        if row_count == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if any(len(value) != row_count for value in (
            deltas, law_keys, quality, queue, mode_ids
        )):
            raise ValueError("batched append fields must have the same row count")
        if len(windows) != row_count:
            raise ValueError("windows must contain one entry per batched row")

        self._ensure_fixed_storage()
        start = len(self)
        end = start + row_count
        if end > self._storage_capacity:
            # This path is only used by deferred topology merges. Ordinary
            # writes call _make_space before reaching the hard bank capacity.
            old_values = {
                field: getattr(self, field)
                for field in self._tensor_state_fields()
            }
            new_capacity = max(self.capacity, end, self._storage_capacity * 2)
            for field, value in old_values.items():
                storage = torch.empty(
                    (new_capacity, *value.shape[1:]),
                    device=value.device,
                    dtype=value.dtype,
                )
                if start:
                    storage[:start].copy_(value)
                setattr(self, f"_storage_{field}", storage)
            self._storage_capacity = new_capacity
            self._bind_fixed_storage_views(start)

        self._storage_keys[start:end].copy_(keys.to(self._storage_keys))
        self._storage_context_keys[start:end].zero_()
        self._storage_context_keys[start:end, 0].copy_(keys.to(self._storage_context_keys))
        self._storage_context_valid[start:end].zero_()
        self._storage_context_valid[start:end, 0] = True
        self._storage_context_support[start:end].zero_()
        self._storage_context_support[start:end, 0] = 1.0
        self._storage_deltas[start:end].copy_(deltas.to(self._storage_deltas))
        self._storage_law_keys[start:end].copy_(law_keys.to(self._storage_law_keys))
        self._storage_write_quality[start:end].copy_(quality.to(self._storage_write_quality))
        self._storage_queue_weight[start:end].copy_(queue.to(self._storage_queue_weight))
        self._storage_support[start:end].fill_(1.0)
        self._storage_quality_mass[start:end].copy_(quality.to(self._storage_quality_mass))
        self._storage_split_mass[start:end].copy_(queue.to(self._storage_split_mass))
        self._storage_mode_ids[start:end].copy_(mode_ids.to(self._storage_mode_ids))
        self._storage_mode_compressed[start:end].zero_()
        self._storage_usage[start:end].zero_()
        self._storage_cycle_usage[start:end].zero_()
        self._storage_stale_cycles[start:end].zero_()
        self._storage_age[start:end].zero_()
        self.windows.extend(windows)
        self._bind_fixed_storage_views(end)
        return torch.arange(start, end, device=self.device, dtype=torch.long)

    def configure_prototype_policy(
        self,
        *,
        duplicate_threshold: float,
        mode_threshold: float,
        mode_capacity: int = 12,
        ema_beta_min: float = 0.01,
        ema_beta_max: float = 0.25,
        retention_support_weight: float = 1.0,
        retention_usage_weight: float = 0.5,
        retention_stale_weight: float = 1.0,
        retention_age_weight: float = 0.1,
        adaptive_history_size: int = 64,
        adaptive_min_samples: int = 18,
        duplicate_quantile: float = 0.85,
        mode_quantile: float = 0.95,
        radius_margin: float = 1e-3,
        gain_quantile: float = 0.95,
        gain_ema_decay: float = 0.8,
        gain_confirmation_min_count: int = 2,
        gain_floor: float = 0.0,
        context_alias_capacity: int = 3,
    ) -> None:
        if not -1.0 <= mode_threshold < duplicate_threshold <= 1.0:
            raise ValueError(
                "prototype thresholds must satisfy -1 <= mode < duplicate <= 1"
            )
        if mode_capacity <= 0:
            raise ValueError("mode_capacity must be positive")
        if not 0.0 < ema_beta_min <= ema_beta_max <= 1.0:
            raise ValueError("EMA bounds must satisfy 0 < min <= max <= 1")
        if min(
            retention_support_weight,
            retention_usage_weight,
            retention_stale_weight,
            retention_age_weight,
        ) < 0.0:
            raise ValueError("retention weights must be non-negative")
        if adaptive_history_size <= 0:
            raise ValueError("adaptive_history_size must be positive")
        if not 1 <= adaptive_min_samples <= adaptive_history_size:
            raise ValueError(
                "adaptive_min_samples must be in [1, adaptive_history_size]"
            )
        if not 0.0 < duplicate_quantile < 1.0:
            raise ValueError("duplicate_quantile must lie in (0, 1)")
        if not 0.0 < mode_quantile < 1.0:
            raise ValueError("mode_quantile must lie in (0, 1)")
        if radius_margin <= 0.0:
            raise ValueError("radius_margin must be positive")
        if not 0.0 < gain_quantile < 1.0:
            raise ValueError("gain_quantile must lie in (0, 1)")
        if not 0.0 <= gain_ema_decay < 1.0:
            raise ValueError("gain_ema_decay must lie in [0, 1)")
        if gain_confirmation_min_count <= 0:
            raise ValueError("gain_confirmation_min_count must be positive")
        if not math.isfinite(gain_floor):
            raise ValueError("gain_floor must be finite")
        if context_alias_capacity <= 0:
            raise ValueError("context_alias_capacity must be positive")
        if (
            len(self) > 0
            and context_alias_capacity != int(
                getattr(self, "context_alias_capacity", 3)
            )
        ):
            raise ValueError(
                "context_alias_capacity cannot change after the bank has entries"
            )
        self.duplicate_threshold = float(duplicate_threshold)
        self.mode_threshold = float(mode_threshold)
        self.mode_capacity = int(mode_capacity)
        self.ema_beta_min = float(ema_beta_min)
        self.ema_beta_max = float(ema_beta_max)
        self.retention_support_weight = float(retention_support_weight)
        self.retention_usage_weight = float(retention_usage_weight)
        self.retention_stale_weight = float(retention_stale_weight)
        self.retention_age_weight = float(retention_age_weight)
        self.adaptive_history_size = int(adaptive_history_size)
        self.adaptive_min_samples = int(adaptive_min_samples)
        self.duplicate_quantile = float(duplicate_quantile)
        self.mode_quantile = float(mode_quantile)
        self.radius_margin = float(radius_margin)
        self.gain_quantile = float(gain_quantile)
        self.gain_ema_decay = float(gain_ema_decay)
        self.gain_confirmation_min_count = int(gain_confirmation_min_count)
        self.gain_floor = float(gain_floor)
        self.context_alias_capacity = int(context_alias_capacity)
        self._trim_adaptive_histories()

    @torch.no_grad()
    def _normalize_law_keys(
        self,
        law_keys: Tensor,
        *,
        batch_size: Optional[int] = None,
        name: str = "law_key",
    ) -> Tensor:
        """Normalize compact law identities and migrate legacy widths.

        New tree banks store the signed-hash physical-law identity in
        ``law_dim`` (normally ``key_dim``). Older callers/checkpoints may
        still provide a retrieval key or an intermediate full-width law
        feature; conversion is kept only for compatibility. Newly computed
        Hawkes identities use the compact width.
        """
        if not torch.is_tensor(law_keys):
            raise ValueError(f"{name} must be a tensor")
        values = law_keys.detach().to(
            device=self.device,
            dtype=self.keys.dtype,
        )
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError(
                f"{name} must have shape [B, {self.law_dim}], "
                f"got {tuple(values.shape)}"
            )
        if batch_size is not None and values.size(0) != int(batch_size):
            raise ValueError(
                f"{name} must have {int(batch_size)} rows, got {values.size(0)}"
            )
        width = int(values.size(-1))
        if width != self.law_dim:
            # ``key_dim`` is the only legacy identity width used by the
            # previous implementation.  ``param_dim`` is accepted as well
            # for intermediate checkpoints that briefly stored raw law
            # features without an explicit ``law_dim`` field.
            if width not in {self.key_dim, self.param_dim}:
                raise ValueError(
                    f"{name} must contain {self.law_dim} values "
                    f"(legacy widths: {self.key_dim}, {self.param_dim}); "
                    f"got {width}"
                )
            migrated = values.new_zeros(values.size(0), self.law_dim)
            copy_width = min(width, self.law_dim)
            if copy_width:
                migrated[:, :copy_width] = values[:, :copy_width]
            values = migrated
        return F.normalize(values, dim=-1)

    @torch.no_grad()
    def _ensure_prototype_state(self) -> None:
        """Repair old checkpoints and topology operations lazily."""
        count = len(self)
        law_dim = int(getattr(self, "law_dim", self.key_dim))
        if law_dim <= 0:
            law_dim = int(self.key_dim)
            self.law_dim = law_dim
        stored_law_keys = getattr(self, "law_keys", None)
        law_keys = None
        if (
            torch.is_tensor(stored_law_keys)
            and stored_law_keys.ndim == 2
            and stored_law_keys.size(0) == count
            and stored_law_keys.size(1) == law_dim
        ):
            law_keys = stored_law_keys.to(
                device=self.keys.device,
                dtype=self.keys.dtype,
            )
        elif torch.is_tensor(stored_law_keys) and stored_law_keys.ndim == 2:
            try:
                migrated_law_keys = self._normalize_law_keys(stored_law_keys)
                if migrated_law_keys.shape == (count, law_dim):
                    law_keys = migrated_law_keys
            except ValueError:
                law_keys = None
        if law_keys is None:
            # A legacy ``keys``-only row has no recoverable physical-law
            # feature.  Keep it usable until the tree-level migration can
            # rebuild it from semantic parameters.
            law_keys = self._normalize_law_keys(self.keys, batch_size=count)
        if (
            not torch.is_tensor(getattr(self, "law_keys", None))
            or self.law_keys.shape != (count, law_dim)
            or self.law_keys.device != law_keys.device
            or self.law_keys.dtype != law_keys.dtype
        ):
            self.law_keys = law_keys
        if self.support.shape != (count,):
            self.support = self.keys.new_ones(count)
        configured_capacity = getattr(self, "context_alias_capacity", None)
        existing_context_keys = getattr(self, "context_keys", None)
        if (
            configured_capacity is None
            and torch.is_tensor(existing_context_keys)
            and existing_context_keys.ndim == 3
            and existing_context_keys.size(-1) == self.key_dim
        ):
            # A few intermediate checkpoints carried the alias tensor before
            # the explicit capacity attribute was serialized. Infer the
            # capacity from that tensor instead of silently truncating it to
            # the default three slots.
            configured_capacity = int(existing_context_keys.size(1))
        context_capacity = int(
            3 if configured_capacity is None else configured_capacity
        )
        if context_capacity <= 0:
            context_capacity = 3
        # Intermediate checkpoints may have serialized alias tensors before
        # the explicit capacity attribute was introduced.  Persist the
        # inferred width so the policy-default pass below cannot overwrite it
        # with the default three slots.
        if getattr(self, "context_alias_capacity", None) is None or int(
            getattr(self, "context_alias_capacity", context_capacity)
        ) <= 0:
            self.context_alias_capacity = context_capacity
        expected_context_shape = (count, context_capacity, self.key_dim)
        context_repaired = False
        if existing_context_keys is None or self.context_keys.shape != expected_context_shape:
            context_keys = self.keys.new_zeros(expected_context_shape)
            if count:
                context_keys[:, 0] = F.normalize(self.keys, dim=-1)
            self.context_keys = context_keys
            context_repaired = True
        elif (
            self.context_keys.device != self.keys.device
            or self.context_keys.dtype != self.keys.dtype
        ):
            self.context_keys = self.context_keys.to(
                device=self.keys.device, dtype=self.keys.dtype
            )
            context_repaired = True
        if (
            getattr(self, "context_valid", None) is None
            or self.context_valid.shape != (count, context_capacity)
        ):
            context_valid = torch.zeros(
                (count, context_capacity), device=self.keys.device, dtype=torch.bool
            )
            if count:
                context_valid[:, 0] = True
            self.context_valid = context_valid
            context_repaired = True
        elif self.context_valid.dtype != torch.bool:
            self.context_valid = self.context_valid.to(dtype=torch.bool)
            context_repaired = True
        if self.context_valid.device != self.keys.device:
            self.context_valid = self.context_valid.to(device=self.keys.device)
            context_repaired = True
        if (
            getattr(self, "context_support", None) is None
            or self.context_support.shape != (count, context_capacity)
        ):
            context_support = self.keys.new_zeros(count, context_capacity)
            if count:
                context_support[:, 0] = self.support
            self.context_support = context_support
            context_repaired = True
        elif (
            self.context_support.device != self.keys.device
            or self.context_support.dtype != self.keys.dtype
        ):
            self.context_support = self.context_support.to(
                device=self.keys.device, dtype=self.keys.dtype
            )
            context_repaired = True
        # A malformed/legacy row should always have one usable retrieval
        # identity.  Keep the first alias synchronized with the legacy key.
        if count:
            normalized_keys = F.normalize(self.keys, dim=-1)
            no_alias = ~self.context_valid.any(dim=-1)
            if bool(no_alias.any()):
                self.context_keys[no_alias, 0] = normalized_keys[no_alias]
                self.context_valid[no_alias, 0] = True
                self.context_support[no_alias, 0] = self.support[no_alias].clamp_min(1.0)
                context_repaired = True
            invalid_support = self.context_valid & (self.context_support <= 0.0)
            if bool(invalid_support.any()):
                self.context_support[invalid_support] = 1.0
                context_repaired = True
            if context_repaired:
                for row in range(count):
                    self._sync_legacy_key(row)
                # Shape/device repairs replace public tensors; any fixed
                # append backing store now points at the old views and must
                # be rebuilt before the next append.
                self._invalidate_fixed_storage()
        if self.quality_mass.shape != (count,):
            self.quality_mass = self.write_quality * self.support
        if self.split_mass.shape != (count,):
            self.split_mass = self.queue_weight * self.support
        if self.mode_ids.shape != (count,):
            self.mode_ids = torch.arange(count, device=self.device, dtype=torch.long)
        if self.mode_compressed.shape != (count,):
            self.mode_compressed = torch.zeros(count, device=self.device, dtype=torch.bool)
        self._next_mode_id = max(
            int(self._next_mode_id),
            int(self.mode_ids.max().item()) + 1 if count else 0,
        )
        defaults = {
            "_mode_duplicate_distances": {},
            "_mode_distances": {},
            "_mode_normal_gains": {},
            "_mode_pending_distances": {},
            "_mode_pending_gain_ema": {},
            "_mode_pending_gain_count": {},
            "_mode_pending_candidates": {},
        }
        for name, default in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, default)
        policy_defaults = {
            "adaptive_history_size": 64,
            "adaptive_min_samples": 18,
            "duplicate_quantile": 0.85,
            "mode_quantile": 0.95,
            "radius_margin": 1e-3,
            "gain_quantile": 0.95,
            "gain_ema_decay": 0.8,
            "gain_confirmation_min_count": 2,
            "gain_floor": 0.0,
            "context_alias_capacity": 3,
        }
        for name, default in policy_defaults.items():
            if not hasattr(self, name):
                setattr(self, name, default)
        # Pending candidates in an old checkpoint can carry retrieval-width
        # identities.  Migrate them before any candidate similarity is
        # evaluated; an unconvertible malformed record is safely discarded.
        for mode_id, candidates in list(
            getattr(self, "_mode_pending_candidates", {}).items()
        ):
            migrated_candidates = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                candidate_law = candidate.get("law_key")
                if not torch.is_tensor(candidate_law):
                    continue
                try:
                    candidate_law = self._normalize_law_keys(candidate_law).reshape(-1)
                except ValueError:
                    continue
                migrated = dict(candidate)
                migrated["law_key"] = candidate_law
                migrated_candidates.append(migrated)
            if migrated_candidates:
                self._mode_pending_candidates[int(mode_id)] = migrated_candidates
                self._sync_legacy_pending_state(int(mode_id))
            else:
                self._mode_pending_candidates.pop(int(mode_id), None)

    @torch.no_grad()
    def _sync_legacy_key(self, index: int) -> None:
        """Expose the first valid context alias through legacy ``keys``."""
        if len(self) == 0:
            return
        valid = self.context_valid[index]
        if bool(valid.any()):
            alias_index = int(torch.nonzero(valid, as_tuple=False)[0].item())
            alias = self.context_keys[index, alias_index]
            if not torch.equal(self.keys[index], alias):
                self.keys[index].copy_(alias)

    @torch.no_grad()
    def _update_context_aliases(
        self,
        index: int,
        context_key: Tensor,
    ) -> Dict[str, int | float | str]:
        """Update a prototype's retrieval aliases using online redundancy.

        Law matching has already classified the write as a duplicate.  This
        routine therefore never changes ``law_keys`` or the dynamics mode; it
        only refreshes, appends, or merges context identities.
        """
        self._ensure_prototype_state()
        query = F.normalize(
            context_key.detach().reshape(1, -1).to(self.context_keys), dim=-1
        ).reshape(-1)
        capacity = int(self.context_alias_capacity)
        valid_indices = torch.nonzero(
            self.context_valid[index], as_tuple=False
        ).flatten()
        alias_count = int(valid_indices.numel())
        if alias_count == 0:
            self.context_keys[index, 0] = query
            self.context_valid[index, 0] = True
            self.context_support[index, 0] = 1.0
            self._sync_legacy_key(index)
            return {
                "context_action": "append",
                "context_alias_index": 0,
                "context_alias_count": 1,
                "context_distance": 1.0,
                "context_redundancy_distance": float("inf"),
            }

        aliases = F.normalize(
            self.context_keys[index, valid_indices], dim=-1
        )
        alias_support = self.context_support[index, valid_indices].clamp_min(1.0)
        similarities = aliases @ query
        distances = (1.0 - similarities).clamp(0.0, 2.0)
        nearest_position = int(torch.argmin(distances).item())
        nearest_slot = int(valid_indices[nearest_position].item())
        d_query = float(distances[nearest_position].item())

        # Warm up a prototype with a second alias directly.  This avoids an
        # arbitrary context threshold before a redundancy estimate exists.
        if alias_count == 1 and capacity > 1:
            free_slot = int(
                torch.nonzero(~self.context_valid[index], as_tuple=False)[0].item()
            )
            self.context_keys[index, free_slot] = query
            self.context_valid[index, free_slot] = True
            self.context_support[index, free_slot] = 1.0
            self._sync_legacy_key(index)
            return {
                "context_action": "append_warmup",
                "context_alias_index": free_slot,
                "context_alias_count": 2,
                "context_distance": d_query,
                "context_redundancy_distance": float("inf"),
            }

        if alias_count >= 2:
            pairwise = (1.0 - aliases @ aliases.transpose(0, 1)).clamp(0.0, 2.0)
            pair_mask = torch.triu(
                torch.ones_like(pairwise, dtype=torch.bool), diagonal=1
            )
            pair_positions = torch.nonzero(pair_mask, as_tuple=False)
            pair_distances = pairwise[pair_mask]
            pair_position = int(torch.argmin(pair_distances).item())
            redundant_distance = float(pair_distances[pair_position].item())
            pair_a_position, pair_b_position = pair_positions[pair_position].tolist()
        else:
            redundant_distance = float("inf")
            pair_a_position = pair_b_position = -1

        if d_query <= redundant_distance:
            old_support = self.context_support[index, nearest_slot].clamp_min(1.0)
            beta = torch.clamp(
                1.0 / (old_support + 1.0),
                min=self.ema_beta_min,
                max=self.ema_beta_max,
            )
            blended = (
                (1.0 - beta) * aliases[nearest_position]
                + beta * query
            )
            self.context_keys[index, nearest_slot] = F.normalize(
                blended.reshape(1, -1), dim=-1
            ).reshape(-1)
            self.context_support[index, nearest_slot] = old_support + 1.0
            self._sync_legacy_key(index)
            return {
                "context_action": "refresh",
                "context_alias_index": nearest_slot,
                "context_alias_count": alias_count,
                "context_distance": d_query,
                "context_redundancy_distance": redundant_distance,
            }

        if alias_count < capacity:
            free_slot = int(
                torch.nonzero(~self.context_valid[index], as_tuple=False)[0].item()
            )
            self.context_keys[index, free_slot] = query
            self.context_valid[index, free_slot] = True
            self.context_support[index, free_slot] = 1.0
            self._sync_legacy_key(index)
            return {
                "context_action": "append",
                "context_alias_index": free_slot,
                "context_alias_count": alias_count + 1,
                "context_distance": d_query,
                "context_redundancy_distance": redundant_distance,
            }

        # The alias bank is full and the new context is more distinct than
        # the closest existing pair. Merge that pair by support, then use the
        # freed slot for the new context.
        slot_a = int(valid_indices[pair_a_position].item())
        slot_b = int(valid_indices[pair_b_position].item())
        support_a = self.context_support[index, slot_a].clamp_min(1.0)
        support_b = self.context_support[index, slot_b].clamp_min(1.0)
        merged = (
            support_a * F.normalize(
                self.context_keys[index, slot_a].reshape(1, -1), dim=-1
            ).reshape(-1)
            + support_b * F.normalize(
                self.context_keys[index, slot_b].reshape(1, -1), dim=-1
            ).reshape(-1)
        )
        self.context_keys[index, slot_a] = F.normalize(
            merged.reshape(1, -1), dim=-1
        ).reshape(-1)
        self.context_support[index, slot_a] = support_a + support_b
        self.context_keys[index, slot_b] = query
        self.context_support[index, slot_b] = 1.0
        self.context_valid[index, slot_a] = True
        self.context_valid[index, slot_b] = True
        self._sync_legacy_key(index)
        return {
            "context_action": "merge_append",
            "context_alias_index": slot_b,
            "context_alias_count": capacity,
            "context_distance": d_query,
            "context_redundancy_distance": redundant_distance,
        }

    def _trim_adaptive_histories(self) -> None:
        if not hasattr(self, "_mode_duplicate_distances"):
            return
        if not hasattr(self, "_mode_pending_distances"):
            self._mode_pending_distances = {}
        if not hasattr(self, "_mode_pending_candidates"):
            self._mode_pending_candidates = {}
        for values_by_mode in (
            self._mode_duplicate_distances,
            self._mode_distances,
            self._mode_normal_gains,
            self._mode_pending_distances,
        ):
            for mode_id, values in list(values_by_mode.items()):
                values_by_mode[int(mode_id)] = list(values)[
                    -self.adaptive_history_size:
                ]
        # Candidate histories and the number of candidates are bounded by the
        # rolling policy window (with at least one slot per possible mode
        # row).  The candidate law key is kept detached so it cannot retain an
        # autograd graph across queue/checkpoint boundaries.
        for mode_id, candidates in list(self._mode_pending_candidates.items()):
            trimmed_candidates: List[Dict[str, Any]] = []
            for candidate in list(candidates)[
                -self._pending_candidate_capacity() :
            ]:
                if not isinstance(candidate, dict):
                    continue
                law_key = candidate.get("law_key")
                if torch.is_tensor(law_key):
                    law_key = law_key.detach().reshape(-1).clone()
                else:
                    # Legacy candidate records without a law identity cannot
                    # participate in identity matching and are dropped rather
                    # than silently reintroducing mode-level aggregation.
                    continue
                history = []
                for value in candidate.get("distance_history", []):
                    value = float(value)
                    if math.isfinite(value):
                        history.append(value)
                candidate = {
                    "law_key": law_key,
                    "gain_ema": float(candidate.get("gain_ema", 0.0)),
                    "count": int(candidate.get("count", 0)),
                    "distance_history": history[-self.adaptive_history_size :],
                }
                trimmed_candidates.append(candidate)
            if trimmed_candidates:
                self._mode_pending_candidates[int(mode_id)] = trimmed_candidates
                self._sync_legacy_pending_state(int(mode_id))
            else:
                self._mode_pending_candidates.pop(int(mode_id), None)

    def _pending_candidate_capacity(self) -> int:
        """Bound temporary candidates without losing multi-outlier identity."""
        return max(
            1,
            int(getattr(self, "adaptive_history_size", 1)),
            int(getattr(self, "mode_capacity", 1)),
            int(getattr(self, "gain_confirmation_min_count", 1)),
        )

    @staticmethod
    def _rolling_quantile(values: Sequence[float], quantile: float) -> float:
        if not values:
            raise ValueError("cannot compute a quantile from an empty history")
        tensor = torch.tensor(tuple(values), dtype=torch.float64)
        return float(torch.quantile(tensor, quantile).item())

    def _append_adaptive_value(
        self,
        values_by_mode: Dict[int, List[float]],
        mode_id: int,
        value: float,
    ) -> None:
        if not math.isfinite(value):
            raise FloatingPointError("adaptive matching statistics must be finite")
        history = values_by_mode.setdefault(int(mode_id), [])
        history.append(float(value))
        if len(history) > self.adaptive_history_size:
            del history[:-self.adaptive_history_size]

    def _sync_legacy_pending_state(self, mode_id: int) -> None:
        """Mirror candidate records into the pre-candidate checkpoint fields.

        Older checkpoints represented pending confirmation with one distance
        list and one gain/count pair per mode.  Keep those fields populated as
        a flattened/representative view so old inspection code continues to
        work, while all new matching decisions use the law-keyed candidates.
        """
        mode_id = int(mode_id)
        candidates = self._mode_pending_candidates.get(mode_id, [])
        if not candidates:
            self._mode_pending_distances.pop(mode_id, None)
            self._mode_pending_gain_ema.pop(mode_id, None)
            self._mode_pending_gain_count.pop(mode_id, None)
            return

        flattened: List[float] = []
        for candidate in candidates:
            flattened.extend(
                float(value)
                for value in candidate.get("distance_history", [])
                if math.isfinite(float(value))
            )
        if flattened:
            self._mode_pending_distances[mode_id] = flattened[
                -self.adaptive_history_size :
            ]
        else:
            self._mode_pending_distances.pop(mode_id, None)

        # The aggregate maps have no exact representation when several
        # candidates coexist.  Expose the most persistent one as a stable
        # compatibility summary; candidate records remain authoritative.
        representative = max(
            enumerate(candidates),
            key=lambda item: (int(item[1].get("count", 0)), item[0]),
        )[1]
        self._mode_pending_gain_ema[mode_id] = float(
            representative.get("gain_ema", 0.0)
        )
        self._mode_pending_gain_count[mode_id] = int(
            representative.get("count", 0)
        )

    def _find_pending_candidate(
        self,
        mode_id: int,
        law_key: Tensor,
    ) -> Optional[int]:
        """Return the matching pending law candidate, if identity is clear.

        Pending candidates use the duplicate-law similarity as their identity
        gate.  Nearest-candidate matching without this gate would make two
        unrelated outliers share a confirmation counter merely because they
        are both nearest to the same old mode.
        """
        candidates = self._mode_pending_candidates.get(int(mode_id), [])
        if not candidates:
            return None
        query = F.normalize(
            law_key.detach().reshape(1, -1).to(
                device=self.device, dtype=self.keys.dtype
            ),
            dim=-1,
        ).reshape(-1)
        best_index: Optional[int] = None
        best_similarity = -1.0
        for index, candidate in enumerate(candidates):
            candidate_key = candidate.get("law_key")
            if not torch.is_tensor(candidate_key):
                continue
            candidate_key = candidate_key.detach().reshape(-1).to(query)
            if candidate_key.numel() != query.numel():
                continue
            candidate_key = F.normalize(candidate_key.reshape(1, -1), dim=-1).reshape(-1)
            similarity = float(torch.dot(candidate_key, query).item())
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index
        if best_index is None or best_similarity < self.duplicate_threshold:
            return None
        return best_index

    def _new_pending_candidate(
        self,
        mode_id: int,
        law_key: Tensor,
        prediction_gain: float,
    ) -> int:
        """Create a law-keyed temporary candidate and return its list index."""
        mode_id = int(mode_id)
        candidates = self._mode_pending_candidates.setdefault(mode_id, [])
        if not candidates:
            # A legacy checkpoint may have aggregate pending state but no law
            # identity.  It cannot be safely attributed to this new candidate.
            self._mode_pending_distances.pop(mode_id, None)
            self._mode_pending_gain_ema.pop(mode_id, None)
            self._mode_pending_gain_count.pop(mode_id, None)
        normalized_law_key = F.normalize(
            law_key.detach().reshape(1, -1).to(
                device=self.device, dtype=self.keys.dtype
            ),
            dim=-1,
        ).reshape(-1).clone()
        candidates.append(
            {
                "law_key": normalized_law_key,
                "gain_ema": float(prediction_gain),
                "count": 1,
                "distance_history": [],
            }
        )
        candidate_capacity = self._pending_candidate_capacity()
        if len(candidates) > candidate_capacity:
            del candidates[:-candidate_capacity]
        self._sync_legacy_pending_state(mode_id)
        return len(candidates) - 1

    def _update_pending_candidate_gain(
        self,
        mode_id: int,
        candidate_index: int,
        prediction_gain: float,
    ) -> Tuple[float, int]:
        """Apply one gain observation to an existing law candidate."""
        mode_id = int(mode_id)
        candidate = self._mode_pending_candidates[mode_id][candidate_index]
        previous = candidate.get("gain_ema")
        gain_ema = (
            float(prediction_gain)
            if previous is None
            else self.gain_ema_decay * float(previous)
            + (1.0 - self.gain_ema_decay) * float(prediction_gain)
        )
        count = int(candidate.get("count", 0)) + 1
        candidate["gain_ema"] = gain_ema
        candidate["count"] = count
        self._sync_legacy_pending_state(mode_id)
        return gain_ema, count

    def _append_pending_candidate_distance(
        self,
        mode_id: int,
        candidate_index: int,
        distance: float,
    ) -> None:
        mode_id = int(mode_id)
        candidate = self._mode_pending_candidates[mode_id][candidate_index]
        history = candidate.setdefault("distance_history", [])
        history.append(float(distance))
        if len(history) > self.adaptive_history_size:
            del history[:-self.adaptive_history_size]
        self._sync_legacy_pending_state(mode_id)

    def _promote_pending_candidate(
        self,
        mode_id: int,
        *,
        law_key: Optional[Tensor] = None,
        candidate_index: Optional[int] = None,
    ) -> bool:
        """Promote one confirmed candidate's queued distances to its mode."""
        mode_id = int(mode_id)
        candidates = self._mode_pending_candidates.get(mode_id, [])
        if candidate_index is None:
            if law_key is None:
                return False
            candidate_index = self._find_pending_candidate(mode_id, law_key)
        if candidate_index is None or not 0 <= int(candidate_index) < len(candidates):
            return False
        candidate = candidates.pop(int(candidate_index))
        for distance in candidate.get("distance_history", []):
            self._append_adaptive_value(
                self._mode_distances, mode_id, float(distance)
            )
        if candidates:
            self._mode_pending_candidates[mode_id] = candidates
        else:
            self._mode_pending_candidates.pop(mode_id, None)
        self._sync_legacy_pending_state(mode_id)
        return True

    def _discard_pending_candidate(
        self,
        mode_id: int,
        *,
        law_key: Optional[Tensor] = None,
        candidate_index: Optional[int] = None,
    ) -> bool:
        """Discard one candidate without affecting unrelated candidates."""
        mode_id = int(mode_id)
        candidates = self._mode_pending_candidates.get(mode_id, [])
        if candidate_index is None:
            if law_key is None:
                return False
            candidate_index = self._find_pending_candidate(mode_id, law_key)
        if candidate_index is None or not 0 <= int(candidate_index) < len(candidates):
            return False
        candidates.pop(int(candidate_index))
        if candidates:
            self._mode_pending_candidates[mode_id] = candidates
        else:
            self._mode_pending_candidates.pop(mode_id, None)
        self._sync_legacy_pending_state(mode_id)
        return True

    def _promote_pending_mode_distances(self, mode_id: int) -> None:
        """Move queued outlier distances into a mode's accepted history.

        A law outlier is not allowed to calibrate ``mode_radius`` merely
        because it was observed once.  Once another observation confirms the
        current mode, however, every queued law distance is evidence of that
        mode's natural variation and should be included retrospectively.
        """
        mode_id = int(mode_id)
        pending = self._mode_pending_distances.pop(mode_id, [])
        for distance in pending:
            self._append_adaptive_value(self._mode_distances, mode_id, distance)
        if not self._mode_pending_candidates.get(mode_id):
            self._mode_pending_gain_ema.pop(mode_id, None)
            self._mode_pending_gain_count.pop(mode_id, None)

    def _discard_pending_mode_distances(self, mode_id: int) -> None:
        """Drop queued law distances when an outlier becomes a new mode."""
        mode_id = int(mode_id)
        self._mode_pending_distances.pop(mode_id, None)
        self._mode_pending_candidates.pop(mode_id, None)
        self._mode_pending_gain_ema.pop(mode_id, None)
        self._mode_pending_gain_count.pop(mode_id, None)

    def adaptive_radii(self, mode_id: int) -> Tuple[float, float]:
        """Return radii from accepted plus retrospectively confirmed samples."""
        self._ensure_prototype_state()
        duplicate_prior = max(0.0, 1.0 - self.duplicate_threshold)
        mode_prior = max(duplicate_prior + self.radius_margin, 1.0 - self.mode_threshold)
        duplicate_history = self._mode_duplicate_distances.get(int(mode_id), [])
        mode_history = self._mode_distances.get(int(mode_id), [])
        duplicate_radius = (
            self._rolling_quantile(duplicate_history, self.duplicate_quantile)
            if len(duplicate_history) >= self.adaptive_min_samples
            else duplicate_prior
        )
        mode_radius = (
            self._rolling_quantile(mode_history, self.mode_quantile)
            if len(mode_history) >= self.adaptive_min_samples
            else mode_prior
        )
        duplicate_radius = min(
            max(duplicate_radius, 0.0),
            max(0.0, 2.0 - self.radius_margin),
        )
        mode_radius = min(
            max(mode_radius, duplicate_radius + self.radius_margin), 2.0
        )
        return duplicate_radius, mode_radius

    def _gain_threshold(self, mode_id: int) -> float:
        history = self._mode_normal_gains.get(int(mode_id), [])
        if not history:
            return self.gain_floor
        return max(
            self.gain_floor,
            self._rolling_quantile(history, self.gain_quantile),
        )

    def _record_in_mode_observation(
        self,
        mode_id: int,
        *,
        distance: float,
        prediction_gain: float,
        duplicate: bool,
        law_key: Optional[Tensor] = None,
    ) -> None:
        # A duplicate/local observation after a queued outlier is a
        # retrospective confirmation of the current mode.  Promote only the
        # pending candidate with the same law identity; unrelated outliers
        # waiting under this mode must remain quarantined.
        mode_id = int(mode_id)
        if self._mode_pending_candidates.get(mode_id):
            self._promote_pending_candidate(mode_id, law_key=law_key)
        elif mode_id in self._mode_pending_distances:
            # Checkpoints written before law-keyed candidates have no identity
            # to match.  Preserve their historical behavior on first use.
            self._promote_pending_mode_distances(mode_id)
        distances = (
            self._mode_duplicate_distances
            if duplicate
            else self._mode_distances
        )
        self._append_adaptive_value(distances, mode_id, distance)
        self._append_adaptive_value(
            self._mode_normal_gains, mode_id, prediction_gain
        )
        if not self._mode_pending_candidates.get(mode_id):
            self._mode_pending_gain_ema.pop(mode_id, None)
            self._mode_pending_gain_count.pop(mode_id, None)

    def _confirm_new_mode(
        self,
        mode_id: int,
        prediction_gain: float,
    ) -> Tuple[bool, float, float, int]:
        mode_id = int(mode_id)
        previous = self._mode_pending_gain_ema.get(mode_id)
        gain_ema = (
            float(prediction_gain)
            if previous is None
            else self.gain_ema_decay * previous
            + (1.0 - self.gain_ema_decay) * float(prediction_gain)
        )
        count = self._mode_pending_gain_count.get(mode_id, 0) + 1
        self._mode_pending_gain_ema[mode_id] = gain_ema
        self._mode_pending_gain_count[mode_id] = count
        threshold = self._gain_threshold(mode_id)
        confirmed = (
            count >= self.gain_confirmation_min_count
            and gain_ema > threshold
        )
        return confirmed, gain_ema, threshold, count

    def _confirm_pending_candidate(
        self,
        mode_id: int,
        candidate_index: int,
        prediction_gain: float,
    ) -> Tuple[bool, float, float, int]:
        """Update one law candidate and evaluate the mode confirmation gate."""
        mode_id = int(mode_id)
        gain_ema, count = self._update_pending_candidate_gain(
            mode_id, candidate_index, prediction_gain
        )
        threshold = self._gain_threshold(mode_id)
        confirmed = (
            count >= self.gain_confirmation_min_count
            and gain_ema > threshold
        )
        return confirmed, gain_ema, threshold, count

    @torch.no_grad()
    def _refresh_prototype(
        self,
        index: int,
        *,
        key: Tensor,
        delta_theta: Tensor,
        law_key: Tensor,
        window: Optional[EventWindow],
        quality: Tensor,
        queue: Tensor,
        law_key_builder: Optional[Callable[[Tensor], Tensor]],
    ) -> Dict[str, int | float | str]:
        old_support = self.support[index].clamp_min(1.0)
        beta = torch.clamp(
            1.0 / (old_support + 1.0),
            min=self.ema_beta_min,
            max=self.ema_beta_max,
        )
        self.deltas[index].mul_(1.0 - beta).add_(delta_theta.reshape(-1), alpha=float(beta))
        context_result = self._update_context_aliases(index, key.reshape(-1))
        if law_key_builder is not None:
            refreshed_law = law_key_builder(self.deltas[index])
        else:
            refreshed_law = (1.0 - beta) * self.law_keys[index] + beta * law_key.reshape(-1)
        refreshed_law = self._normalize_law_keys(
            refreshed_law,
            batch_size=1,
            name="refreshed law_key",
        ).reshape(-1)
        self.law_keys[index] = F.normalize(
            refreshed_law.detach().reshape(1, -1).to(self.law_keys), dim=-1
        ).reshape(-1)
        self.support[index] = old_support + 1.0
        self.quality_mass[index] += quality.reshape(())
        self.split_mass[index] += queue.reshape(())
        self.write_quality[index] = self.quality_mass[index] / self.support[index]
        self.queue_weight[index] = self.split_mass[index] / self.support[index]
        self.age[index] = 0.0
        self.stale_cycles[index] = 0.0
        if window is not None:
            self.windows[index] = window
        result = {
            "action": "refresh",
            "index": int(index),
            "mode_id": int(self.mode_ids[index].item()),
            "support": float(self.support[index].item()),
        }
        result.update(context_result)
        return result

    def _mode_retention(self, mode_id: int) -> Tensor:
        rows = self.mode_ids == int(mode_id)
        support = self.support[rows].sum()
        usage = (self.usage[rows] + self.cycle_usage[rows]).sum()
        stale = self.stale_cycles[rows].max()
        age = (self.age[rows] * self.support[rows]).sum() / support.clamp_min(1.0)
        return (
            self.retention_support_weight * torch.log1p(support)
            + self.retention_usage_weight * torch.log1p(usage)
            - self.retention_stale_weight * stale
            - self.retention_age_weight * torch.log1p(age)
        )

    def _mode_retention_batch(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute retention for every mode with one grouped reduction."""
        mode_ids, inverse, counts = torch.unique(
            self.mode_ids[: len(self)],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        mode_count = mode_ids.numel()
        if mode_count == 0:
            empty = self.support.new_empty(0)
            return mode_ids, empty, counts

        mode_support = self.support.new_zeros(mode_count)
        mode_support.index_add_(0, inverse, self.support[: len(self)])
        mode_usage = self.usage.new_zeros(mode_count)
        mode_usage.index_add_(
            0,
            inverse,
            self.usage[: len(self)] + self.cycle_usage[: len(self)],
        )
        mode_age_mass = self.age.new_zeros(mode_count)
        mode_age_mass.index_add_(
            0,
            inverse,
            self.age[: len(self)] * self.support[: len(self)],
        )
        mode_stale = self.stale_cycles.new_full(
            (mode_count,), -torch.inf
        )
        if hasattr(mode_stale, "scatter_reduce_"):
            mode_stale.scatter_reduce_(
                0,
                inverse,
                self.stale_cycles[: len(self)],
                reduce="amax",
                include_self=True,
            )
        else:
            # Kept for older PyTorch installations; modern CUDA builds use the
            # single scatter-reduce above.
            mode_stale = torch.stack([
                self.stale_cycles[: len(self)][inverse == index].max()
                for index in range(mode_count)
            ])
        mode_age = mode_age_mass / mode_support.clamp_min(1.0)
        retention = (
            self.retention_support_weight * torch.log1p(mode_support)
            + self.retention_usage_weight * torch.log1p(mode_usage)
            - self.retention_stale_weight * mode_stale
            - self.retention_age_weight * torch.log1p(mode_age)
        )
        return mode_ids, retention, counts

    @torch.no_grad()
    def _compress_mode(
        self,
        mode_id: int,
        law_key_builder: Optional[Callable[[Tensor], Tensor]],
    ) -> bool:
        indices = torch.nonzero(self.mode_ids == int(mode_id), as_tuple=False).flatten()
        if indices.numel() <= 1:
            return False
        support = self.support[indices]
        weights = support * self.write_quality[indices].clamp_min(0.0)
        if not bool((weights.sum() > 0.0)):
            weights = support
        normalized = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        target = int(indices[0].item())
        compressed_delta = (normalized[:, None] * self.deltas[indices]).sum(0)
        compressed_key = F.normalize(
            (normalized[:, None] * self.keys[indices]).sum(0, keepdim=True), dim=-1
        ).reshape(-1)
        if law_key_builder is not None:
            compressed_law = law_key_builder(compressed_delta)
        else:
            compressed_law = F.normalize(
                (normalized[:, None] * self.law_keys[indices]).sum(0, keepdim=True),
                dim=-1,
            ).reshape(-1)
        compressed_law = self._normalize_law_keys(
            compressed_law,
            batch_size=1,
            name="compressed law_key",
        ).reshape(-1)
        total_support = support.sum()
        support_average = support / total_support.clamp_min(1.0)
        self.keys[target] = compressed_key
        self.deltas[target] = compressed_delta
        self.law_keys[target] = compressed_law.to(self.law_keys)
        self.support[target] = total_support
        self.quality_mass[target] = self.quality_mass[indices].sum()
        self.split_mass[target] = self.split_mass[indices].sum()
        self.write_quality[target] = self.quality_mass[target] / total_support.clamp_min(1.0)
        self.queue_weight[target] = self.split_mass[target] / total_support.clamp_min(1.0)
        self.usage[target] = (support_average * self.usage[indices]).sum()
        self.cycle_usage[target] = (support_average * self.cycle_usage[indices]).sum()
        self.stale_cycles[target] = self.stale_cycles[indices].max()
        self.age[target] = (support_average * self.age[indices]).sum()
        self.mode_compressed[target] = True

        # Preserve retrieval identity separately from the compressed law key.
        # Keep the strongest aliases explicitly and fold any overflow into a
        # final support-weighted alias so the row's total evidence remains
        # represented after a mode is archived.
        alias_keys = self.context_keys[indices]
        alias_valid = self.context_valid[indices]
        alias_support = self.context_support[indices].clamp_min(0.0)
        flat_keys = alias_keys.reshape(-1, self.key_dim)
        flat_valid = alias_valid.reshape(-1)
        flat_support = alias_support.reshape(-1)
        valid_positions = torch.nonzero(flat_valid, as_tuple=False).flatten()
        self.context_keys[target].zero_()
        self.context_valid[target].zero_()
        self.context_support[target].zero_()
        if valid_positions.numel() > 0:
            order = torch.argsort(
                flat_support[valid_positions], descending=True
            )
            ordered = valid_positions[order]
            keep_count = min(self.context_alias_capacity, int(ordered.numel()))
            if keep_count == self.context_alias_capacity and ordered.numel() > keep_count:
                direct_count = max(0, keep_count - 1)
                direct = ordered[:direct_count]
                remainder = ordered[direct_count:]
                selected_keys = flat_keys[direct]
                selected_support = flat_support[direct]
                rem_weights = flat_support[remainder]
                rem_support = rem_weights.sum().clamp_min(1.0)
                rem_key = F.normalize(
                    (rem_weights[:, None] * flat_keys[remainder]).sum(0, keepdim=True),
                    dim=-1,
                ).reshape(-1)
                selected_keys = torch.cat([selected_keys, rem_key.unsqueeze(0)], dim=0)
                selected_support = torch.cat([selected_support, rem_support.reshape(1)], dim=0)
            else:
                selected = ordered[:keep_count]
                selected_keys = flat_keys[selected]
                selected_support = flat_support[selected]
            self.context_keys[target, :keep_count] = F.normalize(
                selected_keys, dim=-1
            )
            self.context_valid[target, :keep_count] = True
            self.context_support[target, :keep_count] = selected_support
        self._sync_legacy_key(target)
        best_window = int(indices[torch.argmax(weights)].item())
        self.windows[target] = self.windows[best_window]
        keep = torch.ones(len(self), device=self.device, dtype=torch.bool)
        keep[indices[1:]] = False
        self.keep(torch.nonzero(keep, as_tuple=False).flatten())
        return True

    @torch.no_grad()
    def _make_space(
        self,
        law_key_builder: Optional[Callable[[Tensor], Tensor]],
        protected_mode: Optional[int] = None,
    ) -> bool:
        while len(self) >= self.capacity:
            modes, retention, counts = self._mode_retention_batch()
            compressible = counts > 1
            if bool(compressible.any()):
                inf = torch.full_like(retention, torch.inf)
                victim_position = torch.where(
                    compressible, retention, inf
                ).argmin()
                victim = int(modes[victim_position].item())
                self._compress_mode(victim, law_key_builder)
                continue
            # Every mode is already represented by one archive/singleton.
            # Only now is true eviction unavoidable under the hard bank cap.
            candidates = (
                torch.ones_like(modes, dtype=torch.bool)
                if protected_mode is None
                else modes != int(protected_mode)
            )
            if not bool(candidates.any()):
                return False
            inf = torch.full_like(retention, torch.inf)
            victim_position = torch.where(candidates, retention, inf).argmin()
            victim = int(modes[victim_position].item())
            keep = self.mode_ids != victim
            self.keep(torch.nonzero(keep, as_tuple=False).flatten())
            break
        return True

    def _batch_statistic(
        self,
        value: float | Tensor,
        *,
        name: str,
        batch_size: int,
    ) -> Tensor:
        if isinstance(value, Real):
            scalar_value = float(value)
            if not math.isfinite(scalar_value):
                raise FloatingPointError(f"{name} must be finite")
            if not 0.0 <= scalar_value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            result = torch.full(
                (batch_size,),
                scalar_value,
                device=self.device,
                dtype=self.keys.dtype,
            )
        else:
            result = torch.as_tensor(
                value,
                device=self.device,
                dtype=self.keys.dtype,
            ).reshape(-1)
            if result.numel() == 1:
                result = result.expand(batch_size)
            elif result.numel() != batch_size:
                raise ValueError(
                    f"{name} must be scalar or contain one value per batched write"
                )
        if not bool(
            torch.isfinite(result).all()
            & (result >= 0.0).all()
            & (result <= 1.0).all()
        ):
            raise ValueError(f"{name} must be finite values in [0, 1]")
        return result

    def _batch_prediction_gain(
        self,
        value: float | Tensor,
        *,
        batch_size: int,
    ) -> Tensor:
        result = torch.as_tensor(
            value,
            device=self.device,
            dtype=self.keys.dtype,
        ).reshape(-1)
        if result.numel() == 1:
            result = result.expand(batch_size)
        elif result.numel() != batch_size:
            raise ValueError(
                "prediction_gain must be scalar or contain one value per write"
            )
        if not bool(torch.isfinite(result).all()):
            raise ValueError("prediction_gain must contain only finite values")
        return result

    @torch.no_grad()
    def _add_batch_scalar_fallback(
        self,
        keys: Tensor,
        deltas: Tensor,
        law_keys: Tensor,
        quality: Tensor,
        queue: Tensor,
        prediction_gain: Optional[Tensor],
        windows: Sequence[Optional[EventWindow]],
        law_key_builder: Optional[Callable[[Tensor], Tensor]],
    ) -> List[Dict[str, int | float | str]]:
        """Preserve exact causal admission for capacity/mode edge cases."""
        return [
            self.add(
                key=keys[index],
                delta_theta=deltas[index],
                window=windows[index],
                write_quality=quality[index],
                queue_weight=queue[index],
                prediction_gain=(
                    None
                    if prediction_gain is None
                    else prediction_gain[index]
                ),
                law_key=law_keys[index],
                law_key_builder=law_key_builder,
            )
            for index in range(keys.size(0))
        ]

    @torch.no_grad()
    def add_batch(
        self,
        keys: Tensor,
        delta_theta: Tensor,
        windows: Optional[Sequence[Optional[EventWindow]]] = None,
        write_quality: float | Tensor = 1.0,
        queue_weight: float | Tensor = 0.0,
        law_keys: Optional[Tensor] = None,
        law_key_builder: Optional[Callable[[Tensor], Tensor]] = None,
        prediction_gain: Optional[float | Tensor] = None,
    ) -> List[Dict[str, int | float | str]]:
        """Admit a node-local batch in causal order.

        Adaptive radii and prediction confirmation are updated after every
        observation, so later rows in the batch must see earlier decisions.
        Input normalization remains batched; classification delegates to the
        scalar transaction to preserve exactly that order-dependent state.
        """
        if keys.ndim != 2 or keys.size(-1) != self.key_dim:
            raise ValueError(
                f"keys must have shape [B, {self.key_dim}], got {tuple(keys.shape)}"
            )
        if delta_theta.ndim != 2 or delta_theta.size(-1) != self.param_dim:
            raise ValueError(
                "delta_theta must have shape "
                f"[B, {self.param_dim}], got {tuple(delta_theta.shape)}"
            )
        if delta_theta.size(0) != keys.size(0):
            raise ValueError("keys and delta_theta must have the same batch size")
        batch_size = int(keys.size(0))
        if batch_size == 0:
            return []
        if windows is None:
            windows = [None] * batch_size
        elif len(windows) != batch_size:
            raise ValueError("windows must contain one entry per batched write")

        # Keep the same detached persistent-state boundary as add(), while
        # normalizing a complete batch in one kernel.
        keys = F.normalize(
            keys.detach().to(device=self.device, dtype=self.keys.dtype),
            dim=-1,
        )
        deltas = delta_theta.detach().to(
            device=self.device,
            dtype=self.deltas.dtype,
        )
        quality = self._batch_statistic(
            write_quality, name="write_quality", batch_size=batch_size
        )
        queue = self._batch_statistic(
            queue_weight, name="queue_weight", batch_size=batch_size
        )
        gain = (
            None
            if prediction_gain is None
            else self._batch_prediction_gain(
                prediction_gain,
                batch_size=batch_size,
            )
        )

        if law_keys is None:
            if law_key_builder is None:
                law_keys = keys
            else:
                law_keys = law_key_builder(deltas)
        law_keys = self._normalize_law_keys(
            law_keys,
            batch_size=batch_size,
            name="law_keys",
        )
        if not bool(
            torch.isfinite(keys).all()
            & torch.isfinite(deltas).all()
            & torch.isfinite(law_keys).all()
        ):
            raise FloatingPointError(
                "episodic-memory batched key or residual contains NaN or Inf"
            )

        # Adaptive radii and persistence confirmation are causal, mode-local
        # state.  Process the normalized rows in input order so batched and
        # scalar admission have exactly the same semantics.
        return self._add_batch_scalar_fallback(
            keys,
            deltas,
            law_keys,
            quality,
            queue,
            gain,
            windows,
            law_key_builder,
        )

    @torch.no_grad()
    def add(
        self,
        key: Tensor,
        delta_theta: Tensor,
        window: Optional[EventWindow] = None,
        write_quality: float | Tensor = 1.0,
        queue_weight: float | Tensor = 0.0,
        law_key: Optional[Tensor] = None,
        law_key_builder: Optional[Callable[[Tensor], Tensor]] = None,
        prediction_gain: Optional[float | Tensor] = None,
        force_new_mode_confirmation: bool = False,
    ) -> Dict[str, int | float | str]:
        if key.numel() != self.key_dim:
            raise ValueError(
                f"key must contain {self.key_dim} values, got {key.numel()}"
            )
        if delta_theta.numel() != self.param_dim:
            raise ValueError(
                f"delta_theta must contain {self.param_dim} values, "
                f"got {delta_theta.numel()}"
            )
        quality_is_host_scalar = isinstance(write_quality, Real)
        queue_is_host_scalar = isinstance(queue_weight, Real)
        if quality_is_host_scalar:
            quality_value = float(write_quality)
            if not math.isfinite(quality_value):
                raise FloatingPointError("write_quality must be finite")
            if not 0.0 <= quality_value <= 1.0:
                raise ValueError("write_quality must lie in [0, 1]")
        if queue_is_host_scalar:
            queue_value = float(queue_weight)
            if not math.isfinite(queue_value):
                raise FloatingPointError("queue_weight must be finite")
            if not 0.0 <= queue_value <= 1.0:
                raise ValueError("queue_weight must lie in [0, 1]")

        quality = torch.as_tensor(
            write_quality,
            device=self.device,
            dtype=self.keys.dtype,
        ).reshape(-1)
        queue = torch.as_tensor(
            queue_weight,
            device=self.device,
            dtype=self.keys.dtype,
        ).reshape(-1)
        if quality.numel() != 1 or queue.numel() != 1:
            raise ValueError("write_quality and queue_weight must be scalar")
        if not (quality_is_host_scalar and queue_is_host_scalar):
            scalar_values_valid = (
                torch.isfinite(quality).all()
                & torch.isfinite(queue).all()
                & (quality >= 0.0).all()
                & (quality <= 1.0).all()
                & (queue >= 0.0).all()
                & (queue <= 1.0).all()
            )
            if not bool(scalar_values_valid):
                raise ValueError(
                    "write_quality and queue_weight must be finite values in [0, 1]"
                )
        gain_provided = prediction_gain is not None
        gain = torch.as_tensor(
            quality if prediction_gain is None else prediction_gain,
            device=self.device,
            dtype=self.keys.dtype,
        ).reshape(-1)
        if gain.numel() != 1:
            raise ValueError("prediction_gain must be scalar")
        if not bool(torch.isfinite(gain).all()):
            raise ValueError("prediction_gain must be finite")
        gain_value = float(gain.item())

        # Persistent memory is state, not a retained autograd graph. Gradients
        # still flow through the differentiable retrieval weights.
        key = F.normalize(key.detach().reshape(1, -1), dim=-1)
        delta_theta = delta_theta.detach().reshape(1, -1)
        if law_key is None and law_key_builder is not None:
            law_key = law_key_builder(delta_theta.reshape(-1))
        if law_key is None:
            law_key = key.reshape(-1)
        law_key = self._normalize_law_keys(
            law_key,
            batch_size=1,
            name="law_key",
        )
        memory_values_finite = (
            torch.isfinite(key).all()
            & torch.isfinite(delta_theta).all()
            & torch.isfinite(law_key).all()
        )
        if not bool(memory_values_finite):
            raise FloatingPointError(
                "episodic-memory key or residual contains NaN or Inf"
            )

        self._ensure_prototype_state()
        self._ensure_fixed_storage()
        decision: Dict[str, float | int | str] = {}
        append_is_duplicate_seed = True
        match_type = "new_dynamics"
        if len(self) > 0:
            similarities = self.law_keys @ law_key.reshape(-1)
            best_index = int(torch.argmax(similarities).item())
            best_similarity = float(similarities[best_index].item())
            best_distance = min(max(1.0 - best_similarity, 0.0), 2.0)
            best_mode = int(self.mode_ids[best_index].item())
            duplicate_radius, mode_radius = self.adaptive_radii(best_mode)
            decision = {
                "distance": best_distance,
                "duplicate_radius": duplicate_radius,
                "mode_radius": mode_radius,
                "prediction_gain": gain_value,
            }
            if best_distance <= duplicate_radius:
                if bool(self.mode_compressed[best_index]):
                    self.mode_compressed[best_index] = False
                result = self._refresh_prototype(
                    best_index,
                    key=key,
                    delta_theta=delta_theta,
                    law_key=law_key,
                    window=window,
                    quality=quality,
                    queue=queue,
                    law_key_builder=law_key_builder,
                )
                self._record_in_mode_observation(
                    best_mode,
                    distance=best_distance,
                    prediction_gain=gain_value,
                    duplicate=True,
                    law_key=law_key,
                )
                result.update(decision)
                result["match_type"] = "duplicate"
                return result
            # Every non-duplicate write is either a local variation of the
            # nearest mode or a genuinely new mode.  A queued law outlier can
            # be retrospectively admitted to the current mode once repeated
            # low-gain evidence says that it is not a new dynamics; this is
            # what lets the mode radius expand instead of being permanently
            # censored by its cold-start threshold.
            mode_id = best_mode
            if best_distance <= mode_radius:
                append_is_duplicate_seed = False
                match_type = "local_variation"
            else:
                pending_candidate_index: Optional[int] = None
                if gain_provided:
                    pending_candidate_index = self._find_pending_candidate(
                        best_mode, law_key
                    )
                    if pending_candidate_index is None:
                        pending_candidate_index = self._new_pending_candidate(
                            best_mode, law_key, gain_value
                        )
                        pending_candidate = self._mode_pending_candidates[
                            best_mode
                        ][pending_candidate_index]
                        gain_ema = float(pending_candidate["gain_ema"])
                        pending_count = int(pending_candidate["count"])
                        gain_threshold = self._gain_threshold(best_mode)
                        confirmed = (
                            pending_count >= self.gain_confirmation_min_count
                            and gain_ema > gain_threshold
                        )
                    else:
                        confirmed, gain_ema, gain_threshold, pending_count = (
                            self._confirm_pending_candidate(
                                best_mode,
                                pending_candidate_index,
                                gain_value,
                            )
                        )
                    decision.update({
                        "gain_ema": gain_ema,
                        "gain_threshold": gain_threshold,
                        "confirmation_count": pending_count,
                        "pending_candidate_count": len(
                            self._mode_pending_candidates.get(best_mode, [])
                        ),
                    })
                    current_mode_confirmed = (
                        not confirmed
                        and not force_new_mode_confirmation
                        and pending_count >= self.gain_confirmation_min_count
                    )
                else:
                    # Legacy callers do not have the signed local-adaptation
                    # gain needed by the confirmation gate.  Preserve their
                    # historical immediate-admission behavior; production
                    # wake/inference paths pass ``prediction_gain`` and use
                    # the persistent queue/confirmation policy above.
                    confirmed = True
                    current_mode_confirmed = False

                if current_mode_confirmed:
                    # This candidate is now confirmed as natural variation
                    # of ``best_mode``.  Leave promotion until the common
                    # local-variation path below has passed its capacity
                    # checks.  If the mode is full, this observation is
                    # queued and its pending distances must not calibrate the
                    # persistent mode radius before a physical row exists.
                    # Other unrelated candidates waiting under this mode stay
                    # quarantined as well.
                    decision["pending_candidate_count"] = len(
                        self._mode_pending_candidates.get(best_mode, [])
                    )
                    append_is_duplicate_seed = False
                    match_type = "local_variation"
                else:
                    if not (confirmed or force_new_mode_confirmation):
                        # Queue the law distance on this candidate only.  A
                        # different outlier nearest to the same mode receives
                        # its own gain/count/history record.
                        self._append_pending_candidate_distance(
                            best_mode,
                            int(pending_candidate_index),
                            best_distance,
                        )
                        pending_candidate = self._mode_pending_candidates[
                            best_mode
                        ][int(pending_candidate_index)]
                        decision["pending_distance_count"] = len(
                            pending_candidate.get("distance_history", [])
                        )
                        decision["pending_candidate_count"] = len(
                            self._mode_pending_candidates.get(best_mode, [])
                        )
                        return {
                            "action": "queue",
                            "index": -1,
                            "mode_id": best_mode,
                            "support": 0.0,
                            "match_type": "pending_new_dynamics",
                            **decision,
                        }
                    # A positive confirmation (or an explicit force) creates
                    # a new mode.  Remove only the matching candidate so
                    # unrelated pending laws can continue to be confirmed
                    # against the old mode later.
                    if pending_candidate_index is not None:
                        self._discard_pending_candidate(
                            best_mode,
                            candidate_index=pending_candidate_index,
                        )
                        decision["pending_candidate_count"] = len(
                            self._mode_pending_candidates.get(best_mode, [])
                        )
                    mode_id = self._next_mode_id
                    self._next_mode_id += 1
                    self._make_space(law_key_builder)

            if not append_is_duplicate_seed:
                mode_rows = torch.nonzero(
                    self.mode_ids == best_mode, as_tuple=False
                ).flatten()
                if mode_rows.numel() >= self.mode_capacity:
                    # A local variation is a distinct law-level prototype.
                    # Never refresh an existing row just because its mode has
                    # reached the row budget: doing so would overwrite the
                    # law identity and incorrectly turn law variation into a
                    # context-alias update.  Keep the candidate out of the
                    # persistent bank until a row becomes available.
                    return {
                        "action": "queue",
                        "index": -1,
                        "mode_id": best_mode,
                        "support": 0.0,
                        "match_type": "local_variation_capacity",
                        **decision,
                    }
                if bool(self.mode_compressed[mode_rows].any()):
                    self.mode_compressed[mode_rows] = False
                    self.stale_cycles[mode_rows] = 0.0
                if len(self) >= self.capacity and not self._make_space(
                    law_key_builder, protected_mode=best_mode
                ):
                    # The hard bank cap can leave no legal victim even when
                    # this mode itself has spare row capacity (for example a
                    # single-mode bank).  The same identity rule applies:
                    # queue rather than refresh/overwrite an existing law.
                    return {
                        "action": "queue",
                        "index": -1,
                        "mode_id": best_mode,
                        "support": 0.0,
                        "match_type": "bank_capacity",
                        **decision,
                    }
                self.mode_compressed[self.mode_ids == best_mode] = False
        else:
            mode_id = self._next_mode_id
            self._next_mode_id += 1

        index = self._append_batch_rows(
            key.to(device=self.device, dtype=self.keys.dtype),
            delta_theta.to(device=self.device, dtype=self.deltas.dtype),
            law_key.to(device=self.device, dtype=self.law_keys.dtype),
            quality,
            queue,
            torch.tensor([mode_id], device=self.device, dtype=torch.long),
            [window],
        )[0]

        if len(self) > self.capacity:
            raise RuntimeError("prototype admission exceeded the hard bank capacity")
        if append_is_duplicate_seed:
            self._append_adaptive_value(
                self._mode_normal_gains, int(mode_id), gain_value
            )
        else:
            self._record_in_mode_observation(
                int(mode_id),
                distance=float(decision["distance"]),
                prediction_gain=gain_value,
                duplicate=False,
                law_key=law_key,
            )
            if "pending_candidate_count" in decision:
                # A confirmed pending candidate is promoted by the common
                # observation recorder only after capacity checks succeed.
                # Reflect the post-promotion state in the returned decision;
                # a queued capacity result above intentionally retains the
                # candidate count for a later retry.
                decision["pending_candidate_count"] = len(
                    self._mode_pending_candidates.get(int(mode_id), [])
                )
        result = {
            "action": "append",
            "match_type": match_type,
            "index": int(index),
            "mode_id": int(mode_id),
            "support": 1.0,
        }
        result.update(decision)
        return result
        

    @torch.no_grad()
    def step_age(self) -> None:
        """Eager standalone age update.

        TreeEpisodicMemory uses its O(1) logical clock instead. Keeping this
        method preserves the original behavior for an independently used
        MemoryBank.
        """
        if len(self) > 0:
            self.age += 1.0

    def effective_age(self, current_clock: int) -> Tensor:
        """Return chronological age at ``current_clock`` without mutation."""
        current_clock = int(current_clock)
        delta = current_clock - self._age_reference_clock
        if delta < 0:
            raise ValueError("age clock cannot move backwards")
        if delta == 0:
            return self.age
        return self.age + float(delta)

    @torch.no_grad()
    def materialize_age(self, current_clock: int) -> None:
        """Fold the lazy clock offset into ``age`` exactly once."""
        current_clock = int(current_clock)
        delta = current_clock - self._age_reference_clock
        if delta < 0:
            raise ValueError("age clock cannot move backwards")
        if delta > 0 and len(self) > 0:
            self.age.add_(float(delta))
        self._age_reference_clock = current_clock

    def retrieve(
        self,
        query: Tensor,
        retriever: SmoothSparseRetriever,
        update_state: bool = True,
        age_clock: Optional[int] = None,
    ):
        if self.keys.shape[0] == 0:
            return query.new_zeros(self.param_dim), {}
        self._ensure_prototype_state()
        if query.device != self.keys.device:
            raise ValueError(
                f"query is on {query.device}, memory bank is on {self.keys.device}"
            )

        delta_epi, info = retriever(
            query=query,
            keys=self.context_keys,
            deltas=self.deltas,
            usage=self.usage,
            age=(
                self.age
                if age_clock is None
                else self.effective_age(age_clock)
            ),
            write_quality=self.write_quality,
            context_valid=self.context_valid,
        )

        if update_state:
            with torch.no_grad():
                alpha = info["alpha"].to(self.device)
                self.cycle_usage += alpha

        return delta_epi, info

    @torch.no_grad()
    def consolidate_sleep_cycle(
        self,
        usage_decay: float = 0.95,
        effective_usage_threshold: float = 0.0,
    ) -> None:
        """Consolidate wake retrieval mass and update sleep-cycle staleness."""
        if not 0.0 <= usage_decay <= 1.0:
            raise ValueError("usage_decay must be in [0, 1]")
        if effective_usage_threshold < 0.0:
            raise ValueError("effective_usage_threshold must be non-negative")
        if len(self) == 0:
            return

        effectively_used = self.cycle_usage > effective_usage_threshold
        self.usage.mul_(usage_decay).add_(self.cycle_usage)
        self.stale_cycles = torch.where(
            effectively_used,
            torch.zeros_like(self.stale_cycles),
            self.stale_cycles + 1.0,
        )
        self.cycle_usage.zero_()

    @torch.no_grad()
    def keep(self, keep_idx: Tensor) -> None:
        """Keep selected entries while preserving every aligned state field."""
        self._ensure_prototype_state()
        keep_idx = keep_idx.to(device=self.device, dtype=torch.long).sort().values
        self.keys = self.keys[keep_idx]
        self.context_keys = self.context_keys[keep_idx]
        self.context_valid = self.context_valid[keep_idx]
        self.context_support = self.context_support[keep_idx]
        self.deltas = self.deltas[keep_idx]
        self.write_quality = self.write_quality[keep_idx]
        self.queue_weight = self.queue_weight[keep_idx]
        self.law_keys = self.law_keys[keep_idx]
        self.support = self.support[keep_idx]
        self.quality_mass = self.quality_mass[keep_idx]
        self.split_mass = self.split_mass[keep_idx]
        self.mode_ids = self.mode_ids[keep_idx]
        self.mode_compressed = self.mode_compressed[keep_idx]
        self.usage = self.usage[keep_idx]
        self.cycle_usage = self.cycle_usage[keep_idx]
        self.stale_cycles = self.stale_cycles[keep_idx]
        self.age = self.age[keep_idx]
        self.windows = [self.windows[i] for i in keep_idx.detach().cpu().tolist()]
        remaining_modes = set(self.mode_ids.detach().cpu().tolist())
        for values_by_mode in (
            self._mode_duplicate_distances,
            self._mode_distances,
            self._mode_normal_gains,
            self._mode_pending_distances,
            self._mode_pending_gain_ema,
            self._mode_pending_gain_count,
        ):
            for mode_id in list(values_by_mode):
                if int(mode_id) not in remaining_modes:
                    values_by_mode.pop(mode_id, None)
        for mode_id in list(self._mode_pending_candidates):
            if int(mode_id) not in remaining_modes:
                self._mode_pending_candidates.pop(mode_id, None)
            else:
                self._sync_legacy_pending_state(int(mode_id))
        self._sync_fixed_storage()

    @torch.no_grad()
    def append_from(
        self,
        source: "MemoryBank",
        indices: Tensor,
        *,
        deltas: Optional[Tensor] = None,
        node_id: Optional[str] = None,
    ) -> None:
        """Append aligned rows from another bank, optionally replacing deltas."""
        # Repair legacy/intermediate banks before inspecting alias width.  A
        # checkpoint can carry context tensors without the explicit capacity
        # attribute, in which case ``_ensure_prototype_state`` infers it from
        # the serialized tensor shape.
        self._ensure_prototype_state()
        source._ensure_prototype_state()
        target_alias_capacity = int(
            getattr(self, "context_alias_capacity", 3)
        )
        source_alias_capacity = int(
            getattr(source, "context_alias_capacity", 3)
        )
        if (
            self.key_dim != source.key_dim
            or self.param_dim != source.param_dim
            or self.law_dim != source.law_dim
            or target_alias_capacity != source_alias_capacity
        ):
            raise ValueError("source and target MemoryBank dimensions must match")
        indices = indices.to(device=source.device, dtype=torch.long).sort().values
        selected_deltas = source.deltas[indices] if deltas is None else deltas
        selected_deltas = selected_deltas.to(self.device)
        if selected_deltas.shape != (indices.numel(), self.param_dim):
            raise ValueError("replacement deltas have an invalid shape")
        self.keys = torch.cat([self.keys, source.keys[indices].to(self.device)], dim=0)
        self.context_keys = torch.cat(
            [self.context_keys, source.context_keys[indices].to(self.device)], dim=0
        )
        self.context_valid = torch.cat(
            [self.context_valid, source.context_valid[indices].to(self.device)], dim=0
        )
        self.context_support = torch.cat(
            [self.context_support, source.context_support[indices].to(self.device)], dim=0
        )
        self.deltas = torch.cat([self.deltas, selected_deltas], dim=0)
        self.write_quality = torch.cat(
            [
                self.write_quality,
                source.write_quality[indices].to(self.device),
            ],
            dim=0,
        )
        self.queue_weight = torch.cat(
            [
                self.queue_weight,
                source.queue_weight[indices].to(self.device),
            ],
            dim=0,
        )
        self.law_keys = torch.cat(
            [self.law_keys, source.law_keys[indices].to(self.device)], dim=0
        )
        self.support = torch.cat(
            [self.support, source.support[indices].to(self.device)], dim=0
        )
        self.quality_mass = torch.cat(
            [self.quality_mass, source.quality_mass[indices].to(self.device)], dim=0
        )
        self.split_mass = torch.cat(
            [self.split_mass, source.split_mass[indices].to(self.device)], dim=0
        )
        # Mode ids are local to a bank. Allocate fresh ids so unrelated source
        # modes cannot collide after a topology split/merge.
        selected_modes = source.mode_ids[indices]
        mode_map = {}
        remapped = []
        for mode in selected_modes.detach().cpu().tolist():
            if mode not in mode_map:
                mode_map[mode] = self._next_mode_id
                self._next_mode_id += 1
            remapped.append(mode_map[mode])
        self.mode_ids = torch.cat([
            self.mode_ids,
            torch.tensor(remapped, device=self.device, dtype=torch.long),
        ])
        for source_mode, target_mode in mode_map.items():
            for source_values, target_values in (
                (source._mode_duplicate_distances, self._mode_duplicate_distances),
                (source._mode_distances, self._mode_distances),
                (source._mode_normal_gains, self._mode_normal_gains),
                (source._mode_pending_distances, self._mode_pending_distances),
            ):
                if int(source_mode) in source_values:
                    target_values[int(target_mode)] = list(
                        source_values[int(source_mode)]
                    )[-self.adaptive_history_size:]
            if int(source_mode) in source._mode_pending_gain_ema:
                self._mode_pending_gain_ema[int(target_mode)] = float(
                    source._mode_pending_gain_ema[int(source_mode)]
                )
                self._mode_pending_gain_count[int(target_mode)] = int(
                    source._mode_pending_gain_count.get(int(source_mode), 0)
                )
            source_candidates = source._mode_pending_candidates.get(
                int(source_mode), []
            )
            if source_candidates:
                copied_candidates: List[Dict[str, Any]] = []
                for candidate in source_candidates[
                    -source._pending_candidate_capacity() :
                ]:
                    law_key = candidate.get("law_key")
                    if not torch.is_tensor(law_key):
                        continue
                    copied_candidates.append(
                        {
                            "law_key": law_key.detach().to(self.device).reshape(-1).clone(),
                            "gain_ema": float(candidate.get("gain_ema", 0.0)),
                            "count": int(candidate.get("count", 0)),
                            "distance_history": [
                                float(value)
                                for value in candidate.get("distance_history", [])
                            ][-self.adaptive_history_size :],
                        }
                    )
                if copied_candidates:
                    self._mode_pending_candidates[int(target_mode)] = (
                        copied_candidates
                    )
                    self._sync_legacy_pending_state(int(target_mode))
        self.mode_compressed = torch.cat(
            [self.mode_compressed, source.mode_compressed[indices].to(self.device)], dim=0
        )
        self.usage = torch.cat([self.usage, source.usage[indices].to(self.device)], dim=0)
        self.cycle_usage = torch.cat(
            [self.cycle_usage, source.cycle_usage[indices].to(self.device)], dim=0
        )
        self.stale_cycles = torch.cat(
            [self.stale_cycles, source.stale_cycles[indices].to(self.device)], dim=0
        )
        self.age = torch.cat([self.age, source.age[indices].to(self.device)], dim=0)
        for index in indices.detach().cpu().tolist():
            window = source.windows[index]
            if window is not None and node_id is not None:
                window = replace(window, node_id=node_id)
            self.windows.append(window)
        self._sync_fixed_storage()

    @torch.no_grad()
    def clear(self) -> None:
        """Remove every entry while preserving device, dtype, and dimensions."""
        self._ensure_prototype_state()
        self.keys = self.keys[:0]
        self.context_keys = self.context_keys[:0]
        self.context_valid = self.context_valid[:0]
        self.context_support = self.context_support[:0]
        self.deltas = self.deltas[:0]
        self.write_quality = self.write_quality[:0]
        self.queue_weight = self.queue_weight[:0]
        self.law_keys = self.law_keys[:0]
        self.support = self.support[:0]
        self.quality_mass = self.quality_mass[:0]
        self.split_mass = self.split_mass[:0]
        self.mode_ids = self.mode_ids[:0]
        self.mode_compressed = self.mode_compressed[:0]
        self.usage = self.usage[:0]
        self.cycle_usage = self.cycle_usage[:0]
        self.stale_cycles = self.stale_cycles[:0]
        self.age = self.age[:0]
        self.windows = []
        self._mode_duplicate_distances.clear()
        self._mode_distances.clear()
        self._mode_normal_gains.clear()
        self._mode_pending_distances.clear()
        self._mode_pending_gain_ema.clear()
        self._mode_pending_gain_count.clear()
        self._mode_pending_candidates.clear()
        self._bind_fixed_storage_views(0)
    
    @torch.no_grad()
    def prune(self) -> None:
        """
        Enforce hard capacity with the same usage/staleness semantics as sleep.

        A small fresh-memory bonus prevents a just-written residual from losing
        immediately to old cumulative usage when capacity is reached.
        """
        score = (
            torch.log1p(self.usage + self.cycle_usage)
            - self.stale_cycles
            - 0.01 * self.age
            + (self.stale_cycles < 2.0).to(self.usage.dtype)
        )
        keep_idx = torch.topk(score, k=self.capacity).indices.sort().values

        self.keep(keep_idx)
