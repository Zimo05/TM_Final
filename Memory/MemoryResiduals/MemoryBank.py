import math
from dataclasses import dataclass, replace
from numbers import Real
from typing import Callable, Dict, List, Optional, Tuple

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


def _signed_hash_projection(feature: Tensor, output_dim: int) -> Tensor:
    """Deterministically project a physical-law feature without parameters."""
    flat = feature.reshape(-1)
    positions = torch.arange(flat.numel(), device=flat.device, dtype=torch.long)
    buckets = torch.remainder(positions * 2654435761, output_dim)
    signs = torch.where(
        torch.remainder(positions * 2246822519 + 3266489917, 2) == 0,
        flat.new_ones(()),
        -flat.new_ones(()),
    )
    projected = flat.new_zeros(output_dim)
    projected.index_add_(0, buckets, signs * flat)
    return F.normalize(projected.reshape(1, -1), dim=-1).reshape(-1)


def effective_hawkes_law_key(
    semantic_theta: Tensor,
    delta_theta: Tensor,
    decays: Tensor,
    *,
    num_event_types: int,
    num_basis: int,
    key_dim: int,
) -> Tensor:
    """Identity of the effective physical Hawkes law.

    The key is derived from ``semantic_theta + delta_theta`` after mapping the
    unconstrained baseline and kernels to physical parameters.  Consequently
    a Light-Sleep semantic rebase that preserves the effective law also
    preserves this identity.
    """
    effective = semantic_theta.detach().reshape(-1) + delta_theta.detach().reshape(-1)
    expected = num_event_types + num_event_types * num_event_types * num_basis
    if effective.numel() != expected:
        raise ValueError(f"effective Hawkes theta must contain {expected} values")
    decay = decays.detach().reshape(-1).to(effective)
    if decay.numel() != num_basis or bool((decay <= 0).any()):
        raise ValueError("decays must contain one positive value per basis")
    raw_mu = effective[:num_event_types]
    raw_w = effective[num_event_types:].reshape(
        num_event_types, num_event_types, num_basis
    )
    mu = F.softplus(raw_mu)
    integrated_excitation = (F.softplus(raw_w) / decay.reshape(1, 1, -1)).sum(-1)
    return _signed_hash_projection(
        torch.cat([mu, integrated_excitation.reshape(-1)]), key_dim
    )


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
        keys: Tensor,        # [M, d_k]
        deltas: Tensor,      # [M, param_dim]
        usage: Tensor,       # [M]
        age: Tensor,         # [M]
        write_quality: Optional[Tensor] = None,  # [M]
        keep_gate: Optional[Tensor] = None,      # [M]
        null_logit: Optional[float | Tensor] = None,
    ):
        if keys.shape[0] == 0:
            param_dim = deltas.shape[-1]
            return query.new_zeros(param_dim), {}

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
        keys: Tensor,        # [R, M, d_k]
        deltas: Tensor,      # [R, M, P] or shared [B, M, P]
        usage: Tensor,       # [R, M]
        age: Tensor,         # [R, M]
        valid_mask: Tensor,  # [R, M]
        write_quality: Optional[Tensor] = None,  # [R, M]
        keep_gate: Optional[Tensor] = None,      # [R, M]
        null_logit: Optional[float | Tensor] = None,
        row_bank_indices: Optional[Tensor] = None,  # [R] into shared B
    ):
        """Retrieve from padded banks with one independently normalized row.

        No probability mass can enter padded rows. Apart from parallel
        execution, row ``r`` is mathematically identical to ``forward`` on
        ``keys[r, valid_mask[r]]`` and the corresponding bank tensors.
        """
        if query.ndim != 2:
            raise ValueError("query must have shape [R, d_k]")
        if keys.ndim != 3 or keys.shape[:2] != valid_mask.shape:
            raise ValueError("keys and valid_mask must align as [R, M, ...]")
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
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.key_dim = key_dim
        self.param_dim = param_dim
        self.capacity = capacity
        self.device = torch.device(device)

        self.keys = torch.empty(0, key_dim, device=self.device)
        self.deltas = torch.empty(0, param_dim, device=self.device)
        self.write_quality = torch.empty(0, device=self.device)
        self.queue_weight = torch.empty(0, device=self.device)
        # Prototype/mode state.  ``keys`` remain the retrieval (context)
        # identity; ``law_keys`` are used only for duplicate and dynamics-mode
        # matching.
        self.law_keys = torch.empty(0, key_dim, device=self.device)
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

    def __len__(self) -> int:
        return self.keys.shape[0]

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
        self.duplicate_threshold = float(duplicate_threshold)
        self.mode_threshold = float(mode_threshold)
        self.mode_capacity = int(mode_capacity)
        self.ema_beta_min = float(ema_beta_min)
        self.ema_beta_max = float(ema_beta_max)
        self.retention_support_weight = float(retention_support_weight)
        self.retention_usage_weight = float(retention_usage_weight)
        self.retention_stale_weight = float(retention_stale_weight)
        self.retention_age_weight = float(retention_age_weight)

    @torch.no_grad()
    def _ensure_prototype_state(self) -> None:
        """Repair old checkpoints and topology operations lazily."""
        count = len(self)
        if self.law_keys.shape != (count, self.key_dim):
            self.law_keys = self.keys.detach().clone()
        if self.support.shape != (count,):
            self.support = self.keys.new_ones(count)
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
        blended_key = (1.0 - beta) * self.keys[index] + beta * key.reshape(-1)
        self.keys[index] = F.normalize(blended_key.reshape(1, -1), dim=-1).reshape(-1)
        if law_key_builder is not None:
            refreshed_law = law_key_builder(self.deltas[index])
        else:
            refreshed_law = (1.0 - beta) * self.law_keys[index] + beta * law_key.reshape(-1)
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
        return {
            "action": "refresh",
            "index": int(index),
            "mode_id": int(self.mode_ids[index].item()),
            "support": float(self.support[index].item()),
        }

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
            modes = torch.unique(self.mode_ids).detach().cpu().tolist()
            compressible = [
                int(mode) for mode in modes
                if int((self.mode_ids == int(mode)).sum().item()) > 1
            ]
            if compressible:
                victim = min(compressible, key=lambda mode: float(self._mode_retention(mode)))
                self._compress_mode(victim, law_key_builder)
                continue
            # Every mode is already represented by one archive/singleton.
            # Only now is true eviction unavoidable under the hard bank cap.
            eviction_candidates = [
                int(mode) for mode in modes if int(mode) != protected_mode
            ]
            if not eviction_candidates:
                return False
            victim = min(
                eviction_candidates,
                key=lambda mode: float(self._mode_retention(int(mode))),
            )
            keep = self.mode_ids != int(victim)
            self.keep(torch.nonzero(keep, as_tuple=False).flatten())
            break
        return True
    
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

        # Persistent memory is state, not a retained autograd graph. Gradients
        # still flow through the differentiable retrieval weights.
        key = F.normalize(key.detach().reshape(1, -1), dim=-1)
        delta_theta = delta_theta.detach().reshape(1, -1)
        if law_key is None:
            law_key = key.reshape(-1)
        law_key = F.normalize(
            law_key.detach().reshape(1, -1).to(device=self.device, dtype=self.keys.dtype),
            dim=-1,
        )
        if law_key.shape != (1, self.key_dim):
            raise ValueError(f"law_key must contain {self.key_dim} values")
        memory_values_finite = (
            torch.isfinite(key).all()
            & torch.isfinite(delta_theta).all()
        )
        if not bool(memory_values_finite):
            raise FloatingPointError(
                "episodic-memory key or residual contains NaN or Inf"
            )

        self._ensure_prototype_state()
        if len(self) > 0:
            similarities = self.law_keys @ law_key.reshape(-1)
            best_index = int(torch.argmax(similarities).item())
            best_similarity = float(similarities[best_index].item())
            best_mode = int(self.mode_ids[best_index].item())
            if best_similarity >= self.duplicate_threshold:
                if bool(self.mode_compressed[best_index]):
                    self.mode_compressed[best_index] = False
                return self._refresh_prototype(
                    best_index,
                    key=key,
                    delta_theta=delta_theta,
                    law_key=law_key,
                    window=window,
                    quality=quality,
                    queue=queue,
                    law_key_builder=law_key_builder,
                )
            if best_similarity >= self.mode_threshold:
                mode_rows = torch.nonzero(
                    self.mode_ids == best_mode, as_tuple=False
                ).flatten()
                if bool(self.mode_compressed[mode_rows].any()):
                    self.mode_compressed[mode_rows] = False
                    self.stale_cycles[mode_rows] = 0.0
                if mode_rows.numel() >= self.mode_capacity:
                    return self._refresh_prototype(
                        best_index,
                        key=key,
                        delta_theta=delta_theta,
                        law_key=law_key,
                        window=window,
                        quality=quality,
                        queue=queue,
                        law_key_builder=law_key_builder,
                    )
                mode_id = best_mode
                if len(self) >= self.capacity and not self._make_space(
                    law_key_builder, protected_mode=best_mode
                ):
                    return self._refresh_prototype(
                        best_index,
                        key=key,
                        delta_theta=delta_theta,
                        law_key=law_key,
                        window=window,
                        quality=quality,
                        queue=queue,
                        law_key_builder=law_key_builder,
                    )
                self.mode_compressed[self.mode_ids == best_mode] = False
            else:
                mode_id = self._next_mode_id
                self._next_mode_id += 1
                self._make_space(law_key_builder)
        else:
            mode_id = self._next_mode_id
            self._next_mode_id += 1

        self.keys = torch.cat([self.keys, key.to(self.device)], dim=0)
        self.deltas = torch.cat([self.deltas, delta_theta.to(self.device)], dim=0)
        self.law_keys = torch.cat([self.law_keys, law_key], dim=0)
        self.support = torch.cat([self.support, torch.ones_like(quality)], dim=0)
        self.quality_mass = torch.cat([self.quality_mass, quality], dim=0)
        self.split_mass = torch.cat([self.split_mass, queue], dim=0)
        self.mode_ids = torch.cat([
            self.mode_ids,
            torch.tensor([mode_id], device=self.device, dtype=torch.long),
        ])
        self.mode_compressed = torch.cat([
            self.mode_compressed,
            torch.zeros(1, device=self.device, dtype=torch.bool),
        ])
        self.write_quality = torch.cat(
            [self.write_quality, quality],
            dim=0,
        )
        self.queue_weight = torch.cat(
            [self.queue_weight, queue],
            dim=0,
        )

        self.windows.append(window)

        self.usage = torch.cat([self.usage, torch.zeros(1, device=self.device)])
        self.cycle_usage = torch.cat(
            [self.cycle_usage, torch.zeros(1, device=self.device)]
        )
        self.stale_cycles = torch.cat(
            [self.stale_cycles, torch.zeros(1, device=self.device)]
        )
        self.age = torch.cat([self.age, torch.zeros(1, device=self.device)])

        if len(self) > self.capacity:
            raise RuntimeError("prototype admission exceeded the hard bank capacity")
        return {
            "action": "append",
            "index": len(self) - 1,
            "mode_id": int(mode_id),
            "support": 1.0,
        }
        

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
        if query.device != self.keys.device:
            raise ValueError(
                f"query is on {query.device}, memory bank is on {self.keys.device}"
            )

        delta_epi, info = retriever(
            query=query,
            keys=self.keys,
            deltas=self.deltas,
            usage=self.usage,
            age=(
                self.age
                if age_clock is None
                else self.effective_age(age_clock)
            ),
            write_quality=self.write_quality,
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
        if self.key_dim != source.key_dim or self.param_dim != source.param_dim:
            raise ValueError("source and target MemoryBank dimensions must match")
        self._ensure_prototype_state()
        source._ensure_prototype_state()
        indices = indices.to(device=source.device, dtype=torch.long).sort().values
        selected_deltas = source.deltas[indices] if deltas is None else deltas
        selected_deltas = selected_deltas.to(self.device)
        if selected_deltas.shape != (indices.numel(), self.param_dim):
            raise ValueError("replacement deltas have an invalid shape")
        self.keys = torch.cat([self.keys, source.keys[indices].to(self.device)], dim=0)
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

    @torch.no_grad()
    def clear(self) -> None:
        """Remove every entry while preserving device, dtype, and dimensions."""
        self.keys = self.keys[:0]
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
