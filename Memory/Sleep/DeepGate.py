"""Differentiable Light-to-Deep Sleep coordination gate.

The gate consumes detached, inexpensive statistics produced after Light
Sleep.  Only the gate parameters are differentiated; topology proposals and
their estimated prediction-compression gain remain on the slow structural
timescale.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DeepSleepGate(nn.Module):
    """Monotone temporal hazard gate with a Hard-Concrete execution sample."""

    feature_names = (
        "residual",
        "memory",
        "topology",
    )

    def __init__(
        self,
        *,
        accumulator_decay: float = 0.8,
        evidence_temperature: float = 1.0,
        hard_concrete_temperature: float = 2.0 / 3.0,
        hard_concrete_gamma: float = -0.1,
        hard_concrete_zeta: float = 1.1,
        execution_threshold: float = 0.5,
        bias_initial: float = -2.0,
        weight_initial: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 <= accumulator_decay < 1.0:
            raise ValueError("accumulator_decay must lie in [0, 1)")
        if evidence_temperature <= 0.0:
            raise ValueError("evidence_temperature must be positive")
        if hard_concrete_temperature <= 0.0:
            raise ValueError("hard_concrete_temperature must be positive")
        if not hard_concrete_gamma < 0.0:
            raise ValueError("hard_concrete_gamma must be negative")
        if not hard_concrete_zeta > 1.0:
            raise ValueError("hard_concrete_zeta must exceed one")
        if not 0.0 < execution_threshold < 1.0:
            raise ValueError("execution_threshold must lie in (0, 1)")
        if weight_initial <= 0.0:
            raise ValueError("weight_initial must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        raw_initial = math.log(math.expm1(float(weight_initial)))
        self.bias = nn.Parameter(torch.tensor(float(bias_initial)))
        self.raw_weights = nn.Parameter(torch.full(
            (len(self.feature_names),),
            raw_initial,
        ))
        self.register_buffer("accumulator", torch.zeros(()))
        self.register_buffer("cycle_count", torch.zeros((), dtype=torch.long))

        self.accumulator_decay = float(accumulator_decay)
        self.evidence_temperature = float(evidence_temperature)
        self.hard_concrete_temperature = float(
            hard_concrete_temperature
        )
        self.hard_concrete_gamma = float(hard_concrete_gamma)
        self.hard_concrete_zeta = float(hard_concrete_zeta)
        self.execution_threshold = float(execution_threshold)
        self.eps = float(eps)

    @property
    def positive_weights(self) -> Tensor:
        """Return pressure coefficients constrained to be strictly positive."""
        return F.softplus(self.raw_weights)

    @torch.no_grad()
    def reset_after_deep(self) -> None:
        """Clear evidence after paying for a Deep proposal evaluation."""
        self.accumulator.zero_()

    def forward(
        self,
        features: Tensor | Sequence[float],
        *,
        split_demand: float | Tensor,
        deep_availability: float | Tensor,
        sample: bool | None = None,
    ) -> Dict[str, Tensor | bool]:
        reference = self.bias
        feature_tensor = torch.as_tensor(
            features,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(-1).detach()
        if feature_tensor.shape != (len(self.feature_names),):
            raise ValueError(
                "features must contain residual, memory, and topology values"
            )
        if not bool(torch.isfinite(feature_tensor).all()):
            raise FloatingPointError("Deep Sleep features must be finite")
        if bool(((feature_tensor < 0.0) | (feature_tensor > 1.0)).any()):
            raise ValueError("Deep Sleep features must lie in [0, 1]")

        split = torch.as_tensor(
            split_demand,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(()).detach().clamp(0.0, 1.0)
        availability = torch.as_tensor(
            deep_availability,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(()).detach().clamp(0.0, 1.0)

        evidence = self.bias + (
            self.positive_weights * feature_tensor
        ).sum()
        previous = self.accumulator.detach().clone().to(reference)
        accumulated = (
            self.accumulator_decay * previous
            + (1.0 - self.accumulator_decay) * evidence
        )
        with torch.no_grad():
            self.accumulator.copy_(accumulated.detach())
            self.cycle_count.add_(1)

        base_probability = torch.sigmoid(
            accumulated / self.evidence_temperature
        )
        # Smooth probabilistic OR preserves Split demand without the sticky
        # split_queue > 0 immediate-trigger failure mode.  Availability is
        # deliberately applied only after all sources of structural pressure
        # are combined: pressure answers whether Deep is needed; availability
        # answers whether Deep is currently allowed.
        joint_pressure = 1.0 - (
            1.0 - base_probability
        ) * (1.0 - split)
        probability = (availability * joint_pressure).clamp(0.0, 1.0)
        interior_probability = probability.clamp(
            self.eps, 1.0 - self.eps
        )
        log_alpha = torch.logit(interior_probability)
        # Preserve the exact availability boundary.  Merely clamping zero to
        # eps would leave a small chance that Hard-Concrete noise opens Deep
        # while availability is exactly zero.
        log_alpha = torch.where(
            probability <= 0.0,
            log_alpha.new_tensor(float("-inf")),
            torch.where(
                probability >= 1.0,
                log_alpha.new_tensor(float("inf")),
                log_alpha,
            ),
        )

        use_sample = self.training if sample is None else bool(sample)
        if use_sample:
            uniform = torch.rand_like(log_alpha).clamp(
                self.eps,
                1.0 - self.eps,
            )
            logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        else:
            logistic_noise = torch.zeros_like(log_alpha)
        relaxed = torch.sigmoid(
            (log_alpha + logistic_noise)
            / self.hard_concrete_temperature
        )
        stretched = (
            relaxed
            * (self.hard_concrete_zeta - self.hard_concrete_gamma)
            + self.hard_concrete_gamma
        )
        gate = stretched.clamp(0.0, 1.0)
        hard_value = (
            gate >= self.execution_threshold
        ).to(gate.dtype)
        straight_through_gate = hard_value + gate - gate.detach()
        activation_probability = torch.sigmoid(
            log_alpha
            - self.hard_concrete_temperature
            * math.log(
                -self.hard_concrete_gamma / self.hard_concrete_zeta
            )
        )

        return {
            "features": feature_tensor,
            "weights": self.positive_weights,
            "evidence": evidence,
            "accumulator": accumulated,
            "base_probability": base_probability,
            "joint_pressure": joint_pressure,
            # Compatibility name: this is now pressure before availability.
            "pressure_probability": joint_pressure,
            "availability": availability,
            "probability": probability,
            "activation_probability": activation_probability,
            "gate": gate,
            "straight_through_gate": straight_through_gate,
            "hard_gate": bool(hard_value.detach().item()),
        }

    def objective(
        self,
        output: Mapping[str, Any],
        *,
        estimated_gain: float | Tensor,
        computation_cost: float,
        prior_probability: float,
        prior_weight: float,
    ) -> Dict[str, Tensor]:
        """Computation-aware gate loss with detached structural reward."""
        if computation_cost < 0.0:
            raise ValueError("computation_cost must be non-negative")
        if not 0.0 < prior_probability < 1.0:
            raise ValueError("prior_probability must lie in (0, 1)")
        if prior_weight < 0.0:
            raise ValueError("prior_weight must be non-negative")

        probability = torch.as_tensor(output["probability"])
        activation_probability = torch.as_tensor(
            output["activation_probability"]
        )
        gain = torch.as_tensor(
            estimated_gain,
            device=probability.device,
            dtype=probability.dtype,
        ).reshape(()).detach()
        prior = probability.new_tensor(prior_probability)
        p = probability.clamp(self.eps, 1.0 - self.eps)
        kl = (
            p * (torch.log(p) - torch.log(prior))
            + (1.0 - p)
            * (torch.log1p(-p) - torch.log1p(-prior))
        )
        reward_term = -p * gain
        cost_term = float(computation_cost) * activation_probability
        prior_term = float(prior_weight) * kl
        loss = reward_term + cost_term + prior_term
        return {
            "loss": loss,
            "reward_term": reward_term,
            "cost_term": cost_term,
            "prior_term": prior_term,
            "kl": kl,
            "estimated_gain": gain,
        }


__all__ = ["DeepSleepGate"]
