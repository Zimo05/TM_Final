"""Counterfactual residual probe for unexpanded coarse frontier regions."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class RegionalResidualProbe:
    """Vectorized residual regression and sequence pooling for one region."""

    assignment: Tensor
    assignment_confidence: Tensor
    leaf_mass: Tensor
    pooled_embedding: Tensor
    stop_distortion: Tensor
    leaf_distortion: Tensor
    relative_gain: Tensor
    refinement_gain: Tensor
    expand_probability: Tensor


@dataclass(frozen=True)
class CounterfactualEnergyProbe:
    """Detached teacher targets derived from true coarse/leaf NLL energy."""

    teacher: Tensor
    expand_target: Tensor
    conditional_leaf_credit: Tensor
    smoothed_leaf_credit: Tensor
    assignment_confidence: Tensor
    fine_energy: Tensor
    observed_gain: Tensor


def regional_residual_probe(
    residual_signatures: Tensor,
    sequence_embeddings: Tensor,
    coarse_responsibility: Tensor,
    leaf_centers: Tensor,
    coarse_center: Tensor | None = None,
    *,
    residual_temperature: float,
    gain_temperature: float,
    complexity_weight: float,
    eps: float = 1e-12,
) -> RegionalResidualProbe:
    """Compare one fixed coarse residual center with descendant prototypes.

    This performs one ``[S, J, P]`` residual-distance operation. Hawkes NLL is
    never evaluated per descendant leaf. ``leaf_centers`` and the resulting
    assignments are detached so a center cannot move merely to attract the
    sequences it already receives. Gradients flow only through the pooled
    sequence embeddings used by the downstream Router probe.
    """
    if residual_signatures.ndim != 2:
        raise ValueError("residual_signatures must have shape [S, P]")
    if sequence_embeddings.ndim != 2:
        raise ValueError("sequence_embeddings must have shape [S, Z]")
    if coarse_responsibility.ndim != 1:
        raise ValueError("coarse_responsibility must have shape [S]")
    if leaf_centers.ndim != 2:
        raise ValueError("leaf_centers must have shape [J, P]")
    if coarse_center is not None and coarse_center.ndim != 1:
        raise ValueError("coarse_center must have shape [P]")
    sequence_count, parameter_dim = residual_signatures.shape
    if sequence_embeddings.size(0) != sequence_count:
        raise ValueError("sequence embeddings and residuals must align")
    if coarse_responsibility.numel() != sequence_count:
        raise ValueError("coarse responsibility and residuals must align")
    if leaf_centers.size(1) != parameter_dim or leaf_centers.size(0) < 2:
        raise ValueError("leaf centers must contain at least two [P] rows")
    if coarse_center is not None and coarse_center.numel() != parameter_dim:
        raise ValueError("coarse_center must contain P values")
    if residual_temperature <= 0.0 or gain_temperature <= 0.0:
        raise ValueError("probe temperatures must be positive")
    if complexity_weight < 0.0:
        raise ValueError("complexity_weight must be non-negative")

    residual = residual_signatures.detach()
    centers = leaf_centers.detach().to(residual)
    if coarse_center is None:
        warnings.warn(
            "regional_residual_probe called without coarse_center; using "
            "the uniform descendant-prototype center for compatibility. "
            "Synchronize the training objective to pass the target-mass "
            "weighted raw residual center.",
            RuntimeWarning,
            stacklevel=2,
        )
        region_center = centers.mean(dim=0)
    else:
        region_center = coarse_center.detach().to(residual).reshape(-1)
    weight = coarse_responsibility.detach().to(residual)
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("residual signatures must be finite")
    if not bool(torch.isfinite(centers).all()):
        raise ValueError("leaf centers must be finite")
    if not bool(torch.isfinite(region_center).all()):
        raise ValueError("coarse center must be finite")
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0.0).any()):
        raise ValueError("coarse responsibility must be finite and non-negative")
    total_weight = weight.sum()
    if float(total_weight) <= 0.0:
        raise ValueError("coarse responsibility must have positive total mass")

    squared_distance = (
        residual[:, None, :] - centers[None, :, :]
    ).square().sum(dim=-1)
    stop_distortion = (
        weight
        * (residual - region_center).square().sum(dim=-1)
    ).sum() / total_weight.clamp_min(eps)
    normalized_distance = (
        squared_distance
        / stop_distortion.detach().clamp_min(eps)
    )
    assignment = F.softmax(
        -normalized_distance / residual_temperature,
        dim=-1,
    ).detach()
    leaf_count = leaf_centers.size(0)
    entropy = -(
        assignment.clamp_min(eps) * assignment.clamp_min(eps).log()
    ).sum(dim=-1)
    assignment_confidence = (
        1.0 - entropy / residual.new_tensor(float(leaf_count)).log()
    ).clamp(0.0, 1.0).detach()
    weighted_assignment = (
        weight[:, None]
        * assignment_confidence[:, None]
        * assignment
    )
    leaf_mass = weighted_assignment.sum(dim=0)
    pooled_embedding = (
        weighted_assignment.transpose(0, 1)
        @ sequence_embeddings
    ) / leaf_mass.clamp_min(eps).unsqueeze(-1)

    leaf_distortion = (
        weight[:, None] * assignment * squared_distance
    ).sum() / total_weight.clamp_min(eps)
    relative_gain = (
        stop_distortion
        - leaf_distortion
    ) / stop_distortion.detach().clamp_min(eps)
    complexity = residual.new_tensor(float(leaf_count)).log()
    refinement_gain = (
        relative_gain - complexity_weight * complexity
    ).detach()
    expand_probability = torch.sigmoid(
        refinement_gain / gain_temperature
    ).detach()
    return RegionalResidualProbe(
        assignment=assignment,
        assignment_confidence=assignment_confidence,
        leaf_mass=leaf_mass.detach(),
        pooled_embedding=pooled_embedding,
        stop_distortion=stop_distortion.detach(),
        leaf_distortion=leaf_distortion.detach(),
        relative_gain=relative_gain.detach(),
        refinement_gain=refinement_gain,
        expand_probability=expand_probability,
    )


def counterfactual_energy_probe(
    coarse_energy: Tensor,
    leaf_energy: Tensor,
    coarse_responsibility: Tensor,
    leaf_prior: Tensor,
    *,
    teacher_temperature: float,
    gain_temperature: float,
    leaf_smoothing: float,
    eps: float = 1e-12,
) -> CounterfactualEnergyProbe:
    """Construct stop/leaf evidence without consulting ``p_expand``.

    Energies are expected to be mean per-event Hawkes NLL values.  The
    returned teacher is detached: it supervises the expansion head, Router,
    and leaf-local calibration, but cannot move the energy targets merely to
    make their own assignment easier.
    """
    if coarse_energy.ndim != 1:
        raise ValueError("coarse_energy must have shape [S]")
    if leaf_energy.ndim != 2 or leaf_energy.size(0) != coarse_energy.numel():
        raise ValueError("leaf_energy must have shape [S, Kp]")
    if leaf_energy.size(1) <= 0:
        raise ValueError("leaf_energy must contain at least one probe leaf")
    if coarse_responsibility.shape != coarse_energy.shape:
        raise ValueError("coarse_responsibility must have shape [S]")
    if leaf_prior.shape != (leaf_energy.size(1),):
        raise ValueError("leaf_prior must have shape [Kp]")
    if teacher_temperature <= 0.0 or gain_temperature <= 0.0:
        raise ValueError("probe temperatures must be positive")
    if not 0.0 <= leaf_smoothing < 1.0:
        raise ValueError("leaf_smoothing must lie in [0, 1)")
    if not bool(torch.isfinite(coarse_energy).all()):
        raise ValueError("coarse_energy must be finite")
    if not bool(torch.isfinite(leaf_energy).all()):
        raise ValueError("leaf_energy must be finite")
    if (
        not bool(torch.isfinite(coarse_responsibility).all())
        or bool((coarse_responsibility < 0.0).any())
    ):
        raise ValueError("coarse_responsibility must be finite and non-negative")
    if (
        not bool(torch.isfinite(leaf_prior).all())
        or bool((leaf_prior < 0.0).any())
        or float(leaf_prior.sum()) <= 0.0
    ):
        raise ValueError("leaf_prior must be finite, non-negative, and nonzero")

    with torch.no_grad():
        all_energy = torch.cat(
            (coarse_energy[:, None], leaf_energy), dim=1
        )
        teacher = F.softmax(
            -all_energy / teacher_temperature,
            dim=1,
        )
        expand_target = 1.0 - teacher[:, 0]
        # Compute the algebraically identical conditional distribution in its
        # own softmax so a very strong stop option cannot underflow every leaf
        # numerator to zero.
        leaf_credit = F.softmax(
            -leaf_energy / teacher_temperature,
            dim=1,
        )
        leaf_count = leaf_energy.size(1)
        smoothed_credit = (
            (1.0 - leaf_smoothing) * leaf_credit
            + leaf_smoothing / float(leaf_count)
        )
        if leaf_count == 1:
            confidence = torch.ones_like(expand_target)
        else:
            entropy = -(
                leaf_credit.clamp_min(eps)
                * leaf_credit.clamp_min(eps).log()
            ).sum(dim=1)
            confidence = (
                1.0
                - entropy / coarse_energy.new_tensor(float(leaf_count)).log()
            ).clamp(0.0, 1.0)
        prior = leaf_prior / leaf_prior.sum().clamp_min(eps)
        fine_energy = -gain_temperature * torch.logsumexp(
            prior.clamp_min(eps).log()[None, :]
            - leaf_energy / gain_temperature,
            dim=1,
        )
        weight = coarse_responsibility
        observed_gain = (
            weight * (coarse_energy - fine_energy).clamp_min(0.0)
        ).sum() / weight.sum().clamp_min(eps)
    return CounterfactualEnergyProbe(
        teacher=teacher,
        expand_target=expand_target,
        conditional_leaf_credit=leaf_credit,
        smoothed_leaf_credit=smoothed_credit,
        assignment_confidence=confidence,
        fine_energy=fine_energy,
        observed_gain=observed_gain,
    )
