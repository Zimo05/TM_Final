"""Configuration for posterior-guided Wake routing and retrieval."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FrontierRoutingConfig:
    """Hyperparameters corresponding to the Wake construction equations.

    ``frontier_budget`` limits the number of active experts in one
    prediction. It does not limit the number of leaves in the stored tree.
    """

    frontier_budget: int = 4
    frontier_min_experts: int = 2
    routing_temperature: float = 1.5
    # Deprecated compatibility field.  Search now uses the exact
    # fixed-prior distribution softmax(a / tau + log pi_fixed), so no
    # post-softmax exploration mixture is applied.
    exploration_epsilon: float = 0.0

    semantic_weight: float = 1.0
    # Deprecated compatibility field.  The fixed topology prior always has
    # coefficient one so the Router cannot relearn or temper topology mass.
    prior_weight: float = 1.0
    # Deprecated active-frontier-v1 fields retained for checkpoint loading.
    # They are deliberately ignored by the v2 routing hot path.
    data_weight: float = 0.0
    prior_alpha: float = 1.0
    prior_kappa: float = 0.25
    prototype_epsilon: float = 1e-4
    entropy_bonus: float = 0.0
    visit_bonus: float = 0.0
    # Optional leaf target mass in current ``tree.leaf_ids`` order. ``None``
    # initializes uniform mass over the current leaves. Later merge/split
    # operations conserve that mass by node ID. Local child priors are obtained
    # by summing mass over descendant leaves, so an irregular tree has no
    # shallow-leaf advantage.
    target_leaf_mass: Optional[Tuple[float, ...]] = None

    # U(i,n) = stopgrad(m(i,n)) * (G_n + lambda_c C(i,n)) - compute_cost.
    confidence_weight: float = 0.25
    expansion_compute_cost: float = 0.05
    expansion_gain_decay: float = 0.95
    # Cold-start value is just above the K=2 expansion cost at mass 1/2.
    # This lets the model observe at least one non-root refinement and learn
    # its EMA gain, without forcing every prefix all the way to K_max.
    default_expansion_gain: float = 0.12

    posterior_temperature: float = 1.0
    credible_mass: float = 0.90
    owner_confidence_threshold: float = 0.80
    max_writes_per_sequence: int = 8

    def validate(self) -> None:
        if self.frontier_budget < 2:
            raise ValueError("frontier_budget must be at least 2")
        if not 2 <= self.frontier_min_experts <= self.frontier_budget:
            raise ValueError(
                "frontier_min_experts must lie in [2, frontier_budget]"
            )
        if self.routing_temperature <= 0.0:
            raise ValueError("routing_temperature must be positive")
        if not 0.0 <= self.exploration_epsilon <= 1.0:
            raise ValueError("exploration_epsilon must be in [0, 1]")
        if self.target_leaf_mass is not None:
            if not self.target_leaf_mass:
                raise ValueError("target_leaf_mass cannot be empty")
            if any(value < 0.0 for value in self.target_leaf_mass):
                raise ValueError("target_leaf_mass must be non-negative")
            if sum(self.target_leaf_mass) <= 0.0:
                raise ValueError("target_leaf_mass must have positive total mass")
        if self.confidence_weight < 0.0:
            raise ValueError("confidence_weight must be non-negative")
        if self.expansion_compute_cost < 0.0:
            raise ValueError("expansion_compute_cost must be non-negative")
        if not 0.0 <= self.expansion_gain_decay < 1.0:
            raise ValueError("expansion_gain_decay must lie in [0, 1)")
        if self.default_expansion_gain < 0.0:
            raise ValueError(
                "default_expansion_gain must be non-negative"
            )
        if self.posterior_temperature <= 0.0:
            raise ValueError("posterior_temperature must be positive")
        if not 0.0 < self.credible_mass <= 1.0:
            raise ValueError("credible_mass must lie in (0, 1]")
        if not 0.0 < self.owner_confidence_threshold <= 1.0:
            raise ValueError(
                "owner_confidence_threshold must lie in (0, 1]"
            )
        if not 4 <= self.max_writes_per_sequence <= 8:
            raise ValueError(
                "max_writes_per_sequence must lie in [4, 8]"
            )
