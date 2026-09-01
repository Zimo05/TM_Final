"""Active-frontier routing for the dynamic Hawkes tree."""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn


class FrontierRoutingOutput(NamedTuple):
    """Responsibility on the actually computed frontier slots."""

    responsibility: torch.Tensor
    log_responsibility: torch.Tensor
    router_logits: torch.Tensor
    frontier_mask: torch.Tensor
    frontier_node_indices: torch.Tensor
    expanded_node_indices: torch.Tensor
    expanded_probability: torch.Tensor
    expanded_mask: torch.Tensor


class NodeSemanticCompatibility(nn.Module):
    """Shared ``Compat(z_t, u_n)`` scorer used for local child decisions."""

    def __init__(self, z_dim: int, node_dim: int) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.node_dim = int(node_dim)
        self.z_projection = nn.Linear(self.z_dim, self.node_dim)
        self.z_norm = nn.LayerNorm(self.node_dim)
        self.node_norm = nn.LayerNorm(self.node_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(4 * self.node_dim, 2 * self.node_dim),
            nn.GELU(),
            nn.Linear(2 * self.node_dim, 1),
        )
        nn.init.xavier_normal_(self.score_mlp[-1].weight, gain=1)
        nn.init.zeros_(self.score_mlp[-1].bias)

    @torch.no_grad()
    def initialize_for_training(
        self,
        *,
        score_gain: float,
        generator: Optional[torch.Generator] = None,
    ) -> float:
        nn.init.xavier_normal_(
            self.z_projection.weight,
            gain=1.0,
            generator=generator,
        )
        nn.init.zeros_(self.z_projection.bias)
        nn.init.xavier_normal_(
            self.score_mlp[0].weight,
            gain=1.0,
            generator=generator,
        )
        nn.init.zeros_(self.score_mlp[0].bias)
        nn.init.xavier_normal_(
            self.score_mlp[-1].weight,
            gain=score_gain,
            generator=generator,
        )
        nn.init.zeros_(self.score_mlp[-1].bias)
        self.z_norm.reset_parameters()
        self.node_norm.reset_parameters()
        return float(self.score_mlp[-1].weight.norm().cpu())

    def forward(
        self,
        z_t: torch.Tensor,
        node_u: torch.Tensor,
    ) -> torch.Tensor:
        if z_t.ndim not in (2, 3) or z_t.size(-1) != self.z_dim:
            raise ValueError(
                "z_t must have shape [B, z_dim] or [B, N, z_dim]"
            )
        if node_u.ndim not in (2, 3) or node_u.size(-1) != self.node_dim:
            raise ValueError(
                "node_u must have shape [N, node_dim] "
                "or [B, N, node_dim]"
            )
        if z_t.ndim == 2:
            z_t = z_t.unsqueeze(1)
        if node_u.ndim == 2:
            node_u = node_u.unsqueeze(0)
        if z_t.size(0) != node_u.size(0) and node_u.size(0) != 1:
            raise ValueError(
                "z_t and node_u batch dimensions are incompatible"
            )
        if node_u.size(0) == 1 and z_t.size(0) != 1:
            node_u = node_u.expand(z_t.size(0), -1, -1)
        if z_t.size(1) != node_u.size(1):
            if z_t.size(1) == 1:
                z_t = z_t.expand(-1, node_u.size(1), -1)
            elif node_u.size(1) == 1:
                node_u = node_u.expand(-1, z_t.size(1), -1)
            else:
                raise ValueError(
                    "z_t and node_u node dimensions are incompatible"
                )
        return self.score_normalized(
            self.project_z(z_t),
            self.normalize_nodes(node_u),
        )

    def project_z(self, z_t: torch.Tensor) -> torch.Tensor:
        """Project/normalize a query batch once before local branch scoring."""
        if z_t.ndim not in (2, 3) or z_t.size(-1) != self.z_dim:
            raise ValueError(
                "z_t must have shape [B, z_dim] or [B, N, z_dim]"
            )
        return self.z_norm(self.z_projection(z_t))

    def normalize_nodes(self, node_u: torch.Tensor) -> torch.Tensor:
        """Normalize a node table once for all branch decisions."""
        if node_u.ndim not in (2, 3) or node_u.size(-1) != self.node_dim:
            raise ValueError(
                "node_u must have shape [N, node_dim] "
                "or [B, N, node_dim]"
            )
        return self.node_norm(node_u)

    def score_normalized(
        self,
        z_norm: torch.Tensor,
        u_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Score already normalized query/node representations."""
        if z_norm.ndim == 2:
            z_norm = z_norm.unsqueeze(1)
        if u_norm.ndim == 2:
            u_norm = u_norm.unsqueeze(0)
        if z_norm.size(0) != u_norm.size(0) and u_norm.size(0) != 1:
            raise ValueError(
                "z_norm and u_norm batch dimensions are incompatible"
            )
        if u_norm.size(0) == 1 and z_norm.size(0) != 1:
            u_norm = u_norm.expand(z_norm.size(0), -1, -1)
        if z_norm.size(1) != u_norm.size(1):
            if z_norm.size(1) == 1:
                z_norm = z_norm.expand(-1, u_norm.size(1), -1)
            elif u_norm.size(1) == 1:
                u_norm = u_norm.expand(-1, z_norm.size(1), -1)
            else:
                raise ValueError(
                    "z_norm and u_norm node dimensions are incompatible"
                )
        interaction = torch.cat(
            (
                u_norm,
                z_norm,
                u_norm * z_norm,
                torch.abs(u_norm - z_norm),
            ),
            dim=-1,
        )
        return self.score_mlp(interaction).squeeze(-1)


class ExpansionEvidencePredictor(nn.Module):
    """Predict whether a coarse node should be refined for one sequence.

    This head is deliberately separate from the branch Router.  The Regional
    Probe supervises it from counterfactual Hawkes energies, so its prediction
    cannot gate the evidence that is needed to train it.
    """

    def __init__(self, z_dim: int, node_dim: int) -> None:
        super().__init__()
        hidden_dim = max(int(node_dim), 16)
        self.z_dim = int(z_dim)
        self.node_dim = int(node_dim)
        self.network = nn.Sequential(
            nn.Linear(self.z_dim + self.node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_normal_(self.network[0].weight)
        nn.init.zeros_(self.network[0].bias)
        nn.init.xavier_normal_(self.network[-1].weight, gain=0.1)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        sequence_embedding: torch.Tensor,
        node_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if (
            sequence_embedding.ndim != 2
            or sequence_embedding.size(-1) != self.z_dim
        ):
            raise ValueError("sequence_embedding must have shape [B, z_dim]")
        if node_embedding.ndim == 1:
            node_embedding = node_embedding.unsqueeze(0).expand(
                sequence_embedding.size(0), -1
            )
        if (
            node_embedding.ndim != 2
            or node_embedding.shape
            != (sequence_embedding.size(0), self.node_dim)
        ):
            raise ValueError(
                "node_embedding must have shape [node_dim] or [B, node_dim]"
            )
        return self.network(
            torch.cat((sequence_embedding, node_embedding), dim=-1)
        ).squeeze(-1)


class TreeRoutingMixin:
    """Active-frontier methods mixed into ``HawkesTree``."""

    @torch.no_grad()
    def initialize_router_weights(
        self,
        *,
        gain: float = 0.05,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        if gain <= 0.0:
            raise ValueError("router initialization gain must be positive")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device_anchor.device)
            generator.manual_seed(int(seed))
        norm = self.router_compat.initialize_for_training(
            score_gain=gain,
            generator=generator,
        )
        return {node_id: norm for node_id in self.internal_ids}

    def route(
        self,
        z: torch.Tensor,
    ) -> FrontierRoutingOutput:
        """Return responsibility on the actual computed frontier."""
        return self.frontier_route(
            z,
            update_search_state=False,
        )


__all__ = [
    "ExpansionEvidencePredictor",
    "FrontierRoutingOutput",
    "NodeSemanticCompatibility",
    "TreeRoutingMixin",
]
