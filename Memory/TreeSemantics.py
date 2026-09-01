"""Semantic parameter generation for the latent Hawkes tree.

The mixin keeps all parameters directly on ``HawkesTree``. This preserves
historical state-dict keys while separating semantic responsibilities from
topology, routing, and Memory composition.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Sequence

import torch
import torch.nn as nn


class HawkesParamPack(NamedTuple):
    """Unconstrained Hawkes parameters."""

    mu_tilde: torch.Tensor
    W_tilde: torch.Tensor


class HawkesHyperNet(nn.Module):
    """Map cumulative node semantics to raw Hawkes parameters."""

    def __init__(
        self,
        embed_dim: int,
        num_event_types: int,
        num_basis: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.D = num_event_types
        self.M = num_basis
        self.hidden_dim = hidden_dim
        out_dim = self.D + self.D * self.D * self.M
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, embedding: torch.Tensor) -> HawkesParamPack:
        raw = self.net(embedding)
        return HawkesParamPack(
            mu_tilde=raw[..., :self.D],
            W_tilde=raw[..., self.D:].view(
                *raw.shape[:-1], self.D, self.D, self.M
            ),
        )


class TreeSemanticsMixin:
    """Semantic methods mixed into :class:`LatentHawkesTree.HawkesTree`."""

    def _node_embedding_table(self) -> torch.Tensor:
        """Compute every cumulative node embedding with one matrix product."""
        local_embeddings = torch.stack(
            [
                self.node_emb[node_id]
                for node_id in self.all_node_ids
            ],
            dim=0,
        )
        return (
            self.node_path_mask.to(local_embeddings.device)
            @ local_embeddings
        )

    def semantic_theta_table(
        self,
        cumulative_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute every node's path-additive semantic parameters once.

        Routing a sequence can revisit the same frontier node for dozens of
        events. Calling :meth:`semantic_theta` separately for every
        ``(event, node)`` pair repeats both the HyperNet and path reductions.
        This batched form is algebraically identical while keeping one
        autograd graph shared by all selected frontier rows.
        """
        if cumulative_embedding is None:
            cumulative_embedding = self._node_embedding_table()
        expected_shape = (len(self.all_node_ids), self.node_dim)
        if cumulative_embedding.shape != expected_shape:
            raise ValueError(
                "cumulative_embedding must have shape "
                f"{expected_shape}, got {tuple(cumulative_embedding.shape)}"
            )
        params = self.hyper(cumulative_embedding)
        base = torch.cat(
            [
                params.mu_tilde.reshape(len(self.all_node_ids), -1),
                params.W_tilde.reshape(len(self.all_node_ids), -1),
            ],
            dim=-1,
        )
        local_offsets = torch.stack(
            [
                self.semantic_offset[node_id]
                for node_id in self.all_node_ids
            ],
            dim=0,
        )
        cumulative_offset = (
            self.node_path_mask.to(
                device=local_offsets.device,
                dtype=local_offsets.dtype,
            )
            @ local_offsets
        )
        return base + cumulative_offset

    @torch.no_grad()
    def set_residual_probe_prototypes(
        self,
        leaf_ids: Sequence[str],
        prototypes: torch.Tensor,
        target_mass: torch.Tensor,
    ) -> None:
        """Persist the fixed, unscaled residual teacher for Regional Probe."""
        resolved_ids = tuple(str(leaf_id) for leaf_id in leaf_ids)
        prototypes = torch.as_tensor(
            prototypes,
            device=self._device_anchor.device,
            dtype=self.semantic_offset["root"].dtype,
        ).detach()
        target_mass = torch.as_tensor(
            target_mass,
            device=self._device_anchor.device,
            dtype=prototypes.dtype,
        ).detach().reshape(-1)
        expected = (len(resolved_ids), self.param_dim)
        if not resolved_ids or len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("residual probe leaf IDs must be non-empty and unique")
        if prototypes.shape != expected:
            raise ValueError(
                f"residual probe prototypes must have shape {expected}"
            )
        if target_mass.shape != (len(resolved_ids),):
            raise ValueError("residual probe target mass has the wrong shape")
        if not bool(torch.isfinite(prototypes).all()):
            raise ValueError("residual probe prototypes must be finite")
        if (
            not bool(torch.isfinite(target_mass).all())
            or bool((target_mass < 0.0).any())
            or float(target_mass.sum()) <= 0.0
        ):
            raise ValueError(
                "residual probe target mass must be finite, non-negative, "
                "and have positive total mass"
            )
        self.residual_probe_leaf_ids = resolved_ids
        self.residual_probe_prototypes = prototypes.clone()
        self.residual_probe_target_mass = (
            target_mass / target_mass.sum()
        ).clone()

    @torch.no_grad()
    def residual_probe_leaf_table(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the fixed teacher onto the current split/merge leaf topology."""
        source_ids = tuple(self.residual_probe_leaf_ids)
        if not source_ids:
            return (
                self._device_anchor.new_empty(0, self.param_dim),
                self._device_anchor.new_empty(0),
            )
        source_prototypes = self.residual_probe_prototypes.to(
            device=self._device_anchor.device,
            dtype=self.semantic_offset["root"].dtype,
        )
        source_mass = self.residual_probe_target_mass.to(source_prototypes)
        current_ids = tuple(self.leaf_ids)
        current_mass = source_mass.new_zeros(len(current_ids))

        # Conserve the empirical prior: a split shares its old mass, while a
        # merge sums the masses of all original descendant leaves.
        for source_index, source_id in enumerate(source_ids):
            descendants = [
                index
                for index, current_id in enumerate(current_ids)
                if current_id == source_id
                or current_id.startswith(f"{source_id}_")
            ]
            if descendants:
                share = source_mass[source_index] / len(descendants)
                current_mass[descendants] += share
                continue
            ancestors = [
                (index, current_id)
                for index, current_id in enumerate(current_ids)
                if source_id.startswith(f"{current_id}_")
            ]
            if not ancestors:
                raise RuntimeError(
                    f"cannot map residual prototype leaf {source_id!r} "
                    "onto the current topology"
                )
            nearest_index, _ = max(
                ancestors,
                key=lambda item: item[1].count("_"),
            )
            current_mass[nearest_index] += source_mass[source_index]

        current_prototypes = []
        for current_id in current_ids:
            merged_sources = [
                index
                for index, source_id in enumerate(source_ids)
                if source_id == current_id
                or source_id.startswith(f"{current_id}_")
            ]
            if merged_sources:
                indices = torch.tensor(
                    merged_sources,
                    device=source_prototypes.device,
                    dtype=torch.long,
                )
                mass = source_mass.index_select(0, indices)
                prototype = (
                    mass.unsqueeze(-1)
                    * source_prototypes.index_select(0, indices)
                ).sum(dim=0) / mass.sum().clamp_min(1e-12)
            else:
                source_ancestors = [
                    (index, source_id)
                    for index, source_id in enumerate(source_ids)
                    if current_id.startswith(f"{source_id}_")
                ]
                if not source_ancestors:
                    raise RuntimeError(
                        f"cannot map current leaf {current_id!r} to a "
                        "residual prototype"
                    )
                source_index, _ = max(
                    source_ancestors,
                    key=lambda item: item[1].count("_"),
                )
                prototype = source_prototypes[source_index]
            current_prototypes.append(prototype)
        return (
            torch.stack(current_prototypes),
            current_mass / current_mass.sum().clamp_min(1e-12),
        )

    def leaf_embeddings(self) -> torch.Tensor:
        node_embeddings = torch.stack(
            [self.node_emb[node_id] for node_id in self.all_node_ids],
            dim=0,
        )
        return self.path_node_mask.to(node_embeddings.device) @ node_embeddings

    def node_embedding(self, node_id: str) -> torch.Tensor:
        return torch.stack(
            [self.node_emb[item] for item in self.path_to_node(node_id)],
            dim=0,
        ).sum(dim=0)

    def base_semantic_theta(self, node_id: str) -> torch.Tensor:
        params = self.hyper(self.node_embedding(node_id))
        return torch.cat(
            [params.mu_tilde.reshape(-1), params.W_tilde.reshape(-1)],
            dim=0,
        )

    def semantic_theta(self, node_id: str) -> torch.Tensor:
        base = self.base_semantic_theta(node_id)
        offset = torch.stack(
            [
                self.semantic_offset[item]
                for item in self.path_to_node(node_id)
            ],
            dim=0,
        ).sum(dim=0)
        return base + offset

    @torch.no_grad()
    def set_semantic_theta(
        self,
        node_id: str,
        target_theta: torch.Tensor,
    ) -> None:
        target_theta = target_theta.to(
            device=self._device_anchor.device,
            dtype=self.semantic_offset[node_id].dtype,
        ).reshape(-1)
        if target_theta.numel() != self.param_dim:
            raise ValueError(
                f"target_theta must contain {self.param_dim} values"
            )
        base = self.base_semantic_theta(node_id)
        path = self.path_to_node(node_id)
        ancestor_offset = base.new_zeros(self.param_dim)
        for ancestor in path[:-1]:
            ancestor_offset = ancestor_offset + self.semantic_offset[ancestor]
        self.semantic_offset[node_id].copy_(
            target_theta - base - ancestor_offset
        )

    @torch.no_grad()
    def initialize_semantics_from_hawkes(
        self,
        hawkes: nn.Module,
        *,
        semantic_blend: float = 0.0,
    ) -> torch.Tensor:
        if not 0.0 <= semantic_blend <= 1.0:
            raise ValueError("semantic_blend must lie in [0, 1]")
        raw_mu = getattr(hawkes, "raw_mu", None)
        raw_W = getattr(hawkes, "raw_W", None)
        if not torch.is_tensor(raw_mu) or not torch.is_tensor(raw_W):
            raise TypeError("hawkes must expose raw_mu and raw_W tensors")
        if raw_mu.numel() != self.num_event_types:
            raise ValueError("Hawkes/tree event dimensions differ")
        if raw_W.numel() != (
            self.num_event_types
            * self.num_event_types
            * self.num_basis
        ):
            raise ValueError("Hawkes/tree basis dimensions differ")

        cold_target = torch.cat(
            [raw_mu.detach().reshape(-1), raw_W.detach().reshape(-1)],
            dim=0,
        ).to(self._device_anchor.device)
        node_targets = {
            node_id: torch.lerp(
                cold_target,
                self.base_semantic_theta(node_id).detach(),
                float(semantic_blend),
            )
            for node_id in self.nodes
        }
        for node_id in sorted(
            self.nodes,
            key=lambda item: self.nodes[item].depth,
        ):
            self.set_semantic_theta(node_id, node_targets[node_id])
        self.semantic_blend = float(semantic_blend)
        return cold_target.detach().clone()

    @torch.no_grad()
    def break_initial_leaf_symmetry(
        self,
        *,
        relative_scale: float = 0.01,
        seed: int = 0,
    ) -> Dict[str, float]:
        if relative_scale < 0.0:
            raise ValueError("relative_scale must be non-negative")
        leaf_ids = list(self.leaf_ids)
        if len(leaf_ids) < 2 or relative_scale == 0.0:
            return {
                "relative_scale": float(relative_scale),
                "max_abs": 0.0,
                "mean_abs": 0.0,
                "mean_error": 0.0,
            }

        original = torch.stack(
            [self.semantic_theta(leaf_id) for leaf_id in leaf_ids],
            dim=0,
        )
        target_mean = original.mean(dim=0)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        noise = torch.randn(
            len(leaf_ids),
            self.param_dim,
            generator=generator,
            dtype=torch.float32,
        ).to(device=original.device, dtype=original.dtype)
        noise = noise - noise.mean(dim=0, keepdim=True)
        coordinate_scale = target_mean.abs().clamp_min(1.0)
        perturbation = (
            relative_scale * noise * coordinate_scale.unsqueeze(0)
        )
        for leaf_id, target in zip(
            leaf_ids,
            original + perturbation,
        ):
            self.set_semantic_theta(leaf_id, target)

        updated = torch.stack(
            [self.semantic_theta(leaf_id) for leaf_id in leaf_ids],
            dim=0,
        )
        realized = updated - original
        return {
            "relative_scale": float(relative_scale),
            "max_abs": float(realized.abs().max().cpu()),
            "mean_abs": float(realized.abs().mean().cpu()),
            "mean_error": float(
                (updated.mean(dim=0) - target_mean).abs().max().cpu()
            ),
        }

    @torch.no_grad()
    def initialize_semantics_from_residual_prototypes(
        self,
        cold_target: torch.Tensor,
        leaf_prototypes: torch.Tensor,
        target_leaf_mass: torch.Tensor,
        *,
        init_scale: float,
    ) -> Dict[str, float]:
        """Initialize experts from data-related Hawkes residual directions.

        ``leaf_prototypes`` must follow ``self.leaf_ids`` order.  Leaf targets
        are mass-centered around the shared cold-start Hawkes parameters, while
        each internal target is the target-mass-weighted mean of its descendant
        leaves.  No node-ID-dependent random perturbation is introduced.
        """
        if init_scale < 0.0:
            raise ValueError("init_scale must be non-negative")
        leaf_count = len(self.leaf_ids)
        cold_target = cold_target.to(
            device=self._device_anchor.device,
            dtype=self.semantic_offset["root"].dtype,
        ).reshape(-1)
        leaf_prototypes = leaf_prototypes.to(
            device=cold_target.device,
            dtype=cold_target.dtype,
        )
        target_leaf_mass = target_leaf_mass.to(
            device=cold_target.device,
            dtype=cold_target.dtype,
        ).reshape(-1)
        if cold_target.numel() != self.param_dim:
            raise ValueError(
                f"cold_target must contain {self.param_dim} values"
            )
        if leaf_prototypes.shape != (leaf_count, self.param_dim):
            raise ValueError(
                "leaf_prototypes must have shape "
                f"{(leaf_count, self.param_dim)}"
            )
        if target_leaf_mass.shape != (leaf_count,):
            raise ValueError(
                f"target_leaf_mass must have shape {(leaf_count,)}"
            )
        if not torch.isfinite(leaf_prototypes).all():
            raise ValueError("leaf_prototypes must be finite")
        if (
            not torch.isfinite(target_leaf_mass).all()
            or bool((target_leaf_mass < 0).any())
            or float(target_leaf_mass.sum()) <= 0.0
        ):
            raise ValueError(
                "target_leaf_mass must be finite, non-negative, and non-empty"
            )

        target_leaf_mass = target_leaf_mass / target_leaf_mass.sum()
        prototype_center = (
            target_leaf_mass.unsqueeze(-1) * leaf_prototypes
        ).sum(dim=0)
        centered_prototypes = leaf_prototypes - prototype_center
        leaf_targets = (
            cold_target.unsqueeze(0)
            + float(init_scale) * centered_prototypes
        )

        leaf_index = {
            leaf_id: index
            for index, leaf_id in enumerate(self.leaf_ids)
        }
        node_targets: Dict[str, torch.Tensor] = {}
        for node_id in self.all_node_ids:
            descendant_indices = [
                leaf_index[leaf_id]
                for leaf_id in self.leaf_ids
                if node_id in self.path_to_leaf(leaf_id)
            ]
            if not descendant_indices:
                raise RuntimeError(
                    f"node {node_id!r} has no descendant leaves"
                )
            indices = torch.tensor(
                descendant_indices,
                device=cold_target.device,
                dtype=torch.long,
            )
            descendant_mass = target_leaf_mass.index_select(0, indices)
            mass_sum = descendant_mass.sum()
            if float(mass_sum) <= 0.0:
                raise ValueError(
                    f"node {node_id!r} has zero target descendant mass"
                )
            node_targets[node_id] = (
                descendant_mass.unsqueeze(-1)
                * leaf_targets.index_select(0, indices)
            ).sum(dim=0) / mass_sum

        for node_id in sorted(
            self.all_node_ids,
            key=lambda item: self.nodes[item].depth,
        ):
            self.set_semantic_theta(node_id, node_targets[node_id])
        self.semantic_blend = 0.0

        realized_leaves = torch.stack(
            [self.semantic_theta(leaf_id) for leaf_id in self.leaf_ids],
            dim=0,
        )
        weighted_mean = (
            target_leaf_mass.unsqueeze(-1) * realized_leaves
        ).sum(dim=0)
        realized_delta = realized_leaves - cold_target.unsqueeze(0)
        return {
            "init_scale": float(init_scale),
            "prototype_norm_min": float(
                leaf_prototypes.norm(dim=-1).min().cpu()
            ),
            "prototype_norm_mean": float(
                leaf_prototypes.norm(dim=-1).mean().cpu()
            ),
            "prototype_norm_max": float(
                leaf_prototypes.norm(dim=-1).max().cpu()
            ),
            "leaf_delta_mean_abs": float(realized_delta.abs().mean().cpu()),
            "leaf_delta_max_abs": float(realized_delta.abs().max().cpu()),
            "weighted_mean_error": float(
                (weighted_mean - cold_target).abs().max().cpu()
            ),
            "root_error": float(
                (
                    self.semantic_theta("root") - cold_target
                ).abs().max().cpu()
            ),
        }

    def fit_new_node_semantics(
        self,
        node_ids: Sequence[str],
        target_theta: torch.Tensor,
        *,
        num_steps: int = 30,
        learning_rate: float = 1e-2,
    ) -> Dict[str, float]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not node_ids:
            raise ValueError("node_ids must be non-empty")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be unique")
        missing = [
            node_id for node_id in node_ids if node_id not in self.nodes
        ]
        if missing:
            raise KeyError(
                f"unknown semantic initialization nodes: {missing}"
            )
        target = target_theta.detach().to(
            device=self._device_anchor.device,
            dtype=self.semantic_offset[node_ids[0]].dtype,
        )
        if target.shape != (len(node_ids), self.param_dim):
            raise ValueError(
                "target_theta must have shape "
                f"[{len(node_ids)}, {self.param_dim}]"
            )
        parameters = [self.node_emb[node_id] for node_id in node_ids]

        def objective() -> torch.Tensor:
            predicted = torch.stack(
                [self.semantic_theta(node_id) for node_id in node_ids],
                dim=0,
            )
            return (predicted - target).pow(2).sum(dim=-1).mean()

        with torch.enable_grad():
            loss_before = objective().detach()
            if num_steps > 0:
                optimizer = torch.optim.Adam(
                    parameters,
                    lr=learning_rate,
                )
                for _ in range(num_steps):
                    optimizer.zero_grad(set_to_none=True)
                    loss = objective()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            "split semantic initialization became non-finite"
                        )
                    gradients = torch.autograd.grad(loss, parameters)
                    for parameter, gradient in zip(
                        parameters,
                        gradients,
                    ):
                        parameter.grad = gradient
                    optimizer.step()
            loss_after = objective().detach()
        return {
            "loss_before": float(loss_before.cpu()),
            "loss_after": float(loss_after.cpu()),
        }

    def semantic_params(self) -> HawkesParamPack:
        # One HyperNet batch replaces one independent module traversal per
        # leaf. Path-additive offsets remain algebraically identical.
        base = self.hyper(self.leaf_embeddings())
        base_theta = torch.cat(
            [
                base.mu_tilde,
                base.W_tilde.reshape(len(self.leaf_ids), -1),
            ],
            dim=-1,
        )
        local_offsets = torch.stack(
            [
                self.semantic_offset[node_id]
                for node_id in self.all_node_ids
            ],
            dim=0,
        )
        theta = (
            base_theta
            + self.path_node_mask.to(local_offsets.device)
            @ local_offsets
        )
        event_dim = self.hyper.D
        return HawkesParamPack(
            mu_tilde=theta[:, :event_dim],
            W_tilde=theta[:, event_dim:].reshape(
                -1,
                event_dim,
                event_dim,
                self.hyper.M,
            ),
        )

    def mixed_semantic_params(
        self,
        responsibility: torch.Tensor,
        leaf_params: Optional[HawkesParamPack] = None,
    ) -> HawkesParamPack:
        if leaf_params is None:
            leaf_params = self.semantic_params()
        return HawkesParamPack(
            mu_tilde=torch.einsum(
                "bl,ld->bd",
                responsibility,
                leaf_params.mu_tilde,
            ),
            W_tilde=torch.einsum(
                "bl,ldem->bdem",
                responsibility,
                leaf_params.W_tilde,
            ),
        )
