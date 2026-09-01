"""Low-cost per-node sequence prototypes used by local branch routing."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class NodePrototypeStore(nn.Module):
    """Weighted Welford statistics aligned with a dynamic tree.

    The store keeps only ``(mean, diagonal variance accumulator, count)`` for
    each node. A sequence assigned to a leaf is also assigned to every node on
    that leaf's root-to-leaf path, preserving ``D_child subset D_parent``.
    """

    def __init__(
        self,
        feature_dim: int,
        node_ids: Sequence[str] = (),
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.node_ids = tuple(node_ids)
        node_count = len(self.node_ids)
        self.register_buffer(
            "count",
            torch.zeros(node_count, device=device),
        )
        self.register_buffer(
            "mean",
            torch.zeros(node_count, self.feature_dim, device=device),
        )
        self.register_buffer(
            "m2",
            torch.zeros(node_count, self.feature_dim, device=device),
        )

    @property
    def node_index(self) -> dict[str, int]:
        return {
            node_id: index
            for index, node_id in enumerate(self.node_ids)
        }

    @torch.no_grad()
    def sync_nodes(self, node_ids: Sequence[str]) -> None:
        """Preserve existing statistics while adding/removing tree nodes."""
        new_ids = tuple(node_ids)
        if new_ids == self.node_ids:
            return
        old_index = self.node_index
        new_count = self.count.new_zeros(len(new_ids))
        new_mean = self.mean.new_zeros(len(new_ids), self.feature_dim)
        new_m2 = self.m2.new_zeros(len(new_ids), self.feature_dim)
        for new_index, node_id in enumerate(new_ids):
            if node_id not in old_index:
                continue
            old = old_index[node_id]
            new_count[new_index] = self.count[old]
            new_mean[new_index] = self.mean[old]
            new_m2[new_index] = self.m2[old]
        self.node_ids = new_ids
        self.count = new_count
        self.mean = new_mean
        self.m2 = new_m2

    @torch.no_grad()
    def update_weighted(
        self,
        z: Tensor,
        node_weights: Tensor,
    ) -> None:
        """Update all node statistics from weighted sequence assignments.

        Args:
            z: sequence representations with shape ``[B, feature_dim]``.
            node_weights: non-negative membership weights ``[B, N]``.
        """
        if z.ndim != 2 or z.size(1) != self.feature_dim:
            raise ValueError(
                f"z must have shape [B, {self.feature_dim}]"
            )
        if node_weights.shape != (z.size(0), len(self.node_ids)):
            raise ValueError("node_weights must have shape [B, N]")
        if bool((node_weights < 0.0).any()):
            raise ValueError("node_weights must be non-negative")

        weights = node_weights.to(device=z.device, dtype=z.dtype)
        batch_count = weights.sum(dim=0)
        active = batch_count > 0.0
        if not bool(active.any()):
            return
        batch_mean = torch.einsum("bn,bd->nd", weights, z)
        batch_mean = batch_mean / batch_count.clamp_min(1e-12).unsqueeze(-1)
        centered = z.unsqueeze(1) - batch_mean.unsqueeze(0)
        batch_m2 = torch.einsum(
            "bn,bnd->nd",
            weights,
            centered.square(),
        )

        old_count = self.count
        total = old_count + batch_count
        mean_delta = batch_mean - self.mean
        merged_mean = self.mean + (
            mean_delta
            * (batch_count / total.clamp_min(1e-12)).unsqueeze(-1)
        )
        cross = (
            mean_delta.square()
            * (
                old_count * batch_count
                / total.clamp_min(1e-12)
            ).unsqueeze(-1)
        )
        merged_m2 = self.m2 + batch_m2 + cross

        self.count.copy_(torch.where(active, total, old_count))
        self.mean.copy_(
            torch.where(active.unsqueeze(-1), merged_mean, self.mean)
        )
        self.m2.copy_(
            torch.where(active.unsqueeze(-1), merged_m2, self.m2)
        )

    @torch.no_grad()
    def update_leaf_responsibility(
        self,
        z: Tensor,
        leaf_ids: Sequence[str],
        leaf_paths: Sequence[Sequence[str]],
        responsibility: Tensor,
    ) -> None:
        """Propagate leaf responsibilities to all ancestor prototypes."""
        if responsibility.shape != (z.size(0), len(leaf_ids)):
            raise ValueError(
                "responsibility must have shape [B, num_leaves]"
            )
        index = self.node_index
        node_weights = responsibility.new_zeros(
            z.size(0), len(self.node_ids)
        )
        for leaf_index, path in enumerate(leaf_paths):
            for node_id in path:
                if node_id in index:
                    node_weights[:, index[node_id]] += (
                        responsibility[:, leaf_index]
                    )
        self.update_weighted(z, node_weights)

    @torch.no_grad()
    def update_frontier_responsibility(
        self,
        z: Tensor,
        node_indices: Tensor,
        responsibility: Tensor,
        mask: Tensor,
    ) -> None:
        """Update prototypes from posterior mass on computed frontier nodes.

        A frontier node contributes to itself and its already-computed
        ancestors.  It is deliberately *not* projected into unexpanded
        descendant leaves.
        """
        expected = node_indices.shape
        if (
            z.ndim != 2
            or node_indices.ndim != 2
            or responsibility.shape != expected
            or mask.shape != expected
            or expected[0] != z.size(0)
        ):
            raise ValueError(
                "frontier tensors must align as [B, K] with z [B, D]"
            )
        if node_indices.dtype != torch.long or mask.dtype != torch.bool:
            raise ValueError("node_indices must be long and mask must be bool")
        if bool((responsibility.masked_select(mask) < 0.0).any()):
            raise ValueError("responsibility must be non-negative")

        node_count = len(self.node_ids)
        safe_indices = node_indices.clamp_min(0)
        direct = responsibility.new_zeros(z.size(0), node_count)
        direct.scatter_add_(
            1,
            safe_indices,
            responsibility.masked_fill(~mask, 0.0),
        )
        # The store has no tree dependency, so derive ancestor membership from
        # hierarchical node ids (root, root_L, root_L_R, ...).
        ancestor = torch.zeros(
            node_count,
            node_count,
            dtype=torch.bool,
            device=direct.device,
        )
        for descendant_index, descendant_id in enumerate(self.node_ids):
            for ancestor_index, ancestor_id in enumerate(self.node_ids):
                if (
                    ancestor_id == "root"
                    or descendant_id == ancestor_id
                    or descendant_id.startswith(ancestor_id + "_")
                ):
                    ancestor[descendant_index, ancestor_index] = True
        node_weights = direct @ ancestor.to(dtype=direct.dtype)
        self.update_weighted(z, node_weights)

    def variance(self, node_id: str, epsilon: float = 1e-4) -> Tensor:
        index = self.node_index[node_id]
        denominator = (self.count[index] - 1.0).clamp_min(1.0)
        return self.m2[index] / denominator + float(epsilon)

    def data_score(
        self,
        z: Tensor,
        node_ids: Sequence[str],
        *,
        epsilon: float = 1e-4,
    ) -> Tensor:
        """Diagonal-Gaussian prototype score for candidate child nodes."""
        if z.ndim != 2 or z.size(1) != self.feature_dim:
            raise ValueError(
                f"z must have shape [B, {self.feature_dim}]"
            )
        index = self.node_index
        indices = torch.tensor(
            [index[node_id] for node_id in node_ids],
            device=z.device,
            dtype=torch.long,
        )
        mean = self.mean.index_select(0, indices).to(z)
        count = self.count.index_select(0, indices).to(z)
        variance = (
            self.m2.index_select(0, indices).to(z)
            / (count - 1.0).clamp_min(1.0).unsqueeze(-1)
            + float(epsilon)
        )
        score = -0.5 * (
            (z.unsqueeze(1) - mean.unsqueeze(0)).square()
            / variance.unsqueeze(0)
        ).sum(dim=-1)
        # An unseen child must not produce an enormous cold-start penalty.
        return torch.where(
            (count >= 2.0).unsqueeze(0),
            score,
            torch.zeros_like(score),
        )

    def get_extra_state(self):
        return {"node_ids": self.node_ids}

    def set_extra_state(self, state) -> None:
        self.node_ids = tuple(state.get("node_ids", ()))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Resize dynamic prototype buffers before the regular tensor load."""
        saved_count = state_dict.get(prefix + "count")
        saved_mean = state_dict.get(prefix + "mean")
        saved_m2 = state_dict.get(prefix + "m2")
        if saved_count is not None:
            self.count = self.count.new_zeros(saved_count.shape)
        if saved_mean is not None:
            self.mean = self.mean.new_zeros(saved_mean.shape)
        if saved_m2 is not None:
            self.m2 = self.m2.new_zeros(saved_m2.shape)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
