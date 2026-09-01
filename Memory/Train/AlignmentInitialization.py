"""Offline H-tree membership calibration for the causal Encoder and Router.

The initialization objective aligns *routing decisions*, not representations:

    full-leaf q^(0) -> local binary q_off -> semantic compatibility p_sem.

Only ``CausalPrefixEncoder`` and ``tree.router_compat`` are optimized. Hawkes
parameters, node embeddings, semantic experts, and all Memory modules remain
fixed. The phase runs before Wake/Sleep training and never uses topology priors,
Hawkes energy, episodic retrieval, or future-window features.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm


@dataclass(frozen=True)
class LocalBranchTeacher:
    """Offline binary routing targets in ``tree.internal_ids`` order."""

    target: Tensor
    node_mass: Tensor
    confidence: Tensor
    internal_ids: tuple[str, ...]


def build_local_branch_teacher(
    tree: Any,
    leaf_membership: Tensor,
) -> LocalBranchTeacher:
    """Decompose full-leaf membership into every local L/R decision.

    For sequence ``s`` and internal node ``n``:

    ``target[s,n] = [mass(left subtree), mass(right subtree)] / mass(n)``.
    """
    if (
        leaf_membership.ndim != 2
        or leaf_membership.size(1) != len(tree.leaf_ids)
    ):
        raise ValueError(
            "leaf_membership must have shape "
            f"[S, {len(tree.leaf_ids)}]"
        )
    if not tree.internal_ids:
        raise ValueError("alignment requires at least one internal tree node")
    if (
        not torch.isfinite(leaf_membership).all()
        or bool((leaf_membership < 0).any())
    ):
        raise ValueError("leaf_membership must be finite and non-negative")
    row_mass = leaf_membership.sum(dim=-1)
    if not torch.allclose(
        row_mass,
        torch.ones_like(row_mass),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("each leaf_membership row must sum to one")

    leaf_index = {
        leaf_id: index
        for index, leaf_id in enumerate(tree.leaf_ids)
    }
    local_mass = []
    for node_id in tree.internal_ids:
        node = tree.nodes[node_id]
        if node.left is None or node.right is None:
            raise RuntimeError(
                f"internal node {node_id!r} does not have two children"
            )
        left_indices = [
            leaf_index[leaf_id]
            for leaf_id in tree.leaf_ids
            if node.left in tree.path_to_leaf(leaf_id)
        ]
        right_indices = [
            leaf_index[leaf_id]
            for leaf_id in tree.leaf_ids
            if node.right in tree.path_to_leaf(leaf_id)
        ]
        if not left_indices or not right_indices:
            raise RuntimeError(
                f"internal node {node_id!r} has an empty child subtree"
            )
        left_mass = leaf_membership[:, left_indices].sum(dim=-1)
        right_mass = leaf_membership[:, right_indices].sum(dim=-1)
        local_mass.append(torch.stack([left_mass, right_mass], dim=-1))

    child_mass = torch.stack(local_mass, dim=1)
    node_mass = child_mass.sum(dim=-1)
    target = child_mass / node_mass.unsqueeze(-1).clamp_min(1e-12)
    entropy = -torch.where(
        target > 0,
        target * target.clamp_min(1e-12).log(),
        torch.zeros_like(target),
    ).sum(dim=-1)
    confidence = (
        1.0 - entropy / math.log(2.0)
    ).clamp(0.0, 1.0)
    return LocalBranchTeacher(
        target=target,
        node_mass=node_mass,
        confidence=confidence,
        internal_ids=tuple(tree.internal_ids),
    )


def prefix_completion_weight(
    valid_mask: Tensor,
) -> Tensor:
    """Return smooth causal supervision weights ``t / (K_s - 1)``."""
    if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be a boolean [B, T] tensor")
    lengths = valid_mask.sum(dim=-1)
    if bool((lengths <= 0).any()):
        raise ValueError("every alignment sequence must contain an event")
    positions = torch.arange(
        valid_mask.size(1),
        device=valid_mask.device,
        dtype=torch.float32,
    ).unsqueeze(0)
    denominator = (lengths - 1).clamp_min(1).to(positions.dtype)
    return (
        positions / denominator.unsqueeze(-1)
    ).clamp_max(1.0) * valid_mask.to(positions.dtype)


def semantic_branch_logits(
    tree: Any,
    prefix_state: Tensor,
    *,
    detach_node_embeddings: bool = True,
) -> Tensor:
    """Score all local children using compatibility semantics only.

    Returns logits ``[B, T, N_internal, 2]``. No topology prior, exploration
    mixture, frontier utility, or likelihood energy is evaluated here.
    """
    if (
        prefix_state.ndim != 3
        or prefix_state.size(-1) != tree.z_dim
    ):
        raise ValueError(
            f"prefix_state must have shape [B, T, {tree.z_dim}]"
        )
    node_index = {
        node_id: index
        for index, node_id in enumerate(tree.all_node_ids)
    }
    child_indices = torch.tensor(
        [
            [
                node_index[tree.nodes[node_id].left],
                node_index[tree.nodes[node_id].right],
            ]
            for node_id in tree.internal_ids
        ],
        device=prefix_state.device,
        dtype=torch.long,
    )
    node_table = tree._node_embedding_table()
    if detach_node_embeddings:
        node_table = node_table.detach()
    child_table = node_table.index_select(
        0,
        child_indices.reshape(-1),
    ).reshape(len(tree.internal_ids), 2, tree.node_dim)
    normalized_children = tree.router_compat.normalize_nodes(child_table)

    batch, event_count, _ = prefix_state.shape
    projected = tree.router_compat.project_z(
        prefix_state.reshape(batch * event_count, tree.z_dim)
    )
    child_count = 2 * len(tree.internal_ids)
    projected_grid = projected.unsqueeze(1).expand(
        -1,
        child_count,
        -1,
    )
    child_grid = normalized_children.reshape(
        1,
        child_count,
        tree.node_dim,
    ).expand(batch * event_count, -1, -1)
    return tree.router_compat.score_normalized(
        projected_grid,
        child_grid,
    ).reshape(batch, event_count, len(tree.internal_ids), 2)


def membership_alignment_loss(
    semantic_logits: Tensor,
    teacher: LocalBranchTeacher,
    valid_mask: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, Dict[str, Tensor]]:
    """Compute the confidence/mass/prefix-weighted local KL objective."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if semantic_logits.ndim != 4 or semantic_logits.size(-1) != 2:
        raise ValueError(
            "semantic_logits must have shape [B, T, N, 2]"
        )
    batch, event_count, node_count, _ = semantic_logits.shape
    if (
        teacher.target.shape != (batch, node_count, 2)
        or teacher.node_mass.shape != (batch, node_count)
        or teacher.confidence.shape != (batch, node_count)
        or valid_mask.shape != (batch, event_count)
    ):
        raise ValueError("teacher, prefix mask, and logits are misaligned")

    log_student = F.log_softmax(
        semantic_logits / float(temperature),
        dim=-1,
    )
    target = teacher.target.to(log_student)
    local_kl = torch.where(
        target.unsqueeze(1) > 0,
        target.unsqueeze(1)
        * (
            target.unsqueeze(1).clamp_min(1e-12).log()
            - log_student
        ),
        torch.zeros_like(log_student),
    ).sum(dim=-1)
    prefix_weight = prefix_completion_weight(valid_mask).to(log_student)
    weight = (
        prefix_weight.unsqueeze(-1)
        * teacher.node_mass.to(log_student).unsqueeze(1)
        * teacher.confidence.to(log_student).unsqueeze(1)
    )
    denominator = weight.sum()
    if float(denominator.detach()) <= 0.0:
        return semantic_logits.sum() * 0.0, {
            "weight": denominator.detach(),
            "correct": denominator.detach(),
            "target_probability": denominator.detach(),
            "teacher_confidence": denominator.detach(),
        }
    loss = (weight * local_kl).sum() / denominator
    student = log_student.exp()
    correct = (
        student.argmax(dim=-1)
        == target.argmax(dim=-1).unsqueeze(1)
    ).to(weight.dtype)
    target_probability = (
        student * target.unsqueeze(1)
    ).sum(dim=-1)
    return loss, {
        "weight": denominator.detach(),
        "correct": (weight * correct).sum().detach(),
        "target_probability": (
            weight * target_probability
        ).sum().detach(),
        "teacher_confidence": (
            weight
            * teacher.confidence.to(weight).unsqueeze(1)
        ).sum().detach(),
    }


def _pad_sequence_batch(
    sequences: Sequence[Mapping[str, Tensor]],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    if not sequences:
        raise ValueError("cannot pad an empty alignment batch")
    lengths = [int(sequence["times"].numel()) for sequence in sequences]
    if min(lengths) <= 0:
        raise ValueError("alignment does not support empty sequences")
    width = max(lengths)
    times = torch.zeros(
        len(sequences),
        width,
        device=device,
        dtype=sequences[0]["times"].dtype,
    )
    types = torch.zeros(
        len(sequences),
        width,
        device=device,
        dtype=torch.long,
    )
    valid = torch.zeros(
        len(sequences),
        width,
        device=device,
        dtype=torch.bool,
    )
    for row, (sequence, length) in enumerate(zip(sequences, lengths)):
        times[row, :length] = sequence["times"].to(device)
        types[row, :length] = sequence["types"].to(device).long()
        valid[row, :length] = True
    return times, types, valid


@contextmanager
def _alignment_parameter_scope(tree: Any, encoder: nn.Module):
    """Temporarily freeze every tree parameter except compatibility."""
    tree_parameters = list(tree.named_parameters())
    encoder_parameters = list(encoder.parameters())
    original_tree = {
        name: parameter.requires_grad
        for name, parameter in tree_parameters
    }
    original_encoder = [
        parameter.requires_grad
        for parameter in encoder_parameters
    ]
    try:
        for name, parameter in tree_parameters:
            parameter.requires_grad_(name.startswith("router_compat."))
        for parameter in encoder_parameters:
            parameter.requires_grad_(True)
        yield
    finally:
        for name, parameter in tree_parameters:
            parameter.requires_grad_(original_tree[name])
        for parameter, requires_grad in zip(
            encoder_parameters,
            original_encoder,
        ):
            parameter.requires_grad_(requires_grad)


@torch.no_grad()
def evaluate_membership_alignment(
    tree: Any,
    encoder: nn.Module,
    dataset: Sequence[Mapping[str, Tensor]],
    teacher: LocalBranchTeacher,
    *,
    batch_size: int,
    temperature: float,
) -> Dict[str, float]:
    """Evaluate the alignment objective without updating any state."""
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    device = tree._device_anchor.device
    encoder_was_training = encoder.training
    router_was_training = tree.router_compat.training
    encoder.eval()
    tree.router_compat.eval()
    loss_numerator = 0.0
    weight_total = 0.0
    correct_total = 0.0
    target_probability_total = 0.0
    try:
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            indices = list(range(start, stop))
            times, types, valid = _pad_sequence_batch(
                dataset[start:stop],
                device,
            )
            z_prefix, prefix_mask = encoder.forward_padded_prefix(
                times,
                types,
                valid,
            )
            semantic_logits = semantic_branch_logits(
                tree,
                z_prefix,
                detach_node_embeddings=True,
            )
            batch_teacher = LocalBranchTeacher(
                target=teacher.target[indices].to(device),
                node_mass=teacher.node_mass[indices].to(device),
                confidence=teacher.confidence[indices].to(device),
                internal_ids=teacher.internal_ids,
            )
            loss, metrics = membership_alignment_loss(
                semantic_logits,
                batch_teacher,
                prefix_mask,
                temperature=temperature,
            )
            batch_weight = float(metrics["weight"].cpu())
            loss_numerator += float(loss.cpu()) * batch_weight
            weight_total += batch_weight
            correct_total += float(metrics["correct"].cpu())
            target_probability_total += float(
                metrics["target_probability"].cpu()
            )
    finally:
        encoder.train(encoder_was_training)
        tree.router_compat.train(router_was_training)
    if weight_total <= 0.0:
        raise RuntimeError("alignment evaluation received zero usable weight")
    return {
        "loss": loss_numerator / weight_total,
        "weighted_accuracy": correct_total / weight_total,
        "target_probability": target_probability_total / weight_total,
        "weight": weight_total,
    }


def run_membership_alignment(
    tree: Any,
    encoder: nn.Module,
    dataset: Sequence[Mapping[str, Tensor]],
    leaf_membership: Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float = 1e-5,
    temperature: float = 1.0,
    grad_clip: float = 5.0,
    seed: int = 0,
    progress: bool = True,
) -> Dict[str, Any]:
    """Train only Encoder + semantic compatibility from offline membership."""
    if epochs <= 0:
        raise ValueError("alignment epochs must be positive")
    if batch_size <= 0:
        raise ValueError("alignment batch_size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("alignment learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("alignment weight_decay must be non-negative")
    if temperature <= 0.0:
        raise ValueError("alignment temperature must be positive")
    if grad_clip <= 0.0:
        raise ValueError("alignment grad_clip must be positive")
    if len(dataset) != leaf_membership.size(0):
        raise ValueError("dataset and leaf_membership sequence counts differ")
    if not hasattr(encoder, "forward_padded_prefix"):
        raise TypeError(
            "alignment requires an encoder with forward_padded_prefix"
        )

    device = tree._device_anchor.device
    teacher = build_local_branch_teacher(tree, leaf_membership)
    parameters = {
        **{
            f"encoder.{name}": parameter
            for name, parameter in encoder.named_parameters()
        },
        **{
            f"router_compat.{name}": parameter
            for name, parameter in tree.router_compat.named_parameters()
        },
    }
    optimizer = torch.optim.AdamW(
        parameters.values(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    history = []
    encoder_was_training = encoder.training
    router_was_training = tree.router_compat.training

    with _alignment_parameter_scope(tree, encoder):
        baseline = evaluate_membership_alignment(
            tree,
            encoder,
            dataset,
            teacher,
            batch_size=batch_size,
            temperature=temperature,
        )
        encoder.train()
        tree.router_compat.train()
        for epoch in range(1, epochs + 1):
            order = torch.randperm(
                len(dataset),
                generator=generator,
            ).tolist()
            loss_numerator = 0.0
            weight_total = 0.0
            correct_total = 0.0
            target_probability_total = 0.0
            confidence_total = 0.0
            max_gradient_norm = 0.0
            progress_bar = tqdm(
                range(0, len(order), batch_size),
                desc=f"[H-align {epoch:02d}/{epochs:02d}]",
                unit="batch",
                disable=not progress,
            )
            for start in progress_bar:
                indices = order[start : start + batch_size]
                sequences = [dataset[index] for index in indices]
                times, types, valid = _pad_sequence_batch(
                    sequences,
                    device,
                )
                z_prefix, prefix_mask = encoder.forward_padded_prefix(
                    times,
                    types,
                    valid,
                )
                semantic_logits = semantic_branch_logits(
                    tree,
                    z_prefix,
                    detach_node_embeddings=True,
                )
                batch_teacher = LocalBranchTeacher(
                    target=teacher.target[indices].to(device),
                    node_mass=teacher.node_mass[indices].to(device),
                    confidence=teacher.confidence[indices].to(device),
                    internal_ids=teacher.internal_ids,
                )
                loss, metrics = membership_alignment_loss(
                    semantic_logits,
                    batch_teacher,
                    prefix_mask,
                    temperature=temperature,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "non-finite H-tree alignment loss"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    tuple(parameters.values()),
                    max_norm=grad_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()

                batch_weight = float(metrics["weight"].cpu())
                loss_numerator += float(loss.detach().cpu()) * batch_weight
                weight_total += batch_weight
                correct_total += float(metrics["correct"].cpu())
                target_probability_total += float(
                    metrics["target_probability"].cpu()
                )
                confidence_total += float(
                    metrics["teacher_confidence"].cpu()
                )
                max_gradient_norm = max(
                    max_gradient_norm,
                    float(gradient_norm.detach().cpu()),
                )
                progress_bar.set_postfix(
                    loss=loss_numerator / max(weight_total, 1e-12),
                    acc=correct_total / max(weight_total, 1e-12),
                    refresh=False,
                )
            progress_bar.close()
            if weight_total <= 0.0:
                raise RuntimeError(
                    "alignment received zero usable prefix/branch weight"
                )
            epoch_stats = {
                "epoch": epoch,
                "loss": loss_numerator / weight_total,
                "weighted_accuracy": correct_total / weight_total,
                "target_probability": (
                    target_probability_total / weight_total
                ),
                "teacher_confidence": confidence_total / weight_total,
                "gradient_norm_max": max_gradient_norm,
                "weight": weight_total,
            }
            history.append(epoch_stats)
            if progress:
                print(
                    "[H-align] "
                    f"epoch={epoch}/{epochs} "
                    f"loss={epoch_stats['loss']:.6f} "
                    f"acc={epoch_stats['weighted_accuracy']:.4f} "
                    f"p_target={epoch_stats['target_probability']:.4f} "
                    f"confidence={epoch_stats['teacher_confidence']:.4f} "
                    f"grad_max={epoch_stats['gradient_norm_max']:.3e}"
                )

        final_evaluation = evaluate_membership_alignment(
            tree,
            encoder,
            dataset,
            teacher,
            batch_size=batch_size,
            temperature=temperature,
        )

    encoder.train(encoder_was_training)
    tree.router_compat.train(router_was_training)
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "temperature": float(temperature),
        "grad_clip": float(grad_clip),
        "sequence_count": len(dataset),
        "internal_node_count": len(tree.internal_ids),
        "history": history,
        "initial_loss": baseline["loss"],
        "initial_weighted_accuracy": baseline["weighted_accuracy"],
        "initial_target_probability": baseline["target_probability"],
        "final_loss": final_evaluation["loss"],
        "final_weighted_accuracy": final_evaluation["weighted_accuracy"],
        "final_target_probability": final_evaluation[
            "target_probability"
        ],
    }
