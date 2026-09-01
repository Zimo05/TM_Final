"""Data-driven cold-start specialization for Hawkes-tree experts.

This module implements Section 2 of
``WAKE_ROUTING_RETRIEVAL_CONSTRUCTION.md``:

1. compute one low-rank negative Hawkes-gradient signature per sequence;
2. aggregate signatures with the offline H-tree membership;
3. initialize leaf experts with mass-centered prototype directions;
4. initialize internal experts from descendant target mass.

The work runs once before Wake training and is intentionally outside the
event/minibatch hot path.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from AttentionEncoderAdapter import attention_id_to_memory_id
from HawkesBackbone import HawkesFamily
from Wake.HawkesParams import (
    HawkesParams,
    lowrank_project_hawkes_residual,
)


RESIDUAL_SIGNATURE_KEY = "_hawkes_residual_signature"


def _metadata_int(
    sequence: Mapping[str, Any],
    key: str,
) -> int | None:
    value = sequence.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"sequence metadata {key!r} must be scalar")
        value = value.detach().cpu().item()
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"sequence metadata {key!r} must be integer-like"
        ) from error


def _parse_sequence_indices(value: Any) -> tuple[int, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("summary 'sequences' entries must be lists")
    return tuple(int(index) for index in parsed)


def load_h_tree_leaf_membership(
    summary_path: str | Path,
    dataset: Sequence[Mapping[str, Any]],
    leaf_ids: Sequence[str],
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build hard ``q^(0)[sequence, leaf]`` from ``sequence_summary.csv``.

    Cluster metadata is preferred when present on a loaded sequence.  The
    summary's explicit source-row indices are the fallback, allowing the
    construction to work with older datasets that do not carry ``cluster``.
    """
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"H-tree sequence summary not found: {summary_path}"
        )
    summary = pd.read_csv(summary_path)
    required = {"leaf_position", "cluster_id"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(
            "H-tree sequence summary is missing columns: "
            f"{sorted(missing)}"
        )
    if summary.empty:
        raise ValueError("H-tree sequence summary is empty")

    leaf_ids = tuple(str(leaf_id) for leaf_id in leaf_ids)
    leaf_index = {
        leaf_id: index
        for index, leaf_id in enumerate(leaf_ids)
    }
    cluster_to_leaf: Dict[int, str] = {}
    source_to_leaf: Dict[int, str] = {}
    summary_leaves = set()
    for row_index, row in summary.iterrows():
        memory_leaf = attention_id_to_memory_id(str(row["leaf_position"]))
        if memory_leaf not in leaf_index:
            raise ValueError(
                "summary leaf is absent from current H-tree topology: "
                f"{memory_leaf!r}"
            )
        cluster_id = int(row["cluster_id"])
        if (
            cluster_id in cluster_to_leaf
            and cluster_to_leaf[cluster_id] != memory_leaf
        ):
            raise ValueError(
                f"cluster {cluster_id} maps to multiple leaves"
            )
        cluster_to_leaf[cluster_id] = memory_leaf
        summary_leaves.add(memory_leaf)
        if "sequences" in summary.columns:
            for source_index in _parse_sequence_indices(row["sequences"]):
                previous = source_to_leaf.get(source_index)
                if previous is not None and previous != memory_leaf:
                    raise ValueError(
                        f"source sequence {source_index} maps to multiple leaves"
                    )
                source_to_leaf[source_index] = memory_leaf

    missing_leaves = set(leaf_ids).difference(summary_leaves)
    if missing_leaves:
        raise ValueError(
            "H-tree summary does not cover current leaves: "
            f"{sorted(missing_leaves)}"
        )

    membership = torch.zeros(
        len(dataset),
        len(leaf_ids),
        dtype=dtype,
    )
    for dataset_index, sequence in enumerate(dataset):
        cluster_id = _metadata_int(sequence, "cluster_id")
        source_index = _metadata_int(sequence, "source_index")
        if source_index is None:
            source_index = dataset_index

        cluster_leaf = (
            cluster_to_leaf.get(cluster_id)
            if cluster_id is not None
            else None
        )
        source_leaf = source_to_leaf.get(source_index)
        if (
            cluster_leaf is not None
            and source_leaf is not None
            and cluster_leaf != source_leaf
        ):
            raise ValueError(
                "cluster/source membership disagree for dataset sequence "
                f"{dataset_index}: {cluster_leaf!r} != {source_leaf!r}"
            )
        assigned_leaf = cluster_leaf or source_leaf
        if assigned_leaf is None:
            raise ValueError(
                "no H-tree membership for dataset sequence "
                f"{dataset_index} (cluster={cluster_id}, source={source_index})"
            )
        membership[dataset_index, leaf_index[assigned_leaf]] = 1.0

    leaf_mass = membership.sum(dim=0)
    empty = [
        leaf_ids[index]
        for index in torch.nonzero(leaf_mass == 0, as_tuple=False)
        .reshape(-1)
        .tolist()
    ]
    if empty:
        raise ValueError(
            "loaded dataset provides no residual signatures for leaves: "
            f"{empty}"
        )
    return membership


def compute_sequence_residual_signatures(
    hawkes: HawkesFamily,
    dataset: Sequence[Mapping[str, Any]],
    *,
    lowrank_rank: int,
    grad_clip: float = 0.0,
    progress: bool = True,
) -> tuple[Tensor, Dict[str, float]]:
    """Compute ``h_s = P_r(-grad_theta L_Hawkes(s; theta_0))``.

    ``sequence_NLL`` is the complete sequence loss (including an available
    tail interval).  Global norm clipping only rescales a signature and guards
    against one pathological sequence; it does not inject a random direction.
    """
    if not dataset:
        raise ValueError("residual initialization requires a non-empty dataset")
    if lowrank_rank < 0:
        raise ValueError("lowrank_rank must be non-negative")
    if grad_clip < 0.0:
        raise ValueError("grad_clip must be non-negative")

    device = hawkes.raw_mu.device
    base_mu = hawkes.raw_mu.detach()
    base_W = hawkes.raw_W.detach()
    signatures = []
    gradient_norms = []
    clipped_count = 0
    iterator = tqdm(
        dataset,
        total=len(dataset),
        desc="[Residual init]",
        unit="seq",
        disable=not progress,
    )
    was_training = hawkes.training
    hawkes.eval()
    try:
        for sequence in iterator:
            event_count = int(sequence["times"].numel())
            if event_count <= 0:
                raise ValueError(
                    "residual initialization does not support empty sequences"
                )
            model_sequence = {
                "times": sequence["times"].to(device),
                "types": sequence["types"].to(device).long(),
            }
            if "T" in sequence:
                value = sequence["T"]
                model_sequence["T"] = (
                    value.to(device)
                    if torch.is_tensor(value)
                    else torch.as_tensor(
                        value,
                        device=device,
                        dtype=base_mu.dtype,
                    )
                )

            raw_mu = base_mu.clone().requires_grad_(True)
            raw_W = base_W.clone().requires_grad_(True)
            params = HawkesParams(raw_mu, raw_W)
            loss = hawkes.sequence_NLL(
                model_sequence,
                params,
                include_tail=True,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite Hawkes loss during residual initialization"
                )
            grad_mu, grad_W = torch.autograd.grad(
                loss,
                (raw_mu, raw_W),
                create_graph=False,
            )
            grad_norm = torch.sqrt(
                grad_mu.detach().square().sum()
                + grad_W.detach().square().sum()
            )
            gradient_norms.append(float(grad_norm.cpu()))
            clip_multiplier = grad_norm.new_tensor(1.0)
            if grad_clip > 0.0 and float(grad_norm) > grad_clip:
                clip_multiplier = grad_clip / grad_norm.clamp_min(1e-12)
                clipped_count += 1
            negative_gradient = HawkesParams(
                -grad_mu.detach() * clip_multiplier,
                -grad_W.detach() * clip_multiplier,
            )
            projected = lowrank_project_hawkes_residual(
                negative_gradient,
                lowrank_rank,
            )
            signatures.append(torch.cat(
                [
                    projected.mu_tilde.reshape(-1),
                    projected.W_tilde.reshape(-1),
                ],
                dim=0,
            ))
    finally:
        hawkes.train(was_training)

    stacked = torch.stack(signatures, dim=0)
    return stacked, {
        "sequence_count": float(len(dataset)),
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_mean": sum(gradient_norms) / len(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "gradient_clipped_fraction": clipped_count / len(dataset),
    }


def aggregate_leaf_residual_prototypes(
    signatures: Tensor,
    membership: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute H-tree leaf prototypes and their empirical target mass."""
    if signatures.ndim != 2:
        raise ValueError("signatures must have shape [S, P]")
    if membership.ndim != 2 or membership.size(0) != signatures.size(0):
        raise ValueError("membership must have shape [S, L]")
    membership = membership.to(
        device=signatures.device,
        dtype=signatures.dtype,
    )
    if not torch.isfinite(membership).all() or bool((membership < 0).any()):
        raise ValueError("membership must be finite and non-negative")
    row_mass = membership.sum(dim=-1)
    if not torch.allclose(
        row_mass,
        torch.ones_like(row_mass),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("each membership row must sum to one")
    leaf_mass = membership.sum(dim=0)
    if bool((leaf_mass <= 0).any()):
        raise ValueError("every leaf must receive positive membership mass")
    prototypes = (
        membership.transpose(0, 1) @ signatures
    ) / leaf_mass.unsqueeze(-1)
    target_mass = leaf_mass / leaf_mass.sum()
    return prototypes, target_mass


def initialize_tree_from_residual_signatures(
    tree: Any,
    hawkes: HawkesFamily,
    dataset: Sequence[Mapping[str, Any]],
    *,
    summary_path: str | Path,
    init_scale: float,
    lowrank_rank: int,
    grad_clip: float = 0.0,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run the complete residual-signature cold-start construction."""
    membership = load_h_tree_leaf_membership(
        summary_path,
        dataset,
        tree.leaf_ids,
        dtype=hawkes.raw_mu.dtype,
    )
    signatures, gradient_stats = compute_sequence_residual_signatures(
        hawkes,
        dataset,
        lowrank_rank=lowrank_rank,
        grad_clip=grad_clip,
        progress=progress,
    )
    for sequence, signature in zip(dataset, signatures):
        if isinstance(sequence, MutableMapping):
            # Reused by the training-time Regional Probe. This is one cached
            # complete-sequence Hawkes gradient, not one NLL per leaf.
            sequence[RESIDUAL_SIGNATURE_KEY] = signature.detach()
    prototypes, target_mass = aggregate_leaf_residual_prototypes(
        signatures,
        membership,
    )
    # Preserve the original residual-space teacher before semantic
    # initialization applies ``init_scale``. Regional Probe must compare
    # signatures and prototypes in this common, unscaled coordinate system.
    tree.set_residual_probe_prototypes(
        tuple(tree.leaf_ids),
        prototypes,
        target_mass,
    )
    cold_target = torch.cat(
        [
            hawkes.raw_mu.detach().reshape(-1),
            hawkes.raw_W.detach().reshape(-1),
        ],
        dim=0,
    )
    semantic_stats = tree.initialize_semantics_from_residual_prototypes(
        cold_target,
        prototypes,
        target_mass,
        init_scale=init_scale,
    )
    # The same empirical mass that centers semantic initialization must also
    # define the fixed routing prior. Otherwise routing silently falls back to
    # uniform mass over the current leaves and contradicts the data prior.
    tree.frontier_routing.set_target_leaf_mass(
        target_mass,
        leaf_ids=tuple(tree.leaf_ids),
    )
    return {
        **gradient_stats,
        **semantic_stats,
        "lowrank_rank": int(lowrank_rank),
        "grad_clip": float(grad_clip),
        "leaf_ids": list(tree.leaf_ids),
        "leaf_membership_mass": membership.sum(dim=0).tolist(),
        "target_leaf_mass": target_mass.detach().cpu().tolist(),
    }
