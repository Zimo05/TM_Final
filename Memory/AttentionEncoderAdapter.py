"""Offline bridge: Multi-Attention ``H_tree_refined`` -> Memory ``node_emb``.

MESH (PDF Sec. 3.2) separates slow semantic node embeddings ``u_n`` from the
online causal context ``z_t = f_phi(H_t)``.  This module only initializes
``HawkesTree.node_emb`` from a static ``H_tree_refined`` produced by
``AttenEncoderMain_v1.py --node_only``.  Online wake/sleep continues to use
``CausalPrefixEncoder``; there is no runtime ``D_tree`` / cross-attention path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

import torch
from torch import Tensor


PathLike = Union[str, Path]


def attention_id_to_memory_id(node_id: str) -> str:
    """Map Attention IDs (``l_r``) to Memory IDs (``root_L_R``)."""
    if node_id == "root":
        return "root"
    parts = [part for part in node_id.lower().split("_") if part]
    if not parts:
        raise ValueError(f"invalid attention node id: {node_id!r}")
    mapped = []
    for part in parts:
        if part == "l":
            mapped.append("L")
        elif part == "r":
            mapped.append("R")
        else:
            raise ValueError(
                f"attention node id {node_id!r} has unsupported segment {part!r}"
            )
    return "root_" + "_".join(mapped)


def memory_id_to_attention_id(node_id: str) -> str:
    """Map Memory IDs (``root_L_R``) to Attention IDs (``l_r``)."""
    if node_id == "root":
        return "root"
    if node_id.startswith("root_"):
        node_id = node_id[len("root_") :]
    return node_id.lower()


def synchronize_tree_topology_from_node_ids(
    tree: Any,
    node_ids: Sequence[str],
) -> tuple[str, ...]:
    """Expand ``tree`` so its binary topology exactly matches Attention IDs.

    The operation is expansion-only. Call it before creating a trainer or
    optimizer, normally on a depth-0 Memory tree. A malformed or partial binary
    Attention topology is rejected instead of leaving random extra nodes.
    """
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node_ids must be unique")
    target = {attention_id_to_memory_id(str(node_id)) for node_id in node_ids}
    if "root" not in target:
        raise ValueError("Attention topology must contain 'root'")

    existing_extra = set(tree.all_node_ids).difference(target)
    if existing_extra:
        raise ValueError(
            "Memory tree contains nodes absent from Attention topology: "
            f"{sorted(existing_extra)}; construct it with init_depth=0"
        )

    ordered = sorted(target, key=lambda node_id: (node_id.count("_"), node_id))
    for node_id in ordered:
        if node_id not in tree.nodes:
            raise ValueError(
                f"Attention topology is missing the parent path for {node_id!r}"
            )
        left_id = f"{node_id}_L"
        right_id = f"{node_id}_R"
        has_left = left_id in target
        has_right = right_id in target
        if has_left != has_right:
            raise ValueError(
                f"Attention node {memory_id_to_attention_id(node_id)!r} must have "
                "either zero or two children"
            )
        if has_left:
            if not tree.nodes[node_id].is_leaf:
                if (
                    tree.nodes[node_id].left != left_id
                    or tree.nodes[node_id].right != right_id
                ):
                    raise ValueError(f"Memory topology conflicts at {node_id!r}")
            else:
                tree.split_leaf(node_id)

    actual = set(tree.all_node_ids)
    if actual != target:
        raise RuntimeError(
            "failed to synchronize Attention/Memory topology: "
            f"missing={sorted(target - actual)}, extra={sorted(actual - target)}"
        )
    return tuple(tree.all_node_ids)


def load_h_tree(path: PathLike) -> Tuple[Tuple[str, ...], Tensor]:
    """Load ``H_tree_refined`` and node ids from a ``.pt`` saved by the encoder.

    Accepts either ``node_only`` saves (``H_tree_refined``) or full pipeline
    saves that also contain ``D_tree`` (``D_tree`` is ignored).
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"h_tree file must contain a dict, got {type(payload)!r}")

    node_ids = payload.get("node_ids")
    if node_ids is None:
        raise KeyError("h_tree file missing 'node_ids'")
    node_ids = tuple(str(node_id) for node_id in node_ids)

    h_tree = payload.get("H_tree_refined")
    if h_tree is None:
        h_tree = payload.get("H_tree")
    if h_tree is None:
        raise KeyError("h_tree file missing 'H_tree_refined' (or 'H_tree')")
    if not isinstance(h_tree, Tensor):
        raise TypeError("H_tree_refined must be a torch.Tensor")
    if h_tree.ndim != 2:
        raise ValueError(f"H_tree_refined must have shape [N, d], got {tuple(h_tree.shape)}")
    if h_tree.size(0) != len(node_ids):
        raise ValueError(
            f"H_tree_refined rows ({h_tree.size(0)}) != len(node_ids) ({len(node_ids)})"
        )
    return node_ids, h_tree.detach().float()


def h_tree_from_pipeline(pipeline: Any) -> Tuple[Tuple[str, ...], Tensor]:
    """Read refined node embeddings from a prepared attention pipeline."""
    node_ids = getattr(pipeline, "node_ids", None)
    h_tree = getattr(pipeline, "H_tree_refined", None)
    if h_tree is None:
        h_tree = getattr(pipeline, "H_tree", None)
    if not node_ids:
        raise RuntimeError("attention pipeline is missing node_ids")
    if h_tree is None:
        raise RuntimeError(
            "attention pipeline has no H_tree_refined; run Phase 1-3 "
            "(e.g. run(..., node_only=True)) first"
        )
    if not isinstance(h_tree, Tensor) or h_tree.ndim != 2:
        raise ValueError("pipeline H_tree_refined must have shape [N, d]")
    if h_tree.size(0) != len(node_ids):
        raise ValueError("pipeline H_tree_refined rows must match node_ids")
    return tuple(str(node_id) for node_id in node_ids), h_tree.detach().float().cpu()


@torch.no_grad()
def initialize_node_embeddings_from_h_tree(
    tree: Any,
    h_tree: Tensor,
    node_ids: Sequence[str],
    *,
    strict_coverage: bool = False,
) -> dict[str, str]:
    """Copy matching rows of ``H_tree_refined`` into ``tree.node_emb``.

    Only Memory nodes that have an exact Attention counterpart are overwritten.
    Dynamically split leaves without an Attention id keep their existing
    embedding (random init or sleep fit).

    Returns:
        Mapping from Memory node id to the Attention id used for initialization.
    """
    if h_tree.ndim != 2:
        raise ValueError(f"h_tree must have shape [N, d], got {tuple(h_tree.shape)}")
    if h_tree.size(0) != len(node_ids):
        raise ValueError("h_tree rows must match node_ids")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node_ids must be unique")
    if int(tree.node_dim) != int(h_tree.size(-1)):
        raise ValueError(
            f"tree.node_dim={tree.node_dim} must equal H_tree dim={h_tree.size(-1)}; "
            "set --node-dim to the encoder d_model"
        )

    by_attention = {
        memory_id_to_attention_id(node_id): index
        for index, node_id in enumerate(node_ids)
    }
    # Also accept already-canonical attention ids stored in the file.
    for index, node_id in enumerate(node_ids):
        by_attention.setdefault(str(node_id).lower(), index)

    device = tree.node_emb[tree.all_node_ids[0]].device
    dtype = tree.node_emb[tree.all_node_ids[0]].dtype
    h_tree = h_tree.to(device=device, dtype=dtype)

    initialized: dict[str, str] = {}
    for memory_id in tree.all_node_ids:
        attention_id = memory_id_to_attention_id(memory_id)
        index = by_attention.get(attention_id)
        if index is None:
            continue
        tree.node_emb[memory_id].copy_(h_tree[index])
        initialized[memory_id] = attention_id

    if strict_coverage and not initialized:
        raise RuntimeError("no Memory nodes matched Attention node_ids in H_tree")
    return initialized


def initialize_tree_from_h_tree_file(
    tree: Any,
    path: PathLike,
    *,
    strict_coverage: bool = False,
    synchronize_topology: bool = False,
) -> dict[str, str]:
    """Load an encoder ``.pt`` and initialize ``tree.node_emb``."""
    node_ids, h_tree = load_h_tree(path)
    if synchronize_topology:
        synchronize_tree_topology_from_node_ids(tree, node_ids)
    return initialize_node_embeddings_from_h_tree(
        tree,
        h_tree,
        node_ids,
        strict_coverage=strict_coverage,
    )


def initialize_tree_from_pipeline(
    tree: Any,
    pipeline: Any,
    *,
    strict_coverage: bool = False,
) -> dict[str, str]:
    """Initialize ``tree.node_emb`` from an in-memory attention pipeline."""
    node_ids, h_tree = h_tree_from_pipeline(pipeline)
    return initialize_node_embeddings_from_h_tree(
        tree,
        h_tree,
        node_ids,
        strict_coverage=strict_coverage,
    )
