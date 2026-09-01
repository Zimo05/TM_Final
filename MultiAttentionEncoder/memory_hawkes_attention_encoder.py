"""
Memory Hawkes Tree Attention Encoder
====================================

This file implements the pipeline:

    THP sequence embeddings
        -> node prototype pooling
        -> structural tree self-attention
        -> sample THP embedding queries encoded tree by cross-attention
        -> sample-conditioned deterministic tree features for TreeVAE / Memory Hawkes Tree

The file does NOT implement THP itself. It expects THP outputs as tensors:
    node_seq_emb: padded THP embeddings of sequences assigned to each tree node, [N, S, D_seq]
    sample_z: THP embedding of the current query sequence/sample, [B, D_sample]

Main classes:
    - TreeStructureBuilder: builds adjacency mask and pairwise relation tensors from nested tree json.
    - RelationBias: converts pairwise structural relations into per-head attention bias.
    - StructuralTreeEncoder: encodes static tree H_tree with masked relation-biased self-attention.
    - TreeNodeCrossAttention: lets current sample embedding query H_tree and outputs D_tree(x).
    - MemoryHawkesAttentionEncoder: end-to-end wrapper for node pooling + tree encoding + cross-attention.

Author: generated for Memory Hawkes Tree / TreeVAE integration.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Basic neural modules
# -----------------------------------------------------------------------------


class MLP(nn.Module):
    """Simple GELU MLP."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentiveSetPool(nn.Module):
    """
    Pool THP sequence embeddings assigned to one tree node.

    Input:
        seq_emb:    [N, S, D_seq]
        valid_mask: [N, S], True means this sequence slot is valid.

    Output:
        node_proto: [N, D_seq]

    This implements:
        alpha_{i,s} = softmax(score(z_{i,s})) over sequences inside node i
        u_i = sum_s alpha_{i,s} z_{i,s}
    """

    def __init__(self, d_seq: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or d_seq
        self.score = nn.Sequential(
            nn.LayerNorm(d_seq),
            nn.Linear(d_seq, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seq_emb: torch.Tensor, valid_mask: torch.BoolTensor) -> torch.Tensor:
        if seq_emb.ndim != 3:
            raise ValueError("seq_emb must have shape [N, S, D_seq]")
        if valid_mask.ndim != 2:
            raise ValueError("valid_mask must have shape [N, S]")
        if seq_emb.shape[:2] != valid_mask.shape:
            raise ValueError("seq_emb and valid_mask disagree on [N, S]")

        scores = self.score(seq_emb).squeeze(-1)  # [N, S]
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        alpha = F.softmax(scores, dim=-1)  # [N, S]
        alpha = torch.where(valid_mask, alpha, torch.zeros_like(alpha))

        denom = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        alpha = alpha / denom
        return torch.bmm(alpha.unsqueeze(1), seq_emb).squeeze(1)  # [N, D_seq]


class NodeFeatureProjector(nn.Module):
    """
    Build initial tree node feature x_i from:
        u_i: THP pooled sequence prototype
        g_i: Hawkes/statistical node features
        s_i: explicit structural node features

    x_i = Proj([u_i; g_i; s_i])
    """

    def __init__(
        self,
        d_seq: int,
        d_stats: int,
        d_struct: int,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (2 * d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_seq + d_stats + d_struct, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        node_proto: torch.Tensor,
        node_stats: torch.Tensor,
        node_struct_features: torch.Tensor,
    ) -> torch.Tensor:
        if node_proto.ndim != 2 or node_stats.ndim != 2 or node_struct_features.ndim != 2:
            raise ValueError("node_proto, node_stats, node_struct_features must all be [N, D]")
        n = node_proto.size(0)
        if node_stats.size(0) != n or node_struct_features.size(0) != n:
            raise ValueError("all node features must share the same N")
        return self.proj(torch.cat([node_proto, node_stats, node_struct_features], dim=-1))


# -----------------------------------------------------------------------------
# Tree structure building
# -----------------------------------------------------------------------------


@dataclass
class TreeSpec:
    """Canonical tensor-ready representation of one rooted tree."""

    node_ids: List[str]
    parent_index: torch.LongTensor        # [N], root parent = -1
    depth: torch.LongTensor               # [N]
    side: torch.LongTensor                # [N], root=0, left=1, right=2, unknown=0
    is_leaf: torch.BoolTensor             # [N]
    children: List[List[int]]


@dataclass
class TreeRelationTensors:
    """All pairwise structural tensors consumed by RelationBias / StructuralTreeEncoder."""

    adj_mask: torch.BoolTensor            # [N, N], True means attention is allowed
    dist: torch.LongTensor                # [N, N], clipped tree distance
    anc: torch.LongTensor                 # [N, N], 0 none, 1 i ancestor of j, 2 j ancestor of i, 3 same
    sibling: torch.LongTensor             # [N, N], 0/1
    lca_depth: torch.LongTensor           # [N, N], clipped LCA depth
    branch_rel: torch.LongTensor          # [N, N], 0 none/same/ancestor, 1 LL, 2 LR, 3 RL, 4 RR
    node_valid_mask: torch.BoolTensor     # [N]

    def to(self, device: torch.device | str) -> "TreeRelationTensors":
        return TreeRelationTensors(
            adj_mask=self.adj_mask.to(device),
            dist=self.dist.to(device),
            anc=self.anc.to(device),
            sibling=self.sibling.to(device),
            lca_depth=self.lca_depth.to(device),
            branch_rel=self.branch_rel.to(device),
            node_valid_mask=self.node_valid_mask.to(device),
        )


class TreeStructureBuilder:
    """
    Build structural matrices from either:
        1) a nested dict tree with keys like {node_id, left, right, children, leaf}, or
        2) an explicit TreeSpec.

    Attention mask modes:
        - local: self + parent + children
        - local_sibling: local + siblings
        - ancestor: self + ancestors + descendants
        - full: all nodes can attend to all nodes
    """

    SIDE_ROOT = 0
    SIDE_LEFT = 1
    SIDE_RIGHT = 2

    def __init__(
        self,
        max_distance: int = 8,
        max_lca_depth: int = 8,
        mask_mode: str = "local",
    ):
        self.max_distance = max_distance
        self.max_lca_depth = max_lca_depth
        if mask_mode not in {"local", "local_sibling", "ancestor", "full"}:
            raise ValueError("mask_mode must be one of: local, local_sibling, ancestor, full")
        self.mask_mode = mask_mode

    @staticmethod
    def from_nested_dict(tree: Dict, sort_children: bool = False) -> TreeSpec:
        """
        Convert a nested tree dict to TreeSpec.

        Supported patterns:
            {'node_id': 'root', 'left': {...}, 'right': {...}, 'leaf': False}
            {'id': 'root', 'children': [{...}, {...}]}

        For binary trees, left/right are preferred because branch relation can use L/R semantics.
        """
        node_ids: List[str] = []
        parent: List[int] = []
        depth: List[int] = []
        side: List[int] = []
        child_lists: List[List[int]] = []

        def get_node_id(obj: Dict, fallback: str) -> str:
            return str(obj.get("node_id", obj.get("id", obj.get("name", fallback))))

        def add_node(obj: Dict, parent_idx: int, dep: int, side_code: int, fallback: str) -> int:
            idx = len(node_ids)
            node_ids.append(get_node_id(obj, fallback))
            parent.append(parent_idx)
            depth.append(dep)
            side.append(side_code)
            child_lists.append([])

            raw_children: List[Tuple[Dict, int, str]] = []
            if isinstance(obj.get("left"), dict):
                raw_children.append((obj["left"], TreeStructureBuilder.SIDE_LEFT, f"{node_ids[idx]}_L"))
            if isinstance(obj.get("right"), dict):
                raw_children.append((obj["right"], TreeStructureBuilder.SIDE_RIGHT, f"{node_ids[idx]}_R"))
            if "children" in obj and isinstance(obj["children"], list):
                for c_id, child in enumerate(obj["children"]):
                    # For generic children, keep first as left, second as right, remaining unknown/root side.
                    if c_id == 0:
                        c_side = TreeStructureBuilder.SIDE_LEFT
                    elif c_id == 1:
                        c_side = TreeStructureBuilder.SIDE_RIGHT
                    else:
                        c_side = TreeStructureBuilder.SIDE_ROOT
                    raw_children.append((child, c_side, f"{node_ids[idx]}_{c_id}"))

            if sort_children:
                raw_children.sort(key=lambda x: get_node_id(x[0], x[2]))

            for child_obj, child_side, child_fallback in raw_children:
                child_idx = add_node(child_obj, idx, dep + 1, child_side, child_fallback)
                child_lists[idx].append(child_idx)
            return idx

        add_node(tree, -1, 0, TreeStructureBuilder.SIDE_ROOT, "root")
        is_leaf = [len(c) == 0 for c in child_lists]
        return TreeSpec(
            node_ids=node_ids,
            parent_index=torch.tensor(parent, dtype=torch.long),
            depth=torch.tensor(depth, dtype=torch.long),
            side=torch.tensor(side, dtype=torch.long),
            is_leaf=torch.tensor(is_leaf, dtype=torch.bool),
            children=child_lists,
        )

    def build(self, spec: TreeSpec) -> TreeRelationTensors:
        n = len(spec.node_ids)
        parent = spec.parent_index.tolist()
        depth = spec.depth.tolist()
        side = spec.side.tolist()

        ancestors: List[List[int]] = []
        ancestor_sets: List[set[int]] = []
        for i in range(n):
            path = []
            cur = i
            while cur != -1:
                path.append(cur)
                cur = parent[cur]
            path_from_root = list(reversed(path))
            ancestors.append(path_from_root)
            ancestor_sets.append(set(path_from_root))

        lca = torch.zeros(n, n, dtype=torch.long)
        lca_depth = torch.zeros(n, n, dtype=torch.long)
        dist = torch.zeros(n, n, dtype=torch.long)
        anc = torch.zeros(n, n, dtype=torch.long)
        sibling = torch.zeros(n, n, dtype=torch.long)
        branch_rel = torch.zeros(n, n, dtype=torch.long)
        adj_mask = torch.zeros(n, n, dtype=torch.bool)

        for i in range(n):
            for j in range(n):
                if i == j:
                    a = 3
                elif i in ancestor_sets[j]:
                    a = 1
                elif j in ancestor_sets[i]:
                    a = 2
                else:
                    a = 0
                anc[i, j] = a

                l = self._lca(i, j, ancestors)
                lca[i, j] = l
                lca_depth[i, j] = min(depth[l], self.max_lca_depth)
                d_ij = depth[i] + depth[j] - 2 * depth[l]
                dist[i, j] = min(d_ij, self.max_distance)

                is_sib = (i != j) and (parent[i] != -1) and (parent[i] == parent[j])
                sibling[i, j] = 1 if is_sib else 0

                branch_rel[i, j] = self._branch_relation(i, j, l, ancestors, side)
                adj_mask[i, j] = self._allow_attention(i, j, parent, spec.children, ancestor_sets)

        return TreeRelationTensors(
            adj_mask=adj_mask,
            dist=dist,
            anc=anc,
            sibling=sibling,
            lca_depth=lca_depth,
            branch_rel=branch_rel,
            node_valid_mask=torch.ones(n, dtype=torch.bool),
        )

    @staticmethod
    def _lca(i: int, j: int, ancestors: List[List[int]]) -> int:
        ai = ancestors[i]
        aj = ancestors[j]
        lca = ai[0]
        for x, y in zip(ai, aj):
            if x == y:
                lca = x
            else:
                break
        return lca

    @staticmethod
    def _child_after_lca(node: int, lca: int, ancestors: List[List[int]]) -> Optional[int]:
        path = ancestors[node]
        if node == lca:
            return None
        pos = path.index(lca)
        return path[pos + 1]

    def _branch_relation(
        self,
        i: int,
        j: int,
        lca: int,
        ancestors: List[List[int]],
        side: List[int],
    ) -> int:
        """
        Return 0 if same/ancestor relation cannot define two outgoing branches.
        Else map first branch after LCA:
            LL -> 1, LR -> 2, RL -> 3, RR -> 4
        """
        ci = self._child_after_lca(i, lca, ancestors)
        cj = self._child_after_lca(j, lca, ancestors)
        if ci is None or cj is None:
            return 0
        si, sj = side[ci], side[cj]
        if si == self.SIDE_LEFT and sj == self.SIDE_LEFT:
            return 1
        if si == self.SIDE_LEFT and sj == self.SIDE_RIGHT:
            return 2
        if si == self.SIDE_RIGHT and sj == self.SIDE_LEFT:
            return 3
        if si == self.SIDE_RIGHT and sj == self.SIDE_RIGHT:
            return 4
        return 0

    def _allow_attention(
        self,
        i: int,
        j: int,
        parent: List[int],
        children: List[List[int]],
        ancestor_sets: List[set[int]],
    ) -> bool:
        if self.mask_mode == "full":
            return True
        if i == j:
            return True
        if self.mask_mode == "local":
            return parent[i] == j or parent[j] == i
        if self.mask_mode == "local_sibling":
            return parent[i] == j or parent[j] == i or ((parent[i] != -1) and parent[i] == parent[j])
        if self.mask_mode == "ancestor":
            return i in ancestor_sets[j] or j in ancestor_sets[i]
        raise RuntimeError("unreachable")


# -----------------------------------------------------------------------------
# Relation-biased attention
# -----------------------------------------------------------------------------


class RelationBias(nn.Module):
    """
    Convert pairwise tree relation tensors to per-head additive attention bias.

    Inputs are [N, N] integer tensors:
        dist:       clipped tree distance in [0, max_distance]
        anc:        0 none, 1 i ancestor of j, 2 j ancestor of i, 3 same
        sibling:    0/1
        lca_depth:  clipped LCA depth in [0, max_lca_depth]
        branch_rel: 0 none/same/ancestor, 1 LL, 2 LR, 3 RL, 4 RR

    Output:
        rel_bias: [H, N, N]
    """

    def __init__(
        self,
        num_heads: int = 4,
        max_distance: int = 8,
        max_lca_depth: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.max_lca_depth = max_lca_depth
        self.dist_emb = nn.Embedding(max_distance + 1, num_heads)
        self.anc_emb = nn.Embedding(4, num_heads)
        self.sib_emb = nn.Embedding(2, num_heads)
        self.lca_emb = nn.Embedding(max_lca_depth + 1, num_heads)
        self.branch_emb = nn.Embedding(5, num_heads)

    def forward(
        self,
        dist: torch.LongTensor,
        anc: torch.LongTensor,
        sibling: torch.LongTensor,
        lca_depth: torch.LongTensor,
        branch_rel: torch.LongTensor,
    ) -> torch.Tensor:
        dist = dist.clamp(0, self.max_distance)
        lca_depth = lca_depth.clamp(0, self.max_lca_depth)
        bias = (
            self.dist_emb(dist)
            + self.anc_emb(anc)
            + self.sib_emb(sibling)
            + self.lca_emb(lca_depth)
            + self.branch_emb(branch_rel)
        )  # [N, N, H]
        return bias.permute(2, 0, 1).contiguous()  # [H, N, N]


class MultiHeadAttentionWithBias(nn.Module):
    """Standard MHA with optional boolean mask and optional additive relation bias."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        query: torch.Tensor,                    # [B, Lq, D]
        key: torch.Tensor,                      # [B, Lk, D]
        value: torch.Tensor,                    # [B, Lk, D]
        attn_mask: Optional[torch.BoolTensor] = None,   # [Lq, Lk] or [B, Lq, Lk]
        rel_bias: Optional[torch.Tensor] = None,         # [H, Lq, Lk] or [B, H, Lq, Lk]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B, H, Lq, Lk]

        if rel_bias is not None:
            if rel_bias.ndim == 3:
                scores = scores + rel_bias.unsqueeze(0)
            elif rel_bias.ndim == 4:
                scores = scores + rel_bias
            else:
                raise ValueError("rel_bias must have ndim 3 or 4")

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Lq, Lk]
            elif attn_mask.ndim == 3:
                mask = attn_mask.unsqueeze(1)               # [B, 1, Lq, Lk]
            else:
                raise ValueError("attn_mask must have ndim 2 or 3")
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, H, Lq, Hd]
        out = out.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], self.d_model)
        out = self.o_proj(out)
        return out, attn


class StructuralTreeBlock(nn.Module):
    """One relation-biased Transformer block over all nodes in one tree."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionWithBias(d_model, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,                       # [N, D]
        adj_mask: torch.BoolTensor,            # [N, N]
        rel_bias: torch.Tensor,                # [H, N, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.ln1(x).unsqueeze(0)
        y, attn = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=adj_mask,
            rel_bias=rel_bias,
        )
        x = x + y.squeeze(0)
        x = x + self.ff(self.ln2(x))
        return x, attn.squeeze(0)  # [N, D], [H, N, N]


@dataclass
class StructuralTreeEncoderOutput:
    h_tree: torch.Tensor               # [N, D]
    rel_bias: torch.Tensor             # [H, N, N]
    layer_attn: List[torch.Tensor]      # each [H, N, N]


class StructuralTreeEncoder(nn.Module):
    """
    Stage A: static structural tree encoder.

    This is where RelationBias is actually called.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 2,
        max_distance: int = 8,
        max_lca_depth: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.relation_bias = RelationBias(
            num_heads=num_heads,
            max_distance=max_distance,
            max_lca_depth=max_lca_depth,
        )
        self.layers = nn.ModuleList([
            StructuralTreeBlock(d_model, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, node_x: torch.Tensor, relations: TreeRelationTensors) -> StructuralTreeEncoderOutput:
        rel_bias = self.relation_bias(
            dist=relations.dist,
            anc=relations.anc,
            sibling=relations.sibling,
            lca_depth=relations.lca_depth,
            branch_rel=relations.branch_rel,
        )  # [H, N, N]

        h = node_x
        attn_list: List[torch.Tensor] = []
        for layer in self.layers:
            h, attn = layer(h, adj_mask=relations.adj_mask, rel_bias=rel_bias)
            attn_list.append(attn)
        h = self.out_ln(h)
        return StructuralTreeEncoderOutput(h_tree=h, rel_bias=rel_bias, layer_attn=attn_list)


# -----------------------------------------------------------------------------
# Sample-to-tree cross attention
# -----------------------------------------------------------------------------


@dataclass
class CrossAttentionOutput:
    deterministic_tree: torch.Tensor    # [B, N, D]
    retrieved_tree: torch.Tensor        # [B, N, D]
    attn_weights: torch.Tensor          # [B, H, N, N]
    node_relevance: torch.Tensor        # [B, N]


class TreeNodeCrossAttention(nn.Module):
    """
    Convert static encoded tree H_tree = {h_i} into sample-conditioned deterministic features.

    Given a current sample THP embedding z_x:
        q_i(x) = MLP([h_i; z_x; h_i * z_x; |h_i - z_x|])
        c_i(x) = CrossAttention(q_i(x), H_tree, H_tree)
        d_i(x) = MLP([m_i; z_x; m_i * z_x; |m_i - z_x|])

    Output D_tree(x) = {d_i(x)} is the deterministic feature tree for TreeVAE / Memory Hawkes Tree.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        query_hidden: Optional[int] = None,
        fusion_hidden: Optional[int] = None,
        use_residual_det: bool = True,
    ):
        super().__init__()
        query_hidden = query_hidden or (2 * d_model)
        fusion_hidden = fusion_hidden or (2 * d_model)
        self.use_residual_det = use_residual_det

        self.tree_ln = nn.LayerNorm(d_model)
        self.sample_ln = nn.LayerNorm(d_model)
        self.cross_ln = nn.LayerNorm(d_model)

        self.query_mlp = MLP(4 * d_model, query_hidden, d_model, dropout=dropout)
        self.attn = MultiHeadAttentionWithBias(d_model, num_heads, dropout=dropout)

        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.det_mlp = MLP(4 * d_model, fusion_hidden, d_model, dropout=dropout)
        self.det_ln = nn.LayerNorm(d_model)

    @staticmethod
    def _normalize_sample(sample_z: torch.Tensor) -> torch.Tensor:
        if sample_z.ndim == 1:
            sample_z = sample_z.unsqueeze(0)
        elif sample_z.ndim == 3:
            if sample_z.size(1) != 1:
                raise ValueError("sample_z with ndim=3 must have shape [B, 1, D]")
            sample_z = sample_z.squeeze(1)
        elif sample_z.ndim != 2:
            raise ValueError("sample_z must have shape [D], [B, D], or [B, 1, D]")
        return sample_z

    @staticmethod
    def _expand_tree(tree_h: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tree_h.ndim == 2:
            tree_h = tree_h.unsqueeze(0).expand(batch_size, -1, -1)
        elif tree_h.ndim == 3:
            if tree_h.size(0) != batch_size:
                raise ValueError("tree_h batch size must match sample batch size")
        else:
            raise ValueError("tree_h must have shape [N, D] or [B, N, D]")
        return tree_h

    @staticmethod
    def _build_attn_mask(
        node_valid_mask: Optional[torch.BoolTensor],
        batch_size: int,
        num_queries: int,
    ) -> Optional[torch.BoolTensor]:
        if node_valid_mask is None:
            return None
        if node_valid_mask.ndim == 1:
            node_valid_mask = node_valid_mask.unsqueeze(0).expand(batch_size, -1)
        elif node_valid_mask.ndim != 2:
            raise ValueError("node_valid_mask must have shape [N] or [B, N]")
        return node_valid_mask.unsqueeze(1).expand(-1, num_queries, -1)  # [B, Nq, Nk]

    def forward(
        self,
        tree_h: torch.Tensor,                           # [N, D] or [B, N, D]
        sample_z: torch.Tensor,                         # [D], [B, D], or [B, 1, D]
        node_valid_mask: Optional[torch.BoolTensor] = None,
        rel_bias: Optional[torch.Tensor] = None,
    ) -> CrossAttentionOutput:
        sample_z = self._normalize_sample(sample_z)
        b = sample_z.size(0)
        tree_h = self._expand_tree(tree_h, b)
        b, n, d = tree_h.shape

        sample_expand = sample_z.unsqueeze(1).expand(b, n, d)
        h_norm = self.tree_ln(tree_h)
        z_norm = self.sample_ln(sample_expand)

        query_input = torch.cat([h_norm, z_norm, h_norm * z_norm, torch.abs(h_norm - z_norm)], dim=-1)
        query = self.query_mlp(query_input)  # [B, N, D]

        attn_mask = self._build_attn_mask(node_valid_mask, batch_size=b, num_queries=n)
        retrieved, attn = self.attn(
            query=query,
            key=h_norm,
            value=h_norm,
            attn_mask=attn_mask,
            rel_bias=rel_bias,
        )

        gate = self.gate(torch.cat([tree_h, retrieved, sample_expand], dim=-1))
        mixed = (1.0 - gate) * tree_h + gate * retrieved
        mixed = self.cross_ln(mixed + tree_h)

        det_input = torch.cat([mixed, sample_expand, mixed * sample_expand, torch.abs(mixed - sample_expand)], dim=-1)
        deterministic_tree = self.det_mlp(det_input)
        if self.use_residual_det:
            deterministic_tree = self.det_ln(deterministic_tree + mixed)
        else:
            deterministic_tree = self.det_ln(deterministic_tree)

        # This is node relevance, not strict root-to-leaf TreeVAE path probability.
        node_relevance = attn.mean(dim=1).mean(dim=1)  # [B, N]

        if node_valid_mask is not None:
            if node_valid_mask.ndim == 1:
                node_valid_mask = node_valid_mask.unsqueeze(0).expand(b, -1)
            mask_f = node_valid_mask.unsqueeze(-1).to(deterministic_tree.dtype)
            deterministic_tree = deterministic_tree * mask_f
            retrieved = retrieved * mask_f
            node_relevance = node_relevance * node_valid_mask.to(node_relevance.dtype)
            node_relevance = node_relevance / node_relevance.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        return CrossAttentionOutput(
            deterministic_tree=deterministic_tree,
            retrieved_tree=retrieved,
            attn_weights=attn,
            node_relevance=node_relevance,
        )


# -----------------------------------------------------------------------------
# Full wrapper: THP embeddings -> node prototypes -> H_tree -> D_tree(x)
# -----------------------------------------------------------------------------


@dataclass
class AttentionEncoderOutput:
    node_x: torch.Tensor                         # [N, D]
    h_tree: torch.Tensor                         # [N, D]
    deterministic_tree: torch.Tensor             # [B, N, D]
    retrieved_tree: torch.Tensor                 # [B, N, D]
    node_relevance: torch.Tensor                 # [B, N]
    rel_bias: torch.Tensor                       # [H, N, N]
    structural_attn: List[torch.Tensor]          # each [H, N, N]
    cross_attn: torch.Tensor                     # [B, H, N, N]


class MemoryHawkesAttentionEncoder(nn.Module):
    """
    End-to-end attention encoder for the user's current logic:

        THP sequence embeddings
            -> attention pooling into node prototypes
            -> node feature projection
            -> structural relation-biased tree encoder
            -> current sample THP embedding queries tree by cross-attention
            -> deterministic feature tree D_tree(x)

    This module assumes one static tree per forward call. For batching multiple trees with different
    node counts, pad node tensors and extend TreeRelationTensors with batch dimensions.
    """

    def __init__(
        self,
        d_seq: int,
        d_sample: int,
        d_stats: int,
        d_struct: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_tree_layers: int = 2,
        max_distance: int = 8,
        max_lca_depth: int = 8,
        dropout: float = 0.1,
        use_cross_relation_bias: bool = False,
    ):
        super().__init__()
        self.use_cross_relation_bias = use_cross_relation_bias

        self.node_pool = AttentiveSetPool(d_seq=d_seq, hidden_dim=d_model, dropout=dropout)
        self.node_projector = NodeFeatureProjector(
            d_seq=d_seq,
            d_stats=d_stats,
            d_struct=d_struct,
            d_model=d_model,
            dropout=dropout,
        )
        self.sample_proj = nn.Sequential(
            nn.Linear(d_sample, d_model),
            nn.LayerNorm(d_model),
        )
        self.tree_encoder = StructuralTreeEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_tree_layers,
            max_distance=max_distance,
            max_lca_depth=max_lca_depth,
            dropout=dropout,
        )
        self.cross_attention = TreeNodeCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        node_seq_emb: torch.Tensor,                  # [N, S, D_seq], THP seq embeddings per node
        node_seq_mask: torch.BoolTensor,             # [N, S]
        node_stats: torch.Tensor,                    # [N, D_stats]
        node_struct_features: torch.Tensor,           # [N, D_struct]
        relations: TreeRelationTensors,
        sample_z: torch.Tensor,                      # [B, D_sample] or [D_sample]
    ) -> AttentionEncoderOutput:
        device = node_seq_emb.device
        relations = relations.to(device)
        node_seq_mask = node_seq_mask.to(device)
        node_stats = node_stats.to(device)
        node_struct_features = node_struct_features.to(device)
        sample_z = sample_z.to(device)

        # 1) THP embeddings inside each node -> node prototype u_i.
        node_proto = self.node_pool(node_seq_emb, node_seq_mask)  # [N, D_seq]

        # 2) Build x_i = Proj([u_i; g_i; s_i]).
        node_x = self.node_projector(node_proto, node_stats, node_struct_features)  # [N, D]

        # 3) Encode static tree with structural self-attention.
        tree_out = self.tree_encoder(node_x, relations)
        h_tree = tree_out.h_tree  # [N, D]

        # 4) Current sample THP embedding queries encoded tree.
        sample_z_model = self.sample_proj(TreeNodeCrossAttention._normalize_sample(sample_z))
        cross_rel_bias = tree_out.rel_bias if self.use_cross_relation_bias else None
        cross_out = self.cross_attention(
            tree_h=h_tree,
            sample_z=sample_z_model,
            node_valid_mask=relations.node_valid_mask,
            rel_bias=cross_rel_bias,
        )

        return AttentionEncoderOutput(
            node_x=node_x,
            h_tree=h_tree,
            deterministic_tree=cross_out.deterministic_tree,
            retrieved_tree=cross_out.retrieved_tree,
            node_relevance=cross_out.node_relevance,
            rel_bias=tree_out.rel_bias,
            structural_attn=tree_out.layer_attn,
            cross_attn=cross_out.attn_weights,
        )


# -----------------------------------------------------------------------------
# Optional helper: simple structural features from TreeSpec
# -----------------------------------------------------------------------------


def make_basic_node_struct_features(spec: TreeSpec) -> torch.Tensor:
    """
    Build simple explicit structural features s_i.

    Columns:
        depth_norm, is_root, is_leaf, is_left, is_right, num_children_norm

    Output:
        [N, 6]
    """
    n = len(spec.node_ids)
    depth = spec.depth.float()
    max_depth = depth.max().clamp_min(1.0)
    depth_norm = depth / max_depth
    is_root = (spec.parent_index == -1).float()
    is_leaf = spec.is_leaf.float()
    is_left = (spec.side == TreeStructureBuilder.SIDE_LEFT).float()
    is_right = (spec.side == TreeStructureBuilder.SIDE_RIGHT).float()
    num_children = torch.tensor([len(c) for c in spec.children], dtype=torch.float)
    num_children_norm = num_children / num_children.max().clamp_min(1.0)
    return torch.stack([depth_norm, is_root, is_leaf, is_left, is_right, num_children_norm], dim=-1)


# -----------------------------------------------------------------------------
# Minimal runnable shape test
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(0)

    # Example binary tree. In your project, replace this with loaded tree_structure.json.
    tree_json = {
        "node_id": "root",
        "left": {
            "node_id": "root_L",
            "left": {"node_id": "root_L_L", "leaf": True},
            "right": {"node_id": "root_L_R", "leaf": True},
        },
        "right": {
            "node_id": "root_R",
            "left": {"node_id": "root_R_L", "leaf": True},
            "right": {"node_id": "root_R_R", "leaf": True},
        },
    }

    builder = TreeStructureBuilder(max_distance=4, max_lca_depth=4, mask_mode="local_sibling")
    spec = builder.from_nested_dict(tree_json)
    relations = builder.build(spec)

    N = len(spec.node_ids)
    S = 5              # max number of THP sequence embeddings stored/padded per node
    D_seq = 64         # THP sequence embedding dimension
    D_sample = 64      # current sample THP embedding dimension
    D_stats = 10       # Hawkes/statistical feature dimension
    D_struct = 6       # from make_basic_node_struct_features
    D_model = 128
    B = 3

    node_seq_emb = torch.randn(N, S, D_seq)
    node_seq_mask = torch.ones(N, S, dtype=torch.bool)
    node_stats = torch.randn(N, D_stats)
    node_struct = make_basic_node_struct_features(spec)
    sample_z = torch.randn(B, D_sample)

    model = MemoryHawkesAttentionEncoder(
        d_seq=D_seq,
        d_sample=D_sample,
        d_stats=D_stats,
        d_struct=D_struct,
        d_model=D_model,
        num_heads=4,
        num_tree_layers=2,
        max_distance=4,
        max_lca_depth=4,
        dropout=0.1,
        use_cross_relation_bias=False,
    )

    out = model(
        node_seq_emb=node_seq_emb,
        node_seq_mask=node_seq_mask,
        node_stats=node_stats,
        node_struct_features=node_struct,
        relations=relations,
        sample_z=sample_z,
    )

    print("node_ids:", spec.node_ids)
    print("node_x:", tuple(out.node_x.shape))
    print("H_tree:", tuple(out.h_tree.shape))
    print("D_tree(x):", tuple(out.deterministic_tree.shape))
    print("node_relevance:", tuple(out.node_relevance.shape))
    print("rel_bias:", tuple(out.rel_bias.shape))
