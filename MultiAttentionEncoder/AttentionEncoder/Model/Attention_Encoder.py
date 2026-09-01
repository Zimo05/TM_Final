import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
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


class RelationBias(nn.Module):
    def __init__(
        self,
        num_heads: int = 4,
        max_distance: int = 2,
        max_lca_depth: int = 3,
    ):
        super().__init__()
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
        bias = (
            self.dist_emb(dist)
            + self.anc_emb(anc)
            + self.sib_emb(sibling)
            + self.lca_emb(lca_depth)
            + self.branch_emb(branch_rel)
        )  # [N, N, H]
        return bias.permute(2, 0, 1).contiguous()  # [H, N, N]


class MultiHeadAttentionWithBias(nn.Module):
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
        # [B, L, D] -> [B, H, L, Hd]
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        query: torch.Tensor,                   # [B, Lq, D]
        key: torch.Tensor,                     # [B, Lk, D]
        value: torch.Tensor,                   # [B, Lk, D]
        attn_mask: Optional[torch.BoolTensor] = None,   # [Lq, Lk] or [B, Lq, Lk]
        rel_bias: Optional[torch.Tensor] = None,        # [H, Lq, Lk] or [B, H, Lq, Lk]
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
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
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
        x: torch.Tensor,               # [N, D]
        adj_mask: torch.BoolTensor,    # [N, N]
        rel_bias: torch.Tensor,        # [H, N, N]
    ) -> torch.Tensor:
        y, _ = self.attn(
            query=self.ln1(x).unsqueeze(0),
            key=self.ln1(x).unsqueeze(0),
            value=self.ln1(x).unsqueeze(0),
            attn_mask=adj_mask,
            rel_bias=rel_bias,
        )
        x = x + y.squeeze(0)
        x = x + self.ff(self.ln2(x))
        return x


@dataclass
class CrossAttentionOutput:
    deterministic_tree: torch.Tensor  # [B, N, D]
    retrieved_tree: torch.Tensor      # [B, N, D]
    attn_weights: torch.Tensor        # [B, H, N, N]
    route_prob: torch.Tensor          # [B, N]


class TreeNodeCrossAttention(nn.Module):
    """
    Convert a static encoded tree H_tree = {h_i} into sample-conditioned deterministic
    features D_tree(x) = {d_i(x)}.

    Core idea:
      1) one sequence/sample embedding z_x queries the whole tree,
      2) query for each node i is conditioned on both h_i and z_x,
      3) cross-attention retrieves relevant tree context,
      4) fusion produces per-node deterministic features for TreeVAE.

    This follows the design:
        q_i(x) = MLP([h_i ; z_x ; h_i * z_x ; |h_i - z_x|])
        c_i(x) = CrossAttention(q_i(x), H_tree, H_tree)
        d_i(x) = MLP([m_i ; z_x ; m_i * z_x ; |m_i - z_x|])
    where m_i is a gated mix of local node state and retrieved context.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        query_hidden: Optional[int] = None,
        fusion_hidden: Optional[int] = None,
    ):
        super().__init__()
        query_hidden = query_hidden or (2 * d_model)
        fusion_hidden = fusion_hidden or (2 * d_model)

        self.tree_ln = nn.LayerNorm(d_model)
        self.sample_ln = nn.LayerNorm(d_model)
        self.cross_ln = nn.LayerNorm(d_model)

        self.query_mlp = MLP(
            in_dim=4 * d_model,
            hidden_dim=query_hidden,
            out_dim=d_model,
            dropout=dropout,
        )

        self.attn = MultiHeadAttentionWithBias(d_model, num_heads, dropout=dropout)

        # gate decides how much retrieved context to mix into each node state
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # final deterministic feature for TreeVAE
        self.det_mlp = MLP(
            in_dim=4 * d_model,
            hidden_dim=fusion_hidden,
            out_dim=d_model,
            dropout=dropout,
        )

    @staticmethod
    def _normalize_sample(sample_z: torch.Tensor) -> torch.Tensor:
        # Accept [D], [B, D], or [B, 1, D]
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
        # Accept [N, D] or [B, N, D]
        if tree_h.ndim == 2:
            tree_h = tree_h.unsqueeze(0).expand(batch_size, -1, -1)
        elif tree_h.ndim == 3:
            if tree_h.size(0) != batch_size:
                raise ValueError(
                    f"tree_h batch size {tree_h.size(0)} does not match sample batch size {batch_size}"
                )
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

        # Every query node can attend only to valid keys.
        return node_valid_mask.unsqueeze(1).expand(-1, num_queries, -1)

    def forward(
        self,
        tree_h: torch.Tensor,                           # [N, D] or [B, N, D]
        sample_z: torch.Tensor,                         # [D], [B, D], or [B, 1, D]
        node_valid_mask: Optional[torch.BoolTensor] = None,  # [N] or [B, N], True = valid
        rel_bias: Optional[torch.Tensor] = None,             # optional [H, N, N] or [B, H, N, N]
    ) -> CrossAttentionOutput:
        sample_z = self._normalize_sample(sample_z)
        batch_size = sample_z.size(0)
        tree_h = self._expand_tree(tree_h, batch_size)

        b, n, d = tree_h.shape
        sample_expand = sample_z.unsqueeze(1).expand(b, n, d)

        # Node-conditioned sample queries.
        h_norm = self.tree_ln(tree_h)
        z_norm = self.sample_ln(sample_expand)
        query_input = torch.cat(
            [h_norm, z_norm, h_norm * z_norm, torch.abs(h_norm - z_norm)],
            dim=-1,
        )
        query = self.query_mlp(query_input)   # [B, N, D]

        # Cross-attention: sample-conditioned node queries -> static tree memory.
        attn_mask = self._build_attn_mask(node_valid_mask, batch_size=b, num_queries=n)
        retrieved, attn = self.attn(
            query=query,
            key=h_norm,
            value=h_norm,
            attn_mask=attn_mask,
            rel_bias=rel_bias,
        )

        # Gated mix between local node state and retrieved global tree context.
        gate = self.gate(torch.cat([tree_h, retrieved, sample_expand], dim=-1))
        mixed = (1.0 - gate) * tree_h + gate * retrieved
        mixed = self.cross_ln(mixed + tree_h)

        # Deterministic feature tree for TreeVAE.
        det_input = torch.cat(
            [mixed, sample_expand, mixed * sample_expand, torch.abs(mixed - sample_expand)],
            dim=-1,
        )
        deterministic_tree = self.det_mlp(det_input)   # [B, N, D]

        # Route score = mean attention mass received by each node.
        route_prob = attn.mean(dim=1).mean(dim=1)  # [B, N]

        if node_valid_mask is not None:
            if node_valid_mask.ndim == 1:
                node_valid_mask = node_valid_mask.unsqueeze(0).expand(b, -1)
            node_valid_mask_f = node_valid_mask.unsqueeze(-1).to(deterministic_tree.dtype)
            deterministic_tree = deterministic_tree * node_valid_mask_f
            retrieved = retrieved * node_valid_mask_f
            route_prob = route_prob * node_valid_mask.to(route_prob.dtype)
            route_prob = route_prob / route_prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        return CrossAttentionOutput(
            deterministic_tree=deterministic_tree,
            retrieved_tree=retrieved,
            attn_weights=attn,
            route_prob=route_prob,
        )


# Example usage:
# tree_encoder_output = H_tree                      # [N, D]
# sequence_embedding = z_x                         # [B, D]
# cross = TreeNodeCrossAttention(d_model=D, num_heads=4)
# out = cross(tree_h=tree_encoder_output, sample_z=sequence_embedding)
# D_tree_x = out.deterministic_tree                # [B, N, D]
# route = out.route_prob                           # [B, N]
