"""
Tree_Features.py — Structural (s_i) and Hawkes/split (g_i) feature extraction
=============================================================================

For each tree node we build two raw feature blocks that, together with the
pooled sequence embedding ``u_i``, form the node input  ``h_i^0 = [u_i ; s_i ; g_i]``.

s_i  — structural features (derived purely from node-position + topology):
    depth (normalised), is_leaf, is_left_child, is_right_child,
    subtree_size (normalised), num_sequences (normalised)
    + a learnable parent-id embedding (handled in NodeInputFusion via parent_idx).

g_i  — Hawkes / split statistics:
    per-node aggregated Hawkes parameters (mu over event types,
    A summary stats, decay) taken from ``sequence_summary.csv``
    (leaf nodes use their own params, internal nodes average their
    descendant leaves), plus split statistics
    (left_frac, right_frac, balance, num_children).

The class is data-driven and degrades gracefully: if no Hawkes params are
available the Hawkes block is filled with zeros (split stats still computed).
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class TreeFeatureExtractor:
    """Extract structural and Hawkes/split features per tree node.

    Parameters
    ----------
    node_ids : list[str]
        Ordered node-position strings (e.g. ``"root"``, ``"l"``, ``"r_l_r"``).
        The output tensors follow this exact ordering.
    node_sequences : dict[str, list[int]]
        Mapping ``node_position -> list of global sequence IDs``.
    summary_csv : str or Path, optional
        Path to ``sequence_summary.csv`` providing per-leaf Hawkes params
        (columns: ``leaf_position, cluster_id, mu, A, decay, sequences``).
    """

    _DIR_MAP = {"l": 0, "r": 1, "L": 0, "R": 1}

    def __init__(
        self,
        node_ids: List[str],
        node_sequences: Dict[str, List[int]],
        summary_csv: Optional[str] = None,
    ):
        self.node_ids = list(node_ids)
        self.node_sequences = node_sequences
        self.n = len(self.node_ids)
        self.id_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.paths: Dict[str, Tuple[int, ...]] = {
            nid: self._parse_path(nid) for nid in self.node_ids
        }

        # Hawkes params per leaf node-position
        self.leaf_params: Dict[str, Dict[str, object]] = {}
        self.num_types: int = 0
        if summary_csv is not None and Path(summary_csv).exists():
            self._load_leaf_params(Path(summary_csv))

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    @classmethod
    def _parse_path(cls, node_id: str) -> Tuple[int, ...]:
        if node_id == "root":
            return ()
        parts = node_id.split("_")
        return tuple(cls._DIR_MAP.get(p, 0) for p in parts)

    def _is_leaf(self, nid: str) -> bool:
        """A node is a leaf if no other node has it as a strict 1-step prefix."""
        p = self.paths[nid]
        plen = len(p)
        for other in self.node_ids:
            po = self.paths[other]
            if len(po) == plen + 1 and po[:plen] == p:
                return False
        return True

    def _child(self, nid: str, direction: int) -> Optional[str]:
        p = self.paths[nid] + (direction,)
        for other in self.node_ids:
            if self.paths[other] == p:
                return other
        return None

    def _parent(self, nid: str) -> Optional[str]:
        p = self.paths[nid]
        if len(p) == 0:
            return None
        pp = p[:-1]
        for other in self.node_ids:
            if self.paths[other] == pp:
                return other
        return None

    def _subtree_node_count(self, nid: str) -> int:
        p = self.paths[nid]
        plen = len(p)
        return sum(
            1
            for o in self.node_ids
            if len(self.paths[o]) >= plen and self.paths[o][:plen] == p
        )

    def _descendant_leaves(self, nid: str) -> List[str]:
        p = self.paths[nid]
        plen = len(p)
        leaves = []
        for o in self.node_ids:
            po = self.paths[o]
            if len(po) >= plen and po[:plen] == p and self._is_leaf(o):
                leaves.append(o)
        return leaves

    # ------------------------------------------------------------------
    # Hawkes params loading
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_array(raw: str):
        try:
            return ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            return None

    def _load_leaf_params(self, summary_csv: Path) -> None:
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pos = row.get("leaf_position", "").strip()
                if not pos:
                    continue
                mu = self._parse_array(row.get("mu", "[]"))
                A = self._parse_array(row.get("A", "[]"))
                try:
                    decay = float(row.get("decay", 0.0))
                except (TypeError, ValueError):
                    decay = 0.0
                if mu is None:
                    continue
                self.leaf_params[pos] = {"mu": mu, "A": A, "decay": decay}
                self.num_types = max(self.num_types, len(mu))

    # ------------------------------------------------------------------
    # Per-node Hawkes aggregation
    # ------------------------------------------------------------------
    def _node_hawkes(self, nid: str) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Return (mu_mean[num_types], A_mean[num_types,num_types], decay_mean).

        Leaf nodes use their own params; internal nodes average descendant leaves.
        Returns zero tensors when no params are available.
        """
        nt = max(self.num_types, 1)
        leaves = self._descendant_leaves(nid)
        mus, As, decays = [], [], []
        for lf in leaves:
            pr = self.leaf_params.get(lf)
            if pr is None:
                continue
            mu = pr["mu"]
            if mu is not None and len(mu) == nt:
                mus.append(torch.tensor(mu, dtype=torch.float32))
            A = pr["A"]
            if A is not None and len(A) == nt and all(len(r) == nt for r in A):
                As.append(torch.tensor(A, dtype=torch.float32))
            decays.append(float(pr["decay"]))

        mu_mean = torch.stack(mus).mean(0) if mus else torch.zeros(nt)
        A_mean = torch.stack(As).mean(0) if As else torch.zeros(nt, nt)
        decay_mean = float(sum(decays) / len(decays)) if decays else 0.0
        return mu_mean, A_mean, decay_mean

    @staticmethod
    def _a_summary(A: torch.Tensor) -> torch.Tensor:
        """Summarise an excitation matrix into 4 scalars."""
        if A.numel() == 0:
            return torch.zeros(4)
        n = A.size(0)
        diag = torch.diagonal(A)
        diag_mean = diag.mean()
        total = A.sum()
        offdiag_mean = (total - diag.sum()) / max(n * n - n, 1)
        frob = torch.linalg.norm(A)
        return torch.stack([A.mean(), diag_mean, offdiag_mean, frob])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def struct_dim(self) -> int:
        # depth, is_leaf, is_left, is_right, subtree_size, num_seq
        return 6

    @property
    def hawkes_dim(self) -> int:
        # mu(num_types) + A_summary(4) + decay(1) + split(4)
        return max(self.num_types, 1) + 4 + 1 + 4

    def build(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Build feature tensors following ``self.node_ids`` ordering.

        Returns
        -------
        dict with:
            ``struct``     : FloatTensor [N, struct_dim]
            ``hawkes``     : FloatTensor [N, hawkes_dim]
            ``parent_idx`` : LongTensor  [N]   (== N when no parent / root)
        """
        depths = [len(self.paths[nid]) for nid in self.node_ids]
        max_depth = max(depths) if depths else 1
        seq_sizes = [len(self.node_sequences.get(nid, [])) for nid in self.node_ids]
        max_seq = max(seq_sizes) if seq_sizes else 1

        struct_rows: List[torch.Tensor] = []
        hawkes_rows: List[torch.Tensor] = []
        parent_idx: List[int] = []

        for nid in self.node_ids:
            p = self.paths[nid]
            depth = len(p)
            is_leaf = 1.0 if self._is_leaf(nid) else 0.0
            is_left = 1.0 if (depth > 0 and p[-1] == 0) else 0.0
            is_right = 1.0 if (depth > 0 and p[-1] == 1) else 0.0
            subtree_size = self._subtree_node_count(nid)
            num_seq = len(self.node_sequences.get(nid, []))

            struct_rows.append(
                torch.tensor(
                    [
                        depth / max(max_depth, 1),
                        is_leaf,
                        is_left,
                        is_right,
                        subtree_size / max(self.n, 1),
                        num_seq / max(max_seq, 1),
                    ],
                    dtype=torch.float32,
                )
            )

            # ---- Hawkes / split statistics ----
            mu_mean, A_mean, decay_mean = self._node_hawkes(nid)
            a_stats = self._a_summary(A_mean)

            own = max(num_seq, 1)
            left_child = self._child(nid, 0)
            right_child = self._child(nid, 1)
            left_size = len(self.node_sequences.get(left_child, [])) if left_child else 0
            right_size = len(self.node_sequences.get(right_child, [])) if right_child else 0
            left_frac = left_size / own
            right_frac = right_size / own
            balance = abs(left_size - right_size) / own
            n_children = float(bool(left_child)) + float(bool(right_child))
            split_stats = torch.tensor(
                [left_frac, right_frac, balance, n_children / 2.0],
                dtype=torch.float32,
            )

            hawkes_rows.append(
                torch.cat(
                    [
                        mu_mean,
                        a_stats,
                        torch.tensor([decay_mean], dtype=torch.float32),
                        split_stats,
                    ],
                    dim=0,
                )
            )

            # ---- parent index (for learnable embedding) ----
            par = self._parent(nid)
            parent_idx.append(self.id_to_idx[par] if par is not None else self.n)

        return {
            "struct": torch.stack(struct_rows, dim=0).to(device),
            "hawkes": torch.stack(hawkes_rows, dim=0).to(device),
            "parent_idx": torch.tensor(parent_idx, dtype=torch.long, device=device),
        }
