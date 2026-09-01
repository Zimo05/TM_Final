"""
AttenEncoderMain.py — Multi-Attention Encoder Main Pipeline
============================================================

Full pipeline for generating deterministic tree features D_tree(t):

  Phase 1 — THP Global Encoding
     Encode all event sequences using a pretrained THP checkpoint.
     → per-sequence embeddings  Z = {z_k ∈ R^d}

  Phase 2 — Node Input  h_i^0 = [u_i ; s_i ; g_i]  (NodeEmbedding + NodeInputFusion)
     u_i : attention-pool of the node's assigned THP sequence embeddings
     s_i : structural features (depth, is_leaf, is_left/right_child,
           subtree_size, num_sequences) + learnable parent-id embedding
     g_i : Hawkes / split statistics (aggregated mu / A / decay + split fractions)
     → fused initial node embeddings  H_tree^0 = {h_i^0 ∈ R^d}

  Phase 3 — Structural Attention (StructuralTreeBlock)
     Apply tree-topology-aware self-attention with relation biases
     (distance, ancestor, sibling, LCA depth, branch).  The distance / LCA
     embedding ranges adapt to the actual tree depth.
     → refined node embeddings  H_tree = {h_i ∈ R^d}

  Phase 4 — Cross-Attention (TreeNodeCrossAttention)
     Q : sequence embeddings  Z  (conditioned per-sample)
     K, V : structural tree embeddings  H_tree
     → deterministic features  D_tree(t) ∈ R^{S×N×d}
     → route probabilities  ∈ R^{S×N}

Inputs
------
  --thp_json       Path to THP JSON (e.g. THP_8.json)
  --tree_csv       Path to tree_node_sequences.csv
  --summary_csv    Path to sequence_summary.csv (Hawkes params for g_i)
  --checkpoint     Path to pretrained THP checkpoint (.pt)
  --weights        Path to trained attention-module weights (from Train.py)
  --output         Path for output D_tree.pt

Training
--------
  The attention modules (NodeEmbedding / NodeInputFusion / RelationBias /
  StructuralTreeBlock / TreeNodeCrossAttention) are randomly initialised.
  Train them with ``Train.py`` (frozen THP + leaf-routing supervision) and pass
  the resulting weights via ``--weights`` so the encoder is not random.

Usage
-----
  python AttenEncoderMain.py \
      --thp_json   /Volumes/shenzm/Shuang_RA/Data/tree_8/8Cluster/THP_8.json \
      --tree_csv   /Volumes/shenzm/Shuang_RA/Data/tree_8/tree_node_sequences.csv \
      --checkpoint /path/to/checkpoint_best.pt \
      --weights    /Volumes/shenzm/Shuang_RA/Data/tree_8/encoder_weights.pt \
      --output     /Volumes/shenzm/Shuang_RA/Data/tree_8/d_tree.pt
"""

from __future__ import annotations

import ast
import contextlib
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup – make both the project root (for the ``THP`` / ``AttentionEncoder``
# packages) and the THP directory (for THP's flat internal imports such as
# ``from TransformerModel import ...``) importable, regardless of CWD.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # .../MultiAttentionEncoder
_THP_DIR = _PROJECT_ROOT / "THP"
for _p in (str(_THP_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from THP.EncodeMain import THPEncoding
from SplitManifest import build_data_provenance, load_strict_manifest

from AttentionEncoder.Model.Node_Embedding import NodeEmbedding, NodeInputFusion
from AttentionEncoder.Model.Tree_Features import TreeFeatureExtractor
from AttentionEncoder.Model.Attention_Encoder import (
    CrossAttentionOutput,
    RelationBias,
    StructuralTreeBlock,
    TreeNodeCrossAttention,
)


# ============================================================================
# Lightweight node representation for NodeEmbedding
# ============================================================================
@dataclass
class TreeNodeVal:
    """Holds the per-node sequence embeddings that NodeEmbedding will pool."""

    node_id: str
    val: List[torch.Tensor]  # list of [d_model] tensors, one per assigned sequence


# ============================================================================
# Tree-path utilities — compute structural relations from node-position strings
# ============================================================================
class TreePathParser:
    """Parse node-position names like ``"root"``, ``"l"``, ``"l_r"``, ``"r_l_r"``
    into path tuples and derive pairwise structural relations.

    Node-position convention (binary tree):
      - ``"root"``  → path = ()
      - ``"l"``     → path = (0,)     # 0 = left
      - ``"r"``     → path = (1,)     # 1 = right
      - ``"l_r"``   → path = (0, 1)
      - ``"r_l_l"`` → path = (1, 0, 0)

    From these paths we compute five relation matrices used by RelationBias:
      dist[i,j]      – shortest-path distance between nodes i and j
      anc[i,j]       – ancestor relationship  (0=none, 1=i anc of j, 2=j anc of i, 3=same)
      sibling[i,j]   – are i and j siblings?  (0=no, 1=yes)
      lca_depth[i,j] – depth of lowest common ancestor
      branch[i,j]    – branching pattern encoding
    """

    # Direction encoding for left/right
    _DIR_MAP = {"l": 0, "r": 1, "L": 0, "R": 1}

    def __init__(self, node_ids: List[str]):
        self.node_ids = node_ids
        self.n = len(node_ids)
        self._paths: Dict[str, Tuple[int, ...]] = {
            nid: self._parse_path(nid) for nid in node_ids
        }
        self._id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        # Populated by build_relation_tensors() — used to size RelationBias.
        self.max_dist: int = 1
        self.max_lca: int = 1

    # ------------------------------------------------------------------
    @classmethod
    def _parse_path(cls, node_id: str) -> Tuple[int, ...]:
        """Convert a node-position string to a tuple of direction ints."""
        if node_id == "root":
            return ()
        parts = node_id.split("_")
        return tuple(cls._DIR_MAP.get(p, 0) for p in parts)

    # ------------------------------------------------------------------
    @staticmethod
    def _lca(path_a: Tuple[int, ...], path_b: Tuple[int, ...]) -> int:
        """Length of the longest common prefix (= depth of LCA)."""
        d = 0
        for da, db in zip(path_a, path_b):
            if da != db:
                break
            d += 1
        return d

    # ------------------------------------------------------------------
    def build_relation_tensors(
        self, device: torch.device
    ) -> Dict[str, torch.LongTensor]:
        """Build all five relation-index tensors of shape [N, N].

        Returns
        -------
        dict with keys ``"dist"``, ``"anc"``, ``"sibling"``, ``"lca_depth"``,
        ``"branch"`` — each a LongTensor of shape [N, N].
        """
        n = self.n
        paths = [self._paths[nid] for nid in self.node_ids]
        depths = [len(p) for p in paths]

        dist = torch.zeros(n, n, dtype=torch.long)
        anc = torch.zeros(n, n, dtype=torch.long)
        sibling = torch.zeros(n, n, dtype=torch.long)
        lca_depth = torch.zeros(n, n, dtype=torch.long)
        branch = torch.zeros(n, n, dtype=torch.long)

        for i in range(n):
            for j in range(n):
                if i == j:
                    anc[i, j] = 3  # same node
                    continue

                pi, pj = paths[i], paths[j]
                lca = self._lca(pi, pj)

                # Distance
                dist[i, j] = depths[i] + depths[j] - 2 * lca

                # LCA depth
                lca_depth[i, j] = lca

                # Ancestor
                if lca == len(pi) and depths[i] < depths[j]:
                    anc[i, j] = 1  # i is ancestor of j
                elif lca == len(pj) and depths[j] < depths[i]:
                    anc[i, j] = 2  # j is ancestor of i (i is descendant)

                # Sibling: same parent, different node
                if depths[i] == depths[j] and lca == depths[i] - 1 and depths[i] > 0:
                    sibling[i, j] = 1

                # Branch relation encoding:
                # 0 = unrelated at root, 1 = same left branch, 2 = same right branch,
                # 3 = cross-branch L, 4 = cross-branch R
                if lca == 0:
                    branch[i, j] = 0
                else:
                    first_diff_i = paths[i][lca] if depths[i] > lca else -1
                    first_diff_j = paths[j][lca] if depths[j] > lca else -1
                    if first_diff_i == -1 or first_diff_j == -1:
                        # One is the ancestor — encode by the descendant's branch
                        d = max(depths[i], depths[j])
                        deeper = paths[i] if depths[i] > depths[j] else paths[j]
                        first_dir = deeper[lca] if len(deeper) > lca else 0
                        branch[i, j] = 1 + first_dir  # 1=left, 2=right
                    elif first_diff_i == first_diff_j:
                        branch[i, j] = 1 + first_diff_i  # same branch
                    else:
                        branch[i, j] = 3 + first_diff_i  # cross-branch

        # Clamp to valid embedding ranges. ``dist`` and ``lca_depth`` adapt to
        # the actual tree depth instead of fixed maxima (which would collapse
        # distinct structural relations into the same bucket on deep trees).
        self.max_dist = int(dist.max().item()) if n > 1 else 1
        self.max_lca = int(lca_depth.max().item()) if n > 1 else 1
        self.max_dist = max(self.max_dist, 1)
        self.max_lca = max(self.max_lca, 1)

        dist = dist.clamp(0, self.max_dist)
        anc = anc.clamp(0, 3)
        sibling = sibling.clamp(0, 1)
        lca_depth = lca_depth.clamp(0, self.max_lca)
        branch = branch.clamp(0, 4)

        return {
            "dist": dist.to(device),
            "anc": anc.to(device),
            "sibling": sibling.to(device),
            "lca_depth": lca_depth.to(device),
            "branch": branch.to(device),
        }


# ============================================================================
# Main pipeline
# ============================================================================
class MultiAttentionEncoderPipeline:
    """Orchestrates the full attention-encoder pipeline.

    Parameters
    ----------
    thp_json_path : str
        Path to the THP JSON file (e.g. ``THP_8.json``).
    tree_csv_path : str
        Path to ``tree_node_sequences.csv``.
    checkpoint_path : str
        Path to a pretrained THP checkpoint (``.pt``).
    d_model : int
        Embedding dimension for THP and attention modules. Must match the
        checkpoint.
    num_heads : int
        Number of attention heads for StructuralTreeBlock and CrossAttention.
    d_rnn, d_inner_hid, d_k, d_v, n_layers, dropout, batch_size :
        Passed through to THPEncoding.
    device : str or None
        Torch device string (auto-detected if omitted).
    """

    def __init__(
        self,
        thp_json_path: str,
        tree_csv_path: str,
        checkpoint_path: str,
        summary_csv_path: Optional[str] = None,
        d_model: int = 64,
        num_heads: int = 4,
        d_rnn: int = 256,
        d_inner_hid: int = 128,
        d_k: int = 16,
        d_v: int = 16,
        n_layers: int = 4,
        dropout: float = 0.1,
        batch_size: int = 16,
        parent_emb_dim: int = 8,
        device: Optional[str] = None,
    ):
        self.thp_json_path = Path(thp_json_path)
        self.tree_csv_path = Path(tree_csv_path)
        self.checkpoint_path = checkpoint_path
        # Default: sequence_summary.csv next to the tree CSV.
        self.summary_csv_path = (
            Path(summary_csv_path)
            if summary_csv_path is not None
            else self.tree_csv_path.parent / "sequence_summary.csv"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_rnn = d_rnn
        self.d_inner_hid = d_inner_hid
        self.d_k = d_k
        self.d_v = d_v
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.parent_emb_dim = parent_emb_dim

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # When True (default, inference), Phase 2-4 run under torch.no_grad().
        # The trainer sets this to False so gradients can flow.
        self._no_grad = True

        # ------------------------------------------------------------------
        # Internal state — populated by run()
        # ------------------------------------------------------------------
        # Raw data
        self.all_sequences: Dict[str, Any] = {}  # json_key → list of event dicts
        self.node_sequences: Dict[str, List[int]] = {}  # node_position → global seq IDs
        self.global_id_to_key: Dict[int, str] = {}
        self.key_to_global_id: Dict[str, int] = {}

        # Encoded
        self.seq_embeddings: Dict[str, torch.Tensor] = {}  # json_key → [d_model]
        self.node_ids: List[str] = []  # ordered list of node positions
        self.H_tree: Optional[torch.Tensor] = None  # [N, d_model]
        self.H_tree_refined: Optional[torch.Tensor] = None  # [N, d_model] after StructuralTreeBlock
        self.relation_tensors: Dict[str, torch.LongTensor] = {}
        self.node_features: Dict[str, torch.Tensor] = {}  # struct / hawkes / parent_idx

        # Output
        self.D_tree: Optional[torch.Tensor] = None  # [S, N, d_model]
        self.route_prob: Optional[torch.Tensor] = None  # [S, N]
        self.Z_matrix: Optional[torch.Tensor] = None  # [S, d_model] stacked seq embeddings
        self.weights_metadata: Dict[str, Any] = {}
        self.data_provenance: Optional[Dict[str, Any]] = None

        # ------------------------------------------------------------------
        # Sub-modules (built after data is loaded so we know dimensions)
        # ------------------------------------------------------------------
        self.node_embedder: Optional[NodeEmbedding] = None
        self.node_fusion: Optional[NodeInputFusion] = None
        self.feature_extractor: Optional[TreeFeatureExtractor] = None
        self.relation_bias: Optional[RelationBias] = None
        self.structural_block: Optional[StructuralTreeBlock] = None
        self.cross_attention: Optional[TreeNodeCrossAttention] = None

    # ------------------------------------------------------------------
    def _grad_ctx(self):
        """Return torch.no_grad() for inference, or a no-op for training."""
        return torch.no_grad() if self._no_grad else contextlib.nullcontext()

    # ======================================================================
    # Data Loading
    # ======================================================================

    def load_node_sequences(self) -> Dict[str, List[int]]:
        """Read ``tree_node_sequences.csv`` → node_position → list of global seq IDs."""
        with open(self.tree_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_pos = row["node_position"].strip()
                seqs = self._parse_sequence_list(row["sequences"])
                self.node_sequences[node_pos] = seqs

        print(f"[Data] Loaded {len(self.node_sequences)} nodes from {self.tree_csv_path}")
        return self.node_sequences

    @staticmethod
    def _parse_sequence_list(raw: str) -> List[int]:
        """Convert a CSV string like ``"[3, 9, 22]"`` into a list of ints."""
        if isinstance(raw, list):
            return [int(x) for x in raw]
        try:
            parsed = ast.literal_eval(str(raw))
            return [int(x) for x in parsed]
        except (ValueError, SyntaxError):
            return []

    def load_all_sequences(self) -> Dict[str, Any]:
        """Read the full THP JSON into ``self.all_sequences``."""
        with open(self.thp_json_path, "r", encoding="utf-8") as f:
            self.all_sequences = json.load(f)
        print(f"[Data] Loaded {len(self.all_sequences)} sequences from {self.thp_json_path}")
        return self.all_sequences

    def build_global_id_mapping(self) -> Dict[int, str]:
        """Create bidirectional mapping between global IDs (0..N-1) and JSON keys.

        Keys are sorted by ``(cluster_id, seq_index)``.
        """
        if not self.all_sequences:
            raise RuntimeError("Call load_all_sequences() first.")

        # sorted_keys = sorted(
        #     self.all_sequences.keys(),
        #     key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])),
        # )
        def _sort_key(k):
            parts=k.split("_")
            if len(parts)>=2:
                return (int(parts[0]), int(parts[1]))
            return (int(parts[0]), 0)
        sorted_keys = sorted(self.all_sequences.keys(), key=_sort_key)
        self.global_id_to_key = {i: k for i, k in enumerate(sorted_keys)}
        self.key_to_global_id = {k: i for i, k in enumerate(sorted_keys)}
        print(f"[Data] Built global-ID mapping: 0 … {len(sorted_keys) - 1}")
        return self.global_id_to_key

    # ======================================================================
    # Phase 1 — THP Global Encoding
    # ======================================================================

    def encode_all_sequences_thp(self) -> Dict[str, torch.Tensor]:
        """Encode ALL sequences through the THP encoder with the pretrained checkpoint.

        Returns
        -------
        dict[str, Tensor]
            Mapping ``json_key → embedding_tensor`` of shape ``[d_model]``.
        """
        print("\n" + "=" * 60)
        print("Phase 1 — THP Global Encoding")
        print("=" * 60)

        encoder = THPEncoding(
            data=str(self.thp_json_path),
            batch_size=self.batch_size,
            d_model=self.d_model,
            d_rnn=self.d_rnn,
            d_inner_hid=self.d_inner_hid,
            d_k=self.d_k,
            d_v=self.d_v,
            n_head=self.num_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            checkpoint=self.checkpoint_path,
            device=str(self.device),
        )
        self.seq_embeddings = encoder.run_encoding()
        # New checkpoints carry their architecture. Keep downstream attention
        # modules aligned with the actual THP embedding dimension.
        self.d_model = encoder.d_model
        print(f"[Phase 1] Encoded {len(self.seq_embeddings)} sequences "
              f"(dim={self.d_model})")
        return self.seq_embeddings

    # ======================================================================
    # Phase 2 — Per-Node Pooling via NodeEmbedding
    # ======================================================================

    def _build_node_embedder(self) -> NodeEmbedding:
        """Instantiate NodeEmbedding (attention-pooling for u_i)."""
        model = NodeEmbedding(d_model=self.d_model).to(self.device)
        return model

    def encode_nodes(self) -> torch.Tensor:
        """Build initial node embeddings  ``h_i^0 = [u_i ; s_i ; g_i]``.

        Steps
        -----
        1. ``u_i`` — attention-pool each node's assigned THP sequence embeddings.
        2. ``s_i`` — structural features (depth, is_leaf, is_left/right_child,
           subtree_size, num_sequences) + learnable parent-id embedding.
        3. ``g_i`` — Hawkes / split statistics (aggregated mu / A / decay +
           left/right split fractions).
        4. Fuse ``[u_i ; s_i ; g_i]`` → ``d_model`` via NodeInputFusion.

        Returns
        -------
        H_tree : [N, d_model]
            Initial node embeddings (before structural refinement).
        """
        print("\n" + "=" * 60)
        print("Phase 2 — Node Input  h_i^0 = [u_i ; s_i ; g_i]")
        print("=" * 60)

        if not self.seq_embeddings:
            raise RuntimeError("Run Phase 1 first (encode_all_sequences_thp).")

        self.node_ids = sorted(
            self.node_sequences.keys(),
            key=lambda n: (len(TreeFeatureExtractor._parse_path(n)), n),
        )
        N = len(self.node_ids)

        # ---- Build the structural / Hawkes feature extractor ----
        self.feature_extractor = TreeFeatureExtractor(
            node_ids=self.node_ids,
            node_sequences=self.node_sequences,
            summary_csv=str(self.summary_csv_path),
        )
        self.node_features = self.feature_extractor.build(self.device)
        struct_dim = self.feature_extractor.struct_dim
        hawkes_dim = self.feature_extractor.hawkes_dim
        print(f"  s_i (structural) dim = {struct_dim}, "
              f"g_i (Hawkes/split) dim = {hawkes_dim} "
              f"(num_types={self.feature_extractor.num_types})")

        # ---- Build modules (pooling + fusion) ----
        if self.node_embedder is None:
            self.node_embedder = self._build_node_embedder()
        if self.node_fusion is None:
            self.node_fusion = NodeInputFusion(
                d_model=self.d_model,
                struct_dim=struct_dim,
                hawkes_dim=hawkes_dim,
                num_nodes=N,
                parent_emb_dim=self.parent_emb_dim,
                dropout=self.dropout,
            ).to(self.device)
        self._set_module_mode()

        # ---- u_i : attention-pool per node ----
        with self._grad_ctx():
            u_list: List[torch.Tensor] = []
            for node_pos in self.node_ids:
                seq_ids = self.node_sequences[node_pos]
                val_tensors: List[torch.Tensor] = []
                for gid in seq_ids:
                    json_key = self.global_id_to_key.get(gid)
                    if json_key is not None and json_key in self.seq_embeddings:
                        emb = self.seq_embeddings[json_key]
                        if emb.device != self.device:
                            emb = emb.to(self.device)
                        val_tensors.append(emb)

                if not val_tensors:
                    print(f"  [Warn] Node '{node_pos}': no valid sequence embeddings.")
                    u_list.append(torch.zeros(self.d_model, device=self.device))
                    continue

                node_val = TreeNodeVal(node_id=node_pos, val=val_tensors)
                u_list.append(self.node_embedder(node_val))  # [d_model]

            U = torch.stack(u_list, dim=0)  # [N, d_model]

            # ---- Fuse u_i, s_i (+ parent emb), g_i ----
            self.H_tree = self.node_fusion(
                u=U,
                struct_feat=self.node_features["struct"],
                parent_idx=self.node_features["parent_idx"],
                hawkes_feat=self.node_features["hawkes"],
            )  # [N, d_model]

        print(f"\n[Phase 2] H_tree shape: {list(self.H_tree.shape)} "
              f"({N} nodes, d_model={self.d_model})")
        return self.H_tree

    def _set_module_mode(self) -> None:
        """Put all trainable modules in train() or eval() based on ``_no_grad``."""
        training = not self._no_grad
        for m in (
            self.node_embedder,
            self.node_fusion,
            self.relation_bias,
            self.structural_block,
            self.cross_attention,
        ):
            if m is not None:
                m.train(training)

    # ======================================================================
    # Phase 3 — Structural Attention
    # ======================================================================

    def _build_structural_modules(self, max_distance: int, max_lca_depth: int) -> None:
        """Instantiate RelationBias (sized to the actual tree) and StructuralTreeBlock."""
        self.relation_bias = RelationBias(
            num_heads=self.num_heads,
            max_distance=max_distance,
            max_lca_depth=max_lca_depth,
        ).to(self.device)

        self.structural_block = StructuralTreeBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(self.device)

    def apply_structural_attention(self) -> torch.Tensor:
        """Refine node embeddings using tree-topology-aware self-attention.

        Builds relation biases from node-position paths, then applies
        StructuralTreeBlock.

        Returns
        -------
        H_tree_refined : [N, d_model]
        """
        print("\n" + "=" * 60)
        print("Phase 3 — Structural Attention (StructuralTreeBlock)")
        print("=" * 60)

        if self.H_tree is None:
            raise RuntimeError("Run Phase 2 first (encode_nodes).")

        # Build relation tensors from tree-path parsing (depth-adaptive ranges).
        path_parser = TreePathParser(self.node_ids)
        self.relation_tensors = path_parser.build_relation_tensors(self.device)
        print(f"  Relation ranges: max_distance={path_parser.max_dist}, "
              f"max_lca_depth={path_parser.max_lca}")

        if self.relation_bias is None or self.structural_block is None:
            self._build_structural_modules(
                max_distance=path_parser.max_dist,
                max_lca_depth=path_parser.max_lca,
            )
        self._set_module_mode()

        # Compute relation bias: [H, N, N]
        rel_bias = self.relation_bias(
            dist=self.relation_tensors["dist"],
            anc=self.relation_tensors["anc"],
            sibling=self.relation_tensors["sibling"],
            lca_depth=self.relation_tensors["lca_depth"],
            branch_rel=self.relation_tensors["branch"],
        )  # [H, N, N]

        # Build adjacency mask: all nodes can attend to all others (tree is fully connected)
        n = len(self.node_ids)
        adj_mask = torch.ones(n, n, dtype=torch.bool, device=self.device)

        # Apply StructuralTreeBlock
        with self._grad_ctx():
            self.H_tree_refined = self.structural_block(
                x=self.H_tree,
                adj_mask=adj_mask,
                rel_bias=rel_bias,
            )  # [N, d_model]

        print(f"[Phase 3] H_tree_refined shape: {list(self.H_tree_refined.shape)}")
        return self.H_tree_refined

    # ======================================================================
    # Phase 4 — Cross-Attention (Sequence × Tree)
    # ======================================================================

    def _build_cross_attention(self) -> None:
        """Instantiate TreeNodeCrossAttention."""
        self.cross_attention = TreeNodeCrossAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(self.device)
        self._set_module_mode()

    def apply_cross_attention(self) -> CrossAttentionOutput:
        """Cross-attention: sequence embeddings Q × tree embeddings K,V → D_tree.

        For each sequence z_k:
          q_i(x_k) = MLP([h_i ; z_k ; h_i * z_k ; |h_i - z_k|])
          c_i(x_k) = CrossAttention(q_i(x_k), H_tree, H_tree)
          d_i(x_k) = MLP([m_i ; z_k ; m_i * z_k ; |m_i - z_k|])

        Returns
        -------
        CrossAttentionOutput with deterministic_tree, retrieved_tree, attn_weights, route_prob.
        """
        print("\n" + "=" * 60)
        print("Phase 4 — Cross-Attention (TreeNodeCrossAttention)")
        print("=" * 60)

        if self.H_tree_refined is None:
            raise RuntimeError("Run Phase 3 first (apply_structural_attention).")
        if self.cross_attention is None:
            self._build_cross_attention()

        # Build sequence embedding matrix Z: [S, d_model]
        # Order sequences consistently by global ID
        sorted_keys = sorted(self.seq_embeddings.keys(),
                             key=lambda k: self.key_to_global_id.get(k, 0))
        Z_list = []
        for k in sorted_keys:
            emb = self.seq_embeddings[k]
            if emb.device != self.device:
                emb = emb.to(self.device)
            Z_list.append(emb)
        self.Z_matrix = torch.stack(Z_list, dim=0)  # [S, d_model]
        S = self.Z_matrix.shape[0]

        # Build node valid mask (all nodes are valid)
        N = len(self.node_ids)
        node_valid_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        # Process sequences in batches to avoid OOM
        all_deterministic: List[torch.Tensor] = []
        all_retrieved: List[torch.Tensor] = []
        all_attn: List[torch.Tensor] = []
        all_route: List[torch.Tensor] = []

        with torch.no_grad():
            for start in tqdm(range(0, S, self.batch_size), desc="  Cross-Attention"):
                end = min(start + self.batch_size, S)
                z_batch = self.Z_matrix[start:end]  # [B, d_model]

                out: CrossAttentionOutput = self.cross_attention(
                    tree_h=self.H_tree_refined,  # [N, d_model] — expanded to [B, N, d] internally
                    sample_z=z_batch,  # [B, d_model]
                    node_valid_mask=node_valid_mask,
                    rel_bias=None,  # Optional: could reuse structural bias
                )

                all_deterministic.append(out.deterministic_tree.cpu())  # [B, N, d]
                all_retrieved.append(out.retrieved_tree.cpu())
                all_attn.append(out.attn_weights.cpu())
                all_route.append(out.route_prob.cpu())

        # Concatenate along batch dim
        D_tree = torch.cat(all_deterministic, dim=0)  # [S, N, d_model]
        R_tree = torch.cat(all_retrieved, dim=0)
        A_weights = torch.cat(all_attn, dim=0)
        P_route = torch.cat(all_route, dim=0)

        print(f"\n[Phase 4] D_tree shape: {list(D_tree.shape)}")
        print(f"  deterministic_tree : {list(D_tree.shape)}")
        print(f"  retrieved_tree     : {list(R_tree.shape)}")
        print(f"  attn_weights       : {list(A_weights.shape)}")
        print(f"  route_prob         : {list(P_route.shape)}")

        self.D_tree = D_tree
        self.route_prob = P_route

        return CrossAttentionOutput(
            deterministic_tree=D_tree,
            retrieved_tree=R_tree,
            attn_weights=A_weights,
            route_prob=P_route,
        )

    # ======================================================================
    # Differentiable helpers (shared by inference and the trainer)
    # ======================================================================

    def build_Z_matrix(self) -> torch.Tensor:
        """Stack per-sequence THP embeddings into ``Z`` ordered by global ID.

        Row ``j`` of ``Z`` corresponds to global sequence ID ``j``.
        """
        sorted_keys = sorted(
            self.seq_embeddings.keys(),
            key=lambda k: self.key_to_global_id.get(k, 0),
        )
        Z_list = []
        for k in sorted_keys:
            emb = self.seq_embeddings[k]
            if emb.device != self.device:
                emb = emb.to(self.device)
            Z_list.append(emb)
        self.Z_matrix = torch.stack(Z_list, dim=0)  # [S, d_model]
        return self.Z_matrix

    def _cached_rel_bias(self) -> torch.Tensor:
        """Compute (and cache) the structural relation bias  [H, N, N]."""
        return self.relation_bias(
            dist=self.relation_tensors["dist"],
            anc=self.relation_tensors["anc"],
            sibling=self.relation_tensors["sibling"],
            lca_depth=self.relation_tensors["lca_depth"],
            branch_rel=self.relation_tensors["branch"],
        )

    def forward_node_inputs(
        self,
        allowed_sequence_ids: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        """Quiet, differentiable  ``h_i^0 = [u_i ; s_i ; g_i]``  computation.

        Requires modules + features to be built (call :meth:`setup_modules`).
        When ``allowed_sequence_ids`` is provided, the semantic pooling term
        ``u_i`` only sees those global sequence IDs.  Structural and Hawkes
        features are unchanged.
        """
        if allowed_sequence_ids is None:
            allowed_ids = None
        elif isinstance(allowed_sequence_ids, (set, frozenset)):
            allowed_ids = allowed_sequence_ids
        else:
            allowed_ids = frozenset(int(gid) for gid in allowed_sequence_ids)

        u_list: List[torch.Tensor] = []
        for node_pos in self.node_ids:
            seq_ids = self.node_sequences[node_pos]
            val_tensors: List[torch.Tensor] = []
            for gid in seq_ids:
                if allowed_ids is not None and gid not in allowed_ids:
                    continue
                json_key = self.global_id_to_key.get(gid)
                if json_key is not None and json_key in self.seq_embeddings:
                    emb = self.seq_embeddings[json_key]
                    if emb.device != self.device:
                        emb = emb.to(self.device)
                    val_tensors.append(emb)
            if not val_tensors:
                u_list.append(torch.zeros(self.d_model, device=self.device))
            else:
                u_list.append(
                    self.node_embedder(TreeNodeVal(node_id=node_pos, val=val_tensors))
                )
        U = torch.stack(u_list, dim=0)
        return self.node_fusion(
            u=U,
            struct_feat=self.node_features["struct"],
            parent_idx=self.node_features["parent_idx"],
            hawkes_feat=self.node_features["hawkes"],
        )

    def forward_structural(self, H_tree: torch.Tensor) -> torch.Tensor:
        """Quiet, differentiable structural self-attention."""
        n = len(self.node_ids)
        adj_mask = torch.ones(n, n, dtype=torch.bool, device=self.device)
        return self.structural_block(
            x=H_tree, adj_mask=adj_mask, rel_bias=self._cached_rel_bias()
        )

    def forward_cross(
        self, H_refined: torch.Tensor, z_batch: torch.Tensor
    ) -> CrossAttentionOutput:
        """Quiet, differentiable cross-attention for a batch of sequences."""
        n = len(self.node_ids)
        node_valid_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        return self.cross_attention(
            tree_h=H_refined,
            sample_z=z_batch,
            node_valid_mask=node_valid_mask,
            rel_bias=None,
        )

    def setup_modules(self) -> None:
        """One-time setup: build node_ids, features, all modules, relation tensors.

        Assumes data is loaded and Phase 1 (THP encoding) has run.  After this
        call the ``forward_*`` helpers can be used repeatedly (e.g. by the trainer).
        """
        # Node ordering + features
        self.node_ids = sorted(
            self.node_sequences.keys(),
            key=lambda n: (len(TreeFeatureExtractor._parse_path(n)), n),
        )
        N = len(self.node_ids)
        self.feature_extractor = TreeFeatureExtractor(
            node_ids=self.node_ids,
            node_sequences=self.node_sequences,
            summary_csv=str(self.summary_csv_path),
        )
        self.node_features = self.feature_extractor.build(self.device)

        # Pooling + fusion
        if self.node_embedder is None:
            self.node_embedder = self._build_node_embedder()
        if self.node_fusion is None:
            self.node_fusion = NodeInputFusion(
                d_model=self.d_model,
                struct_dim=self.feature_extractor.struct_dim,
                hawkes_dim=self.feature_extractor.hawkes_dim,
                num_nodes=N,
                parent_emb_dim=self.parent_emb_dim,
                dropout=self.dropout,
            ).to(self.device)

        # Relation tensors + structural / cross modules
        path_parser = TreePathParser(self.node_ids)
        self.relation_tensors = path_parser.build_relation_tensors(self.device)
        if self.relation_bias is None or self.structural_block is None:
            self._build_structural_modules(
                max_distance=path_parser.max_dist,
                max_lca_depth=path_parser.max_lca,
            )
        if self.cross_attention is None:
            self._build_cross_attention()

        self.build_Z_matrix()
        self._set_module_mode()

    def trainable_modules(self) -> List[nn.Module]:
        """All attention modules that should be trained (THP stays frozen)."""
        return [
            m
            for m in (
                self.node_embedder,
                self.node_fusion,
                self.relation_bias,
                self.structural_block,
                self.cross_attention,
            )
            if m is not None
        ]

    def trainable_parameters(self):
        for m in self.trainable_modules():
            yield from m.parameters()

    def save_module_weights(
        self,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save the trained attention-module weights (THP excluded)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "node_embedder": self.node_embedder.state_dict(),
                "node_fusion": self.node_fusion.state_dict(),
                "relation_bias": self.relation_bias.state_dict(),
                "structural_block": self.structural_block.state_dict(),
                "cross_attention": self.cross_attention.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )
        print(f"[Weights] Saved attention-module weights to {path}")

    def load_module_weights(self, path: str) -> Dict[str, Any]:
        """Load trained attention-module weights (must call setup_modules first)."""
        state = torch.load(path, map_location=self.device)
        self.node_embedder.load_state_dict(state["node_embedder"])
        self.node_fusion.load_state_dict(state["node_fusion"])
        self.relation_bias.load_state_dict(state["relation_bias"])
        self.structural_block.load_state_dict(state["structural_block"])
        self.cross_attention.load_state_dict(state["cross_attention"])
        self.weights_metadata = state.get("metadata", {})
        print(f"[Weights] Loaded attention-module weights from {path}")
        return self.weights_metadata

    # ======================================================================
    # Output
    # ======================================================================

    def save_output(self, output_path: str) -> None:
        """Save all pipeline outputs to a ``.pt`` file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "D_tree": self.D_tree,  # [S, N, d_model] — deterministic features
            "route_prob": self.route_prob,  # [S, N] — routing probabilities
            "H_tree": self.H_tree,  # [N, d_model] — initial node embeddings
            "H_tree_refined": self.H_tree_refined,  # [N, d_model] — refined
            "Z_matrix": self.Z_matrix,  # [S, d_model] — sequence embeddings
            "node_ids": self.node_ids,
            "sequence_keys": sorted(
                self.seq_embeddings.keys(),
                key=lambda k: self.key_to_global_id.get(k, 0),
            ),
            "global_id_to_key": self.global_id_to_key,
            "weights_metadata": self.weights_metadata,
            "data_provenance": self.data_provenance,
            "evaluation_regime": (
                self.data_provenance.get("evaluation_regime")
                if self.data_provenance else "transductive"
            ),
            "config": {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "num_nodes": len(self.node_ids),
                "num_sequences": self.Z_matrix.shape[0] if self.Z_matrix is not None else 0,
            },
        }
        torch.save(save_dict, output_path)
        print(f"\n[Output] Saved to {output_path}")
        print(f"  Keys: {list(save_dict.keys())}")

    def save_node_output(self, output_path: str) -> None:
        """Save node-only pipeline outputs (Phase 1-3, skipping cross-attention)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "H_tree_refined": self.H_tree_refined,
            "H_tree":         self.H_tree,
            "Z_matrix":       self.Z_matrix,
            "node_ids":       self.node_ids,
            "sequence_keys":  sorted(
                self.seq_embeddings.keys(),
                key=lambda k: self.key_to_global_id.get(k, 0),
            ),
            "global_id_to_key": self.global_id_to_key,
            "weights_metadata": self.weights_metadata,
            "data_provenance": self.data_provenance,
            "evaluation_regime": (
                self.data_provenance.get("evaluation_regime")
                if self.data_provenance else "transductive"
            ),
            "config": {
                "d_model":       self.d_model,
                "num_heads":     self.num_heads,
                "num_nodes":     len(self.node_ids),
                "num_sequences": self.Z_matrix.shape[0] if self.Z_matrix is not None else 0,
                "mode":          "node_only",
            },
        }
        torch.save(save_dict, output_path)
        print(f"\n[Output] Saved node-only output to {output_path}")
        print(f"  Keys: {list(save_dict.keys())}")

    # ======================================================================
    # Full Pipeline
    # ======================================================================

    # def run(
    #     self,
    #     output_path: Optional[str] = None,
    #     weights_path: Optional[str] = None,
    # ) -> CrossAttentionOutput:
    def run(
        self,
        output_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        node_only: bool = False,
    ) -> CrossAttentionOutput:
        """Run the full multi-attention encoder pipeline.

        Parameters
        ----------
        output_path : str or None
            If provided, save results to this ``.pt`` file.
        weights_path : str or None
            If provided, load trained attention-module weights so the encoder
            uses learned (not randomly-initialised) parameters.
        """
        print("=" * 60)
        print("Multi-Attention Encoder Pipeline")
        print(f"  THP JSON    : {self.thp_json_path}")
        print(f"  Tree CSV    : {self.tree_csv_path}")
        print(f"  Summary CSV : {self.summary_csv_path}")
        print(f"  Checkpoint  : {self.checkpoint_path}")
        print(f"  Weights     : {weights_path}")
        print(f"  d_model     : {self.d_model}")
        print(f"  num_heads   : {self.num_heads}")
        print(f"  device      : {self.device}")
        print("=" * 60)

        # ---- Data loading ----
        self.load_node_sequences()
        self.load_all_sequences()
        self.build_global_id_mapping()

        # ---- Phase 1: THP Global Encoding ----
        self.encode_all_sequences_thp()

        # ---- Phase 2: Node Input  h_i^0 = [u_i ; s_i ; g_i] ----
        self.encode_nodes()

        # ---- Phase 3: Structural Attention ----
        self.apply_structural_attention()

        # ---- Optionally load trained weights (modules now exist) ----
        if weights_path is not None:
            if self.cross_attention is None:
                self._build_cross_attention()
            metadata = self.load_module_weights(weights_path)
            self._set_module_mode()
            split = metadata.get("split", {})
            train_pool_ids = split.get("train_indices")
            if train_pool_ids is not None:
                print(
                    f"[Leakage guard] Rebuilding H_tree from "
                    f"{len(train_pool_ids)} training sequences only."
                )
            # Recompute node + structural embeddings with the trained weights.
            with self._grad_ctx():
                self.H_tree = self.forward_node_inputs(
                    allowed_sequence_ids=train_pool_ids
                )
                self.H_tree_refined = self.forward_structural(self.H_tree)

        # ---- Phase 4: Cross-Attention (skipped in node_only mode) ----
        if node_only:
            # Build Z_matrix so it is available for saving (normally populated
            # inside apply_cross_attention, which we skip here).
            self.build_Z_matrix()
            if output_path is not None:
                self.save_node_output(output_path)
            print("\n" + "=" * 60)
            print("Pipeline complete! (node_only mode -- Phase 4 skipped)")
            print(f"  H_tree_refined : {list(self.H_tree_refined.shape)}")
            print(f"  H_tree         : {list(self.H_tree.shape)}")
            print(f"  Z_matrix       : {list(self.Z_matrix.shape)}")
            print("=" * 60)
            return None

        output = self.apply_cross_attention()

        # ---- Save ----
        if output_path is not None:
            self.save_output(output_path)

        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print(f"  D_tree       : {list(self.D_tree.shape) if self.D_tree is not None else 'N/A'}")
        print(f"  route_prob   : {list(self.route_prob.shape) if self.route_prob is not None else 'N/A'}")
        print("=" * 60)

        return output


# ============================================================================
# CLI Entry Point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Attention Encoder: THP → Node Embedding → Structural Attention → Cross Attention → D_tree"
    )

    # Data paths
    parser.add_argument(
        "--thp_json",
        type=str,
        default="/Volumes/shenzm/Shuang_RA/Data/tree_8/8Cluster/THP_8.json",
        help="Path to THP JSON file",
    )
    parser.add_argument(
        "--tree_csv",
        type=str,
        default="/Volumes/shenzm/Shuang_RA/Data/tree_8/tree_node_sequences.csv",
        help="Path to tree_node_sequences.csv",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Path to sequence_summary.csv (Hawkes params for g_i). "
             "Defaults to sequence_summary.csv next to --tree_csv.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained THP checkpoint (.pt)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained attention-module weights (.pt) from Train.py. "
             "If omitted, modules use random init.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/xinye/Hawkes-Memory/Data/tree_20/d_tree.pt",
        help="Output .pt file for D_tree and related tensors",
    )

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_rnn", type=int, default=256)
    parser.add_argument("--d_inner_hid", type=int, default=128)
    parser.add_argument("--d_k", type=int, default=16)
    parser.add_argument("--d_v", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--parent_emb_dim", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--node_only",
        action="store_true",
        help="Skip Phase 4 (cross-attention). Output H_tree_refined [N, d_model] instead of D_tree [S, N, d_model].",
    )
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Strict shared split manifest.")
    parser.add_argument("--split-data-path", type=Path, default=None,
                        help="Source CSV whose SHA-256 is recorded by the manifest.")

    args = parser.parse_args()

    pipeline = MultiAttentionEncoderPipeline(
        thp_json_path=args.thp_json,
        tree_csv_path=args.tree_csv,
        checkpoint_path=args.checkpoint,
        summary_csv_path=args.summary_csv,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_rnn=args.d_rnn,
        d_inner_hid=args.d_inner_hid,
        d_k=args.d_k,
        d_v=args.d_v,
        n_layers=args.n_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        parent_emb_dim=args.parent_emb_dim,
        device=args.device,
    )

    if args.split_manifest is not None:
        if args.split_data_path is None:
            parser.error("--split-data-path is required with --split-manifest")
        pipeline.load_all_sequences()
        pipeline.build_global_id_mapping()
        manifest = load_strict_manifest(
            args.split_manifest,
            data_path=args.split_data_path,
            available_source_ids=pipeline.global_id_to_key.keys(),
        )
        pipeline.data_provenance = build_data_provenance(
            manifest,
            thp_checkpoint=args.checkpoint,
            attention_weights=args.weights,
        )
        if args.weights is None:
            parser.error("strict H-tree construction requires --weights")
        weights_payload = torch.load(args.weights, map_location="cpu")
        weights_provenance = weights_payload.get("metadata", {}).get(
            "data_provenance", {}
        )
        if weights_provenance.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(
                "attention weights were not trained with the requested split manifest"
            )

    output = pipeline.run(output_path=args.output, weights_path=args.weights, node_only=args.node_only)
