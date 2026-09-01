#!/usr/bin/env python3
"""
Evaluate MultiAttentionEncoder output (h_tree.pt / d_tree.pt).

Metrics
-------
  1. Routing confidence  — entropy & max-probability of route_prob (full pipeline only)
  2. Routing purity      — do sequences in each leaf share the same cluster? (full pipeline)
  3. Clustering quality  — NMI / ARI: routed leaf vs true leaf labels (full pipeline)
  4. Embedding separation— within- vs between-cluster cosine similarity + silhouette
                           (Z_matrix baseline, using --tree_csv for ground-truth labels)
  5. Node diversity      — Phase-3 structural attention shift per node;
                           leaf-node pairwise similarity
  6. t-SNE plots         — Z_matrix coloured by true leaf assignment
  7. Route-prob heatmap  — sequences × nodes, sorted by leaf (full pipeline only)

Usage
-----
  python eval_encoder.py --h_tree /path/to/h_tree.pt --tree_csv /path/to/tree_node_sequences.csv
  python eval_encoder.py --h_tree /path/to/h_tree.pt --tree_csv ... --output_dir ./eval_out
  python eval_encoder.py --h_tree /path/to/h_tree.pt --tree_csv ... --no_plot

Interpretation guide
--------------------
  Routing purity   > 0.90  → sequences cleanly routed to correct leaf
  NMI / ARI        > 0.80  → strong agreement with ground-truth leaf labels
  Silhouette (Z)   higher  → THP captures cluster structure well before encoder
  Leaf cosine sim  < 0.80  → leaf nodes distinct (good); > 0.95 → possible collapse
  Phase-3 delta    varies  → depth-dependent variation means structural attn is working
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
        silhouette_score,
    )
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[warn] scikit-learn not found — NMI, ARI, silhouette, t-SNE unavailable.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not found — plots unavailable.")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_pt(path: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"[info] Loaded: {path}")
    print(f"[info] Keys:   {sorted(data.keys())}")
    return data


def identify_leaves(node_ids: list) -> tuple:
    """
    A node is a leaf if neither of its expected children appears in node_ids.
    Naming convention:  root → {l, r},  l → {l_l, l_r},  l_r → {l_r_l, l_r_r}, ...
    """
    node_set = set(node_ids)
    leaf_ids, leaf_indices = [], []
    for i, nid in enumerate(node_ids):
        if nid == "root":
            children = {"l", "r"}
        else:
            children = {f"{nid}_l", f"{nid}_r"}
        if not children & node_set:
            leaf_ids.append(nid)
            leaf_indices.append(i)
    return leaf_ids, leaf_indices


def load_leaf_labels(tree_csv_path: str, node_ids: list) -> dict:
    """
    Read tree_node_sequences.csv and return {global_seq_id (int): leaf_index (int)}.
    leaf_index is the position of the leaf in identify_leaves(node_ids) output.
    """
    import ast
    import csv
    leaf_ids, _ = identify_leaves(node_ids)
    leaf_set = set(leaf_ids)
    seq_to_leaf = {}
    with open(tree_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            pos = row["node_position"].strip()
            if pos in leaf_set:
                idx = leaf_ids.index(pos)
                for sid in ast.literal_eval(row["sequences"]):
                    seq_to_leaf[int(sid)] = idx
    return seq_to_leaf


def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    """[S, S] pairwise cosine similarity matrix."""
    norms = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-9
    n = emb / norms
    return n @ n.T


def within_between(emb: np.ndarray, labels: np.ndarray) -> tuple:
    """Mean within-cluster and between-cluster cosine similarity (upper-triangle only)."""
    sim = cosine_sim_matrix(emb)
    S = len(labels)
    same  = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones((S, S), dtype=bool), k=1)
    w = float(sim[same  & upper].mean()) if (same  & upper).any() else float("nan")
    b = float(sim[~same & upper].mean()) if (~same & upper).any() else float("nan")
    return w, b


def section(title: str) -> None:
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_pt(args.h_tree)

    Z_matrix       = data.get("Z_matrix")          # [S, d_model]
    H_tree         = data.get("H_tree")             # [N, d_model]
    H_tree_refined = data.get("H_tree_refined")     # [N, d_model]
    D_tree         = data.get("D_tree")             # [S, N, d_model]  (None in node_only mode)
    route_prob     = data.get("route_prob")         # [S, N]            (None in node_only mode)
    node_ids       = list(data["node_ids"])
    sequence_keys  = list(data.get("sequence_keys", []))
    config         = data.get("config", {})

    S       = Z_matrix.shape[0]
    N       = len(node_ids)
    d_model = Z_matrix.shape[-1]

    section("Encoder Output — Basic Stats")
    print(f"  Sequences  (S):     {S}")
    print(f"  Nodes      (N):     {N}   {node_ids}")
    print(f"  d_model:            {d_model}")
    print(f"  Config:             {config}")
    print(f"  D_tree present:     {D_tree is not None}")
    print(f"  route_prob present: {route_prob is not None}")

    # ground truth (leaf-based cluster labels from tree_csv)
    true_labels = None
    num_clusters = 0
    leaf_ids, leaf_indices = identify_leaves(node_ids)
    if args.tree_csv and sequence_keys:
        seq_to_leaf = load_leaf_labels(args.tree_csv, node_ids)
        true_labels = np.array([seq_to_leaf[int(k)] for k in sequence_keys])
        num_clusters = len(leaf_ids)
        print(f"\n  Leaf labels loaded from: {args.tree_csv}")
        print(f"  Num leaf clusters:  {num_clusters}  ({leaf_ids})")
    elif sequence_keys:
        print("\n  [warn] --tree_csv not provided — cluster-separation metrics skipped.")
    else:
        print("\n  [warn] No sequence_keys in file — supervised metrics skipped.")

    print(f"\n  Leaf nodes ({len(leaf_ids)}): {leaf_ids}")

    # ── routing prep ──────────────────────────────────────────────────────────
    leaf_assign = None        # [S] index into leaf_ids
    node_assign = None        # [S] index into node_ids  (the assigned leaf)
    max_leaf_prob = None

    if route_prob is not None:
        rp = route_prob.float().numpy()               # [S, N]
        rp_leaves = rp[:, leaf_indices]               # [S, L]
        rp_leaves_norm = rp_leaves / (rp_leaves.sum(axis=-1, keepdims=True) + 1e-9)

        leaf_assign  = rp_leaves_norm.argmax(axis=-1)                # [S]
        node_assign  = np.array([leaf_indices[i] for i in leaf_assign])  # [S]
        max_leaf_prob = rp_leaves_norm.max(axis=-1)                  # [S]

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Routing confidence
    # ─────────────────────────────────────────────────────────────────────────
    if route_prob is not None:
        section("1. Routing Confidence")

        ent_all  = -(rp * np.log(rp + 1e-10)).sum(axis=-1)
        ent_leaf = -(rp_leaves_norm * np.log(rp_leaves_norm + 1e-10)).sum(axis=-1)

        print(f"  Route entropy  (all {N} nodes):     "
              f"mean={ent_all.mean():.3f}  std={ent_all.std():.3f}")
        print(f"  Route entropy  (leaves only):       "
              f"mean={ent_leaf.mean():.3f}  std={ent_leaf.std():.3f}")
        print(f"  Max leaf-route probability:         "
              f"mean={max_leaf_prob.mean():.3f}  std={max_leaf_prob.std():.3f}")
        pct_decisive = (max_leaf_prob > 0.8).mean() * 100
        print(f"  Sequences routed with p>0.80:       {pct_decisive:.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # 2 & 3. Routing purity, NMI, ARI
    # ─────────────────────────────────────────────────────────────────────────
    purity = nmi = ari = None

    if route_prob is not None and true_labels is not None:
        section("2 & 3. Routing Purity / NMI / ARI")

        groups = defaultdict(list)
        for si, li in enumerate(leaf_assign):
            groups[leaf_ids[li]].append(true_labels[si])

        total_correct = 0
        for lid in sorted(groups):
            lbls = np.array(groups[lid])
            cnt  = np.bincount(lbls)
            maj_leaf = leaf_ids[cnt.argmax()]
            total_correct += cnt.max()
            print(f"  Leaf {lid:>14s}: {len(lbls):3d} seqs | "
                  f"majority_true_leaf={maj_leaf} | purity={cnt.max()/len(lbls):.3f}")

        purity = total_correct / S
        print(f"\n  Overall routing purity:  {purity:.4f}   "
              f"({'✓ good' if purity > 0.9 else '✗ low'})")

        if HAS_SKLEARN:
            nmi = normalized_mutual_info_score(true_labels, leaf_assign)
            ari = adjusted_rand_score(true_labels, leaf_assign)
            print(f"  NMI (route vs true):     {nmi:.4f}   "
                  f"({'✓ good' if nmi > 0.8 else '✗ low'})")
            print(f"  ARI (route vs true):     {ari:.4f}   "
                  f"({'✓ good' if ari > 0.8 else '✗ low'})")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Embedding cluster separation
    # ─────────────────────────────────────────────────────────────────────────
    z_sil = d_sil = None

    if true_labels is not None:
        section("4. Embedding Cluster Separation")

        embeddings = [("Z_matrix  [THP baseline]", Z_matrix.numpy())]

        if H_tree_refined is not None and node_assign is not None:
            h_at_leaf = H_tree_refined.numpy()[node_assign]   # [S, d_model]
            embeddings.append(("H_tree_refined @ leaf", h_at_leaf))

        if D_tree is not None and node_assign is not None:
            d_at_leaf = D_tree.numpy()[np.arange(S), node_assign]  # [S, d_model]
            embeddings.append(("D_tree @ assigned leaf", d_at_leaf))

        for name, emb in embeddings:
            w, b = within_between(emb, true_labels)
            sep = w / max(b, 1e-9)
            print(f"\n  [{name}]")
            print(f"    Within-cluster cosine sim:   {w:.4f}")
            print(f"    Between-cluster cosine sim:  {b:.4f}")
            print(f"    Separation ratio (W/B):      {sep:.4f}")

            if HAS_SKLEARN and num_clusters >= 2 and S > num_clusters:
                sil = silhouette_score(emb, true_labels)
                print(f"    Silhouette score:            {sil:.4f}")
                if "Z_matrix" in name:
                    z_sil = sil
                elif "D_tree" in name:
                    d_sil = sil

        if z_sil is not None and d_sil is not None:
            delta = d_sil - z_sil
            sign  = "+" if delta >= 0 else ""
            print(f"\n  Encoder improvement over THP (Δsilhouette): {sign}{delta:.4f}  "
                  f"({'✓ encoder adds structure' if delta > 0 else '✗ no gain'})")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Node embedding diversity
    # ─────────────────────────────────────────────────────────────────────────
    if H_tree is not None and H_tree_refined is not None:
        section("5. Node Embedding Diversity")

        h_np  = H_tree.numpy()
        hr_np = H_tree_refined.numpy()
        delta = np.linalg.norm(hr_np - h_np, axis=-1)

        print(f"  Phase-3 structural attention — L2 shift per node:")
        for i, nid in enumerate(node_ids):
            tag = " ← leaf" if nid in leaf_ids else ""
            print(f"    {nid:>16s}: {delta[i]:.4f}{tag}")

        leaf_embs = hr_np[leaf_indices]  # [L, d_model]
        sim_mat   = cosine_sim_matrix(leaf_embs)
        print(f"\n  Leaf-node pairwise cosine similarity (H_tree_refined):")
        for i in range(len(leaf_ids)):
            for j in range(i + 1, len(leaf_ids)):
                v = sim_mat[i, j]
                flag = "  (similar — possible collapse)" if v > 0.95 else ""
                print(f"    {leaf_ids[i]:>14s} ↔ {leaf_ids[j]:<14s}: {v:.4f}{flag}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6 & 7. Plots
    # ─────────────────────────────────────────────────────────────────────────
    if not args.no_plot and HAS_MPL and true_labels is not None:
        section("6 & 7. Visualizations")

        # --- t-SNE ---
        if HAS_SKLEARN:
            to_plot = [("Z_matrix (THP)", Z_matrix.numpy())]
            if D_tree is not None and node_assign is not None:
                to_plot.append(("D_tree @ assigned leaf",
                                D_tree.numpy()[np.arange(S), node_assign]))

            ncols = len(to_plot)
            fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
            if ncols == 1:
                axes = [axes]
            fig.suptitle("Sequence Embeddings — t-SNE coloured by cluster", fontsize=13)

            cmap = plt.get_cmap("tab20", num_clusters)
            for ax, (name, emb) in zip(axes, to_plot):
                perp = min(30, max(5, S // 5))
                emb2 = TSNE(n_components=2, random_state=42,
                            perplexity=perp).fit_transform(emb)
                sc = ax.scatter(emb2[:, 0], emb2[:, 1],
                                c=true_labels, cmap=cmap,
                                vmin=-0.5, vmax=num_clusters - 0.5,
                                alpha=0.75, s=25)
                ax.set_title(name)
                ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
                cb = plt.colorbar(sc, ax=ax, ticks=range(num_clusters))
                cb.set_ticklabels(leaf_ids)

            fig.tight_layout()
            p = os.path.join(args.output_dir, "tsne.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"  Saved: {p}")

        # --- Route-probability heatmap ---
        if route_prob is not None:
            sort_order = np.argsort(true_labels)
            rp_sorted  = rp[sort_order]

            fig_h = max(6, S * 0.06)
            fig_w = max(8, N * 0.8)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            im = ax.imshow(rp_sorted, aspect="auto", cmap="viridis",
                           vmin=0, vmax=rp_sorted.max())
            ax.set_xticks(range(N))
            ax.set_xticklabels(node_ids, rotation=45, ha="right", fontsize=9)
            ax.set_xlabel("Node")
            ax.set_ylabel("Sequence (sorted by cluster)")
            ax.set_title("Route Probability — sequences sorted by true cluster")
            plt.colorbar(im, ax=ax, label="probability")

            boundaries = np.where(np.diff(true_labels[sort_order]))[0] + 1
            for b in boundaries:
                ax.axhline(b - 0.5, color="red", lw=0.8, ls="--", alpha=0.7)

            fig.tight_layout()
            p = os.path.join(args.output_dir, "route_prob_heatmap.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"  Saved: {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    section("Summary")
    if max_leaf_prob is not None:
        print(f"  Routing confidence (mean max-prob):  {max_leaf_prob.mean():.4f}")
    if purity is not None:
        print(f"  Routing purity:                      {purity:.4f}")
    if nmi is not None:
        print(f"  NMI:                                 {nmi:.4f}")
        print(f"  ARI:                                 {ari:.4f}")
    if z_sil is not None:
        print(f"  Silhouette — Z_matrix (THP):         {z_sil:.4f}")
    if d_sil is not None:
        print(f"  Silhouette — D_tree @ leaf:          {d_sil:.4f}")
        print(f"  Δ silhouette (encoder gain):         {d_sil - z_sil:+.4f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate MultiAttentionEncoder output (h_tree.pt / d_tree.pt)."
    )
    ap.add_argument("--h_tree",     required=True,
                    help="Path to the .pt file saved by the encoder pipeline.")
    ap.add_argument("--tree_csv",   default=None,
                    help="Path to tree_node_sequences.csv (needed for cluster metrics).")
    ap.add_argument("--output_dir", default="./eval_results",
                    help="Directory for plots (default: ./eval_results).")
    ap.add_argument("--no_plot",    action="store_true",
                    help="Skip all matplotlib output.")
    evaluate(ap.parse_args())
