"""
Train.py — Train the Multi-Attention Encoder (avoid random-init features)
=========================================================================

Why training is needed
----------------------
``NodeEmbedding`` / ``NodeInputFusion`` / ``RelationBias`` / ``StructuralTreeBlock``
/ ``TreeNodeCrossAttention`` are randomly initialised.  Run as-is, the encoder
produces (mostly) random deterministic features.  This module trains them
end-to-end while keeping the pretrained THP encoder frozen.

Self-supervision signal
------------------------
We already know which leaf each sequence belongs to (from
``tree_node_sequences.csv``: every leaf node lists its global sequence IDs).
This gives a free, exact label:

    target_leaf(k) = the leaf node whose sequence-set contains sequence k

The cross-attention ``route_prob`` (mass each node receives) is trained so that
each sequence routes to its true leaf.  Data is split into train/dev/test with
the same deterministic rule used by ``THP/Main.py``.  Only training sequences
participate in gradient updates and node-level semantic pooling, so held-out
embeddings cannot leak through ``H_tree``.

    L_route = NLL( log route_prob ,  target_leaf )            (primary)
    L_path  = BCE( route_prob over ancestor path , membership ) (optional)
    L_recon = MSE( recon(d_i*(x)) , z_x )                      (optional aux)

where ``d_i*`` is the deterministic feature at the target leaf.  ``L_recon``
forces the features to retain per-sequence information.

Usage
-----
  python Train.py \
      --thp_json   /Volumes/shenzm/Shuang_RA/Data/tree_8/8Cluster/THP_8.json \
      --tree_csv   /Volumes/shenzm/Shuang_RA/Data/tree_8/tree_node_sequences.csv \
      --checkpoint /path/to/thp_checkpoint.pt \
      --weights_out /Volumes/shenzm/Shuang_RA/Data/tree_8/encoder_weights.pt \
      --epochs 50 --lr 1e-3

The resulting weights file is consumed by ``AttenEncoderMain.py --weights``.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Collection, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from AttenEncoderMain_v1 import MultiAttentionEncoderPipeline

_ENCODER_ROOT = Path(__file__).resolve().parents[1]
if str(_ENCODER_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENCODER_ROOT))
from SplitManifest import build_data_provenance, load_strict_manifest


# ---------------------------------------------------------------------------
# Dataset split (identical ordering rule to THP/Main.py::split_data)
# ---------------------------------------------------------------------------
def split_sequence_indices(
    num_sequences: int,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic, disjoint train/dev/test global sequence IDs."""
    if num_sequences < 3:
        raise ValueError("At least 3 sequences are required for train/dev/test.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError(f"dev_ratio must be in (0, 1), got {dev_ratio}.")
    if train_ratio + dev_ratio >= 1.0:
        raise ValueError(
            "train_ratio + dev_ratio must be < 1 so test is non-empty."
        )

    permutation = np.random.RandomState(seed).permutation(num_sequences)
    n_train = int(num_sequences * train_ratio)
    n_dev = int(num_sequences * dev_ratio)
    n_test = num_sequences - n_train - n_dev
    if min(n_train, n_dev, n_test) <= 0:
        raise ValueError(
            "The requested ratios produce an empty split: "
            f"train={n_train}, dev={n_dev}, test={n_test}."
        )

    return (
        torch.as_tensor(permutation[:n_train], dtype=torch.long),
        torch.as_tensor(
            permutation[n_train:n_train + n_dev], dtype=torch.long
        ),
        torch.as_tensor(permutation[n_train + n_dev:], dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Routing-supervision targets
# ---------------------------------------------------------------------------
def build_routing_targets(
    pipeline: MultiAttentionEncoderPipeline,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build per-sequence routing targets from leaf membership.

    Returns
    -------
    leaf_target : LongTensor [S]
        For each sequence (row of ``Z_matrix`` == global ID), the node index of
        its leaf.  ``-1`` if not found (ignored in the loss).
    path_target : FloatTensor [S, N]
        Multi-label membership: 1 for every ancestor node (root..leaf) that
        contains the sequence, else 0.
    """
    node_ids = pipeline.node_ids
    n = len(node_ids)
    s = pipeline.Z_matrix.shape[0]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    leaf_nodes = [nid for nid in node_ids if pipeline.feature_extractor._is_leaf(nid)]

    leaf_target = torch.full((s,), -1, dtype=torch.long)
    path_target = torch.zeros(s, n, dtype=torch.float32)

    # Multi-label membership: a node contains a sequence iff it is listed.
    for nid in node_ids:
        idx = id_to_idx[nid]
        for gid in pipeline.node_sequences.get(nid, []):
            if 0 <= gid < s:
                path_target[gid, idx] = 1.0

    # Leaf assignment.
    for nid in leaf_nodes:
        idx = id_to_idx[nid]
        for gid in pipeline.node_sequences.get(nid, []):
            if 0 <= gid < s:
                leaf_target[gid] = idx

    n_assigned = int((leaf_target >= 0).sum().item())
    print(f"[Targets] {n_assigned}/{s} sequences assigned to a leaf "
          f"({len(leaf_nodes)} leaves, {n} nodes total)")
    return leaf_target, path_target


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class EncoderTrainer:
    def __init__(
        self,
        pipeline: MultiAttentionEncoderPipeline,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        route_weight: float = 1.0,
        path_weight: float = 0.5,
        recon_weight: float = 0.5,
    ):
        self.p = pipeline
        self.device = pipeline.device
        self.route_weight = route_weight
        self.path_weight = path_weight
        self.recon_weight = recon_weight

        # Optional reconstruction head (training-only; not saved with modules).
        self.recon_head = nn.Sequential(
            nn.Linear(pipeline.d_model, pipeline.d_model),
            nn.GELU(),
            nn.Linear(pipeline.d_model, pipeline.d_model),
        ).to(self.device)

        params = list(pipeline.trainable_parameters()) + list(self.recon_head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def _set_training_mode(self, training: bool) -> None:
        for module in self.p.trainable_modules():
            module.train(training)
        self.recon_head.train(training)

    # ------------------------------------------------------------------
    def _forward_batch(
        self, H_refined: torch.Tensor, z_batch: torch.Tensor
    ):
        out = self.p.forward_cross(H_refined, z_batch)
        return out

    def _compute_loss(
        self,
        out,
        z_batch: torch.Tensor,
        leaf_t: torch.Tensor,
        path_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        route_prob = out.route_prob                      # [B, N]
        log_route = torch.log(route_prob.clamp_min(1e-9))

        valid = leaf_t >= 0
        losses: Dict[str, float] = {}
        total = torch.zeros((), device=self.device)

        # ---- Routing CE (to true leaf) ----
        if valid.any():
            l_route = F.nll_loss(log_route[valid], leaf_t[valid])
            total = total + self.route_weight * l_route
            losses["route"] = float(l_route.item())

        # ---- Ancestor-path BCE (multi-label membership) ----
        if self.path_weight > 0:
            l_path = F.binary_cross_entropy(
                route_prob.clamp(1e-9, 1 - 1e-9), path_t
            )
            total = total + self.path_weight * l_path
            losses["path"] = float(l_path.item())

        # ---- Reconstruction of z from the target-leaf deterministic feature ----
        if self.recon_weight > 0 and valid.any():
            det = out.deterministic_tree                 # [B, N, d]
            idx = leaf_t[valid]                          # [Bv]
            d_target = det[valid][torch.arange(idx.size(0), device=self.device), idx]
            z_hat = self.recon_head(d_target)
            l_recon = F.mse_loss(z_hat, z_batch[valid])
            total = total + self.recon_weight * l_recon
            losses["recon"] = float(l_recon.item())

        losses["total"] = float(total.item())
        return total, losses

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(
        self,
        indices: torch.Tensor,
        leaf_t: torch.Tensor,
        path_t: torch.Tensor,
        train_pool_ids: Collection[int],
        batch_size: int,
    ) -> Dict[str, float]:
        """Evaluate a split while constructing ``H_tree`` from train IDs only."""
        self._set_training_mode(False)
        p = self.p
        Z = p.Z_matrix

        # This shared context is calculated once per evaluation.  Restricting
        # its semantic pool is essential: otherwise dev/test z values enter
        # their own routing context even when their target rows are held out.
        H_tree = p.forward_node_inputs(allowed_sequence_ids=train_pool_ids)
        H_refined = p.forward_structural(H_tree)

        totals: Dict[str, float] = {}
        examples = 0
        correct = 0
        counted = 0
        indices = indices.to(self.device)
        for start in range(0, indices.numel(), batch_size):
            idx = indices[start:start + batch_size]
            z_batch = Z[idx]
            out = self._forward_batch(H_refined, z_batch)
            _, parts = self._compute_loss(
                out, z_batch, leaf_t[idx], path_t[idx]
            )
            batch_n = int(idx.numel())
            examples += batch_n
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * batch_n

            valid = leaf_t[idx] >= 0
            if valid.any():
                pred = out.route_prob[valid].argmax(dim=-1)
                correct += int((pred == leaf_t[idx][valid]).sum().item())
                counted += int(valid.sum().item())

        metrics = {
            key: value / max(examples, 1) for key, value in totals.items()
        }
        metrics["route_acc"] = correct / counted if counted else 0.0
        return metrics

    @staticmethod
    def _format_metrics(metrics: Dict[str, float]) -> str:
        ordered = ("route", "path", "recon", "total", "route_acc")
        return " | ".join(
            f"{key}={metrics[key]:.4f}" for key in ordered if key in metrics
        )

    # ------------------------------------------------------------------
    def train(
        self,
        epochs: int = 50,
        batch_size: int = 64,
        grad_clip: float = 1.0,
        log_every: int = 1,
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        split_seed: int = 42,
        patience: int = 10,
        min_delta: float = 1e-4,
        split_manifest: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        p = self.p
        Z = p.Z_matrix                                   # [S, d]
        S = Z.shape[0]
        leaf_t, path_t = build_routing_targets(p)
        leaf_t = leaf_t.to(self.device)
        path_t = path_t.to(self.device)
        if split_manifest is None:
            train_pos, dev_pos, test_pos = split_sequence_indices(
                S,
                train_ratio=train_ratio,
                dev_ratio=dev_ratio,
                seed=split_seed,
            )
            # Legacy mode preserves the historical insertion-position split.
            thp_order_global_ids = torch.tensor(
                [p.key_to_global_id[key] for key in p.all_sequences.keys()],
                dtype=torch.long,
            )
            train_cpu = thp_order_global_ids[train_pos]
            dev_cpu = thp_order_global_ids[dev_pos]
            test_cpu = thp_order_global_ids[test_pos]
        else:
            splits = split_manifest["splits"]
            train_cpu = torch.tensor(splits["train"], dtype=torch.long)
            dev_cpu = torch.tensor(splits["validation"], dtype=torch.long)
            test_cpu = torch.tensor(splits["test"], dtype=torch.long)
        train_pool_ids = frozenset(train_cpu.tolist())
        train_idx = train_cpu.to(self.device)

        print("\n" + "=" * 60)
        print(
            f"Split: train={train_cpu.numel()}, dev={dev_cpu.numel()}, "
            f"test={test_cpu.numel()} (seed={split_seed})"
        )
        print(
            f"Training: {epochs} epochs, {train_cpu.numel()} train sequences, "
            f"batch={batch_size}"
        )
        print("[Leakage guard] H_tree semantic pooling uses train sequences only.")
        print("=" * 60)

        best_dev_route = float("inf")
        best_epoch = 0
        best_modules = None
        best_recon = None
        stale_epochs = 0

        for epoch in range(1, epochs + 1):
            self._set_training_mode(True)
            order = torch.randperm(train_idx.numel(), device=self.device)
            perm = train_idx[order]
            epoch_losses: Dict[str, float] = {}
            n_batches = 0
            correct = 0
            counted = 0

            for start in range(0, train_idx.numel(), batch_size):
                idx = perm[start:start + batch_size]
                z_batch = Z[idx]

                # Recompute node + structural embeddings so gradients flow.
                # The semantic pool contains no dev/test sequence embeddings.
                H_tree = p.forward_node_inputs(
                    allowed_sequence_ids=train_pool_ids
                )
                H_refined = p.forward_structural(H_tree)
                out = self._forward_batch(H_refined, z_batch)

                loss, parts = self._compute_loss(
                    out, z_batch, leaf_t[idx], path_t[idx]
                )

                self.optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(p.trainable_parameters())
                        + list(self.recon_head.parameters()),
                        grad_clip,
                    )
                self.optimizer.step()

                for key, value in parts.items():
                    epoch_losses[key] = epoch_losses.get(key, 0.0) + value
                n_batches += 1

                valid = leaf_t[idx] >= 0
                if valid.any():
                    pred = out.route_prob[valid].argmax(dim=-1)
                    correct += int((pred == leaf_t[idx][valid]).sum().item())
                    counted += int(valid.sum().item())

            train_metrics = {
                key: value / max(n_batches, 1)
                for key, value in epoch_losses.items()
            }
            train_metrics["route_acc"] = correct / counted if counted else 0.0
            dev_metrics = self._evaluate(
                dev_cpu,
                leaf_t,
                path_t,
                train_pool_ids=train_pool_ids,
                batch_size=batch_size,
            )
            dev_route = dev_metrics.get("route", dev_metrics["total"])

            if dev_route < best_dev_route - min_delta:
                best_dev_route = dev_route
                best_epoch = epoch
                best_modules = [
                    copy.deepcopy(module.state_dict())
                    for module in p.trainable_modules()
                ]
                best_recon = copy.deepcopy(self.recon_head.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if epoch % log_every == 0:
                print(
                    f"  Epoch {epoch:3d}/{epochs}  "
                    f"train[{self._format_metrics(train_metrics)}]  "
                    f"dev[{self._format_metrics(dev_metrics)}]"
                )

            if patience > 0 and stale_epochs >= patience:
                print(
                    f"[Early stopping] dev route loss did not improve by "
                    f"{min_delta:g} for {patience} epochs."
                )
                break

        if best_modules is None or best_recon is None:
            raise RuntimeError("Training finished without a valid dev checkpoint.")
        for module, state in zip(p.trainable_modules(), best_modules):
            module.load_state_dict(state)
        self.recon_head.load_state_dict(best_recon)

        test_metrics = self._evaluate(
            test_cpu,
            leaf_t,
            path_t,
            train_pool_ids=train_pool_ids,
            batch_size=batch_size,
        )
        print("=" * 60)
        print(
            f"Training complete. Restored epoch {best_epoch} "
            f"(best dev route loss={best_dev_route:.6f})."
        )
        print(f"[Test] {self._format_metrics(test_metrics)}")
        return {
            "best_epoch": best_epoch,
            "best_dev_route_loss": best_dev_route,
            "test_metrics": test_metrics,
            "split": {
                "seed": split_seed,
                "train_ratio": train_ratio,
                "dev_ratio": dev_ratio,
                "train_indices": train_cpu.tolist(),
                "dev_indices": dev_cpu.tolist(),
                "test_indices": test_cpu.tolist(),
                "node_pool": "train_only",
            },
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Multi-Attention Encoder.")
    parser.add_argument("--thp_json", type=str,
                        default="/Volumes/shenzm/Shuang_RA/Data/tree_8/8Cluster/THP_8.json")
    parser.add_argument("--tree_csv", type=str,
                        default="/Volumes/shenzm/Shuang_RA/Data/tree_8/tree_node_sequences.csv")
    parser.add_argument("--summary_csv", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pretrained THP checkpoint (.pt), kept frozen.")
    parser.add_argument("--weights_out", type=str,
                        default="/Volumes/shenzm/Shuang_RA/Data/tree_8/encoder_weights.pt")

    # Model hyperparameters (must match AttenEncoderMain / THP checkpoint)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_rnn", type=int, default=256)
    parser.add_argument("--d_inner_hid", type=int, default=128)
    parser.add_argument("--d_k", type=int, default=16)
    parser.add_argument("--d_v", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--parent_emb_dim", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--route_weight", type=float, default=1.0)
    parser.add_argument("--path_weight", type=float, default=0.0,
                        help="Ancestor-path BCE weight. Note route_prob is a "
                             "distribution (sums to 1), so this conflicts with "
                             "multi-label membership; keep 0 unless experimenting.")
    parser.add_argument("--recon_weight", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible attention weights.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stop patience measured on dev routing loss; "
                             "0 disables early stopping.")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="Minimum dev routing-loss improvement.")
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Strict shared split manifest.")
    parser.add_argument("--split-data-path", type=Path, default=None,
                        help="Source CSV whose SHA-256 is recorded by the manifest.")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[Reproducibility] seed={args.seed}")

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

    # Enable gradients for the attention modules.
    pipeline._no_grad = False

    # ---- Data + frozen THP encoding (Phase 1) ----
    pipeline.load_node_sequences()
    pipeline.load_all_sequences()
    pipeline.build_global_id_mapping()
    strict_manifest = None
    if args.split_manifest is not None:
        if args.split_data_path is None:
            parser.error("--split-data-path is required with --split-manifest")
        strict_manifest = load_strict_manifest(
            args.split_manifest,
            data_path=args.split_data_path,
            available_source_ids=pipeline.global_id_to_key.keys(),
        )
    pipeline.encode_all_sequences_thp()   # frozen, run under no_grad internally

    # ---- Build trainable modules + features + relation tensors ----
    pipeline.setup_modules()

    # ---- Train ----
    trainer = EncoderTrainer(
        pipeline,
        lr=args.lr,
        weight_decay=args.weight_decay,
        route_weight=args.route_weight,
        path_weight=args.path_weight,
        recon_weight=args.recon_weight,
    )
    training_result = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        split_seed=args.seed,
        patience=args.patience,
        min_delta=args.min_delta,
        split_manifest=strict_manifest,
    )

    # ---- Save the best dev checkpoint and its split metadata ----
    metadata = {
        "best_epoch": training_result["best_epoch"],
        "best_dev_route_loss": training_result["best_dev_route_loss"],
        "test_metrics": training_result["test_metrics"],
        "split": training_result["split"],
        "evaluation_regime": (
            "strict_inductive" if strict_manifest is not None else "transductive"
        ),
    }
    if strict_manifest is not None:
        metadata["data_provenance"] = build_data_provenance(
            strict_manifest, thp_checkpoint=args.checkpoint
        )
    pipeline.save_module_weights(
        args.weights_out,
        metadata=metadata,
    )
    print(f"\nDone. Use them with:\n"
          f"  python AttenEncoderMain.py --checkpoint {args.checkpoint} "
          f"--weights {args.weights_out}")


if __name__ == "__main__":
    main()
