"""
DataTreeConstructor.py

For each tree node (defined in a CSV file), independently run THP encoding
on its assigned sequences and save the per-node embeddings.

Input:
  - tree_node_sequences.csv : node_position -> list of global sequence IDs
  - THP_*.json             : all sequence data, keyed by "clusterID_seqIndex"

Output:
  - A single .pt file containing:
      Dict[str, Dict[int, Tensor]]
      i.e. {node_position: {global_seq_id: embedding_tensor}}

Usage:
  python DataTreeConstructor.py
"""

import json
import csv
import ast
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Make local THP module importable.
# The THP package uses flat absolute imports (e.g. ``from TransformerModel import ...``)
# so both the project root *and* the THP directory must be on sys.path.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
_THP_DIR = str(Path(__file__).resolve().parent / "THP")
for _p in (_THP_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from THP.EncodeMain import THPEncoding


# ---------------------------------------------------------------------------
# Helper: parse the "sequences" column of the CSV
# ---------------------------------------------------------------------------
def parse_sequence_list(raw: str) -> List[int]:
    """Convert a CSV string like "[3, 9, 22]" into a list of ints."""
    if isinstance(raw, list):
        return [int(x) for x in raw]
    try:
        parsed = ast.literal_eval(str(raw))
        return [int(x) for x in parsed]
    except (ValueError, SyntaxError):
        return []


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class NodeSequenceEncoder:
    """Encode sequences for each tree node independently using THP.

    Parameters
    ----------
    csv_path : str
        Path to ``tree_node_sequences.csv``.
    json_path : str
        Path to the full THP JSON (e.g. ``THP_8.json``).
    output_path : str
        Where to write the final ``.pt`` file.
    checkpoint : str or None
        Path to a pretrained THP checkpoint.  **Strongly recommended** so that
        encodings are consistent across nodes.  Without a checkpoint each node
        is encoded with a fresh randomly-initialised model.
    batch_size, d_model, d_rnn, d_inner_hid, d_k, d_v, n_head, n_layers,
    dropout, device : passed through to ``THPEncoding``.
    """

    def __init__(
        self,
        csv_path: str,
        json_path: str,
        output_path: str,
        batch_size: int = 16,
        d_model: int = 64,
        d_rnn: int = 256,
        d_inner_hid: int = 128,
        d_k: int = 16,
        d_v: int = 16,
        n_head: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.csv_path = Path(csv_path)
        self.json_path = Path(json_path)
        self.output_path = Path(output_path)

        # Encoder kwargs forwarded to THPEncoding
        self.encoder_kwargs = {
            "batch_size": batch_size,
            "d_model": d_model,
            "d_rnn": d_rnn,
            "d_inner_hid": d_inner_hid,
            "d_k": d_k,
            "d_v": d_v,
            "n_head": n_head,
            "n_layers": n_layers,
            "dropout": dropout,
            "checkpoint": checkpoint,
            "device": device,
        }

        # Internal state ---------------------------------------------------
        self.node_sequences: Dict[str, List[int]] = {}
        """node_position -> sorted list of global sequence IDs"""

        self.all_sequences: Dict[str, Any] = {}
        """Raw JSON content: json_key -> list of event dicts"""

        self.global_id_to_key: Dict[int, str] = {}
        """Mapping: global sequence ID (0..N-1) -> JSON key"""

        self.key_to_global_id: Dict[str, int] = {}
        """Reverse mapping: JSON key -> global sequence ID"""

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_node_sequences(self) -> Dict[str, List[int]]:
        """Read the CSV and populate ``self.node_sequences``."""
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_pos = row["node_position"].strip()
                seqs = parse_sequence_list(row["sequences"])
                self.node_sequences[node_pos] = seqs

        print(f"[NodeSequenceEncoder] Loaded {len(self.node_sequences)} nodes "
              f"from {self.csv_path}")
        return self.node_sequences

    def load_all_sequences(self) -> Dict[str, Any]:
        """Read the full JSON into ``self.all_sequences``."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.all_sequences = json.load(f)
        print(f"[NodeSequenceEncoder] Loaded {len(self.all_sequences)} "
              f"sequences from {self.json_path}")
        return self.all_sequences

    # ------------------------------------------------------------------
    # Global-ID ↔ JSON-key mapping
    # ------------------------------------------------------------------
    def build_global_id_mapping(self) -> Dict[int, str]:
        """Create the bidirectional mapping between global IDs and JSON keys.

        Global ordering: keys are sorted by ``(cluster_id, seq_index)``
        where a JSON key has the form ``"{cluster_id}_{seq_index}"``.

        Example
        -------
        JSON key ``"0_29"`` → cluster 0, index 29 → global ID 0
        JSON key ``"0_61"`` → cluster 0, index 61 → global ID 3
        """
        if not self.all_sequences:
            raise RuntimeError("Call load_all_sequences() first.")

        # Sort by (cluster_id, seq_index)
        sorted_keys = sorted(
            self.all_sequences.keys(),
            key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])),
        )

        self.global_id_to_key = {i: k for i, k in enumerate(sorted_keys)}
        self.key_to_global_id = {k: i for i, k in enumerate(sorted_keys)}

        print(f"[NodeSequenceEncoder] Built global-ID mapping: "
              f"0 … {len(sorted_keys) - 1}  →  JSON keys")
        return self.global_id_to_key

    # ------------------------------------------------------------------
    # Per-node encoding
    # ------------------------------------------------------------------
    def _encode_one_node(
        self,
        node_position: str,
        seq_ids: List[int],
        temp_dir: Path,
    ) -> Dict[int, torch.Tensor]:
        """Run THP encoding for a single node.

        Parameters
        ----------
        node_position : str
            Node label (e.g. ``"root"``, ``"l"``, ``"r_r_l"``).
        seq_ids : list[int]
            Global sequence IDs belonging to this node.
        temp_dir : Path
            Directory where the temporary JSON for this node is written.

        Returns
        -------
        dict[int, Tensor]
            Mapping ``global_seq_id → embedding_tensor``.
        """
        if not seq_ids:
            print(f"  [Skip] Node '{node_position}' has no sequences.")
            return {}

        # Build a subset dict with the original JSON keys
        node_data: Dict[str, Any] = {}
        for gid in seq_ids:
            json_key = self.global_id_to_key.get(gid)
            if json_key is None:
                print(f"  [Warn] global ID {gid} has no JSON key – skipping.")
                continue
            if json_key not in self.all_sequences:
                print(f"  [Warn] JSON key '{json_key}' not in data – skipping.")
                continue
            node_data[json_key] = self.all_sequences[json_key]

        # Write temporary JSON for the THP encoder
        safe_name = node_position.replace("/", "_").replace("\\", "_")
        temp_path = temp_dir / f"node_{safe_name}.json"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(node_data, f, ensure_ascii=False)

        # --- Run THP encoding ---
        encoder = THPEncoding(
            data=str(temp_path),
            **self.encoder_kwargs,
        )
        embeddings_by_key: Dict[str, torch.Tensor] = encoder.run_encoding()
        # embeddings_by_key maps JSON key → Tensor

        # Convert back to global-ID space
        result: Dict[int, torch.Tensor] = {}
        for gid in seq_ids:
            json_key = self.global_id_to_key.get(gid)
            if json_key is not None and json_key in embeddings_by_key:
                result[gid] = embeddings_by_key[json_key]

        return result

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(self, keep_temp: bool = False) -> Dict[str, Dict[int, torch.Tensor]]:
        """Run the full pipeline and save the result as a ``.pt`` file.

        Parameters
        ----------
        keep_temp : bool
            If ``False``, delete the temporary JSON files after encoding.

        Returns
        -------
        dict[str, dict[int, Tensor]]
            ``{node_position: {global_seq_id: embedding}}``.
        """
        # 1. Load inputs
        self.load_node_sequences()
        self.load_all_sequences()
        self.build_global_id_mapping()

        # 2. Prepare temp directory
        temp_dir = self.output_path.parent / "temp_node_inputs"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 3. Encode each node independently
        all_embeddings: Dict[str, Dict[int, torch.Tensor]] = {}

        for node_pos, seq_ids in self.node_sequences.items():
            n_seqs = len(seq_ids)
            print(f"\n{'=' * 60}")
            print(f"Node: {node_pos}  ({n_seqs} sequence{'s' if n_seqs != 1 else ''})")
            print(f"{'=' * 60}")

            node_emb = self._encode_one_node(node_pos, seq_ids, temp_dir)
            all_embeddings[node_pos] = node_emb

            # Quick sanity check
            print(f"  → encoded {len(node_emb)} / {n_seqs} sequences")

        # 4. Save
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_embeddings, self.output_path)

        total_embeddings = sum(len(v) for v in all_embeddings.values())
        print(f"\n{'=' * 60}")
        print(f"Saved {len(all_embeddings)} nodes, {total_embeddings} embeddings")
        print(f"Output: {self.output_path}")
        print(f"{'=' * 60}")

        # 5. Cleanup
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Cleaned up temp directory: {temp_dir}")

        return all_embeddings


# ====================================================================
# Example / main
# ====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-node THP encoding for a single tree."
    )
    parser.add_argument(
        "--csv", type=str,
        default="/Volumes/shenzm/Shuang_RA/Data/tree_8/tree_node_sequences.csv",
        help="Path to tree_node_sequences.csv",
    )
    parser.add_argument(
        "--json", type=str,
        default="/Volumes/shenzm/Shuang_RA/Data/tree_8/8Cluster/THP_8.json",
        help="Path to THP JSON file",
    )
    parser.add_argument(
        "--output", type=str,
        default="/Volumes/shenzm/Shuang_RA/Data/tree_8/node_embeddings.pt",
        help="Output .pt file path",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-rnn", type=int, default=256)
    parser.add_argument("--d-inner-hid", type=int, default=128)
    parser.add_argument("--d-k", type=int, default=16)
    parser.add_argument("--d-v", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained THP checkpoint (recommended)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda / cpu), auto-detect if omitted")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary per-node JSON files")

    args = parser.parse_args()

    encoder = NodeSequenceEncoder(
        csv_path=args.csv,
        json_path=args.json,
        output_path=args.output,
        batch_size=args.batch_size,
        d_model=args.d_model,
        d_rnn=args.d_rnn,
        d_inner_hid=args.d_inner_hid,
        d_k=args.d_k,
        d_v=args.d_v,
        n_head=args.n_head,
        n_layers=args.n_layers,
        dropout=args.dropout,
        checkpoint=args.checkpoint,
        device=args.device,
    )

    result = encoder.run(keep_temp=args.keep_temp)
