"""Reproducible cluster-stratified sequence splits for Memory training/eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_stratified_manifest(
    data_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    train_per_cluster: int = 70,
    validation_per_cluster: int = 10,
    test_per_cluster: int = 20,
) -> dict[str, Any]:
    data_path, output_path = Path(data_path), Path(output_path)
    frame = pd.read_csv(data_path)
    if "cluster" not in frame.columns:
        raise ValueError("stratified split requires a 'cluster' column")
    groups: dict[int, list[int]] = defaultdict(list)
    for source_index, cluster in frame["cluster"].items():
        if pd.isna(cluster):
            raise ValueError(f"missing cluster at source row {source_index}")
        groups[int(cluster)].append(int(source_index))
    required = train_per_cluster + validation_per_cluster + test_per_cluster
    splits = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for cluster in sorted(groups):
        values = list(groups[cluster])
        if len(values) != required:
            raise ValueError(
                f"cluster {cluster} has {len(values)} rows; expected exactly {required}"
            )
        rng.shuffle(values)
        splits["train"].extend(values[:train_per_cluster])
        splits["validation"].extend(
            values[train_per_cluster:train_per_cluster + validation_per_cluster]
        )
        splits["test"].extend(values[-test_per_cluster:])
    for values in splits.values():
        values.sort()
    cluster_by_source = {int(i): int(c) for i, c in frame["cluster"].items()}
    manifest: dict[str, Any] = {
        "format_version": 1,
        "seed": int(seed),
        "data_path": str(data_path.resolve()),
        "data_sha256": file_sha256(data_path),
        "source_row_count": int(len(frame)),
        "counts": {name: len(values) for name, values in splits.items()},
        "cluster_distribution": {
            name: dict(sorted(Counter(cluster_by_source[i] for i in values).items()))
            for name, values in splits.items()
        },
        "splits": splits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split_manifest(
    manifest_path: str | Path,
    *,
    data_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError(f"split manifest must contain exactly {sorted(required)}")
    sets = {name: set(map(int, values)) for name, values in splits.items()}
    if any(sets[a] & sets[b] for a in sets for b in sets if a < b):
        raise ValueError("split manifest contains overlapping source indices")
    if data_path is not None and manifest.get("data_sha256") != file_sha256(data_path):
        raise ValueError("split manifest data SHA-256 does not match --data-path")
    return manifest


def select_sequences(
    dataset: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], split: str
) -> list[Mapping[str, Any]]:
    if split not in manifest["splits"]:
        raise ValueError(f"unknown split {split!r}")
    wanted = set(map(int, manifest["splits"][split]))
    selected = []
    for sequence in dataset:
        value = sequence["source_index"]
        source_index = int(value.item()) if hasattr(value, "item") else int(value)
        if source_index in wanted:
            selected.append(sequence)
    found = {int(s["source_index"].item()) if hasattr(s["source_index"], "item") else int(s["source_index"]) for s in selected}
    missing = wanted - found
    if missing:
        raise ValueError(f"dataset is missing {len(missing)} manifest source rows")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed stratified Memory split")
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = create_stratified_manifest(args.data_path, args.output, seed=args.seed)
    print(json.dumps({"output": str(args.output.resolve()), "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
