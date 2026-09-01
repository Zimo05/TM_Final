"""Strict split-manifest validation shared by upstream encoder stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_strict_manifest(
    manifest_path: str | Path,
    *,
    data_path: str | Path,
    available_source_ids: Iterable[int],
) -> Dict[str, Any]:
    """Load a manifest and prove it exactly covers the upstream source IDs."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError(
            f"split manifest must contain exactly {sorted(required)}"
        )
    split_sets = {
        name: set(map(int, values)) for name, values in splits.items()
    }
    names = sorted(split_sets)
    for position, left in enumerate(names):
        for right in names[position + 1:]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(
                    f"split manifest contains {len(overlap)} overlapping source IDs "
                    f"between {left} and {right}"
                )
    actual_data_sha = file_sha256(data_path)
    if manifest.get("data_sha256") != actual_data_sha:
        raise ValueError("split manifest data SHA-256 does not match source data")
    available = set(map(int, available_source_ids))
    declared = set().union(*split_sets.values())
    if declared != available:
        missing = sorted(declared - available)
        undeclared = sorted(available - declared)
        raise ValueError(
            "split/source ID mapping is not one-to-one: "
            f"missing_in_upstream={missing[:10]}, undeclared_upstream={undeclared[:10]}"
        )
    normalized = dict(manifest)
    normalized["splits"] = {
        name: sorted(split_sets[name]) for name in ("train", "validation", "test")
    }
    normalized["manifest_path"] = str(manifest_path.resolve())
    normalized["manifest_sha256"] = file_sha256(manifest_path)
    return normalized


def build_data_provenance(
    manifest: Mapping[str, Any],
    *,
    thp_checkpoint: str | Path | None = None,
    attention_weights: str | Path | None = None,
) -> Dict[str, Any]:
    splits = manifest["splits"]
    provenance: Dict[str, Any] = {
        "evaluation_regime": "strict_inductive",
        "data_sha256": manifest["data_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_path": manifest["manifest_path"],
        "seed": int(manifest.get("seed", 42)),
        "train_source_ids": list(map(int, splits["train"])),
        "validation_source_ids": list(map(int, splits["validation"])),
        "test_source_ids": list(map(int, splits["test"])),
        "node_pool": "train_only",
    }
    if thp_checkpoint is not None:
        provenance["thp_checkpoint_sha256"] = file_sha256(thp_checkpoint)
    if attention_weights is not None:
        provenance["attention_weights_sha256"] = file_sha256(attention_weights)
    return provenance
