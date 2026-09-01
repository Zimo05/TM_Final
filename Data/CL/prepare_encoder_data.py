#!/usr/bin/env python3
"""Prepare the initial CL task for the Multi-Attention Encoder.

The CL benchmark stores one CSV per task, while MultiAttentionEncoder uses
the THP JSON plus the tree-membership CSV used by the DWS pipeline.  Task 0 is
the only data used here: it is the model's initial observation and therefore
does not leak any later task or oracle ground-truth information.

The generated tree intentionally contains one root leaf.  Memory can expand
that leaf online as later CL tasks arrive; the static root embedding still
comes from the complete Multi-Attention Encoder path.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any


def _parse_list(raw: Any, cast: type) -> list[Any]:
    """Parse the JSON/Python-list representation used by CL CSV files."""

    try:
        value = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"cannot parse sequence value {raw!r}") from error
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"sequence value must be a list, got {type(value).__name__}")
    return [cast(item) for item in value]


def _load_task(path: Path, expected_types: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"CL task CSV does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"event_times", "event_types"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader):
            times = _parse_list(row["event_times"], float)
            types = _parse_list(row["event_types"], int)
            if not times or len(times) != len(types):
                raise ValueError(
                    f"{path}:{row_number + 2} has mismatched or empty events"
                )
            if any(not math.isfinite(time) for time in times):
                raise ValueError(f"{path}:{row_number + 2} contains non-finite times")
            if times[0] < 0.0 or any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError(
                    f"{path}:{row_number + 2} event_times must be strictly increasing"
                )
            if any(event_type < 0 or event_type >= expected_types for event_type in types):
                raise ValueError(
                    f"{path}:{row_number + 2} contains an event type outside "
                    f"[0, {expected_types - 1}]"
                )
            rows.append({"times": times, "types": types})

    if len(rows) < 3:
        raise ValueError("MultiAttentionEncoder needs at least three task-00 sequences")
    observed = {event_type for row in rows for event_type in row["types"]}
    missing_types = set(range(expected_types)).difference(observed)
    if missing_types:
        raise ValueError(
            "task_00/train.csv does not expose every configured event type; "
            f"missing {sorted(missing_types)}"
        )
    return rows


def _write_thp_json(rows: list[dict[str, Any]], path: Path) -> None:
    streams: dict[str, list[dict[str, float | int]]] = {}
    for index, row in enumerate(rows):
        times = row["times"]
        event_types = row["types"]
        stream = []
        previous = 0.0
        for time, event_type in zip(times, event_types):
            stream.append(
                {
                    "time_since_start": float(time),
                    "time_since_last_event": float(time - previous),
                    "type_event": int(event_type),
                }
            )
            previous = time
        # AttenEncoderMain_v1 sorts keys by the two numeric components.  This
        # key layout keeps JSON order and tree global IDs unambiguous.
        streams[f"0_{index:06d}"] = stream

    path.write_text(
        json.dumps(streams, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_tree_files(
    rows: list[dict[str, Any]],
    tree_path: Path,
    summary_path: Path,
    expected_types: int,
) -> None:
    sequence_ids = list(range(len(rows)))
    encoded_ids = json.dumps(sequence_ids, separators=(",", ":"))

    with tree_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_position", "sequences"])
        writer.writeheader()
        writer.writerow({"node_position": "root", "sequences": encoded_ids})

    # These are empirical, model-facing features, not oracle parameters.  The
    # CL benchmark's ground_truth directory is deliberately never read.
    exposure = sum(max(float(row["times"][-1]), 1e-8) for row in rows)
    counts = [
        sum(row["types"].count(event_type) for row in rows)
        for event_type in range(expected_types)
    ]
    mu = [count / exposure for count in counts]
    excitation = [[0.0 for _ in range(expected_types)] for _ in range(expected_types)]

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["leaf_position", "cluster_id", "mu", "A", "decay", "sequences"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "leaf_position": "root",
                "cluster_id": 0,
                "mu": json.dumps(mu, separators=(",", ":")),
                "A": json.dumps(excitation, separators=(",", ":")),
                "decay": 1.0,
                "sequences": encoded_ids,
            }
        )


def prepare(input_csv: Path, output_dir: Path, expected_types: int) -> None:
    rows = _load_task(input_csv, expected_types)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_thp_json(rows, output_dir / "task_00_train.json")
    _write_tree_files(
        rows,
        output_dir / "tree_node_sequences.csv",
        output_dir / "sequence_summary.csv",
        expected_types,
    )
    manifest = {
        "source_csv": str(input_csv.resolve()),
        "sequence_count": len(rows),
        "event_count": sum(len(row["times"]) for row in rows),
        "num_event_types": expected_types,
        "tree": "root_only_initialization",
        "oracle_files_used": False,
    }
    (output_dir / "encoder_input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[CL encoder input] sequences={manifest['sequence_count']} "
        f"events={manifest['event_count']} types={expected_types}"
    )
    print(f"[CL encoder input] output={output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CL task_00 train CSV to MultiAttentionEncoder inputs."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-event-types", type=int, default=8)
    args = parser.parse_args()
    if args.num_event_types <= 0:
        parser.error("--num-event-types must be positive")
    prepare(args.input, args.output_dir, args.num_event_types)


if __name__ == "__main__":
    main()
