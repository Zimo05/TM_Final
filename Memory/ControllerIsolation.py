"""Stable hashes and policy partitions for isolated Controller updates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch


HEAD_PARAMETER_NAMES = {
    "adapt": ("bias_assimilate", "raw_assimilate_surprise"),
    "retrieve": (
        "bias_retrieve",
        "raw_retrieve_surprise",
        "raw_retrieve_novelty",
    ),
    "write": (
        "bias_memorize",
        "raw_memorize_surprise",
        "raw_memorize_novelty",
        "raw_memorize_count",
    ),
    "split": (
        "bias_queue_split",
        "raw_queue_surprise",
        "raw_queue_novelty",
        "raw_queue_count",
    ),
}

HEAD_ROW = {"adapt": 0, "retrieve": 1, "write": 2, "split": 3}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tensor_items_sha256(items: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(items):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(items[name]).encode("ascii"))
    return digest.hexdigest()


def controller_item_sha256(module_state: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: tensor_sha256(value)
        for name, value in sorted(module_state.items())
        if torch.is_tensor(value)
    }


def head_policy_items(
    module_state: Mapping[str, Any], head: str
) -> dict[str, torch.Tensor]:
    if head not in HEAD_ROW:
        raise ValueError(f"unknown Controller head: {head}")
    items = {
        name: torch.as_tensor(module_state[name])
        for name in HEAD_PARAMETER_NAMES[head]
        if name in module_state
    }
    context = torch.as_tensor(module_state["context_gate.weight"])
    items[f"context_gate.weight[{HEAD_ROW[head]}]"] = context[HEAD_ROW[head]]
    thresholds = torch.as_tensor(module_state["calibration_thresholds"])
    if head != "split":
        items[f"calibration_thresholds[{HEAD_ROW[head]}]"] = thresholds[
            HEAD_ROW[head]
        ]
    for buffer_name in (
        "utility_temperatures",
        "utility_mean",
        "utility_variance",
        "utility_observations",
    ):
        if buffer_name in module_state:
            values = torch.as_tensor(module_state[buffer_name])
            items[f"{buffer_name}[{HEAD_ROW[head]}]"] = values[HEAD_ROW[head]]
    if head == "split" and "split_enabled" in module_state:
        items["split_enabled"] = torch.as_tensor(module_state["split_enabled"])
    return items


def head_policy_sha256(module_state: Mapping[str, Any], head: str) -> str:
    return tensor_items_sha256(head_policy_items(module_state, head))


def protected_write_only_items(
    module_state: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Everything in Controller state that a Write-only update may not change."""
    protected: dict[str, torch.Tensor] = {}
    for head in ("adapt", "retrieve", "split"):
        for name, value in head_policy_items(module_state, head).items():
            protected[f"{head}.{name}"] = value
    return protected
