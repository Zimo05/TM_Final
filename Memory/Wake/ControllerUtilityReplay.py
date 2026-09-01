"""Stratified, reproducible replay for Controller counterfactual utilities."""

from __future__ import annotations

import random
from copy import deepcopy
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch


ACTION_NAMES = ("adapt", "retrieve", "write", "split")
DEFAULT_CAPACITIES = (1024, 1024, 1536, 512)
DEFAULT_BATCH_SIZES = (64, 64, 96, 32)


def _cpu_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if torch.is_tensor(value):
            detached = value.detach()
            # Batched Wake already transfers the complete replay payload to
            # CPU. Avoid re-entering the device-transfer path for every field
            # when the reservoir stores those rows one at a time.
            result[key] = (
                detached.clone()
                if detached.device.type == "cpu"
                else detached.cpu().clone()
            )
        else:
            result[key] = deepcopy(value)
    return result


def _cpu_group(group: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a grouped replay entry while normalizing its nested rows."""
    result = _cpu_row(group)
    result["rows"] = [_cpu_row(row) for row in group.get("rows", ())]
    return result


class ControllerUtilityReplay:
    """Half uniform reservoir, half hard-example replay, stratified by sign."""

    def __init__(
        self,
        capacities: Sequence[int] = DEFAULT_CAPACITIES,
        *,
        seed: int = 42,
    ) -> None:
        if len(capacities) != 4 or any(int(value) <= 0 for value in capacities):
            raise ValueError("capacities must contain four positive values")
        self.capacities = tuple(map(int, capacities))
        self.rng = random.Random(int(seed))
        self.uniform = {(a, s): [] for a in range(4) for s in (0, 1)}
        self.hard = {(a, s): [] for a in range(4) for s in (0, 1)}
        self.seen = {(a, s): 0 for a in range(4) for s in (0, 1)}
        self.write_ranking_enabled = False
        self.write_pending: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.write_uniform_groups: list[dict[str, Any]] = []
        self.write_hard_groups: list[dict[str, Any]] = []
        self.write_groups_seen = 0

    def __len__(self) -> int:
        base = sum(len(rows) for rows in self.uniform.values()) + sum(
            len(rows) for rows in self.hard.values()
        )
        if not self.write_ranking_enabled:
            return base
        return base + sum(
            len(group["rows"])
            for group in [*self.write_uniform_groups, *self.write_hard_groups]
        )

    def _bucket_capacity(self, action: int, hard: bool) -> int:
        half = self.capacities[action] // 2
        return max(1, half // 2)  # half storage mode, then positive/negative

    def add(self, row: Mapping[str, Any], action: int) -> None:
        action = int(action)
        if action == 2 and self.write_ranking_enabled:
            stored = _cpu_row(row)
            group_id = int(stored.get("group_id", stored.get("source_index", -1)))
            stored["group_id"] = group_id
            stored["raw_write_utility"] = float(
                torch.as_tensor(stored.get("raw_write_utility", stored["utility"][2]))
            )
            stored["probe_propensity"] = float(
                stored.get("probe_propensity", torch.as_tensor(stored["propensity"])[2])
            )
            stored["probe_top"] = bool(stored.get("probe_top", False))
            self.write_pending[group_id].append(stored)
            return
        utility = float(torch.as_tensor(row["utility"])[action])
        sign = int(utility > 0.0)
        key = (action, sign)
        stored = _cpu_row(row)
        self.seen[key] += 1
        uniform_capacity = self._bucket_capacity(action, False)
        uniform = self.uniform[key]
        if len(uniform) < uniform_capacity:
            uniform.append(stored)
        else:
            position = self.rng.randrange(self.seen[key])
            if position < uniform_capacity:
                uniform[position] = stored

        hard_capacity = self._bucket_capacity(action, True)
        hard = self.hard[key]
        gate = float(torch.as_tensor(row["gate"])[action])
        target = float(torch.as_tensor(row["target"])[action])
        score = abs(gate - target) * max(abs(utility), 1e-12)
        stored["hard_score"] = score
        hard.append(stored)
        hard.sort(key=lambda item: float(item["hard_score"]), reverse=True)
        del hard[hard_capacity:]

    @staticmethod
    def _decorate_write_group(
        group_id: int, rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        copied = [_cpu_row(row) for row in rows]
        utilities = torch.tensor(
            [float(row["raw_write_utility"]) for row in copied],
            dtype=torch.float64,
        )
        median = float(torch.quantile(utilities, 0.5))
        q25, q75 = torch.quantile(
            utilities, torch.tensor([0.25, 0.75], dtype=utilities.dtype)
        ).tolist()
        iqr = float(q75 - q25)
        temperature = min(1.0, max(1e-4, iqr / 1.349))
        order = sorted(
            range(len(copied)), key=lambda index: (utilities[index].item(), index)
        )
        quantiles = [0.5] * len(copied)
        if len(copied) > 1:
            for rank, index in enumerate(order):
                quantiles[index] = rank / (len(copied) - 1)
        for index, row in enumerate(copied):
            advantage = float(utilities[index]) - median
            row.update({
                "group_id": int(group_id),
                "raw_write_utility": float(utilities[index]),
                "write_advantage": advantage,
                "group_median": median,
                "group_iqr": iqr,
                "rank_quantile": float(quantiles[index]),
                "relative_target": float(torch.sigmoid(torch.tensor(
                    advantage / temperature, dtype=torch.float64
                ))),
            })
        true_best = max(range(len(copied)), key=lambda index: float(utilities[index]))
        predicted_best = max(
            range(len(copied)),
            key=lambda index: float(torch.as_tensor(copied[index]["gate"])[2]),
        )
        regret = max(
            0.0, float(utilities[true_best] - utilities[predicted_best])
        )
        harmful = max(
            (
                -float(utilities[index])
                * max(float(torch.as_tensor(row["gate"])[2]) - 0.5, 0.0)
                for index, row in enumerate(copied)
                if float(utilities[index]) <= 0.0
            ),
            default=0.0,
        )
        return {
            "group_id": int(group_id),
            "rows": copied,
            "hard_score": regret + harmful,
        }

    def finalize_write_group(self, group_id: int) -> None:
        """Commit one complete sequence of Write probes to grouped replay."""
        if not self.write_ranking_enabled:
            return
        group_id = int(group_id)
        rows = self.write_pending.pop(group_id, [])
        if not rows:
            return
        group = self._decorate_write_group(group_id, rows)
        group_capacity = max(1, self.capacities[2] // (2 * 32))
        self.write_groups_seen += 1
        if len(self.write_uniform_groups) < group_capacity:
            self.write_uniform_groups.append(group)
        else:
            position = self.rng.randrange(self.write_groups_seen)
            if position < group_capacity:
                self.write_uniform_groups[position] = group
        self.write_hard_groups.append(deepcopy(group))
        self.write_hard_groups.sort(
            key=lambda item: float(item["hard_score"]), reverse=True
        )
        del self.write_hard_groups[group_capacity:]

    def sample_write_ranking(
        self, *, max_rows: int = 96, max_pairs: int = 192
    ) -> dict[str, Any]:
        """Sample complete groups and deterministic upper/lower-quartile pairs."""
        if not self.write_ranking_enabled:
            return {"rows": [], "pairs": []}
        by_id = {
            int(group["group_id"]): group
            for group in [*self.write_uniform_groups, *self.write_hard_groups]
        }
        groups = list(by_id.values())
        self.rng.shuffle(groups)
        selected: list[dict[str, Any]] = []
        row_count = 0
        for group in groups:
            size = len(group["rows"])
            if selected and row_count + size > int(max_rows):
                continue
            selected.append(group)
            row_count += size
            if row_count >= int(max_rows):
                break
        rows: list[dict[str, Any]] = []
        pairs: list[tuple[int, int, float, float]] = []
        for group in selected:
            offset = len(rows)
            current = group["rows"]
            rows.extend(current)
            if len(current) < 4:
                continue
            high = [i for i, row in enumerate(current) if row["rank_quantile"] >= 0.75]
            low = [i for i, row in enumerate(current) if row["rank_quantile"] <= 0.25]
            iqr = float(current[0]["group_iqr"])
            minimum_gap = max(1e-4, 0.1 * iqr)
            candidates = []
            for high_index in high:
                for low_index in low:
                    gap = (
                        float(current[high_index]["raw_write_utility"])
                        - float(current[low_index]["raw_write_utility"])
                    )
                    if gap <= minimum_gap:
                        continue
                    high_ipw = min(10.0, max(1.0, 1.0 / max(
                        float(current[high_index]["probe_propensity"]), 1e-6
                    )))
                    low_ipw = min(10.0, max(1.0, 1.0 / max(
                        float(current[low_index]["probe_propensity"]), 1e-6
                    )))
                    gap_weight = min(4.0, max(0.25, gap / (iqr + 1e-6)))
                    candidates.append((
                        offset + high_index,
                        offset + low_index,
                        gap_weight,
                        (high_ipw * low_ipw) ** 0.5,
                    ))
            self.rng.shuffle(candidates)
            pairs.extend(candidates[:64])
        if len(pairs) > int(max_pairs):
            pairs = self.rng.sample(pairs, int(max_pairs))
        return {"rows": rows, "pairs": pairs}

    def rows(self, action: int | None = None) -> list[dict[str, Any]]:
        actions = range(4) if action is None else (int(action),)
        output = [
            row
            for current in actions
            if not (current == 2 and self.write_ranking_enabled)
            for sign in (0, 1)
            for store in (self.uniform, self.hard)
            for row in store[(current, sign)]
        ]
        if self.write_ranking_enabled and (action is None or int(action) == 2):
            output.extend(
                row
                for group in [*self.write_uniform_groups, *self.write_hard_groups]
                for row in group["rows"]
            )
        return output

    def _sample_bucket(self, rows: list[dict[str, Any]], count: int) -> list:
        if not rows or count <= 0:
            return []
        if len(rows) >= count:
            return self.rng.sample(rows, count)
        return [rows[self.rng.randrange(len(rows))] for _ in range(count)]

    def sample(
        self, batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES
    ) -> list[dict[str, Any]]:
        if len(batch_sizes) != 4:
            raise ValueError("batch_sizes must contain four values")
        output = []
        for action, requested in enumerate(map(int, batch_sizes)):
            if action == 2 and self.write_ranking_enabled:
                continue
            positive = [
                *self.uniform[(action, 1)], *self.hard[(action, 1)]
            ]
            negative = [
                *self.uniform[(action, 0)], *self.hard[(action, 0)]
            ]
            if positive and negative:
                negative_count = requested // 2
                positive_count = requested - negative_count
            elif positive:
                positive_count, negative_count = requested, 0
            elif negative:
                positive_count, negative_count = 0, requested
            else:
                continue
            output.extend(self._sample_bucket(positive, positive_count))
            output.extend(self._sample_bucket(negative, negative_count))
        self.rng.shuffle(output)
        return output

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "capacities": self.capacities,
            "uniform": self.uniform,
            "hard": self.hard,
            "seen": self.seen,
            "random_state": self.rng.getstate(),
            "write_ranking_enabled": self.write_ranking_enabled,
            "write_pending": dict(self.write_pending),
            "write_uniform_groups": self.write_uniform_groups,
            "write_hard_groups": self.write_hard_groups,
            "write_groups_seen": self.write_groups_seen,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.capacities = tuple(map(int, state["capacities"]))
        # Checkpoints are loaded with the trainer's ``map_location``.  That
        # can place previously CPU-only replay rows on CUDA, while rows added
        # after resume are normalized to CPU by ``_cpu_row``.  Keep the replay
        # stores CPU-resident regardless of checkpoint load location.
        self.uniform = {
            key: [_cpu_row(row) for row in rows]
            for key, rows in state["uniform"].items()
        }
        self.hard = {
            key: [_cpu_row(row) for row in rows]
            for key, rows in state["hard"].items()
        }
        self.seen = dict(state["seen"])
        self.write_ranking_enabled = bool(state.get("write_ranking_enabled", False))
        self.write_pending = defaultdict(
            list,
            {
                key: [_cpu_row(row) for row in rows]
                for key, rows in state.get("write_pending", {}).items()
            },
        )
        self.write_uniform_groups = [
            _cpu_group(group)
            for group in state.get("write_uniform_groups", ())
        ]
        self.write_hard_groups = [
            _cpu_group(group)
            for group in state.get("write_hard_groups", ())
        ]
        self.write_groups_seen = int(state.get("write_groups_seen", 0))
        self.rng.setstate(state["random_state"])

    def sign_counts(self) -> dict[str, dict[str, int]]:
        result = {
            ACTION_NAMES[action]: {
                "negative": len(self.uniform[(action, 0)]) + len(self.hard[(action, 0)]),
                "positive": len(self.uniform[(action, 1)]) + len(self.hard[(action, 1)]),
            }
            for action in range(4)
        }
        if self.write_ranking_enabled:
            write_rows = [
                row
                for group in [*self.write_uniform_groups, *self.write_hard_groups]
                for row in group["rows"]
            ]
            result["write"] = {
                "negative": sum(row["raw_write_utility"] <= 0.0 for row in write_rows),
                "positive": sum(row["raw_write_utility"] > 0.0 for row in write_rows),
            }
        return result
