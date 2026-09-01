import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import json
import hashlib

from Evaluate import (
    ablation_metrics,
    aggregate_metrics,
    load_dataset,
    make_split,
    validate_evaluation_provenance,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_split_is_sequence_level_and_deterministic(self):
        first = make_split(20, seed=7, validation_ratio=0.2, test_ratio=0.2)
        second = make_split(20, seed=7, validation_ratio=0.2, test_ratio=0.2)
        self.assertEqual(first, second)
        all_indices = first["train"] + first["validation"] + first["test"]
        self.assertEqual(sorted(all_indices), list(range(20)))
        self.assertEqual(len(set(all_indices)), 20)

    def test_prediction_metrics_include_accuracy_error_and_time(self):
        rows = [
            {
                "source_index": 0,
                "event_index": 0,
                "true_type": 0,
                "type_probabilities": [0.8, 0.2],
                "nll": 1.0,
                "predicted_time": 1.5,
                "true_time": 1.0,
            },
            {
                "source_index": 1,
                "event_index": 0,
                "true_type": 1,
                "type_probabilities": [0.7, 0.3],
                "nll": 3.0,
                "predicted_time": 3.0,
                "true_time": 2.0,
            },
        ]
        metrics = aggregate_metrics(rows, num_types=2, seed=0)
        self.assertAlmostEqual(metrics["nll_per_event"], 2.0)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["error_rate"], 0.5)
        self.assertAlmostEqual(metrics["local_time_mae"], 0.75)
        self.assertTrue(math.isfinite(metrics["ece_10bin"]))

    def test_memory_diagnostics_and_write_funnel_are_aggregated(self):
        row = {
            "source_index": 0, "event_index": 0, "true_type": 0,
            "type_probabilities": [0.8, 0.2], "nll": 1.0,
            "predicted_time": 1.0, "true_time": 1.0,
            "visited_bank_count": 2, "visited_nonempty_bank_count": 1,
            "owner_on_retrieval_path": True, "retrieval_alpha_mass": 0.5,
            "retrieval_alpha_per_visited_node": 0.25,
            "raw_episodic_residual_norm": 2.0,
            "gated_episodic_residual_norm": 0.5, "retrieve_gate": 0.25,
            "raw_action_probabilities": [0.0, 0.0, 0.9, 0.0],
            "memorize_argmax": True, "write_candidate": True,
            "write_gate_passed": True, "write_priority_passed": True,
            "write_window_complete": True, "write_accepted": True,
            "write_retrieved_later": True, "write_beneficial": True,
        }
        metrics = aggregate_metrics([row], num_types=2, seed=0)
        self.assertEqual(metrics["read_coverage_fraction"], 1.0)
        self.assertAlmostEqual(metrics["raw_to_gated_residual_ratio"], 0.25)
        self.assertEqual(metrics["write_funnel"]["write_accepted_count"], 1)
        self.assertEqual(metrics["write_funnel"]["accepted_reuse_fraction"], 1.0)

    def test_strict_provenance_rejects_train_test_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "split.json"
            manifest = {
                "data_sha256": "data", "seed": 42,
                "splits": {"train": [0], "validation": [1], "test": [2]},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            checkpoint = {"data_provenance": {
                "evaluation_regime": "strict_inductive",
                "data_sha256": "data", "manifest_sha256": manifest_sha,
                "train_source_ids": [0, 2], "validation_source_ids": [1],
                "test_source_ids": [2], "node_pool": "train_only",
            }}
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_evaluation_provenance(
                    checkpoint, manifest, [2], manifest_path
                )

    def test_ablation_gain_sign_is_baseline_minus_target(self):
        common = {"source_index": 0, "event_index": 0}
        rows = {
            "full_frozen": [{**common, "nll": 1.0}],
            "no_episodic": [{**common, "nll": 1.4}],
            "no_working": [{**common, "nll": 1.2}],
            "semantic_only": [{**common, "nll": 1.8}],
            "full_online": [{**common, "nll": 0.9}],
        }
        output = {item["comparison"]: item for item in ablation_metrics(rows, 0)}
        self.assertAlmostEqual(output["episodic_gain"]["mean_nll_gain"], 0.4)
        self.assertAlmostEqual(output["total_memory_gain"]["mean_nll_gain"], 0.8)
        self.assertAlmostEqual(output["online_vs_frozen_gain"]["mean_nll_gain"], 0.1)

    def test_csv_loader_maps_sparse_event_types_and_cluster(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            pd.DataFrame({
                "event_times": ["0.1,0.3", "0.2,0.8"],
                "event_types": ["10,20", "20,10"],
                "cluster": [3, 4],
            }).to_csv(path, index=False)
            dataset, mapping = load_dataset(path)
        self.assertEqual(mapping, {10: 0, 20: 1})
        self.assertEqual(dataset[0]["types"].tolist(), [0, 1])
        self.assertEqual(dataset[1]["cluster_id"], 4)


if __name__ == "__main__":
    unittest.main()
