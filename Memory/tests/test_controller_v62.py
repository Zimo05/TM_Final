import copy
import unittest

import torch

from CalibrateWriteRollout import paired_policy_improvement
from Evaluate import _write_ranking_metrics
from HawkesBackbone import HawkesFamily
from LatentHawkesTree import HawkesTree
from Train.Train import CausalPrefixEncoder, MemoryTreeTrainer, TrainingConfig
from Wake.ControllerUtilityReplay import ControllerUtilityReplay
from Wake.SequentialController import Controller


def replay_row(source, event, utility, gate, propensity=1.0):
    values = torch.zeros(4)
    values[2] = float(utility)
    targets = torch.zeros(4)
    mask = torch.zeros(4, dtype=torch.bool)
    mask[2] = True
    propensities = torch.ones(4)
    propensities[2] = float(propensity)
    gates = torch.zeros(4)
    gates[2] = float(gate)
    return {
        "inputs": torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.0, 0.0]),
        "utility": values,
        "target": targets,
        "label_mask": mask,
        "propensity": propensities,
        "gate": gates,
        "source_index": source,
        "event_index": event,
        "owner_id": "root",
        "group_id": source,
        "raw_write_utility": float(utility),
        "probe_propensity": float(propensity),
        "probe_top": propensity == 1.0,
    }


class ControllerV62Tests(unittest.TestCase):
    def test_sequence_relative_labels_and_pairs_never_cross_groups(self):
        replay = ControllerUtilityReplay(seed=7)
        replay.write_ranking_enabled = True
        for source in (10, 20):
            for event, utility in enumerate((-1.0, 0.0, 1.0, 2.0)):
                replay.add(replay_row(source, event, utility, event / 4), 2)
            replay.finalize_write_group(source)
        groups = replay.write_uniform_groups
        self.assertEqual({group["group_id"] for group in groups}, {10, 20})
        first = next(group for group in groups if group["group_id"] == 10)
        self.assertAlmostEqual(first["rows"][0]["group_median"], 0.5)
        self.assertAlmostEqual(first["rows"][0]["group_iqr"], 1.5)
        self.assertAlmostEqual(first["rows"][0]["write_advantage"], -1.5)
        sampled = replay.sample_write_ranking(max_rows=96, max_pairs=192)
        for high, low, _, _ in sampled["pairs"]:
            self.assertEqual(
                sampled["rows"][high]["group_id"],
                sampled["rows"][low]["group_id"],
            )

    def test_write_ranking_loss_only_has_write_head_gradients(self):
        controller = Controller(HawkesFamily(2, 1))
        inputs = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.0, 0.0],
            [0.8, 0.7, 0.2, 0.6, 0.4, 0.2, 0.0, 0.0],
        ])
        output = controller.action_distribution_batch(
            inputs[:, 0], inputs[:, 1], inputs[:, 2],
            update_statistics=False,
            owner_confidence=inputs[:, 3], retrieval_similarity=inputs[:, 4],
            retrieval_residual_norm=inputs[:, 5], working_memory_norm=inputs[:, 6],
            pending_write_ratio=inputs[:, 7],
        )
        rows = [
            {"relative_target": 0.1, "probe_propensity": 1.0,
             "raw_write_utility": -0.5},
            {"relative_target": 0.9, "probe_propensity": 0.1,
             "raw_write_utility": 1.0},
        ]
        loss, metrics = controller.write_ranking_loss(
            output, rows, [(1, 0, 1.0, 2.0)]
        )
        loss.backward()
        self.assertGreater(metrics["pair_count"], 0)
        self.assertIsNotNone(controller.bias_memorize.grad)
        self.assertEqual(float(controller.bias_retrieve.grad), 0.0)
        self.assertEqual(float(controller.bias_assimilate.grad), 0.0)
        gradient = controller.context_gate.weight.grad
        self.assertGreater(float(gradient[2].abs().sum()), 0.0)
        self.assertEqual(float(gradient[[0, 1, 3]].abs().sum()), 0.0)

    def test_policy_improvement_requires_positive_effect_and_nonnegative_ci(self):
        baseline = []
        candidate = []
        for index in range(20):
            common = {
                "source_index": index // 10,
                "event_index": index % 10,
            }
            baseline.append({**common, "nll": 1.0})
            candidate.append({**common, "nll": 0.9})
        result = paired_policy_improvement(baseline, candidate, seed=42)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["mean_nll_gain"], 0.1)
        zero = paired_policy_improvement(baseline, copy.deepcopy(baseline), seed=42)
        self.assertFalse(zero["passed"])

    def test_ranking_metrics_report_topk_and_pair_accuracy(self):
        rows = []
        for source in (0, 1):
            for event, utility in enumerate((-1.0, 0.0, 1.0, 2.0)):
                rows.append({
                    "source_index": source,
                    "event_index": event,
                    "write_utility": utility,
                    "raw_action_probabilities": [0.0, 0.0, event / 4, 0.0],
                    "action_probabilities": [0.0, 0.0, event / 4, 0.0],
                })
        metrics = _write_ranking_metrics(rows, seed=42)
        self.assertEqual(metrics["group_count"], 2)
        self.assertEqual(metrics["pairwise_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["ndcg_at_4"], 1.0)
        self.assertGreater(metrics["top4_cumulative_utility"], 0.0)

    def test_one_epoch_write_ranking_training_builds_grouped_replay(self):
        training = TrainingConfig(
            epochs=1, plot_after_training=False,
            controller_write_ranking=True,
        )
        trainer = MemoryTreeTrainer(
            HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3),
            HawkesFamily(2, 1),
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            training=training,
            device="cpu",
        )
        trainer.prepare_controller_only_finetune(
            base_checkpoint="missing_base.pt", target_version=6,
            train_heads=("write",),
        )
        sequence = {
            "times": torch.arange(1, 13, dtype=torch.float32) * 0.1,
            "types": torch.tensor([0, 1] * 6),
            "source_index": 7,
            "cluster_id": 0,
        }
        history = trainer.train([sequence], verbose=False)
        self.assertEqual(len(history), 1)
        self.assertTrue(trainer.controller_utility_replay.write_ranking_enabled)
        self.assertTrue(trainer.controller_utility_replay.write_uniform_groups)
        grouped_rows = trainer.controller_utility_replay.rows(action=2)
        self.assertTrue(grouped_rows)
        self.assertTrue(all(row["group_id"] == 7 for row in grouped_rows))


if __name__ == "__main__":
    unittest.main()
