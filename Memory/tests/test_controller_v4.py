import tempfile
import unittest
from pathlib import Path

import torch

from HawkesBackbone import HawkesFamily
from Evaluate import controller_metrics
from LatentHawkesTree import HawkesTree
from Train.Train import (
    CausalPrefixEncoder, MemoryTreeTrainer, TrainingConfig, WakeObjectiveConfig,
)
from Wake.ControllerUtilityReplay import ControllerUtilityReplay
from Wake.SequentialController import Controller


def replay_row(action, utility, gate, event_index):
    values = torch.zeros(4)
    values[action] = utility
    targets = torch.zeros(4)
    targets[action] = float(utility > 0)
    mask = torch.zeros(4, dtype=torch.bool)
    mask[action] = True
    return {
        "inputs": torch.arange(8, dtype=torch.float32),
        "utility": values,
        "target": targets,
        "label_mask": mask,
        "propensity": torch.ones(4),
        "gate": torch.full((4,), gate),
        "cluster_id": 1,
        "source_index": 2,
        "event_index": event_index,
        "owner_id": "root",
    }


class ControllerV4Tests(unittest.TestCase):
    def setUp(self):
        self.controller = Controller(
            HawkesFamily(num_types=2, num_basis=1),
            write_candidate_threshold=0.6,
            utility_temperature=0.25,
            utility_cost_margin=0.01,
        )

    def test_masked_heads_do_not_supervise_each_other(self):
        output = self.controller.action_distribution_batch(
            torch.tensor([0.2]), torch.tensor([0.4]), torch.tensor([0.3]),
            update_statistics=False,
        )
        output["logits"].retain_grad()
        targets = torch.zeros(1, 4)
        targets[0, 1] = 1.0
        mask = torch.zeros(1, 4, dtype=torch.bool)
        mask[0, 1] = True
        utilities = torch.zeros(1, 4)
        utilities[0, 1] = 0.2
        self.controller.masked_utility_loss(
            output, targets, mask, utilities
        ).backward()
        gradient = output["logits"].grad[0]
        self.assertNotEqual(float(gradient[1]), 0.0)
        self.assertTrue(torch.equal(gradient[[0, 2, 3]], torch.zeros(3)))

    def test_ipw_is_clipped_and_normalized(self):
        propensity = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                                   [1.0, 0.02, 1.0, 1.0]])
        mask = torch.tensor([[False, True, False, False],
                             [False, True, False, False]])
        weights = self.controller.normalized_inverse_propensity(propensity, mask)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(weights[1] / weights[0]), 10.0, places=5)

    def test_write_admission_requires_all_four_conditions(self):
        admissible = self.controller.write_admissible
        self.assertTrue(admissible(0.6, 0.1, 0.2, future_window_complete=True))
        self.assertFalse(admissible(0.59, 0.1, 0.2, future_window_complete=True))
        self.assertFalse(admissible(0.9, 0.0, 0.2, future_window_complete=True))
        self.assertFalse(admissible(0.9, 0.1, 0.0, future_window_complete=True))
        self.assertFalse(admissible(0.9, 0.1, 0.2, future_window_complete=False))

    def test_action_temperatures_use_robust_iqr_and_bounds(self):
        before = self.controller.utility_temperatures.clone()
        self.controller.update_utility_temperatures([
            torch.tensor([-0.2, -0.1, 0.1, 0.2]),
            torch.empty(0), torch.empty(0), torch.empty(0),
        ], ema=0.0)
        self.assertGreaterEqual(float(self.controller.utility_temperatures[0]), 1e-4)
        self.assertLessEqual(float(self.controller.utility_temperatures[0]), 1.0)
        self.assertEqual(float(self.controller.utility_temperatures[1]), float(before[1]))


class ControllerUtilityReplayTests(unittest.TestCase):
    def test_capacity_sign_balance_and_seed_are_reproducible(self):
        capacities = (16, 16, 24, 8)
        first = ControllerUtilityReplay(capacities, seed=42)
        second = ControllerUtilityReplay(capacities, seed=42)
        for event_index in range(100):
            for action in range(4):
                utility = 1.0 if event_index % 2 else -1.0
                row = replay_row(action, utility, 1.0 - float(utility > 0), event_index)
                first.add(row, action)
                second.add(row, action)
        self.assertLessEqual(len(first), sum(capacities))
        self.assertEqual(first.sign_counts(), second.sign_counts())
        sample_a = first.sample((8, 8, 12, 4))
        sample_b = second.sample((8, 8, 12, 4))
        self.assertEqual(
            [row["event_index"] for row in sample_a],
            [row["event_index"] for row in sample_b],
        )
        state = first.state_dict()
        restored = ControllerUtilityReplay(capacities, seed=0)
        restored.load_state_dict(state)
        self.assertEqual(first.sign_counts(), restored.sign_counts())


class ControllerEvaluationTests(unittest.TestCase):
    def test_gate_utility_metrics_follow_action_specific_ablations(self):
        def row(index, nll, retrieve, adapt, write=0.0, accepted=False):
            return {
                "source_index": 1, "event_index": index, "nll": nll,
                "action_probabilities": [adapt, retrieve, write, 0.1],
                "write_utility": None, "write_accepted": accepted,
            }
        frozen = [row(0, 1.0, 0.9, 0.8), row(1, 1.0, 0.1, 0.2)]
        no_episodic = [row(0, 1.2, 0.9, 0.8), row(1, 0.8, 0.1, 0.2)]
        no_working = [row(0, 1.1, 0.9, 0.8), row(1, 0.9, 0.1, 0.2)]
        online = [row(0, 1.0, 0.9, 0.8, 0.8, True), row(1, 1.0, 0.1, 0.2, 0.2)]
        online[0]["write_utility"] = 0.3
        online[1]["write_utility"] = -0.2
        metrics = controller_metrics({
            "full_frozen": frozen, "no_episodic": no_episodic,
            "no_working": no_working, "full_online": online,
        }, {"full_frozen": {"elapsed_seconds": 1.0},
            "full_online": {"elapsed_seconds": 1.0}})
        self.assertGreater(metrics["retrieve"]["spearman_gate_utility"], 0.0)
        self.assertEqual(metrics["write"]["accepted_count"], 1)
        self.assertEqual(metrics["write"]["false_positive_harmful_loss"], 0)

    def test_validation_is_isolated_and_writes_best_last_v4(self):
        sequence = {
            "times": torch.tensor([0.1, 0.3, 0.8]),
            "types": torch.tensor([0, 1, 0]),
            "source_index": 0,
            "cluster_id": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last, best = root / "run_last.pt", root / "run_best.pt"
            trainer = MemoryTreeTrainer(
                HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3),
                HawkesFamily(2, 1, decays=torch.tensor([1.0])),
                CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
                wake=WakeObjectiveConfig(write_horizon=1),
                training=TrainingConfig(
                    epochs=1, checkpoint_path=str(last),
                    best_checkpoint_path=str(best), plot_after_training=False,
                ),
                device="cpu",
            )
            trainer.train([sequence], validation_dataset=[sequence], verbose=False)
            self.assertTrue(last.is_file() and best.is_file())
            last_payload = torch.load(last, map_location="cpu", weights_only=False)
            best_payload = torch.load(best, map_location="cpu", weights_only=False)
            self.assertEqual(last_payload["controller_state"]["controller_version"], 4)
            self.assertEqual(last_payload["checkpoint_identity"]["role"], "last")
            self.assertEqual(best_payload["checkpoint_identity"]["role"], "best")
            before = {
                name: value.detach().clone()
                for name, value in [
                    *trainer.tree.named_parameters(), *trainer.tree.named_buffers()
                ]
            }
            bank_sizes = {
                node_id: len(bank)
                for node_id, bank in trainer.tree.episodic_memory.banks.items()
            }
            trainer._validate_controller_checkpoint(last, [sequence])
            after = dict([
                *trainer.tree.named_parameters(), *trainer.tree.named_buffers()
            ])
            self.assertTrue(all(
                torch.equal(value, after[name]) for name, value in before.items()
            ))
            self.assertEqual(bank_sizes, {
                node_id: len(bank)
                for node_id, bank in trainer.tree.episodic_memory.banks.items()
            })


if __name__ == "__main__":
    unittest.main()
