import copy
import tempfile
import unittest
from pathlib import Path

import torch

from HawkesBackbone import HawkesFamily
from LatentHawkesTree import HawkesTree
from MemoryResiduals.EpisodicMemory import TreeEpisodicMemory
from RecalibrateController import _joint_ra_search
from CalibrateWriteRollout import paired_rollout_metrics
from ControllerIsolation import head_policy_sha256
from PhysicalWriteProbe import paired_physical_write_utility
from Train.Inference import InferenceConfig, MemoryTreeInference
from Train.Train import CausalPrefixEncoder, MemoryTreeTrainer, TrainingConfig
from Wake.SequentialController import Controller


class ControllerV6Tests(unittest.TestCase):
    def setUp(self):
        self.controller = Controller(
            HawkesFamily(num_types=2, num_basis=1),
            write_candidate_threshold=0.6,
        )
        self.controller.controller_version.fill_(6)
        self.controller.split_enabled.fill_(False)
        self.controller.set_calibration_thresholds(0.7, 0.65, 0.75)

    def test_v6_admission_never_uses_future_utility(self):
        self.assertTrue(self.controller.write_admissible(
            0.8, -100.0, 0.1, future_window_complete=True
        ))
        self.assertFalse(self.controller.write_admissible(
            0.7, 100.0, 0.1, future_window_complete=True
        ))
        self.assertFalse(self.controller.write_admissible(
            0.8, 100.0, 0.0, future_window_complete=True
        ))
        self.assertFalse(self.controller.write_admissible(
            0.8, 100.0, 0.1, future_window_complete=False
        ))

    def test_read_without_item_is_read_only(self):
        memory = TreeEpisodicMemory(
            key_dim=3, num_event_types=2, num_basis=1, device="cpu"
        )
        first_key = torch.tensor([1.0, 0.0, 0.0])
        second_key = torch.tensor([0.0, 1.0, 0.0])
        first_delta = torch.ones(6)
        second_delta = -torch.ones(6)
        memory.add_memory("root", first_key, first_delta, write_quality=0.8)
        memory.add_memory("root", second_key, second_delta, write_quality=0.9)
        bank = memory.banks["root"]
        before = {
            name: copy.deepcopy(getattr(bank, name))
            for name in ("keys", "deltas", "usage", "age", "write_quality")
        }
        result, info = memory.read_node_without_item(
            second_key, "root", key=second_key, delta=second_delta
        )
        self.assertEqual(tuple(result.shape), (6,))
        self.assertIn("excluded_index", info)
        for name, value in before.items():
            self.assertTrue(torch.equal(value, getattr(bank, name)))

    def test_joint_search_accounts_for_action_interaction(self):
        def event(nll, adapt_gate=0.8, retrieve_gate=0.8):
            return {
                "nll": nll,
                "action_probabilities": torch.tensor(
                    [adapt_gate, retrieve_gate, 0.2, 0.0]
                ),
                "raw_action_probabilities": torch.tensor(
                    [adapt_gate, retrieve_gate, 0.2, 0.2]
                ),
            }
        semantic = {(0, 0): event(1.0)}
        retrieve = {(0, 0): event(0.9)}
        adapt = {(0, 0): event(0.9)}
        both = {(0, 0): event(1.1)}
        selected, table = _joint_ra_search(semantic, retrieve, adapt, both)
        self.assertTrue(table)
        self.assertFalse(
            selected["retrieve_threshold"] < 1.01
            and selected["adapt_threshold"] < 1.01
        )

    def test_selective_v6_finetune_freezes_adapt_and_split(self):
        trainer = MemoryTreeTrainer(
            HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3),
            HawkesFamily(2, 1),
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            training=TrainingConfig(epochs=1, plot_after_training=False),
            device="cpu",
        )
        trainer.controller.set_calibration_thresholds(0.65, 0.7, 0.75)
        trainer.prepare_controller_only_finetune(
            base_checkpoint="recalibrated.pt", target_version=6,
            train_heads=("retrieve", "write"),
        )
        self.assertEqual(int(trainer.controller.controller_version), 6)
        calibration = trainer.controller.calibration_dict()
        self.assertAlmostEqual(calibration["adapt_threshold"], 0.7, places=6)
        self.assertAlmostEqual(calibration["retrieve_threshold"], 0.65, places=6)
        self.assertAlmostEqual(calibration["write_threshold"], 0.75, places=6)
        self.assertFalse(trainer.controller.bias_assimilate.requires_grad)
        self.assertTrue(trainer.controller.bias_retrieve.requires_grad)
        self.assertTrue(trainer.controller.bias_memorize.requires_grad)
        self.assertFalse(trainer.controller.bias_queue_split.requires_grad)
        before = trainer.controller.context_gate.weight.detach().clone()
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.controller.context_gate.weight.sum().backward()
        trainer.optimizer.step()
        after = trainer.controller.context_gate.weight.detach()
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[3], after[3]))
        self.assertFalse(torch.equal(before[1], after[1]))
        self.assertFalse(torch.equal(before[2], after[2]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v6.pt"
            trainer.save_checkpoint(path, epoch=0)
            payload = torch.load(path, map_location="cpu")
        invariants = payload["controller_only_invariants"]
        self.assertTrue(invariants["verified"])
        self.assertTrue(invariants["inactive_controller_heads_verified"])
        self.assertEqual(
            payload["controller_state"]["trainable_heads"],
            ["retrieve", "write"],
        )

    def test_write_only_step_preserves_retrieve_adapt_and_buffers(self):
        trainer = MemoryTreeTrainer(
            HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3),
            HawkesFamily(2, 1),
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            training=TrainingConfig(epochs=1, plot_after_training=False),
            device="cpu",
        )
        trainer.controller.set_calibration_thresholds(0.65, 0.7, 0.85)
        trainer.controller.utility_temperatures.copy_(torch.tensor([0.2, 0.3, 0.4, 0.5]))
        trainer.prepare_controller_only_finetune(
            base_checkpoint="rollout.pt", target_version=6,
            train_heads=("write",),
        )
        before = trainer.controller.state_dict()
        retrieve_sha = head_policy_sha256(before, "retrieve")
        adapt_sha = head_policy_sha256(before, "adapt")
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = (
            trainer.controller.bias_memorize
            + trainer.controller.context_gate.weight[2].sum()
        )
        loss.backward()
        trainer.optimizer.step()
        after = trainer.controller.state_dict()
        self.assertEqual(head_policy_sha256(after, "retrieve"), retrieve_sha)
        self.assertEqual(head_policy_sha256(after, "adapt"), adapt_sha)
        self.assertAlmostEqual(
            trainer.controller.calibration_dict()["retrieve_threshold"], 0.65
        )
        self.assertAlmostEqual(
            trainer.controller.calibration_dict()["adapt_threshold"], 0.7
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "write_only.pt"
            trainer.save_checkpoint(path, epoch=0)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        invariants = payload["controller_only_invariants"]
        self.assertTrue(invariants["write_isolation_verified"])
        self.assertEqual(invariants["retrieve_policy_sha256"], retrieve_sha)
        self.assertEqual(invariants["adapt_policy_sha256"], adapt_sha)

    def test_zero_rollout_gain_is_not_feasible(self):
        def row(index):
            return {
                "source_index": 0,
                "event_index": index,
                "nll": 1.0,
                "true_type": index % 2,
                "predicted_type_at_event_time": index % 2,
                "type_probabilities": [0.8, 0.2] if index % 2 == 0 else [0.2, 0.8],
                "true_time": float(index + 1),
                "predicted_time": float(index + 1),
                "write_accepted": False,
            }
        frozen = [row(index) for index in range(10)]
        metrics = paired_rollout_metrics(
            frozen, copy.deepcopy(frozen), num_types=2,
            sequence_count=1, seed=42,
        )
        self.assertEqual(metrics["mean_nll_gain"], 0.0)
        self.assertFalse(metrics["feasible"])

    def test_v6_online_inference_commits_after_construction_window(self):
        trainer = MemoryTreeTrainer(
            HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3),
            HawkesFamily(2, 1),
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            training=TrainingConfig(epochs=1, plot_after_training=False),
            device="cpu",
        )
        trainer.prepare_controller_only_finetune(
            base_checkpoint="base.pt", target_version=6,
            train_heads=("retrieve", "write"),
        )
        trainer.controller.set_calibration_thresholds(0.0, 0.0, 0.0)
        trainer.controller.bias_memorize.data.fill_(10.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v6.pt"
            trainer.save_checkpoint(path, epoch=0)
            inference = MemoryTreeInference.from_checkpoint(
                path, device="cpu",
                inference_config=InferenceConfig(
                    adapt_working_memory=True, allow_memory_writes=True,
                    update_memory_usage=True, probe_write_counterfactuals=True,
                    write_probe_seed=42,
                ),
            )
            sequence = {
                "times": torch.arange(1, 13, dtype=torch.float32) * 0.1,
                "types": torch.tensor([0, 1] * 6),
                "source_index": 0,
                "cluster_id": 0,
            }
            result = inference.run_sequence(sequence)
            exact = paired_physical_write_utility(
                path, sequence, 0, device="cpu"
            )
        self.assertGreater(result["accepted_write_count"], 0)
        accepted = [
            event for event in result["events"] if event.get("write_accepted")
        ]
        self.assertTrue(accepted)
        self.assertTrue(all(event["event_index"] <= 7 for event in accepted))
        required_diagnostics = {
            "memorize_argmax", "write_token", "write_candidate",
            "write_gate_active", "write_gate_passed", "write_priority_passed",
            "write_window_complete", "write_accepted",
            "visited_bank_count", "visited_nonempty_bank_count",
            "retrieval_alpha_mass", "retrieval_alpha_per_visited_node",
            "retrieval_similarity", "retrieval_effective_k",
            "retrieval_null_alpha", "raw_episodic_residual_norm",
            "retrieve_gate", "gated_episodic_residual_norm",
            "owner_on_retrieval_path", "retrieval_counterfactual_gain",
        }
        self.assertFalse(required_diagnostics - set(result["events"][0]))
        self.assertEqual(
            result["events"][0]["write_token"], "0:0"
        )
        self.assertTrue(all(event["write_window_complete"] for event in accepted))
        self.assertTrue(all(event["write_gate_passed"] for event in accepted))
        probed = [event for event in result["events"] if event.get("write_probed")]
        self.assertTrue(probed)
        self.assertTrue(all(event.get("write_utility") is not None for event in probed))
        self.assertEqual(exact["construction_window"], [0, 5])
        self.assertEqual(exact["score_window"], [5, 10])
        self.assertTrue(exact["accepted"])


if __name__ == "__main__":
    unittest.main()
