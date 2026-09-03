import copy
import tempfile
import unittest
from pathlib import Path

import torch

from HawkesBackbone import HawkesFamily
from LatentHawkesTree import HawkesTree
from MemoryResiduals.EpisodicMemory import TreeEpisodicMemory
from MemoryResiduals.ProbationMemory import ProbationCandidate
from RecalibrateController import _joint_ra_search
from CalibrateWriteRollout import paired_rollout_metrics
from ControllerIsolation import head_policy_sha256
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

    def test_probation_statistics_keep_signed_gain_and_independent_sequences(self):
        candidate = ProbationCandidate(
            token=(10, 2), source_sequence_id=10, event_index=2,
            owner_id="root", key=torch.ones(3), delta_theta=torch.ones(6),
            window=None, local_utility=1.0, local_quality=0.5,
            write_probability=0.8, split_probability=0.1, queue_weight=0.2,
        )
        self.assertFalse(candidate.record_validation(10, gain=100.0, weight=1.0))
        self.assertTrue(candidate.record_validation(20, gain=2.0, weight=1.0))
        self.assertTrue(candidate.record_validation(30, gain=-2.0, weight=1.0))
        self.assertFalse(candidate.record_validation(30, gain=50.0, weight=1.0))
        self.assertAlmostEqual(candidate.gain_mean, 0.0)
        self.assertAlmostEqual(candidate.gain_variance, 4.0)
        self.assertAlmostEqual(candidate.effective_sample_size, 2.0)
        self.assertLess(candidate.lower_confidence_bound(1.96), 0.0)
        self.assertFalse(candidate.promotion_ready(
            minimum_effective_samples=2.0,
            kappa=1.96,
            persist_threshold=0.0,
        ))

    def test_probation_gate_uses_soft_weight_ess_and_lcb(self):
        candidate = ProbationCandidate(
            token=(1, 0), source_sequence_id=1, event_index=0,
            owner_id="root", key=torch.ones(3), delta_theta=torch.ones(6),
            window=None, local_utility=1.0, local_quality=0.5,
            write_probability=0.8, split_probability=0.1, queue_weight=0.2,
        )
        candidate.record_validation(2, gain=2.0, weight=0.8)
        candidate.record_validation(3, gain=2.0, weight=0.2)
        self.assertAlmostEqual(
            candidate.effective_sample_size, 1.0 / (0.8 ** 2 + 0.2 ** 2)
        )
        self.assertFalse(candidate.promotion_ready(
            minimum_effective_samples=1.5,
            kappa=1.96,
            persist_threshold=0.0,
        ))
        self.assertTrue(candidate.promotion_ready(
            minimum_effective_samples=1.4,
            kappa=1.96,
            persist_threshold=0.0,
        ))
        self.assertAlmostEqual(
            candidate.promoted_quality(1.0), 1.0 - torch.exp(torch.tensor(-2.0)).item()
        )

    def test_sleep_structural_evidence_starts_at_persistent_memory_boundary(self):
        trainer = MemoryTreeTrainer(
            HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3),
            HawkesFamily(2, 1),
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            training=TrainingConfig(epochs=1, plot_after_training=False),
            device="cpu",
        )
        trainer._update_structural_evidence_buffer([{
            "token": (0, 0), "owner_id": "root", "priority": 1.0,
            "item": object(),
        }])
        self.assertEqual(trainer.sleep_state["structural_evidence_buffer"], {})

        # A legacy checkpoint's non-persistent shadow item cannot trigger or
        # populate a Split proposal.
        trainer.sleep_state["structural_evidence_buffer"] = {
            "root": {(0, 0): {"item": object()}}
        }
        trainer.controller.split_queues.clear()
        self.assertEqual(trainer.build_split_proposals(), {})
        self.assertEqual(trainer.sleep_state["structural_evidence_buffer"], {})

        # Raw controller pressure is diagnostic only. It cannot create
        # structural demand without evidence that crossed into a Bank.
        trainer.sleep_state["structural_mass_since_sleep"] = 100.0
        trainer.sleep_state["structural_observations_since_sleep"] = 1
        trainer.sleep_state["structural_demand_ema"] = 0.0
        trainer.controller.split_queues["root"] = 39_000.0
        first = trainer._continuous_split_demand(accepted_writes=0)
        self.assertEqual(first["observation"], 0.0)
        self.assertEqual(first["value"], 0.0)
        self.assertEqual(first["E_bank_struct"], 0.0)
        self.assertEqual(first["Q_decision"], 39_000.0)
        self.assertEqual(first["discarded_raw_structural_mass"], 100.0)

        # Persistent split_mass is the structural source of truth. The
        # controller queue may remain large or be cleared independently.
        trainer.tree.episodic_memory.add_memory(
            "root",
            torch.tensor([1.0, 0.0, 0.0]),
            torch.zeros(trainer.tree.param_dim),
            write_quality=1.0,
            queue_weight=0.5,
        )
        second = trainer._continuous_split_demand(accepted_writes=1)
        self.assertGreater(second["observation"], 0.0)
        self.assertAlmostEqual(second["E_bank_struct"], 0.5, places=6)
        self.assertAlmostEqual(second["N_persistent"], 1.0, places=6)

    def test_read_without_item_is_read_only(self):
        memory = TreeEpisodicMemory(
            key_dim=3, num_event_types=2, num_basis=1, device="cpu"
        )
        first_key = torch.tensor([1.0, 0.0, 0.0])
        second_key = torch.tensor([0.0, 1.0, 0.0])
        first_delta = torch.ones(6)
        second_delta = -torch.ones(6)
        memory.add_memory("root", first_key, first_delta, write_quality=0.8)
        # A law-radius outlier is queued once for prediction confirmation
        # before it is allowed to create a new dynamics mode.
        memory.add_memory("root", second_key, second_delta, write_quality=0.9)
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

    def test_v6_local_acceptance_is_invisible_until_cross_sequence_promotion(self):
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
            # Force this integration fixture through the local utility gate;
            # cross-sequence promotion remains at the production defaults.
            inference.wake_config.controller_write_gain_threshold = -1e9
            sequence = {
                "times": torch.arange(1, 13, dtype=torch.float32) * 0.1,
                "types": torch.tensor([0, 1] * 6),
                "source_index": 0,
                "cluster_id": 0,
            }
            memories_before = sum(
                len(bank) for bank in inference.tree.episodic_memory.banks.values()
            )
            split_queues_before = dict(inference.controller.split_queues)
            result = inference.run_sequence(sequence)
            memories_after = sum(
                len(bank) for bank in inference.tree.episodic_memory.banks.values()
            )
            support_after = sum(
                float(bank.support.sum())
                for bank in inference.tree.episodic_memory.banks.values()
            )
            split_queues_after_local = dict(inference.controller.split_queues)
            probation_metadata = {
                candidate.token: (
                    candidate.owner_id,
                    candidate.queue_weight,
                    candidate.split_probability,
                )
                for candidate in inference.write_probation
            }
            bank_lengths_after_local = {
                node_id: len(bank)
                for node_id, bank in inference.tree.episodic_memory.banks.items()
            }
            bank_split_mass_after_local = {
                node_id: float(bank.split_mass.sum())
                for node_id, bank in inference.tree.episodic_memory.banks.items()
            }
            inference.config.probation_min_effective_samples = 1.0
            inference.config.probation_lcb_kappa = 0.0
            inference.config.probation_persist_threshold = -1e9
            independent = dict(sequence)
            independent["source_index"] = 1
            promoted_result = inference.run_sequence(independent)
            memories_promoted = sum(
                len(bank) for bank in inference.tree.episodic_memory.banks.values()
            )
            support_promoted = sum(
                float(bank.support.sum())
                for bank in inference.tree.episodic_memory.banks.values()
            )
            split_queues_after_promotion = dict(
                inference.controller.split_queues
            )
            promoted_bank_weights = {
                node_id: bank.queue_weight.detach().cpu().tolist()
                for node_id, bank in inference.tree.episodic_memory.banks.items()
            }
            promoted_bank_split_mass = {
                node_id: float(bank.split_mass.sum())
                for node_id, bank in inference.tree.episodic_memory.banks.items()
            }
        self.assertEqual(memories_after, memories_before)
        self.assertEqual(split_queues_after_local, split_queues_before)
        self.assertEqual(result["accepted_write_count"], 0)
        self.assertEqual(result["promoted_write_count"], 0)
        self.assertGreater(result["local_accepted_write_count"], 0)
        self.assertEqual(
            result["probation_size"], result["local_accepted_write_count"]
        )
        self.assertGreater(promoted_result["probation_validation_count"], 0)
        self.assertGreater(promoted_result["promoted_write_count"], 0)
        self.assertLessEqual(
            memories_promoted - memories_after,
            promoted_result["promoted_write_count"],
        )
        self.assertAlmostEqual(
            support_promoted - support_after,
            promoted_result["promoted_write_count"],
        )
        for _, structural_weight, split_probability in probation_metadata.values():
            self.assertAlmostEqual(structural_weight, split_probability)
        promoted_weights_by_owner = {}
        for promotion in promoted_result["promotions"]:
            owner_id, structural_weight, _ = probation_metadata[
                tuple(promotion["token"])
            ]
            self.assertEqual(promotion["owner_id"], owner_id)
            self.assertAlmostEqual(
                promotion["structural_weight"], structural_weight
            )
            promoted_weights_by_owner.setdefault(owner_id, []).append(
                structural_weight
            )
        for owner_id, expected_weights in promoted_weights_by_owner.items():
            self.assertAlmostEqual(
                split_queues_after_promotion.get(owner_id, 0.0)
                - split_queues_after_local.get(owner_id, 0.0),
                sum(expected_weights),
            )
            self.assertAlmostEqual(
                promoted_bank_split_mass[owner_id]
                - bank_split_mass_after_local.get(owner_id, 0.0),
                sum(expected_weights),
            )
        accepted = [
            event for event in result["events"]
            if event.get("write_local_accepted")
        ]
        self.assertTrue(accepted)
        self.assertTrue(all(event["event_index"] <= 7 for event in accepted))
        required_diagnostics = {
            "memorize_argmax", "write_token", "write_candidate",
            "write_gate_active", "write_gate_passed", "write_priority_passed",
            "write_window_complete", "write_accepted",
            "write_local_accepted", "write_probation_enqueued",
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
        self.assertTrue(all(event["write_probation_enqueued"] for event in accepted))
        self.assertTrue(all(not event["write_accepted"] for event in accepted))
        probed = [event for event in result["events"] if event.get("write_probed")]
        self.assertTrue(probed)
        self.assertTrue(all(event.get("write_utility") is not None for event in probed))


if __name__ == "__main__":
    unittest.main()
