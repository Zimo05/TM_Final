import unittest
import tempfile
import copy
from pathlib import Path

import torch

from HawkesBackbone import HawkesFamily
from MemoryResiduals.EpisodicMemory import TreeEpisodicMemory
from LatentHawkesTree import HawkesTree
from Train.Train import CausalPrefixEncoder, MemoryTreeTrainer, TrainingConfig
from Wake.SequentialController import Controller


class ControllerV5Tests(unittest.TestCase):
    def setUp(self):
        self.controller = Controller(
            HawkesFamily(num_types=2, num_basis=1),
            write_candidate_threshold=0.6,
        )

    def test_calibrated_threshold_101_closes_actions_and_split(self):
        self.controller.controller_version.fill_(5)
        self.controller.split_enabled.fill_(False)
        self.controller.set_calibration_thresholds(1.01, 1.01, 1.01)
        output = self.controller.action_distribution_batch(
            torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.0]),
            update_statistics=False,
        )
        self.assertTrue(torch.all(output["probabilities"] == 0))
        self.assertTrue(torch.all(output["raw_probabilities"] > 0))

    def test_single_event_matches_batch_calibration_and_split_mask(self):
        self.controller.controller_version.fill_(5)
        self.controller.split_enabled.fill_(False)
        self.controller.set_calibration_thresholds(0.65, 0.70, 0.75)
        inputs = dict(
            surprise=torch.tensor(0.5), novelty=torch.tensor(0.3),
            count=torch.tensor(0.2), update_statistics=False,
            owner_confidence=torch.tensor(0.8),
            retrieval_similarity=torch.tensor(0.4),
            retrieval_residual_norm=torch.tensor(0.1),
            working_memory_norm=torch.tensor(0.2),
            pending_write_ratio=torch.tensor(0.25),
        )
        single = self.controller.action_distribution(**inputs)
        batch = self.controller.action_distribution_batch(**{
            key: (value.reshape(1) if torch.is_tensor(value) else value)
            for key, value in inputs.items()
        })
        self.assertTrue(torch.allclose(
            single["raw_probabilities"], batch["raw_probabilities"][0]
        ))
        self.assertTrue(torch.allclose(
            single["probabilities"], batch["probabilities"][0]
        ))
        self.assertEqual(float(single["probabilities"][3]), 0.0)
        self.assertTrue(torch.allclose(
            single["gates"].as_tensor(), single["probabilities"]
        ))
        for field in single["features"].__dataclass_fields__:
            scalar_value = getattr(single["features"], field)
            batch_value = getattr(batch["features"], field)[0]
            self.assertTrue(
                torch.allclose(scalar_value, batch_value, atol=1e-7), field
            )

    def test_write_admission_requires_every_condition(self):
        self.controller.set_calibration_thresholds(0.0, 0.0, 0.6)
        self.assertFalse(self.controller.write_admissible(0.59, 1.0, 1.0, future_window_complete=True))
        self.assertFalse(self.controller.write_admissible(0.8, 0.0, 1.0, future_window_complete=True))
        self.assertFalse(self.controller.write_admissible(0.8, 1.0, 0.0, future_window_complete=True))
        self.assertFalse(self.controller.write_admissible(0.8, 1.0, 1.0, future_window_complete=False))
        self.assertTrue(self.controller.write_admissible(0.8, 1.0, 1.0, future_window_complete=True))

    def test_virtual_retrieval_is_read_only(self):
        memory = TreeEpisodicMemory(
            key_dim=3, num_event_types=2, num_basis=1, device="cpu"
        )
        memory.add_memory("root", torch.tensor([1.0, 0.0, 0.0]), torch.ones(6))
        bank = memory.banks["root"]
        before = {
            "keys": bank.keys.clone(),
            "deltas": bank.deltas.clone(),
            "usage": bank.usage.clone(),
            "age": bank.age.clone(),
            "quality": bank.write_quality.clone(),
            "clock": memory._age_clock,
        }
        delta, info = memory.read_node_with_virtual_item(
            torch.tensor([1.0, 0.0, 0.0]), "root",
            key=torch.tensor([0.0, 1.0, 0.0]), delta=-torch.ones(6),
            write_quality=0.8, virtual_usage=1.5, virtual_age=2.0,
        )
        self.assertEqual(tuple(delta.shape), (6,))
        self.assertEqual(info["alpha"].numel(), 2)
        self.assertTrue(torch.equal(bank.keys, before["keys"]))
        self.assertTrue(torch.equal(bank.deltas, before["deltas"]))
        self.assertTrue(torch.equal(bank.usage, before["usage"]))
        self.assertTrue(torch.equal(bank.age, before["age"]))
        self.assertTrue(torch.equal(bank.write_quality, before["quality"]))
        self.assertEqual(memory._age_clock, before["clock"])

    def test_controller_only_preparation_freezes_model_and_resets_state(self):
        trainer = MemoryTreeTrainer(
            HawkesTree(3, 4, 2, 1, init_depth=2, memory_key_dim=3),
            HawkesFamily(2, 1),
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            training=TrainingConfig(epochs=1, plot_after_training=False),
            device="cpu",
        )
        for index, node_id in enumerate(trainer.tree.all_node_ids):
            trainer.tree.episodic_memory.add_memory(
                node_id,
                torch.tensor([1.0, float(index + 1), -0.5]),
                torch.full((6,), 0.01 * (index + 1)),
                write_quality=0.8,
            )
        trainer.history.append({"epoch": 99})
        trainer.prepare_controller_only_finetune(base_checkpoint="base_v4.pt")
        self.assertEqual(int(trainer.controller.controller_version), 5)
        self.assertFalse(bool(trainer.controller.split_enabled))
        self.assertEqual(trainer.history, [])
        self.assertEqual(trainer.completed_epochs, 0)
        self.assertTrue(all(
            not parameter.requires_grad
            for module in (trainer.hawkes, trainer.encoder, trainer.tree)
            for parameter in module.parameters()
        ))
        self.assertTrue(any(
            parameter.requires_grad for parameter in trainer.controller.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for name, parameter in trainer.controller.named_parameters()
            if not name.startswith(("bias_queue_split", "raw_queue_"))
        ))
        self.assertEqual(
            trainer.training_config.frozen_state_sha256,
            trainer.non_controller_state_sha256(),
        )
        frozen_before = {
            f"{prefix}.{name}": copy.deepcopy(value)
            for prefix, module in (
                ("hawkes", trainer.hawkes),
                ("encoder", trainer.encoder),
                ("tree", trainer.tree),
            )
            for name, value in module.state_dict().items()
            if not (prefix == "tree" and name.startswith("working_memory."))
        }
        sequence = {
            "times": torch.tensor([0.1, 0.3, 0.7]),
            "types": torch.tensor([0, 1, 0]),
            "source_index": torch.tensor(0),
            "cluster_id": torch.tensor(0),
        }
        with tempfile.TemporaryDirectory() as directory:
            trainer.training_config.checkpoint_path = str(Path(directory) / "last.pt")
            try:
                trainer.train([sequence], verbose=False)
            except RuntimeError as error:
                frozen_after = {
                    f"{prefix}.{name}": copy.deepcopy(value)
                    for prefix, module in (
                        ("hawkes", trainer.hawkes),
                        ("encoder", trainer.encoder),
                        ("tree", trainer.tree),
                    )
                    for name, value in module.state_dict().items()
                    if not (prefix == "tree" and name.startswith("working_memory."))
                }
                def same(left, right):
                    if torch.is_tensor(left):
                        return torch.equal(left, right)
                    if isinstance(left, dict):
                        return left.keys() == right.keys() and all(
                            same(left[key], right[key]) for key in left
                        )
                    if isinstance(left, (list, tuple)):
                        return len(left) == len(right) and all(
                            same(a, b) for a, b in zip(left, right)
                        )
                    return left == right
                changed = [
                    name for name, value in frozen_before.items()
                    if not same(value, frozen_after[name])
                ]
                raise AssertionError(f"{error}; changed={changed}") from error
            payload = torch.load(
                trainer.training_config.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        self.assertEqual(payload["controller_state"]["controller_version"], 5)
        self.assertTrue(payload["controller_only_invariants"]["verified"])


if __name__ == "__main__":
    unittest.main()
