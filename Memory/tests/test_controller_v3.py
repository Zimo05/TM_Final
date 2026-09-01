import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from DataSplit import create_stratified_manifest, load_split_manifest, select_sequences
from HawkesBackbone import HawkesFamily
from Wake.SequentialController import Controller


class ControllerV3Tests(unittest.TestCase):
    def setUp(self):
        hawkes = HawkesFamily(num_types=2, num_basis=1)
        self.controller = Controller(
            hawkes, utility_temperature=0.25, utility_cost_margin=0.05
        )

    def test_signed_utility_targets_and_zero_cost_margin(self):
        gains = torch.tensor([-0.2, 0.0, 0.2])
        targets = self.controller.utility_target(gains, action_index=2)
        self.assertLess(float(targets[0]), float(targets[1]))
        self.assertLess(float(targets[1]), float(targets[2]))
        self.assertLess(float(targets[1]), 0.5)

    def test_positive_and_negative_utility_move_gate_in_right_direction(self):
        output = self.controller.action_distribution_batch(
            torch.tensor([0.2]), torch.tensor([0.4]), torch.tensor([0.3]),
            update_statistics=False,
        )
        initial = float(output["probabilities"][0, 2])
        optimizer = torch.optim.SGD(self.controller.parameters(), lr=0.2)
        for target_gain in (torch.tensor([0.5]),) * 10:
            optimizer.zero_grad()
            output = self.controller.action_distribution_batch(
                torch.tensor([0.2]), torch.tensor([0.4]), torch.tensor([0.3]),
                update_statistics=False,
            )
            target = self.controller.utility_target(target_gain, action_index=2)
            loss = torch.nn.functional.binary_cross_entropy(
                output["probabilities"][:, 2], target
            )
            loss.backward(); optimizer.step()
        high = float(output["probabilities"][0, 2])
        self.assertGreater(high, initial)
        for target_gain in (torch.tensor([-0.5]),) * 20:
            optimizer.zero_grad()
            output = self.controller.action_distribution_batch(
                torch.tensor([0.2]), torch.tensor([0.4]), torch.tensor([0.3]),
                update_statistics=False,
            )
            target = self.controller.utility_target(target_gain, action_index=2)
            loss = torch.nn.functional.binary_cross_entropy(
                output["probabilities"][:, 2], target
            )
            loss.backward(); optimizer.step()
        self.assertLess(float(output["probabilities"][0, 2]), high)

    def test_queue_split_is_conditioned_on_write(self):
        gates = torch.tensor([[0.1, 0.2, 0.0, 1.0], [0.1, 0.2, 0.5, 0.8]])
        self.assertTrue(torch.allclose(
            self.controller.queue_weight(gates), torch.tensor([0.0, 0.4])
        ))


class StratifiedManifestTests(unittest.TestCase):
    def test_manifest_is_disjoint_complete_and_stratified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path, output = root / "data.csv", root / "split.json"
            rows = []
            for cluster in range(3):
                for _ in range(100):
                    rows.append({"event_times": "[0,1]", "event_types": "[0,1]", "cluster": cluster})
            pd.DataFrame(rows).to_csv(data_path, index=False)
            manifest = create_stratified_manifest(data_path, output, seed=42)
            loaded = load_split_manifest(output, data_path=data_path)
            self.assertEqual(loaded["counts"], {"train": 210, "validation": 30, "test": 60})
            sets = [set(loaded["splits"][name]) for name in ("train", "validation", "test")]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertEqual(len(set.union(*sets)), 300)
            self.assertEqual(loaded["cluster_distribution"]["test"], {"0": 20, "1": 20, "2": 20})
            dataset = [{"source_index": index} for index in range(300)]
            self.assertEqual(len(select_sequences(dataset, manifest, "train")), 210)


if __name__ == "__main__":
    unittest.main()
