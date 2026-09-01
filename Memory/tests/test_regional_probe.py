import unittest

import torch

from HawkesBackbone import HawkesFamily
from LatentHawkesTree import HawkesTree
from Routing_Retrieval_Investigation.routing_retrieval_investigation import (
    FrontierRoutingConfig,
)
from Train.RegionalProbe import counterfactual_energy_probe
from Train.Train import (
    CausalPrefixEncoder,
    MemoryTreeTrainer,
    WakeObjectiveConfig,
)


class RegionalProbeTests(unittest.TestCase):
    def _trainer(self, *, depth: int = 3) -> MemoryTreeTrainer:
        tree = HawkesTree(3, 5, 2, 1, init_depth=depth, memory_key_dim=3)
        tree.configure_frontier_routing(
            config=FrontierRoutingConfig(
                frontier_budget=2,
                frontier_min_experts=2,
            )
        )
        return MemoryTreeTrainer(
            tree,
            HawkesFamily(2, 1, decays=torch.tensor([1.0])),
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=6),
            wake=WakeObjectiveConfig(
                lambda_route_probe=1.0,
                # Deliberately inconsistent: the fixed option is deprecated
                # and each four-leaf coarse region must still select two.
                route_probe_leaves=99,
                route_probe_leaf_smoothing=0.05,
                route_balance_batch_size=2,
            ),
            device="cpu",
        )

    def test_counterfactual_teacher_uses_stop_and_leaf_hawkes_energy(self):
        probe = counterfactual_energy_probe(
            coarse_energy=torch.tensor([2.0, 1.0]),
            leaf_energy=torch.tensor([[0.2, 3.0], [2.0, 3.0]]),
            coarse_responsibility=torch.ones(2),
            leaf_prior=torch.ones(2),
            teacher_temperature=0.2,
            gain_temperature=0.2,
            leaf_smoothing=0.05,
        )
        self.assertGreater(float(probe.expand_target[0]), 0.99)
        self.assertLess(float(probe.expand_target[1]), 0.01)
        self.assertTrue(torch.allclose(
            probe.smoothed_leaf_credit.sum(dim=1), torch.ones(2)
        ))
        self.assertGreater(float(probe.observed_gain), 0.0)
        self.assertFalse(probe.teacher.requires_grad)

    def test_probe_covers_leaves_and_only_calibrates_selected_local_offsets(self):
        torch.manual_seed(811)
        trainer = self._trainer(depth=3)
        tree = trainer.tree
        self.assertEqual(trainer._regional_probe_leaf_count(4), 2)
        self.assertEqual(trainer._regional_probe_leaf_count(5), 3)
        sequences = [
            trainer._move_sequence({
                "times": torch.tensor([0.2, 0.6, 1.1, 1.7]),
                "types": torch.tensor([0, 1, 0, 1]),
            }),
            trainer._move_sequence({
                "times": torch.tensor([0.1, 0.5, 0.9, 1.5]),
                "types": torch.tensor([1, 0, 1, 0]),
            }),
        ]
        z_all, flat = trainer._encode_global_sequence_batch(sequences)
        output = tree(
            z_t=z_all,
            working_delta=torch.zeros(tree.param_dim),
            decays=trainer.hawkes.decays,
            frontier_projected_z=tree.router_compat.project_z(z_all),
            frontier_query=tree.episodic_memory.query_net(z_all),
            update_memory_state=False,
            update_search_state=False,
            detach_routing=True,
            materialize_diagnostics=False,
        )
        frontier_before = output["frontier_node_indices"].clone()
        probe = trainer._regional_probe_objective(
            output,
            output["frontier_mass"].detach(),
            flat["sequence_index"],
            len(sequences),
            z_all,
            flat,
        )
        self.assertEqual(int(probe["regions"]), 2)
        self.assertEqual(int(probe["probe_leaves"]), 4)
        first_selected = {
            leaf_id
            for leaf_id, visits in tree.frontier_routing.probe_leaf_visits.items()
            if visits == 1
        }
        self.assertEqual(len(first_selected), 4)

        probe["loss"].backward()
        for leaf_id in tree.leaf_ids:
            gradient = tree.semantic_offset[leaf_id].grad
            if leaf_id in first_selected:
                self.assertIsNotNone(gradient)
                self.assertTrue(bool((gradient.abs() > 0).any()))
            else:
                self.assertIsNone(gradient)
        self.assertTrue(all(
            tree.semantic_offset[node_id].grad is None
            for node_id in tree.internal_ids
        ))
        self.assertTrue(all(
            parameter.grad is None for parameter in tree.hyper.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None
            and bool((parameter.grad.abs() > 0).any())
            for parameter in tree.expansion_predictor.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None
            and bool((parameter.grad.abs() > 0).any())
            for parameter in tree.router_compat.parameters()
        ))
        self.assertTrue(torch.equal(
            output["frontier_node_indices"], frontier_before
        ))

        trainer._regional_probe_objective(
            output,
            output["frontier_mass"].detach(),
            flat["sequence_index"],
            len(sequences),
            z_all,
            flat,
        )
        self.assertEqual(
            set(tree.frontier_routing.probe_leaf_visits), set(tree.leaf_ids)
        )
        self.assertTrue(all(
            visits == 1
            for visits in tree.frontier_routing.probe_leaf_visits.values()
        ))

    def test_probe_coverage_and_predictor_survive_checkpoint_restore(self):
        tree = HawkesTree(3, 5, 2, 1, init_depth=2, memory_key_dim=3)
        tree.frontier_routing.probe_leaf_visits = {
            tree.leaf_ids[0]: 3,
            tree.leaf_ids[1]: 1,
        }
        state = tree.state_dict()

        restored = HawkesTree(3, 5, 2, 1, init_depth=0, memory_key_dim=3)
        restored.load_state_dict(state)
        self.assertEqual(restored.leaf_ids, tree.leaf_ids)
        self.assertEqual(
            restored.frontier_routing.probe_leaf_visits,
            tree.frontier_routing.probe_leaf_visits,
        )
        for actual, expected in zip(
            restored.expansion_predictor.parameters(),
            tree.expansion_predictor.parameters(),
        ):
            self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
