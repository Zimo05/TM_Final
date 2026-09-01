from __future__ import annotations

import copy
import unittest

import torch

from LatentHawkesTree import HawkesTree
from routing_retrieval_investigation import (
    FrontierRoutingConfig,
    FrontierRoutingRetrieval,
)


class FrontierRoutingRetrievalTests(unittest.TestCase):
    def test_frontier_budget_has_no_fixed_upper_bound(self) -> None:
        config = FrontierRoutingConfig(
            frontier_budget=7,
            frontier_min_experts=2,
        )
        config.validate()
        self.assertEqual(config.frontier_budget, 7)
        model = FrontierRoutingRetrieval(self.tree, config)
        packed = model.route_packed(
            torch.randn(3, self.tree.z_dim),
            update_search_state=False,
        )
        self.assertEqual(tuple(packed.node_indices.shape), (3, 7))
        self.assertEqual(tuple(packed.mask.shape), (3, 7))

    def setUp(self) -> None:
        torch.manual_seed(211)
        self.tree = HawkesTree(
            z_dim=4,
            node_dim=6,
            num_event_types=3,
            num_basis=2,
            init_depth=3,
            memory_key_dim=4,
        )
        self.model = FrontierRoutingRetrieval(
            self.tree,
            FrontierRoutingConfig(
                frontier_budget=4,
                exploration_epsilon=0.05,
            ),
        )

    def test_frontier_is_partition_and_mass_is_conserved(self) -> None:
        samples = self.model.route(
            torch.randn(5, self.tree.z_dim),
            update_search_state=False,
        )
        for sample in samples:
            self.assertGreaterEqual(len(sample.node_ids), 2)
            self.assertLessEqual(len(sample.node_ids), 4)
            self.assertTrue(torch.allclose(
                sample.mass.sum(),
                torch.tensor(1.0),
                atol=1e-6,
            ))
            for leaf_path in self.tree.leaf_paths:
                self.assertEqual(
                    sum(node_id in leaf_path for node_id in sample.node_ids),
                    1,
                )
            for left_index, left_id in enumerate(sample.node_ids):
                for right_id in sample.node_ids[left_index + 1:]:
                    self.assertNotIn(
                        left_id,
                        self.tree.path_to_node(right_id),
                    )
                    self.assertNotIn(
                        right_id,
                        self.tree.path_to_node(left_id),
                    )

    def test_dynamic_ten_leaf_tree_remains_supported(self) -> None:
        self.tree.split_leaf(self.tree.leaf_ids[0])
        self.tree.split_leaf(self.tree.leaf_ids[1])
        self.assertEqual(len(self.tree.leaf_ids), 10)
        sample = self.model.route(
            torch.randn(1, self.tree.z_dim),
            update_search_state=False,
        )[0]
        self.assertLessEqual(len(sample.node_ids), 4)
        for leaf_path in self.tree.leaf_paths:
            self.assertEqual(
                sum(node_id in leaf_path for node_id in sample.node_ids),
                1,
            )
        self.assertEqual(
            self.model.prototypes.node_ids,
            tuple(self.tree.all_node_ids),
        )

    def test_only_active_frontier_branches_are_scored(self) -> None:
        packed = self.model.route_packed(
            torch.randn(1, self.tree.z_dim),
            update_search_state=False,
        )
        branch_count = int(packed.expanded_mask[0].sum())
        self.assertGreaterEqual(branch_count, 1)
        self.assertLess(branch_count, len(self.tree.internal_ids))
        self.assertEqual(
            tuple(packed.expanded_probability.shape[-1:]), (2,)
        )
        self.assertTrue(torch.allclose(
            packed.expanded_probability.sum(dim=-1)[
                packed.expanded_mask
            ],
            torch.ones(branch_count),
        ))

    def test_irregular_tree_uses_descendant_mass_neutral_prior(self) -> None:
        # Make the left child a shallow leaf while the right child retains
        # four leaves. Uniform leaf target mass must yield root prior 1:4,
        # instead of the depth-biased 1:1 prior that caused Leaf-0 collapse.
        tree = HawkesTree(
            z_dim=4,
            node_dim=6,
            num_event_types=3,
            num_basis=2,
            init_depth=1,
            memory_key_dim=4,
        )
        right = tree.nodes["root"].right
        tree.split_leaf(right)
        tree.split_leaf(tree.nodes[right].left)
        model = FrontierRoutingRetrieval(
            tree,
            FrontierRoutingConfig(
                frontier_budget=4,
                # Legacy knobs are intentionally ignored by the exact
                # fixed-prior route distribution.
                prior_weight=0.0,
                exploration_epsilon=1.0,
            ),
        )
        root_prior = model._topology_tensors["child_prior"][0]
        descendant_counts = [
            len(tree.descendant_leaf_indices[child])
            for child in (tree.nodes["root"].left, tree.nodes["root"].right)
        ]
        expected = torch.tensor(descendant_counts, dtype=root_prior.dtype)
        expected /= expected.sum()
        self.assertTrue(torch.allclose(root_prior, expected))
        with torch.no_grad():
            for parameter in tree.router_compat.score_mlp.parameters():
                parameter.zero_()
        packed = model.route_packed(
            torch.randn(1, tree.z_dim),
            update_search_state=False,
        )
        self.assertTrue(torch.allclose(
            packed.expanded_probability[0, 0],
            expected,
            atol=1e-6,
        ))

    def test_batch_routing_does_not_depend_on_row_order(self) -> None:
        z_t = torch.randn(4, self.tree.z_dim)
        forward = self.model.route(
            z_t,
            update_search_state=True,
        )
        visits_after_forward = dict(self.model.expansion_visits)
        self.model.expansion_visits.clear()
        reverse = self.model.route(
            z_t.flip(0),
            update_search_state=True,
        )[::-1]
        self.assertEqual(
            self.model.expansion_visits,
            visits_after_forward,
        )
        for left, right in zip(forward, reverse):
            self.assertEqual(left.node_ids, right.node_ids)
            self.assertTrue(torch.allclose(left.mass, right.mass))

    def test_retrieval_reads_only_frontier_path_union_once(self) -> None:
        for node_index, node_id in enumerate(self.tree.all_node_ids):
            delta = torch.zeros(self.tree.param_dim)
            delta[node_index % self.tree.param_dim] = 0.1
            self.tree.episodic_memory.add_memory(
                node_id=node_id,
                key=torch.randn(self.tree.z_dim),
                delta_theta=delta,
            )
        output = self.model(
            torch.randn(1, self.tree.z_dim),
            working_delta=torch.zeros(self.tree.param_dim),
            update_memory_state=True,
            update_search_state=False,
        )
        sample = output.samples[0]
        self.assertEqual(
            set(output.memory_info[0]),
            set(sample.visited_node_ids),
        )
        for node_id, bank in self.tree.episodic_memory.banks.items():
            if node_id in sample.visited_node_ids:
                self.assertGreater(float(bank.cycle_usage.sum()), 0.0)
            else:
                self.assertEqual(float(bank.cycle_usage.sum()), 0.0)

    def test_effective_theta_matches_frontier_equation(self) -> None:
        working = torch.randn(self.tree.param_dim) * 0.01
        output = self.model(
            torch.randn(2, self.tree.z_dim),
            working_delta=working,
            update_memory_state=False,
            update_search_state=False,
        )
        for batch_index, sample in enumerate(output.samples):
            expected = (
                sample.mass.unsqueeze(-1)
                * output.frontier_theta[batch_index]
            ).sum(dim=0) + working
            self.assertTrue(torch.allclose(
                output.effective_params.theta[batch_index],
                expected,
                atol=1e-6,
            ))

    def test_prototype_statistics_propagate_to_ancestors(self) -> None:
        z = torch.randn(12, self.tree.z_dim)
        responsibility = torch.zeros(12, len(self.tree.leaf_ids))
        responsibility[
            torch.arange(12),
            torch.arange(12) % len(self.tree.leaf_ids),
        ] = 1.0
        self.model.prototypes.update_leaf_responsibility(
            z,
            self.tree.leaf_ids,
            self.tree.leaf_paths,
            responsibility,
        )
        index = self.model.prototypes.node_index
        self.assertEqual(
            float(self.model.prototypes.count[index["root"]]),
            12.0,
        )
        for leaf_index, leaf_id in enumerate(self.tree.leaf_ids):
            expected = float(
                (torch.arange(12) % len(self.tree.leaf_ids)
                 == leaf_index).sum()
            )
            self.assertEqual(
                float(self.model.prototypes.count[index[leaf_id]]),
                expected,
            )

    def test_batch_prototype_merge_matches_sequential_updates(self) -> None:
        torch.manual_seed(223)
        base = copy.deepcopy(self.model.prototypes)
        initial_z = torch.randn(9, self.tree.z_dim)
        initial_responsibility = torch.softmax(
            torch.randn(9, len(self.tree.leaf_ids)),
            dim=-1,
        )
        base.update_leaf_responsibility(
            initial_z,
            self.tree.leaf_ids,
            self.tree.leaf_paths,
            initial_responsibility,
        )
        sequential = copy.deepcopy(base)
        batched = copy.deepcopy(base)

        z = torch.randn(256, self.tree.z_dim)
        responsibility = torch.softmax(
            torch.randn(256, len(self.tree.leaf_ids)),
            dim=-1,
        )
        for row_index in range(z.size(0)):
            sequential.update_leaf_responsibility(
                z[row_index : row_index + 1],
                self.tree.leaf_ids,
                self.tree.leaf_paths,
                responsibility[row_index : row_index + 1],
            )
        batched.update_leaf_responsibility(
            z,
            self.tree.leaf_ids,
            self.tree.leaf_paths,
            responsibility,
        )

        self.assertTrue(torch.allclose(
            batched.count,
            sequential.count,
            atol=5e-5,
            rtol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            batched.mean,
            sequential.mean,
            atol=1e-6,
            rtol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            batched.m2,
            sequential.m2,
            atol=2e-4,
            rtol=2e-6,
        ))

    def test_router_receives_gradient_through_selected_frontier(self) -> None:
        output = self.model(
            torch.randn(3, self.tree.z_dim),
            working_delta=torch.zeros(self.tree.param_dim),
            update_memory_state=False,
            update_search_state=False,
        )
        output.effective_params.theta.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in self.tree.router_compat.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(any(
            bool((gradient.abs() > 0.0).any())
            for gradient in gradients
        ))


if __name__ == "__main__":
    unittest.main()
