import unittest

import torch

from HawkesBackbone import HawkesFamily
from LatentHawkesTree import HawkesTree
try:
    from Routing_Retrieval_Investigation.routing_retrieval_investigation import (
        FrontierRoutingConfig,
    )
except ModuleNotFoundError:
    from Routing_Retrieval.routing_retrieval_investigation import (
        FrontierRoutingConfig,
    )
from Sleep.Merge import commit_merge
from Train.Train import (
    CausalPrefixEncoder,
    MemoryTreeTrainer,
    WakeObjectiveConfig,
)


class FrontierIntegrationTests(unittest.TestCase):
    def _tree(self):
        tree = HawkesTree(
            4,
            6,
            3,
            2,
            init_depth=3,
            memory_key_dim=4,
        )
        tree.configure_frontier_routing(
            config=FrontierRoutingConfig(frontier_budget=4)
        )
        return tree

    def test_frontier_owner_is_not_registered_as_a_child_module(self):
        tree = self._tree()
        frontier = tree.frontier_routing

        self.assertIs(frontier.tree, tree)
        self.assertNotIn("tree", frontier._modules)

        # A future regular assignment must remain safe as well.
        frontier.tree = tree
        self.assertNotIn("tree", frontier._modules)
        self.assertIs(tree.to("cpu"), tree)
        self.assertTrue(all(
            parameter.device.type == "cpu"
            for parameter in tree.parameters()
        ))

    def test_batched_frontier_reuses_one_semantic_hypernet_table(self):
        tree = self._tree()
        expected = torch.stack([
            tree.semantic_theta(node_id)
            for node_id in tree.all_node_ids
        ])
        actual = tree.semantic_theta_table()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

        calls = 0

        def count_hyper_calls(module, inputs, output):
            nonlocal calls
            calls += 1

        handle = tree.hyper.register_forward_hook(count_hyper_calls)
        try:
            tree(
                torch.randn(16, tree.z_dim),
                working_delta=torch.zeros(tree.param_dim),
                update_memory_state=False,
                update_search_state=False,
            )
        finally:
            handle.remove()
        self.assertEqual(calls, 1)

    def test_wake_sequence_reuses_static_tree_tables_across_events(self):
        tree = self._tree()
        hawkes = HawkesFamily(
            3,
            2,
            decays=torch.tensor([0.5, 1.5]),
        )
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            CausalPrefixEncoder(3, 4, type_dim=4, hidden_dim=8),
            device="cpu",
        )
        hyper_calls = 0
        query_calls = 0
        projection_calls = 0

        def count_hyper_calls(module, inputs, output):
            nonlocal hyper_calls
            hyper_calls += 1

        def count_query_calls(module, inputs, output):
            nonlocal query_calls
            query_calls += 1

        def count_projection_calls(module, inputs, output):
            nonlocal projection_calls
            projection_calls += 1

        handles = [
            tree.hyper.register_forward_hook(count_hyper_calls),
            tree.episodic_memory.query_net.register_forward_hook(
                count_query_calls
            ),
            tree.router_compat.z_projection.register_forward_hook(
                count_projection_calls
            ),
        ]
        try:
            trainer.train_wake_sequence({
                "times": torch.tensor([0.1, 0.3, 0.6, 1.0]),
                "types": torch.tensor([0, 1, 2, 0]),
            })
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(hyper_calls, 1)
        self.assertEqual(query_calls, 1)
        self.assertEqual(projection_calls, 1)

    def test_main_forward_uses_bounded_frontier_and_path_union(self):
        torch.manual_seed(701)
        tree = self._tree()
        for node_index, node_id in enumerate(tree.all_node_ids):
            delta = torch.zeros(tree.param_dim)
            delta[node_index % tree.param_dim] = 0.01
            tree.episodic_memory.add_memory(
                node_id,
                torch.randn(tree.z_dim),
                delta,
            )
        output = tree(
            torch.randn(2, tree.z_dim),
            working_delta=torch.zeros(tree.param_dim),
            update_memory_state=False,
            update_search_state=False,
        )
        self.assertEqual(output["r"].shape, (2, 4))
        self.assertTrue(torch.allclose(
            output["r"].masked_fill(
                ~output["frontier_mask"], 0.0
            ).sum(dim=-1),
            torch.ones(2),
            atol=1e-6,
        ))
        self.assertLessEqual(output["frontier_mass"].size(1), 4)
        for batch_index, sample in enumerate(
            output["frontier_samples"]
        ):
            self.assertEqual(
                set(output["memory_info"][batch_index]),
                set(sample.visited_node_ids),
            )
            self.assertLess(
                len(sample.expanded_node_ids),
                len(tree.internal_ids),
            )
        output["effective_params"].theta.square().mean().backward()
        self.assertTrue(any(
            parameter.grad is not None
            and bool((parameter.grad.abs() > 0).any())
            for parameter in tree.router_compat.parameters()
        ))

    def test_dynamic_frontier_state_round_trips_with_tree(self):
        torch.manual_seed(709)
        tree = self._tree()
        tree.split_leaf(tree.leaf_ids[0])
        z_t = torch.randn(3, tree.z_dim)
        before = tree(
            z_t,
            working_delta=torch.zeros(tree.param_dim),
            update_memory_state=False,
            update_search_state=False,
        )
        tree.frontier_routing.prototypes.update_frontier_responsibility(
            z_t,
            before["frontier_node_indices"],
            before["r"],
            before["frontier_mask"],
        )
        expected = tree(
            z_t,
            working_delta=torch.zeros(tree.param_dim),
            update_memory_state=False,
            update_search_state=False,
        )
        state = tree.state_dict()

        restored = HawkesTree(
            4,
            6,
            3,
            2,
            init_depth=0,
            memory_key_dim=4,
        )
        restored.configure_frontier_routing(
            config=FrontierRoutingConfig(frontier_budget=4)
        )
        restored.load_state_dict(state)
        actual = restored(
            z_t,
            working_delta=torch.zeros(restored.param_dim),
            update_memory_state=False,
            update_search_state=False,
        )
        self.assertEqual(restored.leaf_ids, tree.leaf_ids)
        self.assertTrue(torch.allclose(
            actual["effective_params"].theta,
            expected["effective_params"].theta,
        ))
        self.assertTrue(torch.equal(
            restored.frontier_routing.prototypes.count,
            tree.frontier_routing.prototypes.count,
        ))

    def test_empirical_leaf_prior_is_conserved_across_merges(self):
        tree = HawkesTree(
            3,
            4,
            2,
            1,
            init_depth=1,
            memory_key_dim=3,
        )
        tree.split_leaf("root_L")
        tree.split_leaf("root_L_R")
        initial_leaves = tuple(tree.leaf_ids)
        self.assertEqual(initial_leaves, (
            "root_L_L",
            "root_L_R_L",
            "root_L_R_R",
            "root_R",
        ))
        tree.frontier_routing.set_target_leaf_mass(
            (0.25, 0.25, 0.25, 0.25),
            leaf_ids=initial_leaves,
        )

        commit_merge(tree, "root_L_R_L", "root_L_R_R")
        commit_merge(tree, "root_L_L", "root_L_R")
        tree.frontier_routing._sync_topology()

        self.assertEqual(tuple(tree.leaf_ids), ("root_L", "root_R"))
        self.assertTrue(torch.allclose(
            tree.frontier_routing._topology_tensors["leaf_mass"],
            torch.tensor([0.75, 0.25]),
        ))
        root_index = tree.all_node_ids.index("root")
        self.assertTrue(torch.allclose(
            tree.frontier_routing._topology_tensors["child_prior"][
                root_index
            ],
            torch.tensor([0.75, 0.25]),
        ))
        self.assertEqual(
            tree.frontier_routing.config.target_leaf_mass,
            (0.75, 0.25),
        )

    def test_global_training_updates_frontier_prototypes(self):
        torch.manual_seed(719)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3,
            4,
            2,
            1,
            init_depth=2,
            memory_key_dim=3,
        )
        tree.configure_frontier_routing(
            config=FrontierRoutingConfig(frontier_budget=3)
        )
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            wake=WakeObjectiveConfig(route_balance_batch_size=2),
            device="cpu",
        )
        dataset = [
            {
                "times": torch.tensor([0.1, 0.4, 0.8]),
                "types": torch.tensor([0, 1, 0]),
            },
            {
                "times": torch.tensor([0.2, 0.5, 0.9]),
                "types": torch.tensor([1, 0, 1]),
            },
        ]
        prototype_batch_sizes = []
        update_prototypes = (
            tree.frontier_routing.prototypes.update_frontier_responsibility
        )

        def track_prototype_batch(
            z,
            node_indices,
            responsibility,
            mask,
        ):
            prototype_batch_sizes.append(z.size(0))
            return update_prototypes(
                z,
                node_indices,
                responsibility,
                mask,
            )

        tree.frontier_routing.prototypes.update_frontier_responsibility = (
            track_prototype_batch
        )
        result = trainer.train_global_batch_epoch(
            dataset,
            torch.Generator().manual_seed(0),
            epoch=1,
        )
        root_index = tree.frontier_routing.prototypes.node_index["root"]
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(prototype_batch_sizes, [6])
        self.assertEqual(
            float(tree.frontier_routing.prototypes.count[root_index]),
            6.0,
        )

    def test_ambiguous_shallow_leaf_does_not_become_owner_by_argmax(self):
        tree = self._tree()
        hawkes = HawkesFamily(3, 2, decays=torch.tensor([0.5, 1.5]))
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            CausalPrefixEncoder(3, 4, type_dim=4, hidden_dim=8),
            device="cpu",
        )
        frontier = (
            "root_L_L_L",
            "root_L_L_R",
            "root_L_R",
            "root_R",
        )
        posterior = torch.tensor([0.26, 0.25, 0.25, 0.24])
        owner, used_lca, _ = trainer._posterior_owner(
            frontier, posterior
        )
        self.assertEqual(owner, "root")
        self.assertTrue(used_lca)
        self.assertNotEqual(owner, frontier[0])


if __name__ == "__main__":
    unittest.main()
