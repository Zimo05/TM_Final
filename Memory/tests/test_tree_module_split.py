import unittest

import torch

from LatentHawkesTree import (
    FrontierRoutingOutput,
    HawkesHyperNet,
    HawkesTree,
    NodeSemanticCompatibility,
)
from TreeRouting import (
    FrontierRoutingOutput as SplitFrontierRoutingOutput,
    NodeSemanticCompatibility as SplitCompatibility,
)
from TreeSemantics import HawkesHyperNet as SplitHawkesHyperNet


class TreeModuleSplitTests(unittest.TestCase):
    def test_frontier_public_types_are_reexported(self):
        self.assertIs(
            FrontierRoutingOutput,
            SplitFrontierRoutingOutput,
        )
        self.assertIs(NodeSemanticCompatibility, SplitCompatibility)
        self.assertIs(HawkesHyperNet, SplitHawkesHyperNet)
        self.assertEqual(
            HawkesTree.refresh_structure_buffers.__module__,
            "TreeTopology",
        )
        self.assertEqual(HawkesTree.route.__module__, "TreeRouting")
        self.assertEqual(
            HawkesTree.semantic_params.__module__,
            "TreeSemantics",
        )

    def test_vectorized_semantics_match_pathwise_definition(self):
        torch.manual_seed(109)
        tree = HawkesTree(3, 5, 2, 2, init_depth=2)
        vectorized = tree.semantic_params()
        pathwise = torch.stack(
            [
                tree.semantic_theta(leaf_id)
                for leaf_id in tree.leaf_ids
            ]
        )
        packed = torch.cat(
            [
                vectorized.mu_tilde,
                vectorized.W_tilde.reshape(len(tree.leaf_ids), -1),
            ],
            dim=-1,
        )
        self.assertTrue(torch.allclose(
            vectorized.mu_tilde,
            pathwise[:, :tree.num_event_types],
            atol=1e-7,
            rtol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            packed,
            pathwise,
            atol=1e-7,
            rtol=1e-6,
        ))

    def test_dynamic_checkpoint_roundtrip_keeps_outputs_and_keys(self):
        torch.manual_seed(113)
        source = HawkesTree(
            3,
            5,
            2,
            1,
            init_depth=1,
            memory_key_dim=3,
        )
        source.split_leaf(source.leaf_ids[0])
        state = source.state_dict()

        restored = HawkesTree(
            3,
            5,
            2,
            1,
            init_depth=0,
            memory_key_dim=3,
        )
        restored.load_state_dict(state, strict=True)
        self.assertEqual(set(state), set(restored.state_dict()))
        z_t = torch.randn(4, 3)
        source_output = source(
            z_t,
            working_delta=torch.zeros(source.param_dim),
            update_memory_state=False,
        )
        restored_output = restored(
            z_t,
            working_delta=torch.zeros(restored.param_dim),
            update_memory_state=False,
        )
        for key in ("r", "router_logits", "episodic_delta"):
            self.assertTrue(torch.allclose(
                source_output[key],
                restored_output[key],
            ))


if __name__ == "__main__":
    unittest.main()
