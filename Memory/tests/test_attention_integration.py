"""H_tree semantic init + causal online encoder (PDF 1A+2A)."""

import tempfile
import unittest
from pathlib import Path

import torch

from AttentionEncoderAdapter import (
    attention_id_to_memory_id,
    initialize_node_embeddings_from_h_tree,
    initialize_tree_from_h_tree_file,
    load_h_tree,
    memory_id_to_attention_id,
    synchronize_tree_topology_from_node_ids,
)
from HawkesBackbone import HawkesFamily
from LatentHawkesTree import HawkesTree
from Train.Inference import MemoryTreeInference
from Train.Train import CausalPrefixEncoder, MemoryTreeTrainer, TrainingConfig


class HTreeIdMappingTests(unittest.TestCase):
    def test_attention_and_memory_ids_roundtrip(self):
        self.assertEqual(attention_id_to_memory_id("root"), "root")
        self.assertEqual(attention_id_to_memory_id("l"), "root_L")
        self.assertEqual(attention_id_to_memory_id("l_r"), "root_L_R")
        self.assertEqual(memory_id_to_attention_id("root_L_R"), "l_r")
        self.assertEqual(memory_id_to_attention_id("root_L"), "l")


class HTreeSemanticInitTests(unittest.TestCase):
    def test_topology_sync_reconstructs_all_attention_nodes(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        node_ids = ("root", "l", "r", "l_l", "l_r")
        synchronized = synchronize_tree_topology_from_node_ids(tree, node_ids)
        self.assertEqual(
            set(synchronized),
            {"root", "root_L", "root_R", "root_L_L", "root_L_R"},
        )
        self.assertEqual(len(tree.internal_ids), 2)
        self.assertEqual(len(tree.leaf_ids), 3)
        self.assertEqual(set(tree.episodic_memory.banks), set(synchronized))
        routing = tree.route(torch.zeros(1, tree.z_dim)).responsibility
        self.assertTrue(torch.allclose(
            routing.sum(dim=-1),
            torch.ones(1),
            atol=1e-7,
        ))

    def test_topology_sync_rejects_one_sided_branch(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        with self.assertRaises(ValueError):
            synchronize_tree_topology_from_node_ids(tree, ("root", "l"))

    def test_initialize_copies_matching_node_embeddings(self):
        torch.manual_seed(0)
        node_dim = 4
        tree = HawkesTree(3, node_dim, 2, 1, init_depth=1, memory_key_dim=3)
        node_ids = ("root", "l", "r")
        h_tree = torch.randn(3, node_dim)
        mapped = initialize_node_embeddings_from_h_tree(tree, h_tree, node_ids)
        self.assertEqual(set(mapped), {"root", "root_L", "root_R"})
        self.assertTrue(torch.allclose(tree.node_emb["root"], h_tree[0]))
        self.assertTrue(torch.allclose(tree.node_emb["root_L"], h_tree[1]))
        self.assertTrue(torch.allclose(tree.node_emb["root_R"], h_tree[2]))

    def test_dimension_mismatch_raises(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        with self.assertRaises(ValueError):
            initialize_node_embeddings_from_h_tree(
                tree, torch.randn(3, 8), ("root", "l", "r")
            )

    def test_split_leaf_without_attention_id_is_left_unchanged(self):
        torch.manual_seed(1)
        node_dim = 3
        tree = HawkesTree(2, node_dim, 2, 1, init_depth=1, memory_key_dim=2)
        tree.split_leaf("root_L")
        before = tree.node_emb["root_L_L"].detach().clone()
        h_tree = torch.randn(3, node_dim)
        initialize_node_embeddings_from_h_tree(
            tree, h_tree, ("root", "l", "r")
        )
        self.assertTrue(torch.equal(tree.node_emb["root_L_L"], before))
        self.assertTrue(torch.allclose(tree.node_emb["root_L"], h_tree[1]))

    def test_load_h_tree_file_and_initialize(self):
        torch.manual_seed(2)
        node_dim = 5
        tree = HawkesTree(2, node_dim, 2, 1, init_depth=1, memory_key_dim=2)
        payload = {
            "H_tree_refined": torch.randn(3, node_dim),
            "node_ids": ["root", "l", "r"],
            "config": {"mode": "node_only", "d_model": node_dim},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h_tree.pt"
            torch.save(payload, path)
            node_ids, h_tree = load_h_tree(path)
            self.assertEqual(node_ids, ("root", "l", "r"))
            mapped = initialize_tree_from_h_tree_file(tree, path)
            self.assertEqual(len(mapped), 3)
            self.assertTrue(torch.allclose(tree.node_emb["root"], h_tree[0]))


class CausalOnlineEncoderTests(unittest.TestCase):
    def test_padded_prefix_matches_independent_sequences(self):
        torch.manual_seed(2)
        encoder = CausalPrefixEncoder(3, 5, type_dim=4, hidden_dim=7)
        times = torch.tensor([
            [0.1, 0.4, 0.9, 1.2],
            [0.2, 0.5, 0.0, 0.0],
        ])
        types = torch.tensor([
            [0, 1, 2, 0],
            [2, 1, 0, 0],
        ])
        mask = torch.tensor([
            [True, True, True, True],
            [True, True, False, False],
        ])
        padded, padded_mask = encoder.forward_padded_prefix(
            times, types, mask
        )
        for row, length in enumerate((4, 2)):
            reference = encoder.forward_all_prefix(
                times[row, :length],
                types[row, :length],
            )
            self.assertTrue(torch.allclose(
                padded[row, :length], reference, atol=1e-6
            ))
        self.assertTrue(torch.equal(mask, padded_mask))
        self.assertEqual(
            float(padded[1, 2:].detach().abs().sum()), 0.0
        )

    def test_forward_uses_global_z_t_without_node_z(self):
        torch.manual_seed(3)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        times = torch.tensor([0.1, 0.4, 0.9])
        types = torch.tensor([0, 1, 0])
        z_t = encoder(times, types, 2).reshape(1, -1)
        result = tree(
            z_t=z_t,
            decays=hawkes.decays,
            update_memory_state=False,
        )
        self.assertEqual(result["r"].shape, (1, 4))
        self.assertEqual(int(result["frontier_mask"].sum()), 2)
        self.assertEqual(result["memory_query"].shape, (1, 3))
        self.assertNotIn("encoder_node_z", result)
        self.assertNotIn("memory_query_by_node", result)

    def test_prediction_nll_reaches_encoder_router_and_retriever(self):
        torch.manual_seed(41)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        for node_index, node_id in enumerate(tree.all_node_ids):
            for memory_index, key in enumerate(torch.eye(3)):
                delta = torch.zeros(tree.param_dim)
                delta[(node_index + memory_index) % tree.param_dim] = (
                    0.1 * (memory_index + 1)
                )
                tree.episodic_memory.add_memory(node_id, key, delta)
            bank = tree.episodic_memory.get_bank(node_id)
            bank.usage.copy_(torch.tensor([0.0, 1.0, 3.0]))
            bank.age.copy_(torch.tensor([0.0, 2.0, 5.0]))

        sequence = {
            "times": torch.tensor([0.1, 0.4, 0.9]),
            "types": torch.tensor([0, 1, 0]),
        }
        z_t = encoder(sequence["times"], sequence["types"], 2).reshape(1, -1)
        output = tree(
            z_t,
            decays=hawkes.decays,
            update_memory_state=False,
        )
        loss = hawkes.event_NLL(
            sequence, output["effective_params"].select(0), 2
        )
        loss.backward()

        def grad_sum(parameters):
            return sum(
                float(parameter.grad.abs().sum())
                for parameter in parameters
                if parameter.grad is not None
            )

        self.assertGreater(grad_sum(encoder.parameters()), 0.0)
        self.assertGreater(
            grad_sum(tree.episodic_memory.query_net.parameters()), 0.0
        )
        self.assertGreater(
            grad_sum(tree.episodic_memory.retriever.parameters()), 0.0
        )

    def test_checkpoint_rejects_attention_memory_kind(self):
        torch.manual_seed(5)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "integrated.pt"
            trainer = MemoryTreeTrainer(
                tree,
                hawkes,
                encoder,
                training=TrainingConfig(epochs=1, checkpoint_path=str(checkpoint)),
                device="cpu",
            )
            trainer.save_checkpoint(checkpoint, epoch=0)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["model_config"]["encoder_config"] = {"kind": "attention_memory"}
            torch.save(payload, checkpoint)
            with self.assertRaises(ValueError):
                MemoryTreeInference.from_checkpoint(checkpoint, device="cpu")
            with self.assertRaises(ValueError):
                MemoryTreeTrainer.from_checkpoint(checkpoint, device="cpu")

    def test_checkpoint_roundtrip_causal_encoder(self):
        torch.manual_seed(6)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "causal.pt"
            trainer = MemoryTreeTrainer(
                tree,
                hawkes,
                encoder,
                training=TrainingConfig(epochs=1, checkpoint_path=str(checkpoint)),
                device="cpu",
            )
            trainer.save_checkpoint(checkpoint, epoch=0)
            inference = MemoryTreeInference.from_checkpoint(checkpoint, device="cpu")
            self.assertIsInstance(inference.encoder, CausalPrefixEncoder)
            z_t = inference.encoder(
                torch.tensor([0.2, 0.6]),
                torch.tensor([1, 0]),
                2,
            )
            self.assertEqual(z_t.shape, (3,))
            self.assertTrue(torch.isfinite(z_t).all())


if __name__ == "__main__":
    unittest.main()
