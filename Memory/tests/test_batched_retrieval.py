import copy
import unittest

import torch

from MemoryResiduals.MemoryBank import (
    SmoothSparseRetriever,
    entmax15_1d,
    entmax15_masked,
)
from MemoryResiduals.EpisodicMemory import TreeEpisodicMemory


class MaskedEntmaxTests(unittest.TestCase):
    def test_packed_read_with_no_active_memories_stays_zero(self):
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        query = torch.randn(2, 3)
        node_ids = ("root", "root_L")
        indices = torch.tensor([[0, 1], [0, -1]])
        mask = indices >= 0

        retrieved, info = memory.read_packed(
            query,
            indices,
            mask,
            node_ids,
            update_state=False,
        )

        self.assertEqual(retrieved.shape, (2, 2, memory.param_dim))
        self.assertEqual(info["alpha"].shape[:2], (2, 2))
        self.assertEqual(float(retrieved.detach().abs().sum()), 0.0)
        self.assertEqual(float(info["alpha"].detach().abs().sum()), 0.0)

    def test_packed_retrieval_credit_matches_python_path_reference(self):
        torch.manual_seed(83)
        reference = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        node_ids = ("root", "root_L", "root_R")
        for node_id in node_ids:
            for _ in range(2):
                reference.add_memory(
                    node_id,
                    torch.randn(3),
                    torch.randn(reference.param_dim),
                )
        packed_memory = copy.deepcopy(reference)
        query = torch.randn(1, 3)
        visited = torch.tensor([[0, 1, 2]])
        visited_mask = torch.ones_like(visited, dtype=torch.bool)
        _, packed_info = packed_memory.read_packed(
            query,
            visited,
            visited_mask,
            node_ids,
            update_state=False,
        )
        _, reference_info = reference.read_packed(
            query,
            visited,
            visited_mask,
            node_ids,
            update_state=False,
        )
        python_info = [{
            node_id: {
                "alpha": reference_info["alpha"][0, node_index]
            }
            for node_index, node_id in enumerate(node_ids)
        }]
        weights = torch.tensor([[0.4, 0.6]])
        retrieve_probability = torch.tensor(0.7)
        reference.credit_retrieval(
            info_by_batch=python_info,
            leaf_paths=[
                ("root", "root_L"),
                ("root", "root_R"),
            ],
            routing_weights=weights,
            retrieval_probability=retrieve_probability,
        )
        packed_memory.credit_retrieval_packed(
            alpha=packed_info["alpha"],
            visited_node_indices=visited,
            visited_node_mask=visited_mask,
            path_incidence=torch.tensor([[
                [True, True, False],
                [True, False, True],
            ]]),
            routing_weights=weights,
            retrieval_probability=retrieve_probability,
            node_ids=node_ids,
        )
        for node_id in node_ids:
            self.assertTrue(torch.allclose(
                reference.banks[node_id].cycle_usage,
                packed_memory.banks[node_id].cycle_usage,
                atol=1e-7,
                rtol=1e-6,
            ))

    def test_packed_bank_mirror_is_reused_and_invalidated_by_write(self):
        torch.manual_seed(89)
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        node_ids = ("root", "root_L")
        memory.add_memory(
            "root", torch.randn(3), torch.randn(memory.param_dim)
        )
        query = torch.randn(2, 3)
        indices = torch.tensor([[0, 1], [0, -1]])
        mask = indices >= 0

        memory.read_packed(
            query, indices, mask, node_ids, update_state=False
        )
        self.assertEqual(memory._packed_mirror_rebuilds, 1)
        memory.step_age()
        memory.read_packed(
            query, indices, mask, node_ids, update_state=False
        )
        self.assertEqual(memory._packed_mirror_rebuilds, 1)

        memory.add_memory(
            "root_L", torch.randn(3), torch.randn(memory.param_dim)
        )
        memory.read_packed(
            query, indices, mask, node_ids, update_state=False
        )
        self.assertEqual(memory._packed_mirror_rebuilds, 2)

    def test_packed_visited_node_read_matches_reference_calls(self):
        torch.manual_seed(97)
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        node_ids = ("root", "root_L", "root_R")
        for node_index, node_id in enumerate(node_ids):
            for _ in range(node_index + 1):
                memory.add_memory(
                    node_id,
                    torch.randn(3),
                    torch.randn(memory.param_dim),
                )
        query = torch.randn(2, 3)
        indices = torch.tensor([[0, 1, -1], [0, 2, 1]])
        mask = indices >= 0
        packed, _ = memory.read_packed(
            query, indices, mask, node_ids, update_state=False
        )
        for row in range(query.size(0)):
            active = indices[row, mask[row]].tolist()
            delta_by_node, _ = memory.read_nodes(
                query[row],
                [node_ids[index] for index in active],
                update_state=False,
            )
            reference = torch.stack([
                delta_by_node[node_ids[index]] for index in active
            ])
            self.assertTrue(torch.allclose(
                packed[row, mask[row]], reference, atol=1e-6
            ))
            self.assertEqual(
                float(
                    packed[row, ~mask[row]].detach().abs().sum()
                ),
                0.0,
            )

    def test_padded_rows_match_independent_entmax_forward_and_backward(self):
        torch.manual_seed(101)
        lengths = (1, 3, 6, 4)
        logits_batch = torch.randn(4, 6, requires_grad=True)
        valid_mask = (
            torch.arange(6).unsqueeze(0)
            < torch.tensor(lengths).unsqueeze(1)
        )
        batched = entmax15_masked(logits_batch, valid_mask)

        logits_reference = logits_batch.detach().clone().requires_grad_(True)
        reference_rows = []
        for row, length in zip(logits_reference, lengths):
            probabilities = entmax15_1d(row[:length])
            reference_rows.append(torch.nn.functional.pad(
                probabilities,
                (0, 6 - length),
            ))
        reference = torch.stack(reference_rows)
        self.assertTrue(torch.allclose(batched, reference, atol=1e-7, rtol=1e-6))

        weights = torch.randn_like(batched)
        (batched * weights).sum().backward()
        (reference * weights).sum().backward()
        self.assertTrue(torch.allclose(
            logits_batch.grad,
            logits_reference.grad,
            atol=1e-7,
            rtol=1e-6,
        ))

    def test_batched_retriever_matches_independent_bank_calls_and_gradients(self):
        torch.manual_seed(103)
        row_count, width, key_dim, param_dim = 5, 7, 4, 9
        lengths = (1, 4, 7, 2, 5)
        valid_mask = (
            torch.arange(width).unsqueeze(0)
            < torch.tensor(lengths).unsqueeze(1)
        )
        keys = torch.randn(row_count, width, key_dim)
        deltas = torch.randn(row_count, width, param_dim)
        usage = torch.rand(row_count, width)
        age = torch.rand(row_count, width) * 10.0

        batched_retriever = SmoothSparseRetriever()
        reference_retriever = copy.deepcopy(batched_retriever)
        batched_query = torch.randn(
            row_count, key_dim, requires_grad=True
        )
        reference_query = batched_query.detach().clone().requires_grad_(True)

        batched_delta, batched_info = batched_retriever.forward_batched(
            query=batched_query,
            keys=keys,
            deltas=deltas,
            usage=usage,
            age=age,
            valid_mask=valid_mask,
        )
        reference_delta = []
        reference_alpha = []
        for row_index, length in enumerate(lengths):
            delta, info = reference_retriever(
                query=reference_query[row_index],
                keys=keys[row_index, :length],
                deltas=deltas[row_index, :length],
                usage=usage[row_index, :length],
                age=age[row_index, :length],
            )
            reference_delta.append(delta)
            reference_alpha.append(torch.nn.functional.pad(
                info["alpha"],
                (0, width - length),
            ))
        reference_delta = torch.stack(reference_delta)
        reference_alpha = torch.stack(reference_alpha)
        self.assertTrue(torch.allclose(
            batched_delta, reference_delta, atol=1e-6, rtol=1e-5
        ))
        self.assertTrue(torch.allclose(
            batched_info["alpha"],
            reference_alpha,
            atol=1e-6,
            rtol=1e-5,
        ))

        objective_weight = torch.randn_like(batched_delta)
        (batched_delta * objective_weight).sum().backward()
        (reference_delta * objective_weight).sum().backward()
        self.assertTrue(torch.allclose(
            batched_query.grad,
            reference_query.grad,
            atol=2e-6,
            rtol=2e-5,
        ))
        for batched_parameter, reference_parameter in zip(
            batched_retriever.parameters(),
            reference_retriever.parameters(),
        ):
            self.assertTrue(torch.allclose(
                batched_parameter.grad,
                reference_parameter.grad,
                atol=2e-6,
                rtol=2e-5,
            ))


if __name__ == "__main__":
    unittest.main()
