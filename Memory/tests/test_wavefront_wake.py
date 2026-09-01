import copy
import unittest

import torch

from HawkesBackbone import HawkesFamily
from LatentHawkesTree import HawkesTree
from MemoryResiduals.EpisodicMemory import TreeEpisodicMemory
from MemoryResiduals.WorkingMemory import WorkingMemoryAdapter
from Train.Train import (
    CausalPrefixEncoder,
    MemoryTreeTrainer,
    WakeObjectiveConfig,
)
from Wake.SequentialController import Controller


class MaskedWavefrontWakeTests(unittest.TestCase):
    @staticmethod
    def _trainer(seed: int = 0) -> MemoryTreeTrainer:
        torch.manual_seed(seed)
        hawkes = HawkesFamily(
            2,
            1,
            decays=torch.tensor([1.0]),
        )
        tree = HawkesTree(
            3,
            4,
            2,
            1,
            init_depth=1,
            memory_key_dim=3,
        )
        encoder = CausalPrefixEncoder(
            2,
            3,
            type_dim=4,
            hidden_dim=8,
        )
        return MemoryTreeTrainer(
            tree,
            hawkes,
            encoder,
            wake=WakeObjectiveConfig(
                route_balance_batch_size=2,
                write_horizon=2,
            ),
            device="cpu",
        )

    @staticmethod
    def _cached(
        trainer: MemoryTreeTrainer,
        times,
        types,
    ):
        sequence = {
            "times": torch.tensor(times),
            "types": torch.tensor(types),
        }
        trainer.hawkes.prepare_sequence_cache(sequence, inplace=True)
        return sequence

    @staticmethod
    def _run_prepared_batch(
        trainer: MemoryTreeTrainer,
        sequences,
    ):
        prepared = next(
            trainer._iter_masked_wavefront_batches(
                sequences,
                list(range(len(sequences))),
            )
        )
        return trainer.train_wake_batch(
            sequences=prepared["sequences"],
            sequence_indices=prepared["sequence_indices"],
            z_flat=prepared["z_flat"],
            projected_flat=prepared["projected_flat"],
            query_flat=prepared["query_flat"],
            frontier_static_cache=prepared["frontier_static_cache"],
            frontier_flat=prepared["frontier_flat"],
            frontier_rows=prepared["frontier_rows"],
            flat=prepared["flat"],
        )

    def test_batched_working_update_matches_independent_rows(self):
        torch.manual_seed(401)
        adapter = WorkingMemoryAdapter(
            param_dim=7,
            rho=0.73,
            eta=0.04,
            clip_grad_norm=0.6,
        )
        state = torch.randn(4, 7)
        original = state.clone()
        active = torch.tensor([0, 2, 3])
        gradients = torch.randn(3, 7)
        probabilities = torch.tensor([0.2, 0.8, 0.5])

        adapter.update_batch_rows(
            state,
            active,
            gradients,
            adaptation_probability=probabilities,
        )
        expected = original.clone()
        for local_row, state_row in enumerate(active.tolist()):
            scalar = WorkingMemoryAdapter(
                param_dim=7,
                rho=adapter.rho,
                eta=adapter.eta,
                clip_grad_norm=adapter.clip_grad_norm,
            )
            scalar.delta.copy_(original[state_row])
            scalar.update_from_gradient(
                gradients[local_row],
                adaptation_probability=probabilities[local_row],
            )
            expected[state_row] = scalar.delta

        self.assertTrue(torch.allclose(state, expected, atol=1e-7))
        self.assertTrue(torch.equal(state[1], original[1]))

    def test_packed_novelty_matches_scalar_controller(self):
        torch.manual_seed(409)
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        node_ids = ("root", "left", "right")
        memory.sync_nodes(node_ids)
        for _ in range(3):
            memory.add_memory(
                "left",
                torch.randn(3),
                torch.randn(memory.param_dim),
            )
        memory.add_memory(
            "right",
            torch.randn(3),
            torch.randn(memory.param_dim),
        )
        controller = Controller(
            nll_fn=HawkesFamily(
                2,
                1,
                decays=torch.tensor([1.0]),
            ),
            episodic_memory=memory,
        )
        query = torch.randn(3, 3)
        owner_indices = torch.tensor([0, 1, 2])
        packed = memory.novelty_count_packed(
            query,
            owner_indices,
            node_ids,
            temperature=controller.novelty_temperature,
            count_exponent=controller.count_exponent,
            eps=controller.controller_eps,
            count_similarity_low=controller.count_similarity_low,
            count_similarity_high=controller.count_similarity_high,
            count_topk=controller.count_topk,
            count_saturation=controller.count_saturation,
        )
        reference = [
            torch.stack(values)
            for values in zip(*[
                controller.leaf_novelty_count(query[row], node_ids[node])
                for row, node in enumerate(owner_indices.tolist())
            ])
        ]
        for actual, expected in zip(packed, reference):
            self.assertTrue(torch.allclose(
                actual,
                expected,
                atol=1e-6,
                rtol=1e-6,
            ))

    def test_wavefront_batch_size_one_matches_streaming_wake(self):
        base = self._trainer(seed=419)
        streaming = copy.deepcopy(base)
        wavefront = copy.deepcopy(base)
        sequence_stream = self._cached(
            streaming,
            [0.1, 0.4, 0.9],
            [0, 1, 0],
        )
        sequence_wave = self._cached(
            wavefront,
            [0.1, 0.4, 0.9],
            [0, 1, 0],
        )
        expected = streaming.train_wake_sequence(sequence_stream)
        actual = self._run_prepared_batch(
            wavefront,
            [sequence_wave],
        )[0]

        for key in (
            "prediction_nll",
            "wm_penalty",
            "write_penalty",
            "max_gradient_norm",
            "mean_novelty",
            "mean_max_similarity",
            "posterior_entropy",
            "prior_posterior_kl",
        ):
            self.assertAlmostEqual(expected[key], actual[key], places=6)
        self.assertEqual(
            expected["action_counts"],
            actual["action_counts"],
        )
        self.assertEqual(
            expected["memory_assignment_counts"],
            actual["memory_assignment_counts"],
        )

    def test_variable_length_wavefront_masks_finished_rows(self):
        trainer = self._trainer(seed=421)
        sequences = [
            self._cached(trainer, [0.1, 0.4, 0.9], [0, 1, 0]),
            self._cached(trainer, [0.2, 0.5], [1, 0]),
        ]
        tree_calls = []
        hook = trainer.tree.register_forward_hook(
            lambda *_: tree_calls.append(1)
        )
        results = self._run_prepared_batch(trainer, sequences)
        hook.remove()

        self.assertEqual(
            [result["event_count"] for result in results],
            [3, 2],
        )
        # One static tree evaluation serves the complete flat transaction;
        # the recurrent time loop only recomposes its GPU state.
        self.assertEqual(len(tree_calls), 1)
        self.assertEqual(
            [
                sum(result["action_counts"].values())
                for result in results
            ],
            [3, 2],
        )
        for result in results:
            self.assertEqual(
                result["sequence_responsibility"].shape,
                (len(trainer.tree.leaf_ids),),
            )

    def test_flat_wake_retrieval_is_chunked_once_before_time_loop(self):
        trainer = self._trainer(seed=423)
        trainer.wake_config.retrieval_microbatch = 2
        sequences = [
            self._cached(trainer, [0.1, 0.4, 0.9], [0, 1, 0]),
            self._cached(trainer, [0.2, 0.5], [1, 0]),
        ]
        calls = []
        memory = trainer.tree.episodic_memory
        original_read_packed = memory.read_packed

        def counted_read_packed(*args, **kwargs):
            query = kwargs["query"] if "query" in kwargs else args[0]
            calls.append(query.size(0))
            return original_read_packed(*args, **kwargs)

        memory.read_packed = counted_read_packed
        try:
            results = self._run_prepared_batch(trainer, sequences)
        finally:
            memory.read_packed = original_read_packed

        self.assertEqual(calls, [2, 2, 1])
        self.assertEqual(
            [result["event_count"] for result in results],
            [3, 2],
        )


if __name__ == "__main__":
    unittest.main()
