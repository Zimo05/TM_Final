import tempfile
import unittest
from pathlib import Path

import torch

from HawkesBackbone import (
    EVENT_TIME_FEATURES_KEY,
    HAWKES_CACHE_SIGNATURE_KEY,
    HAWKES_HISTORY_STATS_KEY,
    HAWKES_INTERVAL_STATS_KEY,
    HawkesFamily,
)
from LatentHawkesTree import HawkesTree
from MemoryResiduals.MemoryBank import (
    EventWindow,
    SmoothSparseRetriever,
    entmax15_1d,
)
from Train.Inference import InferenceConfig, MemoryTreeInference
from Train.AlignmentInitialization import (
    LocalBranchTeacher,
    build_local_branch_teacher,
    membership_alignment_loss,
    prefix_completion_weight,
    run_membership_alignment,
    semantic_branch_logits,
)
from Train.ResidualInitialization import (
    aggregate_leaf_residual_prototypes,
    compute_sequence_residual_signatures,
    initialize_tree_from_residual_signatures,
    load_h_tree_leaf_membership,
)
from Train.Train import (
    CausalPrefixEncoder,
    MemoryTreeTrainer,
    SleepConfig,
    StructureConfig,
    TrainingConfig,
    WakeObjectiveConfig,
    child_teacher_reliability,
    clip_grad_norm_finite,
    marginal_route_balance_kl,
    normalize_cuda_rng_states,
    reliability_gated_route_state,
    routing_diagnostics,
    sequence_route_information,
)
from Wake.HawkesParams import HawkesParams
from Wake.SequentialController import Action
from Wake.SequentialController import Controller
from Routing_Retrieval_Investigation.routing_retrieval_investigation import (
    FrontierRoutingConfig,
)


class TrainInferenceTests(unittest.TestCase):
    def test_cached_sequence_features_match_uncached_hawkes_value_and_gradient(self):
        hawkes = HawkesFamily(
            3,
            2,
            decays=torch.tensor([0.5, 1.5]),
        )
        # Repeated timestamps verify the original strict-time intensity rule.
        sequence = {
            "times": torch.tensor([0.1, 0.1, 0.4, 0.9]),
            "types": torch.tensor([0, 1, 2, 0]),
        }
        cached = hawkes.prepare_sequence_cache(sequence, inplace=False)
        self.assertEqual(cached[EVENT_TIME_FEATURES_KEY].shape, (4, 2))
        self.assertEqual(cached[HAWKES_HISTORY_STATS_KEY].shape, (4, 3, 2))
        self.assertEqual(cached[HAWKES_INTERVAL_STATS_KEY].shape, (4, 3, 2))

        raw_mu = torch.randn(3)
        raw_W = torch.randn(3, 3, 2)

        def value_and_grad(input_sequence):
            mu = raw_mu.clone().requires_grad_(True)
            W = raw_W.clone().requires_grad_(True)
            params = HawkesParams(mu, W)
            loss = torch.stack([
                hawkes.event_NLL(input_sequence, params, event_index)
                for event_index in range(4)
            ]).sum()
            gradients = torch.autograd.grad(loss, (mu, W))
            return loss.detach(), tuple(item.detach() for item in gradients)

        uncached_value, uncached_gradients = value_and_grad(sequence)
        cached_value, cached_gradients = value_and_grad(cached)
        self.assertTrue(
            torch.allclose(cached_value, uncached_value, atol=1e-6, rtol=1e-6)
        )
        for cached_gradient, uncached_gradient in zip(
            cached_gradients,
            uncached_gradients,
        ):
            self.assertTrue(torch.allclose(
                cached_gradient,
                uncached_gradient,
                atol=1e-6,
                rtol=1e-6,
            ))

        params = HawkesParams(raw_mu, raw_W)
        history = {
            "times": sequence["times"][:3],
            "types": sequence["types"][:3],
        }
        uncached_intensity = hawkes.intensity_at_event(
            history,
            sequence["times"][3],
            params,
        )
        cached_intensity = hawkes.intensity_at_cached_event(cached, 3, params)
        self.assertTrue(torch.allclose(
            cached_intensity,
            uncached_intensity,
            atol=1e-6,
            rtol=1e-6,
        ))

    def test_cached_event_time_features_preserve_encoder_output(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        sequence = {
            "times": torch.tensor([0.1, 0.4, 0.9]),
            "types": torch.tensor([0, 1, 0]),
        }
        cached = hawkes.prepare_sequence_cache(sequence, inplace=False)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        uncached_output = encoder(
            sequence["times"],
            sequence["types"],
            3,
        )
        cached_output = encoder(
            sequence["times"],
            sequence["types"],
            3,
            time_features=cached[EVENT_TIME_FEATURES_KEY],
        )
        self.assertTrue(torch.allclose(
            cached_output,
            uncached_output,
            atol=1e-7,
            rtol=1e-7,
        ))

    def test_resident_sequence_cache_skips_revalidation_and_device_moves(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        trainer = MemoryTreeTrainer(tree, hawkes, encoder, device="cpu")
        resident = hawkes.prepare_sequence_cache(
            {
                "times": torch.tensor([0.1, 0.4, 0.9]),
                "types": torch.tensor([0, 1, 0]),
            },
            inplace=False,
        )

        moved = trainer._move_sequence(resident)
        self.assertEqual(trainer._resident_cache_hits, 1)
        self.assertEqual(trainer._resident_cache_misses, 0)
        self.assertIs(
            moved[HAWKES_HISTORY_STATS_KEY],
            resident[HAWKES_HISTORY_STATS_KEY],
        )

        uncached = dict(resident)
        uncached.pop(HAWKES_CACHE_SIGNATURE_KEY)
        trainer._move_sequence(uncached)
        self.assertEqual(trainer._resident_cache_misses, 1)

    def test_forward_all_prefix_matches_individual_strict_prefix_calls(self):
        torch.manual_seed(5)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        sequence = {
            "times": torch.tensor([0.1, 0.4, 0.9, 1.2]),
            "types": torch.tensor([0, 1, 0, 1]),
        }
        cached = hawkes.prepare_sequence_cache(sequence, inplace=False)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        expected = torch.stack([
            encoder(
                sequence["times"],
                sequence["types"],
                event_index,
                time_features=cached[EVENT_TIME_FEATURES_KEY],
            )
            for event_index in range(sequence["times"].numel())
        ])
        actual = encoder.forward_all_prefix(
            sequence["times"],
            sequence["types"],
            time_features=cached[EVENT_TIME_FEATURES_KEY],
        )
        self.assertTrue(torch.allclose(
            actual,
            expected,
            atol=1e-7,
            rtol=1e-7,
        ))

        changed_types = sequence["types"].clone()
        changed_types[2] = 1 - changed_types[2]
        changed = encoder.forward_all_prefix(
            sequence["times"],
            changed_types,
            time_features=cached[EVENT_TIME_FEATURES_KEY],
        )
        # Changing event 2 may first affect the prediction at event 3.
        self.assertTrue(torch.equal(actual[:3], changed[:3]))

    def test_entmax_sparse_boundary_has_finite_closed_form_backward(self):
        # Large score gaps create inactive support candidates with delta=0.
        # Differentiating the old sqrt-based threshold search produced
        # 0 * inf = NaN for exactly this normal sparse-routing state.
        logits = torch.tensor(
            [100.0, 0.0, -100.0],
            requires_grad=True,
        )
        probabilities = entmax15_1d(logits)
        loss = (
            probabilities
            * torch.tensor([0.2, -0.4, 0.7])
        ).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertAlmostEqual(
            float(probabilities.sum().detach()),
            1.0,
            places=6,
        )
        self.assertEqual(int((probabilities > 0).sum()), 1)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_entmax_closed_form_backward_matches_finite_differences(self):
        logits = torch.tensor(
            [0.4, 0.1, -0.2],
            dtype=torch.float64,
            requires_grad=True,
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                entmax15_1d,
                (logits,),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-4,
            )
        )

    def test_sparse_retriever_boundary_keeps_query_and_parameters_finite(self):
        retriever = SmoothSparseRetriever(init_gamma=100.0)
        query = torch.tensor([1.0, 0.0], requires_grad=True)
        keys = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        deltas = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.5],
        ])
        retrieved, _ = retriever(
            query=query,
            keys=keys,
            deltas=deltas,
            usage=torch.tensor([0.0, 1.0, 2.0]),
            age=torch.tensor([0.0, 3.0, 8.0]),
        )
        retrieved.square().sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        for parameter in retriever.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_cuda_rng_checkpoint_states_are_normalized_for_visible_devices(self):
        saved = [
            torch.tensor([0, 1, 255], dtype=torch.uint8),
            torch.tensor([4, 5, 6], dtype=torch.uint8),
        ]
        normalized = normalize_cuda_rng_states(saved, max_devices=1)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].device.type, "cpu")
        self.assertEqual(normalized[0].dtype, torch.uint8)
        self.assertTrue(normalized[0].is_contiguous())
        self.assertTrue(torch.equal(normalized[0], saved[0]))

    def test_stable_gradient_clip_handles_finite_float32_norm_overflow(self):
        parameter = torch.nn.Parameter(torch.zeros(4))
        parameter.grad = torch.full_like(parameter, 1e30)
        total_norm = clip_grad_norm_finite(
            {"large_parameter": parameter},
            5.0,
            context="unit test",
        )
        self.assertTrue(torch.isfinite(torch.tensor(total_norm, dtype=torch.float64)))
        self.assertAlmostEqual(float(parameter.grad.double().norm()), 5.0, places=5)

    def test_stable_gradient_clip_names_actual_nonfinite_gradient(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([float("nan"), 1.0])
        with self.assertRaisesRegex(
            FloatingPointError,
            r"unit test.*large_parameter.*nan=1",
        ):
            clip_grad_norm_finite(
                {"large_parameter": parameter},
                5.0,
                context="unit test",
            )

    def test_episodic_write_clips_source_gradient_before_persistence(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        controller = Controller(
            nll_fn=hawkes,
            tau_s=2.0,
            tau_n=0.3,
            tau_c=3,
            tau_sim=0.5,
            eta_mem=1e-2,
            memory_write_grad_clip=5.0,
        )

        def large_linear_nll(sequence, params, k):
            return 1e30 * (
                params.mu_tilde.sum() + params.W_tilde.sum()
            )

        controller.nll_fn.event_NLL = large_linear_nll
        theta = HawkesParams(
            torch.zeros(2),
            torch.zeros(2, 2, 1),
        )
        item = controller.write_residual_memory(
            q_t=torch.tensor([1.0, 0.0, 0.0]),
            theta_sem_leaf=theta,
            times=torch.tensor([0.1]),
            types=torch.tensor([0]),
            k=0,
            node_id="root",
        )
        self.assertTrue(torch.isfinite(item.delta_theta).all())
        self.assertLessEqual(
            float(item.delta_theta.double().norm()),
            0.05 + 1e-6,
        )

    def test_batched_residual_writes_match_independent_gradients(self):
        torch.manual_seed(307)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        controller = Controller(
            nll_fn=hawkes,
            eta_mem=1e-2,
            memory_write_grad_clip=5.0,
            write_horizon=2,
        )
        sequence = {
            "times": torch.tensor([0.1, 0.4, 0.8, 1.2]),
            "types": torch.tensor([0, 1, 0, 1]),
        }
        hawkes.prepare_sequence_cache(sequence, inplace=True)
        queries = torch.randn(2, 3)
        mu = torch.randn(2, 2)
        W = torch.randn(2, 2, 2, 1)
        indices = [0, 1]
        node_ids = ["root", "root_L"]
        independent = [
            controller.write_residual_memory(
                q_t=queries[row],
                theta_sem_leaf=HawkesParams(mu[row], W[row]),
                times=sequence["times"],
                types=sequence["types"],
                k=indices[row],
                node_id=node_ids[row],
                cached_sequence=sequence,
                write_quality=0.7 + 0.1 * row,
                queue_weight=0.2 * row,
            )
            for row in range(2)
        ]
        batched = controller.write_residual_memory_batch(
            queries=queries,
            theta_semantic=HawkesParams(mu, W),
            times=sequence["times"],
            types=sequence["types"],
            event_indices=indices,
            node_ids=node_ids,
            cached_sequence=sequence,
            write_quality=torch.tensor([0.7, 0.8]),
            queue_weight=torch.tensor([0.0, 0.2]),
        )
        for expected, actual in zip(independent, batched):
            self.assertTrue(torch.allclose(
                expected.delta_theta,
                actual.delta_theta,
                atol=1e-7,
                rtol=1e-6,
            ))
            self.assertTrue(torch.equal(expected.window.times, actual.window.times))
            self.assertEqual(expected.window.start_idx, actual.window.start_idx)
            self.assertAlmostEqual(
                expected.write_quality, actual.write_quality, places=6
            )

    def test_batch_marginal_balance_does_not_force_each_sample_uniform(self):
        responsibilities = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        balance = marginal_route_balance_kl(responsibilities)
        diagnostics = routing_diagnostics(responsibilities)
        self.assertAlmostEqual(float(balance), 0.0, places=6)
        self.assertAlmostEqual(
            diagnostics["marginal_entropy"],
            float(torch.log(torch.tensor(2.0))),
            places=6,
        )
        self.assertGreater(diagnostics["mutual_information"], 0.0)
        self.assertEqual(diagnostics["hard_counts"], [1, 1])

    def test_sequence_route_information_rewards_confident_diverse_routes(self):
        complementary = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        ambiguous = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        complementary_info = sequence_route_information(complementary)
        ambiguous_info = sequence_route_information(ambiguous)
        self.assertAlmostEqual(
            float(complementary_info["prior_kl"]),
            0.0,
            places=6,
        )
        self.assertGreater(
            float(complementary_info["mutual_information"]),
            float(ambiguous_info["mutual_information"]),
        )

    def test_router_explicitly_depends_on_node_semantics(self):
        torch.manual_seed(171)
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        tree.initialize_router_weights(gain=0.2, seed=17)
        z_t = torch.randn(5, tree.z_dim)
        route_before = tree.route(z_t).responsibility.detach().clone()
        with torch.no_grad():
            tree.node_emb["root_L"].add_(
                3.0 * torch.randn_like(tree.node_emb["root_L"])
            )
        route_after = tree.route(z_t).responsibility.detach()
        self.assertGreater(
            float((route_after - route_before).abs().max()),
            1e-6,
        )

    def test_leaf_symmetry_breaking_is_zero_mean_and_internal_nodes_stay_exact(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=2, memory_key_dim=3)
        target = tree.initialize_semantics_from_hawkes(hawkes)
        internal_before = {
            node_id: tree.semantic_theta(node_id).detach().clone()
            for node_id in tree.internal_ids
        }
        stats = tree.break_initial_leaf_symmetry(
            relative_scale=0.01,
            seed=19,
        )
        leaves = torch.stack([
            tree.semantic_theta(leaf_id)
            for leaf_id in tree.leaf_ids
        ])
        self.assertTrue(torch.allclose(
            leaves.mean(dim=0),
            target,
            atol=1e-6,
            rtol=1e-6,
        ))
        self.assertGreater(
            float(leaves.std(dim=0).mean().detach()),
            0.0,
        )
        self.assertLess(stats["mean_error"], 1e-6)
        for node_id, expected in internal_before.items():
            self.assertTrue(torch.equal(
                tree.semantic_theta(node_id).detach(),
                expected,
            ))

    def test_h_tree_membership_uses_cluster_and_source_metadata(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        dataset = [
            {
                "times": torch.tensor([0.1]),
                "types": torch.tensor([0]),
                "cluster_id": torch.tensor(7),
                "source_index": torch.tensor(41),
            },
            {
                "times": torch.tensor([0.2]),
                "types": torch.tensor([1]),
                "cluster_id": torch.tensor(3),
                "source_index": torch.tensor(19),
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "sequence_summary.csv"
            summary.write_text(
                "leaf_position,cluster_id,sequences\n"
                'l,7,"[41]"\n'
                'r,3,"[19]"\n',
                encoding="utf-8",
            )
            membership = load_h_tree_leaf_membership(
                summary,
                dataset,
                tree.leaf_ids,
            )

        self.assertEqual(tuple(membership.shape), (2, 2))
        self.assertTrue(torch.equal(
            membership,
            torch.eye(2),
        ))

    def test_leaf_membership_decomposes_into_local_binary_teachers(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=2, memory_key_dim=3)
        membership = torch.tensor([
            [0.70, 0.10, 0.15, 0.05],
        ])
        teacher = build_local_branch_teacher(tree, membership)

        self.assertEqual(
            teacher.internal_ids,
            ("root", "root_L", "root_R"),
        )
        self.assertTrue(torch.allclose(
            teacher.target[0, 0],
            torch.tensor([0.8, 0.2]),
        ))
        self.assertTrue(torch.allclose(
            teacher.target[0, 1],
            torch.tensor([0.875, 0.125]),
        ))
        self.assertTrue(torch.allclose(
            teacher.target[0, 2],
            torch.tensor([0.75, 0.25]),
        ))
        self.assertTrue(torch.allclose(
            teacher.node_mass[0],
            torch.tensor([1.0, 0.8, 0.2]),
        ))

    def test_alignment_prefix_weight_is_smooth_and_ignores_empty_prefix(self):
        valid = torch.tensor([
            [True, True, True, True],
            [True, True, False, False],
        ])
        weight = prefix_completion_weight(valid)
        self.assertTrue(torch.allclose(
            weight[0],
            torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]),
        ))
        self.assertTrue(torch.equal(
            weight[1],
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
        ))

    def test_alignment_loss_uses_semantic_logits_and_teacher_weights(self):
        teacher = LocalBranchTeacher(
            target=torch.tensor([[[1.0, 0.0], [0.5, 0.5]]]),
            node_mass=torch.tensor([[1.0, 1.0]]),
            confidence=torch.tensor([[1.0, 0.0]]),
            internal_ids=("root", "ambiguous"),
        )
        valid = torch.tensor([[True, True, True]])
        good = torch.zeros(1, 3, 2, 2)
        good[:, :, 0] = torch.tensor([5.0, -5.0])
        # This branch has confidence zero and must not affect the objective.
        good[:, :, 1] = torch.tensor([-100.0, 100.0])
        good_loss, _ = membership_alignment_loss(
            good,
            teacher,
            valid,
            temperature=1.0,
        )
        bad = good.clone()
        bad[:, 1:, 0] = torch.tensor([-5.0, 5.0])
        bad_loss, _ = membership_alignment_loss(
            bad,
            teacher,
            valid,
            temperature=1.0,
        )
        empty_prefix_changed = good.clone()
        empty_prefix_changed[:, 0, 0] = torch.tensor([-100.0, 100.0])
        empty_loss, _ = membership_alignment_loss(
            empty_prefix_changed,
            teacher,
            valid,
            temperature=1.0,
        )
        self.assertLess(float(good_loss), float(bad_loss))
        self.assertTrue(torch.allclose(good_loss, empty_loss))

    def test_membership_alignment_updates_only_encoder_and_compatibility(self):
        torch.manual_seed(83)
        tree = HawkesTree(4, 6, 2, 1, init_depth=1, memory_key_dim=4)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree.initialize_semantics_from_hawkes(hawkes)
        tree.initialize_router_weights(gain=0.05, seed=83)
        encoder = CausalPrefixEncoder(
            num_event_types=2,
            z_dim=4,
            type_dim=4,
            hidden_dim=8,
        )
        dataset = []
        membership = []
        for cluster in (0, 1):
            for sample_index in range(8):
                dataset.append({
                    "times": torch.tensor([0.1, 0.3, 0.6, 0.9]),
                    "types": torch.tensor(
                        [cluster, cluster, cluster, cluster]
                    ),
                })
                membership.append(
                    [1.0, 0.0] if cluster == 0 else [0.0, 1.0]
                )
        membership = torch.tensor(membership)
        node_before = {
            name: parameter.detach().clone()
            for name, parameter in tree.node_emb.items()
        }
        semantic_before = {
            name: parameter.detach().clone()
            for name, parameter in tree.semantic_offset.items()
        }
        encoder_before = {
            name: parameter.detach().clone()
            for name, parameter in encoder.named_parameters()
        }
        router_before = {
            name: parameter.detach().clone()
            for name, parameter in tree.router_compat.named_parameters()
        }

        stats = run_membership_alignment(
            tree,
            encoder,
            dataset,
            membership,
            epochs=8,
            batch_size=8,
            learning_rate=5e-3,
            temperature=1.0,
            grad_clip=5.0,
            seed=83,
            progress=False,
        )

        for name, expected in node_before.items():
            self.assertTrue(torch.equal(tree.node_emb[name], expected))
        for name, expected in semantic_before.items():
            self.assertTrue(torch.equal(
                tree.semantic_offset[name],
                expected,
            ))
        self.assertTrue(any(
            not torch.equal(parameter, encoder_before[name])
            for name, parameter in encoder.named_parameters()
        ))
        self.assertTrue(any(
            not torch.equal(parameter, router_before[name])
            for name, parameter in tree.router_compat.named_parameters()
        ))
        self.assertLess(stats["final_loss"], stats["initial_loss"])
        self.assertGreater(
            stats["final_target_probability"],
            stats["initial_target_probability"],
        )

    def test_child_energy_reliability_uses_confidence_not_alignment(self):
        node_indices = torch.tensor([[0], [0]])
        mask = torch.ones(2, 1, dtype=torch.bool)
        confident = torch.tensor([
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ])
        aligned = child_teacher_reliability(
            confident,
            confident,
            node_indices,
            mask,
            node_count=1,
        )
        self.assertAlmostEqual(
            float(aligned["reliability"]),
            1.0,
            places=6,
        )

        opposite = torch.tensor([
            [[0.0, 1.0]],
            [[0.0, 1.0]],
        ])
        misaligned = child_teacher_reliability(
            confident,
            opposite,
            node_indices,
            mask,
            node_count=1,
        )
        self.assertAlmostEqual(
            float(misaligned["teacher_student_js"]),
            float(torch.log(torch.tensor(2.0))),
            places=6,
        )
        self.assertAlmostEqual(
            float(misaligned["reliability"]),
            1.0,
            places=6,
        )

        uncertain = torch.full((2, 1, 2), 0.5)
        low_confidence = child_teacher_reliability(
            uncertain,
            uncertain,
            node_indices,
            mask,
            node_count=1,
        )
        self.assertAlmostEqual(
            float(low_confidence["teacher_confidence"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(low_confidence["reliability"]),
            0.0,
            places=6,
        )

    def test_reliability_gate_preserves_value_and_scales_only_gradient(self):
        source = torch.tensor([1.0, -2.0], requires_grad=True)
        routed = reliability_gated_route_state(
            source,
            reliability=0.4,
            alpha_max=0.1,
        )
        self.assertTrue(torch.equal(routed.detach(), source.detach()))
        routed.sum().backward()
        self.assertTrue(torch.allclose(
            source.grad,
            torch.full_like(source, 0.04),
        ))

    def test_child_energy_teacher_uses_fixed_prior_not_learned_mass(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        tree.configure_frontier_routing(
            config=FrontierRoutingConfig(
                frontier_budget=2,
                frontier_min_experts=2,
                target_leaf_mass=(0.8, 0.2),
            ),
        )
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        trainer = MemoryTreeTrainer(tree, hawkes, encoder, device="cpu")
        memory_output = {
            "expanded_probability": torch.tensor([
                [[0.99, 0.01]],
                [[0.01, 0.99]],
            ], requires_grad=True),
            "expanded_semantic_score": torch.zeros(
                2,
                1,
                2,
                requires_grad=True,
            ),
            "expanded_mask": torch.ones(2, 1, dtype=torch.bool),
            "expanded_node_indices": torch.zeros(2, 1, dtype=torch.long),
        }
        child_energy = torch.zeros(2, 1, 2)
        local = trainer._local_frontier_objective(
            memory_output,
            child_energy,
        )
        expected = torch.tensor([0.8, 0.2]).expand(2, 1, 2)
        self.assertTrue(torch.allclose(
            local["teacher"],
            expected,
            atol=1e-6,
            rtol=1e-6,
        ))
        self.assertAlmostEqual(
            float(local["distill"].detach()), 0.0, places=6
        )
        self.assertTrue(torch.equal(
            local["reliability"],
            torch.zeros_like(local["reliability"]),
        ))
        self.assertTrue(torch.equal(
            local["student"],
            memory_output["expanded_probability"].detach(),
        ))
        changed_mass = dict(memory_output)
        changed_mass["expanded_probability"] = torch.full(
            (2, 1, 2),
            0.5,
            requires_grad=True,
        )
        changed = trainer._local_frontier_objective(
            changed_mass,
            child_energy,
        )
        self.assertTrue(torch.equal(
            local["teacher"],
            changed["teacher"],
        ))
        self.assertTrue(torch.equal(
            changed["student"],
            changed_mass["expanded_probability"].detach(),
        ))
        self.assertFalse(torch.equal(
            local["student"],
            changed["student"],
        ))

    def test_router_distillation_detaches_semantic_node_embeddings(self):
        torch.manual_seed(719)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        trainer = MemoryTreeTrainer(tree, hawkes, encoder, device="cpu")
        output = tree(
            torch.randn(3, tree.z_dim, requires_grad=True),
            decays=hawkes.decays,
            update_memory_state=False,
            update_search_state=False,
            materialize_diagnostics=False,
        )
        child_energy = torch.tensor([
            [[0.0, 2.0]],
            [[2.0, 0.0]],
            [[0.0, 1.0]],
        ])
        local = trainer._local_frontier_objective(output, child_energy)
        local["distill"].backward()

        self.assertTrue(any(
            parameter.grad is not None
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in tree.router_compat.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            or float(parameter.grad.abs().sum()) == 0.0
            for parameter in tree.node_emb.values()
        ))

    def test_prediction_with_detached_routing_updates_experts_not_router(self):
        torch.manual_seed(727)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        output = tree(
            torch.randn(3, tree.z_dim, requires_grad=True),
            decays=hawkes.decays,
            update_memory_state=False,
            update_search_state=False,
            detach_routing=True,
            materialize_diagnostics=False,
        )
        output["effective_params"].theta.square().mean().backward()

        self.assertTrue(all(
            parameter.grad is None
            or float(parameter.grad.abs().sum()) == 0.0
            for parameter in tree.router_compat.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in tree.node_emb.values()
        ))

    def test_residual_signature_initialization_is_data_driven_and_centered(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        cold_target = tree.initialize_semantics_from_hawkes(hawkes)
        dataset = [
            {
                "times": torch.tensor([0.1, 0.2, 0.5]),
                "types": torch.tensor([0, 0, 0]),
                "T": torch.tensor(0.6),
                "cluster_id": torch.tensor(0),
                "source_index": torch.tensor(0),
            },
            {
                "times": torch.tensor([0.1, 0.7, 0.9]),
                "types": torch.tensor([1, 1, 1]),
                "T": torch.tensor(1.0),
                "cluster_id": torch.tensor(1),
                "source_index": torch.tensor(1),
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "sequence_summary.csv"
            summary.write_text(
                "leaf_position,cluster_id,sequences\n"
                'l,0,"[0]"\n'
                'r,1,"[1]"\n',
                encoding="utf-8",
            )
            stats = initialize_tree_from_residual_signatures(
                tree,
                hawkes,
                dataset,
                summary_path=summary,
                init_scale=0.01,
                lowrank_rank=1,
                grad_clip=5.0,
                progress=False,
            )

        leaves = torch.stack([
            tree.semantic_theta(leaf_id)
            for leaf_id in tree.leaf_ids
        ])
        self.assertFalse(torch.allclose(leaves[0], leaves[1]))
        self.assertTrue(torch.allclose(
            leaves.mean(dim=0),
            cold_target,
            atol=1e-6,
            rtol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            tree.semantic_theta("root"),
            cold_target,
            atol=1e-6,
            rtol=1e-6,
        ))
        self.assertLess(stats["weighted_mean_error"], 1e-6)
        self.assertLess(stats["root_error"], 1e-6)
        self.assertEqual(stats["leaf_membership_mass"], [1.0, 1.0])
        self.assertEqual(
            tree.frontier_routing.config.target_leaf_mass,
            tuple(stats["target_leaf_mass"]),
        )

    def test_residual_projection_is_rank_bounded_and_aggregates_soft_membership(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        dataset = [
            {
                "times": torch.tensor([0.1, 0.4]),
                "types": torch.tensor([0, 1]),
                "T": torch.tensor(0.5),
            },
            {
                "times": torch.tensor([0.2, 0.3]),
                "types": torch.tensor([1, 1]),
                "T": torch.tensor(0.4),
            },
        ]
        signatures, stats = compute_sequence_residual_signatures(
            hawkes,
            dataset,
            lowrank_rank=1,
            grad_clip=0.0,
            progress=False,
        )
        self.assertEqual(
            tuple(signatures.shape),
            (2, 2 + 2 * 2),
        )
        for signature in signatures:
            projected_W = signature[2:].reshape(2, 2)
            self.assertLessEqual(
                int(torch.linalg.matrix_rank(projected_W)),
                1,
            )
        membership = torch.tensor([
            [0.75, 0.25],
            [0.25, 0.75],
        ])
        prototypes, target_mass = aggregate_leaf_residual_prototypes(
            signatures,
            membership,
        )
        self.assertEqual(tuple(prototypes.shape), (2, signatures.size(1)))
        self.assertTrue(torch.allclose(
            target_mass,
            torch.tensor([0.5, 0.5]),
        ))
        self.assertEqual(stats["gradient_clipped_fraction"], 0.0)

    def test_semantic_blend_preserves_offsets_and_node_differences(self):
        torch.manual_seed(29)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=2, memory_key_dim=3)
        with torch.no_grad():
            hawkes.raw_mu.copy_(torch.tensor([-0.7, 0.2]))
            hawkes.raw_W.copy_(
                torch.arange(4.0).reshape(2, 2, 1) / 10.0
            )
        base_before = {
            node_id: tree.base_semantic_theta(node_id).detach().clone()
            for node_id in tree.all_node_ids
        }
        cold = torch.cat((hawkes.raw_mu, hawkes.raw_W.reshape(-1)))
        alpha = 0.25
        returned = tree.initialize_semantics_from_hawkes(
            hawkes,
            semantic_blend=alpha,
        )
        self.assertTrue(torch.equal(returned, cold))
        self.assertEqual(tree.semantic_blend, alpha)
        for node_id in tree.all_node_ids:
            expected = torch.lerp(cold, base_before[node_id], alpha)
            self.assertTrue(torch.allclose(
                tree.semantic_theta(node_id),
                expected,
                atol=1e-6,
                rtol=1e-6,
            ))
        self.assertTrue(any(
            bool((offset.detach().abs() > 0).any())
            for offset in tree.semantic_offset.values()
        ))

        leaves_before = torch.stack([
            tree.semantic_theta(leaf_id).detach().clone()
            for leaf_id in tree.leaf_ids
        ])
        tree.break_initial_leaf_symmetry(relative_scale=0.01, seed=31)
        leaves_after = torch.stack([
            tree.semantic_theta(leaf_id)
            for leaf_id in tree.leaf_ids
        ])
        self.assertTrue(torch.allclose(
            leaves_after.mean(dim=0),
            leaves_before.mean(dim=0),
            atol=1e-6,
            rtol=1e-6,
        ))
        centered_before = leaves_before - leaves_before.mean(dim=0)
        centered_after = leaves_after - leaves_after.mean(dim=0)
        self.assertGreater(float(centered_before.abs().max()), 0.0)
        self.assertFalse(torch.allclose(centered_after, centered_before))

    def test_leaf_paths_are_cached_and_refreshed_with_topology(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=2, memory_key_dim=3)

        def expected_paths():
            return tuple(
                tuple(tree.path_to_leaf(leaf_id))
                for leaf_id in tree.leaf_ids
            )

        self.assertEqual(tree.leaf_paths, expected_paths())
        self.assertEqual(
            tree.path_node_ids,
            tuple(dict.fromkeys(
                node_id
                for path in tree.leaf_paths
                for node_id in path
            )),
        )

        tree.split_leaf(tree.leaf_ids[0])
        self.assertEqual(tree.leaf_paths, expected_paths())
        self.assertEqual(
            tree.path_node_ids,
            tuple(dict.fromkeys(
                node_id
                for path in tree.leaf_paths
                for node_id in path
            )),
        )

    def test_lazy_event_age_matches_eager_age_and_checkpoint_rebases(self):
        tree = HawkesTree(2, 3, 1, 1, init_depth=0, memory_key_dim=2)
        memory = tree.episodic_memory
        memory.add_memory(
            "root",
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.25, -0.5]),
        )
        bank = memory.get_bank("root")

        memory.step_age()
        memory.step_age()
        # No per-event bank mutation has occurred, but effective age is exact.
        self.assertEqual(bank.age.tolist(), [0.0])
        self.assertEqual(
            bank.effective_age(memory._age_clock).tolist(),
            [2.0],
        )

        # Existing rows materialize at age 2; the new row starts at age 0.
        memory.add_memory(
            "root",
            torch.tensor([0.0, 1.0]),
            torch.tensor([-0.1, 0.4]),
        )
        self.assertEqual(bank.age.tolist(), [2.0, 0.0])
        memory.step_age()
        effective_age = bank.effective_age(memory._age_clock)
        self.assertEqual(effective_age.tolist(), [3.0, 1.0])

        query = torch.tensor([0.7, 0.3])
        lazy_delta, lazy_info = bank.retrieve(
            query,
            memory.retriever,
            update_state=False,
            age_clock=memory._age_clock,
        )
        eager_delta, eager_info = memory.retriever(
            query=query,
            keys=bank.keys,
            deltas=bank.deltas,
            usage=bank.usage,
            age=effective_age,
        )
        self.assertTrue(torch.equal(lazy_delta, eager_delta))
        self.assertTrue(torch.equal(lazy_info["alpha"], eager_info["alpha"]))

        state = memory.get_extra_state()
        self.assertEqual(state["root"]["age"].tolist(), [3.0, 1.0])
        restored = HawkesTree(
            2,
            3,
            1,
            1,
            init_depth=0,
            memory_key_dim=2,
        ).episodic_memory
        restored.set_extra_state(state)
        restored.step_age()
        restored_bank = restored.get_bank("root")
        self.assertEqual(
            restored_bank.effective_age(restored._age_clock).tolist(),
            [4.0, 2.0],
        )
        restored.materialize_all_ages()
        self.assertEqual(restored_bank.age.tolist(), [4.0, 2.0])

    def test_router_mi_encoder_gradient_is_staged(self):
        sequence = {
            "times": torch.tensor([0.1, 0.4, 0.9]),
            "types": torch.tensor([0, 1, 0]),
        }
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        trainer = MemoryTreeTrainer(tree, hawkes, encoder, device="cpu")
        moved = trainer._move_sequence(sequence)

        route = trainer._sequence_route_mean_for_router(
            moved,
            encoder_grad_scale=0.0,
        )
        active_index = int(route.detach().argmax())
        route[active_index].backward()
        self.assertTrue(all(
            parameter.grad is None
            for parameter in encoder.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in tree.router_compat.parameters()
        ))

        trainer.optimizer.zero_grad(set_to_none=True)
        route = trainer._sequence_route_mean_for_router(
            moved,
            encoder_grad_scale=0.1,
        )
        route[active_index].backward()
        self.assertTrue(any(
            parameter.grad is not None
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in encoder.parameters()
        ))

    def test_route_value_vjp_matches_direct_mi_gradient(self):
        torch.manual_seed(43)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        trainer = MemoryTreeTrainer(tree, hawkes, encoder, device="cpu")
        sequences = [
            trainer._move_sequence({
                "times": torch.tensor([0.1, 0.4, 0.9]),
                "types": torch.tensor([0, 1, 0]),
            }),
            trainer._move_sequence({
                "times": torch.tensor([0.2, 0.5, 1.0]),
                "types": torch.tensor([1, 0, 1]),
            }),
        ]
        encoder_scale = 0.1
        parameters = [
            *encoder.parameters(),
            *tree.router_compat.parameters(),
            *tree.node_emb.parameters(),
        ]

        direct_routes = []
        for sequence in sequences:
            z_all = trainer._encode_memory_sequence(sequence)
            routed_z = (
                z_all.detach()
                + encoder_scale * (z_all - z_all.detach())
            )
            direct_routes.append(
                tree.route(routed_z).responsibility.mean(dim=0)
            )
        direct_info = sequence_route_information(
            torch.stack(direct_routes)
        )
        direct_regularizer = (
            -0.2 * direct_info["mutual_information"]
            + 0.05 * direct_info["prior_kl"]
        )
        direct_gradients = torch.autograd.grad(
            direct_regularizer,
            parameters,
            allow_unused=True,
        )

        with torch.no_grad():
            reference = torch.stack([
                tree.route(
                    trainer._encode_memory_sequence(sequence)
                ).responsibility.mean(dim=0)
                for sequence in sequences
            ])
        reference = reference.detach().requires_grad_(True)
        reference_info = sequence_route_information(reference)
        reference_regularizer = (
            -0.2 * reference_info["mutual_information"]
            + 0.05 * reference_info["prior_kl"]
        )
        route_gradient = torch.autograd.grad(
            reference_regularizer,
            reference,
        )[0].detach()

        surrogate = torch.zeros(())
        hooks = []
        for sequence_index, sequence in enumerate(sequences):
            z_all = trainer._encode_memory_sequence(sequence)
            hooks.append(z_all.register_hook(
                lambda gradient, scale=encoder_scale: gradient * scale
            ))
            sequence_route = tree.route(
                z_all
            ).responsibility.mean(dim=0)
            surrogate = surrogate + (
                (sequence_route - sequence_route.detach())
                * route_gradient[sequence_index]
            ).sum()
        vjp_gradients = torch.autograd.grad(
            surrogate,
            parameters,
            allow_unused=True,
        )
        for hook in hooks:
            hook.remove()

        for direct, vjp in zip(direct_gradients, vjp_gradients):
            if direct is None or vjp is None:
                self.assertIsNone(direct)
                self.assertIsNone(vjp)
            else:
                self.assertTrue(torch.allclose(
                    direct,
                    vjp,
                    atol=1e-6,
                    rtol=1e-5,
                ))

    def test_router_optimizer_uses_scaled_lr_after_dynamic_split(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            encoder,
            training=TrainingConfig(
                learning_rate=1e-3,
                router_lr_scale=0.1,
            ),
            device="cpu",
        )
        groups = {
            group["group_name"]: group for group in trainer.optimizer.param_groups
        }
        self.assertAlmostEqual(groups["base"]["lr"], 1e-3)
        self.assertAlmostEqual(groups["router"]["lr"], 1e-4)

        tree.split_leaf("root_L", optimizer=trainer.optimizer)
        trainer._reconcile_optimizer_parameters()
        groups = {
            group["group_name"]: group for group in trainer.optimizer.param_groups
        }
        router_ids = {
            id(parameter)
            for module in (tree.router_compat, tree.expansion_predictor)
            for parameter in module.parameters()
        }
        self.assertEqual(
            {id(parameter) for parameter in groups["router"]["params"]},
            router_ids,
        )
        self.assertAlmostEqual(groups["router"]["lr"], 1e-4)

    def test_controller_uses_smooth_monotone_action_logits(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        controller = Controller(
            nll_fn=hawkes,
        )
        base = controller.action_distribution(
            torch.tensor(0.0),
            torch.tensor(0.2),
            torch.tensor(0.3),
            update_statistics=False,
        )
        high_surprise = controller.action_distribution(
            torch.tensor(1.0),
            torch.tensor(0.2),
            torch.tensor(0.3),
            update_statistics=False,
        )
        high_novelty = controller.action_distribution(
            torch.tensor(0.0),
            torch.tensor(0.8),
            torch.tensor(0.3),
            update_statistics=False,
        )
        high_count = controller.action_distribution(
            torch.tensor(0.0),
            torch.tensor(0.2),
            torch.tensor(0.9),
            update_statistics=False,
        )
        self.assertTrue(torch.all(base["probabilities"] >= 0.0))
        self.assertTrue(torch.all(base["probabilities"] <= 1.0))
        self.assertNotAlmostEqual(
            float(base["probabilities"].sum().detach()), 1.0, places=6
        )
        self.assertLess(high_surprise["logits"][0], base["logits"][0])
        self.assertTrue(torch.all(
            high_surprise["logits"][1:] > base["logits"][1:]
        ))
        self.assertLess(high_novelty["logits"][1], base["logits"][1])
        self.assertGreater(high_novelty["logits"][2], base["logits"][2])
        self.assertGreater(high_novelty["logits"][3], base["logits"][3])
        self.assertLess(high_count["logits"][2], base["logits"][2])
        self.assertGreater(high_count["logits"][3], base["logits"][3])

    def test_controller_novelty_and_local_recurrence_count_match_formulas(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(2, 4, 2, 1, init_depth=0, memory_key_dim=2)
        controller = Controller(
            nll_fn=hawkes,
            episodic_memory=tree.episodic_memory,
            novelty_temperature=10.0,
            count_exponent=2.0,
        )
        empty_novelty, empty_count, _ = controller.leaf_novelty_count(
            torch.tensor([1.0, 0.0]),
            "root",
        )
        self.assertEqual(float(empty_novelty), 1.0)
        self.assertEqual(float(empty_count), 0.0)

        zero_delta = torch.zeros(tree.param_dim)
        tree.episodic_memory.add_memory(
            "root",
            torch.tensor([1.0, 0.0]),
            zero_delta,
        )
        tree.episodic_memory.add_memory(
            "root",
            torch.tensor([-1.0, 0.0]),
            zero_delta,
        )
        novelty, count, weighted_similarity = (
            controller.leaf_novelty_count(
                torch.tensor([1.0, 0.0]),
                "root",
            )
        )
        expected_similarity = torch.tanh(torch.tensor(10.0))
        # Similarity +1 is a full compact-support recurrence match and
        # similarity -1 is outside support, so the bank contributes one unit
        # of local support regardless of its occupied capacity.
        expected_count = 1.0 - torch.exp(
            torch.tensor(-1.0 / controller.count_saturation)
        )
        self.assertTrue(torch.allclose(
            weighted_similarity,
            expected_similarity,
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            novelty,
            (1.0 - expected_similarity) / 2.0,
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(count, expected_count, atol=1e-6))

    def test_retrieval_scales_residual_by_continuous_write_quality(self):
        retriever = SmoothSparseRetriever()
        retrieved, _ = retriever(
            query=torch.tensor([1.0, 0.0]),
            keys=torch.tensor([[1.0, 0.0]]),
            deltas=torch.tensor([[2.0, -4.0]]),
            usage=torch.zeros(1),
            age=torch.zeros(1),
            write_quality=torch.tensor([0.25]),
        )
        self.assertTrue(torch.allclose(
            retrieved,
            torch.tensor([0.5, -1.0]),
            atol=1e-6,
        ))

    def test_load_cold_start_checkpoint_round_trip(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([0.7]))
        with torch.no_grad():
            hawkes.raw_mu.copy_(torch.tensor([-0.3, 0.4]))
            hawkes.raw_W.copy_(torch.arange(4.0).reshape(2, 2, 1) / 7.0)
        payload = {
            "format_version": 1,
            "model_state_dict": hawkes.state_dict(),
            "num_types": 2,
            "num_basis": 1,
            "decays": hawkes.decays,
            "training_result": {"best_validation_nll": 1.23},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "hawkes.pt"
            torch.save(payload, checkpoint)
            restored, restored_payload = HawkesFamily.from_cold_start_checkpoint(
                checkpoint
            )

        self.assertTrue(torch.equal(restored.raw_mu, hawkes.raw_mu))
        self.assertTrue(torch.equal(restored.raw_W, hawkes.raw_W))
        self.assertTrue(torch.equal(restored.decays, hawkes.decays))
        self.assertEqual(
            restored_payload["training_result"]["best_validation_nll"],
            1.23,
        )

    def test_end_to_end_train_checkpoint_and_wake_only_inference(self):
        torch.manual_seed(0)
        dataset = [
            {
                "times": torch.tensor([0.1, 0.4, 0.9]),
                "types": torch.tensor([0, 1, 0]),
            },
            {
                "times": torch.tensor([0.2, 0.5, 1.0]),
                "types": torch.tensor([1, 0, 1]),
            },
        ]
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3,
            4,
            2,
            1,
            init_depth=1,
            memory_key_dim=3,
            memory_capacity_per_node=32,
        )
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "memory_tree.pt"
            trainer = MemoryTreeTrainer(
                tree,
                hawkes,
                encoder,
                wake=WakeObjectiveConfig(
                    tau_surprise=-1.0,
                    tau_novelty=-1.0,
                    tau_count=100,
                    write_horizon=1,
                ),
                sleep=SleepConfig(
                    split_steps=2,
                    split_min_mass=0.1,
                ),
                structure=StructureConfig(
                    merge_kwargs={
                        "min_replay": 20,
                    },
                    promotion_kwargs={
                        "min_support": 100.0,
                        "min_gain": 0.0,
                        "min_balance": 0.5,
                    },
                ),
                training=TrainingConfig(
                    epochs=1,
                    checkpoint_path=str(checkpoint),
                ),
                device="cpu",
            )
            history = trainer.train(dataset, verbose=False)
            self.assertEqual(len(history), 1)
            self.assertTrue(checkpoint.exists())
            self.assertIsNotNone(history[0]["sleep"])
            self.assertEqual(
                sum(len(bank) for bank in trainer.tree.episodic_memory.banks.values()),
                history[0]["writes"],
            )
            self.assertLessEqual(
                history[0]["writes"],
                len(dataset) * 2,
            )
            self.assertFalse(history[0]["topology_prune_enabled"])
            self.assertIn("router_calibration", history[0])
            self.assertIn("global_update", history[0])
            self.assertGreaterEqual(
                history[0]["router_calibration"]["steps"],
                1,
            )
            self.assertEqual(
                sum(history[0]["hard_assignment_counts"].values()),
                len(dataset),
            )
            self.assertEqual(
                sum(history[0]["memory_assignment_counts"].values()),
                history[0]["events"],
            )
            self.assertEqual(
                sum(history[0]["wake_action_counts"].values()),
                history[0]["events"],
            )

            inference = MemoryTreeInference.from_checkpoint(
                checkpoint,
                device="cpu",
                inference_config=InferenceConfig(allow_memory_writes=False),
            )
            self.assertEqual(
                inference.tree.episodic_memory.capacity_per_node, 32
            )
            leaf_ids_before = list(inference.tree.leaf_ids)
            result = inference.run_sequence(dataset[0])
            forecast = inference.predict_next_event(dataset[0])
            self.assertEqual(len(result["events"]), 3)
            self.assertTrue(torch.isfinite(torch.tensor(result["nll_per_event"])))
            self.assertEqual(inference.tree.leaf_ids, leaf_ids_before)
            self.assertAlmostEqual(
                float(forecast["type_probabilities"].sum()), 1.0, places=5
            )

            # Online writes wait until their future horizon has actually arrived.
            writing_inference = MemoryTreeInference.from_checkpoint(
                checkpoint,
                device="cpu",
                inference_config=InferenceConfig(allow_memory_writes=True),
            )
            memories_before = sum(
                len(bank)
                for bank in writing_inference.tree.episodic_memory.banks.values()
            )
            write_result = writing_inference.run_sequence(dataset[0])
            memories_after = sum(
                len(bank)
                for bank in writing_inference.tree.episodic_memory.banks.values()
            )
            self.assertLessEqual(memories_after - memories_before, 2)
            self.assertLessEqual(write_result["pending_write_count"], 1)

    def test_cold_start_parameters_initialize_every_tree_node(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        with torch.no_grad():
            hawkes.raw_mu.copy_(torch.tensor([-0.7, 0.2]))
            hawkes.raw_W.copy_(torch.arange(4.0).reshape(2, 2, 1) / 10.0)
        target = tree.initialize_semantics_from_hawkes(hawkes)
        expected = torch.cat((hawkes.raw_mu, hawkes.raw_W.reshape(-1)))
        self.assertTrue(torch.equal(target, expected))
        for node_id in tree.all_node_ids:
            self.assertTrue(torch.allclose(tree.semantic_theta(node_id), expected))

    def test_training_memory_write_waits_for_future_horizon(self):
        sequence = {
            "times": torch.tensor([0.1, 0.4, 0.9]),
            "types": torch.tensor([0, 1, 0]),
        }
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            encoder,
            wake=WakeObjectiveConfig(
                tau_surprise=-100.0,
                tau_novelty=-100.0,
                tau_count=100,
                write_horizon=2,
            ),
            device="cpu",
        )
        result = trainer.train_wake_sequence(sequence)
        bank = tree.episodic_memory.get_bank("root")
        self.assertEqual(result["write_count"], 1)
        self.assertEqual(result["pending_write_count"], 2)
        self.assertEqual(
            sum(result["action_counts"].values()),
            result["event_count"],
        )
        self.assertEqual(result["action_counts"][Action.MEMORIZE.value], 3)
        self.assertEqual(len(bank), 1)
        self.assertEqual(bank.windows[0].start_idx, 0)
        self.assertEqual(bank.windows[0].end_idx, 3)
        self.assertIsNotNone(bank.windows[0].event_time_features)
        self.assertIsNotNone(bank.windows[0].hawkes_history_stats)
        self.assertIsNotNone(bank.windows[0].hawkes_interval_stats)

    def test_wake_updates_only_sequence_local_working_memory(self):
        sequence = {
            "times": torch.tensor([0.1, 0.4, 0.9]),
            "types": torch.tensor([0, 1, 0]),
        }
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        trainer = MemoryTreeTrainer(tree, hawkes, encoder, device="cpu")
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainer._named_optimized_parameters().items()
        }

        result = trainer.train_wake_sequence(sequence)

        after = trainer._named_optimized_parameters()
        self.assertTrue(all(
            torch.equal(before[name], parameter.detach())
            for name, parameter in after.items()
        ))
        self.assertTrue(all(
            parameter.grad is None for parameter in after.values()
        ))
        self.assertGreater(
            float(tree.working_memory.delta.abs().sum()),
            0.0,
        )
        self.assertEqual(
            result["sequence_responsibility"].shape,
            (len(tree.leaf_ids),),
        )

    def test_global_batch_makes_one_step_for_two_sequences(self):
        dataset = [
            {
                "times": torch.tensor([0.1, 0.4, 0.9]),
                "types": torch.tensor([0, 1, 0]),
            },
            {
                "times": torch.tensor([0.2, 0.5, 1.0]),
                "types": torch.tensor([1, 0, 1]),
            },
        ]
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            encoder,
            wake=WakeObjectiveConfig(route_balance_batch_size=2),
            device="cpu",
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainer._named_optimized_parameters().items()
        }
        generator = torch.Generator(device="cpu").manual_seed(7)

        result = trainer.train_global_batch_epoch(dataset, generator)

        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["sequences"], 2)
        self.assertEqual(result["events"], 6)
        self.assertTrue(any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter
            in trainer._named_optimized_parameters().items()
        ))

    def test_flat_global_losses_match_sequence_reference(self):
        torch.manual_seed(811)
        dataset = [
            {
                "times": torch.tensor([0.1, 0.4, 0.9]),
                "types": torch.tensor([0, 1, 0]),
            },
            {
                "times": torch.tensor([0.2, 0.7]),
                "types": torch.tensor([1, 0]),
            },
        ]
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=2, memory_key_dim=3)
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
        moved = [trainer._move_sequence(sequence) for sequence in dataset]
        trainer.tree.eval()
        trainer.encoder.eval()

        with torch.no_grad():
            z_flat, flat = trainer._encode_global_sequence_batch(moved)
            z_reference = torch.cat([
                trainer._encode_memory_sequence(sequence)
                for sequence in moved
            ])
            self.assertTrue(torch.allclose(
                z_flat, z_reference, atol=1e-7, rtol=1e-6
            ))

            routed = reliability_gated_route_state(
                z_flat,
                reliability=trainer.encoder_routing_reliability,
                alpha_max=trainer.wake_config.route_encoder_grad_scale,
            )
            packed_output = tree(
                z_t=z_flat,
                working_delta=torch.zeros(tree.param_dim),
                decays=hawkes.decays,
                frontier_projected_z=tree.router_compat.project_z(routed),
                frontier_query=tree.episodic_memory.query_net(z_flat),
                update_memory_state=False,
                update_search_state=False,
                materialize_diagnostics=False,
            )
            packed_event = trainer._batched_sequence_event_nll(
                flat, packed_output
            )
            packed_frontier = trainer._batched_frontier_event_nll(
                flat, packed_output
            )
            packed_child = (
                trainer._batched_expanded_child_event_nll(
                    flat, packed_output
                )
            )
            packed_local = (
                trainer._batched_local_frontier_objective(
                    packed_output,
                    packed_child,
                    flat["sequence_index"],
                    len(moved),
                )
            )

            event_reference = []
            frontier_reference = []
            local_reference = []
            offset = 0
            for sequence in moved:
                length = int(sequence["times"].numel())
                z_sequence = z_reference[offset : offset + length]
                offset += length
                routed_sequence = reliability_gated_route_state(
                    z_sequence,
                    reliability=trainer.encoder_routing_reliability,
                    alpha_max=trainer.wake_config.route_encoder_grad_scale,
                )
                output = tree(
                    z_t=z_sequence,
                    working_delta=torch.zeros(tree.param_dim),
                    decays=hawkes.decays,
                    frontier_projected_z=tree.router_compat.project_z(
                        routed_sequence
                    ),
                    frontier_query=tree.episodic_memory.query_net(
                        z_sequence
                    ),
                    update_memory_state=False,
                    update_search_state=False,
                    materialize_diagnostics=False,
                )
                event_reference.append(
                    trainer._sequence_event_nll(sequence, output)
                )
                frontier_reference.append(
                    trainer._frontier_sequence_event_nll(
                        sequence, output
                    )
                )
                child = trainer._expanded_child_sequence_event_nll(
                    sequence, output
                )
                local_reference.append(
                    trainer._local_frontier_objective(output, child)
                )

            self.assertTrue(torch.allclose(
                packed_event,
                torch.cat(event_reference),
                atol=1e-6,
                rtol=1e-6,
            ))
            self.assertTrue(torch.allclose(
                packed_frontier,
                torch.cat(frontier_reference),
                atol=1e-6,
                rtol=1e-6,
            ))
            reliability = torch.cat([
                local["reliability"].reshape(-1)
                for local in local_reference
            ])
            teacher = torch.cat([
                local["teacher"].reshape(-1, 2)
                for local in local_reference
            ])
            student = torch.cat([
                local["student"].reshape(-1, 2)
                for local in local_reference
            ])
            row_kl = (
                teacher
                * (
                    teacher.clamp_min(1e-12).log()
                    - student.clamp_min(1e-12).log()
                )
            ).sum(dim=-1)
            expected_distill = (
                reliability * row_kl
            ).sum() / reliability.sum().clamp_min(1e-12)
            self.assertTrue(torch.allclose(
                packed_local["distill"],
                expected_distill,
                atol=1e-6,
                rtol=1e-6,
            ))

            for key in (
                "mutual_information",
                "balance_kl",
                "conditional_entropy",
                "marginal_entropy",
            ):
                reference = torch.stack([
                    local[key] for local in local_reference
                ]).mean()
                self.assertTrue(
                    torch.allclose(
                        packed_local[key],
                        reference,
                        atol=1e-6,
                        rtol=1e-6,
                    ),
                    msg=key,
                )

    def test_write_penalty_covers_both_actions_and_has_surrogate_gradient(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            encoder,
            wake=WakeObjectiveConfig(lambda_write=0.7),
            device="cpu",
        )
        surprise = torch.tensor(3.0, requires_grad=True)
        novelty = torch.tensor(0.8, requires_grad=True)
        soft = trainer.controller.soft_write_probability(
            surprise,
            novelty,
            temperature=trainer.wake_config.write_surrogate_temperature,
        )
        for action in (Action.MEMORIZE, Action.QUEUE_SPLIT):
            hard = surprise.new_tensor(
                float(action in {Action.MEMORIZE, Action.QUEUE_SPLIT})
            )
            gate = hard + soft - soft.detach()
            self.assertEqual(float(gate.detach()), 1.0)
        (trainer.wake_config.lambda_write * (1.0 + soft - soft.detach())).backward()
        self.assertNotEqual(float(surprise.grad), 0.0)
        self.assertNotEqual(float(novelty.grad), 0.0)

    def test_assimilate_and_retrieve_never_physically_write(self):
        sequence = {
            "times": torch.tensor([0.1, 0.4, 0.9]),
            "types": torch.tensor([0, 1, 0]),
        }
        for forced_action in (Action.ASSIMILATE, Action.RETRIEVE):
            hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
            tree = HawkesTree(
                3, 4, 2, 1, init_depth=0, memory_key_dim=3
            )
            trainer = MemoryTreeTrainer(
                tree,
                hawkes,
                CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
                wake=WakeObjectiveConfig(write_horizon=1),
                device="cpu",
            )
            with torch.no_grad():
                trainer.controller.bias_retrieve.fill_(
                    100.0 if forced_action is Action.RETRIEVE else -100.0
                )
                trainer.controller.bias_memorize.fill_(-100.0)
                trainer.controller.bias_queue_split.fill_(-100.0)
            result = trainer.train_wake_sequence(sequence)
            self.assertEqual(
                result["action_counts"].get(forced_action.value, 0), 3
            )
            self.assertEqual(result["write_count"], 0)
            self.assertEqual(result["write_decision_count"], 0)
            self.assertEqual(
                sum(len(bank) for bank in tree.episodic_memory.banks.values()),
                0,
            )

    def test_physical_writes_use_per_sequence_top_b_budget(self):
        times = torch.arange(1, 11, dtype=torch.float32) / 10.0
        sequence = {
            "times": times,
            "types": torch.arange(10) % 2,
        }
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=0, memory_key_dim=3
        )
        tree.configure_frontier_routing(
            config=FrontierRoutingConfig(
                frontier_budget=4,
                max_writes_per_sequence=4,
            )
        )
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            wake=WakeObjectiveConfig(write_horizon=1),
            device="cpu",
        )
        with torch.no_grad():
            trainer.controller.bias_retrieve.fill_(-100.0)
            trainer.controller.bias_memorize.fill_(100.0)
            trainer.controller.bias_queue_split.fill_(-100.0)
        result = trainer.train_wake_sequence(sequence)
        self.assertEqual(result["write_candidates"], 10)
        self.assertLessEqual(result["write_decision_count"], 4)
        self.assertEqual(result["write_count"], result["write_decision_count"])
        self.assertEqual(
            len(tree.episodic_memory.get_bank("root")), result["write_count"]
        )
        self.assertEqual(result["harmful_write_count"], 0)

    def test_resume_training_restores_dynamic_optimizer_and_history(self):
        torch.manual_seed(71)
        sequence = {
            "times": torch.tensor([0.1, 0.4, 0.9]),
            "types": torch.tensor([0, 1, 0]),
        }
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        with tempfile.TemporaryDirectory() as directory:
            first_checkpoint = Path(directory) / "first.pt"
            resumed_checkpoint = Path(directory) / "resumed.pt"
            trainer = MemoryTreeTrainer(
                tree,
                hawkes,
                encoder,
                wake=WakeObjectiveConfig(
                    tau_surprise=-100.0,
                    tau_novelty=-100.0,
                    tau_count=0,
                    write_horizon=1,
                ),
                sleep=SleepConfig(
                    split_steps=2,
                    split_min_mass=0.0,
                    split_init_steps=2,
                ),
                structure=StructureConfig(
                    merge_kwargs={
                        "min_replay": 100,
                    },
                    promotion_kwargs={
                        "min_support": 100.0,
                        "min_gain": 1e9,
                        "min_balance": 1.0,
                    },
                ),
                training=TrainingConfig(
                    epochs=1,
                    checkpoint_path=str(first_checkpoint),
                ),
                device="cpu",
            )
            # Exercise dynamic optimizer/checkpoint restoration explicitly;
            # split evidence is now hard-gated by actual QUEUE_SPLIT actions.
            trainer.tree.split_leaf("root")
            trainer._sync_split_modules()
            trainer._reconcile_optimizer_parameters()
            trainer.merge_lambda_T = 0.25
            trainer.merge_budget_KT = 3.5
            trainer.train([sequence], verbose=False)
            self.assertEqual(trainer.completed_epochs, 1)
            self.assertGreater(len(trainer.tree.leaf_ids), 1)

            resumed = MemoryTreeTrainer.from_checkpoint(
                first_checkpoint,
                device="cpu",
            )
            self.assertEqual(resumed.completed_epochs, 1)
            self.assertEqual(len(resumed.history), 1)
            self.assertEqual(resumed.tree.leaf_ids, trainer.tree.leaf_ids)
            self.assertAlmostEqual(
                resumed.encoder_routing_reliability,
                trainer.encoder_routing_reliability,
            )
            self.assertAlmostEqual(
                resumed.last_teacher_confidence,
                trainer.last_teacher_confidence,
            )
            self.assertAlmostEqual(
                resumed.last_teacher_student_js,
                trainer.last_teacher_student_js,
            )
            self.assertAlmostEqual(
                resumed.merge_lambda_T,
                trainer.merge_lambda_T,
            )
            self.assertAlmostEqual(
                resumed.merge_budget_KT,
                trainer.merge_budget_KT,
            )
            self.assertEqual(
                sum(len(bank) for bank in resumed.tree.episodic_memory.banks.values()),
                sum(len(bank) for bank in trainer.tree.episodic_memory.banks.values()),
            )
            optimizer_ids = {
                id(parameter)
                for group in resumed.optimizer.param_groups
                for parameter in group["params"]
            }
            self.assertTrue(all(
                id(parameter) in optimizer_ids
                for parameter in list(resumed.tree.parameters())
                + list(resumed.encoder.parameters())
            ))
            resumed.training_config.epochs = 1
            resumed.training_config.checkpoint_path = str(resumed_checkpoint)
            resumed.train([sequence], verbose=False)
            self.assertEqual(resumed.completed_epochs, 2)
            self.assertEqual([item["epoch"] for item in resumed.history], [1, 2])
            self.assertTrue(resumed_checkpoint.exists())

    def test_delayed_queue_split_is_subject_to_unified_null_boundary(self):
        sequence = {
            "times": torch.tensor([0.1, 0.4, 0.9]),
            "types": torch.tensor([0, 1, 0]),
        }
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        encoder = CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8)
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            encoder,
            wake=WakeObjectiveConfig(
                tau_surprise=-100.0,
                tau_novelty=-100.0,
                tau_count=0,
                write_horizon=1,
            ),
            sleep=SleepConfig(
                split_steps=2,
                split_min_mass=0.0,
            ),
            structure=StructureConfig(
                merge_kwargs={
                    "min_replay": 100,
                },
                promotion_kwargs={
                    "min_support": 100.0,
                    "min_gain": 1e9,
                    "min_balance": 1.0,
                },
            ),
            device="cpu",
        )
        with torch.no_grad():
            trainer.controller.bias_retrieve.fill_(-100.0)
            trainer.controller.bias_memorize.fill_(100.0)
            trainer.controller.bias_queue_split.fill_(100.0)
        wake_result = trainer.train_wake_sequence(sequence)
        self.assertGreaterEqual(wake_result["write_count"], 1)
        self.assertLessEqual(wake_result["write_count"], 2)
        bank = tree.episodic_memory.get_bank("root")
        self.assertEqual(len(bank), wake_result["write_count"])
        self.assertTrue(torch.all(
            (bank.write_quality >= 0.0) & (bank.write_quality <= 1.0)
        ))
        self.assertTrue(torch.all(
            (bank.queue_weight >= 0.0) & (bank.queue_weight <= 1.0)
        ))
        self.assertAlmostEqual(
            trainer.controller.split_queues["root"],
            float(bank.queue_weight.sum()),
            places=6,
        )
        sleep_result = trainer.train_sleep(
            torch.stack(wake_result["responsibilities"])
        )
        selected = sleep_result["unified_topology"]["selected_action"]
        self.assertIn(selected, {"null", "split"})
        self.assertEqual(
            [
                action["action"]
                for action in sleep_result["transaction"]["actions"]
                if action["action"] in {"split", "merge", "topology_prune"}
            ],
            [] if selected == "null" else ["split"],
        )
        optimizer_ids = {
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(
            all(id(parameter) in optimizer_ids for parameter in tree.parameters())
        )


if __name__ == "__main__":
    unittest.main()
