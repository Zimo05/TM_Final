import tempfile
import unittest
from pathlib import Path

import torch

from HawkesBackbone import HawkesFamily
from LatentHawkesTree import HawkesTree
from MemoryResiduals.MemoryBank import EventWindow
from MemoryResiduals.Replay import replay_log_likelihood
from Sleep.DeepGate import DeepSleepGate
from Sleep.Light import (
    LightSleepSettings,
    propose_light_absorption,
    run_light_sleep,
)
from Sleep.Split import SplitModule
from Train.Train import (
    CausalPrefixEncoder,
    MemoryTreeTrainer,
    SleepConfig,
    StructureConfig,
)


class LightDeepSleepTests(unittest.TestCase):
    def _improving_residual(
        self,
        tree,
        hawkes,
        window,
    ):
        theta = tree.semantic_theta("root").detach().requires_grad_(True)
        log_likelihood = replay_log_likelihood(
            window,
            theta,
            hawkes,
            decays=hawkes.decays,
            normalize_by_events=True,
        )
        gradient = torch.autograd.grad(log_likelihood, theta)[0]
        return (0.02 * gradient).detach()

    def test_topology_prune_padded_encoder_matches_scalar_prefixes(self):
        torch.manual_seed(397)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=1, memory_key_dim=3
        )
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            device="cpu",
        )
        windows = [
            EventWindow(
                torch.tensor([0.1, 0.4, 0.8]),
                torch.tensor([0, 1, 0]),
                "root_L",
                0,
                2,
                True,
            ),
            EventWindow(
                torch.tensor([0.2, 0.5, 0.9, 1.3]),
                torch.tensor([1, 0, 1, 0]),
                "root_R",
                2,
                4,
                True,
            ),
        ]
        scalar = torch.stack([
            trainer._encode_topology_prune_window(window)
            for window in windows
        ])
        batched = trainer._encode_topology_prune_windows(windows)
        self.assertTrue(torch.allclose(batched, scalar, atol=1e-7))

    def test_light_sleep_rebases_exactly_after_absorption(self):
        torch.manual_seed(401)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=0, memory_key_dim=3
        )
        window = EventWindow(
            times=torch.tensor([0.1, 0.4, 0.8]),
            types=torch.tensor([0, 1, 0]),
            node_id="root",
            start_idx=1,
            end_idx=3,
            has_full_history=True,
        )
        delta = self._improving_residual(tree, hawkes, window)
        for memory_key in torch.eye(3):
            tree.episodic_memory.add_memory(
                "root",
                memory_key,
                delta,
                window,
                write_quality=1.0,
            )
        bank = tree.episodic_memory.get_bank("root")
        theta_before = tree.semantic_theta("root").detach().clone()
        effective_before = theta_before + bank.deltas.clone()

        result = run_light_sleep(
            tree,
            hawkes,
            settings=LightSleepSettings(
                replay_budget=2,
                min_per_leaf=1,
                gain_evaluations_per_direction=2,
                min_direction_support=2,
                coherence_threshold=0.5,
                alpha_max=0.5,
                trust_radius=1.0,
                gain_reference=1e-6,
            ),
        )
        theta_after = tree.semantic_theta("root").detach()
        self.assertEqual(result["replay_windows"], 2)
        self.assertEqual(result["absorbed_leaves"], 1)
        self.assertFalse(torch.allclose(theta_before, theta_after))
        self.assertTrue(torch.allclose(
            theta_after + bank.deltas,
            effective_before,
            atol=2e-6,
            rtol=2e-5,
        ))

    def test_light_replay_evaluations_obey_global_budget(self):
        torch.manual_seed(409)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=1, memory_key_dim=3
        )
        for leaf_id in tree.leaf_ids:
            for index in range(8):
                tree.episodic_memory.add_memory(
                    leaf_id,
                    torch.randn(3),
                    torch.randn(tree.param_dim) * 0.01,
                    EventWindow(
                        times=torch.tensor([0.1, 0.3]),
                        types=torch.tensor([0, 1]),
                        node_id=leaf_id,
                        start_idx=0,
                        end_idx=2,
                        has_full_history=True,
                    ),
                )
        result = run_light_sleep(
            tree,
            hawkes,
            settings=LightSleepSettings(
                replay_budget=3,
                min_per_leaf=1,
                gain_evaluations_per_direction=2,
                min_direction_support=100,
            ),
        )
        self.assertLessEqual(result["replay_windows"], 3)
        self.assertLessEqual(result["scanned_memories"], 12)

    def test_light_scan_cursor_and_direction_index_are_bounded(self):
        torch.manual_seed(419)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=0, memory_key_dim=3
        )
        for _ in range(20):
            tree.episodic_memory.add_memory(
                "root",
                torch.randn(3),
                torch.randn(tree.param_dim) * 0.01,
                EventWindow(
                    times=torch.tensor([0.1, 0.3]),
                    types=torch.tensor([0, 1]),
                    node_id="root",
                    start_idx=0,
                    end_idx=2,
                    has_full_history=True,
                ),
            )
        state = {}
        settings = LightSleepSettings(
            replay_budget=2,
            scan_budget_multiplier=2,
            min_per_leaf=1,
            min_direction_support=100,
        )
        first = run_light_sleep(
            tree,
            hawkes,
            settings=settings,
            state=state,
        )
        first_cursor = state["scan_cursors"]["root"]
        second = run_light_sleep(
            tree,
            hawkes,
            settings=settings,
            state=state,
        )
        self.assertEqual(first["scanned_memories"], 4)
        self.assertEqual(second["scanned_memories"], 4)
        self.assertEqual(first_cursor, 4)
        self.assertEqual(state["scan_cursors"]["root"], 8)
        self.assertIn("root", state["direction_summaries"])
        self.assertLessEqual(
            state["direction_summaries"]["root"]["centroids"].size(0),
            settings.max_directions,
        )

    def test_split_batch_uses_bounded_evidence(self):
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=0, memory_key_dim=3
        )
        bank = tree.episodic_memory.get_bank("root")
        for index in range(10):
            tree.episodic_memory.add_memory(
                "root",
                torch.randn(3),
                torch.randn(tree.param_dim) * 0.01,
                EventWindow(
                    torch.tensor([0.1]),
                    torch.tensor([0]),
                    "root",
                    0,
                    1,
                    True,
                ),
                write_quality=float(index + 1) / 10.0,
                queue_weight=0.1 + 0.08 * index,
            )
        batch = SplitModule.build_split_batch_from_memory_bank(
            bank,
            max_items=4,
        )
        self.assertIsNotNone(batch)
        self.assertEqual(batch.residuals.size(0), 4)
        self.assertEqual(len(batch.windows), 4)

    def test_split_weights_use_replay_count_scale(self):
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=0, memory_key_dim=3
        )
        bank = tree.episodic_memory.get_bank("root")
        for index, structural_weight in enumerate((0.2, 0.4, 0.6, 0.8)):
            tree.episodic_memory.add_memory(
                "root",
                torch.randn(3),
                torch.full((tree.param_dim,), 0.01 * (index + 1)),
                EventWindow(
                    torch.tensor([0.1]),
                    torch.tensor([0]),
                    "root",
                    0,
                    1,
                    True,
                ),
                write_quality=1.0,
                queue_weight=structural_weight,
            )
        bank.usage.copy_(torch.tensor([0.0, 1.0, 3.0, 7.0]))
        bank.age.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))

        batch = SplitModule.build_split_batch_from_memory_bank(bank)

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(batch.weights.sum()), 4.0, places=5)
        expected_weights = batch.base_weights * batch.sample_support
        expected_weights = 4.0 * expected_weights / expected_weights.sum()
        torch.testing.assert_close(batch.weights, expected_weights)
        expected_strength = torch.ones_like(batch.structural_strength)
        self.assertTrue(torch.allclose(
            batch.structural_strength,
            expected_strength,
            atol=1e-6,
        ))

    def test_split_objective_matches_relaxed_formula(self):
        module = SplitModule(
            4,
            3,
            m_min=2.0,
            tau_m=0.5,
            lambda_mass=0.3,
        )
        weights = torch.tensor([1.0, 3.0])
        ell_split = torch.tensor([-2.0, -1.0])
        g_split = torch.tensor(0.25)
        n_eff = torch.tensor([1.5, 3.0])

        terms = module.compute_split_training_objective(
            weights=weights,
            ell_split=ell_split,
            g_split=g_split,
            N_eff=n_eff,
            lambda_T=0.4,
            delta_complexity=2.0,
        )

        expected_prediction = -(
            weights * ell_split
        ).sum() / (weights.sum() + module.eps)
        expected_complexity = torch.tensor(0.4) * g_split * 2.0
        expected_mass = 0.3 * torch.nn.functional.softplus(
            (2.0 - n_eff) / 0.5
        ).sum()
        self.assertTrue(torch.allclose(
            terms["prediction_loss"], expected_prediction
        ))
        self.assertTrue(torch.allclose(
            terms["complexity_penalty"], expected_complexity
        ))
        self.assertTrue(torch.allclose(
            terms["mass_penalty"], expected_mass
        ))
        self.assertTrue(torch.allclose(
            terms["loss"],
            expected_prediction + expected_complexity + expected_mass,
        ))

    def test_split_child_support_is_diagnostic_not_a_hard_gate(self):
        module = SplitModule(4, 3)
        with torch.no_grad():
            module.centers.zero_()
        structure = module.compute_soft_residual_structure(
            residuals=torch.randn(3, 4),
            weights=torch.tensor([3.0, 0.0, 0.0]),
            theta_sem=torch.zeros(4),
        )

        self.assertTrue(torch.allclose(
            structure["N_mass"], torch.tensor([1.5, 1.5])
        ))
        self.assertTrue(torch.allclose(
            structure["N_eff"], torch.tensor([1.0, 1.0]), atol=1e-6
        ))
        self.assertTrue(torch.equal(
            structure["N"], structure["N_mass"]
        ))

        ready = SplitModule.evaluate_split_eligibility(
            {
                "g_split": torch.tensor(0.0),
                "N": torch.tensor([10.0, 10.0]),
                "N_eff": structure["N_eff"],
                "structural_strength": torch.tensor(0.8),
                "effective_sample_size": torch.tensor(3.0),
            },
            module.commit_state,
            m_min=2.0,
            min_structural_strength=0.05,
            min_effective_sample_size=2.0,
        )
        self.assertTrue(ready)

    def test_split_readiness_ignores_strength_and_ess_thresholds(self):
        common = {
            "g_split": torch.tensor(0.95),
            "N": torch.tensor([2.5, 2.5]),
        }
        weak_state = SplitModule(4, 3).commit_state
        weak = SplitModule.evaluate_split_eligibility(
            {
                **common,
                "structural_strength": torch.tensor(1e-6),
                "effective_sample_size": torch.tensor(5.0),
            },
            weak_state,
            m_min=2.0,
            min_structural_strength=0.05,
            min_effective_sample_size=2.0,
        )
        dominant_state = SplitModule(4, 3).commit_state
        dominant = SplitModule.evaluate_split_eligibility(
            {
                **common,
                "structural_strength": torch.tensor(0.8),
                "effective_sample_size": torch.tensor(1.05),
            },
            dominant_state,
            m_min=2.0,
            min_structural_strength=0.05,
            min_effective_sample_size=2.0,
        )

        self.assertTrue(weak)
        self.assertTrue(dominant)

    def test_light_sleep_does_not_mutate_probe_protected_leaf(self):
        torch.manual_seed(421)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=0, memory_key_dim=3
        )
        window = EventWindow(
            torch.tensor([0.1, 0.4, 0.8]),
            torch.tensor([0, 1, 0]),
            "root",
            0,
            3,
            True,
        )
        delta = self._improving_residual(tree, hawkes, window)
        for key in torch.eye(3):
            tree.episodic_memory.add_memory(
                "root", key, delta, window, write_quality=1.0
            )
        bank = tree.episodic_memory.get_bank("root")
        theta_before = tree.semantic_theta("root").detach().clone()
        delta_before = bank.deltas.detach().clone()

        seen = {}

        def structural_probe(leaf_id, current_bank, proposal):
            seen[leaf_id] = {
                "theta": tree.semantic_theta(leaf_id).detach().clone(),
                "deltas": current_bank.deltas.detach().clone(),
                "theta_old": proposal.theta_old.detach().clone(),
            }
            return {"protect": True, "advantage": 1.0}

        result = run_light_sleep(
            tree,
            hawkes,
            settings=LightSleepSettings(replay_budget=2),
            structural_probe=structural_probe,
        )

        self.assertIn("root", seen)
        torch.testing.assert_close(seen["root"]["theta"], theta_before)
        torch.testing.assert_close(seen["root"]["deltas"], delta_before)
        torch.testing.assert_close(
            seen["root"]["theta_old"], theta_before
        )
        torch.testing.assert_close(tree.semantic_theta("root"), theta_before)
        torch.testing.assert_close(bank.deltas, delta_before)
        self.assertEqual(result["protected_leaves"], 1)
        self.assertEqual(result["absorbed_leaves"], 0)
        self.assertIn("root", result["bank_mode_probes"])

    def test_bank_mode_counterfactual_probe_is_non_mutating(self):
        torch.manual_seed(423)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=0, memory_key_dim=3
        )
        bank = tree.episodic_memory.get_bank("root")
        windows = [
            EventWindow(
                torch.tensor([0.1, 0.4, 0.8]),
                torch.tensor([0, 1, 0]),
                "root",
                0,
                3,
                True,
            ),
            EventWindow(
                torch.tensor([0.2, 0.5, 0.9]),
                torch.tensor([1, 0, 1]),
                "root",
                0,
                3,
                True,
            ),
        ]
        law_dim = bank.law_dim
        law_a = torch.zeros(law_dim)
        law_b = torch.zeros(law_dim)
        law_a[0] = 1.0
        law_b[min(1, law_dim - 1)] = 1.0
        delta = self._improving_residual(tree, hawkes, windows[0])
        bank.add(
            torch.tensor([1.0, 0.0, 0.0]),
            delta,
            windows[0],
            law_key=law_a,
            prediction_gain=1.0,
            queue_weight=1.0,
            force_new_mode_confirmation=True,
        )
        bank.add(
            torch.tensor([0.0, 1.0, 0.0]),
            delta,
            windows[0],
            law_key=law_b,
            prediction_gain=1.0,
            queue_weight=1.0,
            force_new_mode_confirmation=True,
        )
        module = SplitModule(tree.param_dim, 3, nll_fn=hawkes)
        theta_before = tree.semantic_theta("root").detach().clone()
        delta_before = bank.deltas.detach().clone()
        modes_before = bank.mode_ids.detach().clone()
        settings = LightSleepSettings(
            replay_budget=2,
            min_per_leaf=1,
            min_direction_support=1,
            coherence_threshold=0.0,
            gain_reference=1e-6,
        )
        proposal = propose_light_absorption(
            bank,
            theta_before,
            hawkes,
            settings,
            [0, 1],
            replay_budget=2,
        )
        self.assertIsNotNone(proposal.selected_direction)
        batch = module.build_split_batch_from_memory_bank(
            bank,
            indices=proposal.replay_indices,
        )
        self.assertIsNotNone(batch)

        probe = module.bank_mode_counterfactual_probe(
            theta_sem=theta_before,
            batch=batch,
            hawkes_ll=hawkes,
            light_proposal=proposal,
        )

        self.assertIsNotNone(probe)
        self.assertTrue(torch.isfinite(torch.tensor(probe["advantage"])))
        torch.testing.assert_close(
            probe["theta_h0"], proposal.theta_candidate
        )
        torch.testing.assert_close(
            probe["shared_delta"], proposal.shared_delta
        )
        self.assertEqual(probe["alpha_light"], proposal.alpha)
        torch.testing.assert_close(
            probe["light_replay_indices"], proposal.replay_indices
        )
        torch.testing.assert_close(tree.semantic_theta("root"), theta_before)
        torch.testing.assert_close(bank.deltas, delta_before)
        torch.testing.assert_close(bank.mode_ids, modes_before)

    def test_split_eligibility_does_not_use_local_gate_or_persistence(self):
        state = SplitModule(4, 3).commit_state
        ready = SplitModule.evaluate_split_eligibility(
            {
                "g_split": torch.tensor(0.0),
                "N": torch.tensor([2.5, 2.5]),
                "structural_strength": torch.tensor(0.8),
                "effective_sample_size": torch.tensor(5.0),
            },
            state,
            m_min=2.0,
            min_structural_strength=0.05,
            min_effective_sample_size=2.0,
        )

        self.assertTrue(ready)
        self.assertEqual(state.consecutive_ready, 1)

    def test_deep_gate_is_monotone_and_has_smooth_availability(self):
        gate = DeepSleepGate(
            accumulator_decay=0.0,
            bias_initial=-3.0,
            weight_initial=1.0,
        )
        gate.eval()
        low = gate(
            torch.tensor([0.0, 0.0, 0.0]),
            split_demand=0.0,
            deep_availability=1.0,
            sample=False,
        )
        gate.reset_after_deep()
        high = gate(
            torch.tensor([1.0, 1.0, 1.0]),
            split_demand=0.0,
            deep_availability=1.0,
            sample=False,
        )
        gate.reset_after_deep()
        refractory = gate(
            torch.tensor([1.0, 1.0, 1.0]),
            split_demand=0.0,
            deep_availability=0.0,
            sample=False,
        )

        self.assertGreater(
            float(high["probability"].detach()),
            float(low["probability"].detach()),
        )
        self.assertLess(
            float(refractory["probability"].detach()),
            float(low["probability"].detach()),
        )
        self.assertTrue(bool((gate.positive_weights > 0.0).all()))

    def test_positive_virtual_gain_pushes_deep_probability_up(self):
        gate = DeepSleepGate(
            accumulator_decay=0.0,
            bias_initial=-1.0,
        )
        gate.eval()
        output = gate(
            torch.tensor([0.5, 0.5, 0.5]),
            split_demand=0.0,
            deep_availability=1.0,
            sample=False,
        )
        objective = gate.objective(
            output,
            estimated_gain=2.0,
            computation_cost=0.0,
            prior_probability=0.15,
            prior_weight=0.0,
        )
        objective["loss"].backward()

        self.assertIsNotNone(gate.bias.grad)
        self.assertLess(float(gate.bias.grad), 0.0)

    def test_deep_gate_separates_pressure_from_external_controls(self):
        gate = DeepSleepGate(
            accumulator_decay=0.0,
            bias_initial=-1.0,
        )
        gate.eval()
        output = gate(
            torch.tensor([0.2, 0.3, 0.4]),
            split_demand=0.25,
            deep_availability=0.5,
            sample=False,
        )
        base = output["base_probability"]
        expected_pressure = 1.0 - (1.0 - base) * (1.0 - 0.25)
        expected_probability = 0.5 * expected_pressure

        self.assertEqual(gate.feature_names, (
            "residual", "memory", "topology"
        ))
        self.assertEqual(gate.raw_weights.numel(), 3)
        self.assertTrue(torch.allclose(
            output["pressure_probability"], expected_pressure
        ))
        self.assertTrue(torch.allclose(
            output["probability"], expected_probability
        ))

    def test_deep_gate_pressures_are_bounded_before_linear_evidence(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=0, memory_key_dim=3
        )
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            device="cpu",
        )
        features = trainer._deep_sleep_features(
            [], residual_energy=1e9, epoch_label=1
        )

        self.assertTrue(bool((features["tensor"] >= 0.0).all()))
        self.assertTrue(bool((features["tensor"] <= 1.0).all()))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            trainer.deep_sleep_gate(
                torch.tensor([2.0, 0.0, 0.0]),
                split_demand=0.0,
                deep_availability=1.0,
                sample=False,
            )

    def test_deep_gate_has_an_optimizer_disjoint_from_the_model(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=0, memory_key_dim=3
        )
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            sleep=SleepConfig(deep_gate_learning_rate=2e-3),
            device="cpu",
        )
        gate_ids = {
            id(parameter)
            for parameter in trainer.deep_sleep_gate.parameters()
        }
        model_ids = {
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        gate_optimizer_ids = {
            id(parameter)
            for group in trainer.deep_gate_optimizer.param_groups
            for parameter in group["params"]
        }

        self.assertTrue(gate_ids.isdisjoint(model_ids))
        self.assertEqual(gate_ids, gate_optimizer_ids)
        self.assertEqual(
            trainer.deep_gate_optimizer.param_groups[0]["lr"],
            2e-3,
        )

    def test_split_demand_uses_persistent_bank_mass_then_decays(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=0, memory_key_dim=3
        )
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            sleep=SleepConfig(
                deep_split_demand_decay=0.5,
                deep_split_queue_scale=1.0,
            ),
            device="cpu",
        )
        trainer.tree.episodic_memory.add_memory(
            "root",
            torch.tensor([1.0, 0.0, 0.0]),
            torch.zeros(tree.param_dim),
            write_quality=1.0,
            queue_weight=0.5,
        )
        first = trainer._continuous_split_demand()
        second = trainer._continuous_split_demand()
        trainer.tree.episodic_memory.add_memory(
            "root",
            torch.tensor([0.0, 1.0, 0.0]),
            torch.zeros(tree.param_dim),
            write_quality=1.0,
            queue_weight=0.5,
        )
        third = trainer._continuous_split_demand()

        self.assertGreater(first["E_bank_struct"], 0.0)
        self.assertGreater(first["value"], 0.0)
        self.assertLess(second["value"], first["value"])
        self.assertGreater(third["value"], second["value"])

    def test_closed_gate_probe_does_not_advance_structural_state(self):
        torch.manual_seed(433)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=1, memory_key_dim=3
        )
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            sleep=SleepConfig(
                deep_probe_interval=1,
                deep_gate_bias_initial=-20.0,
                deep_execution_threshold=0.999,
            ),
            device="cpu",
        )
        lambda_t_before = trainer.merge_lambda_T

        result = trainer.train_sleep(
            torch.ones(1, len(tree.leaf_ids)),
            epoch=1,
        )

        self.assertFalse(result["deep_gate"]["hard_gate"])
        self.assertTrue(result["deep_gate"]["probe"])
        self.assertTrue(result["deep_gate"]["evaluated"])
        self.assertEqual(result["transaction"]["actions"], [])
        self.assertEqual(tree.topology_prune_streak, {})
        self.assertEqual(tree.topology_prune_near_zero_streak, {})
        self.assertEqual(trainer.merge_lambda_T, lambda_t_before)
        self.assertEqual(trainer.sleep_state["last_deep_epoch"], 1)
        self.assertEqual(float(trainer.deep_sleep_gate.accumulator), 0.0)
        self.assertTrue(result["deep_gate"]["reset_after_evaluation"])
        self.assertEqual(result["control_flow"], [
            "freeze_snapshot",
            "bank_mode_probe",
            "light_or_preserve",
            "deep_gate",
            "build_candidates",
            "temporal_smoothing",
            "unified_selector",
            "reset_after_deep_evaluation",
        ])

    def test_open_gate_runs_deep_and_resets_accumulator(self):
        torch.manual_seed(439)
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=0, memory_key_dim=3
        )
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            sleep=SleepConfig(
                deep_gate_bias_initial=20.0,
                deep_execution_threshold=0.001,
                deep_probe_interval=10,
            ),
            device="cpu",
        )

        result = trainer.train_sleep(
            torch.ones(1, 1),
            epoch=1,
        )

        self.assertTrue(result["deep_gate"]["hard_gate"])
        self.assertEqual(result["mode"], "deep")
        self.assertEqual(trainer.sleep_state["last_deep_epoch"], 1)
        self.assertEqual(float(trainer.deep_sleep_gate.accumulator), 0.0)

    def test_sleep_scheduler_state_round_trips_checkpoint(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=0, memory_key_dim=3
        )
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            sleep=SleepConfig(deep_accumulator_decay=0.5),
            device="cpu",
        )
        trainer.sleep_state["last_deep_epoch"] = 7
        trainer.sleep_state["deep_cycle_count"] = 3
        trainer.sleep_state["structural_demand_ema"] = 0.4
        trainer.sleep_state["split_queue_snapshot"] = {"root": 0.75}
        gate_output = trainer.deep_sleep_gate(
            torch.tensor([0.1, 0.2, 0.3]),
            split_demand=0.0,
            deep_availability=1.0,
            sample=False,
        )
        trainer._optimize_deep_sleep_gate(
            gate_output,
            estimated_gain=1.0,
        )
        trainer.deep_sleep_gate.accumulator.fill_(1.25)
        trainer.deep_sleep_gate.bias.data.fill_(-0.75)
        trainer.topology_selector.action_bias.data.fill_(0.25)
        trainer.sleep_state["light_index"] = {
            "scan_cursors": {"root": 5},
            "direction_summaries": {
                "root": {
                    "centroids": torch.ones(1, tree.param_dim),
                    "support": torch.tensor([4.0]),
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sleep.pt"
            trainer.save_checkpoint(checkpoint, epoch=7)
            restored = MemoryTreeTrainer.from_checkpoint(
                checkpoint,
                device="cpu",
            )
        self.assertEqual(restored.sleep_state["last_deep_epoch"], 7)
        self.assertEqual(
            restored.sleep_state["deep_cycle_count"], 3
        )
        self.assertAlmostEqual(
            restored.sleep_state["structural_demand_ema"],
            0.4,
        )
        self.assertAlmostEqual(
            restored.sleep_state["split_queue_snapshot"]["root"],
            0.75,
        )
        self.assertAlmostEqual(
            float(restored.deep_sleep_gate.accumulator),
            1.25,
        )
        self.assertAlmostEqual(
            float(restored.deep_sleep_gate.bias.detach()),
            -0.75,
        )
        self.assertTrue(restored.deep_gate_optimizer.state)
        self.assertTrue(torch.allclose(
            restored.topology_selector.action_bias.detach(),
            torch.full((3,), 0.25),
        ))
        self.assertEqual(
            restored.sleep_state["light_index"]["scan_cursors"]["root"],
            5,
        )
        self.assertTrue(torch.equal(
            restored.sleep_state["light_index"][
                "direction_summaries"
            ]["root"]["support"],
            torch.tensor([4.0]),
        ))

    def test_trainer_topology_prune_evaluator_accepts_normalized_config(self):
        hawkes = HawkesFamily(2, 1, decays=torch.tensor([1.0]))
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=1, memory_key_dim=3
        )
        for child_id in tree.leaf_ids:
            for offset in (0.0, 0.05):
                tree.episodic_memory.add_memory(
                    child_id,
                    torch.randn(3),
                    torch.zeros(tree.param_dim),
                    EventWindow(
                        torch.tensor([0.1 + offset, 0.4 + offset]),
                        torch.tensor([0, 1]),
                        child_id,
                        0,
                        2,
                        True,
                    ),
                )
        trainer = MemoryTreeTrainer(
            tree,
            hawkes,
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=8),
            structure=StructureConfig(topology_prune_kwargs={
                "min_replay": 2,
                "min_effective_replay": 1.0,
                "min_branch_replay": 1,
                "patience": 1,
                "dual_initial": 0.25,
            }),
            device="cpu",
        )

        proposals, metrics = trainer._evaluate_topology_prune(
            max_replay=4,
            allow_candidate=False,
        )

        self.assertIn("root", proposals)
        self.assertTrue(proposals["root"].eligible)
        self.assertEqual(metrics["candidate_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
