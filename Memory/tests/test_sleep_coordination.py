import unittest
from unittest.mock import patch

import torch

from HawkesBackbone import HawkesFamily
from LatentHawkesTree import HawkesTree
from MemoryResiduals.MemoryBank import EventWindow, MemoryBank
from MemoryResiduals.Replay import replay_log_likelihood
from Sleep.Coordinator import run_sleep_cycle
from Sleep.Collapse import collapse_leaf_pair, collapse_snapshot_signature
from Sleep.Merge import (
    branch_support_and_gain,
    commit_merge,
    compute_differentiable_merge_objective,
    find_shared_memory_promotions,
    make_merge_decisions,
)
import Sleep.Prune as sleep_prune
from Sleep.Split import SplitModule, commit_split
from Sleep.TopologyPrune import (
    TopologyPruneProposal,
    _batched_topology_prune_terms,
    _event_local_tpp_divergence,
    apply_prune_persistence,
    evaluate_topology_prune,
    evaluate_topology_prune_candidate,
)
from Sleep.UnifiedTopology import (
    TopologyActionKind,
    UnifiedTopologyCandidate,
    UnifiedTopologySelection,
    UnifiedTopologySelector,
    build_merge_candidate,
    build_split_candidate,
    build_topology_prune_candidate,
)


class SleepCoordinationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.decays = torch.tensor([1.0])
        self.hawkes = HawkesFamily(2, 1, decays=self.decays)

    @staticmethod
    def _null_selection():
        return UnifiedTopologySelector(temperature=1.0)(
            []
        )

    @staticmethod
    def _prune_selection(proposal):
        candidate = build_topology_prune_candidate(
            proposal,
            topology_revision=0,
        )
        return UnifiedTopologySelector(temperature=1.0)(
            [candidate]
        )

    def test_coordinator_rejects_missing_unified_selection(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        with self.assertRaisesRegex(
            ValueError,
            "requires UnifiedTopologySelection",
        ):
            run_sleep_cycle(
                tree,
                torch.ones(1, 1),
                self.decays,
                hawkes_ll=self.hawkes,
                selection=None,
                statistics_prepared=True,
            )

    def test_selector_policy_is_tensor_backed_and_learnable(self):
        selector = UnifiedTopologySelector(temperature=1.0)
        candidate = UnifiedTopologyCandidate(
            action_id="split:0:root",
            kind=TopologyActionKind.SPLIT,
            target="root",
            claims=frozenset(("root",)),
            payload=None,
            eligible=True,
            ready=True,
            raw_gain=torch.tensor(2.0),
            uncertainty=torch.tensor(0.0),
            conservative_gain=torch.tensor(2.0),
            effective_sample_size=8.0,
            replay_size=8,
        )
        selection = selector([candidate])
        objective = selector.objective(selection, entropy_weight=0.0)
        objective["loss"].backward()

        self.assertTrue(selection.probability_tensor.requires_grad)
        self.assertIsNotNone(selector.raw_action_scale.grad)
        self.assertNotEqual(
            float(selector.raw_action_scale.grad[0]), 0.0
        )
        self.assertEqual(selection.selected_action_id, "split:0:root")
        self.assertIsInstance(
            selection.as_log_dict()["probabilities"]["null"], float
        )

    def test_merge_candidate_requires_common_evidence_contract(self):
        common = {
            "node_a": "root_L",
            "node_b": "root_R",
            "parent_id": "root",
            "snapshot_signature": (),
            "delta_keep": 0.1,
            "delta_keep_variance": 4.0,
            "replay_size": 8,
            "topology_revision": 0,
            "lambda_T": 1.0,
            "uncertainty_kappa": 1.0,
            "min_replay_size": 8,
            "min_effective_sample_size": 4.0,
            "min_branch_support": 2,
        }
        insufficient = build_merge_candidate(
            **common,
            effective_sample_size=3.0,
            branch_support=(8.0, 1.0),
        )
        eligible = build_merge_candidate(
            **common,
            effective_sample_size=4.0,
            branch_support=(4.0, 4.0),
        )

        self.assertFalse(insufficient.eligible)
        self.assertEqual(
            insufficient.diagnostics["reason"],
            "insufficient_effective_sample_size",
        )
        self.assertTrue(eligible.eligible)
        self.assertAlmostEqual(float(eligible.uncertainty), 1.0)

    def test_split_candidate_does_not_reintroduce_retired_support_gates(self):
        candidate = build_split_candidate(
            "root",
            torch.nn.Identity(),
            {
                "logp0": torch.tensor([-2.0]),
                "logp_child_mix": torch.tensor([-1.0]),
                "replay_weights": torch.ones(1),
                "N_eff": torch.tensor([0.1, 0.1]),
                "structural_strength": torch.tensor(0.0),
                "effective_sample_size": torch.tensor(1.0),
                "E_bank_struct": 0.0,
            },
            topology_revision=0,
            lambda_T=0.0,
            uncertainty_kappa=0.0,
            min_child_effective_mass=10.0,
            min_structural_strength=0.05,
            min_effective_sample_size=2.0,
        )

        self.assertTrue(candidate.eligible)
        self.assertTrue(candidate.ready)

    def test_prune_persistence_is_eligibility_not_preference(self):
        proposal = TopologyPruneProposal(
            parent_id="root",
            child_ids=("root_L", "root_R"),
            snapshot_signature=(),
            eligible=True,
            reason="ready_to_compare",
            keep_mode="test",
            replay_size=8,
            effective_replay_size=8.0,
            branch_balance=1.0,
            keep_nll=0.0,
            prune_nll=1.0,
            predictive_damage=1.0,
            near_zero_damage=False,
            uncertainty_margin=0.0,
            tpp_divergence=0.0,
            retention_cost=1.0,
            complexity_saving=1.0,
            prior_probability=0.5,
            prune_gain=-1.0,
            prune_probability=0.1,
            locally_positive=False,
            persistence_ok=True,
            persistence=3,
        )
        candidate = build_topology_prune_candidate(
            proposal,
            topology_revision=0,
        )
        selection = UnifiedTopologySelector(temperature=1.0)([candidate])

        self.assertTrue(candidate.ready)
        self.assertTrue(selection.is_null)

    def test_split_commit_requires_unified_authorization(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        module = SplitModule(tree.param_dim, 3, m_min=0.0)
        theta = tree.semantic_theta("root").detach()
        with self.assertRaisesRegex(PermissionError, "authorization"):
            commit_split(
                tree,
                "root",
                module,
                {
                    "theta_plus": theta,
                    "theta_cand": torch.stack((theta, theta)),
                },
            )
        self.assertEqual(tree.leaf_ids, ["root"])

    def test_replay_is_shift_invariant_and_conditions_on_prefix(self):
        theta = torch.tensor([-1.0, -1.5, -2.0, -2.0, -2.0, -2.0])
        first = EventWindow(
            torch.tensor([0.1, 0.4, 0.9]),
            torch.tensor([0, 1, 0]),
            "leaf",
            1,
            3,
            True,
        )
        shifted = EventWindow(
            first.times + 100.0,
            first.types,
            "leaf",
            1,
            3,
            True,
        )
        self.assertTrue(torch.allclose(
            replay_log_likelihood(first, theta, self.hawkes),
            replay_log_likelihood(shifted, theta, self.hawkes),
            atol=1e-5,
        ))

    def test_batched_promotion_score_matches_scalar_replay(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        bank = tree.episodic_memory.get_bank("root")
        for index in range(3):
            tree.episodic_memory.add_memory(
                "root",
                torch.tensor([1.0, float(index + 1), 0.5]),
                0.01 * torch.randn(tree.param_dim),
                EventWindow(
                    torch.tensor([0.1, 0.4, 0.8 + 0.1 * index]),
                    torch.tensor([0, 1, 0]),
                    "root",
                    1,
                    3,
                    True,
                ),
            )
        candidate_key = torch.tensor([0.4, 0.8, 0.2])
        candidate_delta = 0.02 * torch.randn(tree.param_dim)
        theta = tree.semantic_theta("root").detach()
        support, gain = branch_support_and_gain(
            candidate_key,
            candidate_delta,
            bank,
            theta,
            self.decays,
            2,
            1,
            hawkes_ll=self.hawkes,
        )

        normalized_candidate = torch.nn.functional.normalize(
            candidate_key, dim=0
        )
        weights = []
        gains = []
        for index, window in enumerate(bank.windows):
            replay_key = torch.nn.functional.normalize(bank.keys[index], dim=0)
            weights.append(torch.sigmoid(
                (torch.dot(normalized_candidate, replay_key) - 0.7) / 0.1
            ))
            base = replay_log_likelihood(window, theta, self.hawkes)
            corrected = replay_log_likelihood(
                window, theta + candidate_delta, self.hawkes
            )
            gains.append(corrected - base)
        weights = torch.stack(weights)
        gains = torch.stack(gains)
        self.assertTrue(torch.allclose(support, weights.sum(), atol=1e-6))
        self.assertTrue(torch.allclose(
            gain,
            (weights * gains).sum() / (weights.sum() + 1e-8),
            atol=1e-6,
        ))

    def test_promotion_similarity_uses_all_candidate_context_aliases(self):
        """A reusable non-first candidate alias must affect branch support."""
        bank = MemoryBank(
            device="cpu", key_dim=3, param_dim=6, capacity=2
        )
        window = EventWindow(
            torch.tensor([0.1, 0.4, 0.9]),
            torch.tensor([0, 1, 0]),
            "branch",
            0,
            3,
            True,
        )
        branch_key = torch.tensor([1.0, 0.0, 0.0])
        bank.add(
            branch_key,
            torch.zeros(6),
            window=window,
            law_key=branch_key,
        )
        hawkes = HawkesFamily(2, 1, decays=self.decays)
        candidate_aliases = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        support, _ = branch_support_and_gain(
            candidate_aliases,
            torch.zeros(6),
            bank,
            torch.zeros(6),
            self.decays,
            2,
            1,
            hawkes_ll=hawkes,
        )
        expected = torch.sigmoid(torch.tensor((1.0 - 0.7) / 0.1))
        self.assertAlmostEqual(float(support), float(expected), places=6)

    def test_shared_promotion_keeps_all_source_context_aliases(self):
        """Promotion candidates must not collapse to ``source_bank.keys``."""
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        node_a, node_b = tree.leaf_ids
        window_a = EventWindow(
            torch.tensor([0.1]), torch.tensor([0]), node_a, 0, 1, True
        )
        window_b = EventWindow(
            torch.tensor([0.2]), torch.tensor([1]), node_b, 0, 1, True
        )
        first_alias = torch.tensor([1.0, 0.0, 0.0])
        second_alias = torch.tensor([0.0, 1.0, 0.0])
        tree.episodic_memory.add_memory(
            node_a,
            first_alias,
            torch.zeros(tree.param_dim),
            window=window_a,
            law_key=first_alias,
        )
        tree.episodic_memory.add_memory(
            node_b,
            first_alias,
            torch.zeros(tree.param_dim),
            window=window_b,
            law_key=first_alias,
        )
        source_bank = tree.episodic_memory.get_bank(node_a)
        source_bank.context_keys[0, 1] = second_alias
        source_bank.context_valid[0, 1] = True
        source_bank.context_support[0, 1] = 1.0

        captured = []

        def fake_branch_score(candidate_key, *args, **kwargs):
            captured.append(candidate_key.detach().clone())
            return torch.tensor(2.0), torch.tensor(1.0)

        with patch("Sleep.Merge.branch_support_and_gain", side_effect=fake_branch_score):
            records = find_shared_memory_promotions(
                tree,
                "root",
                self.decays,
                min_support=1.0,
                min_gain=0.0,
                max_candidates=1,
            )

        self.assertEqual(len(records), 2)
        self.assertGreaterEqual(len(captured), 2)
        torch.testing.assert_close(
            captured[0], source_bank.context_keys[0, :2]
        )

    def test_topology_prune_batched_terms_match_scalar_replay(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        hawkes = HawkesFamily(
            2,
            1,
            decays=self.decays.to(device),
        ).to(device)
        windows = [
            EventWindow(
                torch.tensor([0.1, 0.4, 0.9]),
                torch.tensor([0, 1, 0]),
                "left",
                1,
                3,
                True,
            ),
            EventWindow(
                torch.tensor([0.2, 0.5, 0.8, 1.2]),
                torch.tensor([1, 0, 1, 0]),
                "right",
                0,
                4,
                True,
            ),
        ]
        keep_theta = torch.tensor(
            [
                [-20.0, -20.0, -20.0, -20.0, -20.0, -20.0],
                [-0.8, -1.4, -1.9, -2.2, -1.6, -1.5],
            ],
            device=device,
        )
        prune_theta = keep_theta + torch.tensor(
            [
                [0.1, -0.05, 0.02, -0.03, 0.04, -0.01],
                [-0.02, 0.08, -0.04, 0.01, -0.03, 0.05],
            ],
            device=device,
        )

        comparison_eps = 1e-5
        keep_nll, prune_nll, divergence = (
            _batched_topology_prune_terms(
                windows,
                keep_theta,
                prune_theta,
                hawkes,
                comparison_eps,
            )
        )
        scalar_keep = torch.stack([
            -replay_log_likelihood(window, theta, hawkes)
            for window, theta in zip(windows, keep_theta)
        ])
        scalar_prune = torch.stack([
            -replay_log_likelihood(window, theta, hawkes)
            for window, theta in zip(windows, prune_theta)
        ])
        scalar_divergence = torch.stack([
            _event_local_tpp_divergence(
                window,
                theta_keep,
                theta_prune,
                hawkes,
                comparison_eps,
            )
            for window, theta_keep, theta_prune in zip(
                windows, keep_theta, prune_theta
            )
        ])

        self.assertEqual(keep_nll.device.type, device.type)
        self.assertTrue(torch.allclose(
            keep_nll, scalar_keep, atol=1e-6, rtol=1e-5
        ))
        self.assertTrue(torch.allclose(
            prune_nll, scalar_prune, atol=1e-6, rtol=1e-5
        ))
        self.assertTrue(torch.allclose(
            divergence,
            scalar_divergence,
            atol=1e-6,
            rtol=1e-5,
        ))

    def test_topology_prune_candidates_share_one_tree_forward(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=2, memory_key_dim=3)
        for leaf_id in tree.leaf_ids:
            tree.episodic_memory.add_memory(
                leaf_id,
                torch.randn(3),
                0.01 * torch.randn(tree.param_dim),
                EventWindow(
                    torch.tensor([0.1, 0.4, 0.9]),
                    torch.tensor([0, 1, 0]),
                    leaf_id,
                    1,
                    3,
                    True,
                ),
            )

        batch_calls = 0

        def scalar_embedding(window):
            return torch.zeros(tree.z_dim)

        def batch_embedding(windows):
            nonlocal batch_calls
            batch_calls += 1
            return torch.zeros(len(windows), tree.z_dim)

        with patch.object(tree, "forward", wraps=tree.forward) as tree_forward:
            proposals, _ = evaluate_topology_prune(
                tree,
                self.hawkes,
                lambda_T=0.0,
                embedding_fn=scalar_embedding,
                embedding_batch_fn=batch_embedding,
                min_replay=2,
                min_effective_replay=1.0,
                min_branch_replay=1,
                max_replay=2,
            )
        self.assertEqual(set(proposals), {"root_L", "root_R"})
        self.assertEqual(batch_calls, 1)
        self.assertEqual(tree_forward.call_count, 1)

        for parent_id, batched in proposals.items():
            scalar = evaluate_topology_prune_candidate(
                tree,
                parent_id,
                self.hawkes,
                lambda_T=0.0,
                embedding_fn=scalar_embedding,
                min_replay=2,
                min_effective_replay=1.0,
                min_branch_replay=1,
                max_replay=2,
            )
            self.assertAlmostEqual(
                batched.retention_cost,
                scalar.retention_cost,
                places=6,
            )
            self.assertAlmostEqual(
                batched.prune_gain,
                scalar.prune_gain,
                places=6,
            )

    def test_merge_uses_frozen_production_child_mixture(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        left, right = tree.leaf_ids
        windows = []
        for node_id in (left, right):
            for offset in (0.0, 0.05):
                window = EventWindow(
                    torch.tensor([0.1 + offset, 0.4 + offset]),
                    torch.tensor([0, 1]),
                    node_id,
                    0,
                    2,
                    True,
                )
                windows.append(window)
                tree.episodic_memory.add_memory(
                    node_id,
                    torch.randn(3),
                    torch.zeros(tree.param_dim),
                    window,
                )

        result = compute_differentiable_merge_objective(
            tree,
            [(left, right)],
            self.decays,
            lambda_T=0.0,
            gate_temperature=1.0,
            min_replay=1,
            hawkes_ll=self.hawkes,
        )
        theta_left = tree.semantic_theta(left)
        theta_right = tree.semantic_theta(right)
        theta_parent = 0.5 * (theta_left + theta_right)
        manual_child = []
        manual_parent = []
        log_half = theta_left.new_tensor(0.5).log()
        for window in windows:
            nll_left = -replay_log_likelihood(
                window, theta_left, self.hawkes
            )
            nll_right = -replay_log_likelihood(
                window, theta_right, self.hawkes
            )
            manual_child.append(-torch.logsumexp(torch.stack((
                log_half - nll_left,
                log_half - nll_right,
            )), dim=0))
            manual_parent.append(-replay_log_likelihood(
                window, theta_parent, self.hawkes
            ))
        # The production KEEP baseline includes the routed frontier mixture
        # and memory state.  The old semantic-only 50/50 calculation above is
        # retained as a sanity diagnostic, but is no longer the commit score.
        self.assertTrue(torch.isfinite(result["delta_keep"][0]))
        child_weights = result["child_weights"][0]
        expected_parent = (
            child_weights[0] * theta_left
            + child_weights[1] * theta_right
        )
        self.assertTrue(torch.allclose(
            result["merge_parent_theta"][0], expected_parent, atol=1e-6
        ))
        self.assertEqual(result["complexity_saving"].tolist(), [1.0])
        self.assertEqual(float(result["effective_sample_size"][0]), 4.0)
        self.assertEqual(result["branch_support"][0].tolist(), [2.0, 2.0])
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertFalse(result["loss"].requires_grad)

        decisions = make_merge_decisions(
            result,
            lambda_T=0.0,
            gate_temperature=1.0,
        )
        self.assertIn((left, right), decisions)
        self.assertIn("keep_probability", decisions[(left, right)])

    def test_sleep_prune_module_no_longer_deletes_memory_rows(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        for index in range(4):
            tree.episodic_memory.add_memory(
                "root",
                torch.randn(3),
                0.02 * torch.randn(tree.param_dim),
                EventWindow(
                    torch.tensor([0.1, 0.3 + 0.02 * index]),
                    torch.tensor([0, 1]),
                    "root",
                    0,
                    2,
                    True,
                ),
            )
        before = len(tree.episodic_memory.get_bank("root"))
        self.assertFalse(hasattr(sleep_prune, "optimize_variational_memory_prune"))
        self.assertFalse(hasattr(sleep_prune, "commit_variational_memory_prune"))
        run_sleep_cycle(
            tree,
            torch.ones(1, 1),
            self.decays,
            hawkes_ll=self.hawkes,
            selection=self._null_selection(),
            statistics_prepared=True,
        )
        self.assertEqual(len(tree.episodic_memory.get_bank("root")), before)

    def test_split_and_merge_preserve_effective_memory_parameters(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        module = SplitModule(tree.param_dim, 3, m_min=0.0)
        module.centers.data[0].fill_(-0.2)
        module.centers.data[1].fill_(0.2)
        old_theta = tree.semantic_theta("root").detach()
        for value in (-0.2, 0.2):
            tree.episodic_memory.add_memory(
                "root",
                torch.randn(3),
                torch.full((tree.param_dim,), value),
                EventWindow(
                    torch.tensor([0.1, 0.4]),
                    torch.tensor([0, 1]),
                    "root",
                    0,
                    2,
                    True,
                ),
            )
        effective_before = old_theta + tree.episodic_memory.banks["root"].deltas.clone()
        left, right = commit_split(
            tree,
            "root",
            module,
            {
                "theta_plus": old_theta + 0.1,
                "theta_cand": torch.stack((old_theta - 0.3, old_theta + 0.4)),
                "logp0": torch.tensor([-2.0, -2.0]),
                "logp_child_each": torch.tensor([
                    [-1.0, -3.0],
                    [-3.0, -1.0],
                ]),
            },
            authorized=True,
        )
        self.assertTrue(torch.allclose(
            tree.semantic_theta(left) + tree.episodic_memory.banks[left].deltas[0],
            effective_before[0],
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            tree.semantic_theta(right) + tree.episodic_memory.banks[right].deltas[0],
            effective_before[1],
            atol=1e-6,
        ))
        left_effective = (
            tree.semantic_theta(left).detach()
            + tree.episodic_memory.banks[left].deltas[0].clone()
        )
        commit_merge(
            tree,
            left,
            right,
            decays=self.decays,
            hawkes_ll=self.hawkes,
            memory_hard_threshold=1e6,
        )
        self.assertTrue(torch.allclose(
            tree.semantic_theta("root") + tree.episodic_memory.banks["root"].deltas[0],
            left_effective,
            atol=1e-6,
        ))

    def test_split_fits_child_hypernetwork_initialization_before_exact_offset(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        module = SplitModule(tree.param_dim, 3, m_min=0.0)
        parent_theta = tree.semantic_theta("root").detach()
        targets = torch.stack((parent_theta - 0.4, parent_theta + 0.6))
        output = {
            "theta_plus": parent_theta,
            "theta_cand": targets,
        }
        left, right = commit_split(
            tree,
            "root",
            module,
            output,
            init_steps=50,
            init_lr=2e-2,
            authorized=True,
        )
        self.assertLess(output["init_loss_after"], output["init_loss_before"])
        self.assertTrue(torch.allclose(tree.semantic_theta(left), targets[0], atol=1e-6))
        self.assertTrue(torch.allclose(tree.semantic_theta(right), targets[1], atol=1e-6))

    def test_merge_combines_all_banks_without_deleting_memory(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        left, right = tree.leaf_ids
        theta_parent = tree.semantic_theta("root").detach().clone()
        child_theta = {
            node_id: tree.semantic_theta(node_id).detach().clone()
            for node_id in (left, right)
        }
        expected_parent = 0.5 * (child_theta[left] + child_theta[right])
        sources = (("root", 0, 1.0), (left, 1, 2.0), (right, 2, 3.0))
        for node_id, marker, usage in sources:
            tree.episodic_memory.add_memory(
                node_id,
                torch.randn(3),
                torch.full((tree.param_dim,), 0.1 * (marker + 1)),
                EventWindow(
                    torch.tensor([0.1, 0.4]),
                    torch.tensor([0, 1]),
                    node_id,
                    marker,
                    marker + 2,
                    True,
                ),
            )
            tree.episodic_memory.get_bank(node_id).usage.fill_(usage)

        parent_scores = {0: -1.0, 1: -2.0, 2: -1.1}
        effective_scores = {0: -1.0, 1: -1.0, 2: -1.0}

        def fake_log_likelihood(window, theta, *args, **kwargs):
            scores = (
                parent_scores
                if torch.allclose(theta, theta_parent)
                else effective_scores
            )
            return theta.new_tensor(scores[window.start_idx])

        with patch(
            "Sleep.Merge._window_log_likelihood",
            side_effect=fake_log_likelihood,
        ):
            parent = commit_merge(
                tree,
                left,
                right,
                decays=self.decays,
                hawkes_ll=self.hawkes,
                memory_hard_threshold=0.2,
            )

        self.assertEqual(parent, "root")
        self.assertFalse(torch.equal(expected_parent, theta_parent))
        self.assertTrue(torch.allclose(
            tree.semantic_theta(parent), expected_parent, atol=1e-6
        ))
        parent_bank = tree.episodic_memory.get_bank(parent)
        # Topology commit preserves all evidence. Memory deletion is a
        # separate variational transaction.
        self.assertEqual(len(parent_bank), 3)
        self.assertEqual(parent_bank.usage.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(
            [window.node_id for window in parent_bank.windows],
            [parent, parent, parent],
        )

    def test_split_parent_bank_keeps_only_parent_explanation_advantage(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=0, memory_key_dim=3)
        module = SplitModule(tree.param_dim, 3, m_min=0.0)
        old_theta = tree.semantic_theta("root").detach()
        old_deltas = []
        for index, value in enumerate((-0.3, 0.1, 0.4, 0.7)):
            delta = torch.full((tree.param_dim,), value)
            old_deltas.append(delta)
            tree.episodic_memory.add_memory(
                "root",
                torch.randn(3),
                delta,
                None if index == 3 else EventWindow(
                    torch.tensor([0.1, 0.4]),
                    torch.tensor([0, 1]),
                    "root",
                    0,
                    2,
                    True,
                ),
            )
        source_bank = tree.episodic_memory.get_bank("root")
        source_bank.usage.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        source_bank.age.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))

        theta_plus = old_theta + 0.2
        child_theta = torch.stack((old_theta - 0.4, old_theta + 0.5))
        left, right = commit_split(
            tree,
            "root",
            module,
            {
                "theta_plus": theta_plus,
                "theta_cand": child_theta,
                # Memory 0 has a parent-only advantage. Memories 1 and 2 are
                # explained at least as well by the left and right child.
                "logp0": torch.tensor([-1.0, -1.0, -1.0]),
                "logp_child_each": torch.tensor([
                    [-1.8, -2.0],
                    [-0.9, -2.0],
                    [-2.0, -0.7],
                ]),
            },
            memory_hard_threshold=0.5,
            authorized=True,
        )

        parent_bank = tree.episodic_memory.get_bank("root")
        left_bank = tree.episodic_memory.get_bank(left)
        right_bank = tree.episodic_memory.get_bank(right)
        self.assertEqual((len(parent_bank), len(left_bank), len(right_bank)), (2, 1, 1))
        # The unscored legacy item is inherited by the parent conservatively.
        self.assertEqual(parent_bank.usage.tolist(), [1.0, 4.0])
        self.assertEqual(parent_bank.age.tolist(), [4.0, 1.0])
        self.assertTrue(torch.allclose(
            tree.semantic_theta("root") + parent_bank.deltas[0],
            old_theta + old_deltas[0],
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            tree.semantic_theta(left) + left_bank.deltas[0],
            old_theta + old_deltas[1],
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            tree.semantic_theta(right) + right_bank.deltas[0],
            old_theta + old_deltas[2],
            atol=1e-6,
        ))

    def test_leaf_prune_is_symmetric_memory_preserving_collapse(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        removed, survivor = tree.leaf_ids
        source_thetas = {
            "root": tree.semantic_theta("root").detach().clone(),
            survivor: tree.semantic_theta(survivor).detach().clone(),
            removed: tree.semantic_theta(removed).detach().clone(),
        }
        records = (
            ("root", 0, 1.0, True),
            (survivor, 1, 2.0, True),
            (removed, 2, 3.0, True),
            (survivor, 3, 4.0, False),
            (removed, 4, 5.0, False),
            (removed, 5, 6.0, True),
        )
        old_effective = {}
        for node_id, marker, usage, has_window in records:
            delta = torch.full((tree.param_dim,), 0.1 * (marker + 1))
            old_effective[usage] = source_thetas[node_id] + delta
            tree.episodic_memory.add_memory(
                node_id,
                torch.randn(3),
                delta,
                EventWindow(
                    torch.tensor([0.1, 0.4]),
                    torch.tensor([0, 1]),
                    node_id,
                    marker,
                    marker + 2,
                    True,
                ) if has_window else None,
            )
            tree.episodic_memory.get_bank(node_id).usage[-1] = usage

        parent_theta = source_thetas["root"]
        collapse = collapse_leaf_pair(tree, "root")
        parent = collapse.parent_id

        self.assertEqual(parent, "root")
        self.assertTrue(torch.equal(tree.semantic_theta(parent), parent_theta))
        parent_bank = tree.episodic_memory.get_bank(parent)
        self.assertEqual(sorted(parent_bank.usage.tolist()), [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(
            window is None or window.node_id == parent
            for window in parent_bank.windows
        ))
        for index, usage in enumerate(parent_bank.usage.tolist()):
            self.assertTrue(torch.allclose(
                tree.semantic_theta(parent) + parent_bank.deltas[index],
                old_effective[usage],
                atol=1e-6,
            ))

    def test_topology_prune_evaluates_then_commits_complete_refinement(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        parent_theta = tree.semantic_theta("root").detach().clone()
        for node_id in tree.leaf_ids:
            for offset in (0.0, 0.05):
                tree.episodic_memory.add_memory(
                    node_id,
                    torch.randn(3),
                    torch.zeros(tree.param_dim),
                    EventWindow(
                        torch.tensor([0.1 + offset, 0.4 + offset]),
                        torch.tensor([0, 1]),
                        node_id,
                        0,
                        2,
                        True,
                    ),
                )
        usage_before = {
            node_id: bank.usage.clone()
            for node_id, bank in tree.episodic_memory.banks.items()
        }
        bank_ids_before = set(tree.episodic_memory.banks)
        proposal = evaluate_topology_prune_candidate(
            tree,
            "root",
            self.hawkes,
            lambda_T=10.0,
            embedding_fn=lambda window: torch.zeros(3),
            min_replay=2,
            min_effective_replay=1.0,
            min_branch_replay=1,
            max_replay=4,
            gate_beta=0.1,
            near_zero_damage_threshold=0.0,
        )
        self.assertTrue(proposal.eligible)
        self.assertEqual(proposal.keep_mode, "production_parameter_mix")
        self.assertGreater(proposal.prune_probability, 0.5)
        self.assertTrue(all(
            torch.equal(tree.episodic_memory.banks[node_id].usage, value)
            for node_id, value in usage_before.items()
        ))
        self.assertEqual(set(tree.episodic_memory.banks), bank_ids_before)
        self.assertTrue(all(
            window.hawkes_cache_signature is None
            for bank in tree.episodic_memory.banks.values()
            for window in bank.windows
            if window is not None
        ))
        proposal = apply_prune_persistence(
            tree,
            {"root": proposal},
            patience=1,
            allow_candidate=True,
        )["root"]
        result = run_sleep_cycle(
            tree,
            torch.tensor([[0.5, 0.5]]),
            self.decays,
            hawkes_ll=self.hawkes,
            selection=self._prune_selection(proposal),
            statistics_prepared=True,
        )
        self.assertEqual(result["actions"][0]["action"], "topology_prune")
        self.assertEqual(tree.leaf_ids, ["root"])
        self.assertTrue(torch.equal(tree.semantic_theta("root"), parent_theta))
        self.assertEqual(len(tree.episodic_memory.get_bank("root")), 4)

    def test_near_zero_damage_waits_once_then_prunes(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)

        def add_replay(offset):
            for child_id in tree.leaf_ids:
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

        add_replay(0.0)
        add_replay(0.05)
        first = evaluate_topology_prune_candidate(
            tree,
            "root",
            self.hawkes,
            lambda_T=0.0,
            embedding_fn=lambda window: torch.zeros(3),
            min_replay=2,
            min_effective_replay=1.0,
            min_branch_replay=1,
            max_replay=8,
            near_zero_damage_threshold=1e9,
        )
        first = apply_prune_persistence(
            tree,
            {"root": first},
            patience=1,
            near_zero_confirmations=2,
            allow_candidate=True,
        )["root"]
        self.assertTrue(first.near_zero_damage)
        self.assertFalse(first.persistence_ok)
        self.assertEqual(first.reason, "near_zero_collect_replay")
        self.assertEqual(first.persistence, 1)

        restored = HawkesTree(
            3, 4, 2, 1, init_depth=1, memory_key_dim=3
        )
        restored.load_state_dict(tree.state_dict())
        tree = restored
        self.assertEqual(
            tree.topology_prune_near_zero_streak["root"],
            1,
        )

        # The next Sleep snapshot contains additional replay. If D_hat is
        # still near zero, the second observation confirms the collapse.
        add_replay(0.1)
        second = evaluate_topology_prune_candidate(
            tree,
            "root",
            self.hawkes,
            lambda_T=0.0,
            embedding_fn=lambda window: torch.zeros(3),
            min_replay=2,
            min_effective_replay=1.0,
            min_branch_replay=1,
            max_replay=8,
            near_zero_damage_threshold=1e9,
        )
        second = apply_prune_persistence(
            tree,
            {"root": second},
            patience=1,
            near_zero_confirmations=2,
            allow_candidate=True,
        )["root"]
        self.assertTrue(second.persistence_ok)
        self.assertEqual(second.reason, "near_zero_confirmed")
        self.assertEqual(second.persistence, 2)

        result = run_sleep_cycle(
            tree,
            torch.tensor([[0.5, 0.5]]),
            self.decays,
            hawkes_ll=self.hawkes,
            selection=self._prune_selection(second),
            statistics_prepared=True,
        )
        self.assertEqual(result["actions"][0]["action"], "topology_prune")
        self.assertEqual(
            result["actions"][0]["decision_reason"],
            "near_zero_confirmed",
        )
        self.assertEqual(tree.leaf_ids, ["root"])

    def test_collapse_rejects_stale_snapshot_without_mutation(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        snapshot = collapse_snapshot_signature(tree, "root")
        left = tree.leaf_ids[0]
        tree.episodic_memory.add_memory(
            left,
            torch.randn(3),
            torch.zeros(tree.param_dim),
        )
        with self.assertRaisesRegex(RuntimeError, "snapshot is stale"):
            collapse_leaf_pair(
                tree,
                "root",
                snapshot_signature=snapshot,
            )
        self.assertEqual(len(tree.leaf_ids), 2)

        parameter_tree = HawkesTree(
            3, 4, 2, 1, init_depth=1, memory_key_dim=3
        )
        parameter_snapshot = collapse_snapshot_signature(
            parameter_tree, "root"
        )
        with torch.no_grad():
            next(parameter_tree.hyper.parameters()).add_(0.01)
        with self.assertRaisesRegex(RuntimeError, "snapshot is stale"):
            collapse_leaf_pair(
                parameter_tree,
                "root",
                snapshot_signature=parameter_snapshot,
            )

    def test_collapse_overflow_capacity_round_trips_checkpoint(self):
        tree = HawkesTree(
            3,
            4,
            2,
            1,
            init_depth=1,
            memory_key_dim=3,
            memory_capacity_per_node=2,
        )
        distinct_keys = (
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([-1.0, 0.0, 0.0]),
        )
        key_index = 0
        for child_id in tree.leaf_ids:
            for index in range(2):
                tree.episodic_memory.add_memory(
                    child_id,
                    distinct_keys[key_index],
                    torch.zeros(tree.param_dim),
                )
                key_index += 1
        collapse_leaf_pair(tree, "root")
        self.assertEqual(len(tree.episodic_memory.get_bank("root")), 4)
        self.assertEqual(tree.episodic_memory.get_bank("root").capacity, 4)
        expected_reconciliation = {
            "root": {"policy": "prune", "delay_cycles": 1}
        }
        self.assertEqual(tree.memory_reconciliation, expected_reconciliation)

        restored = HawkesTree(
            3,
            4,
            2,
            1,
            init_depth=1,
            memory_key_dim=3,
            memory_capacity_per_node=2,
        )
        restored.load_state_dict(tree.state_dict())
        restored_bank = restored.episodic_memory.get_bank("root")
        self.assertEqual(len(restored_bank), 4)
        self.assertEqual(restored_bank.capacity, 4)
        self.assertEqual(
            restored.memory_reconciliation, expected_reconciliation
        )

        run_sleep_cycle(
            restored,
            torch.tensor([[1.0]]),
            self.decays,
            hawkes_ll=self.hawkes,
            selection=self._null_selection(),
            statistics_prepared=True,
        )
        self.assertEqual(len(restored_bank), 4)
        self.assertEqual(restored_bank.capacity, 4)
        self.assertEqual(restored.memory_reconciliation, expected_reconciliation)

    def test_disjoint_collapse_snapshots_commit_in_one_transaction(self):
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=2, memory_key_dim=3
        )
        snapshots = {
            parent_id: collapse_snapshot_signature(tree, parent_id)
            for parent_id in ("root_L", "root_R")
        }

        collapse_leaf_pair(
            tree,
            "root_L",
            snapshot_signature=snapshots["root_L"],
        )
        collapse_leaf_pair(
            tree,
            "root_R",
            snapshot_signature=snapshots["root_R"],
        )

        self.assertEqual(set(tree.leaf_ids), {"root_L", "root_R"})

    def test_collapse_preserves_lazy_effective_ages(self):
        tree = HawkesTree(
            3, 4, 2, 1, init_depth=1, memory_key_dim=3
        )
        left, right = tree.leaf_ids
        tree.episodic_memory.add_memory(
            left,
            torch.ones(3),
            torch.zeros(tree.param_dim),
        )
        tree.episodic_memory.step_age(3)
        tree.episodic_memory.add_memory(
            right,
            torch.ones(3),
            torch.zeros(tree.param_dim),
        )
        snapshot = collapse_snapshot_signature(tree, "root")

        collapse_leaf_pair(
            tree,
            "root",
            snapshot_signature=snapshot,
        )

        bank = tree.episodic_memory.get_bank("root")
        self.assertEqual(sorted(bank.age.tolist()), [0.0, 3.0])
        self.assertEqual(
            bank._age_reference_clock,
            tree.episodic_memory._age_clock,
        )

    def test_sleep_cycle_does_not_apply_stale_memory_deletion(self):
        tree = HawkesTree(2, 3, 1, 1, init_depth=0, memory_key_dim=2)
        tree.episodic_memory.add_memory(
            "root",
            torch.ones(2),
            torch.zeros(tree.param_dim),
            EventWindow(torch.tensor([0.1]), torch.tensor([0]), "root", 0, 1),
        )
        run_sleep_cycle(
            tree,
            torch.ones(1, 1),
            self.decays,
            hawkes_ll=self.hawkes,
            selection=self._null_selection(),
        )
        self.assertEqual(len(tree.episodic_memory.get_bank("root")), 1)

    def test_dynamic_topology_checkpoint_and_conflict_order(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        key = torch.ones(3)
        for leaf in tree.leaf_ids:
            for _ in range(4):
                tree.episodic_memory.add_memory(
                    leaf,
                    key,
                    torch.zeros(tree.param_dim),
                    EventWindow(
                        torch.tensor([0.1, 0.3]),
                        torch.tensor([0, 1]),
                        leaf,
                        0,
                        2,
                        True,
                    ),
                )
        left = tree.leaf_ids[0]
        module = SplitModule(tree.param_dim, 3, m_min=0.0)
        theta = tree.semantic_theta(left).detach()
        merge_candidate = UnifiedTopologyCandidate(
            action_id="merge:0:root",
            kind=TopologyActionKind.MERGE,
            target="root",
            claims=frozenset(("root", *tree.leaf_ids)),
            payload={
                "parent_id": "root",
                "child_ids": tuple(tree.leaf_ids),
                "snapshot_signature": collapse_snapshot_signature(tree, "root"),
            },
            eligible=True,
            ready=True,
            raw_gain=1.0,
            uncertainty=0.0,
            conservative_gain=1.0,
            effective_sample_size=8.0,
            replay_size=8,
        )
        selection = UnifiedTopologySelector(temperature=1.0)(
            [merge_candidate]
        )
        result = run_sleep_cycle(
            tree,
            torch.tensor([[0.5, 0.5]]),
            self.decays,
            hawkes_ll=self.hawkes,
            selection=selection,
            statistics_prepared=True,
        )
        self.assertEqual([item["action"] for item in result["actions"]], ["merge"])
        self.assertEqual(
            set(result["mutated_memory_nodes"]),
            {"root", *result["leaf_snapshot"]},
        )
        self.assertEqual(result["memory"], {})
        tree.mass_ema["root"] = 0.7
        state = tree.state_dict()
        restored = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        restored.load_state_dict(state)
        self.assertEqual(restored.leaf_ids, ["root"])
        self.assertEqual(restored.mass_ema["root"], 0.7)

    def test_topology_prune_warmup_does_not_delete_low_mass_leaf(self):
        tree = HawkesTree(3, 4, 2, 1, init_depth=1, memory_key_dim=3)
        right = "root_R"
        for _ in range(3):
            result = run_sleep_cycle(
                tree,
                torch.tensor([[1.0, 0.0]]),
                self.decays,
                hawkes_ll=self.hawkes,
                selection=self._null_selection(),
                allow_topology_prune=False,
                promotion_kwargs={
                    "min_support": 100.0,
                    "min_gain": 1e9,
                    "min_balance": 1.0,
                },
            )
            self.assertFalse(result["topology_prune_enabled"])
            self.assertEqual(result["actions"], [])
        self.assertIn(right, tree.leaf_ids)
        self.assertEqual(tree.low_mass_streak.get(right, 0), 0)


if __name__ == "__main__":
    unittest.main()
