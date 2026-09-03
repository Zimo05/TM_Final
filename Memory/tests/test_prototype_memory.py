import unittest

import torch

from MemoryResiduals.EpisodicMemory import TreeEpisodicMemory
from MemoryResiduals.MemoryBank import (
    EventWindow,
    MemoryBank,
    SmoothSparseRetriever,
    effective_hawkes_law_key,
)
from Sleep.Split import SplitModule


class PrototypeMemoryTests(unittest.TestCase):
    def test_duplicate_context_aliases_use_online_redundancy(self):
        bank = MemoryBank(device="cpu", key_dim=4, param_dim=2, capacity=8)
        bank.configure_prototype_policy(
            duplicate_threshold=0.98,
            mode_threshold=0.90,
            gain_ema_decay=0.0,
            gain_confirmation_min_count=1,
            context_alias_capacity=3,
        )
        basis = torch.eye(4)
        for key in (basis[0], basis[0], basis[1]):
            bank.add(
                key,
                torch.zeros(2),
                law_key=basis[0],
                prediction_gain=1.0,
                force_new_mode_confirmation=True,
            )
        result = bank.add(
            basis[2],
            torch.zeros(2),
            law_key=basis[0],
            prediction_gain=1.0,
            force_new_mode_confirmation=True,
        )

        self.assertEqual(result["match_type"], "duplicate")
        self.assertEqual(result["context_action"], "merge_append")
        self.assertEqual(int(bank.context_valid[0].sum()), 3)
        self.assertAlmostEqual(float(bank.context_support[0].sum()), 4.0)
        self.assertAlmostEqual(float(bank.support[0]), 4.0)
        self.assertTrue(torch.allclose(bank.law_keys[0], basis[0]))

    def test_retrieval_scores_the_best_context_alias(self):
        bank = MemoryBank(device="cpu", key_dim=3, param_dim=2, capacity=8)
        bank.configure_prototype_policy(
            duplicate_threshold=0.98,
            mode_threshold=0.90,
            gain_ema_decay=0.0,
            gain_confirmation_min_count=1,
        )
        basis = torch.eye(3)
        bank.add(
            basis[0], torch.tensor([1.0, 0.0]), law_key=basis[0],
            prediction_gain=1.0,
        )
        bank.add(
            basis[1], torch.tensor([0.0, 1.0]), law_key=basis[0],
            prediction_gain=1.0,
            force_new_mode_confirmation=True,
        )
        _, info = bank.retrieve(
            basis[1], SmoothSparseRetriever(), update_state=False
        )
        self.assertAlmostEqual(float(info["sim"][0]), 1.0, places=6)

    def test_effective_law_key_is_invariant_to_semantic_rebase(self):
        semantic = torch.tensor([0.2, -0.4])
        delta = torch.tensor([0.3, 0.7])
        shift = torch.tensor([-0.1, 0.25])
        kwargs = dict(
            decays=torch.tensor([1.5]),
            num_event_types=1,
            num_basis=1,
            key_dim=8,
        )
        before = effective_hawkes_law_key(semantic, delta, **kwargs)
        after = effective_hawkes_law_key(semantic + shift, delta - shift, **kwargs)
        torch.testing.assert_close(before, after)

    def test_duplicate_refresh_is_bounded_and_preserves_evidence_mass(self):
        memory = TreeEpisodicMemory(
            key_dim=8,
            num_event_types=1,
            num_basis=1,
            capacity_per_node=8,
            device="cpu",
        )
        memory.configure_prototype_memory(
            duplicate_threshold=0.98,
            mode_threshold=0.90,
            mode_capacity=3,
        )
        semantic = torch.tensor([0.2, -0.4])
        delta = torch.tensor([0.3, 0.7])
        for quality, split in ((0.8, 0.2), (0.6, 0.4), (1.0, 0.0)):
            memory.add_memory(
                "root",
                torch.nn.functional.one_hot(torch.tensor(0), 8).float(),
                delta,
                write_quality=quality,
                queue_weight=split,
                semantic_theta=semantic,
                decays=torch.tensor([1.5]),
            )
        bank = memory.get_bank("root")
        self.assertEqual(len(bank), 1)
        self.assertEqual(float(bank.support[0]), 3.0)
        self.assertAlmostEqual(float(bank.quality_mass[0]), 2.4, places=6)
        self.assertAlmostEqual(float(bank.split_mass[0]), 0.6, places=6)
        self.assertAlmostEqual(float(bank.write_quality[0]), 0.8, places=6)
        self.assertAlmostEqual(float(bank.queue_weight[0]), 0.2, places=6)

    def test_full_bank_compresses_and_can_reactivate_old_mode(self):
        bank = MemoryBank(device="cpu", key_dim=8, param_dim=4, capacity=4)
        bank.configure_prototype_policy(
            duplicate_threshold=0.98,
            mode_threshold=0.90,
            mode_capacity=3,
            gain_ema_decay=0.0,
            gain_confirmation_min_count=1,
        )
        basis = torch.eye(8)
        delta = torch.arange(4, dtype=torch.float32)
        bank.add(basis[0], delta, law_key=basis[0], write_quality=0.8)
        nearby = torch.tensor([0.95, (1.0 - 0.95**2) ** 0.5, 0, 0, 0, 0, 0, 0])
        bank.add(basis[1], delta + 1, law_key=nearby, write_quality=0.6)
        bank.add(basis[2], delta + 2, law_key=basis[2])
        bank.add(basis[3], delta + 3, law_key=basis[3])
        bank.add(basis[4], delta + 4, law_key=basis[4])

        self.assertEqual(len(bank), 4)
        self.assertEqual(float(bank.support.sum()), 5.0)
        archived = torch.nonzero(bank.mode_compressed, as_tuple=False).flatten()
        self.assertEqual(archived.numel(), 1)
        archive_index = int(archived.item())
        archive_mode = int(bank.mode_ids[archive_index])

        bank.add(basis[0], delta, law_key=basis[0])
        self.assertEqual(len(bank), 4)
        self.assertEqual(float(bank.support.sum()), 6.0)
        rows = bank.mode_ids == archive_mode
        self.assertFalse(bool(bank.mode_compressed[rows].any()))

    def test_new_mode_requires_persistent_prediction_confirmation(self):
        bank = MemoryBank(device="cpu", key_dim=3, param_dim=2, capacity=8)
        bank.configure_prototype_policy(
            duplicate_threshold=0.98,
            mode_threshold=0.90,
            gain_ema_decay=0.0,
            gain_confirmation_min_count=2,
        )
        basis = torch.eye(3)
        bank.add(
            basis[0], torch.zeros(2), law_key=basis[0], prediction_gain=0.2
        )

        first = bank.add(
            basis[1], torch.ones(2), law_key=basis[1], prediction_gain=0.9
        )
        self.assertEqual(first["action"], "queue")
        self.assertEqual(first["match_type"], "pending_new_dynamics")
        self.assertEqual(len(bank), 1)

        second = bank.add(
            basis[1], torch.ones(2), law_key=basis[1], prediction_gain=0.9
        )
        self.assertEqual(second["action"], "append")
        self.assertEqual(second["match_type"], "new_dynamics")
        self.assertEqual(len(bank), 2)
        self.assertNotEqual(second["mode_id"], first["mode_id"])
        # A confirmed new dynamics must not feed its queued outlier distance
        # back into the old mode's adaptive radius.
        self.assertEqual(bank._mode_distances.get(0, []), [])
        self.assertEqual(bank._mode_pending_distances, {})

    def test_pending_confirmation_tracks_law_identity_per_mode(self):
        """Distinct outliers nearest one mode must not share a counter."""
        bank = MemoryBank(device="cpu", key_dim=3, param_dim=2, capacity=8)
        bank.configure_prototype_policy(
            duplicate_threshold=0.98,
            mode_threshold=0.90,
            gain_ema_decay=0.0,
            gain_confirmation_min_count=2,
        )
        basis = torch.eye(3)
        bank.add(
            basis[0], torch.zeros(2), law_key=basis[0], prediction_gain=0.2
        )
        candidate_x = torch.tensor([0.8, 0.6, 0.0])
        candidate_y = torch.tensor([0.8, 0.0, 0.6])

        first_x = bank.add(
            candidate_x,
            torch.ones(2),
            law_key=candidate_x,
            prediction_gain=0.9,
        )
        first_y = bank.add(
            candidate_y,
            torch.ones(2),
            law_key=candidate_y,
            prediction_gain=0.9,
        )
        self.assertEqual(first_x["action"], "queue")
        self.assertEqual(first_y["action"], "queue")
        self.assertEqual(first_y["confirmation_count"], 1)
        self.assertEqual(first_y["pending_candidate_count"], 2)
        self.assertEqual(len(bank), 1)

        # X recurs and confirms only X.  Y remains a separate temporary law
        # candidate under the original nearest mode.
        confirmed_x = bank.add(
            candidate_x,
            torch.ones(2),
            law_key=candidate_x,
            prediction_gain=0.9,
        )
        self.assertEqual(confirmed_x["match_type"], "new_dynamics")
        self.assertEqual(len(bank), 2)
        pending = bank._mode_pending_candidates[0]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["count"], 1)
        self.assertTrue(torch.allclose(pending[0]["law_key"], candidate_y))

        # Y then confirms independently and creates its own mode.
        confirmed_y = bank.add(
            candidate_y,
            torch.ones(2),
            law_key=candidate_y,
            prediction_gain=0.9,
        )
        self.assertEqual(confirmed_y["match_type"], "new_dynamics")
        self.assertEqual(len(bank), 3)
        self.assertEqual(bank._mode_pending_candidates, {})

    def test_queued_outlier_is_retrospectively_admitted_to_current_mode(self):
        bank = MemoryBank(device="cpu", key_dim=3, param_dim=2, capacity=8)
        bank.configure_prototype_policy(
            duplicate_threshold=0.98,
            mode_threshold=0.90,
            adaptive_min_samples=1,
            gain_ema_decay=0.0,
            gain_confirmation_min_count=2,
        )
        basis = torch.eye(3)
        bank.add(
            basis[0], torch.zeros(2), law_key=basis[0], prediction_gain=0.2
        )
        candidate = torch.tensor([0.8, 0.6, 0.0])

        first = bank.add(
            candidate,
            torch.ones(2),
            law_key=candidate,
            prediction_gain=0.0,
        )
        self.assertEqual(first["action"], "queue")
        self.assertEqual(first["pending_distance_count"], 1)
        self.assertEqual(len(bank), 1)

        second = bank.add(
            candidate,
            torch.ones(2),
            law_key=candidate,
            prediction_gain=0.0,
        )
        self.assertEqual(second["action"], "append")
        self.assertEqual(second["match_type"], "local_variation")
        self.assertEqual(second["mode_id"], 0)
        self.assertEqual(bank._mode_pending_distances, {})
        self.assertEqual(len(bank._mode_distances[0]), 2)
        # The accepted recurrence and the earlier queued observation both
        # calibrate the mode radius, allowing it to expand beyond the cold
        # start prior of 1 - mode_threshold.
        self.assertGreater(
            bank.adaptive_radii(0)[1], 1.0 - bank.mode_threshold
        )

    def test_local_variation_queues_when_mode_capacity_is_full(self):
        """A full mode must not overwrite a distinct law prototype row."""
        bank = MemoryBank(device="cpu", key_dim=3, param_dim=2, capacity=8)
        bank.configure_prototype_policy(
            duplicate_threshold=0.98,
            mode_threshold=0.90,
            mode_capacity=1,
            gain_ema_decay=0.0,
            gain_confirmation_min_count=2,
        )
        base = torch.tensor([1.0, 0.0, 0.0])
        candidate = torch.tensor([0.95, (1.0 - 0.95**2) ** 0.5, 0.0])
        original_delta = torch.tensor([1.0, -2.0])
        bank.add(
            base,
            original_delta,
            law_key=base,
            prediction_gain=0.2,
        )
        original_key = bank.keys[0].clone()
        original_law_key = bank.law_keys[0].clone()
        original_delta = bank.deltas[0].clone()
        original_context = bank.context_keys[0].clone()
        original_support = bank.support[0].clone()

        result = bank.add(
            candidate,
            torch.tensor([3.0, 4.0]),
            law_key=candidate,
            prediction_gain=0.0,
        )

        self.assertEqual(result["action"], "queue")
        self.assertEqual(result["match_type"], "local_variation_capacity")
        self.assertAlmostEqual(result["distance"], 0.05, places=6)
        self.assertEqual(len(bank), 1)
        torch.testing.assert_close(bank.keys[0], original_key)
        torch.testing.assert_close(bank.law_keys[0], original_law_key)
        torch.testing.assert_close(bank.deltas[0], original_delta)
        torch.testing.assert_close(bank.context_keys[0], original_context)
        torch.testing.assert_close(bank.support[0], original_support)
        self.assertEqual(bank._mode_distances, {})
        self.assertEqual(bank._mode_pending_candidates, {})

    def test_local_variation_queues_when_hard_bank_capacity_has_no_victim(self):
        """Hard-cap fallback must preserve law identity just like mode cap."""
        bank = MemoryBank(device="cpu", key_dim=3, param_dim=2, capacity=1)
        bank.configure_prototype_policy(
            duplicate_threshold=0.98,
            mode_threshold=0.90,
            mode_capacity=2,
        )
        base = torch.tensor([1.0, 0.0, 0.0])
        candidate = torch.tensor([0.95, (1.0 - 0.95**2) ** 0.5, 0.0])
        bank.add(base, torch.zeros(2), law_key=base)
        original_law_key = bank.law_keys[0].clone()

        result = bank.add(
            candidate,
            torch.ones(2),
            law_key=candidate,
        )

        self.assertEqual(result["action"], "queue")
        self.assertEqual(result["match_type"], "bank_capacity")
        self.assertEqual(len(bank), 1)
        torch.testing.assert_close(bank.law_keys[0], original_law_key)

    def test_pending_outlier_distances_survive_tree_checkpoint(self):
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=1,
            num_basis=1,
            capacity_per_node=8,
            device="cpu",
        )
        memory.configure_prototype_memory(
            adaptive_min_samples=1,
            gain_ema_decay=0.0,
            gain_confirmation_min_count=2,
        )
        basis = torch.eye(3)
        memory.add_memory(
            "root", basis[0], torch.zeros(2), law_key=basis[0], prediction_gain=0.2
        )
        candidate = torch.tensor([0.8, 0.6, 0.0])
        memory.add_memory(
            "root", candidate, torch.ones(2), law_key=candidate, prediction_gain=0.0
        )

        restored = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=1,
            num_basis=1,
            capacity_per_node=8,
            device="cpu",
        )
        restored.set_extra_state(memory.get_extra_state())
        restored_bank = restored.get_bank("root")
        self.assertEqual(len(restored_bank._mode_pending_distances[0]), 1)
        self.assertAlmostEqual(restored_bank._mode_pending_distances[0][0], 0.2, places=6)
        self.assertEqual(len(restored_bank._mode_pending_candidates[0]), 1)
        self.assertEqual(restored_bank._mode_pending_candidates[0][0]["count"], 1)

        result = restored.add_memory(
            "root", candidate, torch.ones(2), law_key=candidate, prediction_gain=0.0
        )
        self.assertEqual(result["match_type"], "local_variation")
        self.assertEqual(restored_bank._mode_pending_distances, {})

    def test_batched_admission_uses_the_same_causal_confirmation(self):
        bank = MemoryBank(device="cpu", key_dim=3, param_dim=2, capacity=8)
        bank.configure_prototype_policy(
            duplicate_threshold=0.98,
            mode_threshold=0.90,
            gain_ema_decay=0.0,
            gain_confirmation_min_count=2,
        )
        basis = torch.eye(3)
        first = bank.add_batch(
            keys=basis[:2],
            delta_theta=torch.stack([torch.zeros(2), torch.ones(2)]),
            law_keys=basis[:2],
            prediction_gain=torch.tensor([0.2, 0.9]),
        )
        self.assertEqual([row["action"] for row in first], ["append", "queue"])
        self.assertEqual(len(bank), 1)

        second = bank.add_batch(
            keys=basis[1:2],
            delta_theta=torch.ones(1, 2),
            law_keys=basis[1:2],
            prediction_gain=torch.tensor([0.9]),
        )
        self.assertEqual(second[0]["action"], "append")
        self.assertEqual(len(bank), 2)

    def test_mode_local_radii_are_rolling_quantiles(self):
        bank = MemoryBank(device="cpu", key_dim=4, param_dim=2, capacity=8)
        bank.configure_prototype_policy(
            duplicate_threshold=0.90,
            mode_threshold=0.70,
            adaptive_history_size=4,
            adaptive_min_samples=3,
        )
        base = torch.tensor([1.0, 0.0, 0.0, 0.0])
        bank.add(base, torch.zeros(2), law_key=base, prediction_gain=0.2)
        for cosine in (0.99, 0.98, 0.97):
            candidate = torch.tensor([
                cosine,
                (1.0 - cosine**2) ** 0.5,
                0.0,
                0.0,
            ])
            result = bank.add(
                candidate,
                torch.zeros(2),
                law_key=candidate,
                prediction_gain=0.2,
            )
            self.assertEqual(result["action"], "refresh")

        mode_id = int(bank.mode_ids[0])
        duplicate_radius, mode_radius = bank.adaptive_radii(mode_id)
        expected = MemoryBank._rolling_quantile(
            bank._mode_duplicate_distances[mode_id], 0.90
        )
        self.assertAlmostEqual(duplicate_radius, expected, places=7)
        self.assertGreaterEqual(
            mode_radius, duplicate_radius + bank.radius_margin - 1e-12
        )

        memory = TreeEpisodicMemory(
            key_dim=4,
            num_event_types=1,
            num_basis=1,
            capacity_per_node=8,
            device="cpu",
        )
        memory.banks["root"] = bank
        memory.configure_prototype_memory(
            duplicate_threshold=0.90,
            mode_threshold=0.70,
            adaptive_history_size=4,
            adaptive_min_samples=3,
        )
        restored = TreeEpisodicMemory(
            key_dim=4,
            num_event_types=1,
            num_basis=1,
            capacity_per_node=8,
            device="cpu",
        )
        restored.set_extra_state(memory.get_extra_state())
        restored_bank = restored.get_bank("root")
        self.assertEqual(
            restored_bank._mode_duplicate_distances,
            bank._mode_duplicate_distances,
        )
        self.assertEqual(
            restored_bank.adaptive_radii(mode_id),
            (duplicate_radius, mode_radius),
        )

    def test_split_uses_observation_support_after_physical_compression(self):
        memory = TreeEpisodicMemory(
            key_dim=2,
            num_event_types=1,
            num_basis=1,
            capacity_per_node=4,
            device="cpu",
        )
        window = EventWindow(
            times=torch.tensor([0.1]),
            types=torch.tensor([0]),
            node_id="root",
            start_idx=0,
            end_idx=1,
            has_full_history=True,
        )
        for _ in range(15):
            memory.add_memory(
                "root",
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.2, -0.1]),
                window,
                write_quality=1.0,
                queue_weight=1.0,
            )
        bank = memory.get_bank("root")
        batch = SplitModule.build_split_batch_from_memory_bank(bank)
        self.assertEqual(len(bank), 1)
        self.assertEqual(float(bank.support[0]), 15.0)
        self.assertEqual(float(batch.sample_support[0]), 15.0)
        self.assertAlmostEqual(float(batch.effective_sample_size), 15.0, places=5)

        module = SplitModule(P=2, z_dim=2)
        with torch.no_grad():
            module.centers.zero_()
        structure = module.compute_soft_residual_structure(
            residuals=batch.residuals,
            weights=batch.weights,
            theta_sem=torch.zeros(2),
            sample_support=batch.sample_support,
        )
        self.assertAlmostEqual(
            float(structure["N_mass"].sum().detach()), 15.0, places=5
        )
        torch.testing.assert_close(
            structure["N_eff"], torch.full((2,), 15.0), atol=1e-5, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
