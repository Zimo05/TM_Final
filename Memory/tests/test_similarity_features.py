import unittest

import torch

from MemoryResiduals.SimilarityFeatures import local_recurrence_count


class LocalRecurrenceCountTests(unittest.TestCase):
    def test_empty_bank_has_zero_count(self):
        count, support = local_recurrence_count(torch.empty(0))

        self.assertEqual(float(count), 0.0)
        self.assertEqual(float(support), 0.0)

    def test_orthogonal_bank_is_invariant_to_capacity(self):
        one_count, one_support = local_recurrence_count(
            torch.zeros(1)
        )
        full_count, full_support = local_recurrence_count(
            torch.zeros(128)
        )

        self.assertEqual(float(one_count), 0.0)
        self.assertEqual(float(full_count), 0.0)
        self.assertEqual(float(one_support), 0.0)
        self.assertEqual(float(full_support), 0.0)

    def test_count_is_monotone_in_number_of_similar_memories(self):
        counts = [
            local_recurrence_count(torch.ones(size))[0]
            for size in (1, 3, 6)
        ]

        self.assertLess(float(counts[0]), float(counts[1]))
        self.assertLess(float(counts[1]), float(counts[2]))

    def test_low_similarity_distractors_do_not_change_count(self):
        before, _ = local_recurrence_count(torch.ones(3))
        after, _ = local_recurrence_count(
            torch.cat((torch.ones(3), torch.full((100,), 0.1)))
        )

        self.assertLess(abs(float(before - after)), 1e-8)

    def test_default_sums_the_complete_128_capacity(self):
        count, support = local_recurrence_count(torch.ones(128))
        expected_count = 1.0 - torch.exp(torch.tensor(-128.0 / 3.0))

        self.assertAlmostEqual(float(support), 128.0, places=6)
        self.assertTrue(torch.allclose(count, expected_count))


if __name__ == "__main__":
    unittest.main()

