"""Hand-checkable Good–Turing baselines with successes in the population.

Singletons count failure categories only; both the estimate and confidence
radius use the total number of session draws, including successes.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from av_core import classical_gt_bound
from av_discovery_fixed_primary import build_session_population, run_bootstrap_trial


class TestGoodTuringFullPopulation(unittest.TestCase):
    def setUp(self) -> None:
        # Sorted indices: A, A, B, success, success.
        self.modes, self.probabilities, _ = build_session_population(
            {"a": "A", "b": "A", "c": "B"},
            ["a", "b", "c", "d", "e"],
        )

    def estimate(self, draws):
        return run_bootstrap_trial(
            np.asarray(draws, dtype=np.int64), self.modes, self.probabilities
        )[1]

    def test_successes_only_affect_the_denominator(self) -> None:
        # success, A, success, B, A, A, B.
        # Failure-singleton counts: 0, 1, 1, 2, 1, 1, 0.
        gt = self.estimate([3, 0, 4, 2, 0, 0, 2])
        np.testing.assert_allclose(gt, [0, 1 / 2, 1 / 3, 2 / 4, 1 / 5, 1 / 6, 0])

    def test_singletons_are_categories_not_distinct_sessions(self) -> None:
        # Different sessions a and b share A, so A ceases to be a singleton.
        np.testing.assert_allclose(self.estimate([0, 1, 2]), [1, 0, 1 / 3])

    def test_successes_never_become_singletons(self) -> None:
        # Includes distinct success sessions, then a repeated success.
        np.testing.assert_array_equal(self.estimate([3, 4, 3]), [0, 0, 0])

    def test_uncorrected_bound_uses_all_draws(self) -> None:
        # One failure followed by 999 successes: f1=1, t=1000 (not t=1).
        gt = self.estimate([0] + [3] * 999)
        self.assertEqual(gt[-1], 1 / 1000)
        bound = classical_gt_bound(gt, alpha=0.05, n_bonferroni=1)
        expected = 1 / 1000 + (1 + math.sqrt(2)) * math.sqrt(math.log(80) / 1000)
        self.assertLess(expected, 1)  # Ensure clipping cannot hide a formula error.
        self.assertAlmostEqual(bound[-1], expected, places=13)

    def test_bonferroni_bound_uses_alpha_divided_by_horizon(self) -> None:
        gt = self.estimate([0] + [3] * 999)
        bound = classical_gt_bound(gt, alpha=0.05, n_bonferroni=1000)
        # delta=0.05/1000, so 4/delta=80000.
        expected = 1 / 1000 + (1 + math.sqrt(2)) * math.sqrt(math.log(80000) / 1000)
        self.assertLess(expected, 1)
        self.assertAlmostEqual(bound[-1], expected, places=13)
        uncorrected = classical_gt_bound(gt, alpha=0.05, n_bonferroni=1)
        self.assertTrue(np.all(bound >= uncorrected))
        self.assertGreater(bound[-1], uncorrected[-1])

    def test_no_observed_failures_still_has_positive_confidence_padding(self) -> None:
        gt = self.estimate([3, 4] * 500)
        np.testing.assert_array_equal(gt, np.zeros(1000))
        for correction, log_argument in [(1, 80), (1000, 80000)]:
            with self.subTest(n_bonferroni=correction):
                bound = classical_gt_bound(gt, alpha=0.05, n_bonferroni=correction)
                expected = (1 + math.sqrt(2)) * math.sqrt(math.log(log_argument) / 1000)
                self.assertAlmostEqual(bound[-1], expected, places=13)
                self.assertEqual(bound[0], 1)  # Early confidence bounds are clipped.
                self.assertTrue(np.all((bound > 0) & (bound <= 1)))


if __name__ == "__main__":
    unittest.main()
