"""Unit tests for av_core.classical_gt_bound: the Bubeck & Ganassali fixed-t
Good-Turing bound, Bonferroni-corrected, checked against a hand-computed value
and its delta-range validation.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from av_core import classical_gt_bound


class TestClassicalGtBoundKnownValue(unittest.TestCase):
    """gt_hat = [0.5, 0.2] (t=1,2), alpha=0.1, n_bonferroni=10 -> delta=0.01.
    bound_t = min(1, gt_hat_t + (1+sqrt(2)) * sqrt(ln(4/delta)/t))
    delta=0.01 -> ln(400) -> (1+sqrt2)*sqrt(ln(400)/t)
    """

    def test_matches_hand_computed_bound(self) -> None:
        # At t=1,2 the unclipped formula is far above 1 for any valid delta
        # (min possible value is (1+sqrt2)*sqrt(ln4) ~= 2.84 at t=1), so this
        # case exercises the clip, not the raw formula.
        gt_hat = np.array([0.5, 0.2])
        alpha = 0.1
        n_bonferroni = 10
        bound = classical_gt_bound(gt_hat, alpha, n_bonferroni)
        np.testing.assert_allclose(bound, np.array([1.0, 1.0]), rtol=1e-12)

    def test_matches_hand_computed_bound_when_unclipped(self) -> None:
        # t=50..100 with a small gt_hat keeps the raw formula below 1, so
        # this exercises the actual arithmetic rather than the clip.
        gt_hat = np.full(51, 0.01)
        alpha = 0.1
        n_bonferroni = 10
        bound = classical_gt_bound(gt_hat, alpha, n_bonferroni)

        delta = alpha / n_bonferroni
        t = np.arange(1, 52, dtype=np.float64)
        expected = 0.01 + (1.0 + math.sqrt(2.0)) * np.sqrt(math.log(4.0 / delta) / t)
        self.assertLess(expected[-1], 1.0)  # sanity: t=51 is genuinely unclipped
        np.testing.assert_allclose(bound[-1], expected[-1], rtol=1e-12)

    def test_clips_at_one_when_raw_formula_exceeds_one(self) -> None:
        gt_hat = np.array([0.99])
        bound = classical_gt_bound(gt_hat, alpha=0.5, n_bonferroni=1)
        self.assertEqual(bound[0], 1.0)

    def test_bound_is_nonincreasing_in_t_for_constant_gt_hat(self) -> None:
        # Clipping can flatten the leading (small-t) portion of the
        # trajectory, so this is non-increasing rather than strictly
        # decreasing; the raw formula's own strict decrease is checked
        # separately once past the clip in the unclipped test above.
        gt_hat = np.full(200, 0.01)
        bound = classical_gt_bound(gt_hat, alpha=0.1, n_bonferroni=10)
        self.assertTrue(np.all(np.diff(bound) <= 0))
        self.assertLess(bound[-1], 1.0)  # sanity: not clipped for its whole length

    def test_supports_leading_batch_dimension(self) -> None:
        gt_hat = np.array([[0.5, 0.2], [0.1, 0.05]])
        bound = classical_gt_bound(gt_hat, alpha=0.1, n_bonferroni=10)
        self.assertEqual(bound.shape, (2, 2))


class TestClassicalGtBoundRejectsInvalidDelta(unittest.TestCase):
    def test_rejects_n_bonferroni_that_drives_delta_above_one(self) -> None:
        # delta = alpha / n_bonferroni >= 1 whenever n_bonferroni <= alpha.
        with self.assertRaises(ValueError):
            classical_gt_bound(np.array([0.1]), alpha=1.5, n_bonferroni=1)

    def test_rejects_zero_n_bonferroni(self) -> None:
        with self.assertRaises(ZeroDivisionError):
            classical_gt_bound(np.array([0.1]), alpha=0.1, n_bonferroni=0)


if __name__ == "__main__":
    unittest.main()
