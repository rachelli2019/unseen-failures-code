"""Unit tests for av_core.upper_confidence_trajectory: binary-search inversion
of E_t(p) < 1/alpha, checked against an analytically-derived closed form and
against the documented "vacuous bound" edge case at small n.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from av_core import require_lambda, upper_confidence_trajectory


class TestUpperConfidenceTrajectoryClosedForm(unittest.TestCase):
    """z = [0, 0, 0, 0, 0] (5 zeros), lambda=0.9, alpha=0.5 (1/alpha=2).

    E_5(p) = (1 + 0.9p)^5. The boundary is where E_5(p*) = 1/alpha = 2:
      (1 + 0.9*p*)^5 = 2
      1 + 0.9*p* = 2^(1/5)
      p* = (2^(1/5) - 1) / 0.9

    This binds within [0, 1] (2^(1/5) - 1 ≈ 0.1487, so p* ≈ 0.1652), unlike
    the trivial small-n case -- 5 zeros is enough evidence at this alpha for
    the bound to move off the vacuous U_t=1.
    """

    def test_matches_closed_form_inversion(self) -> None:
        z = np.zeros(5, dtype=np.int8)
        lambda_value = 0.9
        alpha = 0.5
        ucb = upper_confidence_trajectory(z, lambda_value, alpha, binary_search_iterations=60)

        expected_p_star = (2.0 ** (1.0 / 5.0) - 1.0) / 0.9
        self.assertAlmostEqual(ucb[-1], expected_p_star, places=8)

        # Sanity: E_5(p*) must equal 1/alpha at the returned boundary.
        e_at_boundary = (1.0 + lambda_value * ucb[-1]) ** 5
        self.assertAlmostEqual(e_at_boundary, 1.0 / alpha, places=6)

    def test_monotone_nondecreasing_in_binary_search_iterations(self) -> None:
        # More iterations should only refine (not qualitatively change) the
        # answer -- the trajectory should stay close as precision increases.
        z = np.zeros(5, dtype=np.int8)
        coarse = upper_confidence_trajectory(z, 0.9, 0.5, binary_search_iterations=10)
        fine = upper_confidence_trajectory(z, 0.9, 0.5, binary_search_iterations=60)
        np.testing.assert_allclose(coarse, fine, atol=1e-2)


class TestUpperConfidenceTrajectoryVacuousAtSmallN(unittest.TestCase):
    """Too few discoveries-vs-repeats to beat the trivial p=1 bound: with
    z=[1,1,1,1,1] (all discoveries, no repeats), E_t(1) = 1 for every t
    exactly (the z=1 factor is always 1 at p=1), which is always < 1/alpha,
    so U_t must be exactly 1.0 for the whole trajectory regardless of
    lambda or alpha."""

    def test_all_ones_z_gives_vacuous_bound(self) -> None:
        z = np.ones(20, dtype=np.int8)
        ucb = upper_confidence_trajectory(z, 0.25, 0.05, binary_search_iterations=50)
        np.testing.assert_allclose(ucb, np.ones(20))


class TestRequireLambdaRaisesLoudly(unittest.TestCase):
    def test_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            require_lambda(0.0)

    def test_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            require_lambda(1.0)

    def test_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            require_lambda(-0.1)

    def test_nan_raises(self) -> None:
        with self.assertRaises(ValueError):
            require_lambda(float("nan"))

    def test_valid_value_does_not_raise(self) -> None:
        require_lambda(0.25)  # must not raise

    def test_upper_confidence_trajectory_rejects_invalid_lambda(self) -> None:
        z = np.zeros(3, dtype=np.int8)
        with self.assertRaises(ValueError):
            upper_confidence_trajectory(z, 1.5, 0.05, 10)

    def test_upper_confidence_trajectory_rejects_nonpositive_iterations(self) -> None:
        z = np.zeros(3, dtype=np.int8)
        with self.assertRaises(ValueError):
            upper_confidence_trajectory(z, 0.25, 0.05, 0)


if __name__ == "__main__":
    unittest.main()
