"""Unit tests for the hindsight-lambda (closed form) restart mixture: the KL identity
against the raw product, that the closed-form lambda really is the maximiser, and that
the resulting mixture dominates any fixed-lambda mixture on the same restart structure.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from av_closed_form_lambda_mixture import (
    closed_form_log_eprocess_at_u,
    closed_form_upper_confidence_trajectory,
    kl_bernoulli,
)
from av_lambda_restart_mixture import (
    compile_everytime_restart_mixture,
    everytime_lambda_mixture_upper_confidence_trajectory,
)


def raw_product_log_e(z_slice, u):
    """log prod_i (1 + lam(u - z_i)) with lam built from the slice, done literally."""
    z_bar = float(np.mean(z_slice))
    lam = max((u - z_bar) / (u * (1.0 - u)), 0.0)
    return float(np.sum(np.log1p(lam * (u - np.asarray(z_slice, dtype=float)))))


class TestKLIdentity(unittest.TestCase):
    def test_kl_form_matches_the_raw_product(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(25):
            n = int(rng.integers(2, 30))
            z = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(np.int64)
            u = float(rng.uniform(0.05, 0.95))
            if np.mean(z) >= u:
                continue  # lambda = 0 branch, E = 1
            expected = raw_product_log_e(z, u)
            got = n * float(kl_bernoulli(np.array([np.mean(z)]), np.array([u]))[0])
            self.assertAlmostEqual(got, expected, places=10)

    def test_all_zero_slice_is_finite(self) -> None:
        z = np.zeros(6, dtype=np.int64)
        got = 6 * float(kl_bernoulli(np.array([0.0]), np.array([0.3]))[0])
        self.assertAlmostEqual(got, raw_product_log_e(z, 0.3), places=10)
        self.assertTrue(math.isfinite(got))


class TestClosedFormLambdaIsTheMaximiser(unittest.TestCase):
    def test_beats_every_lambda_on_a_fine_grid(self) -> None:
        rng = np.random.default_rng(1)
        z = (rng.random(40) < 0.25).astype(np.int64)
        u = 0.4
        z_bar = float(np.mean(z))
        best_closed = len(z) * float(kl_bernoulli(np.array([z_bar]), np.array([u]))[0])
        grid = np.linspace(1e-4, 1.0 / (1.0 - u) - 1e-4, 4000)
        best_grid = max(
            float(np.sum(np.log1p(lam * (u - z.astype(float))))) for lam in grid
        )
        self.assertGreaterEqual(best_closed, best_grid - 1e-9)


class TestClosedFormRequiresItsThreshold(unittest.TestCase):
    def test_log_t_coefficient_is_required_and_keyword_only(self) -> None:
        """No default: coefficient 0 is the hindsight ("cheating") reading and 1 is the
        discounted one. A caller that omitted it would silently get the invalid bound."""
        z = (np.random.default_rng(3).random(30) < 0.2).astype(np.int64)
        with self.assertRaises(TypeError):
            closed_form_upper_confidence_trajectory(z, 0.05, 20, "log_squared")
        with self.assertRaises(TypeError):
            closed_form_upper_confidence_trajectory(z, 0.05, 20, "log_squared", 0.0)


class TestClosedFormMixture(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(2)
        self.z = (rng.random(120) < 0.2).astype(np.int64)

    def test_mixture_matches_an_explicit_per_restart_sum(self) -> None:
        n = len(self.z)
        u = np.full(n, 0.35)
        got = np.exp(closed_form_log_eprocess_at_u(self.z, u, "log_squared"))
        mixture, reserve = compile_everytime_restart_mixture(self.z, "log_squared")
        weights = mixture.weights
        expected = np.empty(n)
        for t in range(n):
            total = reserve[t + 1]
            for s in range(t + 1):
                total += weights[s] * math.exp(raw_product_log_e(self.z[s:t + 1], float(u[t])))
            expected[t] = total
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_dominates_the_fixed_lambda_mixture_so_the_bound_is_tighter(self) -> None:
        closed = closed_form_upper_confidence_trajectory(
            self.z, 0.05, 30, "log_squared", log_t_coefficient=0.0)
        for lam in (0.1, 0.25, 0.5):
            fixed = everytime_lambda_mixture_upper_confidence_trajectory(
                self.z, np.array([lam]), np.array([1.0]), 0.05, 30, "log_squared")
            self.assertTrue(np.all(closed <= fixed + 1e-6), f"not tighter than lambda={lam}")

    def test_bound_stays_inside_the_unit_interval(self) -> None:
        bound = closed_form_upper_confidence_trajectory(
            self.z, 0.05, 30, "log_squared", log_t_coefficient=0.0)
        self.assertTrue(np.all(bound > 0.0) and np.all(bound < 1.0))

    def test_block_size_does_not_change_the_answer(self) -> None:
        u = np.full(len(self.z), 0.3)
        base = closed_form_log_eprocess_at_u(self.z, u, "log_squared", block_size=256)
        for block in (1, 7, 64):
            got = closed_form_log_eprocess_at_u(self.z, u, "log_squared", block_size=block)
            np.testing.assert_allclose(got, base, rtol=0, atol=1e-12)

    def test_rejects_u_outside_the_open_unit_interval(self) -> None:
        for bad in (0.0, 1.0):
            with self.assertRaises(ValueError):
                closed_form_log_eprocess_at_u(self.z, np.full(len(self.z), bad), "log_squared")

    def test_rejects_wrong_length_u(self) -> None:
        with self.assertRaises(ValueError):
            closed_form_log_eprocess_at_u(self.z, np.full(len(self.z) - 1, 0.3), "log_squared")


if __name__ == "__main__":
    unittest.main()
