"""Unit tests for av_lambda_restart_mixture: the lambda-and-restart double mixture
against a brute-force triple sum, the fast every-time evaluator against the
general one, and the uniform midpoint lambda grid.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from av_core import (
    build_restart_schedule,
    restart_mixture_tail_mass,
    restart_mixture_weights_for_scheme,
    restarted_mixture_upper_confidence_trajectory,
)
from av_lambda_restart_mixture import (
    everytime_lambda_mixture_upper_confidence_trajectory,
    lambda_restart_mixture_log_eprocess_at_p,
    lambda_restart_mixture_upper_confidence_trajectory,
    uniform_lambda_grid,
)


def brute_force_mixture(z, lambda_values, pi_weights, p, weights, tail_mass):
    n = len(z)
    out = np.empty(n)
    for t in range(n):
        total = tail_mass
        for lam, pi in zip(lambda_values, pi_weights):
            for s in range(n):  # 0-based start index; every-time schedule
                e = 1.0
                for i in range(s, t + 1):
                    e *= 1.0 + lam * (p[t] - z[i])
                total += pi * weights[s] * e
        out[t] = total
    return out


class TestLambdaRestartMixture(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.z = (rng.random(40) < 0.2).astype(np.int64)
        self.p = rng.random(40)
        self.starts = build_restart_schedule(len(self.z), 1)
        self.weights, total = restart_mixture_weights_for_scheme("log_squared", len(self.starts))
        self.tail_mass = restart_mixture_tail_mass(self.weights, total)

    def test_matches_brute_force_triple_sum(self) -> None:
        lambda_values = np.array([0.1, 0.4, 0.9])
        pi_weights = np.array([0.2, 0.3, 0.5])
        got = np.exp(lambda_restart_mixture_log_eprocess_at_p(
            self.z, lambda_values, pi_weights, self.p, self.starts, "log_squared"))
        expected = brute_force_mixture(self.z, lambda_values, pi_weights, self.p, self.weights, self.tail_mass)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_fast_everytime_matches_general_for_several_block_sizes(self) -> None:
        lambda_values, pi_weights = uniform_lambda_grid(9)
        expected = lambda_restart_mixture_upper_confidence_trajectory(
            self.z, lambda_values, pi_weights, 0.05, 25, self.starts, "log_squared")
        for block_size in (1, 7, 40, 256):
            got = everytime_lambda_mixture_upper_confidence_trajectory(
                self.z, lambda_values, pi_weights, 0.05, 25, "log_squared", block_size)
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12, err_msg=f"block_size={block_size}")

    def test_single_lambda_grid_matches_av_core_restart_mixture(self) -> None:
        expected = restarted_mixture_upper_confidence_trajectory(
            self.z, 0.25, 0.05, self.starts, self.weights, self.tail_mass, binary_search_iterations=20)
        got = everytime_lambda_mixture_upper_confidence_trajectory(
            self.z, np.array([0.25]), np.array([1.0]), 0.05, 20, "log_squared")
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_rejects_nonpositive_block_size(self) -> None:
        with self.assertRaises(ValueError):
            everytime_lambda_mixture_upper_confidence_trajectory(
                self.z, np.array([0.25]), np.array([1.0]), 0.05, 20, "log_squared", 0)


class TestBothEntryPointsValidateIdentically(unittest.TestCase):
    """The two public entry points must impose the SAME restart-schedule contract.
    They shared no validation code before, so this pins the behaviour rather than
    relying on both happening to reach compile_restart_mixture."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.z = (rng.random(20) < 0.2).astype(np.int64)
        self.lv = np.array([0.25])
        self.pw = np.array([1.0])
        self.p = np.full(20, 0.3)

    def _both(self, starts):
        def at_p():
            lambda_restart_mixture_log_eprocess_at_p(
                self.z, self.lv, self.pw, self.p, starts, "log_squared")

        def ucb():
            lambda_restart_mixture_upper_confidence_trajectory(
                self.z, self.lv, self.pw, 0.05, 5, starts, "log_squared")

        return at_p, ucb

    def test_both_reject_the_same_bad_schedules(self) -> None:
        bad = {
            "unsorted": np.array([5, 3, 8]),
            "zero start": np.array([0, 3, 8]),
            "past the horizon": np.array([1, 5, 99]),
            "duplicate": np.array([1, 3, 3]),
            "two-dimensional": np.array([[1, 2], [3, 4]]),
            "empty": np.array([], dtype=np.int64),
        }
        for name, starts in bad.items():
            for label, fn in zip(("log_eprocess_at_p", "upper_confidence_trajectory"),
                                 self._both(starts)):
                with self.assertRaises(ValueError, msg=f"{label} accepted {name} starts"):
                    fn()

    def test_both_accept_a_valid_schedule(self) -> None:
        for fn in self._both(np.arange(1, 21)):
            fn()


class TestUniformLambdaGrid(unittest.TestCase):
    def test_midpoints_and_equal_weights(self) -> None:
        lambda_values, weights = uniform_lambda_grid(4)
        np.testing.assert_allclose(lambda_values, [0.125, 0.375, 0.625, 0.875])
        np.testing.assert_allclose(weights, [0.25] * 4)

    def test_grid_stays_strictly_inside_unit_interval(self) -> None:
        lambda_values, weights = uniform_lambda_grid(1000)
        self.assertGreater(lambda_values.min(), 0.0)
        self.assertLess(lambda_values.max(), 1.0)
        self.assertAlmostEqual(weights.sum(), 1.0, places=12)

    def test_rejects_nonpositive_points(self) -> None:
        with self.assertRaises(ValueError):
            uniform_lambda_grid(0)


class TestSingleProcessLambdaMixture(unittest.TestCase):
    """No-restart lambda mixture: one process from t = 1."""

    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        self.z = (rng.random(400) < 0.1).astype(np.int8)

    def test_single_lambda_equals_av_core_single_process(self) -> None:
        from av_core import upper_confidence_trajectory
        from av_lambda_restart_mixture import single_process_lambda_mixture_upper_confidence_trajectory
        for lv in (0.1, 0.25, 0.5):
            got = single_process_lambda_mixture_upper_confidence_trajectory(
                self.z, np.array([lv]), np.array([1.0]), 0.05, 50)
            want = upper_confidence_trajectory(self.z, lv, 0.05, 50)
            np.testing.assert_allclose(got, want, atol=1e-12)

    def test_mixture_matches_explicit_per_lambda_sum(self) -> None:
        from av_core import log_eprocess_at_p
        from av_lambda_restart_mixture import single_process_lambda_mixture_upper_confidence_trajectory
        lambdas, weights = np.array([0.1, 0.4, 0.8]), np.array([0.2, 0.3, 0.5])
        bound = single_process_lambda_mixture_upper_confidence_trajectory(self.z, lambdas, weights, 0.05, 50)
        # At the bound the mixture sits at (or just above) 1/alpha; slightly below it, strictly under.
        for p, above in ((bound, True), (np.maximum(bound - 1e-6, 0.0), False)):
            mix = sum(w * np.exp(log_eprocess_at_p(self.z, lv, p)) for lv, w in zip(lambdas, weights))
            interior = bound < 1.0
            if above:
                self.assertTrue(np.all(mix[interior] >= 1 / 0.05 * (1 - 1e-9)))
            else:
                self.assertTrue(np.all(mix[interior] < 1 / 0.05))


class TestResolveArmsRestart(unittest.TestCase):
    def test_restart_key(self) -> None:
        from run_lambda_grid_mixture import resolve_arms
        arms = resolve_arms([
            {"name": "a", "uniform_grid_points": 4},
            {"name": "b", "uniform_grid_points": 4, "restart": "none"},
            {"name": "c", "lambda_values": [0.5], "lambda_weights": [1.0], "restart": "every_time"},
        ])
        self.assertEqual([a["restart"] for a in arms], ["every_time", "none", "every_time"])
        with self.assertRaises(ValueError):
            resolve_arms([{"name": "d", "uniform_grid_points": 4, "restart": "periodic"}])


if __name__ == "__main__":
    unittest.main()
