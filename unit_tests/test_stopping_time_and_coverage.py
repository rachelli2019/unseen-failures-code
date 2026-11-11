"""Unit tests for av_evaluation_metrics: trajectory percentile bands,
cumulative/simultaneous coverage, gap-to-truth, stopping times, and the
combined method_summary, all on small hand-derived matrices.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from av_evaluation_metrics import (
    cumulative_coverage_rate,
    gap_to_truth,
    method_summary,
    simultaneous_violation_rate,
    stopping_time_summary,
    trajectory_summary,
)


class TestTrajectorySummary(unittest.TestCase):
    def test_matches_manual_percentiles_on_small_matrix(self) -> None:
        matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mean, lower, upper = trajectory_summary(matrix, lower_quantile=0.25, upper_quantile=0.75)
        np.testing.assert_allclose(mean, np.array([3.0, 4.0]))
        expected_lower = np.quantile(matrix, 0.25, axis=0)
        expected_upper = np.quantile(matrix, 0.75, axis=0)
        np.testing.assert_allclose(lower, expected_lower)
        np.testing.assert_allclose(upper, expected_upper)

    def test_default_quantiles_are_5_and_95(self) -> None:
        matrix = np.arange(200.0).reshape(20, 10)
        mean, lower, upper = trajectory_summary(matrix, lower_quantile=0.05, upper_quantile=0.95)
        np.testing.assert_allclose(lower, np.quantile(matrix, 0.05, axis=0))
        np.testing.assert_allclose(upper, np.quantile(matrix, 0.95, axis=0))

    def test_rejects_empty_matrix(self) -> None:
        with self.assertRaises(ValueError):
            trajectory_summary(np.zeros((0, 5)), lower_quantile=0.05, upper_quantile=0.95)

    def test_rejects_non_finite_matrix(self) -> None:
        matrix = np.array([[1.0, float("nan")], [3.0, 4.0]])
        with self.assertRaises(ValueError):
            trajectory_summary(matrix, lower_quantile=0.05, upper_quantile=0.95)

    def test_rejects_inverted_quantile_range(self) -> None:
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(ValueError):
            trajectory_summary(matrix, lower_quantile=0.9, upper_quantile=0.1)


class TestCumulativeCoverageRate(unittest.TestCase):
    """estimate = [[0.5, 0.05], [0.5, 0.5]], q = [[0.4, 0.1], [0.4, 0.4]].
    Trial 0: violated at t=2 (0.1 > 0.05) -> ever_violated = [F, T]
    Trial 1: never violated -> ever_violated = [F, F]
    coverage_t1 = 1 - mean([F, F]) = 1.0
    coverage_t2 = 1 - mean([T, F]) = 0.5
    """

    def test_matches_manual_computation(self) -> None:
        estimate = np.array([[0.5, 0.05], [0.5, 0.5]])
        q = np.array([[0.4, 0.1], [0.4, 0.4]])
        coverage = cumulative_coverage_rate(estimate, q)
        np.testing.assert_allclose(coverage, np.array([1.0, 0.5]))

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            cumulative_coverage_rate(np.zeros((2, 3)), np.zeros((2, 4)))


class TestSimultaneousViolationRate(unittest.TestCase):
    def test_matches_manual_computation(self) -> None:
        # Trial 0 violates at t=2; trial 1 never violates -> rate = 0.5.
        estimate = np.array([[0.5, 0.05], [0.5, 0.5]])
        q = np.array([[0.4, 0.1], [0.4, 0.4]])
        self.assertAlmostEqual(simultaneous_violation_rate(estimate, q), 0.5)

    def test_no_violations_gives_zero(self) -> None:
        estimate = np.full((3, 4), 0.9)
        q = np.full((3, 4), 0.1)
        self.assertEqual(simultaneous_violation_rate(estimate, q), 0.0)


class TestGapToTruth(unittest.TestCase):
    def test_matches_manual_subtraction(self) -> None:
        estimate = np.array([[0.5, 0.3], [0.2, 0.9]])
        q = np.array([[0.4, 0.35], [0.1, 0.1]])
        np.testing.assert_allclose(gap_to_truth(estimate, q), estimate - q)

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            gap_to_truth(np.zeros((2, 3)), np.zeros((3, 2)))


class TestStoppingTimeSummary(unittest.TestCase):
    def test_matches_manual_two_trial_computation(self) -> None:
        # Trial 0 crosses 0.1 at t=3 (1-indexed); trial 1 crosses at t=2.
        bound_matrix = np.array([[0.5, 0.3, 0.05], [0.5, 0.05, 0.05]])
        result = stopping_time_summary(bound_matrix, [0.1])
        summary = result["0.1"]
        self.assertEqual(summary["n_stopped"], 2)
        self.assertEqual(summary["n_trials"], 2)
        self.assertEqual(summary["n_sessions"], 3)
        self.assertAlmostEqual(summary["fraction_stopped_by_n"], 1.0)
        np.testing.assert_allclose(
            [summary["tau_q25"], summary["median_tau"], summary["tau_q75"]],
            np.quantile([3, 2], [0.25, 0.5, 0.75]),
        )

    def test_trial_that_never_crosses_is_excluded_from_quantiles(self) -> None:
        bound_matrix = np.array([[0.5, 0.3, 0.05], [0.5, 0.4, 0.3]])
        result = stopping_time_summary(bound_matrix, [0.1])["0.1"]
        self.assertEqual(result["n_stopped"], 1)
        self.assertAlmostEqual(result["fraction_stopped_by_n"], 0.5)
        self.assertEqual(result["median_tau"], 3.0)

    def test_no_trials_stop_gives_none_quantiles(self) -> None:
        bound_matrix = np.full((2, 3), 0.5)
        result = stopping_time_summary(bound_matrix, [0.1])["0.1"]
        self.assertEqual(result["n_stopped"], 0)
        self.assertIsNone(result["median_tau"])
        self.assertIsNone(result["tau_q25"])
        self.assertIsNone(result["tau_q75"])

    def test_rejects_duplicate_epsilons(self) -> None:
        with self.assertRaises(ValueError):
            stopping_time_summary(np.full((2, 3), 0.5), [0.1, 0.1])

    def test_rejects_epsilon_outside_open_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            stopping_time_summary(np.full((2, 3), 0.5), [1.0])


class TestMethodSummary(unittest.TestCase):
    def test_combines_violation_rate_and_stopping_times(self) -> None:
        bound_matrix = np.array([[0.5, 0.3, 0.05], [0.5, 0.05, 0.05]])
        q_matrix = np.array([[0.4, 0.35, 0.02], [0.4, 0.2, 0.01]])
        result = method_summary("my_bound", bound_matrix, q_matrix, [0.1])
        self.assertEqual(result["name"], "my_bound")
        self.assertAlmostEqual(
            result["simultaneous_coverage_rate"], 1.0 - result["simultaneous_violation_rate"]
        )
        self.assertIn("0.1", result["stopping_times"])
        self.assertEqual(result["stopping_times"]["0.1"]["n_stopped"], 2)


if __name__ == "__main__":
    unittest.main()
