"""Unit tests for av_discovery_fixed_primary.py: single-mode-per-session
discovery (novelty_indicator), the Good-Turing estimate, and true missing
mass -- hand-verified at small n, plus regression tests for indexing and
future leakage.

Convention (0-indexed):
    novelty_indicator[t] = whether draw t is novel
    good_turing_estimate[t] = #singleton modes among draws 0..t, divided by (t+1)
    true_missing_mass[t] = P(draw t+1 is novel | draws 0..t)
and any e-process/UCB value at index t must be a function of exactly
z[0], ..., z[t] -- never later draws.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from av_core import upper_confidence_trajectory
from av_discovery_fixed_primary import (
    SUCCESS_MODE_ID,
    build_session_population,
    compute_discovery_statistics,
    compute_true_missing_mass,
    run_bootstrap_trial,
)


class TestDiscoveryFixedPrimaryN3(unittest.TestCase):
    """Population of 3 sessions: session_0 -> mode A, session_1 -> mode B,
    session_2 -> mode A (mode A frequency 2/3, mode B frequency 1/3).
    Draw sequence [0, 1, 0] (with replacement): A, B, A-again.

    Hand-derived expectation:
      t=0: draw A -> first time seen -> z=1, gt_hat=1/1=1.0
           seen={A}; remaining_mass = 1 - 2/3 = 1/3 -> q=1/3
      t=1: draw B -> first time seen -> z=1, gt_hat=2/2=1.0
           seen={A,B}; remaining_mass = 1 - 2/3 - 1/3 = 0 -> q=0
      t=2: draw A again -> already seen -> z=0
           n_singletons drops from 2 to 1 (A no longer a singleton) -> gt_hat=1/3
           seen unchanged -> q=0
    """

    def setUp(self) -> None:
        self.session_to_failure_mode = {"s0": "A", "s1": "B", "s2": "A"}
        self.failure_mode_by_session_index, self.failure_mode_probability, self.n_sessions = (
            build_session_population(self.session_to_failure_mode, list(self.session_to_failure_mode))
        )
        # build_session_population sorts session_ids: s0, s1, s2 -> indices 0, 1, 2
        self.assertEqual(self.n_sessions, 3)
        self.assertEqual(list(self.failure_mode_by_session_index), ["A", "B", "A"])
        self.assertAlmostEqual(self.failure_mode_probability["A"], 2.0 / 3.0)
        self.assertAlmostEqual(self.failure_mode_probability["B"], 1.0 / 3.0)

    def test_discovery_indicator_and_gt_hat(self) -> None:
        indices = np.array([0, 1, 0])
        novelty, gt_hat = compute_discovery_statistics(indices, self.failure_mode_by_session_index)
        np.testing.assert_array_equal(novelty, np.array([1, 1, 0], dtype=np.int8))
        np.testing.assert_allclose(gt_hat, np.array([1.0, 1.0, 1.0 / 3.0]))

    def test_true_missing_mass(self) -> None:
        indices = np.array([0, 1, 0])
        q = compute_true_missing_mass(indices, self.failure_mode_by_session_index, self.failure_mode_probability)
        # atol needed: remaining_mass is a running float subtraction
        # (1 - 2/3 - 1/3), which lands at ~5.55e-17 rather than exact 0.
        np.testing.assert_allclose(q, np.array([1.0 / 3.0, 0.0, 0.0]), atol=1e-9)

    def test_repeated_draw_never_goes_negative(self) -> None:
        # Draw the same session 5 times: only the first counts as new.
        indices = np.array([0, 0, 0, 0, 0])
        novelty, gt_hat = compute_discovery_statistics(indices, self.failure_mode_by_session_index)
        np.testing.assert_array_equal(novelty, np.array([1, 0, 0, 0, 0], dtype=np.int8))
        q = compute_true_missing_mass(indices, self.failure_mode_by_session_index, self.failure_mode_probability)
        # After discovering A once, remaining mass is fixed at 1 - 2/3 = 1/3 forever.
        np.testing.assert_allclose(q, np.full(5, 1.0 / 3.0))


class TestBuildSessionPopulationErrors(unittest.TestCase):
    def test_empty_all_session_ids_raises(self) -> None:
        with self.assertRaises(Exception):
            build_session_population({}, [])

    def test_taxonomy_session_outside_all_session_ids_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_session_population({"s0": "A"}, ["s1"])

    def test_real_mode_named_like_sentinel_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_session_population({"s0": SUCCESS_MODE_ID}, ["s0", "s1"])


class TestToyPopulationIndexing(unittest.TestCase):
    """Population of 4 sessions with modes [A, A, B, C]:
    P(A)=0.5, P(B)=0.25, P(C)=0.25. Bootstrap draws [0,1,2,3] (with
    replacement) observe modes A, A, B, C in that order -- the exact
    worked example this indexing convention is defined against."""

    def setUp(self) -> None:
        session_to_failure_mode = {"s0": "A", "s1": "A", "s2": "B", "s3": "C"}
        self.modes, self.probability, self.n_sessions = build_session_population(session_to_failure_mode, list(session_to_failure_mode))
        self.assertEqual(list(self.modes), ["A", "A", "B", "C"])
        self.assertEqual(self.probability, {"A": 0.5, "B": 0.25, "C": 0.25})

    def test_toy_example_matches_worked_values(self) -> None:
        indices = np.array([0, 1, 2, 3])
        novelty, gt_hat = compute_discovery_statistics(indices, self.modes)
        np.testing.assert_array_equal(novelty, np.array([1, 0, 1, 1], dtype=np.int8))
        np.testing.assert_allclose(gt_hat, np.array([1.0, 0.0, 1.0 / 3.0, 1.0 / 2.0]))

        q = compute_true_missing_mass(indices, self.modes, self.probability)
        np.testing.assert_allclose(q, np.array([1.0 / 2.0, 1.0 / 2.0, 1.0 / 4.0, 0.0]), atol=1e-9)

    def test_run_bootstrap_trial_matches_the_same_worked_values(self) -> None:
        # run_bootstrap_trial is just the three individual functions wired
        # together; confirm it reproduces the identical worked example.
        indices = np.array([0, 1, 2, 3])
        novelty, gt_hat, q = run_bootstrap_trial(indices, self.modes, self.probability)
        np.testing.assert_array_equal(novelty, np.array([1, 0, 1, 1], dtype=np.int8))
        np.testing.assert_allclose(gt_hat, np.array([1.0, 0.0, 1.0 / 3.0, 1.0 / 2.0]))
        np.testing.assert_allclose(q, np.array([1.0 / 2.0, 1.0 / 2.0, 1.0 / 4.0, 0.0]), atol=1e-9)

    def test_t0_boundary(self) -> None:
        """After the very first draw: gt_hat[0] is always 1.0 (a lone draw
        is trivially its own singleton), and true_missing_mass[0] already
        reflects that first draw having been incorporated (not the prior
        1.0)."""
        indices = np.array([2])  # single draw of session index 2 -> mode B
        novelty, gt_hat = compute_discovery_statistics(indices, self.modes)
        self.assertEqual(novelty[0], 1)
        self.assertEqual(gt_hat[0], 1.0)
        q = compute_true_missing_mass(indices, self.modes, self.probability)
        self.assertAlmostEqual(q[0], 1.0 - self.probability["B"])  # 0.75, not 1.0

    def test_oracle_missing_mass_matches_direct_population_count(self) -> None:
        """Cross-check compute_true_missing_mass against an independent,
        from-scratch computation over the actual population array (not the
        probability dict): after each prefix, the fraction of the 4
        population sessions whose mode has not yet been observed."""
        indices = np.array([0, 1, 2, 3])
        q = compute_true_missing_mass(indices, self.modes, self.probability)
        observed_modes: set[str] = set()
        for t, session_index in enumerate(indices):
            observed_modes.add(self.modes[session_index])
            expected = float(np.mean([mode not in observed_modes for mode in self.modes]))
            self.assertAlmostEqual(q[t], expected, places=12)


class TestUnsupportedSessionLabeling(unittest.TestCase):
    """build_session_population must label every session absent from the
    taxonomy as SUCCESS_MODE_ID -- and only those sessions -- regardless of
    how many there are, where they fall in sorted order, or whether any
    real modes exist at all."""

    def test_single_unsupported_session_is_labeled_success(self) -> None:
        session_to_failure_mode = {"s0": "A", "s1": "A", "s3": "B"}
        modes, probability, n_sessions = build_session_population(
            session_to_failure_mode, ["s0", "s1", "s2", "s3"]
        )
        np.testing.assert_array_equal(modes, np.array(["A", "A", SUCCESS_MODE_ID, "B"]))
        self.assertNotIn(SUCCESS_MODE_ID, probability)
        self.assertEqual(n_sessions, 4)

    def test_multiple_unsupported_sessions_all_labeled(self) -> None:
        # s0 and s3 unsupported; s1, s2, s4 have real modes.
        session_to_failure_mode = {"s1": "A", "s2": "B", "s4": "A"}
        modes, probability, n_sessions = build_session_population(
            session_to_failure_mode, ["s0", "s1", "s2", "s3", "s4"]
        )
        np.testing.assert_array_equal(
            modes, np.array([SUCCESS_MODE_ID, "A", "B", SUCCESS_MODE_ID, "A"])
        )
        self.assertEqual(set(probability), {"A", "B"})
        self.assertEqual(n_sessions, 5)

    def test_no_unsupported_sessions_means_no_success_label_at_all(self) -> None:
        # Every session resolved -- SUCCESS_MODE_ID should not appear anywhere.
        session_to_failure_mode = {"s0": "A", "s1": "B"}
        modes, probability, _ = build_session_population(
            session_to_failure_mode, ["s0", "s1"]
        )
        self.assertNotIn(SUCCESS_MODE_ID, modes)
        self.assertNotIn(SUCCESS_MODE_ID, probability)

    def test_all_sessions_unsupported(self) -> None:
        # A population with zero real failure modes at all: every session
        # is a success, and failure_mode_probability is empty.
        modes, probability, n_sessions = build_session_population({}, ["s0", "s1", "s2"])
        np.testing.assert_array_equal(modes, np.array([SUCCESS_MODE_ID] * 3))
        self.assertEqual(probability, {})
        self.assertEqual(n_sessions, 3)

    def test_unsupported_session_labeling_is_independent_of_sort_order(self) -> None:
        # Regardless of dict/list insertion order, labeling follows sorted
        # session_id order, matching the population's own index convention.
        session_to_failure_mode = {"z9": "A", "a1": "B"}
        modes, _, _ = build_session_population(
            session_to_failure_mode, ["z9", "m5", "a1"]
        )
        # sorted(["z9", "m5", "a1"]) == ["a1", "m5", "z9"]
        np.testing.assert_array_equal(modes, np.array(["B", SUCCESS_MODE_ID, "A"]))


class TestSuccessSession(unittest.TestCase):
    """A session absent from the taxonomy is a success, not a failure:
    always non-novel, excluded from singleton/missing-mass bookkeeping, but
    still counted in the population denominator.

    Population: s0, s1 -> A; s3 -> B; s2 has no resolved mode (success).
    n_sessions=4, so P(A)=2/4=0.5, P(B)=1/4=0.25 (success excluded from
    failure_mode_probability; the two real modes' probabilities sum to
    0.75 = 1 - P(success), not 1).
    """

    def setUp(self) -> None:
        session_to_failure_mode = {"s0": "A", "s1": "A", "s3": "B"}
        all_session_ids = ["s0", "s1", "s2", "s3"]
        self.modes, self.probability, self.n_sessions = build_session_population(
            session_to_failure_mode, all_session_ids
        )
        self.assertEqual(list(self.modes), ["A", "A", SUCCESS_MODE_ID, "B"])
        self.assertEqual(self.n_sessions, 4)
        self.assertEqual(self.probability, {"A": 0.5, "B": 0.25})

    def test_success_draw_is_never_novel(self) -> None:
        # Draw s2 (success) first, before any real mode has been observed --
        # even so, novelty must be 0.
        indices = np.array([2, 0, 3])
        novelty, gt_hat = compute_discovery_statistics(indices, self.modes)
        np.testing.assert_array_equal(novelty, np.array([0, 1, 1], dtype=np.int8))

    def test_success_draw_does_not_affect_singleton_count(self) -> None:
        # A -> success -> A again -> B: the success draw must not perturb
        # the singleton count (gt_hat's numerator). gt_hat itself still
        # shifts between the two sequences because its denominator (t+1)
        # counts every draw, successes included -- so compare n_singletons
        # (= gt_hat[t] * (t+1)), not gt_hat directly.
        indices_with_success = np.array([0, 2, 1, 3])
        indices_without_success = np.array([0, 1, 3])
        _, gt_hat_with = compute_discovery_statistics(indices_with_success, self.modes)
        _, gt_hat_without = compute_discovery_statistics(indices_without_success, self.modes)
        n_singletons_with = gt_hat_with * np.arange(1, len(gt_hat_with) + 1)
        n_singletons_without = gt_hat_without * np.arange(1, len(gt_hat_without) + 1)
        # Drop the success draw's slot (index 1) from the "with" trajectory;
        # the remaining singleton counts must match the "without" trajectory
        # exactly.
        n_singletons_with_success_removed = np.delete(n_singletons_with, 1)
        np.testing.assert_allclose(n_singletons_with_success_removed, n_singletons_without)

    def test_success_draw_does_not_reduce_missing_mass(self) -> None:
        indices = np.array([2, 2, 2])  # only ever draw the success session
        q = compute_true_missing_mass(indices, self.modes, self.probability)
        # Nothing real has ever been discovered, so all real mass (0.75) is
        # still missing throughout -- never mistaken for the literal 1.0 a
        # naive (non-success-aware) implementation would report.
        np.testing.assert_allclose(q, np.full(3, 0.75))

    def test_missing_mass_reaches_exactly_zero_once_all_real_modes_seen(self) -> None:
        indices = np.array([2, 0, 3])  # success, then A, then B
        q = compute_true_missing_mass(indices, self.modes, self.probability)
        self.assertAlmostEqual(q[-1], 0.0, places=9)

    def test_oracle_missing_mass_matches_direct_population_count(self) -> None:
        """Cross-check against a from-scratch computation over the actual
        population array: success sessions must never count as
        'undiscovered content' in either direction."""
        indices = np.array([2, 0, 1, 3])
        q = compute_true_missing_mass(indices, self.modes, self.probability)
        observed_modes: set[str] = set()
        for t, session_index in enumerate(indices):
            mode_id = self.modes[session_index]
            if mode_id != SUCCESS_MODE_ID:
                observed_modes.add(mode_id)
            expected = float(np.mean([
                mode not in observed_modes and mode != SUCCESS_MODE_ID
                for mode in self.modes
            ]))
            self.assertAlmostEqual(q[t], expected, places=12)


class TestPrefixInvariance(unittest.TestCase):
    """Two draw sequences sharing a common prefix through index t but
    diverging afterward must produce identical novelty_indicator, gt_hat,
    true_missing_mass, and UCB values through t -- nothing at or before t
    may depend on what comes after."""

    def setUp(self) -> None:
        session_to_failure_mode = {"s0": "A", "s1": "A", "s2": "B", "s3": "C"}
        self.modes, self.probability, _ = build_session_population(session_to_failure_mode, list(session_to_failure_mode))

    def test_shared_prefix_gives_identical_stats_and_ucb_through_t(self) -> None:
        shared_prefix_end = 2  # indices 0, 1, 2 shared; index 3 differs
        indices_1 = np.array([0, 1, 2, 3])  # ..., C
        indices_2 = np.array([0, 1, 2, 0])  # ..., A again

        novelty_1, gt_hat_1 = compute_discovery_statistics(indices_1, self.modes)
        novelty_2, gt_hat_2 = compute_discovery_statistics(indices_2, self.modes)
        np.testing.assert_array_equal(
            novelty_1[: shared_prefix_end + 1], novelty_2[: shared_prefix_end + 1]
        )
        np.testing.assert_allclose(gt_hat_1[: shared_prefix_end + 1], gt_hat_2[: shared_prefix_end + 1])

        q_1 = compute_true_missing_mass(indices_1, self.modes, self.probability)
        q_2 = compute_true_missing_mass(indices_2, self.modes, self.probability)
        np.testing.assert_allclose(q_1[: shared_prefix_end + 1], q_2[: shared_prefix_end + 1])

        ucb_1 = upper_confidence_trajectory(novelty_1, 0.3, 0.1, binary_search_iterations=30)
        ucb_2 = upper_confidence_trajectory(novelty_2, 0.3, 0.1, binary_search_iterations=30)
        np.testing.assert_allclose(ucb_1[: shared_prefix_end + 1], ucb_2[: shared_prefix_end + 1])

        # Sanity: the sequences do genuinely diverge after the shared
        # prefix, so this isn't a vacuous check.
        self.assertNotEqual(novelty_1[shared_prefix_end + 1], novelty_2[shared_prefix_end + 1])


class TestEProcessUsesOnlyPastAndCurrentDraws(unittest.TestCase):
    """If two z arrays differ only at index t, the UCB trajectory must match
    everywhere before t and may differ starting at t -- proving index t's
    value depends on z[t] but never on z[t+1:]."""

    def test_ucb_matches_before_t_and_diverges_at_t(self) -> None:
        t = 6
        z1 = np.zeros(10, dtype=np.int8)
        z2 = np.zeros(10, dtype=np.int8)
        z2[t] = 1  # the only difference between the two sequences

        ucb1 = upper_confidence_trajectory(z1, 0.9, 0.5, binary_search_iterations=50)
        ucb2 = upper_confidence_trajectory(z2, 0.9, 0.5, binary_search_iterations=50)

        np.testing.assert_allclose(ucb1[:t], ucb2[:t])
        self.assertFalse(np.isclose(ucb1[t], ucb2[t]))


if __name__ == "__main__":
    unittest.main()
