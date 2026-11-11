"""Unit tests for av_discovery_fixed_set_valued.py: mode-SET-per-session
discovery (Z_t) and population-level q_t, hand-verified at n=3, including an
unlabeled unit and a mode shared across two units.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from av_discovery_fixed_set_valued import build_population, run_trial


class TestDiscoverySetValuedN3(unittest.TestCase):
    """Population of 3 sessions: s0 -> {A, B}, s1 -> {B}, s2 -> {} (unlabeled).
    Mode B is shared by s0 and s1. Draw sequence [2, 0, 1] (session 2 first).

    Hand-derived expectation:
      unseen_label_count_by_unit starts at [2, 1, 0] (s0 has 2 labels, s1 has
      1, s2 has 0).
      t=0: draw s2 (empty set) -> new_modes=[] -> z=0. Counts unchanged
           [2, 1, 0] -> q = mean([2>0, 1>0, 0>0]) = mean([T,T,F]) = 2/3
      t=1: draw s0 ({A, B}) -> both A and B are new -> z=1 (bool(non-empty)).
           Discovering A decrements s0's count (only unit with A): [1, 1, 0]
           Discovering B decrements s0 AND s1's counts (both have B): [0, 0, 0]
           -> q = mean([F, F, F]) = 0
      t=2: draw s1 ({B}) -> B already seen -> new_modes=[] -> z=0.
           Counts unchanged [0, 0, 0] -> q = 0
    """

    def setUp(self) -> None:
        self.session_mode_sets = {"s0": frozenset({"A", "B"}), "s1": frozenset({"B"})}
        all_session_ids = ["s0", "s1", "s2"]
        self.labels_per_unit, self.mode_to_unit_indices = build_population(
            self.session_mode_sets, all_session_ids
        )
        # s2 is absent from session_mode_sets entirely (unlabeled), but still
        # part of the declared population via all_session_ids -- build_population
        # gives it an empty frozenset() rather than excluding it.
        self.n_population = len(self.labels_per_unit)
        self.assertEqual(self.n_population, 3)
        self.assertEqual(self.labels_per_unit[2], frozenset())

    def test_discovery_indicator_and_true_missing_mass(self) -> None:
        draw_indices = np.array([2, 0, 1])
        z, q = run_trial(draw_indices, self.labels_per_unit, self.mode_to_unit_indices)
        np.testing.assert_array_equal(z, np.array([0, 1, 0], dtype=np.int8))
        np.testing.assert_allclose(q, np.array([2.0 / 3.0, 0.0, 0.0]))

    def test_unlabeled_unit_never_triggers_discovery(self) -> None:
        # Drawing only the unlabeled session repeatedly must never fire z=1.
        draw_indices = np.array([2, 2, 2])
        z, q = run_trial(draw_indices, self.labels_per_unit, self.mode_to_unit_indices)
        np.testing.assert_array_equal(z, np.zeros(3, dtype=np.int8))
        # No mode ever discovered: all units with a nonempty label set still count.
        np.testing.assert_allclose(q, np.full(3, 2.0 / 3.0))

    def test_shared_mode_discovered_via_one_session_clears_both(self) -> None:
        # Draw s1 (which only carries B) first -- B is new via s1, and since
        # mode_to_unit_indices[B] = [s0, s1], s0's count must drop too even
        # though s0 has NOT been drawn yet (population-level bookkeeping).
        draw_indices = np.array([1])
        z, q = run_trial(draw_indices, self.labels_per_unit, self.mode_to_unit_indices)
        np.testing.assert_array_equal(z, np.array([1], dtype=np.int8))
        # After discovering B: s0 still has undiscovered A (count 1 -> >0),
        # s1 now fully discovered (count 0), s2 has no labels (count 0).
        np.testing.assert_allclose(q, np.array([1.0 / 3.0]))


class TestBuildPopulationErrors(unittest.TestCase):
    def test_empty_session_to_failure_modes_gives_all_unlabeled_population(self) -> None:
        # No error: every session in all_session_ids just gets an empty
        # mode set, same as an unlabeled session with a nonempty taxonomy.
        labels_per_unit, mode_to_unit_indices = build_population({}, ["s0", "s1"])
        self.assertEqual(labels_per_unit, [frozenset(), frozenset()])
        self.assertEqual(mode_to_unit_indices, {})

    def test_empty_all_session_ids_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_population({"s0": frozenset({"A"})}, all_session_ids=[])

    def test_negative_count_is_impossible_by_construction_but_checked(self) -> None:
        # Drawing far more than the number of distinct modes must never drive
        # any unit's unseen count negative (internal invariant, guarded by a
        # RuntimeError in run_trial if it ever were violated).
        labels_per_unit, mode_to_unit_indices = build_population(
            {"s0": frozenset({"A"}), "s1": frozenset({"A"})}, ["s0", "s1"]
        )
        draw_indices = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        z, q = run_trial(draw_indices, labels_per_unit, mode_to_unit_indices)
        self.assertTrue(np.all(q >= 0.0))


if __name__ == "__main__":
    unittest.main()
