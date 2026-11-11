"""Unit tests for av_discovery_online_primary.run_trial: causal online
discovery via embedding-similarity greedy cover, hand-derived at n=3 with
orthogonal/duplicate unit-norm embeddings.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from av_discovery_online_primary import run_trial


class TestOnlineDiscoveryN3(unittest.TestCase):
    """3 unit-norm embeddings: e0=[1,0], e1=[0,1] (orthogonal to e0), e2=[1,0]
    (duplicate of e0). threshold=0.5. Draw order [0, 1, 2]:

    t=0: draw e0 -- empty dictionary -> always new. z=1. New dict entry e0
         covers itself and e2 (sim(e0,e2)=1>=0.5), not e1 (sim=0). n_covered=2.
         gt_hat=1/1=1.0. q = 1 - 2/3 = 1/3.
    t=1: draw e1 -- sim to e0 is 0 < 0.5 -> new. z=1. New entry e1 covers only
         e1 itself (sim(e1,e0)=0, sim(e1,e2)=0). n_covered=3 (e1 newly added
         to the covered set). gt_hat=2/2=1.0. q = 1 - 3/3 = 0.
    t=2: draw e2 -- sim to e0=1 >= 0.5 -> matches e0, not new. z=0. e0's hit
         count goes 1->2, so it drops out of the singleton count:
         gt_hat = 1/3 (only e1 remains a singleton). q stays 0 (no new
         discovery).
    """

    def setUp(self) -> None:
        self.embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        self.threshold = 0.5

    def test_matches_hand_derived_trajectory(self) -> None:
        draw_indices = np.array([0, 1, 2])
        z, gt_hat, q = run_trial(draw_indices, self.embeddings, self.threshold, n_population=3)
        np.testing.assert_array_equal(z, np.array([1, 1, 0], dtype=np.int8))
        np.testing.assert_allclose(gt_hat, np.array([1.0, 1.0, 1.0 / 3.0]))
        np.testing.assert_allclose(q, np.array([1.0 / 3.0, 0.0, 0.0]))

    def test_repeated_draws_of_same_unit_never_discover_twice(self) -> None:
        draw_indices = np.array([0, 0, 0])
        z, gt_hat, q = run_trial(draw_indices, self.embeddings, self.threshold, n_population=3)
        np.testing.assert_array_equal(z, np.array([1, 0, 0], dtype=np.int8))
        # After the first draw, e0 and e2 (its duplicate) are covered, but e1
        # never is -- true missing mass stays at 1/3 forever under this draw
        # sequence.
        np.testing.assert_allclose(q, np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))

    def test_all_below_threshold_discovers_every_draw(self) -> None:
        # Raise the threshold above every off-diagonal similarity (0 and 1
        # for duplicates only) so e0 and e1 are each treated as new, but the
        # duplicate e2 still matches e0 exactly (sim=1 >= any threshold<=1).
        draw_indices = np.array([0, 1, 2])
        z, _, _ = run_trial(draw_indices, self.embeddings, similarity_threshold=0.99, n_population=3)
        np.testing.assert_array_equal(z, np.array([1, 1, 0], dtype=np.int8))


class TestOnlineDiscoveryWithSuccesses(unittest.TestCase):
    """Same 3 failure embeddings as above plus 2 success sessions (indices 3
    and 4), n_population=5, threshold=0.5. Draw order [3, 0, 4, 1]:

    t=0: success -- never novel, nothing covered. z=0. q = 3/5. gt_hat = 0/1.
    t=1: e0 -- new. z=1. covers e0, e2. q = 1/5. gt_hat = 1/2.
    t=2: success. z=0. q stays 1/5. gt_hat = 1/3.
    t=3: e1 -- new. z=1. all failures covered. q = 0. gt_hat = 2/4.
    """

    def test_success_draws_are_never_novel_and_count_in_the_denominator(self) -> None:
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        z, gt_hat, q = run_trial(np.array([3, 0, 4, 1]), embeddings, 0.5, n_population=5)
        np.testing.assert_array_equal(z, np.array([0, 1, 0, 1], dtype=np.int8))
        np.testing.assert_allclose(q, np.array([3 / 5, 1 / 5, 1 / 5, 0.0]))
        np.testing.assert_allclose(gt_hat, np.array([0.0, 1 / 2, 1 / 3, 2 / 4]))

    def test_population_smaller_than_failures_raises(self) -> None:
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        with self.assertRaises(ValueError):
            run_trial(np.array([0]), embeddings, 0.5, n_population=2)

    def test_index_beyond_population_raises(self) -> None:
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        with self.assertRaises(ValueError):
            run_trial(np.array([5]), embeddings, 0.5, n_population=5)


if __name__ == "__main__":
    unittest.main()
