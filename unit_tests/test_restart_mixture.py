"""Unit tests for av_core's restarted-mixture construction: schedule/weight
builders and the mixture e-process, checked against known synthetic Z_t
sequences and manual weighted sums.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from av_core import (
    GEOMETRIC_WEIGHTS_TOTAL_MASS,
    LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS,
    build_restart_schedule,
    geometric_weights,
    log_squared_heavy_tailed_weights,
    log_eprocess_at_p,
    restart_mixture_tail_mass,
    restart_mixture_weights_for_scheme,
    restarted_mixture_log_eprocess_at_p,
    restarted_mixture_upper_confidence_trajectory,
    upper_confidence_trajectory,
)


class TestBuildRestartSchedule(unittest.TestCase):
    def test_periodic_starts_and_geometric_weights(self) -> None:
        starts = build_restart_schedule(n=250, restart_interval=100)
        weights = geometric_weights(len(starts))
        np.testing.assert_array_equal(starts, np.array([1, 101, 201]))
        np.testing.assert_allclose(weights, np.array([0.5, 0.25, 0.125]))

    def test_every_time_restart_has_one_start_per_draw(self) -> None:
        starts = build_restart_schedule(n=5, restart_interval=1)
        np.testing.assert_array_equal(starts, np.array([1, 2, 3, 4, 5]))

    def test_rejects_nonpositive_n(self) -> None:
        with self.assertRaises(ValueError):
            build_restart_schedule(n=0, restart_interval=1)

    def test_restart_interval_is_required(self) -> None:
        """No default: the interval selects the method (1 = every-time, 100 = periodic)."""
        with self.assertRaises(TypeError):
            build_restart_schedule(250)

    def test_rejects_nonpositive_interval(self) -> None:
        with self.assertRaises(ValueError):
            build_restart_schedule(n=10, restart_interval=0)


class TestLogSquaredHeavyTailedWeights(unittest.TestCase):
    def test_known_values(self) -> None:
        # w_j = 1 / (2.11 * (j+2) * ln(j+2)^2), j = 0, 1
        weights = log_squared_heavy_tailed_weights(2)
        expected_0 = 1.0 / (2.11 * 2.0 * np.log(2.0) ** 2)
        expected_1 = 1.0 / (2.11 * 3.0 * np.log(3.0) ** 2)
        np.testing.assert_allclose(weights, np.array([expected_0, expected_1]))

    def test_weights_sum_below_one_for_many_terms(self) -> None:
        weights = log_squared_heavy_tailed_weights(1000)
        self.assertLess(float(np.sum(weights)), 1.0)

    def test_rejects_nonpositive_n_starts(self) -> None:
        with self.assertRaises(ValueError):
            log_squared_heavy_tailed_weights(0)


class TestRestartedMixtureEquivalence(unittest.TestCase):
    def test_single_full_weight_process_matches_plain_eprocess(self) -> None:
        """A restart mixture with exactly one process, starting at t=1 with
        weight=1.0 (fully normalized, not a sub-probability), must reduce
        EXACTLY to the plain (unrestarted) e-process -- this is the
        degenerate case of the general construction."""
        z = np.array([1, 0, 1, 0, 0, 1], dtype=np.int8)
        p = np.full(6, 0.3)
        lambda_value = 0.4
        plain_log_e = log_eprocess_at_p(z, lambda_value, p)
        mixture_log_e = restarted_mixture_log_eprocess_at_p(
            z, lambda_value, p, starts=np.array([1]), weights=np.array([1.0]), tail_mass=0.0
        )
        np.testing.assert_allclose(mixture_log_e, plain_log_e, rtol=1e-12)

    def test_two_process_mixture_matches_manual_weighted_sum(self) -> None:
        """z = [1, 0, 0], two processes: s_0=1 (weight 0.6), s_1=2 (weight
        0.3, i.e. starts betting from t=2 onward with capital 1 at t=1).
        At t=3, p=0.5, lambda=0.5:
          process 0 factors: t=1 (z=1): 1+0.5*(0.5-1)=0.75
                              t=2 (z=0): 1+0.5*(0.5-0)=1.25
                              t=3 (z=0): 1.25
                              E_3^(0) = 0.75*1.25*1.25 = 1.171875
          process 1 (starts at t=2): capital 1 at t=1, then
                              t=2 (z=0): 1.25
                              t=3 (z=0): 1.25
                              E_3^(1) = 1*1.25*1.25 = 1.5625
          mixture E_3 = 0.6*1.171875 + 0.3*1.5625 = 0.703125 + 0.46875 = 1.171875
        """
        z = np.array([1, 0, 0], dtype=np.int8)
        p = np.full(3, 0.5)
        starts = np.array([1, 2])
        weights = np.array([0.6, 0.3])
        log_mix = restarted_mixture_log_eprocess_at_p(z, 0.5, p, starts, weights, tail_mass=0.0)
        expected_e3 = 0.6 * 1.171875 + 0.3 * 1.5625
        self.assertAlmostEqual(np.exp(log_mix[-1]), expected_e3, places=10)

    def test_rejects_mismatched_starts_and_weights_length(self) -> None:
        z = np.array([1, 0], dtype=np.int8)
        with self.assertRaises(ValueError):
            restarted_mixture_log_eprocess_at_p(
                z, 0.5, np.full(2, 0.5), starts=np.array([1, 2]),
                weights=np.array([1.0]), tail_mass=0.0
            )

    def test_rejects_weights_summing_above_one(self) -> None:
        z = np.array([1, 0], dtype=np.int8)
        with self.assertRaises(ValueError):
            restarted_mixture_log_eprocess_at_p(
                z, 0.5, np.full(2, 0.5), starts=np.array([1, 2]),
                weights=np.array([0.7, 0.7]), tail_mass=0.0
            )

    def test_tail_mass_adds_exactly_its_own_value_at_every_t(self) -> None:
        """An un-instantiated restart always has E=1 (hasn't started), so
        adding tail_mass must shift E_mix(t) by exactly tail_mass at every t,
        not just asymptotically."""
        z = np.array([1, 0, 1, 0, 0, 1], dtype=np.int8)
        p = np.full(6, 0.3)
        starts = np.array([1, 3])
        weights = np.array([0.4, 0.2])
        tail_mass = 0.1

        without_tail = np.exp(restarted_mixture_log_eprocess_at_p(z, 0.4, p, starts, weights, 0.0))
        with_tail = np.exp(
            restarted_mixture_log_eprocess_at_p(z, 0.4, p, starts, weights, tail_mass=tail_mass)
        )
        np.testing.assert_allclose(with_tail, without_tail + tail_mass, rtol=1e-10)

    def test_explicit_zero_tail_mass_has_no_extra_contribution(self) -> None:
        z = np.array([1, 0, 0], dtype=np.int8)
        p = np.full(3, 0.5)
        starts = np.array([1, 2])
        weights = np.array([0.6, 0.3])
        log_mix = restarted_mixture_log_eprocess_at_p(z, 0.5, p, starts, weights, tail_mass=0.0)
        expected_e3 = 0.6 * 1.171875 + 0.3 * 1.5625
        self.assertAlmostEqual(np.exp(log_mix[-1]), expected_e3, places=10)

    def test_rejects_negative_tail_mass(self) -> None:
        z = np.array([1, 0], dtype=np.int8)
        with self.assertRaises(ValueError):
            restarted_mixture_log_eprocess_at_p(
                z, 0.5, np.full(2, 0.5), starts=np.array([1]), weights=np.array([0.5]), tail_mass=-0.1
            )

    def test_rejects_weights_plus_tail_mass_summing_above_one(self) -> None:
        z = np.array([1, 0], dtype=np.int8)
        with self.assertRaises(ValueError):
            restarted_mixture_log_eprocess_at_p(
                z, 0.5, np.full(2, 0.5), starts=np.array([1]), weights=np.array([0.7]), tail_mass=0.4
            )


class TestRestartMixtureTailMass(unittest.TestCase):
    def test_geometric_tail_mass_matches_closed_form(self) -> None:
        # sum_{j=1}^K 2^-j = 1 - 2^-K exactly, so the tail is exactly 2^-K.
        for n_starts in [1, 5, 20]:
            weights = geometric_weights(n_starts)
            tail = restart_mixture_tail_mass(weights, GEOMETRIC_WEIGHTS_TOTAL_MASS)
            self.assertAlmostEqual(tail, 2.0 ** -n_starts, places=12)

    def test_log_squared_tail_mass_shrinks_as_n_starts_grows(self) -> None:
        small = restart_mixture_tail_mass(
            log_squared_heavy_tailed_weights(5), LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS
        )
        large = restart_mixture_tail_mass(
            log_squared_heavy_tailed_weights(500), LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS
        )
        self.assertGreater(small, large)
        self.assertGreaterEqual(large, 0.0)

    def test_returns_zero_if_weights_exceed_total_mass(self) -> None:
        # total_mass is a deliberate conservative lower bound on the true
        # infinite sum, not the exact value, so weights summing past it just
        # means there is no capital left to recover -- not an error.
        tail = restart_mixture_tail_mass(np.array([0.6, 0.6]), total_mass=1.0)
        self.assertEqual(tail, 0.0)


class TestTailMassNumericalStability(unittest.TestCase):
    """A naive tail_mass implementation (sum_exp + tail_mass*exp(-max)) would
    overflow whenever the active processes' own weighted log-E is extremely
    negative -- exp(-max_over_processes) then overflows even though the
    correct log_mix is just a very negative (but finite) number. The
    logaddexp-based implementation must stay finite and correct here."""

    def _extreme_case(self):
        # lambda close to 1, tiny p, all-ones z: log_if_one is very negative,
        # and n=2000 repetitions drive log_e_process to roughly -13800; a
        # tiny weight pushes weighted (log_weight + log_e_process) to
        # roughly -13639, so naive exp(-weighted) overflows (exp(700+) already
        # exceeds float64 range).
        n = 2000
        z = np.ones(n, dtype=np.int8)
        p = np.full(n, 0.0001)
        lambda_value = 0.999
        starts = np.array([1])
        weights = np.array([1e-6])
        return z, lambda_value, p, starts, weights

    def test_naive_exp_of_negative_max_would_overflow(self) -> None:
        # Sanity check that this scenario actually exercises the failure
        # mode the logaddexp implementation is meant to avoid.
        z, lambda_value, p, starts, weights = self._extreme_case()
        log_if_one = np.log1p(lambda_value * (p[0] - 1.0))
        weighted_final = np.log(weights[0]) + log_if_one * len(z)
        with np.errstate(over="ignore"):
            naive_exp = np.exp(-weighted_final)
        self.assertTrue(np.isinf(naive_exp))

    def test_log_mix_stays_finite_with_tail_mass(self) -> None:
        z, lambda_value, p, starts, weights = self._extreme_case()
        tail_mass = 0.01
        log_mix = restarted_mixture_log_eprocess_at_p(z, lambda_value, p, starts, weights, tail_mass)
        self.assertTrue(np.all(np.isfinite(log_mix)))

    def test_log_mix_matches_tail_mass_when_active_processes_are_negligible(self) -> None:
        # By the final t, the instantiated process's own contribution is
        # astronomically smaller than tail_mass (its weighted log-E is
        # already ~-13639 there), so the mixture is essentially just
        # tail_mass -- log_mix should land almost exactly on log(tail_mass).
        # (At small t, e.g. t=1, the process has only decayed to ~1e-9 next
        # to tail_mass=0.01, which is not yet negligible at this tolerance.)
        z, lambda_value, p, starts, weights = self._extreme_case()
        tail_mass = 0.01
        log_mix = restarted_mixture_log_eprocess_at_p(z, lambda_value, p, starts, weights, tail_mass)
        self.assertAlmostEqual(log_mix[-1], np.log(tail_mass), places=9)

    def test_upper_confidence_trajectory_stays_finite_in_extreme_case(self) -> None:
        # Binary search drives its candidate p down toward 0 over enough
        # iterations, which recreates the same overflow-prone regime as
        # _extreme_case's fixed tiny p; the numerically fragile regime is in
        # the z/lambda_value combination, not the specific weight value.
        z, lambda_value, _, starts, weights = self._extreme_case()
        trajectory = restarted_mixture_upper_confidence_trajectory(
            z, lambda_value, alpha=0.05, starts=starts, weights=weights,
            tail_mass=0.01, binary_search_iterations=50,
        )
        self.assertTrue(np.all(np.isfinite(trajectory)))


class TestRestartedMixtureUpperConfidenceTrajectory(unittest.TestCase):
    def test_single_start_mixture_is_looser_than_plain_ucb(self) -> None:
        """With explicit sub-probability capital, even a single start at t=1
        is looser than the plain e-process once its tail mass is included."""
        z = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0], dtype=np.int8)
        plain = upper_confidence_trajectory(z, 0.25, 0.05, binary_search_iterations=50)
        for weighting_scheme in ["geometric", "log_squared"]:
            starts = np.array([1])
            weights, total_mass = restart_mixture_weights_for_scheme(weighting_scheme, len(starts))
            tail_mass = restart_mixture_tail_mass(weights, total_mass)
            mixture = restarted_mixture_upper_confidence_trajectory(
                z, 0.25, 0.05, starts, weights, tail_mass,
                binary_search_iterations=50,
            )
            self.assertTrue(np.all(mixture >= plain - 1e-9))

    def test_subprobability_mixture_is_looser_than_plain_when_bound_binds(self) -> None:
        """With less than full weight on the single active process (plus its
        scheme's tail_mass), the mixture must require MORE evidence to
        certify the same p -- i.e., its UCB must be >= the plain UCB once the
        plain bound has actually moved off the vacuous 1.0 (enough zeros to
        bind)."""
        z = np.zeros(10, dtype=np.int8)
        plain = upper_confidence_trajectory(z, 0.9, 0.5, binary_search_iterations=50)
        starts = np.array([1])
        weights, total_mass = restart_mixture_weights_for_scheme("geometric", len(starts))
        tail_mass = restart_mixture_tail_mass(weights, total_mass)
        mixture = restarted_mixture_upper_confidence_trajectory(
            z, 0.9, 0.5, starts, weights, tail_mass, binary_search_iterations=50,
        )
        self.assertLess(plain[-1], 1.0)  # sanity: plain bound actually binds here
        self.assertGreaterEqual(mixture[-1], plain[-1] - 1e-9)

    def test_rejects_unknown_weighting_scheme(self) -> None:
        with self.assertRaises(ValueError):
            restart_mixture_weights_for_scheme("bogus", n_starts=1)

    def test_binary_search_iterations_is_required(self) -> None:
        """No default: the iteration count sets the bound's resolution, so a caller that
        omitted it would get a different, still entirely plausible-looking bound."""
        z = np.array([1, 0, 0, 1, 0, 0, 0], dtype=np.int8)
        starts = build_restart_schedule(len(z), restart_interval=1)
        weights, total_mass = restart_mixture_weights_for_scheme("log_squared", len(starts))
        tail_mass = restart_mixture_tail_mass(weights, total_mass)
        with self.assertRaises(TypeError):
            restarted_mixture_upper_confidence_trajectory(z, 0.25, 0.05, starts, weights, tail_mass)

    def test_requires_tail_mass(self) -> None:
        z = np.array([1, 0], dtype=np.int8)
        starts = np.array([1])
        weights = np.array([0.5])
        with self.assertRaises(TypeError):
            restarted_mixture_upper_confidence_trajectory(
                z, 0.25, 0.05, starts, weights,
            )

    def test_rejects_non_increasing_starts(self) -> None:
        z = np.array([1, 0, 1], dtype=np.int8)
        with self.assertRaises(ValueError):
            restarted_mixture_upper_confidence_trajectory(
                z, 0.25, 0.05, np.array([2, 1]), np.array([0.5, 0.25]), 0.25,
                binary_search_iterations=10,
            )

    def test_rejects_starts_out_of_range(self) -> None:
        z = np.array([1, 0, 1], dtype=np.int8)
        with self.assertRaises(ValueError):
            restarted_mixture_upper_confidence_trajectory(
                z, 0.25, 0.05, np.array([1, 5]), np.array([0.5, 0.25]), 0.25,
                binary_search_iterations=10,
            )
        with self.assertRaises(ValueError):
            restarted_mixture_upper_confidence_trajectory(
                z, 0.25, 0.05, np.array([0, 2]), np.array([0.5, 0.25]), 0.25,
                binary_search_iterations=10,
            )


if __name__ == "__main__":
    unittest.main()
