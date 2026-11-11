"""Analytic tests for the uniform lambda-grid mixture (av_lambda_restart_mixture).

Every test here compares the implementation against an answer derived by hand -- a
closed-form polynomial, an exact expectation, an exact quadrature -- rather than against
another implementation sharing the same code path. Here L = lambda_max = 1 (the
implementation restricts lambda to (0, 1)).

Two objects are tested:

  pure mixture    M_t(u) = sum_k pi_k prod_{i<=t} (1 + lambda_k (u - Z_i))
                  (one start at s = 1 with weight 1, no restart layer)
  production      M_t(u) = sum_k pi_k sum_{s<=t} w_s E_{s:t}(u; lambda_k) + (W - sum_{s<=t} w_s)
                  (every-time restarts, log-squared weights w_s, total mass W = 0.999;
                   restarts that have not begun contribute exactly their own weight)

Analytic facts used for the midpoint grid lambda_k = (k - 1/2)/K, weights 1/K:
  * it integrates linear functions of lambda exactly, and
  * for a quadratic a + b lambda + c lambda^2 its average is a + b/2 + c (1/3 - 1/(12 K^2)).
"""
import itertools
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from scipy.special import logsumexp

from av_core import (
    LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS,
    compile_restart_mixture,
    log_squared_heavy_tailed_weights,
)
from av_closed_form_lambda_mixture import kl_bernoulli
from av_lambda_restart_mixture import (
    _everytime_lambda_mixture_log_eprocess,
    _evaluate_lambda_restart_mixture_log_eprocess,
    compile_everytime_restart_mixture,
    everytime_lambda_mixture_upper_confidence_trajectory,
    uniform_lambda_grid,
)

TOTAL_MASS = LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS
ITERS = 40  # bisection iterations for UCB tests; bracket width 2^-40


# ---------------------------------------------------------------------------
# Thin wrappers around the production evaluators (no math re-implemented here)
# ---------------------------------------------------------------------------

def pure_log_mixture(z, lambda_values, pi_weights, u):
    """Production evaluator with a single start at s = 1, weight 1, no tail: the pure
    lambda mixture, with no restart layer."""
    z = np.asarray(z, dtype=np.int64)
    lambda_values = np.asarray(lambda_values, dtype=float)
    pi_weights = np.asarray(pi_weights, dtype=float)
    p = np.broadcast_to(np.asarray(u, dtype=float), (len(z),)).copy()
    mixture = compile_restart_mixture(z, np.array([1]), np.array([1.0]), 0.0)
    return _evaluate_lambda_restart_mixture_log_eprocess(
        mixture, lambda_values, np.log(pi_weights), p,
        np.empty((len(lambda_values), len(z))), np.empty_like(mixture.zeros_in_range),
    )


def production_log_mixture(z, lambda_values, pi_weights, u, block_size=256):
    """The every-time fast-path evaluator the lambda-grid runs actually invert."""
    z = np.asarray(z, dtype=np.int64)
    lambda_values = np.asarray(lambda_values, dtype=float)
    pi_weights = np.asarray(pi_weights, dtype=float)
    n = len(z)
    p = np.broadcast_to(np.asarray(u, dtype=float), (n,)).copy()
    mixture, not_started = compile_everytime_restart_mixture(z, "log_squared")
    return _everytime_lambda_mixture_log_eprocess(
        mixture, not_started, lambda_values, np.log(pi_weights), p, block_size,
        np.empty((len(lambda_values), n)), np.empty((n, min(block_size, n))),
    )


def production_ucb(z, lambda_values, pi_weights, alpha, iters=ITERS):
    return everytime_lambda_mixture_upper_confidence_trajectory(
        np.asarray(z, dtype=np.int64), np.asarray(lambda_values, dtype=float),
        np.asarray(pi_weights, dtype=float), alpha, iters, "log_squared",
    )


# ---------------------------------------------------------------------------
# Independent references (written from the math, not from the implementation)
# ---------------------------------------------------------------------------

def restart_weights(n):
    return log_squared_heavy_tailed_weights(n)


def continuous_uniform_restart_mixture(z, u, n_nodes=None):
    """Exact integral over lambda ~ Uniform[0, 1] of the every-time restart mixture,
    evaluated at a scalar u for every t. The integrand for restart s at time t is the
    polynomial (1 + lambda u)^{n0} (1 - lambda (1 - u))^{n1} of degree t - s + 1 <= n,
    so Gauss-Legendre with ceil((n + 1) / 2) nodes integrates it exactly."""
    z = np.asarray(z, dtype=np.int64)
    n = len(z)
    if n_nodes is None:
        n_nodes = n // 2 + 2
    x, gw = np.polynomial.legendre.leggauss(n_nodes)
    lam = (x + 1.0) / 2.0
    gw = gw / 2.0
    w = restart_weights(n)
    ones_prefix = np.concatenate([[0], np.cumsum(z)])
    out = np.empty(n)
    for t in range(n):  # 0-based: observations z[0..t]
        s = np.arange(t + 1)  # restart s (0-based) sees z[s..t]
        n1 = (ones_prefix[t + 1] - ones_prefix[s]).astype(float)
        n0 = (t + 1 - s) - n1
        log_e = (n0[:, None] * np.log1p(lam[None, :] * u)
                 + n1[:, None] * np.log1p(-lam[None, :] * (1.0 - u)))
        integral = np.exp(log_e) @ gw
        out[t] = float(w[: t + 1] @ integral) + (TOTAL_MASS - float(w[: t + 1].sum()))
    return out


def brute_force_restart_mixture(z, lambda_values, pi_weights, u):
    """Literal triple loop, plain floats (no log space)."""
    n = len(z)
    w = restart_weights(n)
    out = np.empty(n)
    for t in range(n):
        total = TOTAL_MASS - float(w[: t + 1].sum())
        for lam, pi in zip(lambda_values, pi_weights):
            for s in range(t + 1):
                e = 1.0
                for i in range(s, t + 1):
                    e *= 1.0 + lam * (u - z[i])
                total += pi * w[s] * e
        out[t] = total
    return out


def uniform_all_zero(t, u):
    """(1/L) int_0^L (1 + lambda u)^t dlambda = ((1 + u)^{t+1} - 1) / (u (t + 1)), L = 1.
    Written with expm1/log1p so it stays accurate as u -> 0."""
    return np.expm1((t + 1) * np.log1p(u)) / (u * (t + 1))


def uniform_all_one(t, u):
    """(1/L) int_0^L (1 - lambda (1 - u))^t dlambda = (1 - u^{t+1}) / ((1 - u)(t + 1)), L = 1.
    Written with expm1/log1p so it stays accurate as u -> 1."""
    v = 1.0 - u
    return -np.expm1((t + 1) * np.log1p(-v)) / (v * (t + 1))


def random_binary(rng, n, rate):
    return (rng.random(n) < rate).astype(np.int64)


# ---------------------------------------------------------------------------
# 1, 15. Normalisation and the midpoint grid
# ---------------------------------------------------------------------------

class TestNormalisationAndGrid(unittest.TestCase):
    def test_midpoint_grid_endpoints_mean_and_weights(self) -> None:
        for K in (1, 2, 7, 100, 800):
            lam, pi = uniform_lambda_grid(K)
            self.assertEqual(len(lam), K)
            self.assertAlmostEqual(lam[0], 1.0 / (2 * K), places=15)
            self.assertAlmostEqual(lam[-1], 1.0 - 1.0 / (2 * K), places=15)
            self.assertAlmostEqual(float(lam.mean()), 0.5, places=14)
            self.assertAlmostEqual(float(pi.sum()), 1.0, places=14)
            np.testing.assert_allclose(np.diff(lam), 1.0 / K, rtol=1e-12)

    def test_single_point_grid_is_lambda_one_half(self) -> None:
        rng = np.random.default_rng(10)
        z = random_binary(rng, 30, 0.3)
        lam, pi = uniform_lambda_grid(1)
        got = pure_log_mixture(z, lam, pi, 0.4)
        expected = np.cumsum(np.log1p(0.5 * (0.4 - z)))
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13)

    def test_neutral_point_gives_wealth_one_not_K(self) -> None:
        # Z = 0 at u = 0 (and Z = 1 at u = 1) makes every factor exactly 1, so every
        # component wealth is 1: the mixture must be 1, not K (sum instead of average).
        for K in (1, 25, 800):
            lam, pi = uniform_lambda_grid(K)
            np.testing.assert_allclose(pure_log_mixture(np.zeros(12, dtype=np.int64), lam, pi, 0.0), 0.0, atol=1e-13)
            np.testing.assert_allclose(pure_log_mixture(np.ones(12, dtype=np.int64), lam, pi, 1.0), 0.0, atol=1e-13)
            # Production: every restart has E = 1, so the mixture is the total mass W.
            np.testing.assert_allclose(
                production_log_mixture(np.zeros(12, dtype=np.int64), lam, pi, 0.0), math.log(TOTAL_MASS), atol=1e-13)


# ---------------------------------------------------------------------------
# 2, 3. One- and two-step hand calculations (exact for any midpoint grid)
# ---------------------------------------------------------------------------

class TestOneAndTwoStepClosedForms(unittest.TestCase):
    def test_one_step_pure(self) -> None:
        # M_1 is linear in lambda, so the midpoint grid equals the continuous uniform
        # mixture exactly: 1 + u/2 for Z = 0, 1 - (1 - u)/2 for Z = 1.
        for K in (1, 3, 100):
            lam, pi = uniform_lambda_grid(K)
            for u in (0.0, 0.03, 0.5, 0.97, 1.0):
                self.assertAlmostEqual(math.exp(pure_log_mixture([0], lam, pi, u)[0]), 1.0 + u / 2.0, places=13)
                self.assertAlmostEqual(math.exp(pure_log_mixture([1], lam, pi, u)[0]), 1.0 - (1.0 - u) / 2.0, places=13)

    def test_one_step_production(self) -> None:
        w1 = restart_weights(1)[0]
        lam, pi = uniform_lambda_grid(50)
        for u in (0.1, 0.6):
            got0 = math.exp(production_log_mixture([0], lam, pi, u)[0])
            got1 = math.exp(production_log_mixture([1], lam, pi, u)[0])
            self.assertAlmostEqual(got0, w1 * (1.0 + u / 2.0) + (TOTAL_MASS - w1), places=13)
            self.assertAlmostEqual(got1, w1 * (1.0 - (1.0 - u) / 2.0) + (TOTAL_MASS - w1), places=13)

    def test_two_step_polynomial_pure(self) -> None:
        # Z = (0, 1): M_2(u; lambda) = 1 + (2u - 1) lambda - u(1 - u) lambda^2.
        # Continuous: 1 + (2u - 1)/2 - u(1 - u)/3.  Midpoint K-grid: the lambda^2 moment
        # is 1/3 - 1/(12 K^2) instead of 1/3.
        for u in (0.05, 0.3, 0.5, 0.8):
            continuous = 1.0 + (2 * u - 1) / 2.0 - u * (1 - u) / 3.0
            for K in (1, 2, 5, 100, 800):
                lam, pi = uniform_lambda_grid(K)
                exact_grid = 1.0 + (2 * u - 1) / 2.0 - u * (1 - u) * (1.0 / 3.0 - 1.0 / (12.0 * K * K))
                got = math.exp(pure_log_mixture([0, 1], lam, pi, u)[1])
                self.assertAlmostEqual(got, exact_grid, places=13, msg=f"u={u}, K={K}")
                self.assertAlmostEqual(got, continuous, delta=u * (1 - u) / (12.0 * K * K) + 1e-13)

    def test_two_step_polynomial_production(self) -> None:
        # Restart 1 sees (0, 1): the quadratic above; restart 2 sees only (1):
        # 1 - lambda(1 - u), mean 1 - (1 - u)/2. Remainder W - w1 - w2.
        w1, w2 = restart_weights(2)
        for u in (0.2, 0.7):
            for K in (3, 400):
                lam, pi = uniform_lambda_grid(K)
                quad = 1.0 + (2 * u - 1) / 2.0 - u * (1 - u) * (1.0 / 3.0 - 1.0 / (12.0 * K * K))
                expected = w1 * quad + w2 * (1.0 - (1.0 - u) / 2.0) + (TOTAL_MASS - w1 - w2)
                got = math.exp(production_log_mixture([0, 1], lam, pi, u)[1])
                self.assertAlmostEqual(got, expected, places=13)


# ---------------------------------------------------------------------------
# 4, 5. All-zero / all-one closed forms (grid converges to them at rate 1/K^2)
# ---------------------------------------------------------------------------

class TestAllZeroAllOneClosedForms(unittest.TestCase):
    def assert_converges(self, z, u, reference):
        errors = []
        for K in (25, 50, 100, 200, 400, 800):
            lam, pi = uniform_lambda_grid(K)
            got = np.exp(pure_log_mixture(z, lam, pi, u))
            errors.append(float(np.max(np.abs(got / reference - 1.0))))
        # Midpoint error is O(1/K^2): each doubling of K should cut it by ~4.
        for a, b in zip(errors, errors[1:]):
            self.assertGreater(a / b, 3.5, f"errors {errors}")
        self.assertLess(errors[-1], 1e-4, f"errors {errors}")

    def test_all_zero(self) -> None:
        for n, u in ((10, 0.2), (40, 0.05), (40, 0.5)):
            t = np.arange(1, n + 1)
            self.assert_converges(np.zeros(n, dtype=np.int64), u, uniform_all_zero(t, u))

    def test_all_one(self) -> None:
        for n, u in ((10, 0.8), (40, 0.95), (40, 0.5)):
            t = np.arange(1, n + 1)
            self.assert_converges(np.ones(n, dtype=np.int64), u, uniform_all_one(t, u))

    def test_all_zero_production_restart_sum(self) -> None:
        # Restart s sees t - s + 1 zeros, so the continuous version is
        # sum_s w_s * uniform_all_zero(t - s + 1, u) + (W - sum_{s<=t} w_s).
        n, u = 30, 0.3
        w = restart_weights(n)
        expected = np.array([
            sum(w[s] * uniform_all_zero(t - s + 1, u) for s in range(t + 1)) + TOTAL_MASS - w[: t + 1].sum()
            for t in range(n)
        ])
        lam, pi = uniform_lambda_grid(800)
        got = np.exp(production_log_mixture(np.zeros(n, dtype=np.int64), lam, pi, u))
        np.testing.assert_allclose(got, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 6, 7. Sufficient statistics and terminal permutation invariance (pure mixture)
# ---------------------------------------------------------------------------

class TestSufficientStatistics(unittest.TestCase):
    def test_sequential_product_equals_count_formula(self) -> None:
        rng = np.random.default_rng(6)
        lam, pi = uniform_lambda_grid(37)
        for _ in range(20):
            n = int(rng.integers(1, 60))
            z = random_binary(rng, n, float(rng.uniform(0.05, 0.7)))
            u = float(rng.uniform(0.01, 0.99))
            n1 = np.cumsum(z)
            n0 = np.arange(1, n + 1) - n1
            expected = logsumexp(
                np.log(pi)[:, None]
                + n0[None, :] * np.log1p(lam[:, None] * u)
                + n1[None, :] * np.log1p(-lam[:, None] * (1 - u)),
                axis=0,
            )
            np.testing.assert_allclose(pure_log_mixture(z, lam, pi, u), expected, rtol=0, atol=1e-11)

    def test_terminal_value_is_permutation_invariant(self) -> None:
        lam, pi = uniform_lambda_grid(100)
        a = pure_log_mixture([0, 0, 1, 0, 1, 1], lam, pi, 0.35)
        b = pure_log_mixture([1, 0, 1, 0, 0, 1], lam, pi, 0.35)
        self.assertAlmostEqual(float(a[-1]), float(b[-1]), places=13)
        self.assertFalse(np.allclose(a[:-1], b[:-1]))  # intermediate paths differ
        rng = np.random.default_rng(7)
        z = random_binary(rng, 50, 0.3)
        self.assertAlmostEqual(float(pure_log_mixture(z, lam, pi, 0.4)[-1]),
                               float(pure_log_mixture(rng.permutation(z), lam, pi, 0.4)[-1]), places=11)


# ---------------------------------------------------------------------------
# 8. Null expectation by exact enumeration (sign check)
# ---------------------------------------------------------------------------

class TestNullExpectation(unittest.TestCase):
    def expected_wealth(self, evaluator, t, u, q, lam, pi):
        """E_q[M_t(u)] by summing over all 2^t binary sequences."""
        total = 0.0
        for seq in itertools.product((0, 1), repeat=t):
            k = sum(seq)
            prob = q ** k * (1 - q) ** (t - k)
            total += prob * math.exp(evaluator(np.array(seq), lam, pi, u)[-1])
        return total

    def test_single_factor_identity(self) -> None:
        for u in (0.1, 0.5, 0.9):
            for lam in (0.2, 0.99):
                self.assertAlmostEqual((1 - u) * (1 + lam * u) + u * (1 - lam * (1 - u)), 1.0, places=14)

    def test_pure_mixture_expectation(self) -> None:
        # E_q[M_t(u)] = sum_k pi_k (1 + lambda_k (u - q))^t: 1 at q = u, > 1 below, < 1 above.
        lam, pi = uniform_lambda_grid(9)
        t, u = 8, 0.3
        for q in (0.1, 0.3, 0.5):
            got = self.expected_wealth(pure_log_mixture, t, u, q, lam, pi)
            self.assertAlmostEqual(got, float(pi @ (1 + lam * (u - q)) ** t), places=12)
        self.assertAlmostEqual(self.expected_wealth(pure_log_mixture, t, u, u, lam, pi), 1.0, places=12)
        self.assertGreater(self.expected_wealth(pure_log_mixture, t, u, 0.1, lam, pi), 1.0)
        self.assertLess(self.expected_wealth(pure_log_mixture, t, u, 0.5, lam, pi), 1.0)

    def test_production_expectation(self) -> None:
        # Under q: E[E_{s:t}] = (1 + lambda(u - q))^{t-s+1}; at q = u the whole mixture
        # has expectation exactly the total mass W (supermartingale with M_0 = W).
        lam, pi = uniform_lambda_grid(9)
        t, u = 8, 0.3
        w = restart_weights(t)
        for q in (0.1, 0.3, 0.5):
            expected = sum(w[s] * float(pi @ (1 + lam * (u - q)) ** (t - s)) for s in range(t)) + TOTAL_MASS - w.sum()
            got = self.expected_wealth(production_log_mixture, t, u, q, lam, pi)
            self.assertAlmostEqual(got, expected, places=12)
        self.assertAlmostEqual(self.expected_wealth(production_log_mixture, t, u, u, lam, pi), TOTAL_MASS, places=12)


# ---------------------------------------------------------------------------
# 9. Mixture lies inside its component envelope
# ---------------------------------------------------------------------------

class TestComponentEnvelope(unittest.TestCase):
    def test_between_min_and_max_component(self) -> None:
        rng = np.random.default_rng(9)
        lam, pi = uniform_lambda_grid(20)
        for evaluator in (pure_log_mixture, production_log_mixture):
            for _ in range(5):
                z = random_binary(rng, 80, float(rng.uniform(0.02, 0.4)))
                u = float(rng.uniform(0.01, 0.99))
                mix = evaluator(z, lam, pi, u)
                comps = np.stack([evaluator(z, [lv], [1.0], u) for lv in lam])
                self.assertTrue(np.all(mix >= comps.min(axis=0) - 1e-12))
                self.assertTrue(np.all(mix <= comps.max(axis=0) + 1e-12))


# ---------------------------------------------------------------------------
# 10. Hindsight-optimal lambda (the closed-form "oracle" arm)
# ---------------------------------------------------------------------------

class TestHindsightLambda(unittest.TestCase):
    def test_formula_matches_numerical_argmax(self) -> None:
        # log W(lambda) = n0 log(1 + lambda u) + n1 log(1 - lambda (1 - u)) is concave;
        # its stationary point is lambda* = (u - Zbar) / (u (1 - u)).
        rng = np.random.default_rng(10)
        for _ in range(30):
            n = int(rng.integers(5, 80))
            z = random_binary(rng, n, float(rng.uniform(0.02, 0.6)))
            u = float(rng.uniform(0.05, 0.95))
            z_bar = float(z.mean())
            n1 = float(z.sum()); n0 = n - n1
            lam_max = 1.0 / (1.0 - u)  # log W is defined on [0, 1/(1-u))
            grid = np.linspace(0.0, lam_max * (1 - 1e-9), 200001)
            log_w = n0 * np.log1p(grid * u) + (n1 * np.log1p(-grid * (1 - u)) if n1 > 0 else 0.0)
            unconstrained = (u - z_bar) / (u * (1 - u))
            # Unconstrained positive optimum (what the closed-form arm uses via KL).
            if unconstrained > 0 and n1 > 0:
                self.assertAlmostEqual(float(grid[np.argmax(log_w)]), unconstrained, delta=2 * lam_max / 200000)
                self.assertAlmostEqual(float(log_w.max()), n * float(kl_bernoulli(np.array([z_bar]), np.array([u]))[0]),
                                       delta=1e-7)
            # Constrained to [0, L] with L = 1: clip(lambda*, 0, 1).
            in_range = grid <= 1.0
            constrained = min(max(unconstrained, 0.0), 1.0)
            self.assertAlmostEqual(float(grid[in_range][np.argmax(log_w[in_range])]), constrained,
                                   delta=2 * lam_max / 200000)


# ---------------------------------------------------------------------------
# 11. Monotonicity in u
# ---------------------------------------------------------------------------

class TestMonotoneInU(unittest.TestCase):
    def test_nondecreasing_on_dense_grid(self) -> None:
        rng = np.random.default_rng(11)
        lam, pi = uniform_lambda_grid(100)
        u_grid = np.linspace(0.0, 1.0, 1001)
        for evaluator in (pure_log_mixture, production_log_mixture):
            for rate in (0.03, 0.2, 0.6):
                z = random_binary(rng, 60, rate)
                values = np.stack([evaluator(z, lam, pi, u) for u in u_grid])  # (u, t)
                self.assertTrue(np.all(np.diff(values, axis=0) >= -1e-12), f"{evaluator.__name__}, rate={rate}")


# ---------------------------------------------------------------------------
# 12, 13. UCB inversion
# ---------------------------------------------------------------------------

class TestInversion(unittest.TestCase):
    def test_brackets_the_threshold(self) -> None:
        # Check the returned U_t against an independent (non-log-space) brute force:
        # M_t(U_t) >= 1/alpha (U_t is the upper bracket end) and M_t(U_t - 2^-iters) < 1/alpha.
        rng = np.random.default_rng(12)
        lam, pi = uniform_lambda_grid(15)
        alpha = 0.05
        z = random_binary(rng, 60, 0.05)
        ucb = production_ucb(z, lam, pi, alpha)
        crossing = np.flatnonzero(ucb < 1.0)
        self.assertGreater(len(crossing), 10, "test sequence should produce interior crossings")
        for t in crossing:
            at_u = brute_force_restart_mixture(z[: t + 1], lam, pi, float(ucb[t]))[-1]
            below = brute_force_restart_mixture(z[: t + 1], lam, pi, float(ucb[t]) - 2.0 ** -ITERS)[-1]
            self.assertGreaterEqual(at_u, 1.0 / alpha * (1 - 1e-12), f"t={t + 1}")
            self.assertLess(below, 1.0 / alpha, f"t={t + 1}")
        # No crossing: M_t(1) < 1/alpha must return exactly 1.
        for t in np.flatnonzero(ucb == 1.0):
            self.assertLess(brute_force_restart_mixture(z[: t + 1], lam, pi, 1.0)[-1], 1.0 / alpha)

    def test_no_crossing_returns_one(self) -> None:
        lam, pi = uniform_lambda_grid(50)
        ucb = production_ucb(np.ones(40, dtype=np.int64), lam, pi, 0.05)
        np.testing.assert_array_equal(ucb, 1.0)

    def test_one_step_analytic_inversion(self) -> None:
        # t = 1, Z = 0: M_1(u) = w1 (1 + u/2) + (W - w1), linear in lambda so exact for
        # any midpoint grid. Solving M_1(u) = 1/alpha: u* = 2 (1/alpha - W) / w1.
        w1 = restart_weights(1)[0]
        for u_star in (0.1, 0.5, 0.9):
            alpha = 1.0 / (TOTAL_MASS + w1 * u_star / 2.0)
            for K in (1, 100):
                lam, pi = uniform_lambda_grid(K)
                got = float(production_ucb([0], lam, pi, alpha)[0])
                self.assertGreaterEqual(got, u_star - 1e-12)
                self.assertLessEqual(got, u_star + 2.0 ** -ITERS + 1e-12)


# ---------------------------------------------------------------------------
# 14. Grid convergence of the UCB against exact quadrature
# ---------------------------------------------------------------------------

class TestGridConvergence(unittest.TestCase):
    def test_ucb_converges_to_continuous_uniform(self) -> None:
        rng = np.random.default_rng(14)
        alpha = 0.05
        z = random_binary(rng, 120, 0.05)
        n = len(z)
        # Reference UCB: invert the exact Gauss-Legendre integral by 60-step bisection.
        lower, upper = np.zeros(n), np.ones(n)
        ref_at_one = continuous_uniform_restart_mixture(z, 1.0)
        lower[ref_at_one < 1 / alpha] = 1.0
        for _ in range(60):
            mid = (lower + upper) / 2.0
            ok = np.array([continuous_uniform_restart_mixture(z[: t + 1], float(mid[t]))[-1] < 1 / alpha
                           if lower[t] < 1.0 else True for t in range(n)])
            lower = np.where(ok, mid, lower)
            upper = np.where(ok, upper, mid)
        self.assertGreater(int(np.sum(upper < 1.0)), 50)

        errors = {}
        for K in (25, 50, 100, 200, 400, 800):
            lam, pi = uniform_lambda_grid(K)
            errors[K] = float(np.max(np.abs(production_ucb(z, lam, pi, alpha) - upper)))
        Ks = sorted(errors)
        for a, b in zip(Ks, Ks[1:]):
            self.assertGreater(errors[a], errors[b], f"errors {errors}")
        self.assertLess(errors[800], 1e-5, f"errors {errors}")
        print(f"\n  max_t |U_K - U_quad|: " + ", ".join(f"K={K}: {errors[K]:.2e}" for K in Ks))


# ---------------------------------------------------------------------------
# 16. Log-space vs naive arithmetic
# ---------------------------------------------------------------------------

class TestLogSpaceVsNaive(unittest.TestCase):
    def test_matches_naive_for_small_t(self) -> None:
        rng = np.random.default_rng(16)
        lam, pi = uniform_lambda_grid(11)
        for _ in range(5):
            z = random_binary(rng, 25, float(rng.uniform(0.05, 0.5)))
            u = float(rng.uniform(0.05, 0.95))
            naive = brute_force_restart_mixture(z, lam, pi, u)
            np.testing.assert_allclose(np.exp(production_log_mixture(z, lam, pi, u)), naive, rtol=1e-10)

    def test_huge_t_stays_finite_where_naive_overflows(self) -> None:
        lam, pi = uniform_lambda_grid(100)
        n, u = 2500, 0.9
        z = np.zeros(n, dtype=np.int64)
        with np.errstate(over="ignore"):
            self.assertFalse(np.isfinite(float(pi @ (1 + lam * u) ** n)))  # naive overflows
        expected = logsumexp(np.log(pi) + n * np.log1p(lam * u))
        got_pure = pure_log_mixture(z, lam, pi, u)
        self.assertTrue(np.all(np.isfinite(got_pure)))
        self.assertAlmostEqual(float(got_pure[-1]), float(expected), delta=1e-9 * abs(expected))
        got_prod = production_log_mixture(z, lam, pi, u)
        self.assertTrue(np.all(np.isfinite(got_prod)))
        self.assertGreater(float(got_prod[-1]), float(expected) + math.log(restart_weights(1)[0]) - 1e-9)

    def test_huge_t_underflow_side(self) -> None:
        # All ones at small u: wealth is astronomically small; log space must stay
        # finite and match the log-sum-exp of the count formula.
        lam, pi = uniform_lambda_grid(100)
        n, u = 2500, 0.01
        z = np.ones(n, dtype=np.int64)
        expected = logsumexp(np.log(pi) + n * np.log1p(-lam * (1 - u)))
        got_pure = pure_log_mixture(z, lam, pi, u)
        self.assertTrue(np.all(np.isfinite(got_pure)))
        self.assertAlmostEqual(float(got_pure[-1]), float(expected), delta=1e-9 * abs(expected))


# ---------------------------------------------------------------------------
# 17. Boundary values
# ---------------------------------------------------------------------------

class TestBoundaries(unittest.TestCase):
    def test_u_near_zero_and_one(self) -> None:
        lam, pi = uniform_lambda_grid(800)
        t = np.arange(1, 31)
        for u in (1e-12, 1e-6):
            got = np.exp(pure_log_mixture(np.zeros(30, dtype=np.int64), lam, pi, u))
            np.testing.assert_allclose(got, uniform_all_zero(t, u), rtol=1e-6)
        for u in (1 - 1e-6, 1 - 1e-12):
            got = np.exp(pure_log_mixture(np.ones(30, dtype=np.int64), lam, pi, u))
            np.testing.assert_allclose(got, uniform_all_one(t, u), rtol=1e-6)

    def test_lambda_near_upper_limit(self) -> None:
        # lambda -> 1 with Z = 1 at u = 0: factor 1 - lambda is tiny but positive.
        lam = 1.0 - 1e-9
        got = pure_log_mixture(np.ones(20, dtype=np.int64), [lam], [1.0], 0.0)
        np.testing.assert_allclose(got, np.arange(1, 21) * math.log1p(-lam), rtol=1e-9)
        self.assertTrue(np.all(np.isfinite(production_log_mixture(np.ones(20, dtype=np.int64), [lam], [1.0], 0.0))))

    def test_lambda_zero_and_one_are_rejected(self) -> None:
        # At lambda = 0 every sequence has M_t = 1, a component that can never reject;
        # the implementation disallows it (and lambda >= 1) rather than silently accepting.
        for bad in (0.0, 1.0):
            with self.assertRaises(ValueError):
                production_ucb([0, 1, 0], [bad], [1.0], 0.05)

    def test_weights_must_be_normalised(self) -> None:
        with self.assertRaises(ValueError):
            production_ucb([0, 1, 0], [0.25, 0.75], [1.0, 1.0], 0.05)


if __name__ == "__main__":
    unittest.main()
