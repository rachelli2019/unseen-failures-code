import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc, gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from empirical_bernstein import (empirical_bernstein_upper,
                                gamma_exponential_boundary,
                                gamma_exponential_log_mixture)
from run_empirical_bernstein_headline import first_crossing, stopping_statistics


class EmpiricalBernsteinTests(unittest.TestCase):
    def test_closed_form_against_independent_quadrature(self):
        for rho in (1., 25., 100., 400.):
            normalizer = rho*np.log(rho)-gammaln(rho)-np.log(gammainc(rho, rho))
            for s, v in ((0., 0.), (2., 0.), (10., 20.), (40., 300.)):
                def integrand(lam):
                    return np.exp(lam*s + v*(np.log1p(-lam)+lam)
                                  + normalizer + (rho-1)*np.log1p(-lam)-rho*(1-lam))
                expected = quad(integrand, 0, 1, epsabs=1e-10)[0]
                self.assertAlmostEqual(float(gamma_exponential_log_mixture(s, v, rho)),
                                       np.log(expected), places=9)

    def test_boundary_inversion_and_order(self):
        v = np.array([0., .25, 1., 20., 100., 1000.])
        b = gamma_exponential_boundary(v)
        np.testing.assert_allclose(gamma_exponential_log_mixture(b, v), np.log(20), atol=1e-8)
        self.assertTrue(np.all(np.diff(b) > 0))
        self.assertTrue(np.all(gamma_exponential_boundary(v, alpha=.01) > b))

    def test_prediction_is_strictly_past(self):
        y = np.array([1., 0., 1., 0.])
        u, v = empirical_bernstein_upper(y)
        np.testing.assert_allclose(v, np.cumsum([.25, 1., .25, 4/9]))
        prefix, _ = empirical_bernstein_upper(y[:2])
        np.testing.assert_array_equal(prefix, u[:2])
        batch, _ = empirical_bernstein_upper(np.stack([y, y]))
        np.testing.assert_array_equal(batch[0], u)

    def test_exact_coverage_with_history_dependent_depletion(self):
        # Enumerate a finite species experiment, including a non-failure atom.
        # The changing conditional mean is determined by previously seen species.
        paths = np.array(list(itertools.product(range(3), repeat=8)))
        p = np.array([.2, .5, .3])
        y = np.zeros(paths.shape)
        q = np.zeros(paths.shape)
        for row, path in enumerate(paths):
            seen = {0}
            for t, x in enumerate(path):
                y[row, t] = x not in seen
                seen.add(x)
                q[row, t] = sum(p[k] for k in range(3) if k not in seen)
        u, _ = empirical_bernstein_upper(y, alpha=.2, rho=1.)
        violation = np.any(u < q, axis=1)
        self.assertLessEqual(np.prod(p[paths], axis=1)[violation].sum(), .2)

    def test_invalid_inputs(self):
        for y in ([], [np.nan], [-1], [1.1]):
            with self.assertRaises(ValueError):
                empirical_bernstein_upper(y)
        for rho in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                gamma_exponential_boundary([0], rho=rho)

    def test_stopping_censoring_and_strict_threshold(self):
        bound = np.array([[.4, .2, .3], [.4, .3, .3], [.4, .3, .2]])
        np.testing.assert_array_equal(first_crossing(bound, .3), [2, np.inf, 3])
        stats = stopping_statistics(bound, np.zeros_like(bound), .3, bound)
        self.assertEqual(stats['censored_trials'], 1)
        self.assertIsNone(stats['mean_stopping_time'])
        self.assertAlmostEqual(stats['restricted_mean_draws'], 8/3)
        self.assertAlmostEqual(stats['stopped_fraction'], 2/3)
        self.assertEqual(stats['restricted_mean_minus_restart'], 0)


if __name__ == '__main__':
    unittest.main()
