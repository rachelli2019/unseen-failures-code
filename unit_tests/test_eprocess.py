"""Unit tests for av_core.log_eprocess_at_p: the fixed-lambda e-process
E_t(p) = prod_{j<=t} (1 + lambda*(p - z_j)), evaluated at known synthetic Z_t
sequences with hand-computed expected values.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from av_core import log_eprocess_at_p


class TestLogEprocessKnownSequence(unittest.TestCase):
    """z = [1, 0, 1], lambda = 0.5, evaluated at a constant p = 0.5 for every
    prefix t (log_eprocess_at_p is vector-valued in p, but a constant array
    isolates the per-t recursion).

    Hand-derived factors at p=0.5, lambda=0.5:
      j=1 (z=1): 1 + 0.5*(0.5 - 1) = 1 - 0.25 = 0.75
      j=2 (z=0): 1 + 0.5*(0.5 - 0) = 1 + 0.25 = 1.25
      j=3 (z=1): 1 + 0.5*(0.5 - 1) = 0.75
    E_1 = 0.75
    E_2 = 0.75 * 1.25 = 0.9375
    E_3 = 0.9375 * 0.75 = 0.703125
    """

    def test_matches_hand_computed_products(self) -> None:
        z = np.array([1, 0, 1], dtype=np.int8)
        p = np.full(3, 0.5)
        log_e = log_eprocess_at_p(z, 0.5, p)

        expected_e = np.array([0.75, 0.9375, 0.703125])
        np.testing.assert_allclose(np.exp(log_e), expected_e, rtol=1e-12)
        np.testing.assert_allclose(log_e, np.log(expected_e), rtol=1e-12)

    def test_all_zeros_z_grows_with_p_above_zero(self) -> None:
        # z = [0, 0, 0]: E_t(p) = (1 + lambda*p)^t, strictly increasing in t
        # for any p > 0.
        z = np.zeros(3, dtype=np.int8)
        p = np.full(3, 0.4)
        log_e = log_eprocess_at_p(z, 0.5, p)
        expected = np.array([(1 + 0.5 * 0.4) ** t for t in (1, 2, 3)])
        np.testing.assert_allclose(np.exp(log_e), expected, rtol=1e-12)

    def test_p_equals_one_reduces_to_product_over_zeros_only(self) -> None:
        # At p=1, the z=1 factor is exactly 1 (1 + lambda*(1-1) = 1), so only
        # the zeros contribute: E_t(1) = (1+lambda)^(number of zeros so far).
        z = np.array([1, 0, 1, 0, 0], dtype=np.int8)
        p = np.ones(5)
        log_e = log_eprocess_at_p(z, 0.25, p)
        cum_zeros = np.array([0, 1, 1, 2, 3])
        expected = (1.25) ** cum_zeros
        np.testing.assert_allclose(np.exp(log_e), expected, rtol=1e-12)


class TestLogEprocessRaisesOnNonFinite(unittest.TestCase):
    def test_lambda_at_boundary_with_extreme_p_can_raise_or_stay_finite(self) -> None:
        # lambda close to 1 with p driving a factor near/at zero must not
        # silently return -inf/NaN: the function is documented to raise
        # FloatingPointError on non-finite output rather than propagate it.
        z = np.ones(50, dtype=np.int8)
        p = np.zeros(50)  # p=0, z=1 everywhere -> factor = 1 + lambda*(0-1) = 1-lambda
        # lambda=0.999999999 keeps the factor tiny-but-positive, not exactly zero;
        # this should compute finite (very negative) log values, not raise.
        log_e = log_eprocess_at_p(z, 0.999999999, p)
        self.assertTrue(np.all(np.isfinite(log_e)))


if __name__ == "__main__":
    unittest.main()
