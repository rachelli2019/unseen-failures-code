"""Unit tests for the discovery backends' strictness: bad config values must raise
rather than be coerced, and a backend must not be able to hand the bound a z that is
not binary or a q outside [0, 1].
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import lambda_grid_discovery_backends as backends
from lambda_grid_discovery_backends import get_backend


class TestDrawValidation(unittest.TestCase):
    def test_accepts_a_well_formed_draw(self) -> None:
        z = np.array([0, 1, 1, 0])
        q = np.array([0.1, 0.2, 0.3, 0.4])
        got_z, got_q = backends._validated_draw(z, q, 4, "test")
        self.assertEqual(got_z.dtype, np.int8)
        np.testing.assert_array_equal(got_z, z)
        np.testing.assert_allclose(got_q, q)

    def test_rejects_non_binary_z(self) -> None:
        with self.assertRaises(ValueError) as cm:
            backends._validated_draw(np.array([0, 2, 1, 0]), np.full(4, 0.5), 4, "test")
        self.assertIn("non-binary", str(cm.exception))

    def test_rejects_q_above_one_and_below_zero(self) -> None:
        for bad in (np.array([0.1, 1.5, 0.2, 0.3]), np.array([0.1, -0.2, 0.2, 0.3])):
            with self.assertRaises(ValueError) as cm:
                backends._validated_draw(np.zeros(4, dtype=int), bad, 4, "test")
            self.assertIn("outside [0, 1]", str(cm.exception))

    def test_rejects_non_finite_q(self) -> None:
        for bad in (np.nan, np.inf):
            q = np.full(4, 0.5)
            q[2] = bad
            with self.assertRaises(ValueError):
                backends._validated_draw(np.zeros(4, dtype=int), q, 4, "test")

    def test_rejects_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            backends._validated_draw(np.zeros(3, dtype=int), np.full(3, 0.5), 4, "test")

    def test_every_backend_routes_its_draw_through_validation(self) -> None:
        import inspect

        for name, backend in backends.BACKENDS.items():
            src = inspect.getsource(backend["draw"])
            self.assertIn("_validated_draw", src, f"{name} draw() bypasses validation")


class TestConfigValuesAreNotCoerced(unittest.TestCase):
    def test_quoted_number_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            get_backend({"backend": "bernoulli"})["prepare"]({"backend": "bernoulli", "p": "0.05"})

    def test_bool_is_rejected_as_a_number(self) -> None:
        with self.assertRaises(TypeError):
            get_backend({"backend": "bernoulli"})["prepare"]({"backend": "bernoulli", "p": True})

    def test_non_string_path_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            backends._require_str({"taxonomy_json": 3}, "taxonomy_json", "fixed_primary")

    def test_empty_string_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            backends._require_str({"taxonomy_json": ""}, "taxonomy_json", "fixed_primary")

    def test_real_number_passes(self) -> None:
        self.assertEqual(backends._require_number({"p": 0.05}, "p", "bernoulli"), 0.05)
        self.assertEqual(backends._require_number({"p": 1}, "p", "bernoulli"), 1.0)


class TestBackendSelection(unittest.TestCase):
    def test_unknown_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_backend({"backend": "does_not_exist"})

    def test_missing_backend_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_backend({"p": 0.05})

    def test_missing_and_extra_keys_both_raise(self) -> None:
        prepare = get_backend({"backend": "bernoulli"})["prepare"]
        with self.assertRaises(ValueError):
            prepare({"backend": "bernoulli"})                      # missing p
        with self.assertRaises(ValueError):
            prepare({"backend": "bernoulli", "p": 0.05, "q": 1})   # typo'd extra

    def test_bernoulli_draw_is_binary_with_constant_q(self) -> None:
        backend = get_backend({"backend": "bernoulli"})
        state = backend["prepare"]({"backend": "bernoulli", "p": 0.05})
        z, q = backend["draw"](state, np.random.default_rng(0), 500)
        self.assertEqual(set(np.unique(z).tolist()) - {0, 1}, set())
        np.testing.assert_allclose(q, 0.05)


if __name__ == "__main__":
    unittest.main()
