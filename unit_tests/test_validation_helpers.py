"""Unit tests for av_core's fail-loudly validation helpers: require_file and
require_probability (require_lambda is covered in test_upper_confidence_bound.py).
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from av_core import require_file, require_probability


class TestRequireFile(unittest.TestCase):
    def test_missing_path_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            require_file(Path("/nonexistent/definitely/not/here.json"), "--taxonomy-json")

    def test_directory_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                require_file(Path(tmp_dir), "--taxonomy-json")

    def test_existing_file_does_not_raise(self) -> None:
        with tempfile.NamedTemporaryFile() as tmp_file:
            require_file(Path(tmp_file.name), "--taxonomy-json")  # must not raise


class TestRequireProbability(unittest.TestCase):
    def test_zero_raises_when_zero_excluded_by_default(self) -> None:
        with self.assertRaises(ValueError):
            require_probability(0.0, "alpha")

    def test_zero_allowed_when_include_zero_true(self) -> None:
        require_probability(0.0, "alpha", include_zero=True)  # must not raise

    def test_one_always_raises(self) -> None:
        with self.assertRaises(ValueError):
            require_probability(1.0, "alpha")
        with self.assertRaises(ValueError):
            require_probability(1.0, "alpha", include_zero=True)

    def test_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            require_probability(-0.01, "alpha", include_zero=True)

    def test_nan_raises(self) -> None:
        with self.assertRaises(ValueError):
            require_probability(float("nan"), "alpha")

    def test_infinity_raises(self) -> None:
        with self.assertRaises(ValueError):
            require_probability(float("inf"), "alpha")

    def test_mid_range_value_does_not_raise(self) -> None:
        require_probability(0.05, "alpha")  # must not raise


if __name__ == "__main__":
    unittest.main()
