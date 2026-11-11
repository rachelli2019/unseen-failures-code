"""Discover and run every test in unit_tests/, failing loudly: exits non-zero
on ANY test failure or error, with no silent skips. Run directly:

    python unit_tests/run_all.py
"""
import sys
import unittest
from pathlib import Path

if __name__ == "__main__":
    this_dir = Path(__file__).resolve().parent
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(this_dir), pattern="test_*.py", top_level_dir=str(this_dir))

    collected_errors = loader.errors
    if collected_errors:
        for error in collected_errors:
            print(error, file=sys.stderr)
        raise RuntimeError(f"{len(collected_errors)} test module(s) failed to import -- fix before running")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)
    sys.exit(0)
