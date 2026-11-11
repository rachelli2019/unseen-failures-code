"""Checks for historical-report parsing and exact coverage accounting."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/plotting"))
from build_coverage_index import collect, validate
from coverage_record import write_record
import numpy as np


class CoverageIndexTests(unittest.TestCase):
    def test_counts_recover_precision_lost_in_markdown(self):
        self.assertEqual(validate(30000, 10, 1.0000, 1), 1)
        with self.assertRaises(ValueError):
            validate(1000, 10, 0.992, 7)
        with self.assertRaises(ValueError):
            validate(1000, 10, float("nan"))

    def test_simultaneous_coverage_includes_earlier_violations(self):
        with tempfile.TemporaryDirectory() as tmp:
            plot = Path(tmp) / "example_bounds.png"
            plot.touch()
            q = np.array([[0.5, 0.5], [0.5, 0.5]])
            b = np.array([[0.4, 0.6], [0.6, 0.6]])
            write_record(plot, "example", q, [("test", "run.npz", "bound", b, 0.95)])
            # A new explicit record must supersede an old Markdown report.
            plot.with_name("example_coverage.md").write_text(
                "# Simultaneous coverage at t = 2: example\n"
                "Pr(no violation in [1, t]) over 2 bootstrap trials.\n"
                "| test | 1.0000 | 1.0000 | 0 |\n")
            rows, missing = collect(Path(tmp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["violation_count"], 1)
            self.assertEqual(rows[0]["simultaneous_coverage"], 0.5)
            self.assertEqual(missing, [])

    def test_summary_arm_source_and_missing_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "comparison.png").touch()
            (directory / "unreported.png").touch()
            (directory / "comparison_summary.json").write_text(json.dumps(dict(
                n_trials=1000, horizon=3000, arms={"Grid": dict(
                    source="artifacts/grid/run42/combined/trajectories.npz:bound__grid",
                    simultaneous_coverage=0.973)})))
            rows, missing = collect(directory)
            self.assertEqual(rows[0]["run_id"], "run42")
            self.assertEqual(rows[0]["array_key"], "bound__grid")
            self.assertEqual(rows[0]["violation_count"], 27)
            self.assertEqual(missing, [directory / "unreported.png"])


if __name__ == "__main__":
    unittest.main()
