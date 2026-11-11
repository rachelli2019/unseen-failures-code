#!/usr/bin/env python3
"""Rebuild the plot coverage catalog from saved reports and coverage records.

Run from any directory: python src/plotting/build_coverage_index.py
Unknown historical provenance stays blank; no experiment is rerun.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["plot_path", "plot_pdf", "setting", "population_group", "method", "run_id",
          "source_npz", "array_key", "n_trials", "horizon", "violation_count",
          "simultaneous_coverage", "target_coverage", "report_path", "provenance"]


def relative(path: Path) -> str:
    return os.path.relpath(path.resolve(), ROOT)


def validate(n: int, horizon: int, coverage: float, violations=None) -> int:
    if not isinstance(n, int) or n <= 0 or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("Trial count and evaluated horizon must be positive integers")
    if not math.isfinite(coverage) or not 0 <= coverage <= 1:
        raise ValueError(f"Invalid coverage: {coverage}")
    count = round((1 - coverage) * n) if violations is None else violations
    if not isinstance(count, int) or not 0 <= count <= n:
        raise ValueError(f"Invalid violation count: {count}")
    if abs(1 - count / n - coverage) > (0.00005001 if violations is not None else 1e-8):
        raise ValueError("Coverage disagrees with the number of violating trials")
    return count


def row(report, plot, setting, method, n, horizon, coverage, source="", key="",
        violations=None, target="", provenance="saved report; source run unknown"):
    count = validate(n, horizon, coverage, violations)
    source_path = Path(source) if source else None
    run = ""
    if source_path:
        parent = source_path.parent
        run = (parent.parent.name if parent.name == "combined" or parent.name.startswith("task_")
               else source_path.stem if source_path.stem != "trajectories" else parent.name)
    return dict(zip(FIELDS, [relative(plot), relative(plot.with_suffix(".pdf")) if plot.with_suffix(".pdf").exists() else "",
        setting, "full_population" if "full_population" in plot.parts else "other_or_legacy",
        method, run, source, key, n, horizon, count, 1 - count / n, target,
        relative(report), provenance]))


def collect(plot_dir: Path):
    rows, claimed = [], set()
    # Explicit per-figure records take precedence over legacy summary formats.
    for report in sorted(plot_dir.rglob("*_coverage_record.json")):
        record = json.loads(report.read_text())
        plot = report.parent / record["plot_file"]
        if not plot.exists():
            raise ValueError(f"Coverage record references missing plot: {plot}")
        claimed.add(plot)
        for arm in record["arms"]:
            rows.append(row(report, plot, record["setting"], arm["method"], record["n_trials"],
                record["horizon"], arm["coverage"], arm["source_npz"], arm["array_key"],
                arm["violation_count"], arm.get("target_coverage", ""), record["provenance"]))
    for report in sorted(plot_dir.rglob("*_summary.json")):
        summary = json.loads(report.read_text())
        if "arms" not in summary:
            continue
        stem = report.name.removesuffix("_summary.json")
        plots = [report.with_name(stem + ".png")]
        if not plots[0].exists():
            plots = [report.with_name(stem + suffix + ".png") for suffix in ("_bounds", "_ratio")]
        for plot in plots:
            if not plot.exists() or plot in claimed:
                continue
            claimed.add(plot)
            for method, arm in summary["arms"].items():
                source, key = "", ""
                if "source" in arm:
                    source, key = arm["source"].rsplit(":", 1)
                elif "closed_form_npz" in summary:
                    source = summary["fixed_npz"] if method.startswith("Fixed") else (
                        summary["closed_form_npz"] if method.startswith(("Closed", "Best")) else summary["grid_npz"])
                elif "ablation_npz" in summary:
                    source = summary["ablation_npz"] if method.startswith("Single") else summary["grid_npz"]
                rows.append(row(report, plot, summary.get("setting", summary.get("title", stem)),
                    method, summary["n_trials"], summary["horizon"], arm["simultaneous_coverage"],
                    source, key, provenance="saved report; source recorded" if source else "saved report; source run unknown"))
    for report in sorted(plot_dir.rglob("*_coverage.md")):
        plot = report.with_name(report.name.removesuffix("_coverage.md") + "_bounds.png")
        if plot in claimed:
            continue
        if not plot.exists():
            raise ValueError(f"Coverage report has no corresponding plot: {report}")
        text = report.read_text()
        header = re.search(r"coverage at t = (\d+): (.+)", text)
        trials = re.search(r"over (\d+) bootstrap trials", text)
        if not header or not trials:
            raise ValueError(f"Unrecognized coverage report: {report}")
        claimed.add(plot)
        found = 0
        for line in text.splitlines():
            cells = [x.strip() for x in line.strip("|").split("|")]
            if len(cells) == 4 and re.fullmatch(r"\d+\.\d+", cells[1]):
                rows.append(row(report, plot, header[2], cells[0], int(trials[1]), int(header[1]),
                                float(cells[1]), violations=int(cells[3])))
                found += 1
        if not found:
            raise ValueError(f"No coverage rows: {report}")
    missing = sorted(set(plot_dir.rglob("*.png")) - claimed)
    rows.sort(key=lambda r: (r["plot_path"], r["method"]))
    keys = [(r["plot_path"], r["method"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate plot/method records")
    return rows, missing


def build(plot_dir: Path):
    rows, missing = collect(plot_dir)
    csv_path = plot_dir / "coverage_index.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    def link(path, label):
        return f"[{label}]({os.path.relpath(ROOT / path, plot_dir)})"
    lines = ["# Final empirical simultaneous coverage", "",
        f"{len(rows)} method entries across {len(set(r['plot_path'] for r in rows))} figures. PNG/PDF pairs count as one figure.", "",
        "Coverage is the fraction of trials with no violation at any timestep through the evaluated horizon. "
        "Rows from different figures may reuse the same trials; do not pool them. "
        "Blank CSV fields and ‘unknown’ below mean the historical report did not record that detail. "
        "Target coverage is not inferred from a plot's confidence-band level or assumed to be 95%.", "",
        "Refresh from the repository root: `python src/plotting/build_coverage_index.py`.", "",
        "[Download the complete CSV](coverage_index.csv). The CSV includes source paths, array keys and provenance.", ""]
    for group in ("full_population", "other_or_legacy"):
        lines += ["## " + ("Full population" if group == "full_population" else "Other and legacy plots"), ""]
        current = None
        for r in rows:
            if r["population_group"] != group:
                continue
            if current != r["plot_path"]:
                current = r["plot_path"]
                lines += ["", f"### {Path(current).stem}", "", link(current, "Plot") + " · " + link(r["report_path"], "Source report"), "",
                          f"Trials: {r['n_trials']:,}. Evaluated horizon: {r['horizon']:,}.", "",
                          "| Method | Coverage | Violating trials | Source run |", "|---|---:|---:|---|"]
            lines.append(f"| {r['method'].replace('|', '&#124;')} | {r['simultaneous_coverage']:.2%} | {r['violation_count']} | {r['run_id'] or 'unknown'} |")
            # Blank line is inserted before the next heading, not within a table.
        lines.append("")
    lines += ["## Figures without coverage records", ""]
    lines += [f"- {link(relative(p), p.stem)}" for p in missing] or ["None."]
    (plot_dir / "coverage_index.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} entries; {len(missing)} figures without coverage records")
    return rows, missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-dir", type=Path, default=ROOT / "data/plots")
    args = parser.parse_args()
    if not args.plot_dir.is_dir():
        raise ValueError(f"Plot directory does not exist: {args.plot_dir}")
    build(args.plot_dir.resolve())
