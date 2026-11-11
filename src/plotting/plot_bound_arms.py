#!/usr/bin/env python3
"""Compare any set of bound trajectories on identical bootstrap draws: mean bound (with a
CI band) against the oracle, plus simultaneous coverage. Same layout and palette as
plot_lambda_comparison.py, but the arms are given on the command line, e.g.

    grid vs fixed lambda     --arm "lambda grid, K=100=<grid npz>:bound__uniform_K100" \
                             --arm "Fixed lambda = 0.1=<extra npz>:bound__single_lambda_0p1" ...
    restart vs no restart    --arm "Restart mixture=<grid npz>:bound__uniform_K100" \
                             --arm "No restart=<extra npz>:bound__uniform_K100_norestart"

Each --arm is LABEL=NPZ:KEY. The oracle is read from the first arm's npz; every other npz
must carry an identical q_matrix (to 1e-6), so runs that did not share draws cannot be
silently compared.

Writes <prefix>.{png,pdf} and <prefix>_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from av_core import require_file  # noqa: E402
from av_evaluation_metrics import cumulative_coverage_rate  # noqa: E402

# Reference categorical palette, fixed order (as plot_lambda_comparison.py), then extras.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a4fbf", "#c2185b"]
STYLES = ["-", "--", ":", (0, (5, 1, 1, 1)), "-.", (0, (1, 1))]
Q_COLOR = "#3d3d3a"
REPORT_TIMES = (100, 300, 1000, 2000, 3000)
Q_TOLERANCE = 1e-6  # q is float32 in some stored runs


def parse_arm(text: str) -> tuple[str, Path, str]:
    if "=" not in text:
        raise ValueError(f"--arm must be LABEL=NPZ:KEY; got {text!r}")
    label, rest = text.rsplit("=", 1)
    if ":" not in rest:
        raise ValueError(f"--arm must be LABEL=NPZ:KEY; got {text!r}")
    path, key = rest.rsplit(":", 1)
    if not label or not path or not key:
        raise ValueError(f"--arm has an empty LABEL, NPZ or KEY: {text!r}")
    return label, Path(path), key


def load(path: Path, key: str) -> np.ndarray:
    require_file(path, str(path))
    data = np.load(path)
    if key not in data.files:
        raise ValueError(f"{path} is missing {key!r}; has {sorted(data.files)}")
    return data[key].astype(np.float64)


def band(matrix: np.ndarray, ci_level: float):
    half = (1.0 - ci_level) / 2.0
    return matrix.mean(axis=0), np.quantile(matrix, half, axis=0), np.quantile(matrix, 1.0 - half, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, help="LABEL=NPZ:KEY, repeatable")
    parser.add_argument("--title", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ci-level", type=float, required=True)
    parser.add_argument("--coverage-ymin", type=float, required=True,
                        help="coverage-panel floor; extended downward only if a curve would be clipped")
    args = parser.parse_args()

    arms = [parse_arm(a) for a in args.arm]
    if len(arms) > len(COLORS):
        raise ValueError(f"at most {len(COLORS)} arms; got {len(arms)}")
    if len({label for label, _p, _k in arms}) != len(arms):
        raise ValueError("arm labels must be unique")

    q = load(arms[0][1], "q_matrix")
    series = {}
    for label, path, key in arms:
        other_q = load(path, "q_matrix")
        if other_q.shape != q.shape:
            raise ValueError(f"{path} has q shape {other_q.shape}; first arm has {q.shape}")
        gap = float(np.max(np.abs(other_q - q)))
        if gap > Q_TOLERANCE:
            raise ValueError(f"{path} does not share bootstrap draws with {arms[0][1]} (max |q diff| = {gap})")
        series[label] = load(path, key)
        if series[label].shape != q.shape:
            raise ValueError(f"{path}:{key} has shape {series[label].shape}; q has {q.shape}")

    x = np.arange(1, q.shape[1] + 1)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[2.0, 1.0], sharex=True)

    for (label, _p, _k), color, style in zip(arms, COLORS, STYLES):
        mean, lo, hi = band(series[label], args.ci_level)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2.0, linestyle=style, label=label)
    q_mean, q_lo, q_hi = band(q, args.ci_level)
    ax.fill_between(x, q_lo, q_hi, color=Q_COLOR, alpha=0.10, linewidth=0)
    ax.plot(x, q_mean, color=Q_COLOR, linewidth=2.0, label=r"True Missing Mass, $R_t^{\mathrm{unseen}}$")
    ax.set_ylabel("Unseen failure probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_title(args.title, fontsize=12)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    coverage = {label: cumulative_coverage_rate(series[label], q) for label, _p, _k in arms}
    for (label, _p, _k), color, style in zip(arms, COLORS, STYLES):
        ax2.plot(x, coverage[label], color=color, linewidth=2.0, linestyle=style, label=label)
    ax2.axhline(0.95, color=Q_COLOR, linewidth=1.0, alpha=0.6)
    ax2.annotate("nominal $1-\\alpha$", xy=(x[-1], 0.95), xytext=(-6, 4), textcoords="offset points",
                 ha="right", fontsize=8, color=Q_COLOR)
    ax2.set_ylabel("Simultaneous coverage")
    ax2.set_xlabel("Number of sampled sessions, $t$")
    lowest = min(float(c.min()) for c in coverage.values())
    floor = min(args.coverage_ymin, lowest - 0.01)
    ax2.set_ylim(floor, 1.002)

    for a in (ax, ax2):
        a.set_xlim(1, x[-1])
        a.grid(True, alpha=0.2, linewidth=0.8)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.tight_layout()
    out = args.output_dir / args.prefix
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    report_times = [t for t in REPORT_TIMES if t <= q.shape[1]]
    summary = {
        "title": args.title, "n_trials": int(q.shape[0]), "horizon": int(q.shape[1]),
        "report_times": report_times,
        "oracle_mean": [float(q[:, t - 1].mean()) for t in report_times],
        "arms": {
            label: {"source": f"{path}:{key}",
                    "simultaneous_coverage": float(coverage[label][-1]),
                    "mean": [float(series[label][:, t - 1].mean()) for t in report_times]}
            for label, path, key in arms
        },
    }
    (args.output_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out}.png / .pdf and the summary json")


if __name__ == "__main__":
    main()
