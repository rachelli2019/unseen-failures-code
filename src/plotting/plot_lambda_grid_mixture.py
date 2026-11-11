#!/usr/bin/env python3
"""Plots for the uniform-lambda-grid restart mixture (run_lambda_grid_mixture.py)
against single-lambda restart mixtures on the SAME bootstrap draws
(run_lambda_restart_ablation.py's npz, seed 888).

Checks that the two files really share draws (q trajectories must match) before
plotting anything. Writes:
    <prefix>_bounds.png   mean bound per arm (+ central band for the grid arm) vs oracle q
    <prefix>_ratio.png    per-trial ratio grid-arm / single-lambda arm, mean + central band
    <prefix>_summary.json the numbers behind the report table

Usage:
    uv run python src/plotting/plot_lambda_grid_mixture.py \
        --grid-npz artifacts/lambda_grid_mixture/<run_id>/combined/trajectories.npz \
        --grid-arm uniform_K100 \
        --ablation-npz artifacts/reports/e_value_empirical_validation/lambda_restart_ablation.npz \
        --output-dir data/plots/lambda_grid_mixture --prefix primary_fixed_K100
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

# Reference categorical palette, fixed order (validated: scripts/validate_palette.js).
GRID_COLOR = "#2a78d6"
SINGLE_COLORS = {"0p25": "#eb6834", "0p5": "#1baf7a"}
SINGLE_STYLES = {"0p25": "--", "0p5": ":"}
Q_COLOR = "#3d3d3a"
REPORT_TIMES = (100, 300, 1000, 2000, 3000)
SINGLE_LAMBDAS = ("0p1", "0p25", "0p5")
MATCH_TOLERANCE = 1e-6  # q is stored as float32 in the ablation npz


def load(npz_path: Path) -> np.lib.npyio.NpzFile:
    require_file(npz_path, str(npz_path))
    return np.load(npz_path)


def require_key(data, key: str, path: Path) -> np.ndarray:
    if key not in data.files:
        raise ValueError(f"{path} is missing {key!r}; has {sorted(data.files)}")
    return data[key].astype(np.float64)


def band(matrix: np.ndarray, ci_level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = (1.0 - ci_level) / 2.0
    return matrix.mean(axis=0), np.quantile(matrix, half, axis=0), np.quantile(matrix, 1.0 - half, axis=0)


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.2, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-npz", required=True, type=Path)
    parser.add_argument("--grid-arm", required=True)
    parser.add_argument("--ablation-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ci-level", type=float, default=0.95)
    args = parser.parse_args()

    grid = load(args.grid_npz)
    ablation = load(args.ablation_npz)
    grid_meta = json.loads(str(grid["metadata_json"]))
    arm_meta = [a for a in grid_meta["arms"] if a["name"] == args.grid_arm]
    if len(arm_meta) != 1:
        raise ValueError(f"arm {args.grid_arm!r} not in {[a['name'] for a in grid_meta['arms']]}")
    n_lambdas = len(arm_meta[0]["lambda_values"])

    q = require_key(grid, "q_matrix", args.grid_npz)
    q_ablation = require_key(ablation, "q_matrix", args.ablation_npz)
    if q.shape != q_ablation.shape:
        raise ValueError(f"shape mismatch: grid q {q.shape} vs ablation q {q_ablation.shape}")
    q_gap = float(np.max(np.abs(q - q_ablation)))
    if q_gap > MATCH_TOLERANCE:
        raise ValueError(f"grid and ablation runs do not share bootstrap draws (max |q diff| = {q_gap})")

    grid_bound = require_key(grid, f"bound__{args.grid_arm}", args.grid_npz)
    singles = {lam: require_key(ablation, f"restart_mixture__lambda_{lam}", args.ablation_npz)
               for lam in SINGLE_LAMBDAS}
    x = np.arange(1, q.shape[1] + 1)
    grid_label = f"Uniform λ grid, K = {n_lambdas}"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- bounds vs oracle ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    mean, lo, hi = band(grid_bound, args.ci_level)
    ax.fill_between(x, lo, hi, color=GRID_COLOR, alpha=0.15, linewidth=0)
    ax.plot(x, mean, color=GRID_COLOR, linewidth=2.0, label=grid_label)
    for lam in ("0p25", "0p5"):
        ax.plot(x, singles[lam].mean(axis=0), color=SINGLE_COLORS[lam], linewidth=2.0,
                linestyle=SINGLE_STYLES[lam], label=f"Single λ = {lam.replace('p', '.')}")
    q_mean, q_lo, q_hi = band(q, args.ci_level)
    ax.fill_between(x, q_lo, q_hi, color=Q_COLOR, alpha=0.10, linewidth=0)
    ax.plot(x, q_mean, color=Q_COLOR, linewidth=2.0, label=r"True Missing Mass, $R_t^{\mathrm{unseen}}$")
    ax.set_xlim(1, x[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.set_xlabel("Number of sampled sessions, $t$")
    ax.set_ylabel("Unseen failure probability")
    ax.set_title(f"Upper bounds (mean over trials; bands = central {args.ci_level:.0%})", fontsize=11)
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    bounds_path = args.output_dir / f"{args.prefix}_bounds.png"
    fig.savefig(bounds_path, dpi=150)
    fig.savefig(bounds_path.with_suffix(".pdf"))
    plt.close(fig)

    # --- per-trial ratio grid / single ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for lam in ("0p25", "0p5"):
        ratio = grid_bound / singles[lam]
        mean, lo, hi = band(ratio, args.ci_level)
        ax.fill_between(x, lo, hi, color=SINGLE_COLORS[lam], alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=SINGLE_COLORS[lam], linewidth=2.0, linestyle=SINGLE_STYLES[lam],
                label=f"K = {n_lambdas} ÷ single λ = {lam.replace('p', '.')}")
    ax.axhline(1.0, color=Q_COLOR, linewidth=1.0, alpha=0.6)
    ax.set_xlim(1, x[-1])
    ax.set_xlabel("Number of sampled sessions, $t$")
    ax.set_ylabel("Bound ratio (below 1 = grid tighter)")
    ax.set_title("Per-trial bound ratio, same bootstrap draws", fontsize=11)
    style_axes(ax)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    fig.tight_layout()
    ratio_path = args.output_dir / f"{args.prefix}_ratio.png"
    fig.savefig(ratio_path, dpi=150)
    fig.savefig(ratio_path.with_suffix(".pdf"))
    plt.close(fig)

    # --- numbers ---
    arms = {grid_label: grid_bound, **{f"Single λ = {lam.replace('p', '.')}": m for lam, m in singles.items()}}
    summary = {
        "grid_npz": str(args.grid_npz), "ablation_npz": str(args.ablation_npz), "grid_arm": args.grid_arm,
        "n_trials": int(q.shape[0]), "horizon": int(q.shape[1]), "max_abs_q_diff": q_gap,
        "report_times": list(REPORT_TIMES),
        "oracle_mean": [float(q[:, t - 1].mean()) for t in REPORT_TIMES],
        "arms": {},
    }
    for name, matrix in arms.items():
        source = (f"{args.grid_npz}:bound__{args.grid_arm}" if name == grid_label else
                  f"{args.ablation_npz}:restart_mixture__lambda_{name.split(' = ')[1].replace('.', 'p')}")
        summary["arms"][name] = {
            "source": source,
            "simultaneous_coverage": float(cumulative_coverage_rate(matrix, q)[-1]),
            "mean": [float(matrix[:, t - 1].mean()) for t in REPORT_TIMES],
            "median": [float(np.median(matrix[:, t - 1])) for t in REPORT_TIMES],
        }
    for lam in ("0p25", "0p5"):
        diff = grid_bound - singles[lam]
        summary[f"vs_single_{lam}"] = {
            "frac_trial_time_tighter": float((diff < -MATCH_TOLERANCE).mean()),
            "frac_trial_time_looser": float((diff > MATCH_TOLERANCE).mean()),
            "mean_ratio": [float((grid_bound[:, t - 1] / singles[lam][:, t - 1]).mean()) for t in REPORT_TIMES],
        }
    summary_path = args.output_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {bounds_path}, {ratio_path}, {summary_path}")


if __name__ == "__main__":
    main()
