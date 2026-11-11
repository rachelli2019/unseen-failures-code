#!/usr/bin/env python3
"""Per-setting comparison of the three lambda treatments, on identical bootstrap draws:

    fixed lambda = 0.25   one lambda chosen in advance
    lambda grid K = 100   e-values averaged over lambda_k = (k-1/2)/K, then inverted once
    closed-form lambda    per-restart hindsight-optimal lambda, inverted at
                          log(1/alpha) + log(t)
    best in hindsight     the same wealth inverted at log(1/alpha) with no discount --
      (cheating)          the tightest bound available if you pick lambda after seeing
                          the data. Not anytime-valid; shown as a reference floor.

Writes <prefix>_lambda_comparison.{png,pdf} (bounds vs the oracle, and simultaneous
coverage) plus <prefix>_lambda_comparison_summary.json.

All three arms must come from runs that share bootstrap draws; the script refuses to
plot unless their oracle q trajectories agree, so a mismatched pair of runs cannot be
silently compared.

Usage:
    uv run python src/plotting/plot_lambda_comparison.py \
        --grid-npz artifacts/lambda_grid_mixture/<run>/combined/trajectories.npz \
        --closed-form-npz artifacts/closed_form_mixture/<run>/task_000/trajectories.npz \
        --setting-label "Primary fixed taxonomy" \
        --output-dir data/plots/lambda_comparison --prefix primary_fixed
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

# Reference categorical palette, fixed order (validated with scripts/validate_palette.js).
ARMS = [
    ("fixed", "Fixed $\\lambda = 0.25$", "#eb6834", "--"),
    ("grid", "$\\lambda$ grid, $K=100$", "#2a78d6", "-"),
    ("closed", "Closed-form $\\lambda$", "#1baf7a", ":"),
    # Same hindsight wealth as "closed", read against log(1/alpha) with no log(t)
    # discount. Not anytime-valid -- the reference line for what looking at the data buys.
    ("cheating", "Best in hindsight (cheating)", "#eda100", (0, (5, 1, 1, 1))),
]
Q_COLOR = "#3d3d3a"
REPORT_TIMES = (100, 300, 1000, 2000, 3000)
Q_TOLERANCE = 1e-6  # q is float32 in some stored runs


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
    parser.add_argument("--grid-npz", required=True, type=Path)
    parser.add_argument("--closed-form-npz", required=True, type=Path)
    parser.add_argument("--grid-arm", default="uniform_K100")
    parser.add_argument("--fixed-arm", default="single_lambda_0p25",
                        help="arm name inside --grid-npz, or use --fixed-npz/--fixed-key instead")
    parser.add_argument("--fixed-npz", type=Path, default=None,
                        help="separate npz holding the fixed-lambda arm (e.g. the ablation run)")
    parser.add_argument("--fixed-key", default="restart_mixture__lambda_0p25")
    parser.add_argument("--setting-label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--coverage-ymin", type=float, default=0.9,
                        help="lower limit of the coverage panel; automatically extended if a "
                             "curve would otherwise be clipped")
    args = parser.parse_args()

    q = load(args.grid_npz, "q_matrix")
    series = {
        "grid": load(args.grid_npz, f"bound__{args.grid_arm}"),
        "closed": load(args.closed_form_npz, "bound__closed_form_lambda"),
        "cheating": load(args.closed_form_npz, "bound__closed_form_lambda_cheating"),
    }
    if args.fixed_npz is not None:
        series["fixed"] = load(args.fixed_npz, args.fixed_key)
        q_others = [("closed-form", load(args.closed_form_npz, "q_matrix")),
                    ("fixed-lambda", load(args.fixed_npz, "q_matrix"))]
    else:
        series["fixed"] = load(args.grid_npz, f"bound__{args.fixed_arm}")
        q_others = [("closed-form", load(args.closed_form_npz, "q_matrix"))]

    # Refuse to compare arms that did not see the same trials.
    for name, other in q_others:
        if other.shape != q.shape:
            raise ValueError(f"{name} run has q shape {other.shape}, grid run has {q.shape}")
        gap = float(np.max(np.abs(other - q)))
        if gap > Q_TOLERANCE:
            raise ValueError(f"{name} run does not share bootstrap draws with the grid run (max |q diff| = {gap})")

    x = np.arange(1, q.shape[1] + 1)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[2.0, 1.0], sharex=True)

    for key, label, color, style in ARMS:
        mean, lo, hi = band(series[key], args.ci_level)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2.0, linestyle=style, label=label)
    q_mean, q_lo, q_hi = band(q, args.ci_level)
    ax.fill_between(x, q_lo, q_hi, color=Q_COLOR, alpha=0.10, linewidth=0)
    ax.plot(x, q_mean, color=Q_COLOR, linewidth=2.0, label=r"True Missing Mass, $R_t^{\mathrm{unseen}}$")
    ax.set_ylabel("Unseen failure probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_title(f"{args.setting_label}: fixed vs. gridded vs. closed-form $\\lambda$", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    for key, label, color, style in ARMS:
        ax2.plot(x, cumulative_coverage_rate(series[key], q), color=color, linewidth=2.0, linestyle=style,
                 label=label)
    ax2.axhline(0.95, color=Q_COLOR, linewidth=1.0, alpha=0.6)
    ax2.annotate("nominal $1-\\alpha$", xy=(x[-1], 0.95), xytext=(-6, 4), textcoords="offset points",
                 ha="right", fontsize=8, color=Q_COLOR)
    ax2.set_ylabel("Simultaneous coverage")
    ax2.set_xlabel("Number of sampled sessions, $t$")
    # Zoomed: every curve lives in the top few percent, so a full [0, 1] range wastes the
    # panel. The floor drops below the requested one only when a curve would otherwise be
    # cut off -- a coverage line running off the bottom would hide exactly the failure this
    # panel exists to show.
    lowest = min(float(cumulative_coverage_rate(series[key], q).min()) for key, _l, _c, _s in ARMS)
    floor = min(args.coverage_ymin, lowest - 0.01)
    if floor < args.coverage_ymin:
        print(f"note: coverage axis extended down to {floor:.3f} -- an arm reaches "
              f"{lowest:.4f}, below the requested floor of {args.coverage_ymin:.3f}")
    ax2.set_ylim(floor, 1.002)

    for a in (ax, ax2):
        a.set_xlim(1, x[-1])
        a.grid(True, alpha=0.2, linewidth=0.8)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.tight_layout()
    out = args.output_dir / f"{args.prefix}_lambda_comparison"
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    summary = {
        "setting": args.setting_label, "prefix": args.prefix,
        "grid_npz": str(args.grid_npz), "closed_form_npz": str(args.closed_form_npz),
        "fixed_npz": str(args.fixed_npz) if args.fixed_npz else str(args.grid_npz),
        "n_trials": int(q.shape[0]), "horizon": int(q.shape[1]),
        "report_times": list(REPORT_TIMES),
        "oracle_mean": [float(q[:, t - 1].mean()) for t in REPORT_TIMES],
        "arms": {},
    }
    for key, label, _c, _s in ARMS:
        m = series[key]
        source, array_key = {
            "grid": (args.grid_npz, f"bound__{args.grid_arm}"),
            "fixed": (args.fixed_npz or args.grid_npz,
                      args.fixed_key if args.fixed_npz else f"bound__{args.fixed_arm}"),
            "closed": (args.closed_form_npz, "bound__closed_form_lambda"),
            "cheating": (args.closed_form_npz, "bound__closed_form_lambda_cheating"),
        }[key]
        summary["arms"][label] = {
            "source": f"{source}:{array_key}",
            "simultaneous_coverage": float(cumulative_coverage_rate(m, q)[-1]),
            "mean": [float(m[:, t - 1].mean()) for t in REPORT_TIMES],
            "ratio_to_grid": [float((m[:, t - 1] / series["grid"][:, t - 1]).mean()) for t in REPORT_TIMES],
        }
    (args.output_dir / f"{args.prefix}_lambda_comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out}.png / .pdf and the summary json")


if __name__ == "__main__":
    main()
