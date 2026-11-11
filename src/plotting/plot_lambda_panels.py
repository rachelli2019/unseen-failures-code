#!/usr/bin/env python3
"""Two matching panels comparing the headline lambda grid against fixed lambda values, styled
as the headline figure (plot_av_pipeline.py: same figure size, typography, band opacity,
oracle styling, and the headline method's purple) so they can sit beside it or each other:

    <prefix>_bounds.{png,pdf}     mean bound with CI band per arm, against the oracle
    <prefix>_coverage.{png,pdf}   simultaneous coverage per arm, same colors, no legend
                                  (the bounds panel carries it)

The FIRST --arm is the headline method and is drawn in its headline style (solid, thick,
purple); the rest are comparison arms, dashed and drawn ON TOP of it so a near-identical
comparison curve still shows through (the dash gaps reveal the headline line beneath).
--comparison-colors names one palette color per comparison arm (see COMPARISON_COLORS).
Each --arm is LABEL=NPZ:KEY. The oracle is read from the first arm's npz; every other npz
must carry an identical q_matrix (to 1e-6), so runs that did not share draws cannot be
silently compared.

Usage:
    uv run python src/plotting/plot_lambda_panels.py \\
        --arm "lambda grid, K=100 (ours)=<grid npz>:bound__uniform_K100" \\
        --arm "Fixed lambda = 0.1=<extra npz>:bound__single_lambda_0p1" ... \\
        --output-dir data/plots/... --prefix primary_fixed_fullpop_lambda_panels \\
        --ci-level 0.95 --coverage-ymin 0.9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from av_evaluation_metrics import cumulative_coverage_rate  # noqa: E402
from plot_av_pipeline import (  # noqa: E402
    ALPHA_MAIN,
    FIGSIZE,
    FONT_AXIS_LABEL,
    FONT_BASE,
    FONT_LEGEND,
    FONT_TICK,
    Q_COLOR,
    Q_LABEL,
    RESTART_COLOR,
    percentile_band,
    save_figure,
)
from plot_bound_arms import load, parse_arm  # noqa: E402

HEADLINE_STYLE = {"color": RESTART_COLOR, "linewidth": 3.2, "linestyle": "-"}
# Comparison palette. orange / green / red sit outside the purple-blue family of the
# headline purple, so they separate from it and from each other at print size.
COMPARISON_COLORS = {
    "orange": "#E8730C", "green": "#2E9A48", "red": "#CC3340",
    "pink": "#D9559E", "violet": "#9474DB", "blue": "#4A8FE0",
}
COMPARISON_STYLE = {"linewidth": 2.4, "linestyle": (0, (5, 2.5))}
ALPHA_COMPARISON = 0.12
NOMINAL_COVERAGE_COLOR = "0.45"
Q_TOLERANCE = 1e-6


def apply_headline_typography(plt) -> None:
    # The same rcParams plot_av_pipeline.plot_validation sets, so text renders identically.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FONT_BASE,
        "axes.labelsize": FONT_AXIS_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "axes.labelpad": 14,
    })


def style_axes(ax, horizon: int) -> None:
    ax.set_xlim(0, float(horizon))
    ax.set_xlabel("Number of sampled sessions, $t$")
    ax.grid(True, which="major", alpha=0.2, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.2, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, help="LABEL=NPZ:KEY, repeatable; first is ours")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--comparison-colors", required=True,
                        help=f"comma-separated, one per comparison arm, from {sorted(COMPARISON_COLORS)}")
    parser.add_argument("--ci-level", type=float, required=True)
    parser.add_argument("--coverage-ymin", type=float, required=True,
                        help="coverage-panel floor; extended downward only if a curve would be clipped")
    args = parser.parse_args()

    arms = [parse_arm(a) for a in args.arm]
    if len(arms) < 2:
        raise ValueError("need the headline arm plus at least one comparison arm")
    color_names = args.comparison_colors.split(",")
    if len(color_names) != len(arms) - 1:
        raise ValueError(f"--comparison-colors has {len(color_names)} entries for {len(arms) - 1} comparison arms")
    unknown = [c for c in color_names if c not in COMPARISON_COLORS]
    if unknown:
        raise ValueError(f"unknown --comparison-colors {unknown}; choose from {sorted(COMPARISON_COLORS)}")
    if len(set(color_names)) != len(color_names):
        raise ValueError("--comparison-colors must not repeat a color")
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

    styles = {arms[0][0]: (HEADLINE_STYLE, ALPHA_MAIN)}
    for (label, _p, _k), name in zip(arms[1:], color_names):
        styles[label] = ({"color": COMPARISON_COLORS[name], **COMPARISON_STYLE}, ALPHA_COMPARISON)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_headline_typography(plt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizon = q.shape[1]
    x = np.arange(1, horizon + 1)

    # ---- bounds panel
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # Headline first; dashed comparison arms over it, so overlapping curves stay distinguishable.
    for label, _p, _k in arms:
        style, band_alpha = styles[label]
        mean, lower, upper = percentile_band(series[label], args.ci_level)
        ax.plot(x, np.minimum(mean, 1.0), label=label, **style)
        ax.fill_between(x, np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0), color=style["color"],
                        alpha=band_alpha, linewidth=0)
    q_mean, q_lower, q_upper = percentile_band(q, args.ci_level)
    ax.plot(x, q_mean, color=Q_COLOR, linewidth=2.8, linestyle="-", label=Q_LABEL, zorder=10)
    ax.fill_between(x, np.clip(q_lower, 0.0, 1.0), np.clip(q_upper, 0.0, 1.0), color=Q_COLOR, alpha=0.15,
                    linewidth=0, zorder=9)
    ax.set_ylabel("Unseen failure probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.round(np.arange(0.0, 1.01, 0.2), 1))
    ax.set_yticks(np.round(np.arange(0.0, 1.01, 0.1), 1), minor=True)
    ax.tick_params(axis="y", which="minor", length=0)
    style_axes(ax, horizon)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [label for label, _p, _k in arms] + [Q_LABEL]
    ax.legend([by_label[label] for label in order], order,
              loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False)
    fig.tight_layout()
    save_figure(fig, args.output_dir, f"{args.prefix}_bounds")
    plt.close(fig)

    # ---- coverage panel
    coverage = {label: cumulative_coverage_rate(series[label], q) for label, _p, _k in arms}
    lowest = min(float(c.min()) for c in coverage.values())
    floor = min(args.coverage_ymin, np.floor((lowest - 0.005) * 100) / 100)
    if floor < args.coverage_ymin:
        print(f"note: coverage axis extended down to {floor:.2f} -- an arm reaches {lowest:.4f}")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, _p, _k in arms:
        style, _alpha = styles[label]
        ax.plot(x, coverage[label], **style)
    ax.axhline(0.95, color=NOMINAL_COVERAGE_COLOR, linewidth=1.6, linestyle=(0, (2, 2)), zorder=1)
    ax.annotate(r"nominal $1-\alpha$", xy=(float(horizon), 0.95), xytext=(-8, 6), textcoords="offset points",
                ha="right", va="bottom", color=NOMINAL_COVERAGE_COLOR, fontsize=FONT_LEGEND)
    ax.set_ylabel("Simultaneous coverage")
    ax.set_ylim(floor, 1.0 + (1.0 - floor) * 0.02)
    # A labeled tick every 0.01 while the axis spans at most 0.06 (e.g. 0.94-1.00, so the
    # nominal 0.95 gets its own labeled gridline); coarser only if it had to extend further.
    step = 0.01 if (1.0 - floor) <= 0.06 + 1e-9 else (0.02 if (1.0 - floor) <= 0.12 else 0.05)
    ax.set_yticks(np.round(np.arange(1.0, floor - 1e-9, -step)[::-1], 2))
    style_axes(ax, horizon)
    fig.tight_layout()
    save_figure(fig, args.output_dir, f"{args.prefix}_coverage")
    plt.close(fig)

    for label, _p, _k in arms:
        print(f"  {label}: t={horizon} mean bound {series[label][:, -1].mean():.4f}, "
              f"simultaneous coverage {coverage[label][-1]:.3f}")
    from coverage_record import write_record
    for suffix in ("bounds", "coverage"):
        write_record(args.output_dir / f"{args.prefix}_{suffix}.png", args.prefix, q,
                     [(label, path, key, series[label], "") for label, path, key in arms])


if __name__ == "__main__":
    main()
