#!/usr/bin/env python3
"""Render the two ablation figures from run_lambda_restart_ablation.py's npz.

  <label>_lambda_bounds   -- ablation A: the restart mixture at each lambda
  <label>_restart_bounds  -- ablation B: restart mixture vs no restart, at one lambda

Both plot the upper bound against the oracle R_t^unseen, same conventions as
plot_av_pipeline.py: no title (the paper's caption does that), serif type sized for
column width, solid black oracle drawn on top.

Color encodes what the series IS. Ablation A's arms are ORDERED (lambda 0.10 < 0.25 <
0.50), so they move through one blue-purple family, hue and lightness together: hue alone
inside that family cannot separate three lines (every candidate fell below the
normal-vision floor), and lightness alone was hard to read. Ablation B's two arms are
unordered, so they take two distinct hues.

Usage:
    uv run python src/plotting/plot_lambda_restart_ablation.py \
        --input-npz artifacts/reports/e_value_empirical_validation/lambda_restart_ablation.npz \
        --output-dir data/plots/lambda_restart_ablation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from av_core import require_file  # noqa: E402
from av_evaluation_metrics import trajectory_summary  # noqa: E402

# Short: the y-axis already says "Unseen failure probability", so the legend only needs
# the symbol. At the enlarged legend size the long form collided with the curves.
Q_LABEL = r"True Missing Mass, $R_t^{\mathrm{unseen}}$"
Q_COLOR = "black"

# Ablation A: pink-purple -> violet -> blue across lambda. Three distinct hues at similar
# lightness rather than one hue shaded, because a pure light-to-dark ramp in this family
# was hard to tell apart. Validated (dataviz validate_palette.js, light surface): all six
# checks pass, worst adjacent CVD dE 13.5, normal-vision dE 18.6.
LAMBDA_RAMP = {0.10: "#CC5D9E", 0.25: "#7B3FBF", 0.50: "#4E90D9"}

# Ablation B: two distinct hues, all six checks pass (worst CVD dE 8.6, normal 19.8).
RESTART_COLORS = {"restart_mixture": "#7B2D8E", "no_restart": "#1273B8"}
RESTART_LABELS = {"restart_mixture": "Restart mixture", "no_restart": "No restart"}

FIGSIZE = (9, 6.3)  # see plot_av_pipeline.py

# Sized for inclusion at roughly a third of source width -- see plot_av_pipeline.py.
FONT_BASE, FONT_AXIS_LABEL, FONT_TICK, FONT_LEGEND = 22, 26, 22, 22
ALPHA_BAND = 0.18


def band(matrix: np.ndarray, ci_level: float):
    half_tail = (1.0 - ci_level) / 2.0
    return trajectory_summary(matrix, lower_quantile=half_tail, upper_quantile=1.0 - half_tail)


def arm_key(name: str, lambda_value: float) -> str:
    return f"{name}__lambda_{lambda_value:g}".replace(".", "p")


def draw(ax, x, curves, q_mean, q_lower, q_upper, ci_level) -> None:
    """curves: list of (label, color, matrix). Oracle is drawn last, on top."""
    for label, color, matrix in curves:
        mean, lower, upper = band(matrix, ci_level)
        ax.plot(x, np.minimum(mean, 1.0), color=color, linewidth=3.0, label=label)
        ax.fill_between(x, np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0),
                        color=color, alpha=ALPHA_BAND, linewidth=0)

    ax.plot(x, q_mean, color=Q_COLOR, linewidth=2.8, label=Q_LABEL, zorder=10)
    ax.fill_between(x, np.clip(q_lower, 0.0, 1.0), np.clip(q_upper, 0.0, 1.0),
                    color=Q_COLOR, alpha=0.15, linewidth=0, zorder=9)

    ax.set_xlabel("Number of sampled sessions, $t$")
    ax.set_ylabel("Unseen failure probability")
    ax.set_xlim(0, float(x[-1]))
    ax.set_ylim(0.0, 1.0)
    # Gridlines every 0.1, but a label only every 0.2: the finer grid stays readable at
    # print scale without the tick column turning into a wall of numbers.
    ax.set_yticks(np.round(np.arange(0.0, 1.01, 0.2), 1))
    ax.set_yticks(np.round(np.arange(0.0, 1.01, 0.1), 1), minor=True)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(True, which="major", alpha=0.2, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.2, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ci-level", type=float, default=0.95)
    args = parser.parse_args()

    require_file(args.input_npz, "--input-npz")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FONT_BASE,
        "axes.labelsize": FONT_AXIS_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        # Push axis labels clear of the tick numbers; at the enlarged type they crowd.
        "axes.labelpad": 14,
    })

    data = np.load(args.input_npz, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"]))
    label_name = metadata["label_name"]
    lambda_values = metadata["lambda_values"]
    b_lambda = metadata["ablation_b_lambda"]

    q_matrix = data["q_matrix"]
    x = np.arange(1, q_matrix.shape[1] + 1)
    q_mean, q_lower, q_upper = band(q_matrix, args.ci_level)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for suffix, curves in [
        ("lambda_bounds", [
            (rf"Restart mixture, $\lambda$ = {lv:g}",
             LAMBDA_RAMP.get(lv, "#7B3FBF"),
             data[arm_key("restart_mixture", lv)])
            for lv in lambda_values
        ]),
        ("restart_bounds", [
            (RESTART_LABELS[name], RESTART_COLORS[name], data[arm_key(name, b_lambda)])
            for name in ("restart_mixture", "no_restart")
        ]),
    ]:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        draw(ax, x, curves, q_mean, q_lower, q_upper, args.ci_level)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = args.output_dir / f"{label_name}_{suffix}.{extension}"
            fig.savefig(path, dpi=200 if extension == "png" else None)
            print(f"Wrote {path}")
        plt.close(fig)
        from coverage_record import write_record
        keys = ([arm_key("restart_mixture", lv) for lv in lambda_values] if suffix == "lambda_bounds"
                else [arm_key(name, b_lambda) for name in ("restart_mixture", "no_restart")])
        write_record(args.output_dir / f"{label_name}_{suffix}.png", label_name, q_matrix,
                     [(label, args.input_npz, key, matrix, 1 - metadata["alpha"])
                      for (label, _color, matrix), key in zip(curves, keys)])


if __name__ == "__main__":
    main()
