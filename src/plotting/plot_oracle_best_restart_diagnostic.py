#!/usr/bin/env python3
"""One plot: the restart-mixture baseline (every-time restart, log_squared
weights) vs. the oracle best-restart bound (see
oracle_best_restart_diagnostic.py), both vs. the oracle R_t^unseen. This is a
diagnostic -- how much is the mixture losing to hindsight -- not a candidate
method to rank alongside the others.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# these live in src/plotting/ but import their siblings from src/
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from av_core import require_file
from plot_av_pipeline import percentile_band

MIXTURE_LABEL = "Restart mixture (every-time, log_squared)"
MIXTURE_COLOR = "purple"
ORACLE_LABEL = "Oracle best restart (hindsight, not a valid bound)"
ORACLE_COLOR = "black"
# Short: the y-axis already says "Unseen failure probability", so the legend only needs
# the symbol. At the enlarged legend size the long form collided with the curves.
Q_LABEL = r"True Missing Mass, $R_t^{\mathrm{unseen}}$"
Q_COLOR = "#17807E"


def load_matrix(npz_path: Path, key: str) -> np.ndarray:
    require_file(npz_path, "--input-npz")
    data = np.load(npz_path)
    if key not in data.files:
        raise ValueError(f"{npz_path} is missing required array {key!r}; has {sorted(data.files)}")
    return data[key].astype(np.float64)


def plot_mixture_vs_oracle(input_npz: Path, output_plot: Path, ci_level: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_matrix = load_matrix(input_npz, "q_matrix")
    mixture_matrix = load_matrix(input_npz, "mixture_matrix")
    oracle_matrix = load_matrix(input_npz, "oracle_matrix")
    x = np.arange(1, q_matrix.shape[1] + 1)

    fig, (ax_bounds, ax_gap) = plt.subplots(2, 1, figsize=(9, 9), sharex=True, height_ratios=[1.4, 1.0])

    mixture_mean, mixture_lower, mixture_upper = percentile_band(mixture_matrix, ci_level)
    ax_bounds.plot(x, mixture_mean, color=MIXTURE_COLOR, linewidth=2.2, linestyle="-", label=MIXTURE_LABEL)
    ax_bounds.fill_between(x, np.clip(mixture_lower, 0.0, 1.0), np.clip(mixture_upper, 0.0, 1.0),
                            color=MIXTURE_COLOR, alpha=0.15, linewidth=0)

    oracle_mean, oracle_lower, oracle_upper = percentile_band(oracle_matrix, ci_level)
    ax_bounds.plot(x, oracle_mean, color=ORACLE_COLOR, linewidth=2.0, linestyle=(0, (4, 2)), label=ORACLE_LABEL)
    ax_bounds.fill_between(x, np.clip(oracle_lower, 0.0, 1.0), np.clip(oracle_upper, 0.0, 1.0),
                            color=ORACLE_COLOR, alpha=0.10, linewidth=0)

    q_mean, q_lower, q_upper = percentile_band(q_matrix, ci_level)
    ax_bounds.plot(x, q_mean, color=Q_COLOR, linewidth=2.0, linestyle="-", label=Q_LABEL, zorder=10)
    ax_bounds.fill_between(x, np.clip(q_lower, 0.0, 1.0), np.clip(q_upper, 0.0, 1.0), color=Q_COLOR, alpha=0.15,
                            linewidth=0, zorder=9)

    ax_bounds.set_ylabel("Unseen failure probability")
    ax_bounds.set_ylim(0.0, 1.0)
    ax_bounds.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_bounds.set_title("Restart mixture vs. oracle best restart (chosen with hindsight)")
    ax_bounds.grid(True, alpha=0.2, linewidth=0.8)
    ax_bounds.spines["top"].set_visible(False)
    ax_bounds.spines["right"].set_visible(False)
    ax_bounds.legend(loc="upper right", fontsize=8, frameon=False)

    price_of_not_knowing_r = mixture_matrix - oracle_matrix
    price_mean, price_lower, price_upper = percentile_band(price_of_not_knowing_r, ci_level)
    ax_gap.plot(x, price_mean, color=MIXTURE_COLOR, linewidth=2.0)
    ax_gap.fill_between(x, price_lower, price_upper, color=MIXTURE_COLOR, alpha=0.15, linewidth=0)
    ax_gap.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_gap.set_ylabel(r"$U_t^{\mathrm{mixture}} - U_t^{\mathrm{oracle}}$" + "\n(price of not knowing $r$)")
    ax_gap.set_xlabel("Number of sampled sessions, $t$")
    ax_gap.grid(True, alpha=0.2, linewidth=0.8)
    ax_gap.spines["top"].set_visible(False)
    ax_gap.spines["right"].set_visible(False)

    fig.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_plot}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-npz", required=True, type=Path,
                         help="artifacts/reports/e_value_empirical_validation/root_cause_oracle_best_restart.npz")
    parser.add_argument("--output-plot", required=True, type=Path)
    parser.add_argument("--ci-level", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_mixture_vs_oracle(args.input_npz, args.output_plot, args.ci_level)


if __name__ == "__main__":
    main()
