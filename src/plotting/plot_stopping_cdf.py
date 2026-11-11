"""Plot the empirical stopping-time CDF

    F_hat_epsilon(n) = (1/R) * sum_r 1{tau_r(epsilon) <= n}

for every method in a bootstrap trajectory npz, across a fixed set of
stopping epsilons -- the fraction of trials whose bound/estimate has ever
dropped below epsilon by draw n. This is the most directly interpretable
read of stopping-time behavior: "by n samples, what fraction of deployments
could certify q_n < epsilon?"

Equivalent to 1 minus a Kaplan-Meier survival curve, but since every trial
shares the same deterministic censoring horizon N (--n-bootstrap-samples),
the plain empirical CDF is sufficient -- no need for KM's right-censoring
machinery, which exists for the case where the censoring time itself varies
or is random.

Includes the classical GT point estimate even though it carries no validity
guarantee to stop on (its own docstring elsewhere notes this) -- shown here
purely as a diagnostic contrast against the valid bounds/UCBs, exactly as it
already appears in the probability/coverage panels.

Auto-detects online vs. global schema from which classical-GT matrix keys
are present in the npz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from av_lambda_restart_mixture import HEADLINE_KEY  # noqa: E402

STOPPING_EPSILONS = [0.30, 0.20, 0.15]

# Same color-family convention as the other plots: warm for classical-GT,
# cool blue for fixed-lambda e-value UCBs, cool purple for restart-mixture UCBs.
LAMBDA_COLORS = ["tab:blue", "steelblue", "cornflowerblue", "dodgerblue", "royalblue", "navy"]


def require_file(path: Path, argument_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{argument_name} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{argument_name} is not a file: {path}")


def stopping_cdf(bound_matrix: np.ndarray, epsilon: float) -> np.ndarray:
    """F_hat_epsilon(n) for n=1..N: fraction of trials whose bound/estimate
    has dropped below epsilon by draw n at least once -- "ever certified by
    n", not "currently below n" (a bound that dips below epsilon and later
    rises back above it still counts as stopped from that point on)."""
    ever_below = np.cumsum(bound_matrix < epsilon, axis=1) > 0
    return ever_below.mean(axis=0, dtype=np.float64)


def load_methods(data, lambda_names: list[str]) -> list[tuple[str, str, str, str]]:
    """Returns (matrix_key, label, color, linestyle) for every method present
    in the npz, in a fixed legend order. Auto-detects online (gt_bound_matrix)
    vs. global (gt_upper_bound_matrix) schema."""
    methods = [("gt_hat_matrix", "Classical GT point estimate (not a bound)", "tab:orange", "-")]

    if "gt_bound_matrix" in data.files:  # online schema
        methods.append(("gt_bound_matrix", "Classical GT, Bonferroni-corrected", "darkred", "-"))
        methods.append(("gt_bound_uncorrected_matrix", "Classical GT, pointwise-only", "tab:red", "-"))
    else:  # global schema
        methods.append(("gt_bonferroni_bound_matrix", "Classical GT, Bonferroni-corrected", "darkred", "-"))
        methods.append(("gt_upper_bound_matrix", "Classical GT, upper-bound-only", "tab:red", "-"))

    for i, lambda_name in enumerate(lambda_names):
        prefix = lambda_name.replace(".", "p").replace("-", "_")
        if i >= len(LAMBDA_COLORS):
            raise ValueError(f"need a color for lambda index {i}; LAMBDA_COLORS only has {len(LAMBDA_COLORS)}")
        methods.append((f"{prefix}__ucb_matrix", f"e-value UCB: {lambda_name}", LAMBDA_COLORS[i], "-"))

    methods.append(
        ("restart_mixture_periodic_geometric_matrix", "Restart-mixture UCB (periodic, geometric)", "tab:purple", "-.")
    )
    methods.append(
        ("restart_mixture_everytime_heavytail_matrix", "Restart-mixture UCB (every-time, heavy-tailed, lambda=0.25)",
         "darkviolet", "-.")
    )
    methods.append(
        (HEADLINE_KEY, "Anytime-Valid UCB (ours): every-time restart, lambda grid K=100", "purple", "-")
    )
    methods.append(
        ("restart_mixture_discovery_heavytail_matrix", "Restart-mixture UCB (discovery, heavy-tailed)", "indigo", "-.")
    )

    return [(key, label, color, linestyle) for key, label, color, linestyle in methods if key in data.files]


def plot_stopping_cdf(input_npz: Path, output_plot: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require_file(input_npz, "--input-npz")
    data = np.load(input_npz)
    metadata = json.loads(str(data["metadata_json"]))
    label_name = metadata["label_name"]
    lambda_names = [str(name) for name in data["lambda_names"]]

    methods = load_methods(data, lambda_names)
    if not methods:
        raise ValueError(f"{input_npz} has no recognized bound/estimate matrices")

    n = data[methods[0][0]].shape[1]
    x = np.arange(1, n + 1)

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(STOPPING_EPSILONS), figsize=(6.2 * len(STOPPING_EPSILONS), 5), sharey=True)

    for ax, epsilon in zip(axes, STOPPING_EPSILONS, strict=True):
        for key, label, color, linestyle in methods:
            cdf = stopping_cdf(data[key], epsilon)
            ax.step(x, cdf, where="post", color=color, linestyle=linestyle, linewidth=1.4, label=label)
        ax.set_title(f"$\\varepsilon={epsilon:g}$")
        ax.set_xlabel("bootstrap draw n")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$\hat{F}_\varepsilon(n)$ = fraction of runs stopped by $n$")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(f"Stopping-time CDF: {label_name}")
    fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.96))
    fig.savefig(output_plot, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--output-plot", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_stopping_cdf(args.input_npz, args.output_plot)


if __name__ == "__main__":
    main()
