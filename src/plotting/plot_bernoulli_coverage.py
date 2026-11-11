#!/usr/bin/env python3
"""Plot simultaneous coverage vs. t for the i.i.d. Bernoulli(p) simulation
produced by run_bernoulli_simulation.py.

One figure (PNG for viewing, PDF for LaTeX), no title -- the paper's caption
does that job, matching plot_av_pipeline.py.

Coverage IS the figure here, unlike the real-taxonomy pipelines where it is
reported as a table because it sits flat at 1.000. In this simulation novelty
never saturates, so the bound gets an unbounded number of chances to be
violated and the curve actually moves: it can only fall as t grows, and the
question the figure answers is whether it falls to the nominal 1 - alpha (the
bound is tight) or plateaus above it (the bound is conservative).

The band is a Wald standard-error band on the coverage proportion,
C_hat(t) +/- k sqrt(C_hat(t)(1 - C_hat(t))/n) with k = 1 by default
-- see standard_error_band.

Renders only -- everything plotted was already computed by
run_bernoulli_simulation.py.

Usage:
    uv run python src/plotting/plot_bernoulli_coverage.py \
        --input-npz artifacts/bernoulli/trajectories.npz \
        --output-dir data/plots/bernoulli
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# these live in src/plotting/ but import their siblings from src/
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from av_core import require_file
from av_evaluation_metrics import cumulative_coverage_rate

# Shared with plot_av_pipeline.py: purple is our method, and the nominal level is a
# reference line rather than a curve, so it is dashed and neutral.
UCB_LABEL = "Anytime-Valid UCB (ours)"
UCB_COLOR = "purple"
NOMINAL_COLOR = "black"
STYLE_NOMINAL = (0, (5, 3))
ALPHA_MAIN = 0.20  # band opacity, matching plot_av_pipeline.py

FIGSIZE = (9, 6.3)  # see plot_av_pipeline.py

# Sized for inclusion at roughly a third of source width -- see plot_av_pipeline.py.
FONT_BASE = 22
FONT_AXIS_LABEL = 26
FONT_TICK = 22
FONT_LEGEND = 22


def standard_error_band(
    coverage: np.ndarray, n_trials: int, se_multiplier: float
) -> tuple[np.ndarray, np.ndarray]:
    """Wald standard-error band for the coverage proportion.

    Writing T_i for trial i's first violation time (infinite if it never violates),
    coverage is C_hat(t) = (1/n) #{ i : T_i > t }. At fixed t the indicators
    1{T_i > t} are i.i.d. Bernoulli across trials, so n C_hat(t) ~ Binomial(n, C(t)):

        Var[C_hat(t)] = C(t) (1 - C(t)) / n
        SE_hat(t)     = sqrt( C_hat(t) (1 - C_hat(t)) / n )

    The band is C_hat(t) +/- se_multiplier * SE_hat(t), clipped to [0, 1]. The default
    multiplier of 1 gives a +/-1 SE band, which is roughly a 68% interval, NOT a 95%
    one -- at this coverage a 95% band (multiplier 1.96) crosses the nominal line,
    since the estimate sits 0.0027 above 0.95 against a 0.0042 half-width.

    Two properties of the Wald form to keep in mind. Where coverage is exactly 1.0 --
    here t < 319 -- the plug-in gives SE_hat = 0 and the band collapses to zero width,
    although the true uncertainty is not zero; a Wilson interval would give
    [1/(1 + z^2/n), 1] instead. And the band is pointwise: every t is computed from the
    same fixed set of trials, so neighbouring points are strongly dependent and the
    band is not a simultaneous confidence region for the trajectory.
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive; got {n_trials!r}")
    if se_multiplier <= 0.0:
        raise ValueError(f"--se-multiplier must be positive; got {se_multiplier!r}")
    if np.any(coverage < 0.0) or np.any(coverage > 1.0):
        raise ValueError("coverage must lie in [0, 1]")

    half_width = se_multiplier * np.sqrt(coverage * (1.0 - coverage) / n_trials)
    return np.clip(coverage - half_width, 0.0, 1.0), np.clip(coverage + half_width, 0.0, 1.0)


def load_coverage_from_bound(path: Path, bound_key: str) -> tuple[np.ndarray, dict]:
    """Coverage computed from a lambda-mixture run's bound and oracle matrices.

    run_lambda_grid_mixture.py stores per-trial bounds and q rather than the collapsed
    `coverage` vector run_bernoulli_simulation.py writes, so the simultaneous coverage is
    computed here from the same definition. Settings come from the npz's own metadata,
    not a sidecar, so the figure cannot be captioned with another run's parameters.
    """
    require_file(path, "--input-npz")
    data = np.load(path)
    for key in (bound_key, "q_matrix", "metadata_json"):
        if key not in data.files:
            raise ValueError(f"{path} is missing {key!r}; has {sorted(data.files)}")
    bound = np.asarray(data[bound_key], dtype=np.float64)
    q = np.asarray(data["q_matrix"], dtype=np.float64)
    if bound.shape != q.shape:
        raise ValueError(f"{bound_key} has shape {bound.shape}, q_matrix has {q.shape}")

    metadata = json.loads(str(data["metadata_json"]))
    discovery = metadata.get("discovery", {})
    if discovery.get("backend") != "bernoulli":
        raise ValueError(
            f"this figure is the i.i.d. Bernoulli stress test; {path} has backend "
            f"{discovery.get('backend')!r}"
        )
    p_value = discovery["p"]
    if not np.allclose(q, p_value):
        raise ValueError(f"q_matrix is not constant at p = {p_value}")

    coverage = cumulative_coverage_rate(bound, q)
    summary = {"alpha": metadata["alpha"], "p": p_value,
               "n_trials": metadata["n_trials"], "horizon": metadata["n_bootstrap_samples"],
               "bound_key": bound_key}
    if summary["horizon"] != coverage.size:
        raise ValueError(f"horizon {summary['horizon']} != coverage length {coverage.size}")
    return coverage, summary


def load_coverage(path: Path) -> tuple[np.ndarray, dict]:
    require_file(path, "--input-npz")
    data = np.load(path)
    if "coverage" not in data.files:
        raise ValueError(f"--input-npz is missing the 'coverage' array: {path}")
    coverage = np.asarray(data["coverage"], dtype=np.float64)
    if coverage.ndim != 1 or coverage.size == 0:
        raise ValueError(f"'coverage' must be a non-empty 1D array; got shape {coverage.shape}")

    # run_bernoulli_simulation.py writes q_t as a constant array; the summary JSON beside
    # the npz carries alpha. Read alpha from the sidecar rather than assuming 0.05.
    summary_path = path.parent / "summary.json"
    require_file(summary_path, "summary.json beside --input-npz")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("alpha", "p", "n_trials", "horizon"):
        if key not in summary:
            raise ValueError(f"{summary_path} is missing required key {key!r}")
    if summary["horizon"] != coverage.size:
        raise ValueError(
            f"summary horizon {summary['horizon']} does not match coverage length {coverage.size}"
        )
    return coverage, summary


def plot_coverage(
    coverage: np.ndarray,
    summary: dict,
    output_dir: Path,
    stem: str,
    se_multiplier: float,
    title: str | None = None,
) -> None:
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

    nominal = 1.0 - summary["alpha"]
    t = np.arange(1, coverage.size + 1)

    lower, upper = standard_error_band(coverage, int(summary["n_trials"]), se_multiplier)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        t, coverage, color=UCB_COLOR, linewidth=3.2, linestyle="-",
        label=UCB_LABEL, zorder=3,
    )
    # Carries no label of its own, so it collapses into the line's legend entry rather
    # than adding a second row -- same convention as plot_av_pipeline.py.
    ax.fill_between(t, lower, upper, color=UCB_COLOR, alpha=ALPHA_MAIN, linewidth=0, zorder=2)
    ax.axhline(
        nominal, color=NOMINAL_COLOR, linewidth=2.6, linestyle=STYLE_NOMINAL,
        label=f"Coverage guarantee, $1-\\alpha = {nominal:g}$", zorder=2,
    )

    ax.set_xlabel("Number of sampled sessions, $t$")
    # Shortened from "Empirical simultaneous coverage": at the enlarged type the longer
    # string is taller than the axes and gets clipped. The caption says "empirical".
    ax.set_ylabel("Simultaneous coverage")
    # Tight to the data, matching plot_av_pipeline.py: the axis ends on the horizon.
    ax.set_xlim(0, float(t[-1]))
    # Frame the band between the nominal floor and 1: that gap is the whole subject of
    # the figure, and autoscaling to the curve alone would hide how much slack is left.
    bottom = min(nominal, float(lower.min())) - 0.005
    ax.set_ylim(bottom, 1.005)
    ax.set_yticks(np.round(np.arange(np.ceil(bottom * 100) / 100, 1.001, 0.01), 2))
    ax.grid(True, alpha=0.2, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # The curve descends from the top-left, so the upper-right block is the free corner
    # (the lower-left one is occupied by the nominal line). Pinned rather than "best" so
    # placement cannot drift between renders, matching plot_av_pipeline.py.
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False)

    if title:
        ax.set_title(title, loc="center", pad=16)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stem", default="bernoulli_coverage", help="output filename stem")
    parser.add_argument("--bound-key", default=None,
                        help="read this bound array plus q_matrix from a lambda-mixture run "
                             "and compute coverage from them, instead of expecting a "
                             "precomputed 'coverage' array and a sidecar summary.json.")
    parser.add_argument("--title", default=None,
                        help="centered figure title. Omitted by default.")
    parser.add_argument("--se-multiplier", type=float, default=1.0,
                        help="band half-width in standard errors (1 = +/-1 SE, ~68%%)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bound_key:
        coverage, summary = load_coverage_from_bound(args.input_npz, args.bound_key)
    else:
        coverage, summary = load_coverage(args.input_npz)
    plot_coverage(coverage, summary, args.output_dir, args.stem, args.se_multiplier, args.title)
    from build_coverage_index import relative, validate
    n, horizon = int(summary["n_trials"]), int(summary["horizon"])
    violations = validate(n, horizon, float(coverage[-1]))
    record = dict(plot_file=f"{args.stem}.png", setting=f"Bernoulli(p={summary['p']:g})",
                  n_trials=n, horizon=horizon, provenance="recorded during plotting",
                  arms=[dict(method=UCB_LABEL, source_npz=relative(args.input_npz), array_key="coverage",
                             coverage=float(coverage[-1]), violation_count=violations,
                             target_coverage=1 - summary["alpha"])])
    (args.output_dir / f"{args.stem}_coverage_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"Bernoulli(p={summary['p']:g}), {summary['n_trials']} trials, horizon {summary['horizon']}: "
        f"coverage {float(coverage[-1]):.4f} vs nominal {1.0 - summary['alpha']:g}"
    )


if __name__ == "__main__":
    main()
