#!/usr/bin/env python3
"""Plot trajectory arrays produced by any of the three refactored av_* pipeline
scripts (run_primary_fixed.py, run_primary_online.py, run_set_valued_fixed.py).

Produces one figure per input (as both .png and .pdf -- the PDF is the
vector version for LaTeX), plus a table:
  <label_name>_bounds       -- bounds/UCBs vs. the true R_t^unseen
  <label_name>_coverage.md  -- simultaneous coverage at the final horizon

Figures carry no title by default: the paper's caption does that job, and label_name
is an internal identifier readers should never see. --title adds a centered one for
figures used outside the paper, where no caption travels with the image.

Coverage is reported rather than plotted: it is flat at 1.000 across the whole
horizon for every method on this data, so a figure of it is several overlapping
lines at the top of the frame that say less than the numbers do.

This script treats the classical-GT baseline as optional: run_set_valued_fixed.py's
npz genuinely has no GT arrays (the classical Good-Turing estimator has no
well-defined analogue for set-valued draws -- see
av_discovery_fixed_set_valued.py's docstring), so this script detects their
absence and renders the GT lines only when present, rather than raising or
silently faking zeros. Renders only, no resampling or recomputation --
everything plotted was already computed by the run_*.py script.

Line style/color encodes method class:
  solid, black  = the true R_t^unseen -- ground truth, not a method or baseline
  solid, purple = the main proposed method, "Anytime-Valid UCB (ours)": the every-time
                  restart mixture with log-squared weights, e-values averaged over
                  the K = 100 lambda grid (HEADLINE_KEY in av_lambda_restart_mixture)
  short-dashed, light gray = GT UCB (pointwise) -- valid only at one pre-chosen t;
                             shown for context, deliberately recessive
  solid, teal              = GT UCB (simultaneous) -- the actually anytime-valid
                             classical baseline, the real comparator for ours
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
from av_lambda_restart_mixture import HEADLINE_KEY
from av_evaluation_metrics import cumulative_coverage_rate, trajectory_summary

# Method-class line styles -- see module docstring.
STYLE_MAIN = "-"  # solid: the main proposed method

# The headline method's array. Runs made before it was the headline carry only the
# fixed-lambda restart bound; plot those with --ours-npz pointing at a lambda-grid run
# over the same draws.
RESTART_KEY = HEADLINE_KEY
RESTART_LABEL = "Anytime-Valid UCB (ours)"
RESTART_COLOR = "purple"

# Short: the y-axis already says "Unseen failure probability", so the legend only needs
# the symbol. At the enlarged legend size the long form collided with the curves.
Q_LABEL = r"True Missing Mass, $R_t^{\mathrm{unseen}}$"
Q_COLOR = "black"

# GT_SIMULTANEOUS is the classical bound that is actually anytime-valid, so it is the
# real comparator for our method and gets its own color (teal).
#
# GT_POINTWISE is valid only at one pre-chosen t, which is not the guarantee this work is
# about -- it is shown for context, not as a competitor. It is deliberately recessive:
# short-dashed, thin, and light grey, so the eye compares our method against the simultaneous
# bound rather than against this one. It keeps its confidence band, which carries most of
# its visibility at this weight.
STYLE_POINTWISE = (0, (4, 3))  # short-dashed: weakest guarantee, deliberately recessive
STYLE_SIMULTANEOUS = "-"       # solid, like ours: both are anytime-valid bounds
GT_POINTWISE_LABEL = "GT UCB (pointwise)"
GT_POINTWISE_COLOR = "0.72"  # light gray
GT_SIMULTANEOUS_LABEL = "GT UCB (simultaneous)"
GT_SIMULTANEOUS_COLOR = "#17807E"  # teal

# Confidence-band opacity, set per curve rather than derived from linestyle -- the
# simultaneous baseline is solid like ours, so linestyle no longer separates them.
ALPHA_MAIN = 0.20
ALPHA_BAND = 0.18
ALPHA_POINTWISE = 0.12  # lighter still: context line, should not draw the eye

# Typography. Applied via rcParams in plot_validation() so every axis, tick and legend
# picks them up, rather than being passed per call site and drifting apart.
#
# Sized for a figure included at roughly a THIRD of its source width -- two of these sit
# side by side as subpanels in a text-width float, so a 9in-wide source lands near 3.2in
# on the page. At that scale the old 15pt axis label rendered around 5pt. These values put
# axis labels near the caption's own size once shrunk.
# Aspect ratio, not just size: two panels sit side by side in a width-constrained float,
# so a long-and-short figure wastes the axis area it does get. Taller lets the curves and
# the legend breathe at the same page width.
FIGSIZE = (9, 6.3)

FONT_BASE = 22
FONT_AXIS_LABEL = 26
FONT_TICK = 22
FONT_LEGEND = 22


def percentile_band(matrix: np.ndarray, ci_level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """trajectory_summary wrapper taking a single ci_level (e.g. 0.95) instead
    of separate lower/upper quantiles, matching this script's CLI flag."""
    half_tail = (1.0 - ci_level) / 2.0
    return trajectory_summary(matrix, lower_quantile=half_tail, upper_quantile=1.0 - half_tail)


def slice_prefix(values: np.ndarray, prefix_length: int) -> np.ndarray:
    if prefix_length == 0:
        return values
    if prefix_length < 0:
        raise ValueError("--prefix-length must be 0 for full length or a positive integer")
    if prefix_length > values.shape[-1]:
        raise ValueError(f"--prefix-length cannot exceed trajectory length {values.shape[-1]}")
    return values[..., :prefix_length]


# Two GT-baseline key schemas exist across the three pipelines (each script's own
# naming), plus a genuine no-GT case (set-valued): normalize whichever is found to
# (gt_hat, single-t upper bound, Bonferroni-corrected bound) uniformly.
GT_KEY_SCHEMAS = (
    ("gt_hat_matrix", "gt_upper_bound_matrix", "gt_bonferroni_bound_matrix"),  # run_primary_fixed.py
    ("gt_hat_matrix", "gt_bound_uncorrected_matrix", "gt_bound_matrix"),  # run_primary_online.py
)


def detect_gt_schema(available_keys: set[str]) -> tuple[str, str, str] | None:
    """Return the (gt_hat, single_t_upper_bound, bonferroni_bound) key triple that
    matches this npz, or None for a genuine no-GT pipeline (set-valued). Raises if
    the file has a partial/mismatched mix of GT keys from neither known schema."""
    matches = [schema for schema in GT_KEY_SCHEMAS if all(key in available_keys for key in schema)]
    if len(matches) > 1:
        raise ValueError(f"--input-npz matches more than one known GT key schema: {matches}")
    if matches:
        return matches[0]
    any_gt_like = any(
        key in available_keys
        for schema in GT_KEY_SCHEMAS
        for key in schema
    )
    if any_gt_like:
        raise ValueError(
            "--input-npz has some but not a complete set of classical-GT arrays from any known "
            f"schema ({GT_KEY_SCHEMAS}) -- this indicates a malformed/partial npz, not a genuine "
            "no-GT pipeline"
        )
    return None


def load_trajectory_arrays(path: Path, require_headline: bool,
                           ) -> tuple[dict, dict[str, np.ndarray], tuple[str, str, str] | None]:
    require_file(path, "--input-npz")
    data = np.load(path)
    required_keys = {"metadata_json", "q_matrix"} | ({RESTART_KEY} if require_headline else set())
    missing = required_keys - set(data.files)
    if missing:
        raise ValueError(f"--input-npz is missing required arrays: {sorted(missing)}")

    gt_schema = detect_gt_schema(set(data.files))

    metadata = json.loads(str(data["metadata_json"]))
    arrays = {key: data[key] for key in data.files if key not in ("metadata_json", "lambda_names")}
    return metadata, arrays, gt_schema


def plot_bounds(ax, x, q_mean, q_lower, q_upper, ci_level, gt_matrices, restart_mean, restart_lower,
                 restart_upper, hide_gt: bool = False) -> None:
    if hide_gt:
        # Figure-only: the coverage table still reports every method the npz carries.
        gt_matrices = None

    if gt_matrices is not None:
        gt_ub_mean, gt_ub_lower, gt_ub_upper = percentile_band(gt_matrices["gt_upper_bound_matrix"], ci_level)
        gt_bonf_mean, gt_bonf_lower, gt_bonf_upper = percentile_band(gt_matrices["gt_bonferroni_bound_matrix"], ci_level)
        ax.plot(x, np.minimum(gt_ub_mean, 1.0), color=GT_POINTWISE_COLOR, linewidth=2.0, linestyle=STYLE_POINTWISE, label=GT_POINTWISE_LABEL)
        ax.fill_between(x, np.clip(gt_ub_lower, 0.0, 1.0), np.clip(gt_ub_upper, 0.0, 1.0), color=GT_POINTWISE_COLOR, alpha=ALPHA_POINTWISE, linewidth=0)
        ax.plot(x, np.minimum(gt_bonf_mean, 1.0), color=GT_SIMULTANEOUS_COLOR, linewidth=3.0, linestyle=STYLE_SIMULTANEOUS, label=GT_SIMULTANEOUS_LABEL)
        ax.fill_between(x, np.clip(gt_bonf_lower, 0.0, 1.0), np.clip(gt_bonf_upper, 0.0, 1.0), color=GT_SIMULTANEOUS_COLOR, alpha=ALPHA_BAND, linewidth=0)

    ax.plot(x, restart_mean, color=RESTART_COLOR, linewidth=3.2, linestyle=STYLE_MAIN, label=RESTART_LABEL)
    ax.fill_between(x, np.clip(restart_lower, 0.0, 1.0), np.clip(restart_upper, 0.0, 1.0), color=RESTART_COLOR, alpha=ALPHA_MAIN, linewidth=0)

    # Drawn last, at a higher zorder, so the ground-truth line stays visible on top of
    # every method's fill band rather than getting buried under them. The band carries no
    # label of its own (matching every other curve's fill_between above) so it collapses
    # into the line's single legend entry instead of adding a second row.
    ax.plot(x, q_mean, color=Q_COLOR, linewidth=2.8, linestyle=STYLE_MAIN, label=Q_LABEL, zorder=10)
    ax.fill_between(
        x, np.clip(q_lower, 0.0, 1.0), np.clip(q_upper, 0.0, 1.0), color=Q_COLOR, alpha=0.15, linewidth=0,
        zorder=9,
    )

    ax.set_ylabel("Unseen failure probability")
    # Tight to the data: no trailing margin past the final draw, so the axis ends on the
    # horizon (3000) rather than at matplotlib's padded 3150.
    ax.set_xlim(0, float(x[-1]))
    ax.set_xlabel("Number of sampled sessions, $t$")
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
    legend_order = [RESTART_LABEL, Q_LABEL, GT_SIMULTANEOUS_LABEL, GT_POINTWISE_LABEL]
    handles, labels = ax.get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles))
    ordered_labels = [label for label in legend_order if label in handles_by_label]
    ax.legend(
        [handles_by_label[label] for label in ordered_labels], ordered_labels,
        # Pinned rather than "best": placement should not silently move between renders.
        # Every curve descends, so the upper-left block below y=1.0 is free, and sitting
        # inside the axes keeps the frame tight to the 3000 horizon.
        loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False,
    )


NOMINAL_COVERAGE_COLOR = "0.45"


def plot_coverage(ax, x, q_matrix, gt_matrices, restart_matrix, alpha: float, ymin: float,
                  hide_gt: bool = False) -> None:
    """Empirical simultaneous coverage, Pr(bound >= R_t on all of [1, t]), per method: the
    companion to plot_bounds, same colors, line styles and weights. No legend -- it sits
    beside the bounds figure, which carries one."""
    if hide_gt:
        gt_matrices = None
    # Solid curves first, the dashed pointwise baseline last: both GT baselines typically sit
    # at exactly 1.0, and a dashed line drawn over a solid one keeps both visible.
    curves = []
    if gt_matrices is not None:
        curves.append((gt_matrices["gt_bonferroni_bound_matrix"], GT_SIMULTANEOUS_COLOR, STYLE_SIMULTANEOUS, 3.0))
    curves.append((restart_matrix, RESTART_COLOR, STYLE_MAIN, 3.2))
    if gt_matrices is not None:
        curves.append((gt_matrices["gt_upper_bound_matrix"], GT_POINTWISE_COLOR, STYLE_POINTWISE, 2.0))
    coverages = [cumulative_coverage_rate(matrix, q_matrix) for matrix, _c, _s, _w in curves]
    lowest = min(float(c.min()) for c in coverages)
    floor = ymin if lowest >= ymin + 0.005 else np.floor((lowest - 0.005) * 100) / 100
    if floor < ymin:
        print(f"note: coverage axis extended down to {floor:.2f} -- a method reaches {lowest:.4f}")
    for coverage, (_m, color, style, width) in zip(coverages, curves):
        ax.plot(x, coverage, color=color, linestyle=style, linewidth=width)
    nominal = 1.0 - alpha
    ax.axhline(nominal, color=NOMINAL_COVERAGE_COLOR, linewidth=1.6, linestyle=(0, (2, 2)), zorder=1)
    ax.annotate(r"nominal $1-\alpha$", xy=(float(x[-1]), nominal), xytext=(-8, 6), textcoords="offset points",
                ha="right", va="bottom", color=NOMINAL_COVERAGE_COLOR, fontsize=FONT_LEGEND)
    ax.set_ylabel("Simultaneous coverage")
    ax.set_xlabel("Number of sampled sessions, $t$")
    ax.set_xlim(0, float(x[-1]))
    ax.set_ylim(floor, 1.0 + (1.0 - floor) * 0.02)
    # A labeled tick every 0.01 while the axis spans at most 0.06 (e.g. 0.94-1.00, so the
    # nominal 0.95 gets its own labeled gridline); coarser only if it had to extend further.
    step = 0.01 if (1.0 - floor) <= 0.06 + 1e-9 else (0.02 if (1.0 - floor) <= 0.12 else 0.05)
    ax.set_yticks(np.round(np.arange(1.0, floor - 1e-9, -step)[::-1], 2))
    ax.grid(True, which="major", alpha=0.2, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def coverage_table(q_matrix, gt_matrices, restart_matrix, label_name) -> str:
    """Simultaneous coverage at the final horizon, as a markdown table.

    Reported rather than plotted: coverage is flat at 1.000 across the whole
    horizon for every method here, so a figure of it is three overlapping lines
    at the top of the frame and conveys less than the numbers do.
    """
    rows = []
    if gt_matrices is not None:
        rows.append((GT_POINTWISE_LABEL, gt_matrices["gt_upper_bound_matrix"]))
        rows.append((GT_SIMULTANEOUS_LABEL, gt_matrices["gt_bonferroni_bound_matrix"]))
    rows.append((RESTART_LABEL, restart_matrix))

    horizon = q_matrix.shape[1]
    n_trials = q_matrix.shape[0]
    lines = [
        f"# Simultaneous coverage at t = {horizon}: {label_name}",
        "",
        f"Pr(no violation in [1, t]) over {n_trials} bootstrap trials.",
        "",
        "| Method | Coverage at horizon | Minimum over t | Trials with a violation |",
        "|---|---|---|---|",
    ]
    for label, matrix in rows:
        coverage = cumulative_coverage_rate(matrix, q_matrix)
        violations = int(round((1.0 - float(coverage[-1])) * n_trials))
        lines.append(f"| {label} | {float(coverage[-1]):.4f} | "
                     f"{float(np.min(coverage)):.4f} | {violations} |")
    return "\n".join(lines) + "\n"


def save_figure(fig, output_dir: Path, stem: str) -> None:
    """Write a figure as both PNG (quick viewing) and PDF (vector, for LaTeX).

    The PDF is what belongs in a paper: it stays sharp at any zoom and embeds the
    text as text rather than pixels.
    """
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def substitute_ours(arrays: dict, ours_npz: Path, ours_key: str) -> None:
    """Replace the proposed-method array with one from a different run.

    Used to draw the headline figure against the lambda-grid mixture instead of the
    single-lambda one. The substitute must come from a run over the SAME bootstrap
    draws, or the figure would compare methods across different trials; the oracle
    trajectories are the check, and a mismatch raises rather than plots.
    """
    require_file(ours_npz, "--ours-npz")
    other = np.load(ours_npz)
    if ours_key not in other.files:
        raise ValueError(f"{ours_npz} is missing {ours_key!r}; has {sorted(other.files)}")
    if "q_matrix" not in other.files:
        raise ValueError(f"{ours_npz} has no q_matrix, so shared draws cannot be verified")
    q_here = np.asarray(arrays["q_matrix"], dtype=np.float64)
    q_there = np.asarray(other["q_matrix"], dtype=np.float64)
    if q_here.shape != q_there.shape:
        raise ValueError(f"{ours_npz} has q shape {q_there.shape}, input npz has {q_here.shape}")
    gap = float(np.max(np.abs(q_here - q_there)))
    if gap > 1e-6:
        raise ValueError(
            f"{ours_npz} does not share bootstrap draws with the input npz "
            f"(max |q diff| = {gap}); refusing to plot methods scored on different trials"
        )
    arrays[RESTART_KEY] = np.asarray(other[ours_key], dtype=np.float64)


def plot_validation(input_npz: Path, output_dir: Path, prefix_length: int, ci_level: float,
                     hide_gt: bool = False, ours_npz: Path | None = None,
                     ours_key: str | None = None, output_suffix: str = "",
                     title: str | None = None, coverage_ymin: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sized for a figure that gets scaled down to one column width in LaTeX: at 9x5.5in
    # rendered into ~3.5in of page, matplotlib's defaults (10pt base, 8pt legend) land
    # around 4pt on paper. These are the on-figure sizes, not the on-page sizes.
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

    metadata, arrays, gt_schema = load_trajectory_arrays(input_npz, require_headline=ours_npz is None)
    label_name = metadata["label_name"] + output_suffix
    if ours_npz is not None:
        if ours_key is None:
            raise ValueError("--ours-key is required when --ours-npz is given")
        substitute_ours(arrays, ours_npz, ours_key)

    q_matrix = slice_prefix(arrays["q_matrix"], prefix_length)
    restart_matrix = slice_prefix(arrays[RESTART_KEY], prefix_length)
    x = np.arange(1, q_matrix.shape[1] + 1)

    gt_matrices = None
    if gt_schema is not None:
        gt_hat_key, gt_upper_key, gt_bonf_key = gt_schema
        gt_matrices = {
            "gt_hat_matrix": slice_prefix(arrays[gt_hat_key], prefix_length),
            "gt_upper_bound_matrix": slice_prefix(arrays[gt_upper_key], prefix_length),
            "gt_bonferroni_bound_matrix": slice_prefix(arrays[gt_bonf_key], prefix_length),
        }

    q_mean, q_lower, q_upper = percentile_band(q_matrix, ci_level)
    restart_mean, restart_lower, restart_upper = percentile_band(restart_matrix, ci_level)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_bounds(ax, x, q_mean, q_lower, q_upper, ci_level, gt_matrices, restart_mean, restart_lower,
                restart_upper, hide_gt=hide_gt)
    if title:
        ax.set_title(title, loc="center", pad=16)
    fig.tight_layout()
    save_figure(fig, output_dir, f"{label_name}_bounds")
    plt.close(fig)

    if coverage_ymin is not None:
        alpha = metadata.get("alpha")
        if alpha is None:
            raise ValueError(f"{input_npz} metadata has no 'alpha'; the nominal coverage line needs it")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        plot_coverage(ax, x, q_matrix, gt_matrices, restart_matrix, float(alpha), coverage_ymin, hide_gt=hide_gt)
        fig.tight_layout()
        save_figure(fig, output_dir, f"{label_name}_coverage")
        plt.close(fig)


    table = coverage_table(q_matrix, gt_matrices, restart_matrix, label_name)
    coverage_path = output_dir / f"{label_name}_coverage.md"
    coverage_path.write_text(table, encoding="utf-8")
    print(f"Wrote {coverage_path}")
    from coverage_record import write_record
    recorded_arms = []
    if gt_schema is not None:
        recorded_arms.extend([
            (GT_POINTWISE_LABEL, input_npz, gt_schema[1], gt_matrices["gt_upper_bound_matrix"], ""),
            (GT_SIMULTANEOUS_LABEL, input_npz, gt_schema[2], gt_matrices["gt_bonferroni_bound_matrix"], ""),
        ])
    recorded_arms.append((RESTART_LABEL, ours_npz or input_npz,
                          ours_key if ours_npz else RESTART_KEY, restart_matrix, ""))
    write_record(output_dir / f"{label_name}_bounds.png", label_name, q_matrix, recorded_arms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="directory to write the two PNGs and the coverage table into")
    parser.add_argument("--prefix-length", type=int, default=0, help="0 = full trajectory")
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument(
        "--title", default=None,
        help="centered figure title. Omitted by default -- the paper's caption carries this.",
    )
    parser.add_argument(
        "--ours-npz", type=Path, default=None,
        help="take the proposed-method curve from this npz instead of --input-npz (e.g. the "
             "lambda-grid run). Must be over the same bootstrap draws; the oracle trajectories "
             "are checked and a mismatch raises. Baselines still come from --input-npz.",
    )
    parser.add_argument("--ours-key", default=None, help="array name inside --ours-npz")
    parser.add_argument(
        "--output-suffix", default="",
        help="appended to label_name in the output filenames, so a variant figure does not "
             "overwrite the original.",
    )
    parser.add_argument(
        "--hide-gt", action="store_true",
        help="omit the classical Good-Turing baselines from the FIGURE, leaving the proposed "
             "bound against the oracle. Used for the online cell, where the categorization rule "
             "has no species structure for Good-Turing to apply to. The coverage table is "
             "unaffected and still reports every method present in the npz.",
    )
    parser.add_argument(
        "--coverage-ymin", type=float, default=None,
        help="also write <label>_coverage.{png,pdf}: empirical simultaneous coverage per method, "
             "same styling and size as the bounds figure, y-axis from this floor to 1 (extended "
             "down only if a curve would be clipped). Omit to skip the coverage figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_validation(args.input_npz, args.output_dir, args.prefix_length, args.ci_level,
                    hide_gt=args.hide_gt, ours_npz=args.ours_npz, ours_key=args.ours_key,
                    output_suffix=args.output_suffix, title=args.title, coverage_ymin=args.coverage_ymin)


if __name__ == "__main__":
    main()
