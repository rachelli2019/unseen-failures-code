#!/usr/bin/env python3
"""How much later than the oracle does the anytime-valid bound certify "missing mass
below epsilon"?

For each threshold epsilon and each trial:

    tau_true(eps) = inf{ t : R_t^unseen < eps }     the oracle stopping time
    tau_ours(eps) = inf{ t : U_t        < eps }     when our bound can certify it
    delay(eps)    = tau_ours - tau_true

and the figure plots delay against epsilon.

--aggregate chooses what "the" stopping time means:
  mean-trajectory  average the trajectories over trials first, then read off the single t
                   at which each mean curve crosses eps. One number per eps, no band.
  per-trial        cross first, then summarise across trials: solid line is the mean,
                   shaded band the 2.5th-97.5th percentiles (at the default --ci-level 0.95).
These are NOT the same: crossing is a nonlinear functional of the path, so the crossing of
the mean is not the mean of the crossings.

First crossing, not last: U_t is NOT monotone in t -- on the primary fixed full-population
run about 9% of steps increase -- so "the last time the bound was above eps" would be a
different and much later quantity. inf{t : U_t < eps} is the time a practitioner watching
the bound would actually stop, which is what this figure is about.

Censoring is refused, not averaged away. Below some epsilon the bound never crosses within
the horizon (it bottoms out around 0.10 on the primary fixed run), so tau_ours does not
exist for some trials. Averaging over only the trials that did cross would report a delay
biased towards the fast trials and look like a much better result than it is. This script
raises instead, and names the smallest epsilon at which every trial crosses.

Usage:
    uv run python src/plotting/plot_stopping_time_delay.py \
        --input-npz artifacts/lambda_grid_mixture/<run>/combined/trajectories.npz \
        --bound-key bound__uniform_K100 \
        --eps-min 0.12 --eps-max 0.50 \
        --output-dir data/plots/full_population/primary_oneshot_fixed \
        --prefix primary_fixed_fullpop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from av_core import require_file  # noqa: E402

# The delay is a property of the method, so it wears the method colour used everywhere else.
DELAY_COLOR = "purple"
REFERENCE_COLOR = "#3d3d3a"
# Shared with plot_av_pipeline.py so the method keeps one colour across every figure.
# Figure size and type scale copied from plot_av_pipeline.py / plot_bernoulli_coverage.py
# so these sit beside the bound and coverage figures at the same scale on the page.
FIGSIZE = (9, 6.3)
FONT_BASE, FONT_AXIS_LABEL, FONT_TICK, FONT_LEGEND = 22, 26, 22, 22
OURS_COLOR = "purple"
OURS_LABEL = "Anytime-Valid UCB (Ours)"
TRUTH_COLOR = "black"
TRUTH_LABEL = "Oracle Stopping Time"


def first_crossing(matrix: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """(1-indexed first t with value < epsilon, mask of trials that ever cross)."""
    crossed = matrix < epsilon
    ever = np.any(crossed, axis=1)
    return np.argmax(crossed, axis=1) + 1, ever


def delays_at(q: np.ndarray, bound: np.ndarray, epsilon: float) -> np.ndarray:
    tau_true, true_ever = first_crossing(q, epsilon)
    tau_ours, ours_ever = first_crossing(bound, epsilon)
    censored = ~(true_ever & ours_ever)
    if np.any(censored):
        raise ValueError(
            f"epsilon = {epsilon:g}: {int(censored.sum())} of {len(censored)} trials never "
            f"cross within the horizon ({int((~ours_ever).sum())} for the bound, "
            f"{int((~true_ever).sum())} for the truth). Raise --eps-min: averaging over only "
            f"the trials that crossed would understate the delay."
        )
    return (tau_ours - tau_true).astype(np.float64)


def smallest_uncensored_epsilon(q: np.ndarray, bound: np.ndarray, aggregate: str) -> float:
    """Below this epsilon the relevant series never crosses within the horizon.

    per-trial: the largest final value over trials, since ANY uncrossed trial is censoring.
    mean-trajectory: the final value of the mean curves, which is all that is read.
    """
    if aggregate == "per_trial":
        return float(max(q[:, -1].max(), bound[:, -1].max()))
    return float(max(q.mean(axis=0)[-1], bound.mean(axis=0)[-1]))


def mean_trajectory_times(q: np.ndarray, bound: np.ndarray, epsilon: float) -> tuple[float, float]:
    """(tau_true, tau_ours) read off the mean curves."""
    q_mean = q.mean(axis=0)[None, :]
    bound_mean = bound.mean(axis=0)[None, :]
    tau_true, true_ever = first_crossing(q_mean, epsilon)
    tau_ours, ours_ever = first_crossing(bound_mean, epsilon)
    if not (true_ever[0] and ours_ever[0]):
        raise ValueError(
            f"epsilon = {epsilon:g}: the mean "
            f"{'bound' if not ours_ever[0] else 'truth'} never crosses within the horizon"
        )
    return float(tau_true[0]), float(tau_ours[0])


def mean_trajectory_delay(q: np.ndarray, bound: np.ndarray, epsilon: float) -> float:
    """Crossing of the mean curves: average over trials first, then cross."""
    q_mean = q.mean(axis=0)[None, :]
    bound_mean = bound.mean(axis=0)[None, :]
    tau_true, true_ever = first_crossing(q_mean, epsilon)
    tau_ours, ours_ever = first_crossing(bound_mean, epsilon)
    if not (true_ever[0] and ours_ever[0]):
        raise ValueError(
            f"epsilon = {epsilon:g}: the mean "
            f"{'bound' if not ours_ever[0] else 'truth'} never crosses within the horizon"
        )
    return float(tau_ours[0] - tau_true[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--bound-key", required=True)
    parser.add_argument("--eps-min", type=float, default=None,
                        help="default: just above max_trials R_T, the smallest eps the oracle "
                             "still reaches within the horizon.")
    parser.add_argument("--eps-max", type=float, default=None,
                        help="default: just below min_trials R_1, the largest eps at which the "
                             "oracle still needs more than one draw.")
    parser.add_argument("--n-eps", type=int, default=40)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--aggregate", choices=("mean_trajectory", "per_trial"),
                        default="mean_trajectory",
                        help="mean_trajectory: cross the mean curves (one value per epsilon). "
                             "per_trial: cross each trial, then summarise the delays.")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()


    if args.n_eps < 2:
        raise ValueError("--n-eps must be at least 2")

    require_file(args.input_npz, "--input-npz")
    data = np.load(args.input_npz)
    for key in (args.bound_key, "q_matrix"):
        if key not in data.files:
            raise ValueError(f"{args.input_npz} is missing {key!r}; has {sorted(data.files)}")
    q = data["q_matrix"].astype(np.float64)
    bound = data[args.bound_key].astype(np.float64)
    if q.shape != bound.shape:
        raise ValueError(f"q_matrix {q.shape} and {args.bound_key} {bound.shape} disagree")

    # Default x-range is the ORACLE's own usable window: epsilons at which the true missing
    # mass both (a) still needs more than one draw, and (b) is reached within the horizon.
    # Where the bound is additionally censored, its curve is masked and the span shaded --
    # that stretch is a result (the oracle certifies, ours does not), not missing data.
    oracle_lo, oracle_hi = float(q[:, -1].max()), float(q[:, 0].min())
    eps_min = args.eps_min if args.eps_min is not None else oracle_lo * 1.005
    eps_max = args.eps_max if args.eps_max is not None else oracle_hi * 0.999
    if not 0.0 < eps_min < eps_max < 1.0:
        raise ValueError(f"need 0 < eps-min < eps-max < 1; got {eps_min}, {eps_max}")
    if eps_min <= oracle_lo:
        raise ValueError(f"--eps-min {eps_min:g} is at or below {oracle_lo:.4f}; the oracle "
                         f"does not reach it within the horizon in every trial")
    if eps_max >= oracle_hi:
        raise ValueError(f"--eps-max {eps_max:g} is at or above {oracle_hi:.4f}; the oracle "
                         f"already satisfies it at t = 1 in some trial")

    floor = smallest_uncensored_epsilon(q, bound, args.aggregate)
    # Both figures share one range: epsilons where the oracle needs more than one draw, reaches
    # the threshold within the horizon, AND the bound certifies in every trial. Outside it one of
    # the two stopping times does not exist, so there is nothing honest to draw.
    delay_min = max(eps_min, floor * 1.005)
    if delay_min >= eps_max:
        raise ValueError(
            f"the bound only certifies above {floor:.4f}, which is outside the oracle window "
            f"({eps_min:.4f}, {eps_max:.4f}) -- no epsilon has a defined delay"
        )
    epsilons = np.linspace(delay_min, eps_max, args.n_eps)
    half = (1.0 - args.ci_level) / 2.0
    if args.aggregate == "mean_trajectory":
        curve = np.asarray([mean_trajectory_delay(q, bound, float(e)) for e in epsilons])
        lower = upper = None
    else:
        centre, lo, hi = [], [], []
        for eps in epsilons:
            _, ours_ok = first_crossing(bound, float(eps))
            if not ours_ok.all():   # no honest delay exists; the curve stops here
                centre.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            delay = delays_at(q, bound, float(eps))
            centre.append(delay.mean())
            lo.append(np.quantile(delay, half)); hi.append(np.quantile(delay, 1.0 - half))
        curve, lower, upper = np.asarray(centre), np.asarray(lo), np.asarray(hi)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FONT_BASE, "axes.labelsize": FONT_AXIS_LABEL,
        "xtick.labelsize": FONT_TICK, "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND, "axes.labelpad": 14,
    })
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # No legend: one series, and the band is described in the caption rather than a key.
    if lower is not None:
        ax.fill_between(epsilons, lower, upper, color=DELAY_COLOR, alpha=0.15, linewidth=0)
    ax.plot(epsilons, curve, color=DELAY_COLOR, linewidth=2.2)
    ax.axhline(0.0, color=REFERENCE_COLOR, linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"Stopping Threshold, $\epsilon$")
    # Short, like the bound and coverage figures' labels: at the shared 26pt axis size the
    # parenthetical "(Relative to Oracle)" overflows the 6.3in height. It goes in the caption.
    ax.set_ylabel("Certification Delay")
    if args.title:
        ax.set_title(args.title, loc="center", pad=14)
    ax.set_xlim(epsilons[0], epsilons[-1])
    ax.grid(True, alpha=0.2, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Companion figure: the two stopping times themselves, not their difference.
    #
    # This one extends BELOW --eps-min, down to the oracle's own floor, and simply omits the
    # method curve where it is censored. That stretch is a result, not missing data: the
    # oracle certifies every trial there and the bound never does within the horizon. The
    # delay figure above stays truncated, because a delay genuinely does not exist there.

    fig2, ax2 = plt.subplots(figsize=FIGSIZE)
    if args.aggregate == "mean_trajectory":
        times = np.asarray([mean_trajectory_times(q, bound, float(e)) for e in epsilons])
        tau_true_curve, tau_ours_curve = times[:, 0], times[:, 1]
        # A single crossing of one averaged curve -- no per-trial spread exists to band.
        true_lo = true_hi = ours_lo = ours_hi = None
    else:
        tt, to, tl, th, ol, oh = [], [], [], [], [], []
        for eps in epsilons:
            a, a_ok = first_crossing(q, float(eps))
            b, b_ok = first_crossing(bound, float(eps))
            if not a_ok.all():
                raise ValueError(f"epsilon = {eps:g}: the oracle does not cross in every trial")
            # Mean, matching the delay figure, so the delay curve is exactly the
            # difference of these two curves (the mean is linear; the median is not).
            tt.append(a.mean())
            tl.append(np.quantile(a, half)); th.append(np.quantile(a, 1.0 - half))
            if b_ok.all():
                to.append(b.mean())
                ol.append(np.quantile(b, half)); oh.append(np.quantile(b, 1.0 - half))
            else:   # censored: no honest mean exists, so the curve simply stops
                to.append(np.nan); ol.append(np.nan); oh.append(np.nan)
        tau_true_curve, tau_ours_curve = np.asarray(tt), np.asarray(to)
        true_lo, true_hi = np.asarray(tl), np.asarray(th)
        ours_lo, ours_hi = np.asarray(ol), np.asarray(oh)

    if ours_lo is not None:
        ax2.fill_between(epsilons, ours_lo, ours_hi, color=OURS_COLOR, alpha=0.15, linewidth=0)
        ax2.fill_between(epsilons, true_lo, true_hi, color=TRUTH_COLOR, alpha=0.12, linewidth=0)
    ax2.plot(epsilons, tau_ours_curve, color=OURS_COLOR, linewidth=2.4, label=OURS_LABEL)
    ax2.plot(epsilons, tau_true_curve, color=TRUTH_COLOR, linewidth=2.0, label=TRUTH_LABEL)
    ax2.set_xlabel(r"Stopping Threshold, $\epsilon$")
    ax2.set_ylabel("Stopping Time")
    if args.title:
        ax2.set_title(args.title, loc="center", pad=14)
    ax2.set_xlim(epsilons[0], epsilons[-1])
    # Headroom above the widest band so the legend sits clear of it, plus a slightly
    # smaller legend than the axis type -- the two entries are long.
    top = float(np.nanmax(ours_hi if ours_hi is not None else tau_ours_curve))
    ax2.set_ylim(0, top * 1.22)
    ax2.grid(True, alpha=0.2, linewidth=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", frameon=False, fontsize=18,
               borderaxespad=0.3, handlelength=1.6, labelspacing=0.3)
    fig2.tight_layout()
    times_stem = args.output_dir / f"{args.prefix}_stopping_time"
    fig2.savefig(times_stem.with_suffix(".png"), dpi=150)
    fig2.savefig(times_stem.with_suffix(".pdf"))
    plt.close(fig2)

    # The delay figure is superseded by the two-curve comparison above: the same information,
    # with the oracle's own stopping time visible rather than folded into a difference.
    plt.close(fig)
    stem = args.output_dir / f"{args.prefix}_stopping_time_delay"

    report = [0.12, 0.15, 0.20, 0.30, 0.40, 0.50]
    summary = {
        "input_npz": str(args.input_npz), "bound_key": args.bound_key,
        "n_trials": int(q.shape[0]), "horizon": int(q.shape[1]),
        "eps_min": float(eps_min), "eps_max": float(eps_max), "n_eps": args.n_eps,
        "oracle_window": [oracle_lo, oracle_hi],
        "aggregate": args.aggregate, "smallest_uncensored_epsilon": floor,
        "at": {f"{e:g}": (mean_trajectory_delay(q, bound, e) if args.aggregate == "mean_trajectory"
                          else float(delays_at(q, bound, e).mean()))
               for e in report if eps_min <= e <= eps_max and e > floor},
    }
    stem.with_name(stem.name + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {times_stem}.png / .pdf and the summary json")
    print(f"  smallest uncensored epsilon: {floor:.4f}")
    print(f"  aggregate: {args.aggregate}")
    for e, v in summary["at"].items():
        print(f"  eps={e:>5}: delay {v:8.0f}")


if __name__ == "__main__":
    main()
