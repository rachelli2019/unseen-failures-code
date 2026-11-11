#!/usr/bin/env python3
"""I.i.d. Bernoulli(p) simulation scored with the every-time restart-mixture UCB.

The simplest possible check of the anytime-validity guarantee. Draw
Z_t ~ Bernoulli(p) i.i.d. with no discovered-category bookkeeping, so every
Z_t = 1 is by construction a brand-new failure mode and the true unseen-failure
probability is exactly q_t = p for the whole trajectory -- no depletion, no
estimation. Novelty never saturates, so the bound gets an unbounded number of
chances to be violated, which makes this the hardest case for a simultaneous
guarantee and the s -> 1 limit of the Zipf/zeta family.

Reports, per horizon step t:
  - the mean UCB across trials, against the known constant q_t = p
  - simultaneous coverage Pr(q_s <= U_s for all s <= t), which can only fall as
    t grows -- the question is whether it falls to the nominal 1 - alpha (the
    bound is tight) or plateaus above it (the bound is conservative)

Scoring reuses av_core's restart-mixture e-process UCB with an every-time
restart schedule and log-squared heavy-tailed weights -- the same headline
method used for the real failure-mode taxonomies.

Usage:
    uv run python src/run_bernoulli_simulation.py \
        --output-json artifacts/bernoulli/summary.json \
        --output-npz  artifacts/bernoulli/trajectories.npz
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from av_core import (
    build_restart_schedule,
    require_lambda,
    require_probability,
    restart_mixture_tail_mass,
    restart_mixture_weights_for_scheme,
    restarted_mixture_upper_confidence_trajectory,
)

EVERYTIME_RESTART_INTERVAL = 1
WEIGHTING_SCHEME = "log_squared"

_WORKER: dict = {}


def _init_worker(p, horizon, alpha, lambda_value, binary_search_iterations,
                 restart_starts, restart_weights, restart_tail_mass) -> None:
    import threadpoolctl

    threadpoolctl.threadpool_limits(1)
    _WORKER.update(
        p=p, horizon=horizon, alpha=alpha, lambda_value=lambda_value,
        binary_search_iterations=binary_search_iterations,
        restart_starts=restart_starts, restart_weights=restart_weights,
        restart_tail_mass=restart_tail_mass,
    )


def _run_one_trial(seed_sequence: np.random.SeedSequence) -> np.ndarray:
    w = _WORKER
    rng = np.random.default_rng(seed_sequence)
    novelty = (rng.random(w["horizon"]) < w["p"]).astype(np.int8)
    return restarted_mixture_upper_confidence_trajectory(
        novelty,
        w["lambda_value"],
        w["alpha"],
        w["restart_starts"],
        w["restart_weights"],
        w["restart_tail_mass"],
        binary_search_iterations=w["binary_search_iterations"],
    )


def run(
    *,
    p: float,
    horizon: int,
    n_trials: int,
    n_workers: int,
    seed: int,
    alpha: float,
    lambda_value: float,
    binary_search_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ucb_matrix, coverage): shapes (n_trials, horizon) and (horizon,)."""
    require_probability(p, "p", include_zero=False)
    require_probability(alpha, "alpha")
    require_lambda(lambda_value)
    for name, value in (("horizon", horizon), ("n_trials", n_trials),
                        ("n_workers", n_workers),
                        ("binary_search_iterations", binary_search_iterations)):
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")

    starts = build_restart_schedule(horizon, EVERYTIME_RESTART_INTERVAL)
    weights, total_mass = restart_mixture_weights_for_scheme(WEIGHTING_SCHEME, len(starts))
    tail_mass = restart_mixture_tail_mass(weights, total_mass)

    ucb = np.empty((n_trials, horizon), dtype=np.float64)
    child_seeds = np.random.SeedSequence(seed).spawn(n_trials)
    print(f"Bernoulli(p={p:g}): {n_trials} trials x horizon {horizon}, "
          f"every-time restart mixture, alpha={alpha:g}...")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(p, horizon, alpha, lambda_value, binary_search_iterations,
                  starts, weights, tail_mass),
    ) as pool:
        for i, row in enumerate(pool.map(_run_one_trial, child_seeds, chunksize=1)):
            ucb[i] = row

    # q_t = p exactly for every t, so a violation at s is U_s < p. Simultaneous
    # coverage through t requires no violation at any s <= t.
    violated = ucb < p
    coverage = (~np.logical_or.accumulate(violated, axis=1)).mean(axis=0)
    return ucb, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, default=0.05,
                        help="Per-step novelty probability; q_t = p exactly. Default 0.05.")
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--lambda-value", type=float, default=0.25)
    parser.add_argument("--binary-search-iterations", type=int, default=15)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()

    ucb, coverage = run(
        p=args.p, horizon=args.horizon, n_trials=args.n_trials, n_workers=args.n_workers,
        seed=args.seed, alpha=args.alpha, lambda_value=args.lambda_value,
        binary_search_iterations=args.binary_search_iterations,
    )

    mean_ucb = ucb.mean(axis=0)
    summary = {
        "p": args.p,
        "q_t": args.p,
        "horizon": args.horizon,
        "n_trials": args.n_trials,
        "seed": args.seed,
        "alpha": args.alpha,
        "nominal_coverage": 1.0 - args.alpha,
        "lambda_value": args.lambda_value,
        "restart_schedule": "every-time",
        "weighting_scheme": WEIGHTING_SCHEME,
        "mean_ucb_final": float(mean_ucb[-1]),
        "simultaneous_coverage_final": float(coverage[-1]),
        "output_npz": str(args.output_npz),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, ucb_matrix=ucb, coverage=coverage,
                        q_t=np.full(args.horizon, args.p))
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
