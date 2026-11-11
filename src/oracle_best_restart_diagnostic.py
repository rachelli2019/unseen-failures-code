#!/usr/bin/env python3
"""Oracle best-restart diagnostic: at every t, what is the tightest UCB any
SINGLE restart could have achieved, choosing the restart time r
RETROSPECTIVELY (with full knowledge of this trial's own z sequence)?

    U_t^oracle = min_r U_t^(r),   U_t^(r) = the single-restart-at-r UCB.

This is NOT itself a valid anytime bound (r is chosen using information not
available causally, so it isn't a legitimate estimator) -- it exists purely
as a benchmark: the best any single-restart strategy could conceivably do at
t, an upper limit no CAUSAL single-restart heuristic (fixed_500, fixed_1000,
adaptive_cumulative, adaptive_windowed, gated_windowed) could ever beat.

Comparing it to the actual restart-MIXTURE baseline (every-time restart,
log_squared weights) -- U_t^mixture - U_t^oracle -- answers the real
question: is the mixture already paying only a modest price for not knowing
the best restart time in advance? If so, that explains directly why none of
the adaptive single-restart heuristics meaningfully beat the mixture: the
mixture already captures nearly all of the achievable benefit of restarting,
leaving little headroom for being cleverer about WHEN to restart.

Efficient computation: since U_t^(r) is monotone in the restart e-process
value E_{r:t}(p), the oracle's own e-process is

    M_t^oracle(p) = max_r E_{r:t}(p)   (over r <= t only)

i.e. exactly the restart-mixture construction with a MAX reduction over r
instead of a weighted logsumexp. Reuses av_core.compile_restart_mixture (with
starts=1..n and dummy uniform weights, since only its zeros_in_range/
ones_in_range arrays are used here -- the weights themselves are never
consulted by this script) for the O(n) per-restart sufficient statistics,
then does its own max-based binary search inversion. This is an O(n^2)
computation per trial (all n possible restart times x all n timesteps), not
O(n) like the other constructions in this project -- restart_time_variants
scripts are still the right choice for anything meant to run at production
trial counts; this script is a diagnostic, kept separate from all of them.
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from av_core import (
    build_restart_schedule,
    compile_restart_mixture,
    require_binary_z,
    require_file,
    require_lambda,
    require_probability,
    restart_mixture_tail_mass,
    restart_mixture_weights_for_scheme,
    restarted_mixture_upper_confidence_trajectory,
)
from av_discovery_fixed_primary import build_session_population, load_session_to_failure_mode, run_bootstrap_trial
from evaluation_population import load_population_session_ids
from run_config import load_run_config

RESTART_LAMBDA = 0.25
EVERYTIME_RESTART_INTERVAL = 1
WEIGHTING_SCHEME = "log_squared"


def oracle_best_restart_upper_confidence_trajectory(
    z: np.ndarray, lambda_value: float, alpha: float, binary_search_iterations: int,
) -> np.ndarray:
    """U_t^oracle for t=1..len(z): the tightest single-restart UCB achievable
    at each t, choosing the restart time with full hindsight."""
    z = require_binary_z(z)
    require_lambda(lambda_value)
    require_probability(alpha, "alpha")
    if binary_search_iterations <= 0:
        raise ValueError("binary_search_iterations must be positive")

    n = len(z)
    starts = np.arange(1, n + 1, dtype=np.int64)
    # Dummy uniform weights: compile_restart_mixture requires weights.sum() <= 1,
    # but weights/log_weights/tail_mass are never read below -- only
    # zeros_in_range/ones_in_range (which don't depend on weights at all) are used.
    dummy_weights = np.full(n, 1.0 / n, dtype=np.float64)
    mixture = compile_restart_mixture(z, starts, dummy_weights, tail_mass=0.0)

    t_index = np.arange(1, n + 1, dtype=np.float64)
    active = t_index[None, :] >= starts[:, None]  # active[r-1, t-1] = True iff r <= t

    def _oracle_log_m_at_p(p: np.ndarray, scratch: np.ndarray) -> np.ndarray:
        log_if_zero = np.log1p(lambda_value * p)
        log_if_one = np.log1p(lambda_value * (p - 1.0))
        np.multiply(mixture.zeros_in_range, log_if_zero[None, :], out=scratch)
        scratch += mixture.ones_in_range * log_if_one[None, :]
        np.putmask(scratch, ~active, -np.inf)
        log_m = np.max(scratch, axis=0)
        if not np.all(np.isfinite(log_m)):
            raise FloatingPointError("non-finite oracle-best-restart e-process values during inversion")
        return log_m

    boundary = math.log(1.0 / alpha)
    lower = np.zeros(n, dtype=np.float64)
    upper = np.ones(n, dtype=np.float64)
    scratch = np.empty_like(mixture.zeros_in_range)

    log_at_one = _oracle_log_m_at_p(upper, scratch)
    one_is_valid = log_at_one < boundary
    lower[one_is_valid] = 1.0

    active_t = ~one_is_valid
    for _ in range(binary_search_iterations):
        midpoint = (lower + upper) / 2.0
        log_at_midpoint = _oracle_log_m_at_p(midpoint, scratch)
        midpoint_is_valid = log_at_midpoint < boundary
        move_lower = active_t & midpoint_is_valid
        move_upper = active_t & ~midpoint_is_valid
        lower[move_lower] = midpoint[move_lower]
        upper[move_upper] = midpoint[move_upper]

    if not np.all(np.isfinite(lower)):
        raise FloatingPointError("non-finite oracle-best-restart UCB trajectory")
    return lower


_WORKER: dict = {}


def _init_worker(mode_per_unit, mode_frequency, n_population, n_bootstrap_samples, alpha,
                  binary_search_iterations, restart_starts_everytime, restart_weights_everytime,
                  restart_tail_mass_everytime) -> None:
    import threadpoolctl

    threadpoolctl.threadpool_limits(1)
    _WORKER.update(
        mode_per_unit=mode_per_unit, mode_frequency=mode_frequency, n_population=n_population,
        n_bootstrap_samples=n_bootstrap_samples, alpha=alpha, binary_search_iterations=binary_search_iterations,
        restart_starts_everytime=restart_starts_everytime, restart_weights_everytime=restart_weights_everytime,
        restart_tail_mass_everytime=restart_tail_mass_everytime,
    )


def _run_one_trial(seed_sequence: np.random.SeedSequence):
    w = _WORKER
    rng = np.random.default_rng(seed_sequence)
    draw_indices = rng.integers(0, w["n_population"], size=w["n_bootstrap_samples"])
    z, _, q_row = run_bootstrap_trial(draw_indices, w["mode_per_unit"], w["mode_frequency"])
    oracle_bound = oracle_best_restart_upper_confidence_trajectory(
        z, RESTART_LAMBDA, w["alpha"], w["binary_search_iterations"]
    )
    mixture_bound = restarted_mixture_upper_confidence_trajectory(
        z, RESTART_LAMBDA, w["alpha"],
        w["restart_starts_everytime"], w["restart_weights_everytime"], w["restart_tail_mass_everytime"],
        binary_search_iterations=w["binary_search_iterations"],
    )
    return q_row, oracle_bound, mixture_bound


def run_validation(*, taxonomy_json: Path, population_json: Path, label_name: str, n_bootstrap_samples: int, n_trials: int,
                    n_workers: int, seed: int, alpha: float, binary_search_iterations: int,
                    output_npz: Path) -> dict:
    require_probability(alpha, "alpha")
    require_lambda(RESTART_LAMBDA)
    if n_bootstrap_samples <= 0:
        raise ValueError("n_bootstrap_samples must be a positive integer")
    if n_trials <= 0:
        raise ValueError("n_trials must be a positive integer")
    if n_workers <= 0:
        raise ValueError("n_workers must be a positive integer")

    session_to_mode = load_session_to_failure_mode(taxonomy_json)
    mode_per_unit, mode_frequency, n_population = build_session_population(
        session_to_mode, load_population_session_ids(population_json)
    )

    restart_starts_everytime = build_restart_schedule(n_bootstrap_samples, EVERYTIME_RESTART_INTERVAL)
    restart_weights_everytime, total_mass = restart_mixture_weights_for_scheme(
        WEIGHTING_SCHEME, len(restart_starts_everytime)
    )
    restart_tail_mass_everytime = restart_mixture_tail_mass(restart_weights_everytime, total_mass)

    q_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)
    oracle_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)
    mixture_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)

    print(f"Running {n_trials} trials across {n_workers} worker process(es) "
          f"(O(n^2) oracle computation, n={n_bootstrap_samples})...")
    child_seeds = np.random.SeedSequence(seed).spawn(n_trials)
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker,
        initargs=(mode_per_unit, mode_frequency, n_population, n_bootstrap_samples, alpha,
                  binary_search_iterations, restart_starts_everytime, restart_weights_everytime,
                  restart_tail_mass_everytime),
    ) as executor:
        for trial, (q_row, oracle_bound, mixture_bound) in enumerate(
            executor.map(_run_one_trial, child_seeds, chunksize=4)
        ):
            q_matrix[trial] = q_row
            oracle_matrix[trial] = oracle_bound
            mixture_matrix[trial] = mixture_bound

    price_of_not_knowing_r = mixture_matrix - oracle_matrix
    result = {
        "label_name": label_name,
        "taxonomy_json": str(taxonomy_json), "population_json": str(population_json),
        "validation_scope_note": (
            "U_t^oracle = min_r U_t^(r), the best single-restart UCB achievable at t choosing r "
            "retrospectively -- NOT itself a valid anytime bound (r depends on future information), "
            "a diagnostic upper limit on any causal single-restart heuristic. Compared against the "
            "restart-mixture baseline (every-time restart, log_squared weights) via "
            "U_t^mixture - U_t^oracle, 'the price of not knowing the best restart time in advance'."
        ),
        "n_population": n_population, "n_bootstrap_samples": n_bootstrap_samples, "n_trials": n_trials,
        "seed": seed, "alpha": alpha, "restart_lambda": RESTART_LAMBDA,
        "mean_final_price_of_not_knowing_r": float(np.mean(price_of_not_knowing_r[:, -1])),
        "median_final_price_of_not_knowing_r": float(np.median(price_of_not_knowing_r[:, -1])),
        "mean_final_oracle_ucb": float(np.mean(oracle_matrix[:, -1])),
        "mean_final_mixture_ucb": float(np.mean(mixture_matrix[:, -1])),
        "output_npz": str(output_npz),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        metadata_json=np.asarray(json.dumps(result)),
        q_matrix=q_matrix.astype(np.float32),
        oracle_matrix=oracle_matrix.astype(np.float32),
        mixture_matrix=mixture_matrix.astype(np.float32),
    )
    return result


CONFIG_KEYS = {
    "taxonomy_json", "population_json", "label_name", "n_bootstrap_samples", "n_trials", "n_workers", "seed", "alpha",
    "binary_search_iterations", "output_npz",
}
CONFIG_PATH_KEYS = {"taxonomy_json", "population_json", "output_npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path,
                         help="JSON run config -- see configs/run_oracle_best_restart_diagnostic.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(args.config, CONFIG_KEYS, CONFIG_PATH_KEYS)
    require_file(config["taxonomy_json"], "taxonomy_json")
    result = run_validation(**config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
