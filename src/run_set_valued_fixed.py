#!/usr/bin/env python3
"""Set-valued representation + fixed taxonomy: bootstrap-with-replacement scoring
with a session's full mode SET kept intact (no reduction to one winning mode).

Discovery: Z_t = 1{T(A_i) \\ S_{t-1} != empty} for the drawn unit's full mode
set T(A_i); q_t = population fraction of ALL N units still carrying an
undiscovered mode given what's been seen so far. Both defined for
bootstrap-with-replacement draws (repeats allowed), generalizing the original
permutation-sampling algorithm -- see av_discovery_fixed_set_valued.py.

Thin wiring script: sampling/scoring math lives in av_core.py, discovery logic
in av_discovery_fixed_set_valued.py. No Wilson CI, no Good-Turing baseline
(the classical GT point estimate/bound assumes single-label draws and has no
well-defined analogue here -- report only the e-value UCB and restart-mixture
UCB). --n-bootstrap-samples is the simulation horizon.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from av_core import (
    build_restart_schedule,
    require_file,
    require_lambda,
    require_probability,
    restart_mixture_tail_mass,
    restart_mixture_weights_for_scheme,
    restarted_mixture_upper_confidence_trajectory,
    upper_confidence_trajectory,
)
from av_discovery_fixed_set_valued import build_population, load_session_to_failure_modes, run_trial
from evaluation_population import load_population_session_ids, require_within_population
from av_lambda_restart_mixture import (
    HEADLINE_KEY,
    HEADLINE_LAMBDA_GRID_POINTS,
    HEADLINE_WEIGHTING_SCHEME,
    headline_upper_confidence_trajectory,
)
from run_config import load_run_config

RESTART_LAMBDA = 0.25
PERIODIC_RESTART_INTERVAL = 100
EVERYTIME_RESTART_INTERVAL = 1

_WORKER: dict = {}


def _init_worker(failure_modes_by_session_index, mode_to_session_indices, n_population,
                  n_bootstrap_samples, alpha, lambda_values, binary_search_iterations,
                  restart_starts_periodic, restart_weights_periodic, restart_tail_mass_periodic,
                  restart_starts_everytime, restart_weights_everytime, restart_tail_mass_everytime) -> None:
    import threadpoolctl

    threadpoolctl.threadpool_limits(1)
    _WORKER.update(
        failure_modes_by_session_index=failure_modes_by_session_index,
        mode_to_session_indices=mode_to_session_indices,
        n_population=n_population, n_bootstrap_samples=n_bootstrap_samples, alpha=alpha,
        lambda_values=lambda_values, binary_search_iterations=binary_search_iterations,
        restart_starts_periodic=restart_starts_periodic,
        restart_weights_periodic=restart_weights_periodic,
        restart_tail_mass_periodic=restart_tail_mass_periodic,
        restart_starts_everytime=restart_starts_everytime,
        restart_weights_everytime=restart_weights_everytime,
        restart_tail_mass_everytime=restart_tail_mass_everytime,
    )


def _run_one_trial(seed_sequence: np.random.SeedSequence):
    w = _WORKER
    rng = np.random.default_rng(seed_sequence)
    draw_indices = rng.integers(0, w["n_population"], size=w["n_bootstrap_samples"])
    z, q_row = run_trial(draw_indices, w["failure_modes_by_session_index"], w["mode_to_session_indices"])
    ucb_rows = {
        f"lambda-{lv:g}": upper_confidence_trajectory(z, lv, w["alpha"], w["binary_search_iterations"])
        for lv in w["lambda_values"]
    }
    restart_periodic = restarted_mixture_upper_confidence_trajectory(
        z, RESTART_LAMBDA, w["alpha"],
        w["restart_starts_periodic"], w["restart_weights_periodic"], w["restart_tail_mass_periodic"],
        binary_search_iterations=w["binary_search_iterations"],
    )
    restart_everytime = restarted_mixture_upper_confidence_trajectory(
        z, RESTART_LAMBDA, w["alpha"],
        w["restart_starts_everytime"], w["restart_weights_everytime"], w["restart_tail_mass_everytime"],
        binary_search_iterations=w["binary_search_iterations"],
    )
    headline = headline_upper_confidence_trajectory(z, w["alpha"], w["binary_search_iterations"])
    return q_row, ucb_rows, restart_periodic, restart_everytime, headline


def save_trajectory_arrays(path, metadata, q_matrix, ucb_matrices, restart_periodic_matrix,
                            restart_everytime_matrix, headline_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "q_matrix": q_matrix.astype(np.float32),
        "restart_mixture_periodic_geometric_matrix": restart_periodic_matrix.astype(np.float32),
        "restart_mixture_everytime_heavytail_matrix": restart_everytime_matrix.astype(np.float32),
        HEADLINE_KEY: headline_matrix.astype(np.float32),
        "lambda_names": np.asarray(list(ucb_matrices), dtype=str),
    }
    for lambda_name, matrix in ucb_matrices.items():
        prefix = lambda_name.replace(".", "p").replace("-", "_")
        arrays[f"{prefix}__ucb_matrix"] = matrix.astype(np.float32)
    np.savez_compressed(path, **arrays)


def run_validation(*, taxonomy_json: Path, population_json: Path, label_name: str, n_bootstrap_samples: int, n_trials: int,
                    n_workers: int, seed: int, alpha: float, lambda_values: list[float],
                    binary_search_iterations: int, stopping_epsilons: list[float], ci_level: float,
                    output_npz: Path) -> dict:
    require_probability(alpha, "--alpha")
    require_probability(ci_level, "--ci-level")
    if n_bootstrap_samples <= 0:
        raise ValueError("--n-bootstrap-samples must be a positive integer")
    if n_trials <= 0:
        raise ValueError("--n-trials must be a positive integer")
    if n_workers <= 0:
        raise ValueError("--n-workers must be a positive integer")
    if not lambda_values:
        raise ValueError("provide at least one --lambda-value")
    for lv in lambda_values:
        require_lambda(lv)
    if binary_search_iterations <= 0:
        raise ValueError("--binary-search-iterations must be positive")
    if not stopping_epsilons:
        raise ValueError("provide at least one --stopping-epsilon")
    for epsilon in stopping_epsilons:
        require_probability(epsilon, "--stopping-epsilon")

    session_to_failure_modes = load_session_to_failure_modes(taxonomy_json)
    population_session_ids = load_population_session_ids(population_json)
    require_within_population(session_to_failure_modes, population_session_ids, str(taxonomy_json))
    failure_modes_by_session_index, mode_to_session_indices = build_population(
        session_to_failure_modes, population_session_ids
    )
    n_population = len(failure_modes_by_session_index)
    n_modes = len(mode_to_session_indices)
    n_singleton_modes = sum(1 for indices in mode_to_session_indices.values() if len(indices) == 1)

    lambda_names = [f"lambda-{lv:g}" for lv in lambda_values]
    q_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)
    ucb_matrices = {name: np.empty((n_trials, n_bootstrap_samples), dtype=np.float64) for name in lambda_names}
    restart_periodic_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)
    restart_everytime_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)
    headline_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)

    restart_starts_periodic = build_restart_schedule(n_bootstrap_samples, PERIODIC_RESTART_INTERVAL)
    restart_starts_everytime = build_restart_schedule(n_bootstrap_samples, EVERYTIME_RESTART_INTERVAL)
    restart_weights_periodic, total_mass = restart_mixture_weights_for_scheme(
        "geometric", len(restart_starts_periodic)
    )
    restart_tail_mass_periodic = restart_mixture_tail_mass(restart_weights_periodic, total_mass)
    restart_weights_everytime, total_mass = restart_mixture_weights_for_scheme(
        "log_squared", len(restart_starts_everytime)
    )
    restart_tail_mass_everytime = restart_mixture_tail_mass(restart_weights_everytime, total_mass)

    print(f"Running {n_trials} trials across {n_workers} worker process(es)...")
    child_seeds = np.random.SeedSequence(seed).spawn(n_trials)
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker,
        initargs=(failure_modes_by_session_index, mode_to_session_indices, n_population, n_bootstrap_samples,
                  alpha, lambda_values, binary_search_iterations, restart_starts_periodic,
                  restart_weights_periodic, restart_tail_mass_periodic,
                  restart_starts_everytime, restart_weights_everytime, restart_tail_mass_everytime),
    ) as executor:
        for trial, (q_row, ucb_rows, restart_p, restart_e, headline) in enumerate(
            executor.map(_run_one_trial, child_seeds, chunksize=8)
        ):
            q_matrix[trial] = q_row
            for lambda_name in lambda_names:
                ucb_matrices[lambda_name][trial] = ucb_rows[lambda_name]
            restart_periodic_matrix[trial] = restart_p
            restart_everytime_matrix[trial] = restart_e
            headline_matrix[trial] = headline

    result = {
        "label_name": label_name,
        "taxonomy_json": str(taxonomy_json), "population_json": str(population_json),
        "validation_scope_note": (
            "Coverage is evaluated by bootstrap resampling (with replacement) from the fixed session "
            "population in population_json (successes included), treated as the true distribution P. A session's full mode SET is "
            "kept intact (no reduction to one winning mode); q_t is the population-level fraction of "
            "all N units still carrying at least one undiscovered mode. No Good-Turing baseline: the "
            "classical GT point estimate/bound assumes single-label draws and has no well-defined "
            "analogue for genuinely set-valued observations."
        ),
        "n_population": n_population, "n_modes": n_modes, "n_singleton_modes": n_singleton_modes,
        "n_bootstrap_samples": n_bootstrap_samples, "n_trials": n_trials, "seed": seed, "alpha": alpha,
        "ci_level": ci_level, "lambda_values": lambda_values, "stopping_epsilons": stopping_epsilons,
        "headline_method": {"restart_schedule": "every_time", "weighting_scheme": HEADLINE_WEIGHTING_SCHEME,
                            "lambda_grid_points": HEADLINE_LAMBDA_GRID_POINTS},
        "output_npz": str(output_npz),
    }
    save_trajectory_arrays(output_npz, result, q_matrix, ucb_matrices, restart_periodic_matrix,
                            restart_everytime_matrix, headline_matrix)
    return result


CONFIG_KEYS = {
    "taxonomy_json", "population_json", "label_name", "n_bootstrap_samples", "n_trials", "n_workers", "seed",
    "alpha", "lambda_values", "binary_search_iterations", "stopping_epsilons", "ci_level",
    "output_json", "output_npz",
}
CONFIG_PATH_KEYS = {"taxonomy_json", "population_json", "output_json", "output_npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="JSON run config -- see configs/run_set_valued_oneshot_fixed.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(args.config, CONFIG_KEYS, CONFIG_PATH_KEYS)
    require_file(config["taxonomy_json"], "taxonomy_json")
    output_json = config.pop("output_json")
    result = run_validation(**config)
    result["config_path"] = str(args.config)
    rendered = json.dumps(result, indent=2)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
