#!/usr/bin/env python3
"""Primary representation + fixed taxonomy: bootstrap-with-replacement scoring
against a taxonomy built once in advance (one mode per session), read
from the taxonomy files in data/taxonomy/.

Thin wiring script: sampling/scoring math lives in av_core.py, discovery logic
in av_discovery_fixed_primary.py. No Wilson CI -- simultaneous-violation and
threshold-crossing rates are reported as plain empirical fractions.

--n-bootstrap-samples is the simulation horizon (trajectory length per trial),
an explicit required flag -- test it at multiple values (e.g. 2000, 3000,
5000) by just changing this one number; nothing else in the pipeline assumes
a particular horizon.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from av_core import (
    build_restart_schedule,
    classical_gt_bound,
    require_file,
    require_lambda,
    require_probability,
    restart_mixture_tail_mass,
    restart_mixture_weights_for_scheme,
    restarted_mixture_upper_confidence_trajectory,
    upper_confidence_trajectory,
)
from av_discovery_fixed_primary import build_session_population, load_session_to_failure_mode, run_bootstrap_trial
from evaluation_population import load_population_session_ids
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


def _init_worker(
    mode_per_unit: np.ndarray,
    mode_frequency: dict[str, float],
    n_population: int,
    n_bootstrap_samples: int,
    alpha: float,
    lambda_values: list[float],
    binary_search_iterations: int,
    restart_starts_periodic: np.ndarray,
    restart_weights_periodic_geometric: np.ndarray,
    restart_tail_mass_periodic_geometric: float,
    restart_weights_periodic_log_squared: np.ndarray,
    restart_tail_mass_periodic_log_squared: float,
    restart_starts_everytime: np.ndarray,
    restart_weights_everytime_log_squared: np.ndarray,
    restart_tail_mass_everytime_log_squared: float,
) -> None:
    import threadpoolctl

    threadpoolctl.threadpool_limits(1)
    _WORKER.update(
        mode_per_unit=mode_per_unit,
        mode_frequency=mode_frequency,
        n_population=n_population,
        n_bootstrap_samples=n_bootstrap_samples,
        alpha=alpha,
        lambda_values=lambda_values,
        binary_search_iterations=binary_search_iterations,
        restart_starts_periodic=restart_starts_periodic,
        restart_weights_periodic_geometric=restart_weights_periodic_geometric,
        restart_tail_mass_periodic_geometric=restart_tail_mass_periodic_geometric,
        restart_weights_periodic_log_squared=restart_weights_periodic_log_squared,
        restart_tail_mass_periodic_log_squared=restart_tail_mass_periodic_log_squared,
        restart_starts_everytime=restart_starts_everytime,
        restart_weights_everytime_log_squared=restart_weights_everytime_log_squared,
        restart_tail_mass_everytime_log_squared=restart_tail_mass_everytime_log_squared,
    )


def _run_one_trial(seed_sequence: np.random.SeedSequence):
    w = _WORKER
    rng = np.random.default_rng(seed_sequence)
    draw_indices = rng.integers(0, w["n_population"], size=w["n_bootstrap_samples"])
    z, gt_hat, q_row = run_bootstrap_trial(draw_indices, w["mode_per_unit"], w["mode_frequency"])
    ucb_rows = {
        f"lambda-{lv:g}": upper_confidence_trajectory(z, lv, w["alpha"], w["binary_search_iterations"])
        for lv in w["lambda_values"]
    }
    # Three (restart schedule) x (weighting scheme) combinations, not just the two
    # (periodic+geometric, everytime+log_squared) used for the main figure -- lets a
    # diagnostic plot isolate whether restart schedule or weighting scheme drives the
    # everytime+log_squared combination's advantage. everytime+geometric is excluded:
    # geometric weights w_j = 2^-(j+1) underflow to exactly 0.0 in float64 past
    # j~1074, which the everytime schedule reaches well before this horizon (3000
    # restarts) -- not something padding/clipping should paper over, so this
    # combination just isn't numerically well-defined here.
    restart_periodic_geometric = restarted_mixture_upper_confidence_trajectory(
        z, RESTART_LAMBDA, w["alpha"],
        w["restart_starts_periodic"], w["restart_weights_periodic_geometric"],
        w["restart_tail_mass_periodic_geometric"],
        binary_search_iterations=w["binary_search_iterations"],
    )
    restart_periodic_log_squared = restarted_mixture_upper_confidence_trajectory(
        z, RESTART_LAMBDA, w["alpha"],
        w["restart_starts_periodic"], w["restart_weights_periodic_log_squared"],
        w["restart_tail_mass_periodic_log_squared"],
        binary_search_iterations=w["binary_search_iterations"],
    )
    restart_everytime_log_squared = restarted_mixture_upper_confidence_trajectory(
        z, RESTART_LAMBDA, w["alpha"],
        w["restart_starts_everytime"], w["restart_weights_everytime_log_squared"],
        w["restart_tail_mass_everytime_log_squared"],
        binary_search_iterations=w["binary_search_iterations"],
    )
    headline = headline_upper_confidence_trajectory(z, w["alpha"], w["binary_search_iterations"])
    return (gt_hat, q_row, ucb_rows, restart_periodic_geometric, restart_periodic_log_squared,
            restart_everytime_log_squared, headline)


def save_trajectory_arrays(path: Path, metadata: dict, q_matrix, gt_hat_matrix, gt_upper_bound_matrix,
                            gt_bonferroni_bound_matrix, ucb_matrices, restart_periodic_geometric_matrix,
                            restart_periodic_log_squared_matrix, restart_everytime_log_squared_matrix,
                            headline_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "q_matrix": q_matrix.astype(np.float32),
        "gt_hat_matrix": gt_hat_matrix.astype(np.float32),
        "gt_upper_bound_matrix": gt_upper_bound_matrix.astype(np.float32),
        "gt_bonferroni_bound_matrix": gt_bonferroni_bound_matrix.astype(np.float32),
        "restart_mixture_periodic_geometric_matrix": restart_periodic_geometric_matrix.astype(np.float32),
        "restart_mixture_periodic_log_squared_matrix": restart_periodic_log_squared_matrix.astype(np.float32),
        # Kept under its original key name (not renamed to "log_squared") since this is
        # the combination plot_av_pipeline.py's main figure already reads and depends on.
        "restart_mixture_everytime_heavytail_matrix": restart_everytime_log_squared_matrix.astype(np.float32),
        HEADLINE_KEY: headline_matrix.astype(np.float32),
        "lambda_names": np.asarray(list(ucb_matrices), dtype=str),
    }
    for lambda_name, matrix in ucb_matrices.items():
        prefix = lambda_name.replace(".", "p").replace("-", "_")
        arrays[f"{prefix}__ucb_matrix"] = matrix.astype(np.float32)
    np.savez_compressed(path, **arrays)


def run_validation(*, taxonomy_json: Path, population_json: Path, label_name: str, n_bootstrap_samples: int, n_bonferroni: int,
                    n_trials: int, n_workers: int, seed: int, alpha: float, lambda_values: list[float],
                    binary_search_iterations: int, stopping_epsilons: list[float], ci_level: float,
                    output_npz: Path) -> dict:
    require_probability(alpha, "--alpha")
    require_probability(ci_level, "--ci-level")
    if n_bootstrap_samples <= 0:
        raise ValueError("--n-bootstrap-samples must be a positive integer")
    if n_bonferroni <= 0:
        raise ValueError("--n-bonferroni must be a positive integer")
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

    session_to_mode = load_session_to_failure_mode(taxonomy_json)
    mode_per_unit, mode_frequency, n_population = build_session_population(
        session_to_mode, load_population_session_ids(population_json)
    )
    n_modes = len(mode_frequency)
    mode_counts = {m: round(f * n_population) for m, f in mode_frequency.items()}
    n_singleton_modes = sum(1 for c in mode_counts.values() if c == 1)

    lambda_names = [f"lambda-{lv:g}" for lv in lambda_values]
    q_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)
    gt_hat_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)
    ucb_matrices = {name: np.empty((n_trials, n_bootstrap_samples), dtype=np.float64) for name in lambda_names}
    restart_periodic_geometric_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)
    restart_periodic_log_squared_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)
    restart_everytime_log_squared_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)
    headline_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)

    restart_starts_periodic = build_restart_schedule(n_bootstrap_samples, PERIODIC_RESTART_INTERVAL)
    restart_starts_everytime = build_restart_schedule(n_bootstrap_samples, EVERYTIME_RESTART_INTERVAL)
    restart_weights_periodic_geometric, total_mass = restart_mixture_weights_for_scheme(
        "geometric", len(restart_starts_periodic)
    )
    restart_tail_mass_periodic_geometric = restart_mixture_tail_mass(restart_weights_periodic_geometric, total_mass)
    restart_weights_periodic_log_squared, total_mass = restart_mixture_weights_for_scheme(
        "log_squared", len(restart_starts_periodic)
    )
    restart_tail_mass_periodic_log_squared = restart_mixture_tail_mass(restart_weights_periodic_log_squared, total_mass)
    restart_weights_everytime_log_squared, total_mass = restart_mixture_weights_for_scheme(
        "log_squared", len(restart_starts_everytime)
    )
    restart_tail_mass_everytime_log_squared = restart_mixture_tail_mass(restart_weights_everytime_log_squared, total_mass)

    print(f"Running {n_trials} trials across {n_workers} worker process(es)...")
    child_seeds = np.random.SeedSequence(seed).spawn(n_trials)
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker,
        initargs=(mode_per_unit, mode_frequency, n_population, n_bootstrap_samples, alpha, lambda_values,
                  binary_search_iterations, restart_starts_periodic,
                  restart_weights_periodic_geometric, restart_tail_mass_periodic_geometric,
                  restart_weights_periodic_log_squared, restart_tail_mass_periodic_log_squared,
                  restart_starts_everytime,
                  restart_weights_everytime_log_squared, restart_tail_mass_everytime_log_squared),
    ) as executor:
        for trial, (gt_hat, q_row, ucb_rows, restart_pg, restart_pl, restart_el, headline) in enumerate(
            executor.map(_run_one_trial, child_seeds, chunksize=8)
        ):
            gt_hat_matrix[trial] = gt_hat
            q_matrix[trial] = q_row
            for lambda_name in lambda_names:
                ucb_matrices[lambda_name][trial] = ucb_rows[lambda_name]
            restart_periodic_geometric_matrix[trial] = restart_pg
            restart_periodic_log_squared_matrix[trial] = restart_pl
            restart_everytime_log_squared_matrix[trial] = restart_el
            headline_matrix[trial] = headline

    gt_upper_bound_matrix = classical_gt_bound(gt_hat_matrix, alpha, n_bonferroni=1)
    gt_bonferroni_bound_matrix = classical_gt_bound(gt_hat_matrix, alpha, n_bonferroni=n_bonferroni)

    result = {
        "label_name": label_name,
        "taxonomy_json": str(taxonomy_json), "population_json": str(population_json),
        "validation_scope_note": (
            "Coverage is evaluated by bootstrap resampling (with replacement) from the fixed session "
            "population in population_json (successes included), treated as the true distribution P. The equivalence rule (mode_id "
            "per session) is fixed in advance -- no online discovery, no separate oracle clustering."
        ),
        "n_population": n_population, "n_modes": n_modes, "n_singleton_modes": n_singleton_modes,
        "n_bootstrap_samples": n_bootstrap_samples, "n_bonferroni": n_bonferroni, "n_trials": n_trials,
        "seed": seed, "alpha": alpha, "ci_level": ci_level, "lambda_values": lambda_values,
        "stopping_epsilons": stopping_epsilons,
        "headline_method": {"restart_schedule": "every_time", "weighting_scheme": HEADLINE_WEIGHTING_SCHEME,
                            "lambda_grid_points": HEADLINE_LAMBDA_GRID_POINTS},
        "pointwise_classical_gt": {"mean_final_estimate": float(np.mean(gt_hat_matrix[:, -1]))},
        "output_npz": str(output_npz),
    }
    save_trajectory_arrays(output_npz, result, q_matrix, gt_hat_matrix, gt_upper_bound_matrix,
                            gt_bonferroni_bound_matrix, ucb_matrices, restart_periodic_geometric_matrix,
                            restart_periodic_log_squared_matrix, restart_everytime_log_squared_matrix,
                            headline_matrix)
    return result


CONFIG_KEYS = {
    "taxonomy_json", "population_json", "label_name", "n_bootstrap_samples", "n_bonferroni", "n_trials",
    "n_workers", "seed", "alpha", "lambda_values", "binary_search_iterations",
    "stopping_epsilons", "ci_level", "output_json", "output_npz",
}
CONFIG_PATH_KEYS = {"taxonomy_json", "population_json", "output_json", "output_npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="JSON run config -- see configs/run_primary_oneshot_fixed.json")
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
