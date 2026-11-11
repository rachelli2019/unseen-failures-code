#!/usr/bin/env python3
"""Ablation over lambda and over the restart schedule, on the primary fixed taxonomy.

Two questions, one bootstrap pass:

A. Does the proposed bound depend on the lambda it is evaluated at?
   every-time restart mixture at lambda = 0.10, 0.25, 0.50.

B. At lambda = 0.25, what does restarting buy?
   restart mixture  vs  no restart.

The restart mixture uses an every-time schedule with log-squared heavy-tailed weights.
(Geometric weights are not offered: w_j = 2^-(j+1) underflows to exactly 0.0 in float64
past j~1074, which an every-time schedule reaches well before this horizon.)

Every arm is scored on the SAME bootstrap draws within a trial, so differences between
arms are the arms, not Monte Carlo noise. Settings are otherwise identical to
run_primary_fixed.py, so the lambda=0.25 every-time arm reproduces the headline bound.

Usage:
    uv run python src/run_lambda_restart_ablation.py \
        --config configs/run_lambda_restart_ablation.json
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
from av_discovery_fixed_primary import build_session_population, load_session_to_failure_mode, run_bootstrap_trial
from evaluation_population import load_population_session_ids
from run_config import load_run_config

WEIGHTING_SCHEME = "log_squared"
EVERYTIME_RESTART_INTERVAL = 1

# Ablation B is read at this lambda; it must also appear in lambda_values so the
# every-time arm is shared between the two ablations rather than recomputed.
ABLATION_B_LAMBDA = 0.25

_WORKER: dict = {}


def arm_key(name: str, lambda_value: float) -> str:
    return f"{name}__lambda_{lambda_value:g}".replace(".", "p")


def _init_worker(state: dict) -> None:
    import threadpoolctl

    threadpoolctl.threadpool_limits(1)
    _WORKER.update(state)


def _run_one_trial(seed_sequence: np.random.SeedSequence) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    w = _WORKER
    rng = np.random.default_rng(seed_sequence)
    draw_indices = rng.integers(0, w["n_population"], size=w["n_bootstrap_samples"])
    z, _gt_hat, q_row = run_bootstrap_trial(draw_indices, w["mode_per_unit"], w["mode_frequency"])

    arms: dict[str, np.ndarray] = {}

    # A: the proposed bound across the lambda grid.
    for lambda_value in w["lambda_values"]:
        arms[arm_key("restart_mixture", lambda_value)] = restarted_mixture_upper_confidence_trajectory(
            z, lambda_value, w["alpha"],
            w["starts_everytime"], w["weights_everytime"], w["tail_mass_everytime"],
            binary_search_iterations=w["binary_search_iterations"],
        )

    # B: the same bound with no restart at all, at one lambda.
    arms[arm_key("no_restart", ABLATION_B_LAMBDA)] = upper_confidence_trajectory(
        z, ABLATION_B_LAMBDA, w["alpha"], w["binary_search_iterations"],
    )
    return q_row, arms


def run_ablation(*, taxonomy_json: Path, population_json: Path, label_name: str, n_bootstrap_samples: int, n_trials: int,
                 n_workers: int, seed: int, alpha: float, lambda_values: list[float],
                 binary_search_iterations: int, ci_level: float, output_npz: Path) -> dict:
    require_probability(alpha, "alpha")
    require_probability(ci_level, "ci_level")
    for name, value in [("n_bootstrap_samples", n_bootstrap_samples), ("n_trials", n_trials),
                        ("n_workers", n_workers), ("binary_search_iterations", binary_search_iterations)]:
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not lambda_values:
        raise ValueError("provide at least one lambda value")
    for lambda_value in lambda_values:
        require_lambda(lambda_value)
    if ABLATION_B_LAMBDA not in lambda_values:
        raise ValueError(
            f"lambda_values must contain {ABLATION_B_LAMBDA} -- ablation B is read at that lambda "
            f"and shares the restart-mixture arm with ablation A"
        )

    session_to_mode = load_session_to_failure_mode(taxonomy_json)
    mode_per_unit, mode_frequency, n_population = build_session_population(
        session_to_mode, load_population_session_ids(population_json)
    )

    starts_everytime = build_restart_schedule(n_bootstrap_samples, EVERYTIME_RESTART_INTERVAL)
    weights_everytime, total_mass_everytime = restart_mixture_weights_for_scheme(
        WEIGHTING_SCHEME, len(starts_everytime)
    )

    worker_state = dict(
        mode_per_unit=mode_per_unit, mode_frequency=mode_frequency, n_population=n_population,
        n_bootstrap_samples=n_bootstrap_samples, alpha=alpha, lambda_values=lambda_values,
        binary_search_iterations=binary_search_iterations,
        starts_everytime=starts_everytime, weights_everytime=weights_everytime,
        tail_mass_everytime=restart_mixture_tail_mass(weights_everytime, total_mass_everytime),
    )

    print(f"{n_trials} trials x {n_bootstrap_samples} draws over {n_population} sessions, "
          f"{n_workers} worker(s)")
    q_matrix = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)
    arm_matrices: dict[str, np.ndarray] = {}

    child_seeds = np.random.SeedSequence(seed).spawn(n_trials)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(worker_state,)) as executor:
        for trial, (q_row, arms) in enumerate(executor.map(_run_one_trial, child_seeds, chunksize=8)):
            q_matrix[trial] = q_row
            for key, trajectory in arms.items():
                if key not in arm_matrices:
                    arm_matrices[key] = np.empty((n_trials, n_bootstrap_samples), dtype=np.float64)
                arm_matrices[key][trial] = trajectory

    metadata = {
        "label_name": label_name,
        "taxonomy_json": str(taxonomy_json), "population_json": str(population_json),
        "weighting_scheme": WEIGHTING_SCHEME,
        "ablation_b_lambda": ABLATION_B_LAMBDA,
        "n_population": n_population,
        "n_modes": len(mode_frequency),
        "n_bootstrap_samples": n_bootstrap_samples,
        "n_trials": n_trials,
        "seed": seed,
        "alpha": alpha,
        "ci_level": ci_level,
        "lambda_values": lambda_values,
        "arms": sorted(arm_matrices),
    }

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        metadata_json=np.asarray(json.dumps(metadata)),
        q_matrix=q_matrix.astype(np.float32),
        **{key: matrix.astype(np.float32) for key, matrix in arm_matrices.items()},
    )
    print(f"-> {output_npz}")
    return metadata


CONFIG_KEYS = {
    "taxonomy_json", "population_json", "label_name", "n_bootstrap_samples", "n_trials", "n_workers", "seed",
    "alpha", "lambda_values", "binary_search_iterations", "ci_level", "output_json", "output_npz",
}
CONFIG_PATH_KEYS = {"taxonomy_json", "population_json", "output_json", "output_npz"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_run_config(args.config, CONFIG_KEYS, CONFIG_PATH_KEYS)
    require_file(config["taxonomy_json"], "taxonomy_json")
    output_json = config.pop("output_json")

    result = run_ablation(**config)
    result["config_path"] = str(args.config)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
