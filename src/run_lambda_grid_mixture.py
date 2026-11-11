#!/usr/bin/env python3
"""Lambda mixture over a fine uniform grid, one shard of trials.

An arm is a set of lambda values with weights; its bound averages the e-values over
that set and inverts the single averaged e-process by bisection (the per-lambda bounds
are never averaged -- that would not be a valid bound). Arms come in two flavours in
the config:

    {"name": ..., "lambda_values": [...], "lambda_weights": [...]}   explicit grid
    {"name": ..., "uniform_grid_points": K}                           midpoints (k-1/2)/K, weights 1/K

An arm may add "restart": "every_time" (every-time restart schedule, log-squared
weights) or "none" (one process started at t = 1, no restarts). An arm without the
key is every_time, which is what every arm meant before the key existed -- configs
from before it keep their meaning unchanged.

The setting (what counts as a discovery) is the config's "discovery" block, handled by
lambda_grid_discovery_backends.py: fixed_primary, fixed_set_valued, online_primary or
bernoulli. Everything after z/q is identical across settings.

Sharding: trial i always uses np.random.SeedSequence(seed).spawn(n_trials)[i], whatever
shard runs it, so the concatenation of all shards equals one unsharded run -- and, with
seed 888 / 3000 draws and the fixed_primary backend, draws the same bootstrap samples as
run_lambda_restart_ablation.py.

Each arm yields one bound, `bound__<arm>`, inverted at log(1/alpha).

Each shard saves, besides the bound trajectories, the per-trial binary discovery
sequence z and oracle q, so any other lambda grid can be recomputed exactly later.

Usage:
    uv run python src/run_lambda_grid_mixture.py \
        --config configs/run_lambda_grid_mixture_primary_fixed.json --task-index 0 --n-tasks 25 \
        --output-dir artifacts/lambda_grid_mixture/<run_id>
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from av_core import require_lambda, require_probability
from av_lambda_restart_mixture import (
    everytime_lambda_mixture_upper_confidence_trajectory,
    require_lambda_weights,
    single_process_lambda_mixture_upper_confidence_trajectory,
    uniform_lambda_grid,
)
from lambda_grid_discovery_backends import get_backend
from run_config import load_run_config

WEIGHTING_SCHEME = "log_squared"
RESTART_SCHEDULES = ("every_time", "none")

_WORKER: dict = {}


def resolve_arms(arm_specs: list[dict]) -> list[dict]:
    """Turn config arm specs into {name, lambda_values, lambda_weights}, validated."""
    if not arm_specs:
        raise ValueError("arms must be non-empty")
    resolved = []
    seen = set()
    for spec in arm_specs:
        if not isinstance(spec, dict) or "name" not in spec:
            raise ValueError(f"each arm needs a 'name'; got {spec!r}")
        name = spec["name"]
        if name in seen:
            raise ValueError(f"duplicate arm name {name!r}")
        seen.add(name)
        keys = set(spec) - {"name", "restart"}
        restart = spec.get("restart", "every_time")
        if restart not in RESTART_SCHEDULES:
            raise ValueError(f"arm {name!r} has restart={restart!r}; must be one of {RESTART_SCHEDULES}")
        if keys == {"uniform_grid_points"}:
            lambda_values, lambda_weights = uniform_lambda_grid(int(spec["uniform_grid_points"]))
        elif keys == {"lambda_values", "lambda_weights"}:
            lambda_values = np.asarray(spec["lambda_values"], dtype=float)
            lambda_weights = np.asarray(spec["lambda_weights"], dtype=float)
        else:
            raise ValueError(
                f"arm {name!r} must have exactly 'uniform_grid_points' or exactly "
                f"'lambda_values' + 'lambda_weights'; got keys {sorted(keys)}"
            )
        require_lambda_weights(lambda_values, lambda_weights)
        for lv in lambda_values:
            require_lambda(float(lv))
        resolved.append({"name": name, "lambda_values": lambda_values, "lambda_weights": lambda_weights,
                         "restart": restart})
    return resolved


def shard_trial_indices(n_trials: int, task_index: int, n_tasks: int) -> np.ndarray:
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    if not 0 <= task_index < n_tasks:
        raise ValueError(f"task_index must be in [0, {n_tasks}); got {task_index!r}")
    if n_trials % n_tasks != 0:
        raise ValueError(f"n_trials={n_trials} must divide evenly into n_tasks={n_tasks}")
    per_task = n_trials // n_tasks
    return np.arange(task_index * per_task, (task_index + 1) * per_task, dtype=np.int64)


def _init_worker(state: dict) -> None:
    import threadpoolctl

    threadpoolctl.threadpool_limits(1)
    _WORKER.update(state)


def _run_one_trial(seed_sequence: np.random.SeedSequence):
    w = _WORKER
    rng = np.random.default_rng(seed_sequence)
    z, q_row = w["draw"](w["discovery_state"], rng, w["n_bootstrap_samples"])
    if len(z) != w["n_bootstrap_samples"] or len(q_row) != w["n_bootstrap_samples"]:
        raise ValueError(f"backend returned {len(z)} z / {len(q_row)} q, expected {w['n_bootstrap_samples']}")
    bounds = {}
    for arm in w["arms"]:
        if arm["restart"] == "every_time":
            bounds[f"bound__{arm['name']}"] = everytime_lambda_mixture_upper_confidence_trajectory(
                z, arm["lambda_values"], arm["lambda_weights"], w["alpha"],
                w["binary_search_iterations"], WEIGHTING_SCHEME,
            )
        elif arm["restart"] == "none":
            bounds[f"bound__{arm['name']}"] = single_process_lambda_mixture_upper_confidence_trajectory(
                z, arm["lambda_values"], arm["lambda_weights"], w["alpha"], w["binary_search_iterations"],
            )
        else:
            raise ValueError(f"unknown restart schedule {arm['restart']!r}")
    return np.asarray(z, dtype=np.int8), q_row, bounds


def run_shard(*, discovery: dict, label_name: str, n_bootstrap_samples: int, n_trials: int,
              n_workers: int, seed: int, alpha: float, binary_search_iterations: int, arms: list[dict],
              task_index: int, n_tasks: int, output_dir: Path) -> dict:
    require_probability(alpha, "alpha")
    for name, value in [("n_bootstrap_samples", n_bootstrap_samples), ("n_trials", n_trials),
                        ("n_workers", n_workers), ("binary_search_iterations", binary_search_iterations)]:
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    resolved_arms = resolve_arms(arms)
    backend = get_backend(discovery)
    trial_indices = shard_trial_indices(n_trials, task_index, n_tasks)

    shard_dir = output_dir / f"task_{task_index:03d}"
    npz_path = shard_dir / "trajectories.npz"
    summary_path = shard_dir / "summary.json"
    if npz_path.exists() or summary_path.exists():
        raise FileExistsError(f"shard output already exists, refusing to overwrite: {shard_dir}")

    discovery_state = backend["prepare"](discovery)
    worker_state = dict(
        discovery_state=discovery_state, draw=backend["draw"],
        n_bootstrap_samples=n_bootstrap_samples, alpha=alpha,
        binary_search_iterations=binary_search_iterations, arms=resolved_arms,
    )

    all_seeds = np.random.SeedSequence(seed).spawn(n_trials)
    shard_seeds = [all_seeds[i] for i in trial_indices]
    n_shard = len(trial_indices)
    z_matrix = np.empty((n_shard, n_bootstrap_samples), dtype=np.int8)
    q_matrix = np.empty((n_shard, n_bootstrap_samples), dtype=np.float64)
    bound_matrices = {f"bound__{arm['name']}": np.empty((n_shard, n_bootstrap_samples), dtype=np.float64)
                      for arm in resolved_arms}

    print(f"task {task_index}/{n_tasks}: trials {trial_indices[0]}..{trial_indices[-1]} "
          f"({n_shard}) x {n_bootstrap_samples} draws, {n_workers} worker(s), "
          f"arms {[a['name'] for a in resolved_arms]}", flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(worker_state,)) as executor:
        for row, (z_row, q_row, bounds) in enumerate(executor.map(_run_one_trial, shard_seeds, chunksize=1)):
            z_matrix[row] = z_row
            q_matrix[row] = q_row
            for name, trajectory in bounds.items():
                bound_matrices[name][row] = trajectory
            print(f"  trial {row + 1}/{n_shard} done, {time.time() - started:.0f}s elapsed", flush=True)

    for name, matrix in bound_matrices.items():
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0) or np.any(matrix > 1.0):
            raise FloatingPointError(f"arm {name!r} has bounds outside [0, 1] or non-finite")

    metadata = {
        "label_name": label_name,
        "discovery": backend["describe"](discovery, discovery_state),
        "weighting_scheme": WEIGHTING_SCHEME,
        "n_bootstrap_samples": n_bootstrap_samples,
        "n_trials": n_trials,
        "seed": seed,
        "alpha": alpha,
        "binary_search_iterations": binary_search_iterations,
        "task_index": task_index,
        "n_tasks": n_tasks,
        "trial_indices": [int(trial_indices[0]), int(trial_indices[-1])],
        "arms": [{"name": a["name"], "restart": a["restart"], "lambda_values": a["lambda_values"].tolist(),
                  "lambda_weights": a["lambda_weights"].tolist()} for a in resolved_arms],
        "wall_seconds": time.time() - started,
    }
    shard_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        npz_path,
        metadata_json=np.asarray(json.dumps(metadata)),
        trial_indices=trial_indices,
        z_matrix=z_matrix,
        q_matrix=q_matrix,
        **bound_matrices,
    )
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"-> {npz_path} ({metadata['wall_seconds']:.0f}s)", flush=True)
    return metadata


CONFIG_KEYS = {
    "discovery", "label_name", "n_bootstrap_samples", "n_trials", "n_workers", "seed",
    "alpha", "binary_search_iterations", "arms",
}
CONFIG_PATH_KEYS: set[str] = set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--n-tasks", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-trials-override", type=int, default=None,
                        help="smoke tests only: run fewer trials than the config. Trial i still gets the same "
                             "seed (SeedSequence children depend only on their index).")
    args = parser.parse_args()

    config = load_run_config(args.config, CONFIG_KEYS, CONFIG_PATH_KEYS)
    if args.n_trials_override is not None:
        config["n_trials"] = args.n_trials_override
    run_shard(**config, task_index=args.task_index, n_tasks=args.n_tasks, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
