#!/usr/bin/env python3
"""Merge the task_*/trajectories.npz shards of run_lambda_grid_mixture.py into one npz.

Every shard must come from the same experiment (identical metadata apart from the
task index, trial range, and wall time), and together the shards must cover trial
indices 0..n_trials-1 exactly once. Trials are placed by their global index, so the
merged arrays are row-for-row what an unsharded run would have produced.

Usage:
    uv run python src/combine_lambda_grid_shards.py \
        --shard-dir artifacts/lambda_grid_mixture/<run_id> \
        --output-npz artifacts/lambda_grid_mixture/<run_id>/combined/trajectories.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PER_SHARD_KEYS = {"task_index", "trial_indices", "wall_seconds"}


def normalize_restart_metadata(metadata: dict, shard: Path) -> dict:
    """Shards written before arms carried their own "restart" key record the schedule once,
    top-level, as "restart_schedule": "every_time" (the only schedule that runner had).
    Rewrite that into the per-arm form so old and new shards of one experiment compare
    equal; anything else is a genuine mismatch and still fails the comparison."""
    if "restart_schedule" not in metadata or "arms" not in metadata:
        # No arms (e.g. closed-form runs): the top-level schedule is the only description.
        return metadata
    if metadata["restart_schedule"] != "every_time":
        raise ValueError(f"shard {shard} has unexpected restart_schedule {metadata['restart_schedule']!r}")
    if any("restart" in arm for arm in metadata["arms"]):
        raise ValueError(f"shard {shard} has both a top-level restart_schedule and per-arm restart keys")
    normalized = {k: v for k, v in metadata.items() if k != "restart_schedule"}
    normalized["arms"] = [{**arm, "restart": "every_time"} for arm in metadata["arms"]]
    return normalized


def combine(shard_dir: Path, output_npz: Path) -> dict:
    shards = sorted(shard_dir.glob("task_*/trajectories.npz"))
    if not shards:
        raise FileNotFoundError(f"no task_*/trajectories.npz shards under {shard_dir}")
    if output_npz.exists():
        raise FileExistsError(f"refusing to overwrite {output_npz}")

    base: dict | None = None
    merged: dict[str, np.ndarray] = {}
    filled: np.ndarray | None = None
    wall_seconds = []

    for shard in shards:
        data = np.load(shard)
        metadata = normalize_restart_metadata(json.loads(str(data["metadata_json"])), shard)
        shared ={k: v for k, v in metadata.items() if k not in PER_SHARD_KEYS}
        if base is None:
            base = shared
            n_trials, horizon = base["n_trials"], base["n_bootstrap_samples"]
            if len(shards) != base["n_tasks"]:
                raise ValueError(f"found {len(shards)} shards but n_tasks={base['n_tasks']}")
            filled = np.zeros(n_trials, dtype=bool)
            for key in data.files:
                if key in ("metadata_json", "trial_indices"):
                    continue
                merged[key] = np.empty((n_trials, horizon), dtype=data[key].dtype)
        elif shared != base:
            diff = sorted(k for k in set(shared) | set(base) if shared.get(k) != base.get(k))
            raise ValueError(f"shard {shard} disagrees with the first shard on {diff}")

        idx = data["trial_indices"]
        if np.any(filled[idx]):
            raise ValueError(f"shard {shard} repeats trial indices already seen")
        filled[idx] = True
        if set(data.files) - {"metadata_json", "trial_indices"} != set(merged):
            raise ValueError(f"shard {shard} has arrays {sorted(data.files)}, expected {sorted(merged)}")
        for key in merged:
            merged[key][idx] = data[key]
        wall_seconds.append(metadata["wall_seconds"])

    if not np.all(filled):
        raise ValueError(f"missing trials: {np.flatnonzero(~filled)[:20].tolist()} ...")

    result = dict(base, n_shards=len(shards), shard_wall_seconds_max=max(wall_seconds),
                  shard_wall_seconds_total=sum(wall_seconds))
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, metadata_json=np.asarray(json.dumps(result)), **merged)
    output_npz.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"-> {output_npz}: {len(shards)} shards, {base['n_trials']} trials, arrays {sorted(merged)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    args = parser.parse_args()
    combine(args.shard_dir, args.output_npz)


if __name__ == "__main__":
    main()
