#!/usr/bin/env python3
"""Merge the per-task shards of an array-parallelised Bernoulli run into one npz.

run_bernoulli_simulation.py runs a fixed number of trials in one process. To use
more than one node, the array job splits the trial budget across tasks that each
run a distinct seed, and this script concatenates their trajectories back into a
single file with exactly the shape a single large run would have produced.

Trials are i.i.d. and each shard uses its own seed, so concatenating along the
trial axis is the same as having run them together -- there is nothing to
reconcile across shards but the trial count.

Writes `trajectories.npz` (ucb_matrix, coverage) and `summary.json` in the layout
plot_bernoulli_coverage.py expects.

Usage:
    uv run python src/combine_bernoulli_shards.py \
        --shard-dir artifacts/bernoulli/h10000_t10000 \
        --output-dir artifacts/bernoulli/h10000_t10000/combined
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def combine(shard_dir: Path, output_dir: Path) -> dict:
    shards = sorted(shard_dir.glob("task_*/trajectories.npz"))
    if not shards:
        raise FileNotFoundError(f"no task_*/trajectories.npz shards under {shard_dir}")

    matrices: list[np.ndarray] = []
    horizon: int | None = None
    base: dict | None = None
    seeds: list[int] = []

    for shard in shards:
        summary_path = shard.parent / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"shard is missing summary.json: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        data = np.load(shard)
        if "ucb_matrix" not in data.files:
            raise ValueError(f"shard is missing 'ucb_matrix': {shard}")
        ucb = np.asarray(data["ucb_matrix"], dtype=np.float64)

        if horizon is None:
            horizon = ucb.shape[1]
            base = summary
        elif ucb.shape[1] != horizon:
            raise ValueError(f"shard {shard} has horizon {ucb.shape[1]}, expected {horizon}")

        # Every shard must agree on the experiment it is a shard OF; only the seed
        # and the trial count are allowed to differ.
        for key in ("p", "alpha", "lambda_value", "restart_schedule", "weighting_scheme"):
            if summary[key] != base[key]:
                raise ValueError(
                    f"shard {shard} disagrees on {key!r}: {summary[key]!r} vs {base[key]!r}"
                )
        if summary["seed"] in seeds:
            raise ValueError(f"duplicate seed {summary['seed']} -- shards would not be independent")
        seeds.append(summary["seed"])
        matrices.append(ucb)

    ucb_matrix = np.concatenate(matrices, axis=0)
    p = base["p"]
    violated = ucb_matrix < p
    coverage = (~np.logical_or.accumulate(violated, axis=1)).mean(axis=0)

    summary = dict(base)
    summary.update(
        n_trials=int(ucb_matrix.shape[0]),
        horizon=int(horizon),
        seed=int(min(seeds)),           # representative; shard_seeds carries the full set
        shard_seeds=sorted(int(s) for s in seeds),
        n_shards=len(matrices),
        mean_ucb_final=float(ucb_matrix[:, -1].mean()),
        simultaneous_coverage_final=float(coverage[-1]),
    )
    summary.pop("output_npz", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "trajectories.npz",
        ucb_matrix=ucb_matrix,
        coverage=coverage,
        q_t=np.full(horizon, p),
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = combine(args.shard_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
