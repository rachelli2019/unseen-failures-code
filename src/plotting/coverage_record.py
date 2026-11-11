"""Save exact per-method coverage and provenance beside a generated figure."""
import json
from pathlib import Path

import numpy as np

from build_coverage_index import relative, validate


def write_record(plot, setting, q, arms, provenance="recorded during plotting"):
    """arms contains (method, source_npz, array_key, bound_matrix, target)."""
    q = np.asarray(q)
    if q.ndim != 2 or not q.size or not np.isfinite(q).all():
        raise ValueError("Oracle trajectories must be a finite, nonempty trial/time matrix")
    n, horizon = q.shape
    records = []
    for method, source, key, bound, target in arms:
        bound = np.asarray(bound)
        if bound.shape != q.shape or not np.isfinite(bound).all():
            raise ValueError(f"Invalid bound trajectories for {method}")
        violations = int(np.any(q > bound, axis=1).sum())
        coverage = 1 - violations / n
        validate(n, horizon, coverage, violations)
        records.append(dict(method=method, source_npz=relative(Path(source)), array_key=key,
                            coverage=coverage, violation_count=violations, target_coverage=target))
    plot = Path(plot)
    if not plot.exists():
        raise ValueError(f"Cannot record coverage for missing plot: {plot}")
    payload = dict(plot_file=plot.name, setting=setting, n_trials=n, horizon=horizon,
                   provenance=provenance, arms=records)
    plot.with_name(plot.stem + "_coverage_record.json").write_text(json.dumps(payload, indent=2) + "\n")
