"""Load the finite reference population the real-data evaluations sample from.

The population file (data/population/test_population.json, built by
scripts/build_evaluation_population.py) lists every session a bootstrap draw
can land on: successes as well as failed sessions. A taxonomy only covers the
sessions with a failure mode, so it cannot define the population by itself --
every runner takes the population explicitly and the failure-mode sources must
be a subset of it.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_population_session_ids(population_json: Path) -> list[str]:
    """Return the population's session IDs, sorted. Raises on a malformed file."""
    if not population_json.exists():
        raise FileNotFoundError(f"population_json does not exist: {population_json}")
    data = json.loads(population_json.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{population_json} must contain a JSON object")
    session_ids = data.get("session_ids")
    if not isinstance(session_ids, list) or not session_ids:
        raise ValueError(f"{population_json} must contain a non-empty 'session_ids' list")
    if not all(isinstance(s, str) and s for s in session_ids):
        raise ValueError(f"{population_json} 'session_ids' must be non-empty strings")
    if len(set(session_ids)) != len(session_ids):
        raise ValueError(f"{population_json} contains duplicate session IDs")
    if data.get("n_sessions") != len(session_ids):
        raise ValueError(
            f"{population_json} declares n_sessions={data.get('n_sessions')!r} but lists "
            f"{len(session_ids)} session IDs"
        )
    return sorted(session_ids)


def require_within_population(failure_session_ids, population_session_ids, source: str) -> None:
    """Every session with a failure mode must be a member of the population."""
    outside = set(failure_session_ids) - set(population_session_ids)
    if outside:
        raise ValueError(
            f"{len(outside)} session(s) in {source} are not in the evaluation population, "
            f"e.g. {sorted(outside)[0]!r}"
        )
