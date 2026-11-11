"""
Bootstrap sampling and unseen-mode indicators for:

- Failure-mode representation: set-valued
- Novelty rule: fixed taxonomy
- Sampling: uniform with replacement from a finite population of sessions

At index t:
- z[t] = 1 iff draw t contains at least one mode unseen before t.
- q[t] = probability that draw t + 1 contains at least one mode unseen
  after observing draws 0, ..., t.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_session_to_failure_modes(
    taxonomy_json: Path,
) -> dict[str, frozenset[str]]:
    """
    Return session_id -> set of failure modes.

    The taxonomy JSON is this project's native clustering/LLM-merge output
    shape used by data/taxonomy/set_valued_test_taxonomy.json -- a non-empty
    `modes` list. Each mode contains a `mode_id` and a non-empty list of
    `member_raw_ids`, where each raw ID has the form
    "<session_id>::<assignment_id>". Unlike the primary (single-mode) loader,
    a session may legitimately appear under more than one mode here --
    that's the whole point of set-valued; what's invalid is the same raw
    ASSIGNMENT id appearing under two different modes.
    """
    data = json.loads(taxonomy_json.read_text())

    modes = data.get("modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError(f"{taxonomy_json} must contain a non-empty 'modes' list")

    failure_modes_by_session: dict[str, set[str]] = defaultdict(set)
    mode_id_by_raw_id: dict[str, str] = {}

    for mode in modes:
        mode_id = mode["mode_id"]
        member_raw_ids = mode["member_raw_ids"]

        if not member_raw_ids:
            raise ValueError(f"Mode {mode_id!r} in {taxonomy_json} has an empty member_raw_ids list")

        for raw_id in member_raw_ids:
            if not isinstance(raw_id, str) or "::" not in raw_id:
                raise ValueError(f"Malformed member_raw_id {raw_id!r} in {taxonomy_json}")

            if raw_id in mode_id_by_raw_id:
                raise ValueError(
                    f"raw_id {raw_id!r} appears under both {mode_id_by_raw_id[raw_id]!r} and "
                    f"{mode_id!r} in {taxonomy_json} -- the same raw assignment cannot belong "
                    "to two modes"
                )
            mode_id_by_raw_id[raw_id] = mode_id

            session_id = raw_id.split("::", 1)[0]
            failure_modes_by_session[session_id].add(mode_id)

    if not failure_modes_by_session:
        raise ValueError(f"{taxonomy_json} resolved to zero sessions")

    return {
        session_id: frozenset(mode_ids)
        for session_id, mode_ids in failure_modes_by_session.items()
    }


def build_population(
    session_to_failure_modes: dict[str, frozenset[str]],
    all_session_ids: list[str],
) -> tuple[list[frozenset[str]], dict[str, np.ndarray]]:
    """
    Build the finite session population in sorted session-ID order.

    Sessions absent from the taxonomy receive an empty failure-mode set.
    """
    if not all_session_ids:
        raise ValueError("all_session_ids must be non-empty")

    session_ids = sorted(all_session_ids)

    failure_modes_by_session_index = [
        session_to_failure_modes.get(session_id, frozenset())
        for session_id in session_ids
    ]

    indices_by_mode: dict[str, list[int]] = defaultdict(list)
    for session_index, mode_ids in enumerate(failure_modes_by_session_index):
        for mode_id in mode_ids:
            indices_by_mode[mode_id].append(session_index)

    mode_to_session_indices = {
        mode_id: np.asarray(indices, dtype=np.int64)
        for mode_id, indices in indices_by_mode.items()
    }

    return failure_modes_by_session_index, mode_to_session_indices


def run_trial(
    bootstrap_session_indices: np.ndarray,
    failure_modes_by_session_index: list[frozenset[str]],
    mode_to_session_indices: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute z and q for one bootstrap sequence.

    z[t] = 1 iff draw t contains a previously unseen failure mode.

    q[t] = probability that draw t + 1 contains a failure mode not observed
    in draws 0, ..., t.
    """
    n_sessions = len(failure_modes_by_session_index)

    if np.any(bootstrap_session_indices < 0) or np.any(
        bootstrap_session_indices >= n_sessions
    ):
        raise ValueError("bootstrap_session_indices contains an invalid index")

    z = np.zeros(len(bootstrap_session_indices), dtype=np.int8)
    q = np.zeros(len(bootstrap_session_indices), dtype=np.float64)

    seen_modes: set[str] = set()

    # Number of still-unseen modes belonging to each population session.
    unseen_count = np.asarray(
        [len(mode_ids) for mode_ids in failure_modes_by_session_index],
        dtype=np.int32,
    )

    for t, session_index in enumerate(bootstrap_session_indices):
        mode_ids = failure_modes_by_session_index[session_index]
        new_modes = mode_ids - seen_modes

        z[t] = bool(new_modes)

        for mode_id in new_modes:
            unseen_count[mode_to_session_indices[mode_id]] -= 1

        seen_modes.update(new_modes)

        # Fraction of population sessions that would reveal something new
        # if sampled next.
        q[t] = np.mean(unseen_count > 0)

    return z, q