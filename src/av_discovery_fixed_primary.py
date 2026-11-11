"""
Bootstrap sampling and unseen-mode indicators for:

- Failure-mode representation: one failure mode per session
- Novelty rule: fixed taxonomy
- Sampling: uniform with replacement from a finite population of sessions

Bootstrap session indices are generated externally by the run script and
passed into the functions in this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Sentinel mode_id for a session with no resolved failure mode at all -- a
# success, not a failure. It is always non-novel (novelty_indicator[t]=0),
# excluded from singleton/missing-mass bookkeeping entirely, but still
# counted in the population denominator (n_sessions, and the fraction of
# draws it can consume). See build_session_population, compute_discovery_
# statistics, and compute_true_missing_mass.
SUCCESS_MODE_ID = "__success__"


def load_session_to_failure_mode(
    taxonomy_json: Path,
) -> dict[str, str]:
    """
    Load a fixed taxonomy -- this project's native clustering/LLM-merge
    output shape used by data/taxonomy/primary_test_taxonomy.json
    or _llm -- and map each session ID to its single failure mode.

    The taxonomy JSON must contain a non-empty `modes` list. Each mode
    contains a `mode_id` and a non-empty list of `member_session_ids`
    (already one session_id per session; no raw-assignment reduction needed
    for this representation).

    This function enforces the single-valued setting: a session cannot appear
    under more than one failure mode.
    """
    data = json.loads(taxonomy_json.read_text())

    modes = data.get("modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError(
            f"{taxonomy_json} must contain a non-empty 'modes' list"
        )

    session_to_failure_mode: dict[str, str] = {}

    for mode in modes:
        mode_id = mode["mode_id"]
        if mode_id == SUCCESS_MODE_ID:
            raise ValueError(
                f"{taxonomy_json} defines a real failure mode named {SUCCESS_MODE_ID!r}, "
                "which collides with the reserved sentinel for sessions with no failure mode"
            )
        member_session_ids = mode["member_session_ids"]

        if not member_session_ids:
            raise ValueError(
                f"Mode {mode_id!r} in {taxonomy_json} "
                "has an empty member_session_ids list"
            )
        if len(member_session_ids) != len(set(member_session_ids)):
            raise ValueError(
                f"Mode {mode_id!r} in {taxonomy_json} has duplicate session_ids "
                "-- not a primary (one-mode-per-session) taxonomy?"
            )

        for session_id in member_session_ids:
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(
                    f"Malformed member_session_id {session_id!r} in {taxonomy_json}"
                )

            if session_id in session_to_failure_mode:
                previous_mode = session_to_failure_mode[session_id]
                raise ValueError(
                    f"Session {session_id!r} appears more than once in "
                    f"{taxonomy_json}: previously assigned to "
                    f"{previous_mode!r}, encountered again under {mode_id!r}. "
                    "Expected one failure mode per session."
                )

            session_to_failure_mode[session_id] = mode_id

    if not session_to_failure_mode:
        raise ValueError(
            f"{taxonomy_json} resolved to zero sessions"
        )

    return session_to_failure_mode


def build_session_population(
    session_to_failure_mode: dict[str, str],
    all_session_ids: list[str],
) -> tuple[np.ndarray, dict[str, float], int]:
    """
    Construct the finite session population used for bootstrap sampling.

    all_session_ids is the full population (e.g. from session_population.
    load_all_session_ids) -- every session in it is included, whether or not
    it has a resolved failure mode. A session absent from
    session_to_failure_mode is a SUCCESS, not a failure: it is assigned the
    SUCCESS_MODE_ID sentinel, which compute_discovery_statistics and
    compute_true_missing_mass both treat specially (see their docstrings).

    Sessions are ordered deterministically by sorted session ID.

    Returns
    -------
    failure_mode_by_session_index:
        failure_mode_by_session_index[i] is the failure mode (or
        SUCCESS_MODE_ID) assigned to session i in the finite population.

    failure_mode_probability:
        Probability of each REAL failure mode (SUCCESS_MODE_ID excluded)
        under uniform sampling with replacement from the finite session
        population. These probabilities sum to less than 1 whenever any
        session is a success -- the remainder is the success probability,
        which is never "missing" mass to begin with.

    n_sessions:
        Number of sessions in the finite population (successes included).
    """
    if not all_session_ids:
        raise ValueError("all_session_ids is empty")

    if SUCCESS_MODE_ID in session_to_failure_mode.values():
        raise ValueError(
            f"session_to_failure_mode assigns a real session to {SUCCESS_MODE_ID!r}, "
            "which collides with the reserved sentinel for sessions with no failure mode"
        )

    unresolved = set(session_to_failure_mode) - set(all_session_ids)
    if unresolved:
        raise ValueError(
            f"session_to_failure_mode contains {len(unresolved)} session(s) not present in "
            f"all_session_ids, e.g. {next(iter(unresolved))!r} -- the taxonomy and the "
            "declared population have diverged"
        )

    session_ids = sorted(all_session_ids)
    n_sessions = len(session_ids)

    failure_mode_by_session_index = np.asarray(
        [
            session_to_failure_mode.get(session_id, SUCCESS_MODE_ID)
            for session_id in session_ids
        ]
    )

    failure_mode_counts: dict[str, int] = {}
    for mode_id in failure_mode_by_session_index:
        if mode_id == SUCCESS_MODE_ID:
            continue
        failure_mode_counts[mode_id] = (
            failure_mode_counts.get(mode_id, 0) + 1
        )

    failure_mode_probability = {
        mode_id: count / n_sessions
        for mode_id, count in failure_mode_counts.items()
    }

    return (
        failure_mode_by_session_index,
        failure_mode_probability,
        n_sessions,
    )


def validate_bootstrap_indices(
    bootstrap_session_indices: np.ndarray,
    n_sessions: int,
) -> None:
    """
    Validate externally generated bootstrap indices.

    Each entry must be an integer index into the finite session population.
    Repeated indices are allowed because bootstrap sampling is with replacement.
    """
    if bootstrap_session_indices.ndim != 1:
        raise ValueError(
            "bootstrap_session_indices must be one-dimensional"
        )

    if not np.issubdtype(
        bootstrap_session_indices.dtype,
        np.integer,
    ):
        raise TypeError(
            "bootstrap_session_indices must contain integer indices"
        )

    if np.any(bootstrap_session_indices < 0):
        raise ValueError(
            "bootstrap_session_indices contains negative indices"
        )

    if np.any(bootstrap_session_indices >= n_sessions):
        raise ValueError(
            "bootstrap_session_indices contains an index outside the "
            f"session population of size {n_sessions}"
        )


def compute_discovery_statistics(
    bootstrap_session_indices: np.ndarray,
    failure_mode_by_session_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute novelty indicators and Good-Turing estimates for one bootstrap run.

    All indexing is 0-based.

    Parameters
    ----------
    bootstrap_session_indices:
        Sequence of session-population indices generated by the run script.

    failure_mode_by_session_index:
        Mapping from each session-population index to its fixed failure mode.

    Returns
    -------
    novelty_indicator:
        novelty_indicator[t] = 1 iff draw t contains a failure mode not
        observed in draws 0, ..., t - 1.

    good_turing_estimate:
        good_turing_estimate[t] is the Good-Turing missing-mass estimate
        after observing draws 0, ..., t:

            n_singletons / (t + 1),

        where n_singletons is the number of failure modes observed exactly
        once among draws 0, ..., t.

    A draw of SUCCESS_MODE_ID (see build_session_population) is never novel
    and never touches the singleton bookkeeping -- it isn't a failure mode.
    """
    n_draws = len(bootstrap_session_indices)

    novelty_indicator = np.zeros(n_draws, dtype=np.int8)
    good_turing_estimate = np.zeros(n_draws, dtype=np.float64)

    failure_mode_draw_counts: dict[str, int] = {}
    n_singleton_modes = 0

    for t, session_index in enumerate(bootstrap_session_indices):
        mode_id = failure_mode_by_session_index[session_index]

        if mode_id == SUCCESS_MODE_ID:
            # novelty_indicator[t] stays 0; n_singleton_modes is untouched.
            good_turing_estimate[t] = n_singleton_modes / (t + 1)
            continue

        previous_count = failure_mode_draw_counts.get(mode_id, 0)

        if previous_count == 0:
            # First observation of this failure mode.
            novelty_indicator[t] = 1
            failure_mode_draw_counts[mode_id] = 1
            n_singleton_modes += 1

        else:
            # On its second observation, a mode stops being a singleton.
            if previous_count == 1:
                n_singleton_modes -= 1

            failure_mode_draw_counts[mode_id] = previous_count + 1

        good_turing_estimate[t] = n_singleton_modes / (t + 1)
      # use t+1 since t is zero-indexed

    return novelty_indicator, good_turing_estimate


def compute_true_missing_mass(
    bootstrap_session_indices: np.ndarray,
    failure_mode_by_session_index: np.ndarray,
    failure_mode_probability: dict[str, float],
) -> np.ndarray:
    """
    Compute the true missing mass after each draw in one bootstrap run.

    true_missing_mass[t] is the probability that the next draw, t + 1,
    contains a failure mode not observed in draws 0, ..., t.

    Probabilities are defined by uniform sampling with replacement from the
    finite session population. A failure mode's probability is therefore the
    fraction of population sessions assigned to that mode.

    A draw of SUCCESS_MODE_ID (see build_session_population) never reduces
    the missing mass: it was never missing failure content to begin with,
    so remaining_missing_mass starts at the total REAL failure-mode
    probability (failure_mode_probability excludes SUCCESS_MODE_ID by
    construction), not at 1.0.
    """
    n_draws = len(bootstrap_session_indices)
    true_missing_mass = np.zeros(n_draws, dtype=np.float64)

    observed_failure_modes: set[str] = set()
    remaining_missing_mass = sum(failure_mode_probability.values())

    for t, session_index in enumerate(bootstrap_session_indices):
        mode_id = failure_mode_by_session_index[session_index]

        if mode_id != SUCCESS_MODE_ID and mode_id not in observed_failure_modes:
            observed_failure_modes.add(mode_id)
            remaining_missing_mass -= failure_mode_probability[mode_id]

        # Guard against tiny negative values from floating-point arithmetic.
        true_missing_mass[t] = max(
            0.0,
            remaining_missing_mass,
        )

    return true_missing_mass


def run_bootstrap_trial(
    bootstrap_session_indices: np.ndarray,
    failure_mode_by_session_index: np.ndarray,
    failure_mode_probability: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute unseen-mode statistics for one externally generated bootstrap run.

    Returns
    -------
    novelty_indicator:
        novelty_indicator[t] indicates whether draw t discovers a failure
        mode not observed in draws 0, ..., t - 1.

    good_turing_estimate:
        good_turing_estimate[t] is the Good-Turing estimate after observing
        draws 0, ..., t.

    true_missing_mass:
        true_missing_mass[t] is the true probability that the next draw,
        t + 1, discovers a failure mode not observed in draws 0, ..., t.
    """
    validate_bootstrap_indices(
        bootstrap_session_indices,
        n_sessions=len(failure_mode_by_session_index),
    )

    novelty_indicator, good_turing_estimate = compute_discovery_statistics(
        bootstrap_session_indices,
        failure_mode_by_session_index,
    )

    true_missing_mass = compute_true_missing_mass(
        bootstrap_session_indices,
        failure_mode_by_session_index,
        failure_mode_probability,
    )

    return (
        novelty_indicator,
        good_turing_estimate,
        true_missing_mass,
    )