"""
Evaluation metrics:
- Mean trajectory + percentile bands across trials
- Cumulative simultaneous coverage
- Gap between upper bound and true missing mass
- Full-trajectory simultaneous coverage
- Stopping times at fixed epsilon thresholds
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _require_nonempty_2d(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")

    return matrix


def _require_matching_matrices(
    estimate_matrix: np.ndarray,
    q_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Matrices have shape (n_trials, n_sessions).

    Row i of estimate_matrix and q_matrix must correspond to the same trial
    and column t to the same timestep.
    """
    estimate_matrix = _require_nonempty_2d(
        estimate_matrix, "estimate_matrix"
    )
    q_matrix = _require_nonempty_2d(q_matrix, "q_matrix")

    if estimate_matrix.shape != q_matrix.shape:
        raise ValueError(
            "estimate_matrix and q_matrix must have the same shape; "
            f"got {estimate_matrix.shape} and {q_matrix.shape}"
        )

    return estimate_matrix, q_matrix


# ---------------------------------------------------------------------------
# mean trajectories + across-trial percentile bands
# ---------------------------------------------------------------------------

def trajectory_summary(
    matrix: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pointwise mean and percentile band across trials.

    Defaults to the 5th--95th percentile band. This describes variation
    across trials; it is not a confidence interval on the mean.
    """
    matrix = _require_nonempty_2d(matrix, "matrix")

    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError(
            "require 0 <= lower_quantile < upper_quantile <= 1"
        )

    mean = np.mean(matrix, axis=0, dtype=np.float64)
    lower = np.quantile(matrix, lower_quantile, axis=0)
    upper = np.quantile(matrix, upper_quantile, axis=0)

    return mean, lower, upper


# ---------------------------------------------------------------------------
# cumulative simultaneous coverage
# ---------------------------------------------------------------------------

def cumulative_coverage_rate(
    estimate_matrix: np.ndarray,
    q_matrix: np.ndarray,
) -> np.ndarray:
    """
    Empirical probability of no violation through timestep t:

        Pr(q_s <= U_s for every s <= t).

    The comparison is performed separately within each trial, so q may have
    a different trajectory for every trial.
    """
    estimate_matrix, q_matrix = _require_matching_matrices(
        estimate_matrix,
        q_matrix,
    )

    violated = q_matrix > estimate_matrix
    ever_violated = np.maximum.accumulate(violated, axis=1)

    return 1.0 - np.mean(
        ever_violated,
        axis=0,
        dtype=np.float64,
    )


def simultaneous_violation_rate(
    estimate_matrix: np.ndarray,
    q_matrix: np.ndarray,
) -> float:
    """Fraction of trials with at least one violation over the full trajectory."""
    estimate_matrix, q_matrix = _require_matching_matrices(
        estimate_matrix,
        q_matrix,
    )

    violated = q_matrix > estimate_matrix
    return float(np.mean(np.any(violated, axis=1)))


# ---------------------------------------------------------------------------
# gap to true missing mass
# ---------------------------------------------------------------------------

def gap_to_truth(
    estimate_matrix: np.ndarray,
    q_matrix: np.ndarray,
) -> np.ndarray:
    """
    Signed gap U_t - q_t for every trial and timestep.

    Positive = conservative.
    Negative = bound violation.

    Because the gap can be negative, use a symlog rather than ordinary
    log y-axis if violations are shown.
    """
    estimate_matrix, q_matrix = _require_matching_matrices(
        estimate_matrix,
        q_matrix,
    )

    return estimate_matrix - q_matrix


# ---------------------------------------------------------------------------
# Stopping times: main table + taxonomy sensitivity
# ---------------------------------------------------------------------------

def stopping_time_summary(
    bound_matrix: np.ndarray,
    stopping_epsilons: list[float],
) -> dict[str, dict]:
    """
    Summarize

        tau(epsilon) = inf{t : U_t < epsilon}

    across trials.

    Reports median/IQR stopping time among trials that stop, along with the
    fraction of trials that stop within the observed horizon.
    """
    bound_matrix = _require_nonempty_2d(
        bound_matrix,
        "bound_matrix",
    )

    if len(stopping_epsilons) != len(set(stopping_epsilons)):
        raise ValueError("stopping_epsilons must be unique")

    n_trials, n_sessions = bound_matrix.shape
    summary = {}

    for epsilon in stopping_epsilons:
        if not 0.0 < epsilon < 1.0:
            raise ValueError(
                f"stopping epsilon must be in (0, 1); got {epsilon}"
            )

        crossed = bound_matrix < epsilon
        did_cross = np.any(crossed, axis=1)

        # argmax returns the first True. Its value is ignored for trials
        # that never cross.
        first_crossing = np.argmax(crossed, axis=1)

        taus = first_crossing[did_cross] + 1
        n_stopped = int(np.sum(did_cross))

        if n_stopped:
            q25, median, q75 = np.quantile(
                taus,
                [0.25, 0.50, 0.75],
            )
            q25 = float(q25)
            median = float(median)
            q75 = float(q75)
        else:
            q25 = median = q75 = None

        summary[str(epsilon)] = {
            "median_tau": median,
            "tau_q25": q25,
            "tau_q75": q75,
            "fraction_stopped_by_n": n_stopped / n_trials,
            "n_stopped": n_stopped,
            "n_trials": n_trials,
            "n_sessions": n_sessions,
        }

    return summary


# ---------------------------------------------------------------------------
# Main comparison table
# ---------------------------------------------------------------------------

def method_summary(
    name: str,
    bound_matrix: np.ndarray,
    q_matrix: np.ndarray,
    stopping_epsilons: list[float],
) -> dict:
    """
    Metrics reported in the main method-comparison table:
    - full-trajectory simultaneous violation/coverage
    - stopping time at each epsilon
    """
    bound_matrix, q_matrix = _require_matching_matrices(
        bound_matrix,
        q_matrix,
    )

    violation_rate = simultaneous_violation_rate(
        bound_matrix,
        q_matrix,
    )

    return {
        "name": name,
        "simultaneous_coverage_rate": 1.0 - violation_rate,
        "simultaneous_violation_rate": violation_rate,
        "stopping_times": stopping_time_summary(
            bound_matrix,
            stopping_epsilons,
        ),
    }