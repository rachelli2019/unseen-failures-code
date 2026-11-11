"""
Averages over a finite grid of lambda values in (0,1) to approximate uniform mixture over lambda in [0,1]. 

For a fixed lambda, av_core.py already computes the restart-mixture process. 
- sum_s w_s E_{s:t}^{(lambda_k)}(u) + tail_mass 

This script averages over those restart mixtures using uniform weighting on the finite lambda grid.
- M_t(u) = sum_k pi_k M_t^(lambda_k)(u).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from av_core import (  # noqa: E402
    compile_restart_mixture,
    evaluate_restart_mixture_log_eprocess,
    require_binary_z,
    require_lambda,
    require_probability,
    restart_mixture_tail_mass,
    restart_mixture_weights_for_scheme,
)

# A tuning-free default grid: several lambda values weighted equally, so no
# single lambda has to be chosen in advance.
DEFAULT_LAMBDA_GRID = (0.10, 0.25, 0.50, 0.75)
DEFAULT_LAMBDA_WEIGHTS = tuple(1.0 / len(DEFAULT_LAMBDA_GRID) for _ in DEFAULT_LAMBDA_GRID)


def require_lambda_weights(lambda_values: np.ndarray, pi_weights: np.ndarray) -> None:
    if len(lambda_values) == 0:
        raise ValueError("lambda_values must be non-empty")
    if len(pi_weights) != len(lambda_values):
        raise ValueError(
            f"pi_weights and lambda_values must have the same length; "
            f"got {len(pi_weights)!r} and {len(lambda_values)!r}"
        )
    if np.any(pi_weights <= 0):
        raise ValueError("pi_weights must be strictly positive")
    if not math.isclose(float(pi_weights.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"pi_weights must sum to 1; got {pi_weights.sum()!r}")


def _evaluate_lambda_restart_mixture_log_eprocess(
    mixture,
    lambda_values: np.ndarray,
    log_pi: np.ndarray,
    p: np.ndarray,
    per_lambda_workspace: np.ndarray,
    restart_workspace: np.ndarray,
) -> np.ndarray:
    """Evaluate the lambda mixture using one prepared restart mixture."""
    for i, lambda_value in enumerate(lambda_values):
        per_lambda_workspace[i] = evaluate_restart_mixture_log_eprocess(
            mixture, float(lambda_value), p, restart_workspace
        )

    per_lambda_workspace += log_pi[:, None]
    max_over_lambda = np.max(per_lambda_workspace, axis=0)
    np.subtract(per_lambda_workspace, max_over_lambda[None, :], out=per_lambda_workspace)
    np.exp(per_lambda_workspace, out=per_lambda_workspace)
    log_mix = max_over_lambda + np.log(np.sum(per_lambda_workspace, axis=0))

    if not np.all(np.isfinite(log_mix)):
        raise FloatingPointError("non-finite lambda-restart-mixture e-process values during inversion")
    return log_mix


def _prepare_lambda_restart_mixture(
    z: np.ndarray,
    lambda_values: np.ndarray,
    pi_weights: np.ndarray,
    starts: np.ndarray,
    weighting_scheme: str,
):
    """Validate the inputs both public entry points share and compile the mixture.

    Single source of validation so the two entry points cannot drift apart in what
    they accept. The restart-schedule checks are asserted here as well as inside
    compile_restart_mixture: that keeps the error attributable to the caller's own
    argument, and keeps these functions' contract independent of a downstream helper.
    """
    z = require_binary_z(z)
    lambda_values = np.asarray(lambda_values, dtype=float)
    pi_weights = np.asarray(pi_weights, dtype=float)
    require_lambda_weights(lambda_values, pi_weights)
    for lv in lambda_values:
        require_lambda(float(lv))

    starts = np.asarray(starts)
    if starts.ndim != 1 or len(starts) == 0:
        raise ValueError("starts must be a non-empty 1D array")
    if not np.all(np.diff(starts) > 0):
        raise ValueError("starts must be strictly increasing")
    if starts[0] < 1 or starts[-1] > len(z):
        raise ValueError(f"starts must be within [1, {len(z)}]; got range [{starts[0]!r}, {starts[-1]!r}]")

    weights, total_mass = restart_mixture_weights_for_scheme(weighting_scheme, len(starts))
    tail_mass = restart_mixture_tail_mass(weights, total_mass)
    mixture = compile_restart_mixture(z, starts, weights, tail_mass)

    log_pi = np.log(pi_weights)
    per_lambda_workspace = np.empty((len(lambda_values), len(z)), dtype=np.float64)
    restart_workspace = np.empty_like(mixture.zeros_in_range)
    return z, lambda_values, mixture, log_pi, per_lambda_workspace, restart_workspace


def lambda_restart_mixture_log_eprocess_at_p(
    z: np.ndarray,
    lambda_values: np.ndarray,
    pi_weights: np.ndarray,
    p: np.ndarray,
    starts: np.ndarray,
    weighting_scheme: str,
) -> np.ndarray:
    """log M_t(p) for the lambda-and-restart double mixture -- see module
    docstring for the derivation showing this is just a pi_k-weighted,
    log-space average of av_core's single-lambda restart mixtures."""
    z, lambda_values, mixture, log_pi, per_lambda_workspace, restart_workspace = (
        _prepare_lambda_restart_mixture(z, lambda_values, pi_weights, starts, weighting_scheme)
    )
    return _evaluate_lambda_restart_mixture_log_eprocess(
        mixture, lambda_values, log_pi, p, per_lambda_workspace, restart_workspace
    )


def lambda_restart_mixture_upper_confidence_trajectory(
    z: np.ndarray,
    lambda_values: np.ndarray,
    pi_weights: np.ndarray,
    alpha: float,
    binary_search_iterations: int,
    starts: np.ndarray,
    weighting_scheme: str,
) -> np.ndarray:
    require_probability(alpha, "alpha")
    if binary_search_iterations <= 0:
        raise ValueError("binary_search_iterations must be positive")
    z, lambda_values, mixture, log_pi, per_lambda_workspace, restart_workspace = (
        _prepare_lambda_restart_mixture(z, lambda_values, pi_weights, starts, weighting_scheme)
    )

    boundary = math.log(1.0 / alpha)
    n_times = len(z)
    lower = np.zeros(n_times, dtype=float)
    upper = np.ones(n_times, dtype=float)

    log_at_one = _evaluate_lambda_restart_mixture_log_eprocess(
        mixture, lambda_values, log_pi, upper, per_lambda_workspace, restart_workspace
    )
    one_is_valid = log_at_one < boundary
    lower[one_is_valid] = 1.0

    active = ~one_is_valid
    for _ in range(binary_search_iterations):
        midpoint = (lower + upper) / 2.0
        log_at_midpoint = _evaluate_lambda_restart_mixture_log_eprocess(
            mixture, lambda_values, log_pi, midpoint, per_lambda_workspace, restart_workspace
        )
        midpoint_is_valid = log_at_midpoint < boundary
        move_lower = active & midpoint_is_valid
        move_upper = active & ~midpoint_is_valid
        lower[move_lower] = midpoint[move_lower]
        upper[move_upper] = midpoint[move_upper]

    if not np.all(np.isfinite(upper)):
        raise FloatingPointError("non-finite lambda-restart-mixture UCB trajectory")
    # Return the upper bracket endpoint so finite-iteration bisection rounds
    # outward, keeping the reported trajectory conservative as an upper bound.
    return upper


# ---------------------------------------------------------------------------
# Fine uniform lambda grid, every-time restart schedule (fast path)
# ---------------------------------------------------------------------------

def uniform_lambda_grid(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint grid lambda_k = (k - 1/2) / K with equal weights 1/K.

    A discrete stand-in for lambda ~ Uniform[0, 1]. Validity does not depend on
    how well it approximates the integral -- any positive weights summing to 1
    give a valid mixture -- the grid only controls how close the bound is to the
    continuous-uniform one. Midpoints keep every lambda strictly inside (0, 1).
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be positive; got {n_points!r}")
    lambda_values = (np.arange(n_points, dtype=np.float64) + 0.5) / n_points
    return lambda_values, np.full(n_points, 1.0 / n_points)


def compile_everytime_restart_mixture(z: np.ndarray, weighting_scheme: str):
    """Restart-by-time counts for the every-time schedule (starts = 1..n), plus
    not_started_mass[c] = sum_{s > c} w_s + tail_mass: the capital of every
    restart that has not begun by time index c (each has E = 1 there)."""
    z = require_binary_z(z)
    starts = np.arange(1, len(z) + 1, dtype=np.int64)
    weights, total_mass = restart_mixture_weights_for_scheme(weighting_scheme, len(starts))
    tail_mass = restart_mixture_tail_mass(weights, total_mass)
    mixture = compile_restart_mixture(z, starts, weights, tail_mass)
    suffix = np.concatenate([np.cumsum(weights[::-1])[::-1], [0.0]])
    not_started_mass = suffix + tail_mass
    return mixture, not_started_mass


def _everytime_lambda_mixture_log_eprocess(
    mixture,
    not_started_mass: np.ndarray,
    lambda_values: np.ndarray,
    log_pi: np.ndarray,
    p: np.ndarray,
    block_size: int,
    per_lambda_workspace: np.ndarray,
    block_workspace: np.ndarray,
) -> np.ndarray:
    """Same quantity as _evaluate_lambda_restart_mixture_log_eprocess for the
    every-time schedule, but skipping the not-yet-started restarts.

    For times in the block [t0, t1), restarts s > t1 have not started at any of
    those times, so together they contribute exactly not_started_mass[t1]; only
    the first t1 rows of the restart-by-time matrix need to be exponentiated.
    That is ~half the work of the dense evaluation, with the same value.
    """
    n = len(p)
    for k, lambda_value in enumerate(lambda_values):
        log_if_zero = np.log1p(lambda_value * p)
        log_if_one = np.log1p(lambda_value * (p - 1.0))
        for t0 in range(0, n, block_size):
            t1 = min(t0 + block_size, n)
            block = block_workspace[:t1, : t1 - t0]
            np.multiply(mixture.zeros_in_range[:t1, t0:t1], log_if_zero[None, t0:t1], out=block)
            block += mixture.ones_in_range[:t1, t0:t1] * log_if_one[None, t0:t1]
            block += mixture.log_weights[:t1, None]
            max_over_processes = np.max(block, axis=0)
            block -= max_over_processes[None, :]
            np.exp(block, out=block)
            log_mix = max_over_processes + np.log(np.sum(block, axis=0))
            rest = not_started_mass[t1]
            if rest > 0.0:
                log_mix = np.logaddexp(log_mix, math.log(rest))
            per_lambda_workspace[k, t0:t1] = log_mix

    per_lambda_workspace += log_pi[:, None]
    max_over_lambda = np.max(per_lambda_workspace, axis=0)
    per_lambda_workspace -= max_over_lambda[None, :]
    np.exp(per_lambda_workspace, out=per_lambda_workspace)
    log_mix = max_over_lambda + np.log(np.sum(per_lambda_workspace, axis=0))
    if not np.all(np.isfinite(log_mix)):
        raise FloatingPointError("non-finite lambda-restart-mixture e-process values during inversion")
    return log_mix


def everytime_lambda_mixture_upper_confidence_trajectory(
    z: np.ndarray,
    lambda_values: np.ndarray,
    pi_weights: np.ndarray,
    alpha: float,
    binary_search_iterations: int,
    weighting_scheme: str,
    block_size: int = 256,
) -> np.ndarray:
    """UCB trajectory of the lambda-and-restart double mixture, every-time
    restart schedule only. Numerically the same as
    lambda_restart_mixture_upper_confidence_trajectory(..., starts=1..n, ...):
    the e-values are averaged over the lambda grid first, then that single
    mixture is inverted once per t by bisection at log(1/alpha)."""
    z = require_binary_z(z)
    lambda_values = np.asarray(lambda_values, dtype=float)
    pi_weights = np.asarray(pi_weights, dtype=float)
    require_lambda_weights(lambda_values, pi_weights)
    for lv in lambda_values:
        require_lambda(float(lv))
    require_probability(alpha, "alpha")
    if binary_search_iterations <= 0:
        raise ValueError("binary_search_iterations must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    mixture, not_started_mass = compile_everytime_restart_mixture(z, weighting_scheme)
    n_times = len(z)
    log_pi = np.log(pi_weights)
    per_lambda_workspace = np.empty((len(lambda_values), n_times), dtype=np.float64)
    block_workspace = np.empty((n_times, min(block_size, n_times)), dtype=np.float64)

    def evaluate(p: np.ndarray) -> np.ndarray:
        return _everytime_lambda_mixture_log_eprocess(
            mixture, not_started_mass, lambda_values, log_pi, p, block_size,
            per_lambda_workspace, block_workspace,
        )

    boundary = math.log(1.0 / alpha)

    lower = np.zeros(n_times, dtype=float)
    upper = np.ones(n_times, dtype=float)
    one_is_valid = evaluate(upper) < boundary
    lower[one_is_valid] = 1.0

    active = ~one_is_valid
    for _ in range(binary_search_iterations):
        midpoint = (lower + upper) / 2.0
        midpoint_is_valid = evaluate(midpoint) < boundary
        move_lower = active & midpoint_is_valid
        move_upper = active & ~midpoint_is_valid
        lower[move_lower] = midpoint[move_lower]
        upper[move_upper] = midpoint[move_upper]

    if not np.all(np.isfinite(upper)):
        raise FloatingPointError("non-finite lambda-restart-mixture UCB trajectory")
    # Upper bracket endpoint: finite-iteration bisection rounds outward.
    return upper


# The headline method: every-time restart mixture, log-squared restart weights, e-values
# averaged over the K = 100 midpoint lambda grid (a stand-in for lambda ~ Uniform[0, 1]).
# Every headline runner stores it under HEADLINE_KEY; plot_av_pipeline.py reads that key.
HEADLINE_LAMBDA_GRID_POINTS = 100
HEADLINE_WEIGHTING_SCHEME = "log_squared"
HEADLINE_KEY = "headline_lambda_grid_everytime_log_squared_matrix"


def headline_upper_confidence_trajectory(
    z: np.ndarray,
    alpha: float,
    binary_search_iterations: int,
) -> np.ndarray:
    lambda_values, lambda_weights = uniform_lambda_grid(HEADLINE_LAMBDA_GRID_POINTS)
    return everytime_lambda_mixture_upper_confidence_trajectory(
        z, lambda_values, lambda_weights, alpha, binary_search_iterations, HEADLINE_WEIGHTING_SCHEME,
    )


def single_process_lambda_mixture_upper_confidence_trajectory(
    z: np.ndarray,
    lambda_values: np.ndarray,
    pi_weights: np.ndarray,
    alpha: float,
    binary_search_iterations: int,
) -> np.ndarray:
    """UCB trajectory of the lambda mixture WITHOUT restarts: one process started
    at t = 1,

        M_t(u) = sum_k pi_k prod_{i<=t} (1 + lambda_k (u - z_i)),

    inverted once per t by bisection at log(1/alpha). The no-restart counterpart
    of everytime_lambda_mixture_upper_confidence_trajectory; each component is
    av_core.log_eprocess_at_p."""
    z = require_binary_z(z)
    lambda_values = np.asarray(lambda_values, dtype=float)
    pi_weights = np.asarray(pi_weights, dtype=float)
    require_lambda_weights(lambda_values, pi_weights)
    for lv in lambda_values:
        require_lambda(float(lv))
    require_probability(alpha, "alpha")
    if binary_search_iterations <= 0:
        raise ValueError("binary_search_iterations must be positive")

    n_times = len(z)
    n_ones = np.cumsum(z, dtype=np.float64)
    n_zeros = np.arange(1, n_times + 1, dtype=np.float64) - n_ones
    log_pi = np.log(pi_weights)[:, None]
    lam = lambda_values[:, None]

    def evaluate(p: np.ndarray) -> np.ndarray:
        per_lambda = (n_zeros[None, :] * np.log1p(lam * p[None, :])
                      + n_ones[None, :] * np.log1p(lam * (p[None, :] - 1.0)) + log_pi)
        max_over_lambda = np.max(per_lambda, axis=0)
        log_mix = max_over_lambda + np.log(np.sum(np.exp(per_lambda - max_over_lambda[None, :]), axis=0))
        if not np.all(np.isfinite(log_mix)):
            raise FloatingPointError("non-finite single-process lambda-mixture e-process values")
        return log_mix

    boundary = math.log(1.0 / alpha)
    lower = np.zeros(n_times, dtype=float)
    upper = np.ones(n_times, dtype=float)
    one_is_valid = evaluate(upper) < boundary
    lower[one_is_valid] = 1.0

    active = ~one_is_valid
    for _ in range(binary_search_iterations):
        midpoint = (lower + upper) / 2.0
        midpoint_is_valid = evaluate(midpoint) < boundary
        move_lower = active & midpoint_is_valid
        move_upper = active & ~midpoint_is_valid
        lower[move_lower] = midpoint[move_lower]
        upper[move_upper] = midpoint[move_upper]

    if not np.all(np.isfinite(upper)):
        raise FloatingPointError("non-finite single-process lambda-mixture UCB trajectory")
    # Upper bracket endpoint: finite-iteration bisection rounds outward.
    return upper


if __name__ == "__main__":
    # Quick sanity check against a toy sequence: the lambda-mixture bound should
    # sit between the tightest and loosest single-lambda restart mixtures in the
    # grid at every t (a mixture can't beat every one of its own components).
    from av_core import build_restart_schedule, restarted_mixture_upper_confidence_trajectory

    rng = np.random.default_rng(0)
    z_demo = (rng.random(500) < 0.05).astype(np.int64)
    starts_demo = build_restart_schedule(len(z_demo), restart_interval=1)
    weights_demo, total_mass_demo = restart_mixture_weights_for_scheme("log_squared", len(starts_demo))
    tail_mass_demo = restart_mixture_tail_mass(weights_demo, total_mass_demo)

    mixture_bound = lambda_restart_mixture_upper_confidence_trajectory(
        z_demo, np.asarray(DEFAULT_LAMBDA_GRID), np.asarray(DEFAULT_LAMBDA_WEIGHTS),
        alpha=0.05, binary_search_iterations=50, starts=starts_demo, weighting_scheme="log_squared",
    )
    single_lambda_bounds = [
        restarted_mixture_upper_confidence_trajectory(
            z_demo, lv, alpha=0.05, starts=starts_demo, weights=weights_demo,
            tail_mass=tail_mass_demo, binary_search_iterations=50,
        )
        for lv in DEFAULT_LAMBDA_GRID
    ]
    tightest = np.minimum.reduce(single_lambda_bounds)
    loosest = np.maximum.reduce(single_lambda_bounds)
    within_range = np.all((mixture_bound >= tightest - 1e-9) & (mixture_bound <= loosest + 1e-9))
    print(f"lambda grid: {DEFAULT_LAMBDA_GRID}")
    print(f"mixture bound final value: {mixture_bound[-1]:.4f}")
    print(f"single-lambda final values: {[round(float(b[-1]), 4) for b in single_lambda_bounds]}")
    print(f"mixture within [tightest, loosest] envelope at every t: {within_range}")
