"""
Shared functions implementing the statistical evaluation pipeline's methods:
- e-value computation
- e-value UCB inversion
- restart-mixture UCB construction
- classical Good-Turing point estimate, upper bound, Bonferroni correction

This script is agnostic to the failure-mode representation (primary vs set-valued), categorization rule (fixed vs online). It assumes inputs are already-computed z (discovery indicator) / q (oracle target) / gt_hat (pointwise Good-Turing) arrays or matrices. Separate scripts compute these values:
- av_discovery_fixed_primary.py
- av_discovery_fixed_set_valued.py
- av_discovery_online_primary.py
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

def require_file(path: Path, argument_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{argument_name} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{argument_name} is not a file: {path}")


def require_probability(value: float, name: str, *, include_zero: bool = False) -> None:
    lower_ok = value >= 0.0 if include_zero else value > 0.0
    if not math.isfinite(value) or not lower_ok or value >= 1.0:
        lower = "[0, 1)" if include_zero else "(0, 1)"
        raise ValueError(f"{name} must be in {lower}; got {value!r}")

def require_lambda(value: float) -> None:
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(f"--lambda-value must be in (0, 1); got {value!r}")


def require_binary_z(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    if z.ndim != 1 or len(z) == 0:
        raise ValueError("z must be a non-empty 1D array")
    if not np.all((z == 0) | (z == 1)):
        raise ValueError("z must contain only 0/1 values")
    return z


# ---------------------------------------------------------------------------
# Fixed-lambda e-value UCB
# ---------------------------------------------------------------------------

def log_eprocess_at_p(z: np.ndarray, lambda_value: float, p: np.ndarray) -> np.ndarray:
    """ 
    This function computes log E_t(p) for all timesteps t at vector-valued p_t, used for computing the upper confidence trajectory.
    - E_t(p) = prod_{j<=t} (1 + lambda*(p - z_j))
    - log E_t(p) = sum {j <= t} log(1 + lambda*(p-z_j))
    - if z_j = 0, then the inner log term is log(1 + lambda*p)
    - if z_j = 1, then the inner log term is log(1 + lambda*(p-1))
    """
    prefix_t = np.arange(1, len(z) + 1, dtype=float)
  # prefix_z counts how many of the t elements are ones
    prefix_z = np.cumsum(z, dtype=float)
  # np.log1p(x) computes log(1+x) more accurately when x is small.
    log_if_zero = np.log1p(lambda_value * p)
    log_if_one = np.log1p(lambda_value * (p - 1.0))
    values = (prefix_t - prefix_z) * log_if_zero + prefix_z * log_if_one
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite e-process values during inversion")
    return values


def upper_confidence_trajectory(
    z: np.ndarray,
    lambda_value: float,
    alpha: float,
    binary_search_iterations: int,
) -> np.ndarray:
    """
    This function inverts E_t(p) using binary search to solve for U_t = sup {p: E_t(p) < 1/alpha}. Since E_t(p) is monotone increasing in p, we can solve for this independently at each timestep.
    - binary_search_iterations controls precision of the bound.
    """
    z = require_binary_z(z)
    require_lambda(lambda_value)
    require_probability(alpha, "alpha")
    if binary_search_iterations <= 0:
        raise ValueError("--binary-search-iterations must be positive")
    boundary = math.log(1.0 / alpha)
    n_times = len(z)
    lower = np.zeros(n_times, dtype=float)
    upper = np.ones(n_times, dtype=float)

  # check p = 1 before doing binary search
    log_at_one = log_eprocess_at_p(z, lambda_value, upper)
    one_is_valid = log_at_one < boundary
    lower[one_is_valid] = 1.0

  # binary search over all timesteps at the same time (vectorized)
    active = ~one_is_valid
    for _ in range(binary_search_iterations):
        midpoint = (lower + upper) / 2.0
        log_at_midpoint = log_eprocess_at_p(z, lambda_value, midpoint)
        midpoint_is_valid = log_at_midpoint < boundary
        move_lower = active & midpoint_is_valid
        move_upper = active & ~midpoint_is_valid
        lower[move_lower] = midpoint[move_lower]
        upper[move_upper] = midpoint[move_upper]

    if not np.all(np.isfinite(upper)):
        raise FloatingPointError("non-finite UCB trajectory")
    # Return the upper bracket endpoint so finite-iteration bisection rounds
    # outward, keeping the reported trajectory conservative as an upper bound.
    return upper


# ---------------------------------------------------------------------------
# Restarted-mixture UCB (periodic/every-time restart schedules)
# ---------------------------------------------------------------------------
GEOMETRIC_WEIGHTS_TOTAL_MASS = 1.0
LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS = 0.999
# log squared weights using 2.11 approx sums to < 1. We use 0.999 as a lower bound on this infinite sum.

def build_restart_schedule(n: int, restart_interval: int) -> np.ndarray:
    """
    - restart_interval controls the block size
    (e.g. restart_interval = 1 means restart at every timestep, restart_interval = 100 means restart every 100 timesteps)

    Required, not defaulted: the interval selects the METHOD -- 1 is the every-time
    schedule the headline bound uses, 100 is a different construction entirely. A caller
    that forgot the argument would silently get a periodic schedule and a bound that
    looks perfectly plausible.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if restart_interval <= 0:
        raise ValueError("restart_interval must be positive")
    return np.arange(1, n + 1, restart_interval, dtype=np.int64)


def geometric_weights(n_starts: int) -> np.ndarray:
    """Geometric weight sequence w_j = 2^-(j+1)."""
    if n_starts <= 0:
        raise ValueError("n_starts must be positive")
    return 2.0 ** -np.arange(1, n_starts + 1, dtype=np.float64)


def log_squared_heavy_tailed_weights(n_starts: int) -> np.ndarray:
    """
    Heavy-tailed weight sequence w_j = 1/(2.11*(j+2)*ln(j+2)^2).
    2.11 is used since it upper bounds the infinite sequence and ensures the sum of weights is < 1.
    j+2 indexing is used to avoid log(1).
    """
    if n_starts <= 0:
        raise ValueError("n_starts must be positive")
    j = np.arange(n_starts, dtype=np.float64)
    n = j + 2.0
    return 1.0 / (2.11 * n * np.log(n) ** 2)

def restart_mixture_tail_mass(weights: np.ndarray, total_mass: float) -> float:
    """
    Return the capital assigned to restart processes that are part of the
    theoretical infinite mixture but are not explicitly instantiated in code.

    Theory uses M_t = sum_{s=1}^infty w_s E_{s:t}.

    In the function, we instantiate only the first K restart processes, so are missing starting capital from the remaining tail of restart processes.
    """
    return max(total_mass - float(weights.sum()), 0.0)


def restart_mixture_weights_for_scheme(weighting_scheme: str, n_starts: int) -> tuple[np.ndarray, float]:
    if weighting_scheme == "geometric":
        return geometric_weights(n_starts), GEOMETRIC_WEIGHTS_TOTAL_MASS
    if weighting_scheme == "log_squared":
        return log_squared_heavy_tailed_weights(n_starts), LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS
    raise ValueError(f"unknown weighting scheme: {weighting_scheme!r}")


@dataclass(frozen=True)
class RestartMixture:
    """Restart-schedule quantities that do not depend on the candidate p."""

    weights: np.ndarray
    log_weights: np.ndarray
    tail_mass: float
    zeros_in_range: np.ndarray
    ones_in_range: np.ndarray


def compile_restart_mixture(
    z: np.ndarray,
    starts: np.ndarray,
    weights: np.ndarray,
    tail_mass: float,
) -> RestartMixture:
    """Precompute per-restart sufficient statistics for repeated p evaluations.

    The UCB inversion calls the restarted-mixture e-process many times during
    binary search. The restart/time counts below depend only on z and the
    restart schedule, so building them once avoids repeatedly materializing the
    same O(n_starts * n_times) arrays.
    """
    z = require_binary_z(z)
    starts = np.asarray(starts)
    weights = np.asarray(weights, dtype=np.float64)
    n = len(z)
    if starts.ndim != 1 or len(starts) == 0:
        raise ValueError("starts must be a non-empty 1D array")
    if not np.all(np.diff(starts) > 0):
        raise ValueError("starts must be strictly increasing")
    if starts[0] < 1 or starts[-1] > n:
        raise ValueError(f"starts must be within [1, {n}]; got range [{starts[0]!r}, {starts[-1]!r}]")
    if len(starts) != len(weights):
        raise ValueError("starts and weights must have the same length")
    if tail_mass < 0.0:
        raise ValueError(f"tail_mass must be non-negative; got {tail_mass!r}")
    if weights.sum() + tail_mass > 1.0 + 1e-9:
        raise ValueError(
            f"restart mixture weights plus tail_mass must sum to at most 1; "
            f"got weights.sum()={weights.sum()!r}, tail_mass={tail_mass!r}"
        )
    if np.any(weights <= 0):
        raise ValueError("restart mixture weights must be strictly positive")

    t_index = np.arange(1, n + 1, dtype=np.float64)
    prefix_ones = np.concatenate([[0.0], np.cumsum(z, dtype=np.float64)])
    active = t_index[None, :] >= starts[:, None]
    ones_in_range = np.cumsum(z, dtype=np.float64)[None, :] - prefix_ones[starts - 1][:, None]
    total_in_range = t_index[None, :] - starts[:, None] + 1.0
    zeros_in_range = total_in_range - ones_in_range

    return RestartMixture(
        weights=weights,
        log_weights=np.log(weights),
        tail_mass=float(tail_mass),
        zeros_in_range=np.where(active, zeros_in_range, 0.0),
        ones_in_range=np.where(active, ones_in_range, 0.0),
    )


def evaluate_restart_mixture_log_eprocess(
    mixture: RestartMixture,
    lambda_value: float,
    p: np.ndarray,
    scratch: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate log M_t(p) from precomputed restart sufficient statistics."""
    log_if_zero = np.log1p(lambda_value * p)
    log_if_one = np.log1p(lambda_value * (p - 1.0))

    # reuse scratch array if provided, otherwise create a new one
    if scratch is None:
        log_e_process = np.empty_like(mixture.zeros_in_range)
    else:
        if scratch.shape != mixture.zeros_in_range.shape:
            raise ValueError(
                f"scratch has shape {scratch.shape!r}, expected {mixture.zeros_in_range.shape!r}"
            )
        log_e_process = scratch

    # First build log E_{s:t}(p) for each restart s and time t.
    np.multiply(mixture.zeros_in_range, log_if_zero[None, :], out=log_e_process)
    log_e_process += mixture.ones_in_range * log_if_one[None, :]

    # Fold in restart weights and do log-sum-exp over restart processes.
    log_e_process += mixture.log_weights[:, None]
    max_over_processes = np.max(log_e_process, axis=0)
    np.subtract(log_e_process, max_over_processes[None, :], out=log_e_process)
    np.exp(log_e_process, out=log_e_process)
    sum_exp = np.sum(
        log_e_process,
        axis=0,
    )
    log_mix = max_over_processes + np.log(sum_exp)

    if mixture.tail_mass > 0.0:
        log_mix = np.logaddexp(log_mix, math.log(mixture.tail_mass))

    if not np.all(np.isfinite(log_mix)):
        raise FloatingPointError("non-finite restarted-mixture e-process values during inversion")
    return log_mix


def restarted_mixture_log_eprocess_at_p(
    z: np.ndarray,
    lambda_value: float,
    p: np.ndarray,
    starts: np.ndarray,
    weights: np.ndarray,
    tail_mass: float,
) -> np.ndarray:
    """
    Restart e-process:
    - E_{s:t}(p) = 1 if t < s, prod {i=s}^t (1 + lambda * (p - z_i)) if t >= s
    Mixture e-process:
    - M_t(p) = sum {s=1} ^ infty w_s E_{s:t}(p)

    Only finitely many restarts (starts/weights) are instantiated here; every
    restart beyond that has E=1 (not yet started) at every t, so its total
    contribution is exactly its weight, added as the constant tail_mass.
    """
    mixture = compile_restart_mixture(z, starts, weights, tail_mass)
    return evaluate_restart_mixture_log_eprocess(mixture, lambda_value, p)


def restarted_mixture_upper_confidence_trajectory(
    z: np.ndarray,
    lambda_value: float,
    alpha: float,
    starts: np.ndarray,
    weights: np.ndarray,
    tail_mass: float,
    binary_search_iterations: int,
) -> np.ndarray:
    """Return the anytime upper confidence trajectory for a restart mixture.

    This inverts M_t(p) at every time t by binary search, returning the largest
    candidate p whose restarted-mixture e-process has not crossed 1 / alpha.

    For each trajectory, the per-restart zero/one counts are prepared once and
    reused across all binary-search candidates. Those counts depend on z and
    starts, but not on p.
    """
    z = require_binary_z(z)
    require_lambda(lambda_value)
    require_probability(alpha, "alpha")

    starts = np.asarray(starts)
    if starts.ndim != 1 or len(starts) == 0:
        raise ValueError("starts must be a non-empty 1D array")
    if not np.all(np.diff(starts) > 0):
        raise ValueError("starts must be strictly increasing")
    if starts[0] < 1 or starts[-1] > len(z):
        raise ValueError(f"starts must be within [1, {len(z)}]; got range [{starts[0]!r}, {starts[-1]!r}]")

    weights = np.asarray(weights, dtype=np.float64)
    mixture = compile_restart_mixture(z, starts, weights, float(tail_mass))

    if binary_search_iterations <= 0:
        raise ValueError("--binary-search-iterations must be positive")
    boundary = math.log(1.0 / alpha)
    n_times = len(z)
    lower = np.zeros(n_times, dtype=float)
    upper = np.ones(n_times, dtype=float)
    scratch = np.empty_like(mixture.zeros_in_range)

    log_at_one = evaluate_restart_mixture_log_eprocess(mixture, lambda_value, upper, scratch)
    one_is_valid = log_at_one < boundary
    lower[one_is_valid] = 1.0

    active = ~one_is_valid
    for _ in range(binary_search_iterations):
        midpoint = (lower + upper) / 2.0
        log_at_midpoint = evaluate_restart_mixture_log_eprocess(mixture, lambda_value, midpoint, scratch)
        midpoint_is_valid = log_at_midpoint < boundary
        move_lower = active & midpoint_is_valid
        move_upper = active & ~midpoint_is_valid
        lower[move_lower] = midpoint[move_lower]
        upper[move_upper] = midpoint[move_upper]

    if not np.all(np.isfinite(upper)):
        raise FloatingPointError("non-finite restarted-mixture UCB trajectory")
    # Return the upper bracket endpoint so finite-iteration bisection rounds
    # outward, keeping the reported trajectory conservative as an upper bound.
    return upper

# ---------------------------------------------------------------------------
# Classical Good-Turing bound
# ---------------------------------------------------------------------------

def classical_gt_bound(gt_hat: np.ndarray, alpha: float, n_bonferroni: int) -> np.ndarray:
    """
    Computes Good-Turing upper bound with Bonferroni correction for simultaneous upper bounds.
    To compute without Bonferroni correction, set n_bonferroni = 1.
    Upper bound = min(1, gt_hat + (1 + sqrt(2)) * sqrt(log(4/delta)/t)), clipped at 1
    since the missing mass being bounded is itself a probability.
    """
    delta = alpha / n_bonferroni
    if not (0.0 < delta < 1.0):
        raise ValueError(f"Bonferroni-corrected delta={delta!r} is not in (0, 1); check --alpha and --n-bonferroni")
    t = np.arange(1, gt_hat.shape[-1] + 1, dtype=np.float64)
    return np.minimum(
        1.0,
        gt_hat + (1.0 + math.sqrt(2.0))
        * np.sqrt(math.log(4.0 / delta) / t),
    )


def binary_restart_log_e_at_threshold(z, threshold, lambda_values, lambda_weights,
                                      weights=None, total_mass=1.0):
    """O(T K) evaluation at a fixed stopping threshold, with every-time restarts.

    New capital w_t enters BEFORE multiplying by the round-t factor. The
    remaining reserve is added once. This is the same binary product evaluated
    by compile_restart_mixture, without its quadratic intermediate arrays.
    Pass threshold=epsilon-zeta for the PAC procedure. A normalized discrete
    lambda mixture is an e-process; a bare unnormalized integral is not M_0=1.
    """
    z = require_binary_z(z)
    lam = np.asarray(lambda_values, dtype=float)
    mix = np.asarray(lambda_weights, dtype=float)
    if lam.ndim != 1 or mix.shape != lam.shape or np.any(lam <= 0) or np.any(lam >= 1):
        raise ValueError('lambda values must be a vector in (0,1)')
    if np.any(mix <= 0) or not np.isclose(mix.sum(), 1):
        raise ValueError('lambda weights must be positive and sum to one')
    if not 0 < threshold < 1:
        raise ValueError('threshold must be in (0,1)')
    if weights is None:
        weights = normalized_log_squared_weights(len(z))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != z.shape or np.any(weights <= 0) or weights.sum() > total_mass:
        raise ValueError('invalid restart weights')
    log_capital = np.full(len(lam), -np.inf)
    reserve = total_mass - np.cumsum(weights)
    result = np.empty(len(z))
    for i, value in enumerate(z):
        log_capital = np.logaddexp(log_capital, math.log(weights[i])) + np.log1p(lam * (threshold - value))
        result[i] = np.logaddexp(np.logaddexp.reduce(np.log(mix) + log_capital),
                                math.log(reserve[i]) if reserve[i] > 0 else -np.inf)
    return result


def normalized_log_squared_weights(n):
    """w_s proportional to 1/((s+1) log(s+1)^2), s=1,... .

    Euler--Maclaurin tail after N=100000; omitted terms are below double
    precision. Integral tail is 1/log(N), then -f(N)/2-f'(N)/12.
    """
    N = 100000
    x = np.arange(2, N + 1, dtype=float)
    logn = math.log(N)
    f = 1 / (N * logn**2)
    derivative = -(logn**-2 + 2 * logn**-3) / N**2
    C = np.sum(1 / (x * np.log(x)**2)) + 1/logn - f/2 - derivative/12
    return log_squared_heavy_tailed_weights(n) * (2.11 / C)
