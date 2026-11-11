"""Restart mixture with the hindsight-optimal lambda in closed form.

Method 2 of the lambda study. Method 1 (av_lambda_restart_mixture.py) averages the
e-process over a grid of fixed lambdas. This module instead gives every restart its
own lambda, the one that maximises that restart's e-value in hindsight:

    Zbar_{s:t}   = mean of z over [s, t]
    lambda_{s:t}(u) = max( (u - Zbar_{s:t}) / (u (1 - u)), 0 )
    E_{s:t}(u)   = prod_{i=s}^{t} (1 + lambda_{s:t}(u) (u - z_i))
    M_t(u)       = sum_{s<=t} w_s E_{s:t}(u) + sum_{s>t} w_s
    U_t(alpha)   = sup { u in (0,1) : M_t(u) < 1/alpha }

lambda_{s:t}(u) is the exact maximiser, not an approximation: writing N0/N1 for the
zero/one counts in [s, t], d/dlambda [N0 log(1+lambda u) + N1 log(1+lambda(u-1))]
vanishes there. Substituting it back collapses the product to a closed form,

    1 + lambda u       = (1 - Zbar) / (1 - u)
    1 + lambda (u - 1) = Zbar / u

so that

    log E_{s:t}(u) = N (Zbar log(Zbar/u) + (1 - Zbar) log((1 - Zbar)/(1 - u)))
                   = N * KL(Zbar_{s:t} || u)          [0 log 0 = 0]

which is what this module evaluates. That form is what makes the method cheap and
numerically stable: no per-lambda sweep, no log1p of a quantity approaching -1 (the
u -> 1 and Zbar -> 0 corners of the naive product both cancel analytically here).

VALIDITY: lambda_{s:t}(u) is chosen using z_s..z_t, so it is NOT predictable and
E_{s:t} is NOT a martingale. Ville's inequality does not apply and U_t is NOT an
anytime-valid upper bound -- it is the oracle/hindsight benchmark, and it does dip
below the true unseen probability in practice. Use it to measure what the grid
mixture gives up, not as a bound to report. The predictable variant (lambda from
z_s..z_{t-1}) is the valid counterpart and is a separate construction.
"""
from __future__ import annotations

import math

import numpy as np

from av_core import require_binary_z, require_probability
from av_lambda_restart_mixture import compile_everytime_restart_mixture


def kl_bernoulli(z_bar: np.ndarray, u: np.ndarray) -> np.ndarray:
    """KL(Zbar || u) for Bernoulli means, with 0 log 0 = 0.

    Only called where Zbar < u, so u > 0 and 1 - Zbar > 0; the Zbar = 0 corner is the
    one that needs the convention and is handled explicitly.
    """
    z_bar = np.asarray(z_bar, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    first = np.where(z_bar > 0.0, z_bar * np.log(np.where(z_bar > 0.0, z_bar, 1.0) / u), 0.0)
    second = (1.0 - z_bar) * np.log((1.0 - z_bar) / (1.0 - u))
    return first + second


def closed_form_log_eprocess_at_u(
    z: np.ndarray,
    u: np.ndarray,
    weighting_scheme: str,
    block_size: int = 256,
    mixture=None,
    reserve: np.ndarray | None = None,
) -> np.ndarray:
    """log M_t(u) for the hindsight-lambda restart mixture, every-time schedule.

    Restarts that have not begun by the end of a block contribute exactly their own
    weight (E = 1 there), which is the `reserve` term -- the same bookkeeping the grid
    mixture uses, so the two methods mix over the identical restart structure.
    """
    z = require_binary_z(z)
    u = np.asarray(u, dtype=np.float64)
    n = len(z)
    if u.shape != (n,):
        raise ValueError(f"u must have shape {(n,)!r}; got {u.shape!r}")
    if np.any(u <= 0.0) or np.any(u >= 1.0):
        raise ValueError("u must lie strictly inside (0, 1)")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if mixture is None or reserve is None:
        mixture, reserve = compile_everytime_restart_mixture(z, weighting_scheme)
    zeros, ones, log_w = mixture.zeros_in_range, mixture.ones_in_range, mixture.log_weights
    length = zeros + ones

    out = np.empty(n, dtype=np.float64)
    for t0 in range(0, n, block_size):
        t1 = min(t0 + block_size, n)
        n_in_range = length[:t1, t0:t1]
        active = n_in_range > 0
        safe_len = np.where(active, n_in_range, 1.0)
        z_bar = np.where(active, ones[:t1, t0:t1] / safe_len, 0.0)
        u_block = u[t0:t1][None, :]

        # lambda > 0 only where the restart's own mean is below u; elsewhere E = 1.
        betting = active & (z_bar < u_block)
        log_e = np.where(
            betting,
            n_in_range * kl_bernoulli(np.where(betting, z_bar, 0.0), np.broadcast_to(u_block, z_bar.shape)),
            0.0,
        )
        # A restart that has not begun by time t has E = 1 there, i.e. log E = 0, and so
        # contributes exactly its own weight -- `betting` is already False for those, so
        # log_e is 0 and no separate bookkeeping is needed. `reserve[t1]` then covers only
        # the restarts beyond this block. Dropping them here instead would lose that mass.
        log_e = log_e + log_w[:t1, None]

        max_over = np.max(log_e, axis=0)
        log_mix = max_over + np.log(np.sum(np.exp(log_e - max_over[None, :]), axis=0))
        if reserve[t1] > 0.0:
            log_mix = np.logaddexp(log_mix, math.log(reserve[t1]))
        out[t0:t1] = log_mix

    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite closed-form-lambda mixture values during inversion")
    return out


def closed_form_upper_confidence_trajectory(
    z: np.ndarray,
    alpha: float,
    binary_search_iterations: int,
    weighting_scheme: str,
    *,
    log_t_coefficient: float,
    block_size: int = 256,
) -> np.ndarray:
    """Invert the hindsight-lambda mixture at log(1/alpha), once per t, by bisection.

    M_t(u) is not monotone in u the way a fixed-lambda e-process is -- lambda_{s:t}(u)
    moves with u -- but it is increasing wherever it is above the reserve (raising u
    only ever increases each betting restart's KL term), which is the region the
    inversion searches. The bracket is [0, 1) with the endpoints held strictly inside.
    """
    z = require_binary_z(z)
    require_probability(alpha, "alpha")
    if binary_search_iterations <= 0:
        raise ValueError("binary_search_iterations must be positive")

    if not math.isfinite(log_t_coefficient) or log_t_coefficient < 0.0:
        raise ValueError(f"log_t_coefficient must be finite and non-negative; got {log_t_coefficient!r}")

    n = len(z)
    mixture, reserve = compile_everytime_restart_mixture(z, weighting_scheme)
    # The hindsight lambda maximises each restart's e-value, so M~_t sits ABOVE the
    # uniform-lambda integral that the K-point grid approximates. Inverting at
    # log(1/alpha) + c log(t) discounts that inflation, so c = 0 reads the closed form
    # as-is and c > 0 reads it as a proxy for the integrated-lambda wealth. Raising the
    # threshold only ever admits more u, so unlike a negative offset this cannot become
    # infeasible against the mixture's reserve floor.
    boundary = math.log(1.0 / alpha) + log_t_coefficient * np.log(np.arange(1, n + 1, dtype=np.float64))

    # u is held strictly inside (0, 1): lambda_{s:t}(u) divides by u(1 - u).
    eps = 1e-12
    lower = np.full(n, eps)
    upper = np.full(n, 1.0 - eps)

    for _ in range(binary_search_iterations):
        midpoint = (lower + upper) / 2.0
        valid = closed_form_log_eprocess_at_u(
            z, midpoint, weighting_scheme, block_size, mixture, reserve
        ) < boundary
        lower = np.where(valid, midpoint, lower)
        upper = np.where(valid, upper, midpoint)

    if not np.all(np.isfinite(upper)):
        raise FloatingPointError("non-finite closed-form-lambda UCB trajectory")
    # Upper bracket endpoint: finite-iteration bisection rounds outward.
    return upper
