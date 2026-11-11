"""One-sided Howard et al. (2021), Theorem 4 / Proposition 9, for [0,1] data.

The upper bound covers the average conditional mean, not a common iid mean.
For decreasing conditional novelty probabilities it also covers current missing
mass. Use the lower-tail martingale (equivalently apply the theorem to 1-Y).
Reference: https://arxiv.org/html/1810.08240v5#S4.SS1
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammainc, gammaln


def gamma_exponential_log_mixture(s, v, rho=100.0):
    """Proposition 9, equation (65), with scale c=1 and initial wealth 1."""
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    s, v = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(v, dtype=float))
    if np.any(~np.isfinite(s)) or np.any(~np.isfinite(v)) or np.any(s < 0) or np.any(v < 0):
        raise ValueError("s and v must be finite and nonnegative")
    a = v + rho
    x = s + a
    return (rho * np.log(rho) - gammaln(rho) - np.log(gammainc(rho, rho))
            + gammaln(a) + np.log(gammainc(a, x)) - a * np.log(x) + s + v)


def gamma_exponential_boundary(v, alpha=0.05, rho=100.0):
    """One-sided level-alpha uniform boundary; return the upper root bracket."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    v = np.asarray(v, dtype=float)
    # Validate inputs even for empty arrays.
    gamma_exponential_log_mixture(np.zeros_like(v), v, rho)
    lo = np.zeros_like(v)
    hi = np.sqrt(v + rho) + 1
    target = np.log(1 / alpha)
    for _ in range(100):
        below = gamma_exponential_log_mixture(hi, v, rho) < target
        if not np.any(below):
            break
        hi = np.where(below, 2 * hi, hi)
    else:
        raise ArithmeticError("failed to bracket gamma-exponential boundary")
    for _ in range(40):
        mid = (lo + hi) / 2
        below = gamma_exponential_log_mixture(mid, v, rho) < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return hi + 1e-10  # Small outward numerical margin.


def empirical_bernstein_upper(y, alpha=0.05, rho=100.0):
    """Time on last axis; predictable historical mean, initialized at 1/2.

    No restarts, tuning against outcomes, or running intersection. Each method
    in the comparison receives the full one-sided alpha budget independently.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim not in (1, 2) or y.shape[-1] == 0:
        raise ValueError("y must be a nonempty vector or trial-by-time matrix")
    if np.any(~np.isfinite(y)) or np.any((y < 0) | (y > 1)):
        raise ValueError("observations must lie in [0,1]")
    totals = np.cumsum(y, axis=-1)
    t = np.arange(1, y.shape[-1] + 1)
    predictions = np.empty_like(y)
    predictions[..., 0] = 0.5
    predictions[..., 1:] = totals[..., :-1] / t[:-1]
    variance = np.cumsum((y - predictions) ** 2, axis=-1)
    upper = np.minimum(1, totals / t + gamma_exponential_boundary(variance, alpha, rho) / t)
    return upper, variance
