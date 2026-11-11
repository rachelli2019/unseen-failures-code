"""Predictable inverse-propensity PAC calibration under P (no Q).

Round t uses only observations 1,...,t-1. Initialization queries everything
when q is undefined/outside (0,1). L-prime is interpreted as L-check.
The empirical inverse uses strict score < z; consequently the trusted set
also uses score < tau (ties are queried). All queried observations, forced
and random, contribute D/pi. No unqueried ground truth enters calibration.
"""
from __future__ import annotations
import math
import numpy as np


def g_bound(q, a, n, squared_weights):
    if not 0 < q < 1 or n < 1:
        raise ValueError('G requires q in (0,1) and n >= 1')
    L = max(float(squared_weights), 1.)
    logit = math.log(q) - math.log1p(-q)
    ell = (2*math.log(math.log(L)+1)
           + 2*math.log(max(abs(math.ceil(.25*math.sqrt(L)*logit)), 1))
           + math.log(7.06/a))
    shifted = logit + 4*math.sqrt(math.e/L)
    qcheck = 1/(1+math.exp(-shifted)) if shifted >= 0 else math.exp(shifted)/(1+math.exp(shifted))
    return (math.sqrt(2.13*ell*L + 1.76*qcheck*qcheck*ell*ell)
            + 1.33*qcheck*ell)/n + qcheck-q


class PacCalibrator:
    def __init__(self, grid, epsilon, alpha):
        self.grid = np.asarray(grid, dtype=float)
        if len(self.grid) == 0 or not np.all(np.isfinite(self.grid)) or np.any(np.diff(self.grid) <= 0):
            raise ValueError('grid must be finite, sorted and unique')
        if not 0 < epsilon < 1 or not 0 < alpha < 1:
            raise ValueError('epsilon and alpha must be in (0,1)')
        self.epsilon, self.alpha = epsilon, alpha
        self.n = 0
        self.sum_w = self.sum_w2 = 0.
        self.errors = np.zeros(len(grid))

    def threshold(self):
        n = self.n
        if n <= 1:
            return -math.inf
        # The empirical bound uses n past draws, but the upcoming decision
        # uses exploration probability (n+1)^(-1/4). Match that probability
        # here, otherwise the permitted error is slightly too large.
        q = self.epsilon/(2*(1-(n+1)**-.25))
        if not 0 < q < 1:
            return -math.inf
        # F_lower(q; alpha/2) calls G(1-q; alpha/4).
        rhs = max(q+self.sum_w/n-1-g_bound(1-q, self.alpha/4, n, self.sum_w2), 0.)
        if rhs <= 0:
            return -math.inf
        cumulative = np.cumsum(self.errors)/n
        crossing = np.flatnonzero(cumulative >= rhs)
        return float(self.grid[crossing[0]]) if len(crossing) else math.inf

    def update(self, score, prediction, queried_truth, propensity):
        self.n += 1
        if queried_truth is not None:
            w = 1/propensity
            self.sum_w += w
            self.sum_w2 += w*w
            if prediction != queried_truth:
                self.errors[np.searchsorted(self.grid, score)] += w
