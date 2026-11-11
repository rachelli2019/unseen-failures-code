"""Importance-weighted K-lambda, every-time restart mixture.

Native streaming recurrence evaluates the exact mixture at a supplied u grid.
Bounds round outward; coverage and stopping use direct mixture evaluations at
oracle masses and epsilon, respectively, so neither metric is grid-rounded.
"""
import ctypes
from pathlib import Path
import numpy as np
from av_core import log_squared_heavy_tailed_weights, LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS


def length_distribution(lengths):
    a = np.asarray(lengths,dtype=float)
    if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)) or np.any(a<0):
        raise ValueError('Lengths must be a finite nonnegative vector')
    median = float(np.median(a)); long = a>=median
    if np.all(long) or not np.any(long):
        raise ValueError('Both length groups must be nonempty')
    q = np.where(long,.5,1.); q /= q.sum()
    return median,long,q,1/(len(a)*q)


def evaluate(y, candidates, lambdas, oracle, epsilons, library, alpha=.05, restart_weights=None, total_mass=LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS, allow_signed=False):
    y=np.ascontiguousarray(y,dtype=np.float64)
    u=np.ascontiguousarray(candidates,dtype=np.float64)
    lam=np.ascontiguousarray(lambdas,dtype=np.float64)
    oracle=np.asarray(oracle,dtype=float); eps=np.asarray(epsilons,dtype=float)
    if y.ndim!=1 or not len(y) or not np.all(np.isfinite(y)) or (not allow_signed and np.any(y<0)):
        raise ValueError('Invalid weighted observations')
    if u.ndim!=1 or len(u)<2 or u[0]!=0 or u[-1]!=1 or not np.all(np.diff(u)>0):
        raise ValueError('Candidate grid must increase from zero to one')
    if lam.ndim!=1 or not len(lam) or not np.all(np.isfinite(lam)) or np.any(lam<=0) or np.any(lam>=1) or lam.max()*y.max()>=1:
        raise ValueError('Lambdas must give positive factors for all u')
    if oracle.shape!=y.shape or eps.ndim!=1 or not len(eps) or not np.all((eps>0)&(eps<1)):
        raise ValueError('Invalid oracle/epsilon arrays')
    oi=np.searchsorted(u,oracle).astype(np.int32); ei=np.searchsorted(u,eps).astype(np.int32)
    if np.any(oi>=len(u)) or np.any(ei>=len(u)) or not np.all(u[oi]==oracle) or not np.all(u[ei]==eps):
        raise ValueError('Oracle and epsilon values must appear exactly in candidate grid')
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    return _native(y,u,lam,oi,ei,library,alpha,restart_weights,total_mass)


def _native(y,u,lam,oi,ei,library,alpha=.05,restart_weights=None,total_mass=LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS):
    lib=ctypes.CDLL(str(Path(library).resolve()))
    fn=lib.weighted_grid
    double=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    integer=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    byte=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int]*3+[double]*4+[ctypes.c_double]*2+[integer,ctypes.c_int,integer,double,byte,integer]
    fn.restype=ctypes.c_int
    w=log_squared_heavy_tailed_weights(len(y)) if restart_weights is None else np.ascontiguousarray(restart_weights,dtype=np.float64)
    if w.shape != y.shape or not np.all(np.isfinite(w)) or np.any(w <= 0) or not 0 < total_mass <= 1 or w.sum() >= total_mass:
        raise ValueError("Restart weights exhaust total capital at this horizon")
    upper=np.empty(len(y)); covered=np.empty(len(y),dtype=np.uint8); stops=np.empty(len(ei),dtype=np.int32)
    code=fn(len(y),len(u),len(lam),y,u,lam,w,total_mass,alpha,oi,len(ei),ei,upper,covered,stops)
    if code: raise FloatingPointError('Nonfinite native mixture state')
    return upper,covered.astype(bool),stops
