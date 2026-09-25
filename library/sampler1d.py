"""Exact one-dimensional discrete Gaussians DN_Z(mu, sigma^2): Pr[z] ∝ exp(-(z - mu)^2 / (2 sigma^2)), z integer."""
from __future__ import annotations

import math

import numpy as np


def discrete_laplace(t: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Pr[y] ∝ exp(-|y| / t) on Z."""
    p = 1.0 - math.exp(-1.0 / t)
    out = np.empty(size, dtype=np.int64)
    todo = np.arange(size)
    while todo.size:
        u = rng.geometric(p, size=todo.size) - 1
        s = rng.integers(0, 2, size=todo.size) * 2 - 1
        keep = ~((u == 0) & (s == -1))
        out[todo[keep]] = (u * s)[keep]
        todo = todo[~keep]
    return out


def discrete_gaussian_cks(mu, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Canonne-Kamath-Steinke rejection sampler, vectorised over the centres mu (default sampler)."""
    mu = np.asarray(mu, dtype=np.float64)
    m0 = np.rint(mu)
    e = (mu - m0).ravel()
    t = math.floor(sigma) + 1
    out = np.empty(e.size, dtype=np.int64)
    todo = np.arange(e.size)
    while todo.size:
        d = discrete_laplace(t, todo.size, rng).astype(np.float64)
        ee = e[todo]
        logacc = -(d - ee) ** 2 / (2 * sigma**2) + (np.abs(d) - np.abs(ee)) / t - sigma**2 / (2 * t**2)
        acc = rng.random(todo.size) < np.exp(logacc)
        out[todo[acc]] = (m0.ravel()[todo] + d)[acc].astype(np.int64)
        todo = todo[~acc]
    return out.reshape(mu.shape)


def _karney_one(c: float, s: float, rng: np.random.Generator) -> int:
    p = 1.0 - math.exp(-0.5)
    while True:
        k = rng.geometric(p) - 1
        if rng.random() >= math.exp(-k * (k - 1) / 2):
            continue
        sig = 1 if rng.random() < 0.5 else -1
        i = math.ceil(s * k + sig * c)
        j = int(rng.integers(0, math.ceil(s)))
        x = (i - s * k - sig * c + j) / s
        if x >= 1 or (k == 0 and x == 0 and sig == -1):
            continue
        if rng.random() >= math.exp(-x * (2 * k + x) / 2):
            continue
        return sig * (i + j)


def discrete_gaussian_karney(mu, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Karney's sampler (the draft's Figure 1), one entry at a time.  Needs sigma >= 1."""
    if sigma < 1:
        raise ValueError("Karney's sampler needs sigma >= 1")
    mu = np.asarray(mu, dtype=np.float64)
    return np.array([_karney_one(float(c), sigma, rng) for c in mu.ravel()], dtype=np.int64).reshape(mu.shape)


SAMPLERS = {"cks": discrete_gaussian_cks, "karney": discrete_gaussian_karney}


def sample(mu, sigma: float, rng: np.random.Generator, method: str = "cks") -> np.ndarray:
    return SAMPLERS[method](mu, sigma, rng)
