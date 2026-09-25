"""k-means on the released sketch and the certified centre release (draft Section 8.2, steps 6-8)."""
from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans

from . import accounting
from .mechanism import mechanism
from .sampler1d import discrete_gaussian_cks


def partition(W: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Lloyd's algorithm on the columns of W with public randomness."""
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(W.T).labels_


def centres(A: np.ndarray, part: np.ndarray, k: int) -> np.ndarray:
    """Exact (non-private) part means, n x k."""
    return np.stack([A[:, part == c].mean(axis=1) if (part == c).any() else np.zeros(A.shape[0]) for c in range(k)], 1)


def noisy_centres(A: np.ndarray, part: np.ndarray, k: int, rho_c: float, rng: np.random.Generator,
                  adjacency: str = "rank_one", ell_c: int | None = 16, sensitivity: float = 1.0) -> np.ndarray:
    """(alpha, alpha rho_c)-RDP centres for a partition computed from the public sketch.
    rank_one: noise on the means, sensitivity s_min^{-1/2} (Definition 3).
    record:   noise on the sums, sensitivity 1 (one record changed), divided by the public part sizes.
    ell_c:    round to 2^-ell_c and add a discrete Gaussian (draft 7.1); None = continuous Gaussian."""
    n = A.shape[0]
    sizes = np.bincount(part, minlength=k)
    if adjacency == "rank_one":
        stat, div = centres(A, part, k), np.ones(k)
        sens = sensitivity / math.sqrt(sizes[sizes > 0].min())
    else:
        stat = np.stack([A[:, part == c].sum(axis=1) for c in range(k)], axis=1)
        sens, div = sensitivity, np.maximum(sizes, 1).astype(float)
    if ell_c is None:
        noisy = stat + sens / math.sqrt(2 * rho_c) * rng.standard_normal(stat.shape)
    else:
        h = 2.0**-ell_c
        sigma = (sens + h * math.sqrt(n * k)) / math.sqrt(2 * rho_c)
        noisy = h * discrete_gaussian_cks(np.rint(stat / h), sigma / h, rng)
    return noisy / div


def private_kmeans(A: np.ndarray, k: int, eps: float, delta: float, rng: np.random.Generator, r_out: int = 20,
                   frac_centres: float = 0.5, ell: int | None = None, factor: str | None = None,
                   adjacency: str = "rank_one", ell_c: int | None = 16, sensitivity: float = 1.0,
                   sampler: str = "cks"):
    """Sketch -> Lloyd on the sketch -> noisy centres, total (eps, delta)-DP.
    Returns (centres n x k, partition, info)."""
    m = A.shape[1]
    lam, rho_c = accounting.split_budget(eps, r_out, delta, frac_centres, sensitivity=sensitivity, m=m)
    if ell is None:
        ell = accounting.ell_for_utility(r_out, m, lam / sensitivity**2)
    factor = factor or ("cholesky" if m <= 3000 else "lowrank")
    rel = mechanism(A, lam, r_out, ell, rng, factor=factor, sampler=sampler)
    part = partition(rel.Z, k)
    C = noisy_centres(A, part, k, rho_c, rng, adjacency, ell_c, sensitivity)
    return C, part, dict(lam=lam, rho_c=rho_c, ell=ell, r_out=r_out, release=rel)
