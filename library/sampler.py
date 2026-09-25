"""Core sampler: r_out independent rows x in 2^-ell Z^m with Pr[x] ∝ exp(-x^T Sigma^{-1} x / 2).

grid_gaussian_rows     triangular sweep from a dense factor R (Sigma = R^T R)
lowrank_gaussian_rows  same rows for Sigma = M^T M + lam I without any m x m matrix (large m)
gpv_rows               literal GPV SampleD on 2^-ell (R^T)^{-1} Z^m, reference only
All three return identical rows for the same seed and 1-D sampler."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

from .sampler1d import SAMPLERS


def grid_gaussian_rows(R: np.ndarray, r_out: int, ell: int, rng: np.random.Generator, sampler: str = "cks"):
    """Coordinate i ~ DN on 2^-ell Z with centre sum_{j<i} R_ji s_j and width R_ii.
    Returns X (r_out x m) and S (m x r_out) with X = (R^T S)^T."""
    draw = SAMPLERS[sampler]
    m = R.shape[0]
    scale = 2.0**ell
    X = np.zeros((r_out, m))
    S = np.zeros((r_out, m))
    for i in range(m):
        c = S[:, :i] @ R[:i, i] if i else np.zeros(r_out)
        X[:, i] = draw(scale * c, scale * R[i, i], rng) / scale
        S[:, i] = (X[:, i] - c) / R[i, i]
    return X, S.T


def lowrank_gaussian_rows(M: np.ndarray, lam: float, r_out: int, ell: int, rng: np.random.Generator,
                          sampler: str = "cks", block: int = 256):
    """grid_gaussian_rows for Sigma = M^T M + lam I, O(m rows(M)^2) time.  Centres and widths are the Gaussian
    conditionals, computed from the rows(M) x rows(M) posterior of g in x = M^T g + sqrt(lam) n.
    Returns X, S and the diagonal of R."""
    draw = SAMPLERS[sampler]
    r, m = M.shape
    scale = 2.0**ell
    X = np.zeros((r_out, m))
    S = np.zeros((r_out, m))
    Rdiag = np.zeros(m)
    P = np.eye(r)
    b = np.zeros((r, r_out))
    for start in range(0, m, block):
        end = min(start + block, m)
        Mb = M[:, start:end]
        L = cholesky(P, lower=True)
        W = solve_triangular(L, Mb, lower=True)
        mean = (Mb.T @ cho_solve((L, True), b)).T
        Rb = cholesky(W.T @ W + lam * np.eye(end - start), lower=False)
        Sb = np.zeros((r_out, end - start))
        for i in range(end - start):
            c = mean[:, i] + (Sb[:, :i] @ Rb[:i, i] if i else 0.0)
            X[:, start + i] = draw(scale * c, scale * Rb[i, i], rng) / scale
            Sb[:, i] = (X[:, start + i] - c) / Rb[i, i]
        S[:, start:end] = Sb
        Rdiag[start:end] = np.diag(Rb)
        P += Mb @ Mb.T / lam
        b += Mb @ X[:, start:end].T / lam
    return X, S.T, Rdiag


def gram_schmidt(B: np.ndarray) -> np.ndarray:
    Bt = np.zeros_like(B)
    for i in range(B.shape[1]):
        v = B[:, i].copy()
        for j in range(i):
            v -= (B[:, i] @ Bt[:, j]) / (Bt[:, j] @ Bt[:, j]) * Bt[:, j]
        Bt[:, i] = v
    return Bt


def gpv_sample(B: np.ndarray, Bt: np.ndarray, s: float, c: np.ndarray, rng: np.random.Generator, sampler: str = "cks"):
    """GPV08 SampleD: one sample of D_{L(B), s, c}.  Returns (lattice vector, integer coefficients)."""
    draw = SAMPLERS[sampler]
    n = B.shape[1]
    c = c.astype(np.float64).copy()
    v = np.zeros(B.shape[0])
    z = np.zeros(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        bt = Bt[:, i]
        z[i] = draw(np.array([(c @ bt) / (bt @ bt)]), s / np.linalg.norm(bt), rng)[0]
        c -= z[i] * B[:, i]
        v += z[i] * B[:, i]
    return v, z


def gpv_rows(R: np.ndarray, r_out: int, ell: int, rng: np.random.Generator, sampler: str = "cks"):
    m = R.shape[0]
    P = (2.0 ** (-ell) * solve_triangular(R.T, np.eye(m), lower=True))[:, ::-1]
    Pt = gram_schmidt(P)
    X = np.zeros((r_out, m))
    S = np.zeros((m, r_out))
    for r in range(r_out):
        v, z = gpv_sample(P, Pt, 1.0, np.zeros(m), rng, sampler)
        S[:, r] = v
        X[r] = 2.0 ** (-ell) * z[::-1]
    return X, S
