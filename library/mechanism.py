"""The mechanism (Definition 5): public inner JLT, ridge, exact discrete Gaussian outer rows.

    A      n x m, records are columns
    M      = Pi_in A / sqrt(r_in)
    Sigma  = M^T M + lam I
    Z      = [x_1; ...; x_{r_out}] / sqrt(r_out),  x_i ~ DN on 2^-ell Z^m with parameter Sigma
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .factor import ridge_factor
from .inner_jlt import inner_sketch, inner_sketch_fast, next_pow2, sample_inner_jlt
from .sampler import grid_gaussian_rows, lowrank_gaussian_rows


@dataclass
class Release:
    Pi_in: np.ndarray | None   # public inner matrix (None with inner="fwht")
    M: np.ndarray              # inner sketch, r_in x m
    R: np.ndarray | None       # factor of Sigma (None with factor="lowrank")
    S: np.ndarray              # lattice coefficients, m x r_out
    Z_raw: np.ndarray          # rows on 2^-ell Z^m, r_out x m
    Z: np.ndarray              # released sketch Z_raw / sqrt(r_out)


def inner_stage(A, r_in, rng, inner="srht", Pi_in=None):
    """Returns (Pi_in, M).  r_in=None means the full rotation next_pow2(n)."""
    n = A.shape[0]
    r_in = next_pow2(n) if r_in is None else r_in
    if inner == "fwht":
        return None, inner_sketch_fast(A, r_in, rng)
    if Pi_in is None:
        Pi_in = sample_inner_jlt(n, r_in, rng, inner)
    return Pi_in, inner_sketch(A, Pi_in)


def mechanism(A: np.ndarray, lam: float, r_out: int, ell: int, rng: np.random.Generator, r_in: int | None = None,
              inner: str = "srht", sampler: str = "cks", factor: str = "cholesky",
              Pi_in: np.ndarray | None = None) -> Release:
    """Exact mechanism.  factor: "cholesky" (default), "qr", or "lowrank" (no m x m matrix; use for large m).
    All factor choices give identical output for the same seed."""
    Pi_in, M = inner_stage(A, r_in, rng, inner, Pi_in)
    if factor == "lowrank":
        R = None
        Z_raw, S, _ = lowrank_gaussian_rows(M, lam, r_out, ell, rng, sampler)
    else:
        R = ridge_factor(M, lam, factor)
        Z_raw, S = grid_gaussian_rows(R, r_out, ell, rng, sampler)
    return Release(Pi_in, M, R, S, Z_raw, Z_raw / math.sqrt(r_out))


def continuous_mechanism(A: np.ndarray, lam: float, r_out: int, rng: np.random.Generator, r_in: int | None = None,
                         inner: str = "srht", Pi_in: np.ndarray | None = None) -> np.ndarray:
    """Continuous counterpart (Remark 4): rows g^T M + sqrt(lam) n^T, scaled by 1/sqrt(r_out).  Not on a grid."""
    _, M = inner_stage(A, r_in, rng, inner, Pi_in)
    G = rng.standard_normal((r_out, M.shape[0]))
    return (G @ M + math.sqrt(lam) * rng.standard_normal((r_out, M.shape[1]))) / math.sqrt(r_out)
