"""Public inner JLT (data independent).  Data A is n x m with records as columns; the sketch acts on the n rows."""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import hadamard


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def sample_inner_jlt(n: int, r_in: int, rng: np.random.Generator, kind: str = "srht") -> np.ndarray:
    """r_in x n matrix with entries in {-1, 0, 1}.
    srht: rows of a Walsh-Hadamard matrix times random signs; r_in >= next_pow2(n) gives the full rotation
    (kappa_in = 1, delta_in = 0).  rademacher: i.i.d. +-1."""
    if kind == "rademacher":
        return rng.integers(0, 2, size=(r_in, n)) * 2 - 1
    N = next_pow2(n)
    HD = hadamard(N) * (rng.integers(0, 2, size=N) * 2 - 1)
    rows = np.arange(N) if r_in >= N else rng.choice(N, size=r_in, replace=False)
    return HD[rows][:, :n]


def inner_sketch(A: np.ndarray, Pi_in: np.ndarray) -> np.ndarray:
    """M = Pi_in A / sqrt(rows(Pi_in))."""
    return (Pi_in @ A) / math.sqrt(Pi_in.shape[0])


def fwht(A: np.ndarray) -> np.ndarray:
    """In-place unnormalised Walsh-Hadamard transform along axis 0 (length a power of two)."""
    N = A.shape[0]
    h = 1
    while h < N:
        B = A.reshape(N // (2 * h), 2, h, -1)
        a, b = B[:, 0].copy(), B[:, 1].copy()
        B[:, 0], B[:, 1] = a + b, a - b
        h *= 2
    return A


def inner_sketch_fast(A: np.ndarray, r_in: int, rng: np.random.Generator) -> np.ndarray:
    """SRHT sketch without forming Pi_in, O(n log n) per column.  Use when n is large (e.g. regression)."""
    n, m = A.shape
    N = next_pow2(n)
    B = np.zeros((N, m))
    B[:n] = A * (rng.integers(0, 2, size=n) * 2 - 1)[:, None]
    fwht(B)
    if r_in >= N:
        return B / math.sqrt(N)
    return B[rng.choice(N, size=r_in, replace=False)] / math.sqrt(r_in)


def inner_privacy_params(n: int, r_in: int, kind: str = "srht", delta_in: float = 1e-9) -> tuple[float, float]:
    """(eps_in, delta_in) for the privacy accounting.  Full rotation: (0, 0).  Rademacher: Achlioptas tail."""
    if kind == "srht" and r_in >= next_pow2(n):
        return 0.0, 0.0
    if kind != "rademacher":
        raise ValueError("eps_in is only available for the full rotation or a Rademacher sketch")
    target = math.log(1.0 / delta_in) / r_in
    f = lambda e: e * e / 4 - e**3 / 6
    if f(1.0) < target:
        return math.inf, delta_in
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) >= target else (mid, hi)
    return hi, delta_in
