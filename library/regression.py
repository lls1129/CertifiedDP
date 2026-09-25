"""Ridge regression from the sketch.  Records are ROWS here: A = [X | Y] is N x (d + t), the mechanism runs with
n = N (records are the ambient dimension) and m = d + t, and Z^T Z ≈ A^T A + lam I.
Neighbours: one row (x_i, y_i) changed by norm <= sensitivity, or any rank-one change of that size."""
from __future__ import annotations

import numpy as np

from . import accounting
from .mechanism import mechanism


def fit_from_gram(Gxx: np.ndarray, Gxy: np.ndarray, mu: float = 0.0) -> np.ndarray:
    return np.linalg.lstsq(Gxx + mu * np.eye(Gxx.shape[0]), Gxy, rcond=None)[0]


def fit_from_sketch(Z: np.ndarray, d: int, mu: float = 0.0) -> np.ndarray:
    """W (d x t) = (Z_x^T Z_x + mu I)^{-1} Z_x^T Z_y from a released Z (r_out x (d + t))."""
    G = Z.T @ Z
    return fit_from_gram(G[:d, :d], G[:d, d:], mu)


def reference(X: np.ndarray, Y: np.ndarray, mu: float = 0.0) -> np.ndarray:
    """Non-private ridge."""
    return fit_from_gram(X.T @ X, X.T @ Y, mu)


def predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ W


def private_ridge(X: np.ndarray, Y: np.ndarray, eps: float, delta: float, rng: np.random.Generator,
                  r_out: int = 800, mu: float | list = 3e4, ell: int | None = None, sensitivity: float = 1.0):
    """(eps, delta)-DP ridge weights.  X: N x d, Y: N x t (one-hot for classification).
    mu may be a list (one fit per value, all from the same release).  Returns (W or [W...], info)."""
    Y = Y.reshape(len(Y), -1)
    A = np.hstack([X, Y])
    d, m = X.shape[1], A.shape[1]
    lam = accounting.lambda_for_target(eps, r_out, delta, m=m, sensitivity=sensitivity)
    if ell is None:
        ell = accounting.ell_for_utility(r_out, m, lam / sensitivity**2)
    rel = mechanism(A, lam, r_out, ell, rng, inner="fwht")
    Ws = [fit_from_sketch(rel.Z, d, u) for u in np.atleast_1d(mu)]
    return (Ws if np.ndim(mu) else Ws[0]), dict(lam=lam, ell=ell, r_out=r_out, release=rel)
