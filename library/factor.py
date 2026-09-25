"""Upper-triangular R with R^T R = M^T M + lam I and positive diagonal.  Both methods give the same R
(to float64 rounding), hence the same mechanism output for the same seed."""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cholesky


def ridge_cholesky(M: np.ndarray, lam: float) -> np.ndarray:
    return cholesky(M.T @ M + lam * np.eye(M.shape[1]), lower=False)


def ridge_qr(M: np.ndarray, lam: float) -> np.ndarray:
    """Thin QR of [M; sqrt(lam) I] (the draft's Figure 3)."""
    R = np.linalg.qr(np.vstack([M, math.sqrt(lam) * np.eye(M.shape[1])]), mode="r")
    return np.sign(np.diag(R))[:, None] * R


def ridge_factor(M: np.ndarray, lam: float, method: str = "cholesky") -> np.ndarray:
    return {"cholesky": ridge_cholesky, "qr": ridge_qr}[method](M, lam)
