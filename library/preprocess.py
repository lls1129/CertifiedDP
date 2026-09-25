"""Data layout and scaling.  The library uses A of shape n x m with records as COLUMNS (clustering); for regression
the records are rows of A = [X | Y] (see regression.py).

Scale matters: lam protects changes of norm 1 (times `sensitivity`) in the units of A, so choose the units on
purpose (e.g. centre_max_norm => a change of norm 1 is a whole record)."""
from __future__ import annotations

import numpy as np


def from_array(X: np.ndarray) -> np.ndarray:
    """(samples x features) -> A (features x samples)."""
    return np.asarray(X, dtype=np.float64).T


def centre_max_norm(A: np.ndarray) -> np.ndarray:
    """Subtract the mean record, scale so max_j ||a_j|| = 1 (PE-means / Google-LSH preprocessing; non-private)."""
    Ac = A - A.mean(axis=1, keepdims=True)
    return Ac / np.linalg.norm(Ac, axis=0).max()


def unit_norm(A: np.ndarray) -> np.ndarray:
    return A / np.linalg.norm(A, axis=0, keepdims=True)


def clip_norm(A: np.ndarray, C: float = 1.0) -> np.ndarray:
    n = np.linalg.norm(A, axis=0, keepdims=True)
    return A * np.minimum(1.0, C / np.maximum(n, 1e-300))


def one_hot(y: np.ndarray, t: int | None = None) -> np.ndarray:
    return np.eye(int(y.max()) + 1 if t is None else t)[y]
