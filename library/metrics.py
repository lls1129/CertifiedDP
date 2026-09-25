"""Evaluation.  A is n x m (records as columns), centres C are n x k."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score


def cluster_accuracy(y: np.ndarray, part: np.ndarray, k: int) -> float:
    """Hungarian-matched accuracy of a partition."""
    C = np.zeros((max(k, int(y.max()) + 1), max(k, int(part.max()) + 1)), dtype=np.int64)
    np.add.at(C, (y, part), 1)
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(y)


def nmi(y, part) -> float:
    return normalized_mutual_info_score(y, part)


def kmeans_cost(W: np.ndarray, part: np.ndarray) -> float:
    """Sum of squared distances to the part means (Definition 1)."""
    return float(sum(((W[:, part == c] - W[:, part == c].mean(axis=1, keepdims=True)) ** 2).sum()
                     for c in np.unique(part)))


def kmedians_cost(W: np.ndarray, part: np.ndarray) -> float:
    return float(sum(np.abs(W[:, part == c] - np.median(W[:, part == c], axis=1, keepdims=True)).sum()
                     for c in np.unique(part)))


def _sqdist(A, C):
    return np.maximum((A * A).sum(axis=0)[:, None] - 2 * A.T @ C + (C * C).sum(axis=0)[None, :], 0)


def assign(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    return _sqdist(A, C).argmin(axis=1)


def normalized_loss(A: np.ndarray, C: np.ndarray) -> float:
    """PE-means loss: (1/m) sum_j min_c ||a_j - c||^2."""
    return float(_sqdist(A, C).min(axis=1).mean())


def label_accuracy(A: np.ndarray, y: np.ndarray, C: np.ndarray) -> float:
    """Nearest-centre assignment, majority label per centre (PE-means / Google-LSH)."""
    a = assign(A, C)
    pred = np.empty_like(y)
    for c in np.unique(a):
        pred[a == c] = np.bincount(y[a == c]).argmax()
    return float((pred == y).mean())


def loss_auc(eps, losses) -> float:
    """Trapezoidal area under loss vs eps (PE-means uses eps in {0.25, 0.5, 1, 2, 4})."""
    eps, losses = np.asarray(eps, float), np.asarray(losses, float)
    return float(((losses[1:] + losses[:-1]) / 2 * np.diff(eps)).sum())


def mse(P: np.ndarray, Y: np.ndarray) -> float:
    return float(((P - Y) ** 2).sum(axis=1).mean())
