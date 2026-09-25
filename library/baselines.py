"""Simple Gaussian-mechanism baselines with the same RDP conversion as the mechanism."""
from __future__ import annotations

import math

import numpy as np

from .accounting import sigma2_gaussian_mechanism


def input_perturbation(A: np.ndarray, eps: float, delta: float, rng: np.random.Generator, sensitivity: float = 1.0):
    """A + N(0, sigma^2) per entry.  Frobenius sensitivity = operator sensitivity for rank-one changes."""
    return A + math.sqrt(sigma2_gaussian_mechanism(eps, delta, sensitivity)) * rng.standard_normal(A.shape)


def noisy_gram(A: np.ndarray, eps: float, delta: float, rng: np.random.Generator, sensitivity: float):
    """A^T A + symmetric Gaussian noise.  Needs a public bound: for one row changed by norm <= 1 with all rows of
    norm <= B, sensitivity = 2B + 1 (a general rank-one change has no such bound)."""
    E = rng.standard_normal((A.shape[1], A.shape[1])) * math.sqrt(sigma2_gaussian_mechanism(eps, delta, sensitivity))
    return A.T @ A + np.triu(E) + np.triu(E, 1).T
