"""PE-means and HDPE-means (Humphries, Lin, Yekhanin, arXiv 2606.00342), ported to numpy from
github.com/t3humphries/PE-means (MIT License, Copyright (c) 2026 Thomas Humphries).

Their setting: records scaled into the unit ball (preprocess.centre_max_norm), add/remove-one-record adjacency,
delta = 1/N^1.1, eps in EPSILONS.  Same layout as the rest of the library: A is d x N (records as columns),
centres are returned as d x k.

hdpe_means follows the paper's Algorithm 3.  match_repo=True reproduces two differences in their code: a single
scalar noise added to every coordinate of a cluster sum, and the final step's noise multiplier computed for
num_gen instead of num_gen + 2 steps."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import gamma as gamma_fn
from scipy.stats import norm
from sklearn.cluster import KMeans

EPSILONS = [0.25, 0.5, 1.0, 2.0, 4.0]
PARAMS = dict(init_mode="sphere_packing", var_scale=0.01, levy_beta=1.75, L_reduce_threshold=1.0, L_reduce_factor=0.5)


# ---------------------------------------------------------------- Gaussian DP accounting (from Microsoft DPSDA)

def delta_gaussian(eps: float, mu: float) -> float:
    if mu == 0 or eps > math.log(np.finfo(float).max):
        return 0.0
    return norm.cdf(-eps / mu + mu / 2) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_gaussian(delta: float, mu: float) -> float:
    f = lambda x: delta_gaussian(x, mu) - delta
    if f(0.0) <= 0:
        return 0.0
    b = 1.0
    while f(b) > 0:
        b *= 2
    return root_scalar(f, bracket=[0.0, b], method="brentq").root


def compute_epsilon(noise_multiplier: float, num_steps: int, delta: float) -> float:
    """eps of num_steps Gaussian mechanisms with sensitivity 1 and noise std noise_multiplier (mu-GDP)."""
    return math.inf if noise_multiplier == 0 else eps_gaussian(delta, math.sqrt(num_steps) / noise_multiplier)


def get_noise_multiplier(eps: float, num_steps: int, delta: float, lo: float = 0.1, hi: float = 500.0) -> float:
    if eps == math.inf:
        return 0.0
    return root_scalar(lambda x: compute_epsilon(x, num_steps, delta) - eps, bracket=[lo, hi], method="brentq").root


def default_schedule(N: int, d: int, k: int, eps: float):
    """(num_gen, L) as in their experiments/run_ours.py."""
    base = int(4 * math.sqrt(d))
    return max(int(eps * base), 1) if eps > 1 else max(base, 1), int(max(N / (5 * k), 4))


# ---------------------------------------------------------------- evolution operators

def pack_in_sphere(num_points: int, dims: int, radius: float, rng: np.random.Generator, max_retries: int = 100):
    """Sphere-packing initialisation (adapted by them from IBM diffprivlib)."""
    prox = radius / 2.0
    while prox > 0:
        C = np.zeros((num_points, dims))
        count, retry = 0, 0
        while retry < max_retries and count < num_points:
            v = rng.normal(0, 1, dims)
            cand = v / np.linalg.norm(v) * rng.random() ** (1.0 / dims) * (radius - prox)
            if count == 0 or np.sqrt(((C[:count] - cand) ** 2).sum(axis=1).min()) >= 2 * prox:
                C[count] = cand
                count, retry = count + 1, 0
            else:
                retry += 1
        if count >= num_points:
            return C
        prox /= 2.0


def uniform_ball(num_points: int, dims: int, radius: float, rng: np.random.Generator):
    v = rng.standard_normal((num_points, dims))
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300)
    return v * radius * rng.random((num_points, 1)) ** (1.0 / dims)


def levy_mutation(C: np.ndarray, L: int, radius: float, beta: float, step_scale: float, rng: np.random.Generator):
    """L Levy-flight variations of each row of C (Mantegna), projected back into the ball."""
    T = np.repeat(C, L, axis=0)
    if beta >= 2.0:
        steps = rng.normal(0, 1, T.shape)
    else:
        s_u = (gamma_fn(1 + beta) * math.sin(math.pi * beta / 2)
               / (gamma_fn((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        steps = rng.normal(0, s_u, T.shape) / np.abs(rng.normal(0, 1, T.shape)) ** (1 / beta)
    out = T + step_scale * radius * steps
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(n > radius, out / n * radius, out)


def nn_histogram(X: np.ndarray, pop: np.ndarray, noise_multiplier: float, rng: np.random.Generator, chunk: int = 2048):
    """DP nearest-neighbour histogram: each record votes for its nearest candidate, plus N(0, sigma^2) per bin."""
    pn = (pop * pop).sum(axis=1)
    votes = np.concatenate([(pn[None, :] - 2 * X[i:i + chunk] @ pop.T).argmin(axis=1) for i in range(0, len(X), chunk)])
    h = np.bincount(votes, minlength=len(pop)).astype(float)
    return h + noise_multiplier * rng.standard_normal(len(pop)) if noise_multiplier > 0 else h


def find_threshold(h: np.ndarray, N: int) -> float:
    a = np.sort(h)
    idx = np.searchsorted(np.cumsum(a[::-1]), N)
    return 0.0 if idx >= len(a) else a[len(a) - 1 - idx]


def weighted_kmeans(pop: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit(pop, sample_weight=w).cluster_centers_


# ---------------------------------------------------------------- algorithms

def pe_means(A: np.ndarray, k: int, eps: float, delta: float, rng: np.random.Generator, num_gen: int | None = None,
             L: int | None = None, radius: float | None = None, params: dict | None = None, extra_steps: int = 0,
             history: list | None = None) -> np.ndarray:
    """Algorithm 1.  extra_steps: Gaussian mechanisms reserved for later (HDPE-means uses 2)."""
    p = {**PARAMS, **(params or {})}
    X = A.T
    N, d = X.shape
    radius = float(np.linalg.norm(X, axis=1).max()) if radius is None else radius
    g0, L0 = default_schedule(N, d, k, eps)
    num_gen, L = num_gen or g0, L or L0
    sigma = get_noise_multiplier(eps, num_gen + extra_steps, delta)
    size = (L + 1) * k
    pop = pack_in_sphere(size, d, radius, rng) if p["init_mode"] == "sphere_packing" else \
        uniform_ball(size, d, radius, rng)
    for gen in range(num_gen):
        h = nn_histogram(X, pop, sigma, rng)
        h = np.where(h >= find_threshold(h, N), h, 0)
        if p["L_reduce_threshold"] is not None and sigma > 0:
            if (h**2).sum() / (len(h) * sigma**2) < p["L_reduce_threshold"]:
                L = max(int(p["L_reduce_factor"] * L), 4)
        top = weighted_kmeans(pop, h, k)
        if history is not None:
            history.append(top.T.copy())
        if gen == num_gen - 1:
            break
        pop = np.vstack([top, levy_mutation(top, L, radius, p["levy_beta"], p["var_scale"], rng)])
    return top.T


def hdpe_means(A: np.ndarray, k: int, eps: float, delta: float, rng: np.random.Generator, new_dim: int = 16,
               L: int | None = None, radius: float | None = None, params: dict | None = None,
               match_repo: bool = False, proj_seed: int = 42) -> np.ndarray:
    """Algorithm 3: public Gaussian JL projection to new_dim (their fixed seed 42), PE-means there, one noisy
    averaging step in R^d."""
    X = A.T
    N, d = X.shape
    radius = float(np.linalg.norm(X, axis=1).max()) if radius is None else radius
    if d <= 32:
        return pe_means(A, k, eps, delta, rng, L=L, radius=radius, params=params, extra_steps=2)
    Xl = X @ (np.random.default_rng(proj_seed).standard_normal((d, new_dim)) / math.sqrt(new_dim))
    num_gen = int(4 * math.sqrt(new_dim)) - 2
    Cl = pe_means(Xl.T, k, eps, delta, rng, num_gen=num_gen, L=L or default_schedule(N, d, k, eps)[1],
                  radius=radius, params=params, extra_steps=2)
    lab = ((Xl * Xl).sum(1)[:, None] - 2 * Xl @ Cl + (Cl * Cl).sum(0)[None, :]).argmin(axis=1)
    sigma = get_noise_multiplier(eps, num_gen if match_repo else num_gen + 2, delta)
    out = []
    for c in range(k):
        P = X[lab == c]
        if len(P):
            noise = rng.normal() if match_repo else rng.standard_normal(d)
            out.append((P.sum(axis=0) + noise * sigma * radius) / (len(P) + rng.normal() * sigma))
    return np.array(out).T
