"""Privacy accounting (Theorem 5, Corollary 7) and calibration.

Neighbours: A - A~ rank one with operator norm <= sensitivity (Definition 3).  All lam values are for
sensitivity 1 in the units of A; pass sensitivity=Delta to protect changes of norm Delta (lam scales by Delta^2).
kappa_in^2 = 1 + eps_in; the full-rotation inner JLT has eps_in = delta_in = 0.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar


def U_alpha(alpha: float, kappa2: float, lam: float) -> float:
    x = alpha * (alpha - 1) * kappa2 / lam
    return math.inf if x >= 1 else -0.5 / (alpha - 1) * math.log1p(-x)


def lattice_term(alpha: float, lam: float, m: int, ell: int) -> float:
    q = math.exp(-2 * math.pi**2 * 4**ell * lam / alpha)
    return alpha * m / (alpha - 1) * math.log((1 + q) / (1 - q))


def B_alpha_ell(alpha: float, kappa2: float, lam: float, m: int = 1, ell: int = 0) -> float:
    """Per-row RDP bound (Theorem 10)."""
    return U_alpha(alpha, kappa2, lam) + lattice_term(alpha, lam, m, ell)


def rdp_to_dp(eps_rdp: float, alpha: float, delta: float) -> float:
    """Conversion (49)."""
    return max(0.0, eps_rdp + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1))


def eps_dp_at_alpha(alpha, lam, r_out, delta_prime, eps_in=0.0, m=1, ell=0, rho_extra=0.0) -> float:
    B = B_alpha_ell(alpha, 1 + eps_in, lam, m, ell)
    return math.inf if not math.isfinite(B) else rdp_to_dp(r_out * B + alpha * rho_extra, alpha, delta_prime)


def eps_dp(lam, r_out, delta=1e-6, delta_in=0.0, eps_in=0.0, m=1, ell=0, rho_extra=0.0, sensitivity=1.0):
    """Best (eps, alpha) for the sketch; rho_extra adds an (alpha, alpha * rho_extra)-RDP mechanism
    (e.g. the centre release) composed with it."""
    lam = lam / sensitivity**2
    delta_prime = delta - delta_in
    alpha_max = 0.5 * (1 + math.sqrt(1 + 4 * lam / (1 + eps_in))) - 1e-9
    f = lambda a: eps_dp_at_alpha(a, lam, r_out, delta_prime, eps_in, m, ell, rho_extra)
    grid = np.geomspace(1e-3, alpha_max - 1 + 1e-9, 400) + 1.0
    grid = grid[grid < alpha_max]
    if grid.size == 0:
        return math.inf, math.nan
    vals = np.array([f(a) for a in grid])
    j = int(np.argmin(vals))
    res = minimize_scalar(f, bounds=(grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]), method="bounded")
    return (float(res.fun), float(res.x)) if res.fun < vals[j] else (float(vals[j]), float(grid[j]))


def lambda_for_target(eps, r_out, delta=1e-6, delta_in=0.0, eps_in=0.0, m=1, ell=0, rho_extra=0.0, sensitivity=1.0):
    """Smallest lam with eps_dp(lam) <= eps."""
    g = lambda lam: eps_dp(lam, r_out, delta, delta_in, eps_in, m, ell, rho_extra)[0]
    lo, hi = 1e-3, 1e3
    while g(hi) > eps:
        hi *= 2
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if g(mid) <= eps else (mid, hi)
    return hi * sensitivity**2


def rho_for_target(eps, delta=1e-6) -> float:
    """Largest rho such that an (alpha, alpha rho)-RDP mechanism alone is (eps, delta)-DP."""
    a = np.geomspace(1.0001, 1e5, 4000)
    eps_of = lambda rho: float(np.maximum(0.0, a * rho + np.log((a - 1) / a) - (math.log(delta) + np.log(a)) / (a - 1)).min())
    lo, hi = 1e-8, 1e4
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if eps_of(mid) <= eps else (lo, mid)
    return lo


def split_budget(eps, r_out, delta, frac_centres, sensitivity=1.0, **kw):
    """(lam, rho_c) for sketch + Gaussian centre release, total (eps, delta).  The centre noise is that of an
    (frac_centres * eps, delta) release on its own; lam is the smallest ridge keeping the total at eps."""
    rho_c = rho_for_target(frac_centres * eps, delta)
    return lambda_for_target(eps, r_out, delta, rho_extra=rho_c, sensitivity=sensitivity, **kw), rho_c


def lambda_bar_dp(eps, r_out, m, delta=1e-6, delta_in=0.0, eps_in=0.0, ell=0) -> float:
    """Closed-form sufficient lam, (54)."""
    D = math.log(1.0 / (delta - delta_in))
    a0 = 1 + 2 * D / eps
    term1 = a0 * (a0 - 1) * (1 + eps_in) / (1 - math.exp(-D / r_out))
    term2 = a0 * 4.0 ** (-ell) / (2 * math.pi**2) * math.log(1 / math.tanh(D / (4 * a0 * m * r_out)))
    return max(term1, term2)


def lambda_asymptotic(eps, r_out, delta=1e-6, delta_in=0.0, eps_in=0.0) -> float:
    """Remark 14: lam ~ 2 kappa^2 r_out log(1/delta') / eps^2."""
    return 2 * (1 + eps_in) * r_out * math.log(1 / (delta - delta_in)) / eps**2


def lambda_U(t, eps_out, OPT, m, k, eps_in=0.0) -> float:
    """Utility ceiling of Corollary 2 with gamma = 1 and no rounding term."""
    um, up = (1 - eps_in) * (1 - eps_out), (1 + eps_in) * (1 + eps_out)
    return max(0.0, ((1 + t) * um - up) * OPT / (2 * eps_out * (m - k)))


def ell_for_utility(r_out: int, m: int, lam: float, K0_max: float = 0.1) -> int:
    """Smallest ell with K0 = r_out m 4^-ell / (24 lam) <= K0_max (Lemma 3 / Theorem 2)."""
    return max(0, math.ceil(0.5 * math.log2(r_out * m / (24 * lam * K0_max))))


def sigma2_gaussian_mechanism(eps, delta, sensitivity=1.0) -> float:
    """Per-coordinate variance of the Gaussian mechanism, same RDP conversion."""
    return sensitivity**2 / (2 * rho_for_target(eps, delta))


def lambda_blocki(eps, delta, r_out) -> float:
    """[BBDS12] baseline."""
    w = 16 * math.sqrt(r_out * math.log(2 / delta)) * math.log(16 * r_out / delta) / eps
    return w * w


def amplified_eps(lam, r_out, gamma, delta=1e-6, **kw):
    """Poisson gamma-subsampling with record-indexed columns, single-record neighbours only: eps is NOT
    amplified (the release shows whether a record was sampled); delta is (delta_0 = delta / gamma)."""
    if gamma >= 1:
        return eps_dp(lam, r_out, delta, **kw)
    return eps_dp(lam, r_out, min(delta / gamma, 0.5), **kw)
