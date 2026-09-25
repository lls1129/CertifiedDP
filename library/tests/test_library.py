"""python -m pytest library/tests   or   python library/tests/test_library.py   (from the folder containing library/)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from library import accounting as acc, kmeans, metrics, pe_means, regression, sampler, sampler1d
from library.factor import ridge_cholesky, ridge_qr
from library.mechanism import mechanism


def test_paper_tables():
    expect = {939: (1.105, 17), 1130: (1.000, 19), 2980: (0.592, 30), 1e6: (0.026, 479)}
    for lam, (e, a) in expect.items():
        eps, alpha = acc.eps_dp(lam, 40, 1e-6, 1e-9, 0.1)
        assert abs(eps - e) < 0.002 and abs(alpha - a) < 1.5
    assert [round(acc.lambda_for_target(e, 40, 1e-6, 1e-9, 0.1)) for e in (0.5, 1, 2)] == [4084, 1129, 317]
    assert round(acc.lambda_bar_dp(1.0, 40, 10_000, 1e-6, 1e-9, 0.1)) == 2980
    assert abs(acc.lambda_blocki(1, 1e-6, 40) / 6.1e7 - 1) < 0.01
    assert abs(acc.sigma2_gaussian_mechanism(1, 1e-6) - 20.5) < 0.1
    assert abs(acc.lambda_for_target(1, 40, sensitivity=4) / acc.lambda_for_target(1, 40) - 16) < 1e-6


def test_1d_samplers_match_pmf():
    rng = np.random.default_rng(0)
    for name, n in [("cks", 200_000), ("karney", 60_000)]:
        mu, sigma = 3.37, 2.5
        z = sampler1d.sample(np.full(n, mu), sigma, rng, name)
        ks = np.arange(-10, 18)
        p = np.exp(-(ks - mu) ** 2 / (2 * sigma**2))
        p /= p.sum()
        assert np.abs(np.array([(z == k).mean() for k in ks]) - p).max() < 5 / np.sqrt(n), name


def test_samplers_and_factors_agree():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((8, 6)) * 3
    R = ridge_cholesky(M, 50.0)
    assert np.allclose(R, ridge_qr(M, 50.0))
    for seed in range(5):
        X1, _ = sampler.grid_gaussian_rows(R, 1, 0, np.random.default_rng(seed))
        X2, _ = sampler.gpv_rows(R, 1, 0, np.random.default_rng(seed))
        assert np.array_equal(X1, X2)
    X, S = sampler.grid_gaussian_rows(R, 20_000, 0, rng)
    assert np.allclose(X, np.rint(X)) and np.allclose(X, (R.T @ S).T)
    assert np.abs(X.T @ X / 20_000 - R.T @ R).max() < 0.05 * np.abs(R.T @ R).max()
    A = rng.standard_normal((30, 400)) * 3
    for lam, ell in [(40.0, 0), (2.0, 5)]:
        Z = [mechanism(A, lam, 7, ell, np.random.default_rng(9), r_in=32, factor=f).Z_raw
             for f in ("cholesky", "qr", "lowrank")]
        assert np.array_equal(Z[0], Z[1]) and np.array_equal(Z[0], Z[2])


def test_scaling_identity():
    A = np.random.default_rng(1).random((20, 150))
    Z1 = mechanism(A, 30.0, 5, 3, np.random.default_rng(2)).Z_raw
    Z4 = mechanism(4 * A, 16 * 30.0, 5, 1, np.random.default_rng(2)).Z_raw
    assert np.array_equal(Z4, 4 * Z1)


def test_budget_and_metrics():
    lam, rho = acc.split_budget(1.0, 20, 1e-6, 0.5)
    assert abs(acc.eps_dp(lam, 20, 1e-6, rho_extra=rho)[0] - 1.0) < 1e-3
    rng = np.random.default_rng(0)
    A, C = rng.standard_normal((5, 300)), rng.standard_normal((5, 4))
    ref = np.mean([min(((A[:, i] - C[:, j]) ** 2).sum() for j in range(4)) for i in range(300)])
    assert abs(metrics.normalized_loss(A, C) - ref) < 1e-9
    assert metrics.label_accuracy(A, metrics.assign(A, C), C) == 1.0


def test_pipelines_run():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((6, 3)) * 20
    A = np.repeat(centres, 200, axis=1) + rng.standard_normal((6, 600))
    y = np.repeat(np.arange(3), 200)
    C, part, info = kmeans.private_kmeans(A, 3, 2.0, 1e-6, rng, r_out=5)
    assert metrics.cluster_accuracy(y, part, 3) > 0.9
    X = rng.standard_normal((2000, 5)) * 10
    W, _ = regression.private_ridge(X, X @ np.arange(1, 6.0), 2.0, 1e-6, rng, r_out=50, mu=1.0)
    assert np.abs(W.ravel() - np.arange(1, 6)).max() < 1.0


def test_pe_means_accounting_and_run():
    s = pe_means.get_noise_multiplier(1.0, 10, 1e-6)
    assert abs(pe_means.compute_epsilon(s, 10, 1e-6) - 1.0) < 1e-6
    rng = np.random.default_rng(0)
    A = np.repeat(rng.standard_normal((2, 4)), 500, axis=1) * 0.4 + 0.03 * rng.standard_normal((2, 2000))
    A /= np.linalg.norm(A, axis=0).max()
    C = pe_means.pe_means(A, 4, 1.0, 1 / 2000**1.1, rng)
    assert metrics.normalized_loss(A, C) < 0.5 * metrics.normalized_loss(A, np.zeros((2, 1)))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("all tests passed")
