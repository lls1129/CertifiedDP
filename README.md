# CertifiedDP

Importable components of the certified private fast JLT mechanism, plus PE-means for comparison.
Python >= 3.9 with numpy, scipy and scikit-learn; torch and torchvision only for `download.load_mnist_lenet`.
Run everything from the repository root so that `import library` works.

| file | what's in it |
|---|---|
| `inner_jlt.py` | the public inner JLT (Hadamard, fast transform, Rademacher) and its privacy parameters (ε_in, δ_in) |
| `factor.py` | `ridge_cholesky` (default) and `ridge_qr`; they give the same R |
| `sampler1d.py` | 1-D discrete Gaussians: CKS (default) and Karney |
| `sampler.py` | core sampler: the dense sweep, the low-rank version for large m, and the literal GPV reference |
| `mechanism.py` | `mechanism(A, lam, r_out, ell, rng, factor="cholesky" \| "qr" \| "lowrank")` and the continuous version |
| `accounting.py` | ε(λ), λ(ε) with a `sensitivity` argument, budget split with the centroid release, choice of ℓ, λ_U |
| `preprocess.py` | array layout, unit-ball scaling, norm clipping, one-hot |
| `download.py` | MNIST loader, and the LeNet features (need torch, first run only) |
| `metrics.py` | clustering and regression metrics, plus PE-means' loss, label accuracy and AUC |
| `kmeans.py` / `regression.py` | building blocks plus one-call pipelines `private_kmeans` and `private_ridge` |
| `baselines.py` | Gaussian input perturbation, noisy AᵀA |
| `pe_means.py` | PE-means and HDPE-means, ported to numpy from their MIT-licensed repository, with their DP accounting |
| `datasets.py` | the 34 datasets of PE-means' Table 1, downloaded and preprocessed as in their code |

## Conventions

- Clustering: `A` is n x m with **records as columns**; centres are n x k.
- Regression: records are **rows** of `X` (N x d) and `Y` (N x t); `private_ridge` builds `A = [X | Y]`.
- Privacy protects any rank-one change `A - Ã` of operator norm <= `sensitivity` (default 1) **in the units of A**.
  Scale the data on purpose: after `preprocess.centre_max_norm`, norm 1 is a whole record. λ scales with
  `sensitivity**2`. Pass `m` (number of columns) to the accounting functions so the lattice term is included.
- Released rows lie on the grid `2**-ell * Z` before the `1/sqrt(r_out)` scaling; `accounting.ell_for_utility`
  picks `ell`.
- `factor="cholesky" | "qr" | "lowrank"` give identical output for the same seed; use `"lowrank"` when m is in
  the thousands or more (no m x m matrix; about 10 s at m = 70000).
- `pe_means.hdpe_means` follows the paper's Algorithm 3; `match_repo=True` reproduces the authors' code, which
  adds one scalar noise to every coordinate of a cluster sum and calibrates the final step for two fewer steps.

## Use

```python
import numpy as np
from library import accounting, kmeans, metrics, preprocess, regression
from library.mechanism import mechanism

A = preprocess.centre_max_norm(preprocess.from_array(X))      # X: samples x features

# building blocks
lam = accounting.lambda_for_target(1.0, r_out=20, delta=1e-6, m=A.shape[1])
rel = mechanism(A, lam, r_out=20, ell=5, rng=np.random.default_rng(0), factor="lowrank")
Z = rel.Z                                                       # 20 x m released sketch

# pipelines
C, part, info = kmeans.private_kmeans(A, k=10, eps=1.0, delta=1e-6, rng=np.random.default_rng(0))
print(metrics.normalized_loss(A, C))
W, info = regression.private_ridge(X_train, Y_train, eps=1.0, delta=1e-6, rng=np.random.default_rng(0), r_out=800)
```

Examples (downloaded data goes to `./data`):

```bash
python library/tests/test_library.py        # about 3 s
python library/examples/other_dataset.py     # sklearn digits (k-means) and diabetes (regression)
python library/examples/regression_mnist.py  # MNIST one-vs-all ridge
python library/examples/kmeans_mnist.py      # ours vs PE-means / HDPE-means on MNIST-LeNet (about 6 min)
python library/examples/pe_means_table.py --seeds 3 --out results/pe_means_table.csv   # all 34 datasets (hours)
python library/examples/pe_means_table_summary.py results/pe_means_table*.csv > results/pe_means_table.md
```

`results/pe_means_table.md` holds the finished table (3 seeds), next to the numbers reported in their paper.
