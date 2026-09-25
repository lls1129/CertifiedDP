"""Any (samples x features) array: sklearn's digits for k-means, diabetes for regression (both ship with sklearn).
python library/examples/other_dataset.py"""
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes, load_digits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from library import kmeans, metrics, pe_means, preprocess, regression

# clustering: records as columns, scaled into the unit ball (a change of norm 1 = a whole record)
X, y = load_digits(return_X_y=True)
A = preprocess.centre_max_norm(preprocess.from_array(X))
k, delta = 10, 1 / A.shape[1] ** 1.1
C0 = kmeans.centres(A, kmeans.partition(A, k), k)
print(f"digits  non-private loss {metrics.normalized_loss(A, C0):.4f}")
for eps in (1.0, 4.0):
    rng = np.random.default_rng(0)
    C, part, info = kmeans.private_kmeans(A, k, eps, delta, rng, r_out=10)
    Cpe = pe_means.pe_means(A, k, eps, delta, rng)
    print(f"digits  eps={eps}  ours loss {metrics.normalized_loss(A, C):.4f} (lam={info['lam']:.0f}, ell={info['ell']})"
          f"   pe_means loss {metrics.normalized_loss(A, Cpe):.4f}")

# regression: records as rows, each record (x_i, y_i) scaled to norm <= 1
X, t = load_diabetes(return_X_y=True)
XY = preprocess.clip_norm(preprocess.from_array(np.hstack([X, t[:, None] / np.abs(t).max()]))).T
Xs, ts = XY[:, :-1], XY[:, -1:]
W0 = regression.reference(Xs, ts, mu=1e-3)
print(f"diabetes non-private mse {metrics.mse(regression.predict(W0, Xs), ts):.5f}")
for eps in (1.0, 4.0):
    W, info = regression.private_ridge(Xs, ts, eps, 1e-6, np.random.default_rng(0), r_out=50, mu=1.0)
    print(f"diabetes eps={eps}  mse {metrics.mse(regression.predict(W, Xs), ts):.5f} (lam={info['lam']:.0f})")
