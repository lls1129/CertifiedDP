"""DP one-vs-all ridge on MNIST pixels from the sketch.  python library/examples/regression_mnist.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from library import download, preprocess, regression

X, y = download.load_mnist("train", data_dir="data")      # 60000 x 784 in [0, 1]
Xtr, ytr, Xte, yte = X[:50000], y[:50000], X[50000:], y[50000:]
Ytr = preprocess.one_hot(ytr, 10)

W0 = regression.reference(Xtr, Ytr, mu=1e4)
print("non-private acc", (regression.predict(W0, Xte).argmax(1) == yte).mean())
for eps in (1.0, 4.0):
    Ws, info = regression.private_ridge(Xtr, Ytr, eps, 1e-6, np.random.default_rng(0), r_out=800, mu=[1e4, 3e4, 1e5])
    accs = [(regression.predict(W, Xte).argmax(1) == yte).mean() for W in Ws]
    print(f"eps={eps}: lam={info['lam']:.0f} ell={info['ell']}  acc for mu=1e4/3e4/1e5: " + " ".join(f"{a:.3f}" for a in accs))
