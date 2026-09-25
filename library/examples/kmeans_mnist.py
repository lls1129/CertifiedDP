"""DP k-means on MNIST-LeNet (PE-means' setting): ours vs PE-means / HDPE-means, normalized loss and label accuracy.
Run from the folder containing library/:  python library/examples/kmeans_mnist.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from library import download, kmeans, metrics, pe_means, preprocess

X, y = download.load_mnist_lenet("data")                  # 70000 x 84 (needs torch on first use)
A = preprocess.centre_max_norm(preprocess.from_array(X))  # 84 x 70000, records in the unit ball
k, N = 10, A.shape[1]
delta = 1 / N**1.1

C0 = kmeans.centres(A, kmeans.partition(A, k), k)
print(f"non-private  loss {metrics.normalized_loss(A, C0):.4f}  acc {metrics.label_accuracy(A, y, C0):.3f}")
for eps in (1.0, 4.0):
    rng = np.random.default_rng(0)
    runs = {"ours": lambda: kmeans.private_kmeans(A, k, eps, delta, rng, r_out=20)[0],
            "pe_means": lambda: pe_means.pe_means(A, k, eps, delta, rng),
            "hdpe_means": lambda: pe_means.hdpe_means(A, k, eps, delta, rng)}
    for name, run in runs.items():
        C = run()
        print(f"eps={eps}  {name:<11} loss {metrics.normalized_loss(A, C):.4f}  acc {metrics.label_accuracy(A, y, C):.3f}")
