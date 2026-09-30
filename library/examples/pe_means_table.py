"""Reproduce PE-means' Table 1 setting on all 34 datasets: normalized k-means loss for eps in {0.25, 0.5, 1, 2, 4},
delta = 1/N^1.1, for non-private Lloyd, our mechanism (sketch -> Lloyd -> noisy centres), PE-means and HDPE-means
(d > 32, as in their code).  One CSV row per (dataset, method, eps, seed); rows are appended as they finish, so a
partial run can be summarised with pe_means_table_summary.py.

python library/examples/pe_means_table.py --seeds 3 --workers 8 --out results/pe_means_table.csv"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EPS = [0.25, 0.5, 1.0, 2.0, 4.0]
FIELDS = ["dataset", "N", "d", "k", "method", "eps", "seed", "loss", "label_acc", "time_s"]
_cache = {}


def _init():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"


def _data(name, data_dir):
    if name not in _cache:
        from library import datasets
        _cache[name] = datasets.load(name, data_dir)
    return _cache[name]


def job(args):
    name, method, eps, seed, data_dir, r_out = args
    import numpy as np
    from library import kmeans, metrics, pe_means
    A, k, y = _data(name, data_dir)
    d, N = A.shape
    delta = 1 / N**1.1
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if method == "nonpriv":
        C = kmeans.centres(A, kmeans.partition(A, k, seed), k)
    elif method == "ours":
        C = kmeans.private_kmeans(A, k, eps, delta, rng, r_out=r_out)[0]
    elif method == "pe_means":
        C = pe_means.pe_means(A, k, eps, delta, rng)
    elif method == "hdpe_means":
        C = pe_means.hdpe_means(A, k, eps, delta, rng)
    elif method == "hdpe_means_repo":                              # their code rather than their Algorithm 3
        C = pe_means.hdpe_means(A, k, eps, delta, rng, match_repo=True)
    return dict(dataset=name, N=N, d=d, k=k, method=method, eps=eps, seed=seed, loss=metrics.normalized_loss(A, C),
                label_acc=metrics.label_accuracy(A, y, C) if y is not None else "", time_s=round(time.time() - t0, 1))


def main():
    from library import datasets
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=datasets.NAMES)
    ap.add_argument("--methods", nargs="*", default=["nonpriv", "ours", "pe_means", "hdpe_means"])
    ap.add_argument("--eps", type=float, nargs="*", default=EPS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--r-out", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/pe_means_table.csv")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r["dataset"], r["method"], float(r["eps"]), int(r["seed"])) for r in csv.DictReader(f)}
    jobs = []
    for name in a.datasets:
        d = datasets.load(name, a.data_dir)[0].shape[0]                 # also triggers the download
        for method in a.methods:
            if method.startswith("hdpe_means") and d <= 32:
                continue
            for eps in ([math.inf] if method == "nonpriv" else a.eps):
                for seed in range(a.seeds):
                    if (name, method, eps, seed) not in done:
                        jobs.append((name, method, eps, seed, a.data_dir, a.r_out))
    jobs.sort(key=lambda j: -datasets.load(j[0], a.data_dir)[0].shape[1])   # largest first
    print(f"{len(jobs)} jobs, {len(done)} already done", flush=True)
    with open(out, "a", newline="") as f, Pool(a.workers, initializer=_init) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            w.writeheader()
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            w.writerow(r)
            f.flush()
            print(f"[{i}/{len(jobs)}] {r['dataset']:<15} {r['method']:<11} eps={r['eps']:<5} seed={r['seed']} "
                  f"loss={r['loss']:.4f} {r['time_s']}s", flush=True)


if __name__ == "__main__":
    main()
