"""The 34 datasets of PE-means (arXiv 2606.00342, Table 1), built as in their setup/download_datasets.sh and
setup/generate_datasets.ipynb, preprocessed as in their src/helpers.py (centre, then scale to max l2 norm 1).

load(name, data_dir) -> (A, k, y) with A of shape d x N (records as columns), y labels or None.
Files are downloaded into data_dir on first use (scale datasets: one 2.2 GB archive)."""
from __future__ import annotations

import io
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ZENODO = "https://zenodo.org/records/15615521/files/"
SIPU = "https://cs.joensuu.fi/sipu/datasets/"
UCI_GAS = "https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip"

REAL = {"birch2": 100, "iris": 3, "adult": 3, "mnist": 10, "letter": 26, "gas": 6}
G2 = {f"g2_{d}": 2 for d in (4, 16, 64, 128)}
SCALE = {f"scale_{k}_{d}": k for k in (4, 16, 64) for d in (4, 16, 64, 128)}
SKLEARN = {f"sklearn_{k}_{d}": k for d in (4, 16, 64, 128) for k in (4, 16, 64)}
K = {**REAL, **G2, **SCALE, **SKLEARN}
NAMES = list(K)


def load_txt(path) -> np.ndarray:
    """Their loader: whitespace-separated numbers, lines containing 'x' skipped."""
    rows = [[float(v) for v in ln.split()] for ln in open(path) if "x" not in ln]
    return np.array(rows)


def preprocess(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    return Xc / np.linalg.norm(Xc, axis=1).max()


def _fetch(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def _zenodo_tar(name: str, d: Path, members: list[str]):
    if all((d / m).exists() for m in members):
        return
    d.mkdir(parents=True, exist_ok=True)
    tar = _fetch(f"{ZENODO}{name}?download=1", d / name)
    subprocess.run(["tar", "-xJf", str(tar), "-C", str(d)] + members, check=True)


def raw(name: str, data_dir: str = "data"):
    """(X samples x features, y labels or None) before preprocessing."""
    d = Path(data_dir) / "pe_means"
    if name in ("birch2", "iris", "adult"):
        _zenodo_tar("real_datasets.tar.xz", d / "real", ["birch2.txt", "iris.txt", "adult.txt"])
        X = load_txt(d / "real" / f"{name}.txt")
        if name == "birch2":                                          # the archive holds their 25k subset
            C = load_txt(_fetch(SIPU + "b2-gt.txt", d / "real" / "b2-gt.txt"))
            y = _nearest(X, C)
        elif name == "iris":
            rows = [ln.strip() for ln in open(_fetch(SIPU + "iris.data.txt", d / "real" / "iris.data.txt")) if ln.strip()]
            s2i = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
            y = np.array([s2i[r.split(",")[-1]] for r in rows])
        else:
            y = None
        return X, y
    if name == "mnist":
        from .download import load_mnist_lenet
        return load_mnist_lenet(data_dir)
    if name == "letter":
        from sklearn.datasets import fetch_openml
        f = d / "real" / "letter.npz"
        if not f.exists():
            L = fetch_openml(name="letter", version=1, as_frame=True)
            letters = sorted(np.unique(L.target.values))
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, X=L.data.values.astype(float), y=np.array([letters.index(v) for v in L.target.values]))
        z = np.load(f)
        return z["X"], z["y"]
    if name == "gas":
        f = d / "real" / "gas_turbine_full.txt"
        if not f.exists():
            import pandas as pd
            try:                                                      # their path: ucimlrepo (12 columns, with year)
                from ucimlrepo import fetch_ucirepo
                X = fetch_ucirepo(id=551).data.original.values.astype(float)
            except Exception:                                         # fallback: the UCI zip (11 columns, no year)
                z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(UCI_GAS).read()))
                X = pd.concat([pd.read_csv(z.open(n)) for n in sorted(z.namelist()) if n.endswith(".csv")]).values
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(f, X.astype(float), fmt="%.4f")
        return load_txt(f), None
    if name in G2:
        dim = name.split("_")[1]
        _zenodo_tar("g2_datasets.tar.xz", d / "g2", [f"g2-{dd}-50.txt" for dd in (4, 16, 64, 128)])
        X = load_txt(d / "g2" / f"g2-{dim}-50.txt")
        y = None
        if dim in ("16", "128"):
            zf = _fetch(SIPU + "g2-gt-txt.zip", d / "g2" / "g2-gt-txt.zip")
            with zipfile.ZipFile(zf) as z:
                C = np.array([[float(v) for v in ln.split()] for ln in z.read(f"g2-{dim}-50-gt.txt").decode().splitlines() if ln.strip()])
            y = _nearest(preprocess(X), preprocess_like(C, X))
        return X, y
    if name in SCALE:
        _, k, dim = name.split("_")
        _zenodo_tar("scale_datasets.tar.xz", d / "scale", [f"SynthNew_{kk}_{dd}_1.txt" for kk in (4, 16, 64) for dd in (4, 16, 64, 128)])
        return load_txt(d / "scale" / f"SynthNew_{k}_{dim}_1.txt"), None
    if name in SKLEARN:
        from sklearn.datasets import make_blobs
        _, k, dim = name.split("_")
        X, _ = make_blobs(n_samples=20000, n_features=int(dim), centers=int(k), random_state=1)
        return np.round(X, 4), None                                    # they save with fmt='%.4f'
    raise KeyError(name)


def preprocess_like(C: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    return (C - X.mean(axis=0)) / np.linalg.norm(Xc, axis=1).max()


def _nearest(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return ((X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :]).argmin(1)


def load(name: str, data_dir: str = "data"):
    """Preprocessed (A d x N, k, y)."""
    X, y = raw(name, data_dir)
    return preprocess(X).T, K[name], y
