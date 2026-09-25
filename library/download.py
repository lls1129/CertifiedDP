"""Optional dataset loaders.  Each returns (X, y) with X of shape (samples x features); use
preprocess.from_array(X) for the library's column layout."""
from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

MNIST_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"
MNIST_FILES = {"train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
               "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")}


def _read_mnist(d: Path, split: str):
    img, lab = MNIST_FILES[split]
    for f in (img, lab):
        if not (d / f).exists():
            d.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MNIST_URL + f, d / f)
    with gzip.open(d / img, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(d / lab, "rb") as f:
        f.read(8)
        y = np.frombuffer(f.read(), dtype=np.uint8)
    return X.astype(np.float64) / 255.0, y.astype(np.int64)


def load_mnist(split: str = "train", n: int | None = None, seed: int = 0, data_dir: str = "data"):
    """MNIST pixels in [0, 1].  split: "train" (60k), "test" (10k) or "all" (70k, train then test)."""
    d = Path(data_dir)
    if split == "all":
        parts = [_read_mnist(d, s) for s in ("train", "test")]
        X, y = np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    else:
        X, y = _read_mnist(d, split)
    if n is not None:
        idx = np.random.default_rng(seed).choice(len(y), size=n, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def load_mnist_lenet(data_dir: str = "data", seed: int = 0):
    """PE-means' MNIST: 84-dim LeNet-5 features of all 70k images (arXiv 2606.00342).  Cached as .npy;
    built on first use with torch/torchvision (one epoch of Adam, as their notebook does)."""
    d = Path(data_dir)
    fx, fy = d / "mnist_lenet_full.npy", d / "mnist_lenet_labels.npy"
    if not fx.exists():
        _build_lenet(d, seed)
    return np.load(fx), np.load(fy)


def _build_lenet(d: Path, seed: int):
    import torch
    import torch.nn as nn
    from torch.utils.data import ConcatDataset, DataLoader
    from torchvision import datasets, transforms

    torch.manual_seed(seed)
    net = nn.Sequential(nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(6, 16, 5), nn.ReLU(),
                        nn.MaxPool2d(2), nn.Flatten(), nn.Linear(400, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU())
    head = nn.Linear(84, 10)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(str(d), train=True, download=True, transform=tf)
    test = datasets.MNIST(str(d), train=False, download=True, transform=tf)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=1e-3)
    for x, y in DataLoader(train, batch_size=64, shuffle=True):
        opt.zero_grad()
        nn.functional.cross_entropy(head(net(x)), y).backward()
        opt.step()
    with torch.no_grad():
        X = np.vstack([net(x).numpy() for x, _ in DataLoader(ConcatDataset([train, test]), batch_size=2000)])
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "mnist_lenet_full.npy", X)
    np.save(d / "mnist_lenet_labels.npy", np.concatenate([train.targets.numpy(), test.targets.numpy()]))
