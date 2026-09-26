"""MNIST: download it, and parse the IDX container by hand.

Self-contained on purpose. Phase 0 does not import the ``scratchgrad`` package,
including its data loaders -- the point of this directory is that you can read
every line that touches the pixels.

The IDX format is four bytes of magic (two zeros, a type code, a rank), then
``rank`` big-endian ``int32`` dimensions, then the raw payload. Three details
that bite:

* The dimension header is **big-endian** regardless of the host. ``>i4``, not
  ``i4``; on x86 the little-endian read of 60000 is 1,089,470,464 and the
  reshape raises somewhere far away from the cause.
* :func:`numpy.frombuffer` returns a **read-only view that pins the entire
  decompressed buffer alive**. Copy at the parser boundary or a 47 MB temporary
  survives for the length of training.
* Fashion-MNIST ships the *same four filenames*. Cache it in the same directory
  and you will train a digit classifier on trousers, at a plausible-looking 88%
  accuracy. Hence the per-dataset md5s below and a dataset-specific cache dir.

Normalization is ``x / 255`` only -- no mean centring. The browser demo has to
reproduce the preprocessing exactly, and "divide by 255" is a thing a reader can
verify in the JS at a glance; a per-channel mean shipped as a constant is a
thing that silently rots.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import struct
import urllib.request

import numpy as np

__all__ = [
    "FILES",
    "MIRRORS",
    "download",
    "load_mnist",
    "parse_idx",
    "synthetic_mnist",
]

#: ``filename -> md5 of the gzip``. MNIST-specific; see the module docstring.
FILES: dict[str, str] = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

#: yann.lecun.com has served 403 to scripted clients for years; try mirrors.
MIRRORS: tuple[str, ...] = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)

# The repository-level cache, not ``phase0/data``: the package next door caches
# MNIST here too, and pointing Phase 0 somewhere else would download the same
# 11 MB a second time and leave two copies to drift apart. Absolute, so the
# path does not depend on the directory the script was launched from.
_DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mnist"
)

# IDX type codes -> numpy dtypes. Only 0x08 is ever seen in MNIST, but reading
# the code and rejecting the rest beats assuming.
_IDX_DTYPES = {
    0x08: np.dtype(np.uint8),
    0x09: np.dtype(np.int8),
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}


def parse_idx(raw: bytes) -> np.ndarray:
    """Decode an IDX byte string into an array.

    Args:
        raw: The decompressed contents of an ``idx*-ubyte`` file.

    Returns:
        An owning array of the declared shape and dtype.

    Raises:
        ValueError: If the magic number, type code, or payload length is wrong.
    """
    if len(raw) < 4:
        raise ValueError(f"IDX file is {len(raw)} bytes, too short for a header")
    zero_a, zero_b, type_code, rank = raw[0], raw[1], raw[2], raw[3]
    if zero_a or zero_b:
        raise ValueError(f"bad IDX magic {raw[:4]!r}: first two bytes must be zero")
    if type_code not in _IDX_DTYPES:
        raise ValueError(f"unknown IDX type code 0x{type_code:02x}")
    header = 4 + 4 * rank
    if len(raw) < header:
        raise ValueError(f"IDX header claims rank {rank} but the file is {len(raw)} bytes")
    shape = struct.unpack(f">{rank}I", raw[4:header])
    dtype = _IDX_DTYPES[type_code]
    expect = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) - header != expect:
        raise ValueError(
            f"IDX payload is {len(raw) - header} bytes, header implies {expect} for shape {shape}"
        )
    # .copy(): frombuffer hands back a read-only view that keeps `raw` alive.
    return np.frombuffer(raw, dtype=dtype, offset=header).reshape(shape).copy()


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(root: str = _DEFAULT_ROOT, verbose: bool = True) -> str:
    """Fetch the four MNIST archives into ``root``, verifying md5.

    Files already present with the right md5 are left alone.

    Args:
        root: Cache directory. Must not be shared with Fashion-MNIST.
        verbose: Print progress.

    Returns:
        The cache directory.

    Raises:
        RuntimeError: If a file cannot be fetched from any mirror, or the
            downloaded bytes do not match the expected md5.
    """
    os.makedirs(root, exist_ok=True)
    for name, want in FILES.items():
        dest = os.path.join(root, name)
        if os.path.exists(dest) and _md5(dest) == want:
            continue
        errors = []
        for base in MIRRORS:
            url = base + name
            try:
                if verbose:
                    print(f"  fetching {url}")
                with urllib.request.urlopen(url, timeout=60) as resp:
                    blob = resp.read()
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                continue
            got = hashlib.md5(blob).hexdigest()
            if got != want:
                errors.append(f"{url}: md5 {got} != {want}")
                continue
            with open(dest, "wb") as fh:
                fh.write(blob)
            break
        else:
            raise RuntimeError(f"could not fetch {name}:\n    " + "\n    ".join(errors))
    return root


def _read(root: str, name: str) -> np.ndarray:
    with gzip.open(os.path.join(root, name), "rb") as fh:
        return parse_idx(fh.read())


def load_mnist(
    root: str = _DEFAULT_ROOT,
    fetch: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST as flat float64 images in ``[0, 1]`` and int64 labels.

    Args:
        root: Cache directory.
        fetch: Download anything missing. Set ``False`` to fail offline instead.
        verbose: Print progress.

    Returns:
        ``(x_train, y_train, x_test, y_test)`` with images ``(N, 784)`` scaled by
        ``1/255`` -- and nothing else. No mean centring: the browser has to
        reproduce this preprocessing byte for byte.
    """
    if fetch:
        download(root, verbose=verbose)
    x_tr = _read(root, "train-images-idx3-ubyte.gz")
    y_tr = _read(root, "train-labels-idx1-ubyte.gz")
    x_te = _read(root, "t10k-images-idx3-ubyte.gz")
    y_te = _read(root, "t10k-labels-idx1-ubyte.gz")
    if x_tr.shape != (60000, 28, 28) or x_te.shape != (10000, 28, 28):
        raise ValueError(f"unexpected MNIST shapes: {x_tr.shape}, {x_te.shape}")
    return (
        x_tr.reshape(-1, 784).astype(np.float64) / 255.0,
        y_tr.astype(np.int64),
        x_te.reshape(-1, 784).astype(np.float64) / 255.0,
        y_te.astype(np.int64),
    )


def synthetic_mnist(
    rng: np.random.Generator,
    n_train: int = 2000,
    n_test: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fabricate a tiny MNIST-shaped dataset so the pipeline runs offline.

    Ten fixed stroke-like prototypes plus per-sample jitter, quantized to
    ``uint8`` and divided by 255 exactly as the real loader does. It is not
    MNIST and makes no claim to be; it exists so that ``train.py --synthetic``,
    the smoke test and the export path are all exercisable with no network.

    Args:
        rng: Explicit generator.
        n_train: Number of training rows.
        n_test: Number of test rows.

    Returns:
        ``(x_train, y_train, x_test, y_test)`` in the same format as
        :func:`load_mnist`.
    """
    grid_y, grid_x = np.mgrid[0:28, 0:28]
    protos = np.zeros((10, 28, 28))
    for c in range(10):
        # Three Gaussian blobs per class: enough structure that a linear model
        # cannot quite separate them and the ReLU layer earns its keep.
        for _ in range(3):
            cy, cx = rng.uniform(6, 22, size=2)
            sigma = rng.uniform(2.0, 4.0)
            protos[c] += np.exp(-((grid_y - cy) ** 2 + (grid_x - cx) ** 2) / (2 * sigma**2))
        protos[c] /= protos[c].max()

    def make(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, 10, size=n)
        shift = rng.integers(-2, 3, size=(n, 2))
        imgs = np.empty((n, 28, 28))
        for i in range(n):
            img = np.roll(protos[y[i]], shift[i], axis=(0, 1))
            imgs[i] = np.clip(img * rng.uniform(0.7, 1.0) + rng.normal(0, 0.05, (28, 28)), 0, 1)
        pixels = (imgs * 255).astype(np.uint8)
        return pixels.reshape(n, 784).astype(np.float64) / 255.0, y.astype(np.int64)

    x_tr, y_tr = make(n_train)
    x_te, y_te = make(n_test)
    return x_tr, y_tr, x_te, y_te


def main() -> int:
    ap = argparse.ArgumentParser(description="download and summarise MNIST")
    ap.add_argument("--root", default=_DEFAULT_ROOT)
    args = ap.parse_args()
    x_tr, y_tr, x_te, y_te = load_mnist(args.root)
    print(f"train {x_tr.shape} labels {np.bincount(y_tr)}")
    print(f"test  {x_te.shape} labels {np.bincount(y_te)}")
    print(f"pixel range [{x_tr.min():.3f}, {x_tr.max():.3f}]  mean {x_tr.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
