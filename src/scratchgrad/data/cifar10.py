"""CIFAR-10: download the official python tarball and parse its pickled batches.

Two things to know before reading the code.

**The rows are planar.** Each row of ``data`` is 3072 bytes laid out as
``R(1024) G(1024) B(1024)``, i.e. channel-major. The correct reshape is
``(-1, 3, 32, 32)``, which is already the frozen ``(N, C, H, W)`` layout --
no transpose needed. ``(-1, 32, 32, 3)`` also "works": it produces an array of
exactly the right size, full of images that look like three horizontally
squashed colour bands. A convnet trains on it to something like 45% instead of
75%, and every instinct says *architecture* rather than *loader*. Hence
:func:`unpack_rows`, tested directly.

**This format is pickle.** ``pickle.load`` on a file you did not create is
arbitrary code execution -- unpickling is not parsing, it is calling
constructors named by the stream. It is used here anyway, knowingly, because
pickle *is* the official distribution format for CIFAR-10, the tarball is
fetched over TLS from the canonical cs.toronto.edu URL, and it is md5-verified
against a digest recorded from a verified download before anything is
unpickled. Do not point ``root`` at an untrusted directory: a hostile
``data_batch_1`` in a pre-populated cache is a shell, not a bad dataset.
"""

from __future__ import annotations

import pickle
import tarfile
from pathlib import Path

import numpy as np

from ._download import download_to, md5_of
from .dataset import TensorDataset

__all__ = [
    "unpack_rows",
    "load_cifar10",
    "cifar10",
    "CIFAR10_CLASSES",
    "CIFAR10_URL",
    "CIFAR10_MD5",
]

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

#: Recorded by hashing the tarball this module downloaded from the URL above,
#: not copied from memory. Verify with ``md5sum cifar-10-python.tar.gz``.
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

_ARCHIVE = "cifar-10-python.tar.gz"
_MEMBER_DIR = "cifar-10-batches-py"
_TRAIN_BATCHES = tuple(f"data_batch_{i}" for i in range(1, 6))
_TEST_BATCH = "test_batch"


def unpack_rows(rows: np.ndarray) -> np.ndarray:
    """Turn ``(N, 3072)`` planar uint8 rows into ``(N, 3, 32, 32)``.

    Args:
        rows: Raw ``data`` array from a CIFAR-10 batch, ``(N, 3072)`` uint8.

    Returns:
        ``(N, 3, 32, 32)`` uint8, channel-first, matching the frozen image
        convention.

    Raises:
        ValueError: If the row length is not 3072.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 3 * 32 * 32:
        raise ValueError(f"expected (N, 3072) planar rows, got {rows.shape}")
    return rows.reshape(-1, 3, 32, 32)


def _load_batch(fh) -> tuple[np.ndarray, np.ndarray]:
    # encoding="bytes": the batches were pickled under Python 2, so the dict
    # keys are bytes. encoding="latin1" also decodes, but then the keys are str
    # and every reference below would have to change -- pick one and say so.
    obj = pickle.load(fh, encoding="bytes")
    images = unpack_rows(np.asarray(obj[b"data"], dtype=np.uint8))
    labels = np.asarray(obj[b"labels"], dtype=np.int64)
    return images, labels


def load_cifar10(
    root: str | Path = "data/cifar10", download: bool = True
) -> dict[str, np.ndarray]:
    """Download (if needed) and parse CIFAR-10. Returns **raw uint8** arrays.

    Args:
        root: Cache directory for the tarball and its extracted batches.
        download: Fetch the archive when it is missing or fails its md5.

    Returns:
        ``{"train_images": (50000, 3, 32, 32) uint8,
        "train_labels": (50000,) int64,
        "test_images": (10000, 3, 32, 32) uint8,
        "test_labels": (10000,) int64}``.
    """
    root = Path(root)
    archive = root / _ARCHIVE
    if download:
        download_to(archive, [CIFAR10_URL], md5=CIFAR10_MD5, timeout=600.0)
    elif not archive.exists():
        raise FileNotFoundError(f"{archive} is missing and download=False")
    else:
        # The module docstring's containment argument for unpickling this archive
        # is "it is md5-verified before anything is unpickled". That has to hold
        # on *every* path that reaches the unpickle, not only the downloading
        # one, or the stated defence is switched off precisely when a user hands
        # the loader a file of their own.
        actual = md5_of(archive)
        if actual != CIFAR10_MD5:
            raise ValueError(
                f"{archive} has md5 {actual}, expected {CIFAR10_MD5}. Refusing to unpickle an "
                "archive that is not the official CIFAR-10 distribution."
            )

    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    test_x = test_y = None
    with tarfile.open(archive, "r:gz") as tar:
        for name in (*_TRAIN_BATCHES, _TEST_BATCH):
            member = tar.extractfile(f"{_MEMBER_DIR}/{name}")
            if member is None:
                raise ValueError(f"{archive} has no member {_MEMBER_DIR}/{name}")
            images, labels = _load_batch(member)
            if name == _TEST_BATCH:
                test_x, test_y = images, labels
            else:
                train_x.append(images)
                train_y.append(labels)

    return {
        "train_images": np.concatenate(train_x),
        "train_labels": np.concatenate(train_y),
        "test_images": test_x,
        "test_labels": test_y,
    }


def cifar10(
    root: str | Path = "data/cifar10",
    download: bool = True,
    normalize: bool = True,
    dtype=np.float32,
) -> tuple[TensorDataset, TensorDataset]:
    """CIFAR-10 as ``(train, test)`` datasets of ``(3, 32, 32)`` images.

    Args:
        root: Cache directory.
        download: Fetch the archive when missing.
        normalize: Scale to ``[0, 1]`` with ``x / 255``. As with MNIST, no
            per-channel mean subtraction by default -- it is a train/inference
            mismatch waiting to happen and the caller can always do it.
        dtype: Float dtype when ``normalize`` is true.
    """
    raw = load_cifar10(root, download)

    def prep(x: np.ndarray) -> np.ndarray:
        if not normalize:
            return x
        return (np.asarray(x, dtype=dtype) / dtype(255.0)).astype(dtype, copy=False)

    return (
        TensorDataset(prep(raw["train_images"]), raw["train_labels"]),
        TensorDataset(prep(raw["test_images"]), raw["test_labels"]),
    )
