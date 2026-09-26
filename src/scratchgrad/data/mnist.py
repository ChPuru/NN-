"""MNIST and Fashion-MNIST: download, IDX parsing, caching.

No dataset library is involved. The IDX container is forty lines of ``struct``
and this file writes those forty lines, because the format is where the
interesting mistakes live.

Two of those mistakes are worth stating up front.

**IDX is big-endian.** Header fields are ``>I``. A little-endian misread does
not raise: ``0x00000803`` becomes ``0x03080000``, the item count becomes
roughly 16.7 million, and the failure surfaces as a ``reshape`` error several
lines below the actual bug. Hence the explicit magic assertion (2051 for
images, 2049 for labels) immediately after the unpack.

**Fashion-MNIST ships the exact same four filenames as MNIST.** Cache them in
one directory and you get a digit classifier trained on trousers, converging to
a completely plausible ~88%, with nothing anywhere to indicate it. The defence
is structural: a separate cache directory *and* a separate md5 table per
dataset, so a crossed download fails the checksum instead of training.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

from ._download import download_to, md5_of
from .dataset import TensorDataset

__all__ = [
    "IDX_IMAGE_MAGIC",
    "IDX_LABEL_MAGIC",
    "parse_idx_images",
    "parse_idx_labels",
    "load_mnist",
    "load_fashion_mnist",
    "mnist",
    "fashion_mnist",
    "to_float",
    "MNIST_MD5",
    "FASHION_MNIST_MD5",
]

#: ``0x00000803`` -- 3-D unsigned-byte tensor. Images.
IDX_IMAGE_MAGIC = 2051
#: ``0x00000801`` -- 1-D unsigned-byte tensor. Labels.
IDX_LABEL_MAGIC = 2049

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Mirrors in preference order. The first is the one that was actually exercised
# while writing this file; lecun.com is last because it has intermittently
# served 403 to non-browser agents since 2024.
_MNIST_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)

_FASHION_MIRRORS = (
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "https://cdn.jsdelivr.net/gh/zalandoresearch/fashion-mnist@master/data/fashion/",
)

# These eight digests were RECORDED BY HASHING THE FILES THIS MODULE ACTUALLY
# DOWNLOADED from the first mirror of each set -- they are not transcribed from
# memory or from a blog post. Regenerate with:
#     python -c "import hashlib,sys; print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())"
MNIST_MD5 = {
    "train_images": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train_labels": "d53e105ee54ea40749a09fcbcd1e9432",
    "test_images": "9fb629c4189551a2d022fa330f9573f3",
    "test_labels": "ec29112dd5afa0611ce80d1b7f02629c",
}

FASHION_MNIST_MD5 = {
    "train_images": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train_labels": "25c81989df183df01b3e8a0aad5dffbe",
    "test_images": "bef4ecab320f06d8554ea6380940ec79",
    "test_labels": "bb300cfdad3c16e7a12a480ee83cd310",
}

MNIST_CLASSES = tuple(str(i) for i in range(10))
FASHION_CLASSES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


# --------------------------------------------------------------------------- #
# IDX parsing -- the actual unit under test
# --------------------------------------------------------------------------- #
def parse_idx_images(buf: bytes) -> np.ndarray:
    """Parse a decompressed IDX3 image buffer into ``(N, H, W)`` uint8.

    Args:
        buf: The full decompressed file contents.

    Returns:
        A **freshly owned** ``uint8`` array of shape ``(N, H, W)``.

    Raises:
        ValueError: If the buffer is too short to hold a header, the magic
            number is not :data:`IDX_IMAGE_MAGIC`, or the payload length
            disagrees with the header.
    """
    # Check the length before unpacking: `buf[:16]` on a truncated file is a
    # short slice, and struct then raises `struct.error`, which is not a
    # ValueError and so escapes every caller that follows this docstring.
    if len(buf) < 16:
        raise ValueError(f"IDX image buffer is {len(buf)} bytes, too short for 16-byte header")
    magic, count, rows, cols = struct.unpack(">IIII", buf[:16])
    if magic != IDX_IMAGE_MAGIC:
        raise ValueError(
            f"bad IDX image magic {magic} (expected {IDX_IMAGE_MAGIC}). IDX headers are "
            "big-endian: unpack with '>IIII'. Read little-endian, 2051 reads as 50855936."
        )
    expected = count * rows * cols
    body = buf[16:]
    if len(body) != expected:
        raise ValueError(
            f"IDX image payload is {len(body)} bytes, header implies "
            f"{count}x{rows}x{cols} = {expected}"
        )
    # frombuffer hands back a read-only view that pins the entire decompressed
    # buffer alive for as long as any slice of it survives. Copy at the parser
    # boundary: the caller gets a writable array, and the 47 MB gzip scratch
    # buffer becomes collectable immediately.
    return np.frombuffer(body, dtype=np.uint8).reshape(count, rows, cols).copy()


def parse_idx_labels(buf: bytes) -> np.ndarray:
    """Parse a decompressed IDX1 label buffer into ``(N,)`` uint8.

    Raises:
        ValueError: If the buffer is too short to hold a header, the magic
            number is not :data:`IDX_LABEL_MAGIC`, or the payload length
            disagrees with the header.
    """
    # Same reason as in `parse_idx_images`: a short buffer must not surface
    # as `struct.error` from inside the unpack.
    if len(buf) < 8:
        raise ValueError(f"IDX label buffer is {len(buf)} bytes, too short for 8-byte header")
    magic, count = struct.unpack(">II", buf[:8])
    if magic != IDX_LABEL_MAGIC:
        raise ValueError(
            f"bad IDX label magic {magic} (expected {IDX_LABEL_MAGIC}). IDX headers are "
            "big-endian: unpack with '>II'."
        )
    body = buf[8:]
    if len(body) != count:
        raise ValueError(f"IDX label payload is {len(body)} bytes, header implies {count}")
    return np.frombuffer(body, dtype=np.uint8).copy()


def _read_gz(path: Path) -> bytes:
    with gzip.open(path, "rb") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# download + assemble
# --------------------------------------------------------------------------- #
def _load(
    root: Path,
    mirrors: tuple[str, ...],
    md5s: dict[str, str],
    download: bool,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, filename in _FILES.items():
        path = root / filename
        if download:
            download_to(path, [base + filename for base in mirrors], md5=md5s[key])
        elif not path.exists():
            raise FileNotFoundError(f"{path} is missing and download=False")
        else:
            # Verify the cache too. MNIST and Fashion-MNIST ship the *same four
            # filenames*, so a cache directory populated by the wrong loader
            # gives a digit classifier trained on trousers at a plausible 88%.
            # The md5 is the only structural defence against that, and skipping
            # it on the download=False path would leave the documented guard
            # switched off in exactly the case a user reaches for it.
            actual = md5_of(path)
            if actual != md5s[key]:
                raise ValueError(
                    f"{path} has md5 {actual}, expected {md5s[key]}. This cache file is not the "
                    f"{filename} this loader expects -- Fashion-MNIST uses identical filenames, "
                    "so check which dataset populated this directory."
                )
        buf = _read_gz(path)
        out[key] = parse_idx_images(buf) if key.endswith("images") else parse_idx_labels(buf)

    if len(out["train_images"]) != len(out["train_labels"]):
        raise ValueError("train images and labels disagree on length")
    if len(out["test_images"]) != len(out["test_labels"]):
        raise ValueError("test images and labels disagree on length")
    return out


def to_float(images: np.ndarray, dtype=np.float32) -> np.ndarray:
    """The default transform: ``x / 255.0``, mapping ``[0, 255] -> [0, 1]``.

    Deliberately **not** the usual ``(x/255 - 0.1307) / 0.3081``. Mean-centring
    is standard in PyTorch examples and it is not wrong, but it is a poor
    default here for three reasons:

    * It moves the MNIST background from ``0.0`` to ``-0.4242``. Any demo or
      inference path that forgets the exact constants feeds the net a
      distribution it has never seen, and the mismatch is silent -- accuracy
      degrades rather than erroring.
    * A background of exactly zero lets a hand-written forward pass (the JS
      demo, for instance) skip zero inputs entirely. That fast path is worth
      roughly an order of magnitude on a dense first layer, and ``-0.4242``
      destroys it.
    * On a one-hidden-layer net trained with Adam, the accuracy difference
      between the two is inside run-to-run noise. It buys nothing.

    Args:
        images: uint8 array, any shape.
        dtype: Output floating dtype.

    Returns:
        ``images / 255`` as ``dtype``.
    """
    return (np.asarray(images, dtype=dtype) / dtype(255.0)).astype(dtype, copy=False)


def load_mnist(
    root: str | Path = "data/mnist", download: bool = True
) -> dict[str, np.ndarray]:
    """Download (if needed) and parse MNIST. Returns **raw uint8** arrays.

    Args:
        root: Cache directory. Must not be shared with Fashion-MNIST: the four
            filenames are identical.
        download: Fetch missing or checksum-invalid files.

    Returns:
        ``{"train_images": (60000, 28, 28) uint8, "train_labels": (60000,) uint8,
        "test_images": (10000, 28, 28) uint8, "test_labels": (10000,) uint8}``.
        Scale with :func:`to_float` at the point of use.
    """
    return _load(Path(root), _MNIST_MIRRORS, MNIST_MD5, download)


def load_fashion_mnist(
    root: str | Path = "data/fashion_mnist", download: bool = True
) -> dict[str, np.ndarray]:
    """Download (if needed) and parse Fashion-MNIST. Returns raw uint8 arrays.

    Note the different default ``root`` from :func:`load_mnist`, and the
    different md5 table. Both are load-bearing -- see the module docstring.
    """
    return _load(Path(root), _FASHION_MIRRORS, FASHION_MNIST_MD5, download)


def _as_datasets(
    raw: dict[str, np.ndarray], flatten: bool, normalize: bool, dtype
) -> tuple[TensorDataset, TensorDataset]:
    def prep(images: np.ndarray) -> np.ndarray:
        x = to_float(images, dtype) if normalize else images
        if flatten:
            return x.reshape(len(x), -1)
        # (N, 1, 28, 28): a single channel, batch axis first, per the frozen
        # (N, C, H, W) image convention.
        return x.reshape(len(x), 1, *x.shape[1:])

    train = TensorDataset(prep(raw["train_images"]), raw["train_labels"].astype(np.int64))
    test = TensorDataset(prep(raw["test_images"]), raw["test_labels"].astype(np.int64))
    return train, test


def mnist(
    root: str | Path = "data/mnist",
    download: bool = True,
    flatten: bool = True,
    normalize: bool = True,
    dtype=np.float32,
) -> tuple[TensorDataset, TensorDataset]:
    """MNIST as ``(train, test)`` :class:`~scratchgrad.data.dataset.TensorDataset`.

    Args:
        root: Cache directory.
        download: Fetch missing files.
        flatten: ``(N, 784)`` if true, else ``(N, 1, 28, 28)``.
        normalize: Apply :func:`to_float` (``x/255``). If false you get uint8.
        dtype: Float dtype when ``normalize`` is true.
    """
    return _as_datasets(load_mnist(root, download), flatten, normalize, dtype)


def fashion_mnist(
    root: str | Path = "data/fashion_mnist",
    download: bool = True,
    flatten: bool = True,
    normalize: bool = True,
    dtype=np.float32,
) -> tuple[TensorDataset, TensorDataset]:
    """Fashion-MNIST as ``(train, test)`` datasets. Class names in
    :data:`FASHION_CLASSES`. Cached in its own directory; see the module
    docstring for why that is not a stylistic preference."""
    return _as_datasets(load_fashion_mnist(root, download), flatten, normalize, dtype)
