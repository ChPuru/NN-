"""Tests for the ``scratchgrad.data`` subsystem.

No network in the default path. The IDX parser is exercised against synthetic
buffers built here byte by byte -- the parser, not the download, is the unit
under test, and a synthetic buffer can be made *wrong* on purpose, which a real
one cannot.

Network-touching tests are marked ``network`` and skipped unless the marker is
explicitly selected (``-m network``).
"""

from __future__ import annotations

import gzip
import hashlib
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from scratchgrad.data import text as text_mod
from scratchgrad.data._download import DownloadError
from scratchgrad.data.cifar10 import unpack_rows
from scratchgrad.data.dataset import (
    DataLoader,
    Subset,
    TensorDataset,
    default_collate,
    random_split,
)
from scratchgrad.data.mnist import (
    IDX_IMAGE_MAGIC,
    IDX_LABEL_MAGIC,
    MNIST_MD5,
    load_mnist,
    parse_idx_images,
    parse_idx_labels,
    to_float,
)
from scratchgrad.data.text import CharDataset, CharVocab, normalize_text


def make_rng() -> np.random.Generator:
    """Local generator. Named ``make_rng`` so it cannot shadow the ``rng``
    fixture in ``conftest.py``."""
    return np.random.default_rng(20240517)


# --------------------------------------------------------------------------- #
# synthetic IDX buffers
# --------------------------------------------------------------------------- #
def make_idx_images(images: np.ndarray, magic: int = IDX_IMAGE_MAGIC, endian: str = ">") -> bytes:
    """Serialise ``(N, H, W)`` uint8 into an IDX3 buffer, byte for byte."""
    n, h, w = images.shape
    return struct.pack(f"{endian}IIII", magic, n, h, w) + images.astype(np.uint8).tobytes()


def make_idx_labels(labels: np.ndarray, magic: int = IDX_LABEL_MAGIC, endian: str = ">") -> bytes:
    return struct.pack(f"{endian}II", magic, len(labels)) + labels.astype(np.uint8).tobytes()


class TestIDXParser:
    def test_images_round_trip(self):
        images = make_rng().integers(0, 256, size=(7, 11, 5), dtype=np.uint8)
        out = parse_idx_images(make_idx_images(images))
        assert out.shape == (7, 11, 5)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, images)

    def test_labels_round_trip(self):
        labels = make_rng().integers(0, 10, size=13).astype(np.uint8)
        out = parse_idx_labels(make_idx_labels(labels))
        assert out.shape == (13,)
        np.testing.assert_array_equal(out, labels)

    def test_parser_output_is_writable_and_owns_its_data(self):
        """``np.frombuffer`` gives a read-only view pinning the whole buffer."""
        out = parse_idx_images(make_idx_images(np.zeros((3, 4, 5), np.uint8)))
        assert out.flags["WRITEABLE"]
        assert out.flags["OWNDATA"]
        out[0, 0, 0] = 42  # would raise ValueError on the raw frombuffer view

    def test_little_endian_header_is_rejected(self):
        """The whole point of the magic assertion: catch the byte order here."""
        buf = make_idx_images(np.zeros((3, 4, 5), np.uint8), endian="<")
        with pytest.raises(ValueError, match="big-endian"):
            parse_idx_images(buf)

    def test_label_magic_on_image_file_is_rejected(self):
        """2051 vs 2049 -- swapping the two files must not parse as garbage."""
        buf = make_idx_images(np.zeros((3, 4, 5), np.uint8))
        with pytest.raises(ValueError, match="magic"):
            parse_idx_labels(buf)

    def test_truncated_payload_is_rejected(self):
        buf = make_idx_images(make_rng().integers(0, 256, (4, 6, 6), dtype=np.uint8))
        with pytest.raises(ValueError, match="payload"):
            parse_idx_images(buf[:-10])

    def test_a_buffer_too_short_for_the_header_raises_ValueError(self):
        """Both docstrings promise ValueError; `buf[:16]` on a short file used
        to reach `struct.unpack` and raise `struct.error`, which is not one."""
        images = make_idx_images(np.zeros((2, 3, 4), np.uint8))
        for truncated in (b"", images[:4], images[:15]):
            with pytest.raises(ValueError, match="too short"):
                parse_idx_images(truncated)
        labels = make_idx_labels(np.zeros(3, np.uint8))
        for truncated in (b"", labels[:3], labels[:7]):
            with pytest.raises(ValueError, match="too short"):
                parse_idx_labels(truncated)

    def test_magic_constants(self):
        assert IDX_IMAGE_MAGIC == 0x00000803
        assert IDX_LABEL_MAGIC == 0x00000801

    def test_to_float_maps_background_to_zero(self):
        x = np.array([[0, 128, 255]], dtype=np.uint8)
        out = to_float(x)
        assert out.dtype == np.float32
        # Exactly 0.0, not -0.4242: the documented reason the default transform
        # is x/255 and not the mean-centred variant.
        assert out[0, 0] == 0.0
        assert out[0, 2] == pytest.approx(1.0)
        assert out[0, 1] == pytest.approx(128 / 255)


# --------------------------------------------------------------------------- #
# CIFAR-10 planar layout
# --------------------------------------------------------------------------- #
class TestCifarLayout:
    def test_planar_rows_unpack_channel_first(self):
        # A row that is 1024 of R, then 1024 of G, then 1024 of B. If the
        # reshape were (-1, 32, 32, 3), channel 0 would not be constant.
        row = np.concatenate(
            [np.full(1024, 10, np.uint8), np.full(1024, 20, np.uint8), np.full(1024, 30, np.uint8)]
        )[None, :]
        out = unpack_rows(row)
        assert out.shape == (1, 3, 32, 32)
        assert (out[0, 0] == 10).all() and (out[0, 1] == 20).all() and (out[0, 2] == 30).all()

    def test_pixel_order_within_a_plane_is_row_major(self):
        # int32 so the positional index is readable straight off the value;
        # uint8 would wrap at 256 and hide exactly the offsets being checked.
        row = np.arange(3072, dtype=np.int32)[None, :]
        out = unpack_rows(row)
        assert out[0, 0, 0, 0] == 0
        assert out[0, 0, 0, 1] == 1  # next column
        assert out[0, 0, 1, 0] == 32  # next row, same plane
        assert out[0, 1, 0, 0] == 1024  # next plane: one full 32x32 later

    def test_bad_row_length_raises(self):
        with pytest.raises(ValueError, match="3072"):
            unpack_rows(np.zeros((2, 3071), np.uint8))


# --------------------------------------------------------------------------- #
# Dataset / Subset / random_split
# --------------------------------------------------------------------------- #
class TestDatasets:
    def test_tensor_dataset_indexing(self):
        x = make_rng().normal(size=(7, 11))
        y = np.arange(7)
        ds = TensorDataset(x, y)
        assert len(ds) == 7
        xi, yi = ds[3]
        np.testing.assert_array_equal(xi, x[3])
        assert yi == 3

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="first axis"):
            TensorDataset(np.zeros((7, 3)), np.zeros(5))

    def test_subset_is_a_view_not_a_copy(self):
        x = np.zeros((7, 3))
        ds = TensorDataset(x, np.arange(7))
        sub = Subset(ds, [5, 1])
        assert len(sub) == 2
        assert sub[0][1] == 5 and sub[1][1] == 1

    def test_random_split_is_a_disjoint_partition(self):
        ds = TensorDataset(np.arange(23).reshape(23, 1), np.arange(23))
        parts = random_split(ds, [11, 7, 5], rng=make_rng())
        assert [len(p) for p in parts] == [11, 7, 5]
        idx = [set(p.indices.tolist()) for p in parts]
        assert set().union(*idx) == set(range(23))
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                assert not (idx[i] & idx[j])

    def test_random_split_fractions(self):
        ds = TensorDataset(np.zeros((100, 2)), np.zeros(100))
        parts = random_split(ds, [0.8, 0.2], rng=make_rng())
        assert [len(p) for p in parts] == [80, 20]

    def test_random_split_rejects_bad_lengths(self):
        ds = TensorDataset(np.zeros((10, 2)), np.zeros(10))
        with pytest.raises(ValueError, match="sum to"):
            random_split(ds, [3, 3], rng=make_rng())

    def test_random_split_rejects_negative_lengths(self):
        """A negative size passes the sum check, then runs the slice cursor
        backwards: the next split starts before the previous one ended, so the
        subsets overlap and train leaks into test."""
        ds = TensorDataset(np.zeros((10, 2)), np.zeros(10))
        with pytest.raises(ValueError, match="non-negative"):
            random_split(ds, [-3, 13], rng=make_rng())

    def test_random_split_is_deterministic_given_the_same_seed(self):
        ds = TensorDataset(np.zeros((23, 2)), np.zeros(23))
        a = random_split(ds, [13, 10], rng=np.random.default_rng(7))
        b = random_split(ds, [13, 10], rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a[0].indices, b[0].indices)

    def test_random_split_without_rng_is_contiguous(self):
        ds = TensorDataset(np.zeros((10, 2)), np.zeros(10))
        parts = random_split(ds, [4, 6])
        np.testing.assert_array_equal(parts[0].indices, np.arange(4))


# --------------------------------------------------------------------------- #
# DataLoader
# --------------------------------------------------------------------------- #
class TestDataLoader:
    def test_batches_cover_everything_exactly_once(self):
        ds = TensorDataset(np.arange(23).reshape(23, 1), np.arange(23))
        loader = DataLoader(ds, batch_size=5)
        batches = list(loader)
        assert len(batches) == len(loader) == 5
        assert [len(b[0]) for b in batches] == [5, 5, 5, 5, 3]
        seen = np.concatenate([b[1] for b in batches])
        np.testing.assert_array_equal(np.sort(seen), np.arange(23))

    def test_drop_last(self):
        ds = TensorDataset(np.arange(23).reshape(23, 1), np.arange(23))
        loader = DataLoader(ds, batch_size=5, drop_last=True)
        batches = list(loader)
        assert len(batches) == len(loader) == 4
        assert all(len(b[0]) == 5 for b in batches)

    def test_batch_shapes_keep_the_feature_axes(self):
        ds = TensorDataset(np.zeros((23, 3, 32, 32)), np.zeros(23, dtype=np.int64))
        x, y = next(iter(DataLoader(ds, batch_size=7)))
        assert x.shape == (7, 3, 32, 32) and y.shape == (7,)

    def test_shuffle_requires_an_explicit_rng(self):
        ds = TensorDataset(np.zeros((5, 2)), np.zeros(5))
        with pytest.raises(ValueError, match="rng"):
            DataLoader(ds, batch_size=2, shuffle=True)

    def test_shuffle_reorders(self):
        ds = TensorDataset(np.arange(50).reshape(50, 1), np.arange(50))
        loader = DataLoader(ds, batch_size=50, shuffle=True, rng=make_rng())
        _, y = next(iter(loader))
        assert not np.array_equal(y, np.arange(50))
        np.testing.assert_array_equal(np.sort(y), np.arange(50))

    def test_shuffle_keeps_X_AND_Y_ALIGNED(self):
        """The failure this whole design exists to prevent.

        ``y = f(X)`` with an injective ``f``, so if the loader ever drew two
        independent permutations the relation would break for all but a
        vanishing fraction of samples. (Two separate shuffles do not crash: the
        loss then sits at exactly ln(C) forever and overfit-one-batch still
        appears to succeed.)
        """
        x = make_rng().normal(size=(97, 5))
        y = (x.sum(axis=1) * 1000.0).round()  # f(X), injective in practice
        loader = DataLoader(TensorDataset(x, y), batch_size=7, shuffle=True, rng=make_rng())
        n = 0
        for xb, yb in loader:
            np.testing.assert_allclose(yb, (xb.sum(axis=1) * 1000.0).round())
            n += len(xb)
        assert n == 97

    def test_alignment_survives_several_epochs(self):
        x = np.arange(31).reshape(31, 1).astype(float)
        loader = DataLoader(
            TensorDataset(x, x[:, 0] * 3.0), batch_size=4, shuffle=True, rng=make_rng()
        )
        for _ in range(3):
            for xb, yb in loader:
                np.testing.assert_allclose(yb, xb[:, 0] * 3.0)

    def test_two_epochs_use_different_orders(self):
        ds = TensorDataset(np.arange(64).reshape(64, 1), np.arange(64))
        loader = DataLoader(ds, batch_size=64, shuffle=True, rng=make_rng())
        first = next(iter(loader))[1]
        second = next(iter(loader))[1]
        assert not np.array_equal(first, second)

    def test_custom_collate_fn(self):
        ds = TensorDataset(np.arange(10).reshape(10, 1), np.arange(10))
        loader = DataLoader(ds, batch_size=5, collate_fn=lambda samples: len(samples))
        assert list(loader) == [5, 5]

    def test_default_collate_on_bare_samples(self):
        out = default_collate([np.arange(3), np.arange(3, 6)])
        assert out.shape == (2, 3)

    def test_bad_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            DataLoader(TensorDataset(np.zeros((4, 1))), batch_size=0)


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
class TestText:
    def test_crlf_is_normalised_out_of_the_vocabulary(self):
        text = normalize_text(b"To be,\r\nor not\rto be.\n")
        assert "\r" not in text
        assert text == "To be,\nor not\nto be.\n"
        assert "\r" not in CharVocab(text).chars

    def test_vocab_is_the_sorted_distinct_characters(self):
        vocab = CharVocab("banana")
        assert vocab.chars == ("a", "b", "n")
        assert vocab.size == 3
        # Stable across constructions: an embedding checkpoint depends on it.
        assert CharVocab("nabana").chars == vocab.chars

    def test_encode_decode_round_trip(self):
        text = "The quick brown fox\njumps over 13 lazy dogs!"
        vocab = CharVocab(text)
        ids = vocab.encode(text)
        assert ids.dtype == np.int64 and ids.shape == (len(text),)
        assert vocab.decode(ids) == text

    def test_unknown_character_raises(self):
        with pytest.raises(KeyError):
            CharVocab("abc").encode("abz")

    def test_char_dataset_pairs_are_offset_by_one(self):
        ds = CharDataset("abcdefgh", block_size=3)
        x, y = ds[0]
        assert ds.decode(x) == "abc"
        assert ds.decode(y) == "bcd"
        x, y = ds[4]
        assert ds.decode(x) == "efg"
        assert ds.decode(y) == "fgh"

    def test_char_dataset_length_and_shapes(self):
        ds = CharDataset("abcdefgh", block_size=3)
        assert len(ds) == 5  # 8 tokens - block_size
        for i in range(len(ds)):
            x, y = ds[i]
            assert x.shape == (3,) and y.shape == (3,)

    def test_char_dataset_negative_index_counts_from_the_end(self):
        ds = CharDataset("abcdefgh", block_size=3)
        # Before the bounds check, ds[-1] sliced tokens[-1:2] -> x of shape (0,)
        # paired with y of shape (3,): a ragged sample, silently.
        for i in range(1, len(ds) + 1):
            x, y = ds[-i]
            xp, yp = ds[len(ds) - i]
            np.testing.assert_array_equal(x, xp)
            np.testing.assert_array_equal(y, yp)

    def test_char_dataset_out_of_range_index_raises(self):
        ds = CharDataset("abcdefgh", block_size=3)
        for bad in (len(ds), len(ds) + 1, -len(ds) - 1, 10_000):
            with pytest.raises(IndexError):
                ds[bad]

    def test_char_dataset_samples_do_not_alias_the_corpus(self):
        """A returned sample used to be a writable view into ``ds.tokens``, so
        one in-place edit poisoned that window for every later epoch."""
        ds = CharDataset("abcdefgh", block_size=3)
        before = ds.tokens.copy()
        x, y = ds[2]
        x[:] = 0
        y[:] = 0
        np.testing.assert_array_equal(ds.tokens, before)
        np.testing.assert_array_equal(ds[2][0], before[2:5])
        np.testing.assert_array_equal(ds[2][1], before[3:6])

    def test_char_dataset_round_trips_through_a_loader(self):
        ds = CharDataset("the rain in spain stays mainly in the plain\n" * 4, block_size=8)
        x, y = next(iter(DataLoader(ds, batch_size=5, shuffle=True, rng=make_rng())))
        assert x.shape == (5, 8) and y.shape == (5, 8)
        # Targets are the inputs shifted one character; the overlap must match.
        np.testing.assert_array_equal(x[:, 1:], y[:, :-1])

    def test_shared_vocab_between_splits(self):
        text = "abcdefghij" * 5
        train = CharDataset(text[:30], block_size=4)
        val = CharDataset(text[30:], block_size=4, vocab=train.vocab)
        assert val.vocab is train.vocab
        assert val.vocab_size == train.vocab_size

    def test_corpus_shorter_than_block_raises(self):
        with pytest.raises(ValueError, match="block_size"):
            CharDataset("abc", block_size=8)


# --------------------------------------------------------------------------- #
# load_text caching: checksum + atomic write
# --------------------------------------------------------------------------- #
class TestLoadText:
    """No network: ``fetch_bytes`` is replaced inside the ``text`` module."""

    def test_download_writes_the_cache_and_leaves_no_part_file(self, tmp_path, monkeypatch):
        payload = b"To be,\r\nor not\r\n"
        digest = hashlib.md5(payload).hexdigest()
        calls: list[tuple] = []

        def fake_fetch(urls, **kwargs):
            calls.append(tuple(urls))
            return payload, "test://mirror"

        monkeypatch.setattr(text_mod, "fetch_bytes", fake_fetch)
        dest = tmp_path / "corpus.txt"
        expected = "To be,\nor not\n"
        assert text_mod.load_text(("u",), dest, verbose=False, md5=digest) == expected
        assert not (tmp_path / "corpus.txt.part").exists()
        # A checksum-valid cache is reused rather than re-fetched.
        assert text_mod.load_text(("u",), dest, verbose=False, md5=digest) == expected
        assert len(calls) == 1

    def test_a_corrupt_download_raises_and_caches_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            text_mod, "fetch_bytes", lambda urls, **kw: (b"half a corp", "test://mirror")
        )
        dest = tmp_path / "corpus.txt"
        with pytest.raises(DownloadError, match="checksum"):
            text_mod.load_text(
                ("u",), dest, verbose=False, md5=hashlib.md5(b"the whole corpus").hexdigest()
            )
        # A corpus has no length field, so a short one reads as a valid corpus
        # forever after. It must not reach the cache at all.
        assert not dest.exists()

    def test_an_interrupted_write_leaves_no_corpus_behind(self, tmp_path, monkeypatch):
        """The write goes to ``.part`` and is renamed last, so a failure part
        way through cannot leave a truncated file under the real name."""
        monkeypatch.setattr(
            text_mod, "fetch_bytes", lambda urls, **kw: (b"first citizen\n", "test://mirror")
        )

        def interrupted(self, target):
            raise OSError("simulated interruption during rename")

        monkeypatch.setattr(Path, "replace", interrupted)
        dest = tmp_path / "corpus.txt"
        with pytest.raises(OSError, match="simulated interruption"):
            text_mod.load_text(("u",), dest, verbose=False)
        assert not dest.exists()

    def test_a_corrupt_cache_is_refused_when_download_is_off(self, tmp_path):
        dest = tmp_path / "corpus.txt"
        dest.write_bytes(b"truncated")
        with pytest.raises(ValueError, match="md5"):
            text_mod.load_text(
                ("u",), dest, download=False, md5=hashlib.md5(b"the whole corpus").hexdigest()
            )

    def test_a_missing_cache_with_download_off_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            text_mod.load_text(("u",), tmp_path / "nope.txt", download=False)


# --------------------------------------------------------------------------- #
# MNIST cache verification on the download=False path
# --------------------------------------------------------------------------- #
class TestMnistCacheChecksum:
    """MNIST and Fashion-MNIST ship the same four filenames, so the md5 is the
    only thing standing between a crossed cache directory and a digit
    classifier trained on trousers. ``download=False`` must not skip it."""

    @staticmethod
    def _populate(root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        images = np.zeros((2, 28, 28), np.uint8)
        labels = np.zeros(2, np.uint8)
        for name, buf in (
            ("train-images-idx3-ubyte.gz", make_idx_images(images)),
            ("train-labels-idx1-ubyte.gz", make_idx_labels(labels)),
            ("t10k-images-idx3-ubyte.gz", make_idx_images(images)),
            ("t10k-labels-idx1-ubyte.gz", make_idx_labels(labels)),
        ):
            # Well-formed IDX inside well-formed gzip: without the checksum this
            # cache parses cleanly and the wrong dataset trains to a plausible
            # accuracy, which is exactly the failure the md5 exists to stop.
            (root / name).write_bytes(gzip.compress(buf))

    def test_cached_files_with_the_wrong_md5_are_rejected(self, tmp_path):
        root = tmp_path / "mnist"
        self._populate(root)
        with pytest.raises(ValueError, match="md5"):
            load_mnist(root=root, download=False)

    def test_the_error_names_the_expected_digest(self, tmp_path):
        root = tmp_path / "mnist"
        self._populate(root)
        with pytest.raises(ValueError, match=MNIST_MD5["train_images"]):
            load_mnist(root=root, download=False)

    def test_a_missing_cache_with_download_off_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mnist(root=tmp_path / "empty", download=False)


# --------------------------------------------------------------------------- #
# network -- skipped by default, run with: pytest tests/test_data.py -m network
# --------------------------------------------------------------------------- #
#: The ``network`` marker is registered in ``pyproject.toml`` but not deselected
#: by ``addopts``, so a marker alone would still run these on a plain
#: ``pytest tests``. Gate them on an opt-in environment variable as well: the
#: default test path must never touch the network.
_NETWORK_OFF = pytest.mark.skipif(
    not os.environ.get("SCRATCHGRAD_NETWORK_TESTS"),
    reason="set SCRATCHGRAD_NETWORK_TESTS=1 to run downloads",
)


@pytest.mark.network
@_NETWORK_OFF
class TestNetwork:
    def test_mnist_downloads_and_parses(self, tmp_path):
        from scratchgrad.data.mnist import load_mnist

        raw = load_mnist(root=tmp_path / "mnist")
        assert raw["train_images"].shape == (60000, 28, 28)
        assert raw["train_labels"].shape == (60000,)
        assert raw["test_images"].shape == (10000, 28, 28)
        assert set(np.unique(raw["train_labels"]).tolist()) == set(range(10))

    def test_tiny_shakespeare_downloads(self, tmp_path):
        from scratchgrad.data.text import tiny_shakespeare

        text = tiny_shakespeare(root=tmp_path)
        assert text.startswith("First Citizen:")
        assert "\r" not in text
        assert CharVocab(text).size == 65
