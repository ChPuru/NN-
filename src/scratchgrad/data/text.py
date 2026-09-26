"""Character-level corpora and a next-token dataset for language modelling.

The vocabulary is ``sorted(set(text))`` -- deterministic across runs and
machines, which is what makes a checkpoint's embedding matrix mean the same
thing tomorrow. Anything derived from set iteration order is not stable and
would silently permute the embedding rows.

**Line endings are normalised.** The corpus is read as *bytes* and decoded
utf-8 explicitly, then ``\\r\\n`` and bare ``\\r`` are folded to ``\\n``. Fetch
tiny-shakespeare through a proxy or a git checkout with ``core.autocrlf=true``
and the file arrives with CRLF; ``'\\r'`` then joins the vocabulary as a 66th
character with ~1.1M occurrences. Nothing errors. Vocab size, per-character
loss, and every bits-per-character number in the run change, and comparisons to
any published figure become meaningless.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ._download import DownloadError, fetch_bytes, md5_of
from .dataset import Dataset

__all__ = [
    "normalize_text",
    "CharVocab",
    "CharDataset",
    "load_text",
    "tiny_shakespeare",
    "TINY_SHAKESPEARE_URLS",
]

#: raw.githubusercontent.com was observed resetting the connection from some
#: networks (and it is the single point of failure for half the tutorials on the
#: internet), so the jsDelivr mirror of the same repo is a first-class fallback,
#: not a comment. :func:`load_text` reports which one answered.
TINY_SHAKESPEARE_URLS = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt",
)


def normalize_text(raw: bytes) -> str:
    """Decode utf-8 strictly and fold CRLF/CR line endings to ``\\n``.

    Args:
        raw: The undecoded file contents.

    Returns:
        The decoded, newline-normalised text.

    Raises:
        UnicodeDecodeError: On invalid utf-8. Deliberately not silenced with
            ``errors="replace"``: a corpus quietly peppered with U+FFFD is a
            worse outcome than a loud failure at load time.
    """
    return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


class CharVocab:
    """A stable character vocabulary with ``encode``/``decode``.

    Args:
        text: The corpus the vocabulary is derived from.

    Attributes:
        chars: The sorted tuple of distinct characters. Index ``i`` in an
            encoded array means ``chars[i]``.
    """

    def __init__(self, text: str) -> None:
        self.chars: tuple[str, ...] = tuple(sorted(set(text)))
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self.itos: dict[int, str] = {i: c for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars)

    @property
    def size(self) -> int:
        """Number of distinct characters -- the model's output dimension."""
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        """Map a string to ``(len(text),)`` int64 token ids.

        Raises:
            KeyError: On a character absent from the vocabulary. There is no
                ``<unk>`` bucket: for character-level modelling an unknown
                character means the vocabulary was built from a different
                corpus, which should stop the run rather than be absorbed.
        """
        return np.fromiter((self.stoi[c] for c in text), dtype=np.int64, count=len(text))

    def decode(self, ids) -> str:
        """Map an iterable of token ids back to a string."""
        return "".join(self.itos[int(i)] for i in np.asarray(ids).reshape(-1))

    def __repr__(self) -> str:
        return f"CharVocab(size={self.size})"


class CharDataset(Dataset):
    """Next-character prediction pairs over a character-level corpus.

    Sample ``i`` is ``(tokens[i : i+block], tokens[i+1 : i+1+block])`` -- input
    and target are the *same* window offset by one, so the model is trained to
    predict every position in the block at once, not just the last.

    Args:
        text: The corpus. Normalise it with :func:`normalize_text` first if it
            came off the network.
        block_size: Context length ``T``. Each sample is ``(T,)`` / ``(T,)``.
        vocab: Reuse an existing vocabulary (pass the *training* vocab when
            building the validation split, or the two index spaces disagree).
            ``None`` builds one from ``text``.

    Raises:
        ValueError: If the corpus is not longer than ``block_size``; a shorter
            one yields zero samples and an empty DataLoader, which reads as a
            training bug much later.
    """

    def __init__(self, text: str, block_size: int, vocab: CharVocab | None = None) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.vocab = vocab if vocab is not None else CharVocab(text)
        self.block_size = int(block_size)
        self.tokens = self.vocab.encode(text)
        # +1 because the target window extends one past the input window.
        if len(self.tokens) < self.block_size + 1:
            raise ValueError(
                f"corpus has {len(self.tokens)} characters, need at least {self.block_size + 1} "
                f"for block_size={self.block_size}"
            )

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, y)`` for sample ``index``, supporting negative indexing.

        Raises:
            IndexError: If ``index`` is out of range. Without the check the
                slices simply run short: ``ds[-1]`` hands back ``x`` of shape
                ``(0,)`` against ``y`` of shape ``(block_size,)``, and a ragged
                sample only surfaces much later inside ``np.stack``.
        """
        n = len(self)
        i = int(index)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"index {index} is out of range for {n} samples")
        # Copy: a slice of `self.tokens` is a writable view, so a caller who
        # augments a sample in place would silently rewrite the corpus for every
        # later epoch. A read-only view would be cheaper still, but it would
        # break a caller whose in-place edit is legitimate; two blocks of int64
        # is a few kilobytes, and `default_collate` stacks (and so copies) them
        # again anyway.
        return (
            self.tokens[i : i + self.block_size].copy(),
            self.tokens[i + 1 : i + 1 + self.block_size].copy(),
        )

    # -- convenience passthroughs ------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        return self.vocab.encode(text)

    def decode(self, ids) -> str:
        return self.vocab.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.vocab.size

    def __repr__(self) -> str:
        return (
            f"CharDataset(n_tokens={len(self.tokens)}, block_size={self.block_size}, "
            f"vocab_size={self.vocab_size})"
        )


def load_text(
    urls: tuple[str, ...],
    path: str | Path,
    download: bool = True,
    verbose: bool = True,
    md5: str | None = None,
) -> str:
    """Fetch a text corpus (or read the cache) and return normalised text.

    The payload is written to a sibling ``.part`` file and renamed only once
    it is complete, and the optional ``md5`` is checked before that rename. A
    corpus has no header and no length field, so a transfer cut halfway
    through is still a perfectly readable, shorter text: without these two
    guarantees it lands in the cache and every later run returns it verbatim,
    quietly changing the vocabulary and every bits-per-character figure.

    The few lines are repeated from
    :func:`~scratchgrad.data._download.download_to` rather than delegating to
    it because ``verbose`` reports *which mirror answered*, and only
    :func:`~scratchgrad.data._download.fetch_bytes` knows that.

    Args:
        urls: Mirrors in preference order.
        path: Cache file.
        download: Fetch when the cache is absent or fails ``md5``.
        verbose: Print which mirror answered. Which source a corpus came from
            changes the numbers if the mirrors ever drift, so it is reported
            rather than hidden.
        md5: Expected checksum of the raw bytes, or ``None`` to skip the
            check. No digest is pinned for tiny-shakespeare: this package
            only records digests produced by hashing a file it actually
            downloaded, and an invented one would reject every mirror.

    Returns:
        utf-8 text with ``\\n`` line endings.

    Raises:
        FileNotFoundError: If the cache is absent and ``download`` is false.
        ValueError: If a cached file fails ``md5`` and ``download`` is false.
        DownloadError: If a freshly downloaded file fails ``md5``.
    """
    path = Path(path)
    if path.exists() and (md5 is None or md5_of(path) == md5):
        return normalize_text(path.read_bytes())
    if not download:
        if path.exists():
            raise ValueError(f"{path} has md5 {md5_of(path)}, expected {md5}, download=False")
        raise FileNotFoundError(f"{path} is missing and download=False")

    payload, source = fetch_bytes(urls)
    if md5 is not None:
        got = hashlib.md5(payload).hexdigest()
        if got != md5:
            raise DownloadError(
                f"checksum mismatch for {path.name}: expected {md5}, got {got}. "
                "The mirror served a different corpus, or the transfer was truncated."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(payload)
    # Rename only after a complete write: an interrupted run then leaves a
    # `.part` behind and never a short corpus under the real name.
    tmp.replace(path)
    if verbose:
        print(f"fetched {path.name} from {source}")
    return normalize_text(payload)


def tiny_shakespeare(
    root: str | Path = "data/text",
    download: bool = True,
    verbose: bool = True,
) -> str:
    """The 1.1 MB tiny-shakespeare corpus, normalised. 65 characters of vocab."""
    return load_text(
        TINY_SHAKESPEARE_URLS, Path(root) / "tinyshakespeare.txt", download, verbose
    )
