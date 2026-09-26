"""Datasets, subsets, and a single-process ``DataLoader``.

Everything here is plain NumPy and plain Python. There is **no worker-process
pool**: batching a NumPy array is a slice, and a fork/spawn pool would cost more
in pickling than it saves. Single-process is a deliberate choice, not a stub.

The one thing in this file that is worth reading twice is the shuffle. See
:class:`DataLoader`.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence

import numpy as np

__all__ = [
    "Dataset",
    "TensorDataset",
    "Subset",
    "random_split",
    "default_collate",
    "DataLoader",
]


class Dataset:
    """Abstract map-style dataset: integer index in, one sample out.

    Subclasses implement :meth:`__len__` and :meth:`__getitem__`. A sample is
    whatever the subclass wants -- typically an ``(x, y)`` tuple of arrays --
    and :func:`default_collate` stacks a list of them into batch arrays.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]


class TensorDataset(Dataset):
    """Wraps aligned arrays; ``ds[i]`` is the tuple of their ``i``-th rows.

    Args:
        *arrays: Arrays sharing a first axis (the batch axis). Anything
            array-like is accepted and converted with ``np.asarray``.

    Raises:
        ValueError: If the arrays disagree on their first-axis length. This is
            the cheapest possible place to catch an X/y misalignment, so the
            check is unconditional rather than an assert.
    """

    def __init__(self, *arrays: Any) -> None:
        if not arrays:
            raise ValueError("TensorDataset needs at least one array")
        self.arrays = tuple(np.asarray(a) for a in arrays)
        lengths = {len(a) for a in self.arrays}
        if len(lengths) != 1:
            raise ValueError(
                f"all arrays must share the first axis; got lengths {[len(a) for a in self.arrays]}"
            )

    def __len__(self) -> int:
        return len(self.arrays[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        return tuple(a[index] for a in self.arrays)

    def __repr__(self) -> str:
        shapes = ", ".join(str(a.shape) for a in self.arrays)
        return f"TensorDataset({shapes})"


class Subset(Dataset):
    """A view of ``dataset`` restricted to ``indices``, in that order.

    Holds a reference, never a copy: splitting 60k MNIST images must not
    duplicate 47 MB of pixels.
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[int(self.indices[index])]

    def __repr__(self) -> str:
        return f"Subset({self.dataset!r}, n={len(self)})"


def random_split(
    dataset: Dataset,
    lengths: Sequence[int] | Sequence[float],
    rng: np.random.Generator | None = None,
) -> list[Subset]:
    """Split ``dataset`` into disjoint :class:`Subset` views.

    Args:
        dataset: The dataset to split.
        lengths: Either absolute counts summing to ``len(dataset)``, or
            fractions summing to 1.0.
        rng: Generator for the permutation. ``None`` means *no shuffle* -- a
            contiguous split -- rather than a silently seeded one, because a
            global seed is exactly the reproducibility hole this codebase bans.

    Returns:
        One :class:`Subset` per entry of ``lengths``, sharing no index.

    Raises:
        ValueError: If the lengths do not sum to the dataset size.
    """
    n = len(dataset)
    lens = list(lengths)
    if lens and all(isinstance(x, float) for x in lens) and abs(sum(lens) - 1.0) < 1e-9:
        counts = [int(np.floor(f * n)) for f in lens]
        # Hand the rounding remainder to the largest split; dropping it would
        # silently discard samples, which is the sort of thing nobody notices.
        counts[int(np.argmax(counts))] += n - sum(counts)
        lens = counts
    lens = [int(x) for x in lens]
    # Negatives have to be rejected *before* the sum check, which they pass
    # happily. A negative size makes the slice cursor run backwards, so
    # `perm[start : start + size]` is empty and the *next* split starts before
    # the previous one ended -- silently overlapping subsets, which is a
    # train/test leak that shows up only as an implausibly good score.
    if any(x < 0 for x in lens):
        raise ValueError(f"split lengths must be non-negative, got {lens}")
    if sum(lens) != n:
        raise ValueError(f"split lengths {lens} sum to {sum(lens)}, but the dataset has {n} items")

    perm = np.arange(n) if rng is None else rng.permutation(n)
    out: list[Subset] = []
    start = 0
    for size in lens:
        out.append(Subset(dataset, perm[start : start + size]))
        start += size
    return out


def default_collate(samples: list[Any]) -> Any:
    """Stack a list of samples into batch arrays along a new leading axis.

    Tuples/lists are transposed field-wise (``[(x, y), ...] -> (X, Y)``) so the
    per-field alignment of the incoming samples is preserved by construction.
    Anything else is stacked directly.
    """
    first = samples[0]
    if isinstance(first, (tuple, list)):
        return tuple(np.stack([np.asarray(s[i]) for s in samples]) for i in range(len(first)))
    return np.stack([np.asarray(s) for s in samples])


class DataLoader:
    """Iterate a :class:`Dataset` in batches. Plain Python iterable, one process.

    Args:
        dataset: Any map-style dataset (``__len__`` + ``__getitem__``).
        batch_size: Samples per batch.
        shuffle: Draw a fresh permutation at the start of every epoch.
        rng: ``np.random.Generator`` used for that permutation. Required when
            ``shuffle=True`` -- there is no implicit global seeding anywhere in
            this package.
        drop_last: Drop the final partial batch. Useful when a layer needs a
            fixed batch size; off by default because dropping data silently is
            worse than a ragged last batch.
        collate_fn: Turns a list of samples into a batch. Defaults to
            :func:`default_collate`.

    The shuffle is the part that matters:

    **One permutation, indexed into every field of the sample.** The loader
    permutes *indices* and pulls whole samples through them, so ``X`` and ``y``
    cannot come apart. The tempting alternative -- ``rng.shuffle(X)`` then
    ``rng.shuffle(y)`` -- draws two independent permutations and destroys the
    label correspondence. That failure is genuinely nasty: training does not
    crash, it converges to predicting the marginal class distribution, so the
    loss parks at exactly ``ln(C)`` (2.3026 for ten classes) and never moves.
    Worse, the usual sanity check lies: overfit-one-batch still "succeeds",
    because a single fixed batch has a single fixed (wrong) labelling that a big
    enough net memorises perfectly.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
        drop_last: bool = False,
        collate_fn: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if shuffle and rng is None:
            raise ValueError(
                "shuffle=True requires an explicit rng=np.random.Generator; "
                "this package never seeds a global RNG"
            )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = rng
        self.drop_last = bool(drop_last)
        self.collate_fn = collate_fn or default_collate

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        n = len(self.dataset)
        # ONE permutation. It indexes the sample, so every field of that sample
        # travels together -- see the class docstring for what the alternative
        # costs you.
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                break
            yield self.collate_fn([self.dataset[int(i)] for i in idx])

    def __repr__(self) -> str:
        return (
            f"DataLoader(n={len(self.dataset)}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, drop_last={self.drop_last})"
        )
