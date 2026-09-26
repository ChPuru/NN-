"""Seeding and RNG scoping.

Reproducibility is a **graded property** of this project: running the same
script twice with the same seed must produce *bit-identical* losses, not
"similar" ones. Two losses that agree to three decimals are not evidence of a
correct reimplementation -- they are evidence that some draw somewhere is
uncontrolled, and the day that draw matters (a shuffle order, a dropout mask, a
weight init) the discrepancy will be blamed on "randomness" instead of on the
bug it actually is.

The rule that follows from that: **every draw goes through an explicit
``np.random.Generator``** threaded down from the caller. :func:`seed_everything`
exists to mint that generator and, as a belt-and-braces measure, to pin the two
*global* streams that third-party code (and careless code of our own) might
reach for -- ``random`` and the legacy ``np.random`` singleton. Pinning them is
not a licence to use them; it only means an accidental use is reproducible
rather than silently seed-independent.

``PYTHONHASHSEED`` is deliberately **not** set here. It has to be in the
environment before the interpreter starts, so assigning ``os.environ`` from
inside a running process does nothing at all -- a popular ``seed_everything``
snippet does exactly that and reports success. Set it in the shell if set
iteration order ever reaches a numeric result.
"""

from __future__ import annotations

import random as _random
from typing import Any

import numpy as np

__all__ = ["RNGContext", "seed_everything", "spawn"]


def seed_everything(seed: int, *, quiet: bool = True) -> np.random.Generator:
    """Seed the global streams and return the generator you should actually use.

    Args:
        seed: Non-negative integer seed.
        quiet: When False, print the seed -- useful at the top of a training log
            so a run can be replayed from its own output.

    Returns:
        A fresh ``np.random.Generator`` seeded with ``seed``. Thread it through
        explicitly; do not fall back on the globals this function pins.

    Raises:
        TypeError: If ``seed`` is not an integer (a float seed silently
            truncates in some APIs and not others).
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    seed = int(seed)
    _random.seed(seed)
    np.random.seed(seed % (2**32))  # legacy singleton, for third-party code only
    if not quiet:
        print(f"[seed_everything] seed={seed}")
    return np.random.default_rng(seed)


def spawn(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Split ``rng`` into ``n`` independent child generators.

    Use this instead of ``default_rng(seed + i)``: nearby integer seeds are not
    guaranteed to give non-overlapping streams, whereas ``SeedSequence``
    spawning is designed for exactly this.
    """
    ss = rng.bit_generator.seed_seq  # type: ignore[attr-defined]
    return [np.random.default_rng(child) for child in ss.spawn(n)]


class RNGContext:
    """Scope a seed: save the global RNG states, seed, and restore on exit.

    The generator is exposed as the ``with`` target, so the intended use never
    touches a global at all::

        with RNGContext(0) as rng:
            x = rng.normal(size=(7, 11))

    Restoring on exit is what makes an evaluation pass (dropout masks, a
    subsampled validation split) repeatable *without* perturbing the training
    stream -- the classic failure is a "deterministic eval" that reseeds the
    global RNG and thereby changes every subsequent training batch.
    """

    __slots__ = ("_np_state", "_py_state", "rng", "seed")

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        self.seed = int(seed)
        self.rng: np.random.Generator | None = None
        self._py_state: Any = None
        self._np_state: Any = None

    def __enter__(self) -> np.random.Generator:
        self._py_state = _random.getstate()
        self._np_state = np.random.get_state()
        _random.seed(self.seed)
        np.random.seed(self.seed % (2**32))
        self.rng = np.random.default_rng(self.seed)
        return self.rng

    def __exit__(self, exc_type, exc, tb) -> bool:
        _random.setstate(self._py_state)
        np.random.set_state(self._np_state)
        return False

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RNGContext(seed={self.seed})"
