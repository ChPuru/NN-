"""Running metrics, a progress bar, and a timer for training loops.

Nothing here is deep -- but two of the four classes exist because the naive
version reports a number that is subtly not the thing it claims to be:

* :class:`AverageMeter` weights each update by its batch size. Averaging the
  *per-batch means* instead gives the wrong epoch loss whenever the last batch
  is short, and the error is small enough to look like noise.
* :class:`Accuracy` counts correct predictions across the whole epoch rather
  than averaging per-batch accuracies, for the same reason.

:class:`ProgressBar` uses ``tqdm`` when it is installed and falls back to a
plain carriage-return bar when it is not. ``tqdm`` is a convenience, never a
dependency: a training script that cannot run on a bare NumPy install is a
worse artifact than one without a spinner.
"""

from __future__ import annotations

import shutil
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np

__all__ = ["Accuracy", "AverageMeter", "ProgressBar", "Timer", "format_time"]


def _to_numpy(x: Any) -> np.ndarray:
    """Accept a Tensor, a Parameter, or anything ``np.asarray`` understands."""
    data = getattr(x, "data", None)
    return np.asarray(data if isinstance(data, np.ndarray) else x)


def format_time(seconds: float) -> str:
    """``h:mm:ss`` past an hour, ``m:ss`` past a minute, else ``12.3s``."""
    if seconds < 0 or not np.isfinite(seconds):
        return "?"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m:d}:{s:02d}"
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class AverageMeter:
    """A count-weighted running mean.

    ``update(value, n)`` accumulates ``value * n``, so feeding it a batch mean
    together with the batch size reproduces the exact mean over all samples --
    including a ragged final batch, where averaging the batch means does not.

    Attributes:
        val: The most recent value passed to :meth:`update`.
        avg: Count-weighted mean of everything seen since :meth:`reset`.
        sum: Weighted total.
        count: Total weight.
    """

    __slots__ = ("count", "fmt", "name", "sum", "val")

    def __init__(self, name: str = "", fmt: str = ".4f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, value: Any, n: int = 1) -> None:
        """Record ``value`` observed over ``n`` samples."""
        # .mean() rather than float(): a Tensor has no __float__, and a caller
        # who hands over a whole batch of per-sample losses means their mean.
        v = float(_to_numpy(value).mean())
        self.val = v
        self.sum += v * n
        self.count += n

    @property
    def avg(self) -> float:
        # An empty meter is 0.0, not NaN: it is printed in the first log line
        # before any update, and a NaN there reads as a training failure.
        return self.sum / self.count if self.count else 0.0

    def __float__(self) -> float:
        return self.avg

    def __str__(self) -> str:
        label = f"{self.name} " if self.name else ""
        return f"{label}{self.avg:{self.fmt}}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"AverageMeter(name={self.name!r}, avg={self.avg!r}, count={self.count!r})"


class Accuracy:
    """Top-``k`` classification accuracy accumulated over an epoch.

    ``update`` accepts either raw logits ``(N, C)`` or already-argmaxed
    predictions ``(N,)``. Logits are the normal case -- the model emits raw
    logits by contract and the loss owns the softmax, so the metric must argmax
    the logits itself rather than expect probabilities.
    """

    __slots__ = ("correct", "count", "topk")

    def __init__(self, topk: int = 1) -> None:
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        self.topk = int(topk)
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.count = 0

    def update(self, outputs: Any, targets: Any) -> float:
        """Accumulate one batch and return that batch's accuracy.

        Args:
            outputs: Logits ``(N, C)`` or predicted labels ``(N,)``.
            targets: Integer labels ``(N,)``, or one-hot ``(N, C)``.

        Returns:
            The accuracy of this batch alone, in ``[0, 1]``.
        """
        out = _to_numpy(outputs)
        tgt = _to_numpy(targets)
        if tgt.ndim == 2:
            tgt = tgt.argmax(axis=1)
        tgt = tgt.astype(np.int64, copy=False)

        if out.ndim == 1:
            if self.topk != 1:
                raise ValueError("top-k > 1 needs logits, not pre-argmaxed predictions")
            hit = out.astype(np.int64, copy=False) == tgt
        else:
            k = min(self.topk, out.shape[1])
            # argpartition, not a full argsort: O(NC) instead of O(NC log C),
            # and the ordering within the top-k is irrelevant for a hit test.
            top = np.argpartition(-out, kth=k - 1, axis=1)[:, :k]
            hit = (top == tgt[:, None]).any(axis=1)

        n = int(tgt.shape[0])
        c = int(hit.sum())
        self.correct += c
        self.count += n
        return c / n if n else 0.0

    @property
    def value(self) -> float:
        return self.correct / self.count if self.count else 0.0

    def __float__(self) -> float:
        return self.value

    def __str__(self) -> str:
        return f"{100.0 * self.value:.2f}% ({self.correct}/{self.count})"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Accuracy(topk={self.topk}, value={self.value!r}, count={self.count})"


class ProgressBar:
    """Iterate with a progress display, using ``tqdm`` only if it is installed.

    Args:
        iterable: What to iterate. If it has no ``len``, pass ``total``.
        total: Number of steps, when ``iterable`` cannot report it.
        desc: Prefix label.
        width: Bar width in characters for the fallback renderer.
        stream: Where to draw; defaults to ``sys.stderr`` so the bar does not
            pollute a redirected stdout log.
        use_tqdm: Set False to force the fallback (the tests do this so the
            output is deterministic).

    Notes:
        The fallback writes ``\\r`` and never ``\\n`` until :meth:`close`, so a
        non-tty log file gets one very long line rather than thousands. Guard
        with ``disable=not sys.stderr.isatty()`` in scripts that log to a file.
    """

    def __init__(
        self,
        iterable: Iterable[Any] | None = None,
        total: int | None = None,
        desc: str = "",
        width: int = 30,
        stream: Any | None = None,
        use_tqdm: bool = True,
        disable: bool = False,
    ) -> None:
        self.iterable = iterable
        if total is None and iterable is not None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        self.total = total
        self.desc = desc
        self.width = int(width)
        self.stream = stream if stream is not None else sys.stderr
        self.disable = bool(disable)
        self.n = 0
        self.postfix = ""
        self._start = time.perf_counter()
        self._closed = False
        self._tqdm = _load_tqdm() if (use_tqdm and not disable) else None
        self._bar = None

    def _ensure_bar(self):
        """Create the live tqdm instance, once, and keep a handle on it.

        The handle is the point. An earlier version built a *local* tqdm inside
        ``__iter__`` and yielded from it: iteration looked fine, but ``self.n``
        never advanced, ``set_postfix`` wrote to a field nothing read, and
        ``close`` had nothing to close. The two code paths have to share state
        or the tqdm one silently drops every feature the fallback provides.
        """
        if self._bar is None and self._tqdm is not None:
            self._bar = self._tqdm(total=self.total, desc=self.desc, file=self.stream, leave=True)
        return self._bar

    def __iter__(self) -> Iterator[Any]:
        if self.iterable is None:
            raise TypeError("ProgressBar has no iterable; use update() manually")
        try:
            for item in self.iterable:
                yield item
                self.update(1)
        finally:
            # `finally`, so a `break` out of the loop still restores the terminal.
            self.close()

    def update(self, n: int = 1) -> None:
        self.n += n
        if self.disable:
            return
        bar = self._ensure_bar()
        if bar is not None:
            bar.update(n)
        else:
            self._render()

    def set_postfix(self, **kwargs: Any) -> None:
        """Attach ``key=value`` pairs to the right-hand end of the bar."""
        self.postfix = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in kwargs.items()
        )
        if self.disable:
            return
        bar = self._ensure_bar()
        if bar is not None:
            bar.set_postfix_str(self.postfix)
        elif self.n:
            self._render()

    def _render(self) -> None:
        elapsed = time.perf_counter() - self._start
        if self.total:
            frac = min(1.0, self.n / self.total)
            filled = round(self.width * frac)  # round() of a float returns an int
            bar = "#" * filled + "-" * (self.width - filled)
            # Remaining time from the observed rate, not from a per-step guess:
            # the first step of an epoch is always slow (allocation, warm-up).
            eta = (elapsed / self.n) * (self.total - self.n) if self.n else float("inf")
            line = (
                f"{self.desc} [{bar}] {self.n}/{self.total} "
                f"{format_time(elapsed)}<{format_time(eta)}"
            )
        else:
            line = f"{self.desc} {self.n} it {format_time(elapsed)}"
        if self.postfix:
            line += " " + self.postfix
        cols = shutil.get_terminal_size((100, 24)).columns
        self.stream.write("\r" + line[: max(1, cols - 1)])
        self.stream.flush()

    def close(self) -> None:
        if self._closed or self.disable:
            self._closed = True
            return
        self._closed = True
        if self._bar is not None:
            # tqdm restores the terminal itself; writing our own newline on top
            # of that leaves a blank line after every epoch.
            self._bar.close()
            self._bar = None
            return
        self.stream.write("\n")
        self.stream.flush()

    def __enter__(self) -> ProgressBar:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def _load_tqdm():
    """Return ``tqdm.tqdm`` or None. Import is local so absence costs nothing."""
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm


class Timer:
    """Wall-clock timing, as a context manager or by hand.

    Uses ``time.perf_counter``: ``time.time`` is subject to NTP steps and can
    run backwards, which turns a benchmark into a negative duration.

    Example:
        >>> with Timer("epoch") as t:
        ...     pass
        >>> t.elapsed >= 0.0
        True
    """

    __slots__ = ("_elapsed", "_t0", "laps", "name")

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._t0: float | None = None
        self._elapsed = 0.0
        self.laps: list[float] = []

    def start(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._t0 is None:
            raise RuntimeError("Timer.stop() called before start()")
        self._elapsed = time.perf_counter() - self._t0
        self._t0 = None
        self.laps.append(self._elapsed)
        return self._elapsed

    def lap(self) -> float:
        """Record the split since the last lap (or since start) and continue."""
        if self._t0 is None:
            raise RuntimeError("Timer.lap() called before start()")
        now = time.perf_counter()
        dt = now - self._t0
        self._t0 = now
        self.laps.append(dt)
        return dt

    @property
    def elapsed(self) -> float:
        """Seconds of the last completed interval, or of the running one."""
        if self._t0 is not None:
            return time.perf_counter() - self._t0
        return self._elapsed

    @property
    def total(self) -> float:
        return float(sum(self.laps))

    def summary(self, laps: Sequence[float] | None = None) -> str:
        """``mean +/- std`` over the recorded laps -- a one-lap timer has no std."""
        xs = np.asarray(list(laps) if laps is not None else self.laps, dtype=np.float64)
        if xs.size == 0:
            return f"{self.name}: no laps"
        return f"{self.name}: {xs.mean():.4f}s +/- {xs.std():.4f}s over {xs.size} laps"

    def __enter__(self) -> Timer:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Timer(name={self.name!r}, elapsed={self.elapsed:.6f})"
