"""Global configuration: default dtype and the grad-enabled flag.

Kept deliberately tiny and dependency-free so every other module can import it
without creating a cycle.
"""

from __future__ import annotations

import contextlib
import threading

import numpy as np

__all__ = [
    "default_dtype",
    "enable_grad",
    "get_default_dtype",
    "is_grad_enabled",
    "no_grad",
    "set_default_dtype",
    "set_grad_enabled",
]

# Thread-local so a DataLoader worker or a benchmark thread cannot flip the
# mode out from under the main training loop.
_state = threading.local()

_DEFAULT_DTYPE = np.float32


def _s() -> threading.local:
    if not hasattr(_state, "grad_enabled"):
        _state.grad_enabled = True
        _state.dtype = _DEFAULT_DTYPE
    return _state


def get_default_dtype() -> np.dtype:
    """The dtype used for tensors created without an explicit ``dtype=``."""
    return np.dtype(_s().dtype)


def set_default_dtype(dtype) -> None:
    """Set the global default floating dtype (``float32`` or ``float64``).

    Training defaults to ``float32`` (roughly 1.5-2x faster than float64 on this
    class of workload -- not the 7.5x a popular microbenchmark claims, which is
    dominated by writing a much smaller output matrix). Gradient checking
    *requires* float64: in float32 a perfectly correct gradient shows ~1e-2
    relative error and the check has no power at all.
    """
    dt = np.dtype(dtype)
    if dt.kind != "f":
        raise TypeError(f"default dtype must be a floating type, got {dt}")
    _s().dtype = dt


@contextlib.contextmanager
def default_dtype(dtype):
    """Temporarily set the default dtype. Used by ``gradcheck`` to force float64."""
    prev = get_default_dtype()
    set_default_dtype(dtype)
    try:
        yield
    finally:
        set_default_dtype(prev)


def is_grad_enabled() -> bool:
    return bool(_s().grad_enabled)


def set_grad_enabled(mode: bool) -> None:
    _s().grad_enabled = bool(mode)


@contextlib.contextmanager
def no_grad():
    """Disable graph construction inside the block (evaluation, inference)."""
    prev = is_grad_enabled()
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(prev)


@contextlib.contextmanager
def enable_grad():
    """Re-enable graph construction inside a ``no_grad`` block."""
    prev = is_grad_enabled()
    set_grad_enabled(True)
    try:
        yield
    finally:
        set_grad_enabled(prev)
