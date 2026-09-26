"""Checkpointing built on ``np.savez_compressed``.

Three things this file is careful about, each because the obvious version is
wrong in a way that does not announce itself:

**1. ``state_dict()`` aliases the live parameters.**
:meth:`Module.state_dict` returns, by default, the *same* buffers the model is
training on. That is the right default -- saving a checkpoint should not double
the peak memory of a large model -- but it makes the in-memory idiom

.. code-block:: python

    best = model.state_dict()          # WRONG: a view, not a copy
    ...
    model.load_state_dict(best)        # restores whatever the last step wrote

a silent no-op. Every optimizer step writes through those arrays, so "restore
the best epoch" restores the *final* epoch and early stopping quietly stops
doing anything. Use :func:`snapshot`, which is ``state_dict(deep=True)`` with a
name that says why.

**2. No pickle.** ``np.load(..., allow_pickle=True)`` executes arbitrary code
from the file. Everything here is plain arrays plus one JSON manifest, and the
loader passes ``allow_pickle=False`` explicitly.

**3. Shapes are stored, and checked before anything is written.** A checkpoint
whose ``(784, 128)`` layer meets a ``(784, 256)`` model must raise, not
broadcast, not partially load. The check runs over *all* keys first so the model
is never left half-restored from an incompatible file.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "CheckpointError",
    "load_checkpoint",
    "restore",
    "save_checkpoint",
    "snapshot",
]

#: Bumped whenever the on-disk layout changes incompatibly.
SCHEMA_VERSION = 1

_MANIFEST_KEY = "__manifest__"
_ARRAY_TAG = "__ndarray__"
_TUPLE_TAG = "__tuple__"
_KEYED_TAG = "__keyed__"
#: Tags that own the whole JSON object they appear in, so a caller's dict may
#: not use them as keys -- decoding would read the caller's data as a tag.
_RESERVED_TAGS = (_ARRAY_TAG, _TUPLE_TAG, _KEYED_TAG)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be loaded into the given model."""


# --------------------------------------------------------------------------- #
# in-memory snapshots
# --------------------------------------------------------------------------- #
def snapshot(model) -> dict[str, np.ndarray]:
    """A **detached** copy of every parameter and buffer of ``model``.

    This is the function to use for early stopping, best-epoch tracking, or any
    other "remember this state and maybe come back to it" pattern.
    ``model.state_dict()`` alone shares storage with the live parameters, so the
    next optimizer step overwrites the thing you thought you saved.

    Args:
        model: Any :class:`~scratchgrad.nn.module.Module`.

    Returns:
        A plain dict mapping dotted parameter path to a freshly allocated array.
    """
    return dict(model.state_dict(deep=True))


def restore(model, state: dict[str, np.ndarray], strict: bool = True) -> None:
    """Load a :func:`snapshot` back into ``model``. Thin, symmetric alias."""
    model.load_state_dict(state, strict=strict)


# --------------------------------------------------------------------------- #
# JSON skeleton <-> arrays
# --------------------------------------------------------------------------- #
def _encode(obj: Any, arrays: list[np.ndarray]) -> Any:
    """Split ``obj`` into a JSON-safe skeleton plus a flat list of arrays.

    Optimizer state is a nested mix of hyperparameters (floats, ints, strings)
    and per-parameter buffers (arrays). JSON handles the first kind exactly and
    the second not at all, so arrays are lifted out and replaced by an index.

    JSON also has no tuple and no non-string object key, and the obvious
    coercions (``str(k)``, ``list(t)``) are one-way: a checkpoint round trip
    would then hand back ``{'0': 1.0}`` for ``{0: 1.0}`` and a list for a tuple,
    changing the *type* of the caller's metadata without saying so. Both are
    therefore tagged here and restored by :func:`_decode`.
    """
    if isinstance(obj, np.ndarray):
        arrays.append(np.asarray(obj))
        return {_ARRAY_TAG: len(arrays) - 1}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        for tag in _RESERVED_TAGS:
            if tag in obj:
                raise ValueError(f"{tag!r} is reserved and cannot be a dict key")
        if all(isinstance(k, str) for k in obj):
            # The common case stays a plain JSON object: readable, and what
            # every checkpoint written so far already contains.
            return {k: _encode(v, arrays) for k, v in obj.items()}
        return {_KEYED_TAG: [[_encode(k, arrays), _encode(v, arrays)] for k, v in obj.items()]}
    if isinstance(obj, tuple):
        return {_TUPLE_TAG: [_encode(v, arrays) for v in obj]}
    if isinstance(obj, list):
        return [_encode(v, arrays) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(
        f"cannot serialize {type(obj).__name__} into a checkpoint; "
        "use arrays, numbers, strings, lists and dicts"
    )


def _decode(obj: Any, arrays: dict[int, np.ndarray]) -> Any:
    if isinstance(obj, dict):
        if len(obj) == 1:
            if _ARRAY_TAG in obj:
                return arrays[int(obj[_ARRAY_TAG])]
            if _TUPLE_TAG in obj:
                return tuple(_decode(v, arrays) for v in obj[_TUPLE_TAG])
            if _KEYED_TAG in obj:
                return {_decode(k, arrays): _decode(v, arrays) for k, v in obj[_KEYED_TAG]}
        return {k: _decode(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v, arrays) for v in obj]
    return obj


def _with_suffix(path: str | os.PathLike) -> str:
    """``np.savez`` appends ``.npz`` behind your back; do it up front instead.

    Otherwise ``save_checkpoint("ck")`` writes ``ck.npz`` and the matching
    ``load_checkpoint("ck")`` reports a missing file.
    """
    p = os.fspath(path)
    return p if p.endswith(".npz") else p + ".npz"


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str | os.PathLike,
    model,
    optimizer: Any | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """Write model (and optionally optimizer) state to a compressed ``.npz``.

    Args:
        path: Destination. ``.npz`` is appended if absent.
        model: Module whose ``state_dict()`` is saved.
        optimizer: Anything exposing ``state_dict()`` (momentum buffers, step
            counts, hyperparameters). Resuming without it restarts Adam's
            moments at zero, which shows up as a visible loss spike.
        meta: JSON-serializable extras -- epoch, best metric, git sha, the seed.

    Returns:
        The path actually written (with the ``.npz`` suffix).
    """
    target = _with_suffix(path)
    parent = os.path.dirname(os.path.abspath(target))
    if parent:
        os.makedirs(parent, exist_ok=True)

    arrays: list[np.ndarray] = []
    model_entry: dict[str, dict[str, Any]] = {}
    # deep=True so the buffers handed to savez cannot be mutated by a training
    # step between here and the moment the zip is flushed.
    for name, value in model.state_dict(deep=True).items():
        arr = np.asarray(value)
        arrays.append(arr)
        model_entry[name] = {
            "i": len(arrays) - 1,
            "shape": list(arr.shape),
            "dtype": arr.dtype.str,
        }

    opt_state = None
    if optimizer is not None:
        if not hasattr(optimizer, "state_dict"):
            raise TypeError(
                f"optimizer of type {type(optimizer).__name__} has no state_dict(); "
                "pass None or an object that implements it"
            )
        opt_state = _encode(optimizer.state_dict(), arrays)

    manifest = {
        "schema": SCHEMA_VERSION,
        "model": model_entry,
        "optimizer": opt_state,
        "meta": _encode(dict(meta) if meta else {}, arrays),
    }

    payload: dict[str, np.ndarray] = {f"a{i}": a for i, a in enumerate(arrays)}
    payload[_MANIFEST_KEY] = np.array(json.dumps(manifest))
    np.savez_compressed(target, **payload)
    return target


def load_checkpoint(
    path: str | os.PathLike,
    model,
    optimizer: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Restore ``model`` (and optionally ``optimizer``) from a checkpoint.

    Args:
        path: File written by :func:`save_checkpoint`. ``.npz`` optional.
        model: Module to load into, in place.
        optimizer: If given, its ``load_state_dict`` is called with the saved
            optimizer state. Raises if the file has none.
        strict: Forwarded to ``Module.load_state_dict``; when True a missing or
            unexpected key is an error rather than a shrug.

    Returns:
        The ``meta`` dict stored alongside the weights.

    Raises:
        CheckpointError: On a schema mismatch, a shape mismatch, or a request
            for optimizer state the file does not contain.
    """
    target = _with_suffix(path)
    with np.load(target, allow_pickle=False) as z:  # never allow_pickle: it execs
        if _MANIFEST_KEY not in z.files:
            raise CheckpointError(
                f"{target} is not a scratchgrad checkpoint (no {_MANIFEST_KEY} entry)"
            )
        manifest = json.loads(str(z[_MANIFEST_KEY].item()))
        if manifest.get("schema") != SCHEMA_VERSION:
            raise CheckpointError(
                f"checkpoint schema {manifest.get('schema')} != {SCHEMA_VERSION} "
                "(this file was written by a different version of scratchgrad)"
            )
        arrays = {i: z[f"a{i}"] for i in range(len(z.files) - 1)}

    model_entry = manifest["model"]
    live = dict(model.state_dict(deep=False))

    # Validate every key BEFORE mutating anything: a half-loaded model that
    # still runs is far worse than a loud failure.
    bad = [
        (k, tuple(e["shape"]), live[k].shape)
        for k, e in model_entry.items()
        if k in live and tuple(e["shape"]) != tuple(live[k].shape)
    ]
    if bad:
        detail = "; ".join(f"{k}: checkpoint {c} vs model {m}" for k, c, m in bad)
        raise CheckpointError(f"shape mismatch loading {target} -- {detail}")

    state = {k: arrays[int(e["i"])] for k, e in model_entry.items()}
    try:
        model.load_state_dict(state, strict=strict)
    except KeyError as exc:
        raise CheckpointError(f"key mismatch loading {target}: {exc}") from exc
    except ValueError as exc:
        # The pre-check above compares the manifest's *recorded* shapes; this
        # catches the case where the recorded shape and the array actually
        # stored disagree (a truncated or hand-edited file). The docstring
        # promises CheckpointError for a shape mismatch, so numpy's ValueError
        # must not be what a caller has to except on.
        raise CheckpointError(f"shape mismatch loading {target}: {exc}") from exc

    if optimizer is not None:
        if manifest.get("optimizer") is None:
            raise CheckpointError(
                f"{target} contains no optimizer state, but an optimizer was passed"
            )
        optimizer.load_state_dict(_decode(manifest["optimizer"], arrays))

    return _decode(manifest.get("meta", {}), arrays)
