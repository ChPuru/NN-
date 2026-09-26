"""A dense statevector simulator: the state is a complex128 array of shape ``(2,) * n``.

This is a **classical simulation**. Memory and time grow as ``2**n``; nothing
here is, or claims to be, a quantum speed-up.

The qubit-ordering convention, stated once because every simulator picks one and
two of the big ones picked opposite ones:

* Wire ``w`` is **axis ``w``** of the state array.
* Flattening in C order therefore makes **wire 0 the most significant bit**:
  ``|q0 q1 ... q_{n-1}>`` sits at flat index ``sum_w q_w * 2**(n-1-w)``. So on
  three wires ``X`` on wire 0 gives index 4 (``|100>``) and on wire 2 gives
  index 1 (``|001>``). This is the textbook and PennyLane order.
* Qiskit is the reverse (qubit 0 is the *least* significant bit). The trap is
  that the matrix does not change, only its meaning: the 4x4 ``CNOT`` below is
  "control = first wire listed" here, while Qiskit prints the same matrix for a
  CNOT whose control is its qubit 1. Mixing a gate matrix from one source with
  a statevector printout from the other gives a circuit that is plausibly
  wrong -- normalised, unitary, and computing a different function.

A k-qubit gate matrix is written in the basis ``|a b ...>`` of the wires **in the
order they are passed**, first wire most significant. Applying it moves those
axes to the front, reshapes to ``(2**k, rest)``, and does one matmul; nothing is
ever expanded to a ``2**n x 2**n`` matrix.

Rotations are ``R_P(theta) = exp(-i theta P / 2)`` with the **half angle**. The
parameter-shift rule in :mod:`.circuit` (shift ``pi/2``, factor ``1/2``) is exact
for exactly this convention; with ``exp(-i theta P)`` it would be shift ``pi/4``,
factor 1, and the ``pi/2`` rule would return a gradient of the wrong size.

Batching: pass ``batched=True`` and axis 0 is a batch of independent states.
It has to be an explicit flag -- a batch of two states is shape-identical to one
extra qubit, so no shape inspection can tell the two apart.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "CNOT",
    "CZ",
    "H",
    "X",
    "Y",
    "Z",
    "apply_gate",
    "expval_z",
    "rx",
    "ry",
    "rz",
    "zero_state",
]


def _const(m) -> np.ndarray:
    a = np.array(m, dtype=np.complex128)
    a.setflags(write=False)  # a module-level gate edited in place would corrupt every circuit
    return a


_S = 1.0 / np.sqrt(2.0)
H = _const([[_S, _S], [_S, -_S]])
X = _const([[0, 1], [1, 0]])
Y = _const([[0, -1j], [1j, 0]])
Z = _const([[1, 0], [0, -1]])
#: Control = first wire listed, target = second, in basis |control target>.
CNOT = _const([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
#: Symmetric in its two wires, so the ordering trap cannot bite here.
CZ = _const(np.diag([1, 1, 1, -1]))


def _half(theta) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(theta, dtype=np.float64) / 2.0
    return np.cos(t), np.sin(t)


def rx(theta) -> np.ndarray:
    """``exp(-i theta X / 2)``. ``theta`` of shape ``S`` gives matrices ``S + (2, 2)``."""
    c, s = _half(theta)
    out = np.empty((*c.shape, 2, 2), dtype=np.complex128)
    out[..., 0, 0] = c
    out[..., 1, 1] = c
    out[..., 0, 1] = -1j * s
    out[..., 1, 0] = -1j * s
    return out


def ry(theta) -> np.ndarray:
    """``exp(-i theta Y / 2)`` -- real, so ``RY`` alone never leaves the real plane."""
    c, s = _half(theta)
    out = np.empty((*c.shape, 2, 2), dtype=np.complex128)
    out[..., 0, 0] = c
    out[..., 1, 1] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    return out


def rz(theta) -> np.ndarray:
    """``exp(-i theta Z / 2) = diag(e^{-i theta/2}, e^{+i theta/2})``."""
    t = np.asarray(theta, dtype=np.float64) / 2.0
    out = np.zeros((*t.shape, 2, 2), dtype=np.complex128)
    out[..., 0, 0] = np.exp(-1j * t)
    out[..., 1, 1] = np.exp(1j * t)
    return out


def zero_state(n_qubits: int, batch: int | None = None) -> np.ndarray:
    """``|0...0>`` as shape ``(2,)*n``, or ``(batch,) + (2,)*n`` when ``batch`` is given."""
    if n_qubits < 1:
        raise ValueError(f"need at least one qubit, got {n_qubits}")
    lead = () if batch is None else (int(batch),)
    psi = np.zeros(lead + (2,) * n_qubits, dtype=np.complex128)
    psi[(..., *([0] * n_qubits))] = 1.0
    return psi


def _check_wires(wires: Sequence[int], n: int) -> tuple[int, ...]:
    w = (int(wires),) if np.isscalar(wires) else tuple(int(i) for i in wires)
    if len(set(w)) != len(w) or any(not 0 <= i < n for i in w):
        raise ValueError(f"wires {w} invalid for {n} qubits (must be distinct, in [0, {n}))")
    return w


def apply_gate(
    state: np.ndarray, gate: np.ndarray, wires: Sequence[int] | int, batched: bool = False
) -> np.ndarray:
    """Apply a ``k``-qubit gate to ``wires`` and return the new state (input untouched).

    Args:
        state: ``(2,)*n`` or, with ``batched=True``, ``(N,) + (2,)*n``.
        gate: ``(2**k, 2**k)``, or ``(N, 2**k, 2**k)`` -- one matrix per batch
            element, which is how per-sample data-encoding angles are applied.
        wires: The ``k`` target wires, in the order the gate matrix assumes.
        batched: Whether axis 0 of ``state`` is a batch axis.
    """
    off = 1 if batched else 0
    n = state.ndim - off
    w = _check_wires(wires, n)
    k = len(w)
    gate = np.asarray(gate)
    if gate.shape[-2:] != (2**k, 2**k):
        raise ValueError(f"a {k}-qubit gate must be {2**k}x{2**k}, got {gate.shape}")
    if gate.ndim == 3 and not (batched and gate.shape[0] == state.shape[0]):
        raise ValueError("a stack of gates needs batched=True and one matrix per state")
    src = [off + i for i in w]
    dst = list(range(off, off + k))
    psi = np.moveaxis(state, src, dst)
    moved = psi.shape
    psi = psi.reshape((moved[0], 2**k, -1) if batched else (2**k, -1))
    return np.moveaxis((gate @ psi).reshape(moved), dst, src)


def expval_z(state: np.ndarray, wires: Sequence[int] | int, batched: bool = False):
    """``<psi| Z_{w1} Z_{w2} ... |psi>`` -- a Pauli-Z product; ``wires=()`` gives the norm.

    Z-products are diagonal in the computational basis, so this is a signed sum
    of probabilities and needs no matrix at all. Returns a float, or shape
    ``(N,)`` when batched.
    """
    off = 1 if batched else 0
    n = state.ndim - off
    w = _check_wires(wires, n)
    probs = state.real**2 + state.imag**2
    for i in w:
        shape = [1] * probs.ndim
        shape[off + i] = 2
        probs = probs * np.array([1.0, -1.0]).reshape(shape)
    if batched:
        return probs.reshape(probs.shape[0], -1).sum(axis=1)
    return float(probs.sum())
