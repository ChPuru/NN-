"""Quantum neural networks on a classical statevector simulator.

Everything here is a **classical simulation** in NumPy: a statevector of
``2**n`` complex amplitudes, gates applied by tensor contraction, expectations
read off exactly. It is a faithful model of what a variational circuit computes
and how its gradient is obtained (the parameter-shift rule), and it makes no
claim of quantum advantage -- simulating ``n`` qubits costs ``O(2**n)`` here.

* :mod:`.statevector` -- states, gates, ``<Z...Z>``; the qubit-ordering
  convention is documented there (wire 0 = most significant bit).
* :mod:`.circuit` -- :class:`Circuit` / :class:`Op` and the differentiable
  :meth:`Circuit.expval`, whose backward is the parameter-shift rule.
"""

from __future__ import annotations

from .circuit import FIXED_GATES, ROTATIONS, Circuit, Op
from .statevector import CNOT, CZ, H, X, Y, Z, apply_gate, expval_z, rx, ry, rz, zero_state

__all__ = [
    "CNOT",
    "CZ",
    "FIXED_GATES",
    "ROTATIONS",
    "Circuit",
    "H",
    "Op",
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
