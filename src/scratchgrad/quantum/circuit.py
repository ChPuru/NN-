"""Parameterised circuits and an autograd op whose backward is the parameter-shift rule.

A :class:`Circuit` is a flat list of :class:`Op`. A rotation's angle is
``params[op.param] + x[:, op.feature]`` (either term may be absent), so the same
description covers trainable gates, data-encoding gates, and data re-uploading
(the same feature fed in again in a later layer). :meth:`Circuit.expval` returns
``<Z_{w1} ... Z_{wk}>`` for every row of ``x`` as a :class:`~scratchgrad.tensor.Tensor`
wired into the graph like any other op.

The backward never differentiates through the simulator. For a gate
``exp(-i theta P / 2)`` with ``P`` a Pauli, the expectation is exactly
``A + B cos(theta) + C sin(theta)`` in that one angle, so

    dE/dtheta = [E(theta + pi/2) - E(theta - pi/2)] / 2

with no truncation error at all -- the derivation is in
``notes/derivations/12-parameter-shift.md``. It is also what one would run on
hardware, where the only thing available is the expectation itself. The traps:

* **A shared parameter is shifted one occurrence at a time.** If ``theta`` feeds
  ``m`` gates, the derivative is the *sum* of ``m`` two-term shifts, each moving
  one gate. Shifting ``theta`` itself (every occurrence at once) evaluates the
  finite difference of a function that is no longer a single sinusoid: for
  ``RY(theta) RY(theta)`` it returns ``[cos(2t+pi) - cos(2t-pi)]/2 = 0`` for
  every ``theta``, against the true ``-2 sin(2 theta)``.
* **The rule is for generators with eigenvalues +-1/2 only.** ``exp(-i theta P)``
  (no half) needs shift ``pi/4``; a controlled rotation has generator
  eigenvalues ``{0, +-1/2}`` and needs a four-term rule. Neither is offered here.
* **It is not a finite difference.** ``pi/2`` is not a small step and there is
  no step size to tune; agreeing with central differences to ~1e-10 while being
  exact is what the tests check.

Cost: ``2m`` circuit evaluations for ``m`` parameterised gate occurrences --
linear in the parameter count, unlike reverse mode's constant factor, which is
the honest price of a rule that only needs expectation values. All ``2m``
shifted circuits are stacked into one batch and simulated in a single pass.

Inputs ``x`` are treated as constants: :meth:`Circuit.expval` refuses an ``x``
that requires grad rather than silently handing it a zero gradient.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..tensor import Tensor
from .statevector import CNOT, CZ, H, X, Y, Z, apply_gate, expval_z, rx, ry, rz, zero_state

__all__ = ["FIXED_GATES", "ROTATIONS", "Circuit", "Op"]

#: Rotations ``exp(-i theta P / 2)``: the only gates that take an angle.
ROTATIONS = {"RX": rx, "RY": ry, "RZ": rz}
#: Angle-free gates and their arity is read off the matrix.
FIXED_GATES = {"H": H, "X": X, "Y": Y, "Z": Z, "CNOT": CNOT, "CZ": CZ}

_SHIFT = np.pi / 2


@dataclass(frozen=True)
class Op:
    """One gate. Rotations need ``param`` and/or ``feature``; fixed gates take neither.

    Attributes:
        name: A key of :data:`ROTATIONS` or :data:`FIXED_GATES`.
        wires: Target wires, in the order the gate matrix assumes (control first).
        param: Index into the parameter vector, or ``None``.
        feature: Column of ``x`` added to the angle, or ``None``.
    """

    name: str
    wires: tuple[int, ...]
    param: int | None = None
    feature: int | None = None

    def __post_init__(self) -> None:
        wires = (self.wires,) if np.isscalar(self.wires) else tuple(self.wires)
        object.__setattr__(self, "wires", tuple(int(w) for w in wires))
        if self.name in ROTATIONS:
            arity = 1
            if self.param is None and self.feature is None:
                raise ValueError(f"{self.name} needs a param and/or a feature for its angle")
        elif self.name in FIXED_GATES:
            arity = int(np.log2(FIXED_GATES[self.name].shape[0]))
            if self.param is not None or self.feature is not None:
                raise ValueError(f"{self.name} takes no angle")
        else:
            raise ValueError(f"unknown gate {self.name!r}")
        if len(self.wires) != arity:
            raise ValueError(f"{self.name} acts on {arity} wire(s), got {self.wires}")


class Circuit:
    """A fixed gate sequence on ``n_qubits`` wires, measured as a Pauli-Z product.

    Args:
        n_qubits: Number of wires.
        ops: The gates, applied in order to ``|0...0>``.
        observable: Wires of the Z-product measured at the end; ``(0,)`` is
            ``<Z_0>``, ``(0, 1)`` is ``<Z_0 Z_1>``.
    """

    def __init__(self, n_qubits: int, ops: Sequence[Op], observable: Sequence[int] = (0,)):
        self.n_qubits = int(n_qubits)
        self.ops = tuple(ops)
        self.observable = tuple(int(w) for w in observable)
        for op in self.ops:
            if any(not 0 <= w < self.n_qubits for w in op.wires):
                raise ValueError(f"{op} touches a wire outside [0, {self.n_qubits})")
        if any(not 0 <= w < self.n_qubits for w in self.observable):
            raise ValueError(f"observable {self.observable} outside [0, {self.n_qubits})")
        params = [op.param for op in self.ops if op.param is not None]
        feats = [op.feature for op in self.ops if op.feature is not None]
        self.n_params = max(params) + 1 if params else 0
        self.n_features = max(feats) + 1 if feats else 0
        # Every *occurrence* of a parameter is shifted separately; see the module docstring.
        self._shifted = np.array([i for i, op in enumerate(self.ops) if op.param is not None])
        self._shifted_param = np.array([self.ops[i].param for i in self._shifted], dtype=int)

    def __repr__(self) -> str:
        return (
            f"Circuit(n_qubits={self.n_qubits}, gates={len(self.ops)}, "
            f"params={self.n_params}, features={self.n_features}, observable={self.observable})"
        )

    # -- plain numpy -----------------------------------------------------------
    def _inputs(self, x) -> np.ndarray:
        if x is None:
            if self.n_features:
                raise ValueError(f"circuit encodes {self.n_features} feature(s); pass x")
            return np.zeros((1, 0))
        xa = np.asarray(x.data if isinstance(x, Tensor) else x, dtype=np.float64)
        if xa.ndim != 2 or xa.shape[1] < self.n_features:
            raise ValueError(f"x must be (batch, >= {self.n_features}), got {xa.shape}")
        return xa

    def _angles(self, params: np.ndarray, xa: np.ndarray) -> np.ndarray:
        """``(n_ops, batch)`` angles; rows of angle-free gates stay zero and are unused."""
        p = np.asarray(params, dtype=np.float64).reshape(-1)
        if p.size != self.n_params:
            raise ValueError(f"circuit has {self.n_params} params, got {p.size}")
        ang = np.zeros((len(self.ops), xa.shape[0]))
        for i, op in enumerate(self.ops):
            if op.param is not None:
                ang[i] += p[op.param]
            if op.feature is not None:
                ang[i] += xa[:, op.feature]
        return ang

    def _simulate(self, angles: np.ndarray) -> np.ndarray:
        """Run the circuit once per column of ``angles``; returns ``(columns,)``."""
        psi = zero_state(self.n_qubits, batch=angles.shape[1])
        for i, op in enumerate(self.ops):
            gate = ROTATIONS[op.name](angles[i]) if op.name in ROTATIONS else FIXED_GATES[op.name]
            psi = apply_gate(psi, gate, op.wires, batched=True)
        return expval_z(psi, self.observable, batched=True)

    def forward(self, params, x=None) -> np.ndarray:
        """Expectations as a float64 array: ``(batch,)``, or a scalar when ``x`` is None."""
        xa = self._inputs(x)
        e = self._simulate(self._angles(params, xa))
        return e if x is not None else e[0]

    def jacobian(self, params, x=None) -> np.ndarray:
        """``dE_b / dparams_j`` by the parameter-shift rule: ``(batch, n_params)``.

        ``(n_params,)`` when ``x`` is None, mirroring :meth:`forward`.

        Row ``2k`` / ``2k+1`` of the stacked batch shifts occurrence ``k`` by
        ``+pi/2`` / ``-pi/2``; the per-occurrence derivatives are then summed into
        their parameter with ``np.add.at`` (plain fancy-index ``+=`` would keep
        only the last occurrence of a repeated index).
        """
        xa = self._inputs(x)
        base = self._angles(params, xa)
        n_ops, batch = base.shape
        m = len(self._shifted)
        jac = np.zeros((self.n_params, batch))
        if m:
            ang = np.repeat(base[:, None, :], 2 * m, axis=1)  # (n_ops, 2m, batch)
            k = np.arange(m)
            ang[self._shifted, 2 * k, :] += _SHIFT
            ang[self._shifted, 2 * k + 1, :] -= _SHIFT
            e = self._simulate(ang.reshape(n_ops, 2 * m * batch)).reshape(m, 2, batch)
            np.add.at(jac, self._shifted_param, 0.5 * (e[:, 0] - e[:, 1]))
        return jac.T if x is not None else jac[:, 0]

    # -- autograd --------------------------------------------------------------
    def expval(self, params: Tensor, x=None) -> Tensor:
        """Differentiable expectation: ``(batch,)`` Tensor, or 0-d when ``x`` is None.

        The result has ``params``' dtype; the simulation itself always runs in
        complex128. The backward runs the ``2m`` shifted circuits lazily, only
        if ``params`` requires grad and ``.backward()`` is actually called.
        """
        if isinstance(x, Tensor) and x.requires_grad:
            raise ValueError("expval treats x as a constant; it would get no gradient")
        p = np.asarray(params.data, dtype=np.float64).reshape(-1)
        xa = self._inputs(x)
        e = self._simulate(self._angles(p, xa))
        out = (e if x is not None else e[0]).astype(params.data.dtype)

        def grad_params(g: np.ndarray) -> np.ndarray:
            jac = self.jacobian(p, xa)
            return (np.reshape(g, (-1,)).astype(np.float64) @ jac).reshape(params.shape)

        return Tensor._make(np.asarray(out), (params,), (grad_params,), "expval")
