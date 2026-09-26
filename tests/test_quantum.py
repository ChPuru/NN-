"""Tests for ``scratchgrad.quantum``: the statevector simulator and the parameter-shift op.

The simulator is checked against things that do not depend on it: known states
(Bell), a full ``2**n x 2**n`` matrix built index by index (the ordering
convention), unitarity, and closed forms (``<Z>`` after ``RY(t)`` is ``cos t``).
The gradient is checked three ways -- parameter shift, central differences of
the forward pass, and analytic derivatives -- because agreeing with finite
differences alone could not tell an exact rule from a good approximation.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad import optim
from scratchgrad._config import default_dtype
from scratchgrad.nn.module import Parameter
from scratchgrad.quantum import (
    CNOT,
    CZ,
    FIXED_GATES,
    Circuit,
    H,
    Op,
    X,
    Y,
    Z,
    apply_gate,
    expval_z,
    rx,
    ry,
    rz,
    zero_state,
)
from scratchgrad.tensor import Tensor
from scratchgrad.testing.gradcheck import TOL_COMPOSED, gradcheck, rel_error

I2 = np.eye(2)


# --------------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------------- #
def test_bell_state():
    psi = apply_gate(apply_gate(zero_state(2), H, 0), CNOT, (0, 1))
    np.testing.assert_allclose(psi.reshape(-1), np.array([1, 0, 0, 1]) / np.sqrt(2), atol=1e-15)
    assert expval_z(psi, (0, 1)) == pytest.approx(1.0, abs=1e-14)
    assert expval_z(psi, 0) == pytest.approx(0.0, abs=1e-14)
    assert expval_z(psi, ()) == pytest.approx(1.0, abs=1e-14)


def test_wire_zero_is_the_most_significant_bit():
    """The documented convention, pinned: |q0 q1 q2> at index 4*q0 + 2*q1 + q2."""
    assert np.argmax(np.abs(apply_gate(zero_state(3), X, 0).reshape(-1))) == 4
    assert np.argmax(np.abs(apply_gate(zero_state(3), X, 2).reshape(-1))) == 1
    # CNOT's control is the FIRST wire passed, whichever wire that is.
    s01 = apply_gate(zero_state(2), X, 1)  # |01>
    assert np.argmax(np.abs(apply_gate(s01, CNOT, (1, 0)).reshape(-1))) == 3  # -> |11>
    assert np.argmax(np.abs(apply_gate(s01, CNOT, (0, 1)).reshape(-1))) == 1  # unchanged


def _full_matrix(gate: np.ndarray, wires: tuple[int, ...], n: int) -> np.ndarray:
    """The 2**n x 2**n operator, built one matrix element at a time -- no reshapes."""
    k, dim = len(wires), 2**n
    bits = lambda i: [(i >> (n - 1 - w)) & 1 for w in range(n)]  # noqa: E731
    full = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            bi, bj = bits(i), bits(j)
            if any(bi[w] != bj[w] for w in range(n) if w not in wires):
                continue
            r = sum(bi[w] << (k - 1 - t) for t, w in enumerate(wires))
            c = sum(bj[w] << (k - 1 - t) for t, w in enumerate(wires))
            full[i, j] = gate[r, c]
    return full


@pytest.mark.parametrize("wires", [(0,), (2,), (0, 2), (2, 0), (1, 2), (2, 1)])
def test_apply_gate_matches_explicit_operator(wires):
    rng = np.random.default_rng(3)
    n, k = 3, len(wires)
    gate = rng.standard_normal((2**k, 2**k)) + 1j * rng.standard_normal((2**k, 2**k))
    psi = rng.standard_normal((2,) * n) + 1j * rng.standard_normal((2,) * n)
    got = apply_gate(psi, gate, wires).reshape(-1)
    want = _full_matrix(gate, wires, n) @ psi.reshape(-1)
    np.testing.assert_allclose(got, want, atol=1e-13)


def test_gates_are_unitary_and_rotations_use_the_half_angle():
    rng = np.random.default_rng(0)
    for g in FIXED_GATES.values():
        np.testing.assert_allclose(g.conj().T @ g, np.eye(len(g)), atol=1e-15)
    t = rng.uniform(-7, 7, size=5)
    for fn, p in ((rx, X), (ry, Y), (rz, Z)):
        u = fn(t)
        np.testing.assert_allclose(
            u.conj().transpose(0, 2, 1) @ u, np.broadcast_to(I2, u.shape), atol=1e-14
        )
        want = np.cos(t / 2)[:, None, None] * I2 - 1j * np.sin(t / 2)[:, None, None] * p
        np.testing.assert_allclose(u, want, atol=1e-15)


def test_gate_constants_are_read_only():
    with pytest.raises(ValueError):
        H[0, 0] = 0.0


def test_norm_preserved_and_batch_matches_single():
    """A long random circuit, batched with per-sample angles, against one-at-a-time."""
    rng = np.random.default_rng(1)
    n, batch = 4, 5
    psi_b = zero_state(n, batch=batch)
    singles = [zero_state(n) for _ in range(batch)]
    for _ in range(60):
        if rng.random() < 0.5:
            fn, w = (rx, ry, rz)[rng.integers(3)], int(rng.integers(n))
            angles = rng.uniform(0, 2 * np.pi, batch)
            psi_b = apply_gate(psi_b, fn(angles), w, batched=True)
            singles = [apply_gate(s, fn(a), w) for s, a in zip(singles, angles, strict=True)]
        else:
            g = (H, CNOT, CZ)[rng.integers(3)]
            w = tuple(int(i) for i in rng.choice(n, size=int(np.log2(len(g))), replace=False))
            psi_b = apply_gate(psi_b, g, w, batched=True)
            singles = [apply_gate(s, g, w) for s in singles]
    np.testing.assert_allclose(expval_z(psi_b, (), batched=True), 1.0, atol=1e-13)
    np.testing.assert_allclose(psi_b, np.stack(singles), atol=1e-14)
    np.testing.assert_allclose(
        expval_z(psi_b, (0, 3), batched=True), [expval_z(s, (0, 3)) for s in singles], atol=1e-14
    )


def test_expval_after_ry_is_cos():
    t = np.linspace(-np.pi, 2 * np.pi, 13)
    psi = apply_gate(zero_state(1, batch=len(t)), ry(t), 0, batched=True)
    np.testing.assert_allclose(expval_z(psi, 0, batched=True), np.cos(t), atol=1e-14)
    # RX gives the same <Z>; RZ only adds phases and leaves it at 1.
    psi = apply_gate(zero_state(1, batch=len(t)), rx(t), 0, batched=True)
    np.testing.assert_allclose(expval_z(psi, 0, batched=True), np.cos(t), atol=1e-14)
    psi = apply_gate(zero_state(1, batch=len(t)), rz(t), 0, batched=True)
    np.testing.assert_allclose(expval_z(psi, 0, batched=True), 1.0, atol=1e-14)


def test_bad_wires_and_shapes_raise():
    with pytest.raises(ValueError):
        apply_gate(zero_state(2), CNOT, (0, 0))
    with pytest.raises(ValueError):
        apply_gate(zero_state(2), H, 2)
    with pytest.raises(ValueError):
        apply_gate(zero_state(2), CNOT, 0)
    with pytest.raises(ValueError):  # a stack of gates without batched=True
        apply_gate(zero_state(2), ry(np.zeros(2)), 0)


# --------------------------------------------------------------------------- #
# circuits and the parameter-shift rule
# --------------------------------------------------------------------------- #
def _random_circuit(rng: np.random.Generator) -> Circuit:
    """3 qubits, every gate type, 4 params each used 2-3 times, 2 re-uploaded features."""
    ops = [Op("H", 1), Op("RY", 0, feature=0), Op("RX", 2, feature=1)]
    for layer in range(3):
        for q in range(3):
            name = ("RX", "RY", "RZ")[(q + layer) % 3]
            ops.append(Op(name, q, param=int(rng.integers(4)), feature=q % 2 if layer else None))
        ops += [Op("CNOT", (layer % 3, (layer + 1) % 3)), Op("CZ", (0, 2)), Op("X", 1), Op("Y", 2)]
    ops.append(Op("RY", 1, param=3))
    return Circuit(3, ops, observable=(0, 2))


def _central_differences(circ: Circuit, p: np.ndarray, x, h: float = 1e-6) -> np.ndarray:
    cols = []
    for j in range(p.size):
        e = np.zeros_like(p)
        e[j] = h
        cols.append((circ.forward(p + e, x) - circ.forward(p - e, x)) / (2 * h))
    return np.stack(cols, axis=-1)


def test_parameter_shift_matches_analytic():
    """RY(a) q0, RY(b) q1, CNOT(0,1): <Z_1> = cos a cos b."""
    circ = Circuit(2, [Op("RY", 0, param=0), Op("RY", 1, param=1), Op("CNOT", (0, 1))], (1,))
    for a, b in [(0.3, -1.1), (2.0, 0.7), (np.pi / 2, np.pi)]:
        assert circ.forward([a, b]) == pytest.approx(np.cos(a) * np.cos(b), abs=1e-14)
        want = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        np.testing.assert_allclose(circ.jacobian([a, b]), want, atol=1e-14)


def test_shared_parameter_sums_every_occurrence():
    """RY(t) RY(t): E = cos 2t. Shifting t itself (both gates at once) gives exactly 0."""
    circ = Circuit(1, [Op("RY", 0, param=0), Op("RY", 0, param=0)])
    t = np.linspace(-3, 3, 7)
    for ti in t:
        assert circ.jacobian([ti])[0] == pytest.approx(-2 * np.sin(2 * ti), abs=1e-14)
        naive = (circ.forward([ti + np.pi / 2]) - circ.forward([ti - np.pi / 2])) / 2
        assert naive == pytest.approx(0.0, abs=1e-14)


def test_parameter_shift_matches_central_differences():
    rng = np.random.default_rng(7)
    circ = _random_circuit(rng)
    assert circ.n_params == 4 and circ.n_features == 2
    p = rng.uniform(-np.pi, np.pi, circ.n_params)
    x = rng.uniform(-np.pi, np.pi, size=(5, 2))
    shift = circ.jacobian(p, x)
    fd = _central_differences(circ, p, x)
    assert shift.shape == (5, 4)
    assert rel_error(shift, fd) < 1e-8
    assert np.abs(shift).max() > 0.1  # not vacuously zero


def test_expval_backward_gradchecks():
    rng = np.random.default_rng(11)
    circ = _random_circuit(rng)
    x = rng.uniform(-np.pi, np.pi, size=(7, 2))
    theta = Tensor(rng.uniform(-np.pi, np.pi, circ.n_params))
    gradcheck(lambda p: circ.expval(p, x), theta, tol=TOL_COMPOSED)
    # ... and composed inside an ordinary loss.
    y = np.sign(rng.standard_normal(7))
    gradcheck(lambda p: ((circ.expval(p, x) - Tensor(y)) ** 2).mean(), theta, tol=TOL_COMPOSED)


def test_expval_shapes_and_dtype():
    circ = Circuit(1, [Op("RY", 0, param=0)])
    with default_dtype(np.float64):
        t = Tensor([0.4], requires_grad=True)
        e = circ.expval(t)
        assert e.shape == () and e.data.dtype == np.float64
        e.backward()
        np.testing.assert_allclose(t.grad, [-np.sin(0.4)], atol=1e-14)
    t32 = Tensor(np.array([0.4], dtype=np.float32), dtype=np.float32, requires_grad=True)
    assert circ.expval(t32).data.dtype == np.float32
    assert circ.jacobian([0.4]).shape == (1,)


def test_invalid_circuits_and_inputs_raise():
    with pytest.raises(ValueError):
        Op("RY", 0)  # no angle source
    with pytest.raises(ValueError):
        Op("CNOT", (0,))
    with pytest.raises(ValueError):
        Op("H", 0, param=0)
    with pytest.raises(ValueError):
        Op("U3", 0, param=0)
    with pytest.raises(ValueError):
        Circuit(2, [Op("CZ", (0, 2))])
    circ = Circuit(1, [Op("RY", 0, param=0, feature=0)])
    with pytest.raises(ValueError):
        circ.forward([0.1])  # x missing
    with pytest.raises(ValueError):
        circ.forward([0.1, 0.2], np.zeros((3, 1)))  # wrong param count
    with pytest.raises(ValueError):
        circ.expval(Tensor([0.1], requires_grad=True), Tensor(np.zeros((3, 1)), requires_grad=True))


def test_adam_trains_through_the_op():
    """Minimise <Z> after RY(t): the optimum is t = pi, <Z> = -1."""
    circ = Circuit(1, [Op("RY", 0, param=0)])
    with default_dtype(np.float64):
        t = Parameter(np.array([0.3]))
        opt = optim.Adam([t], lr=0.1)
        for _ in range(200):
            loss = circ.expval(t)
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert circ.forward(t.data) < -0.999
