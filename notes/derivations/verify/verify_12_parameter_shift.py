"""Verify the claims in ``12-parameter-shift.md``.

1. With every other gate fixed, ``E(theta)`` for one ``exp(-i theta P/2)`` gate is
   exactly ``A + B cos(theta) + C sin(theta)``: a 3-term least-squares fit over
   40 angles leaves only rounding in the residual.
2. Hence ``[E(t+s) - E(t-s)] / (2 sin s)`` is exact for ANY shift ``s``, not just
   ``pi/2``; ``s = pi/2`` is merely the choice that maximises the denominator.
   The same two evaluations read as a central difference with ``h = pi/2`` are
   off by exactly the factor ``sin(s)/s = 2/pi``.
3. ``Circuit.jacobian`` agrees with the fitted derivative (exact) and with
   central differences of the forward pass (to their noise floor) on a circuit
   with shared parameters and data re-uploading.
4. Shared parameters: shifting ``t`` itself in ``RY(t) RY(t)`` returns exactly 0
   for every ``t``; summing per-occurrence shifts returns ``-2 sin 2t``.
5. Convention trap: for a full-angle gate ``exp(-i theta P)`` the ``pi/2`` rule
   returns exactly 0; the correct rule there is shift ``pi/4``, factor 1.
6. Scope trap: a controlled ``RY`` has generator eigenvalues ``{0, +-1/2}`` and
   the two-term rule is simply wrong for it.

    PYTHONPATH=src python3 notes/derivations/verify/verify_12_parameter_shift.py
"""

from __future__ import annotations

import numpy as np
from _harness import Check, cli

from scratchgrad.quantum import Circuit, Op, apply_gate, expval_z, ry, zero_state
from scratchgrad.testing.gradcheck import rel_error


def _random_unitary(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Haar-ish unitary from the QR of a complex Gaussian (phases fixed so Q is unique)."""
    a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _sandwich(rng: np.random.Generator):
    """E(theta) = <0| V^+ RY_1(theta)^+ W^+ Z0 Z1 W RY_1(theta) V |0> on 2 qubits."""
    v, w = _random_unitary(rng, 4), _random_unitary(rng, 4)

    def energy(theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_1d(theta)
        psi = apply_gate(zero_state(2, batch=theta.size), v, (0, 1), batched=True)
        psi = apply_gate(psi, ry(theta), 1, batched=True)
        psi = apply_gate(psi, w, (0, 1), batched=True)
        return expval_z(psi, (0, 1), batched=True)

    return energy


def _fit_sinusoid(energy) -> tuple[float, np.ndarray]:
    t = np.linspace(-np.pi, np.pi, 40)
    basis = np.stack([np.ones_like(t), np.cos(t), np.sin(t)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, energy(t), rcond=None)
    return float(np.max(np.abs(basis @ coef - energy(t)))), coef


def _mixed_circuit(rng: np.random.Generator) -> Circuit:
    ops = [Op("H", 0), Op("RY", 1, feature=0)]
    for layer in range(3):
        for q in range(3):
            name = ("RX", "RY", "RZ")[(q + layer) % 3]
            ops.append(Op(name, q, param=int(rng.integers(4)), feature=(q + layer) % 2))
        ops += [Op("CNOT", (layer % 3, (layer + 1) % 3)), Op("CZ", (0, 2))]
    ops.append(Op("RY", 2, param=3))
    return Circuit(3, ops, observable=(0, 1))


def _cry(theta: float) -> np.ndarray:
    """Controlled-RY, control = first wire: generator |1><1| (x) Y/2, eigenvalues {0, +-1/2}."""
    u = np.eye(4, dtype=np.complex128)
    u[2:, 2:] = ry(theta)
    return u


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 12-parameter-shift.md."""
    rng = np.random.default_rng(seed)
    energy = _sandwich(rng)
    resid, (_, b, c) = _fit_sinusoid(energy)
    t = rng.uniform(-np.pi, np.pi, 25)
    exact = -b * np.sin(t) + c * np.cos(t)

    def rule(s: float) -> np.ndarray:
        return (energy(t + s) - energy(t - s)) / (2 * np.sin(s))

    any_shift = max(float(np.max(np.abs(rule(s) - exact))) for s in (0.1, 1.0, np.pi / 2, 2.5))
    as_fd = (energy(t + np.pi / 2) - energy(t - np.pi / 2)) / np.pi  # central diff, h = pi/2
    fd_ratio = float(np.median(as_fd / rule(np.pi / 2)))

    circ = _mixed_circuit(rng)
    p = rng.uniform(-np.pi, np.pi, circ.n_params)
    x = rng.uniform(-np.pi, np.pi, size=(6, circ.n_features))
    jac = circ.jacobian(p, x)
    h = 1e-6
    fd = np.stack(
        [
            (circ.forward(p + h * e, x) - circ.forward(p - h * e, x)) / (2 * h)
            for e in np.eye(circ.n_params)
        ],
        axis=-1,
    )

    twice = Circuit(1, [Op("RY", 0, param=0), Op("RY", 0, param=0)])
    ts = np.linspace(-3, 3, 13)
    naive = np.array(
        [(twice.forward([u + np.pi / 2]) - twice.forward([u - np.pi / 2])) / 2 for u in ts]
    )
    summed = np.array([twice.jacobian([u])[0] for u in ts])

    # Full-angle gate exp(-i theta Y) == RY(2 theta): E_full(theta) = E(2 theta).
    full = lambda u: energy(2 * u)  # noqa: E731
    exact_full = 2 * (-b * np.sin(2 * t) + c * np.cos(2 * t))
    full_pi2 = (full(t + np.pi / 2) - full(t - np.pi / 2)) / 2
    full_pi4 = full(t + np.pi / 4) - full(t - np.pi / 4)

    v, w = _random_unitary(rng, 4), _random_unitary(rng, 4)

    def e_cry(theta: float) -> float:
        # W after the gate matters: with an observable diagonal in the control, the
        # control-0 and control-1 branches never interfere, the half-frequency terms
        # cancel, and the two-term rule happens to come out right.
        psi = apply_gate(apply_gate(zero_state(2), v, (0, 1)), _cry(theta), (0, 1))
        return expval_z(apply_gate(psi, w, (0, 1)), 1)

    th = rng.uniform(-np.pi, np.pi, 10)
    cry_rule = np.array([(e_cry(u + np.pi / 2) - e_cry(u - np.pi / 2)) / 2 for u in th])
    cry_fd = np.array([(e_cry(u + 1e-6) - e_cry(u - 1e-6)) / 2e-6 for u in th])

    return [
        Check(
            "single_gate_is_sinusoid",
            resid,
            1e-13,
            note="max residual of A + B cos t + C sin t fitted to E(t) at 40 angles",
        ),
        Check(
            "shift_rule_exact_any_s",
            any_shift,
            1e-13,
            note="[E(t+s)-E(t-s)]/(2 sin s) vs fitted derivative, s in {0.1, 1, pi/2, 2.5}",
        ),
        Check(
            "pi2_as_central_diff_ratio",
            abs(fd_ratio - 2 / np.pi),
            1e-12,
            note=f"same two evaluations read as FD with h=pi/2: x{fd_ratio:.6f} (= 2/pi)",
        ),
        Check(
            "framework_vs_central_diff",
            rel_error(jac, fd),
            1e-8,
            note="Circuit.jacobian vs central differences (h=1e-6), shared params + x",
        ),
        Check(
            "framework_nonvacuous",
            float(np.max(np.abs(jac))),
            0.1,
            kind="above",
            note="largest |dE/dtheta| in that check, so it cannot pass by being zero",
        ),
        Check(
            "shared_naive_is_zero",
            float(np.max(np.abs(naive))),
            1e-14,
            note="RY(t) RY(t): shifting t itself gives 0 for every t",
        ),
        Check(
            "shared_summed_exact",
            float(np.max(np.abs(summed + 2 * np.sin(2 * ts)))),
            1e-14,
            note="per-occurrence sum == -2 sin 2t",
        ),
        Check(
            "full_angle_pi2_is_zero",
            float(np.max(np.abs(full_pi2))),
            1e-14,
            note="exp(-i t Y): the pi/2 rule returns exactly 0",
        ),
        Check(
            "full_angle_pi4_exact",
            float(np.max(np.abs(full_pi4 - exact_full))),
            1e-13,
            note="exp(-i t Y): shift pi/4, factor 1 is exact",
        ),
        Check(
            "controlled_ry_two_term_fails",
            float(np.max(np.abs(cry_rule - cry_fd))),
            1e-2,
            kind="above",
            note="CRY: two-term pi/2 rule vs central differences",
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "12 parameter shift"))
