"""Verify the claims in ``01-backprop-basics.md``.

Four things are measured:

1. ``unbroadcast`` is the exact adjoint of ``broadcast_to`` -- the reason a
   forward broadcast becomes a backward sum.
2. A chain rule composed by hand as VJPs matches the engine's ``backward()``.
3. A tensor consumed twice accumulates (``+=``), it does not overwrite.
4. Reverse mode really is ``O(1)`` passes where forward mode is ``O(n)``: one
   ``backward()`` timed against ``n`` directional forward passes.

    PYTHONPATH=src python3 notes/derivations/verify/verify_01_backprop_basics.py
"""

from __future__ import annotations

import time

import numpy as np
from _harness import Check, cli

import scratchgrad as sg
from scratchgrad.tensor import unbroadcast
from scratchgrad.testing import rel_error


def _adjoint_of_broadcast(rng: np.random.Generator) -> float:
    """max over shape pairs of |<v, broadcast(u)> - <unbroadcast(v), u>| (relative)."""
    pairs = [
        ((11,), (7, 11)),  # a prepended axis: summed away entirely
        ((1, 11), (7, 11)),  # an extent-1 axis: summed with keepdims
        ((7, 1), (7, 11)),
        ((1, 1, 5), (7, 11, 5)),  # both kinds at once
        ((), (7, 11)),  # a scalar
    ]
    worst = 0.0
    for small, big in pairs:
        u = rng.standard_normal(small)
        v = rng.standard_normal(big)
        lhs = float(np.sum(v * np.broadcast_to(u, big)))
        rhs = float(np.sum(unbroadcast(v, small) * u))
        worst = max(worst, abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-30))
    return worst


def _hand_chain(rng: np.random.Generator) -> tuple[float, float]:
    """Hand-composed VJPs vs ``backward()``; returns (dx error, dW1 error)."""
    n, d, k, m = 7, 11, 5, 3
    x = rng.standard_normal((n, d))
    w1 = rng.standard_normal((d, k)) / np.sqrt(d)
    b1 = rng.standard_normal(k)
    w2 = rng.standard_normal((k, m)) / np.sqrt(k)
    v = rng.standard_normal((n, m))

    tx = sg.Tensor(x, requires_grad=True, dtype=np.float64)
    tw1 = sg.Tensor(w1, requires_grad=True, dtype=np.float64)
    tb1 = sg.Tensor(b1, requires_grad=True, dtype=np.float64)
    tw2 = sg.Tensor(w2, requires_grad=True, dtype=np.float64)
    out = (tx @ tw1 + tb1).tanh() @ tw2
    out.backward(v)

    # By hand, right to left. y = h @ W2 with h = tanh(z), z = x @ W1 + b1.
    z = x @ w1 + b1
    h = np.tanh(z)
    dh = v @ w2.T
    dz = dh * (1.0 - h * h)
    dx = dz @ w1.T
    dw1 = x.T @ dz
    return rel_error(tx.grad, dx), rel_error(tw1.grad, dw1)


def _accumulation(rng: np.random.Generator) -> float:
    """A leaf used by two branches must receive the SUM of both cotangents."""
    x = sg.Tensor(rng.standard_normal((7, 11)), requires_grad=True, dtype=np.float64)
    (x.sum() + (x * x).sum()).backward()
    expected = 1.0 + 2.0 * x.data
    return rel_error(x.grad, expected)


def _mode_cost(rng: np.random.Generator) -> tuple[float, int, float, float]:
    """Time one reverse pass against n forward directional derivatives.

    Forward mode needs one pass per *input* to fill a gradient; reverse mode
    needs one per *output*. With a scalar loss and n parameters that is the
    whole argument for reverse mode, and it is a measurable ratio, not a claim.
    """
    d, k = 40, 5
    n_params = d * k + k  # the count forward mode would have to sweep
    w = sg.Tensor(rng.standard_normal((d, k)) / np.sqrt(d), requires_grad=True, dtype=np.float64)
    b = sg.Tensor(rng.standard_normal(k), requires_grad=True, dtype=np.float64)
    x = np.asarray(rng.standard_normal((32, d)))
    target = rng.standard_normal((32, k))

    def loss_of(wd: np.ndarray, bd: np.ndarray) -> float:
        r = np.tanh(x @ wd + bd) - target
        return float((r * r).mean())

    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        w.grad = b.grad = None
        tx = sg.Tensor(x, dtype=np.float64)
        tt = sg.Tensor(target, dtype=np.float64)
        r = (tx @ w + b).tanh() - tt
        (r * r).mean().backward()
    t_rev = (time.perf_counter() - t0) / reps

    h = 1e-5  # the optimum for central differences; see 09-gradient-checking.md
    t0 = time.perf_counter()
    grad = np.zeros(n_params)
    flat = [w.data, b.data]
    idx = 0
    for arr in flat:
        for i in range(arr.size):
            orig = float(arr.flat[i])
            arr.flat[i] = orig + h
            plus = loss_of(w.data, b.data)
            arr.flat[i] = orig - h
            minus = loss_of(w.data, b.data)
            arr.flat[i] = orig
            grad[idx] = (plus - minus) / (2 * h)
            idx += 1
    t_fwd = time.perf_counter() - t0

    analytic = np.concatenate([w.grad.reshape(-1), b.grad.reshape(-1)])
    return t_fwd / t_rev, n_params, rel_error(analytic, grad), t_rev


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 01-backprop-basics.md."""
    rng = np.random.default_rng(seed)
    checks = [
        Check(
            "unbroadcast_adjoint",
            _adjoint_of_broadcast(rng),
            1e-15,
            note="<v, broadcast(u)> == <unbroadcast(v), u>, 5 shape pairs",
        ),
    ]
    dx_err, dw_err = _hand_chain(rng)
    checks += [
        Check("chain_dx", dx_err, 1e-14, note="hand-composed VJP vs backward(), d/dx"),
        Check("chain_dW1", dw_err, 1e-14, note="hand-composed VJP vs backward(), d/dW1"),
        Check(
            "accumulate_two_uses",
            _accumulation(rng),
            1e-15,
            note="a leaf in two branches gets 1 + 2x, not 2x",
        ),
    ]
    ratio, n_params, grad_err, t_rev = _mode_cost(rng)
    checks += [
        Check(
            "reverse_vs_forward_ratio",
            ratio,
            10.0,
            kind="above",
            note=f"n={n_params} directional passes / 1 backward ({t_rev * 1e3:.2f} ms)",
            extras={"n_params": n_params, "backward_ms": t_rev * 1e3},
        ),
        Check(
            "reverse_matches_forward",
            grad_err,
            1e-9,
            note="and the two agree, so the cheap one is not cheating",
        ),
    ]
    return checks


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "01 backprop basics"))
