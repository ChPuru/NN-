"""Verify the claims in ``06-lstm.md``.

1. The gate-by-gate backward for ``i, f, g, o``, written out by hand, against
   ``nn.LSTMCell``'s fused node -- for every one of x, h, c, W_ih, W_hh and b.
2. The constant-error carousel: with the recurrent gate weights zeroed, the
   cell path is *exactly* ``dc_0 = (prod_t f_t) * dc_T``, so the framework's
   gradient can be compared against a closed-form product.
3. What the carousel is actually worth: over 40 steps, the cell path with the
   forget gate open, the same path with it half-closed, and a ``tanh`` RNN's
   hidden path, all on the same data. The LSTM does not solve vanishing
   gradients unconditionally and the middle number says so.
4. What truncation truncates: detaching the state at a chunk boundary gives
   *exactly* the gradient of the objective in which the boundary state is a
   constant -- and that is measurably not the full-BPTT gradient.

    PYTHONPATH=src python3 notes/derivations/verify/verify_06_lstm.py
"""

from __future__ import annotations

import numpy as np
from _harness import Check, cli

import scratchgrad as sg
from scratchgrad import nn
from scratchgrad.testing import check_module, rel_error

N, IN, HID = 7, 5, 3


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _cell(rng: np.random.Generator):
    """A float64 LSTMCell with known weights, plus one input and one state."""
    cell = nn.LSTMCell(IN, HID, rng=np.random.default_rng(1)).double()
    wi = rng.standard_normal((IN, 4 * HID)) * 0.4
    wh = rng.standard_normal((HID, 4 * HID)) * 0.4
    b = rng.standard_normal(4 * HID) * 0.2
    cell.weight_ih.data[...] = wi
    cell.weight_hh.data[...] = wh
    cell.bias.data[...] = b
    x = rng.standard_normal((N, IN))
    h = rng.standard_normal((N, HID))
    c = rng.standard_normal((N, HID))
    return cell, wi, wh, b, x, h, c


def _gate_backward(rng: np.random.Generator) -> dict[str, float]:
    """Hand-derived gate algebra vs the framework, parameter by parameter."""
    cell, wi, wh, b, x, h, c = _cell(rng)
    gh = rng.standard_normal((N, HID))  # cotangent on h'
    gc = rng.standard_normal((N, HID))  # cotangent on c'

    z = x @ wi + h @ wh + b
    i = _sigmoid(z[:, 0:HID])
    f = _sigmoid(z[:, HID : 2 * HID])
    g = np.tanh(z[:, 2 * HID : 3 * HID])
    o = _sigmoid(z[:, 3 * HID : 4 * HID])
    c_new = f * c + i * g
    tc = np.tanh(c_new)
    h_new = o * tc

    # Everything routes through dc: the cotangent that reaches the cell is the
    # one placed on c' PLUS the one that arrives through h' = o * tanh(c').
    dc = gc + gh * o * (1.0 - tc * tc)
    dz = np.concatenate(
        [
            (dc * g) * i * (1.0 - i),  # i is a sigmoid gate on the write
            (dc * c) * f * (1.0 - f),  # f multiplies the OLD cell
            (dc * i) * (1.0 - g * g),  # g is a tanh candidate
            (gh * tc) * o * (1.0 - o),  # o only ever sees h', never c'
        ],
        axis=1,
    )

    tx = sg.Tensor(x, requires_grad=True, dtype=np.float64)
    th = sg.Tensor(h, requires_grad=True, dtype=np.float64)
    tc_in = sg.Tensor(c, requires_grad=True, dtype=np.float64)
    h_out, c_out = cell(tx, (th, tc_in))
    fwd = max(rel_error(h_out.data, h_new), rel_error(c_out.data, c_new))
    (
        (h_out * sg.Tensor(gh, dtype=np.float64)).sum()
        + (c_out * sg.Tensor(gc, dtype=np.float64)).sum()
    ).backward()
    return {
        "forward": fwd,
        "dx": rel_error(tx.grad, dz @ wi.T),
        "dh": rel_error(th.grad, dz @ wh.T),
        "dc": rel_error(tc_in.grad, dc * f),
        "dW_ih": rel_error(cell.weight_ih.grad, x.T @ dz),
        "dW_hh": rel_error(cell.weight_hh.grad, h.T @ dz),
        "db": rel_error(cell.bias.grad, dz.sum(axis=0)),
    }


def _carousel(rng: np.random.Generator, steps: int = 12) -> float:
    """With W_hh = 0 the gates are input-driven, so dc_0 = (prod_t f_t) * dc_T.

    Zeroing the recurrent weights is what makes the statement exact rather than
    approximate: h no longer feeds the gates, so the ONLY path from c_T back to
    c_0 is the multiplicative one, and the product of forget gates is the whole
    Jacobian.
    """
    cell = nn.LSTMCell(IN, HID, rng=np.random.default_rng(2)).double()
    cell.weight_hh.data[...] = 0.0
    cell.weight_ih.data[...] = rng.standard_normal((IN, 4 * HID)) * 0.5
    cell.bias.data[...] = rng.standard_normal(4 * HID) * 0.3
    xs = [rng.standard_normal((N, IN)) for _ in range(steps)]

    wi, b = cell.weight_ih.data, cell.bias.data
    prod = np.ones((N, HID))
    for xt in xs:
        z = xt @ wi + b
        prod = prod * _sigmoid(z[:, HID : 2 * HID])

    c0 = sg.Tensor(rng.standard_normal((N, HID)), requires_grad=True, dtype=np.float64)
    h, c = sg.Tensor(np.zeros((N, HID)), dtype=np.float64), c0
    for xt in xs:
        h, c = cell(sg.Tensor(xt, dtype=np.float64), (h, c))
    v = rng.standard_normal((N, HID))
    (c * sg.Tensor(v, dtype=np.float64)).sum().backward()
    return rel_error(c0.grad, v * prod)


def _decay(rng: np.random.Generator, steps: int = 40) -> dict[str, float]:
    """Gradient magnitude surviving 40 steps: LSTM cell path vs a tanh RNN.

    The honest version of "the LSTM solves vanishing gradients". It does not,
    unconditionally. The cell Jacobian is ``diag(prod_t f_t)`` -- a product of
    numbers the network *chooses*, with no weight matrix and no ``tanh'`` factor
    in it -- so with the forget gate held open the gradient is preserved and
    with it half-closed the gradient still dies. Both are measured here, at
    ``b_f = 5`` (sigmoid 0.993) and ``b_f = 0`` (sigmoid 0.5), against a tanh
    RNN whose Jacobian is ``prod_t W_hh^T diag(1 - h^2)`` and is not a choice at
    all. ``W_hh`` is zeroed in the LSTMs so the forget gate is set purely by its
    bias and the comparison is controlled.
    """
    xs = [rng.standard_normal((N, IN)) * 0.5 for _ in range(steps)]
    v = rng.standard_normal((N, HID))
    ref = float(np.linalg.norm(v))
    out: dict[str, float] = {}

    for tag, bf in (("open", 5.0), ("half", 0.0)):
        cell = nn.LSTMCell(IN, HID, rng=np.random.default_rng(3)).double()
        cell.weight_hh.data[...] = 0.0
        cell.weight_ih.data[...] *= 0.1  # keep the gates near their bias values
        cell.bias.data[...] = 0.0
        cell.bias.data[HID : 2 * HID] = bf
        c0 = sg.Tensor(rng.standard_normal((N, HID)) * 0.1, requires_grad=True, dtype=np.float64)
        h, c = sg.Tensor(np.zeros((N, HID)), dtype=np.float64), c0
        for xt in xs:
            h, c = cell(sg.Tensor(xt, dtype=np.float64), (h, c))
        (c * sg.Tensor(v, dtype=np.float64)).sum().backward()
        out[tag] = float(np.linalg.norm(c0.grad)) / ref

    rnn = nn.RNNCell(IN, HID, rng=np.random.default_rng(3)).double()
    h0 = sg.Tensor(rng.standard_normal((N, HID)) * 0.1, requires_grad=True, dtype=np.float64)
    hh = h0
    for xt in xs:
        hh = rnn(sg.Tensor(xt, dtype=np.float64), hh)
    (hh * sg.Tensor(v, dtype=np.float64)).sum().backward()
    out["rnn"] = float(np.linalg.norm(h0.grad)) / ref
    return out


def _truncation(rng: np.random.Generator, steps: int = 8, cut: int = 4) -> tuple[float, float]:
    """Truncated BPTT == the exact gradient of the truncated objective.

    Two runs. In the first the state is detached at ``cut``; in the second the
    boundary state is fed in as a fresh leaf with the same numbers, so the graph
    genuinely stops there by construction. The two dW must agree bit for bit --
    that is what ``detach`` means. Neither equals the full-BPTT gradient.
    """
    cell = nn.LSTMCell(IN, HID, rng=np.random.default_rng(4)).double()
    xs = [sg.Tensor(rng.standard_normal((N, IN)), dtype=np.float64) for _ in range(steps)]
    v = sg.Tensor(rng.standard_normal((N, HID)), dtype=np.float64)

    def rollout(detach_at: int | None) -> np.ndarray:
        cell.zero_grad()
        h = sg.Tensor(np.zeros((N, HID)), dtype=np.float64)
        c = sg.Tensor(np.zeros((N, HID)), dtype=np.float64)
        for t, xt in enumerate(xs):
            if detach_at is not None and t == detach_at:
                h, c = nn.detach_state((h, c))
            h, c = cell(xt, (h, c))
        (h * v).sum().backward()
        return np.array(cell.weight_hh.grad, copy=True)

    full = rollout(None)
    trunc = rollout(cut)

    # The same objective, but the boundary state is a constant by construction.
    cell.zero_grad()
    h = sg.Tensor(np.zeros((N, HID)), dtype=np.float64)
    c = sg.Tensor(np.zeros((N, HID)), dtype=np.float64)
    with sg.no_grad():
        for xt in xs[:cut]:
            h, c = cell(xt, (h, c))
    h = sg.Tensor(h.data, dtype=np.float64)
    c = sg.Tensor(c.data, dtype=np.float64)
    for xt in xs[cut:]:
        h, c = cell(xt, (h, c))
    (h * v).sum().backward()
    rebuilt = np.array(cell.weight_hh.grad, copy=True)
    return rel_error(trunc, rebuilt), rel_error(trunc, full)


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 06-lstm.md."""
    rng = np.random.default_rng(seed)
    g = _gate_backward(rng)
    dec = _decay(rng)
    trunc_exact, trunc_vs_full = _truncation(rng)
    data = np.random.default_rng(4242).standard_normal((N, 6, IN))
    mod_errs = check_module(
        nn.LSTM(IN, HID, rng=np.random.default_rng(5)),
        (N, 6, IN),
        tol=1e-7,
        input_data=data,
        name="LSTM",
        forward=lambda m, x: m(x)[0],
    )
    return [
        Check("forward_values", g["forward"], 1e-15, note="c' = f*c + i*g, h' = o*tanh(c')"),
        Check("dx", g["dx"], 1e-14, note="dz @ W_ih^T"),
        Check("dh_prev", g["dh"], 1e-14, note="dz @ W_hh^T"),
        Check("dc_prev", g["dc"], 1e-15, note="dc * f -- the carousel, one step"),
        Check("dW_ih", g["dW_ih"], 1e-14, note="x^T dz"),
        Check("dW_hh", g["dW_hh"], 1e-14, note="h^T dz"),
        Check("db", g["db"], 1e-15, note="dz summed over the batch"),
        Check(
            "carousel_is_prod_f",
            _carousel(rng),
            1e-14,
            note="W_hh = 0, 12 steps: dc_0 == (prod_t f_t) * dc_12",
        ),
        Check(
            "carousel_open_gate_survives",
            dec["open"],
            0.5,
            kind="above",
            note="b_f=5, 40 steps: |dc_0| / |dc_40| -- the gradient is still there",
        ),
        Check(
            "carousel_half_gate_vanishes",
            dec["half"],
            1e-9,
            note=f"b_f=0, same 40 steps: {dec['half']:.2e}. The cell is not magic",
        ),
        Check(
            "tanh_rnn_vanishes",
            dec["rnn"],
            1e-2,
            note=f"and a tanh RNN gets {dec['rnn']:.2e} with no say in the matter",
        ),
        Check(
            "truncation_is_exact",
            trunc_exact,
            1e-15,
            note="detach at t=4 == rebuilding the graph from a constant state",
        ),
        Check(
            "truncation_is_not_full_bptt",
            trunc_vs_full,
            1e-3,
            kind="above",
            note="and the discarded 4 steps were carrying real gradient",
        ),
        Check(
            "finite_differences",
            max(mod_errs.values()),
            1e-7,
            note="check_module through 6 unrolled steps of nn.LSTM",
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "06 lstm"))
