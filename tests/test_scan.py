"""Tests for :mod:`scratchgrad.nn.scan`.

The scan is checked three ways: against a plain sequential loop (forward, to
~1e-15 in float64), against central differences (gradcheck), and against the
autograd of the same recurrence unrolled one ``Tensor`` op per step -- the last
pins the reverse-time adjoint to the exact shift, not merely to "a gradient".
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import functional as F
from scratchgrad.nn.scan import (
    MinGRU,
    SelectiveSSM,
    _mingru_scan,
    _ssm_readout,
    linear_scan,
    selective_scan,
)
from scratchgrad.tensor import Tensor, concatenate, stack
from scratchgrad.testing import TOL_COMPOSED, TOL_OP, check_module, gradcheck, rel_error

N, T, D = 7, 13, 5


def _f64(arr) -> Tensor:
    return Tensor(np.asarray(arr), dtype=np.float64)


def _leaf(arr) -> Tensor:
    return Tensor(np.asarray(arr, dtype=np.float64), requires_grad=True, dtype=np.float64)


def _loop(a, b, h0=None):
    """The reference: the recurrence exactly as written, one step at a time."""
    h = np.zeros_like(b[:, 0]) if h0 is None else h0
    out = np.empty_like(b)
    for t in range(b.shape[1]):
        h = a[:, t] * h + b[:, t]
        out[:, t] = h
    return out


# --------------------------------------------------------------------------- #
# linear_scan: forward
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("steps", [1, 2, 3, 7, 13, 64, 257])
@pytest.mark.parametrize("trailing", [(5,), (3, 2)])
@pytest.mark.parametrize("with_h0", [False, True])
def test_scan_matches_sequential_loop(steps, trailing, with_h0):
    rng = np.random.default_rng(steps)
    shape = (N, steps, *trailing)
    a = rng.uniform(-1.0, 1.0, shape)
    b = rng.standard_normal(shape)
    h0 = rng.standard_normal((N, *trailing)) if with_h0 else None
    b_before = b.copy()
    got = linear_scan(_f64(a), _f64(b), None if h0 is None else _f64(h0)).data
    assert np.array_equal(b, b_before), "folding h0 into b_1 must not write to the caller's b"
    assert rel_error(got, _loop(a, b, h0)) < 1e-14


def test_scan_long_sequence_where_the_closed_form_is_nan():
    """T=4096: cumprod(a) underflows, so the textbook closed form is NaN; the scan is exact."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0.3, 0.7, (3, 4096, 4))
    b = rng.standard_normal((3, 4096, 4))
    got = linear_scan(_f64(a), _f64(b)).data
    assert np.all(np.isfinite(got))
    assert rel_error(got, _loop(a, b)) < 1e-12
    with np.errstate(all="ignore"):
        cum = np.cumprod(a, axis=1)
        closed = cum * np.cumsum(b / cum, axis=1)
    assert not np.all(np.isfinite(closed)), "the trap this scan exists to avoid has vanished?"


def test_scan_keeps_float32():
    rng = np.random.default_rng(1)
    a = Tensor(rng.uniform(0, 1, (N, T, D)).astype(np.float32))
    b = Tensor(rng.standard_normal((N, T, D)).astype(np.float32))
    assert linear_scan(a, b).dtype == np.float32


def test_scan_rejects_bad_shapes():
    a = _f64(np.zeros((N, T, D)))
    with pytest.raises(ValueError):
        linear_scan(a, Tensor(np.zeros((N, T, D + 1))))
    with pytest.raises(ValueError):
        linear_scan(a, Tensor(np.zeros((N, T, D))), Tensor(np.zeros((N, T, D))))
    with pytest.raises(ValueError):
        linear_scan(Tensor(np.zeros((N, 0, D))), Tensor(np.zeros((N, 0, D))))


# --------------------------------------------------------------------------- #
# linear_scan: backward
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("with_h0", [False, True])
def test_linear_scan_gradcheck(with_h0):
    rng = np.random.default_rng(2)
    ins = [_leaf(rng.uniform(-1, 1, (N, T, D))), _leaf(rng.standard_normal((N, T, D)))]
    if with_h0:
        ins.append(_leaf(rng.standard_normal((N, D))))
    gradcheck(lambda *ts: linear_scan(*ts), *ins, tol=TOL_OP, name=f"linear_scan[h0={with_h0}]")


def test_linear_scan_backward_equals_unrolled_autograd():
    """The reverse scan must match autograd through T separate multiply-add nodes."""
    rng = np.random.default_rng(3)
    steps = 37
    a_np = rng.uniform(-1, 1, (N, steps, D))
    b_np = rng.standard_normal((N, steps, D))
    h0_np = rng.standard_normal((N, D))
    cot = rng.standard_normal((N, steps, D))

    grads = []
    for fused in (True, False):
        a, b, h0 = _leaf(a_np), _leaf(b_np), _leaf(h0_np)
        if fused:
            out = linear_scan(a, b, h0)
        else:
            h, hs = h0, []
            for t in range(steps):
                h = a[:, t] * h + b[:, t]
                hs.append(h)
            out = stack(hs, axis=1)
        (out * _f64(cot)).sum().backward()
        grads.append((a.grad, b.grad, h0.grad))
    for g_fused, g_loop in zip(*grads, strict=True):
        assert rel_error(g_fused, g_loop) < 1e-13


# --------------------------------------------------------------------------- #
# selective_scan
# --------------------------------------------------------------------------- #
def _ssm_inputs(rng, n=3, steps=7, d=5, s=2):
    x = rng.standard_normal((n, steps, d))
    delta = rng.uniform(0.1, 1.0, (n, steps, d))
    A = -rng.uniform(0.5, 2.0, (d, s))
    B = rng.standard_normal((n, steps, s))
    h0 = rng.standard_normal((n, d, s))
    return x, delta, A, B, h0


def test_selective_scan_matches_loop():
    rng = np.random.default_rng(4)
    x, delta, A, B, h0 = _ssm_inputs(rng, n=4, steps=19, d=6, s=3)
    h = h0.copy()
    want = np.empty((4, 19, 6, 3))
    for t in range(19):
        dt = delta[:, t, :, None]
        h = np.exp(dt * A) * h + dt * B[:, t, None, :] * x[:, t, :, None]
        want[:, t] = h
    got = selective_scan(*(_f64(v) for v in (x, delta, A, B, h0))).data
    assert rel_error(got, want) < 1e-14


@pytest.mark.parametrize("with_h0", [False, True])
def test_selective_scan_gradcheck(with_h0):
    rng = np.random.default_rng(5)
    x, delta, A, B, h0 = _ssm_inputs(rng)
    ins = [_leaf(v) for v in (x, delta, A, B)] + ([_leaf(h0)] if with_h0 else [])
    gradcheck(
        lambda *ts: selective_scan(*ts), *ins, tol=TOL_OP, name=f"selective_scan[h0={with_h0}]"
    )


def test_ssm_readout_gradcheck():
    rng = np.random.default_rng(15)
    h = _leaf(rng.standard_normal((3, 7, D, 2)))
    C = _leaf(rng.standard_normal((3, 7, 2)))
    got = _ssm_readout(h, C).data
    assert rel_error(got, (h.data * C.data[:, :, None, :]).sum(-1)) < 1e-15
    gradcheck(lambda *ts: _ssm_readout(*ts), h, C, tol=TOL_OP, name="_ssm_readout")


# --------------------------------------------------------------------------- #
# MinGRU
# --------------------------------------------------------------------------- #
I_IN, H = 11, 5


@pytest.mark.parametrize("with_h0", [False, True])
def test_mingru_scan_op_gradcheck(with_h0):
    """The fused gate+scan node on its own, at the single-op tolerance."""
    rng = np.random.default_rng(14)
    ins = [_leaf(rng.standard_normal((N, T, 2 * H)))]
    if with_h0:
        ins.append(_leaf(rng.standard_normal((N, H))))
    gradcheck(lambda *ts: _mingru_scan(*ts), *ins, tol=TOL_OP, name=f"_mingru_scan[h0={with_h0}]")


def test_mingru_matches_stepwise_recurrence():
    rng = np.random.default_rng(6)
    gru = MinGRU(I_IN, H, rng=rng).double()
    gru.bias.data[...] = rng.standard_normal(2 * H)
    x = rng.standard_normal((N, T, I_IN))
    h0 = rng.standard_normal((N, H))
    w, bias = gru.weight.data, gru.bias.data
    h = h0
    want = np.empty((N, T, H))
    for t in range(T):
        pre = x[:, t] @ w + bias
        z = 1.0 / (1.0 + np.exp(-pre[:, :H]))
        h = (1.0 - z) * h + z * pre[:, H:]
        want[:, t] = h
    out, h_n = gru(_f64(x), _f64(h0))
    assert rel_error(out.data, want) < 1e-14
    assert np.array_equal(h_n.data, out.data[:, -1])


def test_mingru_polarity_z_gates_the_candidate():
    """z -> 1 writes the candidate; z -> 0 keeps the old state. Opposite to GRUCell."""
    rng = np.random.default_rng(7)
    gru = MinGRU(I_IN, H, rng=rng).double()
    x = _f64(rng.standard_normal((N, 1, I_IN)))
    h0 = _f64(rng.standard_normal((N, H)))
    cand = x.data[:, 0] @ gru.weight.data[:, H:]
    gru.weight.data[:, :H] = 0.0
    gru.bias.data[:H] = 40.0
    assert rel_error(gru(x, h0)[1].data, cand) < 1e-15
    gru.bias.data[:H] = -40.0
    assert rel_error(gru(x, h0)[1].data, h0.data) < 1e-15


def test_mingru_gradcheck():
    check_module(
        MinGRU(I_IN, H, rng=np.random.default_rng(8)),
        (N, T, I_IN),
        tol=TOL_COMPOSED,
        forward=lambda m, x: m(x)[0],
        name="MinGRU",
    )


def test_mingru_gradcheck_through_initial_state():
    rng = np.random.default_rng(9)
    gru = MinGRU(I_IN, H, rng=rng).double()
    x = _leaf(rng.standard_normal((N, T, I_IN)))
    h0 = _leaf(rng.standard_normal((N, H)))
    gradcheck(
        lambda *ts: gru(ts[0], ts[1])[0],
        x,
        h0,
        *gru.parameters(),
        tol=TOL_COMPOSED,
        copy_inputs=False,
        name="MinGRU(h0)",
    )


@pytest.mark.parametrize("layer", ["mingru", "ssm"])
def test_chunked_run_equals_full_run(layer):
    """Carrying h_n into the next chunk is exactly the full-sequence run (truncated BPTT)."""
    rng = np.random.default_rng(10)
    mod = (MinGRU(D, D, rng=rng) if layer == "mingru" else SelectiveSSM(D, 3, rng=rng)).double()
    x = rng.standard_normal((N, T, D))
    full, h_full = mod(_f64(x))
    first, h_mid = mod(_f64(x[:, :6]))
    second, h_end = mod(_f64(x[:, 6:]), h_mid)
    assert rel_error(np.concatenate([first.data, second.data], axis=1), full.data) < 1e-14
    assert rel_error(h_end.data, h_full.data) < 1e-14


# --------------------------------------------------------------------------- #
# SelectiveSSM
# --------------------------------------------------------------------------- #
def test_selective_ssm_parameterisation():
    ssm = SelectiveSSM(D, 4, dt_min=1e-3, dt_max=1e-1, rng=np.random.default_rng(11))
    assert np.all(-np.exp(ssm.A_log.data) < 0)
    dt = F.softplus(ssm.dt_proj.bias).data
    assert np.all((dt >= 1e-3 * (1 - 1e-5)) & (dt <= 1e-1 * (1 + 1e-5)))


def test_selective_ssm_gradcheck():
    def fwd(m, x):
        y, h_n = m(x)
        # h_n carries gradient too; check it through the same random cotangent.
        return concatenate([y.reshape(x.shape[0], -1), h_n.reshape(x.shape[0], -1)], axis=1)

    check_module(
        SelectiveSSM(D, 3, dt_min=0.05, dt_max=0.5, rng=np.random.default_rng(12)),
        (3, 7, D),
        tol=TOL_COMPOSED,
        forward=fwd,
        name="SelectiveSSM",
    )


def test_selective_ssm_gradcheck_through_initial_state():
    rng = np.random.default_rng(13)
    ssm = SelectiveSSM(D, 3, rng=rng).double()
    x = _leaf(rng.standard_normal((3, 7, D)))
    h0 = _leaf(rng.standard_normal((3, D, 3)))
    gradcheck(
        lambda *ts: ssm(ts[0], ts[1])[0],
        x,
        h0,
        *ssm.parameters(),
        tol=TOL_COMPOSED,
        copy_inputs=False,
        name="SelectiveSSM(h0)",
    )
