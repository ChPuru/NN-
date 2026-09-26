"""Tests for ``scratchgrad.quant``.

Fake quantization is piecewise constant, so the usual receipt -- gradcheck
against central differences -- does not apply to it: finite differences of
``round`` are zero on a step and enormous across a jump, and neither is the
straight-through estimate. The STE is a *defined* rule, so it is tested as one:
the backward pass must equal the upstream gradient times the clamp mask,
exactly, and inside the range it must equal the true derivative of the
unrounded surrogate -- which central differences *can* certify. Everything that
is genuinely differentiable (a QATLinear's input path, the surrogate) is
gradchecked in float64 as usual.

The integer path is checked against exact int64 arithmetic and against the
float fake-quant simulation it is meant to reproduce.
"""

from __future__ import annotations

import numpy as np
import pytest

import scratchgrad as sg
from scratchgrad import nn
from scratchgrad.quant import (
    FakeQuantize,
    MinMaxObserver,
    QATLinear,
    QParams,
    QuantizedLinear,
    binarize,
    calc_qparams,
    convert,
    dequantize,
    fake_quant,
    qrange,
    quantize,
    quantize_multiplier,
    requantize,
    set_quant_state,
)
from scratchgrad.tensor import Tensor
from scratchgrad.testing import TOL_OP, gradcheck, numeric_vjp, rel_error

# Prime shapes: a transposed (in, out) weight raises instead of broadcasting.
N, D, M = 7, 11, 5


def _qp(scale, zero_point, qmin, qmax, axis=None):
    return QParams(np.asarray(scale, float), np.asarray(zero_point, np.int64), qmin, qmax, axis)


# --------------------------------------------------------------------------- #
# affine arithmetic
# --------------------------------------------------------------------------- #
def test_qrange():
    assert qrange(8, symmetric=True) == (-127, 127)  # narrow: no -128
    assert qrange(8, symmetric=False) == (0, 255)
    assert qrange(4, symmetric=True) == (-7, 7)
    assert qrange(2, symmetric=True) == (-1, 1)  # ternary
    assert qrange(1, symmetric=False) == (0, 1)
    for bad in [(1, True), (0, False), (17, False)]:
        with pytest.raises(ValueError):
            qrange(*bad)


def test_zero_is_exactly_representable_for_any_range():
    rng = np.random.default_rng(0)
    for _ in range(300):
        lo, hi = np.sort(rng.normal(0, 3, size=2) + rng.normal(0, 5))
        qp = calc_qparams(lo, hi, bits=int(rng.integers(2, 9)))
        assert qp.zero_point.dtype.kind == "i"
        assert dequantize(quantize(0.0, qp), qp) == 0.0  # exactly, not approximately
    # An all-positive range is widened to include 0, so z sits at qmin.
    qp = calc_qparams(0.5, 2.0, 8)
    assert int(qp.zero_point) == 0 and qp.scale == pytest.approx(2.0 / 255, rel=1e-15)


def test_roundtrip_error_is_half_a_step_and_outside_clips():
    rng = np.random.default_rng(1)
    qp = calc_qparams(-1.3, 2.9, 8)
    lo, hi = dequantize(np.array([0, 255]), qp)
    x = rng.uniform(lo, hi, 10_000)
    err = np.abs(dequantize(quantize(x, qp), qp) - x)
    assert err.max() <= qp.scale / 2 * (1 + 1e-9)
    np.testing.assert_array_equal(quantize(np.array([-50.0, 50.0]), qp), [0, 255])


def test_rounding_is_half_to_even():
    qp = _qp(1.0, 0, -127, 127)
    q = quantize(np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5]), qp)
    np.testing.assert_array_equal(q, [0, 2, 2, 0, -2, -2])


def test_per_channel_runs_along_the_output_axis():
    rng = np.random.default_rng(2)
    w = rng.normal(0, 1, (D, M)) * rng.uniform(0.01, 10, M)  # wildly different columns
    qp = calc_qparams(w.min(axis=0), w.max(axis=0), 8, symmetric=True, axis=1)
    q = quantize(w, qp)
    assert q.dtype == np.int8 and qp.scale.shape == (M,)
    np.testing.assert_array_equal(np.abs(q).max(axis=0), 127)  # every column uses its range
    err = np.abs(dequantize(q, qp) - w)
    assert np.all(err <= qp.scale / 2 * (1 + 1e-9))


def test_degenerate_range_gets_unit_scale():
    qp = calc_qparams(np.array([0.0, -1.0]), np.array([0.0, 1.0]), 8, symmetric=True, axis=1)
    assert qp.scale[0] == 1.0
    w = np.array([[0.0, 0.5], [0.0, -1.0]])
    np.testing.assert_array_equal(dequantize(quantize(w, qp), qp)[:, 0], 0.0)


def test_dequantize_widens_uint8_before_subtracting_zero_point():
    qp = _qp(0.5, 10, 0, 255)
    q = np.array([0, 3], dtype=np.uint8)  # uint8 arithmetic would wrap 0 - 10 to 246
    np.testing.assert_array_equal(dequantize(q, qp), [-5.0, -3.5])


def test_observer_running_minmax_ema_per_channel_and_state():
    rng = np.random.default_rng(3)
    batches = [rng.normal(0, s, (N, M)) for s in (1.0, 3.0, 0.5)]
    run = MinMaxObserver()
    for b in batches:
        run(b)
    allb = np.concatenate(batches)
    assert run.min_val == allb.min() and run.max_val == allb.max()

    ema = MinMaxObserver(momentum=0.25)
    ema(batches[0])
    ema(batches[1])
    assert ema.max_val == pytest.approx(
        batches[0].max() + 0.25 * (batches[1].max() - batches[0].max())
    )

    pc = MinMaxObserver(8, symmetric=True, axis=1, channels=M)
    pc(batches[0])
    np.testing.assert_array_equal(pc.max_val, batches[0].max(axis=0))

    fresh = MinMaxObserver(8, symmetric=True, axis=1, channels=M)
    with pytest.raises(RuntimeError):
        fresh.qparams()
    fresh.load_state_dict(pc.state_dict(deep=True))
    np.testing.assert_array_equal(fresh.qparams().scale, pc.qparams().scale)


# --------------------------------------------------------------------------- #
# fake quantization and the STE
# --------------------------------------------------------------------------- #
def test_fake_quant_forward_is_quantize_then_dequantize():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 2, (N, M))
    for qp in (
        calc_qparams(-1.0, 3.0, 4),
        calc_qparams(x.min(0), x.max(0), 4, symmetric=True, axis=1),
    ):
        with sg.default_dtype(np.float64):
            out = fake_quant(Tensor(x), qp).data
        np.testing.assert_array_equal(out, dequantize(quantize(x, qp), qp))


def test_fake_quant_backward_is_exactly_the_clipped_ste():
    rng = np.random.default_rng(5)
    qp = calc_qparams(-0.4, 1.9, 3)
    with sg.default_dtype(np.float64):
        x = Tensor(rng.uniform(-2, 4, (N, D)), requires_grad=True)
        g = rng.normal(size=(N, D))
        fake_quant(x, qp).backward(g)
    r = np.round(x.data / qp.scale) + qp.zero_point
    mask = (r >= qp.qmin) & (r <= qp.qmax)
    assert 0 < mask.mean() < 1  # both regions exercised
    np.testing.assert_array_equal(x.grad, g * mask)


def test_ste_boundary_convention_uses_the_rounded_code():
    # 4-bit symmetric, s = 0.1: codes -7..7. 7.4 rounds to 7 (in), 7.6 to 8 (clamped).
    qp = _qp(0.1, 0, -7, 7)
    with sg.default_dtype(np.float64):
        x = Tensor(np.array([0.74, 0.76, -0.74, -0.76, 0.0]), requires_grad=True)
        fake_quant(x, qp).backward(np.ones(5))
    np.testing.assert_array_equal(x.grad, [1.0, 0.0, 1.0, 0.0, 1.0])


def test_ste_equals_derivative_of_unrounded_surrogate_inside_range():
    """Where the clamp is inactive the STE is the gradient of ``clip(x, lo, hi)``.

    That surrogate is smooth there, so central differences can check it -- and
    they also show why the rounded op itself cannot be checked: its numeric
    gradient is exactly zero on (nearly) every entry.
    """
    rng = np.random.default_rng(6)
    qp = calc_qparams(-1.0, 1.0, 8, symmetric=True)
    lo, hi = -127 * qp.scale, 127 * qp.scale
    with sg.default_dtype(np.float64):
        x = Tensor(rng.uniform(0.9 * lo, 0.9 * hi, (N, D)), requires_grad=True)
        v = rng.normal(size=(N, D))
        fake_quant(x, qp).backward(v)

        def surrogate(t):
            return t.clip(lo, hi)

        gradcheck(surrogate, x, tol=TOL_OP, name="clip surrogate")
        numeric = numeric_vjp(surrogate, [x], v)[0]
        assert rel_error(x.grad, numeric) < TOL_OP
        fq_numeric = numeric_vjp(lambda t: fake_quant(t, qp), [x], v)[0]
    assert np.mean(fq_numeric == 0.0) > 0.95


def test_fake_quant_per_channel_backward():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, (N, M))
    # Channel ranges deliberately narrower than the data so some entries clip.
    qp = calc_qparams(-0.5 * np.ones(M), np.linspace(0.3, 1.5, M), 4, axis=1)
    with sg.default_dtype(np.float64):
        t = Tensor(x, requires_grad=True)
        g = rng.normal(size=x.shape)
        fake_quant(t, qp).backward(g)
    r = np.round(x / qp.scale) + qp.zero_point
    np.testing.assert_array_equal(t.grad, g * ((r >= 0) & (r <= 15)))


def test_binarize_forward_backward_and_optimal_alpha():
    rng = np.random.default_rng(8)
    w = rng.normal(0, 0.5, (D, M))
    w[0, 0], w[1, 1] = 0.0, 1.7  # sign(0) and a clipped STE entry
    with sg.default_dtype(np.float64):
        t = Tensor(w, requires_grad=True)
        out = binarize(t, axis=1)
        g = rng.normal(size=w.shape)
        out.backward(g)
    alpha = np.abs(w).mean(axis=0)
    sign = np.where(w >= 0, 1.0, -1.0)
    np.testing.assert_allclose(out.data, alpha * sign, rtol=1e-15)
    assert out.data[0, 0] > 0
    np.testing.assert_array_equal(t.grad, g * (np.abs(w) <= 1.0))

    def loss(a):
        return ((w - a * sign) ** 2).sum(axis=0)

    assert np.all(loss(alpha) <= loss(alpha * 1.001)) and np.all(loss(alpha) <= loss(alpha * 0.999))


@pytest.mark.parametrize("bits", [8, 4, 2, 1])
def test_qat_linear_gradients(bits):
    """Input path: exact gradcheck (x enters linearly). Weight path: the STE mask is all
    ones because a min-max weight scale never clips, so dW = x^T g at the quantized W."""
    rng = np.random.default_rng(9)
    with sg.default_dtype(np.float64):
        lin = QATLinear(rng.normal(0, 0.3, (D, M)), rng.normal(0, 0.1, M), bits=bits)
        gradcheck(lambda t: lin(t), Tensor(rng.normal(size=(N, D))), tol=TOL_OP)
        x = Tensor(rng.normal(size=(N, D)))
        g = rng.normal(size=(N, M))
        lin.zero_grad()
        lin(x).backward(g)
    np.testing.assert_allclose(lin.weight.grad, x.data.T @ g, rtol=1e-12)
    np.testing.assert_allclose(lin.bias.grad, g.sum(axis=0), rtol=1e-12)


# --------------------------------------------------------------------------- #
# integer-only inference
# --------------------------------------------------------------------------- #
def test_quantize_multiplier_precision_and_carry():
    rng = np.random.default_rng(10)
    m = np.exp(rng.uniform(np.log(1e-6), np.log(1e3), 1000))
    m0, shift = quantize_multiplier(m)
    assert np.all((m0 >= 2**30) & (m0 < 2**31))
    assert np.max(np.abs(m0 * np.exp2(-shift.astype(float)) - m) / m) <= 2.0**-31
    # 1 - 2^-40 rounds its mantissa up to 2^31: must carry to 2^30 and one less shift.
    m0, shift = quantize_multiplier(np.array([1.0 - 2.0**-40]))
    assert m0[0] == 2**30 and shift[0] == 30
    with pytest.raises(ValueError):
        quantize_multiplier(np.array([0.0]))


def test_requantize_rounds_half_up_and_tracks_float_reference():
    m0, shift = quantize_multiplier(0.25)  # exact power of two: ties are real ties
    acc = np.array([2, -2, 6, -6, 3])
    np.testing.assert_array_equal(requantize(acc, m0, shift, 0, -100, 100), [1, 0, 2, -1, 1])
    np.testing.assert_array_equal(np.round(acc * 0.25), [0, -0, 2, -2, 1])  # half-even differs

    rng = np.random.default_rng(11)
    mult = np.exp(rng.uniform(np.log(1e-4), np.log(3.0), M))
    bias = rng.normal(0, 20, M)
    acc = rng.integers(-(2**24), 2**24, (1000, M))
    m0, shift = quantize_multiplier(mult)
    b_fx = np.round(bias * np.exp2(shift.astype(float))).astype(np.int64)
    got = requantize(acc, m0, shift, 3, -(2**40), 2**40, b_fx)
    ref = np.floor(acc * mult + bias + 0.5) + 3
    assert np.max(np.abs(got - ref)) <= 1 and np.mean(got == ref) > 0.999


def test_int8_matmul_wraps_but_quantized_linear_accumulates_in_int32():
    x8 = np.array([[100, 100]], dtype=np.int8)
    assert (x8 @ np.array([[2], [2]], dtype=np.int8))[0, 0] == -112  # the trap itself

    rng = np.random.default_rng(12)
    k = 97
    w_q = rng.choice([-127, 127], (k, M)).astype(np.int8)
    in_qp, out_qp = _qp(1 / 255, 3, 0, 255), _qp(0.37, 128, 0, 255)
    layer = QuantizedLinear.from_codes(w_q, np.full(M, 1e-3), rng.normal(0, 0.1, M), in_qp, out_qp)
    x_q = rng.integers(200, 256, (N, k)).astype(np.uint8)
    acc64 = (x_q.astype(np.int64) - 3) @ w_q.astype(np.int64)
    assert np.abs(acc64).max() > 2**15  # far past anything int8/int16 could hold
    ref = requantize(acc64, layer.m0, layer.shift, 128, 0, 255, layer.b_fx)
    np.testing.assert_array_equal(layer(x_q), ref)
    assert layer(x_q).dtype == np.uint8


def test_accumulator_budget_is_enforced():
    w_q = np.full((784, 3), 127, dtype=np.int8)
    wide_in = _qp(1.0, 0, 0, 2**16 - 1)  # 16-bit activations: 784 * 65535 * 127 > 2^31
    with pytest.raises(ValueError, match="int32 accumulator"):
        QuantizedLinear.from_codes(w_q, np.ones(3), None, wide_in, _qp(1.0, 0, 0, 255))


def _toy_sim(rng, w_bits, a_bits):
    l1, l2 = nn.Linear(13, D, rng=rng), nn.Linear(D, M, rng=rng)
    l1.bias.data[...] = rng.normal(0, 0.2, D)
    l2.bias.data[...] = rng.normal(0, 0.2, M)
    return nn.Sequential(
        FakeQuantize(8),
        QATLinear.from_linear(l1, w_bits),
        nn.ReLU(),
        FakeQuantize(a_bits),
        QATLinear.from_linear(l2, w_bits),
        FakeQuantize(8),
    )


@pytest.mark.parametrize(("w_bits", "a_bits"), [(8, 8), (4, 4), (2, 2), (1, 8)])
def test_integer_model_reproduces_fake_quant_simulation(w_bits, a_bits):
    rng = np.random.default_rng(13)
    sim = _toy_sim(rng, w_bits, a_bits)
    x = rng.uniform(0, 1, (257, 13)).astype(np.float32)
    set_quant_state(sim, observe=True, fake=False)
    with sg.no_grad():
        sim(Tensor(x))
    set_quant_state(sim, observe=False, fake=True)
    sim.eval()
    hidden_qp = list(sim.children())[3].qparams()
    assert int(hidden_qp.zero_point) == 0  # calibrated after the ReLU: the clamp is the ReLU

    with sg.no_grad():
        sim_codes = quantize(sim(Tensor(x)).data, list(sim.children())[5].qparams())
    int_model = convert(sim)
    int_codes = int_model.codes(x)
    assert int_codes.dtype == np.uint8
    assert np.abs(int_codes.astype(int) - sim_codes.astype(int)).max() <= 1
    assert np.mean(int_codes == sim_codes) > 0.99
    np.testing.assert_array_equal(int_model.predict(x), int_codes.argmax(1))


def test_convert_rejects_a_chain_it_cannot_lower():
    rng = np.random.default_rng(14)
    lin = QATLinear(rng.normal(size=(D, M)), None)
    with pytest.raises(ValueError):
        convert(nn.Sequential(lin))
    fq = FakeQuantize(8)
    fq(Tensor(rng.uniform(size=(N, D))))
    with pytest.raises(ValueError):
        convert(nn.Sequential(fq, lin))  # no output quantizer
