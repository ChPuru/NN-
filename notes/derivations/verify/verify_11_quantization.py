"""Verify the claims in ``11-quantization.md``.

1. Round-to-nearest error is at most ``s/2`` inside the range, and its variance
   is ``s^2/12`` for a signal that is smooth on the scale of ``s``.
2. An integer zero point makes 0 exact; the unrounded one does not.
3. The STE equals the derivative of the unrounded surrogate (checked by central
   differences), while central differences of the rounded op itself are 0 on a
   step and ``s/(2h)`` across a jump -- never the STE's 1.
4. ``M0 * 2^-shift`` reproduces ``M`` to ``2^-31``; the integer requantizer
   tracks ``floor(acc M + b + 1/2)`` and rounds ties toward +inf.
5. The overflow budgets: int8 x int8 wraps in NumPy; the int32 accumulator has
   ~84x headroom for a 784-wide 8-bit layer.
6. The textbook int32 bias is off by up to ``M/2`` output codes; the int64
   fixed-point bias is not.
7. ``alpha = mean|w|`` is the least-squares scale for binary weights.
8. The integer-only model reproduces the fake-quant simulation.

    PYTHONPATH=src python3 notes/derivations/verify/verify_11_quantization.py
"""

from __future__ import annotations

import numpy as np
from _harness import Check, cli

import scratchgrad as sg
from scratchgrad import nn
from scratchgrad.quant import (
    FakeQuantize,
    QATLinear,
    calc_qparams,
    convert,
    dequantize,
    fake_quant,
    quantize,
    quantize_multiplier,
    requantize,
    set_quant_state,
)
from scratchgrad.tensor import Tensor
from scratchgrad.testing import numeric_vjp, rel_error


def _error_model(rng: np.random.Generator) -> tuple[float, float]:
    qp = calc_qparams(-4.0, 4.0, 8)
    s = float(qp.scale)
    x = rng.normal(0.0, 1.0, 400_000)  # sigma = 32 steps: smooth on the grid
    lo, hi = dequantize(np.array([0, 255]), qp)
    x = x[(x >= lo) & (x <= hi)]  # the bound is for the representable range only
    e = dequantize(quantize(x, qp), qp) - x
    return float(np.abs(e).max() / (s / 2)), float(e.var() / (s * s / 12))


def _zero_point(rng: np.random.Generator) -> tuple[float, float]:
    worst_int, mean_frac = 0.0, 0.0
    for _ in range(1000):
        # Ranges that straddle 0: a one-signed range puts z at qmin or qmax,
        # where it is an integer by construction and the trap cannot show.
        lo, hi = -rng.uniform(0.1, 5.0), rng.uniform(0.1, 5.0)
        qp = calc_qparams(lo, hi, 8)
        worst_int = max(worst_int, abs(float(dequantize(quantize(0.0, qp), qp))))
        s = (hi - lo) / 255
        z = -lo / s  # the unrounded zero point
        q0 = np.clip(np.round(0.0 / s + z), 0, 255)
        mean_frac += abs(s * (q0 - z)) / s / 1000
    return worst_int, mean_frac


def _ste(rng: np.random.Generator) -> tuple[float, float, float]:
    qp = calc_qparams(-1.0, 1.0, 8, symmetric=True)
    s = float(qp.scale)
    with sg.default_dtype(np.float64):
        x = Tensor(rng.uniform(-0.9, 0.9, (7, 11)), requires_grad=True)
        v = rng.normal(size=(7, 11))
        fake_quant(x, qp).backward(v)
        surrogate = numeric_vjp(lambda t: t.clip(-127 * s, 127 * s), [x], v)[0]
        rounded = numeric_vjp(lambda t: fake_quant(t, qp), [x], v)[0]
        # A probe pair straddling the jump at 10.5 s (1e-9 past it, h = 1e-7).
        h = 1e-7
        x0 = 10.5 * s + 1e-9
        up = fake_quant(Tensor(np.array([x0 + h])), qp).data[0]
        dn = fake_quant(Tensor(np.array([x0 - h])), qp).data[0]
    jump_ratio = ((up - dn) / (2 * h)) / (s / (2 * h))
    return rel_error(x.grad, surrogate), float(np.mean(rounded == 0.0)), float(jump_ratio)


def _requant(rng: np.random.Generator) -> tuple[float, float, float, float]:
    m = np.exp(rng.uniform(np.log(1e-6), np.log(1e3), 10_000))
    m0, shift = quantize_multiplier(m)
    mult_err = float(np.max(np.abs(m0 * np.exp2(-shift.astype(float)) - m) / m))

    mult = np.exp(rng.uniform(np.log(1e-4), np.log(3.0), 13))
    bias = rng.normal(0, 20, 13)
    acc = rng.integers(-(2**24), 2**24, (20_000, 13))
    m0, shift = quantize_multiplier(mult)
    b_fx = np.round(bias * np.exp2(shift.astype(float))).astype(np.int64)
    got = requantize(acc, m0, shift, 0, -(2**40), 2**40, b_fx)
    ref = np.floor(acc * mult + bias + 0.5)
    disagree, maxdiff = float(np.mean(got != ref)), float(np.max(np.abs(got - ref)))

    m0, shift = quantize_multiplier(0.25)
    ties = requantize(np.array([2, -2, 6, -6]), m0, shift, 0, -9, 9)
    tie_err = float(np.abs(ties - np.array([1, 0, 2, -1])).sum())
    return mult_err, disagree, maxdiff, tie_err


def _bias_grid(rng: np.random.Generator) -> tuple[float, float]:
    """Output-code error of the bias, textbook int32 grid vs int64 fixed point.

    Scales of the trained W2A2 MNIST second layer: s_x = 3.07 (2-bit hidden),
    s_y = 0.187 (8-bit logits), and a ternary s_w of 0.68 gives the median
    M = s_x s_w / s_y ~ 11.2 measured there.
    """
    s_x, s_w, s_y = 3.07, 0.68, 0.187
    b = rng.normal(0, 0.5, 100_000)
    textbook = np.round(b / (s_x * s_w)) * (s_x * s_w)
    _, shift = quantize_multiplier(s_x * s_w / s_y)
    fixed = np.round(b / s_y * 2.0 ** float(shift)) * 2.0 ** -float(shift) * s_y
    m = s_x * s_w / s_y
    return float(np.abs(textbook - b).max() / s_y / (m / 2)), float(np.abs(fixed - b).max() / s_y)


def _alpha(rng: np.random.Generator) -> float:
    w = rng.normal(0, 0.1, 1001)
    b = np.where(w >= 0, 1.0, -1.0)
    grid = np.abs(w).mean() * (1 + np.linspace(-1e-2, 1e-2, 200_001))
    # ||w - a b||^2 expanded, so the grid search needs no (200001 x 1001) array.
    losses = (w @ w) - 2 * grid * (w @ b) + grid**2 * (b @ b)
    best = grid[np.argmin(losses)]
    return float(abs(best - np.abs(w).mean()) / np.abs(w).mean())


def _integer_vs_sim(rng: np.random.Generator) -> tuple[float, float]:
    l1, l2 = nn.Linear(13, 11, rng=rng), nn.Linear(11, 5, rng=rng)
    l1.bias.data[...] = rng.normal(0, 0.2, 11)
    l2.bias.data[...] = rng.normal(0, 0.2, 5)
    sim = nn.Sequential(
        FakeQuantize(8),
        QATLinear.from_linear(l1, 2),
        nn.ReLU(),
        FakeQuantize(2),
        QATLinear.from_linear(l2, 2),
        FakeQuantize(8),
    )
    x = rng.uniform(0, 1, (4096, 13)).astype(np.float32)
    set_quant_state(sim, observe=True, fake=False)
    with sg.no_grad():
        sim(Tensor(x))
    set_quant_state(sim, observe=False, fake=True)
    sim.eval()
    with sg.no_grad():
        sim_codes = quantize(sim(Tensor(x)).data, list(sim.children())[5].qparams())
    int_codes = convert(sim).codes(x)
    diff = np.abs(int_codes.astype(int) - sim_codes.astype(int))
    return float(np.mean(diff == 0)), float(diff.max())


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 11-quantization.md."""
    rng = np.random.default_rng(seed)
    half_step, var_ratio = _error_model(rng)
    zero_int, zero_frac = _zero_point(rng)
    ste_err, fd_zero, jump_ratio = _ste(rng)
    mult_err, disagree, maxdiff, tie_err = _requant(rng)
    textbook, fixed = _bias_grid(rng)
    agree, sim_maxdiff = _integer_vs_sim(rng)
    x8 = np.array([[100, 100]], dtype=np.int8) @ np.array([[2], [2]], dtype=np.int8)
    return [
        Check("error_within_half_step", half_step, 1.0 + 1e-9, note="max|x - x_hat| / (s/2)"),
        Check(
            "error_variance_s2_over_12",
            abs(var_ratio - 1.0),
            0.02,
            note=f"var(e) / (s^2/12) = {var_ratio:.4f} for a signal smooth on the grid",
        ),
        Check("zero_exact_integer_zp", zero_int, 0.0, note="max|x_hat(0)| over 1000 ranges"),
        Check(
            "zero_inexact_fractional_zp",
            zero_frac,
            0.1,
            kind="above",
            note="mean|x_hat(0)|/s with an unrounded z (expected ~0.25)",
        ),
        Check("ste_is_surrogate_derivative", ste_err, 1e-9, note="STE vs FD of clip(x)"),
        Check("fd_of_round_is_zero", fd_zero, 0.95, kind="above", note="fraction of FD == 0"),
        Check(
            "fd_across_jump_is_s_over_2h",
            abs(jump_ratio - 1.0),
            1e-6,
            note="FD straddling one jump = s/(2h), not 1",
        ),
        Check("multiplier_rel_error", mult_err, 2.0**-31, note="|M0 2^-shift - M| / M"),
        Check("requant_disagreement", disagree, 1e-3, note="vs floor(acc M + b + 1/2)"),
        Check("requant_max_code_diff", maxdiff, 1.0, note="never more than one code"),
        Check("ties_round_half_up", tie_err, 0.0, note="+-0.5 -> 1, 0 ; +-1.5 -> 2, -1"),
        Check("int8_matmul_wraps", float(x8[0, 0]), -112.0, kind="close", note="[100,100]@[2,2]"),
        Check(
            "int32_headroom_784x255x127",
            2**31 / (784 * 255 * 127),
            50.0,
            kind="above",
            note="accumulator budget of an 8-bit 784-wide layer",
        ),
        Check(
            "textbook_bias_err_over_half_M",
            textbook,
            0.99,
            kind="above",
            note="int32 bias on the acc grid: up to M/2 output codes (M ~ 11)",
        ),
        Check("fixed_point_bias_err_codes", fixed, 1e-6, note="int64 bias on the 2^-shift grid"),
        Check("binary_alpha_is_mean_abs", _alpha(rng), 1e-6, note="grid argmin vs mean|w|"),
        Check("integer_equals_sim_codes", agree, 0.999, kind="above", note="W2A2 toy MLP"),
        Check("integer_vs_sim_max_diff", sim_maxdiff, 1.0, note="in logit codes"),
    ]


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "11 quantization"))
