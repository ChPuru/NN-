"""Verify the claims in ``09-gradient-checking.md``.

1. Error-vs-``h`` sweeps for forward and central differences: the observed
   log-log slopes (1 and 2), the observed optimal ``h``, and the observed best
   attainable error, against the ``eps^(1/2)`` / ``eps^(1/3)`` predictions.
2. Why ``rel_error`` normalises by the array scale: a gradient with one
   legitimately tiny entry is scored both ways and only the array-scale metric
   can pass, at any ``h``.
3. Why the cotangent must be random: a deliberately transposed backward passes
   at machine precision against an all-ones cotangent and fails against a
   Gaussian one.
4. Why float64: the same correct gradient scored in float32.
5. The adjoint identity as the exact, step-free alternative for a linear map.

    PYTHONPATH=src python3 notes/derivations/verify/verify_09_gradient_checking.py
"""

from __future__ import annotations

import math

import numpy as np
from _harness import Check, cli

import scratchgrad as sg
from scratchgrad import nn
from scratchgrad.tensor import Tensor
from scratchgrad.testing import adjoint_test, gradcheck, rel_error

MACH_EPS = float(np.finfo(np.float64).eps)  # 2.22e-16


def _sweep() -> dict[str, float]:
    """Error of both difference formulas over 13 decades of ``h``.

    ``f(x) = exp(sin x)`` at ``x = 1.2``: an entire function with O(1) value and
    O(1) derivatives of every order, so the constants in the truncation and
    rounding terms are both near 1 and the predicted optima are not buried in
    problem-specific scale factors.
    """
    x = 1.2
    f = lambda t: math.exp(math.sin(t))  # noqa: E731
    exact = math.exp(math.sin(x)) * math.cos(x)

    hs = np.logspace(-1, -13, 49)
    fwd = np.array([abs((f(x + h) - f(x)) / h - exact) for h in hs])
    cen = np.array([abs((f(x + h) - f(x - h)) / (2 * h) - exact) for h in hs])

    def slope(err: np.ndarray, lo: float, hi: float) -> float:
        m = (hs <= lo) & (hs >= hi)
        return float(np.polyfit(np.log10(hs[m]), np.log10(err[m]), 1)[0])

    return {
        "fwd_slope": slope(fwd, 1e-1, 1e-5),  # truncation regime: O(h)
        "cen_slope": slope(cen, 1e-1, 1e-4),  # truncation regime: O(h^2)
        "fwd_best_h": float(hs[int(np.argmin(fwd))]),
        "cen_best_h": float(hs[int(np.argmin(cen))]),
        "fwd_best_err": float(fwd.min()),
        "cen_best_err": float(cen.min()),
        "cen_err_at_1e5": float(cen[int(np.argmin(np.abs(hs - 1e-5)))]),
    }


def _small_entry(h: float = 1e-5) -> tuple[float, float, float]:
    """One legitimately tiny gradient entry, scored per-element and array-scale.

    ``f(x) = <c, x>`` with ``c = [3, -2, 1e-6]`` and ``x`` of order 10, so the
    objective is of order 30 and the finite-difference noise floor is
    ``eps * |f| / h`` ~ 7e-10 -- set by the *objective*, not by the entry being
    measured. The 1e-6 entry therefore carries a ~1e-3 error *of its own size*
    however correct the analytic gradient is.
    """
    c = np.array([3.0, -2.0, 1e-6])
    x = np.array([10.0, -7.0, 4.0])
    f = lambda t: float(c @ t)  # noqa: E731
    numeric = np.empty(3)
    for i in range(3):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        numeric[i] = (f(xp) - f(xm)) / (2 * h)
    per_element = float(np.max(np.abs(numeric - c) / (np.abs(c) + np.abs(numeric) + 1e-12)))
    return per_element, rel_error(c, numeric), float(MACH_EPS * abs(f(x)) / h)


def _cotangent(seed: int = 0) -> tuple[float, float]:
    """A transposed backward: rel error against an all-ones vs a Gaussian cotangent.

    The shape is square *on purpose*. That is the whole trap: at (7, 7) the
    wrong-way gradient is still conformable, so nothing raises, and with an
    all-ones cotangent ``<1, x^T>`` is symmetric in the bug and it passes at
    machine precision.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((7, 7))

    def bad_transpose(t: Tensor) -> Tensor:
        # The correct rule is `lambda g: g.T`. This is the classic omission.
        return Tensor._make(t.data.T, (t,), (lambda g: g,), "bad_transpose")

    out = []
    for v in (np.ones((7, 7)), rng.standard_normal((7, 7))):
        t = sg.Tensor(x, requires_grad=True, dtype=np.float64)
        bad_transpose(t).backward(v)
        out.append(rel_error(t.grad, v.T))  # the true VJP of transpose is v^T
    return out[0], out[1]


def _float32_has_no_power(seed: int = 0) -> tuple[float, float]:
    """The same correct gradient, scored in float32 and in float64."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((11, 5)) / math.sqrt(11)
    x = rng.standard_normal((7, 11))
    v = rng.standard_normal((7, 5))
    errs = []
    # Both at the framework's own h_rel=1e-5. That is the point: gradcheck does
    # not get to pick a different step per dtype, so float32's noise floor --
    # eps32 * |f| / h ~ 1.2e-2 * |f| -- is what the check would actually see.
    for dtype, h in ((np.float32, 1e-5), (np.float64, 1e-5)):
        tw = sg.Tensor(w.astype(dtype), requires_grad=True, dtype=dtype)
        (sg.Tensor(x.astype(dtype), dtype=dtype) @ tw).tanh().backward(v.astype(dtype))
        analytic = np.asarray(tw.grad, dtype=np.float64)

        wd = np.array(w, dtype=dtype)
        numeric = np.zeros(wd.shape, dtype=np.float64)
        for i in range(wd.size):
            orig = float(wd.flat[i])
            wd.flat[i] = orig + h
            plus = float(np.sum(np.tanh(x.astype(dtype) @ wd) * v.astype(dtype)))
            wd.flat[i] = orig - h
            minus = float(np.sum(np.tanh(x.astype(dtype) @ wd) * v.astype(dtype)))
            wd.flat[i] = orig
            numeric.flat[i] = (plus - minus) / (2 * h)
        errs.append(rel_error(analytic, numeric))
    return errs[0], errs[1]


def _adjoint_vs_gradcheck(seed: int = 0) -> tuple[float, float]:
    """The same ``Linear``, checked by the adjoint identity and by differences."""
    lin = nn.Linear(11, 5, bias=False, rng=np.random.default_rng(1)).double()

    def forward(u: sg.Tensor) -> sg.Tensor:
        return lin(u)

    def backward_vjp(u: sg.Tensor, v: np.ndarray) -> np.ndarray:
        u.grad = None
        lin(u).backward(v)
        return u.grad

    adj = adjoint_test(forward, backward_vjp, (7, 11), (7, 5), seed=seed, tol=1e-12)
    x = sg.Tensor(
        np.random.default_rng(seed + 99).standard_normal((7, 11)),
        requires_grad=True,
        dtype=np.float64,
    )
    fd = max(gradcheck(lin, x, tol=1e-9, seed=seed, name="Linear").values())
    return adj, fd


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 09-gradient-checking.md."""
    s = _sweep()
    per_el, arr_scale, floor = _small_entry()
    ones_err, rand_err = _cotangent(seed)
    f32, f64 = _float32_has_no_power(seed)
    adj, fd = _adjoint_vs_gradcheck(seed)
    return [
        Check(
            "forward_slope_is_1",
            abs(s["fwd_slope"] - 1.0),
            0.05,
            note=f"log-log slope {s['fwd_slope']:.3f} in the truncation regime",
        ),
        Check(
            "central_slope_is_2",
            abs(s["cen_slope"] - 2.0),
            0.05,
            note=f"log-log slope {s['cen_slope']:.3f}: O(h^2), the whole reason to pay 2x",
        ),
        Check(
            "forward_optimum_near_sqrt_eps",
            abs(math.log10(s["fwd_best_h"]) - math.log10(math.sqrt(MACH_EPS))),
            0.75,
            note=f"best h {s['fwd_best_h']:.2e} vs sqrt(eps)={math.sqrt(MACH_EPS):.2e}",
        ),
        Check(
            "central_optimum_near_cbrt_eps",
            abs(math.log10(s["cen_best_h"]) - math.log10(MACH_EPS ** (1 / 3))),
            0.75,
            note=f"best h {s['cen_best_h']:.2e} vs eps^(1/3)={MACH_EPS ** (1 / 3):.2e}",
        ),
        Check(
            "central_beats_forward",
            s["fwd_best_err"] / s["cen_best_err"],
            100.0,
            kind="above",
            note=f"best errors {s['fwd_best_err']:.2e} vs {s['cen_best_err']:.2e}",
            extras={"fwd_best_err": s["fwd_best_err"], "cen_best_err": s["cen_best_err"]},
        ),
        Check(
            "h_1e5_is_near_optimal",
            s["cen_err_at_1e5"] / s["cen_best_err"],
            10.0,
            note=f"h_rel=1e-5 costs {s['cen_err_at_1e5'] / s['cen_best_err']:.2f}x the best "
            "attainable error",
        ),
        Check(
            "per_element_metric_fails",
            per_el,
            1e-9,
            kind="above",
            note=f"a correct 1e-6 entry scores {per_el:.2e} per-element (floor {floor:.1e})",
            extras={"noise_floor": floor},
        ),
        Check(
            "array_scale_metric_passes",
            arr_scale,
            1e-9,
            note="the same numbers, normalised by the array scale",
        ),
        Check(
            "ones_cotangent_hides_transpose",
            ones_err,
            1e-14,
            note="a wrong-way backward at (7,7) passes at machine precision",
        ),
        Check(
            "random_cotangent_catches_it",
            rand_err,
            0.1,
            kind="above",
            note="and fails outright once the cotangent is Gaussian",
        ),
        Check(
            "float32_has_no_power",
            f32,
            1e-3,
            kind="above",
            note=f"a CORRECT gradient scores {f32:.2e} in float32",
        ),
        Check(
            "float64_has_power",
            f64,
            1e-9,
            note=f"and {f64:.2e} in float64 -- six orders of resolution",
        ),
        Check(
            "adjoint_is_exact",
            adj,
            1e-14,
            note=f"<v, Au> == <A^T v, u> for Linear; gradcheck manages {fd:.2e}",
            extras={"gradcheck_err": fd},
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "09 gradient checking"))
