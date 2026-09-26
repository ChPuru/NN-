"""Verify the claims in ``08-adam.md``.

1. ``m_t = (1 - b1^t) * g`` for a constant gradient, exactly -- which is where
   the bias-correction denominator comes from.
2. The same statement in expectation for *random* stationary gradients,
   measured by Monte Carlo.
3. The framework's first step has magnitude ``lr`` for any gradient scale --
   the invariant that says bias correction is present and correct.
4. The three ways to get correction wrong, with their exact factors:
   omitting it entirely is ``(1-b1)/sqrt(1-b2) = 3.162``; correcting ``m`` but
   not ``v`` is ``1/sqrt(1-b2) = 31.62``; correcting ``v`` but not ``m`` is
   ``(1-b1) = 0.1``. All three are run against ``optim.Adam``.
5. ``eps`` inside the sqrt (``m_hat / sqrt(v_hat + eps)``) floors the
   denominator at ``sqrt(eps) = 1e-4``, four orders too early -- while agreeing
   with the correct form to 5e-9 at |g| = 1, which is why it survives review.

    PYTHONPATH=src python3 notes/derivations/verify/verify_08_adam.py
"""

from __future__ import annotations

import math

import numpy as np
from _harness import Check, cli

import scratchgrad as sg
from scratchgrad import optim
from scratchgrad.nn.module import Parameter

B1, B2, EPS, LR = 0.9, 0.999, 1e-8, 1e-3


def _ema_closed_form(steps: int = 10, g: float = 0.7) -> tuple[float, float]:
    """m_t and v_t against ``(1-b1^t) g`` and ``(1-b2^t) g^2`` for constant g."""
    m = v = 0.0
    worst_m = worst_v = 0.0
    for t in range(1, steps + 1):
        m = B1 * m + (1.0 - B1) * g
        v = B2 * v + (1.0 - B2) * g * g
        worst_m = max(worst_m, abs(m - (1.0 - B1**t) * g) / abs(g))
        worst_v = max(worst_v, abs(v - (1.0 - B2**t) * g * g) / (g * g))
    return worst_m, worst_v


def _monte_carlo(rng: np.random.Generator, trials: int = 200_000, t: int = 5) -> float:
    """E[m_t] / (1 - b1^t) == E[g] for i.i.d. stationary gradients.

    The estimator's standard error is the reason for 200k trials: with
    ``g ~ N(1, 1)`` the corrected mean has SE about ``0.4/sqrt(trials)`` ~ 9e-4,
    so anything below ~5e-3 is consistent with an exactly unbiased estimator and
    anything above it is not.
    """
    mu = 1.0
    m = np.zeros(trials)
    for _ in range(t):
        m = B1 * m + (1.0 - B1) * (rng.standard_normal(trials) + mu)
    return abs(float(m.mean()) / (1.0 - B1**t) - mu)


def _first_step(g_value: float) -> float:
    """|p_1 - p_0| after one ``optim.Adam`` step with a constant gradient.

    ``default_dtype`` is not decoration: ``Parameter`` casts to the default
    dtype, which is float32, and the optimizer then casts its update back to the
    parameter's dtype. In float32 a correct step differs from the closed form by
    ~6e-8 and the comparison below has no resolution left.
    """
    with sg.default_dtype(np.float64):
        p = Parameter(np.zeros(3, dtype=np.float64))
        p.grad = np.full(3, g_value, dtype=np.float64)
        opt = optim.Adam([p], lr=LR, betas=(B1, B2), eps=EPS)
        opt.step()
        return float(np.max(np.abs(p.data)))


def _variants(g_value: float = 1.0) -> dict[str, float]:
    """First-step magnitude for each way of (mis)handling bias correction."""
    m = (1.0 - B1) * g_value
    v = (1.0 - B2) * g_value * g_value
    bc1, bc2 = 1.0 - B1, 1.0 - B2  # t = 1
    return {
        "correct": LR * (m / bc1) / (math.sqrt(v / bc2) + EPS),
        "none": LR * m / (math.sqrt(v) + EPS),
        "m_only": LR * (m / bc1) / (math.sqrt(v) + EPS),
        "v_only": LR * m / (math.sqrt(v / bc2) + EPS),
    }


def _eps_placement(g_value: float) -> tuple[float, float]:
    """``m_hat/(sqrt(v_hat)+eps)`` vs the wrong ``g/sqrt(v+eps)``, at t=1."""
    m = (1.0 - B1) * g_value
    v = (1.0 - B2) * g_value * g_value
    m_hat, v_hat = m / (1.0 - B1), v / (1.0 - B2)
    right = LR * m_hat / (math.sqrt(v_hat) + EPS)
    wrong = LR * m_hat / math.sqrt(v_hat + EPS)  # eps INSIDE the root
    return right, wrong


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 08-adam.md."""
    rng = np.random.default_rng(seed)
    m_err, v_err = _ema_closed_form()
    var = _variants()
    steps = {g: _first_step(g) for g in (1e-3, 1.0, 1e3)}
    right_small, wrong_small = _eps_placement(1e-5)
    right_big, wrong_big = _eps_placement(1.0)

    scale_free = max(abs(s / LR - 1.0) for s in steps.values())
    framework_vs_hand = abs(_first_step(1.0) - var["correct"]) / var["correct"]
    return [
        Check("m_closed_form", m_err, 1e-15, note="m_t == (1 - b1^t) g over 10 steps"),
        Check("v_closed_form", v_err, 1e-15, note="v_t == (1 - b2^t) g^2 over 10 steps"),
        Check(
            "unbiased_in_expectation",
            _monte_carlo(rng),
            5e-3,
            note="E[m_5]/(1 - b1^5) == E[g], 200k i.i.d. draws",
        ),
        Check(
            "framework_matches_hand",
            framework_vs_hand,
            1e-15,
            note="optim.Adam's first step == lr * m_hat/(sqrt(v_hat)+eps)",
        ),
        Check(
            "first_step_is_lr",
            scale_free,
            1e-4,
            note=f"|step| = lr for g in 1e-3, 1, 1e3: "
            f"{steps[1e-3] / LR:.6f}, {steps[1.0] / LR:.6f}, {steps[1e3] / LR:.6f}",
        ),
        # The ratios are compared relatively: `eps` perturbs each variant's
        # denominator slightly differently, at the 1e-7 level.
        Check(
            "no_correction_is_3_16x",
            abs(var["none"] / var["correct"] / (0.1 / math.sqrt(0.001)) - 1.0),
            1e-6,
            note=f"measured {var['none'] / var['correct']:.5f} vs (1-b1)/sqrt(1-b2)="
            f"{0.1 / math.sqrt(0.001):.5f}; the numerator is 10x too small too",
        ),
        Check(
            "m_only_is_31_6x",
            abs(var["m_only"] / var["correct"] * math.sqrt(0.001) - 1.0),
            1e-6,
            note=f"measured {var['m_only'] / var['correct']:.4f} vs 1/sqrt(1-b2)=31.6228 -- "
            "THIS is the explosive mistake",
        ),
        Check(
            "v_only_is_0_1x",
            abs(var["v_only"] / var["correct"] / 0.1 - 1.0),
            1e-6,
            note=f"measured {var['v_only'] / var['correct']:.6f} vs (1-b1)=0.1 -- ten times "
            "too small, and it merely stalls",
        ),
        Check(
            "eps_inside_floors_small_g",
            wrong_small / right_small,
            0.2,
            note=f"at |g|=1e-5, m_hat/sqrt(v_hat+eps) steps {wrong_small:.3e} vs {right_small:.3e}",
            extras={"wrong_step": wrong_small, "right_step": right_small},
        ),
        Check(
            "eps_inside_ok_at_large_g",
            abs(wrong_big / right_big - 1.0),
            1e-7,
            note="at |g|=1 both agree to 1e-8, which is why the bug survives review",
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "08 adam"))
