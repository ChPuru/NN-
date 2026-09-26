"""Verify the claims in ``03-layernorm.md``.

1. The *unsimplified* chain -- ``dmu``, ``dv``, then ``dx_j`` -- reproduces the
   collapsed three-term form. This is the derivation itself, run as arithmetic.
2. The three-term form matches the framework's fused backward exactly.
3. ``dg = sum(dy * xhat)`` and ``db = sum(dy)``.
4. All of it survives central differences (``check_module``).
5. The Annotated Transformer's convention (unbiased std, eps *outside* the
   root) is a genuinely different function -- the size of the disagreement is
   measured, not assumed.

    PYTHONPATH=src python3 notes/derivations/verify/verify_03_layernorm.py
"""

from __future__ import annotations

import numpy as np
from _harness import Check, cli

import scratchgrad as sg
from scratchgrad import nn
from scratchgrad.testing import check_module, rel_error

N, D = 7, 11
EPS = 1e-5


def _pieces(rng: np.random.Generator):
    """A concrete LayerNorm forward and a random cotangent, in float64."""
    x = rng.standard_normal((N, D)) * 2.0 + 0.5
    g = rng.standard_normal(D) * 0.5 + 1.0
    b = rng.standard_normal(D) * 0.1
    dy = rng.standard_normal((N, D))
    mu = x.mean(axis=1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=1, keepdims=True)  # BIASED: 1/D, not 1/(D-1)
    istd = 1.0 / np.sqrt(var + EPS)  # eps INSIDE the root
    xhat = xc * istd
    return x, g, b, dy, mu, xc, var, istd, xhat


def _unsimplified(rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """dx from the raw chain through mu and v, vs the collapsed three-term form."""
    _x, g, _b, dy, _mu, xc, _var, istd, xhat = _pieces(rng)
    dxhat = dy * g

    # v = mean_i (x_i - mu)^2, so d(v)/d(x_j) picks up 2(x_j - mu)/D; and mu
    # contributes nothing extra to v because sum_i (x_i - mu) == 0 identically.
    dv = -0.5 * istd**3 * (dxhat * xc).sum(axis=1, keepdims=True)
    dmu = -istd * dxhat.sum(axis=1, keepdims=True)
    dx_raw = istd * dxhat + dv * 2.0 * xc / D + dmu / D

    dx_three = istd * (
        dxhat
        - dxhat.mean(axis=1, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=1, keepdims=True)
    )
    return rel_error(dx_raw, dx_three), dx_three


def _vs_framework(rng: np.random.Generator) -> tuple[float, float, float]:
    """The three-term dx, dg and db against ``nn.LayerNorm``'s fused backward."""
    x, g, b, dy, _mu, _xc, _var, istd, xhat = _pieces(rng)
    dxhat = dy * g
    dx = istd * (
        dxhat
        - dxhat.mean(axis=1, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=1, keepdims=True)
    )

    ln = nn.LayerNorm(D, eps=EPS).double()
    ln.weight.data[...] = g
    ln.bias.data[...] = b
    tx = sg.Tensor(x, requires_grad=True, dtype=np.float64)
    out = ln(tx)
    fwd_err = rel_error(out.data, xhat * g + b)
    out.backward(dy)
    return (
        rel_error(tx.grad, dx),
        rel_error(ln.weight.grad, (dy * xhat).sum(axis=0)),
        rel_error(ln.bias.grad, dy.sum(axis=0)),
    ), fwd_err


def _annotated_transformer(rng: np.random.Generator) -> tuple[float, float]:
    """How far the ``std_unbiased + eps`` convention is from this one.

    Both train. They are not the same function, they do not gradcheck against
    each other, and a checkpoint written under one is silently rescaled under
    the other. At D = 11 the Bessel factor alone is sqrt(11/10) = 1.0488.
    """
    _x, g, b, _dy, _mu, xc, _var, _istd, xhat = _pieces(rng)
    ours = xhat * g + b
    std_unbiased = np.sqrt((xc * xc).sum(axis=1, keepdims=True) / (D - 1))
    theirs = g * xc / (std_unbiased + EPS) + b
    return float(np.max(np.abs(ours - theirs))), float(np.sqrt(D / (D - 1.0)))


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 03-layernorm.md."""
    rng = np.random.default_rng(seed)
    collapse_err, _ = _unsimplified(rng)
    (dx_err, dg_err, db_err), fwd_err = _vs_framework(rng)
    gap, bessel = _annotated_transformer(rng)
    # The input must come from a DIFFERENT stream than gradcheck's cotangent.
    # check_module seeds both from `seed`, and for a shape-preserving layer that
    # makes <x, norm(x)> a stationary point: the true gradient collapses to
    # ~1e-5 and rel_error, normalised by that collapsed scale, reads ordinary
    # finite-difference noise as a 1e-5 failure. Measured below as `stationary`.
    data = np.random.default_rng(4242).standard_normal((N, D))
    mod_errs = check_module(
        nn.LayerNorm(D, eps=EPS), (N, D), tol=1e-7, input_data=data, name="LayerNorm"
    )
    # tol=inf: measure the same-stream error instead of raising on it.
    same_stream = max(
        check_module(
            nn.LayerNorm(D, eps=EPS), (N, D), tol=float("inf"), name="LayerNorm-same-stream"
        ).values()
    )
    return [
        Check(
            "collapse_to_three_terms",
            collapse_err,
            1e-14,
            note="dmu/dv chain == istd*(dxhat - mean - xhat*mean(dxhat*xhat))",
        ),
        Check("forward_matches", fwd_err, 1e-15, note="biased var, eps inside the sqrt"),
        Check("dx_vs_framework", dx_err, 1e-14, note="three-term dx == fused backward"),
        Check("dg_vs_framework", dg_err, 1e-15, note="dg == sum_n dy * xhat"),
        Check("db_vs_framework", db_err, 1e-15, note="db == sum_n dy"),
        Check(
            "finite_differences",
            max(mod_errs.values()),
            1e-7,
            note="check_module over x, weight and bias, independent input stream",
        ),
        Check(
            "stationary_point_trap",
            same_stream,
            1e-7,
            kind="above",
            note="same stream for x and the cotangent: <x, norm(x)> is stationary",
        ),
        Check(
            "annotated_transformer_gap",
            gap,
            1e-3,
            kind="above",
            note=f"max |ours - theirs| on the same input; Bessel factor {bessel:.4f}",
            extras={"bessel_factor": bessel},
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "03 layernorm"))
