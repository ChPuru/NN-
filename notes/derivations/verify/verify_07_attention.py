"""Verify the claims in ``07-attention.md``.

1. The softmax Jacobian collapse: contracting ``diag(a) - a a^T`` with a
   cotangent, built as a dense ``S x S`` matrix per row, equals the rank-1
   expression ``a * (da - sum(da * a))``.
2. ``dQ``, ``dK`` and ``dV`` derived from that, against the framework.
3. The ``1/sqrt(d)`` scale: the variance of a dot product of two independent
   unit-variance vectors is measured at several head widths, before and after
   scaling, and so is the resulting softmax saturation.
4. Masking: ``-inf`` produces NaN on a fully-masked row; ``finfo.min/2``
   produces a finite uniform row. Both are run.

    PYTHONPATH=src python3 notes/derivations/verify/verify_07_attention.py
"""

from __future__ import annotations

import math

import numpy as np
from _harness import Check, cli

import scratchgrad as sg
from scratchgrad.nn.attention import (
    causal_mask,
    mask_value,
    scaled_dot_product_attention,
)
from scratchgrad.testing import gradcheck, rel_error

B, T, S, D, DV = 2, 7, 5, 11, 3


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _jacobian_collapse(rng: np.random.Generator) -> float:
    """Dense ``diag(a) - a a^T`` per row vs ``a * (da - sum(da*a))``."""
    a = _softmax(rng.standard_normal((13, S)) * 2.0)
    da = rng.standard_normal((13, S))
    dense = np.empty_like(da)
    for r in range(a.shape[0]):
        jac = np.diag(a[r]) - np.outer(a[r], a[r])
        dense[r] = jac @ da[r]  # J is symmetric, so J^T da == J da
    collapsed = a * (da - (da * a).sum(axis=-1, keepdims=True))
    return rel_error(dense, collapsed)


def _qkv_grads(rng: np.random.Generator, causal: bool = False) -> dict[str, float]:
    """Hand-derived dQ, dK, dV against the fused node."""
    q = rng.standard_normal((B, T, D))
    k = rng.standard_normal((B, S, D))
    v = rng.standard_normal((B, S, DV))
    g = rng.standard_normal((B, T, DV))
    sc = 1.0 / math.sqrt(D)

    scores = (q @ np.swapaxes(k, -1, -2)) * sc
    keep = causal_mask(T, S) if causal else None
    if keep is not None:
        scores = np.where(keep, scores, mask_value(scores.dtype))
    a = _softmax(scores)
    da = g @ np.swapaxes(v, -1, -2)
    ds = a * (da - (da * a).sum(axis=-1, keepdims=True))
    if keep is not None:
        ds = np.where(keep, ds, 0.0)
    ds = ds * sc
    dq, dk = ds @ k, np.swapaxes(ds, -1, -2) @ q
    dv = np.swapaxes(a, -1, -2) @ g

    tq = sg.Tensor(q, requires_grad=True, dtype=np.float64)
    tk = sg.Tensor(k, requires_grad=True, dtype=np.float64)
    tv = sg.Tensor(v, requires_grad=True, dtype=np.float64)
    out, weights = scaled_dot_product_attention(tq, tk, tv, is_causal=causal, return_weights=True)
    fwd = rel_error(out.data, a @ v)
    out.backward(g)
    return {
        "forward": fwd,
        "weights": rel_error(weights, a),
        "dq": rel_error(tq.grad, dq),
        "dk": rel_error(tk.grad, dk),
        "dv": rel_error(tv.grad, dv),
    }


def _score_variance(rng: np.random.Generator, d: int, trials: int = 20000) -> tuple[float, float]:
    """Sample var of q.k and of q.k/sqrt(d) for unit-variance entries."""
    qq = rng.standard_normal((trials, d))
    kk = rng.standard_normal((trials, d))
    dots = (qq * kk).sum(axis=1)
    return float(dots.var()), float((dots / math.sqrt(d)).var())


def _saturation(rng: np.random.Generator, d: int = 128, s: int = 64) -> tuple[float, ...]:
    """Saturation and the size of the softmax Jacobian, unscaled vs scaled.

    ``trace(diag(a) - a a^T) = 1 - sum a^2`` is a one-number proxy for how much
    gradient the softmax will pass. It is 0 for a one-hot row and ``1 - 1/S`` for
    a uniform one, so it reads directly as "how dead is this layer".
    """
    qq = rng.standard_normal((256, d))
    kk = rng.standard_normal((s, d))
    raw = qq @ kk.T
    a_raw, a_sc = _softmax(raw), _softmax(raw / math.sqrt(d))
    jac = lambda a: float((1.0 - (a * a).sum(axis=-1)).mean())  # noqa: E731
    return (
        float(a_raw.max(axis=-1).mean()),
        float(a_sc.max(axis=-1).mean()),
        jac(a_raw),
        jac(a_sc),
    )


def _fully_masked_row() -> tuple[float, float, float]:
    """A row where nothing may be attended to. -inf vs finfo.min/2."""
    scores = np.zeros((1, 4))
    with np.errstate(invalid="ignore"):
        inf_row = _softmax(np.full_like(scores, -np.inf))
        # The popular "guard the max with isfinite" patch: the shift is finite,
        # but exp(-inf) is 0 everywhere and the normalisation is still 0/0.
        m = np.full_like(scores, -np.inf)
        guarded = np.exp(m - 0.0)
        guarded = guarded / guarded.sum(axis=-1, keepdims=True)
    finite = _softmax(np.full_like(scores, mask_value(scores.dtype)))
    return (
        float(np.isnan(inf_row).sum()),
        float(np.isnan(guarded).sum()),
        float(np.max(np.abs(finite - 0.25))),
    )


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 07-attention.md."""
    rng = np.random.default_rng(seed)
    plain = _qkv_grads(rng, causal=False)
    causal = _qkv_grads(rng, causal=True)
    var_raw_64, var_scaled_64 = _score_variance(rng, 64)
    var_raw_11, var_scaled_11 = _score_variance(rng, 11)
    sat_raw, sat_scaled, jac_raw, jac_scaled = _saturation(rng)
    nan_inf, nan_guarded, uniform_err = _fully_masked_row()

    tq = sg.Tensor(rng.standard_normal((B, T, D)), requires_grad=True, dtype=np.float64)
    tk = sg.Tensor(rng.standard_normal((B, S, D)), requires_grad=True, dtype=np.float64)
    tv = sg.Tensor(rng.standard_normal((B, S, DV)), requires_grad=True, dtype=np.float64)
    fd = max(
        gradcheck(
            lambda a, b, c: scaled_dot_product_attention(a, b, c),
            tq,
            tk,
            tv,
            tol=1e-7,
            name="sdpa",
        ).values()
    )

    return [
        Check(
            "softmax_jacobian_collapse",
            _jacobian_collapse(rng),
            1e-15,
            note="dense (diag(a) - a a^T) da == a*(da - sum(da*a))",
        ),
        Check("forward_value", plain["forward"], 1e-15, note="out == softmax(qk^T/sqrt(d)) v"),
        Check("weights_match", plain["weights"], 1e-15, note="returned weights == our softmax"),
        Check("dQ", plain["dq"], 1e-14, note="dS . K, with dS the collapsed softmax VJP"),
        Check("dK", plain["dk"], 1e-14, note="dS^T . Q"),
        Check("dV", plain["dv"], 1e-15, note="A^T . dOut"),
        Check("dQ_causal", causal["dq"], 1e-14, note="same, with the causal mask applied"),
        Check("dK_causal", causal["dk"], 1e-14, note="and dS zeroed outside the mask"),
        Check("finite_differences", fd, 1e-7, note="gradcheck on the fused sdpa node"),
        Check(
            "dot_variance_is_d_64",
            var_raw_64 / 64.0,
            1.05,
            kind="below",
            note=f"Var(q.k) = {var_raw_64:.2f} at d=64 (theory: 64)",
            extras={"var_raw_d64": var_raw_64, "var_scaled_d64": var_scaled_64},
        ),
        Check(
            "scaled_variance_is_one_64",
            abs(var_scaled_64 - 1.0),
            0.05,
            note=f"Var(q.k/sqrt(d)) = {var_scaled_64:.3f} at d=64",
        ),
        Check(
            "scaled_variance_is_one_11",
            abs(var_scaled_11 - 1.0),
            0.05,
            note=f"and {var_scaled_11:.3f} at d=11 -- the scale is d-independent",
            extras={"var_raw_d11": var_raw_11},
        ),
        Check(
            "unscaled_softmax_saturates",
            sat_raw,
            0.8,
            kind="above",
            note=f"mean max weight {sat_raw:.3f} unscaled vs {sat_scaled:.3f} scaled (d=128)",
            extras={"max_weight_scaled": sat_scaled},
        ),
        Check(
            "unscaled_jacobian_collapses",
            jac_scaled / jac_raw,
            3.0,
            kind="above",
            note=f"mean trace(J): {jac_raw:.4f} unscaled vs {jac_scaled:.4f} scaled",
            extras={"jac_trace_raw": jac_raw, "jac_trace_scaled": jac_scaled},
        ),
        Check(
            "neg_inf_makes_nan",
            nan_inf,
            4.0,
            kind="close",
            note="all four entries of a fully-masked row are NaN",
        ),
        Check(
            "isfinite_guard_still_nan",
            nan_guarded,
            4.0,
            kind="close",
            note="the popular max-guard patch does not fix it: 0/0",
        ),
        Check(
            "finite_sentinel_is_uniform",
            uniform_err,
            1e-15,
            note="finfo.min/2 gives a finite uniform 1/S row instead",
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "07 attention"))
