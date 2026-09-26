"""Verify the claims in ``05-conv2d.md``.

1. The output-size formula, including the dilation term, over a sweep of 81
   geometries -- compared against the shape ``conv2d`` actually returns.
2. im2col + GEMM equals a naive six-loop cross-correlation (a forward-value
   check: gradcheck alone cannot catch a self-consistently wrong forward).
3. ``dW = dY . col^T`` and ``dX = col2im(W^T . dY)``, hand-computed, against the
   framework's backward.
4. ``col2im`` must **accumulate**: ``col2im(im2col(x)) == x * window_count``,
   and the fancy-assignment version that overwrites is measurably wrong.
5. The adjoint identity ``<v, Au> == <A^T v, u>``, which needs no step size at
   all and is the sharpest available check on a linear map.

    PYTHONPATH=src python3 notes/derivations/verify/verify_05_conv2d.py
"""

from __future__ import annotations

import itertools

import numpy as np
from _harness import Check, cli

import scratchgrad as sg
from scratchgrad import nn
from scratchgrad.nn.conv import col2im, conv2d, conv_output_size, im2col
from scratchgrad.testing import adjoint_test, check_module, rel_error

N, CIN, COUT, HI, WI = 3, 5, 7, 11, 13


def _output_size_sweep() -> tuple[int, int]:
    """Compare the formula against the real output shape over every geometry."""
    bad = total = 0
    for k, s, p, d in itertools.product((1, 2, 3), (1, 2, 3), (0, 1, 2), (1, 2, 3)):
        eff = d * (k - 1) + 1
        if HI + 2 * p < eff or WI + 2 * p < eff:
            continue
        total += 1
        want_h = conv_output_size(HI, p, p, k, s, d)
        want_w = conv_output_size(WI, p, p, k, s, d)
        x = sg.Tensor(np.zeros((1, 1, HI, WI)), dtype=np.float64)
        w = sg.Tensor(np.zeros((1, 1, k, k)), dtype=np.float64)
        got = conv2d(x, w, stride=s, padding=p, dilation=d).data.shape
        bad += int(got[2:] != (want_h, want_w))
    return bad, total


def _naive_forward(
    rng: np.random.Generator, k: int = 3, s: int = 2, p: int = 1, d: int = 2
) -> float:
    """Six explicit loops. Slow, obviously correct, and independent of im2col."""
    x = rng.standard_normal((N, CIN, HI, WI))
    w = rng.standard_normal((COUT, CIN, k, k))
    b = rng.standard_normal(COUT)
    oh = conv_output_size(HI, p, p, k, s, d)
    ow = conv_output_size(WI, p, p, k, s, d)
    xp = np.zeros((N, CIN, HI + 2 * p, WI + 2 * p))
    xp[:, :, p : p + HI, p : p + WI] = x
    ref = np.zeros((N, COUT, oh, ow))
    for n in range(N):
        for m in range(COUT):
            for i in range(oh):
                for j in range(ow):
                    acc = b[m]
                    for c in range(CIN):
                        for a in range(k):
                            for e in range(k):
                                acc += w[m, c, a, e] * xp[n, c, i * s + a * d, j * s + e * d]
                    ref[n, m, i, j] = acc
    got = conv2d(
        sg.Tensor(x, dtype=np.float64),
        sg.Tensor(w, dtype=np.float64),
        sg.Tensor(b, dtype=np.float64),
        stride=s,
        padding=p,
        dilation=d,
    ).data
    return rel_error(got, ref)


def _im2col_backward(
    rng: np.random.Generator, k: int = 3, s: int = 2, p: int = 1, d: int = 2
) -> tuple[float, float, float]:
    """dW, dX and db from the im2col formulation, against the framework."""
    x = rng.standard_normal((N, CIN, HI, WI))
    w = rng.standard_normal((COUT, CIN, k, k))
    b = rng.standard_normal(COUT)
    oh = conv_output_size(HI, p, p, k, s, d)
    ow = conv_output_size(WI, p, p, k, s, d)
    dy = rng.standard_normal((N, COUT, oh, ow))

    cols = im2col(x, k, stride=s, padding=p, dilation=d)  # (N, CK, L)
    wmat = w.reshape(COUT, -1)  # (M, CK)
    gy = dy.reshape(N, COUT, oh * ow)  # (N, M, L)
    dw = np.einsum("nml,ncl->mc", gy, cols, optimize=True).reshape(w.shape)
    dcols = np.einsum("mc,nml->ncl", wmat, gy, optimize=True)
    dx = col2im(dcols, x.shape, k, stride=s, padding=p, dilation=d)
    db = dy.sum(axis=(0, 2, 3))

    tx = sg.Tensor(x, requires_grad=True, dtype=np.float64)
    tw = sg.Tensor(w, requires_grad=True, dtype=np.float64)
    tb = sg.Tensor(b, requires_grad=True, dtype=np.float64)
    conv2d(tx, tw, tb, stride=s, padding=p, dilation=d).backward(dy)
    return rel_error(tw.grad, dw), rel_error(tx.grad, dx), rel_error(tb.grad, db)


def _accumulation(rng: np.random.Generator, k: int = 3, s: int = 1, p: int = 1) -> tuple:
    """col2im(im2col(x)) is x times the per-pixel window count -- never x."""
    x = rng.standard_normal((1, 1, HI, WI))
    cols = im2col(x, k, stride=s, padding=p)
    back = col2im(cols, x.shape, k, stride=s, padding=p)
    counts = col2im(np.ones_like(cols), x.shape, k, stride=s, padding=p)
    interior = float(counts.max())

    # The overwriting version: keep only the last window that touches a pixel.
    oh = conv_output_size(HI, p, p, k, s, 1)
    ow = conv_output_size(WI, p, p, k, s, 1)
    win = cols.reshape(1, 1, k, k, oh, ow)
    over = np.zeros((1, 1, HI + 2 * p, WI + 2 * p))
    for a in range(k):
        for e in range(k):
            over[:, :, a : a + (oh - 1) * s + 1 : s, e : e + (ow - 1) * s + 1 : s] = win[:, :, a, e]
    over = over[:, :, p : p + HI, p : p + WI]
    return rel_error(back, x * counts), rel_error(back, over), interior


def _adjoint(rng: np.random.Generator, k: int = 3, s: int = 2, p: int = 1) -> float:
    """<v, conv(u)> == <conv^T(v), u>, exact to floating-point summation."""
    w = sg.Tensor(rng.standard_normal((COUT, CIN, k, k)), dtype=np.float64)
    oh = conv_output_size(HI, p, p, k, s, 1)
    ow = conv_output_size(WI, p, p, k, s, 1)

    def forward(u: sg.Tensor) -> sg.Tensor:
        return conv2d(u, w, stride=s, padding=p)

    def backward_vjp(u: sg.Tensor, v: np.ndarray) -> np.ndarray:
        u.grad = None
        conv2d(u, w, stride=s, padding=p).backward(v)
        return u.grad

    return adjoint_test(forward, backward_vjp, (N, CIN, HI, WI), (N, COUT, oh, ow), tol=1e-12)


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 05-conv2d.md."""
    rng = np.random.default_rng(seed)
    bad, total = _output_size_sweep()
    dw_err, dx_err, db_err = _im2col_backward(rng)
    acc_err, over_err, interior = _accumulation(rng)
    data = np.random.default_rng(4242).standard_normal((N, CIN, HI, WI))
    mod_errs = check_module(
        nn.Conv2d(CIN, COUT, 3, stride=2, padding=1, dilation=2, rng=np.random.default_rng(1)),
        (N, CIN, HI, WI),
        tol=1e-7,
        input_data=data,
        name="Conv2d",
    )
    return [
        Check(
            "output_size_formula",
            float(bad),
            0.0,
            kind="close",
            note=f"mismatches over {total} (k, stride, pad, dilation) geometries",
            extras={"geometries": total},
        ),
        Check(
            "im2col_vs_naive_loops",
            _naive_forward(rng),
            1e-14,
            note="forward-value check against six explicit loops",
        ),
        Check("dW_is_dY_dot_col", dw_err, 1e-14, note="dW = dY . col^T, reshaped"),
        Check("dX_is_col2im", dx_err, 1e-14, note="dX = col2im(W^T . dY)"),
        Check("db_sums_spatial", db_err, 1e-15, note="db = sum over (N, H_out, W_out)"),
        Check(
            "col2im_accumulates",
            acc_err,
            1e-15,
            note=f"col2im(im2col(x)) == x * count (max count {interior:.0f} at k=3,s=1,p=1)",
            extras={"max_window_count": interior},
        ),
        Check(
            "overwrite_version_is_wrong",
            over_err,
            1e-2,
            kind="above",
            note="fancy assignment keeps the last write; this is how far off that is",
        ),
        Check(
            "adjoint_identity",
            _adjoint(rng),
            1e-12,
            note="<v, Au> == <A^T v, u>, no step size involved",
        ),
        Check(
            "finite_differences",
            max(mod_errs.values()),
            1e-7,
            note="check_module on Conv2d with stride 2, pad 1, dilation 2",
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "05 conv2d"))
