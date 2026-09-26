"""Verify the claims in ``10-parallel-scan.md``.

1. The odd-even scan equals the sequential recurrence, short and long (T=4096).
2. The two tempting alternatives fail where the note says: the cumprod/cumsum
   closed form goes non-finite once ``prod(a)`` underflows, and the log-space form
   is exact for positive ``b`` but NaN as soon as ``b`` takes a sign.
3. The backward: the adjoint ``lam_t = g_t + a_{t+1} lam_{t+1}`` from a hand loop
   against ``_scan_adjoint``, the three gradient formulas against both the
   framework and central differences, and the wrong shift (``a_t``) being caught.
4. minGRU: the fused gate derivative ``dk = z(1-z)(lam h~ - lam h_{t-1})``, and the
   float32 cancellation that ``sigmoid(-k)`` avoids.
5. Selective SSM: ZOH keeps the decay in (0, 1) where Euler leaves it, the
   ``B_bar ~ delta B`` error is ``delta|A|/2`` to first order, and the fused
   backward passes finite differences.

    PYTHONPATH=src python notes/derivations/verify/verify_10_parallel_scan.py
"""

from __future__ import annotations

import numpy as np
from _harness import Check, cli

from scratchgrad.nn.scan import _mingru_scan, _scan, _scan_adjoint, linear_scan, selective_scan
from scratchgrad.tensor import Tensor
from scratchgrad.testing import gradcheck, numeric_vjp, rel_error

N, T, D = 5, 13, 3


def _t(arr, grad: bool = False) -> Tensor:
    return Tensor(np.asarray(arr, dtype=np.float64), requires_grad=grad, dtype=np.float64)


def _loop(a, b, h0=None):
    h = np.zeros_like(b[:, 0]) if h0 is None else h0
    out = np.empty_like(b)
    for t in range(b.shape[1]):
        h = a[:, t] * h + b[:, t]
        out[:, t] = h
    return out


def _forward(rng) -> dict[str, float]:
    a = rng.uniform(-1, 1, (N, T, D))
    b = rng.standard_normal((N, T, D))
    h0 = rng.standard_normal((N, D))
    short = rel_error(linear_scan(_t(a), _t(b), _t(h0)).data, _loop(a, b, h0))

    a = rng.uniform(0.3, 0.7, (2, 4096, D))
    b = rng.standard_normal((2, 4096, D))
    ref = _loop(a, b)
    long = rel_error(_scan(a, b), ref)

    # The closed form h = A * cumsum(b / A), A = cumprod(a).
    first_bad = {}
    for dt in (np.float64, np.float32):
        ac, bc = a.astype(dt), b.astype(dt)
        with np.errstate(all="ignore"):
            cum = np.cumprod(ac, axis=1)
            closed = cum * np.cumsum(bc / cum, axis=1)
        bad = ~np.isfinite(closed).all(axis=(0, 2))
        first_bad[dt.__name__] = int(np.argmax(bad)) if bad.any() else -1
    nonfinite = float(np.mean(~np.isfinite(closed)))  # float32, the training dtype

    # Heinsen's log space: log h_t = a*_t + logcumsumexp(log b - a*), a* = cumsum(log a).
    def log_space(bv):
        with np.errstate(all="ignore"):
            astar = np.cumsum(np.log(a), axis=1)
            return np.exp(astar + np.logaddexp.accumulate(np.log(bv) - astar, axis=1))

    b_pos = np.abs(b) + 0.1
    log_ok = rel_error(log_space(b_pos), _loop(a, b_pos))
    log_nan = float(np.mean(np.isnan(log_space(b))))
    return {
        "short": short,
        "long": long,
        "nonfinite": nonfinite,
        "first_bad_f64": first_bad["float64"],
        "first_bad_f32": first_bad["float32"],
        "log_ok": log_ok,
        "log_nan": log_nan,
    }


def _backward(rng) -> dict[str, float]:
    a = rng.uniform(-1, 1, (N, T, D))
    b = rng.standard_normal((N, T, D))
    h0 = rng.standard_normal((N, D))
    g = rng.standard_normal((N, T, D))
    h = _loop(a, b, h0)

    lam = np.empty_like(g)  # the adjoint, straight from its definition
    nxt = np.zeros_like(g[:, 0])
    for t in range(T - 1, -1, -1):
        nxt = g[:, t] + (a[:, t + 1] * nxt if t + 1 < T else 0.0)
        lam[:, t] = nxt
    h_prev = np.concatenate([h0[:, None], h[:, :-1]], axis=1)
    derived = [lam * h_prev, lam, a[:, 0] * lam[:, 0]]

    ts = [_t(a, True), _t(b, True), _t(h0, True)]
    (linear_scan(*ts) * _t(g)).sum().backward()
    numeric = numeric_vjp(lambda *xs: linear_scan(*xs), ts, g)

    # The tempting wrong adjoint: carried back by a_t instead of a_{t+1}.
    wrong = _scan(a[:, ::-1], g[:, ::-1])[:, ::-1]
    return {
        "adjoint": rel_error(_scan_adjoint(a, g), lam),
        "vs_framework": max(rel_error(t.grad, d) for t, d in zip(ts, derived, strict=True)),
        "vs_numeric": max(rel_error(n, d) for n, d in zip(numeric, derived, strict=True)),
        "wrong_shift": rel_error(wrong, lam),
    }


def _mingru(rng) -> dict[str, float]:
    hs = 4
    pre = rng.standard_normal((N, T, 2 * hs))
    g = rng.standard_normal((N, T, hs))
    k, cand = pre[..., :hs], pre[..., hs:]
    z = 1.0 / (1.0 + np.exp(-k))
    a, b = 1.0 - z, z * cand
    h = _loop(a, b)
    lam = _scan_adjoint(a, g)
    h_prev = np.concatenate([np.zeros_like(h[:, :1]), h[:, :-1]], axis=1)
    dk = z * (1.0 - z) * (lam * cand - lam * h_prev)
    want = np.concatenate([dk, z * lam], axis=2)
    tp = _t(pre, True)
    (_mingru_scan(tp) * _t(g)).sum().backward()
    z17 = np.float32(1.0) / (np.float32(1.0) + np.exp(np.float32(-17.0)))
    return {
        "dpre": rel_error(tp.grad, want),
        "one_minus_sigmoid_f32": float(np.float32(1.0) - z17),
        "sigmoid_neg_f32": float(np.float32(1.0) / (np.float32(1.0) + np.exp(np.float32(17.0)))),
    }


def _ssm(rng) -> dict[str, float]:
    deltas = np.logspace(-3, 2, 200)[:, None]
    A = -np.arange(1, 17, dtype=np.float64)[None, :]
    zoh = np.exp(deltas * A)
    euler = 1.0 + deltas * A

    d_small, a_one = 1e-2, -1.0  # delta * |A| = 1e-2
    bbar_zoh = (np.exp(d_small * a_one) - 1.0) / a_one
    bbar_err = abs(bbar_zoh - d_small) / d_small

    x = rng.standard_normal((3, 7, D))
    delta = rng.uniform(0.1, 1.0, (3, 7, D))
    Am = -rng.uniform(0.5, 2.0, (D, 2))
    B = rng.standard_normal((3, 7, 2))
    h0 = rng.standard_normal((3, D, 2))
    errs = gradcheck(
        lambda *ts: selective_scan(*ts),
        *(_t(v, True) for v in (x, delta, Am, B, h0)),
        tol=1e-9,
        raise_on_fail=False,
    )
    return {
        "zoh_max": float(zoh.max()),
        "zoh_min": float(zoh.min()),
        "euler_max_abs": float(np.abs(euler).max()),
        "bbar_ratio": bbar_err / (d_small * abs(a_one) / 2.0),
        "gradcheck": max(errs.values()),
    }


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 10-parallel-scan.md."""
    rng = np.random.default_rng(seed)
    f, b, m, s = _forward(rng), _backward(rng), _mingru(rng), _ssm(rng)
    return [
        Check("scan_vs_loop", f["short"], 1e-14, note="T=13, a in (-1, 1), with h0"),
        Check("scan_vs_loop_T4096", f["long"], 1e-13, note="float64, a in (0.3, 0.7)"),
        Check(
            "closed_form_nonfinite",
            f["nonfinite"],
            0.9,
            kind="above",
            note=f"float32 cumprod form: first NaN at step {f['first_bad_f32']} "
            f"(float64: {f['first_bad_f64']})",
            extras={"first_bad_f32": f["first_bad_f32"], "first_bad_f64": f["first_bad_f64"]},
        ),
        Check("log_space_ok_if_positive", f["log_ok"], 1e-12, note="Heinsen form, b > 0"),
        Check(
            "log_space_nan_if_signed",
            f["log_nan"],
            0.9,
            kind="above",
            note="same form, signed b: fraction of NaN outputs",
        ),
        Check("adjoint_is_reverse_scan", b["adjoint"], 1e-14, note="hand loop vs _scan_adjoint"),
        Check("grads_vs_framework", b["vs_framework"], 1e-14, note="lam*h_prev, lam, a_1*lam_1"),
        Check("grads_vs_numeric", b["vs_numeric"], 1e-8, note="same formulas vs central diffs"),
        Check(
            "wrong_shift_is_caught",
            b["wrong_shift"],
            1e-2,
            kind="above",
            note="carrying lam back by a_t instead of a_{t+1}",
        ),
        Check("mingru_dpre", m["dpre"], 1e-14, note="dk = z(1-z)(lam h~ - da), dh~ = z lam"),
        Check(
            "one_minus_sigmoid_17_f32",
            m["one_minus_sigmoid_f32"],
            0.0,
            note=f"1 - sigmoid(17) is exactly 0 in float32; sigmoid(-17) = "
            f"{m['sigmoid_neg_f32']:.3g}",
        ),
        Check("zoh_decay_below_1", s["zoh_max"], 1.0, note="exp(delta A), delta in [1e-3, 100]"),
        Check("zoh_decay_nonneg", s["zoh_min"], 0.0, kind="above", note="same grid"),
        Check(
            "euler_decay_unstable",
            s["euler_max_abs"],
            2.0,
            kind="above",
            note=f"|1 + delta A| reaches {s['euler_max_abs']:.0f} on the same grid",
        ),
        Check(
            "bbar_error_is_half_delta_A",
            abs(s["bbar_ratio"] - 1.0),
            1e-2,
            note="|B_zoh - delta B| / (delta B) == delta|A|/2 to first order",
        ),
        Check("selective_scan_gradcheck", s["gradcheck"], 1e-9, note="x, delta, A, B, h0"),
    ]


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "10 parallel scan"))
