"""Verify the claims in ``04-batchnorm.md``.

1. The same three-term backward as LayerNorm, with the group now running over
   ``(N, H, W)`` for a fixed channel instead of over a row.
2. Central differences through the *training* path (``track_running_stats=
   False``, so ``eval()`` still uses batch statistics -- ``check_module`` puts
   the module in ``eval()``).
3. The batch is normalised with the **biased** variance while ``running_var``
   accumulates the **unbiased** one, and the ``n/(n-1)`` gap is measured at the
   ``n = 8`` where it is worth 14%.
4. The frozen-stats backward ``dx = dxhat * istd`` is a *different* function
   from the training one: the size of the difference is measured. Using it
   during training is the classic BatchNorm bug that still descends.

    PYTHONPATH=src python3 notes/derivations/verify/verify_04_batchnorm.py
"""

from __future__ import annotations

import numpy as np
from _harness import Check, cli

import scratchgrad as sg
from scratchgrad import nn
from scratchgrad.testing import check_module, rel_error

N, C, H, W = 7, 5, 3, 2
AXES = (0, 2, 3)  # everything except the channel axis
EPS = 1e-5
BCAST = (1, C, 1, 1)


def _pieces(rng: np.random.Generator):
    """A BatchNorm2d training-mode forward, in float64, with a random cotangent."""
    x = rng.standard_normal((N, C, H, W)) * 2.0 + 0.3
    g = rng.standard_normal(C) * 0.5 + 1.0
    b = rng.standard_normal(C) * 0.1
    dy = rng.standard_normal((N, C, H, W))
    mu = x.mean(axis=AXES, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=AXES, keepdims=True)  # biased, denominator n
    istd = 1.0 / np.sqrt(var + EPS)
    xhat = xc * istd
    return x, g, b, dy, xc, var, istd, xhat


def _train_backward(rng: np.random.Generator):
    """Hand three-term dx / dg / db against the framework's train-mode backward."""
    x, g, b, dy, _xc, _var, istd, xhat = _pieces(rng)
    dxhat = dy * g.reshape(BCAST)
    dx = istd * (
        dxhat
        - dxhat.mean(axis=AXES, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=AXES, keepdims=True)
    )

    bn = nn.BatchNorm2d(C, eps=EPS).double()
    bn.weight.data[...] = g
    bn.bias.data[...] = b
    bn.train()
    tx = sg.Tensor(x, requires_grad=True, dtype=np.float64)
    out = bn(tx)
    fwd_err = rel_error(out.data, xhat * g.reshape(BCAST) + b.reshape(BCAST))
    out.backward(dy)
    return (
        fwd_err,
        rel_error(tx.grad, dx),
        rel_error(bn.weight.grad, (dy * xhat).sum(axis=AXES)),
        rel_error(bn.bias.grad, dy.sum(axis=AXES)),
    )


def _frozen_vs_train(rng: np.random.Generator) -> float:
    """How far ``dxhat * istd`` is from the true training gradient (relative)."""
    _x, g, _b, dy, _xc, _var, istd, xhat = _pieces(rng)
    dxhat = dy * g.reshape(BCAST)
    dx_train = istd * (
        dxhat
        - dxhat.mean(axis=AXES, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=AXES, keepdims=True)
    )
    dx_frozen = dxhat * istd
    return rel_error(dx_train, dx_frozen)


def _running_stats(rng: np.random.Generator, n: int = 8) -> tuple[float, float, float]:
    """momentum=1.0 makes the running buffers exactly this batch's estimators.

    With ``momentum=1.0`` the EMA keeps none of the previous value, so after one
    forward ``running_var`` *is* the batch estimate -- which is the only way to
    read off which estimator was stored without waiting for convergence.
    """
    x = rng.standard_normal((n, C)) * 1.5 + 0.7
    bn = nn.BatchNorm1d(C, eps=EPS, momentum=1.0).double()
    bn.train()
    bn(sg.Tensor(x, dtype=np.float64))
    unbiased = x.var(axis=0, ddof=1)
    biased = x.var(axis=0, ddof=0)
    stored = np.asarray(bn._buffers["running_var"], dtype=np.float64)
    gap = float(np.max(np.abs(unbiased / biased - 1.0)))
    return rel_error(stored, unbiased), rel_error(stored, biased), gap


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 04-batchnorm.md."""
    rng = np.random.default_rng(seed)
    fwd_err, dx_err, dg_err, db_err = _train_backward(rng)
    to_unbiased, to_biased, bessel_gap = _running_stats(rng)
    # Independent stream for the input: with the same one, <x, bn(x)> is
    # stationary and finite differences have nothing to resolve (see 03).
    data = np.random.default_rng(4242).standard_normal((N, C, H, W))
    mod_errs = check_module(
        nn.BatchNorm2d(C, eps=EPS, track_running_stats=False),
        (N, C, H, W),
        tol=1e-7,
        input_data=data,
        name="BatchNorm2d",
    )
    return [
        Check(
            "forward_uses_biased_var",
            fwd_err,
            1e-15,
            note="train-mode output == (x - mu)/sqrt(var_biased + eps) * g + b",
        ),
        Check("dx_vs_framework", dx_err, 1e-14, note="three-term dx over axes (0,2,3)"),
        Check("dg_vs_framework", dg_err, 1e-15, note="dg == sum over (N,H,W) of dy*xhat"),
        Check("db_vs_framework", db_err, 1e-15, note="db == sum over (N,H,W) of dy"),
        Check(
            "finite_differences",
            max(mod_errs.values()),
            1e-7,
            note="check_module through the batch-statistics path",
        ),
        Check(
            "running_var_is_unbiased",
            to_unbiased,
            1e-15,
            note="running_var == var(x, ddof=1) after one momentum=1 step",
        ),
        Check(
            "running_var_is_not_biased",
            to_biased,
            1e-3,
            kind="above",
            note=f"and differs from the biased estimator by {bessel_gap * 100:.1f}% at n=8",
            extras={"bessel_gap_at_n8": bessel_gap},
        ),
        Check(
            "frozen_stats_backward_differs",
            _frozen_vs_train(rng),
            1e-2,
            kind="above",
            note="dxhat*istd is a different gradient; it still descends, which is the trap",
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "04 batchnorm"))
