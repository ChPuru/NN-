"""Verify the claims in ``02-softmax-cross-entropy.md``.

1. ``d/dz [logsumexp(z) - z_y]`` equals ``p - onehot``, and with ``reduction=
   'mean'`` the framework divides it by N.
2. That expression matches central differences (``gradcheck``).
3. The label-smoothing generalisation ``p - q`` with the smoothed target ``q``.
4. The naive ``-log(p + eps)`` form: its loss saturates at ``log(1/eps)`` and
   its gradient is the correct one scaled by ``p_y / (p_y + eps)``, which for a
   badly-wrong sample is *zero* -- measured, not asserted.

    PYTHONPATH=src python3 notes/derivations/verify/verify_02_softmax_cross_entropy.py
"""

from __future__ import annotations

import math

import numpy as np
from _harness import Check, cli

import scratchgrad as sg
from scratchgrad import nn
from scratchgrad.testing import gradcheck, rel_error

N, C = 7, 11


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _closed_form(rng: np.random.Generator) -> tuple[float, float]:
    """(p - onehot)/N against the framework, for reduction='mean' and 'sum'."""
    z = rng.standard_normal((N, C)) * 3.0
    y = rng.integers(0, C, size=N)
    onehot = np.zeros((N, C))
    onehot[np.arange(N), y] = 1.0
    p = _softmax(z)

    errs = []
    for reduction, scale in (("mean", float(N)), ("sum", 1.0)):
        t = sg.Tensor(z, requires_grad=True, dtype=np.float64)
        nn.functional.cross_entropy(t, y, reduction=reduction).backward()
        errs.append(rel_error(t.grad, (p - onehot) / scale))
    return errs[0], errs[1]


def _forward_value(rng: np.random.Generator) -> float:
    """logsumexp(z) - z_y, computed independently, against the framework's loss."""
    z = rng.standard_normal((N, C)) * 3.0
    y = rng.integers(0, C, size=N)
    mx = z.max(axis=1, keepdims=True)
    lse = (mx + np.log(np.exp(z - mx).sum(axis=1, keepdims=True))).reshape(-1)
    mine = float(np.mean(lse - z[np.arange(N), y]))
    theirs = nn.functional.cross_entropy(sg.Tensor(z, dtype=np.float64), y, reduction="mean").item()
    return abs(mine - theirs) / (abs(mine) + abs(theirs))


def _smoothed(rng: np.random.Generator, eps: float = 0.1) -> float:
    """With smoothing the target is q = (1-eps) onehot + eps/C, and dL/dz = (p - q)/N."""
    z = rng.standard_normal((N, C)) * 2.0
    y = rng.integers(0, C, size=N)
    q = np.full((N, C), eps / C)
    q[np.arange(N), y] += 1.0 - eps
    t = sg.Tensor(z, requires_grad=True, dtype=np.float64)
    nn.functional.cross_entropy(t, y, label_smoothing=eps).backward()
    return rel_error(t.grad, (_softmax(z) - q) / N)


def _gradcheck(rng: np.random.Generator) -> float:
    z = sg.Tensor(rng.standard_normal((N, C)), requires_grad=True, dtype=np.float64)
    y = rng.integers(0, C, size=N)
    errs = gradcheck(
        lambda t: nn.functional.cross_entropy(t, y, reduction="none"),
        z,
        tol=1e-9,
        name="cross_entropy",
        raise_on_fail=False,
    )
    return max(errs.values())


def _clipped_gradient(gap: float = 60.0, eps: float = 1e-12) -> dict[str, float]:
    """The ``-log(p + eps)`` fudge, measured on one deliberately very wrong sample.

    Put the true class ``gap`` nats below the winner. Then ``p_y = e^-gap``,
    which for ``gap = 60`` is ``8.8e-27`` -- twelve orders below ``eps``. The
    stable loss is ``gap``; the clipped one reports ``log(1/eps) = 27.6`` and
    stops moving, because

        d/dz -log(p_y + eps) = -(p_y / (p_y + eps)) * (onehot - p)

    and the prefactor is ``8.8e-15``. Both numbers below are central
    differences of each forward pass, so neither is taken on faith.
    """
    z = np.zeros((1, 2))
    z[0, 0] = gap  # the wrong class wins by `gap`
    y = 1  # ... and the true class is the loser

    def stable(zz: np.ndarray) -> float:
        mx = zz.max()
        return float(mx + np.log(np.exp(zz - mx).sum()) - zz[0, y])

    def clipped(zz: np.ndarray) -> float:
        return float(-np.log(_softmax(zz)[0, y] + eps))

    h = 1e-5
    out: dict[str, float] = {}
    for name, fn in (("stable", stable), ("clipped", clipped)):
        zp, zm = z.copy(), z.copy()
        zp[0, y] += h
        zm[0, y] -= h
        out[f"{name}_loss"] = fn(z)
        out[f"{name}_grad"] = (fn(zp) - fn(zm)) / (2 * h)
    p_y = float(_softmax(z)[0, y])
    out["p_y"] = p_y
    out["prefactor"] = p_y / (p_y + eps)
    out["log_1_over_eps"] = math.log(1.0 / eps)
    return out


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 02-softmax-cross-entropy.md."""
    rng = np.random.default_rng(seed)
    mean_err, sum_err = _closed_form(rng)
    c = _clipped_gradient()
    return [
        Check(
            "forward_is_logsumexp",
            _forward_value(rng),
            1e-15,
            note="loss == mean(logsumexp(z) - z_y)",
        ),
        Check("dz_mean", mean_err, 1e-15, note="framework grad == (p - onehot)/N"),
        Check("dz_sum", sum_err, 1e-15, note="reduction='sum' drops the 1/N"),
        Check("dz_label_smoothing", _smoothed(rng), 1e-15, note="== (p - q)/N, q smoothed"),
        Check(
            "finite_differences",
            _gradcheck(rng),
            1e-9,
            note="and the closed form survives central differences",
        ),
        Check(
            "clipped_loss_capped",
            c["clipped_loss"],
            c["log_1_over_eps"],
            kind="close",
            note=f"-log(p+1e-12) reports {c['clipped_loss']:.4f} for a true loss of "
            f"{c['stable_loss']:.1f}",
        ),
        Check(
            "clipped_grad_is_zero",
            abs(c["clipped_grad"]),
            1e-13,
            note=f"prefactor p_y/(p_y+eps) = {c['prefactor']:.3e}",
            extras={"p_y": c["p_y"], "prefactor": c["prefactor"]},
        ),
        Check(
            "stable_grad_is_minus_one",
            abs(c["stable_grad"] + 1.0),
            1e-9,
            note="the stable form still says 'raise this logit', at full strength",
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "02 softmax cross-entropy"))
