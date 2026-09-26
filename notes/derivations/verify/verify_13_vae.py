"""Verify the claims in ``13-vae.md``.

1. The reparameterisation gradients ``dz/dmu = 1`` and
   ``dz/dlogvar = 0.5 (z - mu)`` match the framework's backward and central
   differences.
2. The closed-form KL is exactly zero at the prior, its derived gradients
   ``mu`` and ``0.5 (e^logvar - 1)`` match the framework and central
   differences, and its value matches an independent Monte-Carlo estimate.
3. Why the trick exists: the pathwise estimator of ``d/dmu E[z^2]`` has the
   predicted variance 4 against the score-function estimator's 30.
4. ``exp(lv) - 1 - lv`` in float32 is garbage near ``lv = 0``; ``expm1`` is not.
5. The reduction trap: ``reduction="mean"`` divides the reconstruction by D
   exactly, i.e. beta = D.
6. The whole negative ELBO gradchecks through a tiny VAE's parameters.

    PYTHONPATH=src python notes/derivations/verify/verify_13_vae.py
"""

from __future__ import annotations

import math

import numpy as np
from _harness import Check, cli

import scratchgrad as sg
from scratchgrad.nn import functional as F
from scratchgrad.nn.vae import VAE, gaussian_kl, reparameterize
from scratchgrad.testing import gradcheck, rel_error

N, L = 7, 5


def _reparam(rng: np.random.Generator) -> tuple[float, float, float]:
    mu = sg.Tensor(rng.standard_normal((N, L)), requires_grad=True, dtype=np.float64)
    lv = sg.Tensor(rng.standard_normal((N, L)) * 0.5, requires_grad=True, dtype=np.float64)
    eps = rng.standard_normal((N, L))
    v = rng.standard_normal((N, L))
    z = reparameterize(mu, lv, eps=eps)
    z.backward(v)
    d_mu = rel_error(mu.grad, v * 1.0)
    d_lv = rel_error(lv.grad, v * 0.5 * (z.data - mu.data))
    fd = gradcheck(lambda m, s: reparameterize(m, s, eps=eps), mu, lv, tol=float("inf"))
    return d_mu, d_lv, max(fd.values())


def _kl(rng: np.random.Generator) -> tuple[float, float, float, float]:
    zero = gaussian_kl(sg.Tensor(np.zeros((1, L))), sg.Tensor(np.zeros((1, L)))).data[0]
    mu = sg.Tensor(rng.standard_normal((N, L)), requires_grad=True, dtype=np.float64)
    lv = sg.Tensor(rng.standard_normal((N, L)) * 0.5, requires_grad=True, dtype=np.float64)
    v = rng.standard_normal(N)
    gaussian_kl(mu, lv).backward(v)
    derived = rel_error(mu.grad, v[:, None] * mu.data) + rel_error(
        lv.grad, v[:, None] * 0.5 * (np.exp(lv.data) - 1.0)
    )
    fd = gradcheck(gaussian_kl, mu, lv, tol=float("inf"))
    return float(zero), derived, max(fd.values())


def _kl_monte_carlo(rng: np.random.Generator, m: int = 400_000) -> tuple[float, dict]:
    """|closed - MC| in standard errors. MC uses log-densities, not the formula."""
    mu = np.array([0.7, -1.5, 0.0, 2.0])
    lv = np.array([-1.0, 0.5, 1.5, -0.2])
    f64 = np.float64
    closed = float(
        gaussian_kl(sg.Tensor(mu[None], dtype=f64), sg.Tensor(lv[None], dtype=f64)).data[0]
    )
    std = np.exp(0.5 * lv)
    z = mu + std * rng.standard_normal((m, mu.size))
    log_q = -0.5 * (((z - mu) / std) ** 2 + lv + math.log(2 * math.pi)).sum(1)
    log_p = -0.5 * (z**2 + math.log(2 * math.pi)).sum(1)
    s = log_q - log_p
    se = s.std() / math.sqrt(m)
    return abs(s.mean() - closed) / se, {"closed": closed, "mc": float(s.mean()), "se": se}


def _estimators(rng: np.random.Generator, m: int = 400_000) -> tuple[float, float, dict]:
    """d/dmu E_{z~N(mu,1)}[z^2] at mu = 1 (true value 2mu = 2), two ways.

    pathwise:        2 z                      Var = 4
    score function:  z^2 (z - mu)             Var = E[(1+e)^4 e^2] - 4 = 30
    """
    mu = 1.0
    z = mu + rng.standard_normal(m)
    path, score = 2.0 * z, z**2 * (z - mu)
    zs = max(
        abs(path.mean() - 2.0) / (path.std() / math.sqrt(m)),
        abs(score.mean() - 2.0) / (score.std() / math.sqrt(m)),
    )
    return zs, float(score.var() / path.var()), {"var_path": path.var(), "var_score": score.var()}


def _cancellation() -> tuple[float, float]:
    """Relative error of the per-dimension KL term at lv = 1e-4, float32 vs float64 truth."""
    lv = np.float32(1e-4)
    naive = float(np.exp(lv) - np.float32(1.0) - lv)
    fixed = float(np.expm1(lv) - lv)
    # the truth at the float32-ROUNDED input, so only the formula is being judged
    truth = math.expm1(float(lv)) - float(lv)
    return abs(naive - truth) / truth, abs(fixed - truth) / truth


def _reduction(rng: np.random.Generator) -> float:
    d = 784
    logits = sg.Tensor(rng.standard_normal((N, d)), dtype=np.float64)
    x = (rng.random((N, d)) < 0.3).astype(np.float64)
    s = F.binary_cross_entropy_with_logits(logits, x, reduction="sum").item() / N
    m = F.binary_cross_entropy_with_logits(logits, x, reduction="mean").item()
    return s / m


def _elbo_fd(seed: int) -> float:
    r = np.random.default_rng(seed + 1)
    with sg.default_dtype(np.float64):
        model = VAE(in_features=11, hidden=7, latent=3, rng=r).double()
    x = sg.Tensor((r.random((5, 11)) < 0.4).astype(np.float64))
    eps = r.standard_normal((5, 3))  # hoisted: every probe sees the same noise
    errs = gradcheck(
        lambda *_: model.loss(x, eps=eps)[0],
        *model.parameters(),
        tol=float("inf"),
        copy_inputs=False,
    )
    return max(errs.values())


def run(seed: int = 0) -> list[Check]:
    """Measure every claim in 13-vae.md."""
    rng = np.random.default_rng(seed)
    d_mu, d_lv, r_fd = _reparam(rng)
    kl0, kl_derived, kl_fd = _kl(rng)
    mc_z, mc_extras = _kl_monte_carlo(np.random.default_rng(seed + 10))
    est_z, ratio, est_extras = _estimators(np.random.default_rng(seed + 20))
    naive_err, fixed_err = _cancellation()
    beta = _reduction(rng)
    elbo_fd = _elbo_fd(seed)
    return [
        Check("reparam_dmu", d_mu, 1e-15, note="dz/dmu == 1"),
        Check("reparam_dlogvar", d_lv, 1e-15, note="dz/dlogvar == 0.5 (z - mu)"),
        Check("reparam_finite_diff", r_fd, 1e-9, note="gradcheck with eps held fixed"),
        Check("kl_zero_at_prior", kl0, 0.0, kind="close", note="mu = 0, logvar = 0"),
        Check("kl_grad_derived", kl_derived, 1e-15, note="mu and 0.5 (e^lv - 1)"),
        Check("kl_finite_diff", kl_fd, 1e-9, note="gradcheck of the fused KL op"),
        Check(
            "kl_vs_monte_carlo",
            mc_z,
            5.0,
            note="|closed - E_q[log q - log p]| in standard errors, 400k samples",
            extras=mc_extras,
        ),
        Check("estimators_unbiased", est_z, 5.0, note="both estimators hit 2mu, in SEs"),
        Check(
            "score_vs_pathwise_variance",
            ratio,
            5.0,
            kind="above",
            note="predicted 30 / 4 = 7.5",
            extras=est_extras,
        ),
        Check("naive_kl_float32_garbage", naive_err, 0.1, kind="above", note="e^lv - 1 - lv"),
        Check("expm1_kl_float32", fixed_err, 1e-2, note="expm1(lv) - lv at lv = 1e-4"),
        Check("mean_reduction_is_beta_D", beta, 784.0, kind="close", note="sum / mean == D"),
        Check("elbo_finite_diff", elbo_fd, 1e-7, note="whole -ELBO, every VAE parameter"),
    ]


if __name__ == "__main__":
    raise SystemExit(cli(run, __doc__, "13 vae"))
