"""Tests for :mod:`scratchgrad.nn.vae`.

* ``gradcheck`` on each fused op (``reparameterize``, ``gaussian_kl``) at
  ``TOL_OP``, and on the whole negative ELBO through a tiny VAE's parameters at
  ``TOL_COMPOSED``. The noise ``eps`` is hoisted out of the checked function:
  drawing it inside would give every finite-difference probe a different
  forward pass.
* the closed-form KL against an independent Monte-Carlo estimate of
  ``E_q[log q(z) - log p(z)]`` -- gradcheck cannot catch a wrong forward whose
  backward was derived consistently from it.
* the reduction pinned against a plain-numpy reference: reconstruction summed
  over pixels, KL summed over latents, both averaged over the batch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import scratchgrad as sg
from scratchgrad.nn import functional as F
from scratchgrad.nn.vae import VAE, gaussian_kl, negative_elbo, reparameterize
from scratchgrad.tensor import Tensor
from scratchgrad.testing import TOL_COMPOSED, TOL_OP, gradcheck

N, L = 7, 5


def _f64(rng: np.random.Generator, *shape: int, scale: float = 1.0) -> Tensor:
    return Tensor(rng.standard_normal(shape) * scale, requires_grad=True, dtype=np.float64)


# --------------------------------------------------------------------------- #
# reparameterize
# --------------------------------------------------------------------------- #
def test_reparameterize_forward_uses_half_logvar(rng, f64):
    mu, lv = _f64(rng, N, L), _f64(rng, N, L)
    eps = rng.standard_normal((N, L))
    z = reparameterize(mu, lv, eps=eps)
    assert np.allclose(z.data, mu.data + np.sqrt(np.exp(lv.data)) * eps, rtol=0, atol=1e-14)


def test_reparameterize_gradcheck_with_fixed_eps(rng, f64):
    mu, lv = _f64(rng, N, L), _f64(rng, N, L, scale=0.5)
    eps = rng.standard_normal((N, L))  # hoisted: every probe sees the same noise
    errs = gradcheck(lambda m, v: reparameterize(m, v, eps=eps), mu, lv, tol=TOL_OP)
    assert max(errs.values()) < TOL_OP


def test_reparameterize_samples_have_the_right_moments(f64):
    """With the rng path, z ~ N(mu, exp(logvar)): exp(logvar) as the std would fail."""
    mu = Tensor(np.array([0.3, -1.2, 2.0]), dtype=np.float64)
    lv = Tensor(np.array([0.0, 1.0, -2.0]), dtype=np.float64)
    m = 200_000
    z = reparameterize(
        Tensor(np.broadcast_to(mu.data, (m, 3)).copy()),
        Tensor(np.broadcast_to(lv.data, (m, 3)).copy()),
        rng=np.random.default_rng(0),
    ).data
    var = np.exp(lv.data)
    # 6 standard errors; the wrong-std version misses var by e^1 - 1 = 172% in column 1.
    assert np.all(np.abs(z.mean(0) - mu.data) < 6 * np.sqrt(var / m))
    assert np.all(np.abs(z.var(0) / var - 1.0) < 6 * math.sqrt(2.0 / m))


def test_reparameterize_randomness_is_explicit_and_reproducible(rng, f64):
    mu, lv = _f64(rng, N, L), _f64(rng, N, L)
    with pytest.raises(ValueError):
        reparameterize(mu, lv)
    with pytest.raises(ValueError):
        reparameterize(mu, lv, rng=rng, eps=np.zeros((N, L)))
    a = reparameterize(mu, lv, rng=np.random.default_rng(3)).data
    b = reparameterize(mu, lv, rng=np.random.default_rng(3)).data
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# gaussian_kl
# --------------------------------------------------------------------------- #
def test_kl_is_exactly_zero_at_the_prior(f64):
    z = Tensor(np.zeros((N, L)), requires_grad=True)
    lz = Tensor(np.zeros((N, L)), requires_grad=True)
    kl = gaussian_kl(z, lz)
    assert kl.shape == (N,)
    assert np.all(kl.data == 0.0)
    kl.sum().backward()
    assert np.all(z.grad == 0.0) and np.all(lz.grad == 0.0)  # the prior is the minimum


def test_kl_positive_away_from_the_prior(rng, f64):
    kl = gaussian_kl(_f64(rng, N, L), _f64(rng, N, L))
    assert np.all(kl.data > 0.0)


def test_kl_gradcheck(rng, f64):
    mu, lv = _f64(rng, N, L), _f64(rng, N, L, scale=0.5)
    errs = gradcheck(gaussian_kl, mu, lv, tol=TOL_OP)
    assert max(errs.values()) < TOL_OP


def test_kl_closed_form_matches_monte_carlo(f64):
    """KL = E_q[log q(z) - log p(z)], estimated from 400k samples of q."""
    mu = np.array([0.7, -1.5, 0.0, 2.0])
    lv = np.array([-1.0, 0.5, 1.5, -0.2])
    closed = float(gaussian_kl(Tensor(mu[None]), Tensor(lv[None])).data[0])

    r = np.random.default_rng(1)
    m = 400_000
    std = np.exp(0.5 * lv)
    z = mu + std * r.standard_normal((m, mu.size))
    log_q = -0.5 * (((z - mu) / std) ** 2 + lv + math.log(2 * math.pi)).sum(1)
    log_p = -0.5 * (z**2 + math.log(2 * math.pi)).sum(1)
    samples = log_q - log_p
    se = samples.std() / math.sqrt(m)
    assert abs(samples.mean() - closed) < 5 * se, (closed, samples.mean(), se)


# --------------------------------------------------------------------------- #
# the whole negative ELBO
# --------------------------------------------------------------------------- #
def _tiny(seed: int = 0) -> tuple[VAE, np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed)
    with sg.default_dtype(np.float64):
        model = VAE(in_features=11, hidden=7, latent=3, rng=r).double()
    x = (r.random((5, 11)) < 0.4).astype(np.float64)
    eps = r.standard_normal((5, 3))
    return model, x, eps


def test_negative_elbo_matches_numpy_reference(f64):
    model, x, eps = _tiny()
    total, recon, kl = model.loss(Tensor(x), eps=eps)
    p = {k: v.data for k, v in model.named_parameters()}
    relu = lambda a: np.maximum(a, 0.0)  # noqa: E731
    h = relu(x @ p["enc.weight"] + p["enc.bias"])
    mu = h @ p["enc_mu.weight"] + p["enc_mu.bias"]
    lv = h @ p["enc_logvar.weight"] + p["enc_logvar.bias"]
    z = mu + np.exp(0.5 * lv) * eps
    logits = relu(z @ p["dec.weight"] + p["dec.bias"]) @ p["dec_out.weight"] + p["dec_out.bias"]
    # -log Bernoulli(x | sigmoid(l)) = log(1 + e^l) - x*l, SUMMED over pixels, per image
    recon_ref = (np.logaddexp(0.0, logits) - x * logits).sum(1).mean()
    kl_ref = (0.5 * (mu**2 + np.exp(lv) - 1.0 - lv)).sum(1).mean()
    assert abs(recon.item() - recon_ref) < 1e-12
    assert abs(kl.item() - kl_ref) < 1e-12
    assert abs(total.item() - (recon_ref + kl_ref)) < 1e-12


def test_recon_is_summed_over_pixels_not_averaged(f64):
    """The trap: reduction='mean' is recon / D, i.e. a beta-VAE with beta = D."""
    model, x, eps = _tiny()
    logits, mu, lv = model(Tensor(x), eps=eps)
    _, recon, _ = negative_elbo(logits, x, mu, lv)
    mean_red = F.binary_cross_entropy_with_logits(logits, x).item()
    assert abs(recon.item() / mean_red - x.shape[1]) < 1e-9


def test_negative_elbo_gradcheck_through_the_model(f64):
    model, x, eps = _tiny()
    params = list(model.parameters())
    xt = Tensor(x)
    errs = gradcheck(
        lambda *_: model.loss(xt, eps=eps)[0], *params, tol=TOL_COMPOSED, copy_inputs=False
    )
    assert max(errs.values()) < TOL_COMPOSED


def test_shapes_default_model(rng):
    model = VAE(rng=rng)
    x = Tensor((rng.random((3, 784)) < 0.5).astype(np.float32))
    logits, mu, lv = model(x, rng=rng)
    assert logits.shape == (3, 784) and mu.shape == (3, 20) and lv.shape == (3, 20)
    assert model.num_parameters() == 784 * 400 + 400 + 2 * (400 * 20 + 20) + 20 * 400 + 400 + (
        400 * 784 + 784
    )
