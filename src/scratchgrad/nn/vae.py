"""Variational autoencoder: the reparameterisation trick, the Gaussian KL, the ELBO.

A VAE trains an encoder ``q(z|x) = N(mu(x), diag(exp(logvar(x))))`` and a
decoder ``p(x|z)`` by maximising the evidence lower bound

    ELBO(x) = E_q[log p(x|z)] - KL(q(z|x) || N(0, I))  <=  log p(x).

The loss here is its negative, in **nats per image**. Three places where a
wrong version runs, trains, and produces plausible pictures:

**The standard deviation is ``exp(0.5 * logvar)``, not ``exp(logvar)``.** The
encoder emits a *log-variance* so that the variance is positive without a
constraint. Using ``exp(logvar)`` as the std squares the noise scale the KL
term thinks it is paying for; the two halves of the loss then describe
different posteriors and nothing raises.

**The reduction is part of the objective.** The ELBO is a *sum* over the 784
pixels of ``log p(x_i|z)`` plus a KL summed over latent dimensions, per image.
Calling a loss with its default ``reduction="mean"`` averages the
reconstruction over pixels as well as over the batch, which divides it by 784
while the KL keeps its full weight -- silently a beta-VAE with ``beta = 784``.
The optimiser then buys KL by throwing information away: the posterior
collapses onto the prior for most latent dimensions and samples go blurry. The
mirror mistake (reconstruction summed, KL averaged over latent dimensions)
down-weights the KL by 20 and the "samples from the prior" stop looking like
digits because the aggregate posterior no longer resembles ``N(0, I)``. Both
terms here are summed within an image and averaged over the batch, once.

**``exp(logvar) - 1 - logvar`` cancels near ``logvar = 0``,** which is exactly
where a collapsed dimension sits. ``expm1(logvar) - logvar`` keeps the digits.

Randomness is always explicit: :func:`reparameterize` takes either a seeded
``numpy.random.Generator`` or the noise itself, never a hidden global stream.
Passing ``eps`` is also how a gradcheck holds the noise fixed across probes.
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from . import functional as F
from .layers import Linear
from .module import Module

__all__ = ["VAE", "gaussian_kl", "negative_elbo", "reparameterize"]


def reparameterize(
    mu: Tensor,
    logvar: Tensor,
    rng: np.random.Generator | None = None,
    eps: np.ndarray | None = None,
) -> Tensor:
    """Sample ``z = mu + exp(0.5 * logvar) * eps`` with ``eps ~ N(0, I)``.

    Sampling ``z ~ N(mu, sigma^2)`` directly has no gradient with respect to
    ``mu`` or ``sigma``. Moving the randomness into a parameter-free ``eps``
    makes ``z`` a deterministic, differentiable function of the parameters:

        dz/dmu = 1,    dz/dlogvar = 0.5 * exp(0.5 * logvar) * eps = 0.5 * (z - mu).

    Args:
        mu: Posterior mean, any shape.
        logvar: Posterior log-variance, same shape as ``mu``.
        rng: Generator to draw ``eps`` from. Exactly one of ``rng``/``eps``.
        eps: The standard-normal noise itself, same shape as ``mu``.

    Returns:
        ``z``, same shape as ``mu``.
    """
    if (rng is None) == (eps is None):
        raise ValueError("pass exactly one of rng= or eps=; there is no hidden global stream")
    if mu.shape != logvar.shape:
        raise ValueError(f"mu {mu.shape} and logvar {logvar.shape} must have the same shape")
    dtype = mu.data.dtype
    if eps is None:
        eps = rng.standard_normal(mu.shape).astype(dtype)
    else:
        eps = np.asarray(eps, dtype=dtype)
        if eps.shape != mu.shape:
            raise ValueError(f"eps {eps.shape} must match mu {mu.shape}")
    std = np.exp(0.5 * logvar.data)  # 0.5: logvar is log sigma^2, so sigma = exp(logvar / 2)
    dlogvar = 0.5 * std * eps
    return Tensor._make(
        mu.data + std * eps,
        (mu, logvar),
        (lambda g: g, lambda g: g * dlogvar),
        "reparameterize",
    )


def gaussian_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    """``KL(N(mu, diag(exp(logvar))) || N(0, I))``, summed over the last axis.

    Per dimension the closed form is ``0.5 * (mu^2 + e^logvar - 1 - logvar)``,
    with gradients ``mu`` and ``0.5 * (e^logvar - 1)``. Both vanish at
    ``mu = 0, logvar = 0`` -- the posterior equal to the prior.

    Args:
        mu: ``(..., latent)``.
        logvar: ``(..., latent)``.

    Returns:
        ``(...)`` -- one KL per sample, in nats. Reduce over the batch yourself.
    """
    if mu.shape != logvar.shape:
        raise ValueError(f"mu {mu.shape} and logvar {logvar.shape} must have the same shape")
    m, lv = mu.data, logvar.data
    em1 = np.expm1(lv)  # e^lv - 1 without cancelling when lv ~ 0 (a collapsed dimension)
    out = 0.5 * (m * m + (em1 - lv)).sum(axis=-1)
    dlv = 0.5 * em1
    return Tensor._make(
        np.asarray(out, dtype=m.dtype),
        (mu, logvar),
        (lambda g: g[..., None] * m, lambda g: g[..., None] * dlv),
        "gaussian_kl",
    )


def negative_elbo(
    logits: Tensor, target, mu: Tensor, logvar: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Negative ELBO with a Bernoulli decoder, in nats per image.

    ``recon = sum over pixels of BCE(logits, target)``, ``kl`` summed over
    latent dimensions, both then averaged over the batch -- the same reduction
    for both, which is the whole point (see the module docstring).

    Args:
        logits: ``(N, D)`` decoder logits; the loss owns the sigmoid.
        target: ``(N, D)`` Bernoulli targets in ``[0, 1]``.
        mu: ``(N, latent)``.
        logvar: ``(N, latent)``.

    Returns:
        ``(total, recon, kl)``, each a scalar Tensor; ``total = recon + kl``.
    """
    n = float(logits.shape[0])
    recon = F.binary_cross_entropy_with_logits(logits, target, reduction="sum") / n
    kl = gaussian_kl(mu, logvar).sum() / n
    return recon + kl, recon, kl


class VAE(Module):
    """MLP VAE: ``D -> hidden -> (mu, logvar)``, ``latent -> hidden -> D`` logits.

    Defaults are Kingma & Welling's MNIST setup: 784 -> 400 -> 20 -> 400 -> 784.
    The decoder emits **logits** of a Bernoulli per pixel; ``sigmoid`` is owned
    by the loss, and applied explicitly only when drawing pictures.
    """

    def __init__(
        self, in_features: int = 784, hidden: int = 400, latent: int = 20, rng=None
    ) -> None:
        super().__init__()
        self.in_features, self.hidden, self.latent = int(in_features), int(hidden), int(latent)
        self.enc = Linear(in_features, hidden, rng=rng)
        self.enc_mu = Linear(hidden, latent, nonlinearity="linear", rng=rng)
        self.enc_logvar = Linear(hidden, latent, nonlinearity="linear", rng=rng)
        self.dec = Linear(latent, hidden, rng=rng)
        self.dec_out = Linear(hidden, in_features, nonlinearity="linear", rng=rng)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.enc(x).relu()
        return self.enc_mu(h), self.enc_logvar(h)

    def decode(self, z: Tensor) -> Tensor:
        return self.dec_out(self.dec(z).relu())

    def forward(self, x: Tensor, rng=None, eps=None) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(logits, mu, logvar)`` for one reparameterised sample per row."""
        mu, logvar = self.encode(x)
        return self.decode(reparameterize(mu, logvar, rng=rng, eps=eps)), mu, logvar

    def loss(self, x: Tensor, target=None, rng=None, eps=None) -> tuple[Tensor, Tensor, Tensor]:
        """Single-sample negative ELBO ``(total, recon, kl)``; ``target`` defaults to ``x``."""
        logits, mu, logvar = self(x, rng=rng, eps=eps)
        return negative_elbo(logits, x.data if target is None else target, mu, logvar)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, hidden={self.hidden}, latent={self.latent}"
