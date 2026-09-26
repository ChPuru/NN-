"""A neural network library in one file, with every gradient written by hand.

This is Phase 0 of ``scratchgrad``: a deliberately tiny, deliberately readable
framework that imports **nothing but NumPy**. There is no autograd here and no
package to go and look at -- each layer owns its own ``backward``, so the chain
rule is visible as literal code rather than inferred from a graph.

The layout is the one the rest of the project freezes on:

* ``Dense.W`` is ``(n_in, n_out)``, so the forward pass is ``x @ W + b`` with no
  transpose anywhere. The transpose appears exactly twice, both in ``backward``,
  which is where it belongs.
* Layers emit **raw logits**; :class:`SoftmaxCrossEntropy` owns the softmax. A
  model that ends in a softmax and is then fed to a cross-entropy that softmaxes
  again still has the right argmax, so accuracy looks fine and only the loss
  reveals the bug -- pinned at 1.4615 for ten classes.
* Gradient buffers (``dW``, ``db``) are allocated once and written **in place**.
  Rebinding ``self.dW = ...`` on every backward would leave the optimizer, which
  captured the original array, updating with a gradient from step 1 forever.

Numerically the file is defined by four things that are wrong in most from-scratch
implementations on the internet, each marked at its site:
He init is a *variance* of ``2/fan_in``; softmax subtracts the row max; the loss
is computed in log-sum-exp form and never takes the log of a probability; and
Adam's ``eps`` sits outside the sqrt and after bias correction.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "MLP",
    "Adam",
    "Dense",
    "ReLU",
    "SoftmaxCrossEntropy",
    "accuracy",
    "he_normal",
]


def he_normal(n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
    """He-initialized ``(n_in, n_out)`` weight matrix.

    He/Kaiming specifies a **variance** of ``2 / fan_in``, so the standard
    deviation handed to the sampler is ``sqrt(2 / fan_in)``. Papers write it
    ``N(0, 2/n)`` using the ``N(mean, variance)`` convention; read that as
    ``N(mean, std)`` and for ``fan_in = 784`` the weights come out 55x too
    small. The net still trains, just badly, and it presents as a
    learning-rate problem for an afternoon.

    Args:
        n_in: Number of input features (the fan-in).
        n_out: Number of output features.
        rng: Explicit generator. There is no global seeding in this project.

    Returns:
        A ``(n_in, n_out)`` float64 array.
    """
    return rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))


class Dense:
    """Fully connected layer ``y = x @ W + b``."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator) -> None:
        self.n_in, self.n_out = int(n_in), int(n_out)
        self.W = he_normal(n_in, n_out, rng)
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Cache the input -- ``dL/dW`` needs it -- and return the logits."""
        self.x = x
        return x @ self.W + self.b

    def backward(self, g: np.ndarray) -> np.ndarray:
        """Propagate the cotangent ``g = dL/dy`` back to ``x``, ``W`` and ``b``.

        ``b`` is broadcast over the batch in the forward pass, so its gradient
        is a **sum** over the batch axis, not a mean: any averaging over the
        batch already happened inside the loss.
        """
        assert self.x is not None, "backward() before forward()"
        self.dW[...] = self.x.T @ g
        self.db[...] = g.sum(axis=0)
        return g @ self.W.T

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """``max(x, 0)``, with the subgradient at exactly 0 taken as 0."""

    def __init__(self) -> None:
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, g: np.ndarray) -> np.ndarray:
        assert self.mask is not None, "backward() before forward()"
        return g * self.mask

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return []


class SoftmaxCrossEntropy:
    """Fused softmax + negative log-likelihood over **raw logits**.

    Fusing is not an optimization here, it is the whole point: the gradient of
    the fused op is exactly ``(p - onehot) / N``, which the composed version
    reaches only after a Jacobian cancels.
    """

    def __init__(self) -> None:
        self.p: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy of ``logits`` ``(N, C)`` against labels ``y`` ``(N,)``.

        Two separate stabilizations, both required:

        1. The row max is subtracted before ``exp``. ``exp(800)`` is ``inf`` and
           the loss is ``nan`` from then on; subtracting a per-row constant
           leaves the softmax exactly unchanged.
        2. The loss is evaluated as ``logsumexp(z) - z[y]`` and **never** as
           ``-log(p + 1e-12)``. That epsilon caps the reportable loss at about
           27.6 and, worse, in the clipped regime it hands a *zero* gradient to
           precisely the samples the model got most wrong.
        """
        y = np.asarray(y)
        n = logits.shape[0]
        if y.shape != (n,):
            raise ValueError(f"labels {y.shape} do not match logits {logits.shape}")
        shift = logits - logits.max(axis=1, keepdims=True)
        lse = np.log(np.exp(shift).sum(axis=1, keepdims=True))
        self.p = np.exp(shift - lse)
        self.y = y
        rows = np.arange(n)
        # The row max cancels: (z - m)[y] - logsumexp(z - m) == z[y] - logsumexp(z).
        return float(np.mean(lse[:, 0] - shift[rows, y]))

    def backward(self) -> np.ndarray:
        """``(softmax(z) - onehot(y)) / N``, the cotangent of the logits."""
        assert self.p is not None and self.y is not None, "backward() before forward()"
        n = self.p.shape[0]
        g = self.p.copy()
        g[np.arange(n), self.y] -= 1.0
        return g / n


class MLP:
    """A stack of ``Dense`` layers with ``ReLU`` between them.

    ``MLP([784, 128, 10])`` is the Phase 0 model: 784*128 + 128 + 128*10 + 10 =
    101,770 parameters.
    """

    def __init__(self, sizes: list[int], rng: np.random.Generator) -> None:
        if len(sizes) < 2:
            raise ValueError("an MLP needs at least an input and an output size")
        self.sizes = list(sizes)
        self.layers: list[Dense | ReLU] = []
        for i, (a, b) in enumerate(itertools.pairwise(sizes)):
            self.layers.append(Dense(a, b, rng))
            if i < len(sizes) - 2:  # no activation on the logits
                self.layers.append(ReLU())

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, g: np.ndarray) -> np.ndarray:
        """Run the chain rule backwards through the stack."""
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [pair for layer in self.layers for pair in layer.params()]

    def num_parameters(self) -> int:
        return sum(p.size for p, _ in self.params())

    def dense_layers(self) -> list[Dense]:
        return [layer for layer in self.layers if isinstance(layer, Dense)]


class Adam:
    """Adam, as published: bias-corrected moments, ``eps`` outside the sqrt.

    The update is ``p -= lr * m_hat / (sqrt(v_hat) + eps)``.

    The common corruption is ``g / sqrt(v + eps)``: putting ``eps`` inside the
    sqrt and before bias correction. With ``eps = 1e-8`` that floor engages at
    ``|g| ~ 1e-4`` instead of ``1e-8`` -- four orders of magnitude too early --
    so the last stretch of training is silently damped. It converges, just to a
    worse place, which is the hardest kind of bug to notice.

    Args:
        params: ``(parameter, gradient)`` pairs. Both arrays are updated in
            place, so the pairs must be the layers' own buffers.
        lr: Step size.
        betas: Exponential decay rates for the first and second moments.
        eps: Denominator floor.
    """

    def __init__(
        self,
        params: list[tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = list(params)
        self.lr = float(lr)
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.t = 0
        self.m = [np.zeros_like(p) for p, _ in self.params]
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def zero_grad(self) -> None:
        for _, g in self.params:
            g[...] = 0.0

    def step(self) -> None:
        self.t += 1
        # Bias correction: m and v start at zero, so early estimates are biased
        # toward zero by exactly these factors. Dividing them out is what lets
        # step 1 take a full-size step instead of a ~lr*(1-b1) sized one.
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        for (p, g), m, v in zip(self.params, self.m, self.v, strict=True):
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * (g * g)
            p -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    """Fraction of rows whose argmax logit equals the label."""
    return float(np.mean(np.argmax(logits, axis=1) == np.asarray(y)))
