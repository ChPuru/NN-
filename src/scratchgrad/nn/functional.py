"""Stateless functional forms of the activations and losses.

The layer classes in :mod:`scratchgrad.nn.layers` and
:mod:`scratchgrad.nn.loss` are thin wrappers over these.

Two of these are written as **fused** ops rather than compositions of
primitives. That is deliberate:

* :func:`cross_entropy` fuses softmax and negative log-likelihood because the
  fused gradient is exactly ``(p - onehot) / N`` -- a clean expression that the
  composed version reaches only after cancelling a Jacobian, and one that is
  numerically better behaved besides.
* :func:`gelu` fuses because its derivative is long enough that composing it
  from eight primitives builds eight graph nodes per call for no benefit.
"""

from __future__ import annotations

import math

import numpy as np

from ..tensor import Tensor

__all__ = [
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "dropout",
    "elu",
    "gelu",
    "huber_loss",
    "l1_loss",
    "leaky_relu",
    "linear",
    "log_softmax",
    "mse_loss",
    "nll_loss",
    "one_hot",
    "relu",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
    "swish",
    "tanh",
]


# --------------------------------------------------------------------------- #
# activations
# --------------------------------------------------------------------------- #
def relu(x: Tensor) -> Tensor:
    return x.relu()


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    a = x.data
    mask = np.where(a > 0, 1.0, negative_slope).astype(a.dtype)
    return Tensor._make(a * mask, (x,), (lambda g: g * mask,), "leaky_relu")


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    a = x.data
    pos = a > 0
    ex = np.exp(np.minimum(a, 0.0))  # clamp: exp of a large positive would overflow
    out = np.where(pos, a, alpha * (ex - 1.0)).astype(a.dtype)
    grad = np.where(pos, 1.0, alpha * ex).astype(a.dtype)
    return Tensor._make(out, (x,), (lambda g: g * grad,), "elu")


_GELU_C = math.sqrt(2.0 / math.pi)


def gelu(x: Tensor, approximate: str = "tanh") -> Tensor:
    """Gaussian Error Linear Unit.

    ``approximate='tanh'`` is the form GPT-2 and BERT actually shipped:
    ``0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))``. ``approximate='none'``
    uses the exact ``0.5x(1 + erf(x/sqrt(2)))`` and needs ``scipy``, which is
    not a dependency of this package -- so it raises a clear error rather than
    silently substituting the approximation.
    """
    a = x.data
    if approximate == "tanh":
        # a*a*a, not a**3: float pow() is ~10x slower than two multiplies.
        inner = _GELU_C * (a + 0.044715 * (a * a * a))
        t = np.tanh(inner)
        out = 0.5 * a * (1.0 + t)
        # d/dx = 0.5(1+t) + 0.5x(1-t^2) * sqrt(2/pi)(1 + 3*0.044715 x^2)
        dinner = _GELU_C * (1.0 + 3.0 * 0.044715 * a * a)
        grad = (0.5 * (1.0 + t) + 0.5 * a * (1.0 - t * t) * dinner).astype(a.dtype)
        return Tensor._make(out.astype(a.dtype), (x,), (lambda g: g * grad,), "gelu")
    if approximate == "none":
        try:
            from scipy.special import erf
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "gelu(approximate='none') needs scipy for erf; scipy is not a dependency "
                "of scratchgrad. Use approximate='tanh', which is what GPT-2 shipped."
            ) from exc
        cdf = 0.5 * (1.0 + erf(a / math.sqrt(2.0)))
        pdf = np.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
        out = (a * cdf).astype(a.dtype)
        grad = (cdf + a * pdf).astype(a.dtype)
        return Tensor._make(out, (x,), (lambda g: g * grad,), "gelu_exact")
    raise ValueError(f"approximate must be 'tanh' or 'none', got {approximate!r}")


def silu(x: Tensor) -> Tensor:
    """``x * sigmoid(x)``, a.k.a. Swish."""
    a = x.data
    z = np.exp(-np.abs(a))
    s = np.where(a >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
    out = (a * s).astype(a.dtype)
    grad = (s * (1.0 + a * (1.0 - s))).astype(a.dtype)
    return Tensor._make(out, (x,), (lambda g: g * grad,), "silu")


swish = silu


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """``log(1 + exp(bx))/b``, computed as ``max(bx,0) + log1p(exp(-|bx|))``.

    Above ``threshold`` the function is numerically identical to ``x`` and the
    branch avoids evaluating ``exp`` of anything large.
    """
    a = x.data
    bx = beta * a
    out = np.where(bx > threshold, a, (np.maximum(bx, 0.0) + np.log1p(np.exp(-np.abs(bx)))) / beta)
    z = np.exp(-np.abs(bx))
    s = np.where(bx >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
    grad = np.where(bx > threshold, 1.0, s).astype(a.dtype)
    return Tensor._make(out.astype(a.dtype), (x,), (lambda g: g * grad,), "softplus")


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.softmax(axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.log_softmax(axis)


def dropout(x: Tensor, p: float = 0.5, training: bool = True, rng=None) -> Tensor:
    """Inverted dropout: scale by ``1/(1-p)`` at **train** time.

    Scaling during training (rather than at inference) is what lets ``eval()``
    be a plain identity -- no rescaling, no train/test mismatch to forget.
    """
    if not training or p == 0.0:
        return x
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")
    g = rng if rng is not None else np.random.default_rng()
    keep = (g.random(x.data.shape) >= p).astype(x.data.dtype) / (1.0 - p)
    return Tensor._make(x.data * keep, (x,), (lambda gr: gr * keep,), "dropout")


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """``x @ W + b`` with ``W`` stored ``(in_features, out_features)``."""
    out = x @ weight
    return out if bias is None else out + bias


def one_hot(indices: np.ndarray, num_classes: int, dtype=None) -> np.ndarray:
    idx = np.asarray(indices)
    out = np.zeros(idx.shape + (num_classes,), dtype=dtype or np.float64)
    np.put_along_axis(out, idx[..., None], 1.0, axis=-1)
    return out


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def _reduce(values: np.ndarray, reduction: str) -> tuple[np.ndarray, float]:
    """Return the reduced value and the scale its gradient must be divided by."""
    if reduction == "mean":
        return values.mean(), float(values.size)
    if reduction == "sum":
        return values.sum(), 1.0
    if reduction == "none":
        return values, 1.0
    raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")


def cross_entropy(
    logits: Tensor,
    target,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    ignore_index: int | None = None,
) -> Tensor:
    """Softmax cross-entropy, fused, from **raw logits**.

    ``logits`` is ``(N, C)`` (or ``(N, T, C)``, flattened internally);
    ``target`` is integer class indices of the matching leading shape.

    The forward computes ``logsumexp(z) - z[y]`` and so **never takes the log of
    a probability**. The widespread ``-log(p + 1e-12)`` is a fudge: it caps the
    reportable loss, and in the clipped regime it hands a *zero* gradient to
    precisely the samples the model got most wrong.

    The backward is the fused ``(softmax(z) - onehot) / N``. Deriving it as
    ``dNLL/dp * dp/dz`` gives the same thing only after a Jacobian cancels;
    doing it in one step is both cheaper and the clearer demonstration.
    """
    z = logits.data
    if z.ndim < 2:
        raise ValueError(f"cross_entropy expects at least 2 dims (N, C), got {z.shape}")
    flat_z = z.reshape(-1, z.shape[-1])
    y = np.asarray(target.data if isinstance(target, Tensor) else target).reshape(-1)
    if y.dtype.kind == "f":
        raise TypeError("cross_entropy target must be integer class indices, not floats")
    n_all, C = flat_z.shape
    if y.shape[0] != n_all:
        raise ValueError(f"target has {y.shape[0]} entries but logits imply {n_all}")
    # The ignore sentinel is deliberately out of range (-100 by convention), so
    # both the bounds check and the row mask must be taken from the ORIGINAL
    # targets -- before the sentinel is remapped to a safe index for gathering.
    # Remapping first makes `y != ignore_index` vacuously true and silently
    # ignores nothing.
    keep_mask = None if ignore_index is None else (y != ignore_index)
    checked = y if keep_mask is None else y[keep_mask]
    if checked.size and (checked.min() < 0 or checked.max() >= C):
        raise ValueError(
            f"target values must lie in [0, {C}); got [{checked.min()}, {checked.max()}]"
        )
    if keep_mask is not None:
        y = np.where(keep_mask, y, 0)

    shift = flat_z - flat_z.max(axis=1, keepdims=True)
    exp = np.exp(shift)
    ssum = exp.sum(axis=1, keepdims=True)
    log_probs = shift - np.log(ssum)
    p = exp / ssum

    rows = np.arange(n_all)
    if label_smoothing:
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must lie in [0, 1)")
        eps = label_smoothing
        per = -((1.0 - eps) * log_probs[rows, y] + eps * log_probs.mean(axis=1))
        soft = np.full((n_all, C), eps / C, dtype=p.dtype)
        soft[rows, y] += 1.0 - eps
    else:
        per = -log_probs[rows, y]
        soft = np.zeros((n_all, C), dtype=p.dtype)
        soft[rows, y] = 1.0

    if keep_mask is not None:
        keep = keep_mask.astype(p.dtype)
        per = per * keep
        soft = soft * keep[:, None]
        p = p * keep[:, None]
        denom = float(max(keep.sum(), 1.0))
    else:
        denom = float(n_all)

    # 'mean' divides by `denom`, the number of *contributing* rows -- not by the
    # element count. With ignore_index they differ, and dividing the forward by
    # one while the backward divides by the other is a gradient that is wrong by
    # exactly the ignored fraction. (This also matches PyTorch's convention.)
    if reduction == "mean":
        value = per.sum() / denom
    else:
        value, _ = _reduce(per, reduction)
    scale = denom if reduction == "mean" else 1.0
    base = (p - soft) / scale

    def bw(g, base=base, shape=z.shape, reduction=reduction):
        g = np.asarray(g)
        if reduction == "none":
            return (base * g.reshape(-1, 1)).reshape(shape)
        return (base * g).reshape(shape)

    out = value if reduction != "none" else per.reshape(z.shape[:-1])
    return Tensor._make(np.asarray(out, dtype=z.dtype), (logits,), (bw,), "cross_entropy")


def nll_loss(log_probs: Tensor, target, reduction: str = "mean") -> Tensor:
    """Negative log-likelihood from **already log-softmaxed** inputs.

    Prefer :func:`cross_entropy` on raw logits. This exists for the case where
    the log-probabilities are needed for something else too -- and as the
    explicit reminder that feeding it raw logits is the double-softmax bug.
    """
    lp = log_probs.data
    flat = lp.reshape(-1, lp.shape[-1])
    y = np.asarray(target.data if isinstance(target, Tensor) else target).reshape(-1)
    n_all, C = flat.shape
    rows = np.arange(n_all)
    per = -flat[rows, y]
    value, _ = _reduce(per, reduction)
    scale = float(n_all) if reduction == "mean" else 1.0
    mask = np.zeros((n_all, C), dtype=lp.dtype)
    mask[rows, y] = -1.0 / scale

    def bw(g, mask=mask, shape=lp.shape, reduction=reduction):
        g = np.asarray(g)
        if reduction == "none":
            return (mask * g.reshape(-1, 1)).reshape(shape)
        return (mask * g).reshape(shape)

    out = value if reduction != "none" else per.reshape(lp.shape[:-1])
    return Tensor._make(np.asarray(out, dtype=lp.dtype), (log_probs,), (bw,), "nll_loss")


def _check_same_shape(a: Tensor, b) -> np.ndarray:
    """Shape guard as line 1 of every elementwise loss.

    A ``(N, 1)`` prediction against an ``(N,)`` target broadcasts to ``(N, N)``
    without complaint. The loss then floors at ``Var(y)`` and the model appears
    to predict a constant -- a failure that looks like underfitting.
    """
    t = np.asarray(b.data if isinstance(b, Tensor) else b)
    if a.data.shape != t.shape:
        raise ValueError(
            f"shape mismatch: prediction {a.data.shape} vs target {t.shape}. "
            "These would broadcast silently; reshape the target explicitly."
        )
    return t


def mse_loss(pred: Tensor, target, reduction: str = "mean") -> Tensor:
    t = _check_same_shape(pred, target)
    d = pred.data - t
    value, _ = _reduce(d * d, reduction)
    scale = float(d.size) if reduction == "mean" else 1.0

    def bw(g, d=d, scale=scale, reduction=reduction):
        g = np.asarray(g)
        return (2.0 * d / scale) * g

    out = value if reduction != "none" else d * d
    return Tensor._make(np.asarray(out, dtype=pred.data.dtype), (pred,), (bw,), "mse_loss")


def l1_loss(pred: Tensor, target, reduction: str = "mean") -> Tensor:
    t = _check_same_shape(pred, target)
    d = pred.data - t
    value, _ = _reduce(np.abs(d), reduction)
    scale = float(d.size) if reduction == "mean" else 1.0
    grad = np.sign(d) / scale
    out = value if reduction != "none" else np.abs(d)
    return Tensor._make(
        np.asarray(out, dtype=pred.data.dtype), (pred,), (lambda g: grad * g,), "l1_loss"
    )


def huber_loss(pred: Tensor, target, delta: float = 1.0, reduction: str = "mean") -> Tensor:
    t = _check_same_shape(pred, target)
    d = pred.data - t
    small = np.abs(d) <= delta
    per = np.where(small, 0.5 * d * d, delta * (np.abs(d) - 0.5 * delta))
    value, _ = _reduce(per, reduction)
    scale = float(d.size) if reduction == "mean" else 1.0
    grad = np.where(small, d, delta * np.sign(d)) / scale
    out = value if reduction != "none" else per
    return Tensor._make(
        np.asarray(out, dtype=pred.data.dtype), (pred,), (lambda g: grad * g,), "huber_loss"
    )


def binary_cross_entropy_with_logits(
    logits: Tensor, target, reduction: str = "mean", pos_weight: float | None = None
) -> Tensor:
    """BCE computed from logits, never from a probability.

    Uses ``max(z,0) - z*y + log(1 + exp(-|z|))``, which is exact for every real
    ``z``. Taking ``log(sigmoid(z))`` instead overflows to ``-inf`` for ``z``
    around -750 and needs an epsilon fudge to hide it.
    """
    t = _check_same_shape(logits, target).astype(logits.data.dtype)
    z = logits.data
    per = np.maximum(z, 0.0) - z * t + np.log1p(np.exp(-np.abs(z)))
    zz = np.exp(-np.abs(z))
    s = np.where(z >= 0, 1.0 / (1.0 + zz), zz / (1.0 + zz))
    grad = s - t
    if pos_weight is not None:
        w = 1.0 + (pos_weight - 1.0) * t
        per = per * w
        grad = grad * w
    value, _ = _reduce(per, reduction)
    scale = float(z.size) if reduction == "mean" else 1.0
    grad = grad / scale
    out = value if reduction != "none" else per
    return Tensor._make(
        np.asarray(out, dtype=z.dtype), (logits,), (lambda g: grad * g,), "bce_with_logits"
    )
