"""Weight initialization schemes.

The one trap worth stating loudly, because it is the single most common
off-by-a-square-root in from-scratch implementations:

    **He initialization specifies a variance of ``2 / fan_in``.**
    The standard deviation is therefore ``sqrt(2 / fan_in)``.

Papers and blog posts write it as ``N(0, 2/n)`` -- using the ``N(mean, variance)``
convention. Read that as ``N(mean, std)`` and you initialize with a standard
deviation of ``2/n`` instead of ``sqrt(2/n)``, which for ``fan_in=784`` is 55x
too small. The network still trains; it just trains badly, and it looks like a
learning-rate problem.

All functions here operate **in place** on ``Parameter.data`` and return the
parameter, following the ``torch.nn.init`` trailing-underscore convention.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "calculate_gain",
    "compute_fans",
    "constant_",
    "default_rng",
    "kaiming_normal_",
    "kaiming_uniform_",
    "normal_",
    "ones_",
    "orthogonal_",
    "sparse_",
    "trunc_normal_",
    "uniform_",
    "xavier_normal_",
    "xavier_uniform_",
    "zeros_",
]


def default_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def calculate_gain(nonlinearity: str = "relu", param: float | None = None) -> float:
    """The multiplier that keeps activation variance stable through a layer.

    Derived by asking what scaling keeps ``Var[y] == Var[x]`` when ``y = f(Wx)``.
    ReLU discards half the distribution, hence ``sqrt(2)``.
    """
    nl = nonlinearity.lower()
    if nl in ("linear", "identity", "sigmoid", "conv1d", "conv2d", "conv3d"):
        return 1.0
    if nl == "tanh":
        return 5.0 / 3.0
    if nl == "relu":
        return math.sqrt(2.0)
    if nl == "leaky_relu":
        a = 0.01 if param is None else float(param)
        return math.sqrt(2.0 / (1.0 + a * a))
    if nl == "selu":
        return 0.75
    raise ValueError(f"unsupported nonlinearity {nonlinearity!r}")


def compute_fans(shape: tuple[int, ...]) -> tuple[int, int]:
    """``(fan_in, fan_out)`` for a weight of the given shape.

    Conventions this follows, which have to be stated or the conv case is
    ambiguous:

    * ``Linear`` weights are stored ``(in_features, out_features)`` -- the layout
      that makes the forward pass a plain ``x @ W`` with no transpose.
    * ``Conv2d`` weights are stored ``(out_channels, in_channels, kH, kW)``,
      matching the standard layout, so the receptive field is ``shape[2:]``.
    """
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return shape[0], shape[0]
    if len(shape) == 2:
        return shape[0], shape[1]
    receptive = int(np.prod(shape[2:]))
    return shape[1] * receptive, shape[0] * receptive


def _fan(shape: tuple[int, ...], mode: str) -> int:
    fan_in, fan_out = compute_fans(shape)
    if mode == "fan_in":
        return fan_in
    if mode == "fan_out":
        return fan_out
    if mode == "fan_avg":
        return (fan_in + fan_out) // 2
    raise ValueError(f"unknown mode {mode!r}")


def _apply(param, values: np.ndarray):
    param.data = values.astype(param.data.dtype, copy=False).reshape(param.data.shape)
    return param


def zeros_(param):
    return _apply(param, np.zeros(param.data.shape))


def ones_(param):
    return _apply(param, np.ones(param.data.shape))


def constant_(param, value: float):
    return _apply(param, np.full(param.data.shape, value))


def normal_(param, mean: float = 0.0, std: float = 1.0, rng=None):
    g = default_rng(rng)
    return _apply(param, g.normal(mean, std, size=param.data.shape))


def uniform_(param, low: float = -1.0, high: float = 1.0, rng=None):
    g = default_rng(rng)
    return _apply(param, g.uniform(low, high, size=param.data.shape))


def trunc_normal_(
    param, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0, rng=None
):
    """Normal samples rejected outside ``[mean + a*std, mean + b*std]``."""
    g = default_rng(rng)
    lo, hi = mean + a * std, mean + b * std
    out = g.normal(mean, std, size=param.data.shape)
    for _ in range(100):
        bad = (out < lo) | (out > hi)
        if not bad.any():
            break
        out[bad] = g.normal(mean, std, size=int(bad.sum()))
    return _apply(param, np.clip(out, lo, hi))


def xavier_uniform_(param, gain: float = 1.0, rng=None):
    """Glorot: ``Var[W] = 2 / (fan_in + fan_out)``, uniform form."""
    fan_in, fan_out = compute_fans(param.data.shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std  # Var[U(-b, b)] = b^2 / 3
    return uniform_(param, -bound, bound, rng=rng)


def xavier_normal_(param, gain: float = 1.0, rng=None):
    fan_in, fan_out = compute_fans(param.data.shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(param, 0.0, std, rng=rng)


def kaiming_normal_(
    param, mode: str = "fan_in", nonlinearity: str = "relu", a: float = 0.0, rng=None
):
    """He normal: **standard deviation** ``gain / sqrt(fan)``.

    For ReLU, ``gain = sqrt(2)``, so ``std = sqrt(2 / fan_in)`` -- the variance
    is ``2 / fan_in``. See the module docstring.
    """
    fan = _fan(param.data.shape, mode)
    gain = calculate_gain(nonlinearity, a)
    return normal_(param, 0.0, gain / math.sqrt(fan), rng=rng)


def kaiming_uniform_(
    param, mode: str = "fan_in", nonlinearity: str = "relu", a: float = 0.0, rng=None
):
    fan = _fan(param.data.shape, mode)
    gain = calculate_gain(nonlinearity, a)
    bound = math.sqrt(3.0) * gain / math.sqrt(fan)
    return uniform_(param, -bound, bound, rng=rng)


def orthogonal_(param, gain: float = 1.0, rng=None):
    """Orthogonal init via QR. Standard for recurrent weight matrices.

    The sign correction on ``R``'s diagonal is not cosmetic: without it the Q
    that LAPACK returns is not Haar-distributed, so the columns carry a
    systematic bias.
    """
    g = default_rng(rng)
    shape = param.data.shape
    if len(shape) < 2:
        raise ValueError("orthogonal_ needs at least 2 dimensions")
    rows = shape[0]
    cols = int(np.prod(shape[1:]))
    flat = g.standard_normal((max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(flat)
    q *= np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    return _apply(param, gain * q.reshape(shape))


def sparse_(param, sparsity: float = 0.1, std: float = 0.01, rng=None):
    g = default_rng(rng)
    shape = param.data.shape
    if len(shape) != 2:
        raise ValueError("sparse_ expects a 2-D weight")
    out = g.normal(0.0, std, size=shape)
    n_zero = math.ceil(sparsity * shape[0])  # math.ceil already returns an int
    for col in range(shape[1]):
        rows = g.permutation(shape[0])[:n_zero]
        out[rows, col] = 0.0
    return _apply(param, out)
