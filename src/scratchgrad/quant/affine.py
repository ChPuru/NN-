"""Affine (uniform) quantization: the arithmetic, stated once.

A real value ``x`` is stored as an integer code ``q``::

    q = clamp(round(x / s) + z, qmin, qmax)        x_hat = s * (q - z)

with a positive float **scale** ``s`` and an integer **zero point** ``z``. Every
other file in this subpackage is built on :func:`quantize` and
:func:`dequantize`; the traps below are why they are not one-liners.

**1. The zero point is an integer and the range always contains 0.** A ReLU
layer emits exact zeros for roughly half its outputs, and zero padding is zero
by definition. If ``z`` were the unrounded ``qmin - x_min / s`` -- the
"obvious" formula -- real zero would land between two codes and every one of
those exact zeros would come back as a small non-zero number, a systematic bias
that averages over nothing. So :func:`calc_qparams` widens ``[x_min, x_max]`` to
include 0 and rounds ``z``; then ``dequantize(quantize(0)) == 0`` exactly.

**2. Rounding is ``np.round``: round half to EVEN.** ``2.5 -> 2``, ``3.5 -> 4``,
``-0.5 -> -0``. That is the documented convention for every float-to-code step
here. It is *not* the convention of the integer requantizer in
:mod:`.integer`, which rounds half toward ``+inf`` because that is what "add
half, arithmetic shift right" computes. The two disagree only on exact ties, and
the tests pin both.

**3. Symmetric means narrow range.** Signed symmetric codes use
``[-(2^(b-1) - 1), 2^(b-1) - 1]`` -- ``[-127, 127]`` at 8 bits, not
``[-128, 127]`` -- so the grid is symmetric about zero, ``z = 0`` and
``s = max|x| / qmax``. One consequence worth saying out loud: at **1 bit** a
narrow symmetric grid has a single level, ``{0}``. Binary weights are therefore
not "1-bit affine" at all; they are a separate quantizer
(:func:`scratchgrad.quant.fake.binarize`).

**4. Per-channel means per OUTPUT channel, which is axis 1 here.**
``Linear.weight`` is stored ``(in, out)`` in this framework, so the output
channel is the *last* axis -- the transpose of PyTorch's ``(out, in)``, whose
per-channel axis is 0. Using axis 0 here would give each *input* feature its own
scale, and ``sum_k s_k q_k x_k`` cannot be factored into ``s * sum_k q_k x_k``:
there is then no integer-only matmul at all.

**5. uint8 arithmetic wraps.** ``q - z`` with ``q`` a ``uint8`` array computes in
uint8 under NumPy 2's promotion rules: ``3 - 5 == 254``. Every subtraction of a
zero point here widens to int32 first.

**6. A degenerate range has no scale.** An all-zero channel (a dead unit) gives
``max - min = 0`` and ``s = 0``, then a division by zero. Any positive scale
represents an all-zero tensor exactly, so ``s = 1`` is used there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..nn.module import Module

__all__ = [
    "MinMaxObserver",
    "QParams",
    "calc_qparams",
    "dequantize",
    "qrange",
    "quantize",
]


def qrange(bits: int, symmetric: bool) -> tuple[int, int]:
    """Integer code range for ``bits``-bit codes.

    Args:
        bits: Code width, ``1..16``.
        symmetric: Signed narrow range ``[-(2^(b-1)-1), 2^(b-1)-1]`` if True,
            unsigned ``[0, 2^b - 1]`` if False.

    Raises:
        ValueError: For ``bits`` outside ``1..16``, or ``symmetric`` at 1 bit,
            whose narrow range is the single level ``{0}``.
    """
    bits = int(bits)
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be in 1..16, got {bits}")
    if symmetric:
        if bits < 2:
            raise ValueError("a symmetric 1-bit grid is {0}; use binarize() for binary weights")
        return -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


@dataclass(frozen=True)
class QParams:
    """Scale, zero point and code range of one affine quantizer.

    Attributes:
        scale: Positive float64 scale -- a scalar, or one per channel.
        zero_point: Integer zero point, same shape as ``scale``.
        qmin: Smallest code.
        qmax: Largest code.
        axis: The axis ``scale`` runs along, or None for per-tensor.
    """

    scale: np.ndarray
    zero_point: np.ndarray
    qmin: int
    qmax: int
    axis: int | None = None

    @property
    def dtype(self) -> np.dtype:
        """Smallest integer container the codes fit in."""
        if self.qmin >= 0 and self.qmax <= 255:
            return np.dtype(np.uint8)
        if self.qmin >= -128 and self.qmax <= 127:
            return np.dtype(np.int8)
        return np.dtype(np.int32)

    def broadcast(self, ndim: int) -> tuple[np.ndarray, np.ndarray]:
        """``(scale, zero_point)`` reshaped to broadcast against an ``ndim`` array."""
        if self.axis is None:
            return self.scale, self.zero_point
        shape = [1] * ndim
        shape[self.axis] = -1
        return self.scale.reshape(shape), self.zero_point.reshape(shape)


def calc_qparams(
    x_min,
    x_max,
    bits: int = 8,
    symmetric: bool = False,
    axis: int | None = None,
) -> QParams:
    """Min-max calibration: the quantizer whose grid spans ``[x_min, x_max]``.

    Asymmetric (unsigned codes)::

        lo, hi = min(x_min, 0), max(x_max, 0)          # 0 must be representable
        s = (hi - lo) / (qmax - qmin)
        z = clamp(qmin - round(lo / s), qmin, qmax)     # an INTEGER

    Symmetric (signed narrow codes): ``s = max(|x_min|, |x_max|) / qmax``, ``z = 0``.

    Args:
        x_min: Observed minimum -- scalar, or per channel.
        x_max: Observed maximum, same shape as ``x_min``.
        bits: Code width.
        symmetric: Signed symmetric (weights) vs unsigned asymmetric (activations).
        axis: Channel axis the per-channel statistics belong to.
    """
    qmin, qmax = qrange(bits, symmetric)
    lo = np.minimum(np.asarray(x_min, dtype=np.float64), 0.0)
    hi = np.maximum(np.asarray(x_max, dtype=np.float64), 0.0)
    if symmetric:
        scale = np.maximum(-lo, hi) / qmax
    else:
        scale = (hi - lo) / (qmax - qmin)
    scale = np.where(scale > 0, scale, 1.0)  # trap 6: an all-zero range
    if symmetric:
        zero_point = np.zeros(scale.shape, dtype=np.int64)
    else:
        zero_point = np.clip(qmin - np.round(lo / scale), qmin, qmax).astype(np.int64)
    return QParams(scale, zero_point, qmin, qmax, axis)


def _codes(x: np.ndarray, qp: QParams) -> np.ndarray:
    """Unclamped codes ``round(x / s) + z`` (float64).

    Shared by :func:`quantize` and ``fake_quant`` so both round identically and
    the STE mask is computed from exactly the value the forward pass clamped.
    """
    s, z = qp.broadcast(np.ndim(x))
    return np.round(np.asarray(x, dtype=np.float64) / s) + z


def quantize(x, qp: QParams) -> np.ndarray:
    """Real values to integer codes, in ``qp.dtype`` (uint8 / int8 / int32)."""
    return np.clip(_codes(x, qp), qp.qmin, qp.qmax).astype(qp.dtype)


def dequantize(q, qp: QParams) -> np.ndarray:
    """Codes to float64 reals: ``s * (q - z)``.

    ``q`` is widened to int32 *before* the zero point is subtracted (trap 5).
    """
    s, z = qp.broadcast(np.ndim(q))
    return s * (np.asarray(q).astype(np.int32) - z)


class MinMaxObserver(Module):
    """Tracks the range of what flows past it and turns that into :class:`QParams`.

    With ``momentum=None`` the range is the running min and max over every batch
    seen -- post-training calibration. With a momentum ``m`` it is an
    exponential moving average, ``r <- r + m (batch - r)`` -- what QAT uses, so a
    single outlier batch early in training cannot pin the range for good.

    The range lives in buffers, so it is saved with the model and never handed
    to an optimizer.

    Args:
        bits: Code width of the resulting quantizer.
        symmetric: See :func:`calc_qparams`.
        axis: Channel axis for per-channel ranges; every other axis is reduced.
        channels: Number of channels along ``axis`` (sizes the buffers up front
            so a fresh module can ``load_state_dict`` before seeing data).
        momentum: EMA coefficient in ``(0, 1]``, or None for the running min/max.
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = False,
        axis: int | None = None,
        channels: int | None = None,
        momentum: float | None = None,
    ) -> None:
        super().__init__()
        qrange(bits, symmetric)  # validate early
        if (axis is None) != (channels is None):
            raise ValueError("pass axis and channels together (or neither)")
        if momentum is not None and not 0.0 < momentum <= 1.0:
            raise ValueError(f"momentum must be in (0, 1], got {momentum}")
        self.bits, self.symmetric, self.axis, self.momentum = bits, symmetric, axis, momentum
        shape = () if channels is None else (int(channels),)
        self.register_buffer("min_val", np.full(shape, np.inf))
        self.register_buffer("max_val", np.full(shape, -np.inf))

    def update(self, x: np.ndarray) -> None:
        """Fold one batch into the tracked range."""
        x = np.asarray(x)
        if self.axis is None:
            bmin, bmax = x.min(), x.max()
        else:
            red = tuple(i for i in range(x.ndim) if i != self.axis % x.ndim)
            bmin, bmax = x.min(axis=red), x.max(axis=red)
        # In-place writes into the registered buffers: ``self.min_val += ...``
        # would go through __setattr__ and leave a shadowing instance attribute.
        lo, hi = self._buffers["min_val"], self._buffers["max_val"]
        if not np.all(np.isfinite(lo)):  # first batch: nothing to average with
            lo[...], hi[...] = bmin, bmax
        elif self.momentum is None:
            np.minimum(lo, bmin, out=lo)
            np.maximum(hi, bmax, out=hi)
        else:
            lo += self.momentum * (bmin - lo)
            hi += self.momentum * (bmax - hi)

    def forward(self, x):
        self.update(getattr(x, "data", x))
        return x

    def qparams(self) -> QParams:
        if not np.all(np.isfinite(self._buffers["min_val"])):
            raise RuntimeError("observer has seen no data; calibrate before asking for qparams")
        return calc_qparams(self.min_val, self.max_val, self.bits, self.symmetric, self.axis)

    def extra_repr(self) -> str:
        kind = "symmetric" if self.symmetric else "asymmetric"
        return f"bits={self.bits}, {kind}, axis={self.axis}, momentum={self.momentum}"
