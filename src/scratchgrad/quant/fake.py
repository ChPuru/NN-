"""Fake quantization and the straight-through estimator -- what QAT trains through.

``fake_quant(x) = s * (clamp(round(x / s) + z, qmin, qmax) - z)`` runs the
quantizer and immediately undoes the code step, so the forward pass sees exactly
the values integer inference will see, while everything stays a float tensor
that autograd can carry.

**The gradient of ``round`` is useless.** It is 0 almost everywhere and
undefined on the jumps, so the honest chain rule hands every upstream parameter
a gradient of exactly zero and training stops before it starts. The
straight-through estimator (STE) replaces ``d round(u)/du`` by 1 and
differentiates everything else honestly. The only other non-smooth piece is the
clamp, whose derivative is 1 where it is inactive and 0 where it saturates. So
the **clipped STE** is

    d fake_quant(x) / dx  :=  1  if qmin <= round(x/s) + z <= qmax  else  0

Boundary convention, stated because two defensible choices exist: the mask is
evaluated on the *rounded* code, so an input that rounds to exactly ``qmax`` is
inside (the clamp did not fire on it) even though ``x/s + z`` is up to half a
step above ``qmax``. This matches what the forward pass actually clamped.

**Why this cannot be gradchecked.** Central differences of a piecewise-constant
function are 0 whenever both probes land on the same step, and about
``s / (2h)`` -- astronomically large -- when they straddle a jump. Neither is
the STE value 1, and neither is supposed to be: the STE is a *defined* rule, not
a derivative. The tests therefore check the rule directly (backward equals the
upstream gradient times the mask, exactly) and check that inside the range it
equals the true derivative of the *unrounded* surrogate
``s * (clamp(x/s + z, qmin, qmax) - z)`` -- the one thing finite differences can
certify.

**Binary weights** (:func:`binarize`) are ``alpha * sign(w)``, with
``alpha = mean|w|`` per output channel -- the least-squares optimum for a fixed
sign pattern (``argmin_a ||w - a b||^2 = <w, b>/<b, b> = mean|w|`` when
``b = sign w``). ``sign(0)`` is taken as ``+1`` so every code is ``+-1``. The
STE there is the hardtanh one of BinaryConnect: gradient 1 where ``|w| <= 1``,
0 beyond; ``alpha`` is treated as a constant in the backward pass.
"""

from __future__ import annotations

import numpy as np

from ..nn.module import Module, Parameter
from ..tensor import Tensor
from .affine import MinMaxObserver, QParams, _codes, calc_qparams, quantize

__all__ = [
    "FakeQuantize",
    "QATLinear",
    "binarize",
    "binary_codes",
    "fake_quant",
    "set_quant_state",
]


def fake_quant(x: Tensor, qp: QParams) -> Tensor:
    """Quantize-dequantize with the clipped straight-through estimator.

    Args:
        x: Input tensor (any float dtype; the output keeps it).
        qp: Quantizer parameters, per-tensor or per-channel.

    Returns:
        ``s * (clamp(round(x/s) + z, qmin, qmax) - z)``; backward passes the
        upstream gradient where the clamp is inactive and zero where it saturates.
    """
    a = x.data
    r = _codes(a, qp)
    inside = (r >= qp.qmin) & (r <= qp.qmax)
    s, z = qp.broadcast(a.ndim)
    out = ((np.clip(r, qp.qmin, qp.qmax) - z) * s).astype(a.dtype, copy=False)
    return Tensor._make(out, (x,), (lambda g: g * inside,), "fake_quant")


def binary_codes(w: np.ndarray, axis: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(codes, alpha)`` with codes ``+-1`` (int8) and ``alpha = mean|w|``.

    ``alpha`` is per channel along ``axis`` (shape ``(C,)``), or a scalar.
    """
    w = np.asarray(w)
    codes = np.where(w >= 0, 1, -1).astype(np.int8)  # sign(0) := +1, never 0
    if axis is None:
        return codes, np.asarray(np.abs(w).mean(), dtype=np.float64)
    red = tuple(i for i in range(w.ndim) if i != axis % w.ndim)
    return codes, np.abs(w).mean(axis=red).astype(np.float64)


def binarize(w: Tensor, axis: int | None = None) -> Tensor:
    """``alpha * sign(w)`` with the hardtanh STE (gradient 1 where ``|w| <= 1``)."""
    a = w.data
    codes, alpha = binary_codes(a, axis)
    if axis is not None:
        shape = [1] * a.ndim
        shape[axis] = -1
        alpha = alpha.reshape(shape)
    out = (alpha * codes).astype(a.dtype, copy=False)
    inside = np.abs(a) <= 1.0
    return Tensor._make(out, (w,), (lambda g: g * inside,), "binarize")


class FakeQuantize(Module):
    """Activation fake-quantizer with its own min-max observer.

    Two switches, both plain attributes so a helper can flip them model-wide
    (:func:`set_quant_state`):

    * ``observe`` -- fold each batch into the range. Only ever in ``train()``
      mode: evaluating a model must not move its calibration.
    * ``fake`` -- actually quantize. Off during post-training calibration, when
      the float model should run untouched while the ranges are measured.

    Args:
        bits: Activation code width.
        symmetric: Signed symmetric codes; activations after ReLU want False.
        momentum: Observer EMA coefficient, or None for running min/max.
    """

    def __init__(self, bits: int = 8, symmetric: bool = False, momentum: float | None = None):
        super().__init__()
        self.observer = MinMaxObserver(bits, symmetric, momentum=momentum)
        self.observe = True
        self.fake = True

    def qparams(self) -> QParams:
        return self.observer.qparams()

    def forward(self, x: Tensor) -> Tensor:
        if self.observe and self.training:
            self.observer.update(x.data)
        return fake_quant(x, self.qparams()) if self.fake else x

    def extra_repr(self) -> str:
        return f"observe={self.observe}, fake={self.fake}"


class QATLinear(Module):
    """``x @ Q(W) + b``: a Linear whose weight is fake-quantized every forward.

    Weights are quantized per output channel (axis 1 of the ``(in, out)``
    layout), symmetric, with the scale recomputed from the *current* weights on
    every call. Min-max scaling of the weights themselves means no weight is
    ever clipped, so the STE mask on ``W`` is all ones and the weight gradient
    is exactly ``x^T g`` evaluated at the quantized weights.

    ``bits=1`` switches to :func:`binarize`. The bias stays float here; the
    integer converter puts it on the accumulator grid ``s_x * s_w``.

    Args:
        weight: ``(in, out)`` initial weight (copied).
        bias: ``(out,)`` initial bias (copied), or None.
        bits: Weight code width; 1 means binary ``+-alpha``.
    """

    def __init__(self, weight, bias=None, bits: int = 8) -> None:
        super().__init__()
        self.weight = Parameter(np.array(weight, copy=True))
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = Parameter(np.array(bias, copy=True))
        self.bits = int(bits)
        self.fake = True
        if self.bits != 1:
            calc_qparams(0.0, 1.0, self.bits, symmetric=True)  # validate early

    @classmethod
    def from_linear(cls, linear, bits: int = 8) -> QATLinear:
        """Copy the weights of a trained :class:`~scratchgrad.nn.Linear`."""
        bias = linear._parameters.get("bias")
        return cls(linear.weight.data, None if bias is None else bias.data, bits)

    def weight_qparams(self) -> QParams:
        w = self.weight.data
        return calc_qparams(w.min(axis=0), w.max(axis=0), self.bits, symmetric=True, axis=1)

    def weight_codes(self) -> tuple[np.ndarray, np.ndarray]:
        """``(int8 codes (in, out), per-channel scale (out,))`` for the converter."""
        if self.bits == 1:
            return binary_codes(self.weight.data, axis=1)
        qp = self.weight_qparams()
        return quantize(self.weight.data, qp).astype(np.int8), qp.scale

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self.fake:
            w = binarize(w, axis=1) if self.bits == 1 else fake_quant(w, self.weight_qparams())
        out = x @ w
        bias = self._parameters.get("bias")
        return out if bias is None else out + bias

    def extra_repr(self) -> str:
        n_in, n_out = self.weight.data.shape
        return f"{n_in}, {n_out}, bits={self.bits}, fake={self.fake}"


def set_quant_state(model: Module, observe: bool | None = None, fake: bool | None = None) -> None:
    """Flip ``observe`` / ``fake`` on every quantizer in ``model`` (None = leave)."""
    for m in model.modules():
        if observe is not None and isinstance(m, FakeQuantize):
            m.observe = observe
        if fake is not None and isinstance(m, (FakeQuantize, QATLinear)):
            m.fake = fake
