"""Integer-only inference: int8 x int8 -> int32 accumulate -> fixed-point requantize.

For one layer ``y = x @ W + b`` with ``x = s_x (q_x - z_x)``, per-output-channel
symmetric weights ``W[:, c] = s_w[c] q_w[:, c]`` and ``y = s_y (q_y - z_y)``::

    q_y[c] = z_y + M[c] * sum_k (q_x[k] - z_x) q_w[k, c]  +  b[c] / s_y
    M[c]   = s_x s_w[c] / s_y

The sum is an int32 accumulator; ``M`` is applied as a fixed-point multiply,
``M = M0 * 2^-shift`` with a 31-bit ``M0`` (:func:`quantize_multiplier`), and
the bias enters as the integer ``round(b / s_y * 2^shift)`` added to the int64
product before the rounding shift. No float touches a layer's hot path. The
traps:

**1. ``int8 @ int8`` in NumPy returns int8 and wraps silently.** ``[100, 100] @
[2, 2]`` is ``-112``, not 400. Codes are widened to int32 *before* the matmul.

**2. The int32 accumulator has a budget.** ``|acc| <= K * max|q_x - z_x| *
max|q_w|``. For MNIST's first layer that is ``784 * 255 * 127 ~ 2.5e7``, 85x
under ``2^31``. It is checked at conversion time, because an int32 matmul that
overflows wraps without a word.

**3. The textbook int32 bias is too coarse at low bit-widths.** The usual scheme
quantizes the bias on the accumulator grid, ``q_b = round(b / (s_x s_w))``, and
adds it into the int32 accumulator. Its rounding error is then up to ``M / 2``
*output* codes: negligible at W8A8 (``M ~ 3e-4``), but at W2A2 on MNIST the
second layer has ``M ~ 11``, so the bias alone is off by up to ~5 logit codes,
and the integer model agreed with its own float simulation on only 78% of
predictions. Adding the bias after the multiply, on the output's own
``2^-shift`` grid, costs nothing (the product is int64 anyway) and removes the
error: agreement becomes 100%. It is stored as int64.

**4. Fixed-point rounding is half toward +inf.** ``(acc * M0 + 2^(shift-1)) >>
shift`` floors after adding a half, so ``+0.5 -> 1`` and ``-0.5 -> 0``. The
float path (``np.round``) rounds half to even. They differ only on exact ties;
the tests pin both conventions.

**5. ``acc * M0`` needs int64.** ``|acc| < 2^31`` and ``M0 < 2^31`` so the
product is below ``2^62``; product + bias + rounding half is checked against
``2^63`` at conversion time. int32 never could hold it.

**6. Argmax over raw accumulators is wrong under per-channel scales.** Column
``c`` of the int32 accumulator is in units of ``s_x s_w[c]``, a different unit
per class, so ``argmax(acc)`` compares apples with oranges. The logits are
requantized to one common output scale first; ``argmax`` commutes with that
single positive affine map.

**7. The requantization clamp is the ReLU.** Calibrated after a ReLU, the output
range is ``[0, max]``, so ``z_y = qmin`` and clamping at ``z_y`` *is*
``max(0, .)``. A fused ReLU costs nothing.
"""

from __future__ import annotations

import numpy as np

from ..nn.layers import ReLU
from ..nn.module import Module
from .affine import QParams, quantize
from .fake import FakeQuantize, QATLinear

__all__ = [
    "IntegerMLP",
    "QuantizedLinear",
    "convert",
    "quantize_multiplier",
    "requantize",
]

_INT32_MAX = 2**31 - 1


def quantize_multiplier(m) -> tuple[np.ndarray, np.ndarray]:
    """Split real multipliers ``m > 0`` into ``M0 * 2^-shift``.

    ``np.frexp`` gives ``m = f * 2^e`` with ``f`` in ``[0.5, 1)``; then
    ``M0 = round(f * 2^31)`` lies in ``[2^30, 2^31]``, with the top value carried
    back down to ``2^30`` and ``e + 1``. The relative error of ``M0 * 2^-shift``
    is at most ``2^-31``.

    Returns:
        ``(M0, shift)`` as int64 arrays, ``shift = 31 - e`` in ``1..62``.

    Raises:
        ValueError: If any ``m`` is non-positive or so far from 1 that the shift
            leaves ``1..62`` (an int64 cannot hold the rounding term).
    """
    m = np.asarray(m, dtype=np.float64)
    if np.any(~(m > 0)):
        raise ValueError("requantization multipliers must be positive and finite")
    frac, exp = np.frexp(m)
    m0 = np.round(frac * 2.0**31).astype(np.int64)
    carry = m0 == 2**31
    m0 = np.where(carry, 2**30, m0)
    shift = 31 - (exp.astype(np.int64) + carry)
    if np.any(shift < 1) or np.any(shift > 62):
        raise ValueError(f"multiplier out of fixed-point range: shift {shift.min()}..{shift.max()}")
    return m0, shift


def requantize(acc, m0, shift, zero_point, qmin: int, qmax: int, bias=0) -> np.ndarray:
    """``clamp(z + round_half_up((acc * M0 + bias) / 2^shift), qmin, qmax)``, integers only.

    ``m0``, ``shift`` and ``bias`` (int64, already on the ``2^-shift`` grid)
    broadcast along the last axis -- one per output channel. Returns int64; the
    caller casts to the storage type.
    """
    prod = np.asarray(acc).astype(np.int64) * m0 + bias  # trap 5: int64, never int32
    half = np.left_shift(np.int64(1), shift - 1)
    return np.clip(((prod + half) >> shift) + zero_point, qmin, qmax)


class QuantizedLinear:
    """One integer-only dense layer: codes in, codes out.

    Build it with :meth:`from_codes`. Not a :class:`Module`: nothing here has a
    gradient, and every array is an integer array.

    Attributes:
        w_q: int8 weight codes ``(in, out)``.
        b_fx: int64 bias ``round(b / s_y * 2^shift)`` ``(out,)`` (trap 3).
        m0, shift: Per-channel fixed-point requantization multiplier.
        in_zp: Input zero point.
        out_qp: Output quantizer (its zero point and code range are used).
        relu: Clamp below at the output zero point (a fused ReLU).
        bits: Weight code width, for :meth:`packed_nbytes`.
    """

    def __init__(self, w_q, b_fx, m0, shift, in_zp: int, out_qp: QParams, relu: bool, bits: int):
        self.w_q, self.b_fx, self.m0, self.shift = w_q, b_fx, m0, shift
        self.in_zp, self.out_qp, self.relu, self.bits = int(in_zp), out_qp, bool(relu), int(bits)
        self._w32 = w_q.astype(np.int32)  # widened once (trap 1)
        self._lo = max(out_qp.qmin, int(out_qp.zero_point)) if relu else out_qp.qmin

    @classmethod
    def from_codes(
        cls,
        w_q: np.ndarray,
        w_scale: np.ndarray,
        bias: np.ndarray | None,
        in_qp: QParams,
        out_qp: QParams,
        relu: bool = False,
        bits: int = 8,
    ) -> QuantizedLinear:
        """Fold float scales into integer multipliers; check the int32/int64 budgets.

        Args:
            w_q: Weight codes ``(in, out)``, symmetric (zero point 0).
            w_scale: Per-output-channel weight scale ``(out,)``.
            bias: Float bias ``(out,)`` or None.
            in_qp: Per-tensor input quantizer.
            out_qp: Per-tensor output quantizer.
            relu: Fuse a ReLU into the output clamp.
            bits: Weight code width (bookkeeping only).
        """
        if in_qp.axis is not None or out_qp.axis is not None:
            raise ValueError("activations must be quantized per tensor")
        w_q = np.asarray(w_q, dtype=np.int8)
        s_x, s_y = float(in_qp.scale), float(out_qp.scale)
        span = max(in_qp.qmax - int(in_qp.zero_point), int(in_qp.zero_point) - in_qp.qmin)
        acc_bound = w_q.shape[0] * span * int(np.abs(w_q.astype(np.int32)).max(initial=0))
        if acc_bound > _INT32_MAX:
            raise ValueError(f"int32 accumulator can reach {acc_bound:.3g} > 2^31 - 1")
        m0, shift = quantize_multiplier(s_x * np.asarray(w_scale, dtype=np.float64) / s_y)
        b = np.zeros(w_q.shape[1]) if bias is None else np.asarray(bias, dtype=np.float64)
        b_fx = np.round(b / s_y * np.exp2(shift.astype(np.float64)))
        worst = acc_bound * 2.0**31 + np.abs(b_fx) + np.exp2(shift - 1.0)
        if np.any(worst >= 2.0**63):
            raise ValueError("bias or multiplier too large for the int64 requantizer")
        return cls(w_q, b_fx.astype(np.int64), m0, shift, int(in_qp.zero_point), out_qp, relu, bits)

    def __call__(self, x_q: np.ndarray) -> np.ndarray:
        x32 = np.asarray(x_q).astype(np.int32) - np.int32(self.in_zp)
        # Integer matmul never reaches BLAS. Plain einsum's loop is ~2x faster
        # than ``@`` here; ``optimize=True`` would route through tensordot and be
        # slower -- the opposite of the float conv rule. Same int32 semantics.
        acc = np.einsum("nk,kc->nc", x32, self._w32)
        z = int(self.out_qp.zero_point)
        y = requantize(acc, self.m0, self.shift, z, self._lo, self.out_qp.qmax, self.b_fx)
        return y.astype(self.out_qp.dtype)

    def packed_nbytes(self) -> int:
        """Deployed size: weights bit-packed at ``bits``, int64 bias, int32 M0, int8 shift."""
        n_out = self.w_q.shape[1]
        return -(-self.w_q.size * self.bits // 8) + (8 + 4 + 1) * n_out


class IntegerMLP:
    """A chain of :class:`QuantizedLinear` behind one input quantizer.

    The input quantize is the only float operation, and for MNIST scaled to
    ``[0, 1]`` it just recovers the original pixel bytes (``s = 1/255, z = 0``).
    """

    def __init__(self, in_qp: QParams, layers: list[QuantizedLinear]) -> None:
        self.in_qp, self.layers = in_qp, list(layers)

    def codes(self, x: np.ndarray) -> np.ndarray:
        """Output codes (common output scale) for a float input batch."""
        q = quantize(x, self.in_qp)
        for layer in self.layers:
            q = layer(q)
        return q

    def predict(self, x: np.ndarray, batch: int = 2000) -> np.ndarray:
        """Class indices: argmax over the requantized logit codes (trap 6)."""
        chunks = [self.codes(x[i : i + batch]).argmax(1) for i in range(0, len(x), batch)]
        return np.concatenate(chunks)

    def packed_nbytes(self) -> int:
        return sum(layer.packed_nbytes() for layer in self.layers)


def convert(model: Module) -> IntegerMLP:
    """Turn a calibrated fake-quant chain into an :class:`IntegerMLP`.

    Expects the children of ``model`` in the order
    ``FakeQuantize, (QATLinear, [ReLU], FakeQuantize)+`` -- the input quantizer,
    then each layer followed by the quantizer of its output.
    """
    mods = list(model.children())
    if not mods or not isinstance(mods[0], FakeQuantize):
        raise ValueError("model must start with a FakeQuantize for its input")
    in_qp = mods[0].qparams()
    first_in, layers, i = in_qp, [], 1
    while i < len(mods):
        lin = mods[i]
        if not isinstance(lin, QATLinear):
            raise ValueError(f"expected QATLinear at child {i}, got {type(lin).__name__}")
        relu = i + 1 < len(mods) and isinstance(mods[i + 1], ReLU)
        j = i + 1 + relu
        if j >= len(mods) or not isinstance(mods[j], FakeQuantize):
            raise ValueError(f"QATLinear at child {i} must be followed by a FakeQuantize")
        out_qp = mods[j].qparams()
        w_q, w_scale = lin.weight_codes()
        bias = lin._parameters.get("bias")
        layers.append(
            QuantizedLinear.from_codes(
                w_q, w_scale, None if bias is None else bias.data, in_qp, out_qp, relu, lin.bits
            )
        )
        in_qp, i = out_qp, j + 1
    return IntegerMLP(first_in, layers)
