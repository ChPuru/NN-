"""Quantized neural networks: affine quantization, QAT, and integer-only inference.

Three layers, each built on the one before:

* :mod:`.affine` -- the quantizer itself (scale, zero point, code range,
  per-tensor or per-channel) and a min-max observer for calibration.
* :mod:`.fake` -- ``fake_quant`` with the clipped straight-through estimator,
  binary weights, and the two modules quantization-aware training needs.
* :mod:`.integer` -- int8 x int8 -> int32 -> fixed-point requantize, and a
  converter from a calibrated fake-quant model to an integer-only one.
"""

from __future__ import annotations

from .affine import MinMaxObserver, QParams, calc_qparams, dequantize, qrange, quantize
from .fake import (
    FakeQuantize,
    QATLinear,
    binarize,
    binary_codes,
    fake_quant,
    set_quant_state,
)
from .integer import IntegerMLP, QuantizedLinear, convert, quantize_multiplier, requantize

__all__ = [
    "FakeQuantize",
    "IntegerMLP",
    "MinMaxObserver",
    "QATLinear",
    "QParams",
    "QuantizedLinear",
    "binarize",
    "binary_codes",
    "calc_qparams",
    "convert",
    "dequantize",
    "fake_quant",
    "qrange",
    "quantize",
    "quantize_multiplier",
    "requantize",
    "set_quant_state",
]
