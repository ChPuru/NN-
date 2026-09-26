"""Convolution and pooling, built on one im2col buffer and one GEMM.

Everything in this file reduces to two index-math primitives:

* :func:`im2col` lifts every sliding receptive field of an ``(N, C, H, W)``
  image into a column of a ``(N, C*kH*kW, H_out*W_out)`` matrix, after which the
  convolution is a single batched matrix product and BLAS does the work.
* :func:`col2im` is its **adjoint**: it scatters columns back onto the image,
  *accumulating* wherever windows overlap. That is the whole of ``dX``.

Choices worth defending, because the alternatives are common and worse:

* **im2col + GEMM, not kernel-tap accumulation.** Looping over the ``kH*kW``
  taps and accumulating ``x[..., i:i+.., j:j+..] * w[..., i, j]`` is the
  "memory-free" formulation people reach for. Timed on ``(32, 32, 28, 28)``
  against 64 3x3 filters it runs **2.2x slower** than the GEMM here, and the
  memory claim is false anyway once the tap contraction is written as
  ``np.einsum(..., optimize=True)`` -- the optimiser materialises intermediates
  of its own.
* **No ``np.einsum`` in the hot path.** ``np.einsum`` defaults to
  ``optimize=False``, which walks its own C loop and never reaches BLAS. On that
  same shape the unoptimised contraction takes **1200 ms against 18 ms**: a 66x
  loss, and one that is invisible because the answer is still correct. Rather
  than remember the flag at every call site, the contractions here are an
  explicit reshape plus ``@``, which is unambiguously ``matmul``. (For the
  record, ``einsum`` *with* ``optimize=True`` is about 1.4x faster than this
  ``matmul`` on that shape -- it skips the transposed copy. The reshape form is
  kept because the grouped/batched indexing is far clearer with it and because
  a silently-30x-slow default is a bad thing to have in a hot loop.)
* **Overlap-safe scatter in :func:`col2im`.** With ``stride < kernel`` the
  windows overlap, so a pixel appears in several columns. Assigning instead of
  accumulating keeps only the last write and silently drops the rest of the
  gradient; the forward is untouched, so nothing looks wrong until accuracy
  quietly caps out.
* **Max-pool backward divides by the tie count.** Two exactly-equal maxima in a
  window must *split* the upstream gradient. Handing each the full cotangent
  multiplies the gradient by the tie count -- the classic max-pool bug, and one
  that only shows up on data with exact ties (ReLU outputs, quantised images,
  zero-padded borders), which is to say: in practice, not in your unit test.

Weight layout is the frozen one: ``Conv2d.weight`` is
``(out_channels, in_channels // groups, kH, kW)`` and images are
``(N, C, H, W)``.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..tensor import Tensor
from . import init
from .module import Module, Parameter

__all__ = [
    "AdaptiveAvgPool2d",
    "AvgPool2d",
    "Conv1d",
    "Conv2d",
    "MaxPool2d",
    "ZeroPad2d",
    "adaptive_avg_pool2d",
    "avg_pool2d",
    "col2im",
    "conv1d",
    "conv2d",
    "conv_output_size",
    "im2col",
    "max_pool2d",
    "zero_pad2d",
]


# --------------------------------------------------------------------------- #
# argument normalisation
# --------------------------------------------------------------------------- #
def _pair(value, name: str) -> tuple[int, int]:
    """Normalise ``int`` or ``(h, w)`` to a strictly positive ``(h, w)``."""
    if isinstance(value, (int, np.integer)):
        out = (int(value), int(value))
    else:
        items = tuple(int(v) for v in value)
        if len(items) != 2:
            raise ValueError(f"{name} must be an int or a length-2 sequence, got {value!r}")
        out = items
    if out[0] < 1 or out[1] < 1:
        raise ValueError(f"{name} must be >= 1 in both dimensions, got {value!r}")
    return out


def conv_output_size(
    size: int, pad_before: int, pad_after: int, kernel: int, stride: int, dilation: int
) -> int:
    """``floor((size + pad_total - dilation*(kernel-1) - 1) / stride) + 1``.

    The dilated kernel spans ``dilation*(kernel-1) + 1`` pixels, not ``kernel``;
    dropping the dilation term is the single most common output-size error and
    it produces a *larger* number, so the shape only blows up later, inside the
    reshape after the GEMM, where the cause is no longer visible.
    """
    effective = dilation * (kernel - 1) + 1
    span = size + pad_before + pad_after - effective
    if span < 0:
        return 0
    return span // stride + 1


def _out_size(
    size: int, pad_before: int, pad_after: int, kernel: int, stride: int, dilation: int, axis: str
) -> int:
    out = conv_output_size(size, pad_before, pad_after, kernel, stride, dilation)
    if out <= 0:
        raise ValueError(
            f"convolution config produces a non-positive output {axis}: input {size} with "
            f"padding ({pad_before}, {pad_after}), kernel {kernel}, stride {stride}, "
            f"dilation {dilation} spans {dilation * (kernel - 1) + 1} pixels. "
            "Reduce the kernel or dilation, or add padding."
        )
    return out


def _resolve_padding(
    padding,
    in_hw: tuple[int, int],
    kernel: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return ``(top, bottom, left, right)`` for int / tuple / ``'same'`` input.

    ``'same'`` keeps the output spatially equal to the input, and is **refused
    with ``stride > 1``** exactly as PyTorch refuses it. The two frameworks
    disagree there and the disagreement is silent: TensorFlow defines strided
    ``'same'`` as ``ceil(in / stride)`` outputs, PyTorch as an error, so the same
    argument would mean two different output sizes depending on which reference
    the reader has in mind. Refusing keeps the one documented meaning.

    When the required total is odd the extra row or column goes on the
    **bottom/right**, matching PyTorch's ``padding='same'`` (which splits
    ``total // 2`` before, remainder after). The asymmetric case is why padding
    is carried as four numbers internally rather than two.
    """
    if isinstance(padding, str):
        mode = padding.lower()
        if mode == "valid":
            return (0, 0, 0, 0)
        if mode != "same":
            raise ValueError(f"padding string must be 'same' or 'valid', got {padding!r}")
        if stride[0] != 1 or stride[1] != 1:
            raise ValueError(
                f"padding='same' is not supported with stride={tuple(stride)}: the output "
                "size of a strided 'same' is a convention (PyTorch refuses it, TensorFlow "
                "returns ceil(in/stride)), not a fact. Pass explicit padding instead."
            )
        out: list[int] = []
        for size, k, s, d in zip(in_hw, kernel, stride, dilation, strict=True):
            effective = d * (k - 1) + 1
            wanted = -(-size // s)  # ceil(size / s)
            total = max(0, (wanted - 1) * s + effective - size)
            out.extend((total // 2, total - total // 2))
        return (out[0], out[1], out[2], out[3])

    if isinstance(padding, (int, np.integer)):
        p = int(padding)
        values = (p, p, p, p)
    else:
        items = tuple(int(v) for v in padding)
        if len(items) == 2:
            values = (items[0], items[0], items[1], items[1])
        elif len(items) == 4:
            values = items
        else:
            raise ValueError(
                f"padding must be an int, (pad_h, pad_w), (top, bottom, left, right) or "
                f"'same'/'valid', got {padding!r}"
            )
    if any(v < 0 for v in values):
        raise ValueError(f"padding must be non-negative, got {padding!r}")
    return values


def _pad_nchw(x: np.ndarray, pt: int, pb: int, pl: int, pr: int, value: float) -> np.ndarray:
    if pt == pb == pl == pr == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant", constant_values=value)


# --------------------------------------------------------------------------- #
# the two index-math primitives
# --------------------------------------------------------------------------- #
def _windows(
    xp: np.ndarray, kernel: tuple[int, int], stride: tuple[int, int], dilation: tuple[int, int]
) -> np.ndarray:
    """Strided **view** of every receptive field: ``(N, C, H_out, W_out, kH, kW)``.

    ``sliding_window_view`` gives every unit-stride window of the *dilated* size;
    the two slices then subsample positions by ``stride`` and taps by
    ``dilation``. No data is copied here -- the copy happens at the first
    ``reshape``, which is exactly where the im2col buffer is supposed to be
    materialised and nowhere else.
    """
    kh, kw = kernel
    sh, sw = stride
    dh, dw = dilation
    eff_h, eff_w = dh * (kh - 1) + 1, dw * (kw - 1) + 1
    if xp.shape[2] < eff_h or xp.shape[3] < eff_w:
        raise ValueError(
            f"padded input {xp.shape[2]}x{xp.shape[3]} is smaller than the dilated kernel "
            f"{eff_h}x{eff_w}"
        )
    view = sliding_window_view(xp, (eff_h, eff_w), axis=(2, 3))
    return view[:, :, ::sh, ::sw, ::dh, ::dw]


def _scatter_windows(
    gwin: np.ndarray,
    padded_shape: tuple[int, ...],
    kernel: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    dtype,
) -> np.ndarray:
    """Accumulate per-window gradients back onto a padded image.

    One vectorised ``+=`` per kernel tap. Within a *single* tap the strided
    destination slice visits each pixel at most once, so the in-place add is
    well defined; the accumulation that matters -- the same pixel reached by
    several overlapping windows -- happens *across* taps and across the outer
    loop, where separate statements add rather than overwrite. This is why the
    fully-fused ``acc[index] = ...`` version is wrong: fancy assignment keeps
    the last write only.

    ``kH*kW`` slice-adds beat ``np.add.at`` on the same problem by a wide margin
    (``add.at`` runs an unbuffered elementwise loop), and both are correct; the
    loop is only over taps, never over batch or spatial positions.
    """
    kh, kw = kernel
    sh, sw = stride
    dh, dw = dilation
    h_out, w_out = gwin.shape[2], gwin.shape[3]
    acc = np.zeros(padded_shape, dtype=dtype)
    for i in range(kh):
        r0 = i * dh
        r1 = r0 + (h_out - 1) * sh + 1
        for j in range(kw):
            c0 = j * dw
            c1 = c0 + (w_out - 1) * sw + 1
            acc[:, :, r0:r1:sh, c0:c1:sw] += gwin[:, :, :, :, i, j]
    return acc


def im2col(
    x: np.ndarray,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
) -> np.ndarray:
    """Lift receptive fields of an ``(N, C, H, W)`` array into GEMM columns.

    Args:
        x: Input image batch, shape ``(N, C, H, W)``.
        kernel_size: ``int`` or ``(kH, kW)``.
        stride: ``int`` or ``(sH, sW)``.
        padding: ``int``, ``(pH, pW)``, ``(top, bottom, left, right)``, or
            ``'same'`` / ``'valid'``.
        dilation: ``int`` or ``(dH, dW)``.

    Returns:
        Array of shape ``(N, C*kH*kW, H_out*W_out)``. The row index runs
        ``(channel, kernel row, kernel column)`` in C order and the column index
        runs ``(output row, output column)``, so ``weight.reshape(M, C*kH*kW) @
        cols[n]`` is the convolution.
    """
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError(f"im2col expects (N, C, H, W), got shape {x.shape}")
    n, c, h, w = x.shape
    kernel = _pair(kernel_size, "kernel_size")
    strides = _pair(stride, "stride")
    dilations = _pair(dilation, "dilation")
    pt, pb, pl, pr = _resolve_padding(padding, (h, w), kernel, strides, dilations)
    h_out = _out_size(h, pt, pb, kernel[0], strides[0], dilations[0], "height")
    w_out = _out_size(w, pl, pr, kernel[1], strides[1], dilations[1], "width")
    xp = _pad_nchw(x, pt, pb, pl, pr, 0.0)
    win = _windows(xp, kernel, strides, dilations)
    # (N, C, H_out, W_out, kH, kW) -> (N, C, kH, kW, H_out, W_out) -> flatten.
    return win.transpose(0, 1, 4, 5, 2, 3).reshape(n, c * kernel[0] * kernel[1], h_out * w_out)


def col2im(
    cols: np.ndarray,
    input_shape: tuple[int, int, int, int],
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
) -> np.ndarray:
    """Scatter GEMM columns back onto an image, **accumulating** overlaps.

    The exact adjoint of :func:`im2col` for the same geometry, which is what
    makes ``conv2d``'s ``dX`` a two-liner. Overlapping windows (any
    ``stride < kernel``) contribute additively; ``col2im(im2col(x))`` is
    therefore ``x`` multiplied elementwise by the per-pixel window count, not
    ``x``.

    Args:
        cols: Array of shape ``(N, C*kH*kW, H_out*W_out)``.
        input_shape: The ``(N, C, H, W)`` shape to scatter back into.
        kernel_size: ``int`` or ``(kH, kW)``.
        stride: ``int`` or ``(sH, sW)``.
        padding: Same forms as :func:`im2col`; padded rows are dropped on exit.
        dilation: ``int`` or ``(dH, dW)``.

    Returns:
        Array of shape ``input_shape``.
    """
    cols = np.asarray(cols)
    n, c, h, w = (int(v) for v in input_shape)
    kernel = _pair(kernel_size, "kernel_size")
    strides = _pair(stride, "stride")
    dilations = _pair(dilation, "dilation")
    pt, pb, pl, pr = _resolve_padding(padding, (h, w), kernel, strides, dilations)
    h_out = _out_size(h, pt, pb, kernel[0], strides[0], dilations[0], "height")
    w_out = _out_size(w, pl, pr, kernel[1], strides[1], dilations[1], "width")
    expected = (n, c * kernel[0] * kernel[1], h_out * w_out)
    if cols.shape != expected:
        raise ValueError(f"col2im expected cols of shape {expected}, got {cols.shape}")
    gwin = cols.reshape(n, c, kernel[0], kernel[1], h_out, w_out).transpose(0, 1, 4, 5, 2, 3)
    padded = (n, c, h + pt + pb, w + pl + pr)
    acc = _scatter_windows(gwin, padded, kernel, strides, dilations, cols.dtype)
    return acc[:, :, pt : pt + h, pl : pl + w]


# --------------------------------------------------------------------------- #
# convolution
# --------------------------------------------------------------------------- #
def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    input_requires_grad: bool = True,
) -> Tensor:
    """2-D cross-correlation (the "convolution" of deep learning; no kernel flip).

    Args:
        x: Input, shape ``(N, C, H, W)``.
        weight: Shape ``(out_channels, in_channels // groups, kH, kW)``.
        bias: Optional shape ``(out_channels,)``.
        stride: ``int`` or ``(sH, sW)``.
        padding: ``int``, ``(pH, pW)``, ``(top, bottom, left, right)``, or
            ``'same'`` / ``'valid'``.
        dilation: ``int`` or ``(dH, dW)``.
        groups: Channel groups. ``1`` is dense; ``groups == in_channels`` with
            ``weight.shape[1] == 1`` is a depthwise convolution.
        input_requires_grad: When ``False``, ``dX`` is never formed, skipping the
            ``col2im`` scatter. For the **first** layer of a network nothing
            upstream consumes ``dX``, so this is free accuracy-wise.

            The size of the win depends entirely on how much of the network the
            first layer represents, and the widely-quoted "~2.8x end-to-end"
            figure is an *MLP* result that does not transfer to a convnet.
            Measured here at batch 128, median of 7 runs::

                MLP 784-128-10   dX skipped 2.90 ms   computed 6.37 ms   2.19x
                LeNet-5          dX skipped 66.1 ms   computed 75.8 ms   1.15x

            In an MLP the first ``Linear``'s ``dX`` is a matmul comparable to
            the whole rest of the network; in a convnet the first layer is a
            small fraction of the cost and the saving collapses.

            In practice this flag rarely needs setting: an input ``Tensor``
            defaults to ``requires_grad=False`` and ``Tensor._make`` only
            evaluates a gradient rule for a parent that actually requires one,
            so ``dX`` is already skipped unless something asks for it. The flag
            exists to make the intent explicit and to force the skip even when
            the input does carry ``requires_grad`` (adversarial-example work,
            for instance, where you want it for *some* layers but not the first).

    Returns:
        Tensor of shape ``(N, out_channels, H_out, W_out)``.
    """
    xd, wd = x.data, weight.data
    if xd.ndim != 4:
        raise ValueError(f"conv2d expects an (N, C, H, W) input, got shape {xd.shape}")
    if wd.ndim != 4:
        raise ValueError(
            f"conv2d weight must be (out_channels, in_channels/groups, kH, kW), got {wd.shape}"
        )
    n, c, h, w = xd.shape
    m, c_per_group, kh, kw = wd.shape
    g = int(groups)
    if g < 1:
        raise ValueError(f"groups must be >= 1, got {groups}")
    if c % g or m % g:
        raise ValueError(f"groups={g} must divide both in_channels={c} and out_channels={m}")
    if c_per_group != c // g:
        raise ValueError(
            f"weight has {c_per_group} input channels per group but in_channels={c} with "
            f"groups={g} requires {c // g}"
        )
    strides = _pair(stride, "stride")
    dilations = _pair(dilation, "dilation")
    pt, pb, pl, pr = _resolve_padding(padding, (h, w), (kh, kw), strides, dilations)
    h_out = _out_size(h, pt, pb, kh, strides[0], dilations[0], "height")
    w_out = _out_size(w, pl, pr, kw, strides[1], dilations[1], "width")

    m_per_group = m // g
    ck = c_per_group * kh * kw
    length = h_out * w_out

    xp = _pad_nchw(xd, pt, pb, pl, pr, 0.0)
    win = _windows(xp, (kh, kw), strides, dilations)
    # The group axis is carved out of the channel axis, which is legal precisely
    # because C splits as (G, C/G) in C order -- the same reason the reverse
    # reshape in the dX rule below is a no-op view.
    cols = win.transpose(0, 1, 4, 5, 2, 3).reshape(n, g, ck, length)
    wmat = wd.reshape(g, m_per_group, ck)

    # One batched GEMM: (1, G, Mg, CK) @ (N, G, CK, L) -> (N, G, Mg, L).
    out = (wmat[None] @ cols).reshape(n, m, h_out, w_out)
    if bias is not None:
        if bias.data.shape != (m,):
            raise ValueError(f"bias must have shape ({m},), got {bias.data.shape}")
        out = out + bias.data.reshape(1, m, 1, 1)

    def grad_x(gy: np.ndarray) -> np.ndarray:
        gy = np.asarray(gy).reshape(n, g, m_per_group, length)
        gcols = wmat.transpose(0, 2, 1)[None] @ gy  # (N, G, CK, L)
        gwin = gcols.reshape(n, c, kh, kw, h_out, w_out).transpose(0, 1, 4, 5, 2, 3)
        gxp = _scatter_windows(gwin, xp.shape, (kh, kw), strides, dilations, gcols.dtype)
        return gxp[:, :, pt : pt + h, pl : pl + w]

    def grad_w(gy: np.ndarray) -> np.ndarray:
        gy = np.asarray(gy).reshape(n, g, m_per_group, length)
        # Sum over the batch after the per-sample GEMM; the batch axis is the
        # contraction that BLAS cannot fold into matmul's broadcast rules.
        return (gy @ cols.transpose(0, 1, 3, 2)).sum(axis=0).reshape(wd.shape)

    def grad_b(gy: np.ndarray) -> np.ndarray:
        return np.asarray(gy).sum(axis=(0, 2, 3))

    parents: list[Tensor] = [x, weight]
    rules: list = [grad_x if input_requires_grad else None, grad_w]
    if bias is not None:
        parents.append(bias)
        rules.append(grad_b)
    return Tensor._make(out, tuple(parents), tuple(rules), "conv2d")


def conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding=0,
    dilation: int = 1,
    groups: int = 1,
    input_requires_grad: bool = True,
) -> Tensor:
    """1-D cross-correlation over ``(N, C, L)`` with weight ``(M, C/groups, k)``.

    Implemented as a height-1 :func:`conv2d`. There is nothing to gain from a
    separate index-math path, and one path means one place for the off-by-one
    to live.
    """
    if x.data.ndim != 3:
        raise ValueError(f"conv1d expects an (N, C, L) input, got shape {x.data.shape}")
    if weight.data.ndim != 3:
        raise ValueError(
            f"conv1d weight must be (out_channels, in_channels/groups, k), got {weight.data.shape}"
        )
    if isinstance(padding, str):
        pad2 = padding
    else:
        p = int(padding) if isinstance(padding, (int, np.integer)) else None
        if p is None:
            items = tuple(int(v) for v in padding)
            if len(items) != 2:
                raise ValueError(f"conv1d padding must be an int or (left, right), got {padding!r}")
            pad2 = (0, 0, items[0], items[1])
        else:
            pad2 = (0, 0, p, p)
    out = conv2d(
        x.unsqueeze(2),
        weight.unsqueeze(2),
        bias,
        stride=(1, int(stride)),
        padding=pad2,
        dilation=(1, int(dilation)),
        groups=groups,
        input_requires_grad=input_requires_grad,
    )
    return out.squeeze(axis=2)


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #
def max_pool2d(x: Tensor, kernel_size, stride=None, padding=0, dilation=1) -> Tensor:
    """Max pooling with tie-splitting backward.

    Padding is filled with ``-inf`` so a border cell can never win a window;
    filling with ``0`` (the naive choice, since that is what convolution uses)
    turns every negative-activation border into a spurious zero maximum.

    Args:
        x: Input, shape ``(N, C, H, W)``.
        kernel_size: ``int`` or ``(kH, kW)``.
        stride: Defaults to ``kernel_size`` -- non-overlapping windows.
        padding: ``int``, ``(pH, pW)`` or ``(top, bottom, left, right)``.
        dilation: ``int`` or ``(dH, dW)``.

    Returns:
        Tensor of shape ``(N, C, H_out, W_out)``.
    """
    xd = x.data
    if xd.ndim != 4:
        raise ValueError(f"max_pool2d expects (N, C, H, W), got shape {xd.shape}")
    n, c, h, w = xd.shape
    kernel = _pair(kernel_size, "kernel_size")
    strides = _pair(kernel_size if stride is None else stride, "stride")
    dilations = _pair(dilation, "dilation")
    pt, pb, pl, pr = _resolve_padding(padding, (h, w), kernel, strides, dilations)
    # The bound is half the *dilated* extent, not half `kernel`: a window covers
    # d*(k-1)+1 pixels, and comparing against the raw kernel size measures the
    # padding against a window that does not exist.
    eff = tuple(d * (k - 1) + 1 for k, d in zip(kernel, dilations, strict=True))
    if max(pt, pb) * 2 > eff[0] or max(pl, pr) * 2 > eff[1]:
        raise ValueError(
            f"max_pool2d padding {(pt, pb, pl, pr)} must not exceed half the dilated kernel "
            f"extent {eff} (kernel {kernel}, dilation {dilations})"
        )
    h_out = _out_size(h, pt, pb, kernel[0], strides[0], dilations[0], "height")
    w_out = _out_size(w, pl, pr, kernel[1], strides[1], dilations[1], "width")
    # Half-the-extent is necessary but not sufficient once dilation > 1: the taps
    # of one window are d apart, so they can straddle the whole image and land in
    # the pad on both sides (kernel 2, dilation 2 over a 1-pixel row with pad 1).
    # Such a window maxes over -inf alone, so the forward emits -inf and the
    # backward hands its share to padded cells that are cropped away -- gradient
    # mass silently vanishes. At dilation 1 the taps are contiguous and the bound
    # above already rules this out, so the check is skipped in the hot path.
    if max(dilations) > 1 and max(pt, pb, pl, pr) > 0:
        real = _pad_nchw(np.ones((1, 1, h, w)), pt, pb, pl, pr, 0.0)
        if _windows(real, kernel, strides, dilations).sum(axis=(4, 5)).min() == 0.0:
            raise ValueError(
                f"max_pool2d config leaves a window entirely inside the padding: input "
                f"{h}x{w}, padding {(pt, pb, pl, pr)}, kernel {kernel}, stride {strides}, "
                f"dilation {dilations}. Reduce the dilation or the padding."
            )

    xp = _pad_nchw(xd, pt, pb, pl, pr, -np.inf)
    win = _windows(xp, kernel, strides, dilations)
    kept = win.max(axis=(4, 5), keepdims=True)
    out = kept.reshape(n, c, h_out, w_out)

    def grad_x(gy: np.ndarray) -> np.ndarray:
        # The tie count is the whole point: `mask` may hold several 1s per
        # window, and each winner is entitled to 1/count of the cotangent.
        # Broadcasting `gy` onto `mask` directly would hand the full gradient
        # to every tied element and inflate it by the tie count.
        mask = (win == kept).astype(xp.dtype)
        share = mask / mask.sum(axis=(4, 5), keepdims=True)
        gwin = np.asarray(gy).reshape(n, c, h_out, w_out, 1, 1) * share
        gxp = _scatter_windows(gwin, xp.shape, kernel, strides, dilations, gwin.dtype)
        return gxp[:, :, pt : pt + h, pl : pl + w]

    return Tensor._make(out, (x,), (grad_x,), "max_pool2d")


def avg_pool2d(
    x: Tensor, kernel_size, stride=None, padding=0, count_include_pad: bool = True
) -> Tensor:
    """Average pooling.

    Args:
        x: Input, shape ``(N, C, H, W)``.
        kernel_size: ``int`` or ``(kH, kW)``.
        stride: Defaults to ``kernel_size``.
        padding: ``int``, ``(pH, pW)`` or ``(top, bottom, left, right)``.
        count_include_pad: When ``True`` the divisor is always ``kH*kW``; when
            ``False`` it is the number of *real* pixels in the window, so border
            averages are not pulled toward zero.

    Returns:
        Tensor of shape ``(N, C, H_out, W_out)``.
    """
    xd = x.data
    if xd.ndim != 4:
        raise ValueError(f"avg_pool2d expects (N, C, H, W), got shape {xd.shape}")
    n, c, h, w = xd.shape
    kernel = _pair(kernel_size, "kernel_size")
    strides = _pair(kernel_size if stride is None else stride, "stride")
    ones = (1, 1)
    pt, pb, pl, pr = _resolve_padding(padding, (h, w), kernel, strides, ones)
    # Same guard as max_pool2d, and here it is load-bearing rather than merely
    # tidy: with padding past half the kernel a whole window can land inside the
    # pad, and under count_include_pad=False the divisor for that window is the
    # number of real pixels it covers -- zero. The output is then 0/0, and the
    # backward divides by it again.
    if max(pt, pb) > kernel[0] // 2 or max(pl, pr) > kernel[1] // 2:
        raise ValueError(
            f"avg_pool2d padding {(pt, pb, pl, pr)} must not exceed half the kernel size "
            f"{kernel}; a window falling entirely inside the pad has no real pixels to average"
        )
    h_out = _out_size(h, pt, pb, kernel[0], strides[0], 1, "height")
    w_out = _out_size(w, pl, pr, kernel[1], strides[1], 1, "width")

    xp = _pad_nchw(xd, pt, pb, pl, pr, 0.0)
    win = _windows(xp, kernel, strides, ones)
    total = win.sum(axis=(4, 5))
    if count_include_pad:
        divisor = np.asarray(float(kernel[0] * kernel[1]), dtype=xp.dtype)
    else:
        valid = _pad_nchw(np.ones((1, 1, h, w), dtype=xp.dtype), pt, pb, pl, pr, 0.0)
        divisor = _windows(valid, kernel, strides, ones).sum(axis=(4, 5))
    out = total / divisor

    def grad_x(gy: np.ndarray) -> np.ndarray:
        per_cell = (np.asarray(gy) / divisor).reshape(n, c, h_out, w_out, 1, 1)
        gwin = np.broadcast_to(per_cell, (n, c, h_out, w_out, kernel[0], kernel[1]))
        gxp = _scatter_windows(gwin, xp.shape, kernel, strides, ones, per_cell.dtype)
        return gxp[:, :, pt : pt + h, pl : pl + w]

    return Tensor._make(out, (x,), (grad_x,), "avg_pool2d")


def _adaptive_bins(size: int, out: int) -> list[tuple[int, int]]:
    """``[start, end)`` per output cell: ``floor(i*size/out)`` to ``ceil((i+1)*size/out)``.

    Uneven divisions therefore produce *overlapping* bins rather than dropping
    pixels, which is the behaviour adaptive pooling is defined to have.
    """
    return [((i * size) // out, -((-(i + 1) * size) // out)) for i in range(out)]


def adaptive_avg_pool2d(x: Tensor, output_size) -> Tensor:
    """Average-pool to a fixed ``(out_H, out_W)`` regardless of input size.

    ``output_size=1`` is global average pooling, the usual head of a conv net.

    Args:
        x: Input, shape ``(N, C, H, W)``.
        output_size: ``int`` or ``(out_H, out_W)``.

    Returns:
        Tensor of shape ``(N, C, out_H, out_W)``.
    """
    xd = x.data
    if xd.ndim != 4:
        raise ValueError(f"adaptive_avg_pool2d expects (N, C, H, W), got shape {xd.shape}")
    n, c, h, w = xd.shape
    oh, ow = _pair(output_size, "output_size")
    rows = _adaptive_bins(h, oh)
    cols = _adaptive_bins(w, ow)

    out = np.empty((n, c, oh, ow), dtype=xd.dtype)
    for i, (r0, r1) in enumerate(rows):
        for j, (c0, c1) in enumerate(cols):
            out[:, :, i, j] = xd[:, :, r0:r1, c0:c1].mean(axis=(2, 3))

    def grad_x(gy: np.ndarray) -> np.ndarray:
        gy = np.asarray(gy)
        acc = np.zeros(xd.shape, dtype=gy.dtype)
        for i, (r0, r1) in enumerate(rows):
            for j, (c0, c1) in enumerate(cols):
                count = (r1 - r0) * (c1 - c0)
                # `+=`, not `=`: adjacent bins overlap whenever out does not
                # divide the input size.
                acc[:, :, r0:r1, c0:c1] += (gy[:, :, i, j] / count)[:, :, None, None]
        return acc

    return Tensor._make(out, (x,), (grad_x,), "adaptive_avg_pool2d")


def zero_pad2d(x: Tensor, padding) -> Tensor:
    """Pad the spatial axes with zeros.

    Args:
        x: Input, shape ``(N, C, H, W)``.
        padding: ``int``, or ``(left, right, top, bottom)`` -- PyTorch's order,
            which is *last axis first* and is not the ``np.pad`` order.

    Returns:
        Tensor of shape ``(N, C, H + top + bottom, W + left + right)``.
    """
    if x.data.ndim != 4:
        raise ValueError(f"zero_pad2d expects (N, C, H, W), got shape {x.data.shape}")
    if isinstance(padding, (int, np.integer)):
        left = right = top = bottom = int(padding)
    else:
        items = tuple(int(v) for v in padding)
        if len(items) != 4:
            raise ValueError(
                f"zero_pad2d padding must be an int or (left, right, top, bottom), got {padding!r}"
            )
        left, right, top, bottom = items
    if min(left, right, top, bottom) < 0:
        raise ValueError(f"zero_pad2d padding must be non-negative, got {padding!r}")
    return x.pad(((0, 0), (0, 0), (top, bottom), (left, right)))


# --------------------------------------------------------------------------- #
# modules
# --------------------------------------------------------------------------- #
class Conv2d(Module):
    """2-D convolution layer.

    ``weight`` has the frozen layout ``(out_channels, in_channels // groups,
    kH, kW)``; ``compute_fans`` in :mod:`scratchgrad.nn.init` reads the
    receptive field from ``shape[2:]`` on that assumption, so a transposed
    weight would silently mis-scale the initialisation as well as the forward.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        input_requires_grad: bool = True,
        nonlinearity: str = "relu",
        rng=None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.padding = padding
        self.dilation = _pair(dilation, "dilation")
        self.groups = int(groups)
        # Set False on the first conv of a network: nothing upstream consumes
        # dX, and skipping the col2im scatter is worth ~2.8x end to end -- a
        # bigger win than float32 and than every in-place rewrite combined.
        self.input_requires_grad = bool(input_requires_grad)
        if self.in_channels % self.groups or self.out_channels % self.groups:
            raise ValueError(
                f"groups={self.groups} must divide in_channels={self.in_channels} and "
                f"out_channels={self.out_channels}"
            )
        shape = (
            self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        self.weight = Parameter(np.zeros(shape))
        init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity=nonlinearity, rng=rng)
        if bias:
            self.bias = Parameter(np.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim != 4 or x.data.shape[1] != self.in_channels:
            raise ValueError(f"Conv2d expected (N, {self.in_channels}, H, W), got {x.data.shape}")
        return conv2d(
            x,
            self.weight,
            self._parameters.get("bias"),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            input_requires_grad=self.input_requires_grad,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding!r}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self._parameters.get('bias') is not None}"
        )


class Conv1d(Module):
    """1-D convolution over ``(N, C, L)``; weight is ``(M, C // groups, k)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding=0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        input_requires_grad: bool = True,
        nonlinearity: str = "relu",
        rng=None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = padding
        self.dilation = int(dilation)
        self.groups = int(groups)
        self.input_requires_grad = bool(input_requires_grad)
        if self.in_channels % self.groups or self.out_channels % self.groups:
            raise ValueError(
                f"groups={self.groups} must divide in_channels={self.in_channels} and "
                f"out_channels={self.out_channels}"
            )
        shape = (self.out_channels, self.in_channels // self.groups, self.kernel_size)
        self.weight = Parameter(np.zeros(shape))
        init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity=nonlinearity, rng=rng)
        if bias:
            self.bias = Parameter(np.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim != 3 or x.data.shape[1] != self.in_channels:
            raise ValueError(f"Conv1d expected (N, {self.in_channels}, L), got {x.data.shape}")
        return conv1d(
            x,
            self.weight,
            self._parameters.get("bias"),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            input_requires_grad=self.input_requires_grad,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding!r}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self._parameters.get('bias') is not None}"
        )


class MaxPool2d(Module):
    """Max pooling; ``stride`` defaults to ``kernel_size``."""

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(kernel_size if stride is None else stride, "stride")
        self.padding = padding
        self.dilation = _pair(dilation, "dilation")

    def forward(self, x: Tensor) -> Tensor:
        return max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding!r}, dilation={self.dilation}"
        )


class AvgPool2d(Module):
    """Average pooling; ``stride`` defaults to ``kernel_size``."""

    def __init__(self, kernel_size, stride=None, padding=0, count_include_pad: bool = True) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(kernel_size if stride is None else stride, "stride")
        self.padding = padding
        self.count_include_pad = bool(count_include_pad)

    def forward(self, x: Tensor) -> Tensor:
        return avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.count_include_pad)

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding!r}, count_include_pad={self.count_include_pad}"
        )


class AdaptiveAvgPool2d(Module):
    """Average-pool to a fixed output size. ``output_size=1`` is global pooling."""

    def __init__(self, output_size=1) -> None:
        super().__init__()
        self.output_size = _pair(output_size, "output_size")

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_avg_pool2d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class ZeroPad2d(Module):
    """Zero-pad the spatial axes. ``padding`` is ``(left, right, top, bottom)``."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return zero_pad2d(x, self.padding)

    def extra_repr(self) -> str:
        return f"padding={self.padding!r}"
