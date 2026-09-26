"""Tests for :mod:`scratchgrad.nn.conv`.

The point of every shape here is that it is *awkward*. A conv tested only at
``stride=1, padding=0, dilation=1`` on a square image with a square kernel is a
conv whose transposed-index bugs are all still in it: with ``H == W`` and
``kH == kW`` a swapped axis pair stays conformable and NumPy computes something
plausible without raising. So the shapes below are prime and unequal, the
kernels are non-square, and stride/padding/dilation are all non-default at once.

Three independent kinds of evidence are used, because each is blind to
something the others catch:

* ``gradcheck`` -- backward agrees with central differences of the forward.
  Blind to a *wrong forward* with a self-consistently derived backward.
* ``adjoint_test`` -- ``<v, Au> == <A^T v, u>`` exactly, no step size. Sharper
  than finite differences and immune to that self-consistency trap, but it only
  applies to the linear part (so: no bias).
* hand-computed forward values -- the only thing that pins down what the
  forward actually *is*.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad._config import default_dtype
from scratchgrad.nn.conv import (
    AdaptiveAvgPool2d,
    AvgPool2d,
    Conv1d,
    Conv2d,
    MaxPool2d,
    ZeroPad2d,
    adaptive_avg_pool2d,
    avg_pool2d,
    col2im,
    conv1d,
    conv2d,
    conv_output_size,
    im2col,
    max_pool2d,
)
from scratchgrad.tensor import Tensor
from scratchgrad.testing.gradcheck import (
    TOL_OP,
    adjoint_test,
    check_module,
    gradcheck,
    rel_error,
)


@pytest.fixture(autouse=True)
def _force_float64():
    """Run the whole module in float64.

    ``Tensor._make`` casts every float output to the *global* default dtype, so
    without this every forward-value assertion below would be comparing float32
    results against float64 references and would fail at ~1e-7 -- and every
    gradcheck would be running with no power at all.
    """
    with default_dtype(np.float64):
        yield


def _t(rng, shape) -> Tensor:
    """A float64 leaf that requires grad, built from an explicit generator."""
    return Tensor(rng.standard_normal(shape), requires_grad=True, dtype=np.float64)


# --------------------------------------------------------------------------- #
# forward values, computed by hand
# --------------------------------------------------------------------------- #
def test_conv2d_forward_hand_computed():
    """A 3x3 image against a 2x2 kernel, worked out on paper.

    x = [[1,2,3],       k = [[1,2],
         [4,5,6],            [3,4]]
         [7,8,9]]

    out[0,0] = 1*1 + 2*2 + 4*3 + 5*4 = 37
    out[0,1] = 2*1 + 3*2 + 5*3 + 6*4 = 47
    out[1,0] = 4*1 + 5*2 + 7*3 + 8*4 = 67
    out[1,1] = 5*1 + 6*2 + 8*3 + 9*4 = 77

    Note the kernel is **not** flipped: this is cross-correlation, which is what
    every deep-learning framework calls a convolution. A true convolution would
    give [[23, 33], [53, 63]] here, so the test distinguishes them.
    """
    x = Tensor(np.arange(1.0, 10.0).reshape(1, 1, 3, 3))
    w = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2))
    out = conv2d(x, w)
    assert out.data.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(out.data, [[[[37.0, 47.0], [67.0, 77.0]]]], rtol=0, atol=1e-12)

    biased = conv2d(x, w, Tensor(np.array([10.0])))
    np.testing.assert_allclose(biased.data, [[[[47.0, 57.0], [77.0, 87.0]]]], rtol=0, atol=1e-12)


def test_conv2d_forward_stride_dilation_hand_computed():
    """Same 3x3 image, dilation 2: the 2x2 kernel spans the full 3x3 field.

    Taps land on the corners (0,0), (0,2), (2,0), (2,2) = 1, 3, 7, 9, so
    out = 1*1 + 3*2 + 7*3 + 9*4 = 1 + 6 + 21 + 36 = 64.
    """
    x = Tensor(np.arange(1.0, 10.0).reshape(1, 1, 3, 3))
    w = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2))
    out = conv2d(x, w, dilation=2)
    assert out.data.shape == (1, 1, 1, 1)
    assert out.data.reshape(()) == pytest.approx(64.0, abs=1e-12)


def test_conv2d_nonsquare_kernel_is_not_transposed():
    """A (kH, kW) = (1, 2) kernel differencing along W must not act along H."""
    x = Tensor(np.arange(1.0, 10.0).reshape(1, 1, 3, 3))
    w = Tensor(np.array([[-1.0, 1.0]]).reshape(1, 1, 1, 2))
    out = conv2d(x, w)
    # Horizontal first difference of each row: every entry is exactly 1.
    assert out.data.shape == (1, 1, 3, 2)
    np.testing.assert_allclose(out.data, np.ones((1, 1, 3, 2)), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# output-size arithmetic
# --------------------------------------------------------------------------- #
def test_conv_output_size_matches_the_formula():
    # floor((H + 2p - d*(k-1) - 1)/s) + 1 with H=7, p=1, k=3, s=2, d=2:
    # dilated span 5, (7 + 2 - 5)//2 + 1 = 3.
    assert conv_output_size(7, 1, 1, 3, 2, 2) == 3
    assert conv_output_size(5, 1, 1, 3, 2, 2) == 2
    assert conv_output_size(7, 0, 0, 3, 1, 1) == 5


def test_conv2d_rejects_non_positive_output():
    rng = np.random.default_rng(0)
    x = _t(rng, (1, 1, 7, 5))
    w = _t(rng, (1, 1, 9, 9))
    with pytest.raises(ValueError, match="non-positive output"):
        conv2d(x, w)


def test_conv2d_same_padding_preserves_spatial_size():
    rng = np.random.default_rng(1)
    x = _t(rng, (2, 2, 7, 5))
    for kernel, dilation in (((3, 3), 1), ((5, 3), 1), ((3, 2), 2)):
        w = _t(rng, (3, 2) + kernel)
        out = conv2d(x, w, padding="same", dilation=dilation)
        assert out.data.shape == (2, 3, 7, 5), (kernel, dilation)


def test_same_padding_is_refused_with_stride():
    """PyTorch refuses strided ``padding='same'``; TensorFlow returns ceil(in/s).

    Before the guard the strided call silently returned the TensorFlow answer --
    a (2, 3, 4, 3) output from an argument documented as preserving the input
    size -- so a port of a Torch reference changed shape without changing an
    argument, and the mismatch only surfaced later at the first ``Linear``.
    """
    rng = np.random.default_rng(101)
    x = _t(rng, (2, 2, 7, 5))
    w = _t(rng, (3, 2, 3, 3))
    with pytest.raises(ValueError, match="not supported with stride"):
        conv2d(x, w, padding="same", stride=2)
    with pytest.raises(ValueError, match="not supported with stride"):
        conv2d(x, w, padding="same", stride=(1, 2))
    # Unit stride is the supported case and must keep working.
    assert conv2d(x, w, padding="same", stride=1).data.shape == (2, 3, 7, 5)


def test_conv1d_same_padding_is_refused_with_stride():
    rng = np.random.default_rng(102)
    x = _t(rng, (2, 2, 11))
    w = _t(rng, (3, 2, 3))
    with pytest.raises(ValueError, match="not supported with stride"):
        conv1d(x, w, padding="same", stride=2)
    assert conv1d(x, w, padding="same").data.shape == (2, 3, 11)


def test_conv2d_group_validation():
    rng = np.random.default_rng(2)
    x = _t(rng, (1, 4, 5, 5))
    with pytest.raises(ValueError, match="input channels per group"):
        conv2d(x, _t(rng, (4, 4, 3, 3)), groups=2)
    with pytest.raises(ValueError, match="must divide"):
        conv2d(x, _t(rng, (5, 4, 3, 3)), groups=3)


# --------------------------------------------------------------------------- #
# im2col / col2im
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kernel, stride, padding, dilation",
    [
        ((3, 3), 1, 0, 1),
        ((3, 2), (2, 1), 1, (2, 1)),
        ((2, 3), 1, (1, 2), 1),
        ((3, 3), 1, "same", 1),
    ],
)
def test_col2im_accumulates_the_overlap_count(kernel, stride, padding, dilation):
    """``col2im(im2col(x)) == x * overlap_count``.

    The overlap map is itself produced by ``col2im`` on an all-ones column
    matrix, so this pins the two index computations against each other: if
    ``col2im`` scattered to different positions than ``im2col`` gathered from,
    the identity fails. And if ``col2im`` *assigned* instead of accumulating,
    every count above 1 would come back as 1 and the product would not match.
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 3, 7, 5))
    cols = im2col(x, kernel, stride, padding, dilation)
    count = col2im(np.ones_like(cols), x.shape, kernel, stride, padding, dilation)
    recon = col2im(cols, x.shape, kernel, stride, padding, dilation)

    assert count.max() > 1.0, "shape choice does not actually overlap; test is vacuous"
    np.testing.assert_allclose(recon, x * count, rtol=0, atol=1e-12)


def test_im2col_column_layout():
    """Row order is (channel, kernel row, kernel column); columns are (oh, ow)."""
    x = np.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    cols = im2col(x, 2)
    assert cols.shape == (1, 4, 4)
    # First output window is the top-left 2x2 block, read in row-major order.
    np.testing.assert_allclose(cols[0, :, 0], [1.0, 2.0, 4.0, 5.0])
    # Last output window is the bottom-right 2x2 block.
    np.testing.assert_allclose(cols[0, :, -1], [5.0, 6.0, 8.0, 9.0])


def test_im2col_col2im_are_adjoints():
    """``<c, im2col(x)> == <col2im(c), x>`` -- exact, no step size."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, 3, 7, 5))
    cols = im2col(x, (3, 2), (2, 1), 1, (2, 1))
    c = rng.standard_normal(cols.shape)
    lhs = float(np.sum(c * cols))
    rhs = float(np.sum(col2im(c, x.shape, (3, 2), (2, 1), 1, (2, 1)) * x))
    assert abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12) < 1e-14


# --------------------------------------------------------------------------- #
# Conv2d gradients
# --------------------------------------------------------------------------- #
def test_conv2d_gradcheck_stride_padding_dilation():
    """Non-default stride AND padding AND dilation, at prime unequal shapes."""
    rng = np.random.default_rng(0)
    x = _t(rng, (3, 2, 7, 5))
    w = _t(rng, (3, 2, 3, 3))
    b = _t(rng, (3,))

    def fn(x, w, b):
        return conv2d(x, w, b, stride=2, padding=1, dilation=2)

    assert fn(x, w, b).data.shape == (3, 3, 3, 2)
    gradcheck(fn, x, w, b, tol=TOL_OP, name="conv2d(s=2,p=1,d=2)")


def test_conv2d_gradcheck_nonsquare_everything():
    """Kernel, stride, padding and dilation all differ between H and W."""
    rng = np.random.default_rng(3)
    x = _t(rng, (2, 3, 7, 5))
    w = _t(rng, (5, 3, 3, 2))
    b = _t(rng, (5,))

    def fn(x, w, b):
        return conv2d(x, w, b, stride=(2, 1), padding=(1, 2), dilation=(2, 1))

    assert fn(x, w, b).data.shape == (2, 5, 3, 8)
    gradcheck(fn, x, w, b, tol=TOL_OP, name="conv2d(asymmetric)")


def test_conv2d_gradcheck_same_padding():
    rng = np.random.default_rng(4)
    x = _t(rng, (2, 2, 7, 5))
    w = _t(rng, (3, 2, 3, 2))
    gradcheck(lambda x, w: conv2d(x, w, padding="same"), x, w, tol=TOL_OP, name="conv2d(same)")


def test_conv2d_gradcheck_depthwise_and_grouped():
    rng = np.random.default_rng(5)
    x = _t(rng, (3, 4, 7, 5))

    w_dw = _t(rng, (4, 1, 3, 3))  # depthwise: groups == in_channels
    b_dw = _t(rng, (4,))
    gradcheck(
        lambda x, w, b: conv2d(x, w, b, stride=2, padding=1, groups=4),
        x,
        w_dw,
        b_dw,
        tol=TOL_OP,
        name="conv2d(depthwise)",
    )

    w_g = _t(rng, (6, 2, 3, 2))  # 2 groups, 3 output channels each
    b_g = _t(rng, (6,))
    gradcheck(
        lambda x, w, b: conv2d(x, w, b, padding=(1, 0), dilation=(2, 1), groups=2),
        x,
        w_g,
        b_g,
        tol=TOL_OP,
        name="conv2d(groups=2)",
    )


def test_depthwise_matches_per_channel_dense_convs():
    """groups=C must equal C independent single-channel convolutions."""
    rng = np.random.default_rng(6)
    x = _t(rng, (2, 3, 7, 5))
    w = _t(rng, (3, 1, 3, 2))
    grouped = conv2d(x, w, stride=(2, 1), padding=1, groups=3)
    per_channel = [
        conv2d(
            Tensor(x.data[:, i : i + 1], dtype=np.float64),
            Tensor(w.data[i : i + 1], dtype=np.float64),
            stride=(2, 1),
            padding=1,
        ).data
        for i in range(3)
    ]
    np.testing.assert_allclose(
        grouped.data, np.concatenate(per_channel, axis=1), rtol=0, atol=1e-12
    )


def test_grouped_conv_with_dilation_matches_per_group_dense_convs():
    """``groups > 1`` and ``dilation > 1`` *together*, which is the untested pair.

    Each is pinned separately above, and a gradcheck cannot see a forward that is
    wrong self-consistently. The reference here is two independent dense dilated
    convolutions over the channel halves, which shares no grouping index code
    with the path under test.
    """
    rng = np.random.default_rng(107)
    x = _t(rng, (2, 4, 7, 5))
    w = _t(rng, (6, 2, 3, 2))  # 2 groups of 3 output channels each
    b = _t(rng, (6,))
    grouped = conv2d(x, w, b, stride=(2, 1), padding=(1, 0), dilation=(2, 1), groups=2)
    per_group = [
        conv2d(
            Tensor(x.data[:, 2 * i : 2 * i + 2], dtype=np.float64),
            Tensor(w.data[3 * i : 3 * i + 3], dtype=np.float64),
            Tensor(b.data[3 * i : 3 * i + 3], dtype=np.float64),
            stride=(2, 1),
            padding=(1, 0),
            dilation=(2, 1),
        ).data
        for i in range(2)
    ]
    np.testing.assert_allclose(grouped.data, np.concatenate(per_group, axis=1), rtol=0, atol=1e-12)


def test_conv2d_adjoint_exact():
    """Conv2d is linear in x, so the adjoint identity holds to machine precision.

    bias=False: an affine map is not linear and the identity would pick up the
    bias term as a constant offset.
    """
    rng = np.random.default_rng(0)
    conv = Conv2d(
        2,
        3,
        kernel_size=(3, 2),
        stride=(2, 1),
        padding=(1, 2),
        dilation=(2, 1),
        bias=False,
        rng=rng,
    ).double()

    def vjp(u: Tensor, v: np.ndarray) -> np.ndarray:
        u.grad = None
        conv(u).backward(v)
        return u.grad

    err = adjoint_test(conv, vjp, (3, 2, 7, 5), (3, 3, 3, 8), tol=1e-12)
    assert err < 1e-12


def test_conv2d_module_gradcheck():
    rng = np.random.default_rng(9)
    conv = Conv2d(2, 3, kernel_size=3, stride=2, padding=1, dilation=2, rng=rng)
    check_module(conv, (3, 2, 7, 5), tol=TOL_OP, name="Conv2d")


def test_input_requires_grad_false_skips_dx_only():
    """Weight and bias gradients are unaffected; dX is simply never formed."""
    rng = np.random.default_rng(12)
    x = _t(rng, (2, 2, 7, 5))
    w = _t(rng, (3, 2, 3, 3))
    b = _t(rng, (3,))
    cot = rng.standard_normal((2, 3, 3, 2))

    full = conv2d(x, w, b, stride=2, padding=1, dilation=2)
    full.backward(cot)
    gx, gw, gb = x.grad.copy(), w.grad.copy(), b.grad.copy()

    x.grad = w.grad = b.grad = None
    skipped = conv2d(x, w, b, stride=2, padding=1, dilation=2, input_requires_grad=False)
    skipped.backward(cot)

    assert x.grad is None
    assert rel_error(w.grad, gw) < 1e-15
    assert rel_error(b.grad, gb) < 1e-15
    assert np.abs(gx).max() > 0.0  # the skipped gradient was not trivially zero


# --------------------------------------------------------------------------- #
# Conv1d
# --------------------------------------------------------------------------- #
def test_conv1d_gradcheck_and_shape():
    rng = np.random.default_rng(13)
    x = _t(rng, (3, 2, 11))
    w = _t(rng, (5, 2, 3))
    b = _t(rng, (5,))

    def fn(x, w, b):
        return conv1d(x, w, b, stride=2, padding=1, dilation=2)

    # floor((11 + 2 - 2*(3-1) - 1)/2) + 1 = 5
    assert fn(x, w, b).data.shape == (3, 5, 5)
    gradcheck(fn, x, w, b, tol=TOL_OP, name="conv1d")


def test_conv1d_matches_conv2d_with_unit_height():
    rng = np.random.default_rng(14)
    x = _t(rng, (2, 2, 11))
    w = _t(rng, (3, 2, 3))
    got = conv1d(x, w, stride=2, padding=1)
    want = conv2d(
        Tensor(x.data[:, :, None, :], dtype=np.float64),
        Tensor(w.data[:, :, None, :], dtype=np.float64),
        stride=(1, 2),
        padding=(0, 0, 1, 1),
    )
    np.testing.assert_allclose(got.data, want.data[:, :, 0, :], rtol=0, atol=1e-12)


def test_conv1d_module_gradcheck():
    rng = np.random.default_rng(15)
    conv = Conv1d(2, 3, kernel_size=3, stride=2, padding=1, dilation=2, rng=rng)
    check_module(conv, (3, 2, 11), tol=TOL_OP, name="Conv1d")


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #
def test_max_pool2d_splits_an_exact_tie():
    """Two equal maxima in one window get half the cotangent each.

    Without the divide-by-tie-count the gradient array would be [[1, 1], [0, 0]]
    -- the upstream gradient duplicated, not distributed. This is invisible on
    random floats (ties have measure zero) and ubiquitous on real activations.
    """
    x = Tensor([[[[1.0, 1.0], [0.0, 0.0]]]], requires_grad=True)
    out = max_pool2d(x, 2)
    assert out.data.reshape(()) == pytest.approx(1.0)
    out.backward(np.ones((1, 1, 1, 1)))

    np.testing.assert_allclose(x.grad, [[[[0.5, 0.5], [0.0, 0.0]]]], rtol=0, atol=1e-15)
    assert x.grad.sum() == pytest.approx(1.0), "gradient mass must be conserved"


def test_max_pool2d_splits_a_four_way_tie():
    x = Tensor(np.full((1, 1, 2, 2), 3.0), requires_grad=True)
    max_pool2d(x, 2).backward(np.array([[[[4.0]]]]))
    np.testing.assert_allclose(x.grad, np.full((1, 1, 2, 2), 1.0), rtol=0, atol=1e-15)


def test_max_pool2d_forward_and_gradcheck():
    rng = np.random.default_rng(21)
    x = _t(rng, (2, 3, 7, 5))
    out = max_pool2d(x, (3, 2), stride=(2, 1))
    assert out.data.shape == (2, 3, 3, 4)
    # Spot-check one window against a direct numpy max.
    assert out.data[1, 2, 1, 3] == pytest.approx(x.data[1, 2, 2:5, 3:5].max())
    gradcheck(lambda x: max_pool2d(x, (3, 2), stride=(2, 1)), x, tol=TOL_OP, name="max_pool2d")


def test_max_pool2d_padding_uses_negative_infinity():
    """A padded border must never win against a negative activation."""
    x = Tensor(np.full((1, 1, 2, 2), -5.0), requires_grad=True)
    out = max_pool2d(x, 2, stride=2, padding=1)
    assert out.data.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(out.data, np.full((1, 1, 2, 2), -5.0), rtol=0, atol=1e-15)


def test_max_pool2d_padding_bound_uses_the_dilated_extent():
    """The bound is ``d*(k-1)+1``, the window that actually exists."""
    rng = np.random.default_rng(121)
    x = _t(rng, (1, 2, 5, 5))
    with pytest.raises(ValueError, match="half the dilated kernel"):
        max_pool2d(x, 2, stride=2, padding=2)
    # A kernel 2 dilated by 2 spans 3 pixels, so a pad of 1 is legal here and
    # every window still covers a real pixel; the guard must not reject it.
    out = max_pool2d(x, 2, stride=1, padding=1, dilation=2)
    assert out.data.shape == (1, 2, 5, 5)
    assert np.isfinite(out.data).all()
    gradcheck(
        lambda x: max_pool2d(x, 2, stride=1, padding=1, dilation=2),
        x,
        tol=TOL_OP,
        name="max_pool2d(padded, dilated)",
    )


def test_max_pool2d_rejects_an_all_padding_window():
    """Dilated taps can straddle the image and land in the pad on both sides.

    Kernel 2 with dilation 2 over a one-pixel-tall image padded by 1 puts the
    taps at padded rows 0 and 2 with the only real row at 1. The old guard
    measured the padding against the *raw* kernel (``1*2 > 2`` is false) and let
    it through: the forward returned ``-inf`` and the backward split the
    cotangent between two padded cells that are then cropped away, so the
    gradient reached no input at all.
    """
    x = Tensor(np.arange(3.0).reshape(1, 1, 1, 3), requires_grad=True)
    with pytest.raises(ValueError, match="entirely inside the padding"):
        max_pool2d(x, 2, stride=1, padding=(1, 1, 0, 0), dilation=(2, 1))


def test_avg_pool2d_rejects_padding_past_half_the_kernel():
    """A window inside the pad has no real pixel to average.

    With ``count_include_pad=False`` the divisor for such a window is the number
    of real pixels it covers -- zero -- so the forward is 0/0 and the backward
    divides by it a second time.
    """
    rng = np.random.default_rng(122)
    x = _t(rng, (1, 2, 5, 5))
    with pytest.raises(ValueError, match="no real pixels to average"):
        avg_pool2d(x, 2, stride=2, padding=2)
    with pytest.raises(ValueError, match="no real pixels to average"):
        avg_pool2d(x, (3, 2), stride=2, padding=(1, 1, 2, 2))
    # Exactly half the kernel is the documented limit and must keep working.
    assert avg_pool2d(x, 2, stride=2, padding=1).data.shape == (1, 2, 3, 3)


def test_avg_pool2d_forward_and_gradcheck():
    rng = np.random.default_rng(22)
    x = _t(rng, (2, 3, 7, 5))
    out = avg_pool2d(x, (3, 2), stride=(2, 1))
    assert out.data.shape == (2, 3, 3, 4)
    assert out.data[0, 1, 2, 0] == pytest.approx(x.data[0, 1, 4:7, 0:2].mean())
    gradcheck(lambda x: avg_pool2d(x, (3, 2), stride=(2, 1)), x, tol=TOL_OP, name="avg_pool2d")


def test_avg_pool2d_count_include_pad():
    x = Tensor(np.ones((1, 1, 2, 2)), requires_grad=True)
    included = avg_pool2d(x, 2, stride=2, padding=1, count_include_pad=True)
    excluded = avg_pool2d(x, 2, stride=2, padding=1, count_include_pad=False)
    # Each 2x2 corner window sees exactly one real pixel out of four.
    np.testing.assert_allclose(included.data, np.full((1, 1, 2, 2), 0.25))
    np.testing.assert_allclose(excluded.data, np.ones((1, 1, 2, 2)))

    y = Tensor(np.random.default_rng(23).standard_normal((2, 2, 5, 5)), requires_grad=True)
    gradcheck(
        lambda y: avg_pool2d(y, 3, stride=2, padding=1, count_include_pad=False),
        y,
        tol=TOL_OP,
        name="avg_pool2d(count_include_pad=False)",
    )


def test_adaptive_avg_pool2d_global_and_binned():
    rng = np.random.default_rng(24)
    x = _t(rng, (2, 3, 7, 5))
    glob = adaptive_avg_pool2d(x, 1)
    assert glob.data.shape == (2, 3, 1, 1)
    np.testing.assert_allclose(glob.data[:, :, 0, 0], x.data.mean(axis=(2, 3)), rtol=0, atol=1e-14)
    gradcheck(lambda x: adaptive_avg_pool2d(x, 1), x, tol=TOL_OP, name="global_avg_pool")
    gradcheck(lambda x: adaptive_avg_pool2d(x, (2, 3)), x, tol=TOL_OP, name="adaptive(2,3)")


def test_pool_modules_match_their_functions():
    rng = np.random.default_rng(25)
    x = _t(rng, (2, 3, 7, 5))
    np.testing.assert_allclose(
        MaxPool2d((3, 2), stride=(2, 1))(x).data, max_pool2d(x, (3, 2), stride=(2, 1)).data
    )
    np.testing.assert_allclose(
        AvgPool2d((3, 2), stride=(2, 1))(x).data, avg_pool2d(x, (3, 2), stride=(2, 1)).data
    )
    np.testing.assert_allclose(AdaptiveAvgPool2d(1)(x).data, adaptive_avg_pool2d(x, 1).data)


# --------------------------------------------------------------------------- #
# padding module
# --------------------------------------------------------------------------- #
def test_zero_pad2d_order_is_left_right_top_bottom():
    rng = np.random.default_rng(26)
    x = _t(rng, (2, 3, 7, 5))
    out = ZeroPad2d((1, 2, 3, 4))(x)
    assert out.data.shape == (2, 3, 7 + 3 + 4, 5 + 1 + 2)
    np.testing.assert_allclose(out.data[:, :, 3:10, 1:6], x.data, rtol=0, atol=1e-15)
    assert out.data[:, :, :3, :].sum() == 0.0
    assert out.data[:, :, :, :1].sum() == 0.0
    gradcheck(lambda x: ZeroPad2d((1, 2, 3, 4))(x), x, tol=TOL_OP, name="ZeroPad2d")


def test_conv_stack_composes():
    """A small stack end to end, checked through its scalar loss."""
    rng = np.random.default_rng(27)
    conv = Conv2d(2, 4, 3, padding=1, input_requires_grad=False, rng=rng).double()
    pool = MaxPool2d(2)
    head = AdaptiveAvgPool2d(1)
    x = _t(rng, (3, 2, 7, 5))
    out = head(pool(conv(x).relu()))
    assert out.data.shape == (3, 4, 1, 1)
    loss = (out * out).sum()
    loss.backward()
    assert conv.weight.grad is not None
    assert x.grad is None  # input_requires_grad=False was honoured through the stack
