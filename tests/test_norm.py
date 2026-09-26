"""Tests for ``scratchgrad.nn.norm``.

Two kinds of check appear here and they are not interchangeable. Gradcheck
proves ``backward == d(forward)/dx`` -- it is completely blind to a *wrong
forward* that has been self-consistently differentiated. So every layer also
gets a forward-value assertion: a hand-computed example for LayerNorm and
RMSNorm, standardisation identities for BatchNorm and GroupNorm, and an
expectation test for the two dropouts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import scratchgrad as sg
from scratchgrad.nn.norm import (
    BatchNorm1d,
    BatchNorm2d,
    Dropout,
    Dropout2d,
    GroupNorm,
    LayerNorm,
    RMSNorm,
    dropout2d,
)
from scratchgrad.tensor import Tensor
from scratchgrad.testing import TOL_COMPOSED, TOL_OP, check_module, gradcheck

# Prime shapes throughout: with equal extents a transposed axis stays
# conformable and NumPy computes garbage without raising.
N, C, H, W = 7, 11, 5, 3
# The normalised axis gets its own, larger prime. A 3-element group is a bad
# gradcheck subject through no fault of the gradient: three standard normals
# land within 0.05 of each other often enough that some row has variance ~5e-4,
# ``istd`` ~ 44, and the O(h^2) truncation term of the central difference blows
# past 1e-7. The analytic VJP is unchanged; only the numeric reference degrades.
D = 13


def gradcheck_train(module, x_arr, tol=TOL_COMPOSED, name=""):
    """Gradcheck a module w.r.t. its input and every parameter, in ``train()``.

    ``check_module`` forces ``eval()`` on purpose (a resampled dropout mask
    between two finite-difference probes would make the check meaningless), but
    BatchNorm's train path *is* the interesting backward -- it is the one that
    couples every sample in the batch. So this reproduces what ``check_module``
    does with the mode left alone.

    Passing the module's own parameters with ``copy_inputs=False`` is load
    bearing: the graph records those exact leaf objects, so a copy would be
    perturbed while the forward pass read the originals and every parameter
    gradient would come back an exact zero.
    """
    module.double()
    module.train()
    # dtype= is explicit: Tensor() otherwise casts to the *default* dtype, which
    # is float32 outside gradcheck's own context manager.
    x = Tensor(np.asarray(x_arr, dtype=np.float64), requires_grad=True, dtype=np.float64)
    params = list(module.parameters())
    return gradcheck(
        lambda *ts: module(ts[0]),
        x,
        *params,
        tol=tol,
        copy_inputs=False,
        name=name or type(module).__name__,
    )


def check_norm(module, shape, seed: int = 0, name: str = ""):
    """``check_module`` with an independently drawn input.

    ``check_module`` seeds the input from ``seed`` and hands the same ``seed`` to
    ``gradcheck``, which draws the random cotangent from it. For a layer whose
    output shape equals its input shape -- which is every layer in this file --
    the two draws are then the *same array*, and ``<x, norm(x)>`` is a stationary
    point: the true gradient collapses to ~1e-5 against inputs of order 1, and
    ``rel_error`` normalises by that collapsed scale, so ordinary
    finite-difference noise reads as a 1e-4 relative error. Drawing the input
    from a different stream removes the coincidence.
    """
    data = np.random.default_rng(4242 + seed).standard_normal(shape)
    return check_module(module, shape, tol=TOL_COMPOSED, input_data=data, name=name)


# --------------------------------------------------------------------------- #
# LayerNorm
# --------------------------------------------------------------------------- #
def test_layer_norm_forward_hand_computed(f64):
    """A 4-element example worked out on paper, biased variance, eps inside."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    eps = 1e-5
    # mu = 10/4 = 2.5; deviations [-1.5, -0.5, 0.5, 1.5]
    # var_biased = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25   <- divide by N, not N-1
    denom = math.sqrt(1.25 + eps)  # <- eps INSIDE the sqrt
    expected = np.array([-1.5, -0.5, 0.5, 1.5]) / denom

    ln = LayerNorm(4, eps=eps)
    got = ln(Tensor(x)).data
    assert np.allclose(got, expected, rtol=0, atol=1e-14)

    # ...and with a non-trivial affine on top.
    ln.weight.data[...] = np.array([2.0, 3.0, 4.0, 5.0])
    ln.bias.data[...] = np.array([0.5, -0.5, 1.5, -1.5])
    expected_affine = np.array([2.0, 3.0, 4.0, 5.0]) * expected + np.array([0.5, -0.5, 1.5, -1.5])
    assert np.allclose(ln(Tensor(x)).data, expected_affine, rtol=0, atol=1e-14)


def test_layer_norm_is_not_the_annotated_transformer_variant(f64):
    """The unbiased-std / eps-outside form is a *different function*, not a rounding difference."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    eps = 1e-5
    ours = LayerNorm(4, eps=eps)(Tensor(x)).data
    # g * (x - mu) / (std_unbiased + eps) + b, the widely copied version
    annotated = (x - x.mean()) / (x.std(ddof=1) + eps)
    assert not np.allclose(ours, annotated, rtol=1e-3, atol=1e-3)
    # The gap is the sqrt(N/(N-1)) factor: 4/3 under the root, ~15.5%.
    assert np.allclose(ours / annotated, math.sqrt(4.0 / 3.0), rtol=1e-4)


def test_layer_norm_standardises_each_row(f64, rng):
    x = Tensor(rng.standard_normal((N, H, D)) * 3.0 + 2.0)
    out = LayerNorm(D, eps=1e-12)(x).data
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-12)
    assert np.allclose(out.var(axis=-1), 1.0, atol=1e-9)


def test_layer_norm_gradcheck():
    check_norm(LayerNorm(D), (N, H, D), name="LayerNorm")


def test_layer_norm_multi_axis_gradcheck():
    check_norm(LayerNorm((H, D)), (N, H, D), name="LayerNorm(H,D)")


def test_layer_norm_no_affine_gradcheck():
    check_norm(LayerNorm(D, elementwise_affine=False), (N, H, D), name="LayerNorm-plain")


def test_layer_norm_no_bias_gradcheck():
    check_norm(LayerNorm(D, bias=False), (N, H, D), name="LayerNorm-nobias")


def test_layer_norm_rejects_mismatched_trailing_dims(rng):
    with pytest.raises(ValueError, match="trailing dims"):
        LayerNorm(D)(Tensor(rng.standard_normal((N, C))))


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #
def test_rms_norm_forward_hand_computed(f64):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    eps = 1e-6
    # mean(x^2) = (1 + 4 + 9 + 16) / 4 = 7.5; no mean subtraction anywhere.
    expected = x / math.sqrt(7.5 + eps)
    assert np.allclose(RMSNorm(4, eps=eps)(Tensor(x)).data, expected, rtol=0, atol=1e-14)


def test_rms_norm_does_not_centre(f64, rng):
    """A constant offset survives RMSNorm; under LayerNorm it would vanish."""
    x = Tensor(rng.standard_normal((N, D)) + 10.0)
    out = RMSNorm(D, eps=1e-12)(x).data
    assert np.all(np.abs(out.mean(axis=-1)) > 1e-3)
    assert np.allclose((out * out).mean(axis=-1), 1.0, atol=1e-9)


def test_rms_norm_gradcheck():
    check_norm(RMSNorm(D), (N, H, D), name="RMSNorm")


def test_rms_norm_no_affine_gradcheck():
    check_norm(RMSNorm(D, elementwise_affine=False), (N, H, D), name="RMSNorm-plain")


# --------------------------------------------------------------------------- #
# GroupNorm
# --------------------------------------------------------------------------- #
def test_group_norm_standardises_each_group(f64, rng):
    groups, channels = 3, 12
    gn = GroupNorm(groups, channels, eps=1e-12, affine=False)
    out = gn(Tensor(rng.standard_normal((N, channels, H, W)) * 4.0 - 1.0)).data
    per = out.reshape(N, groups, channels // groups, H, W)
    assert np.allclose(per.mean(axis=(2, 3, 4)), 0.0, atol=1e-11)
    assert np.allclose(per.var(axis=(2, 3, 4)), 1.0, atol=1e-8)


def test_group_norm_one_group_equals_layer_norm(f64, rng):
    x = Tensor(rng.standard_normal((N, C, H, W)))
    gn = GroupNorm(1, C, eps=1e-5, affine=False)(x).data
    ln = LayerNorm((C, H, W), eps=1e-5, elementwise_affine=False)(x).data
    assert np.allclose(gn, ln, rtol=0, atol=1e-13)


def test_group_norm_per_channel_groups_are_instance_norm(f64, rng):
    """``num_groups == num_channels`` normalises each channel of each sample alone."""
    x = rng.standard_normal((N, C, H, W))
    out = GroupNorm(C, C, eps=1e-12, affine=False)(Tensor(x)).data
    manual = (x - x.mean(axis=(2, 3), keepdims=True)) / np.sqrt(
        x.var(axis=(2, 3), keepdims=True) + 1e-12
    )
    assert np.allclose(out, manual, rtol=0, atol=1e-9)


def test_group_norm_gradcheck():
    check_norm(GroupNorm(3, 12), (N, 12, H, W), name="GroupNorm")


def test_group_norm_no_affine_gradcheck():
    check_norm(GroupNorm(3, 12, affine=False), (N, 12, H, W), name="GroupNorm-plain")


def test_group_norm_rejects_indivisible_channels():
    with pytest.raises(ValueError, match="divisible"):
        GroupNorm(3, 11)


@pytest.mark.parametrize("groups, channels", [(0, 6), (-3, 6), (2, 0), (3, -6)])
def test_group_norm_rejects_non_positive_counts(groups, channels):
    """A zero group count must not reach the modulo and leak a ZeroDivisionError.

    ``GroupNorm(0, 6)`` used to escape as ``integer modulo by zero``, and
    ``GroupNorm(-3, 6)`` passed the divisibility test outright (``6 % -3 == 0``)
    and only failed later, inside ``forward``, as a reshape error naming neither
    argument.
    """
    with pytest.raises(ValueError, match=">= 1"):
        GroupNorm(groups, channels)


# --------------------------------------------------------------------------- #
# BatchNorm -- forward identities
# --------------------------------------------------------------------------- #
def test_batchnorm_train_output_is_standardised(f64, rng):
    """Train mode: per-channel mean 0 and **biased** variance 1."""
    x = Tensor(rng.standard_normal((64, C)) * 5.0 + 3.0)
    bn = BatchNorm1d(C, eps=1e-12).train()
    out = bn(x).data
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(out.var(axis=0), 1.0, atol=1e-9)  # np.var is ddof=0


def test_batchnorm2d_train_output_is_standardised(f64, rng):
    x = Tensor(rng.standard_normal((N, C, H, W)) * 2.0 - 4.0)
    out = BatchNorm2d(C, eps=1e-12).train()(x).data
    assert np.allclose(out.mean(axis=(0, 2, 3)), 0.0, atol=1e-12)
    assert np.allclose(out.var(axis=(0, 2, 3)), 1.0, atol=1e-9)


def test_batchnorm_running_var_is_unbiased(f64, rng):
    """The batch is normalised biased; the buffer stores the ``n/(n-1)`` estimate."""
    n = 8
    arr = rng.standard_normal((n, C)) * 2.0 + 1.0
    bn = BatchNorm1d(C, momentum=1.0).train()
    bn(Tensor(arr))

    assert np.allclose(bn.running_mean, arr.mean(axis=0), rtol=0, atol=1e-13)
    assert np.allclose(bn.running_var, arr.var(axis=0, ddof=1), rtol=0, atol=1e-13)
    # And it is *not* the biased one -- a 14% difference at n = 8, which is five
    # orders of magnitude above eps and so cannot be dismissed as noise.
    biased = arr.var(axis=0, ddof=0)
    assert not np.allclose(bn.running_var, biased, rtol=1e-3)
    assert np.allclose(bn.running_var / biased, n / (n - 1.0), rtol=1e-12)


def test_batchnorm_eval_uses_running_stats_exactly(f64, rng):
    arr = rng.standard_normal((N, C))
    bn = BatchNorm1d(C, eps=1e-5)
    bn.running_mean[...] = rng.standard_normal(C)
    bn.running_var[...] = rng.uniform(0.5, 2.0, size=C)
    bn.eval()
    expected = (arr - bn.running_mean) / np.sqrt(bn.running_var + 1e-5)
    assert np.allclose(bn(Tensor(arr)).data, expected, rtol=0, atol=1e-13)


def test_batchnorm_eval_does_not_touch_the_buffers(f64, rng):
    bn = BatchNorm1d(C).train()
    bn(Tensor(rng.standard_normal((N, C))))
    before = (bn.running_mean.copy(), bn.running_var.copy(), int(bn.num_batches_tracked))

    bn.eval()
    bn(Tensor(rng.standard_normal((N, C))))
    assert np.array_equal(bn.running_mean, before[0])
    assert np.array_equal(bn.running_var, before[1])
    assert int(bn.num_batches_tracked) == before[2]

    bn.train()
    bn(Tensor(rng.standard_normal((N, C))))
    assert not np.array_equal(bn.running_mean, before[0])
    assert int(bn.num_batches_tracked) == before[2] + 1


def test_batchnorm_train_and_eval_agree_under_a_matched_estimator(f64, rng):
    """Train/eval agreement, tested the only way that is actually valid.

    Asserting ``train(x) == eval(x)`` on one batch against a standard EMA is
    meaningless -- after one step with momentum 0.1 the running mean is 10% of
    the batch mean and the two outputs have no reason to match. Instead:
    ``momentum=1.0`` makes the running statistics *exactly* this batch's, and
    then overwriting ``running_var`` with the **biased** variance matches the
    estimator eval uses to the one train used. Under that matching the two paths
    are the same function, to the last bit.
    """
    arr = rng.standard_normal((16, C)) * 1.5 + 0.7
    bn = BatchNorm1d(C, momentum=1.0).train()
    bn.weight.data[...] = rng.standard_normal(C)
    bn.bias.data[...] = rng.standard_normal(C)
    train_out = bn(Tensor(arr)).data

    bn.running_var[...] = arr.var(axis=0, ddof=0)  # match the estimator, biased
    bn.eval()
    assert np.allclose(bn(Tensor(arr)).data, train_out, rtol=0, atol=1e-13)


@pytest.mark.parametrize("momentum", [0.1, 0.05])
def test_batchnorm_running_stats_converge(f64, momentum):
    """Over 300 batches from a fixed distribution the EMA lands on the truth."""
    gen = np.random.default_rng(11)
    mu_true = np.array([2.0, -1.0, 0.5])
    sd_true = np.array([3.0, 0.5, 1.5])
    bn = BatchNorm1d(3, momentum=momentum).train()
    for _ in range(300):
        batch = gen.standard_normal((256, 3)) * sd_true + mu_true
        bn(Tensor(batch))

    assert int(bn.num_batches_tracked) == 300
    assert np.allclose(bn.running_mean, mu_true, atol=0.3)
    assert np.allclose(bn.running_var, sd_true**2, rtol=0.15)


def test_batchnorm_cumulative_average_when_momentum_is_none(f64):
    """``momentum=None`` weights every batch equally -- an exact running average."""
    gen = np.random.default_rng(5)
    bn = BatchNorm1d(C, momentum=None).train()
    means, vars_ = [], []
    for _ in range(5):
        batch = gen.standard_normal((13, C)) * 2.0 + 1.0
        bn(Tensor(batch))
        means.append(batch.mean(axis=0))
        vars_.append(batch.var(axis=0, ddof=1))

    assert int(bn.num_batches_tracked) == 5
    assert np.allclose(bn.running_mean, np.mean(means, axis=0), rtol=0, atol=1e-13)
    assert np.allclose(bn.running_var, np.mean(vars_, axis=0), rtol=0, atol=1e-13)


def test_batchnorm_running_stats_are_buffers_not_parameters():
    bn = BatchNorm2d(C)
    params = dict(bn.named_parameters())
    buffers = dict(bn.named_buffers())
    assert set(params) == {"weight", "bias"}
    for name in ("running_mean", "running_var", "num_batches_tracked"):
        assert name in buffers, f"{name} must be a buffer so no optimizer can step it"
        assert name not in params
    assert buffers["num_batches_tracked"].dtype == np.int64
    assert "running_mean" in bn.state_dict()


def test_batchnorm_state_dict_roundtrip(f64, rng):
    src = BatchNorm2d(C).train()
    src(Tensor(rng.standard_normal((N, C, H, W))))
    dst = BatchNorm2d(C)
    dst.load_state_dict(src.state_dict(deep=True))
    assert np.array_equal(dst.running_mean, src.running_mean)
    assert np.array_equal(dst.running_var, src.running_var)
    assert int(dst.num_batches_tracked) == 1


def test_batchnorm_without_running_stats_uses_batch_stats_in_eval(f64, rng):
    arr = rng.standard_normal((N, C)) + 4.0
    bn = BatchNorm1d(C, eps=1e-12, track_running_stats=False).eval()
    out = bn(Tensor(arr)).data
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-12)
    assert dict(bn.named_buffers()) == {}


def test_batchnorm_rejects_a_single_sample_in_train(rng):
    with pytest.raises(ValueError, match="at least 2 elements"):
        BatchNorm1d(C).train()(Tensor(rng.standard_normal((1, C))))


def test_batchnorm_rejects_wrong_rank(rng):
    with pytest.raises(ValueError, match="rank-4"):
        BatchNorm2d(C)(Tensor(rng.standard_normal((N, C))))
    with pytest.raises(ValueError, match="rank-2/3"):
        BatchNorm1d(C)(Tensor(rng.standard_normal((N, C, H, W))))


# --------------------------------------------------------------------------- #
# BatchNorm -- gradients, in both modes
# --------------------------------------------------------------------------- #
def test_batchnorm1d_gradcheck_train(rng):
    gradcheck_train(BatchNorm1d(C), rng.standard_normal((N, C)), name="BatchNorm1d-train")


def test_batchnorm1d_3d_gradcheck_train(rng):
    gradcheck_train(BatchNorm1d(C), rng.standard_normal((N, C, H)), name="BatchNorm1d-3d-train")


def test_batchnorm2d_gradcheck_train(rng):
    gradcheck_train(BatchNorm2d(C), rng.standard_normal((N, C, H, W)), name="BatchNorm2d-train")


def test_batchnorm_no_affine_gradcheck_train(rng):
    gradcheck_train(
        BatchNorm1d(C, affine=False), rng.standard_normal((N, C)), name="BatchNorm-plain-train"
    )


def test_batchnorm1d_gradcheck_eval(rng):
    """Eval mode: the running stats are frozen constants, so the batch decouples."""
    bn = BatchNorm1d(C)
    bn.running_mean[...] = rng.standard_normal(C)
    bn.running_var[...] = rng.uniform(0.5, 2.0, size=C)
    check_norm(bn, (N, C), name="BatchNorm1d-eval")


def test_batchnorm2d_gradcheck_eval(rng):
    bn = BatchNorm2d(C)
    bn.running_mean[...] = rng.standard_normal(C)
    bn.running_var[...] = rng.uniform(0.5, 2.0, size=C)
    check_norm(bn, (N, C, H, W), name="BatchNorm2d-eval")


def test_batchnorm_train_gradient_couples_the_batch(f64, rng):
    """Perturbing one sample changes every other sample's output in train mode.

    This is what separates the train backward from the eval one; a layer that
    quietly used frozen statistics during training would fail here while still
    gradchecking against its own (wrong) forward.
    """
    arr = rng.standard_normal((N, C))
    bn = BatchNorm1d(C).train()
    base = bn(Tensor(arr)).data
    bumped = arr.copy()
    bumped[0] += 1.0
    after = bn(Tensor(bumped)).data
    assert np.all(np.abs(after[1:] - base[1:]) > 1e-6)


# --------------------------------------------------------------------------- #
# Dropout
# --------------------------------------------------------------------------- #
def test_dropout_eval_is_exactly_the_identity(f64, rng):
    x = Tensor(rng.standard_normal((N, C)))
    layer = Dropout(0.5, rng=np.random.default_rng(0)).eval()
    out = layer(x)
    assert out is x  # not merely equal: inverted dropout is a no-op at eval


def test_dropout_p_zero_is_the_identity(f64, rng):
    x = Tensor(rng.standard_normal((N, C)))
    assert Dropout(0.0, rng=np.random.default_rng(0)).train()(x) is x


def test_dropout_expectation_matches_the_input(f64, rng):
    """E[out] == in, which is exactly what the 1/(1-p) rescaling buys."""
    x = rng.standard_normal((N, C))
    layer = Dropout(0.3, rng=np.random.default_rng(2024)).train()
    trials = 8000
    total = np.zeros_like(x)
    for _ in range(trials):
        total += layer(Tensor(x)).data
    assert np.allclose(total / trials, x, rtol=0.05, atol=1e-12)


def test_dropout_actually_drops_and_rescales(f64, rng):
    x = np.abs(rng.standard_normal((N, C))) + 1.0
    out = Dropout(0.4, rng=np.random.default_rng(1)).train()(Tensor(x)).data
    kept = out != 0.0
    assert 0 < kept.sum() < x.size
    assert np.allclose(out[kept], x[kept] / (1.0 - 0.4), rtol=0, atol=1e-13)


def test_dropout_gradcheck_train(rng):
    """The mask must be identical on both sides of every probe pair.

    Re-seeding inside the checked function does exactly that: each of the
    ``2 * size + 1`` forward passes builds a fresh generator from the same seed
    and therefore the same mask. Handing the layer a live generator instead
    would resample per probe and the numeric gradient would be noise.
    """
    layer = Dropout(0.4).train()
    x = Tensor(rng.standard_normal((N, C)), requires_grad=True)
    gradcheck(
        lambda t: layer(t, rng=np.random.default_rng(17)), x, tol=TOL_OP, name="Dropout-train"
    )


def test_dropout_rejects_p_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        Dropout(1.0)


def test_dropout_without_rng_owns_a_generator_seeded_at_construction(f64):
    """No ``rng=`` still means an explicit Generator, drawn once per instance.

    The layer used to build a fresh ``default_rng()`` inside every ``forward``,
    which left an instance with no replayable history at all: two deepcopies of
    the same layer drew from two unrelated streams. Owning the generator makes
    the mask sequence a function of construction, and ``deepcopy`` carries the
    generator *state*, so the copy reproduces the original's next mask exactly.
    """
    layer = Dropout(0.5).train()
    assert isinstance(layer.rng, np.random.Generator)

    x = Tensor(np.ones((N, C)))
    twin = layer.deepcopy()
    # Poking the LEGACY global RNG is the point here -- a layer that reached for
    # `np.random` instead of its own Generator would diverge. Hence the noqa.
    np.random.seed(12345)  # noqa: NPY002
    a = layer(x).data
    np.random.seed(999)  # noqa: NPY002
    b = twin(x).data
    assert np.array_equal(a, b)
    # ...and the stream advances, so the instance is not stuck on one mask.
    assert not np.array_equal(a, layer(x).data)


def test_dropout_per_call_rng_still_overrides_the_owned_one(f64):
    """The construction-time generator must not shadow ``forward(x, rng=...)``."""
    x = Tensor(np.ones((N, C)))
    layer = Dropout(0.5, rng=np.random.default_rng(7)).train()
    first = layer(x, rng=np.random.default_rng(31)).data
    second = layer(x, rng=np.random.default_rng(31)).data
    assert np.array_equal(first, second)


# --------------------------------------------------------------------------- #
# Dropout2d
# --------------------------------------------------------------------------- #
def test_dropout2d_drops_whole_channels(f64, rng):
    x = np.abs(rng.standard_normal((N, C, H, W))) + 1.0
    out = Dropout2d(0.5, rng=np.random.default_rng(3)).train()(Tensor(x)).data
    dropped = np.zeros((N, C), dtype=bool)
    for n in range(N):
        for c in range(C):
            plane = out[n, c]
            all_zero = np.all(plane == 0.0)
            all_kept = np.allclose(plane, x[n, c] * 2.0, rtol=0, atol=1e-13)
            assert all_zero or all_kept, "a channel was partially dropped"
            dropped[n, c] = all_zero
    assert dropped.any() and not dropped.all()


def test_dropout2d_expectation_matches_the_input(f64, rng):
    x = rng.standard_normal((5, 3, 2, 2))
    layer = Dropout2d(0.25, rng=np.random.default_rng(99)).train()
    trials = 8000
    total = np.zeros_like(x)
    for _ in range(trials):
        total += layer(Tensor(x)).data
    assert np.allclose(total / trials, x, rtol=0.05, atol=1e-12)


def test_dropout2d_eval_is_exactly_the_identity(f64, rng):
    x = Tensor(rng.standard_normal((N, C, H, W)))
    assert Dropout2d(0.5, rng=np.random.default_rng(0)).eval()(x) is x


def test_dropout2d_gradcheck_train(rng):
    layer = Dropout2d(0.5).train()
    x = Tensor(rng.standard_normal((N, C, H, W)), requires_grad=True)
    gradcheck(
        lambda t: layer(t, rng=np.random.default_rng(23)), x, tol=TOL_OP, name="Dropout2d-train"
    )


def test_dropout2d_rejects_2d_input(rng):
    with pytest.raises(ValueError, match="at least 3 dims"):
        Dropout2d(0.5, rng=np.random.default_rng(0)).train()(Tensor(rng.standard_normal((N, C))))


@pytest.mark.parametrize("training, p", [(False, 1.5), (False, -0.1), (True, -0.1)])
def test_dropout2d_validates_p_before_the_no_op_return(rng, training, p):
    """The guards sat *after* ``if not training or p == 0: return x``.

    An out-of-range ``p`` was therefore accepted without complaint on every
    no-op path -- the layer would run clean through an eval-mode test and only
    raise once someone called ``train()``.
    """
    x = Tensor(rng.standard_normal((N, C, H, W)))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        dropout2d(x, p=p, training=training, rng=np.random.default_rng(0))


@pytest.mark.parametrize("training, p", [(False, 0.5), (True, 0.0)])
def test_dropout2d_validates_rank_before_the_no_op_return(rng, training, p):
    """Same ordering bug for the rank check: a rank-2 input is wrong at p=0 too."""
    x = Tensor(rng.standard_normal((N, C)))
    with pytest.raises(ValueError, match="at least 3 dims"):
        dropout2d(x, p=p, training=training, rng=np.random.default_rng(0))


def test_dropout2d_valid_no_op_paths_still_return_the_input(rng):
    """Validating first must not cost the identity on the legitimate no-op paths."""
    x = Tensor(rng.standard_normal((N, C, H, W)))
    assert dropout2d(x, p=0.5, training=False) is x
    assert dropout2d(x, p=0.0, training=True) is x


# --------------------------------------------------------------------------- #
# repr, for the summary table
# --------------------------------------------------------------------------- #
def test_reprs_name_their_configuration():
    assert "eps=1e-05" in repr(LayerNorm(4))
    assert "track_running_stats=True" in repr(BatchNorm2d(4))
    assert "p=0.25" in repr(Dropout(0.25))
    assert repr(GroupNorm(2, 4)).startswith("GroupNorm(2, 4")


def test_sg_top_level_still_imports():
    """norm.py must not be reachable through an __init__ edit -- import directly."""
    assert hasattr(sg, "Tensor")
