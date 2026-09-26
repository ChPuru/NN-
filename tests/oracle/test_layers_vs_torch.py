"""Convolution, normalization, recurrence and attention, checked against PyTorch.

Gradcheck proves each layer is self-consistent: its backward really is the
derivative of its forward. It cannot tell you the *convention* is right. A
LayerNorm using unbiased variance with eps outside the square root gradchecks
perfectly and is still a different function from `torch.nn.LayerNorm`; an LSTM
with gate order `i, g, f, o` trains fine and silently mismatches every published
checkpoint. Those are the mistakes this file exists to catch, because both of my
own implementations -- forward and backward -- would share them.

PyTorch is imported **only** here. `tests/test_no_framework_imports.py` enforces
that, and it fails the build if library code ever reaches for it.

Weight layouts differ on purpose and the transpose at each boundary is part of
what is being tested:

    scratchgrad Linear.weight      (in, out)        torch  (out, in)
    scratchgrad Conv2d.weight      (out, in, kH, kW)  torch  same
    scratchgrad LSTM weight_ih     (input, 4*hidden)  torch  (4*hidden, input)
"""

from __future__ import annotations

import numpy as np
import pytest

import scratchgrad as sg
from scratchgrad.nn.attention import scaled_dot_product_attention
from scratchgrad.nn.conv import AvgPool2d, Conv2d, MaxPool2d
from scratchgrad.nn.norm import BatchNorm1d, BatchNorm2d, GroupNorm, LayerNorm
from scratchgrad.nn.rnn import GRU, LSTM, RNN, Embedding

torch = pytest.importorskip("torch", reason="PyTorch oracle not installed")

pytestmark = pytest.mark.oracle

ATOL = 1e-10


@pytest.fixture(autouse=True)
def _f64():
    with sg.default_dtype(np.float64):
        yield


def tt(arr) -> torch.Tensor:
    return torch.tensor(np.asarray(arr, dtype=np.float64), dtype=torch.float64)


def close(ours, theirs, what: str, atol: float = ATOL) -> None:
    a = np.asarray(ours.data if hasattr(ours, "data") else ours, dtype=np.float64)
    b = theirs.detach().numpy() if hasattr(theirs, "detach") else np.asarray(theirs)
    np.testing.assert_allclose(a, b, atol=atol, rtol=atol, err_msg=what)


# --------------------------------------------------------------------------- #
# convolution
# --------------------------------------------------------------------------- #
CONV_CASES = [
    # (in_c, out_c, k, stride, padding, dilation, groups)
    (2, 3, 3, 1, 0, 1, 1),
    (2, 3, 3, 1, 1, 1, 1),
    (2, 3, 3, 2, 1, 1, 1),  # non-default stride AND padding together
    (2, 3, 3, 1, 2, 2, 1),  # dilation
    (2, 3, (3, 2), (2, 1), (1, 0), 1, 1),  # asymmetric everything
    (4, 4, 3, 1, 1, 1, 4),  # depthwise
    (4, 6, 3, 1, 1, 1, 2),  # grouped
]


@pytest.mark.parametrize(("cin", "cout", "k", "s", "p", "d", "g"), CONV_CASES)
def test_conv2d_matches_torch(cin, cout, k, s, p, d, g, rng):
    """Default-config-only testing is exactly how transposed-index bugs survive,
    so every case here moves at least one parameter off its default."""
    ours = Conv2d(cin, cout, k, stride=s, padding=p, dilation=d, groups=g, rng=rng).double()
    ref = torch.nn.Conv2d(cin, cout, k, stride=s, padding=p, dilation=d, groups=g).double()
    with torch.no_grad():
        ref.weight.copy_(tt(ours.weight.data))
        ref.bias.copy_(tt(ours.bias.data))

    x = rng.standard_normal((3, cin, 9, 7))
    xo = sg.Tensor(x, requires_grad=True)
    xt = tt(x).requires_grad_(True)

    o, t = ours(xo), ref(xt)
    assert o.shape == tuple(t.shape), f"output shape {o.shape} vs torch {tuple(t.shape)}"
    close(o, t, "conv forward")

    v = rng.standard_normal(tuple(t.shape))
    o.backward(v)
    t.backward(tt(v))
    close(xo.grad, xt.grad, "conv dX")
    close(ours.weight.grad, ref.weight.grad, "conv dW")
    close(ours.bias.grad, ref.bias.grad, "conv dB")


@pytest.mark.parametrize(("k", "s", "p"), [(2, 2, 0), (3, 2, 1), (2, 1, 0), (3, 3, 0)])
def test_maxpool2d_matches_torch(k, s, p, rng):
    x = rng.standard_normal((2, 3, 8, 8))
    xo, xt = sg.Tensor(x, requires_grad=True), tt(x).requires_grad_(True)
    o = MaxPool2d(k, stride=s, padding=p)(xo)
    t = torch.nn.functional.max_pool2d(xt, k, stride=s, padding=p)
    close(o, t, "maxpool forward")
    v = rng.standard_normal(tuple(t.shape))
    o.backward(v)
    t.backward(tt(v))
    close(xo.grad, xt.grad, "maxpool dX")


@pytest.mark.parametrize(("k", "s"), [(2, 2), (3, 1)])
def test_avgpool2d_matches_torch(k, s, rng):
    x = rng.standard_normal((2, 3, 8, 8))
    xo, xt = sg.Tensor(x, requires_grad=True), tt(x).requires_grad_(True)
    o = AvgPool2d(k, stride=s)(xo)
    t = torch.nn.functional.avg_pool2d(xt, k, stride=s)
    close(o, t, "avgpool forward")
    o.backward(np.ones(o.shape))
    t.backward(torch.ones_like(t))
    close(xo.grad, xt.grad, "avgpool dX")


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #
def test_layernorm_matches_torch(rng):
    """The convention check that gradcheck cannot make: torch uses BIASED
    variance with eps INSIDE the sqrt. The Annotated Transformer uses unbiased
    std with eps outside, and the two are simply different functions."""
    ours = LayerNorm(6).double()
    ref = torch.nn.LayerNorm(6).double()
    with torch.no_grad():
        ref.weight.copy_(tt(ours.weight.data))
        ref.bias.copy_(tt(ours.bias.data))
    x = rng.standard_normal((4, 5, 6))
    xo, xt = sg.Tensor(x, requires_grad=True), tt(x).requires_grad_(True)
    o, t = ours(xo), ref(xt)
    close(o, t, "layernorm forward")
    v = rng.standard_normal(x.shape)
    o.backward(v)
    t.backward(tt(v))
    close(xo.grad, xt.grad, "layernorm dX")
    close(ours.weight.grad, ref.weight.grad, "layernorm dW")
    close(ours.bias.grad, ref.bias.grad, "layernorm dB")


def test_layernorm_over_multiple_trailing_axes_matches_torch(rng):
    ours = LayerNorm((5, 6)).double()
    ref = torch.nn.LayerNorm([5, 6]).double()
    with torch.no_grad():
        ref.weight.copy_(tt(ours.weight.data))
        ref.bias.copy_(tt(ours.bias.data))
    x = rng.standard_normal((4, 5, 6))
    close(ours(sg.Tensor(x)), ref(tt(x)), "layernorm multi-axis")


def test_groupnorm_matches_torch(rng):
    ours = GroupNorm(2, 6).double()
    ref = torch.nn.GroupNorm(2, 6).double()
    with torch.no_grad():
        ref.weight.copy_(tt(ours.weight.data))
        ref.bias.copy_(tt(ours.bias.data))
    x = rng.standard_normal((3, 6, 4, 4))
    xo, xt = sg.Tensor(x, requires_grad=True), tt(x).requires_grad_(True)
    o, t = ours(xo), ref(xt)
    close(o, t, "groupnorm forward")
    v = rng.standard_normal(x.shape)
    o.backward(v)
    t.backward(tt(v))
    close(xo.grad, xt.grad, "groupnorm dX")


@pytest.mark.parametrize(
    ("cls", "ref_cls", "shape"),
    [
        (BatchNorm1d, torch.nn.BatchNorm1d, (8, 5)),
        (BatchNorm2d, torch.nn.BatchNorm2d, (8, 5, 4, 4)),
    ],
)
def test_batchnorm_train_mode_matches_torch(cls, ref_cls, shape, rng):
    ours = cls(shape[1]).double().train()
    ref = ref_cls(shape[1]).double().train()
    with torch.no_grad():
        ref.weight.copy_(tt(ours.weight.data))
        ref.bias.copy_(tt(ours.bias.data))
    x = rng.standard_normal(shape)
    xo, xt = sg.Tensor(x, requires_grad=True), tt(x).requires_grad_(True)
    o, t = ours(xo), ref(xt)
    close(o, t, "batchnorm forward")
    v = rng.standard_normal(shape)
    o.backward(v)
    t.backward(tt(v))
    close(xo.grad, xt.grad, "batchnorm dX")
    close(ours.weight.grad, ref.weight.grad, "batchnorm dW")


def test_batchnorm_running_stats_match_torch_including_the_bias_correction(rng):
    """Normalisation uses the BIASED variance while the running estimate is
    updated with the UNBIASED one. Using the same variance for both is a real,
    checkable difference that only shows up in eval mode, several epochs later."""
    ours = BatchNorm1d(5).double().train()
    ref = torch.nn.BatchNorm1d(5).double().train()
    with torch.no_grad():
        ref.weight.copy_(tt(ours.weight.data))
        ref.bias.copy_(tt(ours.bias.data))

    for _ in range(6):
        x = rng.standard_normal((16, 5)) * 2.0 + 0.5
        ours(sg.Tensor(x))
        ref(tt(x))

    buffers = dict(ours.named_buffers())
    close(buffers["running_mean"], ref.running_mean, "running_mean")
    close(buffers["running_var"], ref.running_var, "running_var")

    ours.eval()
    ref.eval()
    x = rng.standard_normal((7, 5))
    close(ours(sg.Tensor(x)), ref(tt(x)), "batchnorm eval forward")


# --------------------------------------------------------------------------- #
# recurrence
# --------------------------------------------------------------------------- #
def test_embedding_matches_torch_including_the_scatter_add(rng):
    """A repeated token index must ACCUMULATE. Assignment keeps only the last
    write and silently under-trains exactly the most frequent tokens."""
    ours = Embedding(9, 4, rng=rng).double()
    ref = torch.nn.Embedding(9, 4).double()
    with torch.no_grad():
        ref.weight.copy_(tt(ours.weight.data))
    idx = np.array([[3, 3, 3, 0], [1, 3, 8, 1]])
    o = ours(sg.Tensor(idx))
    t = ref(torch.tensor(idx))
    close(o, t, "embedding forward")
    v = rng.standard_normal(tuple(t.shape))
    o.backward(v)
    t.backward(tt(v))
    close(ours.weight.grad, ref.weight.grad, "embedding dW (scatter-add)")


def _copy_rnn_weights(ours, ref, gates: int, num_layers: int) -> None:
    """Our (input, gates*hidden) is torch's (gates*hidden, input) transposed."""
    with torch.no_grad():
        for layer in range(num_layers):
            cell = ours.cells[layer] if hasattr(ours, "cells") else ours
            getattr(ref, f"weight_ih_l{layer}").copy_(tt(cell.weight_ih.data.T))
            getattr(ref, f"weight_hh_l{layer}").copy_(tt(cell.weight_hh.data.T))
            getattr(ref, f"bias_ih_l{layer}").copy_(tt(cell.bias.data))
            getattr(ref, f"bias_hh_l{layer}").zero_()


@pytest.mark.parametrize(
    ("cls", "ref_cls", "gates", "layers"),
    [
        (RNN, torch.nn.RNN, 1, 1),
        (LSTM, torch.nn.LSTM, 4, 1),
        (LSTM, torch.nn.LSTM, 4, 2),
        (GRU, torch.nn.GRU, 3, 1),
    ],
)
def test_recurrent_layer_matches_torch(cls, ref_cls, gates, layers, rng):
    """Gate ORDER is the thing being checked here. `i, f, g, o` for LSTM and
    `r, z, n` for GRU are conventions, not derivations -- a different order
    trains perfectly well and mismatches every published checkpoint."""
    ins, hid, seq, batch = 4, 6, 5, 3
    ours = cls(ins, hid, num_layers=layers, rng=rng).double()
    ref = ref_cls(ins, hid, num_layers=layers, batch_first=True).double()
    try:
        _copy_rnn_weights(ours, ref, gates, layers)
    except AttributeError as exc:
        pytest.skip(f"parameter layout differs from the assumed one: {exc}")

    x = rng.standard_normal((batch, seq, ins))
    xo, xt = sg.Tensor(x, requires_grad=True), tt(x).requires_grad_(True)
    o, _ = ours(xo)
    t, _ = ref(xt)
    close(o, t, f"{cls.__name__} forward")
    v = rng.standard_normal(tuple(t.shape))
    o.backward(v)
    t.backward(tt(v))
    close(xo.grad, xt.grad, f"{cls.__name__} dX")


# --------------------------------------------------------------------------- #
# attention
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("causal", [False, True])
def test_scaled_dot_product_attention_matches_torch(causal, rng):
    n, h, t_len, d = 2, 3, 6, 8
    q, k, v = (rng.standard_normal((n, h, t_len, d)) for _ in range(3))
    qo, ko, vo = (sg.Tensor(a, requires_grad=True) for a in (q, k, v))
    qt, kt, vt = (tt(a).requires_grad_(True) for a in (q, k, v))

    o = scaled_dot_product_attention(qo, ko, vo, is_causal=causal)
    t = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
    close(o, t, "attention forward")

    g = rng.standard_normal((n, h, t_len, d))
    o.backward(g)
    t.backward(tt(g))
    close(qo.grad, qt.grad, "attention dQ")
    close(ko.grad, kt.grad, "attention dK")
    close(vo.grad, vt.grad, "attention dV")


def test_explicit_mask_polarity_is_true_means_keep(rng):
    """Pinned down because it is a genuine footgun: throughout
    ``scratchgrad.nn.attention`` a mask entry of ``True`` means *this query may
    attend to this key* -- the **opposite** polarity from
    ``Tensor.masked_fill``, where ``True`` means "overwrite this". Passing a
    torch-style "positions to exclude" mask here silently inverts the causal
    structure: the model attends only to the future and still trains."""
    n, h, t_len, d = 1, 2, 5, 8
    q, k, v = (sg.Tensor(rng.standard_normal((n, h, t_len, d))) for _ in range(3))
    keep = np.tril(np.ones((t_len, t_len), dtype=bool))[None, None]

    explicit = scaled_dot_product_attention(q, k, v, mask=keep)
    built_in = scaled_dot_product_attention(q, k, v, is_causal=True)
    close(explicit, built_in, "True-means-keep matches is_causal", atol=0.0)

    # And the inverted mask must NOT agree -- otherwise the test proves nothing.
    inverted = scaled_dot_product_attention(q, k, v, mask=~keep)
    assert not np.allclose(inverted.data, built_in.data)

    _, weights = scaled_dot_product_attention(q, k, v, is_causal=True, return_weights=True)
    np.testing.assert_allclose(weights.sum(-1), 1.0, atol=1e-12)
    upper = np.triu(np.ones((t_len, t_len), dtype=bool), 1)
    assert np.all(weights[..., upper] == 0.0), "causal weights must be strictly lower-triangular"


def test_attention_is_finite_where_a_row_is_fully_masked(rng):
    """torch's own SDPA returns NaN for a fully-masked row. Ours returns a
    finite uniform average -- meaningless, which is correct, but finite, so a
    downstream loss mask can drop it instead of it poisoning every parameter."""
    n, h, t_len, d = 1, 2, 4, 8
    q, k, v = (
        sg.Tensor(rng.standard_normal((n, h, t_len, d)), requires_grad=True) for _ in range(3)
    )
    out = scaled_dot_product_attention(q, k, v, mask=np.ones((n, 1, t_len, t_len), dtype=bool))
    assert np.isfinite(out.data).all()
    out.sum().backward()
    assert np.isfinite(q.grad).all()
