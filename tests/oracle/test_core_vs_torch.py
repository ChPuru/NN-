"""Equivalence tests for the core engine against PyTorch.

PyTorch is a **test oracle only**. It is never imported by anything under
``src/`` -- there is a test in ``tests/test_no_framework_imports.py`` that
enforces exactly that, and it would be trivially defeated if library code could
fall back to torch when torch happened to be installed.

What an oracle buys that gradcheck does not: gradcheck proves
``backward == d(forward)/dx``, which a *wrong forward* with a self-consistently
derived backward satisfies forever. Comparing the forward **values** against an
independent implementation is what closes that gap.

These tests skip cleanly when torch is absent. CI installs it via
``uv sync --all-groups`` -- note that ``--all-extras --dev`` does **not** install
PEP 735 dependency groups, so the oracle would be silently missing and CI would
stay green while testing nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

import scratchgrad as sg
from scratchgrad.nn import functional as F
from scratchgrad.nn.layers import Linear

torch = pytest.importorskip("torch", reason="PyTorch oracle not installed")

pytestmark = pytest.mark.oracle

N, D, C = 7, 11, 5
ATOL = 1e-11  # float64 both sides; anything looser would hide a real difference


@pytest.fixture(autouse=True)
def _f64():
    with sg.default_dtype(np.float64):
        yield


def pair(arr: np.ndarray):
    """The same values as a scratchgrad Tensor and a torch tensor, both float64."""
    a = np.asarray(arr, dtype=np.float64)
    return (
        sg.Tensor(a.copy(), requires_grad=True),
        torch.tensor(a.copy(), dtype=torch.float64, requires_grad=True),
    )


def both_match(ours, theirs, atol=ATOL):
    np.testing.assert_allclose(
        np.asarray(ours.data, dtype=np.float64),
        theirs.detach().numpy(),
        atol=atol,
        rtol=atol,
        err_msg="forward values differ",
    )


def grads_match(ours, theirs, atol=ATOL):
    np.testing.assert_allclose(
        np.asarray(ours.grad, dtype=np.float64),
        theirs.grad.numpy(),
        atol=atol,
        rtol=atol,
        err_msg="gradients differ",
    )


# --------------------------------------------------------------------------- #
# activations
# --------------------------------------------------------------------------- #
ACTIVATIONS = [
    ("relu", F.relu, torch.relu),
    ("sigmoid", F.sigmoid, torch.sigmoid),
    ("tanh", F.tanh, torch.tanh),
    ("silu", F.silu, torch.nn.functional.silu),
    ("elu", F.elu, torch.nn.functional.elu),
    ("softplus", F.softplus, torch.nn.functional.softplus),
    ("leaky_relu", F.leaky_relu, lambda t: torch.nn.functional.leaky_relu(t, 0.01)),
    ("gelu_tanh", F.gelu, lambda t: torch.nn.functional.gelu(t, approximate="tanh")),
]


@pytest.mark.parametrize(("name", "ours", "theirs"), ACTIVATIONS, ids=[a[0] for a in ACTIVATIONS])
def test_activation_forward_and_backward(name, ours, theirs, rng):
    x_ours, x_th = pair(rng.standard_normal((N, D)) * 2.0)
    v = rng.standard_normal((N, D))
    o, t = ours(x_ours), theirs(x_th)
    both_match(o, t)
    o.backward(v)
    t.backward(torch.tensor(v, dtype=torch.float64))
    grads_match(x_ours, x_th)


def test_gelu_exact_matches_torch_default(rng):
    pytest.importorskip("scipy")
    x_ours, x_th = pair(rng.standard_normal((N, D)) * 2.0)
    o = F.gelu(x_ours, "none")
    t = torch.nn.functional.gelu(x_th)  # torch's default IS the exact erf form
    both_match(o, t)
    v = rng.standard_normal((N, D))
    o.backward(v)
    t.backward(torch.tensor(v, dtype=torch.float64))
    grads_match(x_ours, x_th)


def test_sigmoid_agrees_in_the_saturated_tails():
    """Agreement to the last bit, including where a naive ``1/(1+exp(-x))``
    would overflow. One ULP is the floor here -- sigmoid(-50) is 1.93e-22 and the
    two implementations branch differently, so demanding exact equality would be
    demanding that float64 rounding be identical, not that the maths is."""
    x_ours, x_th = pair(np.array([-800.0, -50.0, -1.0, 0.0, 1.0, 50.0, 800.0]))
    both_match(x_ours.sigmoid(), torch.sigmoid(x_th), atol=1e-15)


# --------------------------------------------------------------------------- #
# reductions and softmax
# --------------------------------------------------------------------------- #
REDUCTIONS = [
    ("sum_all", lambda x: x.sum(), lambda t: t.sum()),
    ("sum_axis", lambda x: x.sum(axis=1), lambda t: t.sum(dim=1)),
    ("sum_keep", lambda x: x.sum(axis=0, keepdims=True), lambda t: t.sum(dim=0, keepdim=True)),
    ("mean_axis", lambda x: x.mean(axis=1), lambda t: t.mean(dim=1)),
    ("max_axis", lambda x: x.max(axis=1), lambda t: t.max(dim=1).values),
    ("min_axis", lambda x: x.min(axis=0), lambda t: t.min(dim=0).values),
    ("var_biased", lambda x: x.var(axis=1), lambda t: t.var(dim=1, unbiased=False)),
    ("var_unbiased", lambda x: x.var(axis=1, ddof=1), lambda t: t.var(dim=1, unbiased=True)),
    ("std_biased", lambda x: x.std(axis=1), lambda t: t.std(dim=1, unbiased=False)),
    ("logsumexp", lambda x: x.logsumexp(axis=1), lambda t: t.logsumexp(dim=1)),
    ("softmax", lambda x: x.softmax(-1), lambda t: torch.softmax(t, dim=-1)),
    ("log_softmax", lambda x: x.log_softmax(-1), lambda t: torch.log_softmax(t, dim=-1)),
    ("norm2", lambda x: x.norm(2.0), lambda t: t.norm(2)),
]


@pytest.mark.parametrize(("name", "ours", "theirs"), REDUCTIONS, ids=[r[0] for r in REDUCTIONS])
def test_reduction_forward_and_backward(name, ours, theirs, rng):
    x_ours, x_th = pair(rng.standard_normal((N, D)))
    o, t = ours(x_ours), theirs(x_th)
    both_match(o, t)
    v = rng.standard_normal(o.data.shape)
    o.backward(v)
    t.backward(torch.tensor(v, dtype=torch.float64))
    grads_match(x_ours, x_th)


def test_softmax_agrees_at_logits_that_would_overflow(rng):
    x_ours, x_th = pair(rng.standard_normal((N, C)) * 400.0)
    both_match(x_ours.softmax(-1), torch.softmax(x_th, dim=-1))
    both_match(x_ours.logsumexp(axis=1), x_th.logsumexp(dim=1))


# --------------------------------------------------------------------------- #
# matmul and shapes
# --------------------------------------------------------------------------- #
def test_matmul_2d(rng):
    a_ours, a_th = pair(rng.standard_normal((N, D)))
    b_ours, b_th = pair(rng.standard_normal((D, C)))
    o, t = a_ours @ b_ours, a_th @ b_th
    both_match(o, t)
    v = rng.standard_normal((N, C))
    o.backward(v)
    t.backward(torch.tensor(v, dtype=torch.float64))
    grads_match(a_ours, a_th)
    grads_match(b_ours, b_th)


@pytest.mark.parametrize(
    ("ashape", "bshape"),
    [
        ((N, D), (D,)),
        ((D,), (D, C)),
        ((D,), (D,)),
        ((3, N, D), (D, C)),
        ((3, N, D), (3, D, C)),
        ((1, N, D), (4, D, C)),  # broadcast the batch axis: needs unbroadcast on both sides
    ],
)
def test_matmul_rank_and_broadcast_combinations(ashape, bshape, rng):
    a_ours, a_th = pair(rng.standard_normal(ashape))
    b_ours, b_th = pair(rng.standard_normal(bshape))
    o, t = a_ours @ b_ours, a_th @ b_th
    both_match(o, t)
    v = rng.standard_normal(tuple(t.shape))
    o.backward(v)
    t.backward(torch.tensor(v, dtype=torch.float64))
    grads_match(a_ours, a_th)
    grads_match(b_ours, b_th)


def test_broadcast_add_gradient_is_summed(rng):
    x_ours, x_th = pair(rng.standard_normal((N, D)))
    b_ours, b_th = pair(rng.standard_normal(D))
    o, t = (x_ours + b_ours).sum(), (x_th + b_th).sum()
    o.backward()
    t.backward()
    grads_match(b_ours, b_th)


SHAPE_OPS = [
    ("transpose", lambda x: x.transpose(), lambda t: t.T),
    ("reshape", lambda x: x.reshape(D, N), lambda t: t.reshape(D, N)),
    (
        "permute",
        lambda x: x.reshape(N, D, 1).permute(2, 0, 1),
        lambda t: t.reshape(N, D, 1).permute(2, 0, 1),
    ),
    ("slice", lambda x: x[1:5, 2:9], lambda t: t[1:5, 2:9]),
    ("fancy", lambda x: x[np.array([0, 2, 2, 5])], lambda t: t[torch.tensor([0, 2, 2, 5])]),
    ("pad", lambda x: x.pad(((1, 2), (0, 3))), lambda t: torch.nn.functional.pad(t, (0, 3, 1, 2))),
    ("repeat", lambda x: x.repeat(3, axis=0), lambda t: t.repeat_interleave(3, dim=0)),
]


@pytest.mark.parametrize(("name", "ours", "theirs"), SHAPE_OPS, ids=[s[0] for s in SHAPE_OPS])
def test_shape_op_forward_and_backward(name, ours, theirs, rng):
    x_ours, x_th = pair(rng.standard_normal((N, D)))
    o, t = ours(x_ours), theirs(x_th)
    both_match(o, t)
    v = rng.standard_normal(tuple(t.shape))
    o.backward(v)
    t.backward(torch.tensor(v, dtype=torch.float64))
    grads_match(x_ours, x_th)


def test_repeated_fancy_index_accumulates_like_torch(rng):
    """The scatter-add. Assignment would keep only the last write."""
    x_ours, x_th = pair(np.zeros((4, 2)))
    idx = np.array([1, 1, 1, 3])
    x_ours[idx].sum().backward()
    x_th[torch.tensor(idx)].sum().backward()
    grads_match(x_ours, x_th, atol=0.0)


def test_concatenate_and_stack(rng):
    a_ours, a_th = pair(rng.standard_normal((N, D)))
    b_ours, b_th = pair(rng.standard_normal((N, D)))
    for ours, theirs in [
        (sg.concatenate([a_ours, b_ours], axis=0), torch.cat([a_th, b_th], dim=0)),
        (sg.stack([a_ours, b_ours], axis=1), torch.stack([a_th, b_th], dim=1)),
    ]:
        both_match(ours, theirs)
    a_ours.grad = None
    b_ours.grad = None
    a_th.grad = None
    b_th.grad = None
    v = rng.standard_normal((2 * N, D))
    sg.concatenate([a_ours, b_ours], axis=0).backward(v)
    torch.cat([a_th, b_th], dim=0).backward(torch.tensor(v, dtype=torch.float64))
    grads_match(a_ours, a_th)
    grads_match(b_ours, b_th)


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy_matches_torch(reduction, rng):
    z_ours, z_th = pair(rng.standard_normal((N, C)))
    y = rng.integers(0, C, size=N)
    o = F.cross_entropy(z_ours, y, reduction=reduction)
    t = torch.nn.functional.cross_entropy(z_th, torch.tensor(y), reduction=reduction)
    both_match(o, t)
    v = rng.standard_normal(tuple(t.shape)) if t.ndim else None
    if v is None:
        o.backward()
        t.backward()
    else:
        o.backward(v)
        t.backward(torch.tensor(v, dtype=torch.float64))
    grads_match(z_ours, z_th)


def test_cross_entropy_label_smoothing_matches_torch(rng):
    z_ours, z_th = pair(rng.standard_normal((N, C)))
    y = rng.integers(0, C, size=N)
    o = F.cross_entropy(z_ours, y, label_smoothing=0.1)
    t = torch.nn.functional.cross_entropy(z_th, torch.tensor(y), label_smoothing=0.1)
    both_match(o, t)
    o.backward()
    t.backward()
    grads_match(z_ours, z_th)


def test_cross_entropy_ignore_index_matches_torch(rng):
    """'mean' must divide by the number of CONTRIBUTING rows, not by N."""
    z_ours, z_th = pair(rng.standard_normal((N, C)))
    y = rng.integers(0, C, size=N)
    y[:2] = -100
    o = F.cross_entropy(z_ours, y, ignore_index=-100)
    t = torch.nn.functional.cross_entropy(z_th, torch.tensor(y), ignore_index=-100)
    both_match(o, t)
    o.backward()
    t.backward()
    grads_match(z_ours, z_th)
    np.testing.assert_allclose(z_ours.grad[:2], 0.0, atol=0.0)


def test_nll_loss_matches_torch(rng):
    z_ours, z_th = pair(rng.standard_normal((N, C)))
    y = rng.integers(0, C, size=N)
    o = F.nll_loss(z_ours.log_softmax(-1), y)
    t = torch.nn.functional.nll_loss(torch.log_softmax(z_th, dim=-1), torch.tensor(y))
    both_match(o, t)
    o.backward()
    t.backward()
    grads_match(z_ours, z_th)


ELEMENTWISE_LOSSES = [
    ("mse", F.mse_loss, torch.nn.functional.mse_loss),
    ("l1", F.l1_loss, torch.nn.functional.l1_loss),
    (
        "huber",
        lambda p, t, **k: F.huber_loss(p, t, delta=1.0, **k),
        lambda p, t, **k: torch.nn.functional.huber_loss(p, t, delta=1.0, **k),
    ),
    (
        "bce_logits",
        F.binary_cross_entropy_with_logits,
        torch.nn.functional.binary_cross_entropy_with_logits,
    ),
]


@pytest.mark.parametrize(
    ("name", "ours", "theirs"), ELEMENTWISE_LOSSES, ids=[e[0] for e in ELEMENTWISE_LOSSES]
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_elementwise_loss_matches_torch(name, ours, theirs, reduction, rng):
    z_ours, z_th = pair(rng.standard_normal((N, C)))
    tgt = (rng.random((N, C)) > 0.5).astype(np.float64)
    o = ours(z_ours, tgt, reduction=reduction)
    t = theirs(z_th, torch.tensor(tgt), reduction=reduction)
    both_match(o, t)
    o.backward()
    t.backward()
    grads_match(z_ours, z_th)


def test_bce_with_logits_pos_weight_matches_torch(rng):
    z_ours, z_th = pair(rng.standard_normal((N, C)))
    tgt = (rng.random((N, C)) > 0.5).astype(np.float64)
    o = F.binary_cross_entropy_with_logits(z_ours, tgt, pos_weight=2.5)
    t = torch.nn.functional.binary_cross_entropy_with_logits(
        z_th, torch.tensor(tgt), pos_weight=torch.tensor(2.5, dtype=torch.float64)
    )
    both_match(o, t)
    o.backward()
    t.backward()
    grads_match(z_ours, z_th)


# --------------------------------------------------------------------------- #
# Linear
# --------------------------------------------------------------------------- #
def test_linear_matches_torch_linear(rng):
    """Our weight is (in, out); torch's is (out, in). The transpose at the
    boundary here is the whole difference, and getting it wrong is silent."""
    lin = Linear(D, C, rng=rng).double()
    x_ours, x_th = pair(rng.standard_normal((N, D)))
    ref = torch.nn.Linear(D, C).double()
    with torch.no_grad():
        ref.weight.copy_(torch.tensor(lin.weight.data.T.copy()))
        ref.bias.copy_(torch.tensor(lin.bias.data.copy()))

    o, t = lin(x_ours), ref(x_th)
    both_match(o, t)
    v = rng.standard_normal((N, C))
    o.backward(v)
    t.backward(torch.tensor(v, dtype=torch.float64))
    grads_match(x_ours, x_th)
    np.testing.assert_allclose(lin.weight.grad, ref.weight.grad.numpy().T, atol=ATOL)
    np.testing.assert_allclose(lin.bias.grad, ref.bias.grad.numpy(), atol=ATOL)


def test_full_mlp_step_matches_torch(rng):
    """Ten manual SGD steps on identical initial weights must stay in lockstep.
    Divergence at step k localises the disagreement far better than a single
    forward comparison can."""
    from scratchgrad.nn.layers import ReLU
    from scratchgrad.nn.module import Sequential

    ours = Sequential(Linear(D, 8, rng=rng), ReLU(), Linear(8, C, rng=rng)).double()
    ref = torch.nn.Sequential(
        torch.nn.Linear(D, 8), torch.nn.ReLU(), torch.nn.Linear(8, C)
    ).double()
    with torch.no_grad():
        ref[0].weight.copy_(torch.tensor(ours[0].weight.data.T.copy()))
        ref[0].bias.copy_(torch.tensor(ours[0].bias.data.copy()))
        ref[2].weight.copy_(torch.tensor(ours[2].weight.data.T.copy()))
        ref[2].bias.copy_(torch.tensor(ours[2].bias.data.copy()))

    x = rng.standard_normal((N, D))
    y = rng.integers(0, C, size=N)
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y)

    for step in range(10):
        lo = F.cross_entropy(ours(sg.Tensor(x)), y)
        lt = torch.nn.functional.cross_entropy(ref(xt), yt)
        assert lo.item() == pytest.approx(lt.item(), abs=1e-11), f"loss diverged at step {step}"
        ours.zero_grad()
        ref.zero_grad()
        lo.backward()
        lt.backward()
        for p in ours.parameters():
            p.data -= 0.1 * p.grad
        with torch.no_grad():
            for p in ref.parameters():
                p -= 0.1 * p.grad
