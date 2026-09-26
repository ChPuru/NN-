"""The Module system, initialization, dense layers and losses."""

from __future__ import annotations

import math

import numpy as np
import pytest

import scratchgrad as sg
from scratchgrad.nn import functional as F
from scratchgrad.nn import init
from scratchgrad.nn.layers import GELU, Flatten, Identity, Linear, ReLU
from scratchgrad.nn.loss import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from scratchgrad.nn.module import Module, ModuleDict, ModuleList, Parameter, Sequential
from scratchgrad.testing import TOL_COMPOSED, TOL_OP, adjoint_test, check_module, gradcheck

N, D, C = 7, 11, 5


# --------------------------------------------------------------------------- #
# Module system
# --------------------------------------------------------------------------- #
class Tiny(Module):
    def __init__(self, rng=None):
        super().__init__()
        self.fc1 = Linear(D, 13, rng=rng)
        self.fc2 = Linear(13, C, rng=rng)
        self.register_buffer("calls", np.zeros(1))

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())


def test_parameters_are_discovered_through_nesting():
    m = Tiny(np.random.default_rng(0))
    names = [n for n, _ in m.named_parameters()]
    assert names == ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    assert m.num_parameters() == D * 13 + 13 + 13 * C + C


def test_assigning_before_super_init_raises_clearly():
    class Bad(Module):
        def __init__(self):
            self.fc = Linear(2, 2)  # no super().__init__()

    with pytest.raises(AttributeError, match="super"):
        Bad()


def test_reassigning_a_parameter_name_with_a_plain_value_deregisters_it():
    m = Tiny(np.random.default_rng(0))
    m.fc1 = Identity()
    assert [n for n, _ in m.named_parameters()] == ["fc2.weight", "fc2.bias"]
    m.fc2 = 5
    assert list(m.named_parameters()) == []


def test_shared_parameter_is_yielded_once():
    """Weight tying must not double-count, or the optimizer steps it twice."""

    class Tied(Module):
        def __init__(self):
            super().__init__()
            self.a = Linear(4, 4, rng=np.random.default_rng(0))
            self.b = Linear(4, 4, rng=np.random.default_rng(1))
            self.b.weight = self.a.weight

    assert sum(1 for _ in Tied().parameters()) == 3  # two biases + one shared weight


def test_buffers_are_saved_but_never_trainable():
    m = Tiny(np.random.default_rng(0))
    assert "calls" in dict(m.named_buffers())
    assert "calls" not in [n for n, _ in m.named_parameters()]
    assert "calls" in m.state_dict()


def test_train_eval_propagates_to_children():
    m = Tiny(np.random.default_rng(0))
    m.eval()
    assert not any(sub.training for sub in m.modules())
    m.train()
    assert all(sub.training for sub in m.modules())


def test_state_dict_shares_storage_but_deep_copies_on_request():
    """The shallow default is the documented footgun: an early-stopping snapshot
    silently follows the live model."""
    m = Tiny(np.random.default_rng(0))
    shallow = m.state_dict()
    deep = m.state_dict(deep=True)
    m.fc1.weight.data[0, 0] += 1.0
    assert shallow["fc1.weight"][0, 0] == m.fc1.weight.data[0, 0]
    assert deep["fc1.weight"][0, 0] != m.fc1.weight.data[0, 0]


def test_load_state_dict_round_trip_and_strictness():
    a, b = Tiny(np.random.default_rng(0)), Tiny(np.random.default_rng(1))
    b.load_state_dict(a.state_dict(deep=True))
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        np.testing.assert_array_equal(pa.data, pb.data)
    with pytest.raises(KeyError):
        b.load_state_dict({"nonsense": np.zeros(1)})
    bad = a.state_dict(deep=True)
    bad["fc1.weight"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="shape mismatch"):
        b.load_state_dict(bad)


def test_zero_grad_clears_every_parameter():
    m = Tiny(np.random.default_rng(0))
    m(sg.Tensor(np.random.default_rng(0).standard_normal((N, D)))).sum().backward()
    assert all(p.grad is not None for p in m.parameters())
    m.zero_grad()
    assert all(p.grad is None for p in m.parameters())


def test_sequential_modulelist_moduledict():
    s = Sequential(Linear(D, 8, rng=np.random.default_rng(0)), ReLU(), Linear(8, C, rng=None))
    assert len(s) == 3 and isinstance(s[1], ReLU)
    assert s(sg.Tensor(np.zeros((N, D)))).shape == (N, C)
    ml = ModuleList([Linear(2, 2), Linear(2, 2)])
    assert len(ml) == 2 and sum(1 for _ in ml.parameters()) == 4
    md = ModuleDict({"a": Linear(2, 2)})
    assert md["a"].in_features == 2


def test_repr_and_summary_do_not_crash():
    m = Tiny(np.random.default_rng(0))
    assert "Linear" in repr(m)
    assert "total" in m.summary() and "101" not in m.summary().split("total")[0][:0]


# --------------------------------------------------------------------------- #
# initialization
# --------------------------------------------------------------------------- #
def test_he_init_variance_is_two_over_fan_in():
    """He specifies a VARIANCE of 2/n, so the std is sqrt(2/n). Reading the
    `N(0, 2/n)` notation as `N(mean, std)` gives a std 55x too small at n=784."""
    fan_in = 4096
    p = Parameter(np.zeros((fan_in, 512)))
    init.kaiming_normal_(p, nonlinearity="relu", rng=np.random.default_rng(0))
    assert p.data.var() == pytest.approx(2.0 / fan_in, rel=0.05)
    assert p.data.std() == pytest.approx(math.sqrt(2.0 / fan_in), rel=0.03)


def test_xavier_variance_is_two_over_fan_sum():
    p = Parameter(np.zeros((1024, 2048)))
    init.xavier_normal_(p, rng=np.random.default_rng(0))
    assert p.data.var() == pytest.approx(2.0 / (1024 + 2048), rel=0.05)


def test_uniform_forms_match_their_normal_counterparts_in_variance():
    a, b = Parameter(np.zeros((512, 512))), Parameter(np.zeros((512, 512)))
    init.kaiming_normal_(a, rng=np.random.default_rng(0))
    init.kaiming_uniform_(b, rng=np.random.default_rng(0))
    assert b.data.var() == pytest.approx(a.data.var(), rel=0.08)


def test_orthogonal_init_is_actually_orthogonal(f64):
    # float64 on purpose: at the float32 default this is only orthogonal to
    # ~2e-7, which measures the storage dtype rather than the QR.
    p = Parameter(np.zeros((64, 64)))
    init.orthogonal_(p, rng=np.random.default_rng(0))
    np.testing.assert_allclose(p.data @ p.data.T, np.eye(64), atol=1e-12)


def test_conv_fans_use_the_receptive_field():
    # (out, in, kH, kW): fan_in = in * kH * kW, fan_out = out * kH * kW
    assert init.compute_fans((16, 3, 5, 5)) == (75, 400)


def test_calculate_gain_values():
    assert init.calculate_gain("relu") == pytest.approx(math.sqrt(2.0))
    assert init.calculate_gain("linear") == 1.0
    assert init.calculate_gain("tanh") == pytest.approx(5.0 / 3.0)
    with pytest.raises(ValueError):
        init.calculate_gain("banana")


# --------------------------------------------------------------------------- #
# layers
# --------------------------------------------------------------------------- #
def test_linear_gradients_and_adjoint(f64):
    lin = Linear(D, C, rng=np.random.default_rng(0)).double()
    assert max(check_module(lin, (N, D), tol=TOL_OP).values()) < TOL_OP
    w = lin.weight.data
    assert adjoint_test(lambda u: u @ sg.Tensor(w), lambda u, v: v @ w.T, (N, D), (N, C)) < 1e-13


def test_linear_without_bias_has_no_bias_parameter():
    lin = Linear(D, C, bias=False, rng=np.random.default_rng(0))
    assert [n for n, _ in lin.named_parameters()] == ["weight"]


def test_linear_rejects_a_wrong_input_width():
    with pytest.raises(ValueError, match="expected last dim"):
        Linear(D, C, rng=np.random.default_rng(0))(sg.Tensor(np.zeros((N, D + 1))))


def test_flatten_keeps_the_batch_axis(f64):
    x = sg.Tensor(np.zeros((N, 3, 4, 5)), requires_grad=True)
    assert Flatten()(x).shape == (N, 60)
    gradcheck(lambda t: Flatten()(t) * 2.0, x, tol=TOL_OP)


ACTIVATIONS = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "elu": F.elu,
    "gelu_tanh": F.gelu,
    "silu": F.silu,
    "softplus": F.softplus,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
}


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_activation_gradients(name, f64):
    x = sg.Tensor(np.random.default_rng(0).standard_normal((N, D)), requires_grad=True)
    gradcheck(ACTIVATIONS[name], x, tol=TOL_OP, name=name)


def test_gelu_exact_matches_the_tanh_approximation_closely(f64):
    pytest.importorskip("scipy")
    x = sg.Tensor(np.linspace(-4, 4, 101), requires_grad=True)
    approx, exact = F.gelu(x, "tanh").data, F.gelu(x, "none").data
    assert np.abs(approx - exact).max() < 2e-3
    gradcheck(lambda t: F.gelu(t, "none"), x, tol=TOL_OP, name="gelu_exact")


def test_gelu_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="approximate"):
        F.gelu(sg.Tensor(np.zeros(3)), "erf")


def test_dropout_is_inverted_and_identity_in_eval(f64):
    x = sg.Tensor(np.ones((400, 400)))
    out = F.dropout(x, p=0.4, training=True, rng=np.random.default_rng(0))
    assert out.data.mean() == pytest.approx(1.0, rel=0.02)  # E[out] == input
    assert (F.dropout(x, p=0.4, training=False).data == 1.0).all()


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
@pytest.fixture
def logits(f64):
    return sg.Tensor(np.random.default_rng(0).standard_normal((N, C)), requires_grad=True)


@pytest.fixture
def labels():
    return np.random.default_rng(1).integers(0, C, size=N)


def test_cross_entropy_at_uniform_logits_is_exactly_ln_c(f64, labels):
    """The plan's init check. It needs float64: float32 cannot resolve 1e-12
    (it lands 3e-8 away), which is itself worth knowing."""
    loss = F.cross_entropy(sg.Tensor(np.zeros((N, C))), labels)
    assert abs(loss.item() - math.log(C)) < 1e-15


def test_cross_entropy_does_not_overflow_at_extreme_logits(f64):
    z = sg.Tensor(np.array([[1000.0, 0.0, 0.0, 0.0, 0.0]]))
    assert np.isfinite(F.cross_entropy(z, np.array([0])).item())
    assert F.cross_entropy(z, np.array([0])).item() == pytest.approx(0.0, abs=1e-12)


def test_double_softmax_pins_the_loss_at_ln_c(f64, labels):
    """The signature failure: feeding probabilities to cross_entropy preserves
    the argmax, so accuracy looks fine and only the loss reveals it."""
    probs = sg.Tensor(np.random.default_rng(0).standard_normal((N, C))).softmax(-1)
    # With near-uniform probabilities the doubly-softmaxed loss sits at ~ln(C).
    assert F.cross_entropy(probs, labels).item() == pytest.approx(math.log(C), abs=0.25)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy_gradients_by_reduction(logits, labels, reduction):
    gradcheck(lambda z: F.cross_entropy(z, labels, reduction=reduction), logits, tol=TOL_COMPOSED)


def test_cross_entropy_label_smoothing_and_ignore_index(logits, labels):
    gradcheck(lambda z: F.cross_entropy(z, labels, label_smoothing=0.1), logits, tol=TOL_COMPOSED)
    masked = np.where(np.arange(N) < 2, -100, labels)
    gradcheck(lambda z: F.cross_entropy(z, masked, ignore_index=-100), logits, tol=TOL_COMPOSED)
    # Ignored rows contribute exactly no gradient.
    F.cross_entropy(logits, masked, ignore_index=-100).backward()
    np.testing.assert_allclose(logits.grad[:2], 0.0, atol=1e-15)


def test_cross_entropy_rejects_out_of_range_and_float_targets(logits):
    with pytest.raises(ValueError, match=r"\[0, 5\)"):
        F.cross_entropy(logits, np.full(N, 99))
    with pytest.raises(TypeError, match="integer"):
        F.cross_entropy(logits, np.zeros(N, dtype=float))


def test_fused_cross_entropy_gradient_equals_p_minus_onehot(f64, labels):
    z = sg.Tensor(np.random.default_rng(0).standard_normal((N, C)), requires_grad=True)
    F.cross_entropy(z, labels).backward()
    expected = z.detach().softmax(-1).data.copy()
    expected[np.arange(N), labels] -= 1.0
    np.testing.assert_allclose(z.grad, expected / N, rtol=1e-12)


def test_nll_of_log_softmax_matches_cross_entropy(logits, labels):
    a = F.cross_entropy(logits, labels).item()
    b = F.nll_loss(logits.log_softmax(-1), labels).item()
    assert a == pytest.approx(b, rel=1e-12)


def test_elementwise_losses_reject_a_broadcastable_shape_mismatch(f64):
    """(N,1) vs (N,) broadcasts to (N,N) in silence; the loss then floors at
    Var(y) and the model appears to predict a constant."""
    pred = sg.Tensor(np.zeros((N, 1)), requires_grad=True)
    for fn in (F.mse_loss, F.l1_loss, F.huber_loss, F.binary_cross_entropy_with_logits):
        with pytest.raises(ValueError, match="shape mismatch"):
            fn(pred, np.zeros(N))


@pytest.mark.parametrize(
    "fn",
    [F.mse_loss, F.l1_loss, F.huber_loss, F.binary_cross_entropy_with_logits],
)
def test_elementwise_loss_gradients(fn, logits):
    target = (np.random.default_rng(2).random((N, C)) > 0.5).astype(float)
    gradcheck(lambda z: fn(z, target), logits, tol=TOL_COMPOSED)


def test_bce_with_logits_is_stable_where_log_sigmoid_would_overflow(f64):
    z = sg.Tensor(np.array([[-800.0, 800.0]]), requires_grad=True)
    out = F.binary_cross_entropy_with_logits(z, np.array([[1.0, 0.0]]))
    assert np.isfinite(out.item())


def test_loss_modules_match_their_functional_forms(logits, labels):
    assert CrossEntropyLoss()(logits, labels).item() == pytest.approx(
        F.cross_entropy(logits, labels).item()
    )
    t = np.zeros((N, C))
    assert MSELoss()(logits, t).item() == pytest.approx(F.mse_loss(logits, t).item())
    assert BCEWithLogitsLoss()(logits, t).item() == pytest.approx(
        F.binary_cross_entropy_with_logits(logits, t).item()
    )


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_mlp_overfits_a_tiny_batch(f64):
    """If the loss cannot reach ~0 on 32 examples, the model is broken, not
    undertrained. This is the check that runs before any real training."""
    rng = np.random.default_rng(0)
    x = sg.Tensor(rng.standard_normal((32, D)))
    y = rng.integers(0, C, size=32)
    model = Sequential(Linear(D, 64, rng=rng), ReLU(), Linear(64, C, rng=rng)).double()
    params = list(model.parameters())
    for _ in range(2000):
        loss = F.cross_entropy(model(x), y)
        for p in params:
            p.grad = None
        loss.backward()
        for p in params:
            p.data -= 0.5 * p.grad
    assert loss.item() < 1e-3


def test_composed_model_gradcheck(f64):
    model = Sequential(
        Linear(D, 8, rng=np.random.default_rng(0)),
        GELU(),
        Linear(8, C, rng=np.random.default_rng(1)),
    ).double()
    labels = np.random.default_rng(2).integers(0, C, size=N)
    x = np.random.default_rng(3).standard_normal((N, D))
    errs = check_module(
        model,
        (N, D),
        input_data=x,
        tol=TOL_COMPOSED,
        forward=lambda m, t: F.cross_entropy(m(t), labels),
    )
    assert max(errs.values()) < TOL_COMPOSED
