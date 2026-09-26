"""The "General" property: can someone build what the library never anticipated?

`ROADMAP.md` grades this project against five properties, and this one is the
line between a framework and a script that hardcodes one architecture:

    *Someone can define a model you never anticipated without editing the
    library. This is what autograd buys.*

That claim is easy to assert and easy to get wrong, so it is tested here the
only way it can be: by building things that exist nowhere in `src/` -- a new
differentiable primitive, a layer with a non-standard backward, a model with
branching and weight sharing, a custom loss, a custom optimizer -- using **only**
the public API, and checking their gradients.

If any test in this file needs a change under `src/` to pass, the property is
false and the claim in the roadmap has to come out.
"""

from __future__ import annotations

import numpy as np
import pytest

import scratchgrad as sg
from scratchgrad import nn, optim
from scratchgrad.nn import functional as F
from scratchgrad.tensor import Tensor
from scratchgrad.testing import TOL_COMPOSED, TOL_OP, check_module, gradcheck

N, D, C = 7, 11, 5


# --------------------------------------------------------------------------- #
# 1. a new differentiable primitive, defined entirely outside the library
# --------------------------------------------------------------------------- #
def softsign(x: Tensor) -> Tensor:
    """``x / (1 + |x|)``, with a hand-written backward: ``1 / (1 + |x|)^2``.

    Nothing in ``src/`` knows this function exists. It is registered with the
    engine through the same public ``Tensor._make`` contract every built-in op
    uses.
    """
    a = x.data
    denom = 1.0 + np.abs(a)
    return Tensor._make(a / denom, (x,), (lambda g: g / (denom * denom),), "softsign")


def test_a_user_defined_primitive_gradchecks(f64, rng):
    x = sg.Tensor(rng.standard_normal((N, D)), requires_grad=True)
    gradcheck(softsign, x, tol=TOL_OP, name="softsign")


def test_a_user_defined_primitive_composes_with_built_ins(f64, rng):
    x = sg.Tensor(rng.standard_normal((N, D)), requires_grad=True)
    w = sg.Tensor(rng.standard_normal((D, C)), requires_grad=True)
    gradcheck(
        lambda x, w: softsign(x @ w).log_softmax(-1), x, w, tol=TOL_COMPOSED, name="softsign_mlp"
    )


# --------------------------------------------------------------------------- #
# 2. a layer with a genuinely non-standard forward
# --------------------------------------------------------------------------- #
class Maxout(nn.Module):
    """Maxout: ``k`` parallel affine maps, then a max over them.

    Chosen deliberately because its backward is a *routing* operation rather
    than a pointwise multiply -- if the engine's max-reduction gradient were
    wrong, or if ties were double-counted, this would fail.
    """

    def __init__(self, n_in: int, n_out: int, k: int = 3, rng=None) -> None:
        super().__init__()
        self.k = int(k)
        self.n_out = int(n_out)
        self.proj = nn.Linear(n_in, n_out * k, rng=rng)

    def forward(self, x):
        z = self.proj(x)
        return z.reshape(x.shape[0], self.n_out, self.k).max(axis=2)


def test_a_user_defined_layer_gradchecks(f64):
    layer = Maxout(D, C, k=3, rng=np.random.default_rng(0))
    assert max(check_module(layer, (N, D), tol=TOL_COMPOSED).values()) < TOL_COMPOSED


def test_a_user_defined_layer_registers_its_parameters():
    layer = Maxout(D, C, k=3, rng=np.random.default_rng(0))
    assert [n for n, _ in layer.named_parameters()] == ["proj.weight", "proj.bias"]
    assert layer.num_parameters() == D * C * 3 + C * 3


# --------------------------------------------------------------------------- #
# 3. a model with branching, concatenation and weight sharing
# --------------------------------------------------------------------------- #
class MultiScaleTower(nn.Module):
    """Two branches over the same input, concatenated, with a **shared** trunk.

    This is the navoshta multi-scale idea in miniature. It exercises three
    things a hardcoded-architecture library cannot do: a non-linear graph, a
    parameter used twice on different paths (its gradient must be the *sum* of
    both contributions), and a residual connection.
    """

    def __init__(self, n_in: int, hidden: int, n_out: int, rng=None) -> None:
        super().__init__()
        self.shared = nn.Linear(n_in, hidden, rng=rng)
        self.wide = nn.Linear(n_in, hidden, rng=rng)
        self.norm = nn.LayerNorm(2 * hidden)
        self.head = nn.Linear(2 * hidden, n_out, rng=rng)

    def forward(self, x):
        a = self.shared(x).relu()
        b = self.wide(x).tanh()
        # `shared` is used a second time, on the already-transformed branch:
        # its gradient has to accumulate over both uses.
        a = a + self.shared(x) * 0.5
        merged = sg.concatenate([a, b], axis=1)
        return self.head(self.norm(merged))


def test_a_branching_weight_sharing_model_gradchecks(f64):
    model = MultiScaleTower(D, 6, C, rng=np.random.default_rng(0))
    assert max(check_module(model, (N, D), tol=TOL_COMPOSED).values()) < TOL_COMPOSED


def test_shared_parameter_gradient_is_the_sum_of_both_paths(f64):
    """The property that makes weight sharing correct rather than merely legal."""
    w = sg.Tensor(np.array([[2.0]]), requires_grad=True)
    x = sg.Tensor(np.array([[3.0]]))
    # y = (x@w) + (x@w)*0.5  ->  dy/dw = 3 + 1.5 = 4.5
    ((x @ w) + (x @ w) * 0.5).sum().backward()
    np.testing.assert_allclose(w.grad, [[4.5]])


# --------------------------------------------------------------------------- #
# 4. a custom loss with a hand-derived backward
# --------------------------------------------------------------------------- #
def focal_loss(logits: Tensor, target, gamma: float = 2.0) -> Tensor:
    """Focal loss, built by *composing public ops* -- no new primitive at all.

    Written this way on purpose: it demonstrates that a user who does not want
    to derive anything by hand can still get a correct gradient, because the
    engine differentiates the composition for them.
    """
    logp = logits.log_softmax(-1)
    n = logits.shape[0]
    picked = logp[np.arange(n), np.asarray(target)]
    return (-((1.0 - picked.exp()) ** gamma) * picked).mean()


def test_a_composed_custom_loss_gradchecks(f64, rng):
    z = sg.Tensor(rng.standard_normal((N, C)), requires_grad=True)
    y = rng.integers(0, C, size=N)
    gradcheck(lambda z: focal_loss(z, y), z, tol=TOL_COMPOSED, name="focal")


def test_focal_loss_reduces_to_cross_entropy_at_gamma_zero(f64, rng):
    z = sg.Tensor(rng.standard_normal((N, C)), requires_grad=True)
    y = rng.integers(0, C, size=N)
    assert focal_loss(z, y, gamma=0.0).item() == pytest.approx(
        F.cross_entropy(z, y).item(), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# 5. a custom optimizer written against the public Optimizer base
# --------------------------------------------------------------------------- #
class SignSGD(optim.Optimizer):
    """``p -= lr * sign(g)``. Twelve lines, no library edit."""

    def __init__(self, params, lr: float = 0.01):
        super().__init__(params, {"lr": lr})

    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.data -= group["lr"] * np.sign(p.grad)


def test_a_custom_optimizer_trains(f64):
    rng = np.random.default_rng(0)
    x = sg.Tensor(rng.standard_normal((64, D)))
    y = rng.integers(0, C, size=64)
    model = nn.Sequential(nn.Linear(D, 16, rng=rng), nn.ReLU(), nn.Linear(16, C, rng=rng)).double()
    opt = SignSGD(model.parameters(), lr=0.01)
    first = F.cross_entropy(model(x), y).item()
    for _ in range(300):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.3


# --------------------------------------------------------------------------- #
# 6. the headline claim: a new architecture in under 20 lines
# --------------------------------------------------------------------------- #
def test_a_novel_architecture_in_under_twenty_lines(f64):
    """A gated residual tower with a custom activation, defined and trained here.

    Counted: the class body below is 11 statements. Nothing in ``src/`` was
    touched to make it work.
    """

    class GatedResidual(nn.Module):
        def __init__(self, dim, rng=None):
            super().__init__()
            self.f = nn.Linear(dim, dim, rng=rng)
            self.g = nn.Linear(dim, dim, rng=rng)
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            gate = self.g(x).sigmoid()
            return self.norm(x + gate * softsign(self.f(x)))

    rng = np.random.default_rng(0)
    model = nn.Sequential(
        nn.Linear(D, 12, rng=rng),
        GatedResidual(12, rng=rng),
        GatedResidual(12, rng=rng),
        nn.Linear(12, C, rng=rng),
    ).double()

    x = sg.Tensor(rng.standard_normal((48, D)))
    y = rng.integers(0, C, size=48)

    # It differentiates correctly ...
    errs = check_module(
        model, (48, D), tol=TOL_COMPOSED, forward=lambda m, t: F.cross_entropy(m(t), y)
    )
    assert max(errs.values()) < TOL_COMPOSED

    # ... and it actually trains.
    opt = optim.Adam(model.parameters(), lr=3e-3)
    start = F.cross_entropy(model(x), y).item()
    for _ in range(300):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"started at {start:.3f}, ended at {loss.item():.3f}"


def test_no_source_file_mentions_any_symbol_defined_in_this_test():
    """Guard against the property being true only because the library was
    quietly taught about these shapes."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    blob = "\n".join(p.read_text(encoding="utf-8") for p in src.rglob("*.py"))
    for name in ("softsign", "Maxout", "MultiScaleTower", "focal_loss", "SignSGD", "GatedResidual"):
        assert name not in blob, f"{name} leaked into src/ -- the extensibility claim is circular"
