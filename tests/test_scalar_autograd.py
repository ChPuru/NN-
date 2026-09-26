"""Tests for the Tier-1a scalar engine in ``examples/scalar_autograd.py``.

Three things are checked, in the order they matter:

1. the gradients are right -- central differences against the hand-written
   ``_backward`` closures, on composed expressions and on each primitive;
2. the recursive/iterative topological-sort contrast is real, not rhetorical --
   the recursive sort is asserted to raise ``RecursionError`` at a depth the
   iterative one walks without complaint, and both are asserted to produce the
   *same* gradients where both survive;
3. the toy classifier actually learns, at a fixed seed.

``examples`` is not an installed package, so the repo root goes on ``sys.path``
explicitly. Importing the module must not train anything -- everything with a
runtime is behind ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.scalar_autograd import (
    MLP,
    Value,
    accuracy,
    benchmark_node_rate,
    build_fan_in_sum,
    cross_entropy,
    extrapolate_mnist_epoch,
    find_recursive_topo_limit,
    logsumexp,
    make_moons,
    mnist_mlp_node_count,
    topological_order_iterative,
    topological_order_recursive,
    train_toy_model,
)

#: Central differences on float64 with a well-scaled step land around 1e-10.
#: Anything looser than this would pass a gradient that is wrong in the 6th
#: digit, which is exactly the size of a dropped ``0.5`` in a tie rule.
TOL = 1e-8


def numeric_grad(fn, xs: list[float], h: float = 1e-6) -> list[float]:
    """Central-difference gradient of ``fn(list[float]) -> float`` at ``xs``.

    Central, not forward: the forward difference has O(h) truncation error and
    cannot distinguish a correct gradient from one that is off by a percent.
    """
    out = []
    for i in range(len(xs)):
        hi = list(xs)
        lo = list(xs)
        hi[i] += h
        lo[i] -= h
        out.append((fn(hi) - fn(lo)) / (2.0 * h))
    return out


def analytic_grad(fn, xs: list[float]) -> list[float]:
    """Gradient of ``fn`` at ``xs`` through the ``Value`` engine."""
    nodes = [Value(x) for x in xs]
    fn(nodes).backward()
    return [n.grad for n in nodes]


def rel_error(a: list[float], b: list[float]) -> float:
    """Array-scale relative error, matching ``scratchgrad.testing.rel_error``.

    Per-element normalisation cannot pass at 1e-8 for a legitimately tiny
    component; the scale of the whole gradient is the right denominator.
    """
    num = max(abs(x - y) for x, y in zip(a, b, strict=True))
    den = max(abs(x) for x in a) + max(abs(x) for x in b) + 1e-12
    return num / den


# Each entry is (name, symbolic fn over Values, same fn over floats). The float
# version is written independently of the engine on purpose: reusing the engine
# for the reference would make the test check the engine against itself.
EXPRESSIONS = [
    (
        "polynomial",
        lambda v: (v[0] * v[1] + v[2] ** 3) * (v[0] - 2.0 * v[2]) + v[1],
        lambda x: (x[0] * x[1] + x[2] ** 3) * (x[0] - 2.0 * x[2]) + x[1],
    ),
    (
        "division_and_negation",
        lambda v: (-v[0] + 3.0) / (v[1] * v[1] + 1.0) - v[2] / v[0],
        lambda x: (-x[0] + 3.0) / (x[1] * x[1] + 1.0) - x[2] / x[0],
    ),
    (
        "reused_node",  # v0 appears four times: exercises grad += , not grad =
        lambda v: (v[0] * v[0]) * (v[0] + v[1]) + v[0],
        lambda x: (x[0] * x[0]) * (x[0] + x[1]) + x[0],
    ),
    (
        "exp_log_tanh",  # the three primitives micrograd does not have
        lambda v: v[0].exp() * v[1].tanh() + (v[2] * v[2] + 1.0).log(),
        lambda x: math.exp(x[0]) * math.tanh(x[1]) + math.log(x[2] ** 2 + 1.0),
    ),
    (
        "relu_composed",
        lambda v: (v[0] * 2.0 - v[1]).relu() * v[2] + (v[1] - 0.5).relu(),
        lambda x: max(2.0 * x[0] - x[1], 0.0) * x[2] + max(x[1] - 0.5, 0.0),
    ),
    (
        "logsumexp_stability",
        lambda v: logsumexp([v[0], v[1], v[2]]),
        lambda x: math.log(sum(math.exp(t - max(x)) for t in x)) + max(x),
    ),
    (
        "cross_entropy_label_1",
        lambda v: cross_entropy([v[0], v[1], v[2]], 1),
        lambda x: math.log(sum(math.exp(t - max(x)) for t in x)) + max(x) - x[1],
    ),
    (
        "sigmoid_via_tanh",
        lambda v: v[0].sigmoid() * v[1] + v[2].sigmoid(),
        lambda x: x[1] / (1.0 + math.exp(-x[0])) + 1.0 / (1.0 + math.exp(-x[2])),
    ),
]

#: Deliberately non-symmetric, away from 0 and away from every ReLU kink in the
#: table above: a probe pair straddling a kink measures the average of two
#: one-sided derivatives and fails a correct implementation.
POINTS = [
    [1.3, -0.7, 0.45],
    [-0.9, 2.1, 1.7],
    [0.31, 0.83, -1.24],
]


@pytest.mark.parametrize(("name", "sym", "ref"), EXPRESSIONS, ids=[e[0] for e in EXPRESSIONS])
@pytest.mark.parametrize("xs", POINTS, ids=lambda p: ",".join(f"{v:g}" for v in p))
def test_gradients_match_finite_differences(name, sym, ref, xs):
    ana = analytic_grad(sym, xs)
    num = numeric_grad(ref, xs)
    assert rel_error(ana, num) < TOL, f"{name} at {xs}: analytic {ana} vs numeric {num}"


@pytest.mark.parametrize(("name", "sym", "ref"), EXPRESSIONS, ids=[e[0] for e in EXPRESSIONS])
@pytest.mark.parametrize("xs", POINTS, ids=lambda p: ",".join(f"{v:g}" for v in p))
def test_forward_value_matches_reference(name, sym, ref, xs):
    """A self-consistently wrong forward gradchecks perfectly; check it too."""
    got = sym([Value(x) for x in xs]).data
    assert got == pytest.approx(ref(xs), rel=1e-12, abs=1e-12)


def test_backward_accumulates_across_repeated_use():
    """``x * x`` must give ``2x``, not ``x``. The ``grad =`` bug fails here."""
    x = Value(3.0)
    (x * x).backward()
    assert x.grad == pytest.approx(6.0)


def test_backward_resets_grads_between_calls():
    """A second backward on the same graph must not double the first result."""
    x = Value(1.7)
    y = x.exp() * 2.0
    y.backward()
    first = x.grad
    y.backward()
    assert x.grad == pytest.approx(first)


def test_power_rejects_value_exponent():
    with pytest.raises(TypeError):
        _ = Value(2.0) ** Value(3.0)


def test_log_rejects_non_positive():
    with pytest.raises(ValueError, match="non-positive"):
        Value(0.0).log()


# --------------------------------------------------------------------------- #
# the recursion contrast
# --------------------------------------------------------------------------- #
def test_recursive_topo_sort_raises_where_iterative_survives():
    """The headline claim of the example, asserted rather than narrated."""
    limit = sys.getrecursionlimit()
    # Three times the frame limit leaves no room for argument about how many
    # frames one level of `build` costs, or how deep pytest already is.
    deep = 3 * limit
    root = build_fan_in_sum(deep, np.random.default_rng(0))

    with pytest.raises(RecursionError):
        topological_order_recursive(root)

    order = topological_order_iterative(root)
    # Each of `deep` terms contributes a weight, an input, a product and a sum,
    # on top of the Value(0.0) the fold starts from.
    assert len(order) == 4 * deep + 1


def test_mnist_fan_in_is_within_a_factor_of_two_of_the_limit():
    """784 chained additions is one hidden unit, not a pathological graph."""
    survivable = find_recursive_topo_limit(rng=np.random.default_rng(0))
    assert survivable < 2 * 784, (
        f"recursive sort survives fan-in {survivable}; the 784-deep MNIST input "
        "layer is no longer a near-miss and the example's framing needs updating"
    )


def test_both_sorts_give_identical_gradients_where_both_survive():
    """Depth is the only difference between them; the answer must not change."""
    xs = [0.4, -1.1, 2.3]
    rec = [Value(x) for x in xs]
    ite = [Value(x) for x in xs]
    cross_entropy([rec[0] * rec[1], rec[2].exp(), rec[0] + rec[2]], 0).backward(
        topo_fn=topological_order_recursive
    )
    cross_entropy([ite[0] * ite[1], ite[2].exp(), ite[0] + ite[2]], 0).backward(
        topo_fn=topological_order_iterative
    )
    assert [n.grad for n in rec] == pytest.approx([n.grad for n in ite], rel=0, abs=0)


def test_topological_order_is_post_order():
    """Every node must appear after all of its parents, or backward is wrong."""
    a, b = Value(1.5), Value(-2.0)
    root = (a * b + a).tanh() * b
    for order in (topological_order_recursive(root), topological_order_iterative(root)):
        position = {id(n): i for i, n in enumerate(order)}
        for node in order:
            for child in node._prev:
                assert position[id(child)] < position[id(node)]
        assert order[-1] is root


# --------------------------------------------------------------------------- #
# the toy model
# --------------------------------------------------------------------------- #
def test_moons_are_balanced_and_reproducible():
    x1, y1 = make_moons(100, 0.1, np.random.default_rng(7))
    x2, y2 = make_moons(100, 0.1, np.random.default_rng(7))
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    assert x1.shape == (100, 2)
    assert y1.sum() == 50


def test_untrained_model_is_at_chance_level():
    """Guards the accuracy test below: 90% must mean learning, not a lucky init."""
    rng = np.random.default_rng(3)
    X, y = make_moons(100, 0.1, rng)
    model = MLP(2, (8,), 2, rng)
    assert accuracy(model, X, y) < 0.85


def test_toy_model_reaches_ninety_percent():
    """The shipped defaults, at the seed ``main()`` uses. ~20 s on a slow box."""
    _model, _X, _y, acc, history = train_toy_model(np.random.default_rng(0))
    assert acc > 0.90, f"accuracy {acc:.3f}"
    assert history[-1] < history[0]


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2])
def test_toy_model_is_not_seed_luck(seed):
    """One seed clearing 90% could be an accident; two more say it is not."""
    _model, _X, _y, acc, _history = train_toy_model(np.random.default_rng(seed))
    assert acc > 0.90, f"seed {seed}: accuracy {acc:.3f}"


# --------------------------------------------------------------------------- #
# the benchmark plumbing (the numbers are machine-dependent; the shapes are not)
# --------------------------------------------------------------------------- #
def test_node_counter_tracks_construction():
    before = Value.n_constructed
    _ = Value(1.0) * Value(2.0)  # two leaves plus the product
    assert Value.n_constructed - before == 3


def test_benchmark_reports_a_positive_rate():
    bench = benchmark_node_rate(np.random.default_rng(0), n_in=16, n_out=4, repeats=2)
    assert bench["forward_nodes"] > 0
    assert bench["forward_nodes_per_s"] > 0
    assert bench["train_nodes_per_s"] > 0


def test_mnist_node_count_matches_the_documented_model():
    counts = mnist_mlp_node_count(dims=(784, 128, 10), n_samples=60_000)
    # 128 * (2*784 + 1) hidden + 10 * (2*128) output + 3*10 + 2 for the loss.
    expected = 128 * (2 * 784 + 1) + 10 * (2 * 128) + 32
    assert counts["nodes_per_sample"] == expected
    assert counts["nodes_per_epoch"] == expected * 60_000
    assert extrapolate_mnist_epoch(241_000.0, counts["nodes_per_epoch"]) == pytest.approx(
        counts["nodes_per_epoch"] / 241_000.0
    )
