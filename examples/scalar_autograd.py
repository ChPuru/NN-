"""Tier 1a: a micrograd-style **scalar** autograd engine, built to be discarded.

This file deliberately does not import ``scratchgrad``. It is the control arm of
the experiment: the smallest thing that is honestly an autodiff engine, so that
the cost of the array-valued engine in ``scratchgrad.tensor`` can be quoted
against something real rather than against a feeling.

Three claims are made here, and all three are *executed* rather than asserted in
a comment:

1. **Four primitives are not enough.** micrograd defines ``+``, ``*``,
   ``** const`` and ``relu``, and derives ``-``, ``/`` and unary minus from
   them. That closure is the ring operations plus a fixed-exponent power; it
   contains no transcendental function, so ``exp``, ``log`` and ``tanh`` cannot
   be built from it and are added below as primitives. Without them there is no
   softmax and no cross-entropy, which is to say no classifier.
2. **The engine is slow in a quantifiable way.** :func:`benchmark_node_rate`
   measures node constructions per second *on this machine*; the published
   micrograd figure is ~241k/s. :func:`extrapolate_mnist_epoch` turns that into
   an MNIST-MLP epoch time. The measurement is measured; the epoch time is an
   extrapolation from a node count, and both are labelled as such in the output.
3. **CPython's recursion limit bites before the clock does.** micrograd's
   topological sort is recursive and recurses once per *graph level*. A single
   hidden unit reading 784 inputs folds its products with 784 chained additions,
   so the graph is 784 levels deep before anything interesting has happened.
   Both sorts are implemented below; :func:`demonstrate_recursion_limit` runs
   them side by side and reports the depth at which the recursive one dies.

Run it::

    PYTHONPATH=src python3 examples/scalar_autograd.py
"""

from __future__ import annotations

import itertools
import math
import sys
import time
from collections.abc import Callable, Sequence

import numpy as np

__all__ = [
    "MLP",
    "Layer",
    "Neuron",
    "Value",
    "accuracy",
    "benchmark_node_rate",
    "build_fan_in_sum",
    "cross_entropy",
    "demonstrate_recursion_limit",
    "extrapolate_mnist_epoch",
    "find_recursive_topo_limit",
    "logsumexp",
    "main",
    "make_moons",
    "mnist_mlp_node_count",
    "topological_order_iterative",
    "topological_order_recursive",
    "train_toy_model",
]


def _noop() -> None:
    """Backward for a leaf. A shared no-op function, not a fresh closure.

    Every ``Value`` would otherwise carry its own ``lambda``, which is the single
    largest per-node allocation in the engine and shows up directly in the
    node-construction benchmark below.
    """


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #
class Value:
    """A scalar node in a dynamically built expression graph.

    Attributes:
        data: The scalar value, always a Python ``float``.
        grad: Accumulated derivative of the scalar the backward pass was seeded
            from with respect to this node. Reset by hand, never automatically.
        _backward: Closure that pushes ``self.grad`` into ``self._prev``.
        _prev: The nodes this one was computed from.
        _op: Debug label for the operation that produced this node.
    """

    __slots__ = ("_backward", "_op", "_prev", "data", "grad")

    #: Every constructed node is counted, including the ones wrapping Python
    #: floats. This is the quantity the benchmark divides by elapsed time.
    n_constructed: int = 0

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = "") -> None:
        Value.n_constructed += 1
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = _noop
        self._prev = _children
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # -- the four micrograd primitives --------------------------------------
    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # ``+=``, never ``=``: ``x + x`` reaches this node twice and the two
            # contributions must add. Assignment here is the classic micrograd
            # transcription bug and it survives every test where each node is
            # used exactly once.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> Value:
        """``self ** other`` for a **constant** exponent only.

        A ``Value`` exponent would need ``log(self)``, which is undefined for
        ``self <= 0``; micrograd refuses the case rather than returning a
        gradient that is silently wrong on half the domain.
        """
        if not isinstance(other, (int, float)):
            raise TypeError("only int/float powers are supported; use exp/log for a Value exponent")
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # -- primitives micrograd does not have, and why -------------------------
    # The closure of {+, *, **const, relu} over composition is the rational
    # functions of the inputs (plus a piecewise-linear gate). exp, log and tanh
    # are transcendental: no finite composition of the four reaches them. They
    # are therefore *primitives*, each with its own hand-written derivative --
    # not conveniences. Softmax needs exp, cross-entropy needs log, and a
    # log-sum-exp that never exponentiates a raw logit needs both.
    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            # d/dx exp(x) = exp(x): reuse the forward value instead of calling
            # math.exp twice.
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        if self.data <= 0.0:
            raise ValueError(f"log() of a non-positive value ({self.data!r})")
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    # -- everything below is derived, not primitive --------------------------
    def __neg__(self) -> Value:
        return self * -1.0

    def __radd__(self, other: float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-(other if isinstance(other, Value) else Value(other)))

    def __rsub__(self, other: float) -> Value:
        return Value(other) + (-self)

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * (other if isinstance(other, Value) else Value(other)) ** -1.0

    def __rtruediv__(self, other: float) -> Value:
        return Value(other) * self**-1.0

    def sigmoid(self) -> Value:
        """Derived from ``tanh`` rather than from ``exp``.

        ``1/(1+exp(-x))`` written directly overflows for ``x <~ -709``;
        ``0.5*(tanh(x/2) + 1)`` is the same function and saturates instead.
        """
        return 0.5 * self.tanh_half() + 0.5

    def tanh_half(self) -> Value:
        return (self * 0.5).tanh()

    # -- backward ------------------------------------------------------------
    def backward(self, topo_fn: Callable[[Value], list[Value]] | None = None) -> None:
        """Seed this node with 1.0 and propagate to every node it depends on.

        Args:
            topo_fn: Topological sort to use. Defaults to the iterative one;
                pass :func:`topological_order_recursive` to reproduce
                micrograd's behaviour, stack limit included.
        """
        order = (topo_fn or topological_order_iterative)(self)
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()


# --------------------------------------------------------------------------- #
# the two topological sorts -- the whole point of the comparison
# --------------------------------------------------------------------------- #
def topological_order_recursive(root: Value) -> list[Value]:
    """micrograd's sort, transcribed faithfully: post-order DFS by recursion.

    One Python frame per *graph level*. A hidden unit with fan-in 784 is 784
    levels deep on its own, so this dies on the first layer of an MNIST MLP long
    before the wall clock becomes the complaint.
    """
    order: list[Value] = []
    visited: set[int] = set()

    def build(node: Value) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for child in node._prev:
            build(child)
        order.append(node)

    build(root)
    return order


def topological_order_iterative(root: Value) -> list[Value]:
    """The same post-order, on an explicit stack. Depth is bounded by RAM.

    The ``expanded`` flag is what makes this a *post*-order: a node is pushed
    twice, once to schedule its children and once to be emitted after them.
    Emitting on first pop instead gives a pre-order, which reverses to a
    perfectly plausible-looking but wrong backward pass.
    """
    order: list[Value] = []
    visited: set[int] = set()
    stack: list[tuple[Value, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for child in node._prev:
            if id(child) not in visited:
                stack.append((child, False))
    return order


# --------------------------------------------------------------------------- #
# losses that need the extra primitives
# --------------------------------------------------------------------------- #
def logsumexp(values: Sequence[Value]) -> Value:
    """``log(sum(exp(v)))``, max-shifted so no raw logit is ever exponentiated.

    The shift is subtracted as a ``Value``, not as a Python float, so its
    gradient path is part of the graph. It cancels exactly -- d/dm of
    ``m + log(sum(exp(v - m)))`` is zero -- but only if it is differentiated;
    freezing it as a constant gives the same forward value and the same
    gradient, while differentiating it *incorrectly* (a common shortcut is to
    detach only some occurrences) does not.
    """
    m = max(values, key=lambda v: v.data)
    total = Value(0.0)
    for v in values:
        total = total + (v - m).exp()
    return total.log() + m


def cross_entropy(logits: Sequence[Value], label: int) -> Value:
    """Softmax cross-entropy in log-sum-exp form: ``logsumexp(z) - z[label]``.

    Never ``-log(softmax(z)[label])``. That form exponentiates, normalises and
    then takes a log of a number that can already have flushed to zero, and the
    usual ``+1e-12`` guard hands a *zero* gradient to exactly the samples the
    model got most wrong.
    """
    return logsumexp(logits) - logits[label]


# --------------------------------------------------------------------------- #
# a tiny MLP
# --------------------------------------------------------------------------- #
class Neuron:
    """A single unit: ``relu(w . x + b)`` (or the raw affine output)."""

    def __init__(self, n_in: int, rng: np.random.Generator, nonlin: bool = True) -> None:
        # He: variance 2/fan_in, so std = sqrt(2/fan_in). The square root is the
        # single most-dropped detail in from-scratch init code.
        std = math.sqrt(2.0 / n_in)
        self.w = [Value(x) for x in rng.normal(0.0, std, size=n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: Sequence[Value]) -> Value:
        # This fold is what makes the graph deep: n_in chained additions, one
        # graph level each. It is also why the recursive sort cannot survive an
        # MNIST-width input layer.
        act = self.b
        for wi, xi in zip(self.w, x, strict=True):
            act = act + wi * xi
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    """A fully connected layer of :class:`Neuron`."""

    def __init__(
        self, n_in: int, n_out: int, rng: np.random.Generator, nonlin: bool = True
    ) -> None:
        self.neurons = [Neuron(n_in, rng, nonlin=nonlin) for _ in range(n_out)]

    def __call__(self, x: Sequence[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Stack of :class:`Layer`; the last layer is linear and emits raw logits."""

    def __init__(self, n_in: int, hidden: Sequence[int], n_out: int, rng: np.random.Generator):
        sizes = [n_in, *hidden, n_out]
        self.layers = [
            Layer(sizes[i], sizes[i + 1], rng, nonlin=(i < len(sizes) - 2))
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x: Sequence[Value]) -> list[Value]:
        out = list(x)
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


# --------------------------------------------------------------------------- #
# toy data
# --------------------------------------------------------------------------- #
def make_moons(
    n_samples: int, noise: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles: the smallest non-linearly-separable set.

    Args:
        n_samples: Total number of points.
        noise: Standard deviation of the isotropic Gaussian jitter.
        rng: Explicit generator. No global seeding anywhere in this file.

    Returns:
        ``(X, y)`` with ``X`` of shape ``(n_samples, 2)`` and ``y`` in ``{0, 1}``.
    """
    n_out = n_samples // 2
    n_in = n_samples - n_out
    t_out = math.pi * rng.uniform(0.0, 1.0, n_out)
    t_in = math.pi * rng.uniform(0.0, 1.0, n_in)
    outer = np.stack([np.cos(t_out), np.sin(t_out)], axis=1)
    inner = np.stack([1.0 - np.cos(t_in), 0.5 - np.sin(t_in)], axis=1)
    X = np.concatenate([outer, inner], axis=0) + rng.normal(0.0, noise, (n_samples, 2))
    y = np.concatenate([np.zeros(n_out, dtype=np.int64), np.ones(n_in, dtype=np.int64)])
    return X, y


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of rows whose argmax logit matches the label."""
    correct = 0
    for xi, yi in zip(X, y, strict=True):
        logits = model([Value(v) for v in xi])
        correct += int(max(range(len(logits)), key=lambda k: logits[k].data) == yi)
    return correct / len(y)


def train_toy_model(
    rng: np.random.Generator,
    n_samples: int = 100,
    hidden: Sequence[int] = (8,),
    steps: int = 200,
    lr: float = 1.0,
    noise: float = 0.10,
    verbose: bool = False,
) -> tuple[MLP, np.ndarray, np.ndarray, float, list[float]]:
    """Full-batch gradient descent on the moons problem.

    Returns:
        ``(model, X, y, final_accuracy, loss_history)``.
    """
    X, y = make_moons(n_samples, noise, rng)
    model = MLP(2, hidden, 2, rng)
    params = model.parameters()
    history: list[float] = []

    for step in range(steps):
        for p in params:
            p.grad = 0.0
        total = Value(0.0)
        for xi, yi in zip(X, y, strict=True):
            total = total + cross_entropy(model([Value(v) for v in xi]), int(yi))
        loss = total * (1.0 / len(y))
        loss.backward()
        # Cosine-ish decay: a fixed step overshoots once the moons are separated
        # and the accuracy oscillates around the target instead of settling.
        step_lr = lr * (0.5 * (1.0 + math.cos(math.pi * step / steps)) * 0.9 + 0.1)
        for p in params:
            p.data -= step_lr * p.grad
        history.append(loss.data)
        if verbose and (step % 40 == 0 or step == steps - 1):
            print(f"  step {step:4d}  loss {loss.data:.4f}")

    return model, X, y, accuracy(model, X, y), history


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #
def benchmark_node_rate(
    rng: np.random.Generator, n_in: int = 64, n_out: int = 32, repeats: int = 12
) -> dict[str, float]:
    """Measure node constructions per second, forward-only and forward+backward.

    The workload is one dense layer evaluated on one sample -- the same fold
    over ``n_in`` products that an MNIST hidden unit performs, at a width small
    enough to time repeatably.

    Args:
        rng: Generator for the layer's weights.
        n_in: Fan-in of the benchmark layer.
        n_out: Number of units.
        repeats: Forward (and forward+backward) passes to average over.

    Returns:
        Dict with ``forward_nodes_per_s``, ``train_nodes_per_s`` and the raw
        node counts and seconds behind them.
    """
    layer = Layer(n_in, n_out, rng)
    x = [Value(v) for v in rng.normal(size=n_in)]

    # Forward only.
    start_nodes, t0 = Value.n_constructed, time.perf_counter()
    for _ in range(repeats):
        outs = layer(x)
        total = outs[0]
        for o in outs[1:]:
            total = total + o
    fwd_seconds = time.perf_counter() - t0
    fwd_nodes = Value.n_constructed - start_nodes

    # Forward + backward: the rate that actually governs an epoch.
    start_nodes, t0 = Value.n_constructed, time.perf_counter()
    for _ in range(repeats):
        outs = layer(x)
        total = outs[0]
        for o in outs[1:]:
            total = total + o
        total.backward()
    train_seconds = time.perf_counter() - t0
    train_nodes = Value.n_constructed - start_nodes

    return {
        "forward_nodes": float(fwd_nodes),
        "forward_seconds": fwd_seconds,
        "forward_nodes_per_s": fwd_nodes / fwd_seconds,
        "train_nodes": float(train_nodes),
        "train_seconds": train_seconds,
        "train_nodes_per_s": train_nodes / train_seconds,
    }


def mnist_mlp_node_count(
    dims: Sequence[int] = (784, 128, 10), n_samples: int = 60_000
) -> dict[str, float]:
    """Count the ``Value`` objects one epoch of an MNIST MLP would construct.

    Accounting, per output unit of a layer with fan-in ``k``: ``k`` products,
    ``k`` additions (the bias seeds the fold, so there are ``k`` adds, not
    ``k-1``), and one ReLU node on hidden layers. The softmax cross-entropy over
    ``C`` classes adds roughly ``3C + 2`` nodes: a shift, an exp and a fold-add
    per class, plus the log and the label subtraction.

    Note:
        Coercion of Python floats to ``Value`` inside ``__add__`` is not counted;
        the fold above adds ``Value`` to ``Value``. The number is therefore a
        mild **under**-estimate.
    """
    per_sample = 0
    for i, (k, units) in enumerate(itertools.pairwise(dims)):
        hidden = i < len(dims) - 2
        per_sample += units * (2 * k + (1 if hidden else 0))
    per_sample += 3 * dims[-1] + 2
    return {
        "nodes_per_sample": float(per_sample),
        "nodes_per_epoch": float(per_sample) * n_samples,
        "n_samples": float(n_samples),
    }


def extrapolate_mnist_epoch(nodes_per_second: float, nodes_per_epoch: float) -> float:
    """Seconds for one epoch at a given node rate. Pure division -- an estimate."""
    return nodes_per_epoch / nodes_per_second


# --------------------------------------------------------------------------- #
# the recursion-limit demonstration
# --------------------------------------------------------------------------- #
def build_fan_in_sum(fan_in: int, rng: np.random.Generator | None = None) -> Value:
    """Build ``b + sum_i w_i * x_i`` as micrograd does: a chain of ``fan_in`` adds.

    The graph is ``fan_in`` levels deep. This is not a pathological construction
    -- it is one hidden unit of the first layer of an MNIST MLP.
    """
    rng = rng or np.random.default_rng(0)
    w = rng.normal(size=fan_in)
    x = rng.normal(size=fan_in)
    act = Value(0.0)
    for wi, xi in zip(w, x, strict=True):
        act = act + Value(wi) * Value(xi)
    return act


def find_recursive_topo_limit(
    lo: int = 1, hi: int = 4096, rng: np.random.Generator | None = None
) -> int:
    """Largest fan-in whose graph the *recursive* sort can still walk, right now.

    The answer depends on how many frames the caller has already used, so it is
    a property of the moment, not a constant. That is precisely the problem with
    a recursive sort: the ceiling moves under you.
    """
    rng = rng or np.random.default_rng(0)

    def survives(depth: int) -> bool:
        root = build_fan_in_sum(depth, rng)
        try:
            topological_order_recursive(root)
        except RecursionError:
            return False
        return True

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if survives(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def demonstrate_recursion_limit(
    fan_in: int, rng: np.random.Generator | None = None
) -> dict[str, object]:
    """Run both sorts on one ``fan_in``-deep graph and report what each did.

    Returns:
        Dict with ``fan_in``, ``recursive_error`` (the exception type name or
        ``None``) and ``iterative_nodes`` (how many nodes the iterative sort
        walked, or ``None`` if it too failed).
    """
    root = build_fan_in_sum(fan_in, rng)

    recursive_error: str | None = None
    try:
        topological_order_recursive(root)
    except RecursionError as exc:
        recursive_error = type(exc).__name__

    iterative_nodes: int | None
    try:
        iterative_nodes = len(topological_order_iterative(root))
    except RecursionError:
        iterative_nodes = None

    return {
        "fan_in": fan_in,
        "recursive_error": recursive_error,
        "iterative_nodes": iterative_nodes,
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _rule(title: str = "") -> None:
    print(f"\n{title}\n" + "-" * 78 if title else "-" * 78)


def main(seed: int = 0) -> None:
    """Train the toy model, then print the benchmark that justifies Tier 1b."""
    rng = np.random.default_rng(seed)

    _rule("toy problem: two moons, 100 points, MLP 2-8-2, 200 steps of full-batch GD")
    t0 = time.perf_counter()
    _model, _X, _y, acc, history = train_toy_model(rng, verbose=True)
    train_seconds = time.perf_counter() - t0
    print(
        f"  final loss {history[-1]:.4f}   train accuracy {acc:.1%}   "
        f"({train_seconds:.2f} s, {Value.n_constructed:,} nodes built so far)"
    )

    _rule("MEASURED on this machine")
    bench = benchmark_node_rate(rng)
    print(
        f"  forward only        {bench['forward_nodes_per_s']:>12,.0f} node constructions/s"
        f"   ({bench['forward_nodes']:,.0f} nodes / {bench['forward_seconds']:.3f} s)"
    )
    print(
        f"  forward + backward  {bench['train_nodes_per_s']:>12,.0f} node constructions/s"
        f"   ({bench['train_nodes']:,.0f} nodes / {bench['train_seconds']:.3f} s)"
    )
    print("  reference: the published micrograd figure is ~241,000 node constructions/s")

    _rule("EXTRAPOLATED: one epoch of an MNIST MLP 784-128-10 over 60,000 images")
    counts = mnist_mlp_node_count()
    print(
        f"  node count model    {counts['nodes_per_sample']:>12,.0f} nodes/sample"
        f"   -> {counts['nodes_per_epoch']:,.0f} nodes/epoch"
    )
    for label, rate in (
        ("at this machine's fwd+bwd rate", bench["train_nodes_per_s"]),
        ("at the published 241k/s", 241_000.0),
    ):
        secs = extrapolate_mnist_epoch(rate, counts["nodes_per_epoch"])
        print(f"  {label:<32} {secs / 3600:>8.1f} hours  ({secs:,.0f} s)")
    print("  (measured = the rates above; extrapolated = these epoch times. No epoch was run.)")

    _rule("MEASURED: where the recursive topological sort dies")
    print(f"  sys.getrecursionlimit() = {sys.getrecursionlimit()}")
    limit = find_recursive_topo_limit()
    print(f"  deepest graph the recursive sort survives from here: fan-in {limit:,}")
    for fan_in in (784, max(2 * limit, 2000), 50_000):
        r = demonstrate_recursion_limit(fan_in)
        rec = r["recursive_error"] or "ok"
        it = "ok" if r["iterative_nodes"] is not None else "RecursionError"
        note = "  <- MNIST input layer, one hidden unit" if fan_in == 784 else ""
        print(f"  fan-in {fan_in:>6,}:  recursive {rec:<15} iterative {it}{note}")
    print("  The iterative sort is not faster. It is the difference between a program that")
    print("  runs and one that raises before the first weight update -- which is why")
    print("  scratchgrad.tensor._topological_order uses an explicit stack.")
    _rule()
    print(f"  total Value objects constructed by this script: {Value.n_constructed:,}")


if __name__ == "__main__":
    main()
