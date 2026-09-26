"""Reverse-mode automatic differentiation over ``numpy`` arrays.

This is the engine the rest of ``scratchgrad`` is built on. NumPy supplies
array storage and BLAS-backed matmul; **every vector-Jacobian product in this
file is written by hand.** Nothing here imports an autodiff library.

Design notes
------------
* A :class:`Tensor` node holds its value (``.data``), its gradient (``.grad``),
  the tensors it was computed from (``._prev``) and a closure (``._backward``)
  that pushes an incoming cotangent into those parents.
* :meth:`Tensor.backward` builds the topological order with an **explicit
  stack**, not recursion. A recursive topological sort recurses once per graph
  level and CPython's default limit of 1000 frames is reached by ordinary
  unrolled RNNs; the iterative version has no such ceiling.
* Every elementwise backward routes through :func:`unbroadcast`. Broadcasting
  in the forward pass is a *sum* in the backward pass; forgetting it produces a
  shape error at best and a silently wrong gradient at worst.
* Leaves **accumulate** into ``.grad`` (``+=``, never ``=``) so gradients from
  several backward calls add up, exactly as PyTorch does. Non-leaf gradients
  are reset at the start of each backward so a second call cannot read a stale
  intermediate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from ._config import get_default_dtype, is_grad_enabled

__all__ = [
    "Tensor",
    "arange",
    "concatenate",
    "eye",
    "from_numpy",
    "full",
    "ones",
    "stack",
    "tensor",
    "unbroadcast",
    "where",
    "zeros",
]

ArrayLike = Any


# --------------------------------------------------------------------------- #
# broadcasting
# --------------------------------------------------------------------------- #
def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` back to ``shape``, undoing NumPy broadcasting.

    Two distinct things have to be undone, in this order:

    1. axes that broadcasting *prepended* (``(3,) -> (2, 3)``): sum them away
       entirely, dropping the axis;
    2. axes that existed but had extent 1 (``(1, 3) -> (2, 3)``): sum with
       ``keepdims=True`` so the extent-1 axis survives.

    Doing (2) before (1), or using ``keepdims=False`` in (2), yields an array of
    the wrong rank and the bug surfaces far from its cause.
    """
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    if len(shape):
        axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
        if axes:
            grad = grad.sum(axis=axes, keepdims=True)
    if grad.shape != shape:
        grad = grad.reshape(shape)
    return grad


def _topological_order(root: Tensor) -> list[Tensor]:
    """Post-order DFS with an explicit stack: inputs first, ``root`` last."""
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for parent in node._prev:
            if id(parent) not in visited:
                stack.append((parent, False))
    return order


def _is_basic_index(key) -> bool:
    """True when ``key`` is pure basic indexing, which never repeats an element."""
    items = key if isinstance(key, tuple) else (key,)
    return all(
        k is None
        or k is Ellipsis
        or isinstance(k, slice)
        or (isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)))
        for k in items
    )


def _norm_axes(axis, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, (int, np.integer)):
        axis = (int(axis),)
    return tuple(sorted(a % ndim for a in axis))


# --------------------------------------------------------------------------- #
# Tensor
# --------------------------------------------------------------------------- #
class Tensor:
    """An n-dimensional array that records the operations applied to it."""

    __slots__ = ("_backward", "_name", "_op", "_prev", "data", "grad", "requires_grad")

    # NumPy defers to us for ``ndarray + Tensor`` instead of building an object array.
    __array_priority__ = 100.0

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        dtype: Any | None = None,
        name: str | None = None,
    ) -> None:
        arr = data.data if isinstance(data, Tensor) else np.asarray(data)
        if dtype is not None:
            arr = arr.astype(np.dtype(dtype), copy=False)
        elif arr.dtype.kind == "f" or arr.dtype.kind not in "bui":
            arr = arr.astype(get_default_dtype(), copy=False)

        if requires_grad and arr.dtype.kind != "f":
            raise TypeError(
                f"only floating-point tensors can require gradients (got {arr.dtype}); "
                "integer tensors are for indices and labels"
            )

        self.data: np.ndarray = arr
        self.grad: np.ndarray | None = None
        self.requires_grad: bool = bool(requires_grad)
        self._backward: Callable[[np.ndarray], None] | None = None
        self._prev: tuple[Tensor, ...] = ()
        self._op: str = ""
        self._name: str | None = name

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def _wrap(data: np.ndarray, requires_grad: bool = False) -> Tensor:
        """Wrap an array whose dtype is already correct, with no coercion.

        The public constructor casts float input to the *global default* dtype,
        which is what you want when a user writes ``Tensor([1.0, 2.0])``. It is
        emphatically not what you want for the result of an operation: coercing
        there would silently downcast a float64 computation to float32 the moment
        it left a ``default_dtype`` block, so a model cast with ``.double()``
        would quietly run in single precision and every gradcheck outside that
        block would be meaningless.
        """
        out = Tensor.__new__(Tensor)
        out.data = data
        out.grad = None
        out.requires_grad = bool(requires_grad)
        out._backward = None
        out._prev = ()
        out._op = ""
        out._name = None
        return out

    @staticmethod
    def _make(
        data: np.ndarray,
        parents: Sequence[Tensor],
        grad_fns: Sequence[Callable[[np.ndarray], np.ndarray] | None],
        op: str = "",
    ) -> Tensor:
        """Build an output node wired to ``parents``.

        ``grad_fns[i]`` maps the cotangent of the output to the contribution
        made to ``parents[i]``. It is only ever called when that parent actually
        requires a gradient, so a rule that would be expensive or undefined for
        a constant input is never evaluated.
        """
        track = is_grad_enabled() and any(p.requires_grad for p in parents)
        out = Tensor._wrap(np.asarray(data), requires_grad=track)
        if track:
            out._prev = tuple(parents)
            out._op = op

            def _bw(g: np.ndarray, _p=tuple(parents), _f=tuple(grad_fns)) -> None:
                for parent, fn in zip(_p, _f, strict=True):
                    if parent.requires_grad and fn is not None:
                        parent._accumulate(fn(g))

            out._backward = _bw
        return out

    def _accumulate(self, g: np.ndarray) -> None:
        """Add ``g`` into ``.grad``, unbroadcasting and casting as needed."""
        if not self.requires_grad:
            return
        g = np.asarray(g)
        if g.shape != self.data.shape:
            g = unbroadcast(g, self.data.shape)
        if g.dtype != self.data.dtype:
            g = g.astype(self.data.dtype, copy=False)
        if self.grad is None:
            # Copy: the caller may hand us a buffer it intends to reuse, and a
            # later ``+=`` here would corrupt it.
            self.grad = np.array(g, dtype=self.data.dtype, copy=True)
        else:
            self.grad += g

    # -- properties ----------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(self.data.size)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def is_leaf(self) -> bool:
        """True when this tensor was not produced by a tracked operation."""
        return self._backward is None

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        for i in range(len(self.data)):
            yield self[i]

    def __repr__(self) -> str:
        head = f"Tensor(shape={self.shape}, dtype={self.dtype.name}"
        if self.requires_grad:
            head += ", requires_grad=True"
        if self._op:
            head += f", op={self._op!r}"
        body = np.array2string(self.data, precision=4, threshold=32, edgeitems=2)
        return f"{head})\n{body}"

    def __array__(self, dtype=None, copy=None):  # numpy interop
        arr = self.data
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return np.array(arr, copy=True) if copy else arr

    def __hash__(self) -> int:
        return id(self)

    def __bool__(self) -> bool:
        if self.data.size != 1:
            raise RuntimeError(
                "truth value of a multi-element Tensor is ambiguous; use .any()/.all() on .data"
            )
        return bool(self.data.reshape(()))

    def item(self) -> float:
        if self.data.size != 1:
            raise ValueError(f"item() expects exactly one element, got {self.data.size}")
        return self.data.reshape(()).item()

    def numpy(self) -> np.ndarray:
        return self.data

    def tolist(self):
        return self.data.tolist()

    # -- graph ---------------------------------------------------------------
    def backward(self, gradient: ArrayLike | None = None, retain_graph: bool = True) -> None:
        """Accumulate d(self)/d(leaf) into every leaf reachable from here.

        ``gradient`` may be omitted only for a scalar output, where it defaults
        to 1. The graph is never freed (``retain_graph`` is accepted for API
        parity and always honoured), so calling ``backward`` twice is legal --
        leaf gradients accumulate, which is what micro-batch accumulation needs.
        """
        if not self.requires_grad:
            raise RuntimeError(
                "backward() called on a tensor that does not require grad "
                "(no parameter in its history had requires_grad=True)"
            )
        order = _topological_order(self)

        # Reset intermediates. Leaves accumulate across calls; intermediates
        # must not, or a second backward would double-count them.
        for node in order:
            if not (node.is_leaf and node.requires_grad):
                node.grad = None

        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs; this tensor "
                    f"has shape {self.shape}. Pass an explicit cotangent."
                )
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(gradient, dtype=self.data.dtype)
            if seed.shape != self.data.shape:
                raise ValueError(
                    f"gradient shape {seed.shape} does not match tensor shape {self.shape}"
                )
        if self.is_leaf:
            self._accumulate(seed)
        else:
            self.grad = np.array(seed, dtype=self.data.dtype, copy=True)

        for node in reversed(order):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> Tensor:
        """A view of the same buffer with no history. Does not copy data.

        Preserves dtype: detaching a float64 tensor must not hand back a float32
        one just because that is the global default.
        """
        return Tensor._wrap(self.data, requires_grad=False)

    def requires_grad_(self, mode: bool = True) -> Tensor:
        if mode and self.data.dtype.kind != "f":
            raise TypeError(f"cannot require grad on a {self.data.dtype} tensor")
        if mode and not self.is_leaf:
            raise RuntimeError("only leaf tensors can have requires_grad set")
        self.requires_grad = bool(mode)
        return self

    def clone(self) -> Tensor:
        return Tensor._make(self.data.copy(), (self,), (lambda g: g,), "clone")

    def contiguous(self) -> Tensor:
        return self if self.data.flags["C_CONTIGUOUS"] else self.clone()

    def astype(self, dtype) -> Tensor:
        dt = np.dtype(dtype)
        if dt == self.data.dtype:
            return self
        if dt.kind != "f":
            return Tensor(self.data.astype(dt))
        src = self.data.dtype
        return Tensor._make(
            self.data.astype(dt), (self,), (lambda g, s=src: g.astype(s, copy=False),), "astype"
        )

    def float(self) -> Tensor:
        return self.astype(np.float32)

    def double(self) -> Tensor:
        return self.astype(np.float64)

    # -- elementwise binary --------------------------------------------------
    def _coerce(self, other: ArrayLike) -> Tensor:
        if isinstance(other, Tensor):
            return other
        return Tensor(np.asarray(other, dtype=self.data.dtype))

    def __add__(self, other):
        o = self._coerce(other)
        return Tensor._make(self.data + o.data, (self, o), (lambda g: g, lambda g: g), "add")

    __radd__ = __add__

    def __neg__(self):
        return Tensor._make(-self.data, (self,), (lambda g: -g,), "neg")

    def __sub__(self, other):
        o = self._coerce(other)
        return Tensor._make(self.data - o.data, (self, o), (lambda g: g, lambda g: -g), "sub")

    def __rsub__(self, other):
        return self._coerce(other).__sub__(self)

    def __mul__(self, other):
        o = self._coerce(other)
        a, b = self.data, o.data
        return Tensor._make(a * b, (self, o), (lambda g: g * b, lambda g: g * a), "mul")

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._coerce(other)
        a, b = self.data, o.data
        return Tensor._make(a / b, (self, o), (lambda g: g / b, lambda g: -g * a / (b * b)), "div")

    def __rtruediv__(self, other):
        return self._coerce(other).__truediv__(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            a, b = self.data, other.data
            out = a**b
            # log() is only defined for a > 0; the exponent gradient is only
            # requested when `other` requires grad, and a <= 0 there is a user error.
            safe_log = np.log(np.where(a > 0, a, 1.0))
            return Tensor._make(
                out,
                (self, other),
                (lambda g: g * b * a ** (b - 1), lambda g: g * out * safe_log),
                "pow",
            )
        p = other
        a = self.data
        return Tensor._make(a**p, (self,), (lambda g: g * p * a ** (p - 1),), "pow")

    def __rpow__(self, other):
        return self._coerce(other).__pow__(self)

    def __matmul__(self, other):
        return _matmul(self, self._coerce(other))

    def __rmatmul__(self, other):
        return _matmul(self._coerce(other), self)

    def maximum(self, other):
        o = self._coerce(other)
        a, b = self.data, o.data
        # Split a tie evenly rather than handing the full gradient to both sides;
        # otherwise a == b doubles the gradient that flows back.
        tie = a == b
        ga = np.where(a > b, 1.0, 0.0) + 0.5 * tie
        gb = np.where(b > a, 1.0, 0.0) + 0.5 * tie
        return Tensor._make(
            np.maximum(a, b), (self, o), (lambda g: g * ga, lambda g: g * gb), "maximum"
        )

    def minimum(self, other):
        o = self._coerce(other)
        a, b = self.data, o.data
        tie = a == b
        ga = np.where(a < b, 1.0, 0.0) + 0.5 * tie
        gb = np.where(b < a, 1.0, 0.0) + 0.5 * tie
        return Tensor._make(
            np.minimum(a, b), (self, o), (lambda g: g * ga, lambda g: g * gb), "minimum"
        )

    # -- comparisons return plain arrays: they are not differentiable --------
    def __gt__(self, other):
        return self.data > (other.data if isinstance(other, Tensor) else other)

    def __ge__(self, other):
        return self.data >= (other.data if isinstance(other, Tensor) else other)

    def __lt__(self, other):
        return self.data < (other.data if isinstance(other, Tensor) else other)

    def __le__(self, other):
        return self.data <= (other.data if isinstance(other, Tensor) else other)

    def __eq__(self, other):  # type: ignore[override]
        return self.data == (other.data if isinstance(other, Tensor) else other)

    def __ne__(self, other):  # type: ignore[override]
        return self.data != (other.data if isinstance(other, Tensor) else other)

    # -- unary ---------------------------------------------------------------
    def exp(self):
        out = np.exp(self.data)
        return Tensor._make(out, (self,), (lambda g: g * out,), "exp")

    def log(self):
        a = self.data
        return Tensor._make(np.log(a), (self,), (lambda g: g / a,), "log")

    def sqrt(self):
        out = np.sqrt(self.data)
        return Tensor._make(out, (self,), (lambda g: g * 0.5 / out,), "sqrt")

    def rsqrt(self):
        out = 1.0 / np.sqrt(self.data)
        return Tensor._make(out, (self,), (lambda g: g * (-0.5) * out**3,), "rsqrt")

    def abs(self):
        a = self.data
        return Tensor._make(np.abs(a), (self,), (lambda g: g * np.sign(a),), "abs")

    def sin(self):
        a = self.data
        return Tensor._make(np.sin(a), (self,), (lambda g: g * np.cos(a),), "sin")

    def cos(self):
        a = self.data
        return Tensor._make(np.cos(a), (self,), (lambda g: -g * np.sin(a),), "cos")

    def tanh(self):
        out = np.tanh(self.data)
        return Tensor._make(out, (self,), (lambda g: g * (1.0 - out * out),), "tanh")

    def sigmoid(self):
        # Branch on sign: exp() of a large positive number overflows, exp() of a
        # large negative one underflows harmlessly. Computing each half from the
        # stable side is exact across the whole real line.
        x = self.data
        z = np.exp(-np.abs(x))
        out = np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
        return Tensor._make(out, (self,), (lambda g: g * out * (1.0 - out),), "sigmoid")

    def relu(self):
        mask = self.data > 0
        return Tensor._make(self.data * mask, (self,), (lambda g: g * mask,), "relu")

    def clip(self, lo=None, hi=None):
        a = self.data
        out = np.clip(a, lo, hi)
        inside = np.ones(a.shape, dtype=bool)
        if lo is not None:
            inside &= a >= lo
        if hi is not None:
            inside &= a <= hi
        return Tensor._make(out, (self,), (lambda g: g * inside,), "clip")

    # -- reductions ----------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False):
        a = self.data
        out = a.sum(axis=axis, keepdims=keepdims)

        def bw(g, axis=axis, keepdims=keepdims, shape=a.shape):
            g = np.asarray(g)
            if axis is not None and not keepdims:
                g = np.expand_dims(g, _norm_axes(axis, len(shape)))
            return np.broadcast_to(g, shape).copy()

        return Tensor._make(out, (self,), (bw,), "sum")

    def mean(self, axis=None, keepdims: bool = False):
        a = self.data
        out = a.mean(axis=axis, keepdims=keepdims)
        n = a.size if axis is None else int(np.prod([a.shape[i] for i in _norm_axes(axis, a.ndim)]))

        def bw(g, axis=axis, keepdims=keepdims, shape=a.shape, n=n):
            g = np.asarray(g)
            if axis is not None and not keepdims:
                g = np.expand_dims(g, _norm_axes(axis, len(shape)))
            return np.broadcast_to(g / n, shape).copy()

        return Tensor._make(out, (self,), (bw,), "mean")

    def var(self, axis=None, keepdims: bool = False, ddof: int = 0):
        a = self.data
        mu = a.mean(axis=axis, keepdims=True)
        d = a - mu
        n = a.size if axis is None else int(np.prod([a.shape[i] for i in _norm_axes(axis, a.ndim)]))
        denom = n - ddof
        out = (d * d).sum(axis=axis, keepdims=keepdims) / denom

        def bw(g, axis=axis, keepdims=keepdims, shape=a.shape, d=d, denom=denom):
            g = np.asarray(g)
            if axis is not None and not keepdims:
                g = np.expand_dims(g, _norm_axes(axis, len(shape)))
            return (2.0 / denom) * g * d

        return Tensor._make(out, (self,), (bw,), "var")

    def std(self, axis=None, keepdims: bool = False, ddof: int = 0):
        return self.var(axis=axis, keepdims=keepdims, ddof=ddof).sqrt()

    def max(self, axis=None, keepdims: bool = False):
        return self._extremum(np.max, axis, keepdims, "max")

    def min(self, axis=None, keepdims: bool = False):
        return self._extremum(np.min, axis, keepdims, "min")

    def _extremum(self, fn, axis, keepdims, op):
        a = self.data
        kept = fn(a, axis=axis, keepdims=True)
        out = kept if keepdims else fn(a, axis=axis, keepdims=False)
        mask = (a == kept).astype(a.dtype)
        # Ties share the gradient. Handing the full upstream gradient to every
        # tied element multiplies it by the tie count -- the classic max-pool bug.
        share = mask / mask.sum(axis=axis, keepdims=True)

        def bw(g, axis=axis, keepdims=keepdims, shape=a.shape, share=share):
            g = np.asarray(g)
            if axis is not None and not keepdims:
                g = np.expand_dims(g, _norm_axes(axis, len(shape)))
            return g * share

        return Tensor._make(out, (self,), (bw,), op)

    def argmax(self, axis=None):
        return np.argmax(self.data, axis=axis)

    def argmin(self, axis=None):
        return np.argmin(self.data, axis=axis)

    def logsumexp(self, axis=None, keepdims: bool = False):
        """``log(sum(exp(x)))`` without ever exponentiating a raw logit."""
        a = self.data
        m = np.max(a, axis=axis, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        e = np.exp(a - m)
        s = e.sum(axis=axis, keepdims=True)
        out_k = np.log(s) + m
        if keepdims:
            out = out_k
        elif axis is None:
            out = out_k.reshape(())
        else:
            out = np.squeeze(out_k, axis=_norm_axes(axis, a.ndim))
        soft = e / s  # == softmax(a) along `axis`

        def bw(g, axis=axis, keepdims=keepdims, shape=a.shape, soft=soft):
            g = np.asarray(g)
            if axis is not None and not keepdims:
                g = np.expand_dims(g, _norm_axes(axis, len(shape)))
            return g * soft

        return Tensor._make(out, (self,), (bw,), "logsumexp")

    def softmax(self, axis: int = -1):
        return (self - self.logsumexp(axis=axis, keepdims=True)).exp()

    def log_softmax(self, axis: int = -1):
        return self - self.logsumexp(axis=axis, keepdims=True)

    # -- shape ---------------------------------------------------------------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        old = self.data.shape
        return Tensor._make(
            self.data.reshape(shape), (self,), (lambda g, old=old: g.reshape(old),), "reshape"
        )

    def view(self, *shape):
        return self.reshape(*shape)

    def flatten(self, start_dim: int = 1):
        shape = self.data.shape
        new = shape[:start_dim] + (int(np.prod(shape[start_dim:])),)
        return self.reshape(new)

    def ravel(self):
        return self.reshape((-1,))

    def transpose(self, *axes):
        if not axes:
            perm = tuple(reversed(range(self.data.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(axes[0])
        else:
            perm = tuple(axes)
        inv = tuple(int(i) for i in np.argsort(perm))
        return Tensor._make(
            self.data.transpose(perm), (self,), (lambda g, inv=inv: g.transpose(inv),), "transpose"
        )

    def permute(self, *axes):
        return self.transpose(*axes)

    def swapaxes(self, a: int, b: int):
        perm = list(range(self.data.ndim))
        perm[a], perm[b] = perm[b], perm[a]
        return self.transpose(tuple(perm))

    def squeeze(self, axis=None):
        old = self.data.shape
        return Tensor._make(
            np.squeeze(self.data, axis=axis),
            (self,),
            (lambda g, old=old: g.reshape(old),),
            "squeeze",
        )

    def unsqueeze(self, axis: int):
        old = self.data.shape
        return Tensor._make(
            np.expand_dims(self.data, axis),
            (self,),
            (lambda g, old=old: g.reshape(old),),
            "unsqueeze",
        )

    def broadcast_to(self, shape):
        old = self.data.shape
        return Tensor._make(
            np.broadcast_to(self.data, tuple(shape)).copy(),
            (self,),
            (lambda g, old=old: unbroadcast(g, old),),
            "broadcast_to",
        )

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        cur = self.data.shape
        pad = len(shape) - len(cur)
        target = tuple((cur[i - pad] if (i >= pad and s == -1) else s) for i, s in enumerate(shape))
        return self.broadcast_to(target)

    def repeat(self, repeats: int, axis: int | None = None):
        old = self.data.shape

        def bw(g, old=old, repeats=repeats, axis=axis):
            g = np.asarray(g)
            if axis is None:
                return g.reshape(-1, repeats).sum(axis=1).reshape(old)
            idx = np.repeat(np.arange(old[axis]), repeats)
            acc = np.zeros(old, dtype=g.dtype)
            np.add.at(acc, (slice(None),) * (axis % len(old)) + (idx,), g)
            return acc

        return Tensor._make(np.repeat(self.data, repeats, axis=axis), (self,), (bw,), "repeat")

    def __getitem__(self, idx):
        key = idx.data if isinstance(idx, Tensor) else idx
        if isinstance(key, tuple):
            key = tuple(k.data if isinstance(k, Tensor) else k for k in key)
        shape = self.data.shape
        dtype = self.data.dtype

        basic = _is_basic_index(key)

        def bw(g, key=key, shape=shape, dtype=dtype, basic=basic):
            acc = np.zeros(shape, dtype=dtype)
            if basic:
                # Ints, slices, None and Ellipsis can never select an element
                # twice, so a plain write is exact -- and far faster than add.at.
                acc[key] = g
            else:
                # add.at, not acc[key] = g: a repeated index must accumulate, and
                # plain assignment would keep only the last write.
                np.add.at(acc, key, g)
            return acc

        return Tensor._make(self.data[key], (self,), (bw,), "getitem")

    def masked_fill(self, mask, value: float):
        m = mask.data if isinstance(mask, Tensor) else np.asarray(mask)
        out = np.where(m, np.asarray(value, dtype=self.data.dtype), self.data)
        keep = (~m).astype(self.data.dtype)
        return Tensor._make(out, (self,), (lambda g: g * keep,), "masked_fill")

    def pad(self, pad_width, value: float = 0.0):
        pw = tuple((int(a), int(b)) for a, b in pad_width)
        sl = tuple(slice(a, a + n) for (a, _), n in zip(pw, self.data.shape, strict=True))
        return Tensor._make(
            np.pad(self.data, pw, mode="constant", constant_values=value),
            (self,),
            (lambda g, sl=sl: g[sl],),
            "pad",
        )

    # -- convenience ---------------------------------------------------------
    def dot(self, other):
        return self @ other

    def norm(self, p: float = 2.0):
        return (self.abs() ** p).sum() ** (1.0 / p)


def _matmul(a: Tensor, b: Tensor) -> Tensor:
    """Batched matmul with 1-D promotion, unbroadcasting both operands.

    NumPy promotes a 1-D operand to a matrix for the duration of the product and
    then drops the axis it added. The backward pass has to reinsert those axes,
    and it has to do it in the *reverse* of the order they were removed --
    ``sorted([-2, -1])`` is exactly the wrong order and crashes on the 1-D case.
    """
    ad, bd = a.data, b.data
    if ad.ndim > 2 and bd.ndim == 2:
        return _matmul_stacked(a, b)
    a_1d, b_1d = ad.ndim == 1, bd.ndim == 1
    A = ad[None, :] if a_1d else ad
    B = bd[:, None] if b_1d else bd
    out = A @ B
    if a_1d and b_1d:
        out_data = out.reshape(())
    elif a_1d:
        out_data = out.reshape(out.shape[:-2] + out.shape[-1:])
    elif b_1d:
        out_data = out.reshape(out.shape[:-2] + out.shape[-2:-1])
    else:
        out_data = out

    def _lift(g: np.ndarray) -> np.ndarray:
        g = np.asarray(g)
        if a_1d and b_1d:
            return g.reshape((1, 1))
        if a_1d:
            return g[..., None, :]
        if b_1d:
            return g[..., :, None]
        return g

    def ga(g):
        G = _lift(g)
        return unbroadcast(G @ np.swapaxes(B, -1, -2), A.shape).reshape(ad.shape)

    def gb(g):
        G = _lift(g)
        return unbroadcast(np.swapaxes(A, -1, -2) @ G, B.shape).reshape(bd.shape)

    return Tensor._make(out_data, (a, b), (ga, gb), "matmul")


def _matmul_stacked(a: Tensor, b: Tensor) -> Tensor:
    """``(..., M, K) @ (K, N)`` as ONE 2-D GEMM -- the shape of every layer applied over time.

    NumPy's broadcasting matmul runs this as a loop of small per-batch products,
    and the generic ``dB = Aᵀ @ G`` materialises one ``(K, N)`` gradient per batch
    element before ``unbroadcast`` sums them. On ``(32, 64, 128) @ (128, 512)``
    that was ~130x slower than flattening the leading axes, which is exact: the
    rows of ``A`` are independent, so stacking them changes nothing but speed.
    """
    ad, bd = a.data, b.data
    k, n = bd.shape
    a2 = ad.reshape(-1, k)
    out = (a2 @ bd).reshape(ad.shape[:-1] + (n,))

    def ga(g):
        return (np.asarray(g).reshape(-1, n) @ bd.T).reshape(ad.shape)

    def gb(g):
        return a2.T @ np.asarray(g).reshape(-1, n)

    return Tensor._make(out, (a, b), (ga, gb), "matmul")


# --------------------------------------------------------------------------- #
# factories and free functions
# --------------------------------------------------------------------------- #
def tensor(data, requires_grad: bool = False, dtype=None) -> Tensor:
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def from_numpy(arr: np.ndarray, requires_grad: bool = False) -> Tensor:
    return Tensor(arr, requires_grad=requires_grad)


def _shape_args(shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return tuple(shape)


def zeros(*shape, requires_grad: bool = False, dtype=None) -> Tensor:
    return Tensor(
        np.zeros(_shape_args(shape), dtype=dtype or get_default_dtype()),
        requires_grad=requires_grad,
    )


def ones(*shape, requires_grad: bool = False, dtype=None) -> Tensor:
    return Tensor(
        np.ones(_shape_args(shape), dtype=dtype or get_default_dtype()), requires_grad=requires_grad
    )


def full(shape, value, requires_grad: bool = False, dtype=None) -> Tensor:
    return Tensor(
        np.full(tuple(shape), value, dtype=dtype or get_default_dtype()),
        requires_grad=requires_grad,
    )


def arange(*args, dtype=None) -> Tensor:
    return Tensor(np.arange(*args, dtype=dtype or get_default_dtype()))


def eye(n: int, m: int | None = None, dtype=None) -> Tensor:
    return Tensor(np.eye(n, m, dtype=dtype or get_default_dtype()))


def concatenate(tensors: Iterable[Tensor], axis: int = 0) -> Tensor:
    ts = list(tensors)
    sizes = [t.data.shape[axis] for t in ts]
    offs = np.cumsum([0] + sizes)
    out = np.concatenate([t.data for t in ts], axis=axis)

    def make_bw(i):
        def bw(g, i=i):
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(int(offs[i]), int(offs[i + 1]))
            return g[tuple(sl)]

        return bw

    return Tensor._make(out, ts, [make_bw(i) for i in range(len(ts))], "concat")


def stack(tensors: Iterable[Tensor], axis: int = 0) -> Tensor:
    ts = list(tensors)
    out = np.stack([t.data for t in ts], axis=axis)

    def make_bw(i):
        def bw(g, i=i):
            sl = [slice(None)] * g.ndim
            sl[axis] = i
            return g[tuple(sl)]

        return bw

    return Tensor._make(out, ts, [make_bw(i) for i in range(len(ts))], "stack")


def where(cond, a, b) -> Tensor:
    c = cond.data if isinstance(cond, Tensor) else np.asarray(cond)
    at = a if isinstance(a, Tensor) else Tensor(a)
    bt = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor._make(
        np.where(c, at.data, bt.data),
        (at, bt),
        (lambda g: np.where(c, g, 0.0), lambda g: np.where(c, 0.0, g)),
        "where",
    )
