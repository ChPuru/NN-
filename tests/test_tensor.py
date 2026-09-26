"""Gradients, shapes and semantics of the autograd engine itself.

Everything else in the library rests on this file being true.
"""

from __future__ import annotations

import numpy as np
import pytest

import scratchgrad as sg
from scratchgrad.testing import (
    TOL_COMPOSED,
    TOL_OP,
    GradcheckError,
    adjoint_test,
    gradcheck,
    prime_shape,
    rel_error,
)

# Distinct primes. With square or equal shapes a transposed matmul stays
# conformable and NumPy computes garbage without raising.
N, D, M = prime_shape(3)


@pytest.fixture
def xyz(f64, rng):
    x = sg.Tensor(rng.standard_normal((N, D)), requires_grad=True)
    w = sg.Tensor(rng.standard_normal((D, M)), requires_grad=True)
    b = sg.Tensor(rng.standard_normal(M), requires_grad=True)
    return x, w, b


# --------------------------------------------------------------------------- #
# gradients
# --------------------------------------------------------------------------- #
UNARY = {
    "relu": lambda x: x.relu(),
    "tanh": lambda x: x.tanh(),
    "sigmoid": lambda x: x.sigmoid(),
    "exp": lambda x: x.exp(),
    "log": lambda x: (x.abs() + 0.7).log(),
    "sqrt": lambda x: (x.abs() + 0.7).sqrt(),
    "rsqrt": lambda x: (x.abs() + 0.7).rsqrt(),
    "abs": lambda x: x.abs(),
    "sin": lambda x: x.sin(),
    "cos": lambda x: x.cos(),
    "neg": lambda x: -x,
    "clip": lambda x: x.clip(-0.5, 0.5) * 3.0,
    "log_softmax": lambda x: x.log_softmax(-1),
    "logsumexp_axis": lambda x: x.logsumexp(axis=1),
    "logsumexp_all": lambda x: x.logsumexp(),
    "sum_axis": lambda x: x.sum(axis=0),
    "sum_all": lambda x: x.sum(),
    "sum_keepdims": lambda x: x.sum(axis=1, keepdims=True),
    "mean_axis": lambda x: x.mean(axis=1, keepdims=True),
    "var": lambda x: x.var(axis=1),
    "var_ddof1": lambda x: x.var(axis=1, ddof=1),
    "max_axis": lambda x: x.max(axis=1),
    "min_axis": lambda x: x.min(axis=0),
    "max_all": lambda x: x.max(),
    "transpose": lambda x: x.transpose() * 2.0,
    "reshape": lambda x: x.reshape(D, N),
    "flatten": lambda x: x.flatten(0),
    "squeeze": lambda x: x.reshape(N, D, 1).squeeze(2),
    "unsqueeze": lambda x: x.unsqueeze(1),
    "pad": lambda x: x.pad(((1, 2), (0, 3))),
    "repeat_axis": lambda x: x.repeat(3, axis=0),
    "clone": lambda x: x.clone() * 2.0,
    "getitem_slice": lambda x: x[1:5, 2:9],
    "getitem_fancy": lambda x: x[np.array([0, 2, 2, 5])],
    "getitem_bool": lambda x: x[x.data > 0.0],
    "norm2": lambda x: x.norm(2.0),
}

# Compositions accumulate rounding through the chain; TOL_OP is for a single op.
LOOSE = {"softmax", "div", "pow_scalar", "norm2", "var_ddof1", "std"}


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary_gradients(name, xyz):
    x, _, _ = xyz
    tol = TOL_COMPOSED if name in LOOSE else TOL_OP
    gradcheck(UNARY[name], x, tol=tol, name=name)


def test_binary_and_broadcast_gradients(xyz):
    x, w, b = xyz
    gradcheck(lambda x, w, b: x @ w + b, x, w, b, tol=TOL_OP, name="affine")
    gradcheck(lambda x, w: x @ w * 2.0 - 1.0, x, w, tol=TOL_OP, name="mul_sub")
    gradcheck(lambda x: (x * x + 3.0) / (x.abs() + 1.0), x, tol=TOL_COMPOSED, name="div")
    gradcheck(lambda x: (x.abs() + 0.5) ** 2.5, x, tol=TOL_COMPOSED, name="pow_scalar")
    gradcheck(lambda x: x.maximum(x * 0.5), x, tol=TOL_OP, name="maximum")
    gradcheck(lambda x: x.minimum(x * 0.5), x, tol=TOL_OP, name="minimum")
    # b broadcasts over the batch: its backward is a sum, which is the single
    # most commonly omitted step in a hand-written engine.
    gradcheck(lambda x, b: x @ w + b * x.sum(axis=1, keepdims=True), x, b, tol=TOL_COMPOSED)


def test_pow_with_tensor_exponent(f64, rng):
    base = sg.Tensor(rng.standard_normal((N, D)), requires_grad=True)
    expo = sg.Tensor(rng.standard_normal(D), requires_grad=True)
    gradcheck(
        lambda a, e: (a.abs() + 0.5) ** (e.abs() + 0.5), base, expo, tol=TOL_COMPOSED, name="pow_t"
    )


def test_matmul_all_rank_combinations(f64, rng):
    """1-D operands are promoted then squeezed; the backward must reinsert the
    axes in the *reverse* order they were removed."""
    v_n = sg.Tensor(rng.standard_normal(N), requires_grad=True)
    v_d = sg.Tensor(rng.standard_normal(D), requires_grad=True)
    mat = rng.standard_normal((N, D))
    batched = sg.Tensor(rng.standard_normal((3, N, D)), requires_grad=True)
    w = rng.standard_normal((D, M))

    gradcheck(lambda v: sg.Tensor(mat) @ v, v_d, tol=TOL_OP, name="2d@1d")
    gradcheck(lambda v: v @ sg.Tensor(mat), v_n, tol=TOL_OP, name="1d@2d")
    gradcheck(lambda v: v @ v, v_d, tol=TOL_OP, name="1d@1d")
    gradcheck(lambda t: t @ sg.Tensor(w), batched, tol=TOL_OP, name="3d@2d")
    # Hoisted: drawing this inside the lambda would give every finite-difference
    # probe a different forward pass, which reads as a wildly wrong gradient.
    lhs = rng.standard_normal((3, 2, N))
    gradcheck(lambda t: sg.Tensor(lhs) @ t, batched, tol=TOL_OP, name="3d@3d")


def test_free_function_gradients(xyz):
    x, w, _ = xyz
    gradcheck(lambda x, w: sg.concatenate([x @ w, x @ w * 2.0], axis=0), x, w, tol=TOL_OP)
    gradcheck(lambda x: sg.stack([x, x * 3.0], axis=1), x, tol=TOL_OP)
    gradcheck(lambda x: sg.where(x.data > 0, x * 2.0, x * x), x, tol=TOL_OP)
    gradcheck(
        lambda x: x.masked_fill(np.eye(N, D, dtype=bool), -20.0).log_softmax(-1), x, tol=TOL_OP
    )


def test_adjoint_identity_is_exact(f64, rng):
    """For a linear map the adjoint test needs no step size, so it is far sharper
    than finite differences -- and it cannot be fooled by a self-consistent wrong
    forward the way gradcheck can."""
    w = rng.standard_normal((D, M))
    err = adjoint_test(lambda u: u @ sg.Tensor(w), lambda u, v: v @ w.T, (N, D), (N, M))
    assert err < 1e-13


# --------------------------------------------------------------------------- #
# engine semantics
# --------------------------------------------------------------------------- #
def test_leaf_gradients_accumulate_across_backwards(f64):
    x = sg.Tensor(np.array([2.0, 3.0]), requires_grad=True)
    (x * x).sum().backward()
    first = x.grad.copy()
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad, 2 * first)
    x.zero_grad()
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad, first)


def test_intermediate_gradients_reset_between_backwards(f64):
    """Non-leaves must NOT accumulate, or a second backward double-counts them."""
    x = sg.Tensor(np.array([2.0, 3.0]), requires_grad=True)
    h = x * 2.0
    out = (h * h).sum()
    out.backward()
    g1 = h.grad.copy()
    out.backward()
    np.testing.assert_allclose(h.grad, g1)


def test_shared_input_used_twice_sums_both_paths(f64):
    x = sg.Tensor(np.array([3.0]), requires_grad=True)
    (x * x).backward()  # d(x^2)/dx = 2x, only correct if both parents accumulate
    np.testing.assert_allclose(x.grad, [6.0])


def test_diamond_graph(f64, rng):
    x = sg.Tensor(rng.standard_normal((N, D)), requires_grad=True)
    gradcheck(lambda x: (x.relu() + x.tanh() * x.sigmoid()).sum(axis=1), x, tol=TOL_COMPOSED)


def test_deep_chain_does_not_hit_recursion_limit(f64):
    """5000 levels deep. A recursive topological sort dies at CPython's default
    limit of 1000 frames; the iterative one has no ceiling."""
    x = sg.Tensor(np.array([1.0]), requires_grad=True)
    acc = x
    for _ in range(5000):
        acc = acc * 1.0001
    acc.sum().backward()
    np.testing.assert_allclose(x.grad, [1.0001**5000], rtol=1e-9)


def test_no_grad_blocks_graph_construction():
    x = sg.Tensor(np.ones(3), requires_grad=True)
    with sg.no_grad():
        y = x * 2.0
        assert not y.requires_grad and y.is_leaf
        with sg.enable_grad():
            assert (x * 2.0).requires_grad
    assert (x * 2.0).requires_grad


def test_detach_shares_storage_but_cuts_history():
    x = sg.Tensor(np.ones(3), requires_grad=True)
    d = (x * 2.0).detach()
    assert not d.requires_grad and d.is_leaf


def test_backward_requires_scalar_or_explicit_cotangent(f64):
    x = sg.Tensor(np.ones((2, 3)), requires_grad=True)
    with pytest.raises(RuntimeError, match="scalar"):
        (x * 2.0).backward()
    (x * 2.0).backward(np.ones((2, 3)))
    np.testing.assert_allclose(x.grad, np.full((2, 3), 2.0))
    with pytest.raises(ValueError, match="does not match"):
        (x * 2.0).backward(np.ones((3, 2)))


def test_backward_on_non_requiring_tensor_raises():
    with pytest.raises(RuntimeError, match="does not require grad"):
        (sg.Tensor(np.ones(3)) * 2.0).backward()


def test_integer_tensor_cannot_require_grad():
    with pytest.raises(TypeError, match="floating"):
        sg.Tensor(np.arange(5), requires_grad=True)


def test_gradient_is_not_aliased_to_a_reused_buffer(f64):
    """``_accumulate`` copies on first assignment; without it a caller's reused
    buffer would be mutated by the next ``+=``."""
    x = sg.Tensor(np.ones(4), requires_grad=True)
    shared = np.full(4, 5.0)
    x._accumulate(shared)
    x._accumulate(np.ones(4))
    np.testing.assert_allclose(shared, 5.0)


# --------------------------------------------------------------------------- #
# unbroadcast
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("grad_shape", "target"),
    [
        ((2, 3), (2, 3)),
        ((2, 3), (3,)),
        ((2, 3), (1, 3)),
        ((2, 3), (2, 1)),
        ((2, 3), ()),
        ((4, 2, 3), (2, 3)),
        ((4, 2, 3), (1, 1, 3)),
        ((5, 1, 3), (1, 3)),
    ],
)
def test_unbroadcast_shapes_and_sums(grad_shape, target, rng):
    g = rng.standard_normal(grad_shape)
    out = sg.unbroadcast(g, target)
    assert out.shape == target
    # Total mass is conserved: unbroadcasting only ever sums.
    np.testing.assert_allclose(out.sum(), g.sum(), rtol=1e-12)


def test_broadcast_gradient_is_a_sum_not_a_copy(f64):
    b = sg.Tensor(np.zeros(3), requires_grad=True)
    (sg.Tensor(np.ones((5, 3))) + b).sum().backward()
    np.testing.assert_allclose(b.grad, [5.0, 5.0, 5.0])


# --------------------------------------------------------------------------- #
# numerics
# --------------------------------------------------------------------------- #
def test_logsumexp_is_stable_at_extreme_logits(f64):
    big = sg.Tensor(np.array([[1000.0, 999.0, 998.0]]))
    assert np.isfinite(big.logsumexp(axis=1).data).all()
    small = sg.Tensor(np.array([[-1000.0, -1001.0]]))
    assert np.isfinite(small.logsumexp(axis=1).data).all()


def test_softmax_rows_sum_to_one(f64, rng):
    p = sg.Tensor(rng.standard_normal((N, D)) * 30.0).softmax(-1)
    np.testing.assert_allclose(p.data.sum(axis=1), 1.0, atol=1e-12)


def test_sigmoid_is_stable_across_the_real_line(f64):
    x = sg.Tensor(np.array([-800.0, -50.0, 0.0, 50.0, 800.0]))
    s = x.sigmoid().data
    assert np.isfinite(s).all()
    np.testing.assert_allclose(s, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-12)


def test_max_splits_gradient_between_ties(f64):
    """Handing the full upstream gradient to every tied element multiplies it by
    the tie count -- the classic max-pool bug, in miniature."""
    x = sg.Tensor(np.array([[1.0, 1.0, 0.0]]), requires_grad=True)
    x.max(axis=1).sum().backward()
    np.testing.assert_allclose(x.grad, [[0.5, 0.5, 0.0]])


def test_getitem_repeated_index_accumulates(f64):
    x = sg.Tensor(np.zeros((4, 2)), requires_grad=True)
    x[np.array([1, 1, 1])].sum().backward()
    np.testing.assert_allclose(x.grad[1], [3.0, 3.0])
    np.testing.assert_allclose(x.grad[0], [0.0, 0.0])


# --------------------------------------------------------------------------- #
# the checker itself
# --------------------------------------------------------------------------- #
def test_gradcheck_rejects_float32():
    """In float32 a *correct* gradient shows ~1e-2 relative error, so the check
    has no power at all. It must refuse rather than pass everything."""
    x = sg.Tensor(np.ones((3, 3), dtype=np.float32), requires_grad=True)
    with pytest.raises(GradcheckError, match="float32"):
        gradcheck(lambda t: (t * t).astype(np.float32), x, copy_inputs=False)


def test_gradcheck_catches_a_deliberately_wrong_gradient(f64, rng):
    from scratchgrad.tensor import Tensor

    def broken(x):
        # Correct forward, backward missing the factor of 2.
        return Tensor._make(x.data**2, (x,), (lambda g: g * x.data,), "broken")

    with pytest.raises(GradcheckError):
        gradcheck(broken, sg.Tensor(rng.standard_normal((4, 3)) + 2.0, requires_grad=True))


def test_gradcheck_catches_a_transposed_axis(f64, rng):
    """A random cotangent is what makes this detectable. With an all-ones
    cotangent a summed objective is invariant to the transpose and the bug
    passes at 1e-15."""
    from scratchgrad.tensor import Tensor

    def transposed(x):
        return Tensor._make(x.data * 2.0, (x,), (lambda g: (g * 2.0).T.copy(),), "transposed")

    sq = sg.Tensor(rng.standard_normal((5, 5)), requires_grad=True)
    with pytest.raises(GradcheckError):
        gradcheck(transposed, sq)
    ones_only = gradcheck(
        transposed, sq, cotangent=np.ones((5, 5)), raise_on_fail=False, tol=TOL_OP
    )
    assert max(ones_only.values()) < TOL_OP  # the bug hides completely


def test_rel_error_is_normalised_by_array_scale():
    a = np.array([1000.0, 1e-4])
    b = np.array([1000.0, 1e-4 + 1e-9])
    # Per element the second entry disagrees by 1e-5 relative; against the
    # array's scale it is 5e-13, which is the honest reading -- 1e-9 of absolute
    # noise is unavoidable when the objective is of order 1000.
    assert rel_error(a, b) < 1e-11


def test_prime_shapes_are_distinct():
    s = prime_shape(4)
    assert len(set(s)) == 4


def test_stacked_matmul_fast_path_matches_generic():
    """(..., M, K) @ (K, N) takes a flattened single-GEMM path; it must be exact."""
    rng = np.random.default_rng(3)
    x = sg.Tensor(rng.standard_normal((3, 5, 7)), requires_grad=True)
    w = sg.Tensor(rng.standard_normal((7, 11)), requires_grad=True)
    gradcheck(lambda x, w: x @ w, x, w, tol=TOL_OP, name="stacked_matmul")
    x4 = sg.Tensor(rng.standard_normal((2, 3, 5, 7)), requires_grad=True)
    gradcheck(lambda x, w: x @ w, x4, w, tol=TOL_OP, name="stacked_matmul_4d")
    y = x @ w
    y.backward(rng.standard_normal(y.shape))
    np.testing.assert_allclose(y.data, np.matmul(x.data, w.data), rtol=1e-13)


def test_basic_slice_backward_and_repeated_advanced_index():
    rng = np.random.default_rng(4)
    x = sg.Tensor(rng.standard_normal((3, 5, 7)), requires_grad=True)
    gradcheck(lambda x: x[:, 1:4, ::2] * 2.0, x, tol=TOL_OP, name="basic_slice")
    gradcheck(lambda x: x[..., None, 2], x, tol=TOL_OP, name="basic_int_none")
    # Advanced indexing with a repeat must still accumulate, not overwrite.
    v = sg.Tensor(np.zeros(4), requires_grad=True)
    v[np.array([1, 1, 2])].sum().backward()
    np.testing.assert_array_equal(v.grad, [0.0, 2.0, 1.0, 0.0])
