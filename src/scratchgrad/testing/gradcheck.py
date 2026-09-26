"""Finite-difference gradient checking -- the receipt that the backprop is real.

A training loss that goes down is *not* evidence that a gradient is correct. A
transposed axis, a missing broadcast reduction, or a factor of ``1/N`` in the
wrong place all still descend, just to a worse place. The check here compares
each hand-written vector-Jacobian product against central differences of the
forward pass.

Everything in this module exists because of a specific way the naive version of
this check gives the wrong answer:

* **float64 only.** In float32, a *correct* gradient shows roughly ``1e-2``
  relative error -- the check has no power at all and will "pass" anything.
  :func:`gradcheck` refuses to run in float32.
* **A random cotangent, never all-ones.** Backpropagating ones through a summed
  objective is invariant to a transposed axis, so an axis bug passes at 1e-15.
  With ``v ~ N(0, 1)`` it does not.
* **Perturb through ``theta.flat[i]``.** ``theta.ravel()`` returns a *copy* for
  a non-contiguous array, so the perturbation silently never lands and the
  numeric gradient comes out as exactly zero.
* **Repel ReLU kinks.** A probe pair straddling a kink measures the average of
  two different one-sided derivatives. One ``1e-1`` among twenty ``1e-9`` is
  almost always a straddled kink, not a bug.
* **Prime shapes.** With square or equal test shapes, a transposed matmul stays
  conformable and NumPy computes garbage without raising. ``N=7, D=11, M=5``
  turns that class of bug into a ``ValueError``.

And the standing caveat: gradcheck proves ``backward == d(forward)/dx``. A
*wrong forward* with a self-consistently derived backward passes at 1e-15
forever. Pair it with forward-value tests and, for linear maps, with
:func:`adjoint_test`, which needs no step size at all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .._config import default_dtype
from ..tensor import Tensor

__all__ = [
    "TOL_COMPOSED",
    "TOL_OP",
    "GradcheckError",
    "adjoint_test",
    "check_module",
    "gradcheck",
    "numeric_vjp",
    "prime_shape",
    "rel_error",
    "repel_kinks",
]

#: Tolerance for a single primitive op. Anything looser is hiding something.
TOL_OP = 1e-9
#: Tolerance for a composed loss (many ops deep). ``1e-9`` here produces false
#: failures on correct code -- rounding accumulates through the chain.
TOL_COMPOSED = 1e-7


class GradcheckError(AssertionError):
    """Raised when an analytic gradient disagrees with central differences."""


def rel_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Largest disagreement, normalised by the **scale of the whole array**.

    ``max|a - b| / (max|a| + max|b| + eps)``.

    The denominator is deliberately the array's scale rather than each element's
    own magnitude. Central differences have a noise floor set by the magnitude of
    the *function value*, not of the individual gradient entry: with ``h = 1e-5``
    and float64, differencing an objective of order 10 leaves roughly ``1e-10``
    of rounding noise in every entry. Judge a genuinely small entry -- say
    ``2.5e-3`` -- against its own size and that unavoidable ``1e-10`` reads as a
    ``4e-8`` relative error, so a per-element metric can never pass at ``1e-9``
    no matter how correct the gradient is.

    Normalising by the array scale is also *more* sensitive to real bugs, not
    less: a single entry of size ``1e-4`` with a flipped sign contributes a
    numerator of ``2e-4``, which against a scale of ``3`` is ``7e-5`` -- four
    orders of magnitude above the tolerance.

    The blind spot, stated plainly: an entry whose true gradient is smaller than
    the noise floor (about ``1e-11`` of the array scale) is not verified by this
    check at all. Nothing based on finite differences can verify it. That is what
    :func:`adjoint_test` and forward-value tests are for.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch in rel_error: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    scale = float(np.max(np.abs(a))) + float(np.max(np.abs(b)))
    return float(np.max(np.abs(a - b))) / (scale + eps)


def prime_shape(rank: int = 2, start: int = 0) -> tuple[int, ...]:
    """Distinct small primes, so a transposed matmul raises instead of lying."""
    primes = [7, 11, 5, 13, 3, 17, 2, 19]
    return tuple(primes[(start + i) % len(primes)] for i in range(rank))


def repel_kinks(x: np.ndarray, h: float, factor: float = 20.0) -> np.ndarray:
    """Push entries away from 0 so a probe pair cannot straddle a ReLU kink."""
    x = np.array(x, dtype=np.float64, copy=True)
    margin = factor * h
    close = np.abs(x) < margin
    if np.any(close):
        sign = np.where(x[close] >= 0, 1.0, -1.0)
        x[close] = sign * margin
    return x


def numeric_vjp(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    v: np.ndarray,
    h_rel: float = 1e-5,
) -> list[np.ndarray]:
    """Central-difference gradient of ``<v, fn(*inputs)>`` w.r.t. each input.

    The step is scaled per element (``h = h_rel * max(1, |x|)``) so that a
    parameter of magnitude 1e3 is not probed with a step that vanishes into its
    own rounding error.
    """
    grads: list[np.ndarray] = []
    for t in inputs:
        theta = t.data
        g = np.zeros(theta.shape, dtype=np.float64)
        flat_g = g.reshape(-1)
        for i in range(theta.size):
            orig = float(theta.flat[i])
            h = h_rel * max(1.0, abs(orig))
            # ``.flat[i]`` writes through to the buffer; ``.ravel()[i]`` may not.
            theta.flat[i] = orig + h
            plus = float(np.sum(np.asarray(fn(*inputs).data, dtype=np.float64) * v))
            theta.flat[i] = orig - h
            minus = float(np.sum(np.asarray(fn(*inputs).data, dtype=np.float64) * v))
            theta.flat[i] = orig
            flat_g[i] = (plus - minus) / (2.0 * h)
        grads.append(g)
    return grads


def gradcheck(
    fn: Callable[..., Tensor],
    *inputs: Tensor,
    tol: float = TOL_COMPOSED,
    h_rel: float = 1e-5,
    seed: int = 0,
    cotangent: np.ndarray | None = None,
    repel: bool = True,
    repel_factor: float = 20.0,
    name: str = "",
    raise_on_fail: bool = True,
    verbose: bool = False,
    copy_inputs: bool = True,
) -> dict[str, float]:
    """Check every input's analytic gradient against central differences.

    ``fn`` maps the given tensors to a single output tensor of any shape; the
    check is on the VJP with a random cotangent, so a non-scalar output is fine
    and is in fact preferred.

    ``copy_inputs=True`` (the default) works on private float64 copies, leaving
    the caller's tensors untouched. Pass ``False`` when the tensors handed in are
    **the very leaves the graph will be built from** -- a Module's own
    parameters, for instance. A copy would not be those leaves, so every
    parameter gradient would come back as an exact zero: the graph records the
    parameter object, and the perturbation would land on an unrelated array.
    With ``copy_inputs=False`` the inputs are perturbed and kink-repelled **in
    place**; they must already be float64 and require grad.

    Returns a mapping from input index to its max relative error. Raises
    :class:`GradcheckError` when any exceeds ``tol``.
    """
    if not inputs:
        raise ValueError("gradcheck needs at least one input tensor")
    for t in inputs:
        if not isinstance(t, Tensor):
            raise TypeError(f"gradcheck inputs must be Tensors, got {type(t).__name__}")

    rng = np.random.default_rng(seed)
    with default_dtype(np.float64):
        if copy_inputs:
            work = []
            for t in inputs:
                arr = np.array(t.data, dtype=np.float64, copy=True)
                if repel:
                    arr = repel_kinks(
                        arr, h_rel * max(1.0, float(np.max(np.abs(arr)) or 1.0)), repel_factor
                    )
                work.append(Tensor(arr, requires_grad=True))
        else:
            work = list(inputs)
            for t in work:
                if t.data.dtype != np.float64:
                    raise GradcheckError(
                        f"copy_inputs=False requires float64 inputs, got {t.data.dtype}"
                    )
                if not t.requires_grad:
                    raise GradcheckError("copy_inputs=False requires requires_grad=True inputs")
                if repel:
                    t.data[...] = repel_kinks(
                        t.data, h_rel * max(1.0, float(np.max(np.abs(t.data)) or 1.0)), repel_factor
                    )

        out = fn(*work)
        if not isinstance(out, Tensor):
            raise TypeError(f"fn must return a Tensor, got {type(out).__name__}")
        if out.data.dtype != np.float64:
            raise GradcheckError(
                f"gradcheck ran in {out.data.dtype}; float32 gives ~1e-2 relative error on a "
                "correct gradient and the check has no power. Cast inputs to float64."
            )

        v = (
            np.asarray(cotangent, dtype=np.float64)
            if cotangent is not None
            else rng.standard_normal(out.data.shape)
        )
        if v.shape != out.data.shape:
            raise ValueError(f"cotangent shape {v.shape} != output shape {out.data.shape}")

        for t in work:
            t.grad = None
        out.backward(v)
        analytic = [
            (t.grad if t.grad is not None else np.zeros(t.data.shape, dtype=np.float64))
            for t in work
        ]
        numeric = numeric_vjp(fn, work, v, h_rel=h_rel)

    errors: dict[str, float] = {}
    failures: list[str] = []
    for i, (a, n) in enumerate(zip(analytic, numeric, strict=True)):
        err = rel_error(a, n)
        errors[f"input{i}"] = err
        if verbose:
            print(f"  {name or 'gradcheck'} input{i} shape={tuple(a.shape)} rel_err={err:.3e}")
        if not np.isfinite(err) or err > tol:
            worst = int(np.argmax(np.abs(a - n)))
            failures.append(
                f"input{i} (shape {tuple(a.shape)}): rel_err={err:.3e} > tol={tol:.1e}; "
                f"worst element flat[{worst}] analytic={a.reshape(-1)[worst]:.12g} "
                f"numeric={n.reshape(-1)[worst]:.12g}"
            )
    if failures and raise_on_fail:
        label = f" [{name}]" if name else ""
        raise GradcheckError(f"gradcheck failed{label}:\n  " + "\n  ".join(failures))
    return errors


def adjoint_test(
    forward: Callable[[Tensor], Tensor],
    backward_vjp: Callable[[Tensor, np.ndarray], np.ndarray],
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    seed: int = 0,
    tol: float = 1e-12,
) -> float:
    """Check ``<v, A u> == <A^T v, u>`` for a *linear* map. No step size at all.

    Finite differences trade truncation error against rounding error and settle
    around 1e-9 at best. The adjoint identity is exact up to floating-point
    summation, so for anything linear -- ``Linear``, ``Conv2d``, ``Embedding``,
    a reshape -- it is a far sharper instrument than :func:`gradcheck`, and it
    cannot be fooled by a self-consistent wrong forward in the way gradcheck can.
    """
    rng = np.random.default_rng(seed)
    with default_dtype(np.float64):
        u = Tensor(rng.standard_normal(in_shape), requires_grad=True)
        v = rng.standard_normal(out_shape)
        Au = forward(u)
        if Au.data.shape != out_shape:
            raise ValueError(f"forward produced {Au.data.shape}, expected {out_shape}")
        lhs = float(np.sum(np.asarray(Au.data, dtype=np.float64) * v))
        ATv = np.asarray(backward_vjp(u, v), dtype=np.float64)
        rhs = float(np.sum(ATv * np.asarray(u.data, dtype=np.float64)))
    denom = 1e-12 + abs(lhs) + abs(rhs)
    err = abs(lhs - rhs) / denom
    if err > tol:
        raise GradcheckError(
            f"adjoint test failed: <v, Au>={lhs:.16g} != <A^T v, u>={rhs:.16g} (rel {err:.3e})"
        )
    return err


def check_module(
    module,
    input_shape: tuple[int, ...],
    tol: float = TOL_COMPOSED,
    seed: int = 0,
    check_input: bool = True,
    input_data: np.ndarray | None = None,
    name: str = "",
    integer_input: bool = False,
    vocab: int | None = None,
    forward: Callable | None = None,
) -> dict[str, float]:
    """Gradcheck every parameter of an ``nn.Module``, plus optionally its input.

    The module's parameters and float buffers are temporarily swapped for
    float64 copies, the check runs against **those actual objects** (see
    ``copy_inputs`` on :func:`gradcheck`), and the originals are restored in a
    ``finally``. Both shapes a layer can take are handled: a float input, and an
    integer index input such as :class:`Embedding`'s, which has no gradient of
    its own.

    ``forward`` overrides how the module is invoked -- pass e.g.
    ``lambda m, x: m(x, mask)`` for a layer needing extra arguments.
    """
    from ..nn.module import Module, Parameter  # local import: avoids an import cycle

    if not isinstance(module, Module):
        raise TypeError(f"expected an nn.Module, got {type(module).__name__}")

    rng = np.random.default_rng(seed)
    was_training = module.training
    module.eval()  # deterministic: no dropout mask resampled between probes

    saved_params: list[tuple[Module, str, Parameter]] = []
    saved_buffers: list[tuple[Module, str, np.ndarray]] = []
    work_params: list[Parameter] = []
    try:
        with default_dtype(np.float64):
            for _, sub in module.named_modules():
                for pname, p in list(sub._parameters.items()):
                    if p is None:
                        continue
                    saved_params.append((sub, pname, p))
                    q = Parameter(np.array(p.data, dtype=np.float64, copy=True))
                    sub._parameters[pname] = q
                    work_params.append(q)
                for bname, buf in list(sub._buffers.items()):
                    if buf is None or buf.dtype.kind != "f":
                        continue
                    saved_buffers.append((sub, bname, buf))
                    sub._buffers[bname] = np.array(buf, dtype=np.float64, copy=True)

            call = forward if forward is not None else (lambda m, *a: m(*a))

            if integer_input:
                if vocab is None:
                    raise ValueError("integer_input=True requires vocab=")
                idx = Tensor(rng.integers(0, vocab, size=input_shape).astype(np.int64))
                if not work_params:
                    raise ValueError("module has no parameters and no differentiable input")
                return gradcheck(
                    lambda *ps, idx=idx: call(module, idx),
                    *work_params,
                    tol=tol,
                    seed=seed,
                    name=name or type(module).__name__,
                    copy_inputs=False,
                )

            x_arr = (
                np.array(input_data, dtype=np.float64, copy=True)
                if input_data is not None
                else rng.standard_normal(input_shape)
            )
            x = Tensor(x_arr, requires_grad=check_input)
            args = ([x] if check_input else []) + work_params
            if not args:
                raise ValueError("nothing to check: no parameters and check_input=False")

            def fn(*ts, x=x, check_input=check_input):
                return call(module, ts[0] if check_input else x)

            return gradcheck(
                fn,
                *args,
                tol=tol,
                seed=seed,
                name=name or type(module).__name__,
                copy_inputs=False,
            )
    finally:
        for sub, pname, p in saved_params:
            sub._parameters[pname] = p
        for sub, bname, buf in saved_buffers:
            sub._buffers[bname] = buf
        module.train(was_training)
