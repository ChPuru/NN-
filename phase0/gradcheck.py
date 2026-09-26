"""The correctness receipt: every hand-written backward vs. central differences.

Run it standalone::

    python3 phase0/gradcheck.py

A loss that goes down is not evidence that a gradient is right. A transposed
axis, a missing broadcast reduction, or a stray ``1/N`` all still descend -- just
to a worse place, slowly, in a way that reads as a hyperparameter problem. This
file is the only thing in Phase 0 that can tell the difference.

Everything here is the way it is because the naive version of the check gives
the wrong answer:

* **float64 only.** In float32 a *correct* gradient shows ~1e-2 relative error:
  the check has no power and passes anything.
* **A random cotangent, never all-ones.** ``<1, out>`` is a plain sum, and a sum
  is invariant to a transposed axis -- so an axis bug passes at 1e-15. With
  ``v ~ N(0, 1)`` it cannot hide.
* **Perturb through ``theta.flat[i]``.** ``theta.ravel()[i] = ...`` writes into a
  *copy* whenever the array is non-contiguous, the forward never sees the
  perturbation, and the numeric gradient comes back as exact zeros -- which then
  gets read as "the analytic gradient is wrong".
* **Repel ReLU kinks.** A probe pair straddling zero measures the average of two
  different one-sided derivatives. One 1e-1 among twenty 1e-9 is nearly always a
  straddled kink and not a bug, so the kinks are pushed to ``|x| > 20h`` and the
  ambiguity is removed rather than argued about.
* **Prime shapes N=7, D=11, M=5.** With square shapes a transposed matmul stays
  conformable and NumPy computes garbage without raising.
* **Relative error normalised by the array scale**,
  ``max|a-n| / (max|a| + max|n| + 1e-12)``. Central differences have a noise
  floor set by the magnitude of the *objective*, not of each gradient entry, so
  a per-element denominator can never pass at 1e-9 for a legitimately tiny
  entry no matter how correct the gradient is.

And the standing caveat: finite differences prove ``backward == d(forward)/dx``.
A *wrong forward* with a self-consistently derived backward passes forever. That
is why the linear maps also get :func:`adjoint_test`, which uses no step size at
all and is exact to 1e-12.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
from scratchnn import MLP, Dense, ReLU, SoftmaxCrossEntropy

__all__ = [
    "TOL_ADJOINT",
    "TOL_COMPOSED",
    "TOL_OP",
    "check_adjoint",
    "check_composed",
    "check_cross_entropy",
    "check_dense",
    "check_relu",
    "numeric_vjp",
    "rel_error",
    "repel_kinks",
    "run_all",
]

#: A single primitive op. Anything looser is hiding something.
TOL_OP = 1e-9
#: A loss composed of many ops: rounding accumulates through the chain.
TOL_COMPOSED = 1e-7
#: The adjoint identity is exact up to floating-point summation.
TOL_ADJOINT = 1e-12

#: Distinct primes, so a transposed matmul raises instead of lying.
N, D, M, H = 7, 11, 5, 13

_H_REL = 1e-5


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """``max|a-b| / (max|a| + max|b| + 1e-12)`` -- normalised by the array scale.

    Not per element. With ``h = 1e-5`` in float64, differencing an objective of
    order 10 leaves ~1e-10 of rounding noise in every entry; judged against its
    own size, a true gradient entry of 2.5e-3 then shows a 4e-8 "error" and no
    correct implementation could ever pass at 1e-9.

    Array-scale normalisation is also *more* sensitive to real bugs: one entry of
    size 1e-4 with a flipped sign contributes 2e-4 to the numerator, which
    against a scale of 3 is 7e-5 -- four orders above tolerance.

    The blind spot, stated plainly: an entry whose true value is below the noise
    floor (~1e-11 of the array scale) is not verified by this check at all.
    Nothing built on finite differences can verify it; that is what
    :func:`check_adjoint` is for.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch in rel_error: {a.shape} vs {b.shape}")
    scale = float(np.max(np.abs(a))) + float(np.max(np.abs(b)))
    return float(np.max(np.abs(a - b))) / (scale + 1e-12)


def repel_kinks(x: np.ndarray, h: float, factor: float = 20.0) -> np.ndarray:
    """Push entries off zero so no probe pair can straddle a ReLU kink."""
    out = np.array(x, dtype=np.float64, copy=True)
    margin = factor * h
    close = np.abs(out) < margin
    if np.any(close):
        out[close] = np.where(out[close] >= 0, 1.0, -1.0) * margin
    return out


def numeric_vjp(
    forward: Callable[[], np.ndarray],
    thetas: list[np.ndarray],
    v: np.ndarray,
    h_rel: float = _H_REL,
) -> list[np.ndarray]:
    """Central-difference gradient of ``<v, forward()>`` w.r.t. each of ``thetas``.

    ``forward`` takes no arguments and must read the *same array objects* listed
    in ``thetas``, which are perturbed in place.

    Args:
        forward: Closure returning the output array (or a 0-d array for a loss).
        thetas: Arrays to differentiate with respect to, perturbed in place and
            restored exactly.
        v: Cotangent, shaped like ``forward()``'s output.
        h_rel: Relative step. The absolute step is ``h_rel * max(1, |x|)``, so a
            parameter of magnitude 1e3 is not probed with a step that vanishes
            into its own rounding error.

    Returns:
        One float64 array per entry of ``thetas``.
    """
    grads = []
    for theta in thetas:
        if theta.dtype != np.float64:
            raise TypeError(f"gradcheck is float64-only, got {theta.dtype}")
        g = np.zeros(theta.shape, dtype=np.float64)
        for i in range(theta.size):
            orig = float(theta.flat[i])
            h = h_rel * max(1.0, abs(orig))
            # .flat[i] writes through to the buffer; .ravel()[i] may not.
            theta.flat[i] = orig + h
            plus = float(np.sum(np.asarray(forward(), dtype=np.float64) * v))
            theta.flat[i] = orig - h
            minus = float(np.sum(np.asarray(forward(), dtype=np.float64) * v))
            theta.flat[i] = orig
            g.flat[i] = (plus - minus) / (2.0 * h)
        grads.append(g)
    return grads


def _compare(
    label: str,
    names: list[str],
    analytic: list[np.ndarray],
    numeric: list[np.ndarray],
) -> dict[str, float]:
    return {
        f"{label}/{n}": rel_error(a, b) for n, a, b in zip(names, analytic, numeric, strict=True)
    }


# --------------------------------------------------------------------------- #
# per-op checks
# --------------------------------------------------------------------------- #
def check_dense(seed: int = 0) -> dict[str, float]:
    """``y = x @ W + b`` against central differences, w.r.t. ``W``, ``b`` and ``x``."""
    rng = np.random.default_rng(seed)
    layer = Dense(D, M, rng)
    x = rng.standard_normal((N, D))
    v = rng.standard_normal((N, M))  # random, not ones: a sum hides a transpose

    layer.forward(x)
    gx = layer.backward(v)
    analytic = [layer.dW.copy(), layer.db.copy(), gx]
    numeric = numeric_vjp(lambda: layer.forward(x), [layer.W, layer.b, x], v)
    return _compare("dense", ["W", "b", "x"], analytic, numeric)


def check_relu(seed: int = 1) -> dict[str, float]:
    """ReLU's mask backward, with every input pushed clear of the kink."""
    rng = np.random.default_rng(seed)
    act = ReLU()
    x = repel_kinks(rng.standard_normal((N, D)), _H_REL)
    v = rng.standard_normal((N, D))
    act.forward(x)
    analytic = [act.backward(v)]
    numeric = numeric_vjp(lambda: act.forward(x), [x], v)
    return _compare("relu", ["x"], analytic, numeric)


def check_cross_entropy(seed: int = 2) -> dict[str, float]:
    """The fused ``(p - onehot)/N`` against differences of the log-sum-exp loss."""
    rng = np.random.default_rng(seed)
    loss = SoftmaxCrossEntropy()
    logits = rng.standard_normal((N, M)) * 3.0  # scaled up: exercise the max-shift
    y = rng.integers(0, M, size=N)
    v = np.asarray(rng.standard_normal(()))  # the loss is a scalar
    loss.forward(logits, y)
    analytic = [loss.backward() * v]
    numeric = numeric_vjp(lambda: np.asarray(loss.forward(logits, y)), [logits], v)
    return _compare("cross_entropy", ["logits"], analytic, numeric)


# --------------------------------------------------------------------------- #
# composed check
# --------------------------------------------------------------------------- #
def _kink_free_input(model: MLP, rng: np.random.Generator, margin: float) -> np.ndarray:
    """Draw an input whose hidden pre-activations all sit clear of the kink.

    The composed check cannot repel kinks by editing ``x``: the kink lives at the
    *pre-activation*, several ops downstream. Resampling until every hidden
    pre-activation clears ``margin`` is the honest fix -- the alternative,
    loosening the tolerance until the straddled probe fits inside it, would hide
    exactly the class of bug this file exists to find.
    """
    first = model.dense_layers()[0]
    for _ in range(64):
        x = rng.standard_normal((N, D))
        if np.min(np.abs(x @ first.W + first.b)) > margin:
            return x
    raise RuntimeError("could not draw a kink-free input in 64 tries")


def check_composed(seed: int = 3) -> dict[str, float]:
    """The whole loss -- Dense, ReLU, Dense, softmax-CE -- end to end.

    This is the check that catches a per-layer backward that is individually
    right but wired up in the wrong order, which no per-op check can see.
    """
    rng = np.random.default_rng(seed)
    model = MLP([D, H, M], rng)
    crit = SoftmaxCrossEntropy()
    x = _kink_free_input(model, rng, margin=20.0 * _H_REL)
    y = rng.integers(0, M, size=N)
    v = np.asarray(rng.standard_normal(()))

    def forward() -> np.ndarray:
        return np.asarray(crit.forward(model.forward(x), y))

    forward()
    gx = model.backward(crit.backward() * v)
    d0, d1 = model.dense_layers()
    analytic = [d0.dW.copy(), d0.db.copy(), d1.dW.copy(), d1.db.copy(), gx]
    thetas = [d0.W, d0.b, d1.W, d1.b, x]
    numeric = numeric_vjp(forward, thetas, v)
    return _compare("composed", ["W1", "b1", "W2", "b2", "x"], analytic, numeric)


# --------------------------------------------------------------------------- #
# adjoint check -- no step size at all
# --------------------------------------------------------------------------- #
def check_adjoint(seed: int = 4) -> dict[str, float]:
    """``<v, A u> == <A^T v, u>`` for the two linear maps inside ``Dense``.

    Exact to floating-point summation, so it is a far sharper instrument than
    finite differences, and unlike them it cannot be fooled by a self-consistent
    *wrong forward*. The bias is zeroed first: ``x @ W + b`` is affine, and the
    identity holds for the linear part only.
    """
    rng = np.random.default_rng(seed)
    layer = Dense(D, M, rng)
    layer.b[...] = 0.0

    # (1) as a map of the input: A u = u @ W, A^T v = v @ W.T
    u = rng.standard_normal((N, D))
    v = rng.standard_normal((N, M))
    lhs = float(np.sum(layer.forward(u) * v))
    rhs = float(np.sum(layer.backward(v) * u))
    err_x = abs(lhs - rhs) / (1e-12 + abs(lhs) + abs(rhs))

    # (2) as a map of the weight: A W = x @ W, A^T v = x.T @ v
    x = rng.standard_normal((N, D))
    v2 = rng.standard_normal((N, M))
    lhs_w = float(np.sum(layer.forward(x) * v2))
    layer.backward(v2)
    rhs_w = float(np.sum(layer.dW * layer.W))
    err_w = abs(lhs_w - rhs_w) / (1e-12 + abs(lhs_w) + abs(rhs_w))

    return {"adjoint/input": err_x, "adjoint/weight": err_w}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_all(verbose: bool = True) -> tuple[dict[str, float], bool]:
    """Run every check. Returns ``(errors, all_passed)``."""
    groups = [
        (
            f"per-op (tol {TOL_OP:.0e})",
            TOL_OP,
            [check_dense(), check_relu(), check_cross_entropy()],
        ),
        (f"composed (tol {TOL_COMPOSED:.0e})", TOL_COMPOSED, [check_composed()]),
        (f"adjoint, h-free (tol {TOL_ADJOINT:.0e})", TOL_ADJOINT, [check_adjoint()]),
    ]
    all_errors: dict[str, float] = {}
    ok = True
    for title, tol, results in groups:
        if verbose:
            print(f"\n{title}")
            print("  " + "-" * 54)
        for res in results:
            for name, err in res.items():
                all_errors[name] = err
                passed = np.isfinite(err) and err <= tol
                ok &= bool(passed)
                if verbose:
                    print(f"  {name:<26s} rel_err = {err:.3e}   {'ok' if passed else 'FAIL'}")
    if verbose:
        print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
        print("float64, central differences, h = 1e-5*max(1,|x|), random cotangent.")
    return all_errors, ok


def main() -> int:
    ap = argparse.ArgumentParser(description="gradcheck the Phase 0 framework")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    _, ok = run_all(verbose=not args.quiet)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
