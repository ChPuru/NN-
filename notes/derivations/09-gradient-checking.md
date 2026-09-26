# 09 — Gradient checking

Verified by [`verify/verify_09_gradient_checking.py`](verify/verify_09_gradient_checking.py) — 13/13.

Everything else in this directory rests on this file: a derivation is only worth something if the
check that confirmed it has power.

## Central differences, and why not forward

Taylor-expand `f` about `x`:

```
f(x+h) = f + h f' + h²/2 f'' + h³/6 f''' + …
f(x−h) = f − h f' + h²/2 f'' − h³/6 f''' + …
```

Forward difference — subtract `f`, divide by `h`:

```
(f(x+h) − f(x))/h = f' + (h/2) f'' + O(h²)          truncation error O(h)
```

Central difference — subtract the two expansions, so every **even** power cancels:

```
(f(x+h) − f(x−h))/(2h) = f' + (h²/6) f''' + O(h⁴)   truncation error O(h²)
```

The `f''` term cancels by symmetry. That is the whole advantage, and it costs one extra function
evaluation per element — a 2× price for a squared error term.

### The `h` trade-off

Truncation error falls as `h` shrinks. Rounding error rises. `f(x±h)` are each computed to a
relative accuracy of about `ε_mach` (2.22e-16 in float64), so the numerator carries an absolute
error of about `ε_mach·|f|`, and dividing by `2h` amplifies it:

```
E(h) ≈ (h²/6)|f'''|  +  ε_mach |f| / h
       └ truncation ┘    └── rounding ──┘
```

Minimise: `dE/dh = h|f'''|/3 − ε|f|/h² = 0`, so

```
h* = (3 ε |f| / |f'''|)^(1/3)  ~  ε^(1/3)  ≈  6.1e-6
E(h*) ~ ε^(2/3) ≈ 3.7e-11
```

The same argument for the forward difference gives `h* ~ ε^(1/2) ≈ 1.5e-8` and `E ~ ε^(1/2) ≈
1.5e-8`. Central differences are not merely more accurate at a given `h`; their *best attainable*
accuracy is four orders better.

Measured on `f(x) = exp(sin x)` at `x = 1.2`, swept over 13 decades of `h`:

| | log-log slope (truncation regime) | best `h` | best error |
|---|---|---|---|
| forward | **1.003** (predicted 1) | 1.78e-8 (`√ε` = 1.49e-8) | **6.86e-9** |
| central | **2.000** (predicted 2) | 1.78e-6 (`ε^(1/3)` = 6.06e-6) | **5.77e-12** |

Central beats forward by **1189×** at their respective optima. The framework's `h_rel = 1e-5` is
not the argmin, but it is within a factor of **6.93** of the best attainable error — a sensible
place to sit, since the optimum's location depends on `|f'''|/|f|`, which nobody knows in advance,
and the curve is much flatter to the right of the minimum than to the left.

`h` is scaled per element in `numeric_vjp` — `h = h_rel · max(1, |x|)` — so a parameter of
magnitude 1e3 is not probed with a step that vanishes into its own rounding error.

## Why the relative error is normalised by the array scale

`rel_error` is

```
max|a − n| / (max|a| + max|n| + 1e-12)
```

— the denominator is the **array's** scale, not each element's own magnitude. This looks lax and
is the opposite.

The argument is the rounding term above: the finite-difference noise floor is `ε_mach·|f|/h`, set
by the magnitude of the **objective**, not of the gradient entry being measured. Every entry
inherits the same absolute noise. So a legitimately small entry carries a large *relative* error
however correct the analytic gradient is, and a per-element metric can never pass at 1e-9.

Made concrete: `f(x) = ⟨c, x⟩` with `c = [3, −2, 1e-6]` and `x` of order 10. The objective is of
order 30, so at `h = 1e-5` the noise floor is `ε·30/1e-5 ≈ 9.8e-10`. Scoring the *exactly correct*
gradient `c` against central differences:

| metric | measured |
|---|---|
| per-element `\|a−n\|/(\|a\|+\|n\|)` | **4.44e-05** |
| array-scale `max\|a−n\|/(max\|a\|+max\|n\|)` | **7.81e-11** |

The per-element score is 4.4e-5 for a gradient with zero error in it. At a 1e-9 tolerance it fails
forever, and no reduction of `h` helps — shrinking `h` raises the rounding term.

Array-scale normalisation is also *more* sensitive to real bugs, not less. A single entry of size
1e-4 with a flipped sign contributes a numerator of 2e-4; against an array scale of 3 that is 7e-5,
four orders above tolerance.

The blind spot, stated plainly: an entry whose true gradient is below the noise floor (about 1e-11
of the array scale) is not verified by this check at all. Nothing based on finite differences can
verify it. That is what the adjoint test and forward-value tests are for.

## Why the cotangent must be random

`gradcheck` compares `⟨v, J⟩` for a **random** `v`, never all-ones. The reason is that `⟨1, f(x)⟩`
is a sum, and sums are invariant to permutations of what they sum over — including transposes.

The demonstration is a transpose op with the classic wrong backward (`g` where `gᵀ` belongs), at a
**square** shape, because at `(7, 7)` the wrong gradient is still conformable and nothing raises:

| cotangent | relative error of the buggy backward |
|---|---|
| all ones | **0.0** — passes at machine precision |
| `N(0,1)` | **0.593** — fails outright |

Two habits follow, and both are in `testing/gradcheck.py`:

- **Prime shapes** (`N=7, D=11, M=5`). With square or equal extents a transposed matmul stays
  conformable and NumPy computes garbage without raising; with primes the same bug is a
  `ValueError`.
- **Random cotangents**, and preferably a **non-scalar output** so the cotangent has structure to
  disagree about.

There is a matching trap on the other side, found while writing `03`: if the input and the
cotangent are drawn from the *same* seeded stream, a shape-preserving layer is checked at
`⟨x, f(x)⟩`, which for a normalisation layer is near-stationary. The true gradient collapses by
five orders, `rel_error` normalises by that collapsed scale, and ordinary noise reads as a
**1.33e-5** failure on a correct gradient. Draw the input from a different stream.

## Why float64, and other things that quietly ruin a check

In float32, `ε_mach = 1.19e-7` and the noise floor at `h = 1e-5` is `1.2e-2·|f|`. The check has no
power at all: it will "pass" almost anything. Measured, on the same correct gradient at the same
step size:

| dtype | relative error of a **correct** gradient |
|---|---|
| float32 | **3.67e-03** |
| float64 | **4.01e-11** |

Six orders of resolution. `gradcheck` refuses to run in float32 for this reason.

The dtype trap is easy to walk into, because it does not announce itself: `sg.Tensor(x)` on a
float64 array returns **float32** — the default dtype applies at construction — and so does
`Parameter(np.zeros(3, dtype=np.float64))`. Both bit me while writing these verifiers (`01`'s
forward/reverse agreement read 3.2e-8, `08`'s Adam comparison read 5.7e-8). Pass
`dtype=np.float64` explicitly, or wrap the block in `sg.default_dtype(np.float64)`.

Three more, each of which produces a wrong number rather than an error:

- **Perturb through `theta.flat[i]`.** `theta.ravel()[i]` returns a *copy* for a non-contiguous
  array, so the perturbation never lands and the numeric gradient comes back as an exact zero.
- **Repel ReLU kinks.** A probe pair straddling a kink measures the average of two different
  one-sided derivatives. One `1e-1` among twenty `1e-9` is almost always a straddled kink, not a
  bug — check whether the argmax winner changed before believing it.
- **Hoist constants out of the checked function.** Calling an `rng` inside the lambda gives every
  finite-difference probe a different forward pass.
- **Check the module's own parameters, not copies.** `gradcheck(copy_inputs=False)` exists because
  the graph records the parameter *object*; perturbing a copy leaves the graph untouched and every
  parameter gradient comes back as an exact zero — which looks like a passing check if the
  tolerance is on the wrong side.

## The adjoint test: exact, and no step size at all

For a **linear** map `A`, the defining property of the transpose is

```
⟨v, A u⟩ = ⟨Aᵀ v, u⟩        for all u, v
```

A correct backward for a linear op *is* `Aᵀ`, so this identity is a complete check — and it never
evaluates a difference quotient, so it has no truncation error and no noise floor. It is exact up
to floating-point summation order.

Applicable to more of a network than one expects: `Linear`, `Conv2d`, `Embedding`, reshape,
transpose, pad, concatenate, `im2col`/`col2im`, and any composition of them. Measured on the same
`Linear`:

| check | measured |
|---|---|
| adjoint identity `⟨v, Au⟩ = ⟨Aᵀv, u⟩` | **0.0** |
| `gradcheck`, central differences | **2.26e-11** |
| adjoint on `Conv2d` (stride 2, pad 1) | **1.47e-15** (from `05`) |

It also cannot be fooled the way gradcheck can. `gradcheck` proves `backward == d(forward)/dx`; a
*wrong forward* with a self-consistently derived backward passes at 1e-15 forever. The adjoint test
compares the forward and the backward against **each other**, so a forward that is wrong in a way
the backward does not mirror shows up. Neither replaces a forward-value test against hand-computed
numbers — which is why `03`, `05`, `06` and `07` all have one.

## Tolerances

```
TOL_OP       = 1e-9      a single primitive op
TOL_COMPOSED = 1e-7      a loss composed of many ops
```

`1e-9` on a composed loss produces false failures on correct code: rounding accumulates through the
chain. Loosening a tolerance to make a check pass is how a real bug gets shipped, so the two levels
are named and fixed rather than tuned per test.

## Verification

| Check | Measured | Bound |
|---|---|---|
| forward-difference log-log slope | 1.003 | \|·−1\| ≤ 0.05 |
| central-difference log-log slope | 2.000 | \|·−2\| ≤ 0.05 |
| forward optimum near `√ε` | 1.78e-8 | within 0.75 decades |
| central optimum near `ε^(1/3)` | 1.78e-6 | within 0.75 decades |
| best forward error / best central error | 1189× | ≥ 100× |
| cost of `h_rel = 1e-5` vs the optimum | 6.93× | ≤ 10× |
| per-element metric on a correct 1e-6 entry | 4.44e-05 | ≥ 1e-9 (i.e. it fails) |
| array-scale metric, same numbers | 7.81e-11 | ≤ 1e-9 |
| all-ones cotangent vs a transposed backward | 0.0 | ≤ 1e-14 (i.e. it is fooled) |
| Gaussian cotangent, same backward | 0.593 | ≥ 0.1 |
| correct gradient scored in float32 | 3.67e-03 | ≥ 1e-3 |
| the same in float64 | 4.01e-11 | ≤ 1e-9 |
| adjoint identity on `Linear` | 0.0 | ≤ 1e-14 |
