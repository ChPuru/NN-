# 03 — LayerNorm

Verified by [`verify/verify_03_layernorm.py`](verify/verify_03_layernorm.py) — 8/8.

## Setup, and the conventions that are not negotiable

Fix one **group**: for LayerNorm that is one sample's trailing axes, `D` elements sharing a mean
and a variance. Everything below is inside one group; groups do not interact.

```
μ  = (1/D) Σ_i x_i
v  = (1/D) Σ_i (x_i − μ)²           BIASED — denominator D, not D−1
σ  = sqrt(v + ε)                    ε INSIDE the root
x̂_i = (x_i − μ) / σ
y_i = g_i x̂_i + b_i
```

Both conventions are choices, and both are load-bearing. The Annotated Transformer writes
`g·(x−μ)/(std_unbiased + ε) + b` — `ddof=1` and the epsilon *outside*. That is a different function.
It trains fine; it does not gradcheck against this one; a checkpoint written by one is silently
rescaled by the other. At `D = 11` the Bessel factor alone is `sqrt(11/10) = 1.0488`, and the two
forwards differ by a measured **0.188** in absolute value on a standard-normal input.

Write `istd = 1/σ` below; it appears everywhere.

## The derivation

The nuisance is that `μ` and `v` both depend on *every* `x_j`, so `x̂_i` depends on `x_j` for `i ≠ j`.
Do it in the honest order: cotangents on `x̂`, then on `v` and `μ`, then collect.

**Step 0 — through the affine.** `y_i = g_i x̂_i + b_i` gives immediately

```
dx̂_i = dy_i · g_i
dg_i = Σ_n dy_{n,i} x̂_{n,i}          (summed over everything the parameter is shared across)
db_i = Σ_n dy_{n,i}
```

**Step 1 — the variance.** `x̂_i = (x_i − μ)·(v + ε)^(−1/2)`, so

```
∂x̂_i/∂v = (x_i − μ) · (−1/2)(v + ε)^(−3/2) = −½ istd³ (x_i − μ)

dv = Σ_i dx̂_i · ∂x̂_i/∂v = −½ istd³ Σ_i dx̂_i (x_i − μ)
```

**Step 2 — the mean.** Here is the step that is usually asserted. Naively `μ` reaches `x̂` by two
routes: directly, and through `v`. The second route vanishes:

```
∂v/∂μ = (1/D) Σ_i 2(x_i − μ)(−1) = −(2/D) Σ_i (x_i − μ) = 0
```

because `Σ_i (x_i − μ) ≡ 0` by construction of `μ`. So only the direct route survives:

```
dμ = Σ_i dx̂_i · ∂x̂_i/∂μ = −istd Σ_i dx̂_i
```

**Step 3 — collect.** `x_j` reaches the output three ways: directly through its own `x̂_j`, through
`μ` (with `∂μ/∂x_j = 1/D`), and through `v` (with `∂v/∂x_j = 2(x_j − μ)/D`, using Step 2's fact so
that `μ`'s dependence on `x_j` contributes nothing extra here either).

```
dx_j = istd·dx̂_j  +  dv · 2(x_j − μ)/D  +  dμ/D
```

That is a correct, implementable backward. It is also four passes over the data and carries `v`
and `μ` cotangents around. Substitute Steps 1 and 2 into it:

```
dx_j = istd·dx̂_j
     − istd³ (x_j − μ)·(1/D) Σ_i dx̂_i (x_i − μ)
     − istd·(1/D) Σ_i dx̂_i
```

and use `istd(x_j − μ) = x̂_j` twice on the middle term — `istd³(x_j−μ)(x_i−μ) = istd·x̂_j x̂_i`:

```
dx_j = istd · [ dx̂_j − mean_i(dx̂_i) − x̂_j · mean_i(dx̂_i x̂_i) ]
```

which is the implemented form:

```
dx = (1/σ) · (dx̂ − mean(dx̂) − x̂ ⊙ mean(dx̂ ⊙ x̂))
```

The three terms have a clean reading. The first is the naive "just divide by σ". The second
removes the component of `dx̂` along the all-ones direction — the mean subtraction is invariant to
shifting `x`, so no gradient may flow that way. The third removes the component along `x̂` — scaling
`x` leaves `x̂` unchanged, so no gradient may flow that way either. **LayerNorm's backward is a
projection of `dx̂` orthogonal to `span{1, x̂}`, rescaled by `1/σ`.** Both removed directions are
exactly the invariances of the forward pass, which is a good way to remember the formula and a
better way to sanity-check an implementation: the analytic `dx` must satisfy `Σ_i dx_i = 0` and
`Σ_i dx_i x̂_i = 0` in every group.

`verify_03` runs the unsimplified Step-3 expression and the collapsed one on the same numbers:
they agree to **1.17e-16**. The algebra above is not a story about how the formula might have been
found; it is the thing that was executed.

## RMSNorm, for free

RMSNorm drops the centring: `x̂ = x/sqrt(mean(x²) + ε)`. Step 2 never happens, so the mean term is
simply absent:

```
dx = istd · (dx̂ − x̂ ⊙ mean(dx̂ ⊙ x̂))
```

`src/scratchgrad/nn/norm.py::_normalize` implements exactly this with `center=False`, which is why
one function serves LayerNorm, RMSNorm, GroupNorm and BatchNorm — the only thing that changes
between them is *which axes form the group*.

## Verification

| Check | Measured | Bound |
|---|---|---|
| unsimplified `dμ`/`dv` chain = collapsed three-term form | 1.17e-16 | ≤ 1e-14 |
| forward = biased variance, ε inside the root | 0.0 | ≤ 1e-15 |
| three-term `dx` vs the framework's fused backward | 3.07e-17 | ≤ 1e-14 |
| `dg = Σ_n dy·x̂` | 0.0 | ≤ 1e-15 |
| `db = Σ_n dy` | 0.0 | ≤ 1e-15 |
| central differences (`check_module` over x, g, b) | 3.93e-11 | ≤ 1e-7 |
| Annotated-Transformer convention: max forward gap | 0.188 | ≥ 1e-3 |

### One trap found while writing the check

Gradchecking a shape-preserving layer with `check_module` fails at **1.33e-5** if the input and the
random cotangent come from the same seeded stream — `check_module` seeds the input from `seed` and
hands the same `seed` to `gradcheck`, which draws the cotangent from it, so the two draws are the
same array. `⟨x, LN(x)⟩` is then a near-stationary point, the true gradient collapses by five
orders, and `rel_error` — normalised by that collapsed scale — reads ordinary finite-difference
noise as a failure. The fix is one line (`input_data=` from a different stream) and it is what
`tests/test_norm.py::check_norm` already does. Both numbers are recorded in the verifier so the
difference between "the gradient is wrong" and "the *check* is degenerate" stays visible.
