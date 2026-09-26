# 01 — Backprop as vector-Jacobian products

Verified by [`verify/verify_01_backprop_basics.py`](verify/verify_01_backprop_basics.py) — 6/6.

## What a backward function actually is

Take a function `f: Rⁿ → Rᵐ`. Its derivative at a point is the Jacobian `J`, an `m × n` matrix
with `J[i,j] = ∂f_i/∂x_j`. Nobody in this codebase ever builds one. A `(N, 784) → (N, 128)` layer
at `N = 128` has a Jacobian with `(128·128) × (128·784) = 1.64 × 10⁹` entries, and the useful
contraction against it costs almost nothing by comparison.

What is built instead is the **vector-Jacobian product**: given a row vector `v` in the output
space, produce `vᵀJ`, a row vector in the input space.

```
VJP_f(v) = Jᵀ v            (as a column vector; "the cotangent, pulled back")
```

Why that particular contraction is the useful one comes straight from the chain rule. For a
composition `L = f_k ∘ … ∘ f_1` with a **scalar** output `L`, the gradient is

```
∇L = J_1ᵀ J_2ᵀ … J_kᵀ · 1
```

a product of matrices, and matrix products are associative. Two ways to bracket it:

```
forward mode:   (…((J_1ᵀ) J_2ᵀ) …)   — start at the input, carry an n-column object
reverse mode:   J_1ᵀ (J_2ᵀ (… · 1))  — start at the output, carry an m-column object
```

The scalar output makes the right bracketing carry a *single vector* at every stage. The left one
carries `n` of them — one per input. For a network the number of inputs is the number of
parameters and the number of outputs is one, so reverse mode is `n` times cheaper. That is the
entire argument, and it is not asymptotic hand-waving: `verify_01` times one `backward()` on a
205-parameter model against the 205 directional passes forward mode would need, and measures a
ratio of **42–64×** across repeats on this machine. That range is the honest form: it is the one
number in this directory that is not bit-reproducible, `results.json` carries whatever the run
that wrote it measured, and it is listed there under `nondeterministic` so nobody quotes it as a
constant. The two gradients agree to `1.2e-10`.

So a backward function's signature is fixed: *in* comes the cotangent of the output, *out* goes
the contribution to the cotangent of one parent. That is exactly `Tensor._make`:

```python
def my_op(x, y):
    out = ...  # numpy forward

    def gx(g):
        return ...  # d<v, out>/dx  given g, the cotangent of out

    def gy(g):
        return ...

    return Tensor._make(out, (x, y), (gx, gy), "my_op")
```

## Two consequences that are easy to get wrong

### Accumulation, never assignment

A tensor consumed by two downstream ops sits on two paths to the loss. The multivariate chain rule
adds over paths, so its cotangent is the **sum** of what the two paths deliver. In code that is
`+=`, never `=`. `Tensor._accumulate` does this. The check: for `L = Σx + Σx²`, the true gradient
is `1 + 2x`; assignment would give whichever branch ran last. Measured agreement with `1 + 2x`:
`3.7e-17`.

The same rule is why `zero_grad()` exists at all, and why forgetting it is invisible under Adam —
the accumulated gradient is larger but the sign structure survives, so the run still descends,
just with an effective learning rate that grows with the step count.

### Broadcast forward ⇒ sum backward

This is the one that produces silent wrong answers rather than exceptions, so it is worth deriving
rather than remembering.

Broadcasting is a **linear map**. `B: Rⁿ → Rᵐ` copies each input entry to several output positions:

```
(Bu)_i = u_{σ(i)}          σ maps an output index to the input index it was copied from
```

Its Jacobian is the 0/1 matrix `J[i, j] = 1` iff `σ(i) = j`. Then

```
(Jᵀ v)_j = Σ_i J[i, j] v_i = Σ_{i : σ(i) = j} v_i
```

— the sum of the cotangents of every output position that entry `j` was copied into. Broadcast is
a copy; its adjoint is a sum. There is nothing to memorise: the transpose of a copy is an addition
in exactly the way the transpose of an addition is a copy.

`unbroadcast(grad, shape)` implements `Jᵀ` for NumPy's specific broadcasting rules, and the order
of its two steps is forced:

```
1. axes broadcasting PREPENDED, (11,) → (7, 11):  sum them away, dropping the axis
2. axes that existed with extent 1, (1, 11) → (7, 11):  sum with keepdims=True
```

Doing (2) first, or (1) with `keepdims=False`, yields an array of the wrong *rank*, and the error
surfaces several ops later where its cause is invisible.

The clean way to state the requirement is the adjoint identity, which is what `verify_01` actually
tests over five shape pairs:

```
⟨v, broadcast(u)⟩ = ⟨unbroadcast(v), u⟩       for all u, v
```

Measured worst-case relative violation: **4.2e-16** — floating-point summation order, nothing else.
This identity is worth reaching for whenever a backward is linear in its input, because it is
exact: no step size, no truncation error, no noise floor (see `09`).

## Where the elementwise rules come from

Every elementwise binary op ends in `unbroadcast`, and the rules themselves are one line of chain
rule each. For `z = x ⊙ y` with `x`, `y` broadcast to a common shape:

```
∂L/∂x = unbroadcast(dz ⊙ y, x.shape)
∂L/∂y = unbroadcast(dz ⊙ x, y.shape)
```

The `unbroadcast` is not defensive coding. Drop it when `x` is `(1, D)` and `y` is `(N, D)` and
NumPy will happily hand you an `(N, D)` gradient for a `(1, D)` leaf; depending on where it lands
that either raises much later or, worse, broadcasts again on the optimizer step.

## The composition, checked end to end

`verify_01` builds `y = tanh(x W₁ + b₁) W₂` and composes the VJPs by hand, right to left:

```
dh = dy W₂ᵀ
dz = dh ⊙ (1 − h²)          h = tanh(z), so tanh'(z) = 1 − tanh²(z)
dx = dz W₁ᵀ
dW₁ = xᵀ dz
```

`dW₁ = xᵀ dz` deserves a sentence, because the transposes are where this goes wrong. `z = xW₁`
means `z[n,k] = Σ_d x[n,d] W₁[d,k]`, so `∂L/∂W₁[d,k] = Σ_n x[n,d] dz[n,k]` — a contraction over
the *batch* axis, which is `xᵀ dz`. The layout `weight: (in, out)` is frozen precisely so this
never needs re-deriving: forward is `x @ W`, and the two backward rules are `dz @ W.T` and
`x.T @ dz` with no further transposes anywhere.

Measured against `backward()`: `dx` and `dW₁` both **exactly 0.0** relative error — the same
floating-point operations in the same order.

## Verification

| Check | Measured | Bound |
|---|---|---|
| `⟨v, broadcast(u)⟩ = ⟨unbroadcast(v), u⟩`, 5 shape pairs | 4.18e-16 | ≤ 1e-15 |
| hand-composed VJP chain vs `backward()`, `dx` | 0.0 | ≤ 1e-14 |
| hand-composed VJP chain vs `backward()`, `dW₁` | 0.0 | ≤ 1e-14 |
| a leaf in two branches accumulates to `1 + 2x` | 3.72e-17 | ≤ 1e-15 |
| forward-mode cost / reverse-mode cost at `n = 205` (wall clock) | 42–64× | ≥ 10× |
| and the two modes agree | 1.18e-10 | ≤ 1e-9 |

One incidental finding, kept because it cost time: `sg.Tensor(x)` on a float64 array returns a
**float32** tensor — the default dtype applies at construction. The first version of the
forward/reverse agreement check read `3.2e-8` and looked like a real disagreement; it was float32
rounding. Anything comparing gradients needs `dtype=np.float64` passed explicitly, or
`sg.default_dtype(np.float64)` around the block. See `09`.
