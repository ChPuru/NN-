# 05 — Conv2d via im2col

Verified by [`verify/verify_05_conv2d.py`](verify/verify_05_conv2d.py) — 9/9.

## Setup

Deep learning's "convolution" is cross-correlation: no kernel flip. Layout is frozen —
input `(N, C, H, W)`, weight `(M, C, kH, kW)` with `M = out_channels`:

```
Y[n, m, i, j] = b_m + Σ_c Σ_a Σ_e  W[m, c, a, e] · X[n, c, i·s_H + a·d_H − p_top,
                                                          j·s_W + e·d_W − p_left]
```

with `s` stride, `d` dilation, `p` padding. Everything below is an exercise in noticing that this
is a **matrix product wearing six indices**.

### Output size

The dilated kernel spans `d(k−1) + 1` pixels, not `k`. The last valid window start is at
`size + p_total − span`, and starts step by `s`:

```
out = floor((size + p_before + p_after − (d(k−1) + 1)) / s) + 1
```

Dropping the dilation term is the common error and it produces a *larger* number, so nothing
raises until the reshape after the GEMM, by which point the cause is out of sight. `verify_05`
compares the formula against the shape `conv2d` actually returns over **81** `(k, stride, pad,
dilation)` combinations: 0 mismatches.

## im2col

Define `col = im2col(X)` of shape `(N, C·kH·kW, L)` with `L = H_out·W_out`, where the row index
runs `(c, a, e)` in C order and the column index runs `(i, j)`:

```
col[n, (c,a,e), (i,j)] = X[n, c, i·s + a·d − p, j·s + e·d − p]
```

Then with `Wmat = W.reshape(M, C·kH·kW)`:

```
Y[n] = Wmat @ col[n]        (M, CK) @ (CK, L) → (M, L),  reshaped to (M, H_out, W_out)
```

Two things this buys, and one it costs. It buys a BLAS-dispatched GEMM instead of a Python loop
nest, and it buys a backward that is two more GEMMs. It costs a `C·kH·kW/C = kH·kW`× memory
blow-up on the input — for a 3×3 kernel, 9×. The framework builds `col` with
`np.lib.stride_tricks.sliding_window_view`, so the *windowing* is free (a view); only the
`transpose(...).reshape(...)` materialises it.

The forward is checked against six explicit loops — a forward-value test, because gradcheck cannot
catch a wrong forward with a self-consistently derived backward. Measured: **1.76e-16**.

## The backward

`Y[n] = Wmat · col[n]` is bilinear, so both rules are the matmul rules from `01`, and the only work
is bookkeeping the batch axis.

**Bias.** `Y = … + b_m`, broadcast over `(n, i, j)`. Broadcast forward ⇒ sum backward:

```
db = Σ_{n,i,j} dY[n, m, i, j]
```

**Weight.** `Y[n,m,l] = Σ_r Wmat[m,r] col[n,r,l]`, so

```
∂L/∂Wmat[m,r] = Σ_n Σ_l dY[n,m,l] · col[n,r,l]
```

— a contraction over `l` *and* over the batch. Per sample it is `dY[n] @ col[n]ᵀ`; the batch sum is
the part BLAS cannot fold into matmul's broadcasting rules, so it is an explicit `.sum(axis=0)` (or
one `einsum`, which must be given `optimize=True` — the default is `False`, which never dispatches
to BLAS and costs 30–300×).

```
dW = Σ_n dY[n] · col[n]ᵀ       reshaped (M, CK) → (M, C, kH, kW)
```

**Input, part 1 — into the columns.** Same matmul rule, other side:

```
dcol[n] = Wmatᵀ · dY[n]        (CK, M) @ (M, L) → (CK, L)
```

**Input, part 2 — out of the columns.** This is the only genuinely new step. `im2col` is a linear
map `A: X ↦ col`; `dX` is `Aᵀ dcol`. From `01`: the adjoint of a *copy* is a *sum*. And `im2col`
is precisely a copy — every pixel `X[n,c,r,q]` is copied into one entry of `col` for **every
window that contains it**. So

```
dX[n,c,r,q] = Σ over all (a, e, i, j) that read this pixel  of  dcol[n, (c,a,e), (i,j)]
```

That scatter-with-accumulation is `col2im`. It is not "the inverse of im2col" and calling it that
is how the bug gets written: with `stride < kernel` the map is not injective, has no inverse, and
the adjoint genuinely adds.

```
dX = col2im(Wᵀ · dY)
```

### Why accumulation is not optional

The check that settles it: `col2im(im2col(X))` is **not** `X`. It is `X` multiplied elementwise by
the number of windows covering each pixel — because `AᵀA` is diagonal with the coverage counts on
it. At `k=3, s=1, p=1` the interior count is 9. Measured: `col2im(im2col(x)) == x * counts` to
**7.4e-17**, with a max count of 9 exactly as predicted.

The fully-fused version that *looks* right is

```python
acc[index_array] = values  # WRONG
```

NumPy fancy **assignment** keeps only the last write per destination, so every window but one is
discarded. Measured distance from the correct result: **0.8** relative. The framework instead runs
one vectorised `+=` per kernel tap:

```python
for i in range(kh):
    for j in range(kw):
        acc[:, :, r0:r1:sh, c0:c1:sw] += gwin[:, :, :, :, i, j]
```

Within a single tap the strided destination slice touches each pixel at most once, so the in-place
add is well defined; the accumulation that matters happens *across* taps, where separate statements
add rather than overwrite. The loop is over `kH·kW` taps only — never over batch or spatial
position. (`np.add.at` is also correct and much slower: it runs an unbuffered elementwise loop.)

## Groups, and the one reshape that has to be a view

With `groups = G`, the channel axis splits as `(G, C/G)` and the whole thing becomes a *batched*
GEMM: `(1, G, M/G, CK) @ (N, G, CK, L)`. This is legal precisely because `C` splits as `(G, C/G)` in
C order — consecutive channels share a group — which is the same reason the reverse reshape in the
`dX` rule is a free view rather than a copy. Splitting the other way is a different layer whose
checkpoints are not interchangeable.

## The adjoint test

For fixed `W`, `X ↦ conv2d(X, W)` is linear, so it admits the exact check from `01` and `09`:

```
⟨v, conv(u)⟩ = ⟨convᵀ(v), u⟩
```

No step size, no truncation error, no noise floor. Measured: **1.47e-15** — floating-point
summation order and nothing else. For a linear map this is a strictly sharper instrument than
finite differences, which on the same layer manage 5.2e-11.

## Small print

- **Never compute `dX` for the first convolution.** Nothing upstream consumes it, and the `col2im`
  scatter it avoids is the single largest saving available here. Hence
  `conv2d(..., input_requires_grad=False)`.
- **Max-pool backward must divide by the tie count.** Without it, tied maxima each receive the full
  upstream gradient — a different bug in the same file, and one that only shows on quantised or
  ReLU-saturated inputs where exact ties are common.

## Verification

| Check | Measured | Bound |
|---|---|---|
| output-size formula vs actual shape, 81 geometries | 0 mismatches | = 0 |
| im2col+GEMM forward vs six explicit loops | 1.76e-16 | ≤ 1e-14 |
| `dW = dY·colᵀ` | 2.32e-16 | ≤ 1e-14 |
| `dX = col2im(Wᵀ·dY)` | 0.0 | ≤ 1e-14 |
| `db` summed over `(N, H_out, W_out)` | 0.0 | ≤ 1e-15 |
| `col2im(im2col(x)) = x ⊙ count` (max count 9) | 7.44e-17 | ≤ 1e-15 |
| the overwriting version's error | 0.8 | ≥ 1e-2 |
| adjoint identity `⟨v, Au⟩ = ⟨Aᵀv, u⟩` | 1.47e-15 | ≤ 1e-12 |
| central differences, stride 2 / pad 1 / dilation 2 | 5.19e-11 | ≤ 1e-7 |
