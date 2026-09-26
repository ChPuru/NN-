# 10 — Linear-recurrence scan: minGRU and a selective SSM

Verified by [`verify/verify_10_parallel_scan.py`](verify/verify_10_parallel_scan.py) — 16/16.

## Setup

One recurrence, elementwise, over the time axis of a batch-first `(N, T, …)` input:

```
h_t = a_t ⊙ h_{t−1} + b_t          t = 1..T,  h_0 given (zero if absent)
```

The two layers built on it only differ in where `a` and `b` come from:

```
minGRU     k, h̃ = split(x W + b)          a = σ(−k) = 1 − z     b = z ⊙ h̃,  z = σ(k)
SSM        a = exp(Δ ⊙ A)  (ZOH)          b = Δ ⊙ x ⊗ B         h is (N, T, D, S)
```

Neither `a` nor `b` reads `h`. That is the whole trick: the gates for all `T` steps are one GEMM,
and what is left is a *linear* recurrence, which can be evaluated as a tree instead of a chain.

## Why it parallelises

Write one step as the affine map `h ↦ a h + b` and name it by its pair `(a, b)`. Doing `(a₁, b₁)`
then `(a₂, b₂)`:

```
a₂(a₁ h + b₁) + b₂  =  (a₂a₁) h + (a₂b₁ + b₂)       ⇒   (a₁,b₁) ∘ (a₂,b₂) = (a₂a₁, a₂b₁ + b₂)
```

Composition of maps is associative, so any bracketing gives the same `h_t`.
`src/scratchgrad/nn/scan.py::_scan` brackets it by **odd-even reduction**. It pairs step `2i+1`
with step `2i`, which gives a composite step from `h_{2i−1}` straight to `h_{2i+1}`. It then
solves that half-length problem recursively and fills in every even step with one more
multiply-add. Work is `T + T/2 + … = O(T)`, depth is `O(log T)`, and every level is one
vectorised NumPy pass. Against a Python loop, the error at `T = 4096` is **8.8e-17**
(float64).

## The two formulations that break

**Closed form.** Unrolling gives `h_t = A_t (h_0 + Σ_{s≤t} b_s / A_s)` with `A = cumprod(a)`, so
the whole thing is one `cumprod` and one `cumsum`. It fails the moment `A` underflows. `b/A`
becomes `inf`, then `A·inf = 0·inf = NaN`, and every later step is NaN too. With
`a ~ U(0.3, 0.7)` the first NaN comes at **step 118 in float32** and step 972 in float64, and
97% of a 4096-step output is non-finite. Nothing degrades before that: the output is exact right
up to the step where it turns into NaN.

**Log space** (Heinsen 2023): `log h_t = a*_t + logcumsumexp(log b − a*)` with `a* = cumsum(log a)`.
This cures the underflow and matches the loop to 1.5e-13 when `b > 0`. With a signed `b`, 99.9%
of the outputs are NaN, because `log` of a negative number does not exist. That is why the
paper's log-space minGRU forces the candidate through a positive `g()`, and why a Mamba input
term `Δ·B·x`, which takes both signs, would need complex logs. It also costs an `exp` and a `log`
per element to do what one multiply-add does.

The odd-even scan only multiplies and adds. When a product of small `a` underflows to zero, that
is the right answer, because the contribution really has decayed away. It never becomes half of
a `0·inf`. The scan still assumes `|a| ≤ 1`, and both layers guarantee that by construction. With
`|a| > 1` the partial products overflow, just as the recurrence itself would.

## The backward is the same scan, reversed

Let `g_t = ∂L/∂h_t` be the cotangent placed *directly* on output `t`. The total cotangent `λ_t`
also collects everything `h_t` influences later, and it influences later steps only through
`h_{t+1} = a_{t+1} h_t + b_{t+1}`:

```
λ_t = g_t + a_{t+1} λ_{t+1},        λ_T = g_T
```

That is a linear recurrence run backwards in time, so `_scan_adjoint` reverses `g`, shifts the
coefficients by one and calls `_scan`. Each input appears in exactly one step, so:

```
∂L/∂b_t = λ_t
∂L/∂a_t = λ_t ⊙ h_{t−1}         (h_{t−1} = h_0 at t = 1)
∂L/∂h_0 = a_1 ⊙ λ_1
```

**The shift is the trap.** The adjoint of `h_t` is carried back by `a_{t+1}`, the step that
consumes `h_t`. It is not carried back by `a_t`, the step that produced it. Using `a_t` gives
a gradient of the right shape, a model that trains, and an adjoint that is **56% wrong** in
relative terms. `verify_10` measures that error, and `test_linear_scan_backward_equals_unrolled_autograd`
pins the fused backward against autograd through `T` separate multiply-add nodes to 1e-13.

Checked against the framework, the three formulas above agree to **3.1e-17**. Checked against
central differences, they agree to **3.9e-11**.

## minGRU: the gate derivative

Here `a = σ(−k)`, `b = σ(k) h̃ = z h̃`, and `σ' = σ(1−σ)`, so `∂a/∂k = −z(1−z)` and
`∂b/∂k = z(1−z) h̃`. Using the scan gradients above:

```
dk = z(1−z) ⊙ (λ ⊙ h̃ − λ ⊙ h_{t−1})
dh̃ = z ⊙ λ
```

`_mingru_scan` fuses gating and scan into one node. The obvious composition would slice
`pre[..., :H]`, and a slice's backward scatters through `np.add.at`, which at training sizes costs
more than the scan does. The fused node matches the hand-derived formula to 1.1e-16.

`1 − z` is computed as `σ(−k)`. In float32, `σ(17)` rounds to exactly 1, so `1 − σ(17)` is
**exactly 0**. The true retention is `σ(−17) = 4.1e-8`, and the subtraction would erase it.

**Polarity.** minGRU's `z` gates the *new* candidate. `nn.GRUCell`'s `z` gates the *old* state
(the Cho/PyTorch convention in `06`). Moving weights between the two needs `z → 1 − z`.

## Selective SSM: discretisation and its backward

For each channel `d` and state `s`, `A = −exp(A_log)` is negative by construction and
`Δ = softplus(·)` is positive.

- **Zero-order hold** for the decay: `Ā = exp(ΔA)`. Over `Δ ∈ [1e-3, 100]` and `A ∈ [−16, −1]` it
  stays in `[0, 1)`, with max 0.999, so the recurrence is stable at any step size. Euler's
  `1 + ΔA` reaches `|·| = 1599` on the same grid.
- **`B̄ ≈ ΔB`** is the Euler input term, and Mamba uses it too. Exact ZOH is
  `(exp(ΔA) − 1)/A · B`. Their relative gap is `Δ|A|/2` to first order; the measured ratio is
  within 0.3% of that at `Δ|A| = 1e-2`.

The forward is `h = scan(exp(Δ⊗A), (Δx)⊗B)`. Take `λ` from the scan backward and let
`P = λ ⊙ h_{t−1} ⊙ exp(ΔA)` be the cotangent on the exponent `ΔA`. Let `Q = Σ_s λ B` be shared by
`x` and `Δ`. Then:

```
dx = Δ ⊙ Q
dΔ = Σ_s P ⊙ A  +  x ⊙ Q
dA = Σ_{n,t} P ⊙ Δ
dB = Σ_d λ ⊙ (Δ x)
dh_0 = exp(Δ_1 A) ⊙ λ_1
```

`selective_scan` passes gradcheck on all five at 3.6e-11. The readout `y = Σ_s C h + D x` is a
separate small node (`_ssm_readout`) so that the `(N, T, D, S)` product is never held in the graph.

## What it costs, honestly

NumPy already vectorises the plain sequential loop over `N × D`. On a short, wide sequence
(`N=64, T=128, D=256`, float32) the loop takes 6 ms and the odd-even scan takes 19 ms, so the loop
is 3× *faster*. The scan only wins once `T` is long compared with the width: 17 ms against 26 ms
at `T = 4096, D = 64`. The minGRU's speed-up over the Tier-4 LSTM comes from two other places:

- all gates are computed by one GEMM instead of `T` small ones;
- the recurrence is one graph node instead of `T`.

Two NumPy traps showed up while measuring:

- `(N, T, C) @ (C, K)` through NumPy's stacked-matmul path runs **~130× slower** than the same
  product reshaped to `(N·T, C) @ (C, K)` (0.89 s against 7 ms here). Every layer in `scan.py`
  flattens before its GEMM for this reason.
- `functional.gelu` evaluates `a**3` through a generic `pow` that costs more than the GEMM beside
  it, so `char_mingru.py` uses a ReLU MLP.

A third trap is in the machine itself. About 3 s into a sustained load, this laptop
power-throttles: a GEMM goes from 6 ms to 31 ms and an `exp` from 6 ms to 57 ms. Total wall-clock
from runs made at different times therefore compares nothing. `char_mingru.py` reports a
same-load ratio instead, timing one step of each model in turn.

## Results (`examples/char_mingru.py`, same split, val tokens and schedule as the LSTM)

Every run below trained for 400 iterations on 64 × 128 tokens with seed 0.

| model | params | val nats | same-load s/step (vs LSTM) |
|---|---|---|---|
| char-LSTM (Tier 4) | 944,577 | 1.6303 | — |
| minGRU, d=256, 2 blocks | 824,897 | **1.6282** | 3.46 vs 23.75 → 6.9× |
| selective SSM, d=128, S=8, 2 blocks | 320,833 | 1.8257 | 1.10 vs 3.68 → 3.4× |

On identical val tokens, minGRU ties the LSTM (−0.002 nats, one seed, well inside noise) and is
several times faster per step. The tiny SSM **loses by 0.195 nats**. It also has a third of the
parameters and no conv or gating branch, so the comparison says nothing about Mamba.
