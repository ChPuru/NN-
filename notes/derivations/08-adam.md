# 08 — Adam's bias correction

Verified by [`verify/verify_08_adam.py`](verify/verify_08_adam.py) — 10/10.

Not a backward pass, but the same kind of object: a short piece of algebra that is quoted far more
often than it is derived, and misquoted by a factor of 10 when it is.

## Setup

```
m_t = β₁ m_{t−1} + (1 − β₁) g_t            first moment,  m_0 = 0
v_t = β₂ v_{t−1} + (1 − β₂) g_t²           second moment, v_0 = 0
m̂_t = m_t / (1 − β₁ᵗ)
v̂_t = v_t / (1 − β₂ᵗ)
θ_t = θ_{t−1} − lr · m̂_t / (√v̂_t + ε)
```

Defaults `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`.

## Where `1 − βᵗ` comes from

Unroll the recursion from `m_0 = 0`:

```
m_t = (1 − β₁) Σ_{k=1..t} β₁^{t−k} g_k
```

Now assume the gradients are **stationary** — same distribution at every step, `E[g_k] = E[g]`.
That assumption is doing real work and Kingma & Ba say so; it is why bias correction is a
correction for the *initialisation*, not for anything about the loss surface. Taking expectations:

```
E[m_t] = (1 − β₁) E[g] Σ_{k=1..t} β₁^{t−k}
       = (1 − β₁) E[g] · (1 + β₁ + … + β₁^{t−1})
       = (1 − β₁) E[g] · (1 − β₁ᵗ)/(1 − β₁)
       = (1 − β₁ᵗ) · E[g]
```

The `(1 − β₁)` cancels exactly against the geometric sum. So `m_t` is a **biased** estimator of
`E[g]`, low by the factor `(1 − β₁ᵗ)`, and dividing by that factor makes it unbiased. Identically
for `v` with `β₂` and `E[g²]`.

The bias is not small early on: at `t = 1`, `1 − β₁ = 0.1` and `1 − β₂ = 0.001`. The first `m` is
ten times too small and the first `v` is a thousand times too small.

Two measurements. With a **constant** gradient the expectation is exact, and `m_t = (1 − β₁ᵗ)g`,
`v_t = (1 − β₂ᵗ)g²` hold to **7.9e-17** and **5.3e-17** over ten steps. With **random** stationary
gradients (`g ~ N(1, 1)`, 200 000 independent chains, `t = 5`), the corrected estimator's sample
mean lands **1.01e-3** from `E[g] = 1` — inside the ~9e-4 standard error of the Monte Carlo, which
is what "unbiased" looks like when you measure it rather than assert it.

## The invariant a correct implementation has

At `t = 1` with any constant gradient `g`:

```
m̂_1 = (1−β₁)g / (1−β₁) = g
v̂_1 = (1−β₂)g² / (1−β₂) = g²
step = lr · g / (|g| + ε) ≈ lr · sign(g)
```

**The first step has magnitude `lr`, for any gradient scale.** That is the signature of correct
bias correction, and it is one line to test. Measured across `g ∈ {1e-3, 1, 1e3}`: `0.999990`,
`1.000000`, `1.000000` times `lr`. (The 1e-5 deficit at `g = 1e-3` is `ε` in the denominator doing
its job, not an error.)

## The three ways to get it wrong

All three still descend. Their exact first-step factors relative to correct Adam:

### Omitting correction entirely — **3.16×**, not 31.6×

```
step_none / step_correct = [ m/(√v + ε) ] / [ (m/(1−β₁)) / (√(v/(1−β₂)) + ε) ]
                         ≈ (1 − β₁) · √(1/(1 − β₂))          (ε negligible at |g| = 1)
                         = 0.1 / 0.0316…
                         = 3.1623
```

The numerator is 10× too small *at the same time* as the denominator is 31.6× too small, and the
two errors mostly cancel. Measured: **3.16228**, matching `(1−β₁)/√(1−β₂)` to 3.1e-7 relative
(the residual is `ε`).

The "31.6×" figure gets attached to this mistake constantly. It belongs to a different one.

### Correcting `m` but forgetting `v` — **31.62×**

```
step_m_only / step_correct = √(1/(1 − β₂)) = 1/0.0316… = 31.6228
```

This is the genuinely explosive case: a first step 31.6 lr wide. It is also the *easier* mistake to
make, because `m̂ = m/(1−β₁ᵗ)` is the line people remember and `v̂` is the one they forget. Measured:
**31.6228**, to 3.1e-7.

### Correcting `v` but forgetting `m` — **0.1×**

```
step_v_only / step_correct = (1 − β₁) = 0.1
```

Ten times too small. Measured: **0.100000**, to 4.4e-16. This one does not blow up; it just makes
the first ~40 steps (until `β₁ᵗ` decays) quietly ineffective, which is the hardest of the three to
notice.

The whole point of writing all three out: the factor depends on *which half* you forgot, and only
one of the three arrangements is the famous number.

## Where `ε` goes

`ε` is **outside** the square root and **after** bias correction:

```
m̂ / (√v̂ + ε)              correct
m̂ / √(v̂ + ε)              wrong
```

The wrong form floors the denominator at `√ε = 1e-4` for the default `ε = 1e-8`. Any gradient
smaller than about `1e-4` then gets a step proportional to `g` — SGD-like — instead of the
sign-like step Adam exists to take. That is four orders of magnitude too early, and precisely in
the small-gradient regime where Adam earns its keep.

Measured at `t = 1`:

| `\|g\|` | correct step | `ε` inside the root | ratio |
|---|---|---|---|
| 1 | 9.99e-4 | 9.99e-4 | 1 − 5e-9 |
| 1e-5 | 9.99e-4 | **9.95e-5** | **0.0996** |

At `|g| = 1` the two agree to 5e-9. That is why the bug survives review: it is invisible at the
gradient scale anyone spot-checks, and a factor of 10 exactly where it matters.

`optim.Adam`'s own first step was compared against the closed form `lr·m̂/(√v̂+ε)`: **0.0** relative
error, in float64. (In float32 it reads 5.7e-8 — `Parameter` casts to the default dtype and the
optimizer casts its update back, so any comparison at this resolution needs
`sg.default_dtype(np.float64)` around it. See `09`.)

## AdamW, in one paragraph

Coupled L2 (`grad += wd·θ`) sends the decay through both moment estimates, and `1/√v̂` then divides
most of it back out — for exactly the large-gradient weights you wanted to shrink. Decoupled decay
(`θ -= lr·wd·θ`) applies it straight to the parameter, so the amount decayed is independent of the
gradient history. They are different optimizers and do not coincide at any learning rate; only
`wd = 0` makes them agree. That is why `AdamW`'s default `weight_decay` is 1e-2, a hundred times
`Adam`'s — decoupling removed the factor that was quietly shrinking it.

## Verification

| Check | Measured | Bound |
|---|---|---|
| `m_t = (1 − β₁ᵗ)g` over 10 steps | 7.93e-17 | ≤ 1e-15 |
| `v_t = (1 − β₂ᵗ)g²` over 10 steps | 5.31e-17 | ≤ 1e-15 |
| `E[m₅]/(1 − β₁⁵) = E[g]`, 200k draws | 1.01e-3 | ≤ 5e-3 |
| `optim.Adam` first step = `lr·m̂/(√v̂+ε)` | 0.0 | ≤ 1e-15 |
| first step = `lr` for `g ∈ {1e-3, 1, 1e3}` | 1.0e-5 | ≤ 1e-4 |
| no correction: 3.16228 vs `(1−β₁)/√(1−β₂)` | 3.06e-7 | ≤ 1e-6 |
| `m` only: 31.6228 vs `1/√(1−β₂)` | 3.06e-7 | ≤ 1e-6 |
| `v` only: 0.100000 vs `(1−β₁)` | 4.44e-16 | ≤ 1e-6 |
| `ε` inside the root at `\|g\|=1e-5` | 0.0996× | ≤ 0.2 |
| …and at `\|g\|=1` | 5e-9 | ≤ 1e-7 |
