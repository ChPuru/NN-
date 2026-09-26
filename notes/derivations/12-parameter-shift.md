# 12 — The parameter-shift rule

Verified by [`verify/verify_12_parameter_shift.py`](verify/verify_12_parameter_shift.py) — 10/10.

Used by `scratchgrad.quantum.Circuit.expval`, whose backward never differentiates through the
simulator. Everything here runs on a **classical** statevector simulation; the rule is interesting
because it is also what one would run on hardware, where the only thing you can get out of a
circuit is an expectation value.

## Setup

A circuit is `U = W · G(θ) · V`, where `G(θ) = exp(−iθP/2)` is the one gate that depends on `θ`,
`P` is a Pauli (`X`, `Y` or `Z`), and `V`, `W` collect everything before and after it. With
`|ψ⟩ = V|0⟩` and `M = W† O W` for the measured observable `O`:

```
E(θ) = ⟨ψ| G(θ)† M G(θ) |ψ⟩
```

## Why `E` is a single sinusoid

`P² = I`, so the exponential series splits into even and odd powers:

```
G(θ) = cos(θ/2) I − i sin(θ/2) P
```

Substitute and expand. Every term is a product of one factor from `G†` and one from `G`, so each
carries `cos²(θ/2)`, `sin²(θ/2)`, or `cos(θ/2) sin(θ/2)`. The half-angle identities turn those into
`(1 + cos θ)/2`, `(1 − cos θ)/2`, and `sin θ / 2`. Hence, whatever `ψ` and `M` are:

```
E(θ) = A + B cos θ + C sin θ
```

with `A, B, C` real constants (functions of `ψ`, `M`, `P`, not of `θ`). The frequency is exactly 1
because the generator `P/2` has eigenvalues `±1/2`, whose difference is 1. A three-term
least-squares fit of that form to `E` at 40 angles of a random two-qubit sandwich leaves a
residual of **5.6e-16** — rounding, nothing else.

## The rule

Differentiate: `E'(θ) = −B sin θ + C cos θ`. Now look at a symmetric difference of `E` at any
shift `s`:

```
E(θ+s) − E(θ−s) = B[cos(θ+s) − cos(θ−s)] + C[sin(θ+s) − sin(θ−s)]
                = −2B sin θ sin s + 2C cos θ sin s
                = 2 sin s · E'(θ)
```

So `E'(θ) = [E(θ+s) − E(θ−s)] / (2 sin s)` **exactly, for every `s`** that is not a multiple of
`π`. Nothing was Taylor-expanded; there is no truncation error to shrink. The verifier checks
`s ∈ {0.1, 1, π/2, 2.5}` against the fitted derivative: worst error **1.8e-15**.

`s = π/2` is the conventional choice because `sin s = 1` is the largest denominator — on hardware
each `E` is an average of shots, and a big denominator is what keeps the shot noise from being
amplified. That gives the familiar

```
∂E/∂θ = [E(θ + π/2) − E(θ − π/2)] / 2
```

It is not a finite difference with a large step. The same two numbers read as a central
difference with `h = π/2` are off by the factor `sin(h)/h = 2/π` — measured **0.636620**.

## Shared parameters: shift one occurrence at a time

If `θ` feeds `m` gates, the chain rule says `∂E/∂θ = Σ_k ∂E/∂φ_k`, where `φ_k` is the angle of
occurrence `k` with the others held fixed. Each term is a single-gate problem, so each gets its own
two-term shift, and the results are summed: `2m` circuit evaluations.

The subtly wrong version shifts `θ` itself — every occurrence at once. That is the derivative of a
function which is no longer a frequency-1 sinusoid, so the identity above does not apply. For
`RY(θ) RY(θ)` on `|0⟩`, `E = cos 2θ` and the whole-parameter shift gives
`[cos(2θ+π) − cos(2θ−π)]/2 = 0` at every `θ` (measured **1.7e-16** over 13 angles), while the
per-occurrence sum reproduces `−2 sin 2θ` to **5.0e-16**. The same summation is needed for data
re-uploading, where one trainable parameter can appear in several layers.

In `Circuit.jacobian` the per-occurrence derivatives are accumulated with `np.add.at`; a plain
fancy-indexed `+=` keeps only the last write to a repeated index and silently drops the rest.

## Traps

- **The half angle is part of the rule.** For a gate written `exp(−iθP)` (no `1/2`), `E` has
  frequency 2 and the `π/2` rule returns **exactly zero** for every `θ` (measured 1.7e-16). The
  correct rule there is shift `π/4`, factor 1 — measured exact to 5.6e-16. `scratchgrad.quantum`
  uses `R_P(θ) = exp(−iθP/2)` throughout.
- **Only for generators with two eigenvalues `±r`.** A controlled rotation's generator
  `|1⟩⟨1| ⊗ P/2` has eigenvalues `{0, ±1/2}`, so `E` mixes frequencies `1/2` and `1` and needs a
  four-term rule. The two-term rule on a controlled-`RY` sandwiched between random unitaries is off
  by **0.094** against central differences. (With nothing after the gate and an observable
  diagonal in the control, the two branches never interfere and the error vanishes — which is how
  a test can pass on a wrong rule.) No controlled rotations are offered for this reason.
- **Scaled angles need the chain rule.** If a gate's angle is `c·θ`, the shift is applied to the
  angle and the result multiplied by `c`. `Circuit` only allows `θ + x` (coefficient 1), so this
  never arises in the framework; it does in hand-written circuits.
- **Cost is linear in the number of parameterised gates.** `2m` evaluations for `m` occurrences,
  against reverse mode's constant factor. That is the price of needing only expectations. The
  implementation stacks all `2m` shifted circuits into one batch and simulates them in one pass.

## Cross-check against the framework

`Circuit.jacobian` on a 3-qubit circuit of `RX`/`RY`/`RZ`, `H`, `CNOT` and `CZ` gates, with four
parameters each appearing several times and two re-uploaded input features, agrees with central differences of
`Circuit.forward` (`h = 1e-6`) to **1.9e-10** relative error — the finite-difference noise floor,
not the rule's; the largest gradient entry is 0.69, so the check is not vacuous.
