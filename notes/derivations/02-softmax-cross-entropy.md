# 02 — Softmax cross-entropy

Verified by [`verify/verify_02_softmax_cross_entropy.py`](verify/verify_02_softmax_cross_entropy.py) — 8/8.

## Setup

The model emits **raw logits** `z ∈ R^{N×C}`. The target is an integer class index `y_n`. The loss
is the negative log-likelihood of the softmax, but it is never written that way:

```
L_n = logsumexp(z_n) − z_n[y_n]           where logsumexp(z) = log Σ_c exp(z_c)
L   = (1/N) Σ_n L_n                       for reduction='mean'
```

That this equals `−log p_{y}` is one line — `log p_y = z_y − logsumexp(z)` — but the two are not
the same *program*, and the difference is the whole point of the file.

## Deriving `∂L_n/∂z_n`

Two terms. The second is trivial: `∂(−z[y])/∂z_j = −[j = y]`, an indicator. The first is the
interesting one.

```
∂/∂z_j  log Σ_c exp(z_c)  =  (1 / Σ_c exp(z_c)) · ∂/∂z_j Σ_c exp(z_c)
                          =  exp(z_j) / Σ_c exp(z_c)
                          =  p_j
```

So the gradient of logsumexp *is* the softmax. (Which is the standard convex-duality fact —
logsumexp is the convex conjugate of negative entropy on the simplex — but the one-line
computation is all that is needed here.) Combining:

```
∂L_n/∂z_{n,j} = p_{n,j} − [j = y_n]
```

and with `reduction='mean'` the outer `1/N` rides along:

```
dz = (p − onehot(y)) / N
```

Three checks worth doing on the result before trusting it:

- **It sums to zero along `C`.** `Σ_j p_j = 1` and `Σ_j [j = y] = 1`. Correct: adding a constant to
  every logit of a row changes nothing about the softmax, so the gradient must be orthogonal to
  the all-ones direction.
- **It is bounded by 1 in absolute value**, whatever the logits are. The most a single logit can be
  told is "you are completely wrong" — this is what makes the log-sum-exp form numerically safe in
  a way the naive form is not.
- **At uniform logits it is `(1/C − onehot)/N`**, which is exactly the gradient of `ln C`, the loss
  at initialisation. `examples/mnist_mlp.py` asserts the forward half of this to 1e-12.

### Why not derive it in two steps

The textbook route is `∂L/∂p · ∂p/∂z`, which requires the softmax Jacobian
`∂p_i/∂z_j = p_i(δ_ij − p_j)`, and then:

```
∂L/∂z_j = Σ_i (−[i=y]/p_i) · p_i(δ_ij − p_j) = −[j=y] + p_j
```

Same answer, one whole Jacobian contraction later, and it needs `p_y` in a denominator — the exact
quantity that underflows. The fused form never divides by a probability. (`07` does need the
softmax Jacobian, because there the upstream cotangent is arbitrary rather than a one-hot; the
same collapse happens there for the same reason.)

### Label smoothing is the same derivation

With smoothing `ε`, the target distribution becomes `q = (1−ε)·onehot + ε/C`, and
`L_n = −Σ_c q_c log p_c`. Redoing the two lines above with `q` in place of the indicator:

```
dz = (p − q) / N
```

Measured against the framework at `ε = 0.1`: **exactly 0.0** relative error.

## What the code does

`src/scratchgrad/nn/functional.py::cross_entropy` computes `shift = z − max(z)`, `log_probs =
shift − log Σ exp(shift)`, gathers `−log_probs[rows, y]`, and returns a node whose backward is
`(p − soft) / denom`. One fused `_make`, not a composition. The row-max shift is what makes
`exp` safe: the largest exponent is exactly 0, so nothing overflows, and subtracting a constant
from a row is exactly the invariance noted above.

`denom` is the number of *contributing* rows, not the element count — they differ when
`ignore_index` is set, and dividing the forward by one while the backward divides by the other is
a gradient wrong by exactly the ignored fraction.

## The trap: `−log(p + ε)`

The widespread alternative computes probabilities first and guards the log:

```
L_naive = −log(p_y + ε)          typically ε = 1e-12
```

It is usually described as "for numerical stability". It is the opposite. Differentiate it:

```
∂L_naive/∂z_j = −(1/(p_y + ε)) · ∂p_y/∂z_j
              = −(1/(p_y + ε)) · p_y (δ_jy − p_j)
              = −(p_y / (p_y + ε)) · (δ_jy − p_j)
                 └──── prefactor ────┘   └── the correct gradient ──┘
```

The correct gradient, multiplied by `p_y/(p_y + ε)`. That prefactor is ≈ 1 when the model is
roughly right (`p_y ≫ ε`) and **goes to zero** when the model is badly wrong (`p_y ≪ ε`). The fudge
switches off learning for precisely the samples that have the most to teach — and it does it
silently, because the reported loss saturates at `log(1/ε)` and simply looks like a plateau.

Measured on one deliberately wrong sample, with the true class `60` nats below the winner:

| Quantity | Stable form | `−log(p + 1e-12)` |
|---|---|---|
| loss | 60.0 | **27.6310** (= `log(1/ε)`) |
| `∂L/∂z_y` (central differences) | −1.0 | **0.0** |

`p_y = 8.76e-27`, so the prefactor is `8.76e-15` — the gradient is not merely small, it is below
the representable resolution of the difference quotient and comes out as an exact zero. The gap
does not need to be 60 nats: the prefactor is already 0.5 at `p_y = ε`, i.e. a gap of 27.6, and
badly initialised nets produce those routinely.

The forward damage is worth naming separately, because it is what makes the bug survive review:
the *loss curve looks fine*. It is bounded by 27.63 and decreasing. Nothing plots as broken.

## Verification

| Check | Measured | Bound |
|---|---|---|
| forward = `mean(logsumexp(z) − z_y)` | 0.0 | ≤ 1e-15 |
| framework `dz` = `(p − onehot)/N` | 0.0 | ≤ 1e-15 |
| `reduction='sum'` drops the `1/N` | 0.0 | ≤ 1e-15 |
| label smoothing: `dz = (p − q)/N` | 0.0 | ≤ 1e-15 |
| central differences (`gradcheck`, float64) | 2.69e-11 | ≤ 1e-9 |
| clipped loss capped at `log(1/ε)` | 27.6310 | = 27.6310 |
| clipped gradient on a 60-nat-wrong sample | 0.0 | ≤ 1e-13 |
| stable gradient on the same sample | −1.0 | \|·+1\| ≤ 1e-9 |
