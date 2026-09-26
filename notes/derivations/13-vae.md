# 13 — Variational autoencoder

Verified by [`verify/verify_13_vae.py`](verify/verify_13_vae.py) — 13/13.
Code: [`src/scratchgrad/nn/vae.py`](../../src/scratchgrad/nn/vae.py).

## The bound

A latent-variable model `p(x) = ∫ p(x|z) p(z) dz` has an integral nobody can do. Bring in any
distribution `q(z|x)` and multiply by one:

```
log p(x) = log E_q[ p(x|z) p(z) / q(z|x) ]
        ≥ E_q[ log p(x|z) ] + E_q[ log p(z) − log q(z|x) ]          (Jensen: log is concave)
        = E_q[ log p(x|z) ] − KL(q(z|x) ‖ p(z))                        =: ELBO(x)
```

The slack is not mysterious: `log p(x) − ELBO(x) = KL(q(z|x) ‖ p(z|x))`, the distance from the
encoder to the true posterior. Maximising the ELBO over the decoder raises `log p(x)`; maximising
it over the encoder closes the gap.

The loss is `−ELBO`, and its unit is **nats per image**. For a Bernoulli decoder with logits `l`,

```
−log p(x|z) = Σ_i [ log(1 + e^{l_i}) − x_i l_i ]      summed over all D = 784 pixels
```

which is `binary_cross_entropy_with_logits(..., reduction="sum")`. The KL is summed over the latent
dimensions. *Then* both are averaged over the batch — once, together.

## Reparameterisation

The ELBO's first term is `E_{q_φ}[f(z)]` and we need `∇_φ` of it. The sampling step has no
derivative. Two ways round:

- **Score function** (REINFORCE): `∇_φ E_q[f] = E_q[ f(z) ∇_φ log q_φ(z) ]`. Unbiased, works for any
  `q`, but `f` enters as a raw multiplier and the variance is large.
- **Pathwise**: write `z = μ + σ ⊙ ε`, `ε ~ N(0, I)`. The randomness now lives in a parameter-free
  `ε`, so `E_q[f(z)] = E_ε[f(μ + σε)]` and the gradient moves inside: `E_ε[ ∇_z f · ∂z/∂φ ]`.

With the encoder emitting `lv = log σ²` (so `σ > 0` without a constraint), `σ = exp(lv/2)` and

```
∂z/∂μ  = 1
∂z/∂lv = ½ exp(lv/2) ε = ½ (z − μ)
```

so the VJP is `dμ = dz`, `dlv = dz ⊙ ½(z − μ)`. That is the whole fused op.

How much the pathwise form buys, on the smallest possible case — `d/dμ E_{z~N(μ,1)}[z²]` at `μ = 1`,
true value `2μ = 2`. Pathwise: `2z`, variance `4`. Score function: `z²(z − μ)`; with `z = 1 + e`,
`Var = E[(1+e)⁴ e²] − 4 = (1 + 18 + 15) − 4 = 30`. Measured over 400k draws: **3.997** and
**29.75**, ratio **7.44** against a predicted 7.5, both means within 1.9 standard errors of 2.

## The KL in closed form

For one dimension, `q = N(μ, σ²)`, `p = N(0, 1)`:

```
log q(z) − log p(z) = −½ lv − (z − μ)² / (2σ²) + z² / 2
E_q[ · ]            = −½ lv − ½ + ½ (σ² + μ²)
KL                  = ½ (μ² + e^{lv} − 1 − lv)
```

Summed over dimensions for a diagonal Gaussian. Gradients:

```
∂KL/∂μ  = μ
∂KL/∂lv = ½ (e^{lv} − 1)
```

Both vanish at `μ = 0, lv = 0`, and `∂²KL/∂lv² = ½ e^{lv} > 0`, so the prior is the unique minimum:
KL is exactly `0.0` there (measured, not `≈ 0`). Using the closed form rather than a one-sample
estimate `log q(z) − log p(z)` removes the KL's share of the gradient noise entirely.

Independent check: `E_q[log q − log p]` estimated from the log-densities (not from the formula)
with 400k samples lands **1.05 standard errors** from the closed form (4.6285 vs 4.6224, se 0.0058).

## Traps

**`σ = exp(lv/2)`, not `exp(lv)`.** Using `exp(lv)` as the std means the sample's spread and the
KL's idea of it disagree — the KL pays for `σ²` while the sampler draws with `σ⁴`. Nothing raises
and it still trains. `tests/test_vae.py` pins the second moment of 200k samples to `e^{lv}` within
six standard errors; the wrong version misses by 172% at `lv = 1`.

**The reduction is part of the objective.** Most losses default to `reduction="mean"`, which
averages the reconstruction over pixels *and* images. That is exactly `recon / 784` (measured:
the ratio is `784` to machine precision) while the KL keeps its full weight — a β-VAE with
`β = 784`, without anyone having chosen one. The optimiser then prefers to pay no KL at all: the
latent dimensions collapse onto the prior (`q(z_j|x) = N(0,1)` for every `x`), and every
reconstruction becomes the same blurry average digit. Measured on MNIST below: after 5 epochs,
**0 of 20** latent units active and a KL of **3.5e-5** nats. The mirror mistake — reconstruction
summed, KL averaged over latent dimensions instead of summed — down-weights the KL by `1/20`
instead; prior samples should then degrade because the aggregate posterior drifts from `N(0, I)`
(not measured here).

**`e^{lv} − 1 − lv` cancels exactly where it matters.** A collapsed dimension sits at `lv ≈ 0`, where
the true value is `≈ lv²/2`. In float32 at `lv = 1e-4` the naive form has relative error **2.32**
(it is noise); `expm1(lv) − lv` has **1.8e-3**. The gradient `½(e^{lv} − 1)` suffers far less
(one cancellation, not two), so this corrupts the reported per-dimension KL more than training.

**Hoist `ε` out of a gradcheck.** Drawing `ε` inside the checked function gives every
finite-difference probe a different forward pass, and the "numeric gradient" is noise. The
verifier and tests pass `eps=` explicitly; `reparameterize` has no hidden global stream at all —
it demands exactly one of `rng=` or `eps=`.

**Soft targets are not a likelihood.** Feeding MNIST intensities in `[0, 1]` straight into the
Bernoulli BCE trains and draws nice pictures, but `Σ_x p(x)` over `x ∈ [0, 1]` is not 1, so the
number is not a log-likelihood bound and cannot be compared with binarised-MNIST results. The
example binarises dynamically instead: each training pixel is redrawn as `Bernoulli(intensity)`
every time it is seen; validation and test are binarised once with fixed seeds.

## Measured

| Check | Measured | Bound |
|---|---|---|
| `dz/dμ = 1` vs the framework backward | 0 | ≤ 1e-15 |
| `dz/dlv = ½(z − μ)` vs the framework backward | 2.8e-17 | ≤ 1e-15 |
| reparameterise, central differences (ε fixed) | 6.6e-11 | ≤ 1e-9 |
| KL at the prior | 0.0 | == 0 |
| KL gradients `μ`, `½(e^{lv} − 1)` vs the framework | 8.9e-17 | ≤ 1e-15 |
| KL, central differences | 2.2e-11 | ≤ 1e-9 |
| closed form vs Monte Carlo (standard errors) | 1.05 | ≤ 5 |
| both estimators unbiased (standard errors) | 1.84 | ≤ 5 |
| score-function / pathwise variance | 7.44 | ≥ 5 (predicted 7.5) |
| naive float32 KL term at `lv = 1e-4`, rel. error | 2.32 | ≥ 0.1 |
| `expm1` float32 KL term at `lv = 1e-4`, rel. error | 1.8e-3 | ≤ 1e-2 |
| `reduction="sum"` / `"mean"` | 784 | == 784 |
| whole −ELBO through a tiny VAE, central differences | 3.3e-10 | ≤ 1e-7 |

### MNIST ([`examples/vae_mnist.py`](../../examples/vae_mnist.py), seed 0)

784 → 400 → 20 → 400 → 784, 652,824 parameters, Adam `lr = 1e-3`, batch 128, 20 epochs on 50k
dynamically binarised training images; 10k held out from the training set for validation; the
test set (binarised once, fixed seed) evaluated exactly once, with one `z` sample per image.

| Run | −ELBO (nats/image) | recon | KL | active units |
|---|---|---|---|---|
| correct reduction, test, epoch 20 | **101.14** | 73.83 | 27.31 | 20/20 |
| correct reduction, validation, epoch 5 | 108.35 | — | — | 20/20 |
| `reduction="mean"` trap, validation, epoch 5 | 208.61 | 208.61 | 3.5e-5 | 0/20 |

A unit is active when `Var_x(E[z_j|x]) > 0.01` (Burda et al., 2016). For scale: published
one-stochastic-layer VAEs on dynamically binarised MNIST reach about 85–90 nats, but only after
thousands of epochs and measured with an importance-weighted bound. This is a 20-epoch
single-sample ELBO, so it is a looser number by construction.

Timing: without other load an epoch takes 5.7–6.9 s (about 15 ms per step with
`OPENBLAS_NUM_THREADS=4`), so 20 epochs is about 2.5 minutes. The recorded run took **792 s**
because three other training jobs were sharing the CPU and epochs stretched to 40–66 s. The example
sets `OPENBLAS_NUM_THREADS=4` by default (`setdefault`, so an explicit value wins), because with
OpenBLAS's default of 20 threads the same step took 75 ms on this machine, 5× slower. The thread
count also changes float32 summation order: epoch 1 validation reads 123.25 with 4 threads and
123.26 with the default thread count.

Pictures: `assets/vae_samples.png` shows decoder means for `z ~ N(0, I)`, and
`assets/vae_recon.png` shows validation inputs over their reconstructions at `z = μ`.
`assets/vae_trap_recon.png` is the same figure for the trap run: sixteen identical grey blobs.
