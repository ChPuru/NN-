# 04 — BatchNorm

Verified by [`verify/verify_04_batchnorm.py`](verify/verify_04_batchnorm.py) — 8/8.

Read [`03-layernorm.md`](03-layernorm.md) first. The backward is the *same derivation*; only the
group changes. What is genuinely new here is the running statistics, and the fact that BatchNorm is
two different functions depending on `training`.

## Setup

For a `(N, C, H, W)` input the group of a channel `c` is every element that shares it: all `N·H·W`
of them. The reduction axes are `(0, 2, 3)` — everything except the channel.

```
n  = N·H·W                          the group size
μ_c = (1/n) Σ x                     over (N, H, W)
v_c = (1/n) Σ (x − μ_c)²            BIASED
x̂  = (x − μ_c)/sqrt(v_c + ε)
y  = g_c x̂ + b_c
```

The parameters `g, b` are per-channel and broadcast over `(N, H, W)`, so by `01`'s
broadcast-⇒-sum rule their gradients are sums over those same axes.

## The backward is 03's, with different axes

Nothing in `03`'s derivation used the fact that the group was a row. Steps 0–3 go through verbatim
with `D → n` and the mean taken over `(0, 2, 3)`:

```
dx̂ = dy ⊙ g
dx = istd · (dx̂ − mean(dx̂) − x̂ ⊙ mean(dx̂ ⊙ x̂))          means over (0, 2, 3)
dg = Σ_{N,H,W} dy ⊙ x̂
db = Σ_{N,H,W} dy
```

This is why `src/scratchgrad/nn/norm.py` has exactly one `_normalize` kernel that LayerNorm,
GroupNorm, RMSNorm and BatchNorm all call with different `axes`. Measured against the framework:
`dx` at **8.4e-17**, `dg` and `db` at **0.0**.

The consequence people find surprising is worth stating plainly: **the gradient of one sample
depends on every other sample in the batch.** The `mean(dx̂)` and `mean(dx̂·x̂)` terms run over the
batch axis. BatchNorm is not a per-sample function and its backward cannot be.

## The frozen-statistics path is a different function

In `eval()` the layer normalises with the stored `running_mean` / `running_var`. Those are
**constants** — buffers, not parameters, with no graph behind them. The coupling between group
members disappears entirely and the backward collapses to one term:

```
dx = dx̂ · istd            (frozen statistics)
```

`_normalize`'s `stats is not None` branch is exactly this. Reaching for that branch *during
training* is the classic BatchNorm bug: the layer still normalises, the loss still descends, and
the gradient is simply a different function of the input. Measured distance between the two on the
same batch: **0.097** relative — a ten-percent error in every input gradient, invisible in a loss
curve.

Two directions to get right, in the same breath:

- `train()` normalises with **batch** statistics *and* updates the running ones.
- `eval()` normalises with the **running** statistics and updates nothing.

Normalising with running statistics during training also decouples the output from the batch the
gradient is being computed for, and at step 0 (running mean 0, var 1) makes the layer a near no-op.

## Why the running statistics use the *unbiased* variance

This is the part that looks like a nitpick and is not.

The batch normalisation itself uses the biased variance. That is not a statistical claim; it is a
*definitional* one. `x̂ = (x − μ̂)/sqrt(v̂_biased + ε)` is the transform whose output has exactly unit
sample variance on this batch, which is what the layer is for. Bessel's correction would make the
output's sample variance `(n−1)/n`, which is not the intent.

The running buffers are doing something different: they are estimating the **population**
variance of the pre-activation, to be used later on data the layer has never seen. And there the
biased estimator is biased — `E[v̂_biased] = ((n−1)/n)·σ²` — so the estimate that should be
accumulated is `v̂_biased · n/(n−1)`.

Use one estimator for both and the layer's train and eval outputs differ by a systematic
`sqrt(n/(n−1))` factor. At the small batches BatchNorm is used with, that is not a rounding
concern. `verify_04` measures it at `n = 8`: **14.3%**, four orders of magnitude above `ε = 1e-5`.

`_update_running_stats` therefore applies the correction in exactly one place:

```python
rv *= 1.0 - mom
rv += mom * var_biased.reshape(-1) * (n / (n - 1.0))
```

Reading which estimator was actually stored is easy if you pick the right momentum: at
`momentum=1.0` the EMA keeps none of its old value, so after one forward the buffer *is* this
batch's estimate. Measured: `running_var` equals `np.var(x, ddof=1)` to **4.2e-17**, and differs
from `ddof=0` by 6.7e-2 relative.

That momentum trick is also the answer to a testing trap named in the roadmap: train/eval
agreement is **not** a valid test of a standard EMA implementation, because at step 1 the running
stats are still mostly the initial `(0, 1)`. Either set `momentum=1.0` with a matched estimator, or
assert convergence over ≥200 forward passes.

## Small print

- The running buffers go in `register_buffer`, never in a `Parameter`. A `Parameter` is handed to
  the optimizer, which would apply momentum and weight decay to a running mean.
- `num_batches_tracked` is an `int64`, because with `momentum=None` it is the exact denominator of
  a cumulative moving average and must not drift.
- Training with `n = 1` raises rather than warns: the unbiased running variance is undefined.

## Verification

| Check | Measured | Bound |
|---|---|---|
| train forward uses the biased variance | 0.0 | ≤ 1e-15 |
| three-term `dx` over axes (0,2,3) vs framework | 8.43e-17 | ≤ 1e-14 |
| `dg`, `db` summed over (N,H,W) | 0.0 | ≤ 1e-15 |
| central differences through the batch-statistics path | 9.34e-11 | ≤ 1e-7 |
| `running_var` = `var(x, ddof=1)` after one `momentum=1` step | 4.23e-17 | ≤ 1e-15 |
| …and differs from `ddof=0` (14.3% at n=8) | 6.67e-2 | ≥ 1e-3 |
| frozen-stats backward vs the true training gradient | 9.73e-2 | ≥ 1e-2 |

`check_module` calls `module.eval()` to keep the forward deterministic, so gradchecking the
*training* path needs `track_running_stats=False` — with no buffers to fall back on, `eval()` uses
batch statistics too. That configuration is a genuinely different model at test time, not merely an
untracked one, which is worth knowing independently of the test.
