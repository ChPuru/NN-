# 07 — Scaled dot-product attention

Verified by [`verify/verify_07_attention.py`](verify/verify_07_attention.py) — 17/17.

## Setup

`Q` is `(…, T, d)`, `K` is `(…, S, d)`, `V` is `(…, S, d_v)`. Leading axes are batch and head.

```
S_raw = Q Kᵀ · scale             scale = 1/√d
S     = where(keep, S_raw, sentinel)          optional mask; True means "may attend"
A     = softmax(S, axis = −1)                 rows sum to 1
Out   = A V
```

## The softmax VJP, and why it collapses

Everything hard is in the softmax. For one row `a = softmax(s)`:

```
∂a_i/∂s_j = a_i (δ_ij − a_j)
```

(One line: `a_i = e^{s_i}/Σ_c e^{s_c}`; differentiating the quotient gives
`δ_ij a_i − a_i a_j`.) In matrix form the Jacobian is `J = diag(a) − a aᵀ`, symmetric and
positive-semidefinite, with `J·1 = 0` — the row-sum constraint showing up as a null direction.

Building `J` explicitly costs `S²` per row and a matvec per row. But the VJP never needs `J`:

```
(Jᵀ da)_j = (J da)_j = Σ_i (diag(a) − a aᵀ)_{ji} da_i
                     = a_j da_j − a_j Σ_i a_i da_i
                     = a_j ( da_j − ⟨a, da⟩ )
```

so

```
ds = A ⊙ (dA − Σ_last(dA ⊙ A))          one elementwise pass, one row-sum
```

The dense `S × S` Jacobian collapses to a rank-1 correction because `a aᵀ` *is* rank 1. Same
structure as `02`'s `p − onehot`, and the same reason: a softmax's Jacobian only ever subtracts the
component of the incoming cotangent along `a`.

`verify_07` constructs `diag(a) − a aᵀ` densely per row and contracts it, then compares with the
collapsed expression: **9.01e-17**.

## The three gradients

With `dOut` the incoming cotangent, `Out = A V` and `S = QKᵀ·scale`:

```
dA = dOut Vᵀ                                        (…, T, S)
dV = Aᵀ dOut                                        (…, S, d_v)

ds = A ⊙ (dA − Σ_last(dA ⊙ A))                      softmax VJP, above
ds = where(keep, ds, 0) · scale                     mask, then the scale

dQ = ds K
dK = dsᵀ Q
```

`dQ` and `dK` are just the matmul rules from `01` applied to `S = QKᵀ`: `∂L/∂Q = (∂L/∂S)K` and
`∂L/∂K = (∂L/∂S)ᵀQ`. `dV` likewise from `Out = AV`.

Two details in the implementation worth the words:

- `ds` is **shared** by the `Q` and `K` rules, so `src/scratchgrad/nn/attention.py` caches it
  against the cotangent *object* (identity, not `id()`, which the GC may recycle onto a different
  array) and does the softmax-Jacobian work once per backward instead of twice.
- The mask zeroing looks redundant — `A` is already exactly 0 where the sentinel landed, so `ds` is
  too. It is not redundant on a **fully-masked row**, where `A` comes back uniform `1/S` and this
  line is what stops a meaningless gradient reaching `Q` and `K`.

Measured against the framework, plain and causal: `dQ`, `dK`, `dV` all **exactly 0.0** relative
error; `gradcheck` on the fused node, 8.53e-11.

## Why `1/√d`, and why on the scores

Take `q` and `k` with i.i.d. zero-mean unit-variance entries. Then

```
E[q·k] = 0
Var(q·k) = Σ_{c=1..d} Var(q_c k_c) = d          (independent terms, each of variance 1)
```

The scores grow as `√d` in standard deviation. Dividing by `√d` fixes the variance at 1
independently of head width. Measured over 20 000 samples:

| `d` | `Var(q·k)` | `Var(q·k/√d)` |
|---|---|---|
| 64 | **63.90** | **0.998** |
| 11 | — | **1.006** |

What goes wrong without it is not "the softmax is a bit peaky". As the scores spread, `A` goes
one-hot, and `J = diag(a) − a aᵀ → 0`. A one-hot row has *zero* Jacobian: no gradient reaches `Q`
or `K` at all. The layer is dead, not slow. A one-number proxy is `trace(J) = 1 − Σa²`, which is 0
for a one-hot row and `1 − 1/S` for a uniform one. At `d = 128, S = 64`:

| | mean max weight | mean `trace(J)` |
|---|---|---|
| unscaled | **0.851** | **0.2145** |
| scaled by `1/√d` | **0.103** | **0.9622** |

4.5× more gradient through the softmax, and the effect worsens with `d` — which is exactly the
direction models grow.

**On the scores, before the softmax.** Scaling `Out` instead is a different function: softmax is not
homogeneous, `softmax(αs) ≠ softmax(s)`, so the two cannot be reconciled by any downstream constant.

## Why `finfo.min/2` and never `−inf`

The softmax subtracts the row max before exponentiating — that is what makes it safe. On a row that
is masked *everywhere* (a fully-padded position, a cross-attention query that legitimately sees
nothing) the max is itself `−inf`, and the shift computes

```
(−inf) − (−inf) = NaN
```

which spreads through the whole batch on the next matmul. Measured: all 4 entries of a
fully-masked 4-wide row come back NaN.

The popular patch — guard the max with `np.where(np.isfinite(m), m, 0)` — does **not** fix it. The
shift is now finite, but every entry is still `−inf`, `exp` gives an all-zero row, and the
normalisation is `0/0`. Measured: still 4 NaNs, one step later.

A large finite negative avoids both. The row's scores are equal and hugely negative, they cancel
*exactly* in the shift, and the row comes out as a uniform `1/S` average of `V`. Measured deviation
from uniform: **0.0**. That value is meaningless — but the input was meaningless, and it is
*finite*, so it can be dropped downstream by a loss mask instead of poisoning every parameter.

The `/2` is not decoration. Masks are combined additively in other implementations, and
`finfo.min + finfo.min` overflows to `−inf`, reintroducing precisely the bug the sentinel exists to
prevent. Halving leaves the headroom.

One more constraint this puts on testing, from the API contract: **never gradcheck an objective
with a huge magnitude.** Masking with `−1e9` makes `⟨v, out⟩ ~ 1e9`, and the finite-difference
noise floor — set by the magnitude of the objective (`09`) — rises to about 5e-3. Gradcheck the
mask *through the softmax*, where the sentinel has already vanished.

## Small print

- `causal_mask(q_len, kv_len)` is offset by `kv_len − q_len`, not pinned to the main diagonal.
  During incremental decoding the query block is the *tail* of the sequence while the keys are the
  whole cache, so query `i` sits at absolute position `kv_len − q_len + i`. With `k=0` most of the
  context would be masked out and generation would quietly degrade rather than fail.
- Blocks are **pre-LN**: `x + attn(ln(x))`, not `ln(x + attn(x))`. Post-LN renormalises the residual
  branch at every layer, so the path from the loss back to the embeddings is no longer an identity
  and depth needs a warmup to train at all.
- A causal decoder is **not** permutation-equivariant: the mask itself supplies order information.
  Positional embeddings help but are not the sole source.

## Verification

| Check | Measured | Bound |
|---|---|---|
| dense `diag(a) − aaᵀ` contraction = collapsed form | 9.01e-17 | ≤ 1e-15 |
| forward `= softmax(QKᵀ/√d)V` | 0.0 | ≤ 1e-15 |
| `dQ`, `dK`, `dV` vs framework (plain) | 0.0 | ≤ 1e-14 |
| `dQ`, `dK` vs framework (causal) | 0.0 | ≤ 1e-14 |
| central differences on the fused node | 8.53e-11 | ≤ 1e-7 |
| `Var(q·k)` at `d=64` | 63.90 | ≈ 64 |
| `Var(q·k/√d)` at `d=64` / `d=11` | 0.998 / 1.006 | \|·−1\| ≤ 0.05 |
| unscaled mean max attention weight | 0.851 | ≥ 0.8 |
| `trace(J)`: scaled / unscaled | 4.48× | ≥ 3× |
| `−inf` mask ⇒ NaNs on a fully-masked row | 4 of 4 | = 4 |
| the `isfinite` max-guard ⇒ still NaN | 4 of 4 | = 4 |
| `finfo.min/2` ⇒ finite uniform `1/S` | 0.0 | ≤ 1e-15 |
