# 06 — LSTM: gate-by-gate BPTT

Verified by [`verify/verify_06_lstm.py`](verify/verify_06_lstm.py) — 14/14.

## Setup

Gate order along the gate axis is `i, f, g, o` — PyTorch's order. State it or a checkpoint is
unloadable. Weights are `weight_ih: (input, 4H)` and `weight_hh: (hidden, 4H)`, matching
`x @ W_ih + h @ W_hh + b`.

```
z    = x W_ih + h W_hh + b            (N, 4H), sliced into four (N, H) blocks
i    = σ(z_i)      input gate    — how much of the candidate to write
f    = σ(z_f)      forget gate   — how much of the old cell to keep
g    = tanh(z_g)   candidate     — what to write
o    = σ(z_o)      output gate   — how much of the cell to expose
c'   = f ⊙ c + i ⊙ g
h'   = o ⊙ tanh(c')
```

`i, f, o` are sigmoids because they are *gates*: a fraction in `(0,1)`. `g` is a `tanh` because it
is a *value*. Getting a sigmoid where a tanh belongs is a real bug that still trains, just worse.

Useful derivatives: `σ' = σ(1−σ)`, `tanh' = 1 − tanh²`.

## The backward for one step

The cell receives cotangents from two places, and this is the step that gets dropped.

`c'` appears in the output twice: it is a returned state (upstream may place `dc'` on it directly),
and it is read by `h' = o ⊙ tanh(c')`. Sum the paths:

```
dc = dc'  +  dh' ⊙ o ⊙ (1 − tanh²(c'))
      ↑          ↑
   from the   through h'
   next step
```

Dropping the second term is the classic LSTM backward bug. It still trains — it just trains
without the constant-error carousel, which is the entire reason to use an LSTM.

Now each gate. `f` and `i`, `g` reach the loss only through `c'`; `o` reaches it only through `h'`:

```
∂c'/∂f = c        ⇒  df = dc ⊙ c
∂c'/∂i = g        ⇒  di = dc ⊙ g
∂c'/∂g = i        ⇒  dg = dc ⊙ i
∂h'/∂o = tanh(c') ⇒  do = dh' ⊙ tanh(c')
```

Push each through its own nonlinearity to reach the pre-activations:

```
dz_i = di ⊙ i ⊙ (1 − i)
dz_f = df ⊙ f ⊙ (1 − f)
dz_g = dg ⊙ (1 − g²)
dz_o = do ⊙ o ⊙ (1 − o)
dz   = concat([dz_i, dz_f, dz_g, dz_o], axis=1)          (N, 4H)
```

And the affine, straight from `01`:

```
dx      = dz W_ihᵀ
dh_prev = dz W_hhᵀ
dW_ih   = xᵀ dz
dW_hh   = hᵀ dz
db      = Σ_batch dz
dc_prev = dc ⊙ f            ← the only path from c' to c, and it is a multiplication
```

`src/scratchgrad/nn/rnn.py::_lstm_step` emits `(N, 2H) = [h' | c']` from a **single** node so this
algebra runs once per backward rather than once per parameter; the caller slices the halves apart
and that slice backward is exact. Every line above was checked against it: worst relative error
across all six gradients, **1.55e-16**.

## Where the vanishing-gradient path is cut

Look at `dc_prev = dc ⊙ f` and compare it with a vanilla RNN's `dh_prev = (dh ⊙ (1 − h²)) W_hhᵀ`.

```
RNN  :  ∂h_T/∂h_0 = ∏_t  W_hhᵀ · diag(1 − h_t²)      matrices, and |tanh'| ≤ 1
LSTM :  ∂c_T/∂c_0 = diag( ∏_t f_t )                  a product of scalars in (0,1)
```

Three differences, in increasing order of importance:

1. **No weight matrix.** The RNN product is a product of *matrices*; its magnitude is governed by
   the spectral radius of `W_hh`, which the network cannot tune freely without also changing what
   it computes. The cell product has no matrix in it at all.
2. **No `tanh'` factor.** `1 − h²` is strictly less than 1 and typically much less. Every RNN step
   multiplies by it whether the network wants to or not.
3. **`f` is a choice.** The forget gate is computed from the input and the state. If remembering is
   useful, the network can drive `z_f` up and hold `f ≈ 1`, and the product stops shrinking. This
   is why the forget-gate bias is initialised to 1 in practice.

The honest version of the claim is that the LSTM does not *solve* vanishing gradients; it makes the
decay a **learnable parameter** instead of a property of the weight spectrum. `verify_06` measures
exactly that, over 40 steps with `W_hh = 0` so the gate is set purely by its bias:

| Path | `‖dc_0‖ / ‖dc_40‖` |
|---|---|
| LSTM cell, `b_f = 5` (`f ≈ 0.993`) | **0.764** |
| LSTM cell, `b_f = 0` (`f ≈ 0.5`) | **8.94e-13** |
| tanh RNN hidden path, default init | **6.37e-05** |

`σ(5)⁴⁰ = 0.9933⁴⁰ = 0.7645`; the measured 0.764 is that, to three digits. And the
`b_f = 0` row is the part usually left out: a half-closed forget gate vanishes faster than the RNN.

The exactness of `∂c_T/∂c_0 = ∏ f_t` is itself checkable. With `W_hh = 0` the gates depend only on
the inputs, so the product can be computed in closed form ahead of the rollout and compared with
what `backward()` delivers through 12 steps. Measured: **6.97e-17**.

## What truncation actually truncates

Truncated BPTT carries the state's *numbers* across a chunk boundary and cuts the *graph*:

```python
h, c = nn.detach_state((h, c))  # values kept, history dropped
```

The precise statement — and it is exact, not approximate:

> Truncated BPTT computes the **exact** gradient of a **different objective**: the one in which the
> boundary state is a constant.

It is not an approximation of the full gradient with a bounded error term. It is the true gradient
of the truncated problem. The parameter still receives contributions from every step *inside* the
window, including via the state that entered it — but the derivative of that entering state with
respect to the parameters is declared zero.

`verify_06` checks both halves. Run A detaches at `t = 4` of 8. Run B replays the first four steps
under `no_grad`, wraps the resulting `(h, c)` in fresh leaves, and runs the last four — a graph
that stops at the boundary *by construction*, with no `detach` involved.

```
dW_hh(detach at 4)  vs  dW_hh(rebuilt from a constant state) :  0.0        exact
dW_hh(detach at 4)  vs  dW_hh(full 8-step BPTT)              :  6.05e-2    relative
```

The first number is what `detach` means. The second is what it costs on this problem: the four
discarded steps were carrying about 6% of the gradient. That is the real trade — memory and
compute linear in the window, against a bias whose size depends on how far the dependencies
actually reach.

Two related points that belong here because they are dimensional:

- **State the loss reduction.** Per-token cross-entropy at init is `ln V` nats; summed over `T` it
  is `T·ln V`. A clipping threshold is not dimensionless — "clip at 5" without saying which
  reduction is meaningless.
- **Embedding's backward is a scatter-add**, for the same reason `col2im` is (`05`): a repeated
  token index must accumulate. Assignment keeps the last write and silently under-trains the most
  frequent tokens — the ones that matter most.

## Verification

| Check | Measured | Bound |
|---|---|---|
| forward: `c' = f⊙c + i⊙g`, `h' = o⊙tanh(c')` | 4.30e-17 | ≤ 1e-15 |
| `dx = dz W_ihᵀ` | 9.20e-17 | ≤ 1e-14 |
| `dh_prev = dz W_hhᵀ` | 6.93e-17 | ≤ 1e-14 |
| `dc_prev = dc ⊙ f` | 1.42e-17 | ≤ 1e-15 |
| `dW_ih = xᵀdz`, `dW_hh = hᵀdz`, `db = Σdz` | ≤ 1.55e-16 | ≤ 1e-14 |
| `dc_0 = (∏_t f_t)·dc_12` with `W_hh = 0` | 6.97e-17 | ≤ 1e-14 |
| cell path, open gate, 40 steps | 0.764 | ≥ 0.5 |
| cell path, half-open gate, 40 steps | 8.94e-13 | ≤ 1e-9 |
| tanh RNN hidden path, 40 steps | 6.37e-05 | ≤ 1e-2 |
| detach == rebuilding from a constant state | 0.0 | ≤ 1e-15 |
| …and differs from full BPTT | 6.05e-2 | ≥ 1e-3 |
| central differences through 6 unrolled steps | 9.59e-11 | ≤ 1e-7 |
