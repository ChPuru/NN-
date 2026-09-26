# Derivations

Every non-obvious backward pass in `scratchgrad`, worked out rather than quoted.

The project's rubric (`ROADMAP.md`) grades a property it calls **Yours**: *every derivation
written out in your own words*. This directory is that property. It is also the answer to a
practical question — when a gradient is wrong, what do you compare it against? A second
implementation is not an answer; two implementations of the same misunderstanding agree
perfectly. The answer is the algebra, plus a number.

## How to read these

Each file is one topic and follows the same shape:

1. **Setup** — the forward pass, with every convention pinned (which variance, where the epsilon
   sits, which axis is the batch).
2. **The derivation** — from the definition to the implemented expression, with the intermediate
   steps kept in. Where a term cancels, the cancellation is shown, because "this term is zero"
   is exactly the sort of claim that is wrong half the time.
3. **What the code does** — the expression as it actually appears in `src/`, with the file and
   the reason it is written that way.
4. **The trap** — the subtly wrong version that also trains, and how far off it is *in numbers*.
5. **Verification** — what was measured, and the measurement.

Notation is plain unicode inside fenced blocks throughout: `μ` mean, `σ` standard deviation, `∂`
partial, `Σ` sum, `⊙` elementwise product, `Aᵀ` transpose, `⟨u, v⟩` inner product. Cotangents (the
thing a backward pass receives) are written `dL/d·` or, more compactly, `d·` — so `dy` is the
gradient of the loss with respect to `y`, not an infinitesimal.

## Files

| File | Topic | Central result |
|---|---|---|
| [01-backprop-basics.md](01-backprop-basics.md) | the chain rule as VJPs | broadcast forward ⇒ sum backward |
| [02-softmax-cross-entropy.md](02-softmax-cross-entropy.md) | logsumexp(z) − z_y | `dz = (p − onehot)/N` |
| [03-layernorm.md](03-layernorm.md) | the three-term backward | `dx = (1/σ)(dx̂ − mean(dx̂) − x̂·mean(dx̂·x̂))` |
| [04-batchnorm.md](04-batchnorm.md) | the same, over the batch | biased to normalise, unbiased to store |
| [05-conv2d.md](05-conv2d.md) | im2col | `dW = dY·colᵀ`, `dX = col2im(Wᵀ·dY)` |
| [06-lstm.md](06-lstm.md) | gate-by-gate BPTT | the cell path is `diag(∏ f_t)` |
| [07-attention.md](07-attention.md) | softmax(QKᵀ/√d)V | `dS = A ⊙ (dA − Σ dA⊙A)` |
| [08-adam.md](08-adam.md) | bias correction | 3.16×, not 31.6× — and where 31.6× really lives |
| [09-gradient-checking.md](09-gradient-checking.md) | how any of this is checked | `h* ≈ ε^(1/3)`, array-scale normalisation |
| [10-parallel-scan.md](10-parallel-scan.md) | `h_t = a_t·h_{t−1} + b_t` without a loop | the backward is the same scan run in reverse time |
| [11-quantization.md](11-quantization.md) | fake-quant, STE, integer requantisation | the gradient of `round` is a declared rule, not a derivative |
| [12-parameter-shift.md](12-parameter-shift.md) | gradients of quantum circuits | `∂⟨O⟩/∂θ = ½[⟨O⟩(θ+π/2) − ⟨O⟩(θ−π/2)]`, exact |
| [13-vae.md](13-vae.md) | ELBO and the reparameterisation trick | `KL = ½Σ(μ² + σ² − 1 − log σ²)`; reduction mismatch collapses the latent |

## Verification

A derivation nobody ran is a guess with good handwriting. Every file has a companion script under
[`verify/`](verify/) that computes the derived expression from first principles and compares it,
in float64, against **both** the framework's hand-written backward and central differences of the
forward pass. Agreeing with the code alone would only prove the derivation was copied from it.

```
PYTHONPATH=src python3 notes/derivations/verify/run_all.py          # all nine, one table each
PYTHONPATH=src python3 notes/derivations/verify/verify_03_layernorm.py --verbose
```

`run_all.py` writes [`verify/results.json`](verify/results.json), from which every number quoted
in these files is regenerable. As of the run recorded there:

```
93/93 checks passed across 9 derivations
numpy 2.2.3, CPython 3.13.5, Windows-11-10.0.26200-SP0
```

With a fixed `--seed` the run is bit-reproducible, with one exception, named in the JSON's
`nondeterministic` field: the wall-clock ratio in `01` that times reverse mode against forward
mode. It is a real measurement and it moves with machine load; it is flagged rather than dressed
up as a constant.

### What the checks are worth

Two habits, both borrowed from `src/scratchgrad/testing/gradcheck.py` and both load-bearing:

- **Prime shapes** (`N=7, D=11, M=5`). At square shapes a transposed matmul stays conformable and
  NumPy computes garbage in silence.
- **Random cotangents.** Backpropagating all-ones through a summed objective is invariant to a
  transposed axis. `09` measures this: a deliberately wrong-way backward scores `0.0` against an
  all-ones cotangent and `0.59` against a Gaussian one.

And the standing caveat, worth repeating because it limits everything here: finite differences
prove `backward == d(forward)/dx`. A *wrong forward* with a self-consistently derived backward
passes forever. That is why `05` checks im2col against six explicit loops, `03` and `06` check
forward values against hand algebra, and every linear map also gets the adjoint test
`⟨v, Au⟩ = ⟨Aᵀv, u⟩`, which involves no step size and cannot be fooled the same way.
