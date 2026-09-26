# Roadmap — from showpiece to a real framework

Phase 0 (`phase0/`) is a finished thing: draw a digit, watch a network you wrote name it. This
document is the ladder from there to something that is genuinely a neural-network library rather
than a demo dressed as one.

Everything below has been implemented, Tiers 1–11. The tiers are kept as the project's own account of *why*
each piece exists and what would make it fake, and they remain the grading rubric for anything
added later.

## What "real" means here

This is a hobby project and will never compete with PyTorch. But there is a sharp line between *a
hobby project that is genuinely a neural-network library* and *a demo dressed up as one*. The
difference is not scale — it is these five properties, and every tier is graded against them.

| Property | The test |
|---|---|
| **Provably correct** | Every gradient passes finite differences, the adjoint test where it applies, *and* a forward-value check. Not "the loss went down." |
| **General** | Someone can define a model the author never anticipated without editing library code. This is what autograd buys. |
| **Reproducible** | A fixed seed gives bit-identical losses. Every number in the README is regenerable by one command. |
| **Honestly measured** | Benchmarks report where it is slow and where it loses. Accuracy comes from a test set touched once. |
| **Yours** | Every derivation written out in `notes/` in your own words *before* consulting a reference. |

### Anti-goals — what would make it fake

Silent PyTorch fallback in library code · a demo that only works on pre-baked inputs · best-of-N runs
· accuracy measured on tuned-on data · copy-pasting a tutorial and calling it derived · claimed
GPU/distributed support that does not exist · a gradient shipped without a check because "it is
obviously right."

### The recurring proof technique

Each tier must **reproduce the previous tier's result through the new machinery**. When autograd
landed, Phase 0's exact MLP was retrained through it and had to match the accuracy to within seed
noise. That equivalence test is far stronger than a fresh green suite: it proves the general engine
reproduces a special case already verified by hand.

## The tiers

Each is independently finishable; stopping anywhere leaves a coherent project.

### Tier 1 — Autograd *(the biggest single step)*

The line between "hardcoded one architecture" and "built a differentiable programming system."

**1a — Scalar autograd** (`examples/scalar_autograd.py`). A `Value` node with `data`, `grad`, a
`_backward` closure and its parents; topological sort then reverse traversal.

- micrograd defines **only 4 primitives** (`+`, `*`, `**const`, `relu`) and derives the *algebraic*
  operators from them. `exp`/`log`/`tanh` are **not** derivable from that set — they have to be
  added as primitives before softmax or cross-entropy is possible.
- Know the ceiling before you start: pure-Python scalar autograd runs at roughly 241k node
  constructions/second, which puts an MNIST-MLP epoch at **~18 hours**. And CPython's recursion
  limit bites *first* — micrograd's recursive topological sort recurses once per graph level and a
  fan-in-784 sum blows the stack. Build it, benchmark it yourself, then discard it. That benchmark
  *is* the justification for 1b, which is why the example ships both topological sorts and
  demonstrates the recursive one failing.

**1b — Tensor autograd** over `np.ndarray` (`src/scratchgrad/tensor.py`). Arithmetic, `matmul`,
reductions, shape ops, `requires_grad`, `detach`, `no_grad`, and gradient **accumulation**
(`+=`, never `=`).

- The hard part is `unbroadcast(grad, shape)`: sum away prepended axes, then sum with `keepdims`
  where the original extent was 1. Every elementwise backward routes through it.
- `matmul` is a broadcasting op and needs `unbroadcast` too, plus explicit 1-D handling. Axes
  squeezed in the forward must be reinserted in **reverse** order in the backward — `sorted([-2,-1])`
  gives exactly the wrong order and crashes on the 1-D case.
- Leaves must **accumulate** into `.grad`, not assign — assigning wipes the previous `backward()`.

**Proof:** gradcheck every op at prime shapes; Phase 0's MLP retrained through autograd matches.

### Tier 2 — Library shape

`Parameter`, `Module` with `__setattr__` auto-registration, `parameters()`, `train()/eval()`,
`state_dict`, `Sequential`; gain-parameterised init (Xavier/He/orthogonal); SGD/momentum/Nesterov,
RMSProp, Adam, **AdamW**; schedulers; `clip_grad_norm_`; `Dataset`/`DataLoader` with seeded shuffle.

- Apply the LR schedule as a **multiplier on a stored `base_lr` per param group**, not by
  overwriting `lr` — overwriting destroys per-group learning rates.
- Omitting Adam's bias correction makes the first step **3.16×** too large, not 31.6× — the
  numerator is 10× too small at the same time and mostly cancels. (Correcting `m` but *forgetting*
  `v` is the genuinely explosive 31.6× case.)
- `state_dict()` returns a **shallow copy sharing storage** with the live parameters. An in-memory
  snapshot needs a real copy — hence `state_dict(deep=True)` and `utils.snapshot`.

**Proof:** a new architecture in under 20 lines without touching library code; optimizer
trajectories match PyTorch's over 10 steps.

### Tier 3 — Convolutions *(the hardest tier)*

`im2col`/`col2im` via `sliding_window_view`, `Conv2d` fwd+bwd, `MaxPool2d`, `BatchNorm2d`, `Dropout`.

- **Declare the weight layout in writing** — `(F, C, KH, KW)` is standard. Two reference snippets
  with different layouts, cross-checked against each other, is a guaranteed lost afternoon.
- **im2col + GEMM is the default.** Kernel-tap accumulation is ~3× slower and its "zero extra
  memory" claim is false once `optimize=True` is passed. Reach for it only under memory pressure.
- **Every `np.einsum` in a conv needs `optimize=True` explicitly.** The default is `False`, which
  never dispatches to BLAS — a 30–300× loss.
- **Max-pool backward must divide by the tie count.** Without it, tied maxima each receive the full
  upstream gradient.
- **Never compute `dX` for the input layer** — nothing upstream consumes it. Worth ~2.8×
  end-to-end on an MNIST MLP, more than float32 and more than every in-place rewrite combined.
- BatchNorm train/eval agreement is **not** a valid test against a standard EMA implementation —
  either set `momentum=1.0` with a matched estimator, or assert convergence over ≥200 forward passes.
- CIFAR-10 rows are **planar**: `reshape(-1, 3, 32, 32)`, never `(-1, 32, 32, 3)`. The wrong version
  trains badly and you blame the architecture.

**Proof:** gradcheck with non-default stride *and* padding; LeNet-5 on MNIST; a small VGG-ish net on
CIFAR-10.

### Tier 4 — Sequences

`Embedding`, `RNNCell`, `LSTMCell`, `GRUCell`, BPTT with truncation, gradient clipping.

- State the loss reduction explicitly: per-token cross-entropy at init is `ln V` nats; summed over
  T it is `T·ln V`. A clipping threshold is **not** dimensionless — it scales with the reduction, so
  quoting "clip at 5" without saying which reduction is meaningless.
- GRU polarity: PyTorch and Cho gate the **old** state with `z`. Two conventions exist in the wild.
- Embedding's backward is a **scatter-add**. A repeated token index must accumulate; assignment
  keeps only the last write and silently under-trains frequent tokens.

**Proof:** gradcheck through 5 unrolled steps; char-LSTM on tiny-shakespeare beats a trigram
baseline.

### Tier 5 — Attention and a mini-GPT

Scaled dot-product attention, causal masking, multi-head, pre-LN blocks, `LayerNorm` fwd+bwd,
AdamW + cosine warmup.

- **Mask with a large finite negative** (`np.finfo(dtype).min/2`), never `-inf`: a fully-masked row
  gives `max = -inf` and `(-inf) - (-inf) = NaN`. The "guard the max with `isfinite`" trick does
  **not** fix this — it still yields 0/0.
- Implement LayerNorm as `g·(x−μ)/sqrt(var_biased + eps) + b` to match `torch.nn.LayerNorm`. The
  Annotated Transformer uses *unbiased* std with eps *outside* the sqrt — the two will not gradcheck
  against each other.
- **Budget honestly.** In pure NumPy on a laptop CPU only the small config is feasible:
  `n_layer=4, n_head=4, n_embd=128, block_size=64, batch=12, 2000 iters` ≈ 8e12 FLOP ≈ **2–5
  minutes**, target loss ≈1.88. nanoGPT's headline `6×384, T=256, 5000 iters` config is ≈5.8 PFLOP
  ≈ **16–32 hours** in NumPy — that number is an A100-in-PyTorch result and does not transfer.
- Report units: nanoGPT's ~1.47 is **nats**; that is 2.12 bits/char. Random init on V=65 is 4.17 nats.
- A causal decoder is **not** permutation-equivariant — the mask itself supplies order information.
  Positional embeddings help but are not the sole source.

**Proof:** beats the Tier-4 LSTM on identical data and budget; attention rows sum to 1 and are
strictly lower-triangular.

### Tier 6 — Performance, measured honestly

Benchmark harness vs PyTorch CPU on identical models; vectorisation; buffer reuse.

- Expect **~1.5–2×** from float32, not the 7.5× a popular microbenchmark suggests (that one is
  dominated by writing a 400 MB smaller output matrix).
- NumPy dispatch overhead is <1% only above ~10⁵–10⁶ elements per call, not 10⁴.
- Publishing "we are 8× slower than PyTorch CPU, here is exactly why" is more credible than silence.

### Tier 7 — Infrastructure

pytest with a gradcheck per op; `tests/oracle/` vs PyTorch; CI; ruff; docs; pip-installable.

- `--import-mode=importlib` **conflicts with a `tests/helpers.py`** — pytest's own docs list this.
  Shared numeric helpers live in the library as `src/scratchgrad/testing/`.
- `uv sync --all-extras --dev` does **not** install PEP 735 dependency groups. With the torch oracle
  in a group, those tests silently never run and CI stays green. Use `--all-groups`.
- `np.seterr(all='raise')` as a session fixture turns silent overflow into a stack trace.
- `.gitattributes` with `* text=auto eol=lf` — CRLF in `input.txt` adds `\r` to the character
  vocabulary and changes every language-model number.

### Tier 8 — Recurrence without the loop

`linear_scan`, `MinGRU`, a Mamba-style `SelectiveSSM`.

- The cumprod/cumsum closed form divides by a running product that underflows: NaN from step 118
  in float32. The log-space form breaks as soon as `b_t` takes both signs.
- The backward is the same scan run in reverse, carrying `dh` through the **next** step's `a`.
  Using the current step's coefficient is a 56% error that still trains.
- `1 − σ(17)` is exactly 0 in float32. Write it as `σ(−17)`.

**Proof:** the scan equals a sequential loop to ~1e-16 at T = 4,096. minGRU ties the Tier-4 LSTM
(1.628 vs 1.630 nats) at 6.9× less time per step.

### Tier 9 — Quantization

Affine quant, fake-quant with a clipped STE, QAT, binary weights, integer-only inference.

- The STE is a *declared* gradient. Finite differences of `round` are 0 or ∞, so the test checks the
  rule itself, not a derivative.
- The bias belongs after the requantisation multiply. On the `s_x·s_w` grid it is off by up to M/2
  codes.
- int8 matmul wraps, and so does uint8 minus zero-point. Accumulate in int32/int64.

**Proof:** int8 inference matches fp32 (97.54% vs 97.56%). At 2 bits, QAT recovers 91.6% where PTQ
collapses to 22%.

### Tier 10 — Quantum circuits

Statevector simulator plus a parameter-shift autograd op.

- Parameter shift is **exact** for `exp(−iθP/2)` gates, not a finite-difference approximation. It
  gives exactly 0 on a full-angle `exp(−iθP)` gate, and controlled rotations need a four-term rule.
- A parameter reused in several gates needs one shift **per occurrence**, summed. Shifting all of
  them at once returns 0.
- Declare the qubit order. Here wire 0 is the most significant bit, the reverse of Qiskit.

**Proof:** parameter shift = finite differences = analytic, to 1e-10. The classifier reaches 98.5%.
Barren-plateau variance decays ×1.9 per qubit.

### Tier 11 — Variational autoencoder

`reparameterize`, closed-form Gaussian KL, `VAE`.

- The reconstruction is **summed** over pixels. Averaging it reweights the KL term by 784, and every
  latent unit collapses.
- `exp(logvar) − 1 − logvar` is pure noise near 0 in float32. Use `expm1`.

**Proof:** the ELBO gradchecks through the reparameterisation. Test −ELBO is 101.14 nats/image with
20/20 active units.

## Honest capability statement

`scratchgrad` is a working reverse-mode autodiff framework in NumPy that can define and train MLPs,
CNNs, LSTMs, parallel-scan RNNs, a small Transformer, a VAE,
quantized integer-only networks and simulated quantum circuits on CPU, with every gradient independently verified.

It is **not** GPU-accelerated, distributed, fused, JIT-compiled, or competitive with any real
framework on speed. Saying this plainly is what makes the rest believable.

## What would come next

Ideas that fit the same rubric, in rough order of value:

1. **A GPU backend** via CuPy behind a device abstraction — honest, since CuPy would be doing the
   kernels, so it must be labelled as such and never presented as hand-written CUDA.
2. **Graph-level fusion** for the elementwise chains that dominate LayerNorm and GELU.
3. **`vjp`/`jvp` as public API** plus forward-mode dual numbers, which is a genuinely different
   derivation and would exercise the engine's design.
4. **A real serialization format** with backward-compatible schema migration.
5. **Distributed data parallel** over `multiprocessing` with a hand-written ring all-reduce.
