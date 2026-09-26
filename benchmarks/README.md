# Benchmarks — scratchgrad vs PyTorch CPU

Everything on this page came out of one command on one machine. Nothing here is an estimate, a
best-of-N, or a number copied from someone else's blog post. Where the measurement disagrees with a
claim this project has repeated in its own docs, the disagreement is printed rather than smoothed.

## The command

```
PYTHONPATH=src python benchmarks/bench_vs_torch.py --repeat 7 --json benchmarks/results.json
```

`benchmarks/results.json` is written by that run and holds every number below plus the minima, the
per-row `ratio_min`, and the environment block — so this table is regenerable rather than
transcribed. The tables below are the run that produced the checked-in `results.json`.

## Machine

| | |
|---|---|
| CPU | Intel Core i7-13650HX — 6 P-cores + 8 E-cores, 20 logical, 2.6 GHz base |
| OS | Windows 11 (10.0.26200) |
| Python | 3.13.5 |
| NumPy | 2.2.3, OpenBLAS 0.3.28 (`scipy-openblas`, DYNAMIC_ARCH, Haswell kernel selected) |
| PyTorch | 2.6.0+cpu |
| Threads | **1 on both sides** |

Thread pinning is the part that is easy to get wrong, so it is worth spelling out. `torch` is held to
one thread with `torch.set_num_threads(1)`. NumPy's OpenBLAS is held to one thread by setting
`OPENBLAS_NUM_THREADS` (and the OMP/MKL/NUMEXPR equivalents) **before `import numpy`** — OpenBLAS
reads that variable once, when the shared library loads, so setting it afterwards does nothing. The
first version of this harness pinned only torch: NumPy quietly used all 20 cores and the raw-matmul
"floor" row reported a **3.4× win for scratchgrad** that was pure thread-count artifact. With both
sides pinned that row reads 1.04×–1.13×, which is what "both call the same BLAS" should look like.

## How to read these numbers: the noise floor

This is a laptop with heterogeneous cores and other work running on it; a single thread can be parked
on an E-core mid-measurement. The harness therefore times one bit-identical 512×512 f32 matmul 30
times before anything else runs, and publishes the spread:

```
n=30  min 4.89 ms   median 5.87 ms   max 11.08 ms
median/min 1.20x, max/min 2.27x
```

**Any single-row ratio inside about ±30% is not a real effect on this machine.** Both the median and
the minimum are reported for every row; for a deterministic workload the minimum is the cleaner
estimate, and the gap between the two is the honest error bar. Three full runs of the identical
command gave aggregate model-ratio medians of **2.32×, 2.37× and 2.53×** — stable in aggregate, noisy
per row.

## Results

Median of 7 timed runs after 3 warmups. `ratio` is ours ÷ torch; **>1 means we are slower**. Times in
milliseconds.

### Floor — the same BLAS on both sides

| benchmark | ours med | ours min | torch med | torch min | ratio |
|---|---:|---:|---:|---:|---:|
| matmul 256×256 f32 | 0.51 | 0.49 | 0.49 | 0.44 | 1.04× |
| matmul 1024×1024 f32 | 42.69 | 36.77 | 37.72 | 35.81 | 1.13× |

Nothing in this project can beat this floor, and nothing should be slower than it by more than the
autograd bookkeeping it adds.

### Models — identical architecture, identical shapes, identical batch

| benchmark | ours med | ours min | torch med | torch min | ratio (med) | ratio (min) |
|---|---:|---:|---:|---:|---:|---:|
| MLP 784-128-10 forward, batch 128 | 1.18 | 0.84 | 0.51 | 0.46 | **2.32×** | 1.82× |
| MLP fwd+bwd+Adam step | 2.43 | 1.98 | 3.30 | 2.95 | **0.73×** | 0.67× |
| LeNet-5 forward, batch 128 | 52.49 | 47.01 | 15.06 | 12.92 | **3.49×** | 3.64× |
| LeNet-5 fwd+bwd+Adam step | 125.18 | 114.33 | 37.65 | 35.43 | **3.32×** | 3.23× |
| LSTM H=128, T=64, batch 32, forward | 38.30 | 38.16 | 14.61 | 12.49 | **2.62×** | 3.06× |
| LSTM fwd+bwd | 112.84 | 96.59 | 46.16 | 40.70 | **2.44×** | 2.37× |

**Headline: scratchgrad is about 2.4× slower than single-threaded PyTorch CPU on these models
(median over rows; range 0.7×–3.5×).** That is the number to quote. It is not 8×, and it is not 1×.

Two rows deserve honesty rather than a victory lap:

* **MLP fwd+bwd+step comes out at 0.73×, i.e. we are "faster".** Do not read anything into that. The
  MLP step is two small GEMMs; almost all of the measured time on both sides is framework overhead
  on eight parameter tensors, and `torch.optim.Adam`'s per-parameter Python path happens to cost more
  than our `optim.Adam` at this size. Across the three runs this row read 0.92×, 1.73× and 0.73× —
  it is dominated by noise and by optimizer bookkeeping, not by arithmetic. The other MLP row (pure
  forward, 2.32×) is the more informative one.
* **LSTM forward's min-ratio (3.06×) is worse than its median-ratio (2.62×)** because our minimum and
  median nearly coincided (38.16 vs 38.30 ms) while torch's did not. That is what a noisy machine
  looks like; it is not evidence about either implementation.

### mini-GPT — no torch counterpart; this row is about our own budget

| benchmark | ours med | ours min | note |
|---|---:|---:|---|
| mini-GPT forward | 190.16 | 179.44 | 809,856 params, `n_layer=4 n_head=4 n_embd=128 T=64 batch=12` |
| mini-GPT fwd+bwd | 376.95 | 351.96 | ⇒ **≈11.7 min** for the 2000-iteration Tier-5 budget |

That 11.7 minutes is measured wall-clock per step × 2000, not projected from a FLOP count. `ROADMAP`'s
Tier-5 note budgets "2–5 minutes" for this exact config; the real figure on this machine is 2–6× that,
and the ROADMAP estimate should be read as optimistic.

## Claim check

Three claims this project repeats in `ROADMAP.md` were re-derived here instead of taken on faith. The
harness prints its own verdict for each.

### (a) float32 vs float64 matmul — **CONFIRMED**

> Claimed: ~1.5–2×, **not** the 7.5× a popular microbenchmark reports — that one is dominated by
> writing a 400 MB smaller output matrix.

Measured, 1024×1024 GEMM, one thread:

```
matmul 1024 f64   median 81.57 ms   min 78.76 ms
matmul 1024 f32   median 42.97 ms   min 33.92 ms
```

**1.90× on the medians, 2.32× on the minima** (2.12×–2.28× on the other two runs). Right at or just
above the top of the stated 1.5–2× band, and nowhere near 7.5×. This is the ratio you get when the
output matrix is small enough — 4 MB in f32, 8 MB in f64 — that writing it is not the dominant term,
which is exactly the correction the claim makes: 7.5× is an artifact of a benchmark whose output
write dominates, not a property of the FMA units.

### (b) `np.einsum` without `optimize=True` — **CONFIRMED**

> Claimed: the default `optimize=False` never dispatches to BLAS — a 30–300× loss.

Measured on `'nch,co->noh'`, `x=(32,64,100)` f32, `w=(64,128)` f32 — the shape a conv-style einsum
actually has:

```
einsum optimize=False   median 76.15 ms   min 66.82 ms
einsum optimize=True    median  2.02 ms   min  1.97 ms
```

**37.7× on the medians, 33.8× on the minima** (36.6×–43.5× across runs). Inside the claimed range, at
its low end. The claimed 300× upper bound is not reachable at this contraction size; it would need a
much larger reduction axis, where BLAS blocking wins by more. So the claim's *shape* is right and its
*range* is right, but a reader should expect the low end unless the contraction is large. The
practical consequence for this codebase is unchanged: every `np.einsum` in a conv path must pass
`optimize=True` explicitly, or that call runs ~35× slower.

### (c) NumPy per-call dispatch overhead — **DIFFERENT-ON-THIS-MACHINE**

> Claimed: under 1% only above ~10⁵–10⁶ elements per call, not 10⁴.

**How this was measured** (this one had to be designed, not just re-run). Hold the *total* work fixed
at 2²⁴ = 16.8 M float32 elements and vary only how it is chopped up: `total // n` calls of
`np.multiply` on contiguous chunks of length `n`, for `n` = 2¹⁰ … 2²⁴. Four deliberate choices:

* `out=` is supplied, so no allocation is charged to the many-small-calls case — otherwise this
  measures the allocator, not dispatch.
* Chunk sizes are powers of two, so `n × calls == total` exactly: every row does bit-identical
  arithmetic over the same bytes, and per-element cost is directly comparable.
* The operand views are built *outside* the timed region, so only the ufunc call itself is on the
  clock.
* 64 MB per operand keeps everything out of a 24 MB L3, so the per-element floor is real memory
  bandwidth and not a cache artifact.

The cost model is `time = calls · (c₀ + n·tₑ)`, so per-element cost is `tₑ + c₀/n` and the 1%
crossing is `n* = 99·c₀/tₑ`. Both constants are measured directly: `c₀` from 20 000 `np.multiply`
calls on length-1 arrays (min of 5 batches), `tₑ` from the largest chunks, where the `c₀/n` term is
negligible by construction.

```
fixed cost of one np.multiply call (n=1, out= supplied): 0.579 us
     chunk n     calls   ns/elem  measured     model
       1,024     16384     1.981     49.7%     36.2%
       2,048      8192     1.629     38.9%     22.1%
       4,096      4096     1.510     34.0%     12.4%
       8,192      2048     1.207     17.5%      6.6%
      16,384      1024     1.208     17.6%      3.4%
      32,768       512     1.209     17.6%      1.7%
      65,536       256     1.271     21.6%      0.9%
     131,072       128     1.418     29.8%      0.4%
     262,144        64     1.350     26.2%      0.2%
     524,288        32     1.325     24.8%      0.1%
   1,048,576        16     0.996      0.0%      0.1%
   2,097,152         8     1.335     25.4%      0.0%
   4,194,304         4     1.076      7.4%      0.0%
   8,388,608         2     1.068      6.8%      0.0%
  16,777,216         1     1.118     10.9%      0.0%
fixed cost 0.579 us/call, bandwidth floor 0.996 ns/elem
1% crossing from the two measured constants: n = 99*c0/t_e = 57,550
```

**Verdict: `n* ≈ 5.5 × 10⁴ – 7.4 × 10⁴`** — 57 550 here, and 54 861 / 73 726 / 66 610 / 67 159 on four
other runs of the same measurement. That is consistently *just below* the claimed 10⁵ lower bound, by
roughly 1.4–1.8×, so the claim is reported as DIFFERENT-ON-THIS-MACHINE rather than confirmed.

The half of the claim that actually matters is confirmed decisively: **at n = 10⁴ the model says 5%
overhead and the nearest measured row (n = 8 192) reads 17.5%**. Anyone chunking work into
10 000-element NumPy calls is paying a several-percent-to-double-digit dispatch tax, exactly as the
claim warns. The disagreement is only about where the 1% line falls, and here it falls near 6 × 10⁴
rather than 10⁵–10⁶.

Note that the `measured` column is visibly noisier than the `model` column: rows above n = 10⁵ bounce
between 0% and 30% with no trend, because a 1% effect cannot be resolved against a ±30% noise floor.
That is precisely why the crossing is derived from two well-measured constants instead of read off
"the first row that dips under 1%" — which in this run would have reported 1 048 576 and in another
run 16 384, i.e. a number driven entirely by which row got lucky. Publishing that as the answer would
have been the dishonest version of this measurement.

## Why we are slower

Plain language, no excuses. Each of these is a design choice this project made, and each is worth
roughly the factor named.

**No operator fusion.** `x*2 + 1` is two NumPy calls, two output allocations, and two full passes over
memory. PyTorch's CPU backend fuses chains like that into one pass. LayerNorm, GELU and the LSTM gate
arithmetic are almost entirely such chains, which is why the sequence models suffer: an LSTM step is
a dozen elementwise ops around one GEMM, so we pay memory bandwidth a dozen times where a fused
kernel pays it once.

**No threading inside our own code.** Our only parallelism is whatever OpenBLAS does inside a GEMM.
Every elementwise op, every reduction, every `Embedding` gather and the entire autograd traversal is
single-threaded Python. Both sides are pinned to one thread here so this does not show up in the
ratios above — but it is why the ratios would get *worse*, not better, if both sides were given all
20 cores.

**A Python-level graph node per op.** Every forward op allocates a `Tensor`, records its parents, and
closes over a backward function; backward is a Python loop over a topologically sorted list calling
one closure per node. On the MLP that is ~10 nodes and invisible. On the mini-GPT it is thousands of
nodes per step, each costing microseconds of pure interpreter time before any arithmetic happens.
This is the same per-call tax quantified in claim (c) — roughly 0.6 µs of fixed cost — paid once per
graph node instead of once per fused kernel.

**im2col materialises the patch matrix.** `Conv2d` builds an explicit `(N·H_out·W_out, C·KH·KW)`
matrix and calls GEMM on it. For LeNet's first layer at batch 128 that is a 100 352 × 25 array
written to memory and read straight back — traffic a direct-convolution kernel never performs. It is
still the right choice for a NumPy framework (it is the only way to reach BLAS at all, and it is ~3×
faster than kernel-tap accumulation), but it is why the LeNet rows are the worst in the table at
3.3–3.5×.

**Backward does about twice the work of forward with none of the fusion.** `col2im` has to scatter-add
overlapping patches back into the input gradient, and that scatter is bandwidth-bound and unfusable
in NumPy.

**No GPU backend, and none is claimed.** There is no CUDA path, no CuPy path, no device abstraction.
Everything here is CPU float32 through NumPy.

### What we do *not* lose on

The raw-matmul rows are 1.04× and 1.13×. When the work is one big GEMM we *are* PyTorch, because both
of us are calling the same kind of BLAS. **The gap is framework overhead per op, not arithmetic** —
and it grows with the number of ops per unit of arithmetic, which is why it goes MLP forward (2.3×)
→ LSTM (2.4–2.6×) → LeNet (3.3–3.5×) in exactly that order.

## Honest caveats

* Timings are wall-clock on a shared, thermally constrained laptop. The `--repeat 7` median is
  reproducible to a few percent in aggregate and to roughly ±30% per row. The seed makes the *inputs*
  bit-identical between runs; it cannot make the *times* bit-identical, and no benchmark on this
  machine can honestly claim otherwise.
* Only the shapes listed here were measured. Larger batches would favour us slightly (more arithmetic
  per graph node); smaller batches would favour PyTorch.
* The mini-GPT rows have no PyTorch counterpart. Producing one would mean reimplementing our exact
  block layout in torch, and a subtly mismatched reference is worse than no reference.
* Both sides run at one thread. A many-threaded comparison is a different and equally legitimate
  benchmark; it is not this one, and we would lose it by more.
* `bench_dtype` and `bench_einsum_optimize` compare NumPy against NumPy — they occupy the "torch"
  column in the printed table purely so the two variants sit side by side. They are flagged
  `is_model: false` in `results.json` and are excluded from the summary ratios.
