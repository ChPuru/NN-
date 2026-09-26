# 11 — Quantization

Verified by [`verify/verify_11_quantization.py`](verify/verify_11_quantization.py) — 18/18.
Code: `src/scratchgrad/quant/` (`affine.py`, `fake.py`, `integer.py`). Experiment:
`examples/quantize_mnist.py`.

Quantization replaces a float by a small integer and one shared float per tensor or channel. The
arithmetic is short, but there are about half a dozen places where the obvious version is quietly
wrong, and most of them do not show up as an accuracy drop at 8 bits. They show up at 2.

## The affine map

```
q     = clamp(round(x / s) + z, qmin, qmax)
x_hat = s · (q − z)
```

Given an observed range `[lo, hi]` (min-max calibration), first widen it to contain 0, then

```
s = (hi − lo) / (qmax − qmin)
z = qmin − round(lo / s)              ← rounded to an integer
```

**Why `z` must be an integer.** Plug in `x = 0`: `q(0) = round(0/s) + z = z`, so `x_hat(0) = s(z − z)
= 0` exactly. With the unrounded `z' = qmin − lo/s` instead, `q(0) = round(z')` and
`x_hat(0) = s(round(z') − z')`, which is 0 only by luck. A ReLU layer emits exact zeros for about
half its units, so this is a systematic offset on half the activations, not noise. Measured over
1000 ranges straddling 0: integer `z` gives `max|x_hat(0)| = 0`; the unrounded `z'` gives a mean
`|x_hat(0)| = 0.248 s`. The prediction is `0.25 s`, because `round(z') − z'` is uniform on
`[−½, ½]`.

**Rounding convention.** `np.round` rounds half to even: `0.5 → 0`, `1.5 → 2`, `2.5 → 2`. That is
the convention for every float→code step in this package, and it is pinned in a test.

## How big the error is

Inside the representable range, round-to-nearest is never more than half a step off:
`|x − x_hat| ≤ s/2`. If `x` is smooth on the scale of `s` (its density barely changes across one
step), the fractional part of `x/s` is close to uniform, so the error is close to
`U(−s/2, s/2)`:

```
Var(e) = ∫_{−s/2}^{s/2} e² de / s = s²/12
```

Measured with `x ~ N(0, 1)` on an 8-bit grid spanning `[−4, 4]` (σ ≈ 32 steps):
`max|e| = 0.999997 · s/2` and `Var(e) / (s²/12) = 1.0013`.

Each extra bit halves `s`, halves the error's standard deviation, and buys the usual 6.02 dB. The
catch is `s` itself: min-max calibration never clips, so one outlier stretches the grid for every
other value. At 8 bits nobody notices. At 2 bits it is the whole story (see the results).

## Symmetric weights and the per-channel axis

Weights use a signed **narrow** range, `[−(2^(b−1) − 1), 2^(b−1) − 1]`, with `z = 0` and
`s = max|w| / qmax`: `[−127, 127]` at 8 bits, `[−7, 7]` at 4, ternary `{−1, 0, 1}` at 2. That is
symmetric about 0, so `±max|w|` are both exact. At 1 bit a narrow grid is just `{0}`, so binary
weights need their own quantizer (below).

**Per-channel only works along the output axis.** For a dense layer
`y_c = Σ_k x_k w_kc`. If each *output* column gets its own scale, `w_kc = s_c q_kc`, it factors
out of the sum:

```
y_c = s_c · Σ_k x_k q_kc          ← one integer dot product per column, then one multiply
```

If the scale runs along the *input* axis instead, `w_kc = s_k q_kc`, then `s_k` stays inside the
sum and there is no integer matmul at all. In this framework `Linear.weight` is `(in, out)`, so
the output channel is **axis 1**. PyTorch stores `(out, in)` and uses axis 0, which is a trap if
you copy per-channel code between the two layouts.

## Integer-only inference

Write every tensor affinely: `x = s_x(q_x − z_x)`, `w_kc = s_w,c · q_w,kc`, `y = s_y(q_y − z_y)`.
Then

```
y_c   = Σ_k s_x (q_x,k − z_x) · s_w,c q_w,kc + b_c
      = s_x s_w,c · A_c + b_c,            A_c = Σ_k (q_x,k − z_x) q_w,kc      (an integer)
q_y,c = z_y + round( M_c · A_c + b_c / s_y ),      M_c = s_x s_w,c / s_y
```

`A_c` is an int8×int8 dot product accumulated in int32. The only real numbers left are `M_c` and
the bias.

**`M` as fixed point.** `np.frexp` gives `M = f · 2^e` with `f ∈ [½, 1)`. Store
`M0 = round(f · 2^31)` (a 31-bit integer) and `shift = 31 − e`, so `M ≈ M0 · 2^−shift`. The
rounding error in `M0` is at most ½, so the relative error is at most `½ / (f · 2^31) ≤ 2^−31`.
Measured over 10,000 multipliers spanning `[1e−6, 1e3]`: `4.60e−10` against the bound
`2^−31 = 4.66e−10`. If `f · 2^31` rounds up to exactly `2^31`, carry: `M0 = 2^30`, `shift − 1`.

**Rounding in integers.** `round(v / 2^n)` becomes `(v + 2^(n−1)) >> n`. An arithmetic right
shift floors, so this rounds ties toward `+∞`: `+0.5 → 1`, `−0.5 → 0`, `−1.5 → −1`. That differs
from the float path's round-half-to-even, but only on exact ties. Against the float reference
`floor(A·M + b + ½)` over 260,000 outputs, 0.04% differ, and none by more than one code. The
differences come from the `2^−31` error in `M0` pushing a value across a rounding boundary.

**Where the bias goes.** This is the trap I found. The textbook scheme quantizes the bias onto
the accumulator grid, `q_b = round(b / (s_x s_w))`, and adds it into the int32 sum. Its rounding
error is up to `s_x s_w / 2` in real units, which is **`M/2` output codes**. At W8A8 the trained
MNIST model has `M ≈ 2.5e−4` in the first layer and `1e−3` in the second, so that error is
harmless. At W2A2 the second layer's input scale is `s_x = 3.07` (a 2-bit hidden activation)
and its output scale is `s_y = 0.187`, so `M` runs from 9.0 to 13.6 across the ten classes, and
the bias alone can be ~5 codes off. Measured with the textbook bias on that model: 14.5% of logit
codes disagree with the float simulation, by up to 12 codes, and the integer and float
predictions agree on only **78.3%** of the calibration images. The verifier measures the worst
textbook error at `0.9999 · M/2`.

The fix costs nothing, because the product `A · M0` is already int64. Add
`b_fx = round(b / s_y · 2^shift)` to it before the rounding shift. The bias then sits on a
`2^−shift`-code grid, with a measured error of `3.7e−9` codes. After the fix, the integer model
matches the fake-quant simulation on 100% of logit codes, on MNIST and on the verifier's toy W2A2
MLP.

**Overflow budgets.**

* `int8 @ int8` in NumPy returns **int8** and wraps: `[100, 100] @ [2, 2] = −112`. So widen to
  int32 before the matmul.
* The int32 accumulator is bounded by `|A| ≤ K · max|q_x − z_x| · max|q_w|`. For K = 784, uint8
  inputs and int8 weights that is `784 · 255 · 127 = 2.54e7`, which leaves **84.6×** headroom under
  `2^31`. With 16-bit activations it would overflow, and `QuantizedLinear.from_codes` raises.
* `|A · M0| < 2^31 · 2^31 = 2^62`, so product + bias + rounding half fits in int64. That is checked
  at conversion.

**Two free lunches.** (1) If the hidden activation is calibrated *after* the ReLU, its range is
`[0, max]`, so `z_y = qmin = 0` and the requantization clamp *is* the ReLU. (2) MNIST scaled to
`[0, 1]` calibrates to `s = 1/255, z = 0`, so quantizing the input just recovers the pixel bytes.

**Argmax needs a common scale.** Column `c` of the raw accumulator is in units of `s_x s_w,c`,
which differs from class to class. Taking `argmax(A)` compares different units. Requantize the
logits to one per-tensor scale first; `argmax` commutes with a single positive affine map.

## The straight-through estimator

Fake quantization, `fq(x) = s(clamp(round(x/s) + z, qmin, qmax) − z)`, is piecewise constant. Its
true derivative is 0 almost everywhere and undefined at the jumps. Backprop through it gives every
weight upstream a gradient of exactly 0.

Write `round(u) = u + ε(u)`, where `ε` is a sawtooth bounded by ½ with mean zero over any unit
interval. If `ε` is treated as noise that does not depend on the input — the same model that
gave `s²/12` above — then `round` looks like the identity plus noise, and its derivative "on
average" is 1. The **STE** makes that the rule: in the backward pass, replace `d round/du` by 1
and differentiate everything else honestly. The only other non-smooth piece is the clamp, and
its derivative is real: 1 where the clamp is inactive, 0 where it saturates. A value that
saturates is unchanged when you nudge it, so it honestly gets no gradient. So the **clipped
STE** is

```
∂fq/∂x := 1   if qmin ≤ round(x/s) + z ≤ qmax,   else 0
```

Equivalently, the backward pass is the exact derivative of the *unrounded surrogate*
`clip(x, s(qmin − z), s(qmax − z))`.

Boundary convention: the mask is evaluated on the **rounded** code. So `x/s = qmax + 0.4` rounds
to `qmax`, which the clamp did not touch, and gets gradient 1; `qmax + 0.6` gets 0. PyTorch's
`FakeQuantize` uses the same convention.

**Why this cannot be gradchecked.** Central differences of a staircase are 0 when both probes
land on the same step. Measured: 100% of entries. When the probes straddle a jump they give
exactly `s / (2h)`. Measured: equal to `s/(2h)` to machine precision, which is about 39,000 for
`s = 1/127` and `h = 1e−7`. Neither number is the STE's 1, and neither should be: the STE is a
*defined* rule, not a derivative. So the tests check it as a rule. The backward pass must equal
`g · mask`, bit for bit. Inside the range it must equal central differences of the clip
surrogate, and it does, to `1.6e−11`.

In `QATLinear` the weight scale is `max|w| / qmax`, recomputed every step, so no weight is ever
clipped. The weight mask is all ones and `∂L/∂W = xᵀg` evaluated at the quantized weights. The
gradient with respect to the input is exact, since `x` enters linearly with `Q(W)` fixed, and it
is gradchecked at `1e−9`.

## Binary weights

Fix the sign pattern `b = sign(w)` (taking `sign(0) = +1`, so every code is ±1) and choose the
scale that minimises the squared error:

```
L(α)    = ‖w − α b‖² = ‖w‖² − 2α⟨w, b⟩ + α²⟨b, b⟩
dL/dα   = −2⟨w, b⟩ + 2α n = 0     ⇒     α* = ⟨w, b⟩ / n = mean|w|
```

A grid search lands exactly on `mean|w|`. That is a much better scale than min-max's `max|w|`,
and it is why binary PTQ (W1A8) beats ternary PTQ (W2A2) in the table below. The backward pass
uses BinaryConnect's hardtanh STE (1 where `|w| ≤ 1`) and treats `α` as a constant. For integer
inference the codes are ±1 in int8 with a per-channel scale `α` and `z = 0`, so it is the same
kernel.

## Results

`PYTHONPATH=src python examples/quantize_mnist.py` runs the MLP (784 → 128 ReLU → 10) with the
same recipe as `examples/mnist_mlp.py`. Calibration uses 2048 training images. QAT fine-tunes 3
epochs from the calibrated model (Adam 5e−4, EMA observers). Every number below is from the
**integer-only** model, and the test set is touched once per number. The script pins OpenBLAS
to one thread. A different thread count changes the float32 summation order, and the numbers then
move in the second decimal: fp32 is 97.58% multi-threaded against 97.56% single-threaded. The run
takes 145 s of CPU.

| config | PTQ | QAT | deployed bytes |
|---|---|---|---|
| fp32 | 97.56% | — | 407,080 |
| W8A8 | **97.54%** | 97.76% | 103,426 (3.9×) |
| W4A4 | 97.10% | 97.56% | 52,610 (7.7×) |
| W2A2 | 21.96% | 91.59% | 27,202 (15.0×) |
| W1A8 (binary W) | 85.61% | 96.06% | 14,498 (28.1×) |

"W*k*A*k*" means both layers' weights are per-channel symmetric at *k* bits and the hidden
activation is at *k* bits. The input stays 8-bit (it is the pixel bytes) and so do the logits.
Bytes count bit-packed weights plus, per output channel, an int64 bias, an int32 `M0` and an int8
shift. Below 8 bits the codes sit in int8 containers in memory, so the packed size is what a
packer would write. QAT gets extra epochs that PTQ does not, so part of its W8A8 edge is just more
training.

PTQ at W2A2 falls apart for the reason in the error section. A ternary grid with `s = max|w|`
has its threshold at `max|w|/2`, so nearly every weight rounds to 0, and a 2-bit activation with
`s = max/3` does the same to the hidden layer. QAT recovers most of it (91.6%) because the weights
move until the grid suits them. A learned step size (LSQ) is the standard next step and is not
implemented.

## Traps, collected

| trap | what happens | guard |
|---|---|---|
| fractional zero point | exact zeros dequantize to ~s/4 | `z = round(·)`, range widened to include 0 |
| `np.round` is half-to-even | 2.5 → 2 | documented, tested |
| per-channel on the input axis | scales cannot be factored out of the sum | axis 1 for `(in, out)` weights |
| symmetric 1-bit | the grid is `{0}` | `qrange` raises; `binarize` instead |
| `uint8 − z` | wraps (3 − 5 = 254) | widen to int32 before subtracting |
| `int8 @ int8` | returns int8, wraps silently | widen before matmul |
| int32 accumulator | overflows silently for wide or high-bit layers | bound checked at conversion |
| int32 bias on `s_x s_w` grid | up to `M/2` output codes; 78% sim/int agreement at W2A2 | int64 bias on the `2^−shift` grid |
| BLAS thread count | float32 results move in the 2nd decimal | pin `OPENBLAS_NUM_THREADS=1` |
| argmax of raw accumulators | per-class units under per-channel scales | requantize logits to one scale |
| gradchecking `fake_quant` | FD is 0 or `s/(2h)`, never the STE | test the rule + the clip surrogate |
