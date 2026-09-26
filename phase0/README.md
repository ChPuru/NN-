# Phase 0 — a neural network with nothing underneath it

Read this directory top to bottom in one sitting and you have seen every line
that turns a drawn stroke into a probability. There is no framework here.
`scratchnn.py` is about two hundred lines of NumPy; each layer owns its own
`backward`, so the chain rule is code you can point at rather than a graph you
have to trust.

**Phase 0 does not import `scratchgrad`.** Not the package, not its data
loaders, not its gradcheck. That is the point: the rest of the repository builds
a real autograd engine, and this directory is the thing you can hold up next to
it and say *the library is not doing the work*. `tests/test_phase0.py` enforces
it.

```
scratchnn.py          Dense, ReLU, fused softmax-cross-entropy, Adam, an MLP container
mnist.py              download MNIST, parse the IDX container by hand
gradcheck.py          the correctness receipt: analytic vs. central differences
train.py              sanity checks, --smoke, --synthetic, full training, plots
export_web.py         weights.npz -> web/model.js, and --inline -> one shareable file
verify_web.py         drives verify_web.js and scores it against NumPy
verify_web.js         runs the shipped JavaScript over real MNIST digits under node
web/index.html        the canvas demo
web/app.js            mnistify() + the forward pass in vanilla JS
web/model.js          generated; the weights, as `const MODEL = {...}`

weights.npz           generated; W1, b1, W2, b2 and the test accuracy
history.json          generated; per-epoch numbers + the exact config that made them
web_verification.json generated; what the browser demo scored
assets/               generated; training.png, weights.png, mistakes.png
```

## Run it

```bash
# from the repository root

python3 phase0/gradcheck.py       #  0.5s  every gradient vs. finite differences
python3 phase0/train.py --smoke   # 39s    sanity checks only, no training
python3 phase0/train.py           # 27s    5 epochs on MNIST + plots
python3 phase0/export_web.py      #  1s    web/model.js + the consistency check
python3 phase0/export_web.py --inline    # also phase0/mnist_demo.html, one file
python3 phase0/verify_web.py      #  3s    the browser demo, checked against NumPy

# then just double-click it. No server, no build step.
start phase0/web/index.html       # (open, on macOS; xdg-open on Linux)
```

The timings are wall clock on a 20-core CPU, NumPy 2.2.3, Python 3.13. MNIST is
cached in the repository's `data/mnist/`, shared with the package's own loader
so the 11 MB is downloaded once.

`--synthetic` fabricates a tiny fake MNIST from ten Gaussian-blob prototypes so
the whole pipeline — sanity checks, training, export — runs with no network. It
is not MNIST and claims nothing about accuracy; it exists so nothing in this
directory is unrunnable offline.

## Measured results

Every number below came out of the commands above at `--seed 0`, and every one
of them is in `history.json` or `web_verification.json`. Running a command twice
gives the same number twice: the two `train.py` runs behind this table produced
**bit-identical** `W1`, `b1`, `W2` and `b2`.

| | measured |
|---|---|
| test accuracy | **97.52 %** (9752 / 10000) |
| train accuracy | 98.52 % |
| test cross-entropy | 0.0808 |
| training wall clock | **16.8 s**, 5 epochs, 2.7–3.9 s each (15.1 s on an earlier identical run — the clock varies, the accuracy does not) |
| parameters | 101,770 |
| loss at init, head zeroed | 2.302585092994046 (= `ln 10`, exact to 1e-12) |
| overfit 32 examples | 3.29e-05 |
| worst gradcheck error, per op | 1.99e-10 (tolerance 1e-9) |
| worst gradcheck error, composed | 5.70e-11 (tolerance 1e-7) |
| worst adjoint error | 6.96e-17 (tolerance 1e-12) |
| NumPy vs exported-float32 forward | max abs logit diff **2.665e-15** (target < 1e-5) |
| browser demo vs NumPy, same digits | **98.67 %** under node, **99.33 %** in Chrome |
| browser demo, mnistify removed | **20.33 %** |

97.52 % sits at the bottom of the 97.5–98.2 % band this architecture reaches.
It is the number the default command produced on the first run, not the best of
several: no seed was searched and no hyperparameter was chosen by looking at the
test set.

## What the numbers mean

### `gradcheck.py`

```
per-op (tol 1e-09)
  dense/W                    rel_err = 6.622e-12   ok
  dense/b                    rel_err = 1.247e-11   ok
  dense/x                    rel_err = 1.534e-11   ok
  relu/x                     rel_err = 1.529e-11   ok
  cross_entropy/logits       rel_err = 1.990e-10   ok

composed (tol 1e-07)
  composed/W1                rel_err = 5.699e-11   ok
  composed/b1                rel_err = 4.625e-11   ok
  composed/W2                rel_err = 3.726e-11   ok
  composed/b2                rel_err = 4.982e-11   ok
  composed/x                 rel_err = 4.464e-11   ok

adjoint, h-free (tol 1e-12)
  adjoint/input              rel_err = 0.000e+00   ok
  adjoint/weight             rel_err = 6.964e-17   ok
```

Relative error is `max|a-n| / (max|a| + max|n| + 1e-12)` — normalised by the
**scale of the array**, not per element. Central differences have a noise floor
set by the magnitude of the objective, not of each individual gradient entry, so
a per-element denominator can never pass at `1e-9` for a legitimately small
entry no matter how correct the gradient is.

A single `1e-1` among a column of `1e-11`s is almost always a probe pair
straddling a ReLU kink, which is why the inputs are pushed to `|x| > 20h` first
— the ambiguity is removed rather than argued about.

The adjoint test, `<v, Au> == <A^T v, u>`, uses no step size at all and holds to
`7e-17`. It is the sharper instrument for anything linear, and unlike finite
differences it cannot be fooled by a *wrong forward* paired with a
self-consistently derived backward.

### `train.py`

Three sanity checks run before any real training, in this order:

1. **Loss at initialization.** The final layer's `W` and `b` are zeroed, so the
   logits are exactly uniform and the loss must be `ln(10) = 2.302585092994046`
   to `1e-12` in float64. Note what this does *not* say: **without** zeroing,
   `L0` fluctuates with the draw and can legitimately land slightly *below*
   `ln(C)` on a given batch. Only a systematic deficit across seeds is evidence
   of a bug. The classic one is a double softmax, which pins the loss at 1.4615
   for ten classes while leaving accuracy looking fine.
2. **Overfit 32 examples** to ~0 loss, before touching the training set. If the
   model cannot memorise 32 samples, it is broken, not undertrained, and no
   amount of learning-rate tuning will fix it. Measured: `3.29e-05`.
3. **One permutation**, indexed into both `X` and `y`. Two separate shuffles
   decouple images from labels; the loss then parks at exactly `ln(C)` — and,
   nastily, overfit-a-batch still passes, because a network can memorise random
   labels.

Then five epochs of Adam at `lr=1e-3`, batch 128:

```
  epoch 1/5  train 0.3493/0.9498   test 0.1795/0.9480   (3.9s)
  epoch 2/5  train 0.1551/0.9671   test 0.1294/0.9624   (3.2s)
  epoch 3/5  train 0.1119/0.9744   test 0.1041/0.9684   (2.7s)
  epoch 4/5  train 0.0865/0.9791   test 0.0937/0.9701   (2.6s)
  epoch 5/5  train 0.0704/0.9852   test 0.0808/0.9752   (2.7s)
```

**97.5–98.2 % is the ceiling for a 128-unit MLP on raw pixels, and this run got
97.52 %.** A small convolutional net reaches 99.2 %+, because a convolution
knows a stroke means the same thing wherever it lands; this model has to learn
that fact 784 separate times, once per input pixel, from 60,000 examples. A
128-unit MLP reporting 99.x % means something leaked — a test set in the
training data, or a hyperparameter chosen by watching the test number move.

`assets/weights.png` draws the first 64 hidden units' receptive fields. Three
things about that picture are deliberate:

* **A diverging blue–white–red map on a single global symmetric scale**,
  `±max|W1|` shared by every tile. Grayscale would give −0.5 and +0.5 two
  different greys with nothing to say which is which; per-tile normalisation —
  the version most tutorials ship — stretches a dead, all-but-zero filter into
  full-contrast noise indistinguishable from a healthy learned feature, and
  destroys any comparison of magnitude between units.
* **16× nearest-neighbour upscaling applied to the array**, not left to the
  renderer, which would otherwise invent intermediate values no weight has.
* **The first 64 units in index order**, not the 64 best-looking ones. A grid
  picked for looks says nothing about the other half.

### `export_web.py`

```
wrote phase0/web/model.js  (1319 KB, 101,770 params)
JS forward consistency: max|logit diff| = 2.665e-15   max|prob diff| = 2.220e-16   ok (target < 1e-5)
```

After writing the file, the exported numbers are parsed back **out of the
generated JavaScript**, rounded through float32 the way a `Float32Array` does,
and the forward pass is re-run with the same loop nest `app.js` uses. That is
compared against the vectorized NumPy forward. Target is `< 1e-5`; the measured
`2.665e-15` is float64 round-off, i.e. the layouts agree exactly. Anything above
`~1e-13` would mean the flattening order, the bias broadcast, or the
serialization disagrees with the browser — and the demo would be confidently
wrong rather than obviously broken.

`--inline` folds `model.js` and `app.js` into `phase0/mnist_demo.html`, 1341 KB,
one file you can email.

### `verify_web.py`

The export check proves the *weights* survive the trip into JavaScript. It says
nothing about the half of the demo where things actually break: the
preprocessing. So this script renders 300 real MNIST **test** digits onto a
simulated 336×336 drawing pad at random sizes (120–260 px) and positions, runs
the shipped JavaScript over them under node, and scores it.

```
agreement with the NumPy model on the same digits
  JS pipeline (render -> mnistify -> forward) vs NumPy : 98.67%  (296/300)
  same, with mnistify REMOVED (plain 28x28 resize)     : 20.67%  (62/300)

accuracy against the true labels, on the rendered pads
  WITH mnistify    : 98.00%  (294/300)
  WITHOUT mnistify : 20.33%  (61/300)

mnistify recipe, re-derived per digit
  max |long side - 20px|        : 0
  max |paste offset - clamped|  : 0
  max |centre of mass - 14|     : 0.4974 px
  heaviest fed tensor mean      : 0.2605  (MNIST is ~0.13; > 0.5 means inverted)

NumPy re-scoring the exact 784-vectors JavaScript built
  argmax agreement : 100.00%  (300/300)
```

Read those four blocks in order, because they separate two things that a single
number would confuse:

* **The forward pass is exact.** NumPy re-scores the very vectors the JavaScript
  built — same weights, same inputs, two independent implementations — and
  agrees on **300 of 300**. There is no numerical disagreement to explain.
* **The pipeline agreement is 98.67 %, not 100 %, and that is resampling, not a
  bug.** Drawing a 28×28 digit up to a few hundred pixels and letting `mnistify`
  pull it back down to a 20-pixel box is not information-preserving: the median
  pixel moves by 0.033. Of the four digits that disagree, **two are ones the
  browser got right and NumPy got wrong**; accuracy on the rendered pads (98.00 %)
  equals NumPy's accuracy on the same 300 clean rows (98.00 %) exactly. Over
  1000 digits the numbers are 98.70 % agreement, 97.30 % vs 97.50 % accuracy.
* **`mnistify` is worth 78 accuracy points.** 98.00 % with it, **20.33 %**
  without. That is the single most important measurement in this directory, and
  it is why the canvas code is not decoration.

The recipe block re-derives the preprocessing from the outputs rather than
trusting the code that produced them: every one of the 300 tiles has its long
side at exactly 20 px, every paste offset equals
`clamp(round(14 − cx), 0, 28 − nw)` exactly, and the centre of mass of the
finished 28×28 tensor lands within half a pixel of (14, 14) — half a pixel being
the most an integer paste offset can be off by.

## The three things the browser demo gets right

Everything above is textbook. What decides whether the demo impresses or flops
is in `web/app.js`.

**1. `mnistify()` is the demo.** MNIST digits were size-normalised into a 20×20
box and then centred in a 28×28 field *by centre of mass*. Feed a raw canvas
scrape to a genuinely 97.5 %-accurate model and it reads hand-drawn input at
**20.3 %** — measured, above — because the model never saw a digit at that scale
or offset and has no translation invariance to fall back on. The reproduction,
step for step:

```
intensity -> bounding box of I > 25 -> scale the LONG side to 20 (offscreen canvas)
-> cx = sum(x*I)/sum(I), cy = sum(y*I)/sum(I)
-> paste at dx = clamp(round(14 - cx), 0, 28 - nw), dy likewise
-> /255 -> forward -> bars
```

Two details inside that: scaling the long side, not both sides, or every `1`
becomes a blob and every `0` a circle; and *progressive halving* before the final
resample, because a bilinear filter reads a 2×2 neighbourhood, so going 240 → 20
in one step samples 4 of the ~144 source pixels covering each destination pixel
and thin strokes dissolve into speckle.

Normalisation is `x/255` **only**, no mean centring, exactly as `mnist.py` does
it. The panel at the bottom of the page draws the resulting 28×28 so you can
watch it work.

**2. `fetch()` is blocked on `file://`** in Chrome and Firefox: a `file://`
document gets an opaque origin, so reading a sibling file is a cross-origin read
and CORS refuses it. A `<script src>` tag is not subject to that rule, so the
weights ship as `web/model.js` containing `const MODEL = {...}` and a
double-clicked `index.html` just works — no server, no `--allow-file-access`
flag, no build step.

**3. Canvas coordinates and polarity.** Pointer position is
`getBoundingClientRect()` scaled by `canvas.width / rect.width`, never
`e.clientX - canvas.offsetLeft` — that version breaks under scroll and under any
CSS-scaled canvas, and this one *is* CSS-scaled: 336 px internally, 420 px on
screen at desktop width. And MNIST is **white on black**: draw black-on-white and
the model fails silently with confident wrong answers, so `assertPolarity()`
checks the mean of the tensor about to be fed in (MNIST averages ~0.13, the
heaviest tensor measured here was 0.26) and says so on screen.

`verify_web.js` re-checks all three against the shipped sources on every run.

The page uses pointer events, so mouse, touch and stylus take one code path, and
it loads no external CSS, JS or fonts.

### Checked in a real browser too

`verify_web.js` substitutes a hand-written bilinear resampler for the canvas,
because node has no canvas. To close that gap the same 300 digits and placements
were run once through headless Chrome, loading `web/index.html` from `file://`
and evaluating the `mnistify()` source sliced verbatim out of `app.js`:

```
  page boots            : model true, PHASE0 true, pad 336px
  meta line             : "101,770 parameters, 97.52% on the MNIST test set."
  console errors        : none
  agreement vs NumPy    : 99.33%  (298/300)
  accuracy vs labels    : 98.00%  (294/300)
  tiles with long side != 20px : 0
```

Chrome's own `drawImage` filter is slightly kinder than the substitute, so
node's 98.67 % is a conservative lower bound on the real thing. Digits drawn
with synthetic mouse strokes through the CSS-scaled canvas came back
`0` at 97.8 %, `1` at 99.8 %, `3` at 100.0 % and `7` at 100.0 %, with no page
errors. That check is not part of this directory: it needs playwright, and the
showpiece has no dependencies.

## What this is not

A 128-unit MLP on raw pixels tops out around **98 %**. A convolutional network
reaches **99.2 %+** on the same data with fewer parameters, because convolution
builds in the fact that a stroke means the same thing wherever it lands. Nothing
in this directory is competitive with anything; it is here so that every line
between a drawn stroke and a probability is one you can read.

The convolutions, the autograd engine that makes them writable in twenty lines,
and the honest benchmark against PyTorch are in the rest of the repository. See
`ROADMAP.md`.
