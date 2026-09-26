# Verified reference — traps, and verdicts on the source material

This file collects what came out of reviewing the from-scratch neural-network literature this
project started from. Each technical claim was checked against primary sources, and a
second pass tried to refute it. That review overturned a number of widely repeated claims,
including a few in this project's own early design (recorded at the bottom).

Two things are worth keeping: a table of failures that produce **plausible wrong behaviour rather
than an exception**, and a set of verdicts on the reference material — five of the sixteen sources
this project started from contain real, verifiable bugs.

## Traps table

Every row is a failure mode that does not raise. That is the selection criterion: anything that
crashes teaches you what it is. These do not.

| Symptom | Cause | Fix |
|---|---|---|
| Loss pinned at **1.4615** (C=10) but accuracy looks fine | Double softmax — argmax is preserved, so only the loss reveals it | Layers emit raw logits; the loss owns the softmax |
| Loss sits at exactly `ln(C)` forever, overfit-one-batch still "passes" | `X` and `y` shuffled with two separate calls | One permutation index into both |
| MSE floors at `Var(y)`; model predicts a constant | `(N,1)` prediction vs `(N,)` target → silent `(N,N)` broadcast | `assert pred.shape == target.shape` as line 1 of the loss |
| `nan` loss at high logits | `exp()` of a raw logit | Subtract the row max; use log-sum-exp |
| `nan` in attention | Masked with `-inf`, fully-masked row → `(-inf)-(-inf)` | Mask with `finfo.min/2` |
| Gradcheck "fails" everywhere at ~1e-2 | Running it in float32 | float64, asserted at the top of the helper |
| One 1e-1 among twenty 1e-9 | A ReLU/max kink straddled by the two probes | Repel inputs to `\|x\| > 20h`; check whether the argmax winner changed |
| Gradcheck passes, model still wrong | It only proves `backward == d(forward)/dx` | Forward-value tests + adjoint test |
| Perturbation appears to do nothing | `theta.ravel()[i]` returned a copy | `theta.flat[i]` |
| Silent wrong answer, no exception | Test shapes were square/equal, so a transposed matmul stayed conformable | Prime shapes: N=7, D=11, M=5 |
| 88% accuracy on "MNIST" | Fashion-MNIST cached over it — identical filenames | Separate dirs + per-dataset md5 |
| Effective LR grows with step count | `+=` in backward without `zero_grad` (silent under Adam) | Fold zeroing into the loop; assert grads are zero at loop top |
| Demo ~25% on drawn digits, 98% on test | No 20×20 / centre-of-mass normalisation | Implement `mnistify` exactly (measured here: 98.0% → **20.3%**) |
| Demo confidently wrong | Polarity or normalisation mismatch between train and demo | Assert on the mean of the tensor you feed |
| A *reloaded* checkpoint scores far below its recorded accuracy, with a per-class pattern that looks like a broken architecture | The evaluation skipped the preprocessing training used | Store the normalisation constants **in the checkpoint's `history.json`**, not just in the training script. Measured here: the CIFAR-10 net reads 82.81% with per-channel standardisation and **33.60%** without it — automobile 94% → 2%, truck 89% → 1%, ship unchanged at 89% |
| Early-stopping snapshot silently follows the live model | `state_dict()` shares storage with the parameters | `state_dict(deep=True)` / `utils.snapshot` |
| Tied maxima get 2× the gradient | Max-pool backward did not divide by the tie count | `dx /= mask.sum(axis=(3,5), keepdims=True)` |
| Conv is 30–300× slower than expected | `np.einsum` without `optimize=True` never dispatches to BLAS | Pass `optimize=True` explicitly, or reshape + `@` |
| CIFAR-10 trains badly, architecture blamed | Rows are planar; reshaped as `(-1,32,32,3)` | `reshape(-1, 3, 32, 32)` |
| Language-model numbers unreproducible across machines | CRLF added `\r` to the character vocabulary | `.gitattributes` with `* text=auto eol=lf` |
| CI green, oracle tests never ran | `uv sync --all-extras --dev` does not install PEP 735 groups | `uv sync --all-groups` |

## Annotated bibliography — verdicts on the sixteen starting sources

**Five contain real, verifiable bugs.** That is the single most useful finding of the review:
read them for structure, not for gradients.

| Source | Verdict | Note |
|---|---|---|
| **Karpathy, Zero to Hero** | **essential** | The spine of the roadmap. Caveats: `karpathy.ai/zero-to-hero.html` lists 8 lectures; the live playlist has 10 — neither is complete alone. nanoGPT was **deprecated Nov 2025** in favour of `karpathy/nanochat`. |
| **Victor Zhou, intro-to-neural-networks** | **essential** | Best pedagogical ordering in the list. Its 9 flat scalars do not generalise — move to matrices immediately. |
| **Victor Zhou, intro-to-CNNs** | **useful** | Copy the `forward`/`backprop` layer contract. But `Conv3x3.backprop` **returns `None`** — no input gradient, so you cannot stack convs. |
| **Build-Deep-Learning-From-Scratch (34 stages)** | **essential** (as a syllabus) | Best staged curriculum available; the three-file-per-stage contract is worth stealing. `main` is skeletons only; the `solutions` branch covers **only through stage 17**. |
| **SlowTorch** | **useful** (design), **do not copy math** | The storage/strides/offset split is the one thing micrograd cannot teach. But several backward rules are wrong — `SumBackward0` ignores the incoming gradient; `MeanBackward0` is copy-pasted from `max`. |
| **iamtrask, 11 lines** | **useful** | Great for the "depth is just stacking" intuition. Python 2; no learning rate; no bias — the dataset smuggles it in. |
| **500L OCR** | **useful** (architecture) | The five-file client/server/net split is the reason to read it. Python 2 throughout, and the canvas index math `((x-1)*20 + y) - 1` is a genuine bug (correct: `y*W + x`) that really does degrade predictions. |
| **navoshta, traffic signs** | **useful** | The multi-scale branch idea is ~8 lines and the highest-leverage thing here. All TF 0.x/1.x — `tf.concat(1, [...])` uses the pre-1.0 argument order and **silently breaks** on modern TF. |
| **Nature of Code (Coding Train)** | **useful** | Matrix-class-first ordering is genuinely good. But `train()` updates `weights_ho` **before** computing hidden errors — wrong backprop order. |
| **LSTM music generation** | **skip as linked** | The URL now serves a *different article*. The pipeline shape is worth knowing; the code is dead Keras (`np_utils` removed, `music21 .flat` → `.flatten()`). |
| **CodeProject C# OCR** | **skip** | URL is **dead** — a GoDaddy parking page. Use the Wayback snapshot; both demo zips are lost. AForge.NET abandoned 2013; successor Accord.NET archived. |
| **Go: sausheong** | **read the wrappers, not the loader** | The gonum wrapper set is the most reusable artifact. But the CSV loader is **off by one and feeds the label in as pixel 0**. |
| **Go: datadan** | **useful** (structure) | Clean struct-of-matrices model. `sigmoidPrime` is applied to the *activation*, not the pre-activation — a real math wart. |
| **Go: made2591** | **read for the object model only** | `SigmoidalTransferDerivate` literally `return 1.0`. It still descends, but there is no derivative. |
| **F# parts 1 & 2** | **useful** | The Layer/NodesVect split (parameters separate from per-example activations) is worth stealing in any language. Part 1's `backProp` closes over module-level globals; Part 2 fixes it. |
| **JS perceptron (HackerNoon)** | **useful** (intuition) | Bias-as-an-extra-input-pinned-to-1 is the right habit. `learn()` has **no iteration cap** — hangs the tab forever on non-separable data. |

**Worth adding** (not in the original list): CS231n `neural-networks-3` for the gradient-check
section specifically; Karpathy's *A Recipe for Training Neural Networks*; DeZero (*Deep Learning
from Scratch 3*) for a 60-step framework curriculum; HIPS/autograd and JAX's Autodidax for autograd
design; tinygrad for the movement-op view.

## Corrections to this project's early design

1. **Gradcheck tolerance.** The draft specified "max relative error < 1e-7 on every parameter,
   expect ~1e-9". 1e-9 is right for a single op but too tight for the composed MLP loss and would
   have produced false failures on correct code. Corrected to < 1e-7 composed, < 1e-9 per-op, with a
   scale-aware denominator.
2. **MNIST mirror order.** The draft had CVDF first. The verified-live ordering puts
   `ossci-datasets.s3.amazonaws.com` first.
3. **The epsilon fudge.** The draft's code sketch had `-np.log(p + 1e-12)`. That epsilon caps the
   reportable loss and, in the clipped regime, hands a *zero* gradient to the most-wrong samples.
   Replaced with the log-sum-exp form, which needs no epsilon at all.

Two further corrections came out of implementation rather than research, and are recorded here
because they are the same class of error:

4. **The relative-error metric itself.** A per-element denominator (`|a| + |n|`) cannot pass at
   1e-9 for a legitimately small gradient entry, because the finite-difference noise floor is set by
   the magnitude of the *objective*, not of the entry. `max|a-n| / (max|a| + max|n|)` — normalised by
   the array's scale — is both achievable and strictly more sensitive to real bugs. The first
   `matmul + bias` check "failed" at 1.63e-8 with a perfectly correct gradient, which is what
   surfaced it.

5. **"Never compute `dX` for the input layer — worth ~2.8× end-to-end."** Partly confirmed, and the
   claim needs a qualifier the original did not have. Measured here (batch 128, median of 7):

   | model | dX skipped | dX computed | ratio |
   |---|---|---|---|
   | MLP 784-128-10 | 2.90 ms | 6.37 ms | **2.19×** |
   | LeNet-5 | 66.08 ms | 75.80 ms | **1.15×** |

   The large factor is real but it is an *MLP* result: the first `Linear`'s `dX` is a
   `(128, 784) @ (784, 128)` matmul, comparable to the entire rest of the network. In a convnet the
   first layer is a small fraction of the cost and the saving collapses to 1.15×. Quoting 2.8× under
   a convolutions heading, as the source did, invites the wrong conclusion.

   In `scratchgrad` this costs the user nothing to get right: an input `Tensor` defaults to
   `requires_grad=False`, and `Tensor._make` only ever evaluates a gradient rule for a parent that
   actually requires one — so the input layer's `dX` is never computed unless it is explicitly asked
   for. The saving is the default, not an optimisation anyone has to remember.
