# `scratchgrad` internal API contract

The conventions every module in `scratchgrad` follows, and that new code should follow too. The
core (`tensor.py`, `nn/module.py`, `nn/init.py`, `testing/gradcheck.py`) is stable: new layers and
ops are built on top of it rather than by changing it, and any change to it needs a strong reason
and full test coverage.

Package source lives in `src/scratchgrad/`. Run anything from the repository root with
`PYTHONPATH=src`, or after `uv pip install -e .`.

## Hard rules

1. **No deep-learning library may be imported by anything under `src/`.** Not
   torch, not tensorflow, not jax, not autograd. `numpy` only. PyTorch appears
   *exclusively* under `tests/oracle/` as a test oracle, and those tests skip
   cleanly when it is absent.
2. **Every gradient is hand-written.** Every new op or layer ships with a
   gradcheck in `tests/`.
3. `from __future__ import annotations` at the top of every module.
4. Public names in `__all__`. Google-style docstrings. Lines ≤ 100 chars.
5. Comments explain *why*, especially where a subtly wrong version exists in the
   wild. Do not narrate what the code plainly says.
6. No `Date.now()`-style nondeterminism: every random draw goes through an
   explicit `numpy.random.Generator` seeded by the caller.

## Core imports

```python
from scratchgrad.tensor import Tensor, unbroadcast, concatenate, stack, where, zeros, ones
from scratchgrad._config import get_default_dtype, is_grad_enabled, no_grad, default_dtype
from scratchgrad.nn.module import Module, Parameter, Sequential, ModuleList, ModuleDict
from scratchgrad.nn import init
from scratchgrad.testing import gradcheck, adjoint_test, check_module, rel_error, TOL_OP, TOL_COMPOSED
```

## `Tensor` — what already exists

Construction: `Tensor(data, requires_grad=False, dtype=None)`, `zeros/ones/full/arange/eye`,
`from_numpy`, `tensor`.

Attributes: `.data` (np.ndarray), `.grad` (np.ndarray | None), `.requires_grad`,
`.shape`, `.ndim`, `.size`, `.dtype`, `.is_leaf`, `.T`.

Methods — all differentiable, all gradchecked:

| group | members |
|---|---|
| arithmetic | `+ - * / **  @` (and r-variants), `maximum`, `minimum`, `abs`, `clip` |
| unary | `exp log sqrt rsqrt sin cos tanh sigmoid relu` |
| reductions | `sum mean var std max min logsumexp` (all take `axis=`, `keepdims=`), `argmax`, `argmin`, `norm` |
| softmax | `softmax(axis=-1)`, `log_softmax(axis=-1)` — built on `logsumexp`, never `log()` of a probability |
| shape | `reshape view flatten ravel transpose permute swapaxes squeeze unsqueeze broadcast_to expand repeat pad` |
| indexing | `__getitem__` (basic + fancy, accumulates via `np.add.at`), `masked_fill` |
| graph | `backward(gradient=None)`, `zero_grad`, `detach`, `clone`, `contiguous`, `requires_grad_`, `astype/float/double` |

Free functions: `concatenate(tensors, axis=0)`, `stack(tensors, axis=0)`,
`where(cond, a, b)`, `unbroadcast(grad, shape)`.

Comparisons (`> >= < <= == !=`) return **plain numpy bool arrays**, not Tensors.
They are not differentiable and must not be put in a graph.

### Adding a new differentiable op

Use `Tensor._make`. It wires the node and only calls a grad rule when that
parent actually requires grad:

```python
from scratchgrad.tensor import Tensor

def my_op(x: Tensor, y: Tensor) -> Tensor:
    out = ...                      # numpy forward
    def gx(g):                     # d<v, out>/dx given g = cotangent of out
        return ...                 # ANY shape; _accumulate unbroadcasts for you
    def gy(g):
        return ...
    return Tensor._make(out, (x, y), (gx, gy), "my_op")
```

Prefer one fused `_make` per layer over composing twenty primitives: it is
faster, and the backward is the thing being demonstrated.

## `Module` — what already exists

`super().__init__()` **first**, then assign. Assigning a `Parameter` or `Module`
to an attribute registers it automatically.

```python
class Linear(Module):
    def __init__(self, n_in, n_out, bias=True, rng=None):
        super().__init__()
        self.weight = Parameter(np.empty((n_in, n_out)))   # see layout below
        self.bias = Parameter(np.zeros(n_out)) if bias else None
        init.kaiming_normal_(self.weight, nonlinearity="relu", rng=rng)
    def forward(self, x):
        ...
    def extra_repr(self):
        return f"{self.n_in}, {self.n_out}, bias={self.bias is not None}"
```

Available: `parameters()`, `named_parameters()`, `buffers()`, `named_buffers()`,
`children()`, `named_children()`, `modules()`, `named_modules()`, `apply(fn)`,
`train(mode=True)`, `eval()`, `.training`, `zero_grad()`,
`state_dict(deep=False)`, `load_state_dict(state, strict=True)`,
`register_buffer(name, array)`, `register_parameter`, `add_module`,
`num_parameters()`, `summary()`, `astype/float/double`, `deepcopy()`.

Non-trainable state that must be saved (BatchNorm running stats) goes in
`register_buffer`, **never** in a `Parameter` — an optimizer would step it.

## Frozen conventions

| Thing | Convention | Why |
|---|---|---|
| `Linear.weight` | `(in_features, out_features)` | forward is `x @ W + b`, no transpose anywhere |
| `Conv2d.weight` | `(out_channels, in_channels, kH, kW)` | the standard layout; write it down once and never cross-check two references with different ones |
| `Embedding.weight` | `(num_embeddings, embedding_dim)` | |
| RNN/LSTM/GRU | `weight_ih (input, gates)`, `weight_hh (hidden, gates)` | matches `x @ W_ih + h @ W_hh + b` |
| LSTM gate order | `i, f, g, o` along the gate axis | PyTorch's order; state it or a checkpoint is unloadable |
| Batch axis | always axis 0 | |
| Image tensors | `(N, C, H, W)` | CIFAR-10 rows are **planar**: `reshape(-1, 3, 32, 32)`, never `(-1, 32, 32, 3)` |
| Sequence tensors | `(N, T, ...)` — batch first | |
| Logits | layers emit **raw logits**; the loss owns the softmax | a double softmax preserves argmax, so accuracy looks fine and only the loss reveals it (pinned at 1.4615 for C=10) |
| dtype | `float32` default for training; gradcheck forces `float64` | in float32 a *correct* gradient shows ~1e-2 relative error |
| Randomness | explicit `rng: np.random.Generator` argument, default `None` | reproducibility is a graded property |

## Tolerances

- `TOL_OP = 1e-9` — a single primitive op.
- `TOL_COMPOSED = 1e-7` — a loss composed of many ops.

`rel_error` is `max|a-n| / (max|a| + max|n| + 1e-12)` — normalised by the **array
scale**, not per element. Central differences have a noise floor set by the
magnitude of the *function value*; a per-element metric can never pass at 1e-9
for a legitimately small entry. Two consequences for test authors:

- **Never gradcheck an objective with a huge magnitude.** Masking with `-1e9`
  makes `<v, out>` ~1e9 and the noise floor 5e-3. Gradcheck the mask *through a
  softmax*, where the sentinel vanishes.
- **Hoist constants out of the checked function.** Calling `rng` inside the
  lambda gives every finite-difference probe a different forward pass.

## Test layout

```
tests/test_<area>.py          pytest, no torch import
tests/oracle/test_<area>.py   may import torch; must skip cleanly without it:
                              torch = pytest.importorskip("torch")
```

Use prime shapes (`N=7, D=11, M=5`, `prime_shape(rank)`): with square shapes a
transposed matmul stays conformable and NumPy computes garbage silently.

Run: `PYTHONPATH=src python -m pytest tests -q`

## Numerical traps that are non-negotiable

- **He init**: variance `2/fan_in`, so **std** `= sqrt(2/fan_in)`.
- **Softmax**: subtract the row max before `exp`.
- **Cross-entropy**: log-sum-exp form. `-log(p + 1e-12)` caps the reportable
  loss and hands a **zero** gradient to the most-wrong samples.
- **Adam**: `eps` goes **outside** the sqrt and **after** bias correction:
  `m̂/(sqrt(v̂) + eps)`. `g/sqrt(v + eps)` engages the floor four orders of
  magnitude too early.
- **Max-pool backward** divides by the tie count; otherwise tied maxima each get
  the full upstream gradient.
- **`np.einsum` in a conv needs `optimize=True` explicitly** — the default is
  `False`, which never dispatches to BLAS (a 30-300x loss).
- **Attention masking** uses `np.finfo(dtype).min / 2`, never `-inf`: a
  fully-masked row gives `(-inf) - (-inf) = NaN`.
- **LayerNorm** is `g * (x - mu) / sqrt(var_biased + eps) + b` — biased variance,
  eps *inside* the sqrt. The Annotated Transformer uses unbiased std with eps
  outside and the two will not gradcheck against each other.
- **`np.frombuffer`** returns a read-only view that keeps the whole buffer
  alive; `.copy()` at the parser boundary.
- **Fashion-MNIST ships the same four filenames as MNIST** — separate cache
  directories and per-dataset md5, or you train a digit classifier on trousers
  at a plausible-looking 88%.
- **Shuffle with ONE permutation indexed into both `X` and `y`.** Two separate
  calls decouple them; loss then sits at exactly `ln(C)` and overfit-one-batch
  still "succeeds".
