"""Attention, transformer blocks, and a small GPT -- all gradients hand-written.

The three things this file exists to get right, none of which are obvious from
the equations:

1. **Masking uses a large finite negative, never ``-inf``.** See
   :func:`mask_value`. A single fully-masked row is enough to turn an entire
   training run into ``NaN`` with the ``-inf`` version, and the failure is
   silent until the loss prints.
2. **The scale is ``1/sqrt(head_dim)`` and it goes on the scores, before the
   softmax.** Scaling the output instead is a different (wrong) function; the
   softmax is not homogeneous.
3. **Blocks are pre-LN**: ``x + attn(ln(x))``, not ``ln(x + attn(x))``. Post-LN
   is what "Attention Is All You Need" published and it does not train at depth
   without a learning-rate warmup, because the residual branch is renormalised
   at every layer and the path from the loss back to the embeddings is no
   longer an identity. Pre-LN keeps that path clean and trains from step one at
   a flat learning rate.

:func:`scaled_dot_product_attention` is written as a **single fused node**
rather than a composition of matmul/softmax/matmul primitives: the softmax
Jacobian collapses to ``a * (da - sum(da * a))`` in the fused form, which is
both the cheaper thing to compute and the thing actually worth demonstrating.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .._config import no_grad
from ..tensor import Tensor
from . import functional as F
from . import init
from .layers import Linear
from .module import Module, ModuleList, Parameter
from .norm import LayerNorm

__all__ = [
    "GPT",
    "MLP",
    "MultiHeadAttention",
    "PositionalEmbedding",
    "TransformerBlock",
    "causal_mask",
    "mask_value",
    "scaled_dot_product_attention",
]


# --------------------------------------------------------------------------- #
# masking
# --------------------------------------------------------------------------- #
def mask_value(dtype) -> float:
    """The finite stand-in for ``-infinity`` used to knock out masked scores.

    Returns ``np.finfo(dtype).min / 2``.

    **Never use ``-inf``.** The softmax subtracts the row max before
    exponentiating, and for a row that is masked *everywhere* -- a padded
    sequence position, a query in a cross-attention block that legitimately
    sees nothing -- that max is itself ``-inf``, so the shift computes
    ``(-inf) - (-inf) = NaN`` and the NaN spreads through the whole batch on the
    next matmul.

    The popular workaround, guarding the max with ``np.where(np.isfinite(m), m,
    0)``, does **not** fix it: the shift is then finite, but every entry in the
    row is still ``-inf``, ``exp`` gives an all-zero row, and the normalisation
    is ``0 / 0`` -- NaN again, one step later. Only a finite sentinel avoids
    both: the row's scores are equal and hugely negative, they cancel exactly in
    the shift, and the row comes out as a uniform average of ``v``. That value
    is meaningless, which is correct -- the input was meaningless -- but it is
    *finite*, so it can be dropped downstream by a loss mask instead of
    poisoning every parameter in the model.

    The ``/ 2`` is not decoration: masks get combined additively in other
    implementations, and ``finfo.min + finfo.min`` overflows to ``-inf``,
    reintroducing exactly the bug this function exists to prevent. Halving
    leaves the headroom for that.
    """
    return float(np.finfo(np.dtype(dtype)).min) / 2.0


@lru_cache(maxsize=64)
def causal_mask(q_len: int, kv_len: int) -> np.ndarray:
    """Boolean ``(q_len, kv_len)`` mask; ``True`` where attention is allowed.

    Offset by ``kv_len - q_len`` rather than pinned to the main diagonal.
    During incremental decoding the query block is the *tail* of the sequence
    while the keys are the whole cache, so query ``i`` lives at absolute
    position ``kv_len - q_len + i``. With ``k=0`` that case would mask out most
    of the context and generation would quietly degrade instead of failing.

    The returned array is cached and marked read-only; callers must not mutate
    it.
    """
    m = np.tril(np.ones((q_len, kv_len), dtype=bool), k=kv_len - q_len)
    m.flags.writeable = False
    return m


def _as_bool_mask(mask, what: str) -> np.ndarray:
    m = np.asarray(mask.data if isinstance(mask, Tensor) else mask)
    if m.dtype != np.bool_:
        raise TypeError(
            f"{what} must be a boolean array where True means 'may attend', got dtype "
            f"{m.dtype}. Additive float masks are a different convention and would be "
            "added to the scores rather than selecting them."
        )
    return m


def _check_mask_shape(m: np.ndarray, score_shape: tuple[int, ...], what: str) -> None:
    """Reject a mask that does not broadcast *into* the ``(..., T, S)`` scores.

    Args:
        m: The boolean mask.
        score_shape: Shape of the score matrix the mask is about to select on.
        what: Parameter name, for the error message.

    Raises:
        ValueError: If ``m`` does not broadcast to exactly ``score_shape``.

    Broadcasting hides two different wrong answers here, neither of which
    raises on its own. A mask with *more* axes than the scores -- a
    ``(N, 1, T, S)`` one against unbatched ``(T, S)`` scores -- broadcasts the
    scores up instead, and attention returns an output of the wrong rank. And
    the natural ``(N, S)`` key-padding mask is missing its query axis, so numpy
    lines it up against ``(T, S)`` and masks by *query row* rather than by key
    -- silently, whenever ``N`` happens to equal ``T``.
    """
    try:
        broadcast = np.broadcast_shapes(m.shape, score_shape)
    except ValueError:
        broadcast = None
    if broadcast != score_shape:
        raise ValueError(
            f"{what} of shape {m.shape} does not broadcast to the attention scores of "
            f"shape {score_shape}. A key-padding mask needs an explicit query axis: "
            "pass (N, 1, S), not (N, S)."
        )


def _check_dropout_p(p: float, what: str) -> float:
    """Validate a dropout probability at construction time.

    Args:
        p: The probability.
        what: Parameter name, for the error message.

    Returns:
        ``p`` as a float.

    Raises:
        ValueError: If ``p`` is outside ``[0, 1)``.

    The equivalent check inside the forward pass only runs under ``training and
    dropout_p``, so an out-of-range value survives construction and every eval
    pass and first raises deep inside a training step -- possibly hours into a
    run, on a model that was already saved.
    """
    p = float(p)
    if not 0.0 <= p < 1.0:
        raise ValueError(f"{what} must lie in [0, 1), got {p}")
    return p


# --------------------------------------------------------------------------- #
# scaled dot-product attention
# --------------------------------------------------------------------------- #
def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask=None,
    is_causal: bool = False,
    scale: float | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    rng=None,
    return_weights: bool = False,
):
    """``softmax(q k^T / sqrt(d)) v`` as one fused, hand-differentiated node.

    Args:
        q: Queries, ``(..., T, D)``.
        k: Keys, ``(..., S, D)``.
        v: Values, ``(..., S, Dv)``.
        mask: Optional boolean array broadcastable to ``(..., T, S)`` where
            ``True`` means *this query may attend to this key*. That is the
            opposite polarity from :meth:`Tensor.masked_fill`, and it is applied
            consistently across this module: everywhere in
            ``scratchgrad.nn.attention``, ``True`` means keep. The shape is
            checked, not merely broadcast: a ``(N, S)`` key-padding mask has no
            query axis and must be passed as ``(N, 1, S)``.
        is_causal: Also apply :func:`causal_mask`, combined with ``mask`` by
            logical AND when both are given.
        scale: Override for ``1 / sqrt(D)``.
        dropout_p: Dropout on the attention weights, after the softmax and
            before the value matmul, inverted-style so ``eval`` is the identity.
        training: Dropout is applied only when this is True.
        rng: ``np.random.Generator`` for the dropout mask.
        return_weights: Also return the attention matrix.

    Returns:
        ``(..., T, Dv)``; or that and the ``(..., T, S)`` attention weights as a
        plain ``ndarray`` when ``return_weights`` is set. Those weights are the
        **effective** ones -- post-dropout, the exact array multiplied into
        ``v`` -- so an attention map used to explain a prediction is the map the
        prediction was actually made with. They come back detached and
        **read-only**: they are a live internal of the fused node, not a second
        differentiable output, and writing to them would change the gradient a
        later ``backward()`` computes.

    The ``1/sqrt(D)`` is applied to the **scores**, before the softmax. Without
    it, ``q . k`` over ``D`` unit-variance terms has variance ``D``, the softmax
    saturates as the head gets wider, and its Jacobian -- ``diag(a) - a a^T`` --
    goes to zero: a dead layer, not merely a slow one.
    """
    qd, kd, vd = q.data, k.data, v.data
    if qd.shape[-1] != kd.shape[-1]:
        raise ValueError(f"q and k must share a head dim, got {qd.shape[-1]} vs {kd.shape[-1]}")
    if kd.shape[-2] != vd.shape[-2]:
        raise ValueError(f"k and v must share a key length, got {kd.shape[-2]} vs {vd.shape[-2]}")
    t_len, s_len = qd.shape[-2], kd.shape[-2]
    sc = (1.0 / math.sqrt(qd.shape[-1])) if scale is None else float(scale)

    scores = (qd @ np.swapaxes(kd, -1, -2)) * sc

    keep = causal_mask(t_len, s_len) if is_causal else None
    if mask is not None:
        m = _as_bool_mask(mask, "mask")
        _check_mask_shape(m, scores.shape, "mask")
        keep = m if keep is None else (keep & m)

    if keep is not None:
        # np.where with a same-dtype sentinel: a Python float would promote a
        # float32 score matrix to float64 halfway through the forward pass.
        scores = np.where(keep, scores, np.asarray(mask_value(scores.dtype), scores.dtype))

    shifted = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    a = e / e.sum(axis=-1, keepdims=True)

    drop = None
    if training and dropout_p:
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must lie in [0, 1), got {dropout_p}")
        gen = rng if rng is not None else np.random.default_rng()
        drop = (gen.random(a.shape) >= dropout_p).astype(a.dtype) / (1.0 - dropout_p)
    a_eff = a if drop is None else a * drop

    out = a_eff @ vd

    # dScores is shared by the q and k rules. Cache it against the cotangent
    # object itself (identity, not id(), which the garbage collector may
    # recycle onto a different array) so one backward does the softmax-Jacobian
    # work once rather than twice.
    state: dict = {"g": None, "ds": None}

    def dscores(g: np.ndarray) -> np.ndarray:
        if state["g"] is not g:
            da = g @ np.swapaxes(vd, -1, -2)
            if drop is not None:
                da = da * drop
            # Softmax VJP. The full Jacobian is diag(a) - a a^T; contracting it
            # with da collapses to this rank-1 correction, which is the reason
            # the fused form is worth writing out.
            ds = a * (da - (da * a).sum(axis=-1, keepdims=True))
            if keep is not None:
                # d(masked score)/d(raw score) is 0 wherever the sentinel
                # replaced the value. ``a`` is already exactly 0 there, so this
                # is belt and braces -- except on a fully-masked row, where
                # ``a`` is a uniform 1/S and this is the line that stops a
                # meaningless gradient from reaching q and k.
                ds = np.where(keep, ds, 0.0)
            state["g"], state["ds"] = g, ds * sc
        return state["ds"]

    def gq(g):
        return dscores(g) @ kd

    def gk(g):
        return np.swapaxes(dscores(g), -1, -2) @ qd

    def gv(g):
        return np.swapaxes(a_eff, -1, -2) @ g

    node = Tensor._make(out, (q, k, v), (gq, gk, gv), "sdpa")
    if not return_weights:
        return node
    # Read-only, because ``a_eff`` is captured by the gv closure (and ``a`` by
    # gq/gk): a caller who renormalises the returned map in place, or zeroes a
    # head to see what happens, would corrupt the gradient of a graph that has
    # not been backwarded yet -- silently, and far from the write. A *view*
    # rather than a copy because MultiHeadAttention.forward requests the
    # weights on every forward and (N, H, T, S) is the largest array in the
    # layer; the flag costs nothing, a copy would double that footprint.
    weights = a_eff.view()
    weights.flags.writeable = False
    return node, weights


# --------------------------------------------------------------------------- #
# multi-head attention
# --------------------------------------------------------------------------- #
class MultiHeadAttention(Module):
    """Multi-head self-attention with one fused qkv projection.

    Weight layouts follow the frozen convention: :class:`Linear` stores
    ``(in_features, out_features)``, so ``c_attn`` is ``(n_embd, 3 * n_embd)``
    and the three projections are contiguous slices of its output.

    Args:
        n_embd: Model width. Must be divisible by ``n_head``.
        n_head: Number of heads; each gets ``n_embd // n_head`` channels, so the
            total compute is independent of the head count.
        bias: Whether the two projections carry biases.
        causal: Apply the causal mask, making this a decoder layer.
        attn_dropout: Dropout on the attention weights.
        resid_dropout: Dropout on the output projection.
        rng: Generator for initialization.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        bias: bool = True,
        causal: bool = True,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        rng=None,
    ) -> None:
        super().__init__()
        if n_embd % n_head:
            raise ValueError(f"n_embd={n_embd} is not divisible by n_head={n_head}")
        self.n_embd = int(n_embd)
        self.n_head = int(n_head)
        self.head_dim = self.n_embd // self.n_head
        self.causal = bool(causal)
        self.attn_dropout = _check_dropout_p(attn_dropout, "attn_dropout")
        self.resid_dropout = _check_dropout_p(resid_dropout, "resid_dropout")
        # nonlinearity="linear" gives gain 1. The sqrt(2) ReLU gain would
        # inflate every head's scores by 40% before the softmax sees them.
        self.c_attn = Linear(n_embd, 3 * n_embd, bias=bias, nonlinearity="linear", rng=rng)
        self.c_proj = Linear(n_embd, n_embd, bias=bias, nonlinearity="linear", rng=rng)

    def _split(self, t: Tensor, n: int, seq: int) -> Tensor:
        return t.reshape(n, seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, x: Tensor, key_padding_mask=None, rng=None, return_weights: bool = False):
        """Attend over the sequence axis of ``x``.

        Args:
            x: ``(N, T, n_embd)``.
            key_padding_mask: Optional ``(N, T)`` boolean, ``True`` for real
                tokens and ``False`` for padding. Note the polarity: ``True``
                means *keep*, the opposite of PyTorch's flag of the same name
                and the same as every other mask in this module.
            rng: Generator for the dropout masks.
            return_weights: Also return the ``(N, n_head, T, T)`` weights.
        """
        if x.ndim != 3:
            raise ValueError(f"MultiHeadAttention expects (N, T, C), got shape {x.shape}")
        n, seq, chan = x.shape
        if chan != self.n_embd:
            raise ValueError(f"expected last dim {self.n_embd}, got {chan}")

        qkv = self.c_attn(x)
        q = self._split(qkv[..., :chan], n, seq)
        k = self._split(qkv[..., chan : 2 * chan], n, seq)
        v = self._split(qkv[..., 2 * chan :], n, seq)

        mask = None
        if key_padding_mask is not None:
            m = _as_bool_mask(key_padding_mask, "key_padding_mask")
            if m.shape != (n, seq):
                raise ValueError(f"key_padding_mask must be {(n, seq)}, got {m.shape}")
            # (N, 1, 1, S): broadcasts over the heads and over every query row.
            mask = m[:, None, None, :]

        y, weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            is_causal=self.causal,
            dropout_p=self.attn_dropout,
            training=self.training,
            rng=rng,
            return_weights=True,
        )
        y = y.transpose(0, 2, 1, 3).reshape(n, seq, chan)
        y = F.dropout(self.c_proj(y), self.resid_dropout, self.training, rng)
        return (y, weights) if return_weights else y

    def extra_repr(self) -> str:
        return (
            f"n_embd={self.n_embd}, n_head={self.n_head}, head_dim={self.head_dim}, "
            f"causal={self.causal}"
        )


# --------------------------------------------------------------------------- #
# transformer block
# --------------------------------------------------------------------------- #
class MLP(Module):
    """The position-wise feed-forward branch: ``W2 gelu(W1 x)``.

    ``mlp_ratio=4`` is the ratio every transformer since the original paper has
    used, and it is where roughly two thirds of the block's parameters live.
    """

    def __init__(
        self,
        n_embd: int,
        mlp_ratio: int = 4,
        bias: bool = True,
        dropout: float = 0.0,
        rng=None,
    ) -> None:
        super().__init__()
        hidden = int(mlp_ratio) * int(n_embd)
        self.dropout = _check_dropout_p(dropout, "dropout")
        self.c_fc = Linear(n_embd, hidden, bias=bias, nonlinearity="relu", rng=rng)
        self.c_proj = Linear(hidden, n_embd, bias=bias, nonlinearity="linear", rng=rng)

    def forward(self, x: Tensor, rng=None) -> Tensor:
        return F.dropout(self.c_proj(F.gelu(self.c_fc(x))), self.dropout, self.training, rng)


class TransformerBlock(Module):
    """A pre-LN decoder block: ``x + attn(ln1(x))``, then ``x + mlp(ln2(x))``.

    **Why pre-LN.** The post-LN arrangement of the original paper --
    ``ln(x + attn(x))`` -- puts a normalisation *on* the residual path, so the
    identity shortcut from the loss back to the embeddings is broken at every
    layer and the expected gradient norm at initialization grows with depth. In
    practice a post-LN transformer diverges at any sane learning rate and needs
    a warmup schedule to train at all. Pre-LN leaves the residual stream
    untouched end to end (only the branches are normalised), which is why GPT-2
    onward is pre-LN and trains at a flat rate from step one.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        mlp_ratio: int = 4,
        bias: bool = True,
        causal: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-5,
        rng=None,
    ) -> None:
        super().__init__()
        # Validate here as well as in the submodules: the value is handed to
        # four different dropout sites, and naming the block's own argument is
        # what tells the caller which one they got wrong.
        dropout = _check_dropout_p(dropout, "dropout")
        self.ln_1 = LayerNorm(n_embd, eps=eps)
        self.attn = MultiHeadAttention(
            n_embd,
            n_head,
            bias=bias,
            causal=causal,
            attn_dropout=dropout,
            resid_dropout=dropout,
            rng=rng,
        )
        self.ln_2 = LayerNorm(n_embd, eps=eps)
        self.mlp = MLP(n_embd, mlp_ratio=mlp_ratio, bias=bias, dropout=dropout, rng=rng)

    def forward(self, x: Tensor, key_padding_mask=None, rng=None) -> Tensor:
        x = x + self.attn(self.ln_1(x), key_padding_mask=key_padding_mask, rng=rng)
        return x + self.mlp(self.ln_2(x), rng=rng)


# --------------------------------------------------------------------------- #
# positional embedding
# --------------------------------------------------------------------------- #
class PositionalEmbedding(Module):
    """A learned absolute position table, ``(max_len, n_embd)``.

    Learned positions are what GPT-2 shipped. They cost ``max_len * n_embd``
    parameters and cannot extrapolate past ``max_len`` -- the honest tradeoff
    against sinusoidal or rotary schemes, which extrapolate but add no capacity.

    A positional embedding is **not** the only thing telling a causal decoder
    where a token sits: the causal mask alone already breaks permutation
    symmetry, because token ``i`` sees a strictly larger context than token
    ``i - 1``. See :class:`GPT` and ``tests/test_attention.py``.
    """

    def __init__(self, max_len: int, n_embd: int, std: float = 0.02, rng=None) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.n_embd = int(n_embd)
        self.weight = Parameter(np.zeros((self.max_len, self.n_embd)))
        init.normal_(self.weight, 0.0, std, rng=rng)

    def forward(self, seq_len: int) -> Tensor:
        """Return ``(1, seq_len, n_embd)``, ready to broadcast over the batch."""
        seq_len = int(seq_len)
        if seq_len > self.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len {self.max_len}")
        return self.weight[:seq_len].unsqueeze(0)

    def extra_repr(self) -> str:
        return f"{self.max_len}, {self.n_embd}"


# --------------------------------------------------------------------------- #
# GPT
# --------------------------------------------------------------------------- #
class GPT(Module):
    """A decoder-only transformer: token and position embeddings, N pre-LN
    blocks, a final LayerNorm, and a linear head to vocabulary logits.

    Args:
        vocab_size: Number of tokens.
        block_size: Maximum context length; also the size of the position table.
        n_layer: Number of :class:`TransformerBlock` s.
        n_head: Heads per block.
        n_embd: Model width.
        dropout: Applied to the embedding sum, the attention weights, and both
            residual branches.
        bias: Biases in the projections. GPT-2 has them; later models drop them.
        tie_weights: Reuse the token embedding as the LM head.
        mlp_ratio: Feed-forward expansion factor.
        rng: Generator for initialization.

    **Weight tying.** With ``tie_weights=True`` the head is ``x @ wte^T`` using
    the *same* ``Parameter`` object, not a copy. It appears exactly once in
    ``parameters()`` -- ``Module.named_parameters`` dedupes by ``id`` -- so an
    optimizer steps it once, and its gradient is the sum over both uses because
    the autograd graph accumulates both edges into the one leaf. Untied, the
    head is a separate ``n_embd * vocab_size`` matrix, which for a character
    model means spending a third of the parameters twice.

    **The compute budget, honestly.** This is NumPy on a laptop CPU, and the
    only configuration that finishes in a coffee break is a small one:

    * ``n_layer=4, n_head=4, n_embd=128, block_size=64, batch=12, 2000 iters``
      is about ``8e12`` FLOP -- roughly **2-5 minutes**, reaching a loss around
      **1.88** on character-level Shakespeare.
    * nanoGPT's headline ``n_layer=6, n_head=6, n_embd=384, T=256, 5000 iters``
      config is about **5.8 PFLOP**, i.e. **16-32 hours** here. That published
      runtime is an A100-in-PyTorch number: roughly three orders of magnitude of
      hardware and kernel advantage that does not transfer to this file.
      Quoting it as a target for a NumPy implementation is the most common way
      one of these projects ends up looking broken when it is merely small.
    * The loss figures are in different units, too. nanoGPT's ``~1.47`` is in
      **nats** (2.12 bits/char). Random init on a 65-token vocabulary is
      ``ln(65) = 4.17`` nats, not 1.0 -- and if the first printed loss is not
      close to ``ln(vocab_size)``, the bug is in the head or the loss, not in
      the training loop.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        dropout: float = 0.0,
        bias: bool = True,
        tie_weights: bool = True,
        mlp_ratio: int = 4,
        rng=None,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.block_size = int(block_size)
        self.n_layer = int(n_layer)
        self.n_embd = int(n_embd)
        self.dropout = float(dropout)
        self.tie_weights = bool(tie_weights)

        self.wte = Parameter(np.zeros((self.vocab_size, self.n_embd)))
        init.normal_(self.wte, 0.0, 0.02, rng=rng)
        self.wpe = PositionalEmbedding(self.block_size, self.n_embd, rng=rng)
        self.blocks = ModuleList(
            [
                TransformerBlock(
                    n_embd,
                    n_head,
                    mlp_ratio=mlp_ratio,
                    bias=bias,
                    causal=True,
                    dropout=dropout,
                    rng=rng,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.ln_f = LayerNorm(n_embd)
        if tie_weights:
            self.register_parameter("lm_head", None)
        else:
            self.lm_head = Parameter(np.zeros((self.n_embd, self.vocab_size)))
            init.normal_(self.lm_head, 0.0, 0.02, rng=rng)

        # GPT-2's residual scaling: every projection that writes into the
        # residual stream is shrunk by 1/sqrt(2 * n_layer), because each layer
        # adds two branches and without it the stream's variance grows linearly
        # with depth -- the final LayerNorm then divides the signal away.
        scale = 1.0 / math.sqrt(2.0 * max(self.n_layer, 1))
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                p.data = p.data * scale

    # -- forward -------------------------------------------------------------
    def forward(self, idx, rng=None) -> Tensor:
        """Map integer token ids ``(N, T)`` to raw logits ``(N, T, vocab_size)``.

        Raw logits, deliberately: the loss owns the softmax. Applying one here
        and another in the loss preserves the argmax, so accuracy looks correct
        and only the loss value reveals the mistake.
        """
        ids = np.asarray(idx.data if isinstance(idx, Tensor) else idx)
        if ids.dtype.kind not in "iu":
            raise TypeError(f"GPT expects integer token ids, got dtype {ids.dtype}")
        if ids.ndim != 2:
            raise ValueError(f"GPT expects (N, T) token ids, got shape {ids.shape}")
        seq = ids.shape[1]
        if seq > self.block_size:
            raise ValueError(f"sequence length {seq} exceeds block_size {self.block_size}")

        x = self.wte[ids] + self.wpe(seq)
        x = F.dropout(x, self.dropout, self.training, rng)
        for block in self.blocks:
            x = block(x, rng=rng)
        x = self.ln_f(x)
        head = self.wte.transpose() if self.tie_weights else self._parameters["lm_head"]
        return x @ head

    def loss(self, idx, targets) -> Tensor:
        """Mean cross-entropy of the next-token prediction, in nats."""
        return F.cross_entropy(self.forward(idx), targets)

    # -- sampling ------------------------------------------------------------
    def generate(
        self,
        idx,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        rng=None,
    ) -> np.ndarray:
        """Autoregressively extend ``idx`` by ``max_new_tokens`` tokens.

        Args:
            idx: ``(N, T)`` (or ``(T,)``) integer prompt.
            max_new_tokens: How many tokens to append.
            temperature: Divides the logits. Below 1 sharpens, above 1 flattens.
                Must be positive; for greedy decoding pass ``top_k=1``, which is
                exactly argmax at any temperature.
            top_k: Sample only from the ``k`` highest-scoring tokens.
            rng: ``np.random.Generator``. Passing one makes the run
                reproducible; omitting it draws fresh entropy.

        Returns:
            ``(N, T + max_new_tokens)`` int64 array, prompt included.

        Runs in ``eval`` mode under ``no_grad``, restoring the previous mode
        afterwards -- otherwise sampling would build a graph that grows with
        every token and dropout would be live during generation.
        """
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0; use top_k=1 for greedy decoding")
        gen = rng if rng is not None else np.random.default_rng()
        out = np.asarray(idx.data if isinstance(idx, Tensor) else idx, dtype=np.int64)
        if out.ndim == 1:
            out = out[None, :]
        if out.ndim != 2:
            raise ValueError(f"idx must be (N, T) or (T,), got shape {out.shape}")

        was_training = self.training
        self.eval()
        try:
            with no_grad():
                for _ in range(int(max_new_tokens)):
                    # Crop to the context window: the position table has no row
                    # past block_size, and a longer prefix would index off it.
                    cond = out[:, -self.block_size :]
                    logits = np.asarray(self.forward(cond).data[:, -1, :], dtype=np.float64)
                    logits = logits / float(temperature)
                    if top_k is not None:
                        kk = int(top_k)
                        if kk < 1:
                            raise ValueError(f"top_k must be >= 1, got {top_k}")
                        kk = min(kk, logits.shape[-1])
                        kth = np.partition(logits, -kk, axis=-1)[:, -kk][:, None]
                        logits = np.where(logits < kth, mask_value(logits.dtype), logits)
                    p = np.exp(logits - logits.max(axis=-1, keepdims=True))
                    p /= p.sum(axis=-1, keepdims=True)
                    # Inverse-CDF sampling, vectorised over the batch. Rounding
                    # can leave the final cumulative entry a hair below 1, hence
                    # the clip rather than trusting the search to stay in range.
                    cdf = np.cumsum(p, axis=-1)
                    r = gen.random((p.shape[0], 1))
                    nxt = np.clip((cdf < r).sum(axis=-1), 0, self.vocab_size - 1)
                    out = np.concatenate([out, nxt[:, None].astype(np.int64)], axis=1)
        finally:
            self.train(was_training)
        return out

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, block_size={self.block_size}, "
            f"n_layer={self.n_layer}, n_embd={self.n_embd}, tie_weights={self.tie_weights}"
        )
