"""Tests for :mod:`scratchgrad.nn.attention`.

Four independent kinds of evidence, because each is blind to what the others
catch:

* ``gradcheck`` -- the hand-written VJP agrees with central differences. Blind
  to a *wrong forward* carrying a self-consistently derived backward.
* hand-computed forward values -- a naive triple-loop reference pins down what
  the forward actually is, including that the ``1/sqrt(head_dim)`` scale is on
  the scores rather than the output.
* structural properties -- rows of the attention matrix sum to 1; a causal mask
  makes it exactly lower-triangular; a causal decoder is *not* permutation
  equivariant while a bidirectional one is.
* the NaN test -- a fully-masked row must stay finite, and the ``-inf`` version
  (with and without the popular "guard the max" patch) must not.

Shapes are prime and unequal (``N=3, T=7, S=11, n_embd=8, n_head=2`` giving
``head_dim=4``) so that a transposed axis raises instead of quietly computing
something conformable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scratchgrad.nn import functional as F
from scratchgrad.nn.attention import (
    GPT,
    MLP,
    MultiHeadAttention,
    PositionalEmbedding,
    TransformerBlock,
    causal_mask,
    mask_value,
    scaled_dot_product_attention,
)
from scratchgrad.tensor import Tensor
from scratchgrad.testing import TOL_COMPOSED, TOL_OP, check_module, gradcheck


def _f64(rng: np.random.Generator, *shape: int) -> Tensor:
    return Tensor(rng.standard_normal(shape), requires_grad=True, dtype=np.float64)


def _reference_attention(q, k, v, keep=None, scale=None):
    """Naive float64 reference. Deliberately not vectorised over T or S."""
    t_len, d = q.shape[-2], q.shape[-1]
    s_len = k.shape[-2]
    sc = 1.0 / math.sqrt(d) if scale is None else scale
    out = np.zeros(q.shape[:-1] + (v.shape[-1],))
    weights = np.zeros(q.shape[:-1] + (s_len,))
    for idx in np.ndindex(q.shape[:-2]):
        for i in range(t_len):
            scores = np.array([float(q[idx][i] @ k[idx][j]) * sc for j in range(s_len)])
            if keep is not None:
                allowed = np.broadcast_to(keep, q.shape[:-1] + (s_len,))[idx][i]
                scores = np.where(allowed, scores, mask_value(np.float64))
            e = np.exp(scores - scores.max())
            a = e / e.sum()
            weights[idx][i] = a
            out[idx][i] = a @ v[idx]
    return out, weights


# --------------------------------------------------------------------------- #
# scaled_dot_product_attention: forward value
# --------------------------------------------------------------------------- #
def test_sdpa_matches_naive_reference(rng, f64):
    q, k, v = _f64(rng, 2, 7, 5), _f64(rng, 2, 11, 5), _f64(rng, 2, 11, 3)
    out, weights = scaled_dot_product_attention(q, k, v, return_weights=True)
    ref_out, ref_w = _reference_attention(q.data, k.data, v.data)
    assert np.allclose(out.data, ref_out, atol=1e-12)
    assert np.allclose(weights, ref_w, atol=1e-12)


def test_sdpa_scale_is_applied_to_the_scores(rng, f64):
    """Scaling the *output* instead of the scores is a different function.

    softmax is not homogeneous, so the two only coincide when the scores are
    already uniform. This asserts the scaled and unscaled weights differ, and
    that passing ``scale`` explicitly reproduces the default.
    """
    q, k, v = _f64(rng, 7, 4), _f64(rng, 7, 4), _f64(rng, 7, 3)
    _, w_default = scaled_dot_product_attention(q, k, v, return_weights=True)
    _, w_explicit = scaled_dot_product_attention(q, k, v, scale=0.5, return_weights=True)
    _, w_unscaled = scaled_dot_product_attention(q, k, v, scale=1.0, return_weights=True)
    assert np.allclose(w_default, w_explicit, atol=1e-14)  # head_dim=4 -> 1/sqrt(4)=0.5
    assert not np.allclose(w_default, w_unscaled, atol=1e-3)


# --------------------------------------------------------------------------- #
# scaled_dot_product_attention: gradients
# --------------------------------------------------------------------------- #
def test_sdpa_gradcheck_unmasked(rng):
    q, k, v = _f64(rng, 3, 2, 7, 5), _f64(rng, 3, 2, 11, 5), _f64(rng, 3, 2, 11, 3)
    gradcheck(
        lambda a, b, c: scaled_dot_product_attention(a, b, c),
        q,
        k,
        v,
        tol=TOL_OP,
        name="sdpa",
    )


def test_sdpa_gradcheck_causal(rng):
    q, k, v = _f64(rng, 2, 2, 7, 5), _f64(rng, 2, 2, 7, 5), _f64(rng, 2, 2, 7, 3)
    gradcheck(
        lambda a, b, c: scaled_dot_product_attention(a, b, c, is_causal=True),
        q,
        k,
        v,
        tol=TOL_OP,
        name="sdpa-causal",
    )


def test_sdpa_gradcheck_with_key_padding_mask(rng):
    # The mask is hoisted out of the checked function: rebuilding it inside
    # would give every finite-difference probe a different forward pass.
    keep = np.array([[True] * 7 + [False] * 4, [True] * 4 + [False] * 7], dtype=bool)
    q, k, v = _f64(rng, 2, 7, 5), _f64(rng, 2, 11, 5), _f64(rng, 2, 11, 3)
    gradcheck(
        lambda a, b, c: scaled_dot_product_attention(a, b, c, mask=keep[:, None, :]),
        q,
        k,
        v,
        tol=TOL_OP,
        name="sdpa-padded",
    )


# --------------------------------------------------------------------------- #
# structural properties of the weights
# --------------------------------------------------------------------------- #
def test_attention_rows_sum_to_one(rng, f64):
    q, k, v = _f64(rng, 3, 2, 7, 4), _f64(rng, 3, 2, 11, 4), _f64(rng, 3, 2, 11, 5)
    _, w = scaled_dot_product_attention(q, k, v, return_weights=True)
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(w >= 0.0)


def test_causal_weights_are_strictly_lower_triangular(rng, f64):
    q, k, v = _f64(rng, 3, 2, 7, 4), _f64(rng, 3, 2, 7, 4), _f64(rng, 3, 2, 7, 5)
    _, w = scaled_dot_product_attention(q, k, v, is_causal=True, return_weights=True)
    # Exactly zero, not merely small: the sentinel underflows exp() to 0.0.
    assert np.all(np.triu(w, k=1) == 0.0)
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(np.diagonal(w, axis1=-2, axis2=-1) > 0.0)


def test_causal_mask_offset_handles_a_short_query_block():
    """A cached-decoding step (q shorter than k) must see the whole prefix."""
    m = causal_mask(2, 5)
    assert np.array_equal(m[0], np.array([True, True, True, True, False]))
    assert np.array_equal(m[1], np.array([True] * 5))
    assert np.array_equal(causal_mask(3, 3), np.tril(np.ones((3, 3), dtype=bool)))


# --------------------------------------------------------------------------- #
# the NaN test
# --------------------------------------------------------------------------- #
def test_fully_masked_row_stays_finite(rng, f64):
    """The whole reason for the finite sentinel."""
    q, k, v = _f64(rng, 2, 7, 4), _f64(rng, 2, 11, 4), _f64(rng, 2, 11, 3)
    keep = np.ones((2, 1, 11), dtype=bool)
    keep[0] = False  # every key of sample 0 is padding: all 7 rows are dead
    out, w = scaled_dot_product_attention(q, k, v, mask=keep, return_weights=True)
    loss = (out * out).sum()
    loss.backward()

    assert np.isfinite(out.data).all()
    assert np.isfinite(w).all()
    assert np.isfinite(q.grad).all() and np.isfinite(k.grad).all() and np.isfinite(v.grad).all()
    # A dead row falls back to a uniform average of v -- meaningless, as it
    # should be, but finite and therefore maskable downstream.
    assert np.allclose(w[0], 1.0 / 11.0, atol=1e-12)
    assert np.allclose(out.data[0], v.data[0].mean(axis=0), atol=1e-12)
    # No gradient reaches q or k through a row that attends to nothing.
    assert np.allclose(q.grad[0], 0.0, atol=1e-14)


def test_minus_inf_sentinel_would_produce_nan():
    """Demonstrates the bug :func:`mask_value` exists to prevent.

    Both the naive ``-inf`` version and the widely-recommended "guard the max
    with isfinite" patch produce NaN on a fully-masked row; only the finite
    sentinel survives.
    """
    scores = np.array([[1.5, -0.5, 2.0]])
    dead = np.zeros((1, 3), dtype=bool)  # nothing may be attended to

    with np.errstate(invalid="ignore", divide="ignore"):
        neg_inf = np.where(dead, scores, -np.inf)
        shifted = neg_inf - neg_inf.max(axis=-1, keepdims=True)  # (-inf) - (-inf)
        naive = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        assert np.isnan(naive).all()

        # "Guard the max": the shift is finite now, but exp(-inf) is an all-zero
        # row and the normalisation is 0/0. Still NaN, one step later.
        m = neg_inf.max(axis=-1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        guarded_e = np.exp(neg_inf - m)
        guarded = guarded_e / guarded_e.sum(axis=-1, keepdims=True)
        assert np.isnan(guarded).all()

    finite = np.where(dead, scores, mask_value(np.float64))
    e = np.exp(finite - finite.max(axis=-1, keepdims=True))
    good = e / e.sum(axis=-1, keepdims=True)
    assert np.isfinite(good).all()
    assert np.allclose(good, 1.0 / 3.0)


def test_float_mask_is_rejected():
    q = Tensor(np.zeros((2, 3)), dtype=np.float64)
    with pytest.raises(TypeError, match="True means"):
        scaled_dot_product_attention(q, q, q, mask=np.ones((2, 2)))


def test_mask_must_broadcast_to_the_score_shape(rng, f64):
    """A mask that merely *broadcasts somehow* is not a mask that is correct.

    Both rejected shapes here used to sail through: the ``(N, S)`` one is the
    natural key-padding shape and numpy lines it up against ``(T, S)``, masking
    by query row instead of by key; the over-ranked one makes ``np.where``
    broadcast the *scores* up, so attention returns an output of the wrong rank
    with no complaint at all.
    """
    q, k, v = _f64(rng, 3, 7, 5), _f64(rng, 3, 11, 5), _f64(rng, 3, 11, 3)

    with pytest.raises(ValueError) as excinfo:
        scaled_dot_product_attention(q, k, v, mask=np.ones((3, 11), dtype=bool))
    message = str(excinfo.value)
    assert "(3, 11)" in message and "(3, 7, 11)" in message, message

    over_ranked = np.ones((2, 3, 7, 11), dtype=bool)
    with pytest.raises(ValueError, match="does not broadcast"):
        scaled_dot_product_attention(q, k, v, mask=over_ranked)

    # The correct spelling of the same key-padding mask still works, and still
    # produces (N, T, Dv) rather than a silently promoted rank.
    out = scaled_dot_product_attention(q, k, v, mask=np.ones((3, 1, 11), dtype=bool))
    assert out.shape == (3, 7, 3)


# --------------------------------------------------------------------------- #
# the returned attention weights
# --------------------------------------------------------------------------- #
def test_returned_weights_are_read_only(rng, f64):
    """They are the fused node's live internals, captured by its backward.

    Normalising the map in place, or zeroing a head to see what changes, would
    otherwise silently alter the gradient of a graph that has not been
    backwarded yet -- a corruption arbitrarily far from the write that caused
    it. Read-only makes the write fail instead.
    """
    q, k, v = _f64(rng, 2, 7, 5), _f64(rng, 2, 11, 5), _f64(rng, 2, 11, 3)
    out, w = scaled_dot_product_attention(q, k, v, return_weights=True)

    assert not w.flags.writeable
    with pytest.raises(ValueError):
        w[..., 0] = 0.0
    with pytest.raises(ValueError):
        w /= w.sum(axis=-1, keepdims=True)

    # Blocking the write must not have cost the caller the values themselves.
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-12)
    out.sum().backward()
    assert np.isfinite(q.grad).all()


def test_returned_weights_are_the_ones_the_output_was_computed_with(rng, f64):
    """With dropout live, the pre-dropout softmax is not the map that was used.

    An attention map is only evidence about a prediction if it is the array
    that was multiplied into ``v``; returning the raw softmax instead shows a
    dense map for an output computed from a sparsified one.
    """
    q, k, v = _f64(rng, 2, 7, 5), _f64(rng, 2, 11, 5), _f64(rng, 2, 11, 3)
    out, w = scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.5,
        training=True,
        rng=np.random.default_rng(0),
        return_weights=True,
    )
    assert np.any(w == 0.0), "dropout must have zeroed some of the returned weights"
    assert np.allclose(out.data, w @ v.data, atol=1e-12)
    # Sanity: eval is untouched -- no dropout, so the rows still sum to 1.
    _, w_eval = scaled_dot_product_attention(
        q, k, v, dropout_p=0.5, training=False, return_weights=True
    )
    assert np.allclose(w_eval.sum(axis=-1), 1.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# dropout probabilities are validated where they are given
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_modules_reject_an_out_of_range_dropout_at_construction(rng, bad):
    """Not at the first training forward, which may be an hour into a run.

    The forward-pass check is guarded by ``training and dropout_p``, so an
    out-of-range probability used to survive construction *and* every eval
    pass; the model could be built, evaluated and checkpointed before anything
    complained.
    """
    with pytest.raises(ValueError, match="attn_dropout"):
        MultiHeadAttention(8, 2, attn_dropout=bad, rng=rng)
    with pytest.raises(ValueError, match="resid_dropout"):
        MultiHeadAttention(8, 2, resid_dropout=bad, rng=rng)
    with pytest.raises(ValueError, match="dropout"):
        MLP(8, dropout=bad, rng=rng)
    with pytest.raises(ValueError, match="dropout"):
        TransformerBlock(8, 2, dropout=bad, rng=rng)


def test_a_valid_dropout_still_constructs_and_runs_in_eval(rng, f64):
    """The guard must not cost the legitimate boundary values 0.0 and 0.99."""
    for good in (0.0, 0.5, 0.99):
        block = TransformerBlock(8, 2, dropout=good, rng=rng).eval()
        assert np.isfinite(block(Tensor(rng.standard_normal((3, 7, 8)))).data).all()


# --------------------------------------------------------------------------- #
# MultiHeadAttention / TransformerBlock
# --------------------------------------------------------------------------- #
def test_multi_head_attention_gradcheck(rng):
    mha = MultiHeadAttention(8, 2, causal=True, rng=rng)
    assert mha.head_dim == 4
    check_module(mha, (3, 7, 8), tol=TOL_COMPOSED, name="MultiHeadAttention")


def test_multi_head_attention_gradcheck_with_padding(rng):
    mha = MultiHeadAttention(8, 2, causal=True, rng=rng)
    keep = np.ones((3, 7), dtype=bool)
    keep[1, 5:] = False
    keep[2, 6:] = False
    check_module(
        mha,
        (3, 7, 8),
        tol=TOL_COMPOSED,
        name="MultiHeadAttention-padded",
        forward=lambda m, x: m(x, key_padding_mask=keep),
    )


def test_multi_head_attention_shapes_and_weights(rng, f64):
    mha = MultiHeadAttention(8, 2, causal=True, rng=rng)
    x = Tensor(rng.standard_normal((3, 7, 8)))
    y, w = mha(x, return_weights=True)
    assert y.shape == (3, 7, 8)
    assert w.shape == (3, 2, 7, 7)
    assert np.all(np.triu(w, k=1) == 0.0)
    with pytest.raises(ValueError, match="not divisible"):
        MultiHeadAttention(8, 3, rng=rng)


def test_transformer_block_gradcheck(rng):
    block = TransformerBlock(8, 2, rng=rng)
    check_module(block, (3, 7, 8), tol=TOL_COMPOSED, name="TransformerBlock")


def test_transformer_block_is_pre_ln(rng, f64):
    """The residual path is the raw input, not a normalised copy of it.

    With both branch output projections zeroed, a pre-LN block is the identity.
    A post-LN block would return ``ln(x)``, which for a non-degenerate input has
    unit variance per row and so cannot equal ``x``.
    """
    block = TransformerBlock(8, 2, rng=rng)
    block.attn.c_proj.weight.data[...] = 0.0
    block.attn.c_proj.bias.data[...] = 0.0
    block.mlp.c_proj.weight.data[...] = 0.0
    block.mlp.c_proj.bias.data[...] = 0.0
    x = Tensor(3.0 + 2.0 * rng.standard_normal((3, 7, 8)))
    assert np.allclose(block(x).data, x.data, atol=1e-12)


# --------------------------------------------------------------------------- #
# order information comes from the mask, not only from the position table
# --------------------------------------------------------------------------- #
def test_causal_attention_is_not_permutation_equivariant(rng, f64):
    """The mask itself carries order; positional embeddings are not the only
    source of it.

    Bidirectional attention *is* permutation equivariant -- permute the tokens
    and the outputs come back permuted the same way, because every query sees
    the same unordered set of keys. Adding the causal mask breaks that: token
    ``i`` sees a strictly larger prefix than token ``i-1``, so the same token in
    a different slot gets a different answer. Neither model here has any
    positional embedding at all.
    """
    perm = np.array([3, 0, 6, 1, 5, 2, 4])
    x = Tensor(rng.standard_normal((2, 7, 8)))
    x_perm = Tensor(x.data[:, perm])

    bidir = MultiHeadAttention(8, 2, causal=False, rng=rng).eval()
    assert np.allclose(bidir(x_perm).data, bidir(x).data[:, perm], atol=1e-12)

    causal = MultiHeadAttention(8, 2, causal=True, rng=rng).eval()
    y = causal(x).data
    y_perm = causal(x_perm).data
    assert not np.allclose(y_perm, y[:, perm], atol=1e-4)
    # Prefix consistency, which is the property the mask actually buys: the
    # first output row attends only to itself, so it equals what the model
    # returns for that token alone.
    alone = causal(Tensor(x.data[:, perm[:1]])).data
    assert np.allclose(y_perm[:, 0], alone[:, 0], atol=1e-12)


# --------------------------------------------------------------------------- #
# PositionalEmbedding
# --------------------------------------------------------------------------- #
def test_positional_embedding_slices_and_routes_gradient(rng, f64):
    pe = PositionalEmbedding(9, 4, rng=rng)
    out = pe(5)
    assert out.shape == (1, 5, 4)
    assert np.allclose(out.data[0], pe.weight.data[:5])
    out.backward(np.ones((1, 5, 4)))
    assert np.all(pe.weight.grad[:5] == 1.0)
    assert np.all(pe.weight.grad[5:] == 0.0)  # unused rows get nothing
    with pytest.raises(ValueError, match="exceeds max_len"):
        pe(10)


# --------------------------------------------------------------------------- #
# GPT
# --------------------------------------------------------------------------- #
def _tiny_gpt(rng, **kwargs):
    cfg = {
        "vocab_size": 11,
        "block_size": 8,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 8,
        "mlp_ratio": 2,
    }
    cfg.update(kwargs)
    return GPT(rng=rng, **cfg)


def test_gpt_gradcheck(rng):
    model = _tiny_gpt(rng)
    # Gradcheck at a well-conditioned point, not at the GPT-2 init. With the
    # shipped std=0.02 embeddings the input to the first LayerNorm has a
    # standard deviation around 0.03, and LayerNorm's third derivative scales as
    # 1/sigma^3 -- so the central-difference truncation error, h^2/6 * f''',
    # lands at ~1e-7 and the check fails on *correct* code. Rescaling the two
    # embedding tables to unit scale removes the conditioning problem without
    # touching the gradient rules under test.
    model.wte.data = rng.standard_normal(model.wte.shape)
    model.wpe.weight.data = rng.standard_normal(model.wpe.weight.shape)
    check_module(model, (3, 5), tol=TOL_COMPOSED, integer_input=True, vocab=11, name="GPT")


def test_gpt_loss_at_init_is_ln_vocab(rng):
    """Random init on ``V`` tokens costs ``ln(V)`` nats -- 4.17 for V=65.

    Not 1.0, and not nanoGPT's 1.47: that figure is a *trained* loss, also in
    nats (2.12 bits/char). A first loss far from ``ln(V)`` means a broken head
    or a double softmax, not a bad learning rate.
    """
    model = _tiny_gpt(rng, vocab_size=65, block_size=16, n_layer=2, n_embd=16, mlp_ratio=4)
    idx = rng.integers(0, 65, size=(4, 8)).astype(np.int64)
    targets = rng.integers(0, 65, size=(4, 8)).astype(np.int64)
    loss = model.loss(idx, targets)
    assert abs(float(loss.item()) - math.log(65)) < 0.15


def test_gpt_forward_backward_populates_every_gradient(rng):
    model = _tiny_gpt(rng)
    idx = rng.integers(0, 11, size=(3, 5)).astype(np.int64)
    logits = model(idx)
    assert logits.shape == (3, 5, 11)
    loss = F.cross_entropy(logits, rng.integers(0, 11, size=(3, 5)).astype(np.int64))
    model.zero_grad()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert np.isfinite(p.grad).all(), f"{name} gradient is not finite"
    assert np.abs(model.wte.grad).max() > 0.0


def test_gpt_overfits_a_tiny_batch(rng):
    """End-to-end evidence that nothing in the stack is merely self-consistent.

    A gradcheck only proves ``backward == d(forward)/dx``; it says nothing about
    whether the forward is the function it claims to be. Memorising four random
    sequences is the cheapest check that also exercises the residual stream, the
    causal mask and the tied head under an actual optimizer.
    """
    from scratchgrad.optim.optimizer import AdamW  # cross-subsystem: kept local

    model = GPT(vocab_size=23, block_size=12, n_layer=2, n_head=2, n_embd=32, rng=rng)
    x = rng.integers(0, 23, size=(4, 12)).astype(np.int64)
    y = rng.integers(0, 23, size=(4, 12)).astype(np.int64)
    opt = AdamW(model.parameters(), lr=3e-3)

    first = float(model.loss(x, y).item())
    assert abs(first - math.log(23)) < 0.2
    for _ in range(200):
        loss = model.loss(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.item()) < 0.5


def test_gpt_rejects_a_too_long_context(rng):
    model = _tiny_gpt(rng)
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(rng.integers(0, 11, size=(2, 9)).astype(np.int64))


def test_gpt_weight_tying_shares_one_parameter(rng):
    tied = _tiny_gpt(rng, tie_weights=True)
    names = [n for n, _ in tied.named_parameters()]
    ids = [id(p) for p in tied.parameters()]
    assert len(ids) == len(set(ids)), "a shared parameter was yielded twice"
    assert "wte" in names and "lm_head" not in names

    untied = _tiny_gpt(rng, tie_weights=False)
    assert "lm_head" in [n for n, _ in untied.named_parameters()]
    assert untied.num_parameters() == tied.num_parameters() + 11 * 8


def test_gpt_generate_length_and_top_k(rng):
    model = _tiny_gpt(rng)
    prompt = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

    out = model.generate(prompt, 6, rng=np.random.default_rng(0))
    assert out.shape == (2, 9)
    assert np.array_equal(out[:, :3], prompt)
    assert out.min() >= 0 and out.max() < 11

    # top_k=1 is argmax, so two different generators must agree exactly.
    a = model.generate(prompt, 5, top_k=1, rng=np.random.default_rng(1))
    b = model.generate(prompt, 5, top_k=1, rng=np.random.default_rng(999))
    assert np.array_equal(a, b)

    # Every sampled token lies inside the model's own top-k for that prefix.
    k = 3
    sampled = model.generate(prompt, 4, top_k=k, rng=np.random.default_rng(7))
    for step in range(4):
        prefix = sampled[:, : 3 + step]
        logits = model(prefix).data[:, -1, :]
        allowed = np.argsort(logits, axis=-1)[:, -k:]
        for row in range(prefix.shape[0]):
            assert sampled[row, 3 + step] in allowed[row]


def test_gpt_generate_crops_to_the_block_size(rng):
    model = _tiny_gpt(rng)
    prompt = rng.integers(0, 11, size=(1, 8)).astype(np.int64)
    out = model.generate(prompt, 5, rng=np.random.default_rng(3))
    assert out.shape == (1, 13)  # longer than block_size=8, and it did not raise
    assert model.training, "generate() must restore the previous train/eval mode"
    with pytest.raises(ValueError, match="temperature"):
        model.generate(prompt, 1, temperature=0.0, rng=np.random.default_rng(0))
