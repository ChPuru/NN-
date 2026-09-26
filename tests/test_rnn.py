"""Tests for :mod:`scratchgrad.nn.rnn`.

Prime shapes everywhere (``N=7, I=11, H=5, T=13``): with square shapes a
transposed matmul stays conformable and NumPy computes garbage in silence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scratchgrad._config import default_dtype
from scratchgrad.nn import functional as F
from scratchgrad.nn.layers import Linear
from scratchgrad.nn.rnn import (
    GRU,
    LSTM,
    RNN,
    Embedding,
    GRUCell,
    LSTMCell,
    RNNCell,
    detach_state,
)
from scratchgrad.optim import clip_grad_norm_
from scratchgrad.tensor import Tensor, concatenate
from scratchgrad.testing import TOL_COMPOSED, TOL_OP, check_module, gradcheck

N, I, H, T = 7, 11, 5, 13
VOCAB, DIM = 19, 3


def _leaf(arr) -> Tensor:
    """A float64 leaf, which is what ``gradcheck(copy_inputs=False)`` demands."""
    return Tensor(np.asarray(arr, dtype=np.float64), requires_grad=True, dtype=np.float64)


# --------------------------------------------------------------------------- #
# cell gradchecks -- input, hidden state, and every parameter
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
@pytest.mark.parametrize("bias", [True, False])
def test_rnn_cell_gradcheck(nonlinearity, bias):
    rng = np.random.default_rng(0)
    cell = RNNCell(I, H, bias=bias, nonlinearity=nonlinearity, rng=rng).double()
    x = _leaf(rng.standard_normal((N, I)))
    h = _leaf(rng.standard_normal((N, H)))
    gradcheck(
        lambda *ts: cell(ts[0], ts[1]),
        x,
        h,
        *cell.parameters(),
        tol=TOL_OP,
        copy_inputs=False,
        name=f"RNNCell[{nonlinearity}, bias={bias}]",
    )


@pytest.mark.parametrize("bias", [True, False])
def test_lstm_cell_gradcheck(bias):
    rng = np.random.default_rng(1)
    cell = LSTMCell(I, H, bias=bias, rng=rng).double()
    x = _leaf(rng.standard_normal((N, I)))
    h = _leaf(rng.standard_normal((N, H)))
    c = _leaf(rng.standard_normal((N, H)))

    def fn(*ts):
        # Both outputs in one checked object: a cotangent that only ever touched
        # h' would never exercise the dc = gc + ... path out of the cell state.
        h_new, c_new = cell(ts[0], (ts[1], ts[2]))
        return concatenate([h_new, c_new], axis=1)

    gradcheck(
        fn, x, h, c, *cell.parameters(), tol=TOL_OP, copy_inputs=False, name=f"LSTMCell[{bias}]"
    )


@pytest.mark.parametrize("bias", [True, False])
def test_gru_cell_gradcheck(bias):
    rng = np.random.default_rng(2)
    cell = GRUCell(I, H, bias=bias, rng=rng).double()
    x = _leaf(rng.standard_normal((N, I)))
    h = _leaf(rng.standard_normal((N, H)))
    gradcheck(
        lambda *ts: cell(ts[0], ts[1]),
        x,
        h,
        *cell.parameters(),
        tol=TOL_OP,
        copy_inputs=False,
        name=f"GRUCell[{bias}]",
    )


# --------------------------------------------------------------------------- #
# composed graph: the whole unrolled stack
# --------------------------------------------------------------------------- #
def test_lstm_unrolled_gradcheck():
    """The composed-graph check: 7 timesteps, 2 layers, one graph ~14 cells deep."""
    model = LSTM(I, H, num_layers=2, rng=np.random.default_rng(3))
    check_module(
        model,
        (N, 7, I),
        tol=TOL_COMPOSED,
        forward=lambda m, x: m(x)[0],
        name="LSTM(unrolled T=7)",
    )


def test_lstm_final_state_gradcheck():
    """h_n and c_n carry gradient too -- an encoder is trained through them."""
    model = LSTM(I, H, num_layers=2, rng=np.random.default_rng(4))

    def fwd(m, x):
        _, (h_n, c_n) = m(x)
        return concatenate([h_n, c_n], axis=2)

    check_module(model, (N, 6, I), tol=TOL_COMPOSED, forward=fwd, name="LSTM(h_n, c_n)")


@pytest.mark.parametrize("cls", [RNN, GRU])
def test_sequence_gradcheck(cls):
    model = cls(I, H, num_layers=2, rng=np.random.default_rng(5))
    check_module(
        model, (N, 5, I), tol=TOL_COMPOSED, forward=lambda m, x: m(x)[0], name=cls.__name__
    )


@pytest.mark.parametrize("cls", [RNN, LSTM, GRU])
def test_bidirectional_gradcheck(cls):
    model = cls(I, H, num_layers=1, bidirectional=True, rng=np.random.default_rng(6))
    check_module(
        model,
        (N, 5, I),
        tol=TOL_COMPOSED,
        forward=lambda m, x: m(x)[0],
        name=f"{cls.__name__}(bi)",
    )


def test_lstm_gradcheck_through_supplied_initial_state():
    rng = np.random.default_rng(7)
    model = LSTM(I, H, num_layers=1, rng=rng).double()
    x = _leaf(rng.standard_normal((N, 5, I)))
    h0 = _leaf(rng.standard_normal((1, N, H)))
    c0 = _leaf(rng.standard_normal((1, N, H)))
    gradcheck(
        lambda *ts: model(ts[0], (ts[1], ts[2]))[0],
        x,
        h0,
        c0,
        *model.parameters(),
        tol=TOL_COMPOSED,
        copy_inputs=False,
        name="LSTM(hx)",
    )


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def test_embedding_gradcheck():
    emb = Embedding(VOCAB, DIM, rng=np.random.default_rng(8))
    check_module(emb, (N, T), integer_input=True, vocab=VOCAB, tol=TOL_OP, name="Embedding")


def test_embedding_padding_idx_gradcheck_away_from_the_pad_row():
    """Every non-padding row still carries the exact gradient.

    The indices deliberately never touch ``padding_idx``. They cannot: zeroing
    that row is a *deliberate* departure from the true derivative -- the forward
    really does read row 4 when index 4 appears -- so finite differences would
    correctly report a gradient there and gradcheck would correctly fail. The
    zero row is pinned by its own test below instead.
    """
    emb = Embedding(VOCAB, DIM, padding_idx=4, rng=np.random.default_rng(9)).double()
    idx = np.array([[1, 7], [3, 11], [7, 1]], dtype=np.int64)
    gradcheck(
        lambda w, idx=idx: emb(idx),
        emb.weight,
        tol=TOL_OP,
        copy_inputs=False,
        name="Embedding(pad)",
    )


def test_embedding_repeated_index_accumulates():
    """[3, 3, 3] must give row 3 exactly three times the gradient of [3].

    This is the whole reason the backward is ``np.add.at`` and not ``acc[idx]=g``:
    plain assignment keeps only the last write, so a token seen k times in a
    batch trains as if it were seen once. Nothing about the loss curve shows it.
    """
    emb = Embedding(VOCAB, DIM, rng=np.random.default_rng(10)).double()

    out = emb(Tensor(np.array([3], dtype=np.int64)))
    out.backward(np.ones_like(out.data))
    single = np.array(emb.weight.grad[3], copy=True)

    emb.zero_grad()
    out3 = emb(Tensor(np.array([3, 3, 3], dtype=np.int64)))
    out3.backward(np.ones_like(out3.data))
    triple = emb.weight.grad[3]

    assert np.allclose(single, np.ones(DIM))
    assert np.allclose(triple, 3.0 * single)
    # And nothing leaked into a row that was never looked up.
    assert np.all(emb.weight.grad[0] == 0.0)


def test_embedding_padding_idx_gradient_is_exactly_zero():
    emb = Embedding(VOCAB, DIM, padding_idx=2, rng=np.random.default_rng(11)).double()
    assert np.all(emb.weight.data[2] == 0.0)  # zeroed at init as well
    out = emb(Tensor(np.array([[2, 3], [2, 7]], dtype=np.int64)))
    out.backward(np.ones_like(out.data))
    assert np.all(emb.weight.grad[2] == 0.0)
    assert np.all(emb.weight.grad[3] == 1.0)
    assert np.all(emb.weight.grad[7] == 1.0)


def test_embedding_rejects_float_and_out_of_range_indices():
    emb = Embedding(VOCAB, DIM, rng=np.random.default_rng(12))
    with pytest.raises(TypeError):
        emb(np.array([1.0, 2.0]))
    with pytest.raises(IndexError):
        emb(np.array([VOCAB], dtype=np.int64))


# --------------------------------------------------------------------------- #
# forward values, computed by hand
# --------------------------------------------------------------------------- #
def test_lstm_cell_forward_matches_hand_computation():
    """One step of an LSTMCell against arithmetic done on paper.

    Gradcheck proves ``backward == d(forward)``; a self-consistent *wrong*
    forward passes it forever. This pins the forward itself -- and with it the
    ``i, f, g, o`` gate order, since any other order gives a different number.
    """
    w_ih = np.array([[0.1, 0.2, 0.3, 0.4], [-0.5, 0.6, -0.7, 0.8]])
    w_hh = np.array([[0.9, -1.0, 1.1, -1.2]])
    bias = np.array([0.05, -0.05, 0.15, -0.15])

    with default_dtype(np.float64):
        cell = LSTMCell(2, 1, rng=np.random.default_rng(13))
        cell.weight_ih.data[...] = w_ih
        cell.weight_hh.data[...] = w_hh
        cell.bias.data[...] = bias
        h_new, c_new = cell(Tensor([[1.0, -1.0]]), (Tensor([[0.5]]), Tensor([[-0.3]])))

    # Pre-activations, worked out longhand:
    #   x @ W_ih = [0.6, -0.4, 1.0, -0.4]
    #   h @ W_hh = [0.45, -0.5, 0.55, -0.6]
    #   + bias   = [1.10, -0.95, 1.70, -1.15]   (order i, f, g, o)
    sig = lambda v: 1.0 / (1.0 + math.exp(-v))  # noqa: E731
    i, f, g, o = sig(1.10), sig(-0.95), math.tanh(1.70), sig(-1.15)
    c_expect = f * (-0.3) + i * g
    h_expect = o * math.tanh(c_expect)

    assert c_new.data.shape == (1, 1) and h_new.data.shape == (1, 1)
    assert c_new.item() == pytest.approx(c_expect, abs=1e-12)
    assert h_new.item() == pytest.approx(h_expect, abs=1e-12)
    # Literals, so a rewrite of the reference arithmetic cannot drift with it.
    assert c_new.item() == pytest.approx(0.6181346614921708, abs=1e-12)
    assert h_new.item() == pytest.approx(0.1322276163650271, abs=1e-12)


def test_gru_update_gate_holds_the_old_state():
    """``h' = (1 - z) * n + z * h``: z near 1 keeps h, not n.

    The mirrored convention ``h' = z * n + (1 - z) * h`` would return the
    candidate here instead. Both train; only one loads someone else's weights.
    """
    hs = 2
    with default_dtype(np.float64):
        cell = GRUCell(3, hs, rng=np.random.default_rng(14))
        cell.weight_ih.data[...] = 0.0
        cell.weight_hh.data[...] = 0.0
        cell.bias.data[...] = 0.0
        x = Tensor(np.zeros((1, 3)))
        h = Tensor(np.full((1, hs), 0.7))

        cell.bias.data[hs : 2 * hs] = 20.0  # z ~ 1  ->  keep the old state
        assert np.allclose(cell(x, h).data, 0.7, atol=1e-8)

        cell.bias.data[hs : 2 * hs] = -20.0  # z ~ 0  ->  take the candidate
        assert np.allclose(cell(x, h).data, 0.0, atol=1e-8)  # n = tanh(0) = 0


def test_gru_reset_gate_multiplies_the_recurrent_projection():
    """r gates ``h @ W_hn``, not ``h``. The two differ off the diagonal."""
    hs = 2
    m = np.array([[0.3, 1.4], [-0.9, 0.2]])
    with default_dtype(np.float64):
        cell = GRUCell(3, hs, rng=np.random.default_rng(15))
        cell.weight_ih.data[...] = 0.0
        cell.weight_hh.data[...] = 0.0
        cell.weight_hh.data[:, 2 * hs : 3 * hs] = m
        cell.bias.data[...] = np.array([2.0, -2.0, 0.0, 0.0, 0.1, -0.2])
        h_arr = np.array([[0.6, -0.4]])
        out = cell(Tensor(np.zeros((1, 3))), Tensor(h_arr)).data

    r = 1.0 / (1.0 + np.exp(-np.array([[2.0, -2.0]])))
    z = np.full((1, hs), 0.5)
    b_n = np.array([[0.1, -0.2]])
    n_after = np.tanh(b_n + r * (h_arr @ m))  # this implementation / PyTorch
    n_before = np.tanh(b_n + (r * h_arr) @ m)  # Cho's original placement

    assert not np.allclose(n_after, n_before)  # the test would be vacuous otherwise
    assert np.allclose(out, (1.0 - z) * n_after + z * h_arr, atol=1e-12)


def test_bidirectional_output_is_time_aligned():
    """The reverse pass must be un-reversed before stacking.

    Skip the reversal and every backward-direction feature is paired with the
    wrong timestep. Shapes stay correct; only accuracy quietly suffers.
    """
    rng = np.random.default_rng(16)
    model = RNN(I, H, bidirectional=True, rng=rng)
    x = Tensor(rng.standard_normal((N, 4, I)).astype(np.float32))
    out, _ = model(x)
    back_cell = model.cells[1]
    # The reverse direction starts at t = T-1 from a zero state, so its own
    # first step is what belongs at output[:, T-1, H:].
    first_back = back_cell(x[:, 3])
    assert np.allclose(out.data[:, 3, H:], first_back.data, atol=1e-6)


# --------------------------------------------------------------------------- #
# truncated BPTT and deep unrolls
# --------------------------------------------------------------------------- #
def test_detach_state_cuts_the_graph_between_chunks():
    rng = np.random.default_rng(17)
    model = LSTM(I, H, rng=rng)
    x = Tensor(rng.standard_normal((N, 4, I)).astype(np.float32))
    h0 = Tensor(np.zeros((1, N, H), dtype=np.float32), requires_grad=True)
    c0 = Tensor(np.zeros((1, N, H), dtype=np.float32), requires_grad=True)

    _, state = model(x, (h0, c0))
    cut = detach_state(state)
    assert all(s.is_leaf and not s.requires_grad for s in cut)
    assert not np.shares_memory(cut[0].data, np.empty(0))  # values survive
    assert np.allclose(cut[0].data, state[0].data)

    out2, _ = model(x, cut)
    out2.backward(np.ones_like(out2.data))
    # Chunk 2's backward stopped at the boundary: chunk 1's inputs saw nothing.
    assert h0.grad is None and c0.grad is None

    # Without the detach the same call does reach them.
    model.zero_grad()
    out3, _ = model(x, state)
    out3.backward(np.ones_like(out3.data))
    assert h0.grad is not None and np.any(h0.grad != 0.0)


def test_deep_unroll_does_not_recurse():
    """1500 timesteps: a recursive topological sort dies here, an iterative one does not."""
    steps = 1500
    rng = np.random.default_rng(18)
    model = LSTM(3, 4, rng=rng)
    x = Tensor(rng.standard_normal((2, steps, 3)).astype(np.float32))
    out, _ = model(x)
    assert out.shape == (2, steps, 4)
    out.backward(np.ones_like(out.data))  # must not raise RecursionError
    assert model.cells[0].weight_hh.grad is not None
    assert np.all(np.isfinite(model.cells[0].weight_hh.grad))


# --------------------------------------------------------------------------- #
# plumbing: shapes, dropout, clipping-friendliness, state_dict
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [RNN, LSTM, GRU])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_output_and_state_shapes(cls, bidirectional):
    layers, dirs = 2, 2 if bidirectional else 1
    model = cls(I, H, num_layers=layers, bidirectional=bidirectional, rng=np.random.default_rng(19))
    x = Tensor(np.random.default_rng(20).standard_normal((N, T, I)).astype(np.float32))
    out, state = model(x)
    assert out.shape == (N, T, H * dirs)
    for s in state if isinstance(state, tuple) else (state,):
        assert s.shape == (layers * dirs, N, H)


def test_dropout_is_identity_in_eval_and_active_in_train():
    rng = np.random.default_rng(21)
    model = GRU(I, H, num_layers=2, dropout=0.5, rng=rng)
    x = Tensor(rng.standard_normal((N, 4, I)).astype(np.float32))
    model.eval()
    a, _ = model(x)
    b, _ = model(x)
    assert np.allclose(a.data, b.data)
    model.train()
    c, _ = model(x, rng=np.random.default_rng(0))
    d, _ = model(x, rng=np.random.default_rng(1))
    assert not np.allclose(c.data, d.data)


def test_single_layer_dropout_is_never_applied():
    """Dropout goes *between* layers, so a 1-layer stack must be deterministic."""
    rng = np.random.default_rng(22)
    model = RNN(I, H, num_layers=1, dropout=0.9, rng=rng).train()
    x = Tensor(rng.standard_normal((N, 4, I)).astype(np.float32))
    a, _ = model(x)
    b, _ = model(x)
    assert np.allclose(a.data, b.data)


def test_every_parameter_gets_a_gradient_for_the_clipper():
    """A global-norm clipper needs a grad on every parameter, not most of them."""
    rng = np.random.default_rng(23)
    model = LSTM(I, H, num_layers=2, bidirectional=True, rng=rng)
    x = Tensor(rng.standard_normal((N, 5, I)).astype(np.float32))
    out, _ = model(x)
    out.backward(np.ones_like(out.data))
    total = 0.0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert p.grad.shape == p.data.shape
        total += float(np.sum(p.grad.astype(np.float64) ** 2))
    assert np.isfinite(total) and total > 0.0


def test_weight_layout_and_gate_widths():
    for cls, gates in ((RNNCell, 1), (GRUCell, 3), (LSTMCell, 4)):
        cell = cls(I, H, rng=np.random.default_rng(24))
        assert cell.weight_ih.shape == (I, gates * H)
        assert cell.weight_hh.shape == (H, gates * H)
        assert cell.bias.shape == (gates * H,)


def test_recurrent_weight_blocks_are_orthogonal():
    """One orthogonal (H, H) per gate -- singular values 1, so W^T of a long
    unroll neither vanishes nor explodes on step one."""
    cell = LSTMCell(I, H, rng=np.random.default_rng(25))
    w = cell.weight_hh.data.astype(np.float64)
    for k in range(4):
        block = w[:, k * H : (k + 1) * H]
        # float32 storage: orthogonality survives to ~1e-7, not to 1e-10.
        assert np.allclose(block @ block.T, np.eye(H), atol=1e-6)


def test_state_dict_roundtrip():
    a = LSTM(I, H, num_layers=2, rng=np.random.default_rng(26))
    b = LSTM(I, H, num_layers=2, rng=np.random.default_rng(27))
    x = Tensor(np.random.default_rng(28).standard_normal((N, 5, I)).astype(np.float32))
    assert not np.allclose(a(x)[0].data, b(x)[0].data)
    b.load_state_dict(a.state_dict(deep=True))
    assert np.allclose(a(x)[0].data, b(x)[0].data)


def test_shape_errors_are_explicit():
    model = LSTM(I, H, rng=np.random.default_rng(29))
    with pytest.raises(ValueError):
        model(Tensor(np.zeros((N, I), dtype=np.float32)))  # missing the time axis
    with pytest.raises(ValueError):
        model(Tensor(np.zeros((N, T, I + 1), dtype=np.float32)))
    with pytest.raises(ValueError):
        RNNCell(I, H, nonlinearity="gelu", rng=np.random.default_rng(30))


# --------------------------------------------------------------------------- #
# state and index validation
# --------------------------------------------------------------------------- #
def test_embedding_rejects_boolean_indices():
    """A bool array is a numpy *mask*, not a gather.

    ``weight[mask]`` selects the rows where ``mask`` is True, so an all-True
    mask of length ``VOCAB`` returns the whole table -- the right *shape* for
    ``VOCAB`` ids and entirely the wrong rows. The dtype guard used to admit
    kind ``'b'``, so this went through in silence.
    """
    emb = Embedding(VOCAB, DIM, rng=np.random.default_rng(31))
    all_true = np.ones(VOCAB, dtype=bool)  # would have gathered rows 0..V-1
    with pytest.raises(TypeError, match="boolean"):
        emb(all_true)
    partial = np.zeros(VOCAB, dtype=bool)
    partial[[1, 4]] = True  # would have gathered 2 rows for VOCAB ids
    with pytest.raises(TypeError, match="boolean"):
        emb(partial)
    # 0/1 ids spelled as integers remain perfectly legal.
    assert emb(all_true.astype(np.int64)).shape == (VOCAB, DIM)


@pytest.mark.parametrize("cls", [RNNCell, GRUCell, LSTMCell])
def test_cell_rejects_a_state_that_would_broadcast(cls):
    """A batch-mismatched state must fail loudly, not broadcast.

    ``x @ W_ih`` is ``(N, G*H)`` and ``h @ W_hh`` is ``(1, G*H)``: the sum is
    perfectly conformable, so without a check the whole batch quietly shares
    one hidden state and training proceeds on a wrong quantity.
    """
    cell = cls(I, H, rng=np.random.default_rng(32))
    x = Tensor(np.zeros((N, I), dtype=np.float32))
    one_row = Tensor(np.zeros((1, H), dtype=np.float32))
    full = Tensor(np.zeros((N, H), dtype=np.float32))
    state = (one_row, full) if cls is LSTMCell else one_row
    with pytest.raises(ValueError, match="hidden state"):
        cell(x, state)
    if cls is LSTMCell:
        with pytest.raises(ValueError, match="cell state"):
            cell(x, (full, one_row))


@pytest.mark.parametrize("cls", [RNNCell, GRUCell])
def test_cell_rejects_a_state_of_the_wrong_width(cls):
    cell = cls(I, H, rng=np.random.default_rng(33))
    x = Tensor(np.zeros((N, I), dtype=np.float32))
    with pytest.raises(ValueError, match="hidden state"):
        cell(x, Tensor(np.zeros((N, H + 1), dtype=np.float32)))


def test_lstm_cell_rejects_a_bare_hidden_state():
    """``h`` where ``(h, c)`` belongs used to unpack the Tensor by rows."""
    cell = LSTMCell(I, H, rng=np.random.default_rng(34))
    x = Tensor(np.zeros((N, I), dtype=np.float32))
    with pytest.raises(TypeError, match=r"\(h, c\) pair"):
        cell(x, Tensor(np.zeros((N, H), dtype=np.float32)))


def test_sequence_state_of_the_wrong_type_says_what_was_expected():
    """Confusing an RNN's ``h`` for an LSTM's ``(h, c)`` is the obvious slip.

    Both directions used to escape ``_initial`` as a raw unpacking ValueError
    or an AttributeError on ``.shape``, naming neither argument nor shape.
    """
    rng = np.random.default_rng(35)
    x = Tensor(rng.standard_normal((N, 4, I)).astype(np.float32))
    h = Tensor(np.zeros((1, N, H), dtype=np.float32))

    lstm = LSTM(I, H, rng=rng)
    with pytest.raises(TypeError, match=r"\(h0, c0\) pair"):
        lstm(x, h)

    gru = GRU(I, H, rng=rng)
    with pytest.raises(TypeError, match="hx must be a Tensor"):
        gru(x, (h, h))
    with pytest.raises(TypeError, match="hx must be a Tensor"):
        gru(x, np.zeros((1, N, H), dtype=np.float32))


# --------------------------------------------------------------------------- #
# the loss-reduction / clipping-threshold claim, made executable
# --------------------------------------------------------------------------- #
def test_clipping_threshold_is_not_dimensionless_across_reductions():
    """Pin the module docstring's claim about reductions and clipping.

    Per-token cross-entropy at initialisation is ``ln(V)`` nats and the loss
    summed over the ``tokens`` positions is ``tokens * ln(V)``; the gradient
    scales by exactly the same factor. One fixed ``max_norm`` is therefore two
    different operations: a no-op under ``reduction='mean'`` and a ``tokens``-fold
    haircut under ``reduction='sum'``.
    """
    rng = np.random.default_rng(36)
    seqs, steps = 3, T
    emb = Embedding(VOCAB, DIM, rng=rng).double()
    rnn = RNN(DIM, H, rng=rng).double()
    head = Linear(H, VOCAB, rng=rng).double()
    # A zero head makes the logits exactly uniform, which is what turns "about
    # ln(V)" into an equality; a random head sits a little above it.
    head.weight.data[...] = 0.0
    head.bias.data[...] = 0.0
    modules = (emb, rnn, head)
    params = [p for m in modules for p in m.parameters()]

    ids = rng.integers(0, VOCAB, size=(seqs, steps))
    targets = rng.integers(0, VOCAB, size=(seqs, steps))

    def loss(reduction: str) -> Tensor:
        out, _ = rnn(emb(ids))
        return F.cross_entropy(head(out), targets, reduction=reduction)

    tokens = seqs * steps
    ln_v = math.log(VOCAB)
    assert abs(float(loss("mean").data) - ln_v) < 1e-12
    # Per sequence that summed loss is `steps * ln(V)`; over the batch, one
    # factor of `tokens`.
    assert abs(float(loss("sum").data) / (tokens * ln_v) - 1.0) < 1e-12

    def grad_norm(reduction: str, max_norm: float) -> tuple[float, list, list]:
        for m in modules:
            m.zero_grad()
        loss(reduction).backward()
        before = [p.grad.copy() for p in params]
        total = clip_grad_norm_(params, max_norm)
        return total, before, [p.grad.copy() for p in params]

    huge = 1e9  # measure the norms without clipping either of them
    norm_mean, _, _ = grad_norm("mean", huge)
    norm_sum, _, _ = grad_norm("sum", huge)
    assert norm_mean > 0.0
    assert abs(norm_sum / (tokens * norm_mean) - 1.0) < 1e-12

    # A threshold picked while training under 'mean' -- comfortably slack there.
    thr = 2.0 * norm_mean
    _, before, after = grad_norm("mean", thr)
    for b, a in zip(before, after, strict=True):
        assert np.array_equal(b, a), "clipping fired under 'mean' when it should not"

    _, before, after = grad_norm("sum", thr)
    # Same threshold, same model, same data: under 'sum' it is a `tokens / 2`
    # haircut. That factor is the whole content of "a threshold is not
    # dimensionless" -- it is `1 / tokens` of the update the tuning implied.
    coef = thr / (norm_sum + 1e-6)  # `clip_grad_norm_`'s zero-guard
    assert abs(coef - 2.0 / tokens) < 1e-6
    for b, a in zip(before, after, strict=True):
        assert np.allclose(a, b * coef, rtol=1e-9, atol=0.0)
