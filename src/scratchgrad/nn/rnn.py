"""Recurrent layers: token embeddings, RNN/LSTM/GRU cells, and unrolled wrappers.

Everything here is built on one frozen shape convention, restated because a
recurrent checkpoint saved under a different one is silently unloadable:

* ``weight_ih`` is ``(input_size, gates * hidden_size)``;
* ``weight_hh`` is ``(hidden_size, gates * hidden_size)``;
* one step is ``x @ W_ih + h @ W_hh + b`` -- no transpose anywhere.
* **LSTM gate order along the gate axis is ``i, f, g, o``** (input, forget,
  cell-candidate, output), which is PyTorch's order. The forget-gate slice is
  therefore ``[:, H:2H]``; the once-standard trick of initialising the forget
  bias to 1 means writing to *that* slice and nowhere else.
* **GRU gate order is ``r, z, n``** (reset, update, candidate), also PyTorch's.

GRU polarity
------------
This implements the Cho/PyTorch convention, in which ``z`` gates the **old**
state::

    r  = sigmoid(x @ W_ir + h @ W_hr + b_r)
    z  = sigmoid(x @ W_iz + h @ W_hz + b_z)
    n  = tanh(x @ W_in + b_n + r * (h @ W_hn))
    h' = (1 - z) * n + z * h

The other convention in circulation is ``h' = z * n + (1 - z) * h``. The two are
*not* equivalent -- they are related by ``z -> 1 - z``, which a freshly trained
network can absorb into ``W_*z`` and ``b_z`` but an imported checkpoint cannot.
Note also where the reset gate lands: on ``h @ W_hn``, *after* the recurrent
matmul (the CuDNN/PyTorch placement), not on ``h`` before it. Cho's original
paper applies it before, giving ``tanh(x @ W_in + (r * h) @ W_hn)``, and the two
differ numerically as soon as ``W_hn`` is not diagonal.

Biases
------
Each cell carries a single ``bias`` of shape ``(gates * hidden_size,)``, matching
the frozen ``x @ W_ih + h @ W_hh + b``. PyTorch stores two (``b_ih`` and
``b_hh``) purely because CuDNN's kernel signature has two slots; for RNN and
LSTM their sum is the parameter here, so converting a checkpoint just adds them.
For GRU the reduction is not exact: PyTorch keeps ``b_hn`` *inside* the reset
gate, and with one merged vector there is nowhere to put it, so ``b_n`` sits
outside. Conversion is exact only when ``b_hn`` is zero.

Truncated BPTT and graph depth
------------------------------
Unrolling ``T`` steps builds a graph ``T`` nodes deep. That is precisely why
``scratchgrad.tensor``'s topological sort uses an explicit stack instead of
recursion: at ``T = 1500`` a recursive post-order sort blows CPython's
1000-frame limit and a perfectly legitimate model dies with ``RecursionError``.

Depth is still a memory and time cost, so long sequences are trained in chunks
with the hidden state carried across chunk boundaries **detached**::

    h = None
    for chunk in chunks:
        out, h = model(chunk, h)
        loss(out).backward()
        opt.step(); opt.zero_grad()
        h = detach_state(h)      # keep the values, drop the history

Without the detach the graph grows without bound across the whole epoch and each
``backward`` walks every earlier chunk again.

Gradient clipping is not dimensionless
--------------------------------------
Per-token cross-entropy at initialisation is ``ln(V)`` nats (``V`` = vocabulary
size); summed over ``T`` timesteps it is ``T * ln(V)``, and the gradient scales
the same way. A clipping threshold therefore only means something once the
reduction is named: "clip at 5" under ``reduction='mean'`` is "clip at ``5 * T``"
under ``reduction='sum'``. Fix the reduction first, then tune the threshold --
and when a clip value is copied from a paper or a repo, check which reduction it
was measured against before trusting it.

Initialisation
--------------
Recurrent matrices (``weight_hh``) get :func:`scratchgrad.nn.init.orthogonal_`,
one orthogonal ``(H, H)`` block per gate. The reason is the eigenvalues: the
Jacobian of an unrolled RNN is a product of ``T`` recurrent matrices, so the
gradient's magnitude is governed by their singular values raised to the ``T``.
An orthogonal matrix has all singular values exactly 1, the boundary between
exponential decay and explosion -- the only starting point from which a long
unroll neither vanishes nor blows up on step one. Input matrices (``weight_ih``)
are not iterated and get plain Xavier.
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor, concatenate, stack
from . import functional as F
from . import init
from .module import Module, ModuleList, Parameter

__all__ = [
    "GRU",
    "LSTM",
    "RNN",
    "Embedding",
    "GRUCell",
    "LSTMCell",
    "RNNCell",
    "detach_state",
    "embedding",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Branch-on-sign logistic: ``exp`` is only ever applied to a negative number."""
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #
def embedding(weight: Tensor, idx, padding_idx: int | None = None) -> Tensor:
    """Gather rows of ``weight`` at integer positions ``idx``.

    Forward is ``weight[idx]``; the backward is a **scatter-add**, not a scatter.
    A token that appears twice in the batch contributes twice to its row, and
    ``acc[idx] = g`` would keep only the last write -- under-training exactly the
    most frequent tokens while the loss curve looks entirely healthy.
    ``np.add.at`` is the buffered, accumulate-on-collision form.

    Args:
        weight: ``(num_embeddings, embedding_dim)`` table.
        idx: integer array of any shape.
        padding_idx: row that receives an exactly zero gradient, if any.

    Returns:
        Tensor of shape ``idx.shape + (embedding_dim,)``.
    """
    ids = np.asarray(idx)
    if ids.dtype.kind == "b":
        # numpy would take a bool array as a *mask* -- `weight[mask]` selects the
        # rows where True instead of gathering row 0 / row 1, so the result has a
        # data-dependent shape and the "one vector per token" contract is gone.
        raise TypeError(
            "embedding indices must be integers, got a boolean array: numpy would "
            "apply it as a mask over rows, not as an integer gather (cast with "
            ".astype(np.int64) if 0/1 ids were meant)"
        )
    if ids.dtype.kind not in "iu":
        raise TypeError(f"embedding indices must be integers, got dtype {ids.dtype}")
    num, dim = weight.data.shape
    if ids.size and (ids.min() < 0 or ids.max() >= num):
        raise IndexError(f"index out of range for {num} embeddings: [{ids.min()}, {ids.max()}]")
    flat = ids.reshape(-1)
    out = weight.data[flat].reshape(ids.shape + (dim,))

    def bw(g, flat=flat, shape=weight.data.shape, dtype=weight.data.dtype, pad=padding_idx):
        acc = np.zeros(shape, dtype=dtype)
        np.add.at(acc, flat, np.asarray(g).reshape(-1, shape[1]))
        if pad is not None:
            acc[pad] = 0.0
        return acc

    return Tensor._make(out, (weight,), (bw,), "embedding")


class Embedding(Module):
    """A lookup table mapping integer ids to dense vectors.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: width of each row.
        padding_idx: if given, this row is zeroed at init and pinned at a zero
            gradient, so padded positions never pull on the table.
        rng: explicit generator for the ``N(0, 1)`` initialisation.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        rng=None,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        if padding_idx is not None:
            padding_idx = int(padding_idx)
            if not 0 <= padding_idx < self.num_embeddings:
                raise ValueError(f"padding_idx {padding_idx} outside [0, {self.num_embeddings})")
        self.padding_idx = padding_idx
        self.weight = Parameter(np.zeros((self.num_embeddings, self.embedding_dim)))
        init.normal_(self.weight, 0.0, 1.0, rng=rng)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx] = 0.0

    def forward(self, idx) -> Tensor:
        ids = idx.data if isinstance(idx, Tensor) else idx
        return embedding(self.weight, ids, self.padding_idx)

    def extra_repr(self) -> str:
        pad = "" if self.padding_idx is None else f", padding_idx={self.padding_idx}"
        return f"{self.num_embeddings}, {self.embedding_dim}{pad}"


# --------------------------------------------------------------------------- #
# fused single-step kernels
# --------------------------------------------------------------------------- #
def _rnn_step(x: Tensor, h: Tensor, w_ih: Tensor, w_hh: Tensor, b, nonlinearity: str) -> Tensor:
    """One vanilla recurrent step, fused into a single graph node."""
    xd, hd, wi, wh = x.data, h.data, w_ih.data, w_hh.data
    pre = xd @ wi + hd @ wh
    if b is not None:
        pre = pre + b.data
    if nonlinearity == "tanh":
        out = np.tanh(pre)
        dact = 1.0 - out * out
    elif nonlinearity == "relu":
        out = np.maximum(pre, 0.0)
        dact = (pre > 0).astype(pre.dtype)
    else:
        raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}")

    def dz(g):
        return np.asarray(g) * dact

    parents = [x, h, w_ih, w_hh]
    fns = [
        lambda g: dz(g) @ wi.T,
        lambda g: dz(g) @ wh.T,
        lambda g: xd.T @ dz(g),
        lambda g: hd.T @ dz(g),
    ]
    if b is not None:
        parents.append(b)
        fns.append(lambda g: dz(g).sum(axis=0))
    return Tensor._make(out, parents, fns, "rnn_step")


def _lstm_step(x: Tensor, h: Tensor, c: Tensor, w_ih: Tensor, w_hh: Tensor, b) -> Tensor:
    """One LSTM step, emitting ``(N, 2H)`` = ``[h_new | c_new]``.

    Both outputs leave a *single* node so the shared gate algebra is
    differentiated once. The caller slices the halves apart again; that slice
    backward is exact, whereas two nodes over the same parents would redo the
    whole gate backward twice per timestep.
    """
    xd, hd, cd, wi, wh = x.data, h.data, c.data, w_ih.data, w_hh.data
    hs = hd.shape[1]
    z = xd @ wi + hd @ wh
    if b is not None:
        z = z + b.data
    i = _sigmoid(z[:, 0:hs])
    f = _sigmoid(z[:, hs : 2 * hs])
    gg = np.tanh(z[:, 2 * hs : 3 * hs])
    o = _sigmoid(z[:, 3 * hs : 4 * hs])
    c_new = f * cd + i * gg
    tc = np.tanh(c_new)
    h_new = o * tc
    out = np.concatenate([h_new, c_new], axis=1)

    # One-slot cache: ``Tensor._make``'s backward hands the *same* cotangent
    # object to every parent's rule, so the gate algebra runs once per backward
    # pass instead of once per parameter. Holding a reference to ``g`` is what
    # keeps the identity test honest -- a freed array's slot could be reused.
    slot: dict = {"g": None}

    def grads(g):
        g = np.asarray(g)
        if slot["g"] is not g:
            gh, gc = g[:, :hs], g[:, hs:]
            # The cell path receives the upstream cotangent on c' *plus* what
            # arrives through h' = o * tanh(c'). Dropping the second term is the
            # classic LSTM backward bug: it still trains, just without the
            # constant-error carousel that is the entire point of the cell.
            dc = gc + gh * o * (1.0 - tc * tc)
            dzs = np.concatenate(
                [
                    (dc * gg) * i * (1.0 - i),
                    (dc * cd) * f * (1.0 - f),
                    (dc * i) * (1.0 - gg * gg),
                    (gh * tc) * o * (1.0 - o),
                ],
                axis=1,
            )
            slot.update(g=g, dz=dzs, dc=dc * f)
        return slot["dz"], slot["dc"]

    parents = [x, h, c, w_ih, w_hh]
    fns = [
        lambda g: grads(g)[0] @ wi.T,
        lambda g: grads(g)[0] @ wh.T,
        lambda g: grads(g)[1],
        lambda g: xd.T @ grads(g)[0],
        lambda g: hd.T @ grads(g)[0],
    ]
    if b is not None:
        parents.append(b)
        fns.append(lambda g: grads(g)[0].sum(axis=0))
    return Tensor._make(out, parents, fns, "lstm_step")


def _gru_step(x: Tensor, h: Tensor, w_ih: Tensor, w_hh: Tensor, b) -> Tensor:
    """One GRU step. Gate order ``r, z, n``; ``z`` gates the OLD state."""
    xd, hd, wi, wh = x.data, h.data, w_ih.data, w_hh.data
    hs = hd.shape[1]
    gi = xd @ wi
    if b is not None:
        gi = gi + b.data
    gh = hd @ wh
    gh_n = gh[:, 2 * hs : 3 * hs]
    r = _sigmoid(gi[:, 0:hs] + gh[:, 0:hs])
    zg = _sigmoid(gi[:, hs : 2 * hs] + gh[:, hs : 2 * hs])
    n = np.tanh(gi[:, 2 * hs : 3 * hs] + r * gh_n)
    out = (1.0 - zg) * n + zg * hd

    slot: dict = {"g": None}

    def grads(g):
        g = np.asarray(g)
        if slot["g"] is not g:
            an = (g * (1.0 - zg)) * (1.0 - n * n)
            az = (g * (hd - n)) * zg * (1.0 - zg)
            ar = (an * gh_n) * r * (1.0 - r)
            dgi = np.concatenate([ar, az, an], axis=1)
            # The reset gate multiplies the *hidden* branch only, so the
            # candidate column of the recurrent cotangent carries an extra `r`
            # that the input-side one must not have.
            dgh = np.concatenate([ar, az, an * r], axis=1)
            # h reaches the output three ways: through r, through z, and
            # straight through the `z * h` term. That last path is easy to lose.
            slot.update(g=g, gi=dgi, gh=dgh, direct=g * zg)
        return slot

    parents = [x, h, w_ih, w_hh]
    fns = [
        lambda g: grads(g)["gi"] @ wi.T,
        lambda g: grads(g)["direct"] + grads(g)["gh"] @ wh.T,
        lambda g: xd.T @ grads(g)["gi"],
        lambda g: hd.T @ grads(g)["gh"],
    ]
    if b is not None:
        parents.append(b)
        fns.append(lambda g: grads(g)["gi"].sum(axis=0))
    return Tensor._make(out, parents, fns, "gru_step")


# --------------------------------------------------------------------------- #
# cells
# --------------------------------------------------------------------------- #
def _orthogonal_blocks_(weight: Parameter, gates: int, rng) -> Parameter:
    """Fill ``(H, gates*H)`` with one independent orthogonal ``(H, H)`` per gate.

    Calling ``orthogonal_`` on the whole ``(H, gates*H)`` matrix gives
    orthonormal *rows of the concatenation*, which constrains the gate blocks
    against each other instead of making each one unit-singular-valued on its
    own -- and it is each block separately that the unroll raises to the ``T``.
    """
    hs = weight.data.shape[0]
    block = Tensor(np.zeros((hs, hs), dtype=weight.data.dtype))
    for k in range(gates):
        init.orthogonal_(block, rng=rng)
        weight.data[:, k * hs : (k + 1) * hs] = block.data
    return weight


class _CellBase(Module):
    """Shared construction and validation for the three recurrent cells."""

    _gates: int = 1

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, rng=None) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        gates = self._gates
        rng = init.default_rng(rng)
        self.weight_ih = Parameter(np.zeros((self.input_size, gates * self.hidden_size)))
        self.weight_hh = Parameter(np.zeros((self.hidden_size, gates * self.hidden_size)))
        init.xavier_uniform_(self.weight_ih, rng=rng)
        _orthogonal_blocks_(self.weight_hh, gates, rng)
        if bias:
            self.bias = Parameter(np.zeros(gates * self.hidden_size))
        else:
            self.register_parameter("bias", None)

    def _zero_state(self, x: Tensor) -> Tensor:
        return Tensor(np.zeros((x.data.shape[0], self.hidden_size), dtype=x.data.dtype))

    def _check_input(self, x: Tensor) -> None:
        if x.ndim != 2 or x.data.shape[1] != self.input_size:
            raise ValueError(
                f"{type(self).__name__} expects input (N, {self.input_size}), got {x.shape}"
            )

    def _check_state(self, x: Tensor, h: Tensor, what: str = "hidden state") -> None:
        """Validate a state against ``x``'s batch and this layer's width.

        Unchecked, a batch-mismatched state does not fail: ``x @ W_ih`` is
        ``(N, G*H)`` and ``h @ W_hh`` is ``(1, G*H)``, so the sum *broadcasts*
        and every row of the batch shares one hidden state. Training carries on
        at a plausible-looking loss with nothing in the shapes to say why.
        """
        if not isinstance(h, Tensor):
            raise TypeError(
                f"{type(self).__name__} expects {what} as a Tensor, got {type(h).__name__}"
            )
        want = (x.data.shape[0], self.hidden_size)
        if tuple(h.shape) != want:
            raise ValueError(f"{type(self).__name__} expects {what} {want}, got {tuple(h.shape)}")

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}, bias={self.bias is not None}"


class RNNCell(_CellBase):
    """``h' = act(x @ W_ih + h @ W_hh + b)`` for one timestep."""

    _gates = 1

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        rng=None,
    ) -> None:
        super().__init__(input_size, hidden_size, bias=bias, rng=rng)
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}")
        self.nonlinearity = nonlinearity

    def forward(self, x: Tensor, h: Tensor | None = None) -> Tensor:
        self._check_input(x)
        if h is None:
            h = self._zero_state(x)
        else:
            self._check_state(x, h)
        return _rnn_step(x, h, self.weight_ih, self.weight_hh, self.bias, self.nonlinearity)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", nonlinearity={self.nonlinearity!r}"


class LSTMCell(_CellBase):
    """One LSTM step. Gate order along the gate axis is ``i, f, g, o``.

    ``c' = f * c + i * g`` and ``h' = o * tanh(c')``. ``forward`` returns the
    pair ``(h', c')``; pass the same pair back in as ``state`` next timestep.
    """

    _gates = 4

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor] | None = None):
        self._check_input(x)
        if state is None:
            h, c = self._zero_state(x), self._zero_state(x)
        else:
            if not (isinstance(state, (tuple, list)) and len(state) == 2):
                raise TypeError(
                    f"LSTMCell expects its state as an (h, c) pair, got {type(state).__name__}"
                )
            h, c = state
            self._check_state(x, h, "hidden state h")
            self._check_state(x, c, "cell state c")
        out = _lstm_step(x, h, c, self.weight_ih, self.weight_hh, self.bias)
        hs = self.hidden_size
        return out[:, :hs], out[:, hs:]


class GRUCell(_CellBase):
    """One GRU step. Gate order ``r, z, n``; ``h' = (1 - z) * n + z * h``."""

    _gates = 3

    def forward(self, x: Tensor, h: Tensor | None = None) -> Tensor:
        self._check_input(x)
        if h is None:
            h = self._zero_state(x)
        else:
            self._check_state(x, h)
        return _gru_step(x, h, self.weight_ih, self.weight_hh, self.bias)


# --------------------------------------------------------------------------- #
# sequence wrappers
# --------------------------------------------------------------------------- #
def detach_state(state):
    """Strip the autograd history from a hidden state, keeping its values.

    Accepts a Tensor or an arbitrarily nested tuple/list of them, so it covers
    both the ``h`` of an RNN/GRU and the ``(h, c)`` of an LSTM. This is the whole
    of truncated BPTT: carry the numbers across a chunk boundary, cut the graph,
    and the next ``backward`` stops at the boundary instead of walking every
    chunk seen so far.
    """
    if state is None:
        return None
    if isinstance(state, Tensor):
        return state.detach()
    if isinstance(state, (tuple, list)):
        return type(state)(detach_state(s) for s in state)
    raise TypeError(f"cannot detach state of type {type(state).__name__}")


class _SequenceBase(Module):
    """Unrolls a cell over the time axis of a batch-first ``(N, T, input)`` input."""

    _cell_cls: type = RNNCell
    _has_cell_state: bool = False

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        rng=None,
        **cell_kwargs,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1
        self.dropout = float(dropout)
        # One generator for the whole stack: handed `rng=None` each cell would
        # spin up its own OS-seeded generator and the model would stop being
        # reproducible from a single seed.
        rng = init.default_rng(rng)
        self.rng = rng
        cells = []
        for layer in range(self.num_layers):
            in_size = self.input_size if layer == 0 else self.hidden_size * self.num_directions
            for _ in range(self.num_directions):
                cells.append(
                    self._cell_cls(in_size, self.hidden_size, bias=bias, rng=rng, **cell_kwargs)
                )
        self.cells = ModuleList(cells)

    # -- state ---------------------------------------------------------------
    def _check_state(self, t, n_states: int, batch: int, what: str) -> None:
        # A wrong *type* here -- an RNN's bare `h` handed to an LSTM, or an
        # LSTM's `(h, c)` handed to an RNN -- otherwise escapes as a bare
        # AttributeError or an unpacking ValueError from inside `_initial`,
        # naming neither the argument nor the shape that was wanted.
        if not isinstance(t, Tensor):
            raise TypeError(f"{what} must be a Tensor, got {type(t).__name__}")
        want = (n_states, batch, self.hidden_size)
        if tuple(t.shape) != want:
            raise ValueError(f"{what} must have shape {want}, got {tuple(t.shape)}")

    def _initial(self, x: Tensor, hx) -> tuple[list[Tensor], list[Tensor] | None]:
        n_states = self.num_layers * self.num_directions
        batch = x.data.shape[0]

        def zeros_list() -> list[Tensor]:
            return [
                Tensor(np.zeros((batch, self.hidden_size), dtype=x.data.dtype))
                for _ in range(n_states)
            ]

        if self._has_cell_state:
            if hx is None:
                return zeros_list(), zeros_list()
            if not (isinstance(hx, (tuple, list)) and len(hx) == 2):
                raise TypeError(
                    f"{type(self).__name__} expects its state as an (h0, c0) pair of "
                    f"({n_states}, N, {self.hidden_size}) Tensors, got "
                    f"{type(hx).__name__}"
                )
            h0, c0 = hx
            self._check_state(h0, n_states, batch, "h0")
            self._check_state(c0, n_states, batch, "c0")
            return [h0[k] for k in range(n_states)], [c0[k] for k in range(n_states)]
        if hx is None:
            return zeros_list(), None
        self._check_state(hx, n_states, batch, "hx")
        return [hx[k] for k in range(n_states)], None

    # -- forward -------------------------------------------------------------
    def forward(self, x: Tensor, hx=None, rng=None):
        """Run the stack over time.

        Args:
            x: ``(N, T, input_size)`` -- batch first.
            hx: initial state, shaped ``(num_layers * num_directions, N,
                hidden_size)`` for RNN/GRU and a ``(h0, c0)`` pair of those for
                LSTM. ``None`` means zeros.
            rng: generator for the inter-layer dropout mask; defaults to the one
                the module was built with.

        Returns:
            ``(output, h_n)`` for RNN/GRU, ``(output, (h_n, c_n))`` for LSTM.
            ``output`` is ``(N, T, hidden_size * num_directions)``.
        """
        if x.ndim != 3:
            raise ValueError(f"expected a batch-first (N, T, input_size) input, got {x.shape}")
        if x.data.shape[2] != self.input_size:
            raise ValueError(
                f"input has {x.data.shape[2]} features, layer was built for {self.input_size}"
            )
        steps = x.data.shape[1]
        if steps == 0:
            raise ValueError("cannot unroll over a zero-length sequence")

        h_state, c_state = self._initial(x, hx)
        gen = self.rng if rng is None else rng
        layer_in = x
        h_last: list[Tensor] = []
        c_last: list[Tensor] = []

        for layer in range(self.num_layers):
            per_direction: list[Tensor] = []
            for d in range(self.num_directions):
                k = layer * self.num_directions + d
                cell = self.cells[k]
                h = h_state[k]
                c = c_state[k] if c_state is not None else None
                order = range(steps) if d == 0 else range(steps - 1, -1, -1)
                outs: list[Tensor] = []
                for t in order:
                    xt = layer_in[:, t]
                    if c is None:
                        h = cell(xt, h)
                    else:
                        h, c = cell(xt, (h, c))
                    outs.append(h)
                if d == 1:
                    # Re-align with wall-clock time before stacking. Left in
                    # visitation order, the reverse pass pairs each output with
                    # the wrong timestep: still a trainable model, still scores
                    # badly, with nothing in the shapes to say why.
                    outs.reverse()
                per_direction.append(stack(outs, axis=1))
                h_last.append(h)
                if c is not None:
                    c_last.append(c)
            layer_in = (
                per_direction[0] if self.num_directions == 1 else concatenate(per_direction, axis=2)
            )
            # Dropout sits *between* layers only -- never on the final output,
            # and never inside the recurrence, where a fresh mask at every
            # timestep destroys the state it is meant to regularise.
            if self.dropout > 0.0 and layer < self.num_layers - 1:
                layer_in = F.dropout(layer_in, self.dropout, self.training, rng=gen)

        h_n = stack(h_last, axis=0)
        if c_state is not None:
            return layer_in, (h_n, stack(c_last, axis=0))
        return layer_in, h_n

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"bidirectional={self.bidirectional}, dropout={self.dropout}"
        )


class RNN(_SequenceBase):
    """Multi-layer Elman RNN over a batch-first ``(N, T, input_size)`` input."""

    _cell_cls = RNNCell
    _has_cell_state = False


class LSTM(_SequenceBase):
    """Multi-layer LSTM. Gate order ``i, f, g, o``; the state is ``(h, c)``."""

    _cell_cls = LSTMCell
    _has_cell_state = True


class GRU(_SequenceBase):
    """Multi-layer GRU. Gate order ``r, z, n``; ``h' = (1 - z) * n + z * h``."""

    _cell_cls = GRUCell
    _has_cell_state = False
