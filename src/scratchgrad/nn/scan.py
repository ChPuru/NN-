"""Linear-recurrence scan, and the two recurrent layers built on it: MinGRU and a selective SSM.

Both layers reduce to one recurrence, run elementwise over a batch-first ``(N, T, ...)``
input::

    h_t = a_t * h_{t-1} + b_t

The coefficients ``a_t`` and ``b_t`` depend on the *input* only -- never on ``h`` -- so
they are computed for all ``T`` steps at once (one GEMM, not ``T`` small ones), and
the recurrence itself becomes a *scan*: the pair ``(a, b)`` is an affine map, and
composing two of them, first ``(a1, b1)`` then ``(a2, b2)``, gives
``(a2 * a1, a2 * b1 + b2)`` -- associative, so it can be evaluated as a tree.

Why not the closed form
-----------------------
Unrolled, ``h_t = A_t * (h_0 + sum_{s<=t} b_s / A_s)`` with ``A = cumprod(a)``. That
is one ``cumprod`` and one ``cumsum``, fully vectorised -- and wrong on any real
sequence. With ``a ~ 0.5``, ``A_t`` underflows to zero after ~150 steps in float32
and ~1075 in float64; ``b_s / A_s`` is then ``inf`` and ``A_t * inf`` is ``NaN``.
The failure is not gradual: the output is exact up to the step where it is NaN.

The log-space version (Heinsen, 2023) computes ``log A`` with a ``cumsum`` and the sum
with a ``logcumsumexp``, which cures the underflow. It needs ``a > 0``, ``b > 0`` and
``h_0 > 0`` for the logarithms to exist -- hence minGRU's log-space variant squeezing
the candidate through a positive ``g()``. A Mamba input term ``delta * B * x`` takes
both signs, so there it would need complex logs. And it spends an ``exp`` and a ``log``
per element to do what one multiply-add does.

What is used instead
--------------------
Odd-even (cyclic) reduction: pair each odd step with the even step before it into one
composite step, solve the half-length problem recursively, then fill in the even
steps. Every level is a single vectorised pass over half as many steps as the last,
so the work is ``O(T)`` and the depth ``O(log T)``. It only multiplies and adds: a
product of many small ``a`` underflowing to zero is the *correct* answer (that
contribution really has decayed away), not the first half of a ``0 * inf``. It still
assumes a stable recurrence, ``|a| <= 1``, which every gated layer here satisfies by
construction; with ``|a| > 1`` the partial products overflow like the recurrence does.

Honest cost. NumPy already vectorises the plain sequential loop over ``N * D``, so
with a short, wide sequence (``T = 128``, ``D = 256``) the loop is ~3x *faster* than
any parallel scan -- ``O(T)`` Python steps are cheap when each one is wide. The scan
wins once ``T`` is long relative to the width (1.6x at ``T = 4096``, ``D = 64``). The
speed-up over an LSTM comes from somewhere else: gates computed by one GEMM for the
whole sequence, and a recurrence that is one graph node instead of ``T``.

The backward is the same scan, reversed
---------------------------------------
With ``g_t`` the cotangent placed directly on ``h_t``, the total cotangent obeys
``lam_t = g_t + a_{t+1} * lam_{t+1}`` from ``lam_T = g_T`` -- a linear recurrence
running backwards in time, so :func:`_scan` computes it too. Then ``db_t = lam_t``,
``da_t = lam_t * h_{t-1}`` and ``dh_0 = a_1 * lam_1``. The coefficient is ``a_{t+1}``,
not ``a_t``: the step that carries ``h_t`` forward is the *next* one. Shifting by the
wrong one still produces a gradient of the right shape and a model that trains, only
worse; the gradcheck is what catches it.
"""

from __future__ import annotations

import math

import numpy as np

from ..tensor import Tensor
from . import functional as F
from . import init
from .layers import Linear
from .module import Module, Parameter

__all__ = ["MinGRU", "SelectiveSSM", "linear_scan", "selective_scan"]


# --------------------------------------------------------------------------- #
# the numpy kernels
# --------------------------------------------------------------------------- #
def _scan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``h_t = a_t h_{t-1} + b_t`` along axis 1 from ``h_{-1} = 0``, by odd-even reduction.

    Step ``o = 2i+1`` composed with step ``e = 2i`` is the single step
    ``(a_o a_e, a_o b_e + b_o)`` from ``h_{e-1}`` to ``h_o``, so the odd positions
    solve a half-length problem of the same form. Each even position is then one
    step on from the odd position before it. ``a_0`` has no effect on the result:
    it only ever multiplies the zero initial state.
    """
    steps = b.shape[1]
    if steps == 1:
        return np.array(b, dtype=np.result_type(a, b), copy=True)
    m = steps // 2
    a_odd, a_even = a[:, 1 : 2 * m : 2], a[:, 0 : 2 * m : 2]
    h_odd = _scan(a_odd * a_even, a_odd * b[:, 0 : 2 * m : 2] + b[:, 1 : 2 * m : 2])
    h = np.empty(b.shape, dtype=np.result_type(a, b))
    h[:, 1::2] = h_odd
    h[:, 0] = b[:, 0]
    if steps > 2:
        h[:, 2::2] = a[:, 2::2] * h_odd[:, : (steps - 1) // 2] + b[:, 2::2]
    return h


def _fold_state(
    a: np.ndarray, b: np.ndarray, h0: np.ndarray | None, owned: bool = False
) -> np.ndarray:
    """Absorb an initial state into the first input: ``b_1 <- b_1 + a_1 h_0``.

    ``owned=True`` says ``b`` is a temporary the caller built, so it is updated in
    place; otherwise it is copied -- it may be a user's tensor data.
    """
    if h0 is None:
        return b
    if not owned or np.result_type(a, b, h0) != b.dtype:
        b = np.array(b, dtype=np.result_type(a, b, h0), copy=True)
    b[:, 0] += a[:, 0] * h0
    return b


def _scan_adjoint(a: np.ndarray, g: np.ndarray) -> np.ndarray:
    """``lam_t = g_t + a_{t+1} lam_{t+1}``, ``lam_T = g_T``: :func:`_scan` backwards in time."""
    g = np.asarray(g)
    shifted = np.empty(a.shape, dtype=a.dtype)
    # Reversed position s holds original step T-1-s, whose adjoint is carried back
    # by a_{T-s}. Slot 0 has no effect on the result, but it is still multiplied,
    # and np.empty could leave an inf there: inf * 0 is a NaN (a FloatingPointError
    # under strict numpy errors).
    shifted[:, 0] = 0.0
    shifted[:, 1:] = a[:, :0:-1]
    return _scan(shifted, g[:, ::-1])[:, ::-1]


def _prev_times(lam: np.ndarray, h: np.ndarray, h0: np.ndarray | None) -> np.ndarray:
    """``lam_t * h_{t-1}``, with ``h_{-1}`` the initial state (zero if absent)."""
    out = np.empty(h.shape, dtype=h.dtype)
    out[:, 1:] = lam[:, 1:] * h[:, :-1]
    out[:, 0] = 0.0 if h0 is None else lam[:, 0] * h0
    return out


# --------------------------------------------------------------------------- #
# differentiable ops
# --------------------------------------------------------------------------- #
def linear_scan(a: Tensor, b: Tensor, h0: Tensor | None = None) -> Tensor:
    """All states of ``h_t = a_t * h_{t-1} + b_t`` over axis 1, as one graph node.

    Args:
        a: ``(N, T, ...)`` decay coefficients. Keep ``|a| <= 1`` (see module doc).
        b: ``(N, T, ...)`` inputs, same shape as ``a``.
        h0: optional ``(N, ...)`` initial state; ``None`` means zeros.

    Returns:
        ``(N, T, ...)`` tensor of ``h_1 .. h_T``.
    """
    ad, bd = a.data, b.data
    if ad.shape != bd.shape or ad.ndim < 2:
        raise ValueError(f"a and b must share one (N, T, ...) shape, got {ad.shape}, {bd.shape}")
    if ad.shape[1] == 0:
        raise ValueError("cannot scan over a zero-length sequence")
    hd = None
    if h0 is not None:
        want = bd.shape[:1] + bd.shape[2:]
        if tuple(h0.shape) != want:
            raise ValueError(f"h0 must have shape {want}, got {tuple(h0.shape)}")
        hd = h0.data
    h = _scan(ad, _fold_state(ad, bd, hd))

    # One-slot cache keyed on the cotangent object, as in rnn.py: every parent's
    # rule needs the same reverse scan, and it should run once, not three times.
    slot: dict = {"g": None}

    def lam(g):
        if slot["g"] is not g:
            slot.update(g=g, lam=_scan_adjoint(ad, g))
        return slot["lam"]

    parents = [a, b]
    fns = [lambda g: _prev_times(lam(g), h, hd), lam]
    if h0 is not None:
        parents.append(h0)
        fns.append(lambda g: ad[:, 0] * lam(g)[:, 0])
    return Tensor._make(h, parents, fns, "linear_scan")


def selective_scan(
    x: Tensor, delta: Tensor, A: Tensor, B: Tensor, h0: Tensor | None = None
) -> Tensor:
    """The selective-SSM state sequence, discretisation and scan fused into one node.

    Per channel ``d`` and state ``s``::

        h_t[d, s] = exp(delta_t[d] * A[d, s]) * h_{t-1}[d, s] + delta_t[d] * B_t[s] * x_t[d]

    i.e. zero-order hold for ``A`` and the ``B_bar ~ delta * B`` shortcut for the input,
    which is what Mamba itself uses.

    Args:
        x: ``(N, T, D)`` input.
        delta: ``(N, T, D)`` positive step sizes.
        A: ``(D, S)`` negative-real diagonal state matrix.
        B: ``(N, T, S)`` input-dependent input matrix.
        h0: optional ``(N, D, S)`` initial state.

    Returns:
        ``(N, T, D, S)`` states. Memory is ``S`` times the input's -- the price of a
        state that is not collapsed to ``(N, T, D)`` until the readout.
    """
    xd, dd, ad, bd = x.data, delta.data, A.data, B.data
    if xd.ndim != 3 or dd.shape != xd.shape:
        raise ValueError(f"x and delta must both be (N, T, D), got {xd.shape}, {dd.shape}")
    n, steps, d = xd.shape
    if ad.ndim != 2 or ad.shape[0] != d:
        raise ValueError(f"A must be ({d}, S), got {ad.shape}")
    s = ad.shape[1]
    if bd.shape != (n, steps, s):
        raise ValueError(f"B must be ({n}, {steps}, {s}), got {bd.shape}")
    hd = None
    if h0 is not None:
        if tuple(h0.shape) != (n, d, s):
            raise ValueError(f"h0 must be ({n}, {d}, {s}), got {tuple(h0.shape)}")
        hd = h0.data

    decay = np.exp(dd[..., None] * ad)  # in (0, 1) for every delta > 0 once A < 0
    dx = dd * xd
    h = _scan(decay, _fold_state(decay, dx[..., None] * bd[:, :, None, :], hd, owned=True))

    slot: dict = {"g": None}

    def grads(g):
        if slot["g"] is not g:
            lam = _scan_adjoint(decay, g)
            # Cotangent on the exponent delta*A: d/dz exp(z) = exp(z), times what
            # reaches the decay, lam_t * h_{t-1}.
            p = _prev_times(lam, h, hd)
            p *= decay
            # lam . B is shared by x and delta (both enter through delta * x * B).
            q = np.einsum("ntds,nts->ntd", lam, bd, optimize=True)
            slot.update(g=g, lam=lam, p=p, q=q)
        return slot

    parents = [x, delta, A, B]
    fns = [
        lambda g: dd * grads(g)["q"],
        lambda g: np.einsum("ntds,ds->ntd", grads(g)["p"], ad, optimize=True) + xd * grads(g)["q"],
        lambda g: np.einsum("ntds,ntd->ds", grads(g)["p"], dd, optimize=True),
        lambda g: np.einsum("ntds,ntd->nts", grads(g)["lam"], dx, optimize=True),
    ]
    if h0 is not None:
        parents.append(h0)
        fns.append(lambda g: decay[:, 0] * grads(g)["lam"][:, 0])
    return Tensor._make(h, parents, fns, "selective_scan")


def _ssm_readout(h: Tensor, C: Tensor) -> Tensor:
    """``y[n, t, d] = sum_s C[n, t, s] h[n, t, d, s]`` as one node.

    ``(h * C).sum(-1)`` would keep the ``(N, T, D, S)`` product alive in the graph and
    allocate its gradient too; the einsum never materialises it.
    """
    hd, cd = h.data, C.data
    y = np.einsum("ntds,nts->ntd", hd, cd, optimize=True)
    return Tensor._make(
        y,
        (h, C),
        (
            lambda g: np.asarray(g)[..., None] * cd[:, :, None, :],
            lambda g: np.einsum("ntds,ntd->nts", hd, g, optimize=True),
        ),
        "ssm_readout",
    )


def _sigmoid_pair(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(sigmoid(x), sigmoid(-x))`` from one ``exp``, applied only to ``-|x|``."""
    e = np.exp(-np.abs(x))
    big = 1.0 / (1.0 + e)
    small = e * big
    pos = x >= 0
    return np.where(pos, big, small), np.where(pos, small, big)


def _mingru_scan(pre: Tensor, h0: Tensor | None = None) -> Tensor:
    """minGRU gating and scan as one node: ``pre = [k | h~]`` ``(N, T, 2H)`` -> ``(N, T, H)``.

    ``a = 1 - z = sigmoid(-k)``, ``b = z * h~``. With ``lam`` and ``da = lam * h_{t-1}``
    from the scan backward, ``da/dk = -z a`` and ``db/dk = z a h~`` give
    ``dk = z a (lam h~ - da)`` and ``dh~ = z lam``. Fused rather than composed from
    ``pre[..., :H]`` slices: a slice's backward scatters through ``np.add.at``, which
    at these sizes costs more than the whole scan.
    """
    pd = pre.data
    hs = pd.shape[2] // 2
    k, cand = pd[:, :, :hs], pd[:, :, hs:]
    z, a = _sigmoid_pair(k)
    hd = None if h0 is None else h0.data
    h = _scan(a, _fold_state(a, z * cand, hd, owned=True))

    slot: dict = {"g": None}

    def lam(g):
        if slot["g"] is not g:
            slot.update(g=g, lam=_scan_adjoint(a, g))
        return slot["lam"]

    def grad_pre(g):
        lg = lam(g)
        dk = z * a * (lg * cand - _prev_times(lg, h, hd))
        return np.concatenate([dk, z * lg], axis=2)

    parents, fns = [pre], [grad_pre]
    if h0 is not None:
        parents.append(h0)
        fns.append(lambda g: a[:, 0] * lam(g)[:, 0])
    return Tensor._make(h, parents, fns, "mingru_scan")


# --------------------------------------------------------------------------- #
# layers
# --------------------------------------------------------------------------- #
class MinGRU(Module):
    """minGRU (Feng et al., 2024): a GRU whose gates see only the input.

    ::

        z_t  = sigmoid(x_t @ W_z + b_z)
        h~_t = x_t @ W_h + b_h
        h_t  = (1 - z_t) * h_{t-1} + z_t * h~_t

    **Polarity is the opposite of** :class:`scratchgrad.nn.GRUCell`: here ``z`` gates
    the *new* candidate, there it gates the old state. A checkpoint moved between the
    two needs ``z -> 1 - z``, i.e. negated ``W_z`` and ``b_z``.

    ``weight`` is ``(input_size, 2 * hidden_size)`` with gate order ``z, h~``, so the
    whole sequence's gates are one ``x @ W + b``. With no ``h`` inside the gates a
    single layer is a *linear* recurrence in ``h``; expressiveness comes from stacking
    it with nonlinear layers in between, as the paper does. This is the linear-space
    form, so ``h~`` needs no positivity-enforcing ``g()``.

    ``1 - z`` is computed as ``sigmoid(-k)`` rather than ``1 - sigmoid(k)``: in float32
    ``sigmoid(17)`` rounds to exactly 1, and the subtraction would then erase a
    retention of ``4e-8`` to a hard zero. The gating and the scan are one fused node,
    :func:`_mingru_scan`.

    Args:
        input_size: Feature width of ``x``.
        hidden_size: Width of ``h``.
        bias: Learn ``b_z`` and ``b_h``.
        rng: Explicit generator for the Xavier initialisation.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, rng=None) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.weight = Parameter(np.zeros((self.input_size, 2 * self.hidden_size)))
        init.xavier_uniform_(self.weight, rng=init.default_rng(rng))
        if bias:
            self.bias = Parameter(np.zeros(2 * self.hidden_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Run over a batch-first ``(N, T, input_size)`` input.

        Returns:
            ``(output, h_n)``: all states ``(N, T, hidden_size)`` and the last one,
            ``(N, hidden_size)``, to carry into the next chunk.
        """
        if x.ndim != 3 or x.data.shape[2] != self.input_size:
            raise ValueError(f"MinGRU expects (N, T, {self.input_size}), got {x.shape}")
        hs = self.hidden_size
        if h0 is not None and tuple(h0.shape) != (x.data.shape[0], hs):
            raise ValueError(f"h0 must be ({x.data.shape[0]}, {hs}), got {tuple(h0.shape)}")
        n, steps, _ = x.shape
        # One 2-D GEMM for every timestep at once. Flattened on purpose: NumPy's
        # stacked (N, T, I) @ (I, 2H) path is ~100x slower here than the same
        # product as (N*T, I) @ (I, 2H).
        pre = F.linear(x.reshape(n * steps, self.input_size), self.weight, self.bias)
        out = _mingru_scan(pre.reshape(n, steps, 2 * hs), h0)
        return out, out[:, -1]

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}, bias={self.bias is not None}"


class SelectiveSSM(Module):
    """A minimal Mamba-style selective state-space layer, ``(N, T, D) -> (N, T, D)``.

    ::

        delta = softplus(x @ W_dt + b_dt)          (N, T, D)  positive step size
        B, C  = x @ W_B, x @ W_C                   (N, T, S)  input-dependent
        A     = -exp(A_log)                        (D, S)     negative by construction
        h_t   = exp(delta_t A) h_{t-1} + delta_t B_t x_t
        y_t   = sum_s C_t[s] h_t[:, s] + D * x_t

    Why each piece is shaped the way it is:

    * ``A = -exp(A_log)`` makes negativity a property of the parameterisation, not a
      constraint an optimiser step can violate. Init is S4D-real, ``A[d] = -(1..S)``.
    * **Zero-order hold** for ``A``: ``exp(delta * A)`` lies in ``(0, 1)`` for every
      ``delta > 0``, so the recurrence is stable at any step size. The Euler form
      ``1 + delta * A`` leaves ``[0, 1]`` once ``delta * |A| > 1`` and diverges past 2.
    * ``B_bar ~ delta * B`` is the Euler input term Mamba uses too; exact ZOH is
      ``(exp(delta A) - 1) / A * B``. They agree to first order in ``delta * |A|``.
    * ``b_dt`` is the inverse softplus of a log-uniform draw in ``[dt_min, dt_max]``,
      so the initial step sizes span two decades of memory length. Leaving it at 0
      gives ``delta = log 2`` everywhere -- one timescale for every channel.

    Omitted from real Mamba, deliberately: the gated branch, the short causal
    convolution and the low-rank ``delta`` projection. The stack around the layer
    supplies the nonlinearity.

    Args:
        d_model: Channel width ``D``.
        d_state: State size ``S`` per channel.
        dt_min: Smallest initial step size.
        dt_max: Largest initial step size.
        rng: Explicit generator.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        rng=None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        rng = init.default_rng(rng)
        self.dt_proj = Linear(d_model, d_model, nonlinearity="linear", rng=rng)
        self.B_proj = Linear(d_model, d_state, bias=False, nonlinearity="linear", rng=rng)
        self.C_proj = Linear(d_model, d_state, bias=False, nonlinearity="linear", rng=rng)
        dt = np.exp(rng.uniform(math.log(dt_min), math.log(dt_max), self.d_model))
        # softplus^-1(y) = log(e^y - 1) = y + log(1 - e^-y); the second form does not
        # lose everything to cancellation at y = 1e-3.
        self.dt_proj.bias.data[...] = dt + np.log(-np.expm1(-dt))
        a_init = np.tile(np.arange(1, self.d_state + 1, dtype=np.float64), (self.d_model, 1))
        self.A_log = Parameter(np.log(a_init))
        self.D = Parameter(np.ones(self.d_model))

    def forward(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Returns ``(y, h_n)``: ``(N, T, D)`` outputs and the ``(N, D, S)`` last state."""
        if x.ndim != 3 or x.data.shape[2] != self.d_model:
            raise ValueError(f"SelectiveSSM expects (N, T, {self.d_model}), got {x.shape}")
        n, steps, d = x.shape
        rows = x.reshape(n * steps, d)  # 2-D GEMMs: see MinGRU.forward
        delta = F.softplus(self.dt_proj(rows)).reshape(n, steps, d)
        B = self.B_proj(rows).reshape(n, steps, self.d_state)
        C = self.C_proj(rows).reshape(n, steps, self.d_state)
        h = selective_scan(x, delta, -self.A_log.exp(), B, h0)
        y = _ssm_readout(h, C) + x * self.D
        return y, h[:, -1]

    def extra_repr(self) -> str:
        return f"{self.d_model}, d_state={self.d_state}"
