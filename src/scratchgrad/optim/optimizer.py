"""First-order optimizers, hand-written against ``Parameter.data`` and ``.grad``.

Every update rule here is the *reference* formulation of its paper as PyTorch
implements it, because a checkpoint written by one and read by the other has to
train the same. The places where the popular from-scratch version differs are
called out inline; three of them are silent, in the sense that the wrong version
still descends and only reaches a worse place:

* **Adam's ``eps`` belongs outside the sqrt and after bias correction.**
  ``g / sqrt(v + eps)`` puts a floor of ``sqrt(eps)`` under the denominator, so
  with the usual ``eps=1e-8`` the floor engages at ``|g| ~ 1e-4`` -- four orders
  of magnitude earlier than ``m̂ / (sqrt(v̂) + eps)`` does.
* **Bias correction is two corrections, and dropping both is nearly harmless.**
  See :class:`Adam` for the arithmetic; the widely-repeated "31.6x" figure is the
  cost of correcting ``m`` and forgetting ``v``, not of skipping correction.
* **AdamW's weight decay is decoupled.** Adding ``wd * p`` to the gradient makes
  it L2 regularisation, which Adam's second moment then divides back out for
  exactly the parameters that need decaying most. That is :class:`Adam`'s
  ``weight_decay``; it is a different algorithm from :class:`AdamW`'s.

State is keyed by ``id(param)``. Keying by the parameter object would work only
because :class:`~scratchgrad.tensor.Tensor` hashes by identity -- but its
``__eq__`` returns an elementwise bool array, so any container that falls back
to equality (``p in params``, a dict on hash collision) misbehaves. ``id()``
takes the question off the table.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from ..tensor import Tensor

__all__ = [
    "SGD",
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "Optimizer",
    "RMSProp",
    "clip_grad_norm_",
    "clip_grad_value_",
]

ParamsT = Iterable[Tensor] | Iterable[dict[str, Any]]


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class Optimizer:
    """Base class holding parameter groups, per-parameter state, and the step loop.

    Args:
        params: An iterable of :class:`~scratchgrad.nn.module.Parameter`, or an
            iterable of *param-group dicts* ``{"params": [...], "lr": ...}``.
            A group dict may override any key in ``defaults``; keys it omits are
            filled from ``defaults``.
        defaults: Hyperparameters used for any group that does not set them.

    Subclasses implement :meth:`_init_state` and :meth:`_update`.
    """

    def __init__(self, params: ParamsT, defaults: dict[str, Any]) -> None:
        self.defaults: dict[str, Any] = dict(defaults)
        # id(param) -> state. Not param -> state: see the module docstring.
        self.state: dict[int, dict[str, Any]] = {}
        self.param_groups: list[dict[str, Any]] = []

        if isinstance(params, Tensor):
            raise TypeError("params must be an iterable of Parameters, not a single Tensor")
        items = list(params)
        if not items:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(items[0], dict):
            items = [{"params": items}]
        for group in items:
            self.add_param_group(group)

    # -- groups --------------------------------------------------------------
    def add_param_group(self, group: dict[str, Any]) -> None:
        """Register one parameter group, filling missing keys from ``defaults``."""
        if not isinstance(group, dict):
            raise TypeError(f"param group must be a dict, got {type(group).__name__}")
        group = dict(group)
        if "params" not in group:
            raise ValueError("param group is missing the required 'params' key")

        params = group["params"]
        params = [params] if isinstance(params, Tensor) else list(params)
        for p in params:
            if not isinstance(p, Tensor):
                raise TypeError(
                    f"optimizer can only optimize Tensors/Parameters, got {type(p).__name__}"
                )
        group["params"] = params

        # A parameter shared between two groups would be stepped twice per
        # step() with two different learning rates, and the second update would
        # be computed from a stale gradient. Refuse it loudly instead.
        seen: dict[int, int] = {}
        for gi, g in enumerate(self.param_groups):
            for p in g["params"]:
                seen[id(p)] = gi
        for i, p in enumerate(params):
            if id(p) in seen:
                where = f"group {seen[id(p)]}" if seen[id(p)] >= 0 else "this same group"
                raise ValueError(
                    f"parameter at index {i} (shape {p.shape}) already appears in {where}; "
                    "a parameter may belong to exactly one param group"
                )
            seen[id(p)] = -1  # marks 'this group', so intra-group duplicates are caught too

        for key, value in self.defaults.items():
            group.setdefault(key, value)
        self.param_groups.append(group)

    def parameters(self) -> Iterator[Tensor]:
        """Every parameter, in group order -- the order ``state_dict`` indexes by."""
        for group in self.param_groups:
            yield from group["params"]

    # -- gradients -----------------------------------------------------------
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients.

        ``set_to_none=True`` (the default) drops the buffers entirely. That is
        not merely a memory optimisation: a parameter whose gradient is ``None``
        is *skipped* by :meth:`step`, so a frozen branch of the model keeps its
        momentum buffer untouched, while a zero gradient would still let
        momentum and weight decay move it.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad[...] = 0.0

    # -- the step ------------------------------------------------------------
    def step(self) -> None:
        """Apply one update to every parameter that has a gradient.

        Raises:
            ValueError: If a gradient's shape differs from its parameter's.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = np.asarray(p.grad)
                if grad.shape != p.data.shape:
                    # A grad that merely *broadcasts* against the parameter -- a
                    # scalar, or a row against a matrix -- would be applied to
                    # every element and the update would look plausible. PyTorch
                    # blocks this in the `.grad` setter; we have no setter, so
                    # the step is the last place to catch a hand-set gradient.
                    raise ValueError(
                        f"gradient shape {grad.shape} does not match parameter shape {p.data.shape}"
                    )
                state = self.state.get(id(p))
                if state is None:
                    state = self.state[id(p)] = {"step": 0}
                    self._init_state(state, p, group)
                state["step"] += 1
                self._update(p, grad, state, group)

    def _init_state(self, state: dict[str, Any], p: Tensor, group: dict[str, Any]) -> None:
        """Allocate per-parameter buffers. ``state["step"]`` already exists."""

    def _update(
        self, p: Tensor, grad: np.ndarray, state: dict[str, Any], group: dict[str, Any]
    ) -> None:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__} must implement _update()")

    # -- serialization -------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """A picklable snapshot: state keyed by **position**, not by ``id``.

        ``id()`` is only meaningful within one process run, so the saved mapping
        uses each parameter's index in ``self.parameters()`` order. Arrays are
        copied -- a snapshot that aliased the live momentum buffers would be
        silently overwritten by the next :meth:`step`.
        """
        index_of: dict[int, int] = {}
        packed_groups: list[dict[str, Any]] = []
        idx = 0
        for group in self.param_groups:
            packed = {k: v for k, v in group.items() if k != "params"}
            indices = []
            for p in group["params"]:
                index_of[id(p)] = idx
                indices.append(idx)
                idx += 1
            packed["params"] = indices
            packed_groups.append(packed)

        packed_state = {
            index_of[pid]: {
                k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                for k, v in st.items()
            }
            for pid, st in self.state.items()
            if pid in index_of
        }
        return {"state": packed_state, "param_groups": packed_groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state saved by :meth:`state_dict` onto the current parameters."""
        groups = state_dict["param_groups"]
        if len(groups) != len(self.param_groups):
            raise ValueError(
                f"loaded state_dict has {len(groups)} param groups, "
                f"this optimizer has {len(self.param_groups)}"
            )
        for saved, group in zip(groups, self.param_groups, strict=True):
            if len(saved["params"]) != len(group["params"]):
                raise ValueError(
                    "loaded state_dict contains a param group with "
                    f"{len(saved['params'])} params, but this group has {len(group['params'])}"
                )
            for k, v in saved.items():
                if k != "params":
                    group[k] = v

        params = list(self.parameters())
        self.state = {}
        for index, st in state_dict["state"].items():
            index = int(index)
            if index >= len(params):
                raise ValueError(f"state_dict references param index {index}, out of range")
            self.state[id(params[index])] = {
                k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                for k, v in st.items()
            }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        lines = [f"{type(self).__name__}("]
        for i, group in enumerate(self.param_groups):
            opts = ", ".join(f"{k}={v}" for k, v in sorted(group.items()) if k != "params")
            lines.append(f"  group {i}: {len(group['params'])} params, {opts}")
        return "\n".join(lines) + "\n)"


def _as_param_dtype(p: Tensor, arr: np.ndarray) -> np.ndarray:
    return arr.astype(p.data.dtype, copy=False)


def _validate_hyperparams(
    lr: float,
    eps: float | None = None,
    *,
    nonneg: dict[str, float] | None = None,
    unit_interval: dict[str, float] | None = None,
) -> None:
    """Validate hyper-parameters for every optimizer in this module.

    Six hand-rolled copies of these checks is how five of the six end up
    accepting a negative ``weight_decay`` -- which does not regularise but
    *grows* the weights it was asked to shrink, and does so slowly enough to
    read as a bad-but-training run.

    Args:
        lr: Learning rate; must be ``>= 0``.
        eps: Denominator epsilon; must be ``> 0`` when the optimizer has one.
        nonneg: Extra ``name -> value`` pairs required to be ``>= 0``
            (``weight_decay``, ``momentum``, ``lr_decay``).
        unit_interval: ``name -> value`` pairs required to lie in ``[0, 1)``
            (``betas``, ``alpha``, ``rho``). Exactly 1.0 is excluded because a
            running average that never forgets stops tracking the gradient.

    Raises:
        ValueError: Naming the offending hyper-parameter and its value.
    """
    if lr < 0.0:
        raise ValueError(f"invalid lr: {lr}")
    if eps is not None and eps <= 0.0:
        raise ValueError(f"invalid eps: {eps}")
    for name, value in (nonneg or {}).items():
        if value < 0.0:
            raise ValueError(f"invalid {name}: {value}")
    for name, value in (unit_interval or {}).items():
        # Written as `not 0 <= v < 1` rather than `v < 0 or v >= 1` so a NaN,
        # which compares false against everything, is rejected too.
        if not 0.0 <= value < 1.0:
            raise ValueError(f"invalid {name}: {value}")


def _buffer(state: dict[str, Any], key: str, p: Tensor) -> np.ndarray:
    """Return ``state[key]``, allocating a zero buffer on first use.

    ``_init_state`` runs exactly once per parameter, on the step where it first
    has a gradient, so a buffer allocated there *conditionally* on a
    hyper-parameter freezes that hyper-parameter's value at that instant.
    Turning ``momentum``/``centered``/``amsgrad`` on later -- a fine-tuning
    phase, a hand-edited param group -- would then read a key that was never
    created. Allocating on first use instead costs nothing when the option
    stays off, which is the reason not to simply allocate everything eagerly.
    """
    buf = state.get(key)
    if buf is None:
        buf = np.zeros_like(p.data)
        state[key] = buf
    return buf


# --------------------------------------------------------------------------- #
# SGD
# --------------------------------------------------------------------------- #
class SGD(Optimizer):
    """Stochastic gradient descent with momentum, dampening and Nesterov.

    Args:
        params: Parameters or param groups.
        lr: Learning rate.
        momentum: Momentum coefficient ``mu``; 0 disables the buffer entirely.
        dampening: Damps the incoming gradient, ``buf = mu*buf + (1-tau)*g``.
        weight_decay: Coupled L2, added to the gradient before momentum.
        nesterov: Use the look-ahead form.

    The momentum recursion is PyTorch's, which is *not* the textbook
    ``v = mu*v - lr*g; p += v``. Here the learning rate multiplies the buffer at
    use time instead of entering it, so changing ``lr`` mid-run (every scheduler
    does) takes effect immediately rather than decaying in over ``1/(1-mu)``
    steps. The two agree only for a constant ``lr``.

    Nesterov is likewise the PyTorch formulation, ``buf = mu*buf + g`` then
    ``d = g + mu*buf``: the update peeks one momentum step ahead instead of using
    the buffer directly. Writing ``d = mu*buf + g`` (i.e. just ``buf``) is plain
    momentum with an extra line of code.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        _validate_hyperparams(lr, nonneg={"momentum": momentum, "weight_decay": weight_decay})
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("nesterov momentum requires momentum > 0 and dampening == 0")
        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "dampening": dampening,
                "weight_decay": weight_decay,
                "nesterov": nesterov,
            },
        )

    def _update(self, p, grad, state, group):
        wd = group["weight_decay"]
        if wd != 0.0:
            # New array, never `grad +=`: p.grad is the caller's buffer and a
            # second step() before zero_grad() would compound the decay.
            grad = grad + wd * p.data
        mu = group["momentum"]
        if mu != 0.0:
            buf = state.get("momentum_buffer")
            if buf is None:
                # Allocated and seeded on first USE, not at step 1: momentum may
                # be raised from 0 mid-run, and a buffer keyed off the step
                # counter would then be missing. The seed is the raw gradient
                # (no dampening), so a momentum run and a plain run take the
                # same first step. PyTorch seeds on first use too, and it
                # matters for reproducing runs.
                buf = np.zeros_like(p.data)
                buf[...] = _as_param_dtype(p, grad)
                state["momentum_buffer"] = buf
            else:
                buf *= mu
                buf += _as_param_dtype(p, (1.0 - group["dampening"]) * grad)
            grad = grad + mu * buf if group["nesterov"] else buf
        p.data -= _as_param_dtype(p, group["lr"] * grad)


# --------------------------------------------------------------------------- #
# Adagrad
# --------------------------------------------------------------------------- #
class Adagrad(Optimizer):
    """Adagrad: per-parameter lr scaled by the inverse RMS of *all* past gradients.

    Args:
        params: Parameters or param groups.
        lr: Base learning rate.
        lr_decay: Decays the effective lr as ``lr / (1 + (t-1)*lr_decay)``.
        weight_decay: Coupled L2.
        initial_accumulator_value: Starting value of the running sum of squares.
        eps: Added to the *root*, not under it (see :class:`Adam`).

    The accumulator only ever grows, so the effective learning rate decays
    monotonically to zero. That is the algorithm, not a bug -- it is also why
    :class:`RMSProp` exists.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
    ) -> None:
        _validate_hyperparams(lr, eps, nonneg={"weight_decay": weight_decay, "lr_decay": lr_decay})
        super().__init__(
            params,
            {
                "lr": lr,
                "lr_decay": lr_decay,
                "weight_decay": weight_decay,
                "initial_accumulator_value": initial_accumulator_value,
                "eps": eps,
            },
        )

    def _init_state(self, state, p, group):
        state["sum"] = np.full_like(p.data, group["initial_accumulator_value"])

    def _update(self, p, grad, state, group):
        if group["weight_decay"] != 0.0:
            grad = grad + group["weight_decay"] * p.data
        acc = state["sum"]
        acc += _as_param_dtype(p, grad * grad)
        clr = group["lr"] / (1.0 + (state["step"] - 1) * group["lr_decay"])
        p.data -= _as_param_dtype(p, clr * grad / (np.sqrt(acc) + group["eps"]))


# --------------------------------------------------------------------------- #
# RMSProp
# --------------------------------------------------------------------------- #
class RMSProp(Optimizer):
    """RMSProp: Adagrad with an exponentially-weighted accumulator.

    Args:
        params: Parameters or param groups.
        lr: Learning rate.
        alpha: Smoothing constant of the squared-gradient average.
        eps: Added to the root of the denominator.
        weight_decay: Coupled L2.
        momentum: Optional heavy-ball momentum on the *normalised* gradient.
        centered: Subtract the squared running mean, giving an estimate of the
            gradient variance rather than of its second moment.

    Note what ``centered`` divides by: ``sqrt(E[g^2] - E[g]^2)``. On a
    consistently-signed gradient that quantity is near zero, so centring
    *increases* the step there -- the opposite of what "centred = safer"
    suggests. ``eps`` is what keeps it finite.

    Unlike :class:`Adam` there is no bias correction, so the first few steps run
    with an accumulator biased toward its zero initialisation and are therefore
    larger than the steady-state step. Hinton's lecture-slide algorithm has the
    same property; it is not an omission here.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ) -> None:
        _validate_hyperparams(
            lr,
            eps,
            nonneg={"weight_decay": weight_decay, "momentum": momentum},
            unit_interval={"alpha": alpha},
        )
        super().__init__(
            params,
            {
                "lr": lr,
                "alpha": alpha,
                "eps": eps,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "centered": centered,
            },
        )

    def _init_state(self, state, p, group):
        state["square_avg"] = np.zeros_like(p.data)

    def _update(self, p, grad, state, group):
        if group["weight_decay"] != 0.0:
            grad = grad + group["weight_decay"] * p.data
        alpha = group["alpha"]
        sq = state["square_avg"]
        sq *= alpha
        sq += _as_param_dtype(p, (1.0 - alpha) * grad * grad)

        if group["centered"]:
            ga = _buffer(state, "grad_avg", p)
            ga *= alpha
            ga += _as_param_dtype(p, (1.0 - alpha) * grad)
            # Clamp at 0: the variance estimate is a difference of two running
            # averages and can go slightly negative from rounding, which would
            # make sqrt() emit a RuntimeWarning and produce NaN.
            var = np.maximum(sq - ga * ga, 0.0)
            avg = np.sqrt(var) + group["eps"]
        else:
            avg = np.sqrt(sq) + group["eps"]

        if group["momentum"] > 0.0:
            buf = _buffer(state, "momentum_buffer", p)
            buf *= group["momentum"]
            buf += _as_param_dtype(p, grad / avg)
            p.data -= _as_param_dtype(p, group["lr"] * buf)
        else:
            p.data -= _as_param_dtype(p, group["lr"] * grad / avg)


# --------------------------------------------------------------------------- #
# Adam / AdamW
# --------------------------------------------------------------------------- #
class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015), with coupled L2 ``weight_decay``.

    Args:
        params: Parameters or param groups.
        lr: Step size. After bias correction the first step has magnitude
            almost exactly ``lr`` for *any* gradient scale -- that invariant is
            the signature of a correct implementation.
        betas: ``(beta1, beta2)`` for the first and second moment.
        eps: Added to ``sqrt(v_hat)``.
        weight_decay: **Coupled** L2 -- added to the gradient. Use
            :class:`AdamW` for the decoupled form.
        amsgrad: Keep the running maximum of ``v`` so the denominator is
            non-decreasing.

    Two things are easy to get wrong here, and both still descend.

    **Where ``eps`` goes.** It is ``m̂ / (sqrt(v̂) + eps)``: outside the sqrt and
    after bias correction. ``g / sqrt(v + eps)`` floors the denominator at
    ``sqrt(eps) = 1e-4`` for the default ``eps=1e-8``, so any gradient smaller
    than about ``1e-4`` gets a step proportional to ``g`` instead of the
    sign-like step Adam is supposed to take -- four orders of magnitude too
    early, and exactly in the small-gradient regime where Adam earns its keep.

    **Bias correction, and the number everyone misquotes.** At ``t=1``,
    ``m = (1-b1) g`` and ``v = (1-b2) g^2``. Dropping correction *entirely*
    gives a step of ``(1-b1)|g| / sqrt((1-b2) g^2) = 0.1/0.0316 = 3.16``, i.e.
    **3.16x too large, not 31.6x**: the numerator is 10x too small at the same
    time and the two errors mostly cancel. The genuinely explosive case is
    correcting ``m`` but forgetting ``v``: ``1/sqrt(1-b2) = 31.6x``. The "31.6x"
    figure is routinely attached to the wrong mistake.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        b1, b2 = betas
        _validate_hyperparams(
            lr,
            eps,
            nonneg={"weight_decay": weight_decay},
            unit_interval={"beta1": b1, "beta2": b2},
        )
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": (float(b1), float(b2)),
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": amsgrad,
            },
        )

    #: AdamW overrides this to decouple the decay instead of adding it to ``g``.
    _decoupled_decay = False

    def _init_state(self, state, p, group):
        state["exp_avg"] = np.zeros_like(p.data)
        state["exp_avg_sq"] = np.zeros_like(p.data)

    def _update(self, p, grad, state, group):
        lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
        b1, b2 = group["betas"]
        t = state["step"]

        if wd != 0.0:
            if self._decoupled_decay:
                # AdamW: decay the parameter directly, so the amount of decay is
                # independent of the gradient history. Folding it into `grad`
                # instead would send it through the second moment, which divides
                # most of it back out precisely for the large-|g| weights.
                p.data -= _as_param_dtype(p, lr * wd * p.data)
            else:
                grad = grad + wd * p.data

        m, v = state["exp_avg"], state["exp_avg_sq"]
        m *= b1
        m += _as_param_dtype(p, (1.0 - b1) * grad)
        v *= b2
        v += _as_param_dtype(p, (1.0 - b2) * grad * grad)

        bc1 = 1.0 - b1**t
        bc2 = 1.0 - b2**t
        if group["amsgrad"]:
            v_max = _buffer(state, "max_exp_avg_sq", p)
            np.maximum(v_max, v, out=v_max)
            v_used = v_max
        else:
            v_used = v

        m_hat = m / bc1
        v_hat = v_used / bc2
        p.data -= _as_param_dtype(p, lr * m_hat / (np.sqrt(v_hat) + eps))


class AdamW(Adam):
    """Adam with **decoupled** weight decay (Loshchilov & Hutter, 2019).

    Args:
        params: Parameters or param groups.
        lr: Step size.
        betas: ``(beta1, beta2)``.
        eps: Added to ``sqrt(v_hat)``.
        weight_decay: Applied as ``p -= lr * weight_decay * p``, straight to the
            parameter and never through the moment estimates.
        amsgrad: As in :class:`Adam`.

    The default ``weight_decay`` is 1e-2, a hundred times :class:`Adam`'s 0 and
    two orders above the 1e-4 typical of SGD, because decoupling removes the
    ``1/sqrt(v̂)`` factor that was quietly shrinking it.

    ``AdamW(wd=w)`` and ``Adam(weight_decay=w)`` are not the same optimizer and
    do not become the same at any learning rate; only ``w = 0`` makes them agree.
    """

    _decoupled_decay = True

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )


# --------------------------------------------------------------------------- #
# Adadelta
# --------------------------------------------------------------------------- #
class Adadelta(Optimizer):
    """Adadelta (Zeiler, 2012): a unit-correct step with no learning rate.

    Args:
        params: Parameters or param groups.
        lr: A scale on the whole update. The paper has no such term; it exists
            because in practice the unit-matched step is often too small.
        rho: Decay of both running averages.
        eps: Sits **inside** both square roots (see below).
        weight_decay: Coupled L2.

    The update is ``dx = sqrt(E[dx^2] + eps) / sqrt(E[g^2] + eps) * g``, and the
    accumulator of squared *updates* is what makes the step dimensionally
    consistent with the parameter -- the paper's whole point.

    Here ``eps`` really is inside the roots, unlike :class:`Adam`. It is not a
    guard against division by zero but the numerator's seed value: ``E[dx^2]``
    starts at exactly zero, so with ``eps`` outside, the first step would be
    ``0`` and the optimizer would never move at all.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        _validate_hyperparams(
            lr, eps, nonneg={"weight_decay": weight_decay}, unit_interval={"rho": rho}
        )
        super().__init__(params, {"lr": lr, "rho": rho, "eps": eps, "weight_decay": weight_decay})

    def _init_state(self, state, p, group):
        state["square_avg"] = np.zeros_like(p.data)
        state["acc_delta"] = np.zeros_like(p.data)

    def _update(self, p, grad, state, group):
        if group["weight_decay"] != 0.0:
            grad = grad + group["weight_decay"] * p.data
        rho, eps = group["rho"], group["eps"]
        sq, acc = state["square_avg"], state["acc_delta"]
        sq *= rho
        sq += _as_param_dtype(p, (1.0 - rho) * grad * grad)
        delta = np.sqrt(acc + eps) / np.sqrt(sq + eps) * grad
        acc *= rho
        acc += _as_param_dtype(p, (1.0 - rho) * delta * delta)
        p.data -= _as_param_dtype(p, group["lr"] * delta)


# --------------------------------------------------------------------------- #
# gradient clipping
# --------------------------------------------------------------------------- #
def _grad_list(params) -> list[np.ndarray]:
    if isinstance(params, Tensor):
        params = [params]
    return [np.asarray(p.grad) for p in params if p.grad is not None]


def clip_grad_norm_(
    params: Iterable[Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    eps: float = 1e-6,
) -> float:
    """Rescale gradients in place so their **global** norm is at most ``max_norm``.

    Args:
        params: Parameters whose ``.grad`` should be clipped in place.
        max_norm: Ceiling on the total norm.
        norm_type: ``2.0`` for Euclidean, ``float("inf")`` for max-abs, or any
            positive ``p``.
        eps: Guards the division when the total norm is zero.

    Returns:
        The total norm **before** clipping -- the number worth logging, since it
        is what tells you whether clipping fired at all. Returning the post-clip
        norm gives a flat line at ``max_norm`` and hides the exploding step it
        was meant to reveal.

    All gradients are treated as one concatenated vector, and one common factor
    is applied to every tensor. Clipping each tensor to ``max_norm``
    individually is a different operation: it changes the *direction* of the
    update, whereas the global form only changes its length.

    Raises:
        ValueError: If ``max_norm`` or ``norm_type`` is not positive.
    """
    # Both written as `not x > 0` so NaN, which compares false against
    # everything, is rejected as well. Unvalidated, norm_type=0 reaches
    # `acc ** (1/norm_type)` and dies with a bare ZeroDivisionError that names
    # nothing the caller passed; norm_type < 0 silently *inverts* the norm.
    # max_norm == 0 is *legal* and means "scale every gradient to zero" -- PyTorch
    # supports it and it is a real, if blunt, thing to ask for. Only negatives and
    # NaN are rejected here; narrowing this to `> 0` would remove a capability
    # rather than prevent a mistake.
    if not max_norm >= 0.0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}")
    if not norm_type > 0.0:
        raise ValueError(f"norm_type must be positive (or inf), got {norm_type}")
    grads = _grad_list(params)
    if not grads:
        return 0.0
    if norm_type == float("inf"):
        total_norm = max(float(np.max(np.abs(g))) for g in grads)
    else:
        # Sum of p-th powers over every element, then one root: mathematically
        # identical to concatenating first, and it needs no temporary copy.
        acc = sum(float(np.sum(np.abs(g.astype(np.float64)) ** norm_type)) for g in grads)
        total_norm = float(acc ** (1.0 / norm_type))

    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for g in grads:
            g *= np.asarray(clip_coef, dtype=g.dtype)
    return total_norm


def clip_grad_value_(params: Iterable[Tensor], clip_value: float) -> None:
    """Clamp every gradient element in place to ``[-clip_value, clip_value]``.

    This is the elementwise cousin of :func:`clip_grad_norm_` and it *does*
    change the update direction. It is the right tool when a handful of entries
    blow up (an embedding row for a rare token, say) rather than the whole
    gradient growing together.
    """
    v = abs(float(clip_value))
    for g in _grad_list(params):
        np.clip(g, -v, v, out=g)
