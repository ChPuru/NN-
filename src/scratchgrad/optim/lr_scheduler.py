"""Learning-rate schedules, expressed as closed-form multipliers on a base lr.

Every scheduler here computes ``lr(t) = base_lr * factor(t)`` where ``base_lr``
is recorded per param group **at construction time** and never touched again.
The alternative -- reading ``group["lr"]`` and writing back a decayed version --
is the standard from-scratch implementation and it is wrong in three ways at
once:

1. It is not idempotent. ``step()`` called twice for the same epoch (a resumed
   run, a scheduler stepped in both the epoch loop and the batch loop) decays
   twice, and nothing reports it.
2. It destroys per-group learning rates. Any group-specific lr survives only
   until the first update, after which every group is on whatever multiplicative
   path the last write left behind.
3. It cannot be queried. ``get_lr(epoch)`` for an arbitrary epoch is impossible
   without replaying the whole history, which makes plotting a schedule before
   training -- the cheapest possible sanity check -- awkward enough that nobody
   does it.

Because the factor is closed-form, composing schedules is just multiplying or
selecting factors: that is all :class:`ChainedScheduler` and
:class:`SequentialLR` do, and neither needs its children to hold any state.

Convention (PyTorch's): the constructor performs the epoch-0 update, so the
learning rate is correct before the first :meth:`~LRScheduler.step`. The loop is
``forward / backward / optimizer.step() / scheduler.step()``, which means
``scheduler.step()`` at the end of iteration ``t`` sets the lr used by iteration
``t + 1``.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from typing import Any

from .optimizer import Optimizer

__all__ = [
    "ChainedScheduler",
    "ConstantLR",
    "CosineAnnealingLR",
    "ExponentialLR",
    "LRScheduler",
    "LinearWarmup",
    "MultiStepLR",
    "SequentialLR",
    "StepLR",
    "WarmupCosine",
]


class LRScheduler:
    """Base class: owns ``base_lrs``, the epoch counter, and writing lrs back.

    Args:
        optimizer: The optimizer whose ``param_groups`` this schedules.
        last_epoch: Epoch index to resume from. ``-1`` (the default) means a
            fresh run; the constructor then advances to epoch 0 and applies it.

    Subclasses implement :meth:`factor` (one multiplier for every group) or
    :meth:`factors` (one per group).
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"expected an Optimizer, got {type(optimizer).__name__}")
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            # setdefault, not assignment: a second scheduler attached to the
            # same optimizer must anchor to the ORIGINAL lr, not to whatever the
            # first one has already decayed it to.
            group.setdefault("initial_lr", group["lr"])
        self.base_lrs: list[float] = [float(g["initial_lr"]) for g in optimizer.param_groups]
        self.last_epoch = int(last_epoch)
        self._last_lr: list[float] = [float(g["lr"]) for g in optimizer.param_groups]
        self.step()

    # -- the schedule --------------------------------------------------------
    def factor(self, epoch: int) -> float:  # pragma: no cover - abstract
        """The multiplier on ``base_lr`` at ``epoch``, shared by all groups."""
        raise NotImplementedError(f"{type(self).__name__} must implement factor() or factors()")

    def factors(self, epoch: int) -> list[float]:
        """One multiplier per param group. Defaults to :meth:`factor` for all."""
        f = self.factor(epoch)
        return [f] * len(self.base_lrs)

    def get_lr(self, epoch: int | None = None) -> list[float]:
        """The learning rates this schedule prescribes at ``epoch``.

        Pure: it reads nothing but ``base_lrs`` and the epoch, so it is safe to
        call for any epoch, in any order, to plot the schedule before training.
        """
        e = self.last_epoch if epoch is None else int(epoch)
        # strict=True on purpose: base_lrs is frozen at construction, so a param
        # group added to the optimizer afterwards would otherwise be silently
        # left unscheduled forever. Better a loud error than a group that
        # never decays.
        return [b * f for b, f in zip(self.base_lrs, self.factors(e), strict=True)]

    def get_last_lr(self) -> list[float]:
        """The learning rates set by the most recent :meth:`step`."""
        return list(self._last_lr)

    def step(self, epoch: int | None = None) -> None:
        """Advance one epoch (or jump to ``epoch``) and write the new lrs."""
        self.last_epoch = self.last_epoch + 1 if epoch is None else int(epoch)
        for group, lr in zip(self.optimizer.param_groups, self.get_lr(), strict=True):
            group["lr"] = lr
        self._last_lr = [float(g["lr"]) for g in self.optimizer.param_groups]

    # -- serialization -------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Everything but the optimizer, which the caller restores separately."""
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.__dict__.update({k: v for k, v in state_dict.items() if k != "optimizer"})

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(base_lrs={self.base_lrs}, last_epoch={self.last_epoch})"


class ConstantLR(LRScheduler):
    """Holds every group at its base lr. Useful as a no-op arm of a schedule."""

    def factor(self, epoch: int) -> float:
        return 1.0


class StepLR(LRScheduler):
    """Multiply by ``gamma`` every ``step_size`` epochs.

    Args:
        optimizer: Optimizer to schedule.
        step_size: Epochs between drops.
        gamma: Multiplicative drop factor.
        last_epoch: Resume point.

    ``lr(t) = base * gamma ** (t // step_size)``, so the first drop lands at
    ``t == step_size`` and the lr is constant on ``[0, step_size)``.
    """

    def __init__(
        self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1
    ) -> None:
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def factor(self, epoch: int) -> float:
        return self.gamma ** (max(epoch, 0) // self.step_size)


class MultiStepLR(LRScheduler):
    """Multiply by ``gamma`` once at each epoch in ``milestones``.

    Args:
        optimizer: Optimizer to schedule.
        milestones: Epochs at which to drop. Sorted on entry; duplicates are
            honoured (a repeated milestone drops twice), matching PyTorch's
            ``Counter``-based behaviour.
        gamma: Multiplicative drop factor.
        last_epoch: Resume point.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: Sequence[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.milestones = sorted(int(m) for m in milestones)
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def factor(self, epoch: int) -> float:
        # bisect_right, not bisect_left: a milestone at epoch m takes effect AT
        # epoch m. bisect_left delays every drop by one epoch.
        return self.gamma ** bisect.bisect_right(self.milestones, epoch)


class ExponentialLR(LRScheduler):
    """``lr(t) = base * gamma ** t``.

    Args:
        optimizer: Optimizer to schedule.
        gamma: Per-epoch decay factor.
        last_epoch: Resume point.
    """

    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma should be in (0, 1], got {gamma}")
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def factor(self, epoch: int) -> float:
        return self.gamma ** max(epoch, 0)


class CosineAnnealingLR(LRScheduler):
    """Half a cosine from ``base_lr`` down to ``eta_min`` over ``T_max`` epochs.

    Args:
        optimizer: Optimizer to schedule.
        T_max: Epoch at which the lr reaches ``eta_min``.
        eta_min: Floor of the cosine, in absolute lr units.
        last_epoch: Resume point.

    ``lr(t) = eta_min + (base - eta_min) * (1 + cos(pi * t / T_max)) / 2``.

    Deliberately **not clamped** past ``T_max``: this is the full cosine, so at
    ``t = 2*T_max`` the lr is back at ``base``. That is PyTorch's closed form and
    it is what warm restarts are built on, but it surprises anyone who trains
    past ``T_max`` by accident and watches the lr climb. For a schedule that
    holds a floor forever, use :class:`WarmupCosine`.

    ``eta_min`` is absolute, so with several param groups it is *not* a
    multiplier and the group with the smallest base lr may already sit below it.
    Per-group floors are what :class:`WarmupCosine`'s ratio form is for.
    """

    def __init__(
        self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1
    ) -> None:
        if T_max <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def factors(self, epoch: int) -> list[float]:
        cos = (1.0 + math.cos(math.pi * max(epoch, 0) / self.T_max)) / 2.0
        # Expressed as a factor on base_lr so the machinery stays multiplicative
        # even though eta_min is an absolute lr.
        return [
            1.0 if b == 0.0 else (self.eta_min + (b - self.eta_min) * cos) / b
            for b in self.base_lrs
        ]


class LinearWarmup(LRScheduler):
    """Linear ramp from ``start_factor`` to ``end_factor`` over ``warmup_steps``.

    Args:
        optimizer: Optimizer to schedule.
        warmup_steps: Number of epochs/iterations the ramp takes. The factor
            reaches ``end_factor`` **at** ``t == warmup_steps`` and holds there.
        start_factor: Multiplier at ``t = 0``.
        end_factor: Multiplier once the ramp completes.
        last_epoch: Resume point.

    ``start_factor=0.0`` means a literal zero learning rate on the first step,
    which is a wasted step but harmless. Adam users often prefer a small
    non-zero start so the very first update still populates the moments.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        start_factor: float = 0.0,
        end_factor: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
        self.warmup_steps = int(warmup_steps)
        self.start_factor = float(start_factor)
        self.end_factor = float(end_factor)
        super().__init__(optimizer, last_epoch)

    def factor(self, epoch: int) -> float:
        t = min(max(epoch, 0), self.warmup_steps)
        alpha = t / self.warmup_steps
        return self.start_factor + (self.end_factor - self.start_factor) * alpha


class ChainedScheduler(LRScheduler):
    """Apply several schedules at once by multiplying their factors.

    Args:
        optimizer: Optimizer to schedule. Must be the one every child was built
            on, or the base lrs would disagree.
        schedulers: Children whose factors are multiplied together.
        last_epoch: Resume point.

    Because every child's factor is a closed-form function of the epoch, the
    composition is exact and the children are never stepped -- they carry no
    state that matters. PyTorch's version does step each child in turn, which
    makes the composition order-dependent for any child that reads back the
    current lr.
    """

    def __init__(
        self, optimizer: Optimizer, schedulers: Sequence[LRScheduler], last_epoch: int = -1
    ) -> None:
        if not schedulers:
            raise ValueError("ChainedScheduler needs at least one scheduler")
        for s in schedulers:
            if s.optimizer is not optimizer:
                raise ValueError("every chained scheduler must wrap the same optimizer")
        self.schedulers = list(schedulers)
        super().__init__(optimizer, last_epoch)

    def factors(self, epoch: int) -> list[float]:
        out = [1.0] * len(self.base_lrs)
        for s in self.schedulers:
            for i, f in enumerate(s.factors(epoch)):
                out[i] *= f
        return out

    def state_dict(self) -> dict[str, Any]:
        state = {k: v for k, v in self.__dict__.items() if k not in ("optimizer", "schedulers")}
        state["schedulers"] = [s.state_dict() for s in self.schedulers]
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        children = state_dict.get("schedulers", [])
        self.__dict__.update(
            {k: v for k, v in state_dict.items() if k not in ("optimizer", "schedulers")}
        )
        for s, cs in zip(self.schedulers, children, strict=True):
            s.load_state_dict(cs)


class SequentialLR(LRScheduler):
    """Switch between schedules at ``milestones``, restarting each child's clock.

    Args:
        optimizer: Optimizer to schedule.
        schedulers: ``len(milestones) + 1`` schedules, in order.
        milestones: Epochs at which to hand over to the next schedule.
        last_epoch: Resume point.

    The active child sees ``epoch - milestone``, not the global epoch: a cosine
    placed after a 100-step warmup should start at the top of its cosine, not
    100 steps into it. Forgetting the offset is the usual way a warmup-then-decay
    schedule ends up skipping most of the decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: Sequence[LRScheduler],
        milestones: Sequence[int],
        last_epoch: int = -1,
    ) -> None:
        if len(schedulers) != len(milestones) + 1:
            raise ValueError(
                f"expected {len(milestones) + 1} schedulers for {len(milestones)} milestones, "
                f"got {len(schedulers)}"
            )
        for s in schedulers:
            if s.optimizer is not optimizer:
                raise ValueError("every sequential scheduler must wrap the same optimizer")
        self.schedulers = list(schedulers)
        self.milestones = [int(m) for m in milestones]
        if self.milestones != sorted(self.milestones):
            raise ValueError(f"milestones must be increasing, got {self.milestones}")
        super().__init__(optimizer, last_epoch)

    def factors(self, epoch: int) -> list[float]:
        e = max(epoch, 0)
        idx = bisect.bisect_right(self.milestones, e)
        offset = 0 if idx == 0 else self.milestones[idx - 1]
        return self.schedulers[idx].factors(e - offset)

    def state_dict(self) -> dict[str, Any]:
        state = {k: v for k, v in self.__dict__.items() if k not in ("optimizer", "schedulers")}
        state["schedulers"] = [s.state_dict() for s in self.schedulers]
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        children = state_dict.get("schedulers", [])
        self.__dict__.update(
            {k: v for k, v in state_dict.items() if k not in ("optimizer", "schedulers")}
        )
        for s, cs in zip(self.schedulers, children, strict=True):
            s.load_state_dict(cs)


class WarmupCosine(LRScheduler):
    """Linear warmup, then cosine decay to a floor and hold. The nanoGPT schedule.

    Args:
        optimizer: Optimizer to schedule.
        warmup_steps: Steps of linear ramp. The factor is ``(t+1)/(warmup+1)``,
            so it reaches exactly 1.0 at ``t == warmup_steps`` -- where the
            cosine also starts at 1.0, making the join continuous.
        total_steps: Step at which the cosine bottoms out. Must exceed
            ``warmup_steps``.
        min_lr_ratio: The floor, as a fraction of each group's base lr.
        last_epoch: Resume point.

    ``min_lr_ratio`` is a *ratio* rather than an absolute ``min_lr`` on purpose:
    the whole schedule is a multiplier on a per-group ``base_lr``, so an
    absolute floor would mean different things for different groups -- and for a
    group whose base lr is already below it, would mean raising the lr.

    Past ``total_steps`` the factor holds at ``min_lr_ratio`` rather than
    resuming the cosine. Training past the horizon you planned for should coast,
    not silently re-warm the way an unclamped cosine does.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if total_steps <= warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must exceed warmup_steps ({warmup_steps})"
            )
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer, last_epoch)

    def factor(self, epoch: int) -> float:
        t = max(epoch, 0)
        if t < self.warmup_steps:
            return (t + 1) / (self.warmup_steps + 1)
        if t >= self.total_steps:
            return self.min_lr_ratio
        progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + cos * (1.0 - self.min_lr_ratio)
