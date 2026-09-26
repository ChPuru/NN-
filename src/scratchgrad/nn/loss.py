"""Loss functions as modules.

Every loss here takes **raw logits**, not probabilities. Layers emit logits and
the loss owns the softmax; the alternative -- a model ending in ``Softmax`` fed
to a cross-entropy that softmaxes again -- preserves the argmax, so accuracy
looks fine and only the loss reveals the bug, pinned at 1.4615 for ten classes.
"""

from __future__ import annotations

from ..tensor import Tensor
from . import functional as F
from .module import Module

__all__ = [
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "HuberLoss",
    "L1Loss",
    "MSELoss",
    "NLLLoss",
]


class CrossEntropyLoss(Module):
    """Softmax cross-entropy from raw logits and integer class indices."""

    def __init__(
        self,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target) -> Tensor:
        return F.cross_entropy(
            logits,
            target,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}, label_smoothing={self.label_smoothing}"


class NLLLoss(Module):
    """Negative log-likelihood of **already log-softmaxed** inputs."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, log_probs: Tensor, target) -> Tensor:
        return F.nll_loss(log_probs, target, reduction=self.reduction)


class MSELoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target) -> Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)


class L1Loss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target) -> Tensor:
        return F.l1_loss(pred, target, reduction=self.reduction)


class HuberLoss(Module):
    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.delta = float(delta)
        self.reduction = reduction

    def forward(self, pred: Tensor, target) -> Tensor:
        return F.huber_loss(pred, target, delta=self.delta, reduction=self.reduction)


class BCEWithLogitsLoss(Module):
    def __init__(self, reduction: str = "mean", pos_weight: float | None = None) -> None:
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits: Tensor, target) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, target, reduction=self.reduction, pos_weight=self.pos_weight
        )
