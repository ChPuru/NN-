"""Checkpointing, seeding, metering and plotting."""

from __future__ import annotations

from .meters import Accuracy, AverageMeter, ProgressBar, Timer, format_time
from .seed import RNGContext, seed_everything, spawn
from .serialization import (
    CheckpointError,
    load_checkpoint,
    restore,
    save_checkpoint,
    snapshot,
)

__all__ = [
    "Accuracy",
    "AverageMeter",
    "CheckpointError",
    "ProgressBar",
    "RNGContext",
    "Timer",
    "format_time",
    "load_checkpoint",
    "restore",
    "save_checkpoint",
    "seed_everything",
    "snapshot",
    "spawn",
]


def __getattr__(name: str):
    """Plotting is imported lazily so the package works without matplotlib."""
    if name in (
        "plot_weights",
        "plot_history",
        "plot_confusion",
        "plot_filters_conv",
        "confusion_matrix",
    ):
        from . import viz

        return getattr(viz, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
