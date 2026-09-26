"""Plots for weights, training curves, and confusion matrices.

``matplotlib`` is imported **inside** each function, never at module scope, so
``import scratchgrad.utils.viz`` succeeds on a machine that only has NumPy. The
package must not acquire a plotting dependency to be trainable.

Two rendering decisions here are not cosmetic; they decide whether the picture
tells the truth.

**Diverging colormap, global symmetric scale.** Weights are signed and centred
near zero. Rendered in grayscale, ``-0.5`` and ``+0.5`` become two different
greys with nothing to say which one is negative, and the reader silently invents
a sign convention. The fix is a blue-white-red map with ``vmin=-m, vmax=+m``
where ``m = max|W|`` over the *whole* array, so white is exactly zero and the
two signs are distinguishable at a glance.

**Never normalise per tile.** Scaling each filter to its own min/max is the
default in a lot of visualisation code and it is actively misleading: a dead
all-zero filter has a range of zero, gets stretched by floating-point dust into
full-contrast noise, and appears in the grid as a healthy learned feature. On a
global scale a dead filter is what it is -- a flat white square. That is the
single most useful thing this plot can tell you.

**Log-scale loss.** Optimisation is roughly exponential; on a linear y-axis the
first two epochs consume the whole range and every later epoch is flat against
the axis, which is exactly the region you are trying to read.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "confusion_matrix",
    "plot_confusion",
    "plot_filters_conv",
    "plot_history",
    "plot_weights",
]

#: Blue-white-red. White sits at zero given a symmetric vmin/vmax.
DIVERGING_CMAP = "bwr"


def _pyplot(path: str | os.PathLike | None):
    """Import pyplot, forcing Agg when the figure is headed for a file.

    Selecting the backend before ``pyplot`` is first imported is the only point
    at which it can be selected; doing it here keeps every caller from having to
    remember, and keeps a headless CI run from failing on a missing display.
    """
    import matplotlib

    if path is not None and matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, path: str | os.PathLike | None, dpi: int = 130):
    """Save and close, or hand the figure back when ``path`` is None."""
    if path is None:
        return fig
    target = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(target))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    # Closing matters in a training loop: pyplot keeps every unclosed figure
    # alive and a per-epoch plot leaks until the process dies.
    import matplotlib.pyplot as plt

    plt.close(fig)
    return target


def _to_numpy(x: Any) -> np.ndarray:
    data = getattr(x, "data", None)
    return np.asarray(data if isinstance(data, np.ndarray) else x, dtype=np.float64)


def _upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour zoom by an integer factor.

    ``np.kron`` with a block of ones is exact nearest-neighbour: no
    interpolation, so a single pixel of weight stays a single crisp square
    rather than a blurred blob that reads as structure.
    """
    if factor <= 1:
        return img
    return np.kron(img, np.ones((factor, factor), dtype=img.dtype))


def _grid_dims(n: int, cols: int | None) -> tuple[int, int]:
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    cols = max(1, min(int(cols), n))
    return int(np.ceil(n / cols)), cols


def _mosaic(tiles: np.ndarray, rows: int, cols: int, gap: int, fill: float) -> np.ndarray:
    """Lay ``(n, H, W)`` tiles into one image with ``gap`` pixels between them."""
    n, h, w = tiles.shape
    out = np.full((rows * (h + gap) + gap, cols * (w + gap) + gap), fill, dtype=np.float64)
    for i in range(n):
        r, c = divmod(i, cols)
        y, x = gap + r * (h + gap), gap + c * (w + gap)
        out[y : y + h, x : x + w] = tiles[i]
    return out


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def plot_weights(
    W: Any,
    shape: tuple[int, int] = (28, 28),
    path: str | os.PathLike | None = None,
    max_units: int = 64,
    cols: int | None = None,
    upscale: int = 16,
    title: str = "first-layer weights",
):
    """Render first-layer weights as a grid of images on one global scale.

    ``Linear.weight`` is ``(in_features, out_features)``, so a hidden unit's
    receptive field is a **column**. Rows are accepted too when the row length
    matches ``shape``, since a checkpoint from a transposed convention is a
    common thing to be handed.

    Args:
        W: Weight array or Tensor, 2-D.
        shape: Image shape each filter reshapes to, e.g. ``(28, 28)``.
        path: Output file. ``None`` returns the figure instead of saving.
        max_units: Cap on how many filters to draw.
        cols: Grid columns; square-ish by default.
        upscale: Integer nearest-neighbour zoom, ``>= 16`` recommended so a
            28x28 filter is legible in a report at normal size.
        title: Figure title.

    Returns:
        The written path, or the figure when ``path`` is ``None``.

    Raises:
        ValueError: If neither axis of ``W`` matches ``prod(shape)``, or if
            ``max_units`` or the unit count leaves nothing to draw.
    """
    arr = _to_numpy(W)
    if arr.ndim != 2:
        raise ValueError(f"plot_weights expects a 2-D weight array, got shape {arr.shape}")
    n_pix = int(shape[0]) * int(shape[1])
    if arr.shape[0] == n_pix:
        filters = arr.T  # (units, pixels): columns are receptive fields
    elif arr.shape[1] == n_pix:
        filters = arr
    else:
        raise ValueError(
            f"neither axis of W{arr.shape} equals prod(shape)={n_pix}; "
            "pass the shape the first layer actually consumes"
        )

    k = min(int(max_units), filters.shape[0])
    if k <= 0:
        # Without this, an empty slice reaches np.max below and numpy raises
        # "zero-size array to reduction operation maximum", which names neither
        # the argument at fault nor the function the caller actually called.
        raise ValueError(
            f"nothing to plot: max_units={max_units} and W has {filters.shape[0]} units; "
            "both must be >= 1"
        )
    tiles = filters[:k].reshape(k, int(shape[0]), int(shape[1]))
    rows, ncols = _grid_dims(k, cols)

    # ONE scale for every tile, symmetric about zero. Per-tile normalisation
    # would turn a dead filter into convincing noise.
    m = float(np.max(np.abs(filters[:k])))
    if m == 0.0:
        m = 1.0

    tiles = np.stack([_upscale(t, int(upscale)) for t in tiles])
    gap = max(1, int(upscale) // 4)
    img = _mosaic(tiles, rows, ncols, gap, fill=0.0)  # 0.0 -> white on a bwr map

    plt = _pyplot(path)
    fig, ax = plt.subplots(figsize=(1.0 + 1.4 * ncols, 1.4 + 1.4 * rows))
    im = ax.imshow(img, cmap=DIVERGING_CMAP, vmin=-m, vmax=m, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(f"{title}  (n={k}, scale +/-{m:.3g})")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return _finish(fig, path)


def plot_filters_conv(
    W: Any,
    path: str | os.PathLike | None = None,
    max_filters: int = 32,
    upscale: int = 16,
    title: str = "conv filters",
):
    """Render ``(out, in, kH, kW)`` conv weights: one row per output channel.

    The layout is the frozen one from the API contract --
    ``(out_channels, in_channels, kH, kW)``. Every kernel shares the same
    symmetric colour scale for the same reason as :func:`plot_weights`.

    Args:
        W: Conv weight, 4-D.
        path: Output file, or ``None`` to get the figure back.
        max_filters: Cap on output channels drawn.
        upscale: Integer nearest-neighbour zoom (3x3 kernels need a lot of it).
        title: Figure title.

    Raises:
        ValueError: If ``W`` is not 4-D, or if ``max_filters`` or an empty
            channel axis leaves nothing to draw.
    """
    arr = _to_numpy(W)
    if arr.ndim != 4:
        raise ValueError(f"plot_filters_conv expects (out, in, kH, kW), got shape {arr.shape}")
    n_out = min(int(max_filters), arr.shape[0])
    n_in = arr.shape[1]
    if n_out <= 0 or n_in <= 0:
        # Same failure as plot_weights: an empty slice would die inside np.max
        # with a message that mentions neither max_filters nor W.
        raise ValueError(
            f"nothing to plot: max_filters={max_filters} and W is {arr.shape}; "
            "out_channels and in_channels must both be >= 1"
        )
    sub = arr[:n_out]

    m = float(np.max(np.abs(sub)))
    if m == 0.0:
        m = 1.0

    tiles = np.stack([_upscale(t, int(upscale)) for t in sub.reshape(n_out * n_in, *arr.shape[2:])])
    gap = max(1, int(upscale) // 4)
    img = _mosaic(tiles, n_out, n_in, gap, fill=0.0)

    plt = _pyplot(path)
    fig, ax = plt.subplots(figsize=(1.0 + 1.1 * n_in, 1.2 + 1.1 * n_out))
    im = ax.imshow(img, cmap=DIVERGING_CMAP, vmin=-m, vmax=m, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(f"{title}  ({n_out}x{n_in} kernels, scale +/-{m:.3g})")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# history
# --------------------------------------------------------------------------- #
def plot_history(
    history: Mapping[str, Sequence[float]],
    path: str | os.PathLike | None = None,
    title: str = "training history",
):
    """Loss on a log y-axis, accuracy on a linear twin axis.

    Keys are routed by name: anything containing ``loss`` goes on the left
    (log) axis, anything containing ``acc`` on the right (linear) one. Other
    keys are ignored rather than guessed at.

    A log axis cannot show a non-positive value, and a loss can legitimately be
    zero or (for some objectives) negative, so the scale falls back to
    ``symlog`` in that case instead of dropping the point without saying so.

    Args:
        history: e.g. ``{"train_loss": [...], "val_loss": [...],
            "val_acc": [...]}``. Series may have different lengths.
        path: Output file, or ``None`` to get the figure back.
        title: Figure title.

    Raises:
        ValueError: If no key looks like a loss or an accuracy.
    """
    loss_keys = [k for k in history if "loss" in k.lower()]
    acc_keys = [k for k in history if "acc" in k.lower()]
    if not loss_keys and not acc_keys:
        raise ValueError(f"no 'loss' or 'acc' series in history keys {list(history)}")

    plt = _pyplot(path)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    all_loss: list[float] = []
    for key in loss_keys:
        y = np.asarray(history[key], dtype=np.float64)
        ax.plot(np.arange(1, len(y) + 1), y, marker="o", ms=3, lw=1.5, label=key)
        all_loss.extend(y.tolist())

    if all_loss:
        finite = np.asarray([v for v in all_loss if np.isfinite(v)], dtype=np.float64)
        if finite.size and finite.min() > 0:
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_ylabel("loss (log scale)")
    ax.set_xlabel("epoch")
    ax.grid(True, which="both", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if acc_keys:
        ax2 = ax.twinx()
        for i, key in enumerate(acc_keys):
            y = np.asarray(history[key], dtype=np.float64)
            ax2.plot(
                np.arange(1, len(y) + 1),
                y,
                ls="--",
                marker="s",
                ms=3,
                lw=1.3,
                color=f"C{len(loss_keys) + i}",
                label=key,
            )
        ax2.set_ylabel("accuracy (linear)")
        # Accuracy is a fraction; pinning the axis stops a 0.88-0.89 window
        # from being drawn as a dramatic climb.
        ax2.set_ylim(0.0, 1.0)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2

    ax.set_title(title)
    if labels:
        ax.legend(handles, labels, loc="best", fontsize=9)
    fig.tight_layout()
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# confusion
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: Any, y_pred: Any, n_classes: int) -> np.ndarray:
    """Counts of ``(true, predicted)`` pairs, shape ``(n_classes, n_classes)``.

    Built with ``np.add.at``: ``cm[t, p] += 1`` on fancy indices keeps only the
    *last* write per repeated index, which for a confusion matrix means almost
    every count comes out as 1. ``np.add.at`` is the unbuffered version that
    actually accumulates. No sklearn -- it is not installed and must not become
    a dependency.

    Args:
        y_true: Integer labels, shape ``(N,)``.
        y_pred: Integer predictions, shape ``(N,)``, or logits ``(N, C)``.
        n_classes: Number of classes.

    Returns:
        ``(n_classes, n_classes)`` int64 array; rows are true labels.

    Raises:
        ValueError: On a length mismatch or an out-of-range label.
    """
    t = np.asarray(getattr(y_true, "data", y_true))
    p = np.asarray(getattr(y_pred, "data", y_pred))
    if p.ndim == 2:
        p = p.argmax(axis=1)
    t = t.reshape(-1).astype(np.int64, copy=False)
    p = p.reshape(-1).astype(np.int64, copy=False)
    if t.shape != p.shape:
        raise ValueError(f"y_true has {t.shape[0]} entries but y_pred has {p.shape[0]}")
    n = int(n_classes)
    if t.size and (t.min() < 0 or t.max() >= n or p.min() < 0 or p.max() >= n):
        raise ValueError(
            f"labels must lie in [0, {n}); got true[{t.min()},{t.max()}] pred[{p.min()},{p.max()}]"
        )
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    return cm


def plot_confusion(
    cm: Any,
    path: str | os.PathLike | None = None,
    class_names: Sequence[str] | None = None,
    normalize: bool = False,
    title: str = "confusion matrix",
):
    """Draw a confusion matrix with the counts written into the cells.

    Args:
        cm: ``(C, C)`` matrix, rows = true labels (as :func:`confusion_matrix`).
        path: Output file, or ``None`` to get the figure back.
        class_names: Tick labels; indices by default.
        normalize: Divide each row by its total, giving per-class recall on the
            diagonal. Rows with no samples stay at zero rather than becoming
            NaN, so a class absent from the test split does not blank the plot.
        title: Figure title.
    """
    mat = np.asarray(getattr(cm, "data", cm))
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"confusion matrix must be square, got shape {mat.shape}")
    n = mat.shape[0]
    disp = mat.astype(np.float64)
    if normalize:
        row = disp.sum(axis=1, keepdims=True)
        disp = np.divide(disp, row, out=np.zeros_like(disp), where=row > 0)

    plt = _pyplot(path)
    fig, ax = plt.subplots(figsize=(1.4 + 0.62 * n, 1.4 + 0.62 * n))
    # Counts are non-negative, so a sequential map is correct here -- the
    # diverging-map argument applies to signed quantities like weights.
    im = ax.imshow(disp, cmap="Blues", interpolation="nearest", vmin=0.0)
    names = list(class_names) if class_names is not None else [str(i) for i in range(n)]
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title + (" (row-normalized)" if normalize else ""))

    hi = disp.max() if disp.size else 0.0
    for i in range(n):
        for j in range(n):
            v = disp[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}" if normalize else f"{int(mat[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
                # White on the dark cells, black on the light ones; a single
                # colour is unreadable at one end of the ramp or the other.
                color="white" if hi and v > 0.6 * hi else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    return _finish(fig, path)
