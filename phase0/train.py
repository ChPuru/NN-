"""Train the Phase 0 MLP: 784 -> 128 (ReLU) -> 10, Adam, ~5 epochs.

    python3 phase0/train.py --smoke        # 30 seconds, no download, proves the wiring
    python3 phase0/train.py --synthetic    # whole pipeline offline on fake data
    python3 phase0/train.py                # the real thing: ~2 minutes on a CPU

Expect **97.5-98.2%** test accuracy. That is where a plain MLP on raw pixels
tops out; the last two points belong to a convolution, which has the prior that
a stroke means the same thing wherever it lands. Reporting 99.x% from a 128-unit
MLP means something leaked.

Three sanity checks run before any real training, in this order, because each
one fails differently and the order tells you which thing broke:

1. **Loss at initialization.** Zero the *final* layer's ``W`` and ``b``: the
   logits are then exactly uniform and the loss must be ``ln(10) =
   2.302585092994046`` to 1e-12 in float64. Note what this does *not* say --
   without zeroing, ``L0`` fluctuates with the draw and can legitimately land
   slightly *below* ``ln(C)`` on a given batch. Only a systematic deficit across
   seeds is evidence of a bug (usually a label leak or a double softmax, which
   pins the loss at 1.4615 for C=10).
2. **Overfit 32 examples.** If the model cannot drive 32 samples to ~0 loss, it
   is broken, not undertrained, and no amount of real training will fix it.
3. **One permutation, indexed into both X and y.** Shuffling them with two
   separate calls decouples the labels from the images; the loss then parks at
   exactly ``ln(C)``, and -- the nasty part -- overfit-one-batch still succeeds,
   because memorising random labels is a thing a network can do.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
from mnist import load_mnist, synthetic_mnist
from scratchnn import MLP, Adam, SoftmaxCrossEntropy, accuracy

__all__ = [
    "check_loss_at_init",
    "epoch_batches",
    "evaluate",
    "plot_first_layer_weights",
    "save_history",
    "save_plots",
    "save_weights",
    "smoke_overfit",
    "train",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WEIGHTS = os.path.join(_HERE, "weights.npz")
# phase0/assets, not the repository-level one: this directory is meant to be
# copyable on its own, and its README points at these files by relative path.
_DEFAULT_ASSETS = os.path.join(_HERE, "assets")


# --------------------------------------------------------------------------- #
# sanity checks
# --------------------------------------------------------------------------- #
def check_loss_at_init(model: MLP, x: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """Zero the final layer and assert the loss is exactly ``ln(n_classes)``.

    With the last ``W`` and ``b`` at zero every logit is 0, so the softmax is
    exactly uniform and the cross-entropy is ``-log(1/C) = ln(C)`` -- an
    algebraic identity, not an approximation, which is why the tolerance is
    1e-12 and the arithmetic is float64. A failure here is a broken loss, and
    checking it against a *deterministic* value is the whole reason for zeroing:
    the un-zeroed initial loss is a random variable.

    Args:
        model: The MLP. Its final layer is restored before returning.
        x: A batch of inputs.
        y: The matching labels.
        n_classes: Number of classes.

    Returns:
        The measured loss.

    Raises:
        AssertionError: If the loss differs from ``ln(C)`` by more than 1e-12.
    """
    last = model.dense_layers()[-1]
    saved_w, saved_b = last.W.copy(), last.b.copy()
    try:
        last.W[...] = 0.0
        last.b[...] = 0.0
        loss = SoftmaxCrossEntropy().forward(model.forward(x), y)
    finally:
        last.W[...] = saved_w
        last.b[...] = saved_b
    target = math.log(n_classes)
    if abs(loss - target) > 1e-12:
        raise AssertionError(
            f"loss with a zeroed final layer is {loss!r}, expected ln({n_classes}) = {target!r} "
            f"(off by {loss - target:.3e}); the cross-entropy is wrong"
        )
    return loss


def smoke_overfit(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
    n: int = 32,
    steps: int = 600,
    lr: float = 3e-3,
    verbose: bool = True,
) -> float:
    """Drive ``n`` examples to ~zero loss. The cheapest possible integration test.

    This exercises forward, backward, and the optimizer against the one target a
    correct implementation can always hit: a training set small enough to
    memorise. If it plateaus, stop tuning the learning rate -- something in the
    chain rule is wrong.

    Returns:
        The final loss.
    """
    xs, ys = x[:n], y[:n]
    model = MLP([xs.shape[1], 128, n_classes], rng)
    crit = SoftmaxCrossEntropy()
    opt = Adam(model.params(), lr=lr)
    loss = float("nan")
    for step in range(steps):
        loss = crit.forward(model.forward(xs), ys)
        opt.zero_grad()
        model.backward(crit.backward())
        opt.step()
        if verbose and (step % 150 == 0 or step == steps - 1):
            acc = accuracy(model.forward(xs), ys)
            print(f"    step {step:4d}  loss {loss:.6f}  acc {acc:.3f}")
    return loss


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def epoch_batches(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    """Yield shuffled minibatches.

    **One** permutation, indexed into both arrays. Two separate shuffles is the
    classic silent killer: the images and labels decouple, the loss settles at
    exactly ``ln(C)``, and because a network can memorise random labels the
    overfit-a-batch check still passes.
    """
    perm = rng.permutation(x.shape[0])
    xs, ys = x[perm], y[perm]
    for start in range(0, xs.shape[0], batch_size):
        yield xs[start : start + batch_size], ys[start : start + batch_size]


def evaluate(
    model: MLP, x: np.ndarray, y: np.ndarray, batch_size: int = 1000
) -> tuple[float, float]:
    """Mean loss and accuracy over ``x``, in batches so memory stays flat."""
    crit = SoftmaxCrossEntropy()
    total_loss, correct = 0.0, 0
    for start in range(0, x.shape[0], batch_size):
        xb, yb = x[start : start + batch_size], y[start : start + batch_size]
        logits = model.forward(xb)
        total_loss += crit.forward(logits, yb) * xb.shape[0]
        correct += int(np.sum(np.argmax(logits, axis=1) == yb))
    return total_loss / x.shape[0], correct / x.shape[0]


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    rng: np.random.Generator,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden: int = 128,
    verbose: bool = True,
) -> tuple[MLP, dict[str, list[float]]]:
    """Run the full training loop and return the model plus its history."""
    n_classes = int(y_train.max()) + 1
    model = MLP([x_train.shape[1], hidden, n_classes], rng)
    crit = SoftmaxCrossEntropy()
    opt = Adam(model.params(), lr=lr)

    if verbose:
        print(f"  model {model.sizes}  parameters {model.num_parameters():,}")
    check_loss_at_init(model, x_train[:256], y_train[:256], n_classes)
    if verbose:
        print(f"  sanity: zeroed-head loss == ln({n_classes}) to 1e-12  ok")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_seconds": [],
    }
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        running, seen = 0.0, 0
        for xb, yb in epoch_batches(x_train, y_train, batch_size, rng):
            loss = crit.forward(model.forward(xb), yb)
            opt.zero_grad()
            model.backward(crit.backward())
            opt.step()
            running += loss * xb.shape[0]
            seen += xb.shape[0]
        elapsed = time.perf_counter() - t0
        train_loss, train_acc = running / seen, evaluate(model, x_train, y_train)[1]
        test_loss, test_acc = evaluate(model, x_test, y_test)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_seconds"].append(elapsed)
        if verbose:
            print(
                f"  epoch {epoch}/{epochs}  train {train_loss:.4f}/{train_acc:.4f}   "
                f"test {test_loss:.4f}/{test_acc:.4f}   ({elapsed:.1f}s)"
            )
    return model, history


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def save_weights(model: MLP, path: str, test_acc: float, sizes: list[int]) -> str:
    """Write ``weights.npz``: ``W1, b1, W2, b2`` plus the metadata the demo shows."""
    d0, d1 = model.dense_layers()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        W1=d0.W,
        b1=d0.b,
        W2=d1.W,
        b2=d1.b,
        sizes=np.asarray(sizes),
        test_accuracy=np.asarray(test_acc),
    )
    return path


def save_history(
    path: str,
    history: dict[str, list[float]],
    config: dict[str, object],
    extras: dict[str, object],
) -> str:
    """Write ``history.json`` beside the checkpoint so every quoted number is regenerable.

    The file records the exact command-line configuration alongside the numbers,
    which is the difference between a result and an anecdote: a reader can re-run
    it and diff, rather than take the README's word for it.

    Args:
        path: Destination ``.json`` path.
        history: Per-epoch lists as returned by :func:`train`.
        config: Seed, epochs, batch size, lr, hidden width -- everything that
            changes the answer.
        extras: Measured scalars (final accuracies, wall clock, parameter count).

    Returns:
        The path written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {"config": config, "history": history, **extras}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def plot_first_layer_weights(W1: np.ndarray, path: str, n_show: int = 64, upscale: int = 16) -> str:
    """Render the first ``n_show`` hidden units' receptive fields as one mosaic.

    Three choices here are the whole reason this plot is worth looking at, and
    the obvious version of each is wrong:

    * **A diverging blue-white-red map, not grayscale.** These weights are signed
      and roughly symmetric about zero. In grayscale -0.5 and +0.5 are simply two
      different greys with nothing to say which is which, so the picture reads as
      texture rather than as "this unit votes *against* ink here".
    * **One global symmetric scale**, ``vmin = -max|W1|``, ``vmax = +max|W1|``,
      shared by every tile. Per-tile normalisation is the common version and it
      lies twice over: it stretches a dead, all-but-zero filter into full-contrast
      noise that is indistinguishable from a healthy learned feature, and it
      destroys any comparison of magnitude between units.
    * **Nearest-neighbour upscaling by an integer factor**, done to the array
      before it reaches the renderer. A 28x28 tile drawn into a few hundred screen
      pixels is otherwise resampled by whatever filter the backend picks, which
      invents intermediate values that no weight has.

    The units shown are the first ``n_show`` in index order -- not the most
    structured ones. A grid picked for looks would say nothing about the other 64.

    Args:
        W1: The ``(n_in, n_hidden)`` first-layer weight matrix.
        path: Destination PNG.
        n_show: Number of hidden units to draw, laid out in a square grid.
        upscale: Integer nearest-neighbour magnification, at least 16.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    side = round(math.sqrt(W1.shape[0]))
    if side * side != W1.shape[0]:
        raise ValueError(f"first-layer fan-in {W1.shape[0]} is not a square image")
    n_show = min(n_show, W1.shape[1])
    cols = math.ceil(math.sqrt(n_show))
    rows = math.ceil(n_show / cols)

    # NaN in the gutters, not zero: zero is white here and would be
    # indistinguishable from a genuinely dead filter.
    gap = 1
    mosaic = np.full((rows * (side + gap) - gap, cols * (side + gap) - gap), np.nan)
    for k in range(n_show):
        r, c = divmod(k, cols)
        tile = W1[:, k].reshape(side, side)
        mosaic[
            r * (side + gap) : r * (side + gap) + side, c * (side + gap) : c * (side + gap) + side
        ] = tile

    vmax = float(np.max(np.abs(W1[:, :n_show])))
    big = np.kron(mosaic, np.ones((upscale, upscale)))  # nearest neighbour, exactly

    cmap = plt.get_cmap("bwr").copy()
    cmap.set_bad("#f2f2f2")
    dpi = 100
    h_px, w_px = big.shape
    pad_l, pad_r, pad_t, pad_b = 40, 260, 130, 60
    fig_w, fig_h = (w_px + pad_l + pad_r) / dpi, (h_px + pad_t + pad_b) / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes(
        [pad_l / (fig_w * dpi), pad_b / (fig_h * dpi), w_px / (fig_w * dpi), h_px / (fig_h * dpi)]
    )
    im = ax.imshow(big, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    cax = fig.add_axes(
        [
            (pad_l + w_px + 40) / (fig_w * dpi),
            pad_b / (fig_h * dpi),
            60 / (fig_w * dpi),
            h_px / (fig_h * dpi),
        ]
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=18)
    fig.text(
        pad_l / (fig_w * dpi),
        1 - 45 / (fig_h * dpi),
        f"first-layer weights: hidden units 0-{n_show - 1}, {side}x{side} each",
        fontsize=22,
        va="center",
    )
    fig.text(
        pad_l / (fig_w * dpi),
        1 - 90 / (fig_h * dpi),
        f"blue = negative, white = 0, red = positive; one global symmetric scale "
        f"+/-{vmax:.3f}; {upscale}x nearest neighbour",
        fontsize=14,
        va="center",
        color="#444444",
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def save_plots(
    history: dict[str, list[float]], model: MLP, x_test, y_test, assets: str
) -> list[str]:
    """Training curves, the learned first-layer weights, and the model's mistakes."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display on a build box; must precede pyplot
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed; skipping plots")
        return []

    os.makedirs(assets, exist_ok=True)
    written = []
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(10, 4))
    ax_l.plot(epochs, history["train_loss"], "o-", label="train")
    ax_l.plot(epochs, history["test_loss"], "s-", label="test")
    ax_l.set_xlabel("epoch")
    ax_l.set_ylabel("cross-entropy")
    ax_l.set_title("loss")
    ax_l.legend()
    ax_l.grid(alpha=0.3)
    ax_a.plot(epochs, history["train_acc"], "o-", label="train")
    ax_a.plot(epochs, history["test_acc"], "s-", label="test")
    ax_a.set_xlabel("epoch")
    ax_a.set_ylabel("accuracy")
    ax_a.set_title(f"accuracy (final test {history['test_acc'][-1]:.4f})")
    ax_a.legend()
    ax_a.grid(alpha=0.3)
    fig.tight_layout()
    curves = os.path.join(assets, "training.png")
    fig.savefig(curves, dpi=130)
    plt.close(fig)
    written.append(curves)

    written.append(
        plot_first_layer_weights(model.dense_layers()[0].W, os.path.join(assets, "weights.png"))
    )

    logits = model.forward(x_test)
    pred = np.argmax(logits, axis=1)
    wrong = np.flatnonzero(pred != y_test)[:24]
    if wrong.size:
        cols = 8
        rows = math.ceil(wrong.size / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.5))
        for ax, idx in zip(np.atleast_1d(axes).ravel(), wrong, strict=False):
            ax.imshow(x_test[idx].reshape(28, 28), cmap="gray_r")
            ax.set_title(f"{y_test[idx]}->{pred[idx]}", fontsize=8)
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        fig.suptitle("test digits the MLP gets wrong")
        fig.tight_layout()
        misses = os.path.join(assets, "mistakes.png")
        fig.savefig(misses, dpi=130)
        plt.close(fig)
        written.append(misses)
    return written


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="overfit 32 examples and stop")
    ap.add_argument("--synthetic", action="store_true", help="fake offline data, no download")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default=_DEFAULT_WEIGHTS)
    ap.add_argument("--assets", default=_DEFAULT_ASSETS)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    if args.synthetic:
        print("data: SYNTHETIC (not MNIST -- pipeline exercise only)")
        x_tr, y_tr, x_te, y_te = synthetic_mnist(rng, n_train=3000, n_test=600)
    else:
        print("data: MNIST")
        x_tr, y_tr, x_te, y_te = load_mnist()
    n_classes = int(y_tr.max()) + 1
    print(
        f"  train {x_tr.shape}  test {x_te.shape}  pixels in [{x_tr.min():.2f}, {x_tr.max():.2f}]"
    )

    # Sanity 1 runs against a fresh model before anything else touches it.
    probe = MLP([x_tr.shape[1], args.hidden, n_classes], np.random.default_rng(args.seed))
    loss0 = check_loss_at_init(probe, x_tr[:256], y_tr[:256], n_classes)
    print(f"sanity 1: zeroed-head loss = {loss0!r} == ln({n_classes})  ok")
    print("          (un-zeroed L0 is a random variable; only a systematic")
    print("           deficit below ln(C) across seeds indicates a bug)")

    print("sanity 2: overfitting 32 examples")
    smoke_loss = smoke_overfit(x_tr, y_tr, n_classes, np.random.default_rng(args.seed + 1))
    if smoke_loss > 1e-3:
        print(
            f"  FAILED: 32 examples plateaued at {smoke_loss:.6f}; the model is broken, "
            "not undertrained"
        )
        return 1
    print(f"  ok: final loss {smoke_loss:.3e}")
    if args.smoke:
        return 0

    print(f"training {args.epochs} epochs, batch {args.batch_size}, Adam lr {args.lr}")
    t_start = time.perf_counter()
    model, history = train(
        x_tr,
        y_tr,
        x_te,
        y_te,
        rng,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
    )
    wall = time.perf_counter() - t_start
    test_acc = history["test_acc"][-1]
    test_loss = history["test_loss"][-1]
    print(f"\ntrained in {wall:.1f}s wall clock on CPU")
    print(
        f"final test accuracy : {test_acc * 100:.2f}%  ({round(test_acc * len(y_te))}/{len(y_te)})"
    )
    print(f"final train accuracy: {history['train_acc'][-1] * 100:.2f}%")
    print(f"parameters          : {model.num_parameters():,}")

    save_weights(model, args.weights, test_acc, model.sizes)
    print(f"wrote {args.weights}  (test accuracy {test_acc:.4f})")
    hist_path = os.path.join(os.path.dirname(os.path.abspath(args.weights)), "history.json")
    save_history(
        hist_path,
        history,
        {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden": args.hidden,
            "data": "synthetic" if args.synthetic else "mnist",
            "command": "python3 phase0/train.py",
        },
        {
            "loss_at_init": loss0,
            "ln_n_classes": math.log(n_classes),
            "smoke_overfit_loss": smoke_loss,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "train_accuracy": history["train_acc"][-1],
            "train_loss": history["train_loss"][-1],
            "wall_seconds": wall,
            "parameters": model.num_parameters(),
            "n_train": int(x_tr.shape[0]),
            "n_test": int(x_te.shape[0]),
        },
    )
    print(f"wrote {hist_path}")
    if not args.no_plots:
        for path in save_plots(history, model, x_te, y_te, args.assets):
            print(f"wrote {path}")
    if not args.synthetic:
        print(
            "\n97.5-98.2% is the ceiling for a 128-unit MLP on raw pixels. A small CNN\n"
            "reaches 99.2%+ because convolution knows a stroke means the same thing\n"
            "wherever it lands; this model has to learn that fact 784 separate times."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
