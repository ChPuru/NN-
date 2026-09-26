"""A small VGG-ish CNN on CIFAR-10 -- the Tier-3 proof that convolutions work.

Six 3x3 convolutions in three BN+ReLU pairs, max-pooled twice between them, and
a single linear head::

    [Conv(3->32) BN ReLU] x2   MaxPool(2)   32x32 -> 16x16
    [Conv(32->64) BN ReLU] x2  MaxPool(2)   16x16 -> 8x8
    [Conv(64->128) BN ReLU] x2 MaxPool(2)    8x8  -> 4x4
    Flatten  Dropout  Linear(128*4*4 -> 10)

Two things this script is careful about, because both are silent when wrong:

* **CIFAR-10 rows are planar.** The loader hands back ``(N, 3, 32, 32)`` already;
  the ``(-1, 32, 32, 3)`` misreading is the same number of bytes and trains to
  something like 45% while every instinct blames the architecture. So the shape
  is asserted *and* a sample grid is rendered to ``assets/cifar10_samples.png``
  with class names -- a channel-order mistake has to be visible, not inferred.
* **The test set is touched exactly once**, in the final evaluation. The last
  5000 training images are held out as a validation split and every decision
  (epoch count, learning rate, dropout) was made against that split.

    PYTHONPATH=src python examples/cifar10_cnn.py                 # full run
    PYTHONPATH=src python examples/cifar10_cnn.py --smoke         # 32 images, must reach ~0
    PYTHONPATH=src python examples/cifar10_cnn.py --time-steps 20 # cost per epoch, then stop
    PYTHONPATH=src python examples/cifar10_cnn.py --epochs 8 --seed 0

Why 8 epochs: ``--time-steps 25`` measured a median 2.094 s per batch-128 step
on this 20-core CPU, i.e. 352 steps = ~12 min per epoch, and epochs are the only
knob that trades wall clock for accuracy here. Eight is the smallest count whose
cosine schedule finished above the 70% target on the validation split; the run
recorded in ``runs/cifar10_cnn/history.json`` took 102.8 min.

Measured on that run (seed 0, one command, test set evaluated once):
**82.81% test accuracy**, 83.38% validation, 85.68% train, 307,946 parameters.

Every number the README quotes for this model comes out of this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import scratchgrad as sg
from scratchgrad import nn, optim
from scratchgrad.data import CIFAR10_CLASSES, cifar10
from scratchgrad.utils import Accuracy, AverageMeter, ProgressBar, save_checkpoint

VAL_SIZE = 5000  # the last 5000 training images; never trained on, tuned on freely


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def conv_block(in_c: int, out_c: int, rng: np.random.Generator, first: bool = False) -> list:
    """One ``Conv -> BatchNorm -> ReLU`` triple.

    Args:
        in_c: Input channels.
        out_c: Output channels.
        rng: Generator for the weight init.
        first: True for the network's first convolution, which switches off
            ``dX``. Nothing upstream of it consumes the input gradient and the
            ``col2im`` scatter it skips is the single largest saving available.

    Returns:
        The three modules, ready to splice into a ``Sequential``.
    """
    conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False, input_requires_grad=not first, rng=rng)
    # bias=False because BatchNorm's beta immediately follows and re-adds a
    # per-channel shift; the conv bias would be an exactly redundant parameter
    # whose gradient is cancelled by the normalisation.
    return [conv, nn.BatchNorm2d(out_c), nn.ReLU()]


def build_model(rng: np.random.Generator, dropout: float = 0.3) -> nn.Module:
    """The VGG-ish stack. 32 -> 64 -> 128 channels, halving the map each stage."""
    layers: list = []
    layers += conv_block(3, 32, rng, first=True) + conv_block(32, 32, rng)
    layers += [nn.MaxPool2d(2)]
    layers += conv_block(32, 64, rng) + conv_block(64, 64, rng)
    layers += [nn.MaxPool2d(2)]
    layers += conv_block(64, 128, rng) + conv_block(128, 128, rng)
    layers += [nn.MaxPool2d(2)]
    # Dropout takes the generator at construction: with rng=None it draws from a
    # fresh default_rng() per call and the run stops being reproducible.
    layers += [nn.Flatten(), nn.Dropout(dropout, rng=rng), nn.Linear(128 * 4 * 4, 10, rng=rng)]
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# augmentation -- by hand, in numpy
# --------------------------------------------------------------------------- #
def augment(x: np.ndarray, rng: np.random.Generator, pad: int = 4) -> np.ndarray:
    """Random 32x32 crop out of a reflect-padded 40x40, plus a random h-flip.

    Reflect padding rather than zero padding because the images are already
    per-channel standardised: a zero border would be the channel *mean*, i.e. a
    grey frame that the first convolution learns to key on. Reflection needs to
    know nothing about the normalisation.

    Both draws are per-image, vectorised over the batch -- a Python loop over
    128 images costs more than the convolution that follows it.

    Args:
        x: ``(N, 3, 32, 32)`` float batch.
        rng: Generator; the caller's, so a run replays exactly.
        pad: Border added before cropping.

    Returns:
        A new ``(N, 3, 32, 32)`` array. The input is never modified.
    """
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")
    oy = rng.integers(0, 2 * pad + 1, size=n)
    ox = rng.integers(0, 2 * pad + 1, size=n)
    rows = oy[:, None] + np.arange(h)  # (N, H) absolute row per image
    cols = ox[:, None] + np.arange(w)  # (N, W)
    out = xp[
        np.arange(n)[:, None, None, None],
        np.arange(c)[None, :, None, None],
        rows[:, None, :, None],
        cols[:, None, None, :],
    ]
    flip = rng.random(n) < 0.5
    out[flip] = out[flip][:, :, :, ::-1]
    return out


# --------------------------------------------------------------------------- #
# sanity checks
# --------------------------------------------------------------------------- #
def loss_at_init_is_ln_c(x: np.ndarray, y: np.ndarray) -> float:
    """Zero the head so the logits are exactly uniform, then check ``ln(10)``.

    Zeroing matters: without it the initial loss fluctuates and can legitimately
    land below ``ln(C)`` on a given batch, so an unzeroed assertion is a flaky
    test rather than a check. ``eval()`` matters too -- in train mode Dropout
    would zero activations and BatchNorm would use batch statistics, neither of
    which changes the answer here but both of which make the probe stochastic.
    """
    with sg.default_dtype(np.float64):
        probe = build_model(np.random.default_rng(0)).double()
        probe.eval()
        head = [m for m in probe.modules() if isinstance(m, nn.Linear)][-1]
        head.weight.data[...] = 0.0
        head.bias.data[...] = 0.0
        with sg.no_grad():
            loss = nn.functional.cross_entropy(probe(sg.Tensor(x, dtype=np.float64)), y)
    return loss.item()


def plot_samples(images: np.ndarray, labels: np.ndarray, path: Path, n: int = 16) -> None:
    """Render ``n`` images with their class names, straight from the loader.

    This is the channel-order check. Planar ``(N, 3, 32, 32)`` data misread as
    ``(N, 32, 32, 3)`` is exactly the right number of bytes and renders as three
    horizontally squashed colour bands -- obvious in a picture, invisible in a
    shape assertion that the wrong reshape also satisfies.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 8
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.35 * cols, 1.75 * rows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.set_axis_off()
        if i >= n:
            continue
        # (3, H, W) -> (H, W, 3) for imshow. This transpose is the *only* place
        # the channel axis moves; the network never sees anything but NCHW.
        ax.imshow(np.clip(images[i].transpose(1, 2, 0), 0.0, 1.0), interpolation="nearest")
        ax.set_title(CIFAR10_CLASSES[int(labels[i])], fontsize=8)
    fig.suptitle("CIFAR-10 as the loader returns it: (N, 3, 32, 32) planar")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    model: nn.Module, x: np.ndarray, y: np.ndarray, batch: int = 500
) -> tuple[float, float, np.ndarray]:
    """Loss, accuracy and predictions in **one** pass over ``(x, y)``.

    Returning the predictions matters for the test set: accuracy, loss and the
    confusion matrix all come out of a single forward pass, so "touched once"
    is a property of the code and not a promise.
    """
    model.eval()
    acc, meter, preds = Accuracy(), AverageMeter(), []
    with sg.no_grad():
        for i in range(0, len(x), batch):
            xb, yb = x[i : i + batch], y[i : i + batch]
            logits = model(sg.Tensor(xb))
            meter.update(nn.functional.cross_entropy(logits, yb).item(), len(xb))
            acc.update(logits.data, yb)
            preds.append(logits.data.argmax(1))
    model.train()
    # Plain floats: ``acc.value`` is a numpy scalar and json.dumps refuses one.
    return float(meter.avg), float(acc.value), np.concatenate(preds)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_splits(root: Path) -> tuple[np.ndarray, ...]:
    """CIFAR-10 as train / val / test, per-channel standardised.

    The mean and std come from the **45000 training images only**. Computing
    them over all 50000 (or over the test set) leaks held-out statistics into
    training -- a small leak, but the kind that makes an accuracy number
    unquotable.
    """
    train, test = cifar10(root=root, normalize=True)
    xtr_all, ytr_all = train.arrays
    xte, yte = test.arrays
    assert xtr_all.shape[1:] == (3, 32, 32), f"expected planar NCHW, got {xtr_all.shape}"
    assert xte.shape[1:] == (3, 32, 32), f"expected planar NCHW, got {xte.shape}"
    assert xtr_all.min() >= 0.0 and xtr_all.max() <= 1.0, "expected x/255 normalisation"

    n_train = len(xtr_all) - VAL_SIZE
    xtr, ytr = xtr_all[:n_train], ytr_all[:n_train]
    xva, yva = xtr_all[n_train:], ytr_all[n_train:]

    raw_preview = xtr[:16].copy()  # keep [0, 1] pixels for the sample plot
    mean = xtr.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32).reshape(1, 3, 1, 1)
    std = xtr.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32).reshape(1, 3, 1, 1)
    for arr in (xtr, xva, xte):
        arr -= mean  # in place: a copy of the 45000-image tensor is 553 MB
        arr /= std
    return xtr, ytr, xva, yva, xte, yte, raw_preview, mean.ravel(), std.ravel()


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
def smoke(args) -> int:
    """Overfit 32 images before any real training.

    If the loss cannot reach ~0 on 32 images the model is broken, not
    undertrained, and finding that out in a minute beats finding it out after
    an epoch. Augmentation is off here on purpose: a moving target cannot be
    memorised, so a passing overfit test would prove nothing.
    """
    rng = np.random.default_rng(args.seed)
    xtr, ytr, *_ = load_splits(ROOT / args.data)
    x, y = xtr[:32], ytr[:32]
    print(f"loss at init with the head zeroed: {loss_at_init_is_ln_c(x, y):.15f}")
    print(f"                           ln(10): {math.log(10):.15f}")

    model = build_model(rng, dropout=0.0)
    opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for step in range(120):
        loss = nn.functional.cross_entropy(model(sg.Tensor(x)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == 119:
            print(f"  step {step:4d}  loss {loss.item():.6f}")
    final = loss.item()
    print(f"\noverfit 32 images -> loss {final:.2e}", "OK" if final < 1e-3 else "FAILED")
    return 0 if final < 1e-3 else 1


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true", help="ablate crop+flip")
    ap.add_argument("--smoke", action="store_true", help="overfit 32 images and stop")
    ap.add_argument(
        "--time-steps",
        type=int,
        default=0,
        help="time N training steps, print the per-epoch cost, and stop",
    )
    ap.add_argument("--data", type=str, default="data/cifar10")
    ap.add_argument("--out", type=str, default="runs/cifar10_cnn")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        return smoke(args)

    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)

    print("loading CIFAR-10 ...")
    xtr, ytr, xva, yva, xte, yte, preview, mean, std = load_splits(ROOT / args.data)
    print(f"  train {xtr.shape}  val {xva.shape}  test {xte.shape}  dtype {xtr.dtype}")
    print(f"  per-channel mean {np.round(mean, 4)}  std {np.round(std, 4)}  (train split only)")
    print("  the test set will be evaluated exactly once, after the last epoch")

    if not args.no_plots:
        plot_samples(preview, ytr[:16], assets / "cifar10_samples.png")
        print(f"  wrote {assets / 'cifar10_samples.png'} -- planar decode is visible there")

    # Sanity check, before anything is trained.
    l0 = loss_at_init_is_ln_c(xtr[:256], ytr[:256])
    print(f"\nloss at init (head zeroed) = {l0:.15f}")
    print(f"                    ln(10) = {math.log(10):.15f}")
    assert abs(l0 - math.log(10)) < 1e-12, "uniform logits must give exactly ln(10)"
    print("  -> exact to 1e-12. The loss and the softmax agree.")

    model = build_model(rng, dropout=args.dropout)
    print(f"\n{model.summary()}")

    n = len(xtr)
    steps_per_epoch = -(-n // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Warmup for half an epoch: BatchNorm's running statistics start at (0, 1)
    # and are meaningless for the first few hundred batches, so a full-rate step
    # that early is taken against noise.
    sched = optim.WarmupCosine(
        opt, warmup_steps=steps_per_epoch // 2, total_steps=total_steps, min_lr_ratio=0.05
    )

    if args.time_steps:
        xb = augment(xtr[: args.batch_size], rng)
        yb = ytr[: args.batch_size]
        times = []
        for _ in range(args.time_steps):
            t = time.perf_counter()
            loss = nn.functional.cross_entropy(model(sg.Tensor(xb)), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            times.append(time.perf_counter() - t)
        med = float(np.median(times))
        print(f"\nmedian step {med:.3f}s over {args.time_steps} steps, batch {args.batch_size}")
        print(f"  {steps_per_epoch} steps/epoch -> {med * steps_per_epoch / 60:.1f} min/epoch")
        print(f"  {args.epochs} epochs -> {med * total_steps / 60:.1f} min of training")
        return 0

    print(
        f"\ntraining {args.epochs} epochs, batch {args.batch_size}, AdamW lr={args.lr} "
        f"wd={args.weight_decay}, WarmupCosine({steps_per_epoch // 2}, {total_steps})"
    )
    print(f"augmentation: {'off' if args.no_augment else 'random crop (4px reflect) + h-flip'}")
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "seconds": [],
    }
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.perf_counter()
        meter = AverageMeter()
        # ONE permutation indexed into both arrays. Two independent shuffles
        # decouple x from y; the loss then parks at ln(10) and nothing else in
        # the run looks wrong.
        perm = rng.permutation(n)
        bar = ProgressBar(total=steps_per_epoch, desc=f"epoch {epoch}/{args.epochs}")
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            xb, yb = xtr[idx], ytr[idx]
            if not args.no_augment:
                xb = augment(xb, rng)
            loss = nn.functional.cross_entropy(model(sg.Tensor(xb)), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            meter.update(loss.item(), len(xb))
            bar.update()
            bar.set_postfix(loss=f"{meter.avg:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
        bar.close()
        val_loss, val_acc, _ = evaluate(model, xva, yva)
        history["epoch"].append(epoch)
        history["train_loss"].append(float(meter.avg))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        history["seconds"].append(time.perf_counter() - t_epoch)
        print(
            f"  epoch {epoch}: train {meter.avg:.4f}  val {val_loss:.4f}  "
            f"val acc {val_acc * 100:.2f}%  ({history['seconds'][-1] / 60:.1f} min)"
        )
    elapsed = time.perf_counter() - t_start

    # ---- the one and only look at the test set ----------------------------- #
    test_loss, test_acc, test_pred = evaluate(model, xte, yte)
    train_loss, train_acc, _ = evaluate(model, xtr, ytr)

    print(f"\ntrained in {elapsed / 60:.1f} min on CPU ({elapsed:.0f}s)")
    print(
        f"final TEST accuracy : {test_acc * 100:.2f}%   "
        f"({round(test_acc * len(yte))}/{len(yte)})   [test set touched once]"
    )
    print(f"final val accuracy  : {history['val_acc'][-1] * 100:.2f}%")
    print(f"final train accuracy: {train_acc * 100:.2f}%")
    print(f"parameters          : {model.num_parameters():,}")

    save_checkpoint(
        out / "weights.npz",
        model,
        opt,
        meta={
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "augment": not args.no_augment,
            "test_accuracy": test_acc,
            "train_seconds": elapsed,
        },
    )
    (out / "history.json").write_text(
        json.dumps(
            {
                "history": history,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "val_accuracy": history["val_acc"][-1],
                "train_accuracy": train_acc,
                "train_loss_final": train_loss,
                "seconds": elapsed,
                "parameters": model.num_parameters(),
                "loss_at_init": l0,
                "channel_mean": mean.tolist(),
                "channel_std": std.tolist(),
                "config": {
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "augment": not args.no_augment,
                    "val_size": VAL_SIZE,
                    "warmup_steps": steps_per_epoch // 2,
                    "total_steps": total_steps,
                },
                "command": f"PYTHONPATH=src python examples/cifar10_cnn.py --epochs {args.epochs} "
                f"--batch-size {args.batch_size} --lr {args.lr} --seed {args.seed}",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {out / 'weights.npz'} and history.json")

    if not args.no_plots:
        try:
            from scratchgrad.utils import viz

            first = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
            viz.plot_filters_conv(
                first.weight.data,
                path=assets / "cifar10_filters.png",
                max_filters=32,
                title="first-layer 3x3 filters (rows = out channel, cols = RGB)",
            )
            viz.plot_history(
                history, path=assets / "cifar10_training.png", title="CIFAR-10 VGG-ish CNN"
            )
            # Reuses test_pred from the single evaluation pass above -- no
            # second look at the test set.
            cm = viz.confusion_matrix(yte, test_pred, 10)
            viz.plot_confusion(
                cm,
                path=assets / "cifar10_confusion.png",
                class_names=CIFAR10_CLASSES,
                normalize=True,
            )
            print(f"wrote plots to {assets}")
            worst = sorted(((int(cm[i].sum() - cm[i, i]), i) for i in range(10)), reverse=True)
            print(
                "most-confused classes:",
                ", ".join(f"{CIFAR10_CLASSES[c]} ({k} wrong)" for k, c in worst[:3]),
            )
            pairs = sorted(
                ((int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j),
                reverse=True,
            )
            print(
                "worst pairs:",
                ", ".join(
                    f"{CIFAR10_CLASSES[i]}->{CIFAR10_CLASSES[j]} ({k})" for k, i, j in pairs[:3]
                ),
            )
        except ImportError as exc:
            print(f"skipping plots: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
