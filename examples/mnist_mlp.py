"""MNIST MLP through the autograd engine -- and the Tier-1 equivalence proof.

This trains the *same* architecture as ``phase0/`` (784 -> 128 ReLU -> 10, Adam)
but through the general framework instead of the hand-rolled per-layer backward.
Matching Phase 0's accuracy to within seed noise is a much stronger statement
than a fresh green test suite: it proves the general engine reproduces a special
case that was already verified by hand, independently.

    PYTHONPATH=src python examples/mnist_mlp.py                 # full run
    PYTHONPATH=src python examples/mnist_mlp.py --smoke         # 32 examples, must reach ~0
    PYTHONPATH=src python examples/mnist_mlp.py --epochs 5 --seed 0

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
from scratchgrad.data import DataLoader, mnist
from scratchgrad.utils import Accuracy, AverageMeter, ProgressBar, save_checkpoint


def build_model(rng: np.random.Generator) -> nn.Module:
    return nn.Sequential(
        nn.Linear(784, 128, rng=rng),
        nn.ReLU(),
        nn.Linear(128, 10, rng=rng),
    )


def loss_at_init_is_ln_c(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    """Zero the final layer so the logits are exactly uniform, then check ln(10).

    Zeroing matters. Without it the initial loss fluctuates and can legitimately
    land slightly *below* ``ln(C)`` on any given batch -- only a systematic
    deficit across seeds is evidence of a bug, so an unzeroed assertion is a
    flaky test rather than a check. With the final layer zeroed the logits are
    identically zero and the loss is exactly ``ln(C)``, to float64 precision.
    """
    with sg.default_dtype(np.float64):
        probe = build_model(np.random.default_rng(0)).double()
        last = [m for m in probe.modules() if isinstance(m, nn.Linear)][-1]
        last.weight.data[...] = 0.0
        last.bias.data[...] = 0.0
        with sg.no_grad():
            loss = nn.functional.cross_entropy(probe(sg.Tensor(x, dtype=np.float64)), y)
    return loss.item()


def evaluate(
    model: nn.Module, x: np.ndarray, y: np.ndarray, batch: int = 1000
) -> tuple[float, float]:
    model.eval()
    acc, meter = Accuracy(), AverageMeter()
    with sg.no_grad():
        for i in range(0, len(x), batch):
            xb, yb = x[i : i + batch], y[i : i + batch]
            logits = model(sg.Tensor(xb))
            meter.update(nn.functional.cross_entropy(logits, yb).item(), len(xb))
            acc.update(logits.data, yb)
    model.train()
    return meter.avg, acc.value


def smoke(args) -> int:
    """Overfit 32 examples before any real training.

    If the loss cannot reach ~0 on 32 examples, the model is broken, not
    undertrained -- and finding that out in four seconds beats finding it out
    after an epoch.
    """
    rng = np.random.default_rng(args.seed)
    train, _ = mnist(flatten=True, normalize=True)
    x, y = train.arrays[0][:32], train.arrays[1][:32]
    print(f"loss at init with the final layer zeroed: {loss_at_init_is_ln_c(None, x, y):.15f}")
    print(f"                                   ln(10): {math.log(10):.15f}")

    model = build_model(rng)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for step in range(400):
        loss = nn.functional.cross_entropy(model(sg.Tensor(x)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 399:
            print(f"  step {step:4d}  loss {loss.item():.6f}")
    final = loss.item()
    print(f"\noverfit 32 examples -> loss {final:.2e}", "OK" if final < 1e-3 else "FAILED")
    return 0 if final < 1e-3 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="overfit 32 examples and stop")
    ap.add_argument("--out", type=str, default="runs/mnist_mlp")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        return smoke(args)

    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print("loading MNIST ...")
    train, test = mnist(flatten=True, normalize=True)
    xtr, ytr = train.arrays
    xte, yte = test.arrays
    print(f"  train {xtr.shape}  test {xte.shape}  dtype {xtr.dtype}")
    assert xtr.max() <= 1.0 and xtr.min() >= 0.0, "expected x/255 normalisation"

    # Sanity check 1, before anything is trained.
    l0 = loss_at_init_is_ln_c(None, xtr[:256], ytr[:256])
    print(f"\nloss at init (final layer zeroed) = {l0:.15f}")
    print(f"                           ln(10) = {math.log(10):.15f}")
    assert abs(l0 - math.log(10)) < 1e-12, "uniform logits must give exactly ln(10)"
    print("  -> exact to 1e-12. The loss and the softmax agree.")

    model = build_model(rng)
    print(f"\n{model}\n")
    print(model.summary())

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, rng=rng)
    history: dict[str, list[float]] = {"loss": [], "val_loss": [], "val_acc": [], "epoch": []}

    print(f"\ntraining {args.epochs} epochs, batch {args.batch_size}, Adam lr={args.lr}")
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        meter = AverageMeter()
        bar = ProgressBar(total=len(loader), desc=f"epoch {epoch}/{args.epochs}")
        for xb, yb in loader:
            loss = nn.functional.cross_entropy(model(sg.Tensor(xb)), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            meter.update(loss.item(), len(xb))
            bar.update()
            bar.set_postfix(loss=f"{meter.avg:.4f}")
        bar.close()
        val_loss, val_acc = evaluate(model, xte, yte)
        history["epoch"].append(epoch)
        history["loss"].append(meter.avg)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"  epoch {epoch}: train {meter.avg:.4f}  test {val_loss:.4f}  acc {val_acc * 100:.2f}%"
        )
    elapsed = time.perf_counter() - t_start

    final_loss, final_acc = evaluate(model, xte, yte)
    _, train_acc = evaluate(model, xtr, ytr)
    print(f"\ntrained in {elapsed:.1f}s on CPU")
    print(
        f"final test accuracy : {final_acc * 100:.2f}%   ({int(final_acc * len(yte))}/{len(yte)})"
    )
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
            "test_accuracy": final_acc,
            "train_seconds": elapsed,
        },
    )
    (out / "history.json").write_text(
        json.dumps(
            {
                "history": history,
                "test_accuracy": final_acc,
                "train_accuracy": train_acc,
                "test_loss": final_loss,
                "seconds": elapsed,
                "parameters": model.num_parameters(),
                "seed": args.seed,
                "loss_at_init": l0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {out / 'weights.npz'} and history.json")

    if not args.no_plots:
        try:
            from scratchgrad.utils import viz

            assets = ROOT / "assets"
            assets.mkdir(exist_ok=True)
            first = next(m for m in model.modules() if isinstance(m, nn.Linear))
            viz.plot_weights(
                first.weight.data,
                shape=(28, 28),
                path=assets / "mnist_mlp_weights.png",
                title="learned first-layer weights (blue = negative, red = positive)",
            )
            viz.plot_history(history, path=assets / "mnist_mlp_training.png")
            with sg.no_grad():
                pred = np.concatenate(
                    [
                        model(sg.Tensor(xte[i : i + 1000])).data.argmax(1)
                        for i in range(0, len(xte), 1000)
                    ]
                )
            cm = viz.confusion_matrix(yte, pred, 10)
            viz.plot_confusion(cm, path=assets / "mnist_mlp_confusion.png")
            print(f"wrote plots to {assets}")
            worst = [(cm[i].sum() - cm[i, i], i) for i in range(10)]
            worst.sort(reverse=True)
            print("most-confused digits:", ", ".join(f"{d} ({n} wrong)" for n, d in worst[:3]))
        except ImportError as exc:
            print(f"skipping plots: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
