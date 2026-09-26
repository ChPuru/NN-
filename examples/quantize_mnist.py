"""Quantize the MNIST MLP: int8 PTQ with integer-only inference, then QAT down to 1 bit.

    PYTHONPATH=src python examples/quantize_mnist.py              # everything, ~2.5 min CPU
    PYTHONPATH=src python examples/quantize_mnist.py --qat-epochs 1

Protocol, fixed before any number was looked at:

* **fp32 baseline** -- the exact recipe, seed and RNG call order of
  ``examples/mnist_mlp.py`` (784 -> 128 ReLU -> 10, Adam 1e-3, batch 128, 5
  epochs), so the weights are bit-identical to that run. No test evaluation
  during training.
* **Calibration** -- 2048 training images drawn by a separately seeded
  permutation; running min/max of the input, the post-ReLU hidden activation and
  the logits.
* **Configurations** -- ``W8A8, W4A4, W2A2`` (k-bit per-output-channel symmetric
  weights in *both* layers, k-bit hidden activation) and ``W1A8`` (binary
  ``+-alpha`` weights, 8-bit activation). The input stays 8-bit -- it *is* the
  pixel bytes -- and the logits are requantized to 8 bits so argmax has
  resolution. Biases are int32 on the accumulator grid.
* **PTQ** -- calibrate, convert, done: no training.
* **QAT** -- start from the calibrated PTQ model, fine-tune ``--qat-epochs``
  epochs (Adam ``--qat-lr``) through fake-quant with the clipped STE, observers
  switched to an EMA (momentum 0.01), then convert. QAT therefore sees extra
  training epochs that PTQ does not; that is what QAT *is*, but it is stated.
* **Every reported accuracy is the integer-only model** (int8 x int8 -> int32 ->
  fixed-point requantize), and the test set is touched exactly once per reported
  number: 1 fp32 + 4 PTQ + 4 QAT = 9 passes. Agreement between the float
  fake-quant simulation and the integer model is measured on the calibration
  images, never on test.
* **One BLAS thread** (``OPENBLAS_NUM_THREADS=1`` unless already set). OpenBLAS
  splits a float32 GEMM differently for different thread counts, which changes
  the summation order, and after 17 epochs the accuracies differ in the second
  decimal (fp32 97.58% multi-threaded vs 97.56% single-threaded here). Pinning
  one thread makes the numbers independent of the machine's core count.
* **Bytes** are the deployed size: weights bit-packed at k bits, int32 bias,
  int32 multiplier and int8 shift per output channel. Below 8 bits the codes sit
  in int8 containers in memory; the packed size is what a packer would write.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Before NumPy loads OpenBLAS: see "One BLAS thread" above.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import scratchgrad as sg
from scratchgrad import nn, optim
from scratchgrad.data import DataLoader, mnist
from scratchgrad.quant import (
    FakeQuantize,
    MinMaxObserver,
    QATLinear,
    convert,
    set_quant_state,
)
from scratchgrad.utils import AverageMeter

CONFIGS = [(8, 8), (4, 4), (2, 2), (1, 8)]  # (weight bits, hidden-activation bits)


def build_model(rng: np.random.Generator) -> nn.Module:
    """Identical to ``examples/mnist_mlp.py`` -- same shapes, same RNG draws."""
    return nn.Sequential(nn.Linear(784, 128, rng=rng), nn.ReLU(), nn.Linear(128, 10, rng=rng))


def train_epochs(model, loader, opt, epochs: int, label: str) -> list[float]:
    losses = []
    for epoch in range(1, epochs + 1):
        meter = AverageMeter()
        for xb, yb in loader:
            loss = nn.functional.cross_entropy(model(sg.Tensor(xb)), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            meter.update(loss.item(), len(xb))
        losses.append(meter.avg)
        print(f"    {label} epoch {epoch}/{epochs}: train loss {meter.avg:.4f}")
    return losses


def float_predict(model, x: np.ndarray, batch: int = 2000) -> np.ndarray:
    was = model.training
    model.eval()
    with sg.no_grad():
        pred = np.concatenate(
            [model(sg.Tensor(x[i : i + batch])).data.argmax(1) for i in range(0, len(x), batch)]
        )
    model.train(was)
    return pred


def build_sim(fp32: nn.Module, w_bits: int, a_bits: int) -> nn.Module:
    """The fake-quant twin of the fp32 model; order is what ``convert`` expects."""
    l1, l2 = (m for m in fp32.modules() if isinstance(m, nn.Linear))
    return nn.Sequential(
        FakeQuantize(8),  # input: recovers the pixel bytes exactly
        QATLinear.from_linear(l1, w_bits),
        nn.ReLU(),
        FakeQuantize(a_bits),  # hidden, calibrated AFTER the ReLU so z = 0
        QATLinear.from_linear(l2, w_bits),
        FakeQuantize(8),  # logits
    )


def calibrate(sim: nn.Module, x_cal: np.ndarray, batch: int = 512) -> None:
    """Measure ranges on the float model, then freeze them and switch quantization on."""
    sim.train()
    set_quant_state(sim, observe=True, fake=False)
    with sg.no_grad():
        for i in range(0, len(x_cal), batch):
            sim(sg.Tensor(x_cal[i : i + batch]))
    set_quant_state(sim, observe=False, fake=True)


def plot(results: list[dict], fp32_acc: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # WkAk is one family and gets lines; W1A8 keeps 8-bit activations, so it is
    # drawn as separate markers rather than joined to W2A2 as if on one curve.
    wk = [r for r in results if r["w_bits"] == r["a_bits"]]
    other = [r for r in results if r["w_bits"] != r["a_bits"]]
    styles = (("ptq_acc", "PTQ", "o", "C0", -13), ("qat_acc", "QAT", "s", "C1", 6))
    for key, label, marker, color, dy in styles:
        ax.plot(
            [r["w_bits"] for r in wk],
            [100 * r[key] for r in wk],
            marker=marker,
            color=color,
            label=f"{label} WkAk (integer-only)",
        )
        ax.plot(
            [r["w_bits"] for r in other],
            [100 * r[key] for r in other],
            ls="none",
            marker=marker,
            mfc="none",
            color=color,
        )
        for r in results:
            a = 100 * r[key]
            ax.annotate(
                f"{a:.1f}",
                (r["w_bits"], a),
                textcoords="offset points",
                xytext=(5, dy),
                fontsize=8,
                color=color,
            )
    ax.axhline(100 * fp32_acc, color="gray", ls="--", lw=1, label=f"fp32 {100 * fp32_acc:.2f}%")
    ax.set_xscale("log", base=2)
    ax.set_xticks([r["w_bits"] for r in results], [r["name"] for r in results])
    ax.set_xlim(0.8, 11)
    ax.set_xlabel("weight bits / hidden-activation bits  (hollow: binary W, 8-bit A)")
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_ylim(0, 104)
    ax.set_title("MNIST MLP: post-training quantization vs QAT")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5, help="fp32 epochs (mnist_mlp recipe)")
    ap.add_argument("--qat-epochs", type=int, default=3)
    ap.add_argument("--qat-lr", type=float, default=5e-4)
    ap.add_argument("--calib", type=int, default=2048, help="training images used to calibrate")
    ap.add_argument("--out", type=str, default="runs/quantize_mnist")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    t0, cpu0 = time.perf_counter(), time.process_time()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    train, test = mnist(flatten=True, normalize=True)
    xtr = train.arrays[0]
    xte, yte = test.arrays

    # fp32 -- same rng call order as mnist_mlp.py: build_model, then the loader.
    rng = np.random.default_rng(args.seed)
    model = build_model(rng)
    loader = DataLoader(train, batch_size=128, shuffle=True, rng=rng)
    print(f"fp32: {args.epochs} epochs, Adam 1e-3, batch 128")
    t = time.perf_counter()
    train_epochs(model, loader, optim.Adam(model.parameters(), lr=1e-3), args.epochs, "fp32")
    fp32_seconds = time.perf_counter() - t
    fp32_acc = float(np.mean(float_predict(model, xte) == yte))  # test pass 1
    fp32_bytes = 4 * model.num_parameters()
    print(f"  fp32 test accuracy {100 * fp32_acc:.2f}%   {fp32_bytes:,} bytes\n")

    x_cal = xtr[np.random.default_rng(args.seed + 1).permutation(len(xtr))[: args.calib]]

    results = []
    for w_bits, a_bits in CONFIGS:
        name = f"W{w_bits}A{a_bits}"
        print(f"{name}")
        sim = build_sim(model, w_bits, a_bits)
        calibrate(sim, x_cal)
        ptq = convert(sim)
        ptq_acc = float(np.mean(ptq.predict(xte) == yte))  # one test pass
        ptq_agree = float(np.mean(ptq.predict(x_cal) == float_predict(sim, x_cal)))
        print(f"  PTQ  {100 * ptq_acc:6.2f}%   sim/int agreement on calib {100 * ptq_agree:.2f}%")

        for m in sim.modules():
            if isinstance(m, MinMaxObserver):
                m.momentum = 0.01
        set_quant_state(sim, observe=True, fake=True)
        qrng = np.random.default_rng([args.seed, w_bits, a_bits])
        qloader = DataLoader(train, batch_size=128, shuffle=True, rng=qrng)
        t = time.perf_counter()
        opt = optim.Adam(sim.parameters(), lr=args.qat_lr)
        losses = train_epochs(sim, qloader, opt, args.qat_epochs, f"QAT {name}")
        qat_seconds = time.perf_counter() - t
        set_quant_state(sim, observe=False)
        qat = convert(sim)
        qat_acc = float(np.mean(qat.predict(xte) == yte))  # one test pass
        qat_agree = float(np.mean(qat.predict(x_cal) == float_predict(sim, x_cal)))
        nbytes = qat.packed_nbytes()
        print(
            f"  QAT  {100 * qat_acc:6.2f}%   sim/int agreement on calib {100 * qat_agree:.2f}%"
            f"   {nbytes:,} bytes ({fp32_bytes / nbytes:.1f}x smaller)\n"
        )
        results.append(
            {
                "name": name,
                "w_bits": w_bits,
                "a_bits": a_bits,
                "ptq_acc": ptq_acc,
                "qat_acc": qat_acc,
                "bytes": nbytes,
                "ptq_sim_int_agreement": ptq_agree,
                "qat_sim_int_agreement": qat_agree,
                "qat_train_loss": losses,
                "qat_seconds": qat_seconds,
            }
        )

    total, cpu = time.perf_counter() - t0, time.process_time() - cpu0
    int8 = results[0]
    print(f"fp32 {100 * fp32_acc:.2f}% {fp32_bytes:,} B")
    print(f"int8 PTQ {100 * int8['ptq_acc']:.2f}% {int8['bytes']:,} B (integer-only)")
    print(f"total {total:.1f}s wall, {cpu:.1f}s process CPU")

    (out / "history.json").write_text(
        json.dumps(
            {
                "fp32": {"acc": fp32_acc, "bytes": fp32_bytes, "seconds": fp32_seconds},
                "configs": results,
                "settings": {
                    "seed": args.seed,
                    "fp32_epochs": args.epochs,
                    "qat_epochs": args.qat_epochs,
                    "qat_lr": args.qat_lr,
                    "calib_images": args.calib,
                    "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                    "observer_momentum_qat": 0.01,
                    "test_passes": 1 + 2 * len(CONFIGS),
                },
                "seconds": total,
                "cpu_seconds": cpu,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out / 'history.json'}")
    if not args.no_plots:
        (ROOT / "assets").mkdir(exist_ok=True)
        plot(results, fp32_acc, ROOT / "assets" / "quantize_bits.png")
        print(f"wrote {ROOT / 'assets' / 'quantize_bits.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
