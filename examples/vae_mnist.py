"""A VAE on binarised MNIST, trained through the autograd engine.

Likelihood: **dynamically binarised** MNIST. Every time a training image is
drawn, each pixel is resampled as ``x_i ~ Bernoulli(intensity_i)`` from the
script's seeded generator; validation and test sets are binarised once, with
fixed seeds. The decoder is a Bernoulli per pixel, so the loss is a genuine
discrete negative log-likelihood bound in nats and is comparable with the
binarised-MNIST literature. (Feeding the raw intensities to BCE as "soft"
targets also trains and gives nice pictures, but its number is not the log
likelihood of anything -- the Bernoulli pmf does not normalise over ``[0, 1]``.)

    PYTHONPATH=src python examples/vae_mnist.py                     # full run
    PYTHONPATH=src python examples/vae_mnist.py --epochs 1          # quick look
    PYTHONPATH=src python examples/vae_mnist.py --recon-reduction mean --epochs 5 \
        --out runs/vae_mnist/trap_mean                              # the trap

``--recon-reduction mean`` reproduces the reduction trap on purpose: BCE
averaged over pixels as well as images, KL left alone, i.e. beta = 784. Its
validation -ELBO is still computed with the correct sum reduction, so the two
runs are comparable; it never touches the test set. Measured at epoch 5 (seed
0): the trap sits at 208.6 nats with 0/20 active latent units and a KL of 0.00
-- total posterior collapse, every reconstruction the same grey average digit
-- against 108.4 nats and 20/20 for the correct reduction.

Protocol: 50k train / 10k validation carved from the training set; the test
set is touched exactly once, after training, for the reported negative ELBO.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# OpenBLAS reads this once, at load, so it must precede `import numpy`. These
# matmuls are small (128 x 784 x 400); on the 20-thread dev machine the default
# thread count measured ~5x slower per step than 4 threads. An explicit
# environment setting still wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import scratchgrad as sg
from scratchgrad import optim
from scratchgrad.data import mnist
from scratchgrad.nn import functional as F
from scratchgrad.nn.vae import VAE, gaussian_kl
from scratchgrad.utils import AverageMeter, ProgressBar


def binarise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(x.shape, dtype=np.float32) < x).astype(np.float32)


def evaluate(model: VAE, x: np.ndarray, seed: int, batch: int = 1000) -> dict[str, float]:
    """Single-sample negative ELBO in nats/image, plus active latent units.

    A unit is *active* when ``Var_x(E[z_j | x]) > 0.01`` (Burda et al. 2016): the
    encoder mean actually moves with the input. A collapsed unit has
    ``q(z_j|x) = N(0, 1)`` for every ``x``, carries no information, and pays no KL.
    """
    rng = np.random.default_rng(seed)
    tot = rec = kl = 0.0
    mus = []
    with sg.no_grad():
        for i in range(0, len(x), batch):
            xb = sg.Tensor(x[i : i + batch])
            t, r, k = model.loss(xb, rng=rng)
            n = len(xb.data)
            tot, rec, kl = tot + t.item() * n, rec + r.item() * n, kl + k.item() * n
            mus.append(model.encode(xb)[0].data)
    active = int((np.concatenate(mus).var(axis=0) > 0.01).sum())
    n = len(x)
    return {"nelbo": tot / n, "recon": rec / n, "kl": kl / n, "active_units": active}


def grid(tiles: np.ndarray, cols: int, gap: int = 2) -> np.ndarray:
    n = len(tiles)
    rows = -(-n // cols)
    out = np.ones((rows * (28 + gap) + gap, cols * (28 + gap) + gap))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[gap + r * 30 : gap + r * 30 + 28, gap + c * 30 : gap + c * 30 + 28] = t.reshape(28, 28)
    return out


def save_image(img: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(img.shape[1] / 50, img.shape[0] / 50 + 0.4))
    ax.imshow(img, cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recon-reduction", choices=["sum", "mean"], default="sum")
    ap.add_argument("--out", type=str, default="runs/vae_mnist")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    train, test = mnist(flatten=True, normalize=True)
    x_all, x_test = train.arrays[0], test.arrays[0]
    perm = np.random.default_rng(12345).permutation(len(x_all))  # fixed split, any --seed
    x_tr, x_val = x_all[perm[:50_000]], binarise(x_all[perm[50_000:]], np.random.default_rng(1))
    x_te = binarise(x_test, np.random.default_rng(2))
    print(f"train {x_tr.shape} (rebinarised per batch)  val {x_val.shape}  test {x_te.shape}")

    model = VAE(784, 400, args.latent, rng=rng)
    print(model)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    history: dict[str, list[float]] = {
        k: [] for k in ("epoch", "loss", "recon", "kl", "val_nelbo", "val_kl", "val_active")
    }

    steps = -(-len(x_tr) // args.batch_size)
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_tr))
        m_tot, m_rec, m_kl = AverageMeter(), AverageMeter(), AverageMeter()
        bar = ProgressBar(total=steps, desc=f"epoch {epoch}/{args.epochs}")
        for i in range(0, len(x_tr), args.batch_size):
            xb = binarise(x_tr[order[i : i + args.batch_size]], rng)
            n = len(xb)
            if args.recon_reduction == "sum":
                loss, recon, kl = model.loss(sg.Tensor(xb), rng=rng)
            else:  # the trap, on purpose: per-pixel mean recon + per-image KL
                logits, mu, logvar = model(sg.Tensor(xb), rng=rng)
                recon = F.binary_cross_entropy_with_logits(logits, xb, reduction="mean")
                kl = gaussian_kl(mu, logvar).sum() / float(n)
                loss = recon + kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            m_tot.update(loss.item(), n)
            m_rec.update(recon.item(), n)
            m_kl.update(kl.item(), n)
            bar.update()
            bar.set_postfix(nelbo=f"{m_tot.avg:.2f}")
        bar.close()
        val = evaluate(model, x_val, seed=100)
        for k, v in (
            ("epoch", epoch),
            ("loss", m_tot.avg),
            ("recon", m_rec.avg),
            ("kl", m_kl.avg),
            ("val_nelbo", val["nelbo"]),
            ("val_kl", val["kl"]),
            ("val_active", val["active_units"]),
        ):
            history[k].append(v)
        print(
            f"  epoch {epoch:2d}: train loss {m_tot.avg:7.2f}  val -ELBO {val['nelbo']:7.2f} "
            f"(recon {val['recon']:6.2f} + KL {val['kl']:5.2f})  active {val['active_units']}"
            f"/{args.latent}"
        )
    elapsed = time.perf_counter() - t0

    print(f"\ntrained {args.epochs} epochs in {elapsed:.1f}s on CPU")
    test_res = None
    if args.recon_reduction == "sum":  # the trap run is judged on validation only
        test_res = evaluate(model, x_te, seed=200)  # the one and only look at the test set
        print(
            f"test negative ELBO: {test_res['nelbo']:.2f} nats/image "
            f"(recon {test_res['recon']:.2f} + KL {test_res['kl']:.2f}), "
            f"active units {test_res['active_units']}/{args.latent}"
        )

    (out / "history.json").write_text(
        json.dumps(
            {
                "history": history,
                "test": test_res,
                "seconds": elapsed,
                "config": vars(args),
                "likelihood": "Bernoulli; dynamically binarised train, fixed-binarised val/test",
                "parameters": model.num_parameters(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {out / 'history.json'}")

    if not args.no_plots:
        assets = ROOT / "assets"
        assets.mkdir(exist_ok=True)
        tag = "vae" if args.recon_reduction == "sum" else "vae_trap"
        with sg.no_grad():
            z = np.random.default_rng(7).standard_normal((100, args.latent)).astype(np.float32)
            samples = model.decode(sg.Tensor(z)).sigmoid().data
            xv = x_val[:16]
            recon = model.decode(model.encode(sg.Tensor(xv))[0]).sigmoid().data  # z = mu
        save_image(grid(samples, 10), assets / f"{tag}_samples.png", "z ~ N(0, I), decoder means")
        pairs = np.concatenate([xv, recon])
        save_image(
            grid(pairs, 16), assets / f"{tag}_recon.png", "top: validation input; bottom: z = mu"
        )
        print(f"wrote {assets / f'{tag}_samples.png'} and {assets / f'{tag}_recon.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
