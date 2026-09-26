"""A variational quantum classifier, and barren plateaus -- on a classical simulator.

**This is a classical simulation.** Every amplitude is a complex128 in a NumPy
array, and the cost is ``O(2**n)`` per gate. Nothing here demonstrates, or
claims, a quantum advantage; the point is the mechanics -- a circuit as a
differentiable layer whose gradient comes from the parameter-shift rule, trained
by scratchgrad's own Adam -- and one honest measurement of why deep random
circuits are hard to train.

    PYTHONPATH=src python examples/quantum_classifier.py                 # both experiments
    PYTHONPATH=src python examples/quantum_classifier.py --skip-barren   # classifier only

Part 1 trains a 2-qubit **data re-uploading** classifier (Perez-Salinas et al.
2020) on seeded two moons: each layer encodes ``(x0, x1)`` as ``RY`` angles, then
applies trainable ``RY RZ`` on each wire and a ``CZ``. Re-uploading is what lets
a two-qubit model draw a curved boundary at all -- with a single encoding layer
``<Z>`` is a degree-1 trigonometric polynomial in each feature, and every extra
upload raises that degree by one.

Part 2 measures ``Var[dE/dtheta_1]`` for ``E = <Z_0 Z_1>`` over random deep
hardware-efficient circuits (McClean et al. 2018) as the width grows. The
gradient is computed per circuit by the same parameter-shift code, and the
random circuits are simply a batch: every angle except the differentiated one
is an "input" feature, so hundreds of circuits run as one stacked simulation.
Unlike McClean, the rotation axes are fixed (``RY`` then ``RZ``) rather than
drawn at random per gate. Depth grows with width (``5n`` layers by default)
because a 1-D circuit needs depth ~ ``n`` before it scrambles like a 2-design;
at a *fixed* depth the widest circuits are under-scrambled and their variance
decays more slowly than the ``~2**-n`` a 2-design predicts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import scratchgrad as sg
from scratchgrad import optim
from scratchgrad.nn.module import Parameter
from scratchgrad.quantum import Circuit, Op


# --------------------------------------------------------------------------- #
# part 1: classifier
# --------------------------------------------------------------------------- #
def two_moons(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved half circles, labels in {+1, -1}, shuffled with ONE permutation."""
    n1 = n // 2
    t1, t2 = rng.uniform(0, np.pi, n1), rng.uniform(0, np.pi, n - n1)
    upper = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    lower = np.stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)
    x = np.concatenate([upper, lower]) + noise * rng.standard_normal((n, 2))
    y = np.concatenate([np.ones(n1), -np.ones(n - n1)])
    perm = rng.permutation(n)
    return x[perm], y[perm]


def reuploading_circuit(n_layers: int) -> Circuit:
    """2 qubits; per layer: RY(x_q), then trainable RY RZ on each wire, then CZ."""
    ops, p = [], 0
    for _ in range(n_layers):
        for q in range(2):
            ops.append(Op("RY", q, feature=q))
        for q in range(2):
            ops += [Op("RY", q, param=p), Op("RZ", q, param=p + 1)]
            p += 2
        ops.append(Op("CZ", (0, 1)))
    return Circuit(2, ops, observable=(0,))


def train_classifier(args, rng: np.random.Generator) -> dict:
    xtr, ytr = two_moons(args.n_train, args.noise, rng)
    xte, yte = two_moons(args.n_test, args.noise, rng)
    # Standardise on the TRAIN statistics only, then map to angles.
    mu, sd = xtr.mean(0), xtr.std(0)
    to_angle = lambda x: args.angle_scale * (x - mu) / sd  # noqa: E731

    circ = reuploading_circuit(args.layers)
    print(f"\n[classifier] {circ}")
    theta = Parameter(rng.uniform(0, 2 * np.pi, circ.n_params))
    opt = optim.Adam([theta], lr=args.lr)
    atr, ate = to_angle(xtr), to_angle(xte)
    target = sg.Tensor(ytr)

    hist: dict[str, list[float]] = {"step": [], "loss": [], "train_acc": [], "test_acc": []}
    t0 = time.perf_counter()
    for step in range(args.steps):
        out = circ.expval(theta, atr)
        loss = ((out - target) ** 2).mean()  # square loss on <Z> in [-1, 1]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            tr_acc = float(np.mean(np.sign(out.data) == ytr))
            te_acc = float(np.mean(np.sign(circ.forward(theta.data, ate)) == yte))
            for k, v in zip(hist, (step, loss.item(), tr_acc, te_acc), strict=True):
                hist[k].append(v)
            print(
                f"  step {step:4d}  loss {loss.item():.4f}  train {tr_acc:.3f}  test {te_acc:.3f}"
            )
    seconds = time.perf_counter() - t0

    train_acc = float(np.mean(np.sign(circ.forward(theta.data, atr)) == ytr))
    test_acc = float(np.mean(np.sign(circ.forward(theta.data, ate)) == yte))
    print(f"  trained in {seconds:.1f}s: train acc {train_acc:.3f}, TEST ACC {test_acc:.3f}")

    if not args.no_plots:
        plot_boundary(circ, theta.data, to_angle, xtr, ytr, xte, yte, test_acc)
    return {
        "history": hist,
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "seconds": seconds,
        "n_params": circ.n_params,
        "circuit": repr(circ),
        "config": {k: v for k, v in vars(args).items() if k not in ("out",)},
    }


def plot_boundary(circ, theta, to_angle, xtr, ytr, xte, yte, test_acc) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = np.minimum(xtr.min(0), xte.min(0)) - 0.5, np.maximum(xtr.max(0), xte.max(0)) + 0.5
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 120), np.linspace(lo[1], hi[1], 120))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    z = circ.forward(theta, to_angle(grid)).reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # RdBu_r: <Z> = +1 is red, matching the red class-+1 markers.
    cf = ax.contourf(gx, gy, z, levels=np.linspace(-1, 1, 21), cmap="RdBu_r", alpha=0.75)
    ax.contour(gx, gy, z, levels=[0.0], colors="k", linewidths=1.2)
    for x, y, marker, lab in ((xtr, ytr, "o", "train"), (xte, yte, "^", "test")):
        ax.scatter(
            *x[y > 0].T,
            c="#b2182b",
            marker=marker,
            s=14,
            edgecolors="k",
            linewidths=0.3,
            label=f"{lab} +1",
        )
        ax.scatter(
            *x[y < 0].T,
            c="#2166ac",
            marker=marker,
            s=14,
            edgecolors="k",
            linewidths=0.3,
            label=f"{lab} -1",
        )
    fig.colorbar(cf, ax=ax, label=r"$\langle Z_0 \rangle$")
    ax.set_title(
        f"2-qubit re-uploading classifier, classical sim -- test acc {test_acc:.1%}", fontsize=10
    )
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    path = ROOT / "assets" / "quantum_classifier.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# part 2: barren plateaus
# --------------------------------------------------------------------------- #
def hardware_efficient(n_qubits: int, n_layers: int) -> Circuit:
    """Layers of RY RZ on every wire then a CZ ladder; measure Z0 Z1.

    Every angle is an *input feature* (one random draw per circuit instance)
    except the first gate, which also carries the single parameter ``theta_1``
    that is differentiated. With ``theta_1 = 0`` its angle is just its feature.
    """
    ops, f = [], 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            first = layer == 0 and q == 0
            ops += [Op("RY", q, param=0 if first else None, feature=f), Op("RZ", q, feature=f + 1)]
            f += 2
        ops += [Op("CZ", (q, q + 1)) for q in range(n_qubits - 1)]
    return Circuit(n_qubits, ops, observable=(0, 1))


def barren_plateau(args, rng: np.random.Generator) -> dict:
    qubits = list(range(2, args.max_qubits + 1))
    var, mean, secs = [], [], []
    print(f"\n[barren plateau] {args.circuits} random circuits per width, {args.bp_depth}n layers")
    for n in qubits:
        circ = hardware_efficient(n, args.bp_depth * n)
        angles = rng.uniform(0, 2 * np.pi, size=(args.circuits, circ.n_features))
        t0 = time.perf_counter()
        grad = circ.jacobian(np.zeros(1), angles)[:, 0]
        secs.append(time.perf_counter() - t0)
        var.append(float(grad.var(ddof=1)))
        mean.append(float(grad.mean()))
        print(
            f"  n={n:2d}  Var[dE/dtheta_1] = {var[-1]:.3e}  mean = {mean[-1]:+.1e}  "
            f"({secs[-1]:.1f}s)"
        )
    slope, intercept = np.polyfit(qubits, np.log2(var), 1)
    print(
        f"  fit: log2 Var = {intercept:.2f} {slope:+.3f} n   "
        f"(variance shrinks x{2**-slope:.2f} per added qubit; a 2-design predicts ~x2)"
    )

    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        # Standard error of a sample variance, assuming roughly Gaussian gradients.
        err = np.array(var) * np.sqrt(2.0 / (args.circuits - 1))
        ax.errorbar(qubits, var, yerr=err, fmt="o", color="#2166ac", capsize=3, label="measured")
        ns = np.array(qubits, dtype=float)
        ax.plot(
            ns,
            2.0 ** (intercept + slope * ns),
            "--",
            color="#b2182b",
            label=f"fit: slope {slope:.2f} in log$_2$ (x{2**-slope:.2f} per qubit)",
        )
        ax.set_yscale("log")
        ax.set_xlabel("qubits n")
        ax.set_ylabel(r"Var$[\partial \langle Z_0 Z_1\rangle / \partial\theta_1]$")
        ax.set_title(f"Barren plateau: random {args.bp_depth}n-layer circuits (classical sim)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        path = ROOT / "assets" / "quantum_barren_plateau.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  wrote {path.relative_to(ROOT)}")
    return {
        "qubits": qubits,
        "variance": var,
        "mean": mean,
        "seconds": secs,
        "log2_slope": float(slope),
        "log2_intercept": float(intercept),
        "shrink_per_qubit": float(2**-slope),
        "layers": [args.bp_depth * n for n in qubits],
        "circuits": args.circuits,
        "observable": "Z0 Z1",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4, help="re-uploading layers")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--angle-scale", type=float, default=1.0)
    ap.add_argument("--max-qubits", type=int, default=10)
    ap.add_argument("--bp-depth", type=int, default=5, help="barren-plateau layers per qubit")
    ap.add_argument("--circuits", type=int, default=300, help="random circuits per width")
    ap.add_argument("--skip-barren", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out", type=str, default="runs/quantum_classifier")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "assets").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with sg.default_dtype(np.float64):
        result = train_classifier(args, rng)
    if not args.skip_barren:
        result["barren_plateau"] = barren_plateau(args, np.random.default_rng(args.seed + 1))
    result["seed"] = args.seed
    result["note"] = "classical statevector simulation; no quantum-advantage claim"
    (out / "history.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {(out / 'history.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
