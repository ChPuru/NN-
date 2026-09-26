"""Honest benchmarks: scratchgrad vs PyTorch CPU on identical models.

The point of this file is not to win. It is to publish exactly where the project
loses and why, because "we are 8x slower than PyTorch CPU, here is the reason" is
a far more credible claim than silence.

Method
------
* Both sides run the **same shapes, same dtype, same batch size**, and the same
  number of threads. PyTorch is pinned with ``torch.set_num_threads`` and NumPy's
  OpenBLAS is pinned with the ``*_NUM_THREADS`` environment variables *before*
  NumPy is imported -- OpenBLAS reads them once, at load time. Pinning only one
  side turns the comparison into "our 20 BLAS threads vs their one", which
  measures something nobody asked about and flatters us by ~3x on matmul.
* Every measurement is the **median** of ``--repeat`` timed runs after
  ``--warmup`` untimed ones. The median, not the mean: a single OS scheduling
  hiccup skews a mean by more than the effect being measured. The **minimum** is
  reported next to it, since for a deterministic workload it is the cleanest
  estimate of the true cost -- and on a busy hybrid-core laptop the gap between
  median and min is itself the honest error bar, which ``bench_noise``
  quantifies before anything else runs.
* Timing uses ``time.perf_counter``.

Three published claims are checked against these measurements rather than
repeated on faith; see ``bench_dtype``, ``bench_einsum_optimize`` and
``bench_dispatch_overhead``.

Run:  PYTHONPATH=src python benchmarks/bench_vs_torch.py --repeat 7
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


def _pin_blas_threads(argv: list[str]) -> str:
    """Pin NumPy's BLAS to the same thread count as torch, before NumPy loads.

    OpenBLAS reads ``OPENBLAS_NUM_THREADS`` exactly once, when the shared library
    is loaded, so this has to happen above ``import numpy``. Without it NumPy
    quietly uses every core while torch is held to one, and the matmul "floor"
    benchmark reports a 3x win that is entirely a thread-count artifact.

    Args:
        argv: Command-line arguments, scanned for ``--threads``.

    Returns:
        The thread count that was applied, as a string.
    """
    n = "1"
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = argv[i + 1]
        elif a.startswith("--threads="):
            n = a.split("=", 1)[1]
    for var in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, n)
    return n


_THREADS = _pin_blas_threads(sys.argv[1:])

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import scratchgrad as sg
from scratchgrad import nn, optim

try:
    import torch

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    """One row of the comparison table. All times are milliseconds; lower is better."""

    name: str
    ours_ms: float
    ours_min_ms: float
    torch_ms: float | None
    torch_min_ms: float | None
    note: str = ""
    is_model: bool = False

    @property
    def ratio(self) -> float | None:
        """Slowdown factor ours/torch computed from the two medians."""
        if not self.torch_ms:
            return None
        return self.ours_ms / self.torch_ms

    @property
    def ratio_min(self) -> float | None:
        """Slowdown factor ours/torch computed from the two minima (less noisy)."""
        if not self.torch_min_ms:
            return None
        return self.ours_min_ms / self.torch_min_ms


@dataclass
class Claim:
    """A published claim, the measurement made here, and the verdict."""

    name: str
    claim: str
    measured: str
    verdict: str

    def show(self) -> None:
        """Print the claim as a three-line verdict block."""
        print(f"  [{self.verdict}] {self.name}")
        print(f"      claimed : {self.claim}")
        print(f"      measured: {self.measured}")


@dataclass
class Suite:
    """Accumulates table rows and claim verdicts, printing each as it lands."""

    results: list[Result] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)

    def add(self, r: Result) -> None:
        """Record and print one timing row."""
        self.results.append(r)
        ratio = f"{r.ratio:6.2f}x" if r.ratio else "     --"
        torch_ms = f"{r.torch_ms:9.2f}" if r.torch_ms else "       --"
        torch_min = f"{r.torch_min_ms:9.2f}" if r.torch_min_ms else "       --"
        print(
            f"  {r.name:<30} {r.ours_ms:8.2f} {r.ours_min_ms:8.2f} "
            f"{torch_ms} {torch_min}  {ratio}  {r.note}",
            flush=True,
        )

    def claim(self, c: Claim) -> None:
        """Record and print one claim verdict."""
        self.claims.append(c)
        c.show()


def timeit(fn, repeat: int, warmup: int) -> tuple[float, float]:
    """Time a callable.

    Args:
        fn: Zero-argument callable to time.
        repeat: Number of timed calls.
        warmup: Number of untimed calls first.

    Returns:
        ``(median_ms, min_ms)``.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), min(samples)


# --------------------------------------------------------------------------- #
# models, defined once per framework from the same spec
# --------------------------------------------------------------------------- #
def build_mlp(rng):
    """784-128-10 MLP in scratchgrad."""
    return nn.Sequential(nn.Linear(784, 128, rng=rng), nn.ReLU(), nn.Linear(128, 10, rng=rng))


def build_mlp_torch():
    """The same MLP in PyTorch."""
    return torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))


def build_lenet(rng):
    """LeNet-5 in scratchgrad."""
    return nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2, rng=rng, input_requires_grad=False),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5, rng=rng),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120, rng=rng),
        nn.ReLU(),
        nn.Linear(120, 84, rng=rng),
        nn.ReLU(),
        nn.Linear(84, 10, rng=rng),
    )


def build_lenet_torch():
    """The same LeNet-5 in PyTorch."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 6, 5, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(6, 16, 5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 5 * 5, 120),
        torch.nn.ReLU(),
        torch.nn.Linear(120, 84),
        torch.nn.ReLU(),
        torch.nn.Linear(84, 10),
    )


# --------------------------------------------------------------------------- #
# machine noise -- the error bar every other row must be read against
# --------------------------------------------------------------------------- #
def bench_noise(suite: Suite, args) -> None:
    """Time one fixed, deterministic workload many times and report the spread.

    Every ratio below is a quotient of two noisy numbers. Publishing the noise
    first is the only way a reader can tell a real 1.4x from scheduler jitter on
    a hybrid-core laptop where a single thread may be parked on an E-core.
    """
    print("\nNoise floor -- one identical 512x512 f32 matmul, timed many times:")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((512, 512)).astype(np.float32)
    b = rng.standard_normal((512, 512)).astype(np.float32)
    for _ in range(5):
        a @ b
    s = []
    for _ in range(max(30, args.repeat * 4)):
        t0 = time.perf_counter()
        a @ b
        s.append((time.perf_counter() - t0) * 1e3)
    med, lo, hi = statistics.median(s), min(s), max(s)
    print(f"  n={len(s)}  min {lo:.2f} ms   median {med:.2f} ms   max {hi:.2f} ms")
    print(
        f"  median/min {med / lo:.2f}x, max/min {hi / lo:.2f}x  <-- read every ratio against this"
    )
    suite.claims.append(
        Claim(
            "machine noise floor",
            "n/a -- calibration, not a published claim",
            f"median/min {med / lo:.2f}x, max/min {hi / lo:.2f}x over {len(s)} runs "
            f"of a bit-identical 512x512 f32 matmul",
            "MEASURED",
        )
    )


# --------------------------------------------------------------------------- #
# benchmarks
# --------------------------------------------------------------------------- #
def bench_matmul(suite: Suite, args) -> None:
    """Raw GEMM on both sides. Both call the same kind of BLAS, so this is the floor."""
    print("\nRaw matmul -- both sides call a single-threaded BLAS, so this is the floor:")
    for n in (256, 1024):
        a = np.random.default_rng(0).standard_normal((n, n))
        b = np.random.default_rng(1).standard_normal((n, n))
        af, bf = a.astype(np.float32), b.astype(np.float32)
        ours, ours_min = timeit(lambda af=af, bf=bf: af @ bf, args.repeat, args.warmup)
        th = thm = None
        if HAVE_TORCH:
            ta, tb = torch.tensor(af), torch.tensor(bf)
            th, thm = timeit(lambda ta=ta, tb=tb: ta @ tb, args.repeat, args.warmup)
        suite.add(Result(f"matmul {n}x{n} f32", ours, ours_min, th, thm, "same BLAS"))


def bench_dtype(suite: Suite, args) -> None:
    """Claim (a): float32 buys ~1.5-2x on matmul, not 7.5x."""
    print("\nCLAIM (a)  float32 vs float64 matmul")
    rng = np.random.default_rng(0)
    a64, b64 = rng.standard_normal((1024, 1024)), rng.standard_normal((1024, 1024))
    a32, b32 = a64.astype(np.float32), b64.astype(np.float32)
    m64, min64 = timeit(lambda: a64 @ b64, args.repeat, args.warmup)
    m32, min32 = timeit(lambda: a32 @ b32, args.repeat, args.warmup)
    print(f"  matmul 1024 f64   median {m64:8.2f} ms   min {min64:8.2f} ms")
    print(f"  matmul 1024 f32   median {m32:8.2f} ms   min {min32:8.2f} ms")
    sp_med, sp_min = m64 / m32, min64 / min32
    verdict = "CONFIRMED" if 1.3 <= sp_min <= 2.5 else "DIFFERENT-ON-THIS-MACHINE"
    suite.claim(
        Claim(
            "float32 vs float64 matmul",
            "~1.5-2x, not the 7.5x a popular microbenchmark reports "
            "(that one is dominated by writing a 400 MB smaller output matrix)",
            f"{sp_min:.2f}x on minima, {sp_med:.2f}x on medians "
            f"(1024x1024 GEMM, {args.threads} thread(s))",
            verdict,
        )
    )
    suite.add(Result("f64 -> f32 matmul", m32, min32, m64, min64, f"f32 is {sp_min:.2f}x faster"))


def bench_einsum_optimize(suite: Suite, args) -> None:
    """Claim (b): ``np.einsum`` defaults to ``optimize=False`` and never reaches BLAS."""
    print("\nCLAIM (b)  np.einsum optimize=")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 64, 100)).astype(np.float32)
    w = rng.standard_normal((64, 128)).astype(np.float32)
    off, off_min = timeit(lambda: np.einsum("nch,co->noh", x, w, optimize=False), args.repeat, 1)
    on, on_min = timeit(
        lambda: np.einsum("nch,co->noh", x, w, optimize=True), args.repeat, args.warmup
    )
    print(f"  einsum optimize=False   median {off:8.2f} ms   min {off_min:8.2f} ms")
    print(f"  einsum optimize=True    median {on:8.2f} ms   min {on_min:8.2f} ms")
    sp_med, sp_min = off / on, off_min / on_min
    verdict = "CONFIRMED" if 30.0 <= sp_min <= 300.0 else "DIFFERENT-ON-THIS-MACHINE"
    suite.claim(
        Claim(
            "einsum optimize=True",
            "the default optimize=False never dispatches to BLAS -- a 30-300x loss",
            f"{sp_min:.1f}x on minima, {sp_med:.1f}x on medians "
            f"('nch,co->noh', x=(32,64,100) f32, w=(64,128) f32)",
            verdict,
        )
    )
    suite.add(
        Result(
            "einsum optimize=True", on, on_min, off, off_min, f"{sp_min:.0f}x faster than default"
        )
    )


def bench_dispatch_overhead(suite: Suite, args) -> None:
    """Claim (c): NumPy per-call overhead is under 1% only above ~1e5-1e6 elements.

    Design. Hold the *total* work fixed at ``total`` elements and vary only how it
    is chopped up: ``total // n`` calls of ``np.multiply`` on contiguous chunks of
    length ``n``, writing through ``out=`` so that no allocation is charged to the
    small-call case -- otherwise this measures the allocator rather than dispatch.
    Chunk sizes are powers of two so ``n * calls == total`` exactly and every row
    does bit-identical arithmetic; the operand views are built *outside* the timed
    region so that only the ufunc call itself is on the clock. The per-element cost
    at the largest chunk is the hardware floor; the excess at chunk ``n`` is
    per-call overhead, and the crossing point is the smallest ``n`` whose excess is
    under 1%. The fixed cost of a single call is also measured directly on
    length-1 arrays, which predicts the same crossing independently
    (crossing ~= fixed_cost / floor_per_element / 0.01).
    """
    print("\nCLAIM (c)  NumPy per-call dispatch overhead")
    total = 1 << 24  # 16.8M f32 elements = 64 MB per operand, safely out of L3
    a = np.ones(total, dtype=np.float32)
    b = np.ones(total, dtype=np.float32)
    out = np.empty(total, dtype=np.float32)

    # Direct measurement of the fixed cost of one np.multiply call.
    tiny_a, tiny_b, tiny_o = a[:1], b[:1], out[:1]
    reps = 20000

    def tiny_batch():
        for _ in range(reps):
            np.multiply(tiny_a, tiny_b, out=tiny_o)

    _tmed, tmin = timeit(tiny_batch, 5, 1)
    fixed_us = tmin / reps * 1e3
    print(f"  fixed cost of one np.multiply call (n=1, out= supplied): {fixed_us:.3f} us")

    rows = []
    for k in range(10, 25):
        n = 1 << k
        calls = total // n
        chunks = [
            (a[i * n : (i + 1) * n], b[i * n : (i + 1) * n], out[i * n : (i + 1) * n])
            for i in range(calls)
        ]

        def work(chunks=chunks):
            for av, bv, ov in chunks:
                np.multiply(av, bv, out=ov)

        # Many timed runs, minimum taken: a 1% effect is far below this machine's
        # ~30% run-to-run spread, so the median is useless here and only the
        # minimum is a usable estimate of the true cost.
        _med, lo = timeit(work, max(15, args.repeat * 2), 2)
        rows.append((n, calls, lo / total * 1e6))  # ns per element, from the min run

    # Bandwidth floor: the asymptotic per-element cost, estimated from the largest
    # chunks where the per-call term is negligible by construction.
    floor = min(r[2] for r in rows if r[0] >= (1 << 20))
    crossing_model = 99.0 * (fixed_us * 1e-6) / (floor * 1e-9)
    print(f"  {'chunk n':>12} {'calls':>9} {'ns/elem':>9} {'measured':>9} {'model':>9}")
    crossing = None
    for n, calls, per_elem_ns in rows:
        ov = max(0.0, (per_elem_ns - floor) / per_elem_ns)
        model_ov = (fixed_us * 1e3 / n) / (floor + fixed_us * 1e3 / n)
        mark = ""
        if ov < 0.01 and crossing is None:
            crossing = n
            mark = "  <-- first measured row under 1%"
        print(
            f"  {n:>12,} {calls:>9} {per_elem_ns:>9.3f} "
            f"{ov * 100:>8.1f}% {model_ov * 100:>8.1f}%{mark}"
        )
    print(f"  fixed cost {fixed_us:.3f} us/call, bandwidth floor {floor:.3f} ns/elem")
    print(f"  1% crossing from the two measured constants: n = 99*c0/t_e = {crossing_model:,.0f}")
    empirical = (
        f"first measured row under 1% at n={crossing:,}" if crossing else "no measured row under 1%"
    )
    model_at_1e4 = (fixed_us * 1e3 / 1e4) / (floor + fixed_us * 1e3 / 1e4)
    at_8k = next(r for r in rows if r[0] == (1 << 13))[2] / floor - 1
    ok = 1e5 <= crossing_model <= 1e6
    suite.claim(
        Claim(
            "NumPy per-call dispatch overhead",
            "under 1% only above ~1e5-1e6 elements per call, not 1e4",
            f"n = 99*c0/t_e = {crossing_model:,.0f} elements from c0={fixed_us:.3f} us/call and "
            f"t_e={floor:.3f} ns/elem; at n=1e4 the model says {model_at_1e4 * 100:.0f}% "
            f"overhead, measured {at_8k:.0%} at n=8,192; "
            f"{empirical}",
            "CONFIRMED" if ok else "DIFFERENT-ON-THIS-MACHINE",
        )
    )


def bench_model(
    suite: Suite, args, label, ours_model, ours_x, ours_y, torch_model, torch_x, torch_y
):
    """Time forward, and forward+backward+optimizer step, on identical shapes."""

    def ours_fwd():
        with sg.no_grad():
            ours_model(ours_x)

    def ours_train():
        loss = nn.functional.cross_entropy(ours_model(ours_x), ours_y)
        ours_opt.zero_grad()
        loss.backward()
        ours_opt.step()

    ours_opt = optim.Adam(ours_model.parameters(), lr=1e-3)
    f_ours, f_ours_min = timeit(ours_fwd, args.repeat, args.warmup)
    t_ours, t_ours_min = timeit(ours_train, args.repeat, args.warmup)

    f_th = f_th_min = t_th = t_th_min = None
    if HAVE_TORCH and torch_model is not None:
        topt = torch.optim.Adam(torch_model.parameters(), lr=1e-3)

        def th_fwd():
            with torch.no_grad():
                torch_model(torch_x)

        def th_train():
            loss = torch.nn.functional.cross_entropy(torch_model(torch_x), torch_y)
            topt.zero_grad()
            loss.backward()
            topt.step()

        f_th, f_th_min = timeit(th_fwd, args.repeat, args.warmup)
        t_th, t_th_min = timeit(th_train, args.repeat, args.warmup)

    suite.add(Result(f"{label} forward", f_ours, f_ours_min, f_th, f_th_min, is_model=True))
    suite.add(Result(f"{label} fwd+bwd+step", t_ours, t_ours_min, t_th, t_th_min, is_model=True))


def bench_mlp(suite: Suite, args) -> None:
    """784-128-10 MLP, both frameworks, identical batch."""
    print(f"\nMLP 784-128-10, batch {args.batch}:")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((args.batch, 784)).astype(np.float32)
    y = rng.integers(0, 10, size=args.batch)
    ours = build_mlp(rng)
    th = build_mlp_torch() if HAVE_TORCH else None
    bench_model(
        suite,
        args,
        "MLP",
        ours,
        sg.Tensor(x),
        y,
        th,
        torch.tensor(x) if HAVE_TORCH else None,
        torch.tensor(y) if HAVE_TORCH else None,
    )


def bench_lenet(suite: Suite, args) -> None:
    """LeNet-5 on MNIST-shaped input, both frameworks."""
    print(f"\nLeNet-5 on 1x28x28, batch {args.batch}:")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((args.batch, 1, 28, 28)).astype(np.float32)
    y = rng.integers(0, 10, size=args.batch)
    ours = build_lenet(rng)
    th = build_lenet_torch() if HAVE_TORCH else None
    bench_model(
        suite,
        args,
        "LeNet-5",
        ours,
        sg.Tensor(x),
        y,
        th,
        torch.tensor(x) if HAVE_TORCH else None,
        torch.tensor(y) if HAVE_TORCH else None,
    )


def bench_lstm(suite: Suite, args) -> None:
    """Single-layer LSTM over T=64, both frameworks.

    ``nn.LSTM`` returns ``(output, state)`` on both sides; the tuple is unpacked
    where the gradient is needed and discarded in the forward-only case.
    """
    print("\nLSTM 128-hidden, batch 32, T=64:")
    rng = np.random.default_rng(0)
    n, t, h = 32, 64, 128
    x = rng.standard_normal((n, t, h)).astype(np.float32)
    ours = nn.LSTM(h, h, rng=rng)

    def ours_fwd():
        with sg.no_grad():
            ours(sg.Tensor(x))

    def ours_train():
        out, _ = ours(sg.Tensor(x))
        loss = out.sum()
        for p in ours.parameters():
            p.grad = None
        loss.backward()

    f_ours, f_min = timeit(ours_fwd, max(3, args.repeat // 2), 1)
    t_ours, t_min = timeit(ours_train, max(3, args.repeat // 2), 1)
    f_th = f_thm = t_th = t_thm = None
    if HAVE_TORCH:
        ref = torch.nn.LSTM(h, h, batch_first=True)
        xt = torch.tensor(x)

        def th_fwd():
            with torch.no_grad():
                ref(xt)

        def th_train():
            out, _ = ref(xt)
            ref.zero_grad()
            out.sum().backward()

        f_th, f_thm = timeit(th_fwd, args.repeat, args.warmup)
        t_th, t_thm = timeit(th_train, args.repeat, args.warmup)
    suite.add(
        Result("LSTM T=64 forward", f_ours, f_min, f_th, f_thm, "vs fused kernel", is_model=True)
    )
    suite.add(Result("LSTM T=64 fwd+bwd", t_ours, t_min, t_th, t_thm, is_model=True))


def bench_gpt(suite: Suite, args) -> None:
    """The Tier-5 mini-GPT. No torch counterpart: this one is about our own budget."""
    print("\nmini-GPT n_layer=4 n_head=4 n_embd=128 T=64, batch 12:")
    rng = np.random.default_rng(0)
    vocab, block, batch = 65, 64, 12
    idx = rng.integers(0, vocab, size=(batch, block))
    model = nn.GPT(vocab, block, n_layer=4, n_head=4, n_embd=128, rng=rng)
    print(f"  parameters: {model.num_parameters():,}")

    def fwd():
        with sg.no_grad():
            model(sg.Tensor(idx))

    def train():
        logits = model(sg.Tensor(idx))
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), idx.reshape(-1))
        for p in model.parameters():
            p.grad = None
        loss.backward()

    f, fm = timeit(fwd, max(3, args.repeat // 2), 1)
    t, tm = timeit(train, max(3, args.repeat // 2), 1)
    suite.add(Result("mini-GPT forward", f, fm, None, None, f"{model.num_parameters():,} params"))
    suite.add(
        Result(
            "mini-GPT fwd+bwd",
            t,
            tm,
            None,
            None,
            f"~{tm * 2000 / 1000 / 60:.1f} min for 2000 iters",
        )
    )


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3, help="untimed calls before timing")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument(
        "--threads", type=int, default=1, help="threads for BOTH torch and NumPy's BLAS"
    )
    ap.add_argument("--json", type=str, default="", help="write results here")
    ap.add_argument("--quick", action="store_true", help="skip the slow model benchmarks")
    args = ap.parse_args()

    print("=" * 100)
    print("scratchgrad vs PyTorch CPU -- honest benchmarks")
    print("=" * 100)
    print(f"python  {platform.python_version()}   numpy {np.__version__}")
    print(f"cpu     {platform.processor() or platform.machine()}")
    print(f"os      {platform.platform()}")
    blas_threads = os.environ.get("OPENBLAS_NUM_THREADS")
    print(f"blas    numpy pinned to {blas_threads} thread(s) via *_NUM_THREADS")
    if HAVE_TORCH:
        torch.set_num_threads(args.threads)
        print(f"torch   {torch.__version__}  pinned to {torch.get_num_threads()} thread(s)")
    else:
        print("torch   NOT INSTALLED -- comparison columns will be blank")
    print(f"median of {args.repeat} runs after {args.warmup} warmups; lower is better")
    print(
        f"\n  {'benchmark':<30} {'ours med':>8} {'ours min':>8} "
        f"{'torch med':>9} {'torch min':>9}  {'ratio':>7}"
    )
    print("  " + "-" * 97)

    suite = Suite()
    bench_noise(suite, args)
    bench_matmul(suite, args)
    bench_dtype(suite, args)
    bench_einsum_optimize(suite, args)
    bench_dispatch_overhead(suite, args)
    bench_mlp(suite, args)
    if not args.quick:
        bench_lenet(suite, args)
        bench_lstm(suite, args)
        bench_gpt(suite, args)

    print("\n" + "=" * 100)
    print("Claim check")
    print("=" * 100)
    for c in suite.claims:
        c.show()

    ratios = [r.ratio for r in suite.results if r.is_model and r.ratio]
    ratios_min = [r.ratio_min for r in suite.results if r.is_model and r.ratio_min]
    print("\n" + "=" * 100)
    if ratios:
        # Some rows can land below 1.0; say so rather than writing "0.3x slower".
        print(
            f"Summary over the {len(ratios)} model benchmarks that have a torch counterpart:\n"
            f"  ratio ours/torch from medians: min {min(ratios):.2f}x  "
            f"median {statistics.median(ratios):.2f}x  max {max(ratios):.2f}x\n"
            f"  ratio ours/torch from minima : min {min(ratios_min):.2f}x  "
            f"median {statistics.median(ratios_min):.2f}x  max {max(ratios_min):.2f}x\n"
            f"  >1 means scratchgrad is slower on that row; <1 means it is faster."
        )
    print(
        "Why we lose: no operator fusion, no threading inside our own code, a Python-level graph\n"
        "node per op, and an im2col convolution that materialises the patch matrix. Every\n"
        "elementwise op is a separate NumPy call with its own allocation and its own pass over\n"
        "memory; PyTorch fuses and reuses buffers. There is no GPU backend and none is claimed."
    )
    print("=" * 100)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "env": {
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "torch": torch.__version__ if HAVE_TORCH else None,
                        "threads": args.threads,
                        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                        "cpu": platform.processor(),
                        "machine": platform.machine(),
                        "platform": platform.platform(),
                    },
                    "args": vars(args),
                    "results": [
                        {
                            "name": r.name,
                            "ours_ms": r.ours_ms,
                            "ours_min_ms": r.ours_min_ms,
                            "torch_ms": r.torch_ms,
                            "torch_min_ms": r.torch_min_ms,
                            "ratio": r.ratio,
                            "ratio_min": r.ratio_min,
                            "is_model": r.is_model,
                            "note": r.note,
                        }
                        for r in suite.results
                    ],
                    "claims": [
                        {
                            "name": c.name,
                            "claim": c.claim,
                            "measured": c.measured,
                            "verdict": c.verdict,
                        }
                        for c in suite.claims
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
