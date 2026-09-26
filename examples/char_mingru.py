"""Char-level minGRU (or a selective SSM) on tiny-shakespeare, scored against the LSTM.

Tier 8's question is whether a recurrence whose gates see only the *input* -- and
which therefore trains as one parallel scan instead of ``T`` dependent steps -- can
match the Tier-4 LSTM. So everything that makes a number comparable is imported
from ``examples/char_lstm.py`` rather than rewritten: the 90/10 split by position,
the contiguous-stream truncated BPTT layout, and the deterministic full-coverage
validation pass (32 streams, one un-scored warm-up block, then the same 106,496
val tokens each scored exactly once). Same tokens, same units (per-token mean
nats), same seed, same optimiser schedule. The LSTM's numbers are read from its
``history.json``.

Model: embedding -> ``layers`` x pre-LN residual blocks ``[x + mix(ln(x)); x +
mlp(ln(x))]`` -> LayerNorm -> linear head, where ``mix`` is :class:`MinGRU` or
:class:`SelectiveSSM`. A single minGRU layer is linear in its state, so the MLPs
between them are not decoration: they are the only nonlinearity on the depth path.

Timing is measured the way ``char_lstm.py`` measures it: wall clock around the
whole training loop, with the periodic evaluations included. That number is only
as good as the machine was steady, though. On the laptop these runs were made on,
a sustained load gets power-throttled about 3 s in: a GEMM goes from 6 ms to 31 ms
and an ``exp`` from 6 ms to 57 ms. Runs made minutes apart, or alongside other jobs,
cannot be compared on their totals. So after training, ``--bench-steps`` times a
fresh copy of this model and a fresh char-LSTM *interleaved*, one step of each in
turn, and that ratio is the speed claim.

    PYTHONPATH=src python examples/char_mingru.py                 # minGRU
    PYTHONPATH=src python examples/char_mingru.py --model ssm --d-model 128 --mlp-ratio 4
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
sys.path.insert(0, str(ROOT / "examples"))

from char_lstm import build_model, evaluate, fetch, sample, scored_positions, stream_layout

import scratchgrad as sg
from scratchgrad import nn, optim
from scratchgrad.data import CharVocab, tiny_shakespeare
from scratchgrad.nn.scan import MinGRU, SelectiveSSM
from scratchgrad.utils import AverageMeter

LN2 = math.log(2.0)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    """Pre-LN residual block around a scan layer and a ReLU MLP.

    The residual stream is kept as ``(N*T, d)`` rows and reshaped to ``(N, T, d)``
    only around the scan: every Linear then runs as one 2-D GEMM, where NumPy's
    stacked 3-D matmul path is ~100x slower. ReLU rather than GELU for the same
    reason -- ``functional.gelu`` evaluates ``a**3``, a generic ``pow`` that costs
    more than the GEMM beside it.
    """

    def __init__(self, d: int, kind: str, d_state: int, mlp_ratio: int, rng) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        if kind == "mingru":
            self.mix = MinGRU(d, d, rng=rng)
        else:
            self.mix = SelectiveSSM(d, d_state, rng=rng)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, mlp_ratio * d, rng=rng)
        self.proj = nn.Linear(mlp_ratio * d, d, nonlinearity="linear", rng=rng)

    def forward(self, x, shape, h0=None):
        y, h = self.mix(self.ln1(x).reshape(shape), h0)
        x = x + y.reshape(x.shape)
        return x + self.proj(self.fc(self.ln2(x)).relu()), h


class CharScanLM(nn.Module):
    """``(N, T)`` ids -> ``(N, T, V)`` raw logits, plus one carried state per block."""

    def __init__(self, vocab_size: int, args, rng: np.random.Generator) -> None:
        super().__init__()
        d = self.d_model = args.d_model
        self.embed = nn.Embedding(vocab_size, d, rng=rng)
        self.blocks = nn.ModuleList(
            [Block(d, args.model, args.d_state, args.mlp_ratio, rng) for _ in range(args.layers)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, nonlinearity="linear", rng=rng)

    def forward(self, idx, state=None):
        n, steps = np.shape(idx)
        x = self.embed(idx).reshape(n * steps, self.d_model)
        new_state = []
        for k, block in enumerate(self.blocks):
            x, h = block(x, (n, steps, self.d_model), None if state is None else state[k])
            new_state.append(h)
        return self.head(self.ln_f(x)).reshape(n, steps, -1), tuple(new_state)


def loss_at_init_is_ln_v(vocab_size: int, x: np.ndarray, y: np.ndarray, args) -> float:
    """Zeroed head -> uniform logits -> the loss must be exactly ``ln V`` (float64)."""
    with sg.default_dtype(np.float64):
        probe = CharScanLM(vocab_size, args, np.random.default_rng(0)).double()
        probe.head.weight.data[...] = 0.0
        probe.head.bias.data[...] = 0.0
        with sg.no_grad():
            return nn.functional.cross_entropy(probe(x)[0], y).item()


def seconds_per_step(models, x, y, steps: int) -> list[float]:
    """Median seconds per training step (forward, backward, Adam) for each model.

    The steps are *interleaved* -- one of each model, then again -- so every model
    sees the same machine load. On a shared CPU two runs made minutes apart are not
    comparable: another process's BLAS threads can slow a GEMM twentyfold.
    """
    opts = [optim.Adam(m.parameters(), lr=1e-3) for m in models]
    times: list[list[float]] = [[] for _ in models]
    for r in range(steps + 1):  # round 0 is a warm-up and is not recorded
        for k, (m, opt) in enumerate(zip(models, opts, strict=True)):
            t0 = time.perf_counter()
            loss = nn.functional.cross_entropy(m(x)[0], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if r:
                times[k].append(time.perf_counter() - t0)
    return [float(np.median(t)) for t in times]


# --------------------------------------------------------------------------- #
# the comparison figure
# --------------------------------------------------------------------------- #
def plot_comparison(path: Path, trigram_nats: float | None) -> None:
    """Val nats against tokens seen and against wall-clock, for every run on disk.

    The LSTM's history records only its total time, so its time axis assumes a
    constant cost per iteration -- true for a fixed-shape loop, and labelled.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [
        ("char-LSTM", "runs/char_lstm"),
        ("minGRU", "runs/char_mingru"),
        ("selective SSM", "runs/char_ssm"),
    ]
    fig, (ax_tok, ax_sec) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for label, rel in runs:
        f = ROOT / rel / "history.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        hist, cfg = rec["history"], rec["config"]
        tokens = np.array(hist["iter"]) * cfg["batch_size"] * cfg["block_size"] / 1e6
        if "seconds" in hist:
            secs, suffix = np.array(hist["seconds"]), ""
        else:
            secs, suffix = (
                np.array(hist["iter"]) * rec["seconds"] / cfg["iters"],
                " (time interpolated)",
            )
        bench = (rec.get("vs_char_lstm") or {}).get("same_load_timing")
        if bench:
            suffix += (
                f" [same load: {bench['scan_s_per_step']:.1f} vs LSTM "
                f"{bench['lstm_s_per_step']:.1f} s/step]"
            )
        line = ax_tok.plot(tokens, hist["val_loss"], marker="o", ms=3, label=label)[0]
        ax_sec.plot(
            secs, hist["val_loss"], marker="o", ms=3, color=line.get_color(), label=label + suffix
        )
    for ax in (ax_tok, ax_sec):
        if trigram_nats is not None:
            ax.axhline(trigram_nats, color="grey", ls="--", lw=1, label="trigram")
        ax.grid(alpha=0.3)
    ax_tok.set_xlabel("training tokens (M)")
    ax_sec.set_xlabel("recorded wall-clock s (evals incl.; machine load differed between runs)")
    ax_tok.set_ylabel("val nats/char (per-token mean)")
    ax_sec.set_xscale("log")
    ax_sec.legend(fontsize=8)
    fig.suptitle("tiny-shakespeare, identical val tokens: LSTM vs parallel-scan recurrences")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["mingru", "ssm"], default="mingru")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--mlp-ratio", type=int, default=2)
    ap.add_argument("--d-state", type=int, default=8, help="SSM state size per channel")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip", type=float, default=1.0, help="against the per-token MEAN loss")
    ap.add_argument("--val-streams", type=int, default=32)
    ap.add_argument("--val-warmup-blocks", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--sample-chars", type=int, default=400)
    ap.add_argument("--prompt", type=str, default="KING RICHARD III:\n")
    ap.add_argument("--out", type=str, default=None, help="default runs/char_<model>")
    ap.add_argument(
        "--bench-steps",
        type=int,
        default=3,
        help="interleaved same-load timing steps against a fresh char-LSTM (0 = skip)",
    )
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    args.out = args.out or ("runs/char_mingru" if args.model == "mingru" else "runs/char_ssm")

    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    text = tiny_shakespeare(root=ROOT / "data" / "text")
    vocab = CharVocab(text)
    tokens = vocab.encode(text)
    cut = int(0.9 * len(tokens))
    train_tokens, val_tokens = tokens[:cut], tokens[cut:]
    block = args.block_size
    val_starts, val_blocks = stream_layout(len(val_tokens), args.val_streams, block)
    val_positions = scored_positions(val_starts, val_blocks, block, args.val_warmup_blocks)
    print(f"{len(text):,} chars, vocab {vocab.size}; val scores {len(val_positions):,} tokens")

    x0, y0, _ = fetch(train_tokens, np.array([0, 1000]), 0, 64)
    l0 = loss_at_init_is_ln_v(vocab.size, x0, y0, args)
    assert abs(l0 - math.log(vocab.size)) < 1e-12, "uniform logits must give exactly ln(V)"

    model = CharScanLM(vocab.size, args, rng)
    print(f"{model}\nparameters: {model.num_parameters():,}")
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.WarmupCosine(
        opt, warmup_steps=args.warmup, total_steps=args.iters, min_lr_ratio=0.1
    )
    tr_starts, tr_blocks = stream_layout(len(train_tokens), args.batch_size, block)
    tokens_per_iter = args.batch_size * block

    history: dict[str, list[float]] = {
        "iter": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "grad_norm": [],
        "seconds": [],
    }
    meter = AverageMeter()
    state = None
    t_start = time.perf_counter()
    for it in range(1, args.iters + 1):
        b = (it - 1) % tr_blocks
        if b == 0:
            state = None  # streams jump back to their starts; the old state is stale
        x, y, _ = fetch(train_tokens, tr_starts, b, block)
        logits, state = model(x, state)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        gnorm = optim.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        state = nn.detach_state(state)
        meter.update(loss.item(), int(y.size))

        if it % args.eval_every == 0 or it == args.iters:
            val = evaluate(model, val_tokens, val_starts, val_blocks, block, args.val_warmup_blocks)
            history["iter"].append(it)
            history["train_loss"].append(meter.avg)
            history["val_loss"].append(val)
            history["lr"].append(float(sched.get_last_lr()[0]))
            history["grad_norm"].append(float(gnorm))
            history["seconds"].append(time.perf_counter() - t_start)
            print(
                f"  iter {it:5d}  train {meter.avg:.4f}  val {val:.4f} nats  "
                f"({time.perf_counter() - t_start:.0f}s)",
                flush=True,
            )
            meter.reset()
    elapsed = time.perf_counter() - t_start
    val_nats = history["val_loss"][-1]  # the final eval above ran on the final weights

    lstm_file = ROOT / "runs" / "char_lstm" / "history.json"
    lstm = json.loads(lstm_file.read_text(encoding="utf-8")) if lstm_file.exists() else None
    name = "minGRU" if args.model == "mingru" else "selective SSM"
    print(
        f"\n{'model':<16}{'params':>10}{'val nats':>10}{'bits/char':>11}{'seconds':>9}{'tok/s':>8}"
    )
    rows = [
        (name, model.num_parameters(), val_nats, elapsed, args.iters * tokens_per_iter / elapsed)
    ]
    if lstm is not None:
        rows.append(
            (
                "char-LSTM",
                lstm["parameters"],
                lstm["val_nats"],
                lstm["seconds"],
                lstm["tokens_per_second"],
            )
        )
    for label, params, nats, secs, tps in rows:
        print(f"{label:<16}{params:>10,}{nats:>10.4f}{nats / LN2:>11.4f}{secs:>9.0f}{tps:>8,.0f}")
    comparison = None
    if lstm is not None:
        bench = None
        if args.bench_steps > 0:
            bx, by, _ = fetch(train_tokens, tr_starts, 0, block)
            lstm_cfg = argparse.Namespace(**lstm["config"])
            fresh = [
                CharScanLM(vocab.size, args, np.random.default_rng(1)),
                build_model(vocab.size, lstm_cfg, np.random.default_rng(1)),
            ]
            s_scan, s_lstm = seconds_per_step(fresh, bx, by, args.bench_steps)
            bench = {"scan_s_per_step": s_scan, "lstm_s_per_step": s_lstm}
            print(
                f"same-load timing, {args.bench_steps} interleaved steps: {name} {s_scan:.2f} s/step,"
                f" char-LSTM {s_lstm:.2f} s/step -> {s_lstm / s_scan:.1f}x"
            )
        comparison = {
            "same_load_timing": bench,
            "lstm_val_nats": lstm["val_nats"],
            "lstm_seconds": lstm["seconds"],
            "lstm_parameters": lstm["parameters"],
            "lstm_tokens_per_second": lstm["tokens_per_second"],
            "val_nats_minus_lstm": val_nats - lstm["val_nats"],
            "speedup_vs_lstm": lstm["seconds"] / elapsed,
            "trigram_val_nats": lstm["trigram_val_nats"],
        }
        verdict = "beats" if val_nats < lstm["val_nats"] else "LOSES TO"
        print(
            f"{name} {verdict} the LSTM by {abs(val_nats - lstm['val_nats']):.4f} nats on identical "
            f"val tokens, in {elapsed / lstm['seconds']:.2f}x its wall-clock"
        )

    text_sample = sample(
        model,
        vocab,
        args.prompt,
        args.sample_chars,
        args.temperature,
        np.random.default_rng(args.seed + 1),
    )
    print(f"\n--- sample, temperature {args.temperature} ---\n{text_sample}\n--- end sample ---")
    (out / "sample.txt").write_text(text_sample, encoding="utf-8")
    (out / "history.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "history": history,
                "reduction": "per-token mean cross-entropy",
                "val_nats": val_nats,
                "val_bits_per_char": val_nats / LN2,
                "val_tokens_scored": len(val_positions),
                "seconds": elapsed,
                "seconds_note": "wall clock around the training loop, evals included (as char_lstm.py)",
                "tokens_per_second": args.iters * tokens_per_iter / elapsed,
                "parameters": model.num_parameters(),
                "loss_at_init": l0,
                "vocab_size": vocab.size,
                "seed": args.seed,
                "vs_char_lstm": comparison,
                "config": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {out / 'history.json'} and sample.txt")

    if not args.no_plots:
        png = ROOT / "assets" / "char_mingru.png"
        png.parent.mkdir(exist_ok=True)
        plot_comparison(png, None if lstm is None else lstm["trigram_val_nats"])
        print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
