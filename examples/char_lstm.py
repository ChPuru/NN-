"""Char-level LSTM on tiny-shakespeare -- and the Tier-4 trigram comparison.

The point of this example is not that an LSTM produces vaguely Shakespearean
noise. It is that it *beats a baseline that is actually trying*. A trigram
model with add-alpha smoothing and backoff is not a straw man: on 1M characters
of a single author it is a genuinely strong local predictor, and any recurrent
net that cannot beat it has learned nothing a count table does not already
know. So both models are scored on **the same validation tokens, in the same
units**, and both numbers are printed side by side.

Conventions this script commits to, because a loss without them is unreadable:

* **Reduction: per-token mean cross-entropy.** Not summed over the block. A
  clipping threshold is not dimensionless either -- ``--clip 1.0`` here is a
  global grad-norm ceiling *on the gradient of the per-token mean*, and would
  have to be ``block_size`` times larger against a summed loss.
* **Units: nats, and bits/char = nats / ln 2.** Random init on ``V=65`` is
  ``ln 65 = 4.174`` nats = 6.022 bits/char -- the zero-information reference.
* **Split: 90/10 by position**, never at random. Random blocks would put a
  character's own neighbours on the other side of the boundary, and the
  validation loss would then be measuring memorisation of adjacent context.

    PYTHONPATH=src python examples/char_lstm.py                  # full run
    PYTHONPATH=src python examples/char_lstm.py --smoke          # overfit 1 batch
    PYTHONPATH=src python examples/char_lstm.py --baseline-only  # trigram, ~10 s
    PYTHONPATH=src python examples/char_lstm.py --iters 500 --seed 0

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
from scratchgrad.data import CharVocab, tiny_shakespeare
from scratchgrad.utils import AverageMeter, ProgressBar, save_checkpoint

LN2 = math.log(2.0)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class CharLSTM(nn.Module):
    """Embedding -> stacked LSTM -> linear head, emitting raw logits.

    Args:
        vocab_size: Number of distinct characters ``V``.
        embed_dim: Width of the embedding table.
        hidden: LSTM hidden width.
        layers: Number of stacked LSTM layers.
        dropout: Inter-layer dropout (never inside the recurrence).
        rng: Explicit generator; one generator seeds every submodule so the
            whole model is reproducible from a single integer.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed = nn.Embedding(vocab_size, embed_dim, rng=rng)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers, dropout=dropout, rng=rng)
        # nonlinearity="linear" (gain 1) because nothing follows the head but the
        # softmax inside the loss; a relu gain would inflate the initial logits
        # and start training at a loss above ln(V) for no reason.
        self.head = nn.Linear(hidden, vocab_size, nonlinearity="linear", rng=rng)

    def forward(self, idx, hx=None):
        """Map ``(N, T)`` int ids to ``(N, T, V)`` logits and the final state."""
        out, state = self.lstm(self.embed(idx), hx)
        return self.head(out), state


def build_model(vocab_size: int, args, rng: np.random.Generator) -> CharLSTM:
    return CharLSTM(
        vocab_size,
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        rng=rng,
    )


# --------------------------------------------------------------------------- #
# the baseline: a smoothed trigram model, in numpy
# --------------------------------------------------------------------------- #
class TrigramLM:
    """Add-alpha trigram with backoff to bigram then unigram.

    ``P(c3 | c1, c2) = (N(c1,c2,c3) + a) / (N(c1,c2) + a*V)`` whenever the
    context ``(c1, c2)`` was seen in training at all; otherwise the bigram, and
    if ``c2`` is itself an unseen context, the unigram.

    The backoff is the part that matters. Add-alpha alone already assigns a
    probability to every unseen context -- but a *uniform* one, ``1/V``, which
    is 4.17 nats and strictly worse than what the bigram already knows. Backing
    off on an unseen context is therefore not a fallback for a numerical edge
    case; it is what makes the model a fair opponent rather than a rigged one.

    This is deliberately the honest-but-simple smoother, not Kneser-Ney: the
    claim being made is "the LSTM beats a competent count model", and stating
    exactly which count model is part of the claim.

    Args:
        vocab_size: ``V``.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self.uni: np.ndarray | None = None

    def fit(self, tokens: np.ndarray) -> TrigramLM:
        """Count 1-, 2- and 3-grams over ``tokens`` (a flat int array)."""
        v = self.vocab_size
        t = np.asarray(tokens, dtype=np.int64)
        # bincount over a flattened n-gram id, not np.add.at: same result, and
        # it is a single pass in C rather than a buffered scatter.
        self.uni = np.bincount(t, minlength=v).astype(np.float64)
        self.bi = np.bincount(t[:-1] * v + t[1:], minlength=v * v).astype(np.float64)
        self.bi = self.bi.reshape(v, v)
        tri_ids = (t[:-2] * v + t[1:-1]) * v + t[2:]
        self.tri = np.bincount(tri_ids, minlength=v * v * v).astype(np.float64)
        self.tri = self.tri.reshape(v * v, v)
        self.tri_ctx = self.tri.sum(axis=1)
        self.bi_ctx = self.bi.sum(axis=1)
        self.n_tokens = float(self.uni.sum())
        return self

    def nats(self, tokens: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
        """Per-token cross-entropy in nats for ``tokens[targets]``.

        Args:
            tokens: The stream being scored.
            targets: Positions in ``tokens`` to score. Each must be >= 1; the
                two preceding characters supply the context.
            alpha: Smoothing constant.

        Returns:
            ``(len(targets),)`` of ``-log P(token | context)``.
        """
        if self.uni is None:
            raise RuntimeError("fit() the trigram model before scoring")
        v = self.vocab_size
        t = np.asarray(tokens, dtype=np.int64)
        idx = np.asarray(targets, dtype=np.int64)
        if idx.size and idx.min() < 1:
            raise ValueError("every scored position needs at least one character of context")
        c3 = t[idx]
        c2 = t[idx - 1]
        # Clamp rather than index with -1: a negative index wraps to the end of
        # the corpus and would silently score the first token against the last.
        c1 = t[np.maximum(idx - 2, 0)]
        ctx = c1 * v + c2

        p_uni = (self.uni[c3] + alpha) / (self.n_tokens + alpha * v)
        p_bi = (self.bi[c2, c3] + alpha) / (self.bi_ctx[c2] + alpha * v)
        p_tri = (self.tri[ctx, c3] + alpha) / (self.tri_ctx[ctx] + alpha * v)

        use_tri = (idx >= 2) & (self.tri_ctx[ctx] > 0)
        use_bi = self.bi_ctx[c2] > 0
        p = np.where(use_tri, p_tri, np.where(use_bi, p_bi, p_uni))
        return -np.log(p)


def choose_alpha(
    train_tokens: np.ndarray,
    vocab_size: int,
    grid: tuple[float, ...],
    dev_frac: float = 0.05,
) -> tuple[float, list[tuple[float, float]]]:
    """Pick the smoothing constant on a dev slice held out of **train**.

    Selecting alpha on the validation split would be exactly the anti-goal the
    project names -- a number tuned on the data it is reported on. So the last
    ``dev_frac`` of the training split is carved off, alpha is chosen there, and
    the model is then refit on the whole training split before it ever touches
    validation. This gives the baseline its best honest shot, which is the
    point: a baseline that was quietly handicapped proves nothing.

    Args:
        train_tokens: The training stream.
        vocab_size: ``V``.
        grid: Candidate alphas.
        dev_frac: Fraction of train held out for the choice.

    Returns:
        ``(best_alpha, [(alpha, dev_nats), ...])``.
    """
    cut = int(len(train_tokens) * (1.0 - dev_frac))
    fit_part, dev_part = train_tokens[:cut], train_tokens[cut:]
    probe = TrigramLM(vocab_size).fit(fit_part)
    dev_targets = np.arange(2, len(dev_part))
    scored = [(a, float(probe.nats(dev_part, dev_targets, a).mean())) for a in grid]
    best = min(scored, key=lambda kv: kv[1])[0]
    return best, scored


# --------------------------------------------------------------------------- #
# truncated BPTT: contiguous parallel streams
# --------------------------------------------------------------------------- #
def stream_layout(n_tokens: int, n_streams: int, block: int) -> tuple[np.ndarray, int]:
    """Split a token stream into ``n_streams`` contiguous, equal-length streams.

    Truncated BPTT means the hidden state survives across consecutive blocks
    while the *graph* does not, so the batch axis has to be a set of positions
    that each advance by exactly ``block`` per iteration. Sampling a fresh
    random offset per iteration -- the usual transformer recipe -- is a
    different algorithm: it resets the state every block and the model then
    never sees more than ``block`` characters of context.

    Args:
        n_tokens: Length of the stream.
        n_streams: Batch size ``N``.
        block: Truncation length ``T``.

    Returns:
        ``(starts, n_blocks)``: ``starts[s]`` is where stream ``s`` begins and
        every stream has ``n_blocks`` usable blocks.

    Raises:
        ValueError: If the stream is too short to give every stream one block.
    """
    seg = n_tokens // n_streams
    # -1 because the target window reads one token past the input window.
    n_blocks = (seg - 1) // block
    if n_blocks < 1:
        raise ValueError(
            f"{n_tokens} tokens / {n_streams} streams = {seg} each, too short for block {block}"
        )
    return np.arange(n_streams, dtype=np.int64) * seg, n_blocks


def fetch(tokens: np.ndarray, starts: np.ndarray, b: int, block: int):
    """Block ``b`` of every stream as ``(x, y, target_positions)``."""
    off = starts + b * block
    idx = off[:, None] + np.arange(block, dtype=np.int64)[None, :]
    return tokens[idx], tokens[idx + 1], idx + 1


def scored_positions(starts: np.ndarray, n_blocks: int, block: int, warmup: int) -> np.ndarray:
    """The token positions an evaluation with ``warmup`` burn-in blocks scores."""
    return np.concatenate([fetch_positions(starts, b, block) for b in range(warmup, n_blocks)])


def fetch_positions(starts: np.ndarray, b: int, block: int) -> np.ndarray:
    off = starts + b * block
    return (off[:, None] + np.arange(block, dtype=np.int64)[None, :] + 1).ravel()


def evaluate(
    model: CharLSTM,
    tokens: np.ndarray,
    starts: np.ndarray,
    n_blocks: int,
    block: int,
    warmup: int,
) -> float:
    """Per-token mean cross-entropy in nats over the scored positions.

    The first ``warmup`` blocks of each stream run forward but are **not**
    scored. Their only job is to fill the hidden state, so that every token that
    does count is predicted with at least ``warmup * block`` characters of
    context behind it -- the same footing the trigram gets, which never has a
    cold start because its context is only two characters wide.
    """
    model.eval()
    meter = AverageMeter()
    state = None
    with sg.no_grad():
        for b in range(n_blocks):
            x, y, _ = fetch(tokens, starts, b, block)
            logits, state = model(x, state)
            if b >= warmup:
                meter.update(nn.functional.cross_entropy(logits, y).item(), int(y.size))
    model.train()
    return meter.avg


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def sample(
    model: CharLSTM,
    vocab: CharVocab,
    prompt: str,
    n_chars: int,
    temperature: float,
    rng: np.random.Generator,
) -> str:
    """Autoregressively extend ``prompt`` by ``n_chars`` characters.

    The prompt is consumed in one forward pass to build the state, then each new
    character is fed back one at a time -- an LSTM's state is a sufficient
    statistic for the whole prefix, so re-running the full context every step
    (what a transformer must do) would be pure waste.

    Args:
        temperature: Divides the logits before the softmax. ``->0`` is greedy
            and loops; ``1.0`` is the model's own distribution; ``0.8`` is the
            usual compromise. It is applied to *logits*, not probabilities:
            renormalising ``p ** (1/T)`` is the same thing only when the
            softmax has not already been taken, which is why the head must emit
            raw logits.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    model.eval()
    ids = vocab.encode(prompt)
    out: list[int] = []
    with sg.no_grad():
        logits, state = model(ids[None, :])
        z = logits.data[0, -1]
        for _ in range(n_chars):
            z = z.astype(np.float64) / temperature
            z = z - z.max()  # stabilise before exp; the loss does this too
            p = np.exp(z)
            p /= p.sum()
            nxt = int(rng.choice(len(p), p=p))
            out.append(nxt)
            logits, state = model(np.array([[nxt]], dtype=np.int64), state)
            z = logits.data[0, -1]
    model.train()
    return prompt + vocab.decode(out)


# --------------------------------------------------------------------------- #
# sanity checks
# --------------------------------------------------------------------------- #
def loss_at_init_is_ln_v(vocab_size: int, x: np.ndarray, y: np.ndarray, args) -> float:
    """Zero the head so the logits are exactly uniform, then check ``ln V``.

    Zeroing matters. An unzeroed net's initial loss fluctuates around ``ln V``
    and can legitimately land below it on any one batch, so asserting on it is a
    flaky test rather than a check. With the head zeroed the logits are
    identically zero and the loss is exactly ``ln V`` to float64 precision --
    which is a real test that the loss and its softmax agree.
    """
    with sg.default_dtype(np.float64):
        probe = build_model(vocab_size, args, np.random.default_rng(0)).double()
        probe.head.weight.data[...] = 0.0
        probe.head.bias.data[...] = 0.0
        with sg.no_grad():
            logits, _ = probe(x)
            loss = nn.functional.cross_entropy(logits, y)
    return loss.item()


def smoke(args) -> int:
    """Overfit a single batch before spending half an hour on the real run.

    If a 2-layer LSTM cannot drive one batch of ``batch x block`` characters to
    near-zero loss, the model is broken rather than undertrained, and finding
    that out in a minute beats finding it out in thirty.
    """
    rng = np.random.default_rng(args.seed)
    text = tiny_shakespeare(root=ROOT / "data" / "text")
    vocab = CharVocab(text)
    tokens = vocab.encode(text)
    n, t = 4, 64
    x = np.stack([tokens[i * t : i * t + t] for i in range(n)])
    y = np.stack([tokens[i * t + 1 : i * t + t + 1] for i in range(n)])

    l0 = loss_at_init_is_ln_v(vocab.size, x, y, args)
    print(f"loss at init with the head zeroed: {l0:.15f}")
    print(f"                          ln({vocab.size}):  {math.log(vocab.size):.15f}")
    assert abs(l0 - math.log(vocab.size)) < 1e-12, "uniform logits must give exactly ln(V)"

    model = build_model(vocab.size, args, rng)
    opt = optim.Adam(model.parameters(), lr=3e-3)
    loss = None
    for step in range(200):
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if step % 40 == 0 or step == 199:
            print(f"  step {step:4d}  loss {loss.item():.6f}")
    final = loss.item()
    print(f"\noverfit {n}x{t} characters -> loss {final:.2e}", "OK" if final < 1e-2 else "FAILED")
    return 0 if final < 1e-2 else 1


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=500, help="optimizer steps")
    ap.add_argument("--batch-size", type=int, default=64, help="parallel BPTT streams")
    ap.add_argument("--block-size", type=int, default=128, help="BPTT truncation length T")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="global grad-norm ceiling, against the per-token MEAN loss",
    )
    ap.add_argument("--val-streams", type=int, default=32)
    ap.add_argument("--val-warmup-blocks", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--sample-chars", type=int, default=400)
    ap.add_argument("--prompt", type=str, default="KING RICHARD III:\n")
    ap.add_argument("--smoke", action="store_true", help="overfit one batch and stop")
    ap.add_argument("--baseline-only", action="store_true", help="trigram only, no training")
    ap.add_argument("--out", type=str, default="runs/char_lstm")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        return smoke(args)

    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print("loading tiny-shakespeare ...")
    text = tiny_shakespeare(root=ROOT / "data" / "text")
    assert "\r" not in text, "CRLF would add a 66th vocabulary entry and change every number"
    vocab = CharVocab(text)
    tokens = vocab.encode(text)
    # The vocabulary is built from the whole corpus, not just train: a character
    # that appears only after the split point would otherwise raise KeyError at
    # encode time. This leaks the *alphabet*, not the text -- V is 65 either way
    # here, and the alternative is a model that cannot represent its own input.
    cut = int(0.9 * len(tokens))
    train_tokens, val_tokens = tokens[:cut], tokens[cut:]
    print(f"  {len(text):,} characters, vocab {vocab.size}")
    print(f"  split by position: train {len(train_tokens):,}  val {len(val_tokens):,}")

    block = args.block_size
    val_starts, val_blocks = stream_layout(len(val_tokens), args.val_streams, block)
    val_positions = scored_positions(val_starts, val_blocks, block, args.val_warmup_blocks)
    print(
        f"  val scored on {len(val_positions):,} tokens "
        f"({100 * len(val_positions) / len(val_tokens):.1f}% of the split)"
    )

    # -- the baseline, first, so the target is known before anything is trained.
    print("\ntrigram baseline")
    grid = (3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
    alpha, dev_scores = choose_alpha(train_tokens, vocab.size, grid)
    for a, s in dev_scores:
        mark = "  <- chosen" if a == alpha else ""
        print(f"  alpha {a:<6g} dev {s:.4f} nats{mark}")
    t_tri = time.perf_counter()
    trigram = TrigramLM(vocab.size).fit(train_tokens)
    tri_nats = float(trigram.nats(val_tokens, val_positions, alpha).mean())
    tri_laplace = float(trigram.nats(val_tokens, val_positions, 1.0).mean())
    tri_seconds = time.perf_counter() - t_tri
    print(f"  fit on {len(train_tokens):,} train characters in {tri_seconds:.2f}s")
    print(f"  val  {tri_nats:.4f} nats = {tri_nats / LN2:.4f} bits/char  (alpha={alpha:g})")
    print(f"  val  {tri_laplace:.4f} nats = {tri_laplace / LN2:.4f} bits/char  (alpha=1, Laplace)")
    print(
        f"  reference: uniform over {vocab.size} chars = {math.log(vocab.size):.4f} nats "
        f"= {math.log(vocab.size) / LN2:.4f} bits/char"
    )

    if args.baseline_only:
        return 0

    # -- sanity check 1, before anything is trained.
    x0, y0, _ = fetch(train_tokens, np.array([0, 1000]), 0, 64)
    l0 = loss_at_init_is_ln_v(vocab.size, x0, y0, args)
    print(f"\nloss at init (head zeroed) = {l0:.15f}")
    print(f"                    ln({vocab.size}) = {math.log(vocab.size):.15f}")
    assert abs(l0 - math.log(vocab.size)) < 1e-12, "uniform logits must give exactly ln(V)"
    print("  -> exact to 1e-12. The loss and the softmax agree.")

    model = build_model(vocab.size, args, rng)
    print(f"\n{model}\n")
    print(model.summary())

    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.WarmupCosine(
        opt, warmup_steps=args.warmup, total_steps=args.iters, min_lr_ratio=0.1
    )
    tr_starts, tr_blocks = stream_layout(len(train_tokens), args.batch_size, block)
    tokens_per_iter = args.batch_size * block
    print(
        f"\ntraining {args.iters} iters x {args.batch_size} streams x {block} chars "
        f"= {args.iters * tokens_per_iter / 1e6:.2f}M tokens "
        f"({args.iters * tokens_per_iter / len(train_tokens):.2f} epochs)"
    )
    print(
        f"  Adam lr={args.lr} (warmup {args.warmup}, cosine to 0.1x), clip {args.clip}, "
        f"{tr_blocks} blocks per epoch"
    )

    history: dict[str, list[float]] = {
        "iter": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "grad_norm": [],
    }
    meter = AverageMeter()
    state = None
    # The fallback renderer never emits a newline until close(), so a redirected
    # log would otherwise become one multi-megabyte line.
    bar = ProgressBar(total=args.iters, desc="train", disable=not sys.stderr.isatty())
    t_start = time.perf_counter()
    for it in range(1, args.iters + 1):
        b = (it - 1) % tr_blocks
        if b == 0:
            # New pass over the corpus: the streams jump back to their starts, so
            # the carried state describes the wrong position and must be dropped.
            state = None
        x, y, _ = fetch(train_tokens, tr_starts, b, block)
        logits, state = model(x, state)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        gnorm = optim.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        # Truncated BPTT: keep the *values* of the state, drop the graph that
        # produced them. Without the detach the tape grows without bound and
        # backward() would eventually replay the entire epoch.
        state = nn.detach_state(state)
        meter.update(loss.item(), int(y.size))
        bar.update()
        bar.set_postfix(loss=f"{meter.avg:.4f}", gn=f"{gnorm:.2f}")

        if it % args.eval_every == 0 or it == args.iters:
            val = evaluate(model, val_tokens, val_starts, val_blocks, block, args.val_warmup_blocks)
            history["iter"].append(it)
            history["train_loss"].append(meter.avg)
            history["val_loss"].append(val)
            history["lr"].append(float(sched.get_last_lr()[0]))
            history["grad_norm"].append(float(gnorm))
            print(
                f"  iter {it:5d}  train {meter.avg:.4f}  val {val:.4f} nats "
                f"({val / LN2:.4f} bits/char)  lr {sched.get_last_lr()[0]:.2e}",
                flush=True,
            )
            # Reset so each reported train loss covers only the window since the
            # last eval; a running mean over the whole run hides the curve.
            meter.reset()
    bar.close()
    elapsed = time.perf_counter() - t_start

    val_nats = evaluate(model, val_tokens, val_starts, val_blocks, block, args.val_warmup_blocks)
    # Same protocol on a same-sized slice of train, so the gap is comparable.
    tr_eval_starts, tr_eval_blocks = stream_layout(len(val_tokens), args.val_streams, block)
    train_nats = evaluate(
        model,
        train_tokens[: len(val_tokens)],
        tr_eval_starts,
        tr_eval_blocks,
        block,
        args.val_warmup_blocks,
    )

    print(
        f"\ntrained in {elapsed:.1f}s on CPU "
        f"({args.iters * tokens_per_iter / elapsed:,.0f} tokens/s)"
    )
    print(f"parameters   : {model.num_parameters():,}")
    print(f"\n{'model':<22}{'nats':>9}{'bits/char':>12}")
    print("-" * 43)
    print(f"{'uniform (V=65)':<22}{math.log(vocab.size):>9.4f}{math.log(vocab.size) / LN2:>12.4f}")
    print(f"{'trigram (val)':<22}{tri_nats:>9.4f}{tri_nats / LN2:>12.4f}")
    print(f"{'char-LSTM (val)':<22}{val_nats:>9.4f}{val_nats / LN2:>12.4f}")
    print(f"{'char-LSTM (train)':<22}{train_nats:>9.4f}{train_nats / LN2:>12.4f}")
    print("-" * 43)
    print(
        f"LSTM beats trigram by {tri_nats - val_nats:.4f} nats "
        f"({(tri_nats - val_nats) / LN2:.4f} bits/char, "
        f"{100 * (1 - val_nats / tri_nats):.1f}% lower) on identical validation tokens"
    )

    text_sample = sample(
        model,
        vocab,
        args.prompt,
        args.sample_chars,
        args.temperature,
        np.random.default_rng(args.seed + 1),
    )
    print(
        f"\n--- sample, temperature {args.temperature} "
        f"({args.sample_chars} chars after the prompt) ---"
    )
    print(text_sample)
    print("--- end sample ---")

    save_checkpoint(
        out / "weights.npz",
        model,
        opt,
        meta={
            "seed": args.seed,
            "iters": args.iters,
            "batch_size": args.batch_size,
            "block_size": block,
            "lr": args.lr,
            "clip": args.clip,
            "val_nats": val_nats,
            "trigram_val_nats": tri_nats,
            "train_seconds": elapsed,
            "vocab": "".join(vocab.chars),
        },
    )
    (out / "sample.txt").write_text(text_sample, encoding="utf-8")
    (out / "history.json").write_text(
        json.dumps(
            {
                "history": history,
                "reduction": "per-token mean cross-entropy",
                "val_nats": val_nats,
                "val_bits_per_char": val_nats / LN2,
                "train_nats": train_nats,
                "train_bits_per_char": train_nats / LN2,
                "trigram_val_nats": tri_nats,
                "trigram_val_bits_per_char": tri_nats / LN2,
                "trigram_alpha": alpha,
                "trigram_dev_scores": dev_scores,
                "trigram_laplace_val_nats": tri_laplace,
                "uniform_nats": math.log(vocab.size),
                "gap_nats": tri_nats - val_nats,
                "val_tokens_scored": len(val_positions),
                "seconds": elapsed,
                "tokens_per_second": args.iters * tokens_per_iter / elapsed,
                "parameters": model.num_parameters(),
                "loss_at_init": l0,
                "vocab_size": vocab.size,
                "seed": args.seed,
                "config": vars(args) | {"out": str(args.out)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved {out / 'weights.npz'}, history.json and sample.txt")

    if not args.no_plots:
        try:
            from scratchgrad.utils import viz

            assets = ROOT / "assets"
            assets.mkdir(exist_ok=True)
            # The baseline goes on the same axes as a flat line: a learning curve
            # without the thing it has to beat is a decoration, not a result.
            plotted = dict(history)
            plotted["trigram_val_loss"] = [tri_nats] * len(history["iter"])
            viz.plot_history(
                {k: v for k, v in plotted.items() if "loss" in k},
                path=assets / "char_lstm_training.png",
                # viz.plot_history hard-labels the x-axis "epoch"; here one point
                # is one eval checkpoint, so the title says what the axis cannot.
                title=(
                    f"char-LSTM on tiny-shakespeare -- nats/char, per-token mean "
                    f"(1 point = {args.eval_every} iters)"
                ),
            )
            print(f"wrote {assets / 'char_lstm_training.png'}")
        except ImportError as exc:
            print(f"skipping plots: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
