"""A mini-GPT on tiny-shakespeare -- the Tier-5 attention result, measured honestly.

Four pre-LN decoder blocks, four heads, width 128, context 64, trained with AdamW
under a linear-warmup/cosine schedule and gradient clipping. Everything -- the
softmax Jacobian, the causal mask, LayerNorm's backward, AdamW's decoupled decay
-- runs through ``scratchgrad``; NumPy is the only dependency.

    PYTHONPATH=src python examples/mini_gpt.py                  # the real run
    PYTHONPATH=src python examples/mini_gpt.py --smoke          # overfit 8 seqs
    PYTHONPATH=src python examples/mini_gpt.py --rescore        # re-measure, no retrain
    PYTHONPATH=src python examples/mini_gpt.py --iters 500 --seed 1

**The compute budget, stated up front.** This configuration --
``n_layer=4, n_head=4, n_embd=128, block_size=64, batch=12, 2000 iters`` -- is
about ``8e12`` FLOP, which is the largest thing that finishes in a coffee break
in pure NumPy on a laptop CPU. nanoGPT's *headline* configuration
(``n_layer=6, n_head=6, n_embd=384, T=256, 5000 iters``) is about **5.8 PFLOP**,
i.e. **16-32 hours** here. That published runtime is an A100-in-PyTorch number
and roughly three orders of magnitude of hardware and kernel advantage do not
transfer to this file. Quoting it as a target for a NumPy implementation is the
most common way one of these projects ends up looking broken when it is merely
small. This script does not run that config and neither should you.

**Units, which are widely misquoted.** Cross-entropy here is the mean per-token
negative log-likelihood in **nats**; divide by ``ln 2`` for bits/character.
nanoGPT's often-cited ``~1.47`` is nats, i.e. 2.12 bits/char. Random init on the
65-character vocabulary is ``ln(65) = 4.174`` nats = 6.02 bits/char, not 1.0.
The target for *this* budget is roughly 1.88 nats; the script reports what it
actually measured, whichever side of that it lands on.

**The comparisons.** Two, both on the same 90/10 positional split of the same
corpus: the ``ln(65)`` random-init reference, and the Tier-4 char-LSTM from
``examples/char_lstm.py`` if its ``runs/char_lstm/history.json`` exists. That
second one is scored twice -- once at each script's own headline protocol, and
once on the *identical* target positions the LSTM scores, because the two
protocols differ (the LSTM skips a warmup block per stream; this script scores
every position of every window) by enough to swamp the gap being measured. If
the LSTM run is absent the script says so rather than guessing, and ``--rescore``
re-measures against a frozen checkpoint once it lands.

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
from scratchgrad.utils import (
    AverageMeter,
    ProgressBar,
    load_checkpoint,
    save_checkpoint,
)

#: nats -> bits. Kept as a named constant because mixing the two units is the
#: single most common way a language-model number gets misreported.
NATS_TO_BITS = 1.0 / math.log(2.0)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def split_corpus(text: str, val_frac: float) -> tuple[CharVocab, np.ndarray, np.ndarray]:
    """Split the corpus into a train and a validation half by character position.

    The vocabulary is built from the **whole** corpus, not from the training
    half. A character-level vocabulary has no ``<unk>`` bucket, so a symbol that
    happens to appear only in the tail would raise at encode time; building it
    once over everything keeps both index spaces identical and the split honest
    (no token in validation is unseen *as a symbol*, only as context).

    Args:
        text: Newline-normalised corpus.
        val_frac: Fraction held out, taken as the contiguous **tail**. A random
            per-character split would leak: neighbouring characters are the
            model's own context window.

    Returns:
        ``(vocab, train_tokens, val_tokens)``.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must lie in (0, 1), got {val_frac}")
    vocab = CharVocab(text)
    cut = int(len(text) * (1.0 - val_frac))
    return vocab, vocab.encode(text[:cut]), vocab.encode(text[cut:])


def get_batch(tokens: np.ndarray, batch: int, block: int, rng: np.random.Generator):
    """Sample ``batch`` random windows and their one-position-shifted targets."""
    starts = rng.integers(0, len(tokens) - block - 1, size=batch)
    x = np.stack([tokens[s : s + block] for s in starts])
    y = np.stack([tokens[s + 1 : s + 1 + block] for s in starts])
    return x, y


def eval_starts(tokens: np.ndarray, block: int, limit: int | None) -> np.ndarray:
    """Deterministic, non-overlapping window starts covering the split.

    Non-overlapping and fixed, so the same command reports the same validation
    number twice -- a validation loss estimated from freshly drawn random
    windows is a random variable and cannot be compared across runs at the third
    decimal place. ``limit`` strides evenly across the whole split rather than
    truncating to its beginning, so the cheap in-training estimate is not biased
    towards one scene of the play.
    """
    starts = np.arange(0, len(tokens) - block - 1, block)
    if limit is not None and len(starts) > limit:
        starts = starts[:: max(1, len(starts) // limit)][:limit]
    return starts


def lstm_scored_positions(n_tokens: int, streams: int, block: int, warmup: int) -> np.ndarray:
    """Reconstruct exactly the target positions ``examples/char_lstm.py`` scores.

    That script lays ``streams`` equal segments over the validation split, runs
    each as a continuous BPTT stream, and skips the first ``warmup`` blocks of
    every stream so no token is scored from a cold hidden state. This mirrors
    its ``stream_layout`` + ``scored_positions`` arithmetic so the transformer
    can be scored on the *same tokens* rather than on a different sample of the
    same split -- comparing two models on two token sets is not a comparison.

    Args:
        n_tokens: Length of the validation split.
        streams: ``--val-streams`` from the LSTM run.
        block: ``--block-size`` from the LSTM run (its BPTT length, which need
            not equal this model's context window).
        warmup: ``--val-warmup-blocks`` from the LSTM run.

    Returns:
        Sorted target positions, as indices into the validation token array.
    """
    seg = n_tokens // streams
    n_blocks = (seg - 1) // block
    if n_blocks <= warmup:
        raise ValueError(
            f"{n_tokens} tokens / {streams} streams gives {n_blocks} blocks "
            f"of {block}, not more than the {warmup} warmup blocks"
        )
    starts = np.arange(streams, dtype=np.int64) * seg
    runs = [
        (starts + b * block)[:, None] + np.arange(block, dtype=np.int64)[None, :] + 1
        for b in range(warmup, n_blocks)
    ]
    return np.unique(np.concatenate([r.ravel() for r in runs]))


def _window_plan(positions: np.ndarray, block: int, stride: int) -> list[tuple[int, int]]:
    """Cover ``positions`` with ``(window_start, n_scored)`` sliding windows.

    A window ``tokens[s : s + block]`` predicts targets ``s+1 .. s+block``; only
    the **last** ``n_scored`` of those are counted, so every scored token carries
    at least ``block - stride + 1`` characters of context. Scoring a whole
    non-overlapping block instead would charge the model for position 0, which
    has one character of context, and that is not a property of the model -- it
    is a property of where the window happened to be cut.
    """
    plan: list[tuple[int, int]] = []
    runs = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
    for run in runs:
        lo, hi = int(run[0]), int(run[-1])
        p = lo
        while p <= hi:
            k = min(stride, hi - p + 1)
            start = p + k - 1 - block
            if start < 0:  # too close to the corpus start to fill a window
                start, k = 0, min(k, block)
            plan.append((start, k))
            p += k
    return plan


def evaluate_positions(
    model: nn.Module, tokens: np.ndarray, positions: np.ndarray, block: int, stride: int, batch: int
) -> tuple[float, int]:
    """Mean nats over exactly ``positions``, using sliding windows of ``block``.

    Returns ``(nats, n_scored)``. Token-weighted, so a short final chunk does
    not get the same weight as a full one.
    """
    plan = _window_plan(positions, block, stride)
    by_k: dict[int, list[int]] = {}
    for start, k in plan:
        by_k.setdefault(k, []).append(start)
    was_training = model.training
    model.eval()
    meter, scored = AverageMeter(), 0
    with sg.no_grad():
        for k, starts in sorted(by_k.items()):
            arr = np.asarray(starts, dtype=np.int64)
            for i in range(0, len(arr), batch):
                chunk = arr[i : i + batch]
                x = np.stack([tokens[s : s + block] for s in chunk])
                y = np.stack([tokens[s + 1 : s + 1 + block] for s in chunk])
                logits = model(x)[:, block - k :, :]
                meter.update(
                    nn.functional.cross_entropy(logits, y[:, block - k :]).item(), len(chunk) * k
                )
                scored += len(chunk) * k
    model.train(was_training)
    return meter.avg, scored


def evaluate(
    model: nn.Module, tokens: np.ndarray, block: int, batch: int, limit: int | None = None
) -> float:
    """Mean per-token cross-entropy in nats over fixed windows of ``tokens``.

    Token-weighted, not batch-weighted: the final batch is usually short and a
    plain mean over batches would silently over-weight it.
    """
    starts = eval_starts(tokens, block, limit)
    was_training = model.training
    model.eval()
    meter = AverageMeter()
    with sg.no_grad():
        for i in range(0, len(starts), batch):
            chunk = starts[i : i + batch]
            x = np.stack([tokens[s : s + block] for s in chunk])
            y = np.stack([tokens[s + 1 : s + 1 + block] for s in chunk])
            meter.update(model.loss(x, y).item(), len(chunk) * block)
    model.train(was_training)
    return meter.avg


# --------------------------------------------------------------------------- #
# model + sanity checks
# --------------------------------------------------------------------------- #
def build_model(vocab_size: int, args, rng: np.random.Generator) -> nn.GPT:
    return nn.GPT(
        vocab_size,
        args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        tie_weights=True,
        rng=rng,
    )


def loss_at_init_is_ln_v(vocab_size: int, args, x: np.ndarray, y: np.ndarray) -> float:
    """Zero the tied embedding so the logits are exactly uniform, then check ln(V).

    Zeroing matters. An untouched initialization gives a loss *near* ``ln(V)``
    that fluctuates by a few thousandths per batch and can legitimately land
    below it, so asserting on the raw value is a flaky test rather than a check.
    With ``wte`` zeroed the token embedding is zero **and** the tied LM head is
    zero, so the logits are identically zero and the loss is exactly ``ln(V)``
    to float64 precision -- which pins the head, the softmax and the reduction
    all at once. If this number is 1.0, or ln of anything but the vocabulary
    size, the bug is in the head or the loss and not in the training loop.
    """
    with sg.default_dtype(np.float64):
        probe = build_model(vocab_size, args, np.random.default_rng(0)).double()
        probe.wte.data[...] = 0.0
        with sg.no_grad():
            loss = probe.loss(x, y)
    return loss.item()


def attention_weights(model: nn.GPT, ids: np.ndarray, layer: int) -> np.ndarray:
    """Re-run the forward pass up to ``layer`` and return that block's weights.

    Returns ``(N, n_head, T, T)``. The blocks before ``layer`` are run in full,
    so these are the weights the model actually computes on this input -- not
    the weights of a freshly initialised block fed the raw embeddings.
    """
    if not 0 <= layer < model.n_layer:
        raise ValueError(f"layer must lie in [0, {model.n_layer}), got {layer}")
    was_training = model.training
    model.eval()
    try:
        with sg.no_grad():
            x = model.wte[ids] + model.wpe(ids.shape[1])
            for i, block in enumerate(model.blocks):
                if i == layer:
                    _, weights = block.attn(block.ln_1(x), return_weights=True)
                    return np.asarray(weights)
                x = block(x)
    finally:
        model.train(was_training)
    raise AssertionError("unreachable: layer index was validated above")


def check_attention_structure(model: nn.GPT, ids: np.ndarray) -> dict:
    """Verify rows sum to 1 and the causal mask leaves a lower-triangular matrix.

    Both are checked on the **trained** model, on a real sequence. Rows summing
    to 1 is the softmax's normalisation; the strictly-zero upper triangle is the
    mask actually doing its job. The second is the sharper test: a mask applied
    after the softmax, or with the wrong polarity, still gives rows that sum to
    1 and a loss that goes down -- while letting every position read its own
    future, which is how a character model ends up reporting an impossibly good
    validation loss.
    """
    stats = {"per_layer": []}
    worst_row, worst_tri = 0.0, 0.0
    for layer in range(model.n_layer):
        w = attention_weights(model, ids, layer)
        row_err = float(np.abs(w.sum(axis=-1) - 1.0).max())
        # k=1 excludes the diagonal: a token attending to itself is causal.
        tri = float(np.abs(np.triu(w, k=1)).max())
        stats["per_layer"].append(
            {"layer": layer, "row_sum_max_err": row_err, "upper_triangle_max_abs": tri}
        )
        worst_row, worst_tri = max(worst_row, row_err), max(worst_tri, tri)
    stats["row_sum_max_err"] = worst_row
    stats["upper_triangle_max_abs"] = worst_tri
    stats["shape"] = list(attention_weights(model, ids, 0).shape)
    return stats


# --------------------------------------------------------------------------- #
# comparisons
# --------------------------------------------------------------------------- #
def load_lstm_baseline(path: Path) -> dict | None:
    """Read the Tier-4 char-LSTM's ``history.json`` if that run exists.

    Returns ``None`` when the file is absent. The roadmap's Tier-5 claim is that
    attention beats the LSTM on identical data and budget; with no LSTM run on
    disk that claim is simply untested here, and saying so is the only honest
    option -- a remembered number from another machine is not a measurement.
    """
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # a half-written file mid-run
        print(f"  could not read {path}: {exc}")
        return None
    val = None
    # "val_nats" first: that is the key examples/char_lstm.py actually writes,
    # and it is the post-training number rather than the last curve point.
    for key in ("val_nats", "val_loss_nats", "val_loss", "final_val_loss", "test_loss"):
        if isinstance(blob.get(key), (int, float)):
            val = float(blob[key])
            break
    if val is None:
        final = blob.get("final")
        if isinstance(final, dict):
            for key in ("val_loss", "val_loss_nats", "test_loss"):
                if isinstance(final.get(key), (int, float)):
                    val = float(final[key])
                    break
    if val is None:
        hist = blob.get("history", {})
        series = hist.get("val_loss") if isinstance(hist, dict) else None
        if isinstance(series, list) and series:
            val = float(series[-1])
    if val is None:
        print(f"  {path} exists but has no recognisable validation loss field")
        return None
    cfg = blob.get("config") if isinstance(blob.get("config"), dict) else {}
    return {
        "path": str(path),
        "val_loss_nats": val,
        "val_bits_per_char": val * NATS_TO_BITS,
        "trigram_val_nats": blob.get("trigram_val_nats"),
        "seconds": blob.get("seconds"),
        "parameters": blob.get("parameters"),
        "val_tokens_scored": blob.get("val_tokens_scored"),
        "eval_protocol": {
            "block_size": cfg.get("block_size"),
            "val_streams": cfg.get("val_streams"),
            "val_warmup_blocks": cfg.get("val_warmup_blocks"),
            "iters": cfg.get("iters"),
            "batch_size": cfg.get("batch_size"),
            "seed": cfg.get("seed"),
        },
        "raw_keys": sorted(blob),
    }


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
def smoke(args) -> int:
    """Overfit 8 sequences before spending twelve minutes on the real run.

    A transformer that cannot drive 8 sequences of 64 characters to a near-zero
    loss is broken, not undertrained, and finding that out in a minute beats
    finding it out after the full budget.
    """
    rng = np.random.default_rng(args.seed)
    text = tiny_shakespeare(root=str(ROOT / "data" / "text"), verbose=False)
    vocab, train_tok, _ = split_corpus(text, args.val_frac)
    x, y = get_batch(train_tok, 8, args.block_size, rng)

    l0 = loss_at_init_is_ln_v(vocab.size, args, x, y)
    print(f"loss at init with wte zeroed: {l0:.15f}")
    print(f"                    ln({vocab.size}) = {math.log(vocab.size):.15f}")
    assert abs(l0 - math.log(vocab.size)) < 1e-12, "uniform logits must give exactly ln(V)"

    model = build_model(vocab.size, args, rng)
    opt = optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.0)
    for step in range(200):
        loss = model.loss(x, y)
        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if step % 50 == 0 or step == 199:
            print(f"  step {step:4d}  loss {loss.item():.6f} nats")
    final = loss.item()
    print(f"\noverfit 8 sequences -> loss {final:.4f} nats", "OK" if final < 0.1 else "FAILED")
    return 0 if final < 0.1 else 1


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def finalize(model, opt, args, vocab, train_tok, val_tok, ln_v, l0, history, elapsed, out) -> dict:
    """Score, compare, sample, save, and plot. Returns the blob written to disk.

    Split out of :func:`main` so ``--rescore`` can re-run every measurement
    against a saved checkpoint without retraining -- the Tier-4 baseline this
    tier is graded against comes from a separate script and may land after this
    one has finished. Re-scoring is not re-tuning: the weights are frozen and
    only the comparison is recomputed.
    """
    # Final numbers: the WHOLE validation split, not the strided subset used for
    # the curve, and a matching-size slice of train so the gap is comparable.
    print("evaluating on the full validation split ...", flush=True)
    n_val_windows = len(eval_starts(val_tok, args.block_size, None))
    val_loss = evaluate(model, val_tok, args.block_size, args.eval_batch, None)
    train_eval = evaluate(model, train_tok, args.block_size, args.eval_batch, n_val_windows)
    print(
        f"\nfinal validation loss : {val_loss:.4f} nats "
        f"= {val_loss * NATS_TO_BITS:.4f} bits/char  ({n_val_windows} windows)"
    )
    print(
        f"train loss (same size): {train_eval:.4f} nats = {train_eval * NATS_TO_BITS:.4f} bits/char"
    )
    print(f"random-init reference : {ln_v:.4f} nats = {ln_v * NATS_TO_BITS:.4f} bits/char")
    print(
        f"improvement over random init: {ln_v - val_loss:.4f} nats "
        f"({math.exp(ln_v - val_loss):.2f}x lower perplexity)"
    )
    print(f"parameters            : {model.num_parameters():,}")

    # Structural properties of the TRAINED model, on a real validation sequence.
    ids = np.stack([val_tok[: args.block_size], val_tok[args.block_size : 2 * args.block_size]])
    attn = check_attention_structure(model, ids)
    print(f"\nattention on a real {tuple(attn['shape'])} batch, all {model.n_layer} layers:")
    print(f"  max |row sum - 1|    = {attn['row_sum_max_err']:.3e}  (softmax normalisation)")
    print(f"  max |upper triangle| = {attn['upper_triangle_max_abs']:.3e}  (causal mask)")
    assert attn["row_sum_max_err"] < 1e-6, "attention rows must sum to 1"
    assert attn["upper_triangle_max_abs"] == 0.0, "causal attention must be lower-triangular"
    print("  -> rows sum to 1 and the upper triangle is exactly zero.")

    # Comparison against the Tier-4 LSTM, if that run exists on disk.
    lstm = load_lstm_baseline(ROOT / args.lstm_history)
    matched: dict | None = None
    print("\nvs the Tier-4 char-LSTM:")
    if lstm is None:
        print(
            f"  {ROOT / args.lstm_history} does not exist -- there is no LSTM run to "
            "compare against. The roadmap's 'attention beats the LSTM' claim is "
            "UNTESTED here; re-run with --rescore once that file lands."
        )
    else:
        proto = lstm["eval_protocol"]
        print(
            f"  LSTM val {lstm['val_loss_nats']:.4f} nats "
            f"({lstm['val_bits_per_char']:.4f} bits/char), "
            f"{lstm['val_tokens_scored']} tokens scored"
        )
        print(
            f"  GPT  val {val_loss:.4f} nats ({val_loss * NATS_TO_BITS:.4f} bits/char), "
            "own protocol (non-overlapping windows, every position scored)"
        )
        if proto.get("block_size") and proto.get("val_streams"):
            # The two scripts score different token sets by default: the LSTM
            # skips a warmup block per stream so nothing is predicted from a cold
            # hidden state, while this one scores every position of every window,
            # including the one that has a single character behind it. Neither is
            # wrong, the difference is worth order 0.1 nats, and comparing across
            # it would be comparing protocols rather than models.
            pos = lstm_scored_positions(
                len(val_tok),
                int(proto["val_streams"]),
                int(proto["block_size"]),
                int(proto.get("val_warmup_blocks") or 0),
            )
            stride = max(1, args.block_size // 2)
            nats, scored = evaluate_positions(
                model, val_tok, pos, args.block_size, stride, args.eval_batch
            )
            delta = lstm["val_loss_nats"] - nats
            gpt_tokens = args.iters * args.batch_size * args.block_size
            lp = lstm["eval_protocol"]
            lstm_tokens = (
                (lp.get("iters") or 0) * (lp.get("batch_size") or 0) * (lp.get("block_size") or 0)
            ) or None
            matched = {
                "protocol": (
                    f"the LSTM's scored positions; GPT context {args.block_size} "
                    f"with stride {stride}, so every scored token has at least "
                    f"{args.block_size - stride + 1} characters of context"
                ),
                "positions": len(pos),
                "gpt_tokens_scored": int(scored),
                "gpt_val_nats": nats,
                "gpt_val_bits_per_char": nats * NATS_TO_BITS,
                "lstm_val_nats": lstm["val_loss_nats"],
                "delta_nats": delta,
                "winner": "gpt" if delta > 0 else "lstm",
                # "Identical budget" is a claim, so it is recorded rather than
                # asserted: the two scripts ship different defaults and a win by
                # the model that saw twice the characters is not a win for its
                # architecture.
                "budget": {
                    "gpt_train_tokens": int(gpt_tokens),
                    "lstm_train_tokens": lstm_tokens,
                    "gpt_seconds": elapsed,
                    "lstm_seconds": lstm.get("seconds"),
                    "gpt_parameters": model.num_parameters(),
                    "lstm_parameters": lstm.get("parameters"),
                },
            }
            print(f"  matched protocol ({len(pos)} identical target positions):")
            print(f"    GPT  {nats:.4f} nats ({nats * NATS_TO_BITS:.4f} bits/char)")
            print(
                f"    LSTM {lstm['val_loss_nats']:.4f} nats "
                f"({lstm['val_bits_per_char']:.4f} bits/char)"
            )
            print(
                f"    -> {'GPT' if delta > 0 else 'LSTM'} wins by {abs(delta):.4f} nats "
                f"({abs(delta) * NATS_TO_BITS:.4f} bits/char)"
            )
            print(
                "    caveat: after its warmup the LSTM carries unbounded context while "
                f"this model sees at most {args.block_size} characters. That is an "
                "architectural difference, not a protocol one."
            )
            bt = matched["budget"]
            print(
                f"    budgets: GPT {bt['gpt_train_tokens']:,} train tokens / "
                f"{bt['gpt_seconds']:.0f}s / {bt['gpt_parameters']:,} params vs "
                f"LSTM {bt['lstm_train_tokens']:,} / {bt['lstm_seconds']:.0f}s / "
                f"{bt['lstm_parameters']:,}"
            )
            if bt["lstm_train_tokens"] and bt["gpt_train_tokens"] < bt["lstm_train_tokens"]:
                print(
                    "    -> NOT an equal-budget comparison; the LSTM saw "
                    f"{bt['lstm_train_tokens'] / bt['gpt_train_tokens']:.2f}x the "
                    "characters. Re-run with --iters scaled to match before reading "
                    "this as a statement about the architectures."
                )
        else:
            print(
                "  the LSTM history records no eval protocol (block_size / val_streams), "
                "so its tokens cannot be reproduced and only the headline numbers are "
                "comparable -- treat the gap as indicative, not matched."
            )

    # Sampling.
    prompt = vocab.encode("\n")[None, :]
    sample_rng = np.random.default_rng(args.seed + 1)
    tokens = model.generate(
        prompt, args.sample_tokens, temperature=args.temperature, rng=sample_rng
    )
    sample = vocab.decode(tokens[0])
    print(f"\n--- sample, {args.sample_tokens} tokens at temperature {args.temperature} ---")
    print(sample)
    print("--- end sample ---")

    if opt is not None:
        save_checkpoint(
            out / "weights.npz",
            model,
            opt,
            meta={
                "seed": args.seed,
                "iters": args.iters,
                "batch_size": args.batch_size,
                "block_size": args.block_size,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
                "lr": args.lr,
                "val_loss_nats": val_loss,
                "train_seconds": elapsed,
            },
        )
    (out / "sample.txt").write_text(sample, encoding="utf-8")
    blob = {
        "config": vars(args),
        "vocab_size": vocab.size,
        "parameters": model.num_parameters(),
        "split": {
            "corpus": "tiny-shakespeare (newline-normalised)",
            "scheme": "contiguous tail held out; vocab built on the full corpus",
            "val_frac": args.val_frac,
            "train_chars": len(train_tok),
            "val_chars": len(val_tok),
            "eval_windows": "non-overlapping, stride == block_size, every position scored",
            "n_val_windows": int(n_val_windows),
        },
        "history": history,
        "final": {
            "val_loss_nats": val_loss,
            "val_bits_per_char": val_loss * NATS_TO_BITS,
            "train_loss_nats": train_eval,
            "train_bits_per_char": train_eval * NATS_TO_BITS,
        },
        "reference": {
            "random_init_nats": ln_v,
            "random_init_bits_per_char": ln_v * NATS_TO_BITS,
            "loss_at_init_zeroed_wte": l0,
        },
        "attention_check": attn,
        "comparison_char_lstm": lstm,
        "comparison_matched_positions": matched,
        "seconds": elapsed,
        "ms_per_iter": (elapsed / args.iters * 1e3) if elapsed else None,
        "seed": args.seed,
        "sample_temperature": args.temperature,
    }
    (out / "history.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    what = (
        "weights.npz, history.json and sample.txt"
        if opt is not None
        else "history.json and sample.txt (weights untouched)"
    )
    print(f"\nsaved {what} in {out}")

    if not args.no_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            assets = ROOT / "assets"
            assets.mkdir(exist_ok=True)

            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.plot(
                history["iter"],
                history["train_loss"],
                marker="o",
                ms=3,
                lw=1.5,
                label="train (mean over the interval)",
            )
            ax.plot(
                history["iter"],
                history["val_loss"],
                marker="o",
                ms=3,
                lw=1.5,
                label="val (fixed windows)",
            )
            ax.axhline(
                ln_v,
                ls="--",
                lw=1.0,
                color="grey",
                label=f"random init, ln({vocab.size}) = {ln_v:.2f}",
            )
            if lstm is not None:
                ax.axhline(
                    lstm["val_loss_nats"],
                    ls=":",
                    lw=1.2,
                    color="firebrick",
                    label=f"char-LSTM val = {lstm['val_loss_nats']:.2f}",
                )
            ax.set_xlabel("iteration")
            ax.set_ylabel("cross-entropy (nats/char)")
            ax.set_title(
                f"mini-GPT on tiny-shakespeare -- final val {val_loss:.3f} nats "
                f"({val_loss * NATS_TO_BITS:.2f} bits/char)"
            )
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            # Second axis in bits/char so both conventions are readable at once.
            bits = ax.secondary_yaxis(
                "right", functions=(lambda v: v * NATS_TO_BITS, lambda v: v / NATS_TO_BITS)
            )
            bits.set_ylabel("bits/char")
            fig.tight_layout()
            fig.savefig(assets / "mini_gpt_training.png", dpi=140)
            plt.close(fig)

            head, layer = 0, model.n_layer - 1
            w = attention_weights(model, ids, layer)[0, head]
            fig, ax = plt.subplots(figsize=(5.4, 4.6))
            im = ax.imshow(w, cmap="magma", interpolation="nearest")
            ax.set_xlabel("key position")
            ax.set_ylabel("query position")
            ax.set_title(f"layer {layer}, head {head} -- upper triangle exactly 0")
            fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")
            fig.tight_layout()
            fig.savefig(assets / "mini_gpt_attention.png", dpi=140)
            plt.close(fig)
            print(f"wrote plots to {assets}")
        except ImportError as exc:
            print(f"skipping plots: {exc}")
    return blob


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    # 2e-3 was chosen by a 300-iteration probe over {1, 2, 3, 5}e-3 that read the
    # *training* loss only (2.428 / 2.364 / 2.399 / 2.476 nats). The validation
    # split was never consulted while tuning, so the number this script reports
    # is not measured on data it was tuned on.
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument(
        "--eval-blocks",
        type=int,
        default=96,
        help="validation windows for the in-training curve; the final "
        "number is always measured on the whole split",
    )
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--sample-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="overfit 8 sequences and stop")
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="reload the saved checkpoint and redo every measurement without "
        "retraining; use it when the LSTM baseline lands after this run",
    )
    ap.add_argument("--out", type=str, default="runs/mini_gpt")
    ap.add_argument("--lstm-history", type=str, default="runs/char_lstm/history.json")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        return smoke(args)

    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print("loading tiny-shakespeare ...")
    text = tiny_shakespeare(root=str(ROOT / "data" / "text"), verbose=True)
    vocab, train_tok, val_tok = split_corpus(text, args.val_frac)
    print(f"  {len(text):,} characters, vocab {vocab.size}")
    print(
        f"  train {len(train_tok):,} tokens   val {len(val_tok):,} tokens "
        f"(contiguous tail {args.val_frac:.0%})"
    )
    assert "\r" not in text, "CRLF in the corpus would add a 66th character to the vocab"

    # Sanity check 1, before anything is trained: the loss and the softmax agree.
    ln_v = math.log(vocab.size)
    probe_x, probe_y = get_batch(train_tok, 4, args.block_size, np.random.default_rng(0))
    l0 = loss_at_init_is_ln_v(vocab.size, args, probe_x, probe_y)
    print(f"\nloss at init (wte zeroed) = {l0:.15f}")
    print(f"                  ln({vocab.size}) = {ln_v:.15f}")
    assert abs(l0 - ln_v) < 1e-12, "uniform logits must give exactly ln(V)"
    print(
        f"  -> exact to 1e-12. Random-init reference: {ln_v:.4f} nats "
        f"= {ln_v * NATS_TO_BITS:.4f} bits/char."
    )

    model = build_model(vocab.size, args, rng)
    if args.rescore:
        # The weights are frozen; only the measurements are recomputed. The
        # training history and wall-clock are carried over from the run that
        # produced the checkpoint rather than invented.
        meta = load_checkpoint(out / "weights.npz", model)
        old = json.loads((out / "history.json").read_text(encoding="utf-8"))
        print(
            f"\nrescoring {out / 'weights.npz'} (trained {meta.get('iters')} iters, "
            f"seed {meta.get('seed')}) -- no training in this pass"
        )
        finalize(
            model,
            None,
            args,
            vocab,
            train_tok,
            val_tok,
            ln_v,
            old.get("reference", {}).get("loss_at_init_zeroed_wte", l0),
            old.get("history", {}),
            float(old.get("seconds") or 0.0),
            out,
        )
        return 0

    print(f"\n{model}\n")
    print(model.summary())

    opt = optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.weight_decay
    )
    sched = optim.WarmupCosine(opt, args.warmup, args.iters, min_lr_ratio=args.min_lr_ratio)

    history: dict[str, list[float]] = {
        "iter": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "grad_norm": [],
    }
    print(
        f"\ntraining {args.iters} iters, batch {args.batch_size} x block {args.block_size} "
        f"= {args.batch_size * args.block_size} tokens/step"
    )
    print(
        f"AdamW lr={args.lr} wd={args.weight_decay} betas=(0.9, {args.beta2}), "
        f"warmup {args.warmup} then cosine to {args.min_lr_ratio:g}x, clip {args.clip}"
    )

    meter = AverageMeter()
    # The fallback bar writes '\r' and never '\n' until close(), so a redirected
    # log would get one enormous line; disable it when stderr is not a terminal.
    bar = ProgressBar(total=args.iters, desc="train", disable=not sys.stderr.isatty())
    t_start = time.perf_counter()
    for it in range(1, args.iters + 1):
        x, y = get_batch(train_tok, args.batch_size, args.block_size, rng)
        loss = model.loss(x, y)
        opt.zero_grad()
        loss.backward()
        # The returned norm is the pre-clip one -- the number worth logging,
        # because it is what says whether clipping ever fired.
        gnorm = optim.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        meter.update(loss.item(), 1)
        bar.update()
        bar.set_postfix(loss=f"{meter.avg:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
        if it % args.eval_every == 0 or it == args.iters:
            val = evaluate(model, val_tok, args.block_size, args.eval_batch, args.eval_blocks)
            history["iter"].append(it)
            history["train_loss"].append(meter.avg)
            history["val_loss"].append(val)
            history["lr"].append(float(sched.get_last_lr()[0]))
            history["grad_norm"].append(float(gnorm))
            print(
                f"  iter {it:5d}  train {meter.avg:.4f}  val {val:.4f} nats "
                f"({val * NATS_TO_BITS:.4f} bits/char)  |g| {gnorm:.2f}",
                flush=True,
            )
            # Reset so each printed train loss covers only its own interval; a
            # running mean since step 1 would lag the curve by half the run.
            meter = AverageMeter()
    bar.close()
    elapsed = time.perf_counter() - t_start

    print(
        f"\ntrained in {elapsed:.1f}s ({elapsed / 60:.1f} min) on CPU, "
        f"{elapsed / args.iters * 1e3:.0f} ms/iter"
    )

    finalize(model, opt, args, vocab, train_tok, val_tok, ln_v, l0, history, elapsed, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
