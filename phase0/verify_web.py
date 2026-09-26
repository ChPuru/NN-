"""Drive ``verify_web.js``: does the browser demo agree with the NumPy model?

    python3 phase0/verify_web.py                    # 300 real MNIST test digits
    python3 phase0/verify_web.py --n 1000 --seed 1

The export check in ``export_web.py`` proves the *weights* survive the trip into
JavaScript. It says nothing about the other half of the demo, which is where the
demo actually fails: the preprocessing. A drawing pad hands the network a
336x336 canvas; MNIST hands it a 28x28 image whose ink has been scaled to a
20-pixel box and centred by centre of mass. Skip that translation and a
genuinely 97.5%-accurate model reads hand-drawn input at chance-adjacent rates,
with no error anywhere -- the model is simply being shown a distribution it has
never seen.

So this script renders real MNIST test digits onto a simulated pad at arbitrary
sizes and positions, runs the shipped JavaScript over them under node, and
compares three things:

1. **Agreement.** The JS pipeline's prediction against the NumPy model's
   prediction on the same digit. Below ~99% means the JS forward or the
   preprocessing disagrees with training.
2. **The mnistify contrast.** Accuracy on those rendered pads with the
   preprocessing and with it removed. That gap is the entire argument for the
   hundred lines of canvas code in ``app.js``.
3. **Numerics.** NumPy re-scores the exact 784-vectors the JavaScript built, so
   the forward pass is compared on identical inputs rather than on inputs that
   merely ought to be identical.

Results land in ``phase0/web_verification.json`` next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from export_web import numpy_forward, parse_model_js
from mnist import load_mnist

__all__ = ["build_job", "main", "run_node", "score_tensors"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEB = os.path.join(_HERE, "web")

#: The drawing pad in ``index.html`` is 336x336 in canvas coordinates.
PAD = 336


def build_job(
    n: int, seed: int, rng: np.random.Generator, pad: int = PAD
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Assemble the digits, placements and NumPy reference predictions.

    The placements are drawn here rather than in JavaScript so that one seed
    controls the whole experiment and the exact pads are reconstructible from
    this file alone.

    Args:
        n: Number of test digits.
        seed: Recorded in the job for provenance.
        rng: Explicit generator for the sample and the placements.
        pad: Canvas side length in pixels.

    Returns:
        ``(job, x_test_rows, labels)`` -- the job dict handed to node, the
        ``(n, 784)`` float64 MNIST rows in ``[0, 1]``, and the true labels.
    """
    _, _, x_te, y_te = load_mnist(verbose=False)
    idx = rng.choice(len(x_te), size=n, replace=False)
    x, y = x_te[idx], y_te[idx]
    digits = np.round(x * 255.0).astype(np.uint8)

    # Size and position of the drawn digit on the pad. The range is wide on
    # purpose: a demo that only works when the user fills the canvas edge to
    # edge is a demo that only works for the person who built it.
    sizes = rng.integers(120, 261, size=n)
    ox = np.array([rng.integers(0, pad - s + 1) for s in sizes])
    oy = np.array([rng.integers(0, pad - s + 1) for s in sizes])

    with open(os.path.join(_WEB, "model.js"), encoding="utf-8") as fh:
        model = parse_model_js(fh.read())
    W1 = model["W1"].reshape(model["nIn"], model["nHidden"])
    W2 = model["W2"].reshape(model["nHidden"], model["nOut"])
    # float32 input, because the browser holds the tensor in a Float32Array.
    preds = np.array(
        [
            int(
                np.argmax(
                    numpy_forward(
                        np.asarray(row, dtype=np.float32).astype(np.float64),
                        W1,
                        model["b1"],
                        W2,
                        model["b2"],
                    )[0]
                )
            )
            for row in x
        ]
    )

    job = {
        "seed": seed,
        "padSize": pad,
        "digits": digits.tolist(),
        "labels": y.tolist(),
        "numpyPred": preds.tolist(),
        "placements": [[int(s), int(a), int(b)] for s, a, b in zip(sizes, ox, oy, strict=True)],
        "testIndices": idx.tolist(),
    }
    return job, x, y


def run_node(job: dict[str, object], node: str) -> tuple[dict[str, object], str, int]:
    """Write the job to a temporary file and run ``verify_web.js`` over it.

    Returns:
        ``(result, stdout, returncode)``.
    """
    work = tempfile.mkdtemp(prefix="phase0-verify-")
    try:
        job_path = os.path.join(work, "job.json")
        out_path = os.path.join(work, "out.json")
        with open(job_path, "w", encoding="utf-8") as fh:
            json.dump(job, fh)
        proc = subprocess.run(
            [node, os.path.join(_HERE, "verify_web.js"), job_path, out_path],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if not os.path.exists(out_path):
            raise RuntimeError(f"verify_web.js produced no output:\n{proc.stdout}\n{proc.stderr}")
        with open(out_path, encoding="utf-8") as fh:
            return json.load(fh), proc.stdout + proc.stderr, proc.returncode
    finally:
        shutil.rmtree(work, ignore_errors=True)


def score_tensors(tensors: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Re-run the NumPy forward on the exact vectors the JavaScript produced.

    This is the check with no wiggle room in it: same weights, same inputs, two
    independent implementations. Anything but exact argmax agreement is a bug in
    one of them.

    Returns:
        ``(predictions, probabilities)``.
    """
    with open(os.path.join(_WEB, "model.js"), encoding="utf-8") as fh:
        model = parse_model_js(fh.read())
    W1 = model["W1"].reshape(model["nIn"], model["nHidden"])
    W2 = model["W2"].reshape(model["nHidden"], model["nOut"])
    probs = []
    for row in tensors:
        x = np.asarray(row, dtype=np.float32).astype(np.float64)
        probs.append(numpy_forward(x, W1, model["b1"], W2, model["b2"])[1])
    probs = np.asarray(probs)
    return probs.argmax(axis=1), probs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=300, help="number of MNIST test digits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_HERE, "web_verification.json"))
    args = ap.parse_args(argv)

    node = shutil.which("node")
    if node is None:
        print("node is not installed; the browser numerics cannot be verified", file=sys.stderr)
        return 2
    if not os.path.exists(os.path.join(_WEB, "model.js")):
        print("web/model.js is missing; run phase0/export_web.py first", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    job, x, y = build_job(args.n, args.seed, rng)
    result, output, code = run_node(job, node)
    print(output.rstrip())

    js_pred = np.asarray(result["predsPrep"])
    np_pred, _ = score_tensors(result["tensors"])
    same = int(np.sum(js_pred == np_pred))
    print("")
    print("NumPy re-scoring the exact 784-vectors JavaScript built")
    print(f"  argmax agreement : {100 * same / args.n:.2f}%  ({same}/{args.n})")

    # Where the residual pipeline disagreement comes from. Rendering a 28x28
    # digit up to a few hundred pixels and letting mnistify pull it back down to
    # a 20-pixel box is not information-preserving: the two resamples leave a
    # few hundredths of intensity per pixel. Measuring that separates "the JS
    # disagrees with training" -- which would be a bug -- from "a borderline
    # digit moved across the decision boundary", which is arithmetic.
    tensors = np.asarray(result["tensors"])
    ref_pred = np.asarray(job["numpyPred"])
    sizes = np.asarray([p[0] for p in job["placements"]])
    pixdiff = np.abs(tensors - x).mean(axis=1)
    print("")
    print("round-trip cost of drawing the digit and re-extracting it")
    print(
        f"  mean |x_js - x_mnist| per pixel : median {np.median(pixdiff):.4f}   "
        f"p95 {np.percentile(pixdiff, 95):.4f}   max {pixdiff.max():.4f}"
    )
    print(
        f"  tensor mean                     : MNIST {x.mean():.4f}   JS pads {tensors.mean():.4f}"
    )
    print(f"  accuracy on the clean test rows : NumPy {100 * (ref_pred == y).mean():.2f}%")
    print(f"  accuracy on the rendered pads   : JS    {100 * result['accPrep'] / args.n:.2f}%")
    disagree = np.flatnonzero(js_pred != ref_pred)
    print(f"  the {len(disagree)} disagreements, digit by digit:")
    for i in disagree:
        who = (
            "JS right"
            if js_pred[i] == y[i]
            else ("NumPy right" if ref_pred[i] == y[i] else "both wrong")
        )
        print(
            f"    label {y[i]}  NumPy {ref_pred[i]}  JS {js_pred[i]}  "
            f"drawn at {sizes[i]}px  pixdiff {pixdiff[i]:.4f}  ({who})"
        )

    summary = {
        "n": args.n,
        "seed": args.seed,
        "node": result["node"],
        "pad": PAD,
        "agreement_js_pipeline_vs_numpy": result["agree"] / args.n,
        "agreement_without_mnistify": result["agreeNaive"] / args.n,
        "accuracy_with_mnistify": result["accPrep"] / args.n,
        "accuracy_without_mnistify": result["accNaive"] / args.n,
        "forward_argmax_agreement_same_inputs": same / args.n,
        "numpy_accuracy_on_clean_rows": float((ref_pred == y).mean()),
        "median_abs_pixel_reconstruction_error": float(np.median(pixdiff)),
        "p95_abs_pixel_reconstruction_error": float(np.percentile(pixdiff, 95)),
        "max_long_side_error_px": result["maxLongSideErr"],
        "max_paste_offset_error_px": result["maxOffsetErr"],
        "max_centre_of_mass_error_px": result["maxComErr"],
        "heaviest_fed_tensor_mean": result["worstMean"],
        "traps": {name: bool(ok) for name, ok in result["traps"]},
        "node_exit_code": code,
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {args.out}")

    # Two different kinds of claim, gated differently.
    #
    # The forward pass on identical inputs is a claim about code: two
    # implementations, one set of weights, so anything but *every* digit
    # agreeing is a bug and the threshold is 100%.
    #
    # The end-to-end pipeline rate is a claim about a lossy round trip, and the
    # honest floor is set by what the round trip costs, not by what would read
    # well. 0.98 is deliberately below the measured rate so the check has room
    # to be a regression guard rather than a coin flip; the number it actually
    # produces is printed above and written to the JSON, and a drop toward the
    # low nineties would mean the preprocessing had genuinely drifted.
    pipeline = result["agree"] / args.n
    ok = code == 0 and same == args.n and pipeline >= 0.98
    print("\nVERIFIED" if ok else "\nVERIFICATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
