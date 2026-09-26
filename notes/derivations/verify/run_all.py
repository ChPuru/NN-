"""Run every derivation verifier and write ``results.json`` beside this file.

Each derivation in ``notes/derivations/`` states an expression; each verifier
next to this script compares that expression against the framework's actual
gradient and, wherever a step size makes sense, against central differences.
This runner is what makes the whole directory one command:

    PYTHONPATH=src python3 notes/derivations/verify/run_all.py

Every number in the Markdown files comes out of the JSON this writes. With a
fixed ``--seed`` the run is bit-reproducible with two named exceptions, both in
``01``: a wall-clock ratio and a wall-clock duration, which depend on machine
load. They are listed under ``nondeterministic`` in the JSON so no one quotes
them as if they were.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
from _harness import Check, report

HERE = Path(__file__).resolve().parent

MODULES = [
    ("01-backprop-basics", "verify_01_backprop_basics"),
    ("02-softmax-cross-entropy", "verify_02_softmax_cross_entropy"),
    ("03-layernorm", "verify_03_layernorm"),
    ("04-batchnorm", "verify_04_batchnorm"),
    ("05-conv2d", "verify_05_conv2d"),
    ("06-lstm", "verify_06_lstm"),
    ("07-attention", "verify_07_attention"),
    ("08-adam", "verify_08_adam"),
    ("09-gradient-checking", "verify_09_gradient_checking"),
    ("10-parallel-scan", "verify_10_parallel_scan"),
    ("11-quantization", "verify_11_quantization"),
    ("12-parameter-shift", "verify_12_parameter_shift"),
    ("13-vae", "verify_13_vae"),
]

#: Wall-clock measurements. Real, but not reproducible to the digit.
NONDETERMINISTIC = ["01-backprop-basics/reverse_vs_forward_ratio"]


def main() -> int:
    """Run everything, print one table per file, write ``results.json``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default=str(HERE / "results.json"))
    args = ap.parse_args()

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    results: dict[str, dict] = {}
    failed = 0
    total = 0
    for title, modname in MODULES:
        checks: list[Check] = importlib.import_module(modname).run(args.seed)
        failed += report(title, checks, args.verbose)
        total += len(checks)
        results[title] = {
            c.name: {
                "measured": c.measured,
                "bound": c.bound,
                "kind": c.kind,
                "note": c.note,
                "ok": c.ok,
                **c.extras,
            }
            for c in checks
        }

    n_bad = sum(1 for f in results.values() for c in f.values() if not c["ok"])
    payload = {
        "seed": args.seed,
        "checks": total,
        "failed": n_bad,
        "nondeterministic": NONDETERMINISTIC,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "files": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{total - n_bad}/{total} checks passed across {len(MODULES)} derivations")
    print(f"wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
