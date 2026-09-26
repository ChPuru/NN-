"""Shared plumbing for the derivation verifiers: path setup, a check record, a CLI.

Every script in this directory answers one question -- *does the expression
derived in the neighbouring Markdown file agree with what the framework
actually computes?* -- so they all have the same shape: a ``run(seed)`` that
returns a list of :class:`Check`, and a CLI that prints them and exits non-zero
if any failed. Keeping that shape in one place is what lets ``run_all.py``
aggregate every file into a single results JSON.

The checks are deliberately *two-sided* wherever it is possible: the derived
expression is compared both against the framework's hand-written backward and
against central differences of the forward pass. Agreeing with the code alone
would only prove the derivation copies the code.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@dataclass
class Check:
    """One measured quantity and the bound the derivation predicts for it.

    Attributes:
        name: Short identifier, unique within a file.
        measured: The number actually observed. Never a target, never a guess.
        bound: The threshold ``measured`` is compared against.
        kind: ``"below"``, ``"above"``, or ``"close"`` (relative to ``bound``).
        note: One line saying what the number means.
    """

    name: str
    measured: float
    bound: float
    kind: str = "below"
    note: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        m, b = float(self.measured), float(self.bound)
        if m != m:  # NaN never passes, whatever the comparison
            return False
        if self.kind == "below":
            return m <= b
        if self.kind == "above":
            return m >= b
        if self.kind == "close":
            return abs(m - b) <= 1e-9 * max(1.0, abs(b))
        raise ValueError(f"unknown check kind {self.kind!r}")


def report(title: str, checks: Sequence[Check], verbose: bool = False) -> int:
    """Print a check table and return a process exit code."""
    print(f"\n=== {title} ===")
    width = max(len(c.name) for c in checks) if checks else 4
    for c in checks:
        flag = "ok  " if c.ok else "FAIL"
        rel = {"below": "<=", "above": ">=", "close": "=="}[c.kind]
        print(f"  [{flag}] {c.name:<{width}}  {c.measured:>12.6g} {rel} {c.bound:<10.4g}  {c.note}")
        if verbose and c.extras:
            for k, v in c.extras.items():
                print(f"           {k}: {v}")
    bad = [c.name for c in checks if not c.ok]
    print(f"  {len(checks) - len(bad)}/{len(checks)} passed" + (f"; failed: {bad}" if bad else ""))
    return 1 if bad else 0


def cli(run: Callable[[int], list[Check]], doc: str, title: str) -> int:
    """Standard CLI: ``--seed``, ``--verbose``, ``--json PATH``."""
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", type=str, default=None, help="write the measured numbers here")
    args = ap.parse_args()
    checks = run(args.seed)
    code = report(title, checks, args.verbose)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    c.name: {
                        "measured": c.measured,
                        "bound": c.bound,
                        "kind": c.kind,
                        "note": c.note,
                        "ok": c.ok,
                        **c.extras,
                    }
                    for c in checks
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return code
