"""Run the derivation verifiers as part of the suite.

`notes/derivations/` is the project's answer to the "Yours" property in
`ROADMAP.md`: every non-obvious backward pass derived by hand rather than
copied. A derivation that has drifted out of agreement with the code is worse
than no derivation at all -- it is a confident, wrong explanation of what the
software does. So the prose ships with executable checks, and those checks run
here.

Each verifier re-derives its result independently (from the algebra in the
matching Markdown file) and compares against the framework's actual gradient.
A failure means the prose and the code disagree, and one of them is wrong.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFY_DIR = ROOT / "notes" / "derivations" / "verify"


def _verifier_scripts() -> list[Path]:
    if not VERIFY_DIR.exists():
        return []
    return sorted(VERIFY_DIR.glob("verify_*.py"))


pytestmark = pytest.mark.skipif(
    not _verifier_scripts(), reason="notes/derivations/verify/ not present"
)


@pytest.mark.slow
def test_every_derivation_agrees_with_the_code():
    """The whole battery, in one subprocess, with its JSON summary asserted."""
    runner = VERIFY_DIR / "run_all.py"
    if not runner.exists():
        pytest.skip("run_all.py not present")

    proc = subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert proc.returncode == 0, (
        f"derivation verifiers failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}"
    )

    results = json.loads((VERIFY_DIR / "results.json").read_text(encoding="utf-8"))
    assert results["failed"] == 0, f"{results['failed']} derivation checks failed"
    assert results["checks"] > 50, f"only {results['checks']} checks ran; expected the full battery"


@pytest.mark.parametrize("script", _verifier_scripts(), ids=lambda p: p.stem)
def test_derivation_script_runs_clean(script):
    """Each verifier individually, so a failure names the derivation it belongs to."""
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert proc.returncode == 0, (
        f"{script.name} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-1500:]}"
    )


def test_every_derivation_note_has_a_verifier():
    """Prose without an executable check is exactly what this directory is not
    allowed to become."""
    notes = sorted(p for p in (ROOT / "notes" / "derivations").glob("*.md") if p.stem != "00-index")
    if not notes:
        pytest.skip("derivation notes not written yet")
    verifiers = {p.stem.replace("verify_", "").replace("_", "-") for p in _verifier_scripts()}
    missing = []
    for note in notes:
        # 03-layernorm.md  ->  verify_03_layernorm.py
        key = note.stem.replace("-", "-")
        stem = note.stem.replace("-", "_")
        if not (VERIFY_DIR / f"verify_{stem}.py").exists() and key not in verifiers:
            missing.append(note.name)
    assert not missing, f"derivation notes with no verifier: {missing}"
