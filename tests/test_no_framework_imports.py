"""The test that keeps the project's central claim honest.

"Built from scratch" is only true if it is enforced. A silent PyTorch fallback in
library code -- a `try: import torch / except ImportError:` that quietly uses the
fast path when torch happens to be installed -- would make every benchmark and
every gradient claim meaningless, and it would be invisible in review. So it is a
test, and it fails the build.

The rule: nothing under ``src/`` or ``phase0/`` may import a deep-learning
framework. PyTorch is allowed **only** under ``tests/oracle/``.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BANNED = {"torch", "tensorflow", "jax", "autograd", "keras", "theano", "chainer", "mxnet", "flax"}

# Everything the library is permitted to depend on at all.
ALLOWED_THIRD_PARTY = {"numpy", "scipy", "matplotlib", "tqdm"}


def _python_files(*relative: str) -> list[Path]:
    out: list[Path] = []
    for rel in relative:
        base = ROOT / rel
        if base.exists():
            out += [p for p in base.rglob("*.py") if "__pycache__" not in p.parts]
    return out


LIBRARY_FILES = _python_files("src", "phase0")


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_there_are_library_files_to_check():
    """Guard the guard: an empty file list would make every test below vacuous."""
    assert len(LIBRARY_FILES) > 15, f"only found {len(LIBRARY_FILES)} library files"


@pytest.mark.parametrize("path", LIBRARY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_deep_learning_framework_is_imported(path):
    offending = _imported_roots(path) & BANNED
    assert not offending, f"{path.relative_to(ROOT)} imports {sorted(offending)}"


@pytest.mark.parametrize("path", LIBRARY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_framework_named_in_a_string_or_importlib_call(path):
    """Catches the dynamic dodge: ``importlib.import_module("torch")`` and
    ``__import__("torch")`` are invisible to the AST import walk above."""
    src = path.read_text(encoding="utf-8")
    for name in BANNED:
        pattern = rf"""(import_module|__import__)\s*\(\s*['"]{name}"""
        assert not re.search(pattern, src), f"{path.relative_to(ROOT)} dynamically imports {name}"


@pytest.mark.parametrize("path", LIBRARY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_library_only_depends_on_the_declared_stack(path):
    stdlib = sys.stdlib_module_names
    first_party = {"scratchgrad"}
    roots = _imported_roots(path)
    unexpected = roots - stdlib - first_party - ALLOWED_THIRD_PARTY
    # phase0 is standalone by design and imports its own sibling modules.
    if path.parts[-2] == "phase0" or "phase0" in path.parts:
        unexpected -= {p.stem for p in (ROOT / "phase0").glob("*.py")}
    assert not unexpected, f"{path.relative_to(ROOT)} imports unexpected {sorted(unexpected)}"


def test_phase0_does_not_import_the_package():
    """Phase 0's whole point is being readable end to end with no library behind
    it. Importing scratchgrad would quietly destroy that."""
    for path in _python_files("phase0"):
        assert "scratchgrad" not in _imported_roots(path), (
            f"{path.relative_to(ROOT)} imports scratchgrad; Phase 0 must stay self-contained"
        )


def test_torch_is_confined_to_the_oracle_directory():
    for path in _python_files("tests"):
        if "oracle" in path.parts:
            continue
        assert not (_imported_roots(path) & BANNED), (
            f"{path.relative_to(ROOT)} imports a framework outside tests/oracle/"
        )


def test_importing_scratchgrad_does_not_pull_in_a_framework():
    """Even transitively, and even lazily at import time."""
    import subprocess

    code = (
        "import sys; import scratchgrad; import scratchgrad.nn; "
        "bad=[m for m in sys.modules if m.split('.')[0] in "
        f"{sorted(BANNED)!r}]; "
        "print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
    )
    assert out.stdout.strip() == "[]", f"importing scratchgrad loaded {out.stdout.strip()}"
