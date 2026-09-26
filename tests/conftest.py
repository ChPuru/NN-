"""Shared pytest configuration.

Note there is deliberately no ``tests/helpers.py``: pytest's own docs record that
``--import-mode=importlib`` conflicts with a helper module living inside the test
package. Shared numeric helpers live in the library instead, as
``scratchgrad.testing``, which also means they are covered by the same checks as
everything else.
"""

from __future__ import annotations

import numpy as np
import pytest

import scratchgrad as sg


@pytest.fixture(autouse=True)
def _strict_numpy_errors():
    """Turn silent numerical failures into stack traces for the whole suite.

    ``divide``, ``over`` and ``invalid`` are raised; ``underflow`` is not.
    Underflow to zero is the *correct* behaviour of the stable sigmoid and
    softmax paths -- ``exp(-800)`` is meant to become 0 -- so raising on it would
    reject working code.
    """
    old = np.seterr(divide="raise", over="raise", invalid="raise", under="ignore")
    try:
        yield
    finally:
        np.seterr(**old)


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    """Guarantee no test leaks a float64 default into the next one."""
    before = sg.get_default_dtype()
    try:
        yield
    finally:
        sg.set_default_dtype(before)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260909)


@pytest.fixture
def f64():
    """Run the body with float64 as the default dtype."""
    with sg.default_dtype(np.float64):
        yield
