"""scratchgrad -- a deep-learning framework written from scratch in pure NumPy.

NumPy supplies array storage and BLAS-backed matrix multiply. Every gradient in
this package is hand-derived and independently verified against finite
differences, an adjoint identity where the operation is linear, and forward-value
checks against hand-computed constants. No autodiff library is imported anywhere
under ``src/``; PyTorch appears only in ``tests/oracle/`` as a test oracle, and
``tests/test_no_framework_imports.py`` fails the build if that ever changes.

    >>> import numpy as np, scratchgrad as sg
    >>> from scratchgrad import nn, optim
    >>> model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    >>> opt = optim.Adam(model.parameters(), lr=1e-3)
    >>> logits = model(sg.Tensor(np.zeros((32, 784))))
    >>> loss = nn.functional.cross_entropy(logits, np.zeros(32, dtype=int))
    >>> opt.zero_grad(); loss.backward(); opt.step()
"""

from __future__ import annotations

from ._config import (
    default_dtype,
    enable_grad,
    get_default_dtype,
    is_grad_enabled,
    no_grad,
    set_default_dtype,
    set_grad_enabled,
)
from .tensor import (
    Tensor,
    arange,
    concatenate,
    eye,
    from_numpy,
    full,
    ones,
    stack,
    tensor,
    unbroadcast,
    where,
    zeros,
)

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "__version__",
    "arange",
    "concatenate",
    "default_dtype",
    "enable_grad",
    "eye",
    "from_numpy",
    "full",
    "get_default_dtype",
    "is_grad_enabled",
    "no_grad",
    "ones",
    "set_default_dtype",
    "set_grad_enabled",
    "stack",
    "tensor",
    "unbroadcast",
    "where",
    "zeros",
]


def __getattr__(name: str):
    """Lazy submodule access so ``import scratchgrad`` stays cheap."""
    if name in ("nn", "optim", "data", "utils", "testing"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
