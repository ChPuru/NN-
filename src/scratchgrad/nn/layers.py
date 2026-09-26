"""Dense layers, activations as modules, and small structural helpers."""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from . import functional as F
from . import init
from .module import Module, Parameter

__all__ = [
    "ELU",
    "GELU",
    "Flatten",
    "Identity",
    "Lambda",
    "LeakyReLU",
    "Linear",
    "LogSoftmax",
    "ReLU",
    "Reshape",
    "SiLU",
    "Sigmoid",
    "Softmax",
    "Softplus",
    "Tanh",
]


class Linear(Module):
    """``y = x @ W + b``.

    ``weight`` is stored ``(in_features, out_features)`` -- the layout that makes
    the forward pass a plain ``x @ W`` with no transpose anywhere in either
    direction. (PyTorch stores the transpose and transposes in ``F.linear``;
    both are fine, but mixing the two conventions across a codebase is a
    guaranteed lost afternoon.)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        nonlinearity: str = "relu",
        rng=None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter(np.zeros((self.in_features, self.out_features)))
        init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity=nonlinearity, rng=rng)
        if bias:
            self.bias = Parameter(np.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        if x.data.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear expected last dim {self.in_features}, got {x.data.shape[-1]} "
                f"(input shape {x.data.shape})"
            )
        return F.linear(x, self.weight, self._parameters.get("bias"))

    def extra_repr(self) -> str:
        has_bias = self._parameters.get("bias") is not None
        return f"{self.in_features}, {self.out_features}, bias={has_bias}"


class Identity(Module):
    def forward(self, x):
        return x


class Flatten(Module):
    """Collapse every axis from ``start_dim`` onward. Batch axis survives."""

    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = int(start_dim)

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}"


class Reshape(Module):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape((x.data.shape[0],) + self.shape)

    def extra_repr(self) -> str:
        return str(self.shape)


class Lambda(Module):
    """Wrap a plain function so it can sit inside a ``Sequential``."""

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class _Activation(Module):
    _fn: staticmethod

    def forward(self, x: Tensor) -> Tensor:
        return type(self)._fn(x)


class ReLU(_Activation):
    _fn = staticmethod(F.relu)


class Tanh(_Activation):
    _fn = staticmethod(F.tanh)


class Sigmoid(_Activation):
    _fn = staticmethod(F.sigmoid)


class SiLU(_Activation):
    _fn = staticmethod(F.silu)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class ELU(Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: Tensor) -> Tensor:
        return F.elu(x, self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class GELU(Module):
    def __init__(self, approximate: str = "tanh") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x, self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"


class Softplus(Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = float(beta)

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x, self.beta)


class Softmax(Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = int(axis)

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, self.axis)

    def extra_repr(self) -> str:
        return f"axis={self.axis}"


class LogSoftmax(Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = int(axis)

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(x, self.axis)

    def extra_repr(self) -> str:
        return f"axis={self.axis}"
