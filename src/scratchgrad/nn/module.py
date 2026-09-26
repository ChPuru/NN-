"""``Parameter`` and ``Module`` -- the container layer that makes this a library.

The line between "a script that hardcodes one architecture" and "a framework"
is that someone can define a model the author never anticipated without editing
library code. That needs two things: an autograd engine (``scratchgrad.tensor``)
and a way for a composite object to *find* its own trainable state. This module
is the second one.

Registration happens in ``__setattr__``: assigning a :class:`Parameter` or a
:class:`Module` to an attribute records it, so ``self.weight = Parameter(...)``
is all a layer author has to write and ``model.parameters()`` finds it through
any depth of nesting.
"""

from __future__ import annotations

import copy as _copy
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from ..tensor import Tensor

__all__ = ["Module", "ModuleDict", "ModuleList", "Parameter", "Sequential"]


class Parameter(Tensor):
    """A :class:`Tensor` that is trainable state of a :class:`Module`.

    The only behavioural difference from ``Tensor`` is that it defaults to
    ``requires_grad=True`` and that ``Module.__setattr__`` recognises it.
    """

    __slots__ = ()

    def __init__(self, data, requires_grad: bool = True, dtype=None, name: str | None = None):
        super().__init__(data, requires_grad=requires_grad, dtype=dtype, name=name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "Parameter " + super().__repr__()


class Module:
    """Base class for every layer and model.

    Subclasses call ``super().__init__()`` first, assign parameters and
    submodules as attributes, and implement :meth:`forward`.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    # -- attribute registration ---------------------------------------------
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (Parameter, Module)) and "_parameters" not in self.__dict__:
            raise AttributeError(
                f"cannot assign {type(value).__name__} '{name}' before "
                f"{type(self).__name__}.__init__() -- call super().__init__() first"
            )
        if isinstance(value, Parameter):
            self._parameters.pop(name, None)
            self.__dict__.pop(name, None)
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules.pop(name, None)
            self.__dict__.pop(name, None)
            self._modules[name] = value
        else:
            # Reassigning a registered name with a non-parameter clears it,
            # otherwise the old parameter would linger in parameters().
            if "_parameters" in self.__dict__:
                self._parameters.pop(name, None)
                self._modules.pop(name, None)
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails, so the three registries are
        # the last place to look.
        for registry in ("_parameters", "_modules", "_buffers"):
            store = self.__dict__.get(registry)
            if store is not None and name in store:
                return store[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __delattr__(self, name: str) -> None:
        for registry in ("_parameters", "_modules", "_buffers"):
            store = self.__dict__.get(registry)
            if store is not None and name in store:
                del store[name]
                return
        object.__delattr__(self, name)

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        self._parameters[name] = param

    def register_buffer(self, name: str, value: np.ndarray | None) -> None:
        """Register non-trainable state that still belongs in ``state_dict``.

        BatchNorm's running mean and variance are the canonical case: they are
        updated by the forward pass, must be saved and restored with the model,
        and must never be handed to an optimizer.
        """
        self._buffers[name] = None if value is None else np.asarray(value)

    def add_module(self, name: str, module: Module | None) -> None:
        self._modules[name] = module

    # -- traversal -----------------------------------------------------------
    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()
        for name, p in self._parameters.items():
            if p is None or id(p) in seen:
                continue
            seen.add(id(p))
            yield (f"{prefix}{name}", p)
        if recurse:
            for mod_name, mod in self._modules.items():
                if mod is None:
                    continue
                for name, p in mod.named_parameters(prefix=f"{prefix}{mod_name}.", recurse=True):
                    if id(p) in seen:
                        continue
                    seen.add(id(p))
                    yield (name, p)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, p in self.named_parameters(recurse=recurse):
            yield p

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, np.ndarray]]:
        for name, b in self._buffers.items():
            if b is not None:
                yield (f"{prefix}{name}", b)
        if recurse:
            for mod_name, mod in self._modules.items():
                if mod is None:
                    continue
                yield from mod.named_buffers(prefix=f"{prefix}{mod_name}.", recurse=True)

    def buffers(self, recurse: bool = True) -> Iterator[np.ndarray]:
        for _, b in self.named_buffers(recurse=recurse):
            yield b

    def children(self) -> Iterator[Module]:
        for m in self._modules.values():
            if m is not None:
                yield m

    def named_children(self) -> Iterator[tuple[str, Module]]:
        for name, m in self._modules.items():
            if m is not None:
                yield name, m

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        yield (prefix.rstrip("."), self)
        for name, m in self._modules.items():
            if m is not None:
                yield from m.named_modules(prefix=f"{prefix}{name}.")

    def modules(self) -> Iterator[Module]:
        for _, m in self.named_modules():
            yield m

    def apply(self, fn) -> Module:
        for m in self.children():
            m.apply(fn)
        fn(self)
        return self

    # -- modes ---------------------------------------------------------------
    def train(self, mode: bool = True) -> Module:
        self.training = bool(mode)
        for m in self.children():
            m.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    # -- serialization -------------------------------------------------------
    def state_dict(self, prefix: str = "", deep: bool = False) -> OrderedDict[str, np.ndarray]:
        """Every parameter and buffer, keyed by dotted path.

        By default the returned arrays **share storage with the live model**, so
        a snapshot taken for early stopping is silently overwritten by the next
        optimizer step. Pass ``deep=True`` for a genuine copy.
        """
        out: OrderedDict[str, np.ndarray] = OrderedDict()
        for name, p in self.named_parameters(prefix=prefix):
            out[name] = np.array(p.data, copy=True) if deep else p.data
        for name, b in self.named_buffers(prefix=prefix):
            out[name] = np.array(b, copy=True) if deep else b
        return out

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        own = dict(self.named_parameters())
        own_buf = dict(self.named_buffers())
        missing = [k for k in list(own) + list(own_buf) if k not in state]
        unexpected = [k for k in state if k not in own and k not in own_buf]
        if strict and (missing or unexpected):
            raise KeyError(
                f"state_dict mismatch. missing={missing[:8]}{'...' if len(missing) > 8 else ''} "
                f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
            )
        for name, p in own.items():
            if name in state:
                v = np.asarray(state[name])
                if v.shape != p.data.shape:
                    raise ValueError(
                        f"shape mismatch for {name}: checkpoint {v.shape} vs model {p.data.shape}"
                    )
                p.data = v.astype(p.data.dtype, copy=True)
        for name, b in own_buf.items():
            if name in state:
                v = np.asarray(state[name])
                if v.shape != b.shape:
                    raise ValueError(
                        f"shape mismatch for buffer {name}: checkpoint {v.shape} vs model {b.shape}"
                    )
                b[...] = v.astype(b.dtype, copy=False)

    def astype(self, dtype) -> Module:
        """Cast every parameter and buffer in place (e.g. to float64 for gradcheck)."""
        dt = np.dtype(dtype)
        for p in self.parameters():
            p.data = p.data.astype(dt, copy=False)
            if p.grad is not None:
                p.grad = p.grad.astype(dt, copy=False)
        for _, store in self._named_buffer_stores():
            for k, v in list(store.items()):
                if v is not None and v.dtype.kind == "f":
                    store[k] = v.astype(dt, copy=False)
        return self

    def double(self) -> Module:
        return self.astype(np.float64)

    def float(self) -> Module:
        return self.astype(np.float32)

    def _named_buffer_stores(self):
        for name, m in self.named_modules():
            yield name, m._buffers

    def deepcopy(self) -> Module:
        return _copy.deepcopy(self)

    # -- introspection -------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.size for p in self.parameters() if p.requires_grad or not trainable_only)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        lines = []
        extra = self.extra_repr()
        for name, m in self.named_children():
            block = repr(m).split("\n")
            block = [block[0]] + ["  " + ln for ln in block[1:]]
            lines.append(f"  ({name}): " + "\n".join(block))
        head = f"{type(self).__name__}({extra}"
        if not lines:
            return head + ")"
        return head + "\n" + "\n".join(lines) + "\n)"

    def summary(self) -> str:
        """A parameter table. Every number in the README should come from here."""
        rows = [(n, tuple(p.shape), p.size) for n, p in self.named_parameters()]
        width = max([len(r[0]) for r in rows] + [9])
        out = [f"{'parameter'.ljust(width)}  {'shape':<22} {'count':>10}", "-" * (width + 36)]
        for n, s, c in rows:
            out.append(f"{n.ljust(width)}  {s!s:<22} {c:>10,}")
        out.append("-" * (width + 36))
        out.append(f"{'total'.ljust(width)}  {'':<22} {self.num_parameters():>10,}")
        return "\n".join(out)

    # -- call ----------------------------------------------------------------
    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Sequential(Module):
    """Apply the contained modules in order."""

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = tuple(layers[0])
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __getitem__(self, idx):
        items = list(self._modules.values())
        if isinstance(idx, slice):
            return Sequential(*items[idx])
        return items[idx]

    def append(self, module: Module) -> Sequential:
        self.add_module(str(len(self._modules)), module)
        return self

    def forward(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x


class ModuleList(Module):
    """A list of submodules that are registered but not automatically applied."""

    def __init__(self, modules: Iterable[Module] | None = None) -> None:
        super().__init__()
        if modules:
            for m in modules:
                self.append(m)

    def append(self, module: Module) -> ModuleList:
        self.add_module(str(len(self._modules)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> ModuleList:
        for m in modules:
            self.append(m)
        return self

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __getitem__(self, idx):
        items = list(self._modules.values())
        if isinstance(idx, slice):
            return ModuleList(items[idx])
        return items[idx]

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("ModuleList has no forward(); index into it and call the children")


class ModuleDict(Module):
    """A string-keyed collection of submodules."""

    def __init__(self, modules: dict[str, Module] | None = None) -> None:
        super().__init__()
        for k, v in (modules or {}).items():
            self.add_module(k, v)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, value: Module) -> None:
        self.add_module(key, value)

    def __len__(self) -> int:
        return len(self._modules)

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def values(self):
        return self._modules.values()

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("ModuleDict has no forward(); index into it and call the children")
