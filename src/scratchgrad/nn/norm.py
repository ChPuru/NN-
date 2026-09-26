"""Normalisation layers, and the two dropout variants that live beside them.

Every layer in this file is a **single fused** :meth:`Tensor._make` node with a
hand-derived backward, not a composition of ``mean`` / ``var`` / ``sqrt``
primitives. Composing works and gradchecks identically -- it just builds a dozen
graph nodes and several full-size temporaries per call for a backward that
collapses into one line.

The backward, derived once for all of them
------------------------------------------
Take one *group* of ``N`` elements that share a mean and a variance: a row for
LayerNorm, a channel across the whole batch for BatchNorm, a (sample, group)
slab for GroupNorm. With ``mu`` and ``v`` the **biased** moments,
``istd = 1 / sqrt(v + eps)``, ``xhat_i = (x_i - mu) * istd`` and
``y_i = g_i * xhat_i + b_i``::

    dxhat_i = dy_i * g_i
    dv      = -(istd**3 / 2) * sum_i dxhat_i * (x_i - mu)
    dmu     = -istd * sum_i dxhat_i
    dx_j    = istd * dxhat_j  +  dv * 2 * (x_j - mu) / N  +  dmu / N

``dmu`` carries no ``dv`` contribution because ``v`` depends on ``mu`` only
through ``sum_i (x_i - mu)``, which is identically zero. Substituting
``istd**3 * (x_j - mu) * (x_i - mu) == istd * xhat_j * xhat_i`` collapses the
whole thing to the three-term form actually implemented below::

    dx = istd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))

with both means taken over the group. RMSNorm is the same expression without the
centring term, since it never subtracts a mean::

    dx = istd * (dxhat - xhat * mean(dxhat * xhat))

Conventions that are not negotiable
-----------------------------------
* **Biased variance, epsilon inside the sqrt.** The Annotated Transformer's
  LayerNorm is ``g * (x - mu) / (std_unbiased + eps) + b`` -- ``ddof=1`` and the
  epsilon *outside*. Both versions train. They are different functions: they do
  not gradcheck against each other, and a checkpoint written by one is silently
  wrong under the other. This file implements the PyTorch form.
* **BatchNorm normalises with the biased batch variance and stores the unbiased
  one** in ``running_var``. That ``n/(n-1)`` correction is what PyTorch does and
  it is a real, checkable difference -- at ``n = 8`` the two estimators differ by
  14%, four orders of magnitude above eps.
* **Running statistics are buffers, never Parameters.** A ``Parameter`` is handed
  to the optimizer, which would then apply momentum and weight decay to a running
  mean.
* **Dropout is inverted dropout**: the ``1/(1-p)`` rescaling happens at *train*
  time, so ``eval()`` is exactly the identity and there is no train/test mismatch
  left to forget about.
"""

from __future__ import annotations

import numpy as np

from .._config import get_default_dtype
from ..tensor import Tensor
from .module import Module, Parameter

__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "Dropout",
    "Dropout2d",
    "GroupNorm",
    "LayerNorm",
    "RMSNorm",
    "dropout2d",
    "layer_norm",
    "rms_norm",
]


# --------------------------------------------------------------------------- #
# the fused kernel every normalisation layer delegates to
# --------------------------------------------------------------------------- #
def _normalize(
    x: Tensor,
    axes: tuple[int, ...],
    eps: float,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    center: bool = True,
    stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Tensor, np.ndarray | None, np.ndarray]:
    """Normalise ``x`` over ``axes`` and apply an optional affine map.

    Args:
        x: Input tensor.
        axes: Axes reduced to form each group's statistics.
        eps: Added to the variance **inside** the square root.
        weight: Scale, already reshaped to broadcast against ``x`` (same rank,
            extent 1 on every axis it is broadcast along). ``None`` skips it.
        bias: Shift, same shape rule as ``weight``.
        center: Subtract the mean. ``False`` gives the RMSNorm variant.
        stats: ``(mean, var)`` to normalise with instead of computing them from
            ``x`` -- BatchNorm's eval path. They are **constants**: no gradient
            flows into them, which is exactly right for a buffer.

    Returns:
        ``(output, mean, biased_var)``. The moments come back as keepdims arrays
        so BatchNorm can fold them into its running statistics; ``mean`` is
        ``None`` when ``center`` is off.
    """
    a = x.data
    for p in (weight, bias):
        if p is not None and p.data.ndim != a.ndim:
            raise ValueError(
                f"affine parameter of rank {p.data.ndim} cannot broadcast against a rank-"
                f"{a.ndim} input; reshape it to full rank before calling"
            )

    if stats is None:
        mu = a.mean(axis=axes, keepdims=True) if center else None
        centered = a - mu if center else a
        var = (centered * centered).mean(axis=axes, keepdims=True)
    else:
        mu, var = stats
        centered = a - mu if center else a

    istd = 1.0 / np.sqrt(var + eps)
    xhat = centered * istd

    wd = weight.data if weight is not None else None
    out = xhat * wd if wd is not None else xhat
    if bias is not None:
        out = out + bias.data

    def gx(g: np.ndarray) -> np.ndarray:
        dxhat = g * wd if wd is not None else g
        if stats is not None:
            # mu and var are constants on this path, so the coupling between the
            # members of a group disappears entirely. Reaching for this branch
            # during training is the classic BatchNorm that descends but is wrong.
            return dxhat * istd
        corr = xhat * (dxhat * xhat).mean(axis=axes, keepdims=True)
        if center:
            corr = corr + dxhat.mean(axis=axes, keepdims=True)
        return (dxhat - corr) * istd

    parents: list[Tensor] = [x]
    grad_fns: list = [gx]
    if weight is not None:
        w_axes = tuple(i for i, s in enumerate(wd.shape) if s == 1)

        def gw(g: np.ndarray, w_axes: tuple[int, ...] = w_axes) -> np.ndarray:
            return (g * xhat).sum(axis=w_axes, keepdims=True)

        parents.append(weight)
        grad_fns.append(gw)
    if bias is not None:
        b_axes = tuple(i for i, s in enumerate(bias.data.shape) if s == 1)

        def gb(g: np.ndarray, b_axes: tuple[int, ...] = b_axes) -> np.ndarray:
            return g.sum(axis=b_axes, keepdims=True)

        parents.append(bias)
        grad_fns.append(gb)

    op = "rms_norm" if not center else ("norm_frozen_stats" if stats is not None else "norm")
    return Tensor._make(out, tuple(parents), tuple(grad_fns), op), mu, var


def _as_shape(normalized_shape) -> tuple[int, ...]:
    if isinstance(normalized_shape, (int, np.integer)):
        return (int(normalized_shape),)
    return tuple(int(s) for s in normalized_shape)


def _trailing_setup(x: Tensor, shape: tuple[int, ...], what: str):
    """Axes and affine broadcast shape for a norm over the trailing dims."""
    k = len(shape)
    if x.data.ndim < k or x.data.shape[-k:] != shape:
        raise ValueError(f"{what} expected trailing dims {shape}, got input shape {x.data.shape}")
    ndim = x.data.ndim
    return tuple(range(ndim - k, ndim)), (1,) * (ndim - k) + shape


# --------------------------------------------------------------------------- #
# functional forms
# --------------------------------------------------------------------------- #
def layer_norm(
    x: Tensor,
    normalized_shape,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """``g * (x - mu) / sqrt(var_biased + eps) + b`` over the trailing axes."""
    shape = _as_shape(normalized_shape)
    axes, bcast = _trailing_setup(x, shape, "layer_norm")
    w = weight.reshape(bcast) if weight is not None else None
    b = bias.reshape(bcast) if bias is not None else None
    return _normalize(x, axes, eps, w, b)[0]


def rms_norm(
    x: Tensor, normalized_shape, weight: Tensor | None = None, eps: float = 1e-6
) -> Tensor:
    """``g * x / sqrt(mean(x**2) + eps)`` -- LayerNorm with the centring removed."""
    shape = _as_shape(normalized_shape)
    axes, bcast = _trailing_setup(x, shape, "rms_norm")
    w = weight.reshape(bcast) if weight is not None else None
    return _normalize(x, axes, eps, w, None, center=False)[0]


def _keep_mask(shape, p: float, dtype, rng: np.random.Generator) -> np.ndarray:
    """The inverted-dropout keep mask: ``1/(1-p)`` where kept, ``0`` where dropped."""
    return (rng.random(shape) >= p).astype(dtype) / (1.0 - p)


def dropout2d(
    x: Tensor, p: float = 0.5, training: bool = True, rng: np.random.Generator | None = None
) -> Tensor:
    """Zero **whole channels** of an ``(N, C, ...)`` input.

    Spatial dropout, not elementwise: adjacent pixels of a feature map are
    strongly correlated, so zeroing individual activations leaves nearly all of a
    channel's information reachable and regularises far less than the nominal
    ``p`` suggests.
    """
    # Validate *before* the no-op early return. A bad ``p`` or a rank-2 input is
    # a caller bug on every path; guarding only the sampling path means the same
    # broken call is silently accepted in eval() and at p=0, so it survives a
    # test suite and first raises in production the moment training is enabled.
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")
    if x.data.ndim < 3:
        raise ValueError(f"dropout2d expects (N, C, ...) with at least 3 dims, got {x.data.shape}")
    if not training or p == 0.0:
        return x
    g = rng if rng is not None else np.random.default_rng()
    mask_shape = x.data.shape[:2] + (1,) * (x.data.ndim - 2)
    keep = _keep_mask(mask_shape, p, x.data.dtype, g)
    return Tensor._make(x.data * keep, (x,), (lambda gr: gr * keep,), "dropout2d")


# --------------------------------------------------------------------------- #
# LayerNorm / RMSNorm / GroupNorm
# --------------------------------------------------------------------------- #
class LayerNorm(Module):
    """Normalise each sample over its trailing ``normalized_shape`` axes.

    Args:
        normalized_shape: Int or tuple naming the trailing axes to normalise.
        eps: Added to the biased variance inside the square root.
        elementwise_affine: Learn a per-element ``weight`` (and ``bias``).
        bias: Learn the shift. Ignored when ``elementwise_affine`` is False.
    """

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = _as_shape(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape, dtype=get_default_dtype()))
            if bias:
                self.bias = Parameter(np.zeros(self.normalized_shape, dtype=get_default_dtype()))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(
            x,
            self.normalized_shape,
            self._parameters.get("weight"),
            self._parameters.get("bias"),
            self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"
        )


class RMSNorm(Module):
    """Root-mean-square normalisation: LayerNorm with the mean subtraction removed.

    Zhang & Sennrich (2019) observed that re-centring contributes little and
    dropped it, which removes one reduction and one full-size temporary per call.
    There is no bias by convention -- the shift is precisely the part the paper
    argues is unnecessary.
    """

    def __init__(
        self, normalized_shape, eps: float = 1e-6, elementwise_affine: bool = True
    ) -> None:
        super().__init__()
        self.normalized_shape = _as_shape(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape, dtype=get_default_dtype()))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.normalized_shape, self._parameters.get("weight"), self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


class GroupNorm(Module):
    """Normalise over ``(C / G, *spatial)`` within each sample.

    ``num_groups == 1`` is LayerNorm over the whole feature map;
    ``num_groups == num_channels`` is InstanceNorm. Neither depends on the batch
    size, which is the reason to reach for this instead of BatchNorm when the
    batch is small.

    The channel axis splits as ``(G, C // G)`` -- consecutive channels share a
    group. Splitting the other way (``(C // G, G)``, i.e. strided groups) is a
    different layer whose checkpoints are not interchangeable with this one.
    """

    def __init__(
        self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True
    ) -> None:
        super().__init__()
        # Checked before the divisibility test, which is where a zero would land
        # as a bare ZeroDivisionError from the modulo. A negative count is worse:
        # it passes the modulo (6 % -3 == 0) and only surfaces as an unrelated
        # reshape failure inside forward(), one call site away from the mistake.
        if num_groups < 1 or num_channels < 1:
            raise ValueError(
                f"GroupNorm needs num_groups >= 1 and num_channels >= 1, got "
                f"num_groups={num_groups}, num_channels={num_channels}"
            )
        if num_channels % num_groups:
            raise ValueError(f"num_channels {num_channels} is not divisible by {num_groups} groups")
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.affine = bool(affine)
        if affine:
            self.weight = Parameter(np.ones(self.num_channels, dtype=get_default_dtype()))
            self.bias = Parameter(np.zeros(self.num_channels, dtype=get_default_dtype()))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.data.shape
        if len(shape) < 2 or shape[1] != self.num_channels:
            raise ValueError(
                f"GroupNorm({self.num_groups}, {self.num_channels}) got input shape {shape}; "
                "expected (N, num_channels, ...)"
            )
        per_group = self.num_channels // self.num_groups
        spatial = shape[2:]
        grouped = (shape[0], self.num_groups, per_group) + spatial
        # Statistics are per (sample, group) while the affine map is per channel,
        # so the parameters reshape into the *split* channel axis and ride along
        # in the same fused node instead of costing a second pass over the data.
        bcast = (1, self.num_groups, per_group) + (1,) * len(spatial)
        w = self._parameters.get("weight")
        b = self._parameters.get("bias")
        out, _, _ = _normalize(
            x.reshape(grouped),
            tuple(range(2, len(grouped))),
            self.eps,
            w.reshape(bcast) if w is not None else None,
            b.reshape(bcast) if b is not None else None,
        )
        return out.reshape(shape)

    def extra_repr(self) -> str:
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}"


# --------------------------------------------------------------------------- #
# BatchNorm
# --------------------------------------------------------------------------- #
class _BatchNorm(Module):
    """Shared implementation of the ``BatchNorm{1,2}d`` pair.

    Statistics are taken over the batch axis and every spatial axis, leaving one
    mean and one variance per channel.

    Two things the subtly-wrong versions in the wild get backwards:

    * ``train()`` normalises with the **batch** statistics *and* updates the
      running ones; ``eval()`` normalises with the **running** statistics and
      updates nothing. Normalising with the running stats during training makes
      the layer a near no-op at step 0 and decouples the output from the batch
      that the gradient is being computed for.
    * The batch is normalised with the **biased** variance while the running
      estimate accumulates the **unbiased** one. Using a single estimator for
      both is the usual shortcut, and it leaves a systematic ``sqrt(n/(n-1))``
      scale mismatch between train and eval outputs.
    """

    _valid_ndim: tuple[int, ...] = ()

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = None if momentum is None else float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)

        if affine:
            self.weight = Parameter(np.ones(self.num_features, dtype=get_default_dtype()))
            self.bias = Parameter(np.zeros(self.num_features, dtype=get_default_dtype()))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            dt = get_default_dtype()
            self.register_buffer("running_mean", np.zeros(self.num_features, dtype=dt))
            self.register_buffer("running_var", np.ones(self.num_features, dtype=dt))
            # An int64 count, not a float: with momentum=None it is the exact
            # denominator of a cumulative average and must not drift.
            self.register_buffer("num_batches_tracked", np.array(0, dtype=np.int64))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    # -- running state -------------------------------------------------------
    def reset_running_stats(self) -> None:
        """Restore the buffers to the identity transform (mean 0, variance 1)."""
        if not self.track_running_stats:
            return
        self._buffers["running_mean"][...] = 0.0
        self._buffers["running_var"][...] = 1.0
        self._buffers["num_batches_tracked"][...] = 0

    def _update_running_stats(self, mu: np.ndarray, var_biased: np.ndarray, n: int) -> None:
        """Fold this batch's moments into the running estimates.

        ``var_biased`` arrives divided by ``n``; Bessel's ``n / (n - 1)``
        correction is applied here and only here, so the batch itself is still
        normalised by the biased estimator.

        The updates are in-place writes into the buffer arrays. Rebinding
        ``self.running_mean = ...`` would install a plain attribute that shadows
        the registered buffer, and the shadow would never reach ``state_dict``.
        """
        if n < 2:
            raise ValueError(
                f"BatchNorm in train() needs at least 2 elements per channel, got {n}. "
                "The unbiased running variance is undefined for a single sample."
            )
        counter = self._buffers["num_batches_tracked"]
        counter += 1
        # momentum=None is a cumulative moving average: every batch seen so far
        # is weighted equally, so the estimate converges instead of tracking.
        mom = 1.0 / float(counter) if self.momentum is None else self.momentum

        rm = self._buffers["running_mean"]
        rv = self._buffers["running_var"]
        rm *= 1.0 - mom
        rm += mom * mu.reshape(-1)
        rv *= 1.0 - mom
        rv += mom * var_biased.reshape(-1) * (n / (n - 1.0))

    # -- forward -------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        shape = x.data.shape
        if len(shape) not in self._valid_ndim:
            expected = "/".join(str(d) for d in self._valid_ndim)
            raise ValueError(
                f"{type(self).__name__} expects a rank-{expected} input, got shape {shape}"
            )
        if shape[1] != self.num_features:
            raise ValueError(
                f"{type(self).__name__}({self.num_features}) got {shape[1]} channels "
                f"(input shape {shape})"
            )

        axes = (0,) + tuple(range(2, len(shape)))
        bcast = (1, self.num_features) + (1,) * (len(shape) - 2)
        w = self._parameters.get("weight")
        b = self._parameters.get("bias")

        # Without running stats there is nothing to fall back on, so eval() also
        # uses the batch -- which is what makes that configuration a different
        # model at test time, not merely an untracked one.
        use_batch_stats = self.training or not self.track_running_stats
        stats = None
        if not use_batch_stats:
            stats = (
                self._buffers["running_mean"].reshape(bcast),
                self._buffers["running_var"].reshape(bcast),
            )

        out, mu, var = _normalize(
            x,
            axes,
            self.eps,
            w.reshape(bcast) if w is not None else None,
            b.reshape(bcast) if b is not None else None,
            stats=stats,
        )
        if self.training and self.track_running_stats:
            self._update_running_stats(mu, var, x.data.size // self.num_features)
        return out

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class BatchNorm1d(_BatchNorm):
    """BatchNorm for ``(N, C)`` or ``(N, C, L)`` inputs."""

    _valid_ndim = (2, 3)


class BatchNorm2d(_BatchNorm):
    """BatchNorm for ``(N, C, H, W)`` images."""

    _valid_ndim = (4,)


# --------------------------------------------------------------------------- #
# dropout
# --------------------------------------------------------------------------- #
class Dropout(Module):
    """Inverted dropout over individual activations.

    Args:
        p: Probability that an activation is zeroed. Must lie in ``[0, 1)``.
        rng: Generator used to draw masks. Passing one at construction makes a
            whole training run reproducible; ``forward(x, rng=...)`` overrides it
            per call, which is what gradchecking a stochastic layer needs -- a
            finite-difference probe pair must see the *same* mask on both sides.

            Omitting it does **not** hand the draw to numpy's global state. The
            layer seeds a generator of its own, once, at construction. What that
            guarantees: the entire mask sequence is fixed the instant the layer
            is built, so ``np.random.seed`` and every other consumer of the
            global stream cannot perturb it, and a ``deepcopy`` replays the same
            masks. What it does **not** guarantee: the same masks in the next
            process. Only an explicit ``rng`` buys that.
    """

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = float(p)
        # Seeded here rather than per call: contract rule 6 says every draw goes
        # through an explicit Generator, and a fresh `default_rng()` inside
        # forward() would make the instance's own history unreplayable -- two
        # copies of the same layer could never be made to agree.
        self.rng = np.random.default_rng() if rng is None else rng

    def _generator(self, rng: np.random.Generator | None) -> np.random.Generator:
        return self.rng if rng is None else rng

    def forward(self, x: Tensor, rng: np.random.Generator | None = None) -> Tensor:
        # Returning `x` itself rather than a rescaled copy is the whole point of
        # the inverted formulation: eval() is the identity map, node for node.
        if not self.training or self.p == 0.0:
            return x
        keep = _keep_mask(x.data.shape, self.p, x.data.dtype, self._generator(rng))
        return Tensor._make(x.data * keep, (x,), (lambda g: g * keep,), "dropout")

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Dropout2d(Dropout):
    """Inverted dropout that zeroes entire ``(N, C)`` channels of a feature map."""

    def forward(self, x: Tensor, rng: np.random.Generator | None = None) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        return dropout2d(x, self.p, True, self._generator(rng))
