"""Tests for ``scratchgrad.optim``.

Imports are from the concrete modules (``scratchgrad.optim.optimizer``) rather
than the package, because the package ``__init__`` is owned elsewhere.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scratchgrad._config import default_dtype
from scratchgrad.nn.module import Parameter
from scratchgrad.optim.lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    ExponentialLR,
    LinearWarmup,
    MultiStepLR,
    SequentialLR,
    StepLR,
    WarmupCosine,
)
from scratchgrad.optim.optimizer import (
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    Optimizer,
    RMSProp,
    clip_grad_norm_,
    clip_grad_value_,
)
from scratchgrad.tensor import Tensor
from scratchgrad.testing.gradcheck import TOL_COMPOSED, gradcheck

# Optimizers do float arithmetic on parameters directly; float32 rounding would
# swamp the exact-equality checks (state_dict round trip) and the 1e-8 tolerances.
F64 = np.float64


@pytest.fixture(autouse=True)
def float64_everywhere():
    """Force the framework default dtype to float64 for every test in this file.

    ``Tensor`` casts the result of *every* op to the global default, so a
    float64 leaf still produces a float32 loss under the float32 training
    default -- and the exact-trajectory and 1e-9 checks below would be measuring
    float32 rounding, not the optimizer.
    """
    with default_dtype(F64):
        yield


def p64(arr) -> Parameter:
    """A float64 Parameter. The default dtype is float32, which is too coarse here."""
    return Parameter(np.asarray(arr, dtype=F64), dtype=F64)


# --------------------------------------------------------------------------- #
# the quadratic bowl
# --------------------------------------------------------------------------- #
def make_bowl(rng: np.random.Generator, n: int = 5, cond: float = 8.0):
    """``f(x) = 0.5 (x - c)^T A (x - c)`` with ``A`` SPD. Minimum 0 at ``x = c``.

    ``cond`` is the ratio of largest to smallest eigenvalue -- the knob that
    decides how much momentum is worth.
    """
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.linspace(1.0, cond, n)
    a = (q * eigs) @ q.T
    a = 0.5 * (a + a.T)  # kill the asymmetry qr/rounding leaves behind
    c = rng.standard_normal(n)
    return a, c


def bowl_loss(x: Tensor, a: np.ndarray, c: np.ndarray) -> Tensor:
    d = x - Tensor(c, dtype=F64)
    return 0.5 * ((d @ Tensor(a, dtype=F64)) * d).sum()


def run_bowl(opt_cls, steps: int, n: int = 5, seed: int = 0, cond: float = 8.0, **kwargs) -> float:
    """Optimize the bowl from a fixed start; return the final distance to ``c``."""
    rng = np.random.default_rng(seed)
    a, c = make_bowl(rng, n, cond)
    x = p64(rng.standard_normal(n))
    opt = opt_cls([x], **kwargs)
    for _ in range(steps):
        opt.zero_grad()
        bowl_loss(x, a, c).backward()
        opt.step()
    return float(np.linalg.norm(x.data - c))


def test_bowl_gradient_is_exact():
    """The objective the convergence tests descend is itself gradchecked."""
    rng = np.random.default_rng(3)
    a, c = make_bowl(rng, 5)
    x = Tensor(rng.standard_normal(5), requires_grad=True, dtype=F64)
    gradcheck(lambda t: bowl_loss(t, a, c), x, tol=TOL_COMPOSED, name="bowl")

    # And against the analytic gradient A(x - c), which finite differences alone
    # cannot distinguish from a self-consistently wrong forward.
    x.grad = None
    bowl_loss(x, a, c).backward()
    assert np.allclose(x.grad, a @ (x.data - c), rtol=1e-12, atol=1e-12)


def run_bowl_decayed(opt_cls, steps: int, gamma: float, seed: int = 0, **kwargs) -> float:
    """Same bowl, with the learning rate annealed geometrically. See below for why."""
    rng = np.random.default_rng(seed)
    a, c = make_bowl(rng, 5)
    x = p64(rng.standard_normal(5))
    opt = opt_cls([x], **kwargs)
    sched = ExponentialLR(opt, gamma=gamma)
    for _ in range(steps):
        opt.zero_grad()
        bowl_loss(x, a, c).backward()
        opt.step()
        sched.step()
    return float(np.linalg.norm(x.data - c))


#: ``(id, class, steps, gamma, kwargs)``. Every variant of every optimizer.
BOWL_CASES = [
    ("sgd", SGD, 1500, 0.995, {"lr": 0.1}),
    ("sgd-momentum", SGD, 1500, 0.995, {"lr": 0.05, "momentum": 0.9}),
    ("sgd-nesterov", SGD, 1500, 0.995, {"lr": 0.05, "momentum": 0.9, "nesterov": True}),
    ("sgd-dampening", SGD, 1500, 0.995, {"lr": 0.05, "momentum": 0.9, "dampening": 0.5}),
    ("sgd-weight-decay", SGD, 1500, 0.995, {"lr": 0.05, "momentum": 0.9, "weight_decay": 1e-8}),
    ("adagrad", Adagrad, 1500, 0.995, {"lr": 0.5}),
    ("adagrad-lr-decay", Adagrad, 1500, 0.995, {"lr": 0.5, "lr_decay": 1e-3}),
    ("rmsprop", RMSProp, 1500, 0.995, {"lr": 0.01}),
    ("rmsprop-momentum", RMSProp, 1500, 0.995, {"lr": 0.01, "momentum": 0.9}),
    ("rmsprop-centered", RMSProp, 1500, 0.995, {"lr": 0.01, "centered": True}),
    ("adam", Adam, 1500, 0.995, {"lr": 0.05}),
    ("adam-amsgrad", Adam, 1500, 0.995, {"lr": 0.05, "amsgrad": True}),
    ("adamw", AdamW, 1500, 0.995, {"lr": 0.05, "weight_decay": 0.0}),
    # Adadelta has no learning rate in the paper and its step size is sticky --
    # the accumulator of past updates is what sets the next one -- so it needs a
    # gentler anneal and more steps to settle.
    ("adadelta", Adadelta, 3000, 0.998, {"lr": 1.0}),
]


@pytest.mark.parametrize(
    "opt_cls,steps,gamma,kwargs",
    [c[1:] for c in BOWL_CASES],
    ids=[c[0] for c in BOWL_CASES],
)
def test_every_optimizer_finds_the_minimum(opt_cls, steps, gamma, kwargs):
    """Every optimizer drives ``||x - c||`` to zero on a well-conditioned bowl.

    The learning rate is annealed, and that is not a crutch: with a *constant*
    lr the adaptive methods do not converge to a point at all. Their update is
    close to ``lr * sign(g)`` no matter how small ``g`` gets, so they end in a
    limit cycle of radius ``O(lr)`` around the minimum. Testing them at a fixed
    lr therefore measures the lr, not the implementation.
    """
    assert run_bowl_decayed(opt_cls, steps, gamma, **kwargs) < 1e-6


def test_constant_lr_leaves_an_adaptive_method_hovering():
    """The O(lr) limit cycle above, made visible rather than assumed."""
    rng = np.random.default_rng(0)
    _, c = make_bowl(rng, 5)
    start = float(np.linalg.norm(rng.standard_normal(5) - c))
    assert start > 1.0
    hovering = run_bowl(RMSProp, 2000, lr=0.01)
    landed = run_bowl_decayed(RMSProp, 1500, 0.995, lr=0.01)
    assert hovering < 0.05  # it did descend
    assert landed < 1e-9 < hovering  # but only the annealed run actually lands


def test_sgd_momentum_beats_plain_sgd_on_an_ill_conditioned_bowl():
    """Momentum is not decoration: it is what makes a stretched bowl tractable."""
    plain = run_bowl(SGD, 200, cond=200.0, lr=0.004)
    momentum = run_bowl(SGD, 200, cond=200.0, lr=0.004, momentum=0.9)
    assert momentum < plain / 10.0


# --------------------------------------------------------------------------- #
# Adam: the bias-correction signature
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, 1e6])
def test_adam_first_step_magnitude_equals_lr_at_any_gradient_scale(scale):
    """With correct bias correction the first Adam step is ``lr * sign(g)``.

    At ``t=1``: ``m_hat = g`` and ``sqrt(v_hat) = |g|`` exactly, so the ratio is
    ``sign(g)`` regardless of how big ``g`` is. This is the cleanest observable
    test that both corrections are present and applied to the right moments.
    """
    lr = 1e-3
    p = p64([2.0, -3.0, 0.5])
    p.grad = np.array([scale, -scale, scale], dtype=F64)
    before = p.data.copy()
    # eps=1e-16 so the eps term is negligible even for a gradient of 1e-6; with
    # the default 1e-8 the tiny-gradient case is legitimately damped by ~1%.
    Adam([p], lr=lr, eps=1e-16).step()

    delta = p.data - before
    assert np.allclose(np.abs(delta), lr, rtol=1e-9)
    assert np.allclose(np.sign(delta), [-1.0, 1.0, -1.0])


def _first_step_ratio(correct_m: bool, correct_v: bool, b1=0.9, b2=0.999) -> float:
    """First-step size relative to ``lr`` for a given bias-correction mistake."""
    g = 7.0  # any value: the ratio is scale-free
    m = (1.0 - b1) * g
    v = (1.0 - b2) * g * g
    if correct_m:
        m /= 1.0 - b1**1
    if correct_v:
        v /= 1.0 - b2**1
    return float(m / math.sqrt(v))


def test_bias_correction_error_magnitudes_are_the_real_numbers():
    """Omitting *both* corrections is 3.16x, not 31.6x. Only the m-only slip is 31.6x.

    The two errors partly cancel: the uncorrected numerator is ``(1-b1) = 0.1``
    of what it should be, and the uncorrected denominator is
    ``sqrt(1-b2) = 0.0316`` of what it should be, leaving ``0.1/0.0316 = 3.16``.
    Correcting ``m`` and forgetting ``v`` removes the cancelling factor and
    leaves the full ``1/sqrt(1-b2) = 31.6``.
    """
    assert _first_step_ratio(correct_m=True, correct_v=True) == pytest.approx(1.0, rel=1e-12)
    assert _first_step_ratio(correct_m=False, correct_v=False) == pytest.approx(
        3.16227766, rel=1e-6
    )
    assert _first_step_ratio(correct_m=True, correct_v=False) == pytest.approx(31.6227766, rel=1e-6)
    # The remaining slip is the harmless direction: v corrected, m forgotten.
    assert _first_step_ratio(correct_m=False, correct_v=True) == pytest.approx(0.1, rel=1e-12)

    # And the implementation sits at the 1.0 end of that table.
    p = p64([1.0])
    p.grad = np.array([12345.0], dtype=F64)
    Adam([p], lr=0.01, eps=1e-16).step()
    assert abs(1.0 - p.data[0]) == pytest.approx(0.01, rel=1e-9)


def test_adam_eps_is_outside_the_sqrt():
    """``m_hat/(sqrt(v_hat)+eps)`` vs ``g/sqrt(v+eps)``: the floor sits 1e4 apart.

    With ``eps=1e-8`` the wrong form floors the denominator at ``1e-4``, so a
    gradient of ``1e-6`` takes a step ~100x smaller than ``lr``. The correct
    form still takes a step of essentially ``lr``.
    """
    lr, eps = 1e-3, 1e-8
    p = p64([0.0])
    p.grad = np.array([1e-6], dtype=F64)
    Adam([p], lr=lr, eps=eps).step()
    correct_step = abs(p.data[0])

    g = 1e-6
    v_uncorrected = (1.0 - 0.999) * g * g
    wrong_step = lr * g / math.sqrt(v_uncorrected + eps)

    assert correct_step == pytest.approx(lr, rel=1e-2)
    assert correct_step / wrong_step > 90.0  # the floored version steps ~99x shorter


def test_adamw_decouples_weight_decay_and_adam_does_not():
    """AdamW's decay is ``lr*wd*p``; Adam's goes through the second moment."""
    lr, wd = 0.1, 0.5
    grad = np.array([3.0], dtype=F64)

    pw = p64([2.0])
    pw.grad = grad.copy()
    AdamW([pw], lr=lr, weight_decay=wd, eps=1e-16).step()
    # decoupled: p*(1 - lr*wd), then the usual unit-magnitude Adam step
    assert pw.data[0] == pytest.approx(2.0 * (1 - lr * wd) - lr, rel=1e-9)

    pa = p64([2.0])
    pa.grad = grad.copy()
    Adam([pa], lr=lr, weight_decay=wd, eps=1e-16).step()
    # coupled: the decay entered `grad`, and the first Adam step normalises the
    # whole gradient to unit size, so the decay changes nothing but the sign path.
    assert pa.data[0] == pytest.approx(2.0 - lr, rel=1e-9)

    assert pw.data[0] != pytest.approx(pa.data[0], rel=1e-6)


def test_adamw_decay_shrinks_a_zero_gradient_parameter():
    """A parameter with zero gradient still decays under AdamW. Under Adam it cannot.

    Adam's coupled decay reaches the parameter only through ``m`` and ``v``,
    which a zero gradient leaves untouched... except that ``wd*p`` is itself the
    gradient here, so Adam moves it by a full ``lr``. AdamW moves it by
    ``lr*wd*p``, which is the shape of decay people expect.
    """
    pw, pa = p64([4.0]), p64([4.0])
    pw.grad = np.zeros(1, dtype=F64)
    pa.grad = np.zeros(1, dtype=F64)
    AdamW([pw], lr=0.01, weight_decay=0.1, eps=1e-16).step()
    Adam([pa], lr=0.01, weight_decay=0.1, eps=1e-16).step()
    assert pw.data[0] == pytest.approx(4.0 * (1 - 0.01 * 0.1), rel=1e-12)
    assert pa.data[0] == pytest.approx(4.0 - 0.01, rel=1e-9)


def test_sgd_nesterov_uses_the_pytorch_formulation():
    """``buf = mu*buf + g`` then ``d = g + mu*buf`` -- not ``d = buf``."""
    mu, lr = 0.9, 0.1
    g1, g2 = 2.0, -1.0
    p = p64([0.0])
    opt = SGD([p], lr=lr, momentum=mu, nesterov=True)

    p.grad = np.array([g1], dtype=F64)
    opt.step()
    buf = g1  # first step seeds the buffer with the raw gradient
    expected = -lr * (g1 + mu * buf)
    assert p.data[0] == pytest.approx(expected, rel=1e-12)

    p.grad = np.array([g2], dtype=F64)
    opt.step()
    buf = mu * buf + g2
    expected -= lr * (g2 + mu * buf)
    assert p.data[0] == pytest.approx(expected, rel=1e-12)


def test_adadelta_moves_on_its_very_first_step():
    """``eps`` inside the roots is what lets a zero ``acc_delta`` produce a step."""
    p = p64([1.0])
    p.grad = np.array([0.3], dtype=F64)
    Adadelta([p], lr=1.0, rho=0.9, eps=1e-6).step()
    assert p.data[0] != 1.0


# --------------------------------------------------------------------------- #
# param groups and state
# --------------------------------------------------------------------------- #
def test_param_groups_carry_independent_hyperparameters():
    a, b = p64([1.0]), p64([1.0])
    opt = SGD([{"params": [a], "lr": 0.1}, {"params": [b]}], lr=0.01)
    a.grad = np.array([1.0], dtype=F64)
    b.grad = np.array([1.0], dtype=F64)
    opt.step()
    assert a.data[0] == pytest.approx(0.9, rel=1e-12)
    assert b.data[0] == pytest.approx(0.99, rel=1e-12)
    assert opt.param_groups[0]["momentum"] == 0.0  # filled from defaults


def test_duplicate_parameter_across_groups_is_rejected():
    a = p64([1.0])
    with pytest.raises(ValueError, match="already appears"):
        SGD([{"params": [a], "lr": 0.1}, {"params": [a], "lr": 0.2}], lr=0.01)


def test_duplicate_parameter_within_one_group_is_rejected():
    a = p64([1.0])
    with pytest.raises(ValueError, match="already appears"):
        SGD([a, a], lr=0.01)


def test_state_is_keyed_by_identity_not_by_value():
    """Two parameters that happen to hold equal values must not share state."""
    a, b = p64([1.0, 2.0]), p64([1.0, 2.0])
    assert np.array_equal(a.data, b.data)
    opt = Adam([a, b], lr=0.1)
    a.grad = np.array([1.0, 1.0], dtype=F64)
    b.grad = np.array([-5.0, 4.0], dtype=F64)
    opt.step()
    assert len(opt.state) == 2
    assert not np.allclose(opt.state[id(a)]["exp_avg"], opt.state[id(b)]["exp_avg"])


def test_zero_grad_set_to_none_skips_the_parameter_entirely():
    """A ``None`` gradient must not advance momentum; a zero gradient does."""
    a, b = p64([1.0]), p64([1.0])
    opt = SGD([a, b], lr=0.1, momentum=0.9)
    a.grad = np.array([1.0], dtype=F64)
    b.grad = np.array([1.0], dtype=F64)
    opt.step()
    after = (a.data[0], b.data[0])

    opt.zero_grad(set_to_none=True)
    a.grad = None
    b.grad = np.zeros(1, dtype=F64)  # explicit zero: momentum still carries it
    opt.step()
    assert a.data[0] == after[0]
    assert b.data[0] < after[1]


# --------------------------------------------------------------------------- #
# clipping
# --------------------------------------------------------------------------- #
def test_clip_grad_norm_returns_pre_clip_norm_and_scales_to_max_norm():
    rng = np.random.default_rng(11)
    params = [p64(rng.standard_normal(s)) for s in [(7, 11), (5,), (3, 2)]]
    for p in params:
        p.grad = rng.standard_normal(p.data.shape) * 10.0
    flat = np.concatenate([p.grad.ravel() for p in params])
    expected = float(np.linalg.norm(flat))
    direction = flat / expected

    total = clip_grad_norm_(params, max_norm=1.0)
    assert total == pytest.approx(expected, rel=1e-12)

    after = np.concatenate([p.grad.ravel() for p in params])
    assert float(np.linalg.norm(after)) == pytest.approx(1.0, rel=1e-5)
    # A global rescale changes length only -- the direction must be untouched.
    assert np.allclose(after / np.linalg.norm(after), direction, atol=1e-12)


def test_clip_grad_norm_is_a_noop_below_the_threshold():
    p = p64([0.0, 0.0])
    p.grad = np.array([0.3, 0.4], dtype=F64)
    total = clip_grad_norm_([p], max_norm=10.0)
    assert total == pytest.approx(0.5, rel=1e-12)
    assert np.allclose(p.grad, [0.3, 0.4])


def test_clip_grad_norm_inf_type_and_empty_input():
    p = p64([0.0, 0.0, 0.0])
    p.grad = np.array([1.0, -6.0, 2.0], dtype=F64)
    total = clip_grad_norm_([p], max_norm=3.0, norm_type=float("inf"))
    assert total == pytest.approx(6.0, rel=1e-12)
    assert float(np.max(np.abs(p.grad))) == pytest.approx(3.0, rel=1e-5)

    q = p64([1.0])
    assert clip_grad_norm_([q], max_norm=1.0) == 0.0  # grad is None: nothing to do


def test_clip_grad_value_clamps_elementwise():
    p = p64(np.zeros(4))
    p.grad = np.array([-9.0, -0.5, 0.25, 12.0], dtype=F64)
    clip_grad_value_([p], 1.0)
    assert np.allclose(p.grad, [-1.0, -0.5, 0.25, 1.0])


# --------------------------------------------------------------------------- #
# schedulers
# --------------------------------------------------------------------------- #
def one_group_opt(lr: float = 1.0) -> SGD:
    return SGD([p64([0.0])], lr=lr)


def lr_sequence(sched, n: int) -> list[float]:
    """The lr in effect on each of the first ``n`` steps, stepping at the end."""
    out = []
    for _ in range(n):
        out.append(sched.optimizer.param_groups[0]["lr"])
        sched.step()
    return out


def test_step_lr_exact_sequence():
    sched = StepLR(one_group_opt(1.0), step_size=3, gamma=0.1)
    assert lr_sequence(sched, 8) == pytest.approx(
        [1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01], rel=1e-12
    )


def test_multi_step_lr_exact_sequence():
    sched = MultiStepLR(one_group_opt(1.0), milestones=[2, 5], gamma=0.5)
    assert lr_sequence(sched, 7) == pytest.approx([1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25], rel=1e-12)


def test_exponential_lr_exact_sequence():
    sched = ExponentialLR(one_group_opt(0.5), gamma=0.9)
    assert lr_sequence(sched, 5) == pytest.approx([0.5 * 0.9**t for t in range(5)], rel=1e-12)


def test_cosine_annealing_exact_sequence():
    t_max, eta_min, base = 4, 0.1, 1.0
    sched = CosineAnnealingLR(one_group_opt(base), T_max=t_max, eta_min=eta_min)
    expected = [
        eta_min + (base - eta_min) * (1 + math.cos(math.pi * t / t_max)) / 2 for t in range(6)
    ]
    assert lr_sequence(sched, 6) == pytest.approx(expected, rel=1e-12)
    assert expected[t_max] == pytest.approx(eta_min, abs=1e-15)


def test_linear_warmup_exact_sequence():
    sched = LinearWarmup(one_group_opt(0.2), warmup_steps=4, start_factor=0.0)
    assert lr_sequence(sched, 6) == pytest.approx(
        [0.0, 0.05, 0.10, 0.15, 0.20, 0.20], rel=1e-12, abs=1e-15
    )


def test_warmup_cosine_matches_the_nanogpt_shape():
    warmup, total, floor, base = 3, 13, 0.1, 1.0  # (total - warmup) even, so a true midpoint exists
    sched = WarmupCosine(
        one_group_opt(base), warmup_steps=warmup, total_steps=total, min_lr_ratio=floor
    )
    seq = lr_sequence(sched, 16)

    assert seq[:warmup] == pytest.approx([base * (t + 1) / (warmup + 1) for t in range(warmup)])
    assert seq[warmup] == pytest.approx(base, rel=1e-12)  # ramp joins the cosine at 1.0
    assert seq[total] == pytest.approx(base * floor, rel=1e-12)
    assert seq[total + 1] == pytest.approx(base * floor, rel=1e-12)  # holds, never re-warms
    assert all(seq[i] >= seq[i + 1] for i in range(warmup, total))  # monotone decay
    mid = warmup + (total - warmup) // 2
    assert seq[mid] == pytest.approx(base * (floor + 0.5 * (1 - floor)), rel=1e-12)


def test_scheduler_keeps_per_group_learning_rates_independent():
    """The multiplier form is what makes this hold; overwriting group['lr'] loses it."""
    a, b = p64([0.0]), p64([0.0])
    opt = SGD([{"params": [a], "lr": 1e-1}, {"params": [b], "lr": 1e-3}], lr=1e-2)
    sched = StepLR(opt, step_size=2, gamma=0.5)
    for _ in range(7):
        g0, g1 = opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]
        assert g0 / g1 == pytest.approx(100.0, rel=1e-12)
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-1 * 0.5**3, rel=1e-12)
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-3 * 0.5**3, rel=1e-12)


def test_schedule_is_idempotent_and_queryable_off_the_current_epoch():
    """Stepping to the same epoch twice must not decay twice."""
    opt = one_group_opt(1.0)
    sched = ExponentialLR(opt, gamma=0.5)
    sched.step(epoch=4)
    first = opt.param_groups[0]["lr"]
    sched.step(epoch=4)
    assert opt.param_groups[0]["lr"] == pytest.approx(first, rel=1e-15)
    assert first == pytest.approx(0.5**4, rel=1e-12)
    # get_lr is pure: asking about a future epoch does not move anything.
    assert sched.get_lr(9)[0] == pytest.approx(0.5**9, rel=1e-12)
    assert opt.param_groups[0]["lr"] == pytest.approx(first, rel=1e-15)


def test_two_schedulers_on_one_optimizer_anchor_to_the_same_base_lr():
    """``initial_lr`` is set once. A second scheduler must not re-anchor to a decayed lr."""
    opt = one_group_opt(1.0)
    first = StepLR(opt, step_size=1, gamma=0.5)
    for _ in range(3):
        first.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5**3, rel=1e-12)
    second = ExponentialLR(opt, gamma=0.9)
    assert second.base_lrs == [1.0]
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0, rel=1e-12)


def test_chained_scheduler_multiplies_factors():
    opt = one_group_opt(1.0)
    warm = LinearWarmup(opt, warmup_steps=3, start_factor=0.0)
    decay = ExponentialLR(opt, gamma=0.9)
    sched = ChainedScheduler(opt, [warm, decay])
    expected = [min(t, 3) / 3 * 0.9**t for t in range(6)]
    assert lr_sequence(sched, 6) == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_sequential_lr_restarts_the_child_clock_at_the_milestone():
    opt = one_group_opt(1.0)
    warm = LinearWarmup(opt, warmup_steps=2, start_factor=0.5)
    cos = CosineAnnealingLR(opt, T_max=4, eta_min=0.0)
    sched = SequentialLR(opt, [warm, cos], milestones=[3])
    seq = lr_sequence(sched, 8)
    assert seq[:3] == pytest.approx([0.5, 0.75, 1.0], rel=1e-12)
    # The cosine sees local epochs 0..4 from epoch 3 onward, so it starts at 1.0.
    assert seq[3:8] == pytest.approx(
        [(1 + math.cos(math.pi * t / 4)) / 2 for t in range(5)], rel=1e-12, abs=1e-15
    )


def test_scheduler_state_dict_round_trip():
    opt = one_group_opt(1.0)
    sched = StepLR(opt, step_size=2, gamma=0.5)
    for _ in range(5):
        sched.step()
    saved = sched.state_dict()
    expected = sched.get_last_lr()

    other = StepLR(one_group_opt(1.0), step_size=99, gamma=0.99)
    other.load_state_dict(saved)
    assert other.last_epoch == sched.last_epoch
    assert other.get_last_lr() == pytest.approx(expected, rel=1e-15)
    assert "optimizer" not in saved


def test_constant_lr_is_a_true_noop():
    sched = ConstantLR(one_group_opt(0.7))
    assert lr_sequence(sched, 4) == pytest.approx([0.7] * 4, rel=1e-15)


# --------------------------------------------------------------------------- #
# optimizer state_dict round trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "opt_cls,kwargs",
    [
        (SGD, {"lr": 0.05, "momentum": 0.9}),
        (Adam, {"lr": 0.05}),
        (AdamW, {"lr": 0.05, "weight_decay": 0.01}),
        (RMSProp, {"lr": 0.02, "momentum": 0.5, "centered": True}),
        (Adagrad, {"lr": 0.2}),
        (Adadelta, {"lr": 1.0}),
    ],
)
def test_state_dict_round_trip_reproduces_the_exact_trajectory(opt_cls, kwargs):
    """Save mid-training, corrupt the optimizer, restore, and get identical steps."""
    rng = np.random.default_rng(7)
    a, c = make_bowl(rng, 5)
    start = rng.standard_normal(5)

    def train(steps: int, x, opt):
        for _ in range(steps):
            opt.zero_grad()
            bowl_loss(x, a, c).backward()
            opt.step()

    x = p64(start)
    opt = opt_cls([x], **kwargs)
    train(5, x, opt)

    snapshot = opt.state_dict()
    params_snapshot = x.data.copy()

    train(5, x, opt)
    reference = x.data.copy()

    # Perturb hard: garbage steps from a different point, so any surviving
    # momentum/moment buffer would show up in the next five steps.
    x.data[...] = rng.standard_normal(5) * 10.0
    train(4, x, opt)

    x.data[...] = params_snapshot
    opt.load_state_dict(snapshot)
    train(5, x, opt)
    assert np.array_equal(x.data, reference)


def test_state_dict_snapshot_does_not_alias_live_buffers():
    x = p64([1.0, 2.0])
    opt = Adam([x], lr=0.1)
    x.grad = np.array([1.0, -1.0], dtype=F64)
    opt.step()
    snap = opt.state_dict()
    saved = snap["state"][0]["exp_avg"].copy()
    opt.step()
    assert np.array_equal(snap["state"][0]["exp_avg"], saved)


def test_state_dict_indexes_by_position_across_groups():
    a, b = p64([1.0]), p64([2.0])
    opt = SGD([{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.2}], lr=0.01, momentum=0.9)
    a.grad = np.array([1.0], dtype=F64)
    b.grad = np.array([1.0], dtype=F64)
    opt.step()
    sd = opt.state_dict()
    assert [g["params"] for g in sd["param_groups"]] == [[0], [1]]
    assert sorted(sd["state"]) == [0, 1]
    assert sd["param_groups"][0]["lr"] == 0.1

    a2, b2 = p64([1.0]), p64([2.0])
    opt2 = SGD([{"params": [a2], "lr": 0.0}, {"params": [b2], "lr": 0.0}], lr=0.0, momentum=0.9)
    opt2.load_state_dict(sd)
    assert opt2.param_groups[0]["lr"] == 0.1
    assert id(a2) in opt2.state and id(b2) in opt2.state


def test_load_state_dict_rejects_a_mismatched_layout():
    a, b = p64([1.0]), p64([2.0])
    sd = SGD([a, b], lr=0.1).state_dict()
    with pytest.raises(ValueError, match="param group"):
        SGD([p64([1.0])], lr=0.1).load_state_dict(sd)


def test_optimizer_rejects_non_tensors_and_empty_lists():
    with pytest.raises(ValueError, match="empty"):
        SGD([], lr=0.1)
    with pytest.raises(TypeError, match="Tensors"):
        SGD([np.zeros(3)], lr=0.1)
    with pytest.raises(TypeError, match="single Tensor"):
        SGD(p64([1.0]), lr=0.1)
    with pytest.raises(ValueError, match="nesterov"):
        SGD([p64([1.0])], lr=0.1, nesterov=True)


def test_base_optimizer_has_no_update_rule():
    opt = Optimizer([p64([1.0])], {"lr": 0.1})
    next(iter(opt.parameters())).grad = np.ones(1, dtype=F64)
    with pytest.raises(NotImplementedError):
        opt.step()


# --------------------------------------------------------------------------- #
# optimizer + scheduler together
# --------------------------------------------------------------------------- #
def test_training_loop_with_warmup_cosine_converges():
    rng = np.random.default_rng(5)
    a, c = make_bowl(rng, 5)
    x = p64(rng.standard_normal(5))
    opt = AdamW([x], lr=0.2, weight_decay=0.0)
    sched = WarmupCosine(opt, warmup_steps=20, total_steps=400, min_lr_ratio=0.01)
    for _ in range(400):
        opt.zero_grad()
        bowl_loss(x, a, c).backward()
        opt.step()
        sched.step()
    assert float(np.linalg.norm(x.data - c)) < 1e-5
    assert sched.get_last_lr()[0] == pytest.approx(0.2 * 0.01, rel=1e-12)


# --------------------------------------------------------------------------- #
# shape guards, uniform validation, and mid-run hyper-parameter changes
# --------------------------------------------------------------------------- #
ALL_OPTIMIZERS = [SGD, Adagrad, RMSProp, Adam, AdamW, Adadelta]


def test_step_rejects_a_gradient_that_only_broadcasts():
    """A grad of the wrong shape must not be broadcast over the whole parameter."""
    p = p64([1.0, 2.0, 3.0])
    opt = SGD([p], lr=0.1)

    p.grad = np.array([1.0], dtype=F64)  # broadcasts silently over all three entries
    with pytest.raises(ValueError, match=r"\(1,\).*\(3,\)"):
        opt.step()

    p.grad = np.array(1.0, dtype=F64)  # 0-d, broadcasts too
    with pytest.raises(ValueError, match="does not match parameter shape"):
        opt.step()

    # The guard fires before any state is allocated or any update applied.
    assert np.allclose(p.data, [1.0, 2.0, 3.0])
    assert opt.state == {}


@pytest.mark.parametrize("opt_cls", ALL_OPTIMIZERS)
def test_every_optimizer_rejects_a_negative_weight_decay(opt_cls):
    """Negative decay does not regularise -- it grows the weights it should shrink."""
    with pytest.raises(ValueError, match="weight_decay"):
        opt_cls([p64([1.0])], weight_decay=-1e-4)


@pytest.mark.parametrize("opt_cls", ALL_OPTIMIZERS)
def test_every_optimizer_rejects_a_negative_lr(opt_cls):
    with pytest.raises(ValueError, match="lr"):
        opt_cls([p64([1.0])], lr=-0.1)


def test_rmsprop_rejects_a_negative_momentum():
    with pytest.raises(ValueError, match="momentum"):
        RMSProp([p64([1.0])], momentum=-0.5)


def test_adagrad_rejects_a_negative_lr_decay():
    with pytest.raises(ValueError, match="lr_decay"):
        Adagrad([p64([1.0])], lr_decay=-1.0)


@pytest.mark.parametrize(
    ("opt_cls", "kwargs", "match"),
    [
        (Adam, {"betas": (1.0, 0.999)}, "beta1"),
        (Adam, {"betas": (0.9, 1.0)}, "beta2"),
        (AdamW, {"betas": (-0.1, 0.999)}, "beta1"),
        (RMSProp, {"alpha": 1.0}, "alpha"),
        (Adadelta, {"rho": 1.0}, "rho"),
        (Adam, {"eps": 0.0}, "eps"),
        (AdamW, {"eps": -1e-8}, "eps"),
        (Adagrad, {"eps": 0.0}, "eps"),
        (RMSProp, {"eps": 0.0}, "eps"),
        (Adadelta, {"eps": 0.0}, "eps"),
    ],
)
def test_decay_rates_and_eps_are_validated(opt_cls, kwargs, match):
    with pytest.raises(ValueError, match=match):
        opt_cls([p64([1.0])], **kwargs)


def test_clip_grad_norm_validates_max_norm_and_norm_type():
    """``norm_type=0`` used to reach ``acc ** (1/0)`` and raise a bare ZeroDivisionError."""
    p = p64([1.0, 2.0])
    p.grad = np.array([3.0, 4.0], dtype=F64)

    for bad in (0.0, -2.0):
        with pytest.raises(ValueError, match="norm_type"):
            clip_grad_norm_([p], max_norm=1.0, norm_type=bad)
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="max_norm"):
            clip_grad_norm_([p], max_norm=bad)
    assert np.allclose(p.grad, [3.0, 4.0])  # no gradient was touched

    # max_norm == 0 is NOT an error: it means "scale every gradient to zero",
    # which PyTorch supports. Rejecting it would remove a capability rather than
    # prevent a mistake.
    zeroed = p64([1.0, 2.0])
    zeroed.grad = np.array([3.0, 4.0], dtype=F64)
    assert clip_grad_norm_([zeroed], max_norm=0.0) == pytest.approx(5.0)  # pre-clip norm
    assert np.allclose(zeroed.grad, [0.0, 0.0])

    # Validation happens before the "no gradients" short circuit, and inf is
    # still a legal norm_type.
    with pytest.raises(ValueError, match="max_norm"):
        clip_grad_norm_([p64([1.0])], max_norm=-1.0)
    assert clip_grad_norm_([p], max_norm=10.0, norm_type=float("inf")) == pytest.approx(4.0)


def test_sgd_momentum_raised_after_the_first_step_allocates_its_buffer():
    """Buffers are allocated on first use, so momentum may be switched on mid-run."""
    p = p64([1.0])
    opt = SGD([p], lr=0.1, momentum=0.0)
    p.grad = np.array([1.0], dtype=F64)
    opt.step()
    assert "momentum_buffer" not in opt.state[id(p)]  # not paid for while unused

    opt.param_groups[0]["momentum"] = 0.9
    p.grad = np.array([1.0], dtype=F64)
    opt.step()  # used to raise KeyError: 'momentum_buffer'
    # Seeded with the raw gradient, exactly as a run that started with momentum.
    assert np.allclose(opt.state[id(p)]["momentum_buffer"], [1.0])
    assert p.data[0] == pytest.approx(0.8, rel=1e-12)


def test_rmsprop_momentum_and_centering_can_be_switched_on_mid_run():
    p = p64([1.0])
    opt = RMSProp([p], lr=0.01, momentum=0.0, centered=False)
    p.grad = np.array([1.0], dtype=F64)
    opt.step()
    assert "momentum_buffer" not in opt.state[id(p)]
    assert "grad_avg" not in opt.state[id(p)]

    opt.param_groups[0]["momentum"] = 0.9
    opt.param_groups[0]["centered"] = True
    p.grad = np.array([1.0], dtype=F64)
    opt.step()  # used to raise KeyError on either buffer
    assert "momentum_buffer" in opt.state[id(p)]
    assert "grad_avg" in opt.state[id(p)]


def test_adam_amsgrad_can_be_switched_on_mid_run():
    p = p64([1.0])
    opt = Adam([p], lr=0.1, amsgrad=False)
    p.grad = np.array([1.0], dtype=F64)
    opt.step()
    assert "max_exp_avg_sq" not in opt.state[id(p)]

    opt.param_groups[0]["amsgrad"] = True
    p.grad = np.array([1.0], dtype=F64)
    opt.step()  # used to raise KeyError: 'max_exp_avg_sq'
    st = opt.state[id(p)]
    assert np.allclose(st["max_exp_avg_sq"], st["exp_avg_sq"])


def test_param_group_added_after_the_scheduler_is_a_loud_error():
    """base_lrs is frozen at construction, so a later group would never be scheduled."""
    opt = one_group_opt(lr=1.0)
    sched = StepLR(opt, step_size=1, gamma=0.5)
    opt.add_param_group({"params": [p64([0.0])]})
    with pytest.raises(ValueError, match="zip"):
        sched.step()
