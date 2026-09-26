"""Tests for ``scratchgrad.utils``: checkpoints, seeding, meters, plots.

There are no differentiable ops in this subsystem, so there is nothing here to
gradcheck; the corresponding "did it really work" question for utils is whether
a restored checkpoint is *bitwise* identical and whether a snapshot actually
survives the next optimizer step. Both are asserted below.

Imports are from concrete modules (``scratchgrad.utils.serialization``) rather
than from the package, so that nothing here depends on an ``__init__.py`` edit.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg", force=True)  # before any pyplot import, or it has no effect

import numpy as np
import pytest

from scratchgrad.nn.layers import Linear
from scratchgrad.nn.module import Module, Sequential
from scratchgrad.utils.meters import Accuracy, AverageMeter, ProgressBar, Timer, format_time
from scratchgrad.utils.seed import RNGContext, seed_everything, spawn
from scratchgrad.utils.serialization import (
    SCHEMA_VERSION,
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
    snapshot,
)
from scratchgrad.utils.viz import (
    confusion_matrix,
    plot_confusion,
    plot_filters_conv,
    plot_history,
    plot_weights,
)

# Prime shapes: with square ones a transposed matmul stays conformable and the
# resulting garbage never raises.
N, D, H, C = 7, 11, 5, 3


class TinySGD:
    """A minimal in-place SGD, standing in for ``scratchgrad.optim``.

    ``p.data -= lr * g`` -- in place, exactly as a real optimizer writes -- is
    precisely what makes a shallow ``state_dict()`` snapshot worthless, so the
    stand-in has to behave that way for the test below to mean anything.
    """

    def __init__(self, params, lr: float = 0.1) -> None:
        self.params = list(params)
        self.lr = float(lr)
        self.step_count = 0
        self.momentum_buffers = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, buf in zip(self.params, self.momentum_buffers, strict=True):
            g = p.grad if p.grad is not None else np.ones_like(p.data)
            buf *= 0.9
            buf += g
            p.data -= self.lr * buf
        self.step_count += 1

    def state_dict(self):
        return {
            "lr": self.lr,
            "step_count": self.step_count,
            "momentum_buffers": list(self.momentum_buffers),
        }

    def load_state_dict(self, state) -> None:
        self.lr = float(state["lr"])
        self.step_count = int(state["step_count"])
        self.momentum_buffers = [np.asarray(b) for b in state["momentum_buffers"]]


def make_model(rng: np.random.Generator) -> Module:
    return Sequential(Linear(D, H, rng=rng), Linear(H, C, rng=rng))


def randomize_(model: Module, rng: np.random.Generator) -> None:
    """Overwrite every parameter with fresh noise, in place."""
    for p in model.parameters():
        p.data[...] = rng.normal(size=p.data.shape).astype(p.data.dtype)


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip_is_bitwise_exact(tmp_path):
    rng = np.random.default_rng(0)
    model = make_model(rng)
    randomize_(model, rng)
    saved = {k: np.array(v, copy=True) for k, v in model.state_dict(deep=True).items()}

    path = save_checkpoint(tmp_path / "ck", model, meta={"epoch": 3, "loss": 0.25})
    assert os.path.getsize(path) > 0

    other = make_model(np.random.default_rng(1))
    randomize_(other, np.random.default_rng(99))
    meta = load_checkpoint(path, other)

    assert meta == {"epoch": 3, "loss": 0.25}
    restored = other.state_dict(deep=True)
    assert set(restored) == set(saved)
    for k in saved:
        # array_equal, not allclose: a checkpoint that only round-trips to
        # 1e-7 has lost the low bits of float32 and is not a checkpoint.
        assert np.array_equal(restored[k], saved[k]), f"{k} changed across save/load"
        assert restored[k].dtype == saved[k].dtype


def test_checkpoint_suffix_is_normalized(tmp_path):
    model = make_model(np.random.default_rng(0))
    path = save_checkpoint(tmp_path / "ck", model)
    assert path.endswith(".npz")
    # Loading through the un-suffixed name must find the same file.
    load_checkpoint(tmp_path / "ck", make_model(np.random.default_rng(0)))


def test_checkpoint_stores_schema_and_shapes(tmp_path):
    model = make_model(np.random.default_rng(0))
    path = save_checkpoint(tmp_path / "ck", model)
    import json

    with np.load(path, allow_pickle=False) as z:
        manifest = json.loads(str(z["__manifest__"].item()))
    assert manifest["schema"] == SCHEMA_VERSION
    assert manifest["model"]["0.weight"]["shape"] == [D, H]
    assert manifest["model"]["1.bias"]["shape"] == [C]


def test_shape_mismatch_raises_and_does_not_broadcast(tmp_path):
    model = make_model(np.random.default_rng(0))
    randomize_(model, np.random.default_rng(0))
    path = save_checkpoint(tmp_path / "ck", model)

    wider = Sequential(Linear(D, H + 4, rng=np.random.default_rng(2)), Linear(H + 4, C))
    before = {k: np.array(v, copy=True) for k, v in wider.state_dict(deep=True).items()}

    with pytest.raises(CheckpointError, match="shape mismatch"):
        load_checkpoint(path, wider)

    # The failure must be atomic: nothing partially loaded.
    after = wider.state_dict(deep=True)
    for k in before:
        assert np.array_equal(before[k], after[k]), f"{k} was mutated by a failed load"


def test_optimizer_state_roundtrips(tmp_path):
    model = make_model(np.random.default_rng(0))
    opt = TinySGD(model.parameters(), lr=0.03)
    opt.step()
    opt.step()
    path = save_checkpoint(tmp_path / "ck", model, optimizer=opt, meta={"note": "resume me"})

    model2 = make_model(np.random.default_rng(0))
    opt2 = TinySGD(model2.parameters(), lr=1.0)
    meta = load_checkpoint(path, model2, optimizer=opt2)

    assert meta["note"] == "resume me"
    assert opt2.lr == pytest.approx(0.03)
    assert opt2.step_count == 2
    for a, b in zip(opt.momentum_buffers, opt2.momentum_buffers, strict=True):
        assert np.array_equal(a, b)


def test_real_adam_state_roundtrips(tmp_path):
    """Interop with the actual optimizer: nested groups, int keys, tuples.

    ``Adam.state_dict()`` mixes arrays (the moments) with hyperparameters and
    uses **integer** state keys. JSON stringifies those keys, so the loader must
    tolerate it -- which it does, since it calls ``int(index)``.
    """
    Adam = pytest.importorskip("scratchgrad.optim.optimizer").Adam
    rng = np.random.default_rng(11)
    model = make_model(rng)
    opt = Adam(model.parameters(), lr=0.004)
    for p in model.parameters():
        p.grad = rng.normal(size=p.data.shape).astype(p.data.dtype)
    opt.step()
    opt.step()

    path = save_checkpoint(tmp_path / "adam", model, optimizer=opt, meta={"epoch": 1})
    model2 = make_model(np.random.default_rng(0))
    opt2 = Adam(model2.parameters(), lr=99.0)
    load_checkpoint(path, model2, optimizer=opt2)

    assert opt2.param_groups[0]["lr"] == pytest.approx(0.004)
    # Compare through state_dict(), which is keyed by parameter *position*.
    # opt.state is keyed by id(), and the two optimizers hold different objects.
    before, after = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert set(before) == set(after) and before
    for idx, a in before.items():
        b = after[idx]
        for k, v in a.items():
            if isinstance(v, np.ndarray):
                assert np.array_equal(v, b[k]), f"Adam state {k} changed for param {idx}"
            else:
                assert v == b[k]


def test_optimizer_requested_but_absent_raises(tmp_path):
    model = make_model(np.random.default_rng(0))
    path = save_checkpoint(tmp_path / "ck", model)  # no optimizer saved
    with pytest.raises(CheckpointError, match="no optimizer state"):
        load_checkpoint(path, make_model(np.random.default_rng(0)), optimizer=TinySGD([]))


def test_load_rejects_a_foreign_npz(tmp_path):
    bogus = tmp_path / "bogus.npz"
    np.savez_compressed(bogus, x=np.arange(4))
    with pytest.raises(CheckpointError, match="not a scratchgrad checkpoint"):
        load_checkpoint(bogus, make_model(np.random.default_rng(0)))


def test_meta_round_trip_preserves_key_types_and_tuples(tmp_path):
    """A checkpoint must not silently change the *type* of the metadata it holds.

    JSON has no tuple and no non-string object key, so the naive encoding
    (``str(k)``, ``list(t)``) hands back ``{'0': 1.0}`` for ``{0: 1.0}`` -- a
    per-class metric dict that no longer indexes by class id, and a shape tuple
    that is now a list. Both are tagged on the way out and restored on the way
    back in.
    """
    meta = {
        "per_class": {0: 1.0, 1: 0.5},
        "input_shape": (1, 28, 28),
        "grid": {(0, 1): "cell"},
        "layers": [(3, 3), (5, 5)],
        "epoch": 4,
        "tags": ["a", "b"],
    }
    model = make_model(np.random.default_rng(0))
    path = save_checkpoint(tmp_path / "meta", model, meta=meta)
    back = load_checkpoint(path, make_model(np.random.default_rng(0)))

    assert back == meta
    assert list(back["per_class"]) == [0, 1], "int keys came back stringified"
    assert isinstance(back["input_shape"], tuple)
    assert isinstance(next(iter(back["grid"])), tuple), "a tuple key must stay hashable"
    assert isinstance(back["layers"][0], tuple)
    assert isinstance(back["tags"], list), "a list must not be promoted to a tuple"


def test_reserved_tags_cannot_be_metadata_keys(tmp_path):
    """Each tag owns the whole JSON object it appears in.

    A caller key spelled the same way would be read back as an array, a tuple
    or a re-keyed dict instead of as itself, so it is refused at write time
    rather than corrupted at read time.
    """
    model = make_model(np.random.default_rng(0))
    for tag in ("__ndarray__", "__tuple__", "__keyed__"):
        with pytest.raises(ValueError, match="reserved"):
            save_checkpoint(tmp_path / "bad", model, meta={tag: [1, 2]})


def test_manifest_that_lies_about_a_shape_still_raises_checkpoint_error(tmp_path):
    """The up-front check trusts the manifest; the arrays are the ground truth.

    When the two disagree -- a truncated or hand-edited file -- the mismatch
    only surfaces inside ``Module.load_state_dict``, and the documented
    ``CheckpointError`` must be what escapes, not a bare numpy ``ValueError``.
    """
    model = make_model(np.random.default_rng(0))
    path = save_checkpoint(tmp_path / "ck", model)
    with np.load(path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["a0"] = np.zeros((2, 3), dtype=payload["a0"].dtype)  # manifest still says (D, H)
    np.savez_compressed(path, **payload)

    with pytest.raises(CheckpointError, match="shape mismatch"):
        load_checkpoint(path, make_model(np.random.default_rng(0)))


def test_snapshot_survives_a_step_but_shallow_state_dict_does_not():
    """The headline hazard: ``state_dict()`` aliases the live parameters.

    A shallow snapshot taken for early stopping is overwritten in place by the
    very next optimizer step, so "restore the best epoch" silently restores the
    latest one. ``snapshot()`` (``deep=True``) is the version that works.
    """
    rng = np.random.default_rng(7)
    model = make_model(rng)
    randomize_(model, rng)

    shallow = model.state_dict()  # deep=False: views onto the live buffers
    deep = snapshot(model)

    truth = {k: np.array(v, copy=True) for k, v in model.state_dict(deep=True).items()}

    opt = TinySGD(model.parameters(), lr=0.5)
    for p in model.parameters():
        p.grad = np.ones_like(p.data)
    opt.step()

    key = "0.weight"
    # The step actually moved the parameters, or the test proves nothing.
    live = dict(model.state_dict(deep=False))
    assert not np.array_equal(live[key], truth[key])

    # The shallow "snapshot" tracked the step -- it was never a snapshot.
    assert np.shares_memory(shallow[key], live[key])
    assert np.array_equal(shallow[key], live[key])
    assert not np.array_equal(shallow[key], truth[key])

    # The deep snapshot still holds the pre-step values, bit for bit.
    assert not np.shares_memory(deep[key], live[key])
    for k, v in truth.items():
        assert np.array_equal(deep[k], v)

    # And restoring it genuinely rewinds the model.
    model.load_state_dict(deep)
    for k, v in model.state_dict(deep=True).items():
        assert np.array_equal(v, truth[k])


def test_saved_checkpoint_is_immune_to_a_later_step(tmp_path):
    """Saving must capture the state at call time, not a live view of it."""
    rng = np.random.default_rng(3)
    model = make_model(rng)
    randomize_(model, rng)
    truth = {k: np.array(v, copy=True) for k, v in model.state_dict(deep=True).items()}

    path = save_checkpoint(tmp_path / "ck", model)
    opt = TinySGD(model.parameters(), lr=0.5)
    for p in model.parameters():
        p.grad = np.ones_like(p.data)
    opt.step()

    load_checkpoint(path, model)
    for k, v in model.state_dict(deep=True).items():
        assert np.array_equal(v, truth[k])


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def test_seed_everything_gives_bit_identical_draws():
    a = seed_everything(1234).normal(size=(N, D))
    b = seed_everything(1234).normal(size=(N, D))
    assert np.array_equal(a, b)  # bit-identical, not allclose
    c = seed_everything(1235).normal(size=(N, D))
    assert not np.array_equal(a, c)


def test_seed_everything_pins_the_global_streams():
    import random

    seed_everything(0)
    a = (random.random(), float(np.random.rand()))
    seed_everything(0)
    b = (random.random(), float(np.random.rand()))
    assert a == b


def test_seed_everything_rejects_a_non_int():
    with pytest.raises(TypeError):
        seed_everything(1.5)


def test_rng_context_restores_the_global_state():
    import random

    seed_everything(5)
    before_py, before_np = random.getstate(), np.random.get_state()

    with RNGContext(99) as rng:
        inner = rng.normal(size=(D,))
    with RNGContext(99) as rng2:
        assert np.array_equal(rng2.normal(size=(D,)), inner)

    after_py, after_np = random.getstate(), np.random.get_state()
    assert before_py == after_py
    assert np.array_equal(before_np[1], after_np[1])


def test_spawn_children_are_reproducible_and_distinct():
    kids = spawn(np.random.default_rng(0), 3)
    draws = [k.normal(size=(D,)) for k in kids]
    again = [k.normal(size=(D,)) for k in spawn(np.random.default_rng(0), 3)]
    for x, y in zip(draws, again, strict=True):
        assert np.array_equal(x, y)
    assert not np.array_equal(draws[0], draws[1])


# --------------------------------------------------------------------------- #
# meters
# --------------------------------------------------------------------------- #
def test_average_meter_weights_by_count():
    m = AverageMeter("loss")
    m.update(1.0, n=8)
    m.update(0.0, n=2)
    # Count-weighted: 8/10, not the unweighted 0.5 of the two batch means.
    assert m.avg == pytest.approx(0.8)
    assert m.count == 10
    assert m.val == pytest.approx(0.0)
    m.reset()
    assert m.avg == 0.0 and m.count == 0


def test_accuracy_over_batches_and_topk():
    acc = Accuracy()
    logits = np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 5.0]])
    targets = np.array([0, 1, 0])
    assert acc.update(logits, targets) == pytest.approx(2 / 3)
    acc.update(np.array([[9.0, 0.0, 0.0]]), np.array([0]))
    assert acc.value == pytest.approx(3 / 4)

    top2 = Accuracy(topk=2)
    # Row 2's true class 0 is the runner-up, so top-2 catches it and top-1 did not.
    assert top2.update(logits, targets) == pytest.approx(1.0)

    plain = Accuracy()
    assert plain.update(np.array([0, 1, 2]), np.array([0, 1, 1])) == pytest.approx(2 / 3)


def test_accuracy_accepts_one_hot_targets():
    acc = Accuracy()
    logits = np.array([[3.0, 1.0], [0.0, 2.0]])
    onehot = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert acc.update(logits, onehot) == pytest.approx(0.5)


def test_progress_bar_works_without_tqdm(capsys):
    import io

    buf = io.StringIO()
    bar = ProgressBar(range(5), desc="epoch", stream=buf, use_tqdm=False, width=10)
    bar.set_postfix(loss=0.5)
    assert list(bar) == [0, 1, 2, 3, 4]
    out = buf.getvalue()
    assert "\r" in out and "epoch" in out and "5/5" in out
    assert out.endswith("\n")


def test_progress_bar_uses_tqdm_when_available():
    import io

    tqdm = pytest.importorskip("tqdm")
    assert tqdm is not None
    buf = io.StringIO()
    assert list(ProgressBar(range(3), stream=buf, use_tqdm=True)) == [0, 1, 2]


def test_timer_and_format_time():
    with Timer("t") as t:
        _ = sum(range(1000))
    assert t.elapsed >= 0.0
    assert t.total >= t.elapsed - 1e-9
    assert "t:" in t.summary()
    with pytest.raises(RuntimeError):
        Timer().stop()
    assert format_time(12.34) == "12.3s"
    assert format_time(75) == "1:15"
    assert format_time(3725) == "1:02:05"


# --------------------------------------------------------------------------- #
# viz
# --------------------------------------------------------------------------- #
def test_confusion_matrix_hand_computed():
    y_true = np.array([0, 1, 2, 2, 0, 1, 1])
    y_pred = np.array([0, 1, 2, 0, 0, 2, 1])
    # By hand, rows = true:
    #   true 0 -> pred 0 twice
    #   true 1 -> pred 1 twice, pred 2 once
    #   true 2 -> pred 2 once, pred 0 once
    expected = np.array([[2, 0, 0], [0, 2, 1], [1, 0, 1]], dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, 3)
    assert np.array_equal(cm, expected)
    assert cm.sum() == y_true.size  # np.add.at accumulated; assignment would not
    assert cm.dtype == np.int64


def test_confusion_matrix_accumulates_repeats():
    """The np.add.at point: fancy-index assignment keeps only the last write."""
    y = np.zeros(50, dtype=np.int64)
    cm = confusion_matrix(y, y, 2)
    assert cm[0, 0] == 50


def test_confusion_matrix_accepts_logits_and_validates():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    assert np.array_equal(confusion_matrix(np.array([0, 1]), logits, 2), np.eye(2, dtype=np.int64))
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 1]), np.array([0]), 2)
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 5]), np.array([0, 0]), 2)


def _assert_written(path) -> None:
    assert os.path.exists(path), f"{path} was not created"
    assert os.path.getsize(path) > 0, f"{path} is empty"


def test_plot_weights_writes_a_file(tmp_path):
    rng = np.random.default_rng(0)
    W = rng.normal(size=(28 * 28, 12)).astype(np.float32)
    W[:, 3] = 0.0  # a dead filter: on a global scale it must stay flat
    out = plot_weights(W, shape=(28, 28), path=tmp_path / "w.png", max_units=12, upscale=16)
    _assert_written(out)


def test_plot_weights_accepts_the_transposed_layout_and_rejects_junk(tmp_path):
    W = np.random.default_rng(1).normal(size=(6, 28 * 28))
    _assert_written(plot_weights(W, shape=(28, 28), path=tmp_path / "wt.png"))
    with pytest.raises(ValueError):
        plot_weights(np.zeros((5, 5)), shape=(28, 28), path=tmp_path / "bad.png")


def test_plot_weights_survives_an_all_zero_matrix(tmp_path):
    # max|W| == 0 would make vmin == vmax; the code must not divide by it.
    out = plot_weights(np.zeros((28 * 28, 4)), path=tmp_path / "zero.png")
    _assert_written(out)


def test_plot_filters_conv_writes_a_file(tmp_path):
    W = np.random.default_rng(2).normal(size=(8, 3, 3, 3))  # (out, in, kH, kW)
    _assert_written(plot_filters_conv(W, path=tmp_path / "conv.png", max_filters=8))
    with pytest.raises(ValueError):
        plot_filters_conv(np.zeros((3, 3)), path=tmp_path / "nope.png")


def test_plot_history_writes_a_file(tmp_path):
    history = {
        "train_loss": [2.30, 1.10, 0.62, 0.41, 0.30],
        "val_loss": [2.31, 1.20, 0.71, 0.52, 0.44],
        "val_acc": [0.31, 0.72, 0.83, 0.88, 0.90],
    }
    _assert_written(plot_history(history, path=tmp_path / "h.png"))


def test_plot_history_handles_a_nonpositive_loss(tmp_path):
    # A log axis cannot render 0.0; the symlog fallback must keep the point.
    _assert_written(plot_history({"loss": [1.0, 0.5, 0.0]}, path=tmp_path / "h0.png"))
    with pytest.raises(ValueError):
        plot_history({"lr": [0.1, 0.01]}, path=tmp_path / "hx.png")


def test_plot_confusion_writes_a_file(tmp_path):
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, C, size=60)
    y_pred = np.where(rng.random(60) < 0.7, y_true, rng.integers(0, C, size=60))
    cm = confusion_matrix(y_true, y_pred, C)
    _assert_written(plot_confusion(cm, path=tmp_path / "cm.png"))
    _assert_written(
        plot_confusion(cm, path=tmp_path / "cmn.png", normalize=True, class_names=list("abc"))
    )
    with pytest.raises(ValueError):
        plot_confusion(np.zeros((2, 3)), path=tmp_path / "bad.png")


def test_plot_confusion_normalizes_an_empty_row_without_nan(tmp_path):
    cm = np.array([[3, 1, 0], [0, 0, 0], [0, 2, 4]], dtype=np.int64)  # class 1 absent
    _assert_written(plot_confusion(cm, path=tmp_path / "cme.png", normalize=True))


def test_plot_weights_rejects_an_empty_selection(tmp_path):
    # Unguarded, np.max on the empty slice says "zero-size array to reduction
    # operation maximum" -- naming neither the argument nor the function.
    with pytest.raises(ValueError, match="max_units"):
        plot_weights(np.zeros((28 * 28, 4)), path=tmp_path / "e0.png", max_units=0)
    with pytest.raises(ValueError, match="max_units"):
        plot_weights(np.zeros((28 * 28, 0)), path=tmp_path / "e1.png")


def test_plot_filters_conv_rejects_an_empty_selection(tmp_path):
    with pytest.raises(ValueError, match="max_filters"):
        plot_filters_conv(np.zeros((8, 3, 3, 3)), path=tmp_path / "c0.png", max_filters=0)
    with pytest.raises(ValueError, match="in_channels"):
        plot_filters_conv(np.zeros((8, 0, 3, 3)), path=tmp_path / "c1.png")


def test_degenerate_history_and_confusion_still_render(tmp_path):
    """Empty series, an all-zero matrix and a single class are legal inputs.

    An all-zero confusion matrix is exactly what an untrained model produces,
    so these must give a dull figure rather than a reduction error.
    """
    _assert_written(plot_history({"loss": [], "acc": []}, path=tmp_path / "d0.png"))
    zeros = np.zeros((3, 3), dtype=np.int64)
    _assert_written(plot_confusion(zeros, path=tmp_path / "d1.png"))
    _assert_written(plot_confusion(zeros, normalize=True, path=tmp_path / "d2.png"))
    _assert_written(plot_confusion(np.array([[7]]), path=tmp_path / "d3.png"))
    _assert_written(plot_confusion(np.array([[0]]), normalize=True, path=tmp_path / "d4.png"))
