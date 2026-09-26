"""The "Reproducible" property: a fixed seed gives bit-identical results.

`ROADMAP.md` grades this project on it:

    *A fixed seed gives bit-identical losses. Every number in the README is
    regenerable by one command.*

"Bit-identical" is the load-bearing word. Approximately-equal is what you get
anyway from a deterministic library; it would pass even if the shuffle order
drifted or an uninitialised buffer leaked in. Exact equality is what actually
catches those.

The one thing deliberately *not* asserted is cross-machine bit-identity. BLAS
picks different reduction orders for different SIMD widths and thread counts, so
the same seed on a different CPU can differ in the last few bits of a float32
accumulation. Claiming otherwise would be a claim this project cannot honour.
"""

from __future__ import annotations

import numpy as np
import pytest

import scratchgrad as sg
from scratchgrad import nn, optim
from scratchgrad.data import DataLoader, TensorDataset
from scratchgrad.nn import functional as F
from scratchgrad.utils import seed_everything, snapshot

D, C = 11, 5


def train_a_little(seed: int, steps: int = 25) -> tuple[list[float], dict[str, np.ndarray]]:
    """A complete miniature training run: init, shuffle, forward, backward, step."""
    rng = seed_everything(seed)
    x = rng.standard_normal((64, D)).astype(np.float32)
    y = rng.integers(0, C, size=64)
    model = nn.Sequential(
        nn.Linear(D, 16, rng=rng), nn.ReLU(), nn.Dropout(0.2, rng=rng), nn.Linear(16, C, rng=rng)
    )
    opt = optim.Adam(model.parameters(), lr=1e-2)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True, rng=rng)
    losses: list[float] = []
    for _ in range(steps // 4 + 1):
        for xb, yb in loader:
            loss = F.cross_entropy(model(sg.Tensor(xb)), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return losses[:steps], snapshot(model)


def test_same_seed_gives_bit_identical_losses():
    a, _ = train_a_little(1234)
    b, _ = train_a_little(1234)
    assert a == b, "loss sequences differ despite the same seed"


def test_same_seed_gives_bit_identical_weights():
    _, wa = train_a_little(1234)
    _, wb = train_a_little(1234)
    for k in wa:
        np.testing.assert_array_equal(wa[k], wb[k], err_msg=f"parameter {k} differs")


def test_different_seeds_actually_differ():
    """Guard the guard: a run that ignores its seed would pass every test above."""
    a, _ = train_a_little(1)
    b, _ = train_a_little(2)
    assert a != b


def test_initialization_is_seed_determined():
    def build(seed):
        return nn.Linear(D, C, rng=np.random.default_rng(seed)).weight.data.copy()

    np.testing.assert_array_equal(build(7), build(7))
    assert not np.array_equal(build(7), build(8))


def test_dataloader_shuffle_is_seed_determined_and_keeps_pairs_aligned():
    """Two separate shuffle calls decouple X from y. The failure is quiet: the
    loss then sits at exactly ln(C) forever and overfit-one-batch still
    'succeeds', so it looks like a modelling problem rather than a data bug."""
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = (x[:, 0] * 2).astype(np.int64)  # an exact relation that must survive
    ds = TensorDataset(x, y)

    def order(seed):
        out = []
        for xb, yb in DataLoader(ds, batch_size=4, shuffle=True, rng=np.random.default_rng(seed)):
            np.testing.assert_array_equal(yb, (xb[:, 0] * 2).astype(np.int64))
            out.append(xb[:, 0].copy())
        return np.concatenate(out)

    np.testing.assert_array_equal(order(3), order(3))
    assert not np.array_equal(order(3), order(4))


def test_dropout_mask_is_seed_determined():
    x = sg.Tensor(np.ones((50, 50)))
    a = F.dropout(x, 0.5, True, rng=np.random.default_rng(11)).data
    b = F.dropout(x, 0.5, True, rng=np.random.default_rng(11)).data
    c = F.dropout(x, 0.5, True, rng=np.random.default_rng(12)).data
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_no_global_numpy_state_leaks_into_results():
    """A layer that reached for ``np.random`` instead of its ``rng=`` argument
    would still look deterministic in a test that seeds the global RNG. Perturb
    the global state between runs so that shortcut cannot hide."""
    np.random.seed(0)
    a, _ = train_a_little(99)
    np.random.seed(12345)
    _ = np.random.random(1000)
    b, _ = train_a_little(99)
    assert a == b, "results moved when only the GLOBAL numpy RNG changed"


def test_evaluation_is_deterministic_and_dropout_free():
    rng = seed_everything(5)
    model = nn.Sequential(
        nn.Linear(D, 8, rng=rng), nn.Dropout(0.5, rng=rng), nn.Linear(8, C, rng=rng)
    )
    x = sg.Tensor(rng.standard_normal((16, D)).astype(np.float32))
    model.eval()
    with sg.no_grad():
        first = model(x).data.copy()
        for _ in range(5):
            np.testing.assert_array_equal(model(x).data, first)
    model.train()
    with sg.no_grad():
        assert not np.array_equal(model(x).data, first), "train mode should resample the mask"


def test_snapshot_survives_a_later_optimizer_step():
    """``state_dict()`` shares storage with the live parameters, so a snapshot
    taken for early stopping silently follows the model unless it is copied."""
    rng = seed_everything(0)
    model = nn.Linear(D, C, rng=rng)
    opt = optim.Adam(model.parameters(), lr=0.1)
    shallow = model.state_dict()
    deep = snapshot(model)
    F.cross_entropy(
        model(sg.Tensor(rng.standard_normal((8, D)).astype(np.float32))), rng.integers(0, C, size=8)
    ).backward()
    opt.step()
    assert np.array_equal(shallow["weight"], model.weight.data)
    assert not np.array_equal(deep["weight"], model.weight.data)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_forward_is_deterministic_in_both_dtypes(dtype):
    with sg.default_dtype(dtype):
        rng = np.random.default_rng(0)
        model = nn.Sequential(nn.Linear(D, 9, rng=rng), nn.GELU(), nn.Linear(9, C, rng=rng))
        x = sg.Tensor(rng.standard_normal((13, D)), dtype=dtype)
        with sg.no_grad():
            a = model(x).data.copy()
            b = model(x).data.copy()
    np.testing.assert_array_equal(a, b)
    assert a.dtype == dtype
