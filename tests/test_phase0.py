"""Tests for the Phase 0 showpiece in ``phase0/``.

Phase 0 is deliberately not part of the ``scratchgrad`` package -- it imports
nothing from ``src/`` -- so these tests load it from its directory by path.

The tests fall into four groups:

* **gradients**: every hand-written backward against central differences, plus
  the step-size-free adjoint identity.
* **numerical traps**: He init, the softmax max-shift, the log-sum-exp loss, and
  Adam's ``eps`` placement, each checked against the specific wrong version.
* **pipeline**: IDX parsing, the shuffle coupling, the ``ln(C)`` loss identity,
  overfitting 32 examples, and the ``model.js`` export.
* **the browser demo**: the numerics in ``web/app.js`` are executed under
  ``node`` and compared against NumPy, so the JS the page ships is verified
  rather than eyeballed. Skipped cleanly when ``node`` is absent.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PHASE0 = os.path.join(_REPO, "phase0")
_WEB = os.path.join(_PHASE0, "web")
if _PHASE0 not in sys.path:
    sys.path.append(_PHASE0)

import export_web  # noqa: E402
import gradcheck as p0gradcheck  # noqa: E402
import mnist as p0mnist  # noqa: E402
import scratchnn  # noqa: E402
import train as p0train  # noqa: E402

NODE = shutil.which("node")


def _js_code_only(text: str) -> str:
    """Strip JS comments.

    Several checks below assert that an anti-pattern is *absent* from the code
    -- and the code documents those very anti-patterns by name in its comments.
    Without this the tests would fail on their own explanations.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


# --------------------------------------------------------------------------- #
# gradients
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "check", [p0gradcheck.check_dense, p0gradcheck.check_relu, p0gradcheck.check_cross_entropy]
)
def test_per_op_gradients(check):
    for name, err in check().items():
        assert err <= p0gradcheck.TOL_OP, f"{name}: {err:.3e}"


def test_composed_gradient():
    """Dense -> ReLU -> Dense -> softmax-CE, end to end.

    Catches a per-layer backward that is individually right but wired into the
    stack in the wrong order, which no per-op check can see.
    """
    for name, err in p0gradcheck.check_composed().items():
        assert err <= p0gradcheck.TOL_COMPOSED, f"{name}: {err:.3e}"


def test_adjoint_identity():
    """``<v, Au> == <A^T v, u>``: no step size, so no truncation error at all."""
    for name, err in p0gradcheck.check_adjoint().items():
        assert err <= p0gradcheck.TOL_ADJOINT, f"{name}: {err:.3e}"


def test_run_all_reports_pass():
    _, ok = p0gradcheck.run_all(verbose=False)
    assert ok


def test_gradcheck_refuses_float32():
    """The check has no power in float32 and must say so rather than pass."""
    x = np.ones((3, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        p0gradcheck.numeric_vjp(lambda: x, [x], np.ones((3, 3)))


def test_rel_error_is_array_scaled():
    """One flipped small entry must dominate, and noise on a big array must not."""
    a = np.array([3.0, 1e-4])
    b = np.array([3.0, -1e-4])
    assert p0gradcheck.rel_error(a, b) == pytest.approx(2e-4 / 6.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# numerical traps
# --------------------------------------------------------------------------- #
def test_he_init_is_a_variance_not_a_std():
    """std = sqrt(2/fan_in). Reading ``N(0, 2/n)`` as a std is 55x too small at 784."""
    rng = np.random.default_rng(0)
    W = scratchnn.he_normal(784, 4096, rng)
    assert W.std() == pytest.approx(math.sqrt(2.0 / 784), rel=0.02)
    assert not np.isclose(W.std(), 2.0 / 784, rtol=0.5)


def test_softmax_survives_huge_logits():
    """Without the row-max shift, exp(1000) is inf and the loss is nan forever."""
    loss = scratchnn.SoftmaxCrossEntropy()
    logits = np.array([[1000.0, 1001.0, 999.0], [-900.0, -901.0, -899.0]])
    value = loss.forward(logits, np.array([1, 2]))
    assert np.isfinite(value)
    assert np.all(np.isfinite(loss.p))
    assert loss.p.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_loss_never_clips_the_most_wrong_sample():
    """``-log(p + 1e-12)`` caps the loss near 27.6 and zeroes its gradient.

    The log-sum-exp form reports the true 50 and still hands back a full-size
    gradient, which is the only reason a badly wrong sample can be corrected.
    """
    loss = scratchnn.SoftmaxCrossEntropy()
    value = loss.forward(np.array([[0.0, 50.0]]), np.array([0]))
    assert value == pytest.approx(50.0, abs=1e-9)
    assert value > 27.7, "loss is capped -- the epsilon fudge is present"
    g = loss.backward()
    assert abs(g[0, 0]) == pytest.approx(1.0, abs=1e-9)


def test_adam_eps_is_outside_the_sqrt_and_after_bias_correction():
    """``m/(sqrt(v)+eps)``, not ``g/sqrt(v+eps)``.

    With a small gradient the two differ by two orders of magnitude: the wrong
    form's floor engages at ``|g| ~ 1e-4`` instead of ``1e-8``, silently damping
    the whole late phase of training.
    """
    lr, eps = 0.1, 1e-8
    g_value = 1e-6
    p = np.array([1.0])
    g = np.array([g_value])
    scratchnn.Adam([(p, g)], lr=lr, eps=eps).step()

    m_hat, v_hat = g_value, g_value**2
    expected = 1.0 - lr * m_hat / (math.sqrt(v_hat) + eps)
    assert p[0] == pytest.approx(expected, abs=1e-15)

    wrong = 1.0 - lr * m_hat / math.sqrt(v_hat + eps)
    assert abs(p[0] - wrong) > 1e-3, "eps placement is indistinguishable -- check the test"


def test_adam_bias_correction_takes_a_full_first_step():
    """Step 1 must move by about ``lr``; without bias correction it moves by ~lr/10."""
    p = np.array([0.0])
    g = np.array([1.0])
    scratchnn.Adam([(p, g)], lr=0.1).step()
    assert abs(p[0]) == pytest.approx(0.1, rel=1e-6)


def test_gradient_buffers_are_written_in_place():
    """Rebinding ``self.dW`` would leave the optimizer holding a stale array."""
    rng = np.random.default_rng(0)
    layer = scratchnn.Dense(11, 5, rng)
    before = layer.dW
    layer.forward(rng.standard_normal((7, 11)))
    layer.backward(rng.standard_normal((7, 5)))
    assert layer.dW is before
    assert np.any(layer.dW != 0.0)


def test_parameter_count_is_101770():
    model = scratchnn.MLP([784, 128, 10], np.random.default_rng(0))
    assert model.num_parameters() == 784 * 128 + 128 + 128 * 10 + 10 == 101_770


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def _idx_bytes(arr: np.ndarray) -> bytes:
    """Encode a uint8 array as an IDX blob, big-endian header."""
    header = bytes([0, 0, 0x08, arr.ndim]) + struct.pack(f">{arr.ndim}I", *arr.shape)
    return header + arr.astype(np.uint8).tobytes()


def test_parse_idx_round_trip():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(3, 28, 28), dtype=np.uint8)
    out = p0mnist.parse_idx(_idx_bytes(arr))
    assert out.shape == (3, 28, 28)
    np.testing.assert_array_equal(out, arr)


def test_parse_idx_returns_a_writable_copy():
    """frombuffer hands back a read-only view that pins the whole buffer alive."""
    arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = p0mnist.parse_idx(_idx_bytes(arr))
    assert out.flags["WRITEABLE"] and out.flags["OWNDATA"]
    out[0, 0] = 99  # must not raise


def test_parse_idx_header_is_big_endian():
    """A little-endian read of 60000 is 1,089,470,464 and fails far from the cause."""
    blob = _idx_bytes(np.zeros((260, 2), dtype=np.uint8))
    assert struct.unpack(">2I", blob[4:12]) == (260, 2)
    assert p0mnist.parse_idx(blob).shape == (260, 2)


@pytest.mark.parametrize(
    "blob, match",
    [
        (b"\x01\x00\x08\x01\x00\x00\x00\x01\x00", "magic"),
        (b"\x00\x00\xff\x01\x00\x00\x00\x01\x00", "type code"),
        (b"\x00\x00\x08\x01\x00\x00\x00\x09\x00", "payload"),
    ],
)
def test_parse_idx_rejects_garbage(blob, match):
    with pytest.raises(ValueError, match=match):
        p0mnist.parse_idx(blob)


def test_mnist_and_fashion_do_not_share_a_cache():
    """Same four filenames; only the md5s separate them."""
    assert set(p0mnist.FILES) == {
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    }
    assert len(set(p0mnist.FILES.values())) == 4


def test_gzip_round_trip_through_the_real_reader(tmp_path):
    """The on-disk path: gzip -> parse_idx -> array."""
    arr = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    path = tmp_path / "train-images-idx3-ubyte.gz"
    with gzip.open(path, "wb") as fh:
        fh.write(_idx_bytes(arr))
    with gzip.open(path, "rb") as fh:
        np.testing.assert_array_equal(p0mnist.parse_idx(fh.read()), arr)


def test_synthetic_data_is_normalised_by_255_only():
    """x/255 and nothing else: the browser has to reproduce it byte for byte."""
    x, y, _, _ = p0mnist.synthetic_mnist(np.random.default_rng(0), n_train=64, n_test=8)
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert not np.isclose(x.mean(), 0.0, atol=1e-3), "data looks mean-centred"
    # Every value must be a whole number of 1/255 steps.
    np.testing.assert_allclose(x * 255.0, np.round(x * 255.0), atol=1e-9)
    assert y.dtype == np.int64


def test_loss_at_init_is_exactly_ln_c():
    """Zero the final layer -> uniform logits -> ln(10), to 1e-12 in float64."""
    rng = np.random.default_rng(0)
    model = scratchnn.MLP([784, 128, 10], rng)
    x = rng.random((16, 784))
    y = rng.integers(0, 10, size=16)
    loss = p0train.check_loss_at_init(model, x, y, 10)
    assert abs(loss - math.log(10)) < 1e-12


def test_loss_at_init_restores_the_final_layer():
    rng = np.random.default_rng(0)
    model = scratchnn.MLP([11, 7, 5], rng)
    saved = model.dense_layers()[-1].W.copy()
    p0train.check_loss_at_init(model, rng.random((4, 11)), rng.integers(0, 5, 4), 5)
    np.testing.assert_array_equal(model.dense_layers()[-1].W, saved)


def test_loss_at_init_detects_non_uniform_logits():
    """The assertion has to actually fire, or it is decoration.

    Note the limit of this check, since it is easy to overclaim: a *double*
    softmax is idempotent on uniform logits, so zeroing the head hides it. The
    double softmax shows up instead in the un-zeroed initial loss, pinned at
    1.4615 for C=10 -- a systematic deficit below ln(10).
    """

    class LeakyHead(scratchnn.MLP):
        def forward(self, x):
            return super().forward(x) + np.linspace(-1.0, 1.0, 10)

    rng = np.random.default_rng(0)
    model = LeakyHead([784, 32, 10], rng)
    with pytest.raises(AssertionError, match="cross-entropy is wrong"):
        p0train.check_loss_at_init(model, rng.random((8, 784)), rng.integers(0, 10, 8), 10)


def test_shuffle_uses_one_permutation():
    """Two separate shuffles decouple X from y; encode the label in the row."""
    n = 257
    x = np.arange(n, dtype=np.float64).reshape(n, 1)
    y = np.arange(n)
    seen = 0
    for xb, yb in p0train.epoch_batches(x, y, 32, np.random.default_rng(0)):
        np.testing.assert_array_equal(xb[:, 0].astype(np.int64), yb)
        seen += xb.shape[0]
    assert seen == n


def test_shuffle_actually_shuffles():
    x = np.arange(64, dtype=np.float64).reshape(64, 1)
    y = np.arange(64)
    first = next(iter(p0train.epoch_batches(x, y, 64, np.random.default_rng(0))))[1]
    assert not np.array_equal(first, y)


def test_smoke_overfit_reaches_zero_loss():
    """32 examples must be memorisable. If not, the model is broken, not undertrained."""
    rng = np.random.default_rng(0)
    x, y, _, _ = p0mnist.synthetic_mnist(rng, n_train=64, n_test=8)
    loss = p0train.smoke_overfit(x, y, 10, np.random.default_rng(1), steps=400, verbose=False)
    assert loss < 1e-3, f"plateaued at {loss:.3e}"


def test_short_training_run_learns(tmp_path):
    """The whole loop, offline, on fabricated data. Not an accuracy claim."""
    rng = np.random.default_rng(0)
    x_tr, y_tr, x_te, y_te = p0mnist.synthetic_mnist(rng, n_train=1500, n_test=300)
    model, hist = p0train.train(x_tr, y_tr, x_te, y_te, rng, epochs=2, verbose=False)
    assert hist["test_acc"][-1] > 0.8
    assert hist["test_loss"][-1] < hist["test_loss"][0]
    path = p0train.save_weights(model, str(tmp_path / "w.npz"), hist["test_acc"][-1], model.sizes)
    with np.load(path) as z:
        assert z["W1"].shape == (784, 128) and z["W2"].shape == (128, 10)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Export untrained He weights to a private ``model.js``; no training needed."""
    out = tmp_path_factory.mktemp("export") / "model.js"
    path, d_logit, d_prob, ok = export_web.export(
        weights="", out_js=str(out), rng=np.random.default_rng(0), synthetic=True, verbose=False
    )
    return {"path": path, "d_logit": d_logit, "d_prob": d_prob, "ok": ok}


def test_export_consistency_under_1e5(exported):
    assert exported["ok"]
    assert exported["d_logit"] < 1e-5
    assert exported["d_prob"] < 1e-5


def test_model_js_is_script_tag_loadable(exported):
    """No fetch(): a file:// document has an opaque origin and CORS refuses it."""
    source = Path(exported["path"]).read_text(encoding="utf-8")
    assert source.lstrip().startswith("//")
    assert "const MODEL = {" in source
    code = _js_code_only(source)
    assert "fetch(" not in code
    assert "export " not in code and "import " not in code


def test_model_js_round_trips_float32_exactly(exported):
    """9 significant digits is FLT_DECIMAL_DIG; through a Float32Array it is exact."""
    rng = np.random.default_rng(1)
    W1, b1, W2, b2 = export_web._synthetic_weights(rng, (784, 128, 10))
    W1 = np.asarray(W1, dtype=np.float32).astype(np.float64)
    parsed = export_web.parse_model_js(export_web.render_model_js(W1, b1, W2, b2, 0.98))
    np.testing.assert_array_equal(parsed["W1"], W1.reshape(-1))


def test_model_js_is_row_major_in_the_layout_app_js_indexes(exported):
    """``W1[i * nHidden + j]``. A transposed export is the bug this catches."""
    rng = np.random.default_rng(2)
    W1, b1, W2, b2 = export_web._synthetic_weights(rng, (7, 5, 3))
    parsed = export_web.parse_model_js(export_web.render_model_js(W1, b1, W2, b2, 0.0))
    flat = parsed["W1"]
    W1_32 = np.asarray(W1, dtype=np.float32).astype(np.float64)
    for i in range(7):
        for j in range(5):
            assert flat[i * 5 + j] == W1_32[i, j]


def test_export_rejects_mismatched_shapes(tmp_path):
    npz = tmp_path / "bad.npz"
    np.savez(npz, W1=np.zeros((4, 3)), b1=np.zeros(5), W2=np.zeros((3, 2)), b2=np.zeros(2))
    with pytest.raises(ValueError, match="inconsistent shapes"):
        export_web.export(str(npz), str(tmp_path / "m.js"), np.random.default_rng(0), verbose=False)


def test_inline_html_is_a_single_self_contained_file(tmp_path, exported):
    index = Path(os.path.join(_WEB, "index.html")).read_text(encoding="utf-8")
    app = Path(os.path.join(_WEB, "app.js")).read_text(encoding="utf-8")
    model = Path(exported["path"]).read_text(encoding="utf-8")
    html = export_web.inline_html(index, model, app)
    assert 'src="model.js"' not in html and 'src="app.js"' not in html
    assert "const MODEL = {" in html and "function mnistify" in html
    assert "<script src=" not in html


# --------------------------------------------------------------------------- #
# the shipped page
# --------------------------------------------------------------------------- #
def test_web_files_exist():
    for name in ("index.html", "app.js"):
        assert os.path.exists(os.path.join(_WEB, name))


def test_page_loads_nothing_external():
    """It has to run from file://: no CDN, no Google Fonts, no absolute URLs."""
    for name in ("index.html", "app.js"):
        text = Path(os.path.join(_WEB, name)).read_text(encoding="utf-8")
        assert 'src="http' not in text
        assert 'href="http' not in text
        assert "@import" not in text
        assert "cdn." not in text


def test_canvas_coordinates_use_the_bounding_rect():
    """``e.clientX - canvas.offsetLeft`` breaks under scroll and CSS scaling."""
    app = Path(os.path.join(_WEB, "app.js")).read_text(encoding="utf-8")
    assert "getBoundingClientRect()" in app
    assert "pad.width / rect.width" in app
    assert "offsetLeft" not in _js_code_only(app)


def test_page_uses_pointer_events_not_mouse_events():
    """One code path for mouse, touch and stylus."""
    app = Path(os.path.join(_WEB, "app.js")).read_text(encoding="utf-8")
    for name in ("pointerdown", "pointermove", "pointerup", "pointercancel", "setPointerCapture"):
        assert name in app
    assert "mousedown" not in _js_code_only(app)
    assert "touch-action: none" in Path(os.path.join(_WEB, "index.html")).read_text(
        encoding="utf-8"
    )


def test_mnistify_follows_the_documented_recipe():
    """Threshold 25, long side 20, centre of mass, clamped paste, /255."""
    app = Path(os.path.join(_WEB, "app.js")).read_text(encoding="utf-8")
    assert "inkBounds(src, w, h, 25)" in app
    assert "20 / Math.max(bw, bh)" in app
    assert re.search(r"clamp\(Math\.round\(14 - cx\), 0, 28 - nw\)", app)
    assert re.search(r"clamp\(Math\.round\(14 - cy\), 0, 28 - nh\)", app)
    assert "/ 255" in app


def test_every_element_app_js_reaches_for_exists_in_the_page():
    """A typo'd id is `null` at load and the whole script dies on the first access.

    This is the closest a non-browser test gets to "the page boots"; the numerics
    below are executed for real under node, but the DOM wiring is only checked
    here, statically.
    """
    app = Path(os.path.join(_WEB, "app.js")).read_text(encoding="utf-8")
    html = Path(os.path.join(_WEB, "index.html")).read_text(encoding="utf-8")
    wanted = set(re.findall(r"getElementById\('([^']+)'\)", app))
    present = set(re.findall(r'\bid="([^"]+)"', html))
    assert wanted, "no getElementById calls found -- the regex is stale"
    assert wanted <= present, f"app.js reaches for missing ids: {sorted(wanted - present)}"


def test_page_draws_white_on_black():
    """MNIST polarity. Black-on-white fails silently with confident wrong answers."""
    app = Path(os.path.join(_WEB, "app.js")).read_text(encoding="utf-8")
    assert "ctx.fillStyle = '#000000'" in app
    assert "ctx.strokeStyle = '#ffffff'" in app
    assert "assertPolarity" in app


# --------------------------------------------------------------------------- #
# node parity: run the shipped JavaScript and compare it to NumPy
# --------------------------------------------------------------------------- #
_HARNESS = r"""
const fs = require('fs');
const P = require(process.argv[2]);
const MODEL = eval(fs.readFileSync(process.argv[3], 'utf8') + '\nMODEL');
const job = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));

const out = {};
out.probs = job.inputs.map(function (x) {
  return Array.from(P.forward(Float32Array.from(x), MODEL).probs);
});
out.logits = job.inputs.map(function (x) {
  return Array.from(P.forward(Float32Array.from(x), MODEL).logits);
});
out.softmax = Array.from(P.softmax(Float64Array.from(job.logits)));
const rgba = Uint8ClampedArray.from(job.rgba);
out.bounds = P.inkBounds(rgba, job.w, job.h, 25);
const small = Uint8ClampedArray.from(job.small);
out.com = P.centreOfMass(small, job.sw, job.sh);
out.offsets = P.pasteOffsets(out.com.cx, out.com.cy, job.sw, job.sh);
const tensor = P.toTensor(small, job.sw, job.sh, out.offsets.dx, out.offsets.dy);
out.tensor = Array.from(tensor);
out.mean = P.meanOf(tensor);
out.polarity = [P.assertPolarity(0.13), P.assertPolarity(0.8)];
process.stdout.write(JSON.stringify(out));
"""


def _rgba(gray: np.ndarray) -> list[int]:
    """Widen an (H, W) uint8 intensity map into a flat RGBA buffer."""
    h, w = gray.shape
    buf = np.zeros((h, w, 4), dtype=np.uint8)
    buf[..., 0] = buf[..., 1] = buf[..., 2] = gray
    buf[..., 3] = 255
    return buf.reshape(-1).tolist()


@pytest.fixture(scope="module")
def node_result(tmp_path_factory, exported):
    if NODE is None:
        pytest.skip("node not installed; the browser numerics are not exercised")
    work = tmp_path_factory.mktemp("node")
    harness = work / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")

    rng = np.random.default_rng(7)
    inputs = [
        np.where(rng.random(784) < 0.18, rng.random(784), 0.0),
        np.zeros(784),
        rng.random(784),
    ]
    canvas = np.zeros((40, 60), dtype=np.uint8)
    canvas[8:30, 15:35] = 200  # a block of ink
    canvas[0:3, 0:3] = 20  # below the threshold of 25: must be ignored
    small = (rng.random((9, 13)) * 255).astype(np.uint8)

    job = {
        "inputs": [list(map(float, x)) for x in inputs],
        "logits": [1000.0, 1001.0, 999.0],
        "rgba": _rgba(canvas),
        "w": 60,
        "h": 40,
        "small": _rgba(small),
        "sw": 13,
        "sh": 9,
    }
    job_path = work / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")

    proc = subprocess.run(
        [NODE, str(harness), os.path.join(_WEB, "app.js"), exported["path"], str(job_path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    return {
        "out": json.loads(proc.stdout),
        "inputs": inputs,
        "canvas": canvas,
        "small": small,
        "model_js": exported["path"],
    }


def test_js_forward_matches_numpy(node_result):
    """The forward pass the page actually ships, against NumPy on the same weights."""
    with open(node_result["model_js"], encoding="utf-8") as fh:
        model = export_web.parse_model_js(fh.read())
    W1 = model["W1"].reshape(model["nIn"], model["nHidden"])
    W2 = model["W2"].reshape(model["nHidden"], model["nOut"])
    for x, js_probs, js_logits in zip(
        node_result["inputs"],
        node_result["out"]["probs"],
        node_result["out"]["logits"],
        strict=True,
    ):
        # float32 input: the browser holds the tensor in a Float32Array.
        x32 = np.asarray(x, dtype=np.float32).astype(np.float64)
        logits, probs = export_web.numpy_forward(x32, W1, model["b1"], W2, model["b2"])
        assert np.max(np.abs(logits - np.asarray(js_logits))) < 1e-5
        assert np.max(np.abs(probs - np.asarray(js_probs))) < 1e-5


def test_js_softmax_survives_huge_logits(node_result):
    """No max-shift and exp(1000) is Infinity, so every bar renders NaN."""
    p = np.asarray(node_result["out"]["softmax"])
    assert np.all(np.isfinite(p))
    assert p.sum() == pytest.approx(1.0)
    z = np.array([1000.0, 1001.0, 999.0])
    e = np.exp(z - z.max())
    np.testing.assert_allclose(p, e / e.sum(), atol=1e-12)


def test_js_ink_bounds_ignores_the_sub_threshold_fringe(node_result):
    b = node_result["out"]["bounds"]
    assert (b["minX"], b["maxX"], b["minY"], b["maxY"]) == (15, 34, 8, 29)


def test_js_centre_of_mass_matches_numpy(node_result):
    small = node_result["small"].astype(np.float64)
    ys, xs = np.mgrid[0 : small.shape[0], 0 : small.shape[1]]
    total = small.sum()
    com = node_result["out"]["com"]
    assert com["cx"] == pytest.approx((xs * small).sum() / total, rel=1e-12)
    assert com["cy"] == pytest.approx((ys * small).sum() / total, rel=1e-12)


def test_js_paste_offsets_centre_the_mass_and_stay_in_bounds(node_result):
    com = node_result["out"]["com"]
    off = node_result["out"]["offsets"]
    sw, sh = node_result["small"].shape[1], node_result["small"].shape[0]
    assert off["dx"] == min(max(round(14 - com["cx"]), 0), 28 - sw)
    assert off["dy"] == min(max(round(14 - com["cy"]), 0), 28 - sh)
    assert 0 <= off["dx"] <= 28 - sw and 0 <= off["dy"] <= 28 - sh


def test_js_tensor_is_the_pasted_patch_over_255(node_result):
    """The exact 784-vector the net is fed, rebuilt in NumPy."""
    out = node_result["out"]
    sw, sh = node_result["small"].shape[1], node_result["small"].shape[0]
    expect = np.zeros((28, 28))
    dy, dx = out["offsets"]["dy"], out["offsets"]["dx"]
    expect[dy : dy + sh, dx : dx + sw] = node_result["small"] / 255.0
    got = np.asarray(out["tensor"]).reshape(28, 28)
    np.testing.assert_allclose(got, expect, atol=1e-6)
    assert out["mean"] == pytest.approx(expect.mean(), rel=1e-6)


def test_js_polarity_assertion_fires_only_when_inverted(node_result):
    ok_msg, inverted_msg = node_result["out"]["polarity"]
    assert ok_msg == ""
    assert "inverted" in inverted_msg


# --------------------------------------------------------------------------- #
# the standing constraint
# --------------------------------------------------------------------------- #
def test_phase0_imports_no_deep_learning_library_and_no_scratchgrad():
    """Phase 0 stands alone: numpy only, and not even the package next door."""
    banned = re.compile(
        r"^\s*(?:import|from)\s+(torch|tensorflow|jax|autograd|scratchgrad)\b", re.M
    )
    for name in os.listdir(_PHASE0):
        if not name.endswith(".py"):
            continue
        text = Path(os.path.join(_PHASE0, name)).read_text(encoding="utf-8")
        assert not banned.search(text), f"{name} imports a forbidden module"
