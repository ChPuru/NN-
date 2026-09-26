/*
 * Phase 0 draw-a-digit demo: preprocessing, forward pass, UI.
 * Vanilla JS, no dependencies, runs from file://.
 *
 * The file is in two halves. The top half is pure numerics -- no DOM, no
 * globals -- which is what lets `tests/test_phase0.py` run these exact
 * functions under node and compare them against NumPy. The bottom half is
 * canvas and UI, and only runs when there is a document to run it in.
 *
 * The interesting part is mnistify(). Everything else is plumbing.
 */
'use strict';

var PHASE0 = (function () {

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /*
   * Bounding box of the ink in an RGBA buffer.
   *
   * Intensity is the red channel: the stroke is pure white on pure black, so
   * R == G == B and one channel is enough. Threshold 25 out of 255 keeps the
   * anti-aliased fringe of a stroke from inflating the box.
   */
  function inkBounds(rgba, w, h, threshold) {
    var minX = w, minY = h, maxX = -1, maxY = -1;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        if (rgba[(y * w + x) * 4] > threshold) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    return maxX < 0 ? null : { minX: minX, minY: minY, maxX: maxX, maxY: maxY };
  }

  /*
   * Intensity-weighted centre of mass.
   *
   * Not the centre of the bounding box: the ink in a 7 or a 4 sits well off the
   * geometric middle, and MNIST centred by centre of mass. Using the box centre
   * shifts every asymmetric digit by a pixel or two, which is exactly the kind
   * of distribution shift this model has no invariance to absorb.
   */
  function centreOfMass(rgba, w, h) {
    var sum = 0, sx = 0, sy = 0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var I = rgba[(y * w + x) * 4];
        sum += I; sx += x * I; sy += y * I;
      }
    }
    if (sum === 0) return null;
    return { cx: sx / sum, cy: sy / sum, sum: sum };
  }

  /* Paste offsets that put the centre of mass on (14, 14), clamped so the
     digit cannot be pushed off the edge of the 28x28 field. */
  function pasteOffsets(cx, cy, nw, nh) {
    return {
      dx: clamp(Math.round(14 - cx), 0, 28 - nw),
      dy: clamp(Math.round(14 - cy), 0, 28 - nh)
    };
  }

  /* Copy the scaled ink into a 784-vector, divided by 255 and nothing else --
     no mean centring. mnist.py normalises exactly this way, and a preprocessing
     mismatch between training and inference is invisible right up until the
     accuracy is quietly wrong. */
  function toTensor(rgba, nw, nh, dx, dy) {
    var out = new Float32Array(784);
    for (var y = 0; y < nh; y++) {
      for (var x = 0; x < nw; x++) {
        out[(dy + y) * 28 + (dx + x)] = rgba[(y * nw + x) * 4] / 255;
      }
    }
    return out;
  }

  function meanOf(vec) {
    var s = 0;
    for (var i = 0; i < vec.length; i++) s += vec[i];
    return s / vec.length;
  }

  /*
   * Polarity assertion on the tensor about to be fed in.
   *
   * MNIST is white-on-black and averages about 0.13. A vector averaging 0.8 is
   * an inverted image, which the network will classify confidently and wrongly
   * with no error raised anywhere -- an inverted digit is a perfectly valid
   * input that the model has simply never seen. Cheap to check, so check it.
   */
  function assertPolarity(mean) {
    if (mean > 0.5) {
      return 'polarity looks inverted (mean ' + mean.toFixed(3) +
             '): MNIST is white ink on a black ground.';
    }
    if (mean > 0.4) {
      return 'unusually heavy input (mean ' + mean.toFixed(3) + '); try a thinner stroke.';
    }
    return '';
  }

  /* Subtract the max before exp. exp(800) is Infinity and every probability
     downstream becomes NaN; subtracting a constant leaves the softmax exactly
     unchanged, so there is no reason not to. */
  function softmax(z) {
    var m = -Infinity, i;
    for (i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
    var s = 0;
    var p = new Float64Array(z.length);
    for (i = 0; i < z.length; i++) { p[i] = Math.exp(z[i] - m); s += p[i]; }
    for (i = 0; i < z.length; i++) p[i] /= s;
    return p;
  }

  /*
   * 784 -> 128 ReLU -> 10 -> softmax, over the flat row-major weights.
   *
   * W1 is (nIn, nHidden) flattened, so W1[i * nHidden + j]. The loop is
   * input-major rather than the textbook output-major: the weight reads are
   * then sequential in memory, and a zero pixel -- about 80% of them after
   * mnistify -- is skipped exactly rather than multiplied by zero.
   * phase0/export_web.py runs this same loop nest in NumPy and asserts the two
   * agree, which is what catches a transposed export.
   */
  function forward(x, model) {
    var nIn = model.nIn, nH = model.nHidden, nOut = model.nOut;
    var W1 = model.W1, b1 = model.b1, W2 = model.W2, b2 = model.b2;
    var i, j, k, off;

    var h = new Float64Array(nH);
    for (j = 0; j < nH; j++) h[j] = b1[j];
    for (i = 0; i < nIn; i++) {
      var xi = x[i];
      if (xi === 0) continue;
      off = i * nH;
      for (j = 0; j < nH; j++) h[j] += xi * W1[off + j];
    }
    for (j = 0; j < nH; j++) if (h[j] < 0) h[j] = 0;   // ReLU

    var logits = new Float64Array(nOut);
    for (k = 0; k < nOut; k++) logits[k] = b2[k];
    for (j = 0; j < nH; j++) {
      var hj = h[j];
      if (hj === 0) continue;
      off = j * nOut;
      for (k = 0; k < nOut; k++) logits[k] += hj * W2[off + k];
    }
    return { logits: logits, probs: softmax(logits) };
  }

  return {
    clamp: clamp,
    inkBounds: inkBounds,
    centreOfMass: centreOfMass,
    pasteOffsets: pasteOffsets,
    toTensor: toTensor,
    meanOf: meanOf,
    assertPolarity: assertPolarity,
    softmax: softmax,
    forward: forward
  };
})();

// Node imports the numerics above for the parity test in tests/test_phase0.py.
// A browser never sees this branch.
if (typeof module !== 'undefined' && module.exports) { module.exports = PHASE0; }

// --------------------------------------------------------------------------
// Everything below needs a DOM.
// --------------------------------------------------------------------------
if (typeof document !== 'undefined') (function () {

  var pad = document.getElementById('pad');
  var preview = document.getElementById('preview');
  var barsHost = document.getElementById('bars');
  var alertLine = document.getElementById('alert');
  var stepsLine = document.getElementById('steps');
  var meanLine = document.getElementById('meanline');
  var verdictLine = document.getElementById('verdict');
  var metaLine = document.getElementById('meta');

  var ctx = pad.getContext('2d', { willReadFrequently: true });
  var previewCtx = preview.getContext('2d');
  var READY = typeof MODEL !== 'undefined' && MODEL && MODEL.W1;

  // MNIST is WHITE ON BLACK. Drawing black-on-white does not error: it produces
  // confident, consistent, wrong answers. Black ground, white ink -- and
  // assertPolarity() re-checks the tensor right before it is fed in.
  function resetPad() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, pad.width, pad.height);
    ctx.strokeStyle = '#ffffff';
    // ~2px wide once the digit is scaled into the 20x20 box, which is what
    // MNIST's own anti-aliased strokes look like.
    ctx.lineWidth = pad.width / 14;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }

  // Canvas coordinates, correctly. getBoundingClientRect() is in CSS pixels and
  // already accounts for scroll; the ratio converts into the canvas's own
  // coordinate system, which differs whenever CSS sizes the element -- it always
  // does here, the canvas is 336 wide internally and fluid on screen.
  // `e.clientX - canvas.offsetLeft` is the version everyone writes, and it is
  // wrong under scroll, under any CSS scaling, and inside a transformed parent.
  function padPoint(e) {
    var rect = pad.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (pad.width / rect.width),
      y: (e.clientY - rect.top) * (pad.height / rect.height)
    };
  }

  var drawing = false;

  // Pointer events, not mouse events: one code path covers mouse, touch and
  // stylus, and setPointerCapture keeps a stroke alive when the pointer leaves
  // the canvas mid-drag.
  pad.addEventListener('pointerdown', function (e) {
    e.preventDefault();
    pad.setPointerCapture(e.pointerId);
    drawing = true;
    var p = padPoint(e);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(p.x, p.y);          // a tap alone must leave a dot
    ctx.stroke();
    predict();
  });

  pad.addEventListener('pointermove', function (e) {
    if (!drawing) return;
    e.preventDefault();
    var p = padPoint(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    predict();
  });

  function endStroke(e) {
    if (!drawing) return;
    drawing = false;
    if (e && e.pointerId !== undefined && pad.hasPointerCapture(e.pointerId)) {
      pad.releasePointerCapture(e.pointerId);
    }
    predict();
  }
  pad.addEventListener('pointerup', endStroke);
  pad.addEventListener('pointercancel', endStroke);
  pad.addEventListener('pointerleave', endStroke);

  document.getElementById('clear').addEventListener('click', function () {
    resetPad();
    clearOutputs();
  });

  /*
   * Progressive halving before the final resample.
   *
   * drawImage() downscaling is a bilinear filter reading a 2x2 neighbourhood.
   * Going 240 -> 20 in one step therefore samples 4 of the ~144 source pixels
   * covering each destination pixel and discards the rest: thin strokes
   * dissolve into speckle. Halving until within 2x keeps every source pixel in
   * the average, which is what a box filter would have done.
   */
  function downscale(src, sx, sy, sw, sh, dw, dh) {
    var cur = document.createElement('canvas');
    cur.width = sw;
    cur.height = sh;
    cur.getContext('2d').drawImage(src, sx, sy, sw, sh, 0, 0, sw, sh);
    var cw = sw, ch = sh;
    while (cw > dw * 2 || ch > dh * 2) {
      var nw = Math.max(dw, Math.floor(cw / 2));
      var nh = Math.max(dh, Math.floor(ch / 2));
      var next = document.createElement('canvas');
      next.width = nw;
      next.height = nh;
      var nctx = next.getContext('2d');
      nctx.imageSmoothingEnabled = true;
      nctx.imageSmoothingQuality = 'high';
      nctx.drawImage(cur, 0, 0, cw, ch, 0, 0, nw, nh);
      cur = next; cw = nw; ch = nh;
    }
    var out = document.createElement('canvas');
    out.width = dw;
    out.height = dh;
    var octx = out.getContext('2d', { willReadFrequently: true });
    octx.imageSmoothingEnabled = true;
    octx.imageSmoothingQuality = 'high';
    octx.drawImage(cur, 0, 0, cw, ch, 0, 0, dw, dh);
    return octx.getImageData(0, 0, dw, dh);
  }

  /*
   * Turn the drawing into the 784-vector MNIST would have produced.
   *
   *   1. threshold and take the bounding box of the ink
   *   2. scale so the LONG side is 20 pixels, aspect ratio preserved
   *   3. centre of mass of the scaled image
   *   4. paste into a 28x28 field so the centre of mass lands on (14, 14)
   *   5. divide by 255
   *
   * Skip this and a genuinely 98%-accurate model reads hand-drawn input at
   * about 25%. It is not polish; it is the demo.
   */
  function mnistify(canvas) {
    var w = canvas.width, h = canvas.height;
    var src = canvas.getContext('2d', { willReadFrequently: true })
                    .getImageData(0, 0, w, h).data;

    var box = PHASE0.inkBounds(src, w, h, 25);
    if (!box) return null;

    // Scaling both sides to 20 instead of the long side would turn every 1 into
    // a blob and every 0 into a circle.
    var bw = box.maxX - box.minX + 1, bh = box.maxY - box.minY + 1;
    var scale = 20 / Math.max(bw, bh);
    var nw = Math.max(1, Math.round(bw * scale));
    var nh = Math.max(1, Math.round(bh * scale));
    var small = downscale(canvas, box.minX, box.minY, bw, bh, nw, nh).data;

    var com = PHASE0.centreOfMass(small, nw, nh);
    if (!com) return null;
    var off = PHASE0.pasteOffsets(com.cx, com.cy, nw, nh);
    var x = PHASE0.toTensor(small, nw, nh, off.dx, off.dy);
    return {
      x: x, nw: nw, nh: nh, dx: off.dx, dy: off.dy,
      cx: com.cx, cy: com.cy, mean: PHASE0.meanOf(x)
    };
  }

  // ---------------------------------------------------------------------- UI
  var rows = [];
  for (var d = 0; d < 10; d++) {
    var row = document.createElement('div');
    row.className = 'bar';
    row.innerHTML = '<span class="digit">' + d + '</span>' +
                    '<span class="track"><span class="fill"></span></span>' +
                    '<span class="pct">0.0%</span>';
    barsHost.appendChild(row);
    rows.push({ row: row, fill: row.querySelector('.fill'), pct: row.querySelector('.pct') });
  }

  function renderBars(p, top) {
    for (var i = 0; i < 10; i++) {
      var v = p ? p[i] : 0;
      rows[i].fill.style.width = (v * 100).toFixed(1) + '%';
      rows[i].pct.textContent = (v * 100).toFixed(1) + '%';
      rows[i].row.classList.toggle('top', p !== null && i === top);
    }
  }

  function renderPreview(vec) {
    var img = previewCtx.createImageData(28, 28);
    for (var i = 0; i < 784; i++) {
      var v = vec ? Math.round(vec[i] * 255) : 0;
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    previewCtx.putImageData(img, 0, 0);
  }

  function clearOutputs() {
    renderBars(null, -1);
    renderPreview(null);
    stepsLine.textContent = '-';
    meanLine.textContent = '-';
    verdictLine.textContent = '-';
    alertLine.textContent = '';
  }

  function predict() {
    if (!READY) return;
    var prep = mnistify(pad);
    if (!prep) { clearOutputs(); return; }
    renderPreview(prep.x);

    var warning = PHASE0.assertPolarity(prep.mean);
    alertLine.textContent = warning;
    if (warning) console.warn('[phase0] ' + warning);

    var out = PHASE0.forward(prep.x, MODEL);
    var p = out.probs, top = 0;
    for (var i = 1; i < 10; i++) if (p[i] > p[top]) top = i;
    renderBars(p, top);

    stepsLine.textContent = 'box ' + prep.nw + 'x' + prep.nh +
      '  com (' + prep.cx.toFixed(1) + ', ' + prep.cy.toFixed(1) + ')' +
      '  paste (' + prep.dx + ', ' + prep.dy + ')';
    meanLine.textContent = prep.mean.toFixed(4);
    verdictLine.textContent = top + '  at  ' + (p[top] * 100).toFixed(1) + '%';
  }

  // -------------------------------------------------------------------- boot
  resetPad();
  clearOutputs();
  if (READY) {
    metaLine.textContent = ' ' + MODEL.params.toLocaleString() + ' parameters, ' +
      (MODEL.testAccuracy > 0
        ? (MODEL.testAccuracy * 100).toFixed(2) + '% on the MNIST test set.'
        : 'untrained weights (run train.py, then export_web.py).');
  } else {
    metaLine.textContent = '';
    alertLine.textContent =
      'model.js not found. Run: python3 phase0/train.py && python3 phase0/export_web.py';
  }

})();
