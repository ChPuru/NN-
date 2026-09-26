/*
 * End-to-end verification of the browser demo, under node.
 *
 *   node phase0/verify_web.js <job.json> <out.json>
 *
 * Driven by phase0/verify_web.py, which supplies real MNIST test digits and the
 * NumPy model's prediction for each one. This file is the JavaScript half: it
 * loads the shipped web/model.js and web/app.js, rebuilds the canvas pipeline
 * that app.js's DOM half performs, and reports how often the JS answer matches
 * NumPy's.
 *
 * What is reused and what is re-implemented, stated plainly because it bounds
 * what this test can claim:
 *
 * * REUSED, byte for byte, from the shipped web/app.js: inkBounds,
 *   centreOfMass, pasteOffsets, toTensor, meanOf, assertPolarity, softmax and
 *   forward. Those are the functions the page exports for exactly this purpose.
 * * RE-IMPLEMENTED here: mnistify's control flow and the canvas resampling it
 *   leans on, because both live inside app.js's `typeof document !== 'undefined'`
 *   branch and there is no canvas in node. The re-implementation mirrors the
 *   original line for line -- same threshold, same 20px long side, same
 *   progressive halving, same uint8 quantisation at every intermediate step --
 *   and the recipe is re-derived independently below and asserted against what
 *   the pipeline produced.
 *
 * The one thing this cannot pin down is Chrome's exact bilinear filter kernel:
 * `drawImage` downscaling is not specified to the last bit. That affects the
 * grey levels of the resampled strokes by a hair, never the crop, the scale, the
 * centre of mass or the layout -- which is where the bugs live.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, 'web');
const PHASE0 = require(path.join(WEB, 'app.js'));
// eval, deliberately: model.js is *executable JS* by design (Float32Array.from
// literals, so a <script src> tag works on file:// where fetch does not), which
// means JSON.parse cannot read it. The input is a local file this repository
// generates from weights.npz -- the browser runs it the same way, via a script
// tag, and evaluating it here is what makes this the same program.
const MODEL = eval(fs.readFileSync(path.join(WEB, 'model.js'), 'utf8') + '\nMODEL');

// --------------------------------------------------------------------------
// a canvas substitute: bilinear resampling with uint8 quantisation
// --------------------------------------------------------------------------

/*
 * Bilinear resample of a single-channel uint8 image.
 *
 * The half-pixel offset is the whole story: destination pixel centres sit at
 * dx + 0.5 in destination space, which maps to (dx + 0.5) * sw/dw - 0.5 in
 * source space. Dropping the 0.5s -- the version everyone writes first --
 * shifts the whole image by half a destination pixel, which after a 336 -> 20
 * downscale is eight source pixels of drift and moves the centre of mass.
 *
 * Rounding to uint8 at the end is not cosmetic: a real canvas stores 8-bit
 * channels, so every intermediate of the halving chain is quantised, and a
 * float64 chain would diverge slightly from the browser's.
 */
function resizeBilinear(src, sw, sh, dw, dh) {
  const out = new Uint8ClampedArray(dw * dh);
  const rx = sw / dw, ry = sh / dh;
  for (let y = 0; y < dh; y++) {
    let syf = (y + 0.5) * ry - 0.5;
    if (syf < 0) syf = 0;
    if (syf > sh - 1) syf = sh - 1;
    const y0 = Math.floor(syf), y1 = Math.min(y0 + 1, sh - 1), wy = syf - y0;
    for (let x = 0; x < dw; x++) {
      let sxf = (x + 0.5) * rx - 0.5;
      if (sxf < 0) sxf = 0;
      if (sxf > sw - 1) sxf = sw - 1;
      const x0 = Math.floor(sxf), x1 = Math.min(x0 + 1, sw - 1), wx = sxf - x0;
      const a = src[y0 * sw + x0], b = src[y0 * sw + x1];
      const c = src[y1 * sw + x0], d = src[y1 * sw + x1];
      const top = a + (b - a) * wx, bot = c + (d - c) * wx;
      out[y * dw + x] = Math.round(top + (bot - top) * wy);
    }
  }
  return out;
}

/* Crop a sub-rectangle, exactly as app.js's first drawImage does. */
function crop(src, sw, sh, sx, sy, cw, ch) {
  const out = new Uint8ClampedArray(cw * ch);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) out[y * cw + x] = src[(sy + y) * sw + (sx + x)];
  }
  return out;
}

/*
 * app.js's downscale(): halve until within 2x of the target, then resample.
 *
 * A bilinear filter reads a 2x2 neighbourhood. Going 240 -> 20 in one step
 * therefore samples 4 of the ~144 source pixels covering each destination pixel
 * and throws the rest away, so thin strokes dissolve into speckle. Halving keeps
 * every source pixel in the average, which is what a box filter would have done.
 */
function downscale(src, sw, sh, sx, sy, cw, ch, dw, dh) {
  let cur = crop(src, sw, sh, sx, sy, cw, ch);
  let w = cw, h = ch;
  while (w > dw * 2 || h > dh * 2) {
    const nw = Math.max(dw, Math.floor(w / 2));
    const nh = Math.max(dh, Math.floor(h / 2));
    cur = resizeBilinear(cur, w, h, nw, nh);
    w = nw; h = nh;
  }
  return resizeBilinear(cur, w, h, dw, dh);
}

/* Widen a single-channel image into the RGBA layout app.js's helpers index. */
function toRgba(gray, w, h) {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    out[i * 4] = gray[i];
    out[i * 4 + 1] = gray[i];
    out[i * 4 + 2] = gray[i];
    out[i * 4 + 3] = 255;
  }
  return out;
}

// --------------------------------------------------------------------------
// the pipeline under test
// --------------------------------------------------------------------------

/*
 * mnistify: the five steps app.js performs on the drawing pad.
 *
 *   1. threshold at 25 and take the bounding box of the ink
 *   2. scale so the LONG side is 20 pixels, aspect ratio preserved
 *   3. centre of mass of the scaled image
 *   4. paste into 28x28 so the centre of mass lands on (14, 14)
 *   5. divide by 255
 */
function mnistify(pad, w, h) {
  const box = PHASE0.inkBounds(toRgba(pad, w, h), w, h, 25);
  if (!box) return null;
  const bw = box.maxX - box.minX + 1, bh = box.maxY - box.minY + 1;
  const scale = 20 / Math.max(bw, bh);
  const nw = Math.max(1, Math.round(bw * scale));
  const nh = Math.max(1, Math.round(bh * scale));
  const small = downscale(pad, w, h, box.minX, box.minY, bw, bh, nw, nh);
  const rgbaSmall = toRgba(small, nw, nh);
  const com = PHASE0.centreOfMass(rgbaSmall, nw, nh);
  if (!com) return null;
  const off = PHASE0.pasteOffsets(com.cx, com.cy, nw, nh);
  const x = PHASE0.toTensor(rgbaSmall, nw, nh, off.dx, off.dy);
  return {
    x: x, nw: nw, nh: nh, dx: off.dx, dy: off.dy,
    cx: com.cx, cy: com.cy, bw: bw, bh: bh, mean: PHASE0.meanOf(x)
  };
}

/*
 * The baseline: skip mnistify, just resize the pad to 28x28 and divide by 255.
 *
 * This is not a straw man -- it is what the demo does if you delete the
 * preprocessing, and the accuracy it gets is the number that justifies the
 * hundred lines above.
 */
function naiveResize(pad, w, h) {
  const small = downscale(pad, w, h, 0, 0, w, h, 28, 28);
  const x = new Float32Array(784);
  for (let i = 0; i < 784; i++) x[i] = small[i] / 255;
  return x;
}

/*
 * Draw one MNIST digit onto the pad the way a user's stroke would land there:
 * white ink on a black ground, at an arbitrary size and position.
 *
 * Placement is what makes the comparison mean something. Drawing every digit
 * dead centre at exactly 28/336 scale would let the naive baseline succeed for
 * a reason no real user reproduces.
 */
function renderToPad(digit, padSize, size, ox, oy) {
  const pad = new Uint8ClampedArray(padSize * padSize);  // 0 = black ground
  const up = resizeBilinear(Uint8ClampedArray.from(digit), 28, 28, size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) pad[(oy + y) * padSize + (ox + x)] = up[y * size + x];
  }
  return pad;
}

function argmax(v) {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

// --------------------------------------------------------------------------
// the three file:// traps, checked against the shipped sources
// --------------------------------------------------------------------------

/*
 * Strip comments before searching for an anti-pattern.
 *
 * Both sources document the very traps these checks look for, by name, in
 * prose: index.html's opening comment explains why fetch() is unusable on
 * file://, so searching the raw text finds the explanation and reports a
 * failure. HTML comments have to go too, not only the two JavaScript forms.
 */
function codeOnly(text) {
  return text
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/[^\n]*/g, '');
}

function checkTraps() {
  const html = fs.readFileSync(path.join(WEB, 'index.html'), 'utf8');
  const app = fs.readFileSync(path.join(WEB, 'app.js'), 'utf8');
  const modelSrc = fs.readFileSync(path.join(WEB, 'model.js'), 'utf8');
  const appCode = codeOnly(app), htmlCode = codeOnly(html);
  return [
    ['weights arrive via <script src>, not fetch',
     html.indexOf('<script src="model.js"></script>') >= 0 &&
     modelSrc.indexOf('const MODEL = {') >= 0 &&
     appCode.indexOf('fetch(') < 0 && htmlCode.indexOf('fetch(') < 0 &&
     appCode.indexOf('XMLHttpRequest') < 0],
    ['canvas coords use getBoundingClientRect() scaled by width/rect.width',
     appCode.indexOf('getBoundingClientRect()') >= 0 &&
     appCode.indexOf('pad.width / rect.width') >= 0 &&
     appCode.indexOf('pad.height / rect.height') >= 0 &&
     appCode.indexOf('offsetLeft') < 0 && appCode.indexOf('offsetTop') < 0],
    ['polarity is white ink on black, with an assertion on the fed mean',
     appCode.indexOf("ctx.fillStyle = '#000000'") >= 0 &&
     appCode.indexOf("ctx.strokeStyle = '#ffffff'") >= 0 &&
     appCode.indexOf('PHASE0.assertPolarity(prep.mean)') >= 0 &&
     PHASE0.assertPolarity(0.13) === '' &&
     PHASE0.assertPolarity(0.85).indexOf('inverted') >= 0]
  ];
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
function main() {
  const job = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const padSize = job.padSize;
  const n = job.digits.length;

  let agree = 0, accPrep = 0, accNaive = 0, agreeNaive = 0;
  const tensors = [], predsPrep = [], predsNaive = [];
  let maxLongSideErr = 0, maxOffsetErr = 0, maxComErr = 0, worstMean = 0;

  for (let i = 0; i < n; i++) {
    const place = job.placements[i];
    const pad = renderToPad(job.digits[i], padSize, place[0], place[1], place[2]);

    const prep = mnistify(pad, padSize, padSize);
    if (!prep) throw new Error('mnistify found no ink in digit ' + i);

    // Re-derive the recipe independently of the code that just ran it.
    maxLongSideErr = Math.max(maxLongSideErr, Math.abs(Math.max(prep.nw, prep.nh) - 20));
    const wantDx = Math.min(Math.max(Math.round(14 - prep.cx), 0), 28 - prep.nw);
    const wantDy = Math.min(Math.max(Math.round(14 - prep.cy), 0), 28 - prep.nh);
    maxOffsetErr = Math.max(maxOffsetErr, Math.abs(prep.dx - wantDx), Math.abs(prep.dy - wantDy));

    // The centre of mass of the finished 28x28 tensor must sit on (14, 14),
    // up to the integer rounding of the paste offset (so within half a pixel).
    let s = 0, sx = 0, sy = 0;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = prep.x[y * 28 + x];
        s += v; sx += x * v; sy += y * v;
      }
    }
    maxComErr = Math.max(maxComErr, Math.abs(sx / s - 14), Math.abs(sy / s - 14));
    worstMean = Math.max(worstMean, prep.mean);

    const pPrep = argmax(PHASE0.forward(prep.x, MODEL).probs);
    const xNaive = naiveResize(pad, padSize, padSize);
    const pNaive = argmax(PHASE0.forward(xNaive, MODEL).probs);

    if (pPrep === job.numpyPred[i]) agree++;
    if (pNaive === job.numpyPred[i]) agreeNaive++;
    if (pPrep === job.labels[i]) accPrep++;
    if (pNaive === job.labels[i]) accNaive++;

    tensors.push(Array.from(prep.x));
    predsPrep.push(pPrep);
    predsNaive.push(pNaive);
  }

  const traps = checkTraps();
  const pct = (k) => (100 * k / n).toFixed(2) + '%';

  console.log('node ' + process.version + ', ' + n + ' real MNIST test digits');
  console.log('model.js: ' + MODEL.arch + ', ' + MODEL.params +
              ' params, testAccuracy ' + (MODEL.testAccuracy * 100).toFixed(2) + '%');
  console.log('');
  console.log('agreement with the NumPy model on the same digits');
  console.log('  JS pipeline (render -> mnistify -> forward) vs NumPy : ' +
              pct(agree) + '  (' + agree + '/' + n + ')');
  console.log('  same, with mnistify REMOVED (plain 28x28 resize)     : ' +
              pct(agreeNaive) + '  (' + agreeNaive + '/' + n + ')');
  console.log('');
  console.log('accuracy against the true labels, on the rendered pads');
  console.log('  WITH mnistify    : ' + pct(accPrep) + '  (' + accPrep + '/' + n + ')');
  console.log('  WITHOUT mnistify : ' + pct(accNaive) + '  (' + accNaive + '/' + n + ')');
  console.log('');
  console.log('mnistify recipe, re-derived per digit');
  console.log('  max |long side - 20px|        : ' + maxLongSideErr);
  console.log('  max |paste offset - clamped|  : ' + maxOffsetErr);
  console.log('  max |centre of mass - 14|     : ' + maxComErr.toFixed(4) + ' px');
  console.log('  heaviest fed tensor mean      : ' + worstMean.toFixed(4) +
              '  (MNIST is ~0.13; > 0.5 means inverted)');
  console.log('');
  console.log('file:// traps');
  traps.forEach(function (t) {
    console.log('  [' + (t[1] ? 'ok' : 'FAIL') + '] ' + t[0]);
  });

  fs.writeFileSync(process.argv[3], JSON.stringify({
    n: n,
    agree: agree,
    agreeNaive: agreeNaive,
    accPrep: accPrep,
    accNaive: accNaive,
    maxLongSideErr: maxLongSideErr,
    maxOffsetErr: maxOffsetErr,
    maxComErr: maxComErr,
    worstMean: worstMean,
    traps: traps,
    tensors: tensors,
    predsPrep: predsPrep,
    predsNaive: predsNaive,
    node: process.version
  }));

  /*
   * This process gates only on the invariants that are exactly true or exactly
   * false -- the recipe and the three traps. The agreement *rate* is a
   * statistic about resampling, not about the code, so the decision on it lives
   * in verify_web.py where the reconstruction error that explains it is also
   * measured. Encoding a rate threshold in two places invites moving whichever
   * one is inconvenient.
   */
  const allTraps = traps.every(function (t) { return t[1]; });
  const ok = maxLongSideErr === 0 && maxOffsetErr === 0 && maxComErr < 0.51 && allTraps;
  process.exit(ok ? 0 : 1);
}

main();
