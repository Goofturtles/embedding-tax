// Asserts that nothing the film shows depicts a model over the 50,000,000 parameter
// cap, and that no frame shows more layers than its on-screen vocabulary can pay for.
//
// Why this exists: an earlier version of scenes.js interpolated the layer count
// independently of the vocabulary, so mid-transition frames showed 13 layers while the
// embedding table was still ~19,000 tokens wide. scenes.js now clamps layers to
// layersThatFit(vocab), and this test guards that clamp.
//
// Three earlier weaknesses of this test, each fixed and each worth remembering:
//   - It scraped numbers out of rendered text, quantising the embedding count to 0.1%
//     of the cap (50,000 params) - a wider band than the ~30,000 of real headroom, so
//     it could not certify the boundary it existed to check. Now reads exact values.
//   - A failed parse produced NaN, and every NaN comparison is false, so broken samples
//     were skipped in silence. Now every field must be finite or the run throws.
//   - It only checked the animated grid, while scene 3 makes the same
//     vocabulary-to-layers claim with fixed captions. Now both are checked.
//
// Run:  node verify_cap.mjs

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = resolve(fileURLToPath(new URL('.', import.meta.url)));
const CAP = 50_000_000;
const N_EMBD = 512;
// Derived, not copied: attention (4d^2) + MLP (8d^2) + two weight-only norms (2d).
// Writing the literal 3_146_752 here would duplicate scenes.js rather than check it.
const PER_LAYER = 12 * N_EMBD ** 2 + 2 * N_EMBD;

// Scene 2 occupies t in (6.8, 38.0), so ~31.2s of a 50s film depicts the budget.
// Expressed in seconds: an absolute frame count would fail spuriously at --fps 30.
const MIN_LIVE_SECONDS = 30;
const MIN_CLAMPED_SECONDS = 0.25;

const TYPES = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json',
  '.ttf': 'font/ttf', '.woff2': 'font/woff2',
};

// render.mjs takes --fps from argv and records what it used. Sampling on frame
// boundaries is exact only while the two grids coincide: 30fps is a subset of 60,
// but 24 or 50 is not, and the test would then sweep instants the film never contains.
async function renderFps() {
  try {
    const j = JSON.parse(await readFile(resolve(HERE, 'render.json'), 'utf8'));
    if (j.fps === undefined) return 60;              // no prior render
    // Refuse a present-but-unusable value rather than silently sweeping a grid the
    // film never contained: 59.94 would put every sample between rendered frames.
    if (!Number.isInteger(j.fps) || j.fps < 24 || j.fps > 240) {
      throw new Error(`render.json fps is ${j.fps}; expected an integer 24..240`);
    }
    return j.fps;
  } catch (e) {
    if (e instanceof SyntaxError || e.code === 'ENOENT') return 60;
    throw e;
  }
}

const server = createServer(async (req, res) => {
  try {
    const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'scene/index.html';
    const file = resolve(HERE, rel);
    // Resolve first, then check containment: decoding before normalising is how
    // %2e%2e%2f escapes. The sep guard stops a sibling like film-secrets/ matching.
    if (file !== HERE && !file.startsWith(HERE + sep)) {
      res.writeHead(403).end('forbidden');
      return;
    }
    const body = await readFile(file);
    res.writeHead(200, {
      'content-type': TYPES[extname(file)] || 'application/octet-stream',
      'cache-control': 'no-store',
    });
    res.end(body);
  } catch {
    // Covers a malformed percent-escape too, which would otherwise reject unhandled
    // and hard-exit the process past the finally block below.
    res.writeHead(404).end('not found');
  }
});

const FPS = await renderFps();
let browser;
let failed = false;

try {
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  server.unref();
  const port = server.address().port;

  browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') pageErrors.push(`console: ${m.text()}`); });

  // Load the stage exactly as render.mjs does. Both knobs are inert in scenes.js today,
  // but if either ever gates behaviour, this test must sweep the page render captures.
  await page.goto(`http://127.0.0.1:${port}/scene/index.html?static=1`, { waitUntil: 'load' });
  await page.evaluate(() => { window.__manual = true; });
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 20_000 });

  const duration = await page.evaluate(() => window.__duration);
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error(`window.__duration is not a positive number (${duration})`);
  }
  const frames = Math.round(duration * FPS);

  // One round trip instead of 3001. t = f / FPS is the identical single division
  // render.mjs performs, so these are exactly the instants that get rendered.
  const samples = await page.evaluate(({ frames, fps }) => {
    const out = [];
    for (let f = 0; f <= frames; f++) {
      const st = window.__state(f / fps);
      if (st && st.active === true) {
        out.push({ f, vocab: st.vocab, embParams: st.embParams, layers: st.layers,
                   total: st.total, affordable: st.affordable, wanted: st.wanted,
                   labelText: st.labelText, lineText: st.lineText });
      }
    }
    return out;
  }, { frames, fps: FPS });

  const claims = await page.evaluate(() => {
    if (typeof window.__claims !== 'function') throw new Error('window.__claims is missing');
    return window.__claims();
  });
  // Exact count, not a minimum: if #s3 .col ever stops matching, querySelectorAll
  // returns empty, forEach no-ops, and scene 3's two bar claims go unverified in
  // silence while the grid-label pair still satisfies a "2 or more" check.
  const EXPECTED_CLAIMS = 4;
  if (claims.length !== EXPECTED_CLAIMS) {
    throw new Error(`__claims returned ${claims.length} entries, expected exactly ${EXPECTED_CLAIMS}`);
  }

  // Flush: with one batched evaluate there are no longer 3001 round trips for CDP
  // console events to interleave with, so a late error could land after this check.
  await page.evaluate(() => {});

  if (pageErrors.length) throw new Error(`page errors: ${pageErrors.join(' | ')}`);

  const overCap = [];
  const unaffordable = [];
  const disagreed = [];
  const badEmb = [];
  const badLabel = [];
  const badLine = [];
  const claimFails = [];
  // Exact string, not a substring test: a label could otherwise contain "shrinking"
  // alongside an unparseable vocabulary and take the escape hatch.
  const TRANSITIONAL_LABEL = 'shrinking the vocabulary...';
  // The closing line spells its layer count. Any numeral word on screen must match
  // the number rendered beside it.
  const NUMERALS = {
    one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9,
    ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15,
    sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20,
  };
  // Number-ish words with no mapped value. Seeing one means the line states a count
  // this check cannot read, which must fail rather than pass quietly.
  const UNMAPPED_NUMBER_WORDS = new Set([
    'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred',
    'thousand', 'million', 'dozen', 'zero', 'none', 'no',
  ]);
  // The label names a fixed vocabulary while the readout shows the live one. An
  // absolute bound, not a percentage: a relative tolerance is loosest exactly where
  // the numbers are largest, which is where a mismatch is most visible.
  const MAX_LABEL_DRIFT = 100;
  let worst = null;
  let clampedFrames = 0;
  const affordableAt = (v) => Math.floor((CAP - (v * N_EMBD + N_EMBD)) / PER_LAYER);
  // Read the number that actually precedes "tokens": stripping all non-digits swallows
  // the 2 in "GPT-2" and reports 250,257 for a 50,257-token vocabulary.
  const tokenCount = (str) => {
    const m = String(str ?? '').match(/([0-9][0-9,]*)\s*tokens/);
    return m ? Number(m[1].replace(/,/g, '')) : NaN;
  };

  for (const s of samples) {
    for (const k of ['vocab', 'embParams', 'layers', 'total', 'affordable', 'wanted']) {
      if (!Number.isFinite(s[k])) {
        throw new Error(`frame ${s.f}: __state.${k} is not finite (${s[k]})`);
      }
    }
    // Re-derive the total here rather than trusting the page's arithmetic. Without
    // this the test would verify scenes.js against itself: a mistyped PER_LAYER there
    // would move total and affordable together and everything would still pass.
    const expected = s.embParams + s.layers * PER_LAYER + N_EMBD;
    if (Math.abs(s.total - expected) > 1e-6) {
      disagreed.push({ ...s, expected });
    }
    // Recompute affordability here too, so no assertion leans on the page's own
    // arithmetic. Once the table has finished filling, the painted embedding must match
    // the vocabulary it claims, or the two halves describe different models.
    const affordable = affordableAt(s.vocab);
    if (s.layers > 0 && Math.abs(s.embParams - s.vocab * N_EMBD) > 1) badEmb.push(s);
    if (!worst || s.total > worst.total) worst = s;
    if (s.total > CAP) overCap.push(s);
    // Subsumes the over-layer check and makes the clamp non-gameable: layers must be
    // exactly the clamped value, not merely below the ceiling.
    if (s.layers !== Math.min(s.wanted, affordable)) {
      unaffordable.push({ ...s, affordable });
    }
    if (s.wanted > affordable) clampedFrames++;   // the clamp only proves itself here

    // The grid label states a vocabulary in prose beside the live layer count. Check
    // it on EVERY frame of its display window, not just the two saturated endpoints:
    // widening the threshold that governs the window is a cosmetic edit that would
    // otherwise put "50,257 tokens" next to 9 layers (54,052,864 params) undetected.
    const labelVocab = tokenCount(s.labelText);
    if (Number.isFinite(labelVocab)) {
      // Both directions. Affordability alone only catches a label that OVERSTATES the
      // vocabulary; labelling the GPT-2 grid "16,384 tokens" understates it, is
      // trivially affordable, and would otherwise pass.
      const overstates = s.layers > affordableAt(labelVocab);
      const disagrees = Math.abs(labelVocab - s.vocab) > MAX_LABEL_DRIFT;
      if (overstates || disagrees) {
        badLabel.push({ ...s, labelVocab, labelAffordable: affordableAt(labelVocab),
                        why: overstates ? 'cannot afford the layers shown'
                                        : 'disagrees with the vocabulary readout' });
      }
    } else if (String(s.labelText || '') !== TRANSITIONAL_LABEL) {
      // Neither a token count nor the known transitional copy: it states something
      // this test cannot read, which is the silent-skip mode the header warns about.
      badLabel.push({ ...s, labelVocab: NaN, labelAffordable: NaN });
    }

    // Same class of claim as the label: a numeral spelled out beside the readout.
    // Fail closed, matching the label check above: a digit form ("13 layers") or a
    // number word outside the map states a count this cannot read, and silence there
    // is exactly the hole this file exists to prevent.
    const lineRaw = String(s.lineText || '');
    if (lineRaw) {
      const words = lineRaw.toLowerCase().match(/[a-z]+/g) || [];
      const bad = words.find((w) => (w in NUMERALS && NUMERALS[w] !== s.layers) ||
                                    UNMAPPED_NUMBER_WORDS.has(w));
      if (bad) {
        badLine.push({ ...s, said: bad, saidValue: NUMERALS[bad] ?? NaN });
      } else if (/[0-9]/.test(lineRaw)) {
        badLine.push({ ...s, said: lineRaw, saidValue: NaN });
      }
    }
  }

  const n = (x) => Math.round(x).toLocaleString('en-US');

  console.log(`swept ${frames + 1} frames at ${FPS}fps across ${duration}s (${samples.length} depicting the budget)`);
  if (worst) {
    console.log(`peak depicted total: ${n(worst.total)} at frame ${worst.f} ` +
                `(vocab ${n(worst.vocab)}, ${worst.layers} layers, ${affordableAt(worst.vocab)} affordable)`);
    console.log(`cap:                 ${n(CAP)}`);
    console.log(`headroom at peak:    ${n(CAP - worst.total)}`);
  }
  for (const c of claims) {
    const total = c.vocab * N_EMBD + c.layers * PER_LAYER + N_EMBD;
    const affordable = Math.floor((CAP - (c.vocab * N_EMBD + N_EMBD)) / PER_LAYER);
    // Round-trip the parse: digit-stripping turns "50.2k tokens" into 502, which
    // looks affordable. The rendered text must contain the number we think it states.
    const roundTrips = String(c.text || '').includes(n(c.vocab));
    const ok = Number.isFinite(c.vocab) && Number.isFinite(c.layers) &&
               c.vocab > 0 && c.layers > 0 && roundTrips &&
               total <= CAP && c.layers <= affordable;
    console.log(`${c.where}: ${c.layers} layers at vocab ${n(c.vocab)} = ${n(total)} ` +
                `(${affordable} affordable)${roundTrips ? '' : ' [caption did not round-trip]'} ` +
                `${ok ? 'ok' : 'FAIL'}`);
    if (!ok) claimFails.push({ ...c, total, affordable });
  }

  // Guards against the test passing vacuously: if the animation were shortened so the
  // clamp never actually binds, nothing above would fail, yet the regression this file
  // exists for would be unguarded.
  if (disagreed.length) {
    failed = true;
    console.error(`\nFAIL: ${disagreed.length} frames where scenes.js arithmetic disagrees with this test.`);
    for (const d of disagreed.slice(0, 3)) {
      console.error(`  frame ${d.f}: page says ${n(d.total)}, recomputed ${n(d.expected)}`);
    }
  }
  if (overCap.length) {
    failed = true;
    console.error(`\nFAIL: ${overCap.length} frames exceed the cap. First few:`);
    for (const v of overCap.slice(0, 5)) {
      console.error(`  frame ${v.f}: ${n(v.total)} params (vocab ${n(v.vocab)}, ${v.layers} layers)`);
    }
  }
  if (unaffordable.length) {
    failed = true;
    console.error(`\nFAIL: ${unaffordable.length} frames show more layers than the vocabulary affords. First few:`);
    for (const v of unaffordable.slice(0, 5)) {
      console.error(`  frame ${v.f}: ${v.layers} shown, ${v.affordable} affordable (vocab ${n(v.vocab)})`);
    }
  }
  if (claimFails.length) {
    failed = true;
    console.error(`\nFAIL: ${claimFails.length} fixed claims are infeasible:`);
    for (const c of claimFails) {
      console.error(`  ${c.where}: ${c.layers} layers at vocab ${n(c.vocab)} = ` +
                    `${n(c.total)} params, ${c.affordable} affordable`);
    }
  }
  if (badLabel.length) {
    failed = true;
    const b = badLabel[0];
    console.error(`\nFAIL: ${badLabel.length} frames where the grid label is wrong.`);
    console.error(`  frame ${b.f}: label "${b.labelText}" ${b.why ?? 'is unreadable'} ` +
                  `- readout says ${n(b.vocab)}, ${b.layers} layers shown, ` +
                  `${b.labelAffordable} affordable at the labelled vocabulary`);
  }
  if (badLine.length) {
    failed = true;
    const b = badLine[0];
    console.error(`\nFAIL: ${badLine.length} frames where the on-screen line spells a layer ` +
                  `count that does not match the readout.`);
    console.error(`  frame ${b.f}: "${b.lineText}" says ${b.saidValue}, readout shows ${b.layers}`);
  }
  if (badEmb.length) {
    failed = true;
    const b = badEmb[0];
    console.error(`\nFAIL: ${badEmb.length} frames where the painted embedding does not ` +
                  `match the stated vocabulary.`);
    console.error(`  frame ${b.f}: embParams ${n(b.embParams)}, ` +
                  `vocab ${n(b.vocab)} implies ${n(b.vocab * N_EMBD)}`);
  }

  // Coverage guards last, so a run that both shrank coverage and went over cap reports
  // the substantive failures rather than only the first guard to trip.
  const minLive = Math.round(MIN_LIVE_SECONDS * FPS);
  const minClamped = Math.max(3, Math.round(MIN_CLAMPED_SECONDS * FPS));
  if (samples.length < minLive) {
    throw new Error(`only ${samples.length} live samples, expected at least ${minLive} ` +
                    `(${MIN_LIVE_SECONDS}s at ${FPS}fps); coverage has shrunk`);
  }
  if (clampedFrames < minClamped) {
    throw new Error(`the cap clamp bound on only ${clampedFrames} frames, expected at least ` +
                    `${minClamped}; it is barely exercised and this test would nearly pass ` +
                    `even if the clamp were deleted`);
  }

  if (!failed) {
    console.log(`\nPASS: every budget-grid frame and every fixed claim is under the cap ` +
                `and within what its vocabulary affords ` +
                `(clamp bound on ${clampedFrames} frames).`);
  }
} finally {
  // Cleanup must never replace the real result. A browser.close() rejection thrown
  // from here would mask an over-cap failure with an unrelated teardown error, so it
  // is reported and swallowed; server.close() runs regardless, since its listening
  // handle would otherwise keep the loop alive.
  try {
    if (browser) await browser.close();
  } catch (e) {
    console.error(`(teardown) browser.close failed: ${e.message}`);
  } finally {
    server.close();
  }
}

// Set the code rather than calling process.exit, so stderr drains first: on Windows,
// console writes to a pipe are asynchronous and exiting immediately truncates them.
// A thrown error still exits non-zero via Node's unhandled top-level-await rejection.
if (failed) process.exitCode = 1;
