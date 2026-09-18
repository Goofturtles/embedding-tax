// The Embedding Tax - film timeline.
//
// Every visual property is a pure function of one number: the film time `t`.
// No CSS animations, no wall clock, no state carried between frames. That is what
// lets render.mjs ask for frame 1,847 directly and get identical pixels every run.

const CAP = 50_000_000;
const PARAMS_PER_CELL = 1_000;
const COLS = 250, ROWS = 200;              // 50,000 cells = the whole budget
const CELLS = COLS * ROWS;
if (CELLS * PARAMS_PER_CELL !== CAP) {
  throw new Error(`grid depicts ${CELLS * PARAMS_PER_CELL} parameters, not the ${CAP} cap`);
}
const PER_LAYER = 3_146_752;               // attn 4d^2 + mlp 8d^2 + 2 norms, d=512

const GPT2_VOCAB = 50_257, OURS_VOCAB = 16_384, N_EMBD = 512;
const GPT2_EMB = GPT2_VOCAB * N_EMBD;      // 25,731,584
const OURS_EMB = OURS_VOCAB * N_EMBD;      //  8,388,608
const GPT2_LAYERS = 7, OURS_LAYERS = 13;

const DUR = 50;                            // total film length in seconds

/* ---------- helpers ---------- */

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
/** Layers affordable under the cap at this vocabulary. Mirrors tools/param_count.py. */
function layersThatFit(vocab) {
  return Math.floor((CAP - (vocab * N_EMBD + N_EMBD)) / PER_LAYER);
}
const lerp = (a, b, p) => a + (b - a) * p;
// Ramp from 0 to 1 across [a,b], eased.
const ramp = (t, a, b, ease = easeInOut) => ease(clamp01((t - a) / (b - a)));
function easeInOut(x) { return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; }
function easeOut(x) { return 1 - Math.pow(1 - x, 3); }

/** Opacity envelope for a scene occupying [start,end], with fade in/out. */
function sceneAlpha(t, start, end, fin = 0.6, fout = 0.6) {
  if (t < start - fin || t > end + fout) return 0;
  if (t < start) return clamp01((t - (start - fin)) / fin);
  if (t > end) return clamp01(1 - (t - end) / fout);
  return 1;
}

function setScene(el, alpha, rise = 0) {
  el.style.opacity = alpha.toFixed(4);
  el.style.transform = alpha >= 1 && rise === 0 ? "none" : `translateY(${((1 - alpha) * rise).toFixed(2)}px)`;
}

const commas = (n) => Math.round(n).toLocaleString("en-US");

/* ---------- deterministic grain, built once ---------- */

(function buildGrain() {
  const N = 180;
  const c = document.createElement("canvas");
  c.width = N; c.height = N;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(N, N);
  let seed = 20260908;
  const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < N * N; i++) {
    const v = 120 + rand() * 135;
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  document.documentElement.style.setProperty("--grain", `url(${c.toDataURL()})`);
})();

/* ---------- the budget grid ---------- */

const gridCanvas = document.getElementById("grid");
const gctx = gridCanvas.getContext("2d");
const cellBuf = document.createElement("canvas");
cellBuf.width = COLS; cellBuf.height = ROWS;
const cellCtx = cellBuf.getContext("2d");
const cellImg = cellCtx.createImageData(COLS, ROWS);

const RGB = {
  tax: [232, 86, 58],
  depth: [79, 180, 148],
  empty: [35, 33, 30],
  hatch: [255, 255, 255],
  off: [12, 12, 11],
};

/**
 * Paint the budget.
 * @param reveal 0..1 how much of the grid has materialised at all
 * @param embCells how many cells the embedding table occupies
 * @param layerCells how many cells the transformer layers occupy
 */
function drawGrid(reveal, embCells, layerCells) {
  if (embCells + layerCells > CELLS) {
    throw new Error(`budget overflow: ${embCells} + ${layerCells} > ${CELLS} cells`);
  }
  const shown = Math.round(CELLS * clamp01(reveal));
  for (let i = 0; i < CELLS; i++) {
    const x = i % COLS, y = (i / COLS) | 0;
    let c;
    if (i >= shown) c = RGB.off;
    else if (i < embCells) c = (x + y) % 4 === 0 ? RGB.hatch : RGB.tax;   // hatched: survives greyscale
    else if (i < embCells + layerCells) c = RGB.depth;
    else c = RGB.empty;
    const o = i * 4;
    cellImg.data[o] = c[0]; cellImg.data[o + 1] = c[1]; cellImg.data[o + 2] = c[2]; cellImg.data[o + 3] = 255;
  }
  cellCtx.putImageData(cellImg, 0, 0);
  gctx.imageSmoothingEnabled = false;
  gctx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);
  gctx.drawImage(cellBuf, 0, 0, gridCanvas.width, gridCanvas.height);
}

/* ---------- scene 1: the cap ---------- */

const S1 = { a: 0, b: 7.5 };
const el = (id) => document.getElementById(id);

function scene1(t) {
  const alpha = sceneAlpha(t, S1.a, S1.b - 0.6, 0.5, 0.7);
  setScene(el("s1"), alpha, 26);
  // Count up fast then settle, so the final number lands rather than crawls.
  const p = ramp(t, S1.a + 0.35, S1.a + 3.4, easeOut);
  el("s1-num").textContent = commas(CAP * p);
  el("s1-cap").style.opacity = ramp(t, S1.a + 2.6, S1.a + 3.8).toFixed(3);
}

/* ---------- scene 2: the grid ---------- */

const S2 = { a: 7.5, b: 38 };

// Exact values from the most recent scene2 evaluation, for verify_cap.mjs. Reading
// these off the rendered text loses precision the cap check cannot afford.
let lastState = { active: false };

function scene2(t) {
  const alpha = sceneAlpha(t, S2.a, S2.b - 0.7, 0.7, 0.7);
  setScene(el("s2"), alpha);
  if (alpha <= 0) { lastState = { active: false }; return; }

  const u = t - S2.a;                       // local time, 0 .. 30.5

  // Beat 1 (0-3): the budget materialises, still empty.
  const reveal = ramp(u, 0.2, 3.0, easeOut);

  // Beat 2 (3-8): GPT-2's word list floods in.
  const taxIn = ramp(u, 3.0, 7.6, easeInOut);
  // Beat 4 (16-21): the vocabulary shrinks, the red retreats and the green grows
  // into the space it frees. One motion, because that is the whole argument.
  const shrink = ramp(u, 16.0, 21.0, easeInOut);

  const vocab = lerp(GPT2_VOCAB, OURS_VOCAB, shrink);
  const embParams = lerp(GPT2_EMB, OURS_EMB, shrink) * taxIn;
  const embCells = Math.round(embParams / PARAMS_PER_CELL);

  // Beat 3 (9.5-14): 7 layers stack. Then they grow to 13 across the shrink,
  // never dropping back to zero in between.
  //
  // The count is clamped to what the CURRENT vocabulary can actually pay for. Without
  // this, an intermediate frame shows 11 layers while the embedding table is still
  // 30,509 tokens wide, which totals 50,234,880 parameters: over the cap the film is
  // about. No frame may ever depict an infeasible model. verify_cap.mjs enforces this.
  const stackA = ramp(u, 9.5, 14.0, easeOut);
  const wanted = Math.round(lerp(GPT2_LAYERS * stackA, OURS_LAYERS, shrink));
  const layers = Math.min(wanted, layersThatFit(vocab));
  const layerCells = Math.round((layers * PER_LAYER) / PARAMS_PER_CELL);

  // The label states a vocabulary in prose next to the live layer count, so it is a
  // claim in its own right. Computed here and carried in the state so every frame of
  // its display window is checkable, not just the two saturated endpoints.
  const LABEL_EPS = 0.0014;              // 0.0014 * (50257 - 16384) = 47 tokens
  const labelText =
    shrink < LABEL_EPS ? "GPT-2 tokenizer - 50,257 tokens"
    : shrink > 1 - LABEL_EPS ? "our tokenizer - 16,384 tokens"
    : "shrinking the vocabulary...";

  lastState = {
    active: true,
    vocab, embParams, layers, embCells, layerCells, labelText,
    // What the animation asked for before the cap clamped it.
    wanted,
    // What the frame actually depicts, exactly. Includes the final norm, matching
    // model/config.py's accounting: embedding + layers + final norm.
    total: embParams + layers * PER_LAYER + N_EMBD,
    // The ceiling the vocabulary on screen can pay for.
    affordable: layersThatFit(vocab),
  };

  drawGrid(reveal, embCells, layerCells);

  el("grid-label").textContent = labelText;
  el("grid-label").style.opacity = ramp(u, 2.6, 3.4).toFixed(3);

  const roAlpha = ramp(u, 3.2, 4.2);
  el("ro-vocab").style.opacity = roAlpha.toFixed(3);
  el("ro-emb").style.opacity = roAlpha.toFixed(3);
  el("ro-layers").style.opacity = ramp(u, 9.2, 10.0).toFixed(3);

  el("ro-vocab").querySelector("b").textContent = commas(vocab);
  el("ro-emb").querySelector("b").textContent = `${(100 * embParams / CAP).toFixed(1)}%`;
  el("ro-layers").querySelector("b").textContent = String(layers);

  // The two lines of the argument, one per half.
  const line = el("s2-line");
  let text = "", vis = 0;
  if (u >= 7.4 && u < 15.4) { text = "Half of it is just a list of words."; vis = ramp(u, 7.6, 8.4) * (1 - ramp(u, 14.6, 15.4)); }
  else if (u >= 15.4 && u < 21.4) { text = "So use fewer words."; vis = ramp(u, 15.8, 16.6) * (1 - ramp(u, 20.6, 21.4)); }
  else if (u >= 21.4) { text = "Same cap. Thirteen layers."; vis = ramp(u, 21.8, 22.8); }
  line.textContent = text;
  line.style.opacity = vis.toFixed(3);
  // The closing line spells a layer count in words. Carry it so verify_cap.mjs can
  // hold it against the number actually on screen: it is the same kind of claim as
  // the grid label, and a pacing change could put "Thirteen" beside a readout of 12.
  lastState.lineText = vis > 0.01 ? text : "";
}

/* ---------- scene 3: the comparison ---------- */

const S3 = { a: 38, b: 44 };
const BAR_MAX = 470;

function scene3(t) {
  const alpha = sceneAlpha(t, S3.a, S3.b - 0.6, 0.6, 0.6);
  setScene(el("s3"), alpha, 20);
  if (alpha <= 0) return;
  const u = t - S3.a;
  const grow = ramp(u, 0.3, 2.4, easeOut);
  el("bar-a").style.height = `${(BAR_MAX * (GPT2_LAYERS / OURS_LAYERS) * grow).toFixed(1)}px`;
  el("bar-b").style.height = `${(BAR_MAX * grow).toFixed(1)}px`;
  el("bar-a").style.background = "var(--tax)";
  el("num-a").textContent = String(Math.round(GPT2_LAYERS * grow));
  el("num-b").textContent = String(Math.round(OURS_LAYERS * grow));
  el("s3-line").style.opacity = ramp(u, 2.6, 3.5).toFixed(3);
}

/* ---------- scene 4: title ---------- */

const S4 = { a: 44, b: DUR };

function scene4(t) {
  const alpha = sceneAlpha(t, S4.a, S4.b, 0.8, 0.4);
  setScene(el("s4"), alpha, 18);
  if (alpha <= 0) return;
  const u = t - S4.a;
  el("s4-sub").style.opacity = ramp(u, 1.1, 2.2).toFixed(3);
}

/* ---------- seek ---------- */

let lastT = 0;

function seek(t) {
  lastT = t;
  scene1(t);
  scene2(t);
  scene3(t);
  scene4(t);
}

window.__seek = seek;
window.__duration = DUR;
/**
 * The fixed vocabulary-to-layer claims made outside the animated grid, i.e. scene 3's
 * two bars and their captions in index.html. Their on-screen counts are
 * round(N * grow) with grow in [0,1], so they only ever fall below these endpoints;
 * checking the endpoints therefore covers every intermediate frame.
 */
window.__claims = () => {
  const t0 = lastT;
  // Read the number that actually precedes "tokens". Stripping all non-digits
  // swallows the 2 in "GPT-2" and reports 250,257 for a 50,257-token vocabulary.
  const tokenCount = (str) => {
    const m = String(str).match(/([0-9][0-9,]*)\s*tokens/);
    return m ? Number(m[1].replace(/,/g, "")) : NaN;
  };
  const digits = (str) => Number(String(str).replace(/[^0-9]/g, ""));
  const out = [];

  // Scene 3's two bars, sampled where grow has saturated.
  seek(S3.a + 3.0);
  Array.from(document.querySelectorAll("#s3 .col")).forEach((col, i) => {
    const text = col.querySelector(".barcap").textContent.trim();
    out.push({
      where: `scene3 ${i === 0 ? "left" : "right"} bar`,
      vocab: tokenCount(text), text,
      layers: digits(col.querySelector(".barnum").textContent),
    });
  });

  // The grid label, at both saturated ends, paired with the layer readout beside it.
  [["grid label, GPT-2 end", S2.a + 15], ["grid label, ours end", S2.a + 27]].forEach(([where, t]) => {
    seek(t);
    const text = el("grid-label").textContent.trim();
    out.push({
      where, vocab: tokenCount(text), text,
      layers: digits(el("ro-layers").querySelector("b").textContent),
    });
  });

  seek(t0);
  return out;
};
/** Seek and return the exact budget state at t. Used by verify_cap.mjs. */
window.__state = (t) => { seek(t); return lastState; };
seek(0);

// render.mjs waits for this before capturing. Gate it on the bundled face actually
// being ready, otherwise frame 0 can rasterise in a fallback font.
document.fonts.ready.then(() => {
  seek(0);
  window.__ready = true;
});
