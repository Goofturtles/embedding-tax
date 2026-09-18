// Score generator for the mem film.
//
// Writes a 65-second 48kHz stereo WAV. Everything is synthesised from scratch
// — additive sines with real envelopes, a Schroeder reverb, a soft limiter —
// so there is no licence attached to the output and the arrangement can be
// aligned to the picture to the exact second.
//
//   node music.mjs            → music.wav
//
// The progression is A minor: i – VI – III – VII, resolving home on the
// sign-off. It's warm rather than triumphant, which suits a film about
// remembering things rather than about winning.

import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SR = 48000;
const DUR = 65.0;
const N = Math.round(SR * DUR);

// ---------- notes ----------

const NOTE = {
  E2: 82.41, F2: 87.31, G2: 98.00, A2: 110.00, B2: 123.47,
  C3: 130.81, D3: 146.83, E3: 164.81, F3: 174.61, G3: 196.00, A3: 220.00, B3: 246.94,
  C4: 261.63, D4: 293.66, E4: 329.63, F4: 349.23, G4: 392.00, A4: 440.00,
  C5: 523.25, E5: 659.25, A5: 880.00,
};

// Sections are aligned to the film's own cuts, so the harmony changes where
// the picture does rather than on an arbitrary bar line.
const SECTIONS = [
  { at: 0.0,  chord: [NOTE.A2, NOTE.E3, NOTE.A3],                     level: 0.34 }, // cold open
  { at: 7.2,  chord: [NOTE.A2, NOTE.F3, NOTE.C4],                     level: 0.40 }, // the problem — unresolved
  { at: 15.4, chord: [NOTE.A2, NOTE.E3, NOTE.A3, NOTE.C4],            level: 0.62 }, // reveal
  { at: 21.4, chord: [NOTE.F2, NOTE.C3, NOTE.F3, NOTE.A3],            level: 0.58 }, // ask
  { at: 27.0, chord: [NOTE.C3, NOTE.G3, NOTE.C4, NOTE.E4],            level: 0.58 },
  { at: 32.4, chord: [NOTE.G2, NOTE.D3, NOTE.G3, NOTE.B3],            level: 0.58 }, // passage
  { at: 38.0, chord: [NOTE.A2, NOTE.E3, NOTE.A3, NOTE.C4],            level: 0.60 },
  { at: 41.2, chord: [NOTE.F2, NOTE.C3, NOTE.F3, NOTE.A3],            level: 0.60 }, // episodes
  { at: 45.0, chord: [NOTE.C3, NOTE.G3, NOTE.C4, NOTE.E4],            level: 0.60 },
  { at: 48.8, chord: [NOTE.G2, NOTE.D3, NOTE.G3, NOTE.B3],            level: 0.62 }, // notices
  { at: 52.2, chord: [NOTE.E2, NOTE.B2, NOTE.E3, NOTE.G3],            level: 0.62 },
  { at: 55.4, chord: [NOTE.F2, NOTE.C3, NOTE.F3, NOTE.A3],            level: 0.70 }, // on-device — the lift
  { at: 58.2, chord: [NOTE.G2, NOTE.D3, NOTE.G3, NOTE.B3],            level: 0.74 },
  { at: 60.6, chord: [NOTE.A2, NOTE.E3, NOTE.A3, NOTE.C4, NOTE.E4],   level: 0.72 }, // home
];

function sectionAt(t) {
  let s = SECTIONS[0];
  for (const sec of SECTIONS) if (t >= sec.at) s = sec;
  return s;
}
function sectionIndexAt(t) {
  let i = 0;
  for (let k = 0; k < SECTIONS.length; k++) if (t >= SECTIONS[k].at) i = k;
  return i;
}

// ---------- buffers ----------

const L = new Float64Array(N);
const R = new Float64Array(N);

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Equal-power crossfade weight between adjacent sections. */
const XFADE = 1.6;
function sectionWeights(t) {
  const i = sectionIndexAt(t);
  const cur = SECTIONS[i];
  const next = SECTIONS[i + 1];
  if (!next) return [{ sec: cur, w: 1 }];
  const into = clamp01((t - next.at + XFADE) / XFADE);
  if (into <= 0) return [{ sec: cur, w: 1 }];
  // Equal power keeps the total energy flat through the change, so chords
  // don't dip in the middle of every transition.
  return [
    { sec: cur, w: Math.cos(into * Math.PI / 2) },
    { sec: next, w: Math.sin(into * Math.PI / 2) },
  ];
}

// ---------- pad ----------
//
// Each chord tone is three slightly detuned sines. The detune is what stops it
// sounding like a test tone: the partials drift in and out of phase, which is
// most of what "warm" means for a synth pad.

const DETUNE = [-0.12, 0, 0.14];   // in Hz
for (let n = 0; n < N; n++) {
  const t = n / SR;
  const parts = sectionWeights(t);
  let l = 0, r = 0;

  for (const { sec, w } of parts) {
    if (w <= 0.0001) continue;
    for (let ci = 0; ci < sec.chord.length; ci++) {
      const f = sec.chord[ci];
      // Higher chord tones sit quieter, the way a real voicing does.
      const voiceGain = 1 / (1 + ci * 0.55);
      for (let d = 0; d < DETUNE.length; d++) {
        const ff = f + DETUNE[d];
        const ph = 2 * Math.PI * ff * t;
        const v = Math.sin(ph) * voiceGain * w * sec.level * 0.13;
        // Spread the detuned copies across the stereo field.
        const pan = d === 0 ? -0.6 : d === 2 ? 0.6 : 0;
        l += v * (1 - Math.max(0, pan));
        r += v * (1 + Math.min(0, pan));
      }
    }
  }
  L[n] += l;
  R[n] += r;
}

// ---------- sub bass ----------
//
// The root, an octave down, following the progression. Carries the weight so
// the pad doesn't have to be loud to feel present.

for (let n = 0; n < N; n++) {
  const t = n / SR;
  let v = 0;
  for (const { sec, w } of sectionWeights(t)) {
    const root = sec.chord[0] / 2;
    // A touch of second harmonic so it survives on laptop speakers that can't
    // reproduce 55Hz at all.
    v += (Math.sin(2 * Math.PI * root * t) * 0.9 + Math.sin(2 * Math.PI * root * 2 * t) * 0.18)
       * w * sec.level * 0.30;
  }
  L[n] += v;
  R[n] += v;
}

// ---------- pulse ----------
//
// A soft eighth-note pluck through the product beats only. It gives the middle
// of the film forward motion; the open and the close are deliberately without
// it so they feel still.

const BPM = 72;
const BEAT = 60 / BPM;
const PULSE_IN = 21.0;
const PULSE_OUT = 56.0;

function pluck(startT, freq, gain, decay) {
  const start = Math.round(startT * SR);
  const len = Math.round(decay * 3 * SR);
  for (let i = 0; i < len && start + i < N; i++) {
    const tt = i / SR;
    const env = Math.exp(-tt / decay);
    // Two partials and a click transient — enough to read as plucked rather
    // than as a sine fading out.
    const s = (Math.sin(2 * Math.PI * freq * tt) * 0.7
             + Math.sin(2 * Math.PI * freq * 2 * tt) * 0.22
             + Math.sin(2 * Math.PI * freq * 3.01 * tt) * 0.08) * env * gain;
    const pan = ((freq % 7) / 7 - 0.5) * 0.5;
    L[start + i] += s * (1 - Math.max(0, pan));
    R[start + i] += s * (1 + Math.min(0, pan));
  }
}

let step = 0;
for (let t = PULSE_IN; t < PULSE_OUT; t += BEAT / 2, step++) {
  const sec = sectionAt(t);
  // Cycle the upper chord tones, an octave up, skipping a beat here and there
  // so it breathes instead of machine-gunning.
  if (step % 8 === 3 || step % 8 === 6) continue;
  const tone = sec.chord[1 + (step % (sec.chord.length - 1))] * 2;
  // Fade the pulse in and out at the edges of its span.
  const edge = Math.min(clamp01((t - PULSE_IN) / 2.5), clamp01((PULSE_OUT - t) / 3.0));
  pluck(t, tone, 0.055 * edge, 0.34);
}

// ---------- bells ----------
//
// Two only: the logo lands, and the sign-off lands. Anything more and they
// stop being punctuation.

function bell(startT, freq, gain) {
  const start = Math.round(startT * SR);
  const len = Math.round(6 * SR);
  // Inharmonic partials are what separate a bell from an organ.
  const partials = [[1, 1], [2.01, 0.5], [2.99, 0.28], [4.21, 0.15], [5.43, 0.09]];
  for (let i = 0; i < len && start + i < N; i++) {
    const tt = i / SR;
    let s = 0;
    for (const [mult, amp] of partials) {
      s += Math.sin(2 * Math.PI * freq * mult * tt) * amp * Math.exp(-tt / (2.4 / Math.sqrt(mult)));
    }
    const v = s * gain * (1 - Math.exp(-tt * 300)); // soft attack, no click
    L[start + i] += v;
    R[start + i] += v;
  }
}

bell(15.7, NOTE.A4, 0.085);   // the brand lock
bell(60.8, NOTE.A4, 0.075);   // the sign-off
bell(61.0, NOTE.E5, 0.040);

// ---------- transition swells ----------
//
// Filtered noise rising into each scene change. Almost subliminal, but it's
// what makes a cut feel intentional rather than abrupt.

let noiseState = 0;
function swell(atT, dur, gain) {
  const start = Math.round((atT - dur) * SR);
  const len = Math.round(dur * SR);
  for (let i = 0; i < len; i++) {
    const idx = start + i;
    if (idx < 0 || idx >= N) continue;
    const p = i / len;
    // Deterministic noise via a simple LCG, low-passed by a one-pole.
    noiseState = (noiseState * 1664525 + 1013904223) >>> 0;
    const white = (noiseState / 4294967296) * 2 - 1;
    const lp = white * 0.3;
    const env = Math.pow(p, 2.2) * (1 - Math.pow(p, 8));
    const v = lp * env * gain;
    L[idx] += v;
    R[idx] += v * 0.85;
  }
}
for (const sec of SECTIONS.slice(1)) swell(sec.at, 1.4, 0.10);

// ---------- reverb ----------
//
// Schroeder: four parallel combs into two series allpasses. Cheap, and it is
// the difference between "synth" and "a room with something playing in it".

function reverb(buf, mix) {
  const combs = [1557, 1617, 1491, 1422].map((d) => ({ d, buf: new Float64Array(d), i: 0, fb: 0.82 }));
  const allpass = [225, 556].map((d) => ({ d, buf: new Float64Array(d), i: 0, fb: 0.5 }));
  const out = new Float64Array(buf.length);

  for (let n = 0; n < buf.length; n++) {
    let s = 0;
    for (const c of combs) {
      const y = c.buf[c.i];
      c.buf[c.i] = buf[n] + y * c.fb;
      c.i = (c.i + 1) % c.d;
      s += y;
    }
    s *= 0.25;
    for (const a of allpass) {
      const y = a.buf[a.i];
      const v = s + y * a.fb;
      a.buf[a.i] = v;
      a.i = (a.i + 1) % a.d;
      s = y - v * a.fb;
    }
    out[n] = buf[n] * (1 - mix) + s * mix;
  }
  return out;
}

const Lw = reverb(L, 0.30);
const Rw = reverb(R, 0.32);   // slightly different mix widens the tail

// ---------- master ----------

// Fades scaled to the film: in under the cold open, out under the sign-off.
const FADE_IN = 2.5;
const FADE_OUT = 5.0;

let peak = 0;
for (let n = 0; n < N; n++) {
  const t = n / SR;
  const fin = clamp01(t / FADE_IN);
  const fout = clamp01((DUR - t) / FADE_OUT);
  // Cosine fades rather than linear — a linear audio fade is audible as a
  // corner at both ends.
  const env = (0.5 - 0.5 * Math.cos(fin * Math.PI)) * (0.5 - 0.5 * Math.cos(fout * Math.PI));
  Lw[n] *= env;
  Rw[n] *= env;
  peak = Math.max(peak, Math.abs(Lw[n]), Math.abs(Rw[n]));
}

// Normalise to leave headroom, then soft-clip anything that still pokes out.
const target = 0.72;
const gain = peak > 0 ? target / peak : 1;
const softClip = (x) => Math.tanh(x * 1.15) / Math.tanh(1.15);

const pcm = Buffer.alloc(N * 4);
for (let n = 0; n < N; n++) {
  const l = softClip(Lw[n] * gain);
  const r = softClip(Rw[n] * gain);
  pcm.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(l * 32767))), n * 4);
  pcm.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(r * 32767))), n * 4 + 2);
}

// ---------- WAV ----------

const header = Buffer.alloc(44);
header.write('RIFF', 0);
header.writeUInt32LE(36 + pcm.length, 4);
header.write('WAVE', 8);
header.write('fmt ', 12);
header.writeUInt32LE(16, 16);
header.writeUInt16LE(1, 20);          // PCM
header.writeUInt16LE(2, 22);          // stereo
header.writeUInt32LE(SR, 24);
header.writeUInt32LE(SR * 4, 28);     // byte rate
header.writeUInt16LE(4, 32);          // block align
header.writeUInt16LE(16, 34);         // bits
header.write('data', 36);
header.writeUInt32LE(pcm.length, 40);

const out = path.join(HERE, 'music.wav');
writeFileSync(out, Buffer.concat([header, pcm]));
console.log(`✓ ${out}`);
console.log(`  ${DUR}s · ${SR}Hz stereo · pre-normalisation peak ${peak.toFixed(3)} · gain ${gain.toFixed(3)}`);
