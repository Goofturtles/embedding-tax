// Encoder. Turns the captured frames into a 4K master on the 4090's NVENC
// silicon rather than the CPU.
//
//   node encode.mjs                 H.264 4K, widest compatibility
//   node encode.mjs --codec hevc    HEVC, smaller file, same quality
//   node encode.mjs --codec av1     AV1, best compression (Ada supports it)
//   node encode.mjs --no-audio      skip the score
//
// Quality note: NVENC is tuned here with a constant-quality target rather than
// a bitrate cap. Flat near-black fields and slow gradients are exactly what
// bitrate-capped encoders band, and banding is the one artefact that would
// make this look cheap.

import { spawn } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, renameSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRAMES = path.join(HERE, 'frames');
const argv = process.argv.slice(2);
const flag = (n, d) => { const i = argv.indexOf('--' + n); return i === -1 ? d : argv[i + 1]; };
const has = (n) => argv.includes('--' + n);

const meta = existsSync(path.join(HERE, 'render.json'))
  ? JSON.parse(readFileSync(path.join(HERE, 'render.json'), 'utf8'))
  : { fps: 60, first: 0 };

const FPS = Number(flag('fps', meta.fps || 60));
const CODEC = flag('codec', 'h264');
const OUT = flag('out', path.join(HERE, `embedding-tax-${CODEC === 'h264' ? '4k' : CODEC + '-4k'}.mp4`));
const AUDIO = !has('no-audio');

const frames = readdirSync(FRAMES).filter((f) => f.endsWith('.jpg')).sort();
if (frames.length === 0) { console.error('no frames — run render.mjs first'); process.exit(1); }
const startNumber = Number(frames[0].replace(/\D/g, ''));
const durationSec = frames.length / FPS;

const run = (cmd, args) => new Promise((resolve, reject) => {
  console.log('\n$', cmd, args.join(' '), '\n');
  const p = spawn(cmd, args, { stdio: 'inherit' });
  p.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}`))));
  p.on('error', reject);
});

// ---------- score ----------
//
// Synthesised from scratch so there is no licensing question attached to the
// file. Deliberately minimal: a low sustained bed, a fifth above it, and a
// slow filter opening. It sits under the picture and never asks for attention.

// A rendered score from music.mjs takes precedence over the synthesised bed.
// Regenerate it with `node music.mjs`.
const SCORE_WAV = path.join(HERE, 'music.wav');
const HAS_SCORE = existsSync(SCORE_WAV);

function scoreArgs() {
  if (!HAS_SCORE) return synthBedArgs();
  // Seek into the score by the same offset the frames start at. music.mjs
  // composes against absolute film time — sections and bell hits are placed
  // to match specific cuts — so a subrange render (`render.mjs --from`) would
  // otherwise play the score from 0:00 under picture that starts later.
  const offset = startNumber / FPS;
  return offset > 0 ? ['-ss', offset.toFixed(3), '-i', SCORE_WAV] : ['-i', SCORE_WAV];
}

/**
 * Integrated loudness of the score, measured rather than assumed.
 *
 * The level is then applied as a single static gain. Using loudnorm directly
 * on the output would put a *dynamic* processor after the fades music.mjs
 * already drew — the exact mistake the drone graph below was restructured to
 * avoid. A measured constant cannot touch the envelope by construction.
 */
async function measureLoudness(file) {
  return new Promise((resolve) => {
    const p = spawn('ffmpeg', [
      '-hide_banner', '-nostats', '-i', file,
      '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json',
      '-f', 'null', process.platform === 'win32' ? 'NUL' : '/dev/null',
    ], { stdio: ['ignore', 'ignore', 'pipe'] });
    let err = '';
    p.stderr.on('data', (d) => { err += d; });
    p.on('close', () => {
      try {
        const json = JSON.parse(err.slice(err.lastIndexOf('{'), err.lastIndexOf('}') + 1));
        resolve({ lufs: Number(json.input_i), truePeak: Number(json.input_tp) });
      } catch {
        resolve(null);
      }
    });
    p.on('error', () => resolve(null));
  });
}

// Fallback used only when music.wav is absent: the original five-voice drone.
function synthBedArgs() {
  const d = durationSec.toFixed(3);
  // A minor-ninth-ish stack in A. Low fundamentals only — anything in the
  // mid-range would fight the on-screen type for attention.
  const voices = [
    { f: 55.00, g: 0.34 },   // A1
    { f: 82.41, g: 0.24 },   // E2
    { f: 110.00, g: 0.18 },  // A2
    { f: 164.81, g: 0.10 },  // E3
    { f: 220.00, g: 0.055 }, // A3
  ];
  // Gain is chained straight onto each oscillator. Splitting it into a second
  // pass and writing back to the same label is invalid in a filtergraph — a
  // label can only be produced once.
  const src = voices.map((v, i) => `sine=frequency=${v.f}:duration=${d},volume=${v.g}[g${i}]`).join(';');
  const mix = voices.map((_, i) => `[g${i}]`).join('') + `amix=inputs=${voices.length}:normalize=0[bed]`;

  // Shaping chain. Note there is no output label: as a lavfi *input*, the
  // graph's final pad must stay unlabelled or ffmpeg rejects it outright.
  // Order matters here. loudnorm is a *dynamic* normaliser with look-ahead:
  // put it after the fades and it rides the gain back up through both the
  // opening swell and the sign-off fade, flattening the two moments the
  // envelope exists to create. So it normalises the steady-state bed first,
  // and the fades are applied to an already-levelled signal.
  //
  // It replaced a hand-set `volume=0.5`, which looked reasonable and landed at
  // -47.5 dB mean / -39.7 dB peak — inaudible — because the aecho in/out gains
  // and the lowpass each take a bite the arithmetic doesn't make obvious.
  // -20 LUFS is the usual seat for a bed with nothing spoken over it.
  // Both fades scale down together on a short render. Clamping only the
  // out-fade's start still let the two overlap into a dip in the middle for
  // anything under ~7s — a 5s render faded out from 0.8s while it was still
  // fading in. Proportional caps keep them disjoint at any duration, and are
  // a no-op at feature length (65s → 3.5s in, 4.2s out, unchanged).
  const fadeIn = Math.min(3.5, durationSec * 0.25);
  const fadeOut = Math.min(4.2, durationSec * 0.3);

  const shape =
    `[bed]` +
    `highpass=f=32,` +
    `lowpass=f=520,` +
    // A little air, so it reads as a room rather than a synth.
    `aecho=0.45:0.6:530|870:0.22|0.14,` +
    `loudnorm=I=-20:TP=-2.0:LRA=11,` +
    // loudnorm resamples to 192kHz internally; without this the muxer
    // negotiates down to 96kHz and the master ships at an odd rate.
    `aresample=48000,` +
    // Gentle swell in, long fade out under the sign-off.
    `afade=t=in:st=0:d=${fadeIn.toFixed(2)},` +
    `afade=t=out:st=${(durationSec - fadeOut).toFixed(2)}:d=${fadeOut.toFixed(2)}`;

  return ['-f', 'lavfi', '-i', `${src};${mix};${shape}`];
}

// ---------- video ----------

const encoders = {
  h264: ['-c:v', 'h264_nvenc', '-preset', 'p7', '-tune', 'hq', '-rc', 'vbr',
         '-cq', '19', '-b:v', '0', '-maxrate', '90M', '-bufsize', '180M',
         '-profile:v', 'high', '-bf', '3', '-rc-lookahead', '32'],
  hevc: ['-c:v', 'hevc_nvenc', '-preset', 'p7', '-tune', 'hq', '-rc', 'vbr',
         '-cq', '21', '-b:v', '0', '-maxrate', '70M', '-bufsize', '140M',
         '-profile:v', 'main10', '-bf', '3', '-rc-lookahead', '32',
         '-tag:v', 'hvc1'],
  av1:  ['-c:v', 'av1_nvenc', '-preset', 'p7', '-tune', 'hq', '-rc', 'vbr',
         '-cq', '26', '-b:v', '0', '-maxrate', '50M', '-bufsize', '100M',
         '-rc-lookahead', '32'],
};
if (!encoders[CODEC]) { console.error(`unknown codec ${CODEC}`); process.exit(1); }

// Measure the score once, up front, so the level can be applied as a constant.
const TARGET_LUFS = -16;
let scoreGainDb = 0;
if (AUDIO && HAS_SCORE) {
  const measured = await measureLoudness(SCORE_WAV);
  if (measured && Number.isFinite(measured.lufs)) {
    scoreGainDb = TARGET_LUFS - measured.lufs;
    console.log(`score measured at ${measured.lufs.toFixed(1)} LUFS → static ${scoreGainDb >= 0 ? '+' : ''}${scoreGainDb.toFixed(2)} dB`);

    // The limiter after the gain is the one remaining dynamic stage, and a
    // dynamic stage is exactly what the static gain exists to avoid. It stays
    // idle only while the gained true peak clears the threshold. music.mjs
    // pins the WAV's *peak* but not its *integrated loudness*, so a sparser
    // re-composition can move one without the other and quietly re-introduce
    // gain riding over the fades. Say so rather than let it happen silently.
    const gainedPeak = measured.truePeak + scoreGainDb;
    if (Number.isFinite(gainedPeak) && gainedPeak > -0.54) {
      console.warn(
        `  warning: gained true peak is ${gainedPeak.toFixed(2)} dBTP — the limiter will engage and ` +
        'the score’s fades will no longer be preserved exactly. Lower the normalisation ' +
        'target in music.mjs, or raise TARGET_LUFS here.'
      );
    }
  } else {
    console.warn('could not measure score loudness; passing it through at its rendered level (limiter still applies)');
  }
}

const args = [
  '-y',
  '-hide_banner',
  '-stats',
  '-framerate', String(FPS),
  '-start_number', String(startNumber),
  '-i', path.join(FRAMES, 'f%06d.jpg'),
  ...(AUDIO ? scoreArgs() : []),
  // JPEG frames are full-range, so ffmpeg silently picks yuvj420p and tags
  // the file `pc`. Any player that ignores that tag reads 0 as 16 and lifts
  // every black in the film — which, on something this dark, is the worst
  // possible failure. Convert explicitly to limited range and say so.
  '-vf', `scale=in_range=full:out_range=limited,format=${CODEC === 'h264' ? 'yuv420p' : 'p010le'}`,
  ...encoders[CODEC],
  '-color_range', 'tv',
  '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
  // A real score is the content, not a bed, so it sits louder (-16 LUFS) and
  // gets a higher bitrate than the fallback drone did.
  ...(AUDIO
    ? ['-map', '0:v', '-map', '1:a',
       // A static gain, not loudnorm. Dynamic normalisation applied after the
       // fades music.mjs drew would be free to ride them back up; a constant
       // cannot touch the envelope at all.
       ...(HAS_SCORE ? ['-af', `volume=${scoreGainDb.toFixed(2)}dB,alimiter=level_in=1:level_out=1:limit=0.94,aresample=48000`] : []),
       '-c:a', 'aac', '-b:a', HAS_SCORE ? '256k' : '192k',
       '-ar', '48000', '-ac', '2', '-shortest']
    : ['-an']),
  // Front-load the index so it starts playing before it finishes downloading.
  '-movflags', '+faststart',
  OUT,
];

console.log(`encoding ${frames.length} frames · ${FPS}fps · ${durationSec.toFixed(1)}s · ${CODEC} on NVENC`);
await run('ffmpeg', args);

// NVENC writes the colour matrix and primaries but leaves transfer
// characteristics unset, so the file reports `color_transfer=unknown` and a
// strict player is entitled to guess. Stamping it into the bitstream is a
// stream copy — seconds, no re-encode, no quality cost — so it belongs in the
// pipeline rather than in a README step someone will forget.
if (CODEC === 'h264') {
  const tagged = OUT.replace(/\.mp4$/, '.tagged.mp4');
  await run('ffmpeg', [
    '-y', '-hide_banner', '-loglevel', 'error',
    '-i', OUT,
    '-c', 'copy',
    '-bsf:v', 'h264_metadata=transfer_characteristics=1:colour_primaries=1:matrix_coefficients=1:video_full_range_flag=0',
    '-movflags', '+faststart',
    tagged,
  ]);
  renameSync(tagged, OUT);
}

console.log(`\n✓ ${OUT}`);
