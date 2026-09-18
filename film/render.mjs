// Frame renderer.
//
// Drives the stage's deterministic `__seek(t)` and captures one screenshot per
// frame at deviceScaleFactor 2. The stage is authored at 1920×1080 CSS pixels,
// so every frame rasterises natively at 3840×2160 — real 4K, not an upscale.
//
//   node render.mjs                 full film
//   node render.mjs --fps 30        cheaper pass while iterating
//   node render.mjs --from 20 --to 26   just one scene
//   node render.mjs --probe 3.5,12,17   single stills for eyeballing

import { chromium } from 'playwright';
import { mkdirSync, rmSync, existsSync, writeFileSync, readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, 'frames');

// The stage is an ES module, and modules are blocked over file:// by CORS, so
// the renderer serves its own directory. Self-contained: no external server to
// start, no port to keep track of, nothing left running afterwards.
const MIME = {
  '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript',
  '.mjs': 'text/javascript', '.json': 'application/json',
  '.ttf': 'font/ttf', '.woff2': 'font/woff2', '.woff': 'font/woff',
};
function serveStage() {
  return new Promise((resolve) => {
    const server = createServer((req, res) => {
      const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'scene/index.html';
      const file = path.join(HERE, rel);
      if (!file.startsWith(HERE)) { res.writeHead(403).end(); return; }
      try {
        const body = readFileSync(file);
        res.writeHead(200, {
          'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
          'Cache-Control': 'no-store',
        });
        res.end(body);
      } catch {
        res.writeHead(404).end('not found');
      }
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

const argv = process.argv.slice(2);
const flag = (name, def) => {
  const i = argv.indexOf('--' + name);
  return i === -1 ? def : argv[i + 1];
};
const has = (name) => argv.includes('--' + name);

const FPS = Number(flag('fps', 60));
const FROM = Number(flag('from', 0));
const TO = flag('to', null);
const PROBE = flag('probe', null);
const QUALITY = Number(flag('quality', 94));

const run = async () => {
  const { server, port } = await serveStage();
  const STAGE = `http://127.0.0.1:${port}/scene/index.html?static=1`;

  const browser = await chromium.launch({
    args: [
      // Deterministic rasterisation: no GPU variance between frames, and the
      // font stack resolves identically every run.
      '--force-color-profile=srgb',
      '--font-render-hinting=none',
      '--disable-lcd-text',
      '--hide-scrollbars',
      '--disable-frame-rate-limit',
    ],
  });
  const page = await browser.newPage({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 2,
    colorScheme: 'dark',
  });

  page.on('console', (m) => { if (m.type() === 'error') console.error('  [page]', m.text()); });
  page.on('pageerror', (e) => console.error('  [page error]', e.message));

  await page.goto(STAGE, { waitUntil: 'load' });
  await page.evaluate(() => { window.__manual = true; });
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 20000 });
  await page.evaluate((f) => { window.__fps = f; }, FPS);

  // The bundled face must actually be the one that rendered. Silently falling
  // back to a system font would change every glyph in the film.
  const fontOk = await page.evaluate(() => document.fonts.check('700 76px InterFilm'));
  if (!fontOk) throw new Error('bundled InterFilm did not load — check scene/fonts/');

  const duration = await page.evaluate(() => window.__duration);

  // Probe mode: a few stills, named by timestamp, for looking at.
  if (PROBE) {
    const dir = path.join(HERE, 'probe');
    mkdirSync(dir, { recursive: true });
    for (const raw of PROBE.split(',')) {
      const t = Number(raw);
      await page.evaluate((tt) => window.__seek(tt), t);
      await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
      const file = path.join(dir, `t${t.toFixed(2).replace('.', '_')}.png`);
      await page.screenshot({ path: file, type: 'png' });
      console.log('probe →', file);
    }
    await browser.close();
    server.close();
    return;
  }

  const end = TO === null ? duration : Number(TO);
  const first = Math.round(FROM * FPS);
  const last = Math.round(end * FPS);
  const total = last - first;

  // Preserve by default. The documented iteration flags render a subrange, and
  // wiping on every run meant `--to 22` silently destroyed a completed 3,900
  // frame render. Frames are overwritten by name, so keeping them is safe;
  // pass --clean when the frame count or fps actually changes.
  if (has('clean') && existsSync(OUT)) rmSync(OUT, { recursive: true, force: true });
  mkdirSync(OUT, { recursive: true });

  console.log(`rendering ${total} frames · ${FPS}fps · ${FROM}s→${end.toFixed(1)}s · 3840×2160`);
  const t0 = Date.now();

  for (let f = first; f < last; f++) {
    const t = f / FPS;
    await page.evaluate((tt) => window.__seek(tt), t);
    // Two rAFs guarantee the compositor has applied this frame's transforms
    // before the capture. One is not always enough.
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    await page.screenshot({
      path: path.join(OUT, `f${String(f).padStart(6, '0')}.jpg`),
      type: 'jpeg',
      quality: QUALITY,
    });

    const done = f - first + 1;
    if (done % 30 === 0 || done === total) {
      const el = (Date.now() - t0) / 1000;
      const rate = done / el;
      const eta = (total - done) / rate;
      process.stdout.write(
        `\r  ${done}/${total}  ${(done / total * 100).toFixed(1)}%  ` +
        `${rate.toFixed(1)} fps  eta ${Math.round(eta)}s     `
      );
    }
  }

  process.stdout.write('\n');
  writeFileSync(path.join(HERE, 'render.json'), JSON.stringify({
    fps: FPS, from: FROM, to: end, first, last, total,
    width: 3840, height: 2160,
    renderedAt: new Date().toISOString(),
    seconds: (Date.now() - t0) / 1000,
  }, null, 2));
  console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  await browser.close();
  server.close();
};

run().catch((e) => { console.error(e); process.exit(1); });
