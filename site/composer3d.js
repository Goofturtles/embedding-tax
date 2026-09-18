/* composer3d.js: the composer as a glass object standing in the Mostar scene (styles: composer3d.css).
   The pointer stands in for the viewer's eye. As the eye moves, the scene layers slide with it by
   depth through their .par wrappers (sky most, towers least), the composer plane holds still so
   nothing you click drifts, the glass turns a few degrees, its sheen slides and its raised parts
   slide against it.
   It shares the page with the zoom-through camera (home.js) without ever writing what the camera
   writes: the camera owns transform on .cam and .zt-hero, this file owns transform on .par and
   .c3d-tilt. The glass turns and its parts slide only while the camera rests (html.zt-rest): it
   stands straight the frame the camera moves and eases back in (its own weight, ~330ms) when the
   camera settles on the first screen again, pointer moving or not. The scene parallax keeps going
   through the scroll before the camera starts (the output card is read there) and fades to zero
   over the first 12% of a screen of camera travel (the hero passes the camera anyway).
   rAF only while easing or while that fade changes, writes only inside the frame, no layout reads
   per frame (the object's rect is read at most once after a resize, scroll or content change,
   lazily on the next pointer move). Off on coarse pointers and under reduced motion, with live
   listeners; paused while focus is inside (typing), while the settings popover is open, while the
   tab is hidden, and frozen while a button is held. Chromium with GPU compositing gets the real 3D
   tilt and the SVG rim lens; everything else gets the 2D parallax, sheen and light only (software
   compositing drops the backdrop blur under a 3D transform). */
(function () {
  'use strict';

  var doc = document.documentElement;
  var stage = document.querySelector('[data-c3d]');
  var tilt = stage && stage.querySelector('.c3d-tilt');
  var scene = document.querySelector('.scene');
  if (!stage || !tilt || !window.matchMedia) return;

  var MAX_X = 2.5;          // deg of rotateX at a full vertical eye offset
  var MAX_Y = 3.5;          // deg of rotateY at a full horizontal eye offset
  var OVER = 0.5;           // with the pointer over the object the tilt halves: targets never slide out from under it
  var RATE = 9;             // 1/s exponential approach: 95% settled in ~330ms, no overshoot
  var EPS = 0.002;
  var FADE = 0.12;          // the parallax is gone after this fraction of a screen of camera travel (of scroll on the static page)

  /* scene layers at home.js's camera depths: slide = 16px x (1 - 1/z) at a full eye offset, 60% of it
     vertically. `over` names the edges a layer covers, which a slide must never uncover: the sky fills
     the camera box, the towers stand on the bottom edge and meet both sides. */
  var LAYERS = [
    { sel: '.par--sky', z: 30, over: { l: 1, r: 1, t: 1, b: 1 } },
    { sel: '.par--four', z: 5 },
    { sel: '.par--town', z: 2.2 },
    { sel: '.par--bridge', z: 1.45 },
    { sel: '.par--tw', z: 1.25, over: { l: 1, r: 1, b: 1 } }
  ].map(function (L) { L.el = scene && scene.querySelector(L.sel); L.px = 16 * (1 - 1 / L.z); L.t = ''; return L; })
   .filter(function (L) { return !!L.el; });

  var fineMQ = window.matchMedia('(hover: hover) and (pointer: fine)');
  var reduceMQ = window.matchMedia('(prefers-reduced-motion: reduce)');
  var chromium = !!navigator.userAgentData;              // secure-context Chromium; WebKit loses the blur under 3D
  var gpu = null, gpuWait = false;                       // checked once, at idle after motion first turns on (checkGpu)
  var lensMap = document.getElementById('c3d-lens-map'); // <feImage> inside #c3d-lens
  var tabRow = stage.querySelector('.c-tabs');
  var settings = stage.querySelector('.c-settings');

  var on = false, frame = 0, last = 0, measureFrame = 0;
  var vw = window.innerWidth, vh = window.innerHeight, persp = 1200, marginX = 0, marginTop = 0;
  var rect = null, rectDirty = true, tabsH = 40, lensKey = '', gapL = 0, gapR = 0, dropTimer = 0;
  var ex = 0, ey = 0, amp = 1;                           // eased eye (-1..1) and tilt amplitude
  var tx = 0, ty = 0, ta = 1;                            // their targets
  var wg = 1, twg = 1;                                   // the glass's own weight (tilt, sheen, knob, shadow) and its target: 0 while the camera moves
  var px = 0, py = 0, hasPointer = false;
  var focusIn = false, popOpen = false, held = false;
  var sy = window.scrollY || window.pageYOffset || 0;
  var gain = 1, gainDirty = false, tiltT = '', parOk = false;

  var cache = {};
  function put(name, value) {
    if (cache[name] === value) return;
    cache[name] = value;
    stage.style.setProperty(name, value);
  }
  function clamp(v) { return v < -1 ? -1 : v > 1 ? 1 : v; }
  function ss(x) { x = x < 0 ? 0 : x > 1 ? 1 : x; return x * x * (3 - 2 * x); }
  // how far the zoom camera has travelled (home.js etZoom: yAt(0) is where it starts, after the hold that shows a tall hero), else the scroll
  function travel() {
    var z = window.etZoom;
    return z && doc.classList.contains('zt-on') ? Math.max(0, sy - z.yAt(0)) : sy;
  }
  function gainAt() { return 1 - ss(travel() / (FADE * vh)); }
  function hasGpu() {
    if (!chromium) return false;
    try {
      var gl = document.createElement('canvas').getContext('webgl', { failIfMajorPerformanceCaveat: true });
      if (!gl) return false;                             // null = blocklisted GPU or software rendering
      // headless and VM Chromium hand out a software WebGL (SwiftShader) even with the caveat flag (measured)
      var info = gl.getExtension('WEBGL_debug_renderer_info');
      var ren = info ? String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL)) : '';
      var lose = gl.getExtension('WEBGL_lose_context');
      if (lose) lose.loseContext();
      return !/swiftshader|llvmpipe|softpipe|software|basic render/i.test(ren);
    } catch (err) { return false; }
  }
  /* a WebGL context is a main-thread moment, so the check waits for idle time after load: the tilt and the lens only
     matter once the page rests, and until the answer comes the object is the 2D version everyone else gets */
  function checkGpu() {
    gpuWait = false;
    gpu = hasGpu();
    if (!on || !gpu) return;
    doc.classList.add('c3d-3d');
    lensKey = '';
    scheduleMeasure();                                   // builds the lens map and writes the tilt once
  }

  /* ----- discrete measures: viewport, tab gap in the top rim, tab height, rim lens map ----- */
  function scheduleMeasure() { rectDirty = true; if (!measureFrame) measureFrame = requestAnimationFrame(measure); }
  function measure() {
    measureFrame = 0;
    vw = window.innerWidth; vh = window.innerHeight;
    // the camera box (home.css .scene --sw): 16:9, covering the viewport, centred, standing on the bottom edge
    var boxW = Math.max(vw, vh * 16 / 9);
    marginX = (boxW - vw) / 2; marginTop = boxW * 0.5625 - vh;
    persp = Math.round(0.75 * boxW);                     // ~67deg horizontal field of view, a phone main camera, as the photo
    var sel = stage.querySelector('.c-tab[aria-selected="true"]');
    var panel = stage.querySelector('.composer:not([hidden])');
    if (!panel) return;
    if (tabRow) { tabsH = tabRow.offsetHeight; put('--tabs-h', tabsH + 'px'); }
    if (sel && tabRow) {
      // offsets, not rects: a rect read mid-tilt or mid-zoom is skewed. Tab, row and panel share .c3d-tilt as
      // offsetParent unless .c-tabs is positioned, in which case the tab's offset is relative to the row.
      var l = sel.offsetLeft + (sel.offsetParent === tabRow ? tabRow.offsetLeft : 0) - panel.offsetLeft;
      gapL = l; gapR = l + sel.offsetWidth;
      put('--tab-l', gapL + 'px');
      put('--tab-r', gapR + 'px');
    }
    if (on && gpu && lensMap) {
      var w = panel.offsetWidth, h = panel.offsetHeight, key = w + 'x' + h + ':' + gapL + '-' + gapR;
      if (w && h && key !== lensKey) { lensKey = key; buildLens(w, h); }
      doc.classList.toggle('c3d-lens', !!lensKey);    // never before a map exists: an empty feImage would shear the whole card
    }
    if (on) { tiltT = null; gainDirty = true; kick(); }  // a new perspective or overscan margin: write once
  }

  // map for #c3d-lens: R,G = displacement, neutral inside and pointing inward within a 16px band along
  // the rounded rim (radius 22), eased so the bend grows toward the edge; B = the band weight.
  // Under the selected tab (gapL..gapR) the top rim is not an edge but where the tab opens into the card:
  // no band there, with 16px soft ends, like the rim line's gap in composer3d.css. Half resolution, stretched 2x.
  function buildLens(w, h) {
    var S = 2, R = 22, BAND = 16;
    var cw = Math.ceil(w / S), ch = Math.ceil(h / S);
    var canvas = document.createElement('canvas');
    canvas.width = cw; canvas.height = ch;
    var ctx = canvas.getContext('2d');
    if (!ctx) return;
    var img = ctx.createImageData(cw, ch), d = img.data, hw = w / 2, hh = h / 2;
    for (var j = 0; j < ch; j++) {
      for (var i = 0; i < cw; i++) {
        var x = (i + 0.5) * S, y = (j + 0.5) * S;
        var qx = Math.abs(x - hw) - (hw - R), qy = Math.abs(y - hh) - (hh - R);
        var ox = Math.max(qx, 0), oy = Math.max(qy, 0), len = Math.sqrt(ox * ox + oy * oy);
        var sd = len + Math.min(Math.max(qx, qy), 0) - R;              // signed distance, < 0 inside
        var nx, ny;
        if (len > 0) { nx = ox / len; ny = oy / len; } else if (qx > qy) { nx = 1; ny = 0; } else { nx = 0; ny = 1; }
        if (x < hw) nx = -nx;
        if (y < hh) ny = -ny;                                          // outward normal
        var t = Math.min(1, Math.max(0, 1 + sd / BAND));
        if (ny < 0 && gapR > gapL) t *= 1 - Math.min(ss((x - gapL + 8) / 16), ss((gapR + 8 - x) / 16)) * -ny;
        var k = t * t;
        var p = (j * cw + i) * 4;
        d[p] = 128 - 127 * nx * k;                                     // sample inward: never outside the backdrop
        d[p + 1] = 128 - 127 * ny * k;
        d[p + 2] = 255 * t * t * (3 - 2 * t);                          // how much clear rim shows over the frost
        d[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    lensMap.setAttribute('width', w);
    lensMap.setAttribute('height', h);
    lensMap.setAttribute('href', canvas.toDataURL('image/png'));
  }

  /* ----- the camera's rest: html.zt-rest (home.js) while the zoom-through sits settled on its first frame, or no
     zoom at all (the static page). Only at rest does the glass turn, do its parts slide and is its rect read.
     The frame the camera starts to move they stand straight (the zoom itself masks it) and only the scene
     parallax keeps fading with the scroll: a 3D-turned backdrop blur or freshly promoted layers inside the
     scaling hero cost 25-70ms frames at that hand-off (measured). */
  function atRest() { return !doc.classList.contains('zt-on') || doc.classList.contains('zt-rest'); }
  var rest = atRest();
  function onRootClass() {
    var r = atRest();
    if (r === rest) return;
    rest = r;
    if (!on) return;
    if (r) {                                           // back on the first screen: the glass eases back in, pointer moving or not
      rectDirty = true; twg = 1; gainDirty = true; tiltT = null;
      retarget();
      return;
    }
    if (frame) { cancelAnimationFrame(frame); frame = 0; }
    ex = tx; ey = ty; amp = ta; wg = twg = 0;
    write(); settle(true);                             // in this same frame, before the camera's first moving paint
  }

  /* ----- targets (event handlers: arithmetic only, plus one lazy rect read after a change) ----- */
  function paused() { return focusIn || popOpen || document.hidden; }
  function retarget() {
    if (held) return;                                  // frozen between pointerdown and pointerup
    if (rest && (rectDirty || !rect)) { rect = tilt.getBoundingClientRect(); rectDirty = false; }   // never a rect the zoom has scaled
    if (!on || !hasPointer || paused() || !rect) { tx = 0; ty = 0; ta = 1; }
    else {
      var cx = rect.left + rect.width / 2;
      var cy = rect.top + (rect.height + tabsH) / 2;   // the card's centre, below the tabs
      tx = clamp((px - cx) / (vw * 0.5));
      ty = clamp((py - cy) / (vh * 0.5));
      var inside = px >= rect.left && px <= rect.right && py >= rect.top && py <= rect.bottom;
      ta = inside ? OVER : 1;
    }
    if (tx === 0 && ty === 0) wg = twg;                // nothing to turn: the weight needs no frames
    if (!rest || gain === 0) {                         // the camera is moving or the hero is gone: nothing eases, it resumes from here
      ex = tx; ey = ty; amp = ta;
      if (gain > 0) { gainDirty = true; kick(); }
      return;
    }
    kick();
  }
  function easing() { return Math.abs(tx - ex) >= EPS || Math.abs(ty - ey) >= EPS || Math.abs(ta - amp) >= EPS || Math.abs(twg - wg) >= EPS; }
  function kick() {
    var moving = easing();
    if (!moving && !gainDirty) { if (!frame) settle(); return; }
    if (moving) { clearTimeout(dropTimer); dropTimer = 0; live(true); stage.classList.remove('is-still'); }
    if (!frame) { last = 0; frame = requestAnimationFrame(step); }
  }

  /* ----- the loop: writes only ----- */
  function step(now) {
    frame = 0;
    var dt = last ? Math.min(0.05, (now - last) / 1000) : 1 / 60;
    last = now;
    var k = 1 - Math.exp(-RATE * dt);
    ex += (tx - ex) * k; ey += (ty - ey) * k; amp += (ta - amp) * k; wg += (twg - wg) * k;
    var done = !easing();
    if (done) { ex = tx; ey = ty; amp = ta; wg = twg; }
    write();
    if (done) settle();
    else frame = requestAnimationFrame(step);
  }
  function write() {
    gainDirty = false;
    var gx = ex * gain, gy = ey * gain, moved = gx !== 0 || gy !== 0;
    var w = rest ? wg : 0;                             // the object itself moves only while the camera rests, by its own weight
    var glass = moved && w > 0;
    stage.classList.toggle('is-tilted', glass);
    put('--mx', (glass ? gx * w : 0).toFixed(3));
    put('--my', (glass ? gy * w : 0).toFixed(3));

    var t = '';
    if (glass && gpu) {
      var rx = gy * w * MAX_X * amp, ry = -gx * w * MAX_Y * amp;
      if (Math.abs(rx) >= 0.005 || Math.abs(ry) >= 0.005) t = 'perspective(' + persp + 'px) rotateX(' + rx.toFixed(2) + 'deg) rotateY(' + ry.toFixed(2) + 'deg)';
    }
    if (t !== tiltT) { tiltT = t; tilt.style.transform = t; }

    for (var i = 0; i < LAYERS.length; i++) {
      var L = LAYERS[i], s = '';
      if (moved && parOk) {
        var dx = gx * L.px, dy = gy * L.px * 0.6, need = 0, o = L.over, sc = '';
        if (o) {
          // scaled about the viewport centre, each side grows by (scale - 1) x half the viewport
          if ((o.l || o.r) && Math.abs(dx) > marginX) need = Math.max(need, (Math.abs(dx) - marginX) / (vw / 2));
          if (o.b && dy < 0) need = Math.max(need, -dy / (vh / 2));                        // the box stands on the bottom edge
          if (o.t && dy > marginTop) need = Math.max(need, (dy - marginTop) / (vh / 2));
          // a hair of overscan against sub-pixel seams, ramped in with the eye so no frame steps it on (it used to switch on at need > 0)
          sc = ' scale(' + (1 + 0.002 * Math.min(1, 10 * Math.max(Math.abs(gx), Math.abs(gy))) + need).toFixed(4) + ')';
        }
        s = 'translate(' + dx.toFixed(2) + 'px, ' + dy.toFixed(2) + 'px)' + sc;
      }
      if (s !== L.t) { L.t = s; L.el.style.transform = s; }
    }
  }
  /* layers only while easing, and easing only at rest: dropped when settled so text re-rasters crisp, and never
     added or held while the zoom camera moves (measured: layers kept while the pointer was in the window made
     the first scroll frame 21ms at the median against 13ms without) */
  function live(want) {
    stage.classList.toggle('is-live', want);
    if (scene) scene.classList.toggle('is-par', want && parOk);
  }
  /* settled, the layers drop (and the lens comes back) 350ms later, so a short pause between pointer moves does not
     repeat the promote-and-drop (measured: a 13-17ms frame at each start and stop); the camera starting drops them at once */
  function drop() {
    dropTimer = 0;
    if (frame || easing()) return;
    live(false);
    stage.classList.toggle('is-still', sy <= 4 && !popOpen);   // the rim lens only at rest (CSS also waits for html.zt-rest)
  }
  function settle(now) {
    clearTimeout(dropTimer); dropTimer = 0;
    if (now || !stage.classList.contains('is-live')) { live(false); stage.classList.toggle('is-still', sy <= 4 && !popOpen); return; }
    dropTimer = setTimeout(drop, 350);
  }
  function snapHome() {
    if (frame) { cancelAnimationFrame(frame); frame = 0; }
    ex = ey = tx = ty = 0; amp = ta = 1; wg = twg = 1;
    write(); settle(true);
  }

  /* ----- events ----- */
  function onMove(e) {
    if (e.pointerType === 'touch') return;
    px = e.clientX; py = e.clientY; hasPointer = true;
    retarget();
  }
  function onOut(e) { if (!e.relatedTarget) { hasPointer = false; retarget(); } }   // left the window
  function onDown() { held = true; }
  function onUp() { if (!held) return; held = false; retarget(); }
  function onFocusIn() { focusIn = true; retarget(); }
  function onFocusOut(e) { focusIn = !!(e.relatedTarget && stage.contains(e.relatedTarget)); retarget(); }
  function onToggle() { popOpen = !!(settings && settings.open); retarget(); }
  function onScroll() {
    rectDirty = true;
    sy = window.scrollY || window.pageYOffset || 0;
    if (!on) return;
    var g = gainAt();
    if (g === gain) { if (!frame && sy <= 8) settle(); return; }   // deep in the page there is nothing to do
    var wasZero = gain === 0;
    gain = g; gainDirty = true;
    if (wasZero) retarget();                           // back near the top the pointer picks up again
    else kick();
  }
  function onResize() { scheduleMeasure(); }
  function onVisibility() { if (document.hidden) { hasPointer = false; snapHome(); } }

  function listen(add) {
    var m = add ? 'addEventListener' : 'removeEventListener';
    window[m]('pointermove', onMove, { passive: true });
    window[m]('pointerout', onOut, { passive: true });
    window[m]('pointerdown', onDown, true);
    window[m]('pointerup', onUp, true);
    window[m]('pointercancel', onUp, true);
    stage[m]('focusin', onFocusIn);
    stage[m]('focusout', onFocusOut);
    if (settings) settings[m]('toggle', onToggle);
    document[m]('visibilitychange', onVisibility);
  }

  function sync() {
    var want = fineMQ.matches && !reduceMQ.matches;
    if (want === on) return;
    on = want;
    doc.classList.toggle('c3d-on', on);
    if (on && gpu === null && !gpuWait) {
      gpuWait = true;
      if (window.requestIdleCallback) window.requestIdleCallback(checkGpu, { timeout: 2000 });
      else setTimeout(checkGpu, 200);
    }
    doc.classList.toggle('c3d-3d', !!(on && gpu));
    doc.classList.remove('c3d-lens');                  // measure() adds it back once the map is built
    listen(on);
    if (on) {
      // a transformed wrapper without its composer3d.css box would become the photos' containing block and drop them
      parOk = LAYERS.length > 0 && getComputedStyle(LAYERS[0].el).position === 'absolute';
      focusIn = stage.contains(document.activeElement);
      popOpen = !!(settings && settings.open);
      gain = gainAt();
      lensKey = '';
      scheduleMeasure();
      settle();
    } else {
      hasPointer = false; held = false;
      snapHome();
      stage.classList.remove('is-still', 'is-tilted');
    }
  }
  function onMedia() { sync(); }

  /* ----- wiring that runs for everyone (the rim gap is static design, not motion) ----- */
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onResize);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(scheduleMeasure);
  if (window.ResizeObserver) new ResizeObserver(scheduleMeasure).observe(tilt);
  if (window.MutationObserver) {
    new MutationObserver(onRootClass).observe(doc, { attributes: true, attributeFilter: ['class'] });
    var mo = new MutationObserver(scheduleMeasure);
    Array.prototype.forEach.call(stage.querySelectorAll('.c-tab, .composer'), function (el) {
      mo.observe(el, { attributes: true, attributeFilter: ['aria-selected', 'hidden'] });
    });
  }
  [fineMQ, reduceMQ].forEach(function (mq) {
    if (mq.addEventListener) mq.addEventListener('change', onMedia);
    else if (mq.addListener) mq.addListener(onMedia);
  });

  // QA hook
  window.c3d = {
    state: function () {
      return { on: on, chromium: chromium, gpu: gpu, ex: ex, ey: ey, amp: amp, tx: tx, ty: ty, wg: wg, gain: gain, frame: !!frame,
               focusIn: focusIn, popOpen: popOpen, held: held, sy: sy, cls: stage.className, tilt: tiltT,
               par: LAYERS.map(function (L) { return L.t; }) };
    }
  };

  sync();
  scheduleMeasure();
})();
