/* home.js: everything on the home page that is not the model.
   1. The composer tabs (Write | Budget, WAI-ARIA tabs with arrow keys), and the settings panel: the
      side it opens on, and how it closes (Escape, a press outside, focus moving on out of it).
   2. The budget, pure arithmetic on the cap: the same formula tools/param_count.py verifies, so
      16,384 gives 13 layers and 49,296,896 parameters, and GPT-2's 50,257 gives 7. It fills the
      Budget tab and the budget chapter.
   3. The zoom-through: the first screen is a picture in depth and scrolling moves one camera
      forward through it. Every layer scales about a point inside the arch opening (nearer layers
      faster), the hero UI swells and fades past the camera, the stone of the bridge leaves every
      edge, and three glass chapters grow out of the valley, hold, and pass the camera in turn; the
      tower pair comes back once as a gate between the last two.
      rAF + scroll, transform and opacity only, no layout reads while ticking, idle when settled.
   The model itself is playground.js. */
(function () {
  'use strict';

  var doc = document.documentElement;
  var scheduleMeasure = function () {};   // replaced by the zoom once it runs

  /* ---------- 1. tabs ---------- */
  var hero = document.querySelector('.hero');
  var tabs = [document.getElementById('tabWrite'), document.getElementById('tabBudget')];
  var panels = [document.getElementById('panelWrite'), document.getElementById('panelBudget')];
  var tries = [document.getElementById('tryWrite'), document.getElementById('tryBudget')];
  var hasTabs = !!(hero && tabs[0] && tabs[1] && panels[0] && panels[1]);

  function select(i, focus) {
    if (!hasTabs) return;
    for (var k = 0; k < 2; k++) {
      var on = k === i;
      tabs[k].setAttribute('aria-selected', String(on));
      tabs[k].tabIndex = on ? 0 : -1;
      panels[k].hidden = !on;
      if (tries[k]) tries[k].hidden = !on;
    }
    hero.setAttribute('data-tab', i ? 'budget' : 'write');
    if (focus) tabs[i].focus();
    scheduleMeasure();
  }
  if (hasTabs) {
    tabs.forEach(function (tab, i) {
      tab.addEventListener('click', function () { select(i, false); });
      tab.addEventListener('keydown', function (e) {
        var next = { ArrowRight: 1 - i, ArrowLeft: 1 - i, Home: 0, End: 1 }[e.key];
        if (next === undefined) return;
        e.preventDefault();
        select(next, true);
      });
    });
  }

  /* ---------- 1b. the settings panel: given a side once, as it opens ----------
     Above the knob when the whole panel fits under the header band, else below when it fits there, else
     the roomier side; its max-height is always the room on that side, so a guards list or a lock note
     opening later scrolls inside it instead of running off the screen. Layout is read on the toggle and
     on resize while open, never per frame. Under 641px the panel sits in the composer's flow (home.css). */
  var settings = document.querySelector('.c-settings');
  var pop = settings && settings.querySelector('.c-pop');
  var knob = settings && settings.querySelector('summary');
  function placeSettings() {
    if (!settings.open || getComputedStyle(pop).position !== 'absolute') return;
    var k = knob.getBoundingClientRect();
    var band = parseFloat(getComputedStyle(doc).getPropertyValue('--header-band')) || 72;
    var above = Math.floor(k.top - 24 - band), below = Math.floor(window.innerHeight - k.bottom - 24);   // 12 to the knob, 12 to the edge
    var h = pop.scrollHeight;
    var down = h > above && (h <= below || below > above);
    settings.setAttribute('data-place', down ? 'down' : 'up');
    // never taller than the room itself: on a short landscape screen 120px would run past .zt-view's crop
    pop.style.setProperty('--pop-max', Math.max(0, down ? below : above) + 'px');
  }
  if (settings && pop && knob) {
    settings.addEventListener('toggle', placeSettings);
    window.addEventListener('resize', placeSettings);

    /* Dismissal, as any disclosure panel that floats over content: Escape closes it (focus back on the knob when
       it was inside), a press anywhere outside closes it, and so does focus moving on to anything outside, so a
       Shift+Tab to the prompt never lands under the panel (WCAG 2.2 2.4.11). A press closes it only once the press
       is over, and only while the panel floats: below 641px it sits in the composer's flow and covers nothing, and
       closing it under a finger would move the tapped button away before its click lands. A cancelled press is a
       scroll swipe, never a close. Escape is handled in the capture phase and marked handled, so playground.js
       does not also read it as "stop the run". */
    var pressing = false, pressedInside = false, closeAfterPress = false;
    var floating = function () { return getComputedStyle(pop).position === 'absolute'; };   // read once per press, never per frame
    var closeSettings = function (refocus) {
      if (!settings.open) return;
      settings.open = false;
      if (refocus) { try { knob.focus({ preventScroll: true }); } catch (err) { knob.focus(); } }
    };
    document.addEventListener('keydown', function (e) {
      if (e.key !== 'Escape' || !settings.open || e.defaultPrevented) return;
      e.preventDefault();
      closeSettings(settings.contains(document.activeElement));
    }, true);
    document.addEventListener('pointerdown', function (e) {
      pressing = true;
      pressedInside = settings.contains(e.target);
      if (settings.open && !pressedInside && floating()) closeAfterPress = true;
    }, true);
    var pressEnd = function () {
      pressing = false;
      if (!closeAfterPress) return;
      closeAfterPress = false;
      setTimeout(function () { closeSettings(false); }, 0);   // after this press's click has reached its target
    };
    document.addEventListener('pointerup', pressEnd, true);
    document.addEventListener('pointercancel', function () { pressing = false; closeAfterPress = false; }, true);
    window.addEventListener('blur', function () { pressing = false; closeAfterPress = false; });   // a press whose pointerup never arrives
    settings.addEventListener('focusout', function (e) {
      var to = e.relatedTarget;
      if (!settings.open || !to || settings.contains(to)) return;   // no target: the window lost focus, or a press on plain panel text
      if (pressing) { if (!pressedInside && floating()) closeAfterPress = true; return; }
      closeSettings(false);
    });
  }

  /* ---------- 2. the budget: width 512, tied embeddings, RMSNorm weight-only, no biases ---------- */
  var CAP = 50000000, D = 512, PER_LAYER = 12 * D * D + 2 * D;   // 3,146,752
  var VOCABS = [1024, 2048, 4096, 8192, 16384, 32000, 50257, 65536];
  var fmt = function (n) { return n.toLocaleString('en-US'); };

  function budget(v) {
    var emb = v * D;                                                      // one tied matrix
    var layers = Math.max(0, Math.floor((CAP - emb - D) / PER_LAYER));   // D: the final norm
    return { emb: emb, layers: layers, layerParams: layers * PER_LAYER };
  }

  var range = document.getElementById('bVocab');
  if (range) {
    var $ = function (id) { return document.getElementById(id); };
    var chips = document.querySelectorAll('[data-vocab]');
    var render = function () {
      var idx = Number(range.value);
      var v = VOCABS[idx];
      var b = budget(v);
      $('bVocabOut').textContent = fmt(v) + ' tokens';
      range.setAttribute('aria-valuetext', fmt(v) + ' tokens, ' + b.layers + (b.layers === 1 ? ' layer fits' : ' layers fit'));
      range.style.setProperty('--pct', (100 * idx / (VOCABS.length - 1)) + '%');
      $('bLayers').textContent = String(b.layers);
      $('bLayersUnit').textContent = b.layers === 1 ? 'layer fits' : 'layers fit';
      $('bEmb').textContent = (100 * b.emb / CAP).toFixed(1) + '%';
      $('bBarEmb').parentNode.style.setProperty('--emb', Math.min(1, b.emb / CAP).toFixed(4));
      $('bBarEmb').parentNode.style.setProperty('--lay', (b.layerParams / CAP).toFixed(4));
      for (var i = 0; i < chips.length; i++) chips[i].setAttribute('aria-pressed', String(Number(chips[i].getAttribute('data-vocab')) === idx));
    };
    range.addEventListener('input', render);
    for (var c = 0; c < chips.length; c++) {
      chips[c].addEventListener('click', function () {
        range.value = this.getAttribute('data-vocab');
        render();
      });
    }
    render();
  }

  // the budget chapter: two vocabularies, the same arithmetic, written over the shipped values
  var budRows = document.querySelectorAll('[data-bud]');
  for (var r = 0; r < budRows.length; r++) {
    var bv = budget(Number(budRows[r].getAttribute('data-bud')));
    var lEl = budRows[r].querySelector('[data-bud-l]'), eEl = budRows[r].querySelector('[data-bud-e]'), bar = budRows[r].querySelector('.bud-bar');
    if (lEl) lEl.textContent = String(bv.layers);
    if (eEl) eEl.textContent = (100 * bv.emb / CAP).toFixed(1) + '%';
    if (bar) {
      bar.style.setProperty('--emb', Math.min(1, bv.emb / CAP).toFixed(4));
      bar.style.setProperty('--lay', (bv.layerParams / CAP).toFixed(4));
    }
  }

  /* ---------- 3. the zoom-through ---------- */
  function clamp(v, lo, hi) { lo = lo === undefined ? 0 : lo; hi = hi === undefined ? 1 : hi; return v < lo ? lo : v > hi ? hi : v; }
  function ss(x) { return x * x * (3 - 2 * x); }
  function seg(v, a, b) { return clamp((v - a) / (b - a)); }

  // write discipline: each element keeps the last string per property and class, identical writes are skipped
  function setVar(el, name, value) {
    var c = el.__etv || (el.__etv = {});
    if (c[name] === value) return;
    c[name] = value;
    el.style.setProperty(name, value);
  }
  function setClass(el, cls, on) {
    var c = el.__etc || (el.__etc = {});
    if (c[cls] === on) return;
    c[cls] = on;
    el.classList.toggle(cls, on);
  }
  function clearWrites(el) {
    var name;
    if (el.__etv) { for (name in el.__etv) el.style.removeProperty(name); el.__etv = null; }
    if (el.__etc) { for (name in el.__etc) el.classList.remove(name); el.__etc = null; }
  }
  function jump(top) {
    try { window.scrollTo({ top: top, behavior: 'instant' }); }
    catch (err) { window.scrollTo(0, top); }
  }

  /* The camera. u is the camera distance in svh (one small-viewport height of scroll = 1). A layer
     at depth z scales z / (z - c), so nearer layers grow faster. The camera never quite stops: while a
     chapter holds it creeps forward .05, so the scene keeps breathing under the reader's scroll. */
  var L_SVH = 5.60;                         // the whole track: arch 1.60, holds .65 / .65 / .65, moves 1.05 / 1.00
  function camAt(u) {
    if (u <= 0) return 0;
    if (u < 1.60) { var t = u / 1.60; return 1.25 * (1 - (1 - t) * (1 - t)); }   // ease-out: instant response, soft landing
    if (u <= 2.25) return 1.25 + 0.05 * seg(u, 1.60, 2.25);                   // hold #device
    if (u < 3.30) return 1.30 + 0.55 * ss(seg(u, 2.25, 3.30));
    if (u <= 3.95) return 1.85 + 0.05 * seg(u, 3.30, 3.95);                   // hold #budget
    if (u < 4.95) return 1.90 + 0.50 * ss(seg(u, 3.95, 4.95));
    return 2.40 + 0.05 * seg(u, 4.95, 5.60);                                  // hold #pages, then release
  }
  function depth(z, c, cap) { return c < z ? Math.min(cap, z / (z - c)) : cap; }
  function heroFade(hs) { return 1 - clamp((hs - 1) / 0.42); }       // the hero is gone by 1.42x, before it turns into a see-through slab
  function heroTopFade(hs) { return 1 - clamp((hs - 1) / 0.22); }    // the pill and headline leave before they reach the header
  /* a chapter comes from depth (.30 -> 1), holds (drifting 1.5%), then passes the camera (gone by 1.15x).
     Its glass and its text fade in together from 40% size (the text a little faster, as the two
     opacities multiply), and leave on one curve, so no blank frosted slab ever arrives ahead of its
     words or outlives them on the way out (the #budget exit is the tower-gate frame).
     On touch the text has no fade of its own and so is exactly as opaque as its glass: the panels span a phone's
     width, and a glass at .53 over text at .18 read as an empty frosted tile arriving (iPhone 15, u 1.175). */
  var touchMQ = window.matchMedia ? window.matchMedia('(hover: none) and (pointer: coarse)') : null;
  var textWithGlass = !!(touchMQ && touchMQ.matches);
  function chapter(u, w) {                  // w = [in0, in1, out0, out1] in svh; out may be null
    var t = ss(seg(u, w[0], w[1]));
    var s = 1 / (1 + 2.3333 * (1 - t));
    var o = ss(seg(s, 0.40, 0.56));    // solid by 56% of its travel, not 72%: a half-there card
    var ci = textWithGlass ? 1 : ss(seg(s, 0.44, 0.60));   // over a photograph reads as a glitch, not as distance
    if (w[2] != null) {
      s *= 1 + 0.015 * seg(u, w[1], w[2]);
      var k = seg(u, w[2], w[3]);
      var so = 1 / (1 - 0.75 * k * k);      // accelerating toward the camera
      var out = 1 - ss(seg(so, 1, 1.15));
      s *= so; o *= out;
      if (!textWithGlass) ci *= out;
    }
    return [s, o, ci];
  }
  // #device already waits inside the arch opening while the stone still frames it
  var CHAP = { device: [0.70, 1.60, 2.25, 2.75], budget: [2.87, 3.30, 3.95, 4.45], pages: [4.45, 4.95, null, null] };
  var HOLD_AT = { 'try': 0, device: 1.925, budget: 3.625, pages: 5.275 };

  function initZoom() {
    var zt = document.getElementById('zt');
    var probe = zt && zt.querySelector('.zt-probe');
    var view = zt && zt.querySelector('.zt-view');
    var heroWrap = zt && zt.querySelector('.zt-hero');
    var scene = document.querySelector('.scene');
    if (!zt || !probe || !view || !heroWrap || !scene) return;

    var camSky = scene.querySelector('.cam--sky'), camFour = scene.querySelector('.cam--four'), camTown = scene.querySelector('.cam--town');
    var camBridge = scene.querySelector('.cam--bridge');
    var camSplits = scene.querySelectorAll('.cam--split'), camDoors = scene.querySelectorAll('.cam--door');
    var scrim = scene.querySelector('.scene-scrim');
    if (!camSky || !camFour || !camTown || !camBridge) return;
    var chaps = [];
    ['device', 'budget', 'pages'].forEach(function (id) {
      var el = document.getElementById(id);
      if (el && el.classList.contains('zt-chap')) chaps.push({ id: id, el: el, w: CHAP[id], left: 0, top: 0, wd: 0, h: 0, dTop: 0, dBot: 0, dRight: 0, s: .3, o: 0 });
    });
    if (chaps.length !== 3) return;
    var composerWrap = heroWrap.querySelector('.composer-wrap');
    var askPill = document.querySelector('.ask');
    var foot = document.querySelector('.home-foot');
    var promptEl = document.getElementById('hPrompt');
    var reduceMQ = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };
    var canSticky = !!(window.CSS && CSS.supports && CSS.supports('position', 'sticky'));

    var zoom = false, noFit = {};
    var svh = window.innerHeight, vw = doc.clientWidth, heroH = 0, viewH = svh, pre = 0, trackTop = 0, Vx = 0, Vy = 0, composerBottom = 0;
    var footTop = 0, footBot = 0, footRight = 0;   // document px: the ask pill stays off the footer (and the stacked chapters)
    var y = window.scrollY || window.pageYOffset || 0, dT = 0, d = 0, u = 0, c = 0, ho = 1;
    var mTrackTop = 0, mPre = 0, mSvh = 0, mZoom = false;   // the layout the current scroll position was last read under
    var snapNext = true, last = 0, settled = true;
    var measureFrame = 0, tickFrame = 0, suppressFocus = false, measuring = 0;
    var pendingHash = hashId(location.hash);
    var inputAt = -1e9, movedAt = -1e9;

    function hashId(h) {
      var id = '';
      try { id = decodeURIComponent((h || '').replace(/^#/, '')); } catch (err) { id = ''; }
      return Object.prototype.hasOwnProperty.call(HOLD_AT, id) ? id : null;
    }

    /* ----- scene images: the town and the mountains wait for the towers so their edges never flash ----- */
    var late = scene.querySelectorAll('img.sc-late');
    var lateState = [];
    function settleImages() {
      var splitsDone = true, allDone = true;
      for (var i = 0; i < lateState.length; i++) {
        if (lateState[i].split && !lateState[i].done) splitsDone = false;
        if (!lateState[i].done) allDone = false;
      }
      for (var k = 0; k < lateState.length; k++) {
        var s = lateState[k];
        if (s.split ? s.done : (splitsDone && allDone)) s.img.classList.remove('is-wait');
      }
    }
    Array.prototype.forEach.call(late, function (img) {
      // settled = decoded, failed (errors count as ready) or the 1px GIF a portrait phone gets
      var s = { img: img, split: img.classList.contains('sc-split'), done: img.complete || /^data:/.test(img.currentSrc || '') };
      lateState.push(s);
      if (!s.done) img.classList.add('is-wait');
      var mark = function () { s.done = true; settleImages(); };
      img.addEventListener('load', mark);
      img.addEventListener('error', mark);
    });
    settleImages();

    /* ----- measure: the only layout reads; coalesced to one frame ----- */
    scheduleMeasure = function () { if (!measureFrame) measureFrame = requestAnimationFrame(measure); };

    function resetZoom() {
      clearWrites(scene); clearWrites(zt); clearWrites(heroWrap); clearWrites(doc);
      [camSky, camFour, camTown, camBridge].forEach(clearWrites);
      if (scrim) clearWrites(scrim);
      for (var i = 0; i < camSplits.length; i++) clearWrites(camSplits[i]);
      for (var j = 0; j < camDoors.length; j++) clearWrites(camDoors[j]);
      chaps.forEach(function (ch) { clearWrites(ch.el); ch.s = .3; ch.o = 0; });
      snapNext = true; mZoom = false; ho = 1;
    }
    function setMode(on) {
      zoom = on;
      doc.classList.toggle('zt-on', on);
      doc.classList.remove('zt-pending');
      if (!on) resetZoom();
      snapNext = true;
    }

    function measure() {
      measureFrame = 0;
      var prevSvh = svh;
      svh = probe.offsetHeight || window.innerHeight;
      vw = doc.clientWidth;
      var key = vw + 'x' + svh;
      var want = !reduceMQ.matches && canSticky && svh >= 560 && !noFit[key];
      if (want !== zoom) {
        setMode(want);
        // the layout just changed: read it again now, so no frame paints the new mode unmeasured
        if (measuring < 2) { measuring++; try { measure(); } finally { measuring--; } }
        else scheduleMeasure();
        return;
      }
      doc.classList.remove('zt-pending');

      if (zoom) {
        // every chapter must fit between the header band and the bottom margin, or the page stays static
        var band = parseFloat(getComputedStyle(doc).getPropertyValue('--header-band')) || 72;
        var room = svh - band - 12 - 24;
        for (var f = 0; f < chaps.length; f++) {
          if (chaps[f].el.offsetHeight > room) {
            noFit[key] = true;
            setMode(false);
            if (measuring < 2) { measuring++; try { measure(); } finally { measuring--; } }
            else scheduleMeasure();
            return;
          }
        }
      }

      var sy = window.scrollY || window.pageYOffset || 0;
      y = sy;
      // the camera distance under the layout we last measured (the scroll position itself never moves with layout: overflow-anchor none)
      var dist = mZoom && zoom ? sy - mTrackTop - mPre : 0, oldSvh = mSvh;
      heroH = heroWrap.offsetHeight;
      viewH = Math.max(svh, heroH);
      pre = viewH - svh;
      trackTop = zt.getBoundingClientRect().top + sy;
      if (composerWrap) composerBottom = composerWrap.getBoundingClientRect().bottom + sy;
      if (foot) { var fr = foot.getBoundingClientRect(); footTop = fr.top + sy; footBot = fr.bottom + sy; footRight = fr.right; }
      if (!zoom) chaps.forEach(function (ch) { var cr = ch.el.getBoundingClientRect(); ch.dTop = cr.top + sy; ch.dBot = cr.bottom + sy; ch.dRight = cr.right; });

      if (zoom) {
        if (prevSvh && prevSvh !== svh) d *= svh / prevSvh;
        setVar(zt, '--svh-px', svh + 'px');
        setVar(zt, '--view-h', viewH + 'px');
        setVar(zt, '--view-top', (svh - viewH) + 'px');
        setVar(zt, '--zt-len', (L_SVH * svh).toFixed(1) + 'px');

        Vx = scene.clientWidth / 2;
        Vy = scene.clientHeight - 0.22 * camBridge.offsetHeight;
        setVar(heroWrap, '--hox', Vx.toFixed(1) + 'px');
        setVar(heroWrap, '--hoy', (Vy + pre).toFixed(1) + 'px');
        chaps.forEach(function (ch) {
          // offsets are relative to .zt-stage, which covers the viewport while the view is pinned
          ch.left = ch.el.offsetLeft; ch.top = ch.el.offsetTop; ch.wd = ch.el.offsetWidth; ch.h = ch.el.offsetHeight;
          setVar(ch.el, '--ox', (Vx - ch.left).toFixed(1) + 'px');
          setVar(ch.el, '--oy', (Vy - ch.top).toFixed(1) + 'px');
        });
      }

      // anchoring: an address hash lands on its hold; otherwise a resize or growing output keeps the
      // camera where it was (past the end of the track, the reader keeps their place in the footer)
      var target = null, snap = false;
      if (pendingHash !== null && performance.now() - movedAt > 800) {
        var id = pendingHash;
        if (id === 'try') target = 0;
        else if (zoom) target = trackTop + pre + HOLD_AT[id] * svh;
        if (target !== null) {
          snap = true;
          var head = id === 'try' ? null : document.getElementById(id + 'Head');
          if (head) { suppressFocus = true; try { head.focus({ preventScroll: true }); } catch (err) {} finally { suppressFocus = false; } }
        }
        // the browser may scroll to the fragment again as the page finishes loading: hold the hash until then
        if (document.readyState === 'complete') pendingHash = null;
      } else {
        pendingHash = null;
        if (dist > 0 && oldSvh > 0) {
          var oldL = L_SVH * oldSvh;
          target = trackTop + pre + (dist <= oldL ? dist / oldSvh * svh : L_SVH * svh + (dist - oldL));
        }
      }
      if (target !== null) {
        if (Math.abs(target - sy) > .5) jump(target);
        y = target;
        if (snap) snapNext = true;
      }
      mTrackTop = trackTop; mPre = pre; mSvh = svh; mZoom = zoom;

      if (tickFrame) { cancelAnimationFrame(tickFrame); tickFrame = 0; }
      update(performance.now());
    }

    function requestTick() { if (!tickFrame) tickFrame = requestAnimationFrame(update); }

    // the pill fades and lifts in and out (home.css .is-out); [hidden] only once it has faded
    var askShown = false, askTimer = 0;
    function setAsk(show) {
      if (!askPill || askShown === show) return;
      askShown = show;
      clearTimeout(askTimer);
      if (show) {
        askPill.hidden = false;
        requestAnimationFrame(function () { requestAnimationFrame(function () { if (askShown) askPill.classList.remove('is-out'); }); });
      } else {
        askPill.classList.add('is-out');
        if (reduceMQ.matches) askPill.hidden = true;
        else askTimer = setTimeout(function () { if (!askShown) askPill.hidden = true; }, 260);
      }
    }
    // the pill's band, bottom right: a box (document px) that reaches into it blocks the pill
    function askBlockedAt(top, bottom, right) {
      return right > vw - 220 && top < y + svh - 16 && bottom > y + svh - 76;
    }

    /* ----- update: writes only ----- */
    function update(now) {
      tickFrame = 0;
      var footBlocks = !!foot && askBlockedAt(footTop, footBot, footRight);
      if (!zoom) {
        settled = true;
        var chapBlocks = chaps.some(function (ch) { return askBlockedAt(ch.dTop, ch.dBot, ch.dRight); });
        setAsk(!!composerWrap && composerBottom < y && !footBlocks && !chapBlocks);
        return;
      }
      var L = L_SVH * svh;
      dT = clamp(y - trackTop - pre, 0, L);

      // time-based damping: the same feel at 30, 60 or 120 Hz
      var gap = last ? now - last : Infinity;
      var dt = gap > 100 ? 1 / 60 : Math.min(.064, gap / 1000);   // a tick after idle starts at one frame, not a lurch
      last = now;
      if (snapNext) { d = dT; snapNext = false; }
      else d += (dT - d) * (1 - Math.exp(-dt * 9));
      if (Math.abs(dT - d) < .5) d = dT;
      settled = d === dT;
      setClass(doc, 'zt-rest', settled && d === 0);   // the first screen at rest: layers unpromoted (home.css)

      u = d / svh;
      c = camAt(u);

      // the town fades while #device still covers the centre (gone by u 2.45, while the panel is still at .5 or
      // more, so the town never lingers half-transparent over the valley); a phone stops it at 2.6x, where its source
      // runs out, and has it gone by 2.3x, before the #device hold (it used to sit at .1-.2 under the card through the hold)
      var narrow = vw < 700, tCap = narrow ? 2.6 : 3.5;   // 4.4 pushed the town to 2.3x its own pixels
      var sTown = depth(2.2, c, tCap), oTown = 1 - ss(narrow ? seg(sTown, 1.56, 2.3) : seg(sTown, 2.45, 2.56));
      var sBr = depth(1.45, c, 6), oBr = 1 - ss(seg(sBr, 4.5, 6));
      var sTw = depth(1.25, c, 3.2);
      // the gate: as #budget starts to pass, the tower pair cuts in (a .07 svh dissolve, never a lingering ghost)
      // at the size of the first screen and parts past the camera, leaving by scale like the first pair
      var sDoor = 1 + 2.2 * ss(seg(u, 3.95, 4.75)), oDoor = ss(seg(u, 3.95, 4.02));
      // caps keep each photograph near its own resolution: the sky and the mountains used to reach
      // 9x, several times their source, and the late frames went soft.
      // Each value is written on the box that reads it, not on .scene: a custom property on .scene restyled every
      // scene element each frame (about 5ms a frame at a phone's CPU, with the hero and the chapters)
      setVar(camSky, '--s-sky', depth(30, c, 4).toFixed(4));
      setVar(camFour, '--s-four', depth(5, c, narrow ? 2.6 : 3.4).toFixed(4));
      setVar(camFour, '--o-four', (.9 + .1 * ss(seg(c, 1.25, 1.85))).toFixed(3));
      setVar(camTown, '--s-town', sTown.toFixed(4));
      setVar(camTown, '--o-town', oTown.toFixed(3));
      setVar(camBridge, '--s-br', sBr.toFixed(4));
      setVar(camBridge, '--o-br', oBr.toFixed(3));
      var sTwS = sTw.toFixed(4), sDoorS = sDoor.toFixed(4), oDoorS = oDoor.toFixed(3);
      setClass(camTown, 'is-gone', oTown <= .001);
      setClass(camBridge, 'is-gone', oBr <= .001);
      for (var s = 0; s < camSplits.length; s++) { setVar(camSplits[s], '--s-tw', sTwS); setClass(camSplits[s], 'is-gone', sTw >= 3.2); }
      for (var g = 0; g < camDoors.length; g++) {
        setVar(camDoors[g], '--s-door', sDoorS); setVar(camDoors[g], '--o-door', oDoorS);
        setClass(camDoors[g], 'is-gone', oDoor <= .001 || sDoor >= 3.2);
      }

      // the hero UI passes the camera first (depth 1)
      var hs = depth(1, c, 2.2);
      ho = heroFade(hs);
      setVar(heroWrap, '--hs', hs.toFixed(4));
      setVar(heroWrap, '--ho', ho.toFixed(3));
      setVar(heroWrap, '--ho-top', heroTopFade(hs).toFixed(3));
      setClass(heroWrap, 'is-off', ho < .05);

      // beyond the arch: each chapter from depth, a hold, then past the camera
      var rel = Math.max(0, y - (trackTop + pre + L));   // how far the released view has moved up
      var blocking = false, mo = 0;
      for (var k = 0; k < chaps.length; k++) {
        var ch = chaps[k];
        var so = chapter(u, ch.w);
        ch.s = so[0]; ch.o = so[1];
        setVar(ch.el, '--cs', ch.s.toFixed(4));
        setVar(ch.el, '--co', ch.o.toFixed(3));
        setVar(ch.el, '--ci', so[2].toFixed(3));
        setClass(ch.el, 'is-off', ch.o < .05);
        if (ch.o > .05 && ch.top + ch.h - rel > svh - 76 && ch.left + ch.wd > vw - 220) blocking = true;
        if (ch.o > mo) mo = ch.o;
      }
      // the scrim only protects text: it follows the most visible chapter, and the photo is clear in between
      if (scrim) setVar(scrim, '--scrim-o', (.34 * mo).toFixed(3));

      setAsk(ho < .05 && !blocking && !footBlocks);

      if (!settled) requestTick();
    }

    /* ----- keyboard: focus never lands on something faded; the camera moves to where it is readable -----
       Judged one frame after focusin, once the browser has scrolled the target into view. */
    function offsetWithin(el, anc) {
      var top = 0;
      for (var n = el; n && n !== anc; n = n.offsetParent) top += n.offsetTop;
      return top;
    }
    function snapTo(top) {
      jump(top);
      y = top; snapNext = true;
      if (tickFrame) { cancelAnimationFrame(tickFrame); tickFrame = 0; }
      update(performance.now());
    }
    function revealFocus(t) {
      if (!zoom || document.activeElement !== t) return;
      var sy = window.scrollY || window.pageYOffset || 0;
      var uT = clamp(sy - trackTop - pre, 0, L_SVH * svh) / svh;   // where the camera is headed, not where it is
      if (heroWrap.contains(t)) {
        var tr = t.getBoundingClientRect();
        var out = tr.bottom < 0 || tr.top > window.innerHeight;
        if (heroFade(depth(1, camAt(uT), 2.2)) >= .95 && !out) return;
        snapTo(clamp(offsetWithin(t, heroWrap) - svh / 3, 0, pre) + trackTop);
        return;
      }
      for (var k = 0; k < chaps.length; k++) {
        var ch = chaps[k];
        if (!ch.el.contains(t)) continue;
        if (chapter(uT, ch.w)[1] >= .95) return;
        snapTo(trackTop + pre + HOLD_AT[ch.id] * svh);
        return;
      }
    }
    // only focus that follows the visitor's own key press or tap moves the page; a focus that
    // playground.js makes seconds later (a load finishing) never pulls a reader back
    var markInput = function () { inputAt = performance.now(); };
    document.addEventListener('keydown', markInput, true);
    document.addEventListener('pointerdown', markInput, true);
    // any move of the reader's own cancels a pending address-hash landing
    var markMoved = function () { movedAt = performance.now(); };
    ['keydown', 'pointerdown', 'wheel', 'touchstart'].forEach(function (type) { document.addEventListener(type, markMoved, { capture: true, passive: true }); });
    document.addEventListener('focusin', function (e) {
      if (!zoom || suppressFocus || performance.now() - inputAt > 800) return;
      var t = e.target;
      if (!t || !t.closest) return;
      try { if (!t.matches(':focus-visible')) return; } catch (err) { return; }
      requestAnimationFrame(function () {
        revealFocus(t);
        // the phone card row: Chrome does not scroll a partly visible card into view, so the row does (never the page)
        var row = t.parentNode;
        if (row && row.classList && row.classList.contains('pages-grid') && row.scrollWidth > row.clientWidth + 1 && row.firstElementChild) {
          var left = t.offsetLeft - row.firstElementChild.offsetLeft;
          try { row.scrollTo({ left: left, behavior: reduceMQ.matches ? 'auto' : 'smooth' }); } catch (err) { row.scrollLeft = left; }
        }
      });
    });

    /* ----- the way back: chapter 1's button and the floating pill return to the composer ----- */
    var askers = document.querySelectorAll('[data-ask]');
    for (var a = 0; a < askers.length; a++) {
      askers[a].addEventListener('click', function () {
        select(0, false);
        // re-anchor to the Write tab's height now, so no later measure interrupts the glide back
        if (measureFrame) { cancelAnimationFrame(measureFrame); measureFrame = 0; }
        measure();
        if (reduceMQ.matches) jump(0);
        else window.scrollTo({ top: 0, behavior: 'smooth' });
        if (promptEl) {
          suppressFocus = true;
          try { promptEl.focus({ preventScroll: true }); } finally { suppressFocus = false; }
        }
      });
    }

    /* ----- output: when a run finishes, the whole card comes into view -----
       The tray and the disclaimer can end below the fold (1440x900: 58px under it). The hero is taller than the
       screen by then, and the scroll before the camera starts (pre) moves it up without any zoom, so the page
       glides the card's bottom into view within that hold, never further: the camera stays at u 0. Only for a
       reader still on the first screen; layout is read once per finished run. */
    var pgRoot = heroWrap.querySelector('[data-playground]');
    var outCard = heroWrap.querySelector('.h-out');
    if (pgRoot && outCard && window.MutationObserver) {
      var runState = pgRoot.getAttribute('data-state');
      new MutationObserver(function () {
        var st = pgRoot.getAttribute('data-state');
        if (st === runState) return;
        runState = st;
        if (st !== 'done' || !zoom) return;
        var sy = window.scrollY || window.pageYOffset || 0;
        var hold = trackTop + Math.max(0, heroWrap.offsetHeight - svh);   // this layout's pre, measured or not yet
        if (sy >= hold) return;
        var r = outCard.getBoundingClientRect();
        var band = parseFloat(getComputedStyle(doc).getPropertyValue('--header-band')) || 72;
        var by = Math.min(r.bottom + 16 - window.innerHeight, r.top - band - 12);   // never past the card's own top under the header
        if (!(r.height > 0) || by <= 0) return;
        try { window.scrollTo({ top: Math.min(hold, sy + by), behavior: 'smooth' }); }
        catch (err) { window.scrollTo(0, Math.min(hold, sy + by)); }
      }).observe(pgRoot, { attributes: true, attributeFilter: ['data-state'] });
    }

    /* ----- output: follow the newest text while the reader is at the bottom of the card ----- */
    var cont = heroWrap.querySelector('[data-pg="cont"]');
    var contText = heroWrap.querySelector('[data-pg="conttext"]');
    if (cont && contText && window.MutationObserver) {
      var stick = true, contHidden = cont.hidden;
      cont.addEventListener('scroll', function () {
        stick = cont.scrollHeight - cont.scrollTop - cont.clientHeight < 24;
      }, { passive: true });
      new MutationObserver(function () {
        if (!contText.textContent) stick = true;
        if (stick) cont.scrollTop = cont.scrollHeight;
      }).observe(contText, { childList: true, characterData: true, subtree: true });
      new MutationObserver(function () {
        if (contHidden && !cont.hidden) stick = true;
        contHidden = cont.hidden;
      }).observe(cont, { attributes: true, attributeFilter: ['hidden'] });
    }

    /* ----- first pass: rasterise the big photos once at full size while nothing moves -----
       The first time a layer grows past its on-screen size it needs its full-size raster, and that
       frame stalls (80ms and more). A 10px, nearly transparent box behind the scene draws each
       downloaded photo at natural size for three frames, once, so the first scroll is as smooth as the
       second. Same URLs, from cache: nothing new is downloaded. Zoom mode only. */
    var warmed = false;
    function warmUp() {
      if (warmed || !zoom || !settled || d !== 0) return;
      var imgs = scene.querySelectorAll('.cam img');
      for (var i = 0; i < imgs.length; i++) if (!imgs[i].complete) return;
      warmed = true;
      var box = document.createElement('div');
      box.setAttribute('aria-hidden', 'true');
      box.style.cssText = 'position:fixed;left:0;top:0;width:10px;height:10px;overflow:hidden;opacity:.01;pointer-events:none;z-index:-2';
      ['.cam--sky img', '.cam--four img', '.cam--town img', '.cam--bridge img', '.cam--split-l img', '.cam--split-r img'].forEach(function (sel) {
        var src = scene.querySelector(sel);
        var url = src && (src.currentSrc || src.src);
        if (!url || /^data:/.test(url) || !src.naturalWidth) return;
        var im = new Image();
        im.alt = '';
        im.src = url;
        im.style.cssText = 'position:absolute;left:0;top:0;max-width:none;height:auto;will-change:transform;width:' + src.naturalWidth + 'px';
        box.appendChild(im);
      });
      document.body.appendChild(box);
      requestAnimationFrame(function () { requestAnimationFrame(function () { requestAnimationFrame(function () { box.remove(); }); }); });
    }
    function scheduleWarm() {
      if (warmed) return;
      if (window.requestIdleCallback) requestIdleCallback(warmUp, { timeout: 1500 });
      else setTimeout(warmUp, 0);
    }
    Array.prototype.forEach.call(scene.querySelectorAll('.cam img'), function (img) {
      if (!img.complete) { img.addEventListener('load', scheduleWarm); img.addEventListener('error', scheduleWarm); }
    });

    /* ----- wiring ----- */
    window.addEventListener('scroll', function () {
      y = window.scrollY || window.pageYOffset || 0;
      requestTick();
    }, { passive: true });
    window.addEventListener('resize', scheduleMeasure);
    window.addEventListener('load', scheduleMeasure);
    window.addEventListener('load', scheduleWarm);
    window.addEventListener('hashchange', function () {
      pendingHash = hashId(location.hash);
      movedAt = -1e9;                       // following a link is a deliberate move, not a stray scroll
      scheduleMeasure();
    });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(scheduleMeasure);
    if (reduceMQ.addEventListener) reduceMQ.addEventListener('change', scheduleMeasure);
    else if (reduceMQ.addListener) reduceMQ.addListener(scheduleMeasure);
    if (touchMQ) {
      var onTouch = function () { textWithGlass = touchMQ.matches; scheduleMeasure(); };
      if (touchMQ.addEventListener) touchMQ.addEventListener('change', onTouch);
      else if (touchMQ.addListener) touchMQ.addListener(onTouch);
    }
    if (window.ResizeObserver) {
      var ro = new ResizeObserver(scheduleMeasure);
      ro.observe(heroWrap);
      chaps.forEach(function (ch) { ro.observe(ch.el); });
    }

    // QA hook
    window.etZoom = {
      state: function () {
        // settled is judged against the latest scroll position, so a scroll the camera has not ticked for yet reads unsettled
        var target = zoom ? clamp(y - trackTop - pre, 0, L_SVH * svh) : 0;
        return { y: y, dT: target, d: d, u: u, p: u / L_SVH, c: c, pre: pre, svh: svh, viewH: viewH, L: L_SVH * svh, trackTop: trackTop,
                 V: [Vx, Vy], fly: zoom, settled: settled && d === target, ticking: !!tickFrame };
      },
      yAt: function (uu) { return trackTop + pre + uu * svh; }
    };

    measure();
    scheduleWarm();
  }

  try { initZoom(); }
  catch (err) {
    // a thrown error must never leave content faded, scaled or pinned: fall back to the static page
    doc.classList.remove('zt-on');
    doc.classList.remove('zt-pending');
    ['.scene', '.cam', '.scene-scrim', '.zt', '.zt-hero', '.zt-chap'].forEach(function (sel) {
      var els = document.querySelectorAll(sel);
      for (var i = 0; i < els.length; i++) els[i].removeAttribute('style');
    });
    if (window.console) console.error(err);
  }
})();
