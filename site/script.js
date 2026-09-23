/* The Embedding Tax — landing scroll engine.

   Two layers of motion, deliberately:
   1. Owed's smooth scroll drives the window. A wheel tick moves a target and the page
      eases toward it every frame, so the scroll itself carries inertia.
   2. The choreography reads the resulting scroll distance and smooths it again at 0.14 a
      frame, which is what keeps the layers from snapping on a fast flick. Below 60fps the
      step grows with the frame time, so a slow device settles in the same time, not the same
      number of frames. On touch screens it does not smooth at all: the OS already does.

   Performance: update() never reads layout. Every size and offset it needs is cached by
   measure() on resize / font load / image load, the scroll position is cached by the
   scroll listener, and each of the rig's custom properties is written only when its value
   changes, on the element that consumes it (STAGE_TARGETS). A custom property is inherited,
   so one written on .stage restyled all of the stage's elements every frame; one written on
   .shade restyles .shade.

   Reduced motion: the smooth scroll is off, values snap, the pointer parallax is zero, and
   every large move (bridge exit and zoom, splitframe zoom, card fly-in, panel slides) plays
   at a fraction of its amplitude or becomes a fade. The CSS block does the rest.

   Short screens (max-height: 560px): the stage unpins in CSS and every panel sits in normal
   flow; this script stops driving the rig and marks every reveal as done. */
(function () {
  "use strict";

  var root = document.documentElement;
  root.classList.add("js");   // also set inline in <head>; harmless duplicate
  var section = document.querySelector(".cinema-scroll");
  if (!section) return;

  /* ---------- timing: every scroll range in one table (px of stage scroll) ---------- */
  /* Paced in scroll distance so every beat gets a fully visible plateau (about 100vh) with
     eased ramps of about 360px; the band map's starting points, validated by the flick test. */
  var TIMING = {
    STAGE_LENGTH: 6300,              // = the +6300px in .cinema-scroll's height (styles.css): change both together
    INTRO_EXIT: [520, 1000],         // hero and sentence leave (a 520px plateau first)
    BUDGET: [820, 1180, 2120, 2480], // frame 2: starts in, fully in, starts out, gone
    DEPTH: [2620, 2980, 3920, 4280], // frame 3
    PROGRESS_END: 4280,              // the shared camera push ends here
    CARDS_IN: [4360, 5620],          // spec cards travel in
    CONTROLS_IN: [5360, 5760],       // carousel controls fade in
    CHAPTER_AT: [0, 1000, 2550, 6300],// Intro | Budget | Depth | end (= STAGE_LENGTH)
    // In-page links land in the middle of the range where their panel is fully shown.
    ANCHOR_AT: { "#cinema": 0, "#budget": 1650, "#depth": 3450 }
  };
  var RAMP_MS = 2400;                // band 1's one-time load ramp (--k1), from scene-open
  var CHAPTER_NAMES = ["Intro", "Budget", "Depth"];
  var ANCHOR_FOCUS = { "#cinema": ".hero-title", "#budget": ".story-panel-bridge h2", "#depth": ".story-panel-bazaar h2" };
  // One source for the easings: the tokens shared.css declares (the anchor's curves).
  var EASE = (function () {
    var cs = getComputedStyle(root);
    function tok(name, fallback) { var v = cs.getPropertyValue(name).trim(); return v || fallback; }
    var out = tok("--ease-out-power2", "cubic-bezier(.25,1,.5,1)");
    return { out: out, expo: out, inn: tok("--ease-in-quart", "cubic-bezier(.895,.03,.685,.22)") };
  })();

  /* ---------- elements ---------- */
  var stage = section.querySelector(".stage");
  var heroTitle = document.querySelector(".hero-title");
  var heroNum = document.getElementById("heroNum");
  var introCopy = document.querySelector(".intro-copy");
  var budgetPanel = document.querySelector(".story-panel-bridge");
  var depthPanel = document.querySelector(".story-panel-bazaar");
  // The depth panel fades to opacity 0 but stays in the DOM, so its link leaves the tab
  // order while it is invisible (WCAG 2.4.7). Only the link: the text stays readable.
  var depthLink = depthPanel ? depthPanel.querySelector(".note-button") : null;
  var slider = document.querySelector(".sights-slider");
  var track = document.querySelector(".sights-track");
  var sightsControls = document.querySelector(".sights-controls");
  var prevBtn = document.querySelector(".sight-prev");
  var nextBtn = document.querySelector(".sight-next");
  var playBtn = document.getElementById("sightPlay");
  var dotSet = document.getElementById("sightDots");
  var sightCount = document.getElementById("sightCount");
  var countVis = document.getElementById("sightCountVis");
  var countSr = document.getElementById("sightCountSr");
  var capNav = document.getElementById("capNav");
  var capPill = document.querySelector(".cap-pill");
  var capLinks = toArray(document.querySelectorAll(".cap-links a[data-sec]"));
  var rail = document.getElementById("secRail");
  var sectionLinks = toArray(document.querySelectorAll(".cap-links a[data-sec], .sec-rail a[data-sec]"));
  var chapFlag = document.getElementById("chapFlag");
  var chapFlagWord = document.getElementById("chapFlagWord");
  var chapFlagNum = document.getElementById("chapFlagNum");
  var scrollCue = document.getElementById("scrollCue");
  var skipStory = document.getElementById("skipStory");
  var pauseMotion = document.getElementById("pauseMotion");
  var veil = document.querySelector(".dawn-veil");
  var backStack = document.querySelector(".back-stack");
  var shadeEl = document.querySelector(".stage .shade");
  var scrimHero = document.querySelector(".band-scrim--hero");
  var scrimBudget = document.querySelector(".band-scrim--budget");
  var scrimDepth = document.querySelector(".band-scrim--depth");
  var closing = document.querySelector(".closing");
  var closingScene = document.querySelector(".closing-scene");

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  var shortScreen = window.matchMedia("(max-height: 560px)");
  var finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
  var coarsePointer = window.matchMedia("(pointer: coarse)");

  /* ---------- state ---------- */
  // vh: the window's height (the closing band, short screens, the pointer). sh: the pinned
  // stage's own height (100svh), which is what the rig is laid out against; on iOS the window
  // grows when the toolbar collapses while the stage does not.
  var L = { vh: window.innerHeight, sh: window.innerHeight, vw: window.innerWidth, sectionTop: 0, sectionH: 0, maxDist: 0,
            budgetTop: 0, depthTop: 0, closingTop: 0, closingH: 1, pill: [], sightStep: 0 };
  var sy = window.scrollY;
  var isShort = shortScreen.matches;
  var targetMouseX = 0, targetMouseY = 0, mouseX = 0, mouseY = 0;
  var smoothScroll = 0, initialized = false, rafPending = false, lastTick = 0;
  var activeSection = -2, factsCounted = false, depthBuilt = false, budgetEntered = false;
  var cueGone = false, controlsReady = false, firstActivation = false;
  var introStart = -1, introOn = false, closingNear = false;
  var rampStart = -1, scrollDir = 0, userPaused = false;
  var stageCache = {}, closingCache = {}, segShown = [-1, -1, -1];
  var syncScrollRig = function () {};   // replaced by the smooth-scroll rig when it runs

  /* ---------- helpers ---------- */
  function toArray(list) { return Array.prototype.slice.call(list); }
  function clamp(v, min, max) {
    if (min === undefined) min = 0;
    if (max === undefined) max = 1;
    return Math.min(max, Math.max(min, v));
  }
  function smoothstep(e0, e1, v) { var x = clamp((v - e0) / (e1 - e0)); return x * x * (3 - 2 * x); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function segmentInOut(s, r) {
    var enter = smoothstep(r[0], r[1], s), exit = smoothstep(r[2], r[3], s);
    return { enter: enter, exit: exit, active: enter * (1 - exit) };
  }
  function pad2(n) { return (n < 10 ? "0" : "") + n; }
  function raf2(fn) { requestAnimationFrame(function () { requestAnimationFrame(fn); }); }
  function wait(ms) { return new Promise(function (res) { setTimeout(res, ms); }); }
  function onChange(mq, fn) { if (mq.addEventListener) mq.addEventListener("change", fn); else if (mq.addListener) mq.addListener(fn); }
  function setVar(el, cache, name, value) {
    if (!el || cache[name] === value) return;
    cache[name] = value;
    el.style.setProperty(name, value);
  }
  // Where each live variable is consumed (styles.css). Anything not listed goes on .stage.
  var STAGE_TARGETS = {
    "--back-opacity": [backStack], "--back-x": [backStack], "--back-y": [backStack], "--back-scale": [backStack],
    "--shade-z": [shadeEl], "--shade-top-alpha": [shadeEl], "--shade-mid-alpha": [shadeEl], "--shade-bottom-alpha": [shadeEl],
    "--title-y": [heroTitle], "--title-scale": [heroTitle], "--title-opacity": [heroTitle, scrimHero],
    "--intro-copy-y": [introCopy], "--intro-copy-opacity": [introCopy],
    "--panel2-opacity": [budgetPanel, scrimBudget], "--panel2-y": [budgetPanel],
    "--panel3-opacity": [depthPanel, scrimDepth], "--panel3-y": [depthPanel],
    "--sights-opacity": [slider], "--sights-enter-x": [slider], "--sights-visibility": [slider],
    "--sights-scale": [slider], "--sights-top": [slider],
    "--sights-controls-opacity": [sightsControls], "--sights-screen-top": [sightsControls],
    "--k1": [heroTitle, introCopy, scrollCue], "--k2": [budgetPanel], "--k3": [depthPanel], "--k4": [slider]
  };
  function setStage(name, value) {
    if (stageCache[name] === value) return;
    stageCache[name] = value;
    var els = STAGE_TARGETS[name] || [stage];
    for (var i = 0; i < els.length; i++) if (els[i]) els[i].style.setProperty(name, value);
  }

  /* ---------- layout cache ----------
     The only place that reads layout. Runs at init, on resize (once per frame), when the
     web font lands, when images load, and when the short-screen mode flips. */
  function measure() {
    L.vh = window.innerHeight;
    L.vw = window.innerWidth;
    L.sh = isShort ? L.vh : (stage.clientHeight || L.vh);
    sy = window.scrollY;
    L.sectionTop = section.getBoundingClientRect().top + sy;
    L.sectionH = section.offsetHeight;
    L.maxDist = Math.max(0, L.sectionH - L.sh);
    if (isShort) {
      // In flow and untransformed only in this mode.
      if (budgetPanel) L.budgetTop = budgetPanel.getBoundingClientRect().top + sy;
      if (depthPanel) L.depthTop = depthPanel.getBoundingClientRect().top + sy;
    }
    if (closing) {
      var c = closing.getBoundingClientRect();
      L.closingTop = c.top + sy;
      L.closingH = Math.max(1, c.height);
    }
    L.pill = [];
    capLinks.forEach(function (a) { L.pill[Number(a.dataset.sec)] = { left: a.offsetLeft, width: a.offsetWidth }; });
    if (sightCards.length && track) {
      var gap = parseFloat(getComputedStyle(track).columnGap || "0") || 0;
      L.sightStep = sightCards[0].offsetWidth + gap;
    }
  }

  var measurePending = false;
  function scheduleMeasure() {
    if (measurePending) return;
    measurePending = true;
    requestAnimationFrame(function () {
      measurePending = false;
      measure();
      updateSightSlider();
      placePill();
      requestTick();
    });
  }

  /* ---------- Owed's smooth scroll ----------
     Skipped on touch (the OS already does this well) and under reduced motion. */
  (function smoothScrollRig() {
    if (reduceMotion.matches) return;
    if (window.matchMedia("(pointer: coarse)").matches) return;

    var target = window.scrollY, current = window.scrollY;
    var running = false, last = 0;
    // goToAnchor moves the page directly. Without resyncing, the next frame of a wheel
    // scroll still in flight would ease the page back to where the wheel was heading.
    syncScrollRig = function () { target = current = window.scrollY; running = false; };

    function maxScroll() { return Math.max(0, document.documentElement.scrollHeight - window.innerHeight); }

    function frame(now) {
      var dt = Math.min(0.064, (now - last) / 1000);
      last = now;
      current += (target - current) * (1 - Math.exp(-dt * 6.5));   // frame-rate independent
      var settled = Math.abs(target - current) < 0.5;
      if (settled) current = target;
      window.scrollTo({ top: current, behavior: "auto" });
      if (settled) { running = false; return; }
      requestAnimationFrame(frame);
    }

    window.addEventListener("wheel", function (e) {
      if (e.ctrlKey || reduceMotion.matches) return;   // let the browser zoom / scroll natively
      e.preventDefault();
      var remaining = Math.abs(target - current);
      var damp = 1 / (1 + remaining / 900);
      target = clamp(target + e.deltaY * damp, 0, maxScroll());
      if (!running) { running = true; last = performance.now(); requestAnimationFrame(frame); }
    }, { passive: false });
    window.addEventListener("scroll", function () { if (!running) target = current = window.scrollY; }, { passive: true });
    window.addEventListener("resize", function () { target = clamp(target, 0, maxScroll()); });
  })();

  /* ---------- the choreography ---------- */

  function update(now) {
    rafPending = false;
    if (typeof now !== "number") now = performance.now();
    var reduce = reduceMotion.matches;

    var target = clamp(sy - L.sectionTop, 0, L.maxDist);
    // 0.14 a frame at 60fps and above, as it always was; below 60fps the step follows the time
    // (k = 9.05/s is 0.14 at 16.7ms), so a slow frame no longer stretches the tail. After an
    // idle spell the first frame counts as one 60fps frame, not as the whole pause.
    var dt = now - lastTick > 100 ? 1 / 60 : Math.min(0.1, (now - lastTick) / 1000);
    lastTick = now;
    if (!initialized || reduce || isShort || coarsePointer.matches) { smoothScroll = target; initialized = true; }
    else smoothScroll = lerp(smoothScroll, target, Math.max(0.14, 1 - Math.exp(-dt * 9.05)));
    if (Math.abs(smoothScroll - target) < 0.08) smoothScroll = target;

    if (reduce) { mouseX = mouseY = 0; targetMouseX = targetMouseY = 0; }
    else { mouseX = lerp(mouseX, targetMouseX, 0.12); mouseY = lerp(mouseY, targetMouseY, 0.12); }

    if (!cueGone && target > 24 && scrollCue) { cueGone = true; scrollCue.classList.add("is-gone"); }

    var past = sy > L.sectionTop + L.sectionH - L.sh * 0.5;
    updateClosing(reduce);
    // Hide-on-scroll-down inside the stage; back on any upward scroll, at the stage's end and
    // on short screens (the CSS also brings it back whenever focus is inside the capsule).
    if (capNav) capNav.classList.toggle("hidden", !isShort && !past && scrollDir > 0 && target > 160);

    if (isShort) {
      updateShort(past);
      if (Math.abs(mouseX - targetMouseX) > 0.001 || Math.abs(mouseY - targetMouseY) > 0.001) requestTick();
      return;
    }

    var s = smoothScroll;
    var frame2 = segmentInOut(s, TIMING.BUDGET);
    var frame3 = segmentInOut(s, TIMING.DEPTH);
    var progress = clamp(s / TIMING.PROGRESS_END);
    var introExit = smoothstep(TIMING.INTRO_EXIT[0], TIMING.INTRO_EXIT[1], s);
    // The cards enter over 1,400px of scroll and travel 130vw.
    var sightsEnter = Math.pow(smoothstep(TIMING.CARDS_IN[0], TIMING.CARDS_IN[1], s), 1.55);
    var controlsEnter = smoothstep(TIMING.CONTROLS_IN[0], TIMING.CONTROLS_IN[1], s);
    var blurActive = clamp(frame2.active + frame3.active);
    var panel2Opacity = frame2.active * (1 - frame2.exit);
    var panel3Opacity = frame3.active * (1 - frame3.exit);
    var backScale = 0.76 + progress * 0.2 + frame2.enter * 0.18 + frame3.enter * 0.16;
    // amp: how much of each large translate plays; zoom: how much of each large scale ramp.
    var amp = 1, zoom = 1;
    if (reduce) {
      amp = 0.2; zoom = 0.25;
      blurActive *= 0.25;
      backScale = 0.76 + (backScale - 0.76) * 0.4;
    }

    // The arrival settle: 1.5s, ease-out quart, folded additively into the rig's values.
    var intro = 0;
    if (introStart >= 0) {
      var t = clamp((now - introStart) / 1500);
      intro = Math.pow(1 - t, 4);
      if (t >= 1) introStart = -2;
    }

    // Band 1's one-time load ramp: k1 = max(scrollK, loadK). scrollK assembles within the
    // first 160px so a reader arriving mid-page never waits; loadK runs RAMP_MS from the
    // moment the veil lifts and holds at 1 (there is nothing above band 1 to scroll back to).
    var loadK = 1;
    if (rampStart >= 0) { loadK = clamp((now - rampStart) / RAMP_MS); if (loadK >= 1) rampStart = -2; }
    else if (rampStart === -1 && introOn) loadK = 0;
    var k1 = reduce ? 1 : Math.max(clamp(s / 160), loadK);

    // Placed with the scale the stage actually uses (after the reduced-motion damping).
    var sightsScreenTop = clamp(L.sh * 0.19, 112, 220) - 36;
    var sightsParentTop = L.sh - (L.sh - sightsScreenTop) / backScale;
    // The slider sits inside .back-stack, which is 106vw wide (left -3vw) and scaled by
    // --back-scale about its bottom centre, so its left edge lands at 50vw - 53vw * backScale
    // on screen. This puts the slider's origin back at the viewport's left edge (in the
    // back-stack's own pre-scale units); the 130vw fly-in rides on top of it.
    var sightsLeft = L.vw * (0.53 - 0.5 / backScale);

    // The photo layers' variables (--four-*, --bazaar-*, --bridge-*, --split-*, --frame2-*,
    // --blur-px, --back-brightness, --sky-settle, and --mx/--my, which only composer3d.css
    // reads) are no longer written: their layers were removed from this page.
    setStage("--back-opacity", (1 - frame2.active * 0.06).toFixed(4));
    setStage("--back-x", (mouseX * -12).toFixed(2) + "px");
    setStage("--back-y", (mouseY * -4 + 28 * intro).toFixed(2) + "px");
    setStage("--back-scale", (backScale * (1 + 0.045 * intro)).toFixed(4));
    setStage("--shade-z", frame2.active > 0.02 ? "2" : "0");
    setStage("--shade-top-alpha", (blurActive * 0.465).toFixed(4));
    setStage("--shade-mid-alpha", (blurActive * 0.42).toFixed(4));
    setStage("--shade-bottom-alpha", (blurActive * 0.51).toFixed(4));

    setStage("--title-y", (introExit * -210 * amp).toFixed(2) + "px");
    setStage("--title-scale", (1 - introExit * 0.08 * zoom).toFixed(4));
    setStage("--title-opacity", (1 - introExit).toFixed(4));

    setStage("--intro-copy-y", (introExit * 90 * amp).toFixed(2) + "px");
    setStage("--intro-copy-opacity", (1 - introExit).toFixed(4));
    setStage("--panel2-opacity", panel2Opacity.toFixed(4));
    setStage("--panel2-y", "calc(-50% + " + ((-frame2.exit * 86 + (1 - frame2.enter) * 58) * amp).toFixed(2) + "px)");
    setStage("--panel3-opacity", panel3Opacity.toFixed(4));
    setStage("--panel3-y", "calc(-50% + " + ((-frame3.exit * 86 + (1 - frame3.enter) * 58) * amp).toFixed(2) + "px)");

    // Full motion: the cards fly 130vw at full opacity. Reduced: they fade in where they sit.
    setStage("--sights-opacity", (reduce ? sightsEnter : 1).toFixed(4));
    setStage("--sights-enter-x", ((reduce ? 0 : (1 - sightsEnter) * 1.3 * L.vw) + sightsLeft).toFixed(2) + "px");
    setStage("--sights-visibility", sightsEnter > 0.01 ? "visible" : "hidden");
    setStage("--sights-controls-opacity", controlsEnter.toFixed(4));
    setStage("--sights-scale", (1 / backScale).toFixed(4));
    setStage("--sights-top", sightsParentTop.toFixed(2) + "px");
    setStage("--sights-screen-top", sightsScreenTop.toFixed(2) + "px");

    // Per-band assembly progress (styles.css scrubs every entrance off these; reversible on
    // scroll-up; pinned at 1 under reduced motion so every part sits in its final state).
    setStage("--k1", k1.toFixed(3));
    setStage("--k2", (reduce ? 1 : frame2.enter).toFixed(3));
    setStage("--k3", (reduce ? 1 : frame3.enter).toFixed(3));
    setStage("--k4", (reduce ? 1 : sightsEnter).toFixed(3));

    // ---- class states (classList.toggle with a force is a no-op when nothing changes) ----
    var hideLink = panel3Opacity < 0.05;
    if (depthLink && depthLink.inert !== hideLink) depthLink.inert = hideLink;
    if (introCopy) introCopy.classList.toggle("is-offstage", introExit > 0.995);
    // The intro card's three links leave the tab order as soon as the card has faded.
    var introGone = introExit > 0.9;
    if (introCopy && introCopy.inert !== introGone) introCopy.inert = introGone;
    if (budgetPanel) {
      if (!budgetEntered && panel2Opacity > 0.05) { budgetEntered = true; budgetPanel.classList.add("is-entered"); }
      budgetPanel.classList.toggle("is-shown", panel2Opacity > 0.6);
      // Instrument's figures count up once, the first time panel 1 is really on screen.
      if (!factsCounted && panel2Opacity > 0.6) { factsCounted = true; budgetPanel.classList.add("is-counted"); countFacts(); }
    }
    if (depthPanel && !depthBuilt && panel3Opacity > 0.35) { depthBuilt = true; depthPanel.classList.add("is-built"); }

    var ready = controlsEnter > 0.98;
    if (ready !== controlsReady) {
      controlsReady = ready;
      if (sightsControls) sightsControls.classList.toggle("is-ready", ready);
      setHold("notReady", !ready);
      if (ready && !firstActivation) { firstActivation = true; setTimeout(activateSight, 120); }
    }
    // The rail steps aside while the cards cross the right edge, and past the stage.
    if (rail) rail.classList.toggle("is-away", sightsEnter > 0.3 || past);

    // Where the story is: header pill, header bars, rail thumb and rail track.
    var CH = TIMING.CHAPTER_AT;
    writeSegs([0, 1, 2].map(function (c) { return past ? 1 : clamp((s - CH[c]) / (CH[c + 1] - CH[c])); }));
    setSection(past ? -1 : s < CH[1] ? 0 : s < CH[2] ? 1 : 2);

    if (Math.abs(smoothScroll - target) > 0.08 ||
        Math.abs(mouseX - targetMouseX) > 0.001 ||
        Math.abs(mouseY - targetMouseY) > 0.001 ||
        introStart >= 0 || rampStart >= 0) {
      requestTick();
    }
  }

  /* Short screens: no rig. The chapter follows the panels' cached document positions. */
  function updateShort(past) {
    var probe = sy + L.vh * 0.4;
    var b = [L.sectionTop, L.budgetTop, L.depthTop, L.sectionTop + L.sectionH];
    writeSegs([0, 1, 2].map(function (c) { return past ? 1 : clamp((probe - b[c]) / Math.max(1, b[c + 1] - b[c])); }));
    setSection(past ? -1 : probe >= L.depthTop ? 2 : probe >= L.budgetTop ? 1 : 0);
  }

  function applyMode() {
    isShort = shortScreen.matches;
    if (!isShort) return;
    // Every panel is in flow and fully shown, so every reveal is simply done.
    if (depthLink) depthLink.inert = false;
    if (introCopy) { introCopy.classList.remove("is-offstage"); introCopy.inert = false; }
    ["--k1", "--k2", "--k3", "--k4"].forEach(function (name) { setStage(name, "1"); });
    if (capNav) capNav.classList.remove("hidden");
    if (budgetPanel) { budgetPanel.classList.add("is-entered", "is-shown", "is-counted"); budgetEntered = true; factsCounted = true; }
    if (depthPanel) { depthPanel.classList.add("is-built"); depthBuilt = true; }
    if (sightsControls) sightsControls.classList.add("is-ready");
    controlsReady = true;
    firstActivation = true;
    setHold("notReady", false);
    if (rail) rail.classList.remove("is-away");
  }

  function writeSegs(vals) {
    for (var c = 0; c < 3; c++) {
      var v = Math.round(vals[c] * 500) / 500;
      if (v === segShown[c]) continue;
      segShown[c] = v;
      if (capNav) capNav.style.setProperty("--seg" + c, v);
      if (rail) rail.style.setProperty("--seg" + c, v);
    }
  }

  function requestTick() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(update);
  }

  /* ---------- where you are ----------
     Sana's sliding highlight in the header and Air's filled rail slot, keyed to one index. */
  function setSection(idx) {
    if (idx === activeSection) return;
    var first = activeSection === -2;
    activeSection = idx;
    sectionLinks.forEach(function (a) {
      var on = Number(a.dataset.sec) === idx;
      a.classList.toggle("is-current", on);
      if (on) a.setAttribute("aria-current", "location");
      else a.removeAttribute("aria-current");
    });
    placePill();
    if (rail) {
      rail.classList.toggle("has-current", idx >= 0);
      if (idx >= 0) rail.style.setProperty("--rail-idx", idx);
      // The first placement snaps; later ones slide.
      if (first) raf2(function () { rail.classList.add("is-settled"); });
    }
    if (!first && idx >= 0) flagChapter(idx);
  }

  function placePill() {
    if (!capPill) return;
    var g = activeSection >= 0 ? L.pill[activeSection] : null;
    if (!g || !g.width) { capPill.style.opacity = "0"; return; }
    capPill.style.opacity = "1";
    capPill.style.width = g.width + "px";
    capPill.style.transform = "translateX(" + g.left + "px)";
  }

  /* ---------- chapter flag: Raw Materials' "Now entering", as Vercel's blue toast ---------- */
  var flagAnims = [];
  function flagChapter(idx) {
    if (!chapFlag || !chapFlag.animate || document.hidden || L.vw <= 900) return;
    flagAnims.forEach(function (a) { a.cancel(); });   // rapid changes restart, never stack
    flagAnims = [];
    if (chapFlagWord) chapFlagWord.textContent = CHAPTER_NAMES[idx];
    if (chapFlagNum) chapFlagNum.textContent = pad2(idx + 1);
    if (reduceMotion.matches) {
      flagAnims.push(chapFlag.animate(
        [{ opacity: 0 }, { opacity: 1, offset: 0.14 }, { opacity: 1, offset: 0.88 }, { opacity: 0 }],
        { duration: 2000, easing: "linear" }));
      return;
    }
    flagAnims.push(chapFlag.animate([
      { opacity: 0, transform: "translateY(10px)", easing: EASE.expo },
      { opacity: 1, transform: "none", offset: 0.14 },
      { opacity: 1, transform: "none", offset: 0.88, easing: EASE.inn },
      { opacity: 0, transform: "translateY(6px)" }
    ], { duration: 2000 }));
    if (chapFlagWord) flagAnims.push(chapFlagWord.animate(
      [{ transform: "translateY(105%)" }, { transform: "none" }],
      { duration: 460, delay: 90, easing: EASE.out, fill: "backwards" }));
    if (chapFlagNum) flagAnims.push(chapFlagNum.animate(
      [{ opacity: 0 }, { opacity: 1 }], { duration: 300, delay: 160, fill: "backwards" }));
  }

  /* ---------- count-ups ----------
     The final figure is in the HTML, so crawlers, screenshots and visitors without scripts
     get the true number; a figure is reset only while it animates. The box is held at its
     final width so the digits never shove the layout. A new count cancels a running one. */
  function countUp(el, to, ms, decimals) {
    if (!el || reduceMotion.matches || document.hidden) return;
    var id = (el._countId || 0) + 1;
    el._countId = id;
    var f = Math.pow(10, decimals);
    var show = function (v) { return v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }); };
    if (!el.style.minWidth) el.style.minWidth = el.getBoundingClientRect().width + "px";
    var t0 = performance.now();
    (function step(now) {
      if (el._countId !== id) return;
      var t = Math.min(1, (now - t0) / ms);
      var e = t === 1 ? 1 : 1 - Math.pow(2, -10 * t);   // easeOutExpo
      el.textContent = show(Math.round(to * e * f) / f);
      if (t < 1) requestAnimationFrame(step);
      else el.textContent = show(to);
    })(t0);
  }

  function countFacts() {
    // 520ms apart, so 16.8 counts in lockstep with its fill (CSS delays it 520ms too).
    toArray(document.querySelectorAll(".fact-num")).forEach(function (el, i) {
      var to = parseFloat(el.dataset.to);
      if (!isFinite(to)) return;
      setTimeout(function () { countUp(el, to, 1400, 1); }, i * 520);
    });
  }

  /* ---------- word splitting ----------
     One helper for every per-word reveal (Savor, Readymag). Words become span.wd with a
     running --i and inherit data-mark from the phrase they sit in; spaces stay text nodes,
     so wrapping and text-wrap:balance are unchanged. The visual copy is aria-hidden and a
     single sr-only copy carries the accessible text. No overflow mask anywhere. */
  function splitWords(el) {
    if (!el || el.dataset.split) return;
    el.dataset.split = "1";
    var text = el.textContent.replace(/\s+/g, " ").trim();
    var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    var i = 0;
    // Punctuation that touches the word before it rides on that word. As its own inline-block
    // it was a line-break opportunity, and the comma after "eats half of it" wrapped alone.
    var last = null, gap = true;
    nodes.forEach(function (node) {
      var markHost = node.parentNode && node.parentNode.closest ? node.parentNode.closest("[data-mark]") : null;
      if (markHost && !el.contains(markHost)) markHost = null;
      var frag = document.createDocumentFragment();
      node.nodeValue.split(/(\s+)/).forEach(function (part) {
        if (!part) return;
        if (/^\s+$/.test(part)) { frag.appendChild(document.createTextNode(part)); gap = true; return; }
        var lead = !gap && last ? (part.match(/^[,.;:!?’”)\]]+/) || [""])[0] : "";
        if (lead) {
          last.appendChild(document.createTextNode(lead));
          if (last.dataset.mark) last.dataset.trail = "1";   // keep the underline off the comma
          part = part.slice(lead.length);
          if (!part) return;
        }
        gap = false;
        var w = document.createElement("span");
        w.className = "wd";
        w.style.setProperty("--i", i++);
        if (markHost) w.dataset.mark = markHost.dataset.mark;
        w.textContent = part;
        frag.appendChild(w);
        last = w;
      });
      node.parentNode.replaceChild(frag, node);
    });
    var vis = document.createElement("span");
    vis.setAttribute("aria-hidden", "true");
    while (el.firstChild) vis.appendChild(el.firstChild);
    var sr = document.createElement("span");
    sr.className = "sr-only";
    sr.textContent = text;
    el.appendChild(sr);
    el.appendChild(vis);
  }

  /* The footer wordmark as separate letters (Open), each rising from the footer's edge. */
  function splitFootWord() {
    var host = document.querySelector(".big-foot-word .bfw");
    if (!host || host.dataset.split) return;
    host.dataset.split = "1";
    var words = host.textContent.trim().split(/\s+/);
    var i = 0;
    host.textContent = "";
    words.forEach(function (word, wi) {
      if (wi) host.appendChild(document.createTextNode(" "));
      var w = document.createElement("span");
      w.className = "bfw-word";
      word.split("").forEach(function (ch) {
        var l = document.createElement("span");
        l.className = "bfw-l";
        l.style.setProperty("--i", i++);
        var g = document.createElement("span");
        g.className = "bfw-g";
        g.textContent = ch;
        l.appendChild(g);
        w.appendChild(l);
      });
      host.appendChild(w);
    });
  }

  /* ---------- hero: the 50,000,000 odometer ----------
     Vercel's per-digit reels and Shopify's odometer. It rolls once, on arrival at the top,
     and stops; it never re-rolls or ticks (a moving number would imply live data). */
  function mk(tag, cls) { var el = document.createElement(tag); el.className = cls; return el; }

  function settleHero(animated) {
    if (!heroNum || !heroTitle) return;
    heroNum.classList.add("is-shown");
    if (!animated) heroTitle.classList.add("no-anim");
    heroTitle.classList.add("is-settled");
  }

  function rollHero() {
    var text = heroNum.textContent.trim();
    var odo = mk("span", "odo");
    odo.setAttribute("aria-hidden", "true");
    var di = 0, lastReel = null;
    for (var c = 0; c < text.length; c++) {
      var ch = text.charAt(c);
      if (ch < "0" || ch > "9") { odo.appendChild(document.createTextNode(ch)); continue; }
      var d = Number(ch), n = di === 0 ? d : d + 10;   // the first digit rolls up to 5, the rest a full turn
      var win = mk("span", "odo-win");
      var ghost = mk("span", "odo-ghost");
      ghost.textContent = ch;
      win.appendChild(ghost);
      var reel = mk("span", "odo-reel");
      reel.style.setProperty("--n", n);
      reel.style.setProperty("--d", (1000 + 70 * di) + "ms");
      reel.style.setProperty("--w", (30 * di) + "ms");
      for (var k = 0; k <= n; k++) { var sp = document.createElement("span"); sp.textContent = String(k % 10); reel.appendChild(sp); }
      win.appendChild(reel);
      odo.appendChild(win);
      lastReel = reel;
      di++;
    }
    heroNum.textContent = "";
    heroNum.appendChild(odo);
    heroNum.classList.add("is-rolling", "is-shown");
    var done = false;
    function collapse() {
      if (done) return;
      done = true;
      heroNum.textContent = text;
      heroNum.classList.remove("is-rolling", "is-rolled");
      heroTitle.classList.add("is-settled");   // draws the tax bar under the figure
    }
    raf2(function () { heroNum.classList.add("is-rolled"); });
    if (lastReel) lastReel.addEventListener("transitionend", function (e) { if (e.propertyName === "transform") collapse(); });
    setTimeout(collapse, 2100);
  }

  function startHero() {
    if (!heroNum || !heroTitle) return;
    if (!introOn) { settleHero(false); return; }
    var fonts = document.fonts && document.fonts.ready ? Promise.race([document.fonts.ready, wait(600)]) : wait(0);
    Promise.all([fonts, wait(300)]).then(function () {
      var fontOk = !document.fonts || !document.fonts.check || document.fonts.check('400 50px "Newsreader"');
      if (!document.hidden && fontOk && !reduceMotion.matches && window.scrollY < 40) rollHero();
      else settleHero(false);
    });
  }

  /* ---------- arrival: the veil lifts once the photo can paint (Paraform) ---------- */
  function startIntro() {
    var atTop = (!location.hash || location.hash === "#cinema") && window.scrollY < 40;
    introOn = atTop && !reduceMotion.matches && !isShort;
    if (!atTop) { root.classList.add("no-veil"); return; }
    if (!introOn) { root.classList.add("scene-open"); return; }   // the veil's lift only, no settle
    var decodes = toArray(document.querySelectorAll(".sky-img, .bridge-img")).map(function (img) {   // none since the scene left; the veil then lifts on the next frame
      return img.decode ? img.decode().catch(function () {}) : Promise.resolve();
    });
    Promise.race([Promise.all(decodes), wait(500)]).then(function () {
      introStart = performance.now();
      rampStart = introStart;   // band 1's --k1 ramp starts with the settle
      root.classList.add("scene-open");
      requestTick();
    });
  }
  if (veil) veil.addEventListener("animationend", function () { if (veil.parentNode) veil.parentNode.removeChild(veil); });

  /* ---------- entrances: one-shot observers ---------- */
  function watchReveals() {
    var els = toArray(document.querySelectorAll("[data-reveal]"));
    if (!("IntersectionObserver" in window)) {
      els.forEach(function (el) { el.classList.add("is-in"); });
      if (stage) stage.classList.add("is-onstage");
      if (closing) { closingNear = true; closing.classList.add("is-near"); }
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add("is-in"); io.unobserve(en.target); } });
    }, { threshold: 0.2 });
    els.forEach(function (el) { io.observe(el); });
    // The stage's motes run only while the stage is on screen (.is-onstage).
    if (stage) {
      new IntersectionObserver(function (entries) {
        stage.classList.toggle("is-onstage", entries[0].isIntersecting);
      }, { threshold: 0.05 }).observe(stage);
    }
    // The closing parallax and its motes run only while the band is near the viewport.
    if (closing) {
      new IntersectionObserver(function (entries) {
        closingNear = entries[0].isIntersecting;
        closing.classList.toggle("is-near", closingNear);
        if (closingNear) requestTick();
      }, { rootMargin: "240px 0px" }).observe(closing);
    }
  }

  /* Closing band parallax: the sky and the city drift at two speeds past the still bridge,
     written on .closing only while it is near. Zero under reduced motion. */
  function updateClosing(reduce) {
    if (!closing || !closingNear) return;
    var cp = clamp((sy + L.vh - L.closingTop) / (L.vh + L.closingH));
    var a = reduce ? 0 : (cp - 0.5) * 2;
    setVar(closing, closingCache, "--close-sky-y", (a * 56).toFixed(2) + "px");
    setVar(closing, closingCache, "--close-city-y", (a * 22).toFixed(2) + "px");
  }

  /* =========================================================================================
     THE CAROUSEL — an infinite slider (three identical sets, only the middle one real),
     with the page's one ambient loop (autoplay, with Pause) and direct manipulation.
     ========================================================================================= */
  var originalCards = toArray(document.querySelectorAll(".sight-card"));
  var N = originalCards.length;
  var sightCards = [], dots = [];
  var activeSight = N, shownReal = 0;

  function setupSightSlider() {
    if (!track || !N) return;
    var frag = [];
    for (var set = 0; set < 3; set++) {
      for (var i = 0; i < N; i++) {
        var clone = originalCards[i].cloneNode(true);
        clone.dataset.sightIndex = String(set * N + i);
        if (set !== 1) {
          // Duplicates for a seamless wrap: never announced (they hold no focusables).
          clone.setAttribute("aria-hidden", "true");
        } else {
          clone.setAttribute("role", "group");
          clone.setAttribute("aria-roledescription", "slide");
          clone.setAttribute("aria-label", (i + 1) + " of " + N);
        }
        frag.push(clone);
      }
    }
    track.replaceChildren.apply(track, frag);
    sightCards = toArray(track.querySelectorAll(".sight-card"));
    activeSight = N;

    // Mistral AI's pagination: one mark per card, the current one a long dash.
    if (dotSet) {
      originalCards.forEach(function (card, i) {
        var dot = document.createElement("button");
        dot.type = "button";
        dot.className = "sight-dot" + (i === 0 ? " is-on" : "");
        var h = card.querySelector("h3");
        dot.setAttribute("aria-label", "Card " + (i + 1) + ": " + (h ? h.textContent.replace(/\s+/g, " ").trim() : ""));
        if (i === 0) dot.setAttribute("aria-current", "true");
        dot.addEventListener("click", function () {
          var target = N + i;                  // the middle set, the real one
          stopRotation();
          if (target !== activeSight) goTo(target);
        });
        dotSet.appendChild(dot);
        dots.push(dot);
      });
    }

    // Clicking a neighbouring card brings it forward (a pointer convenience; the cards are
    // not controls, so keyboard users have the arrows, dots and arrow keys).
    track.addEventListener("click", function (e) {
      var card = e.target.closest ? e.target.closest(".sight-card") : null;
      if (!card) return;
      var i = Number(card.dataset.sightIndex);
      if (!isFinite(i) || i === activeSight) return;
      stopRotation();
      goTo(i);
    });
    // Only the track's own transform ends a slide. Card scale/translate/colour transitions
    // bubble up too, and without this filter a wrap jumped mid-slide.
    track.addEventListener("transitionend", function (e) {
      if (e.target !== track || e.propertyName !== "transform") return;
      normalizeSightSlider();
    });

    if (prevBtn) prevBtn.addEventListener("click", function () { stopRotation(); goTo(activeSight - 1); });
    if (nextBtn) nextBtn.addEventListener("click", function () { stopRotation(); goTo(activeSight + 1); });
    setupRotation();
    setupDrag();
    setupKeys();
  }

  function updateSightSlider() {
    if (!track || !sightCards.length) return;
    track.style.setProperty("--sights-shift", (-(L.sightStep || 0) * activeSight).toFixed(2) + "px");
    sightCards.forEach(function (card, i) { card.classList.toggle("is-active", i === activeSight); });
    var real = ((activeSight % N) + N) % N;
    if (real === shownReal) return;
    shownReal = real;
    dots.forEach(function (dot, i) {
      dot.classList.toggle("is-on", i === real);
      if (i === real) dot.setAttribute("aria-current", "true");
      else dot.removeAttribute("aria-current");
    });
    // The live region (#sightCount) is written only when the card changes, and says it
    // in words; the "02 / 05" beside it is visual only.
    if (countVis) countVis.textContent = pad2(real + 1) + " / " + pad2(N);
    if (countSr) countSr.textContent = "Card " + (real + 1) + " of " + N;
  }

  function goTo(i) {
    // Bounded to two cards past the middle set: normalize only runs on transitionend, so
    // a burst of moves inside one 650ms slide used to walk the track clean off the screen.
    i = clamp(i, N - 2, 2 * N + 1);
    var changed = i !== activeSight;
    activeSight = i;
    updateSightSlider();
    if (changed) activateSight();
    afterSlide();
  }

  /* Activation beats: the badge pops, the figure rises and recounts, the counter rolls. */
  function restartClass(el, cls) { el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls); }
  function activateSight() {
    var card = sightCards[activeSight];
    if (!card) return;
    sightCards.forEach(function (c) { if (c !== card) c.classList.remove("is-activating"); });
    restartClass(card, "is-activating");
    var num = card.querySelector(".sight-num");
    var to = num ? parseFloat(num.dataset.to) : NaN;
    // 600ms, shorter than the 650ms slide, so a wrap jump never cuts a count short.
    if (to >= 8) countUp(num, to, 600, 0);
    if (countVis) restartClass(countVis, "is-rolling");
  }

  /* transitionend drives the wrap normally. Under reduced motion the track has no
     transition and it never fires, so normalise on the next frame instead. */
  function afterSlide() { if (reduceMotion.matches) requestAnimationFrame(normalizeSightSlider); }

  function jumpSightSlider(i) {
    track.classList.add("is-jumping");
    activeSight = i;
    updateSightSlider();
    // Two frames: one for the class to land, one for the transition-free layout to flush.
    raf2(function () { track.classList.remove("is-jumping"); });
  }
  function normalizeSightSlider() {
    if (activeSight >= N * 2) jumpSightSlider(activeSight - N);
    else if (activeSight < N) jumpSightSlider(activeSight + N);
  }

  /* ---------- autoplay: the page's only ambient loop ----------
     Zellerfeld's timed segment: the active dash fills over 6.5s in CSS and its animationend
     advances the card. No JS timers. It pauses while held (not yet on screen, off screen,
     tab hidden, hovered, focused, dragging), stops for good on any manual navigation (APG
     carousel), and never starts on its own under reduced motion. Pause/Play is visible. */
  var rotating = false;
  var holds = { notReady: true, offstage: false, hidden: document.hidden, hover: false, focus: false, drag: false };
  function held() { for (var k in holds) if (holds[k]) return true; return false; }
  function syncRotation() {
    if (!sightsControls) return;
    var h = held();
    sightsControls.classList.toggle("is-playing", rotating);
    sightsControls.classList.toggle("is-paused", rotating && h);
    if (playBtn) {
      playBtn.dataset.state = rotating ? "playing" : "paused";
      playBtn.setAttribute("aria-label", rotating ? "Pause card rotation" : "Play card rotation");
    }
    // While it rotates unattended, the counter stays quiet; otherwise it announces.
    if (sightCount) sightCount.setAttribute("aria-live", rotating && !h ? "off" : "polite");
  }
  function setHold(k, v) { if (holds[k] === v) return; holds[k] = v; syncRotation(); }
  function stopRotation() { if (!rotating) return; rotating = false; syncRotation(); }
  function startRotation() {
    rotating = true;
    if (sightsControls) { sightsControls.classList.remove("is-playing"); void sightsControls.offsetWidth; }
    syncRotation();   // re-adding .is-playing restarts the fill from 0
  }

  function setupRotation() {
    if (!sightsControls || !dotSet) return;
    if (playBtn) {
      playBtn.hidden = false;
      playBtn.addEventListener("click", function () { if (rotating) stopRotation(); else startRotation(); });
    }
    dotSet.addEventListener("animationend", function (e) {
      if (rotating && !held() && /^sight-dash-(fill|hold)$/.test(e.animationName)) goTo(activeSight + 1);
    });
    var hoverOn = function (e) { if (e.pointerType === "mouse") setHold("hover", true); };
    var hoverOff = function (e) { if (e.pointerType === "mouse") setHold("hover", false); };
    [slider, sightsControls].forEach(function (el) {
      if (!el) return;
      el.addEventListener("pointerenter", hoverOn);
      el.addEventListener("pointerleave", hoverOff);
    });
    sightsControls.addEventListener("focusin", function () { setHold("focus", true); });
    sightsControls.addEventListener("focusout", function (e) {
      if (!e.relatedTarget || !sightsControls.contains(e.relatedTarget)) setHold("focus", false);
    });
    document.addEventListener("visibilitychange", function () { setHold("hidden", document.hidden); });
    if (slider && "IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        var en = entries[0];
        setHold("offstage", !en.isIntersecting || en.intersectionRatio < 0.5);
      }, { threshold: [0, 0.5] }).observe(slider);
    }
    rotating = !reduceMotion.matches;
    syncRotation();
  }

  /* ---------- direct manipulation: Badoo's follow-the-finger, then commit or spring back ---------- */
  var drag = null, dragRaf = 0, justDragged = false;
  function setupDrag() {
    track.addEventListener("pointerdown", function (e) {
      if (e.pointerType === "mouse" && e.button !== 0) return;
      justDragged = false;
      drag = { id: e.pointerId, x0: e.clientX, y0: e.clientY, dx: 0, axis: null, t: e.timeStamp, lastX: e.clientX, vx: 0, moved: false };
    });
    track.addEventListener("pointermove", function (e) {
      if (!drag || e.pointerId !== drag.id) return;
      var dx = e.clientX - drag.x0, dy = e.clientY - drag.y0;
      if (!drag.axis) {
        if (Math.abs(dx) < 8 && Math.abs(dy) < 8) return;
        drag.axis = Math.abs(dx) > Math.abs(dy) ? "x" : "y";
        if (drag.axis === "y") { drag = null; return; }   // a vertical swipe keeps scrolling the story
        try { track.setPointerCapture(e.pointerId); } catch (err) { /* capture is optional */ }
        track.classList.add("is-dragging");
        setHold("drag", true);
      }
      var dt = Math.max(1, e.timeStamp - drag.t);
      drag.vx = 0.8 * ((e.clientX - drag.lastX) / dt) + 0.2 * drag.vx;
      drag.t = e.timeStamp;
      drag.lastX = e.clientX;
      drag.dx = dx;
      drag.moved = true;
      // One write per frame, on the track itself; no layout reads. The net scale is 1.
      if (!dragRaf) dragRaf = requestAnimationFrame(function () {
        dragRaf = 0;
        if (drag) track.style.setProperty("--sights-drag", drag.dx.toFixed(1) + "px");
      });
    });
    function endDrag(e, cancelled) {
      if (!drag || e.pointerId !== drag.id) return;
      var d = drag;
      drag = null;
      if (d.axis !== "x") return;
      track.classList.remove("is-dragging");
      try { track.releasePointerCapture(e.pointerId); } catch (err) { /* already released */ }
      setHold("drag", false);
      if (dragRaf) { cancelAnimationFrame(dragRaf); dragRaf = 0; }
      var vx = e.timeStamp - d.t > 90 ? 0 : d.vx;       // a pause before release kills the flick
      var steps = cancelled ? 0 : clamp(Math.round(-(d.dx + vx * 220) / (L.sightStep || 1)), -2, 2);
      if (!cancelled && steps === 0 && Math.abs(vx) > 0.45 && Math.abs(d.dx) > 24) steps = vx < 0 ? 1 : -1;
      track.style.setProperty("--sights-drag", "0px");
      if (d.moved) justDragged = true;
      if (steps) { stopRotation(); goTo(activeSight + steps); }
    }
    track.addEventListener("pointerup", function (e) { endDrag(e, false); });
    track.addEventListener("pointercancel", function (e) { endDrag(e, true); });
    // The click that follows a real drag must not also select a card.
    track.addEventListener("click", function (e) {
      if (!justDragged) return;
      justDragged = false;
      e.stopPropagation();
      e.preventDefault();
    }, true);
    // A sideways trackpad swipe steps one card (60px of travel, then a 550ms lock).
    var acc = 0, lastWheel = 0, lockUntil = 0;
    if (slider) slider.addEventListener("wheel", function (e) {
      if (!controlsReady || Math.abs(e.deltaX) <= 1.5 * Math.abs(e.deltaY)) return;
      var t = e.timeStamp;
      if (t - lastWheel > 200) acc = 0;
      lastWheel = t;
      if (t < lockUntil) return;
      acc += e.deltaX;
      if (Math.abs(acc) >= 60) { stopRotation(); goTo(activeSight + (acc > 0 ? 1 : -1)); acc = 0; lockUntil = t + 550; }
    }, { passive: true });
  }

  function setupKeys() {
    if (!sightsControls) return;
    sightsControls.addEventListener("keydown", function (e) {
      if (e.repeat) return;               // a held arrow is one move, not a move per repeat
      var target = null;
      if (e.key === "ArrowLeft") target = activeSight - 1;
      else if (e.key === "ArrowRight") target = activeSight + 1;
      else if (e.key === "Home") target = N;
      else if (e.key === "End") target = N * 2 - 1;
      if (target === null) return;
      e.preventDefault();
      stopRotation();
      if (target !== activeSight) goTo(target);
    });
  }

  /* ---------- in-page links ----------
     #budget and #depth are panels inside the sticky stage, so the browser's own jump lands
     on their DOM offset, where both are still at opacity 0. Each link goes instead to the
     middle of its panel's hold (TIMING.ANCHOR_AT). On short screens the panels are in flow,
     so the browser's own jump is right and is left alone. */
  function goToAnchor(hash) {
    if (isShort || !Object.prototype.hasOwnProperty.call(TIMING.ANCHOR_AT, hash)) return false;
    window.scrollTo({ top: L.sectionTop + TIMING.ANCHOR_AT[hash], behavior: "auto" });
    sy = window.scrollY;
    scrollDir = -1;   // a deliberate jump keeps the capsule on screen
    syncScrollRig();
    requestTick();
    // Move focus with the jump, to the heading, so a screen reader announces the panel.
    var target = document.querySelector(ANCHOR_FOCUS[hash]);
    if (target) {
      if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
      target.focus({ preventScroll: true });
    }
    return true;
  }

  document.addEventListener("click", function (e) {
    var a = e.target.closest ? e.target.closest('a[href^="#"]') : null;
    if (!a || !goToAnchor(a.getAttribute("href"))) return;
    e.preventDefault();
    if (history.replaceState) history.replaceState(null, "", a.getAttribute("href"));
  });
  window.addEventListener("hashchange", function () { goToAnchor(location.hash); });

  /* ---------- tooltips: Esc dismisses (WCAG 1.4.13); the next pointer or focus restores ---------- */
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") root.classList.add("tips-off"); });
  var tipsOn = function () { if (root.classList.contains("tips-off")) root.classList.remove("tips-off"); };
  document.addEventListener("pointerover", tipsOn, { passive: true });
  document.addEventListener("focusin", tipsOn);

  /* ---------- the stage's two controls ----------
     "Skip the story" jumps to the depth panel (WCAG 2.4.1; on short screens the browser's
     own jump is right). "Pause motion" toggles body.paused, which shared.css turns into
     animation-play-state: paused on every element and pseudo-element (the motes, the glows,
     the environment); the choice is remembered for the session. A hidden tab pauses too. */
  if (skipStory) skipStory.addEventListener("click", function () { if (!goToAnchor("#depth")) location.hash = "#depth"; });
  function syncPaused() { document.body.classList.toggle("paused", userPaused || document.hidden); }
  if (pauseMotion) {
    try { userPaused = sessionStorage.getItem("et-paused") === "1"; } catch (err) { userPaused = false; }
    var labelPause = function () { pauseMotion.textContent = userPaused ? "Play motion" : "Pause motion"; };
    labelPause();
    pauseMotion.addEventListener("click", function () {
      userPaused = !userPaused;
      try { sessionStorage.setItem("et-paused", userPaused ? "1" : "0"); } catch (err) { /* storage off: the session forgets */ }
      labelPause();
      syncPaused();
    });
  }
  document.addEventListener("visibilitychange", syncPaused);
  syncPaused();

  /* ---------- listeners ---------- */
  window.addEventListener("scroll", function () {
    var y = window.scrollY;
    if (y !== sy) scrollDir = y > sy ? 1 : -1;
    sy = y;
    requestTick();
  }, { passive: true });
  window.addEventListener("resize", scheduleMeasure);
  window.addEventListener("load", scheduleMeasure);
  if (closing) {
    var closingBridge = closing.querySelector(".closing-bridge");
    if (closingBridge && !closingBridge.complete) closingBridge.addEventListener("load", scheduleMeasure);
  }
  // The pill is sized from the links, which change width once the web font lands.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(scheduleMeasure);
  window.addEventListener("pointermove", function (e) {
    if (reduceMotion.matches || !finePointer.matches) return;
    targetMouseX = e.clientX / L.vw - 0.5;
    targetMouseY = e.clientY / L.vh - 0.5;
    requestTick();
  }, { passive: true });
  onChange(reduceMotion, function () {
    if (reduceMotion.matches) { stopRotation(); targetMouseX = targetMouseY = 0; }
    requestTick();
  });
  onChange(shortScreen, function () { applyMode(); scheduleMeasure(); });

  /* =========================================================================================
     WATCH IT WRITE: the landing's compact playground against window.EmbeddingTax (infer.js).
     The manifest (a few hundred bytes) is fetched now to fill the download size, the
     checkpoint step and the run's state; the weights download only on the visitor's click.
     States on #lpRoot's data-state: packaging (no infer.js or no manifest), idle, loading,
     ready, writing, done, failed. Every write is textContent; streamed text is appended in
     one rAF batch; the live region is debounced 600ms. Nothing is simulated: with the files
     missing the panel says the model is being packaged and offers the run report.
     ========================================================================================= */
  function setupWatch() {
    var rootEl = document.getElementById("lpRoot");
    if (!rootEl || !window.fetch || !window.AbortController) return;
    var $ = function (id) { return document.getElementById(id); };
    var ui = {
      presets: toArray(rootEl.querySelectorAll("[data-preset]")), prompt: $("lpPrompt"),
      load: $("lpLoad"), mb: $("lpMb"), progress: $("lpProgress"), bar: $("lpBar"), fill: $("lpBarFill"), barText: $("lpBarText"),
      write: $("lpWrite"), stop: $("lpStop"), tokens: $("lpTokens"),
      pack: $("lpPack"), idle: $("lpIdle"), fail: $("lpFail"), retry: $("lpRetry"), model: $("lpModel"),
      out: $("lpOut"), echo: $("lpEcho"), cont: $("lpCont"), copy: $("lpCopy"), status: $("lpStatus"), backend: $("lpBackend"),
      step: $("lpStep"), run: $("lpRun"), summary: $("lpSummary"),
      temp: $("lpTemp"), tempN: $("lpTempN"), topk: $("lpTopk"), topkN: $("lpTopkN"), max: $("lpMax"), maxN: $("lpMaxN"),
      seed: $("lpSeed"), seedLock: $("lpSeedLock"), reset: $("lpReset"), prov: $("lpProv"), arch: $("watchArch"),
      modes: toArray(rootEl.querySelectorAll('input[name="lpMode"]')), modeName: $("lpModeName"), modeDesc: $("lpModeDesc"),
      lockNote: $("lpLockNote"), unlimited: $("lpUnlimited"), unlimitedBadge: $("lpUnlimitedBadge"), guardState: $("lpGuardState")
    };
    /* The decoding presets: presets.js (window.EmbeddingTaxPresets), the one table the explorer and the
       home composer read too. Balanced, Focused and Creative share the guards; Published is
       tools/sample.py's temperature 0.8 / top-k 40 / 60 tokens with every guard off, and locks the
       sliders. If presets.js failed to load, the panel offers Published alone, with no guards. */
    var T = window.EmbeddingTaxPresets || {
      PRESETS: [{ id: "published", name: "Published", guards: false, temperature: 0.8, topK: 40, maxNewTokens: 60, untilDone: false,
        desc: "The settings used for the published samples: temperature 0.8, top-k 40, 60 tokens, no other rules." }],
      CUSTOM_DESC: { guarded: "", unguarded: "Published's sampler, run until the passage ends (at most 256 tokens). None of the guards are on." },
      DEFAULT_PRESET: "published", DEFAULT_SEED: 1337, GUARDS: null,
      NO_GUARDS: { topP: 1, minP: 0, repetitionPenalty: 1, repetitionWindow: 0, frequencyPenalty: 0, presencePenalty: 0, noRepeatNgram: 0,
                   noLeadingBreak: 0, noMidsentenceBreak: false, cadAlpha: 0, bestOf: 1, bestOfFluencyWeight: 0 },
      RANGES: { temperature: [0.6, 1.4, 0.05], topK: [40, 200, 1], maxNewTokens: [8, 90, 1] }, UNTIL_DONE_CAP: 256,
      minPFor: function () { return 0; }
    };
    var WATCH_PRESETS = T.PRESETS, CUSTOM_DESC = T.CUSTOM_DESC, GUARDS = T.GUARDS, NO_GUARDS = T.NO_GUARDS, minPFor = T.minPFor, UNTIL_DONE_CAP = T.UNTIL_DONE_CAP;
    var DEFAULT_PRESET = WATCH_PRESETS.filter(function (p) { return p.id === T.DEFAULT_PRESET; })[0] || WATCH_PRESETS[0], DEFAULT_SEED = T.DEFAULT_SEED;
    // the sliders and their boxes take the tested ranges from the table (bindPair reads them below)
    [[ui.temp, ui.tempN, T.RANGES.temperature], [ui.topk, ui.topkN, T.RANGES.topK], [ui.max, ui.maxN, T.RANGES.maxNewTokens]].forEach(function (r) {
      r[0].min = r[1].min = String(r[2][0]); r[0].max = r[1].max = String(r[2][1]); r[0].step = r[1].step = String(r[2][2]);
    });
    function presetOf(s) {   // the preset these settings are exactly, or null (Custom)
      for (var i = 0; i < WATCH_PRESETS.length; i++) {
        var p = WATCH_PRESETS[i];
        if (p.guards === s.guards && p.temperature === s.temperature && p.topK === s.topK && p.untilDone === s.untilDone &&
            (s.untilDone || p.maxNewTokens === s.maxNewTokens)) return p;
      }
      return null;
    }
    var settings = { guards: DEFAULT_PRESET.guards, temperature: DEFAULT_PRESET.temperature, topK: DEFAULT_PRESET.topK,
                     maxNewTokens: DEFAULT_PRESET.maxNewTokens, untilDone: DEFAULT_PRESET.untilDone, seed: DEFAULT_SEED };
    var manifest = null, state = "packaging", controller = null;
    var lastPrompt = "", runs = 0, tokensOut = 0, pending = "", flushRaf = 0;
    var fmt = function (n) { return Number(n).toLocaleString("en-US"); };
    var mb = function (b) { return Math.round(b / 1e6); };

    function setState(s) { state = s; rootEl.dataset.state = s; render(); }
    function render() {
      var live = state === "ready" || state === "writing" || state === "done";
      ui.pack.hidden = state !== "packaging";
      ui.idle.hidden = !(state === "idle" || state === "loading");
      ui.fail.hidden = state !== "failed";
      ui.load.hidden = state !== "idle";
      ui.progress.hidden = state !== "loading";
      ui.write.hidden = !(state === "ready" || state === "done");
      ui.stop.hidden = state !== "writing";
      ui.out.hidden = !live;
      ui.copy.hidden = state !== "done";
      ui.backend.hidden = !live;
      ui.prompt.disabled = state === "packaging" || state === "failed";
      ui.presets.forEach(function (b) { b.disabled = state === "packaging" || state === "failed" || state === "writing"; });
      syncWrite();
    }
    function syncWrite() {
      var has = ui.prompt.value.trim().length > 0;
      ui.write.disabled = !has || !(state === "ready" || state === "done");
      ui.write.textContent = state === "done" && ui.prompt.value === lastPrompt ? "Write again" : "Write";
    }
    // The live region (WCAG 4.1.3), debounced so a stream never floods a screen reader.
    var statusTimer = 0, statusText = "";
    var wasLocked = null;          // the last lock state announced
    function announce(text) {
      statusText = text;
      if (statusTimer) return;
      statusTimer = setTimeout(function () { statusTimer = 0; ui.status.textContent = statusText; }, 600);
    }

    /* ---- the manifest: small, fetched now; the weights wait for the click ---- */
    function fillManifest(m) {
      var params = typeof m.params_total === "number" ? m.params_total : 49296896;
      var step = typeof m.steps_completed === "number" ? m.steps_completed : null;
      var finished = m.training_finished === true;   // the one data field allowed to say so
      if (m.bytes) { ui.mb.textContent = String(mb(m.bytes)); ui.barText.textContent = "0 of " + mb(m.bytes) + " MB"; }
      if (step !== null) {
        ui.step.textContent = "step " + fmt(step);
        ui.run.textContent = finished ? "a training run that has finished" : "a training run that is still going";
      }
      ui.model.textContent = fmt(params) + " parameters" + (step !== null ? " · checkpoint step " + fmt(step) : "") +
        " · " + (finished ? "run finished" : "run still in progress");
      var prov = [];
      if (m.exported_at) prov.push("Exported " + String(m.exported_at).slice(0, 10));
      if (step !== null) prov.push("from step " + fmt(step));
      if (/int8/i.test(String(m.quantization || m.quantization_mode || ""))) prov.push("int8 weights");
      if (m.bytes) prov.push(mb(m.bytes) + " MB");
      prov.push("runs with onnxruntime-web on this device");
      ui.prov.textContent = prov.join(" · ");
    }
    function boot() {
      fetch("model/manifest.json", { cache: "default" })
        .then(function (r) { if (!r.ok) throw new Error("manifest " + r.status); return r.json(); })
        .then(function (m) {
          manifest = m;
          fillManifest(m);
          // infer.js is a deferred script after this one: give it up to 2s to define the API.
          var tries = 0;
          (function waitApi() {
            var api = window.EmbeddingTax;
            if (api && typeof api.load === "function") { if (api.ready) onLoaded(api.backend); else { setState("idle"); resume(api, m); } return; }
            if (++tries > 20) { setState("packaging"); return; }
            setTimeout(waitApi, 100);
          })();
        })
        .catch(function () { setState("packaging"); });
    }

    /* ---- load: one explicit action, a real progress bar (it never eases back) ---- */
    // a visit after a successful load reopens the copy saved in Cache Storage by itself; on a phone
    // or tablet (EmbeddingTax.constrained) only on a tap, as in playground.js, so a build of a few
    // hundred MB never lands on the page's arrival. cacheOnly never downloads, and any failure
    // while reopening by itself just leaves the Load button as before
    // (the button stays as short as "Load the model (50 MB)"; the idle note, which wraps, says it downloads nothing)
    var saved = false, savedSpan = null, idleNote = ui.idle.querySelector("p"), idleNoteText = idleNote ? idleNote.textContent : "";
    function savedLabel(on) {
      saved = on;
      if (!savedSpan) { savedSpan = document.createElement("span"); savedSpan.textContent = "Open the saved model"; ui.load.appendChild(savedSpan); }
      savedSpan.hidden = !on;
      if (ui.load.firstElementChild !== savedSpan) ui.load.firstElementChild.hidden = on;
      if (idleNote) idleNote.textContent = on ? "Already saved on this device: opening it downloads nothing. Nothing you type leaves this page." : idleNoteText;
    }
    function resume(api, m) {
      if (typeof api.isCached !== "function") return;
      api.isCached(m).then(function (hit) {
        if (!hit || state !== "idle" || api.ready) return;
        if (typeof api.constrained === "function" && api.constrained()) { savedLabel(true); return; }
        setState("loading");
        ui.barText.textContent = "Opening the copy saved on this device";
        api.load({ cacheOnly: true }).then(function (res) {
          onLoaded((res && res.backend) || api.backend);
        }, function () {
          if (state === "loading") setState(api.ready ? "ready" : "idle");
        });
      });
    }
    function onLoaded(backend) {
      ui.backend.textContent = (backend === "webgpu" ? "WebGPU" : "WASM") + " on this device";
      if (ui.arch) ui.arch.classList.add("is-lit");   // the crown's glow starts breathing: the load's reward
      setState("ready");
    }
    // focus follows the panel only if the visitor is still in it: a 50 MB load or a slow WASM run
    // can end after they have scrolled on, and an unconditional focus() would drag the page back
    // (a container of the panel counts as nowhere in particular, like body: Safari focuses the
    // nearest focusable ancestor, <main tabindex="-1">, on a tap, never the button, as in playground.js)
    function keepFocus(el) {
      var a = document.activeElement;
      if (el && (!a || a === document.body || rootEl.contains(a) || a.contains(rootEl))) el.focus({ preventScroll: true });
    }
    function load() {
      setState("loading");
      announce("Loading the model");
      var lastSaid = 0;
      if (saved) ui.barText.textContent = "Opening the copy saved on this device";
      window.EmbeddingTax.load(saved ? { cacheOnly: true } : { onProgress: function (loaded, total) {
        var t = total || (manifest && manifest.bytes) || 0;
        var p = t ? clamp(loaded / t) : 0;
        ui.fill.style.setProperty("--p", p.toFixed(3));
        ui.bar.setAttribute("aria-valuenow", String(Math.round(p * 100)));
        ui.barText.textContent = mb(loaded) + " of " + mb(t) + " MB";
        var now = performance.now();
        if (now - lastSaid > 600) { lastSaid = now; announce("Loading the model, " + mb(loaded) + " of " + mb(t) + " MB"); }
      } }).then(function (res) {
        onLoaded((res && res.backend) || window.EmbeddingTax.backend);
        announce("Model loaded. Write when ready.");
        keepFocus(ui.prompt.value.trim() ? ui.write : ui.prompt);
      }).catch(function (err) {
        if (window.console) console.error(err);
        if (err && err.code === "NOT_CACHED") {   // the saved copy is gone: back to the download button
          savedLabel(false);
          setState("idle");
          announce("The saved copy is no longer on this device. Load downloads it again.");
          return;
        }
        // iPhone and iPad run WebKit in every browser: "try another browser" cannot help there
        var title = ui.fail.querySelector(".wp-note-title");
        if (title && (/iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1))) {
          title.textContent = err && err.code === "OUT_OF_MEMORY" ? "This device ran out of memory for the model. Close other tabs, then try again." : "The model could not run on this device just now. Try again.";
        }
        setState("failed");
        announce("The model could not load on this device.");
        keepFocus(ui.retry);
      });
    }

    /* ---- write / stop: the stream is real; the caret is the only decoration ---- */
    function flush() {
      flushRaf = 0;
      ui.tokens.textContent = tokensOut + " tokens";
      if (!pending) return;
      var caret = ui.cont.querySelector(".wp-caret");
      var text = document.createTextNode(pending);
      pending = "";
      var gen = ui.cont.querySelector(".wp-gen");
      if (gen) gen.appendChild(text); else if (caret) ui.cont.insertBefore(text, caret); else ui.cont.appendChild(text);
    }
    function write() {
      if (!(state === "ready" || state === "done")) return;
      var prompt = ui.prompt.value;
      if (!prompt.trim()) return;
      // "Write again" moves the seed on by one so the text reads differently; "Same seed" keeps it.
      // The seed box and summary move with it, so the seed on screen is always the one that ran.
      if (prompt === lastPrompt && runs > 0 && !ui.seedLock.checked) { var r = runs; setSeed(settings.seed + 1); runs = r; }
      else if (prompt !== lastPrompt) runs = 0;
      var seed = settings.seed;
      runs++;
      lastPrompt = prompt;
      tokensOut = 0; pending = "";
      // one document: the prompt in grey, the model's words on a highlighter after it
      ui.cont.textContent = "";
      var echo = document.createElement("span");
      echo.className = "wp-echo-in"; echo.textContent = prompt;
      var gen = document.createElement("span");
      gen.className = "wp-gen";
      ui.cont.appendChild(echo); ui.cont.appendChild(gen);
      var caret = document.createElement("span");
      caret.className = "wp-caret"; caret.setAttribute("aria-hidden", "true");
      ui.cont.appendChild(caret);
      ui.tokens.textContent = "0 tokens";
      controller = new AbortController();
      setState("writing");
      announce("Writing");
      keepFocus(ui.stop);
      // every decoding option spelled out: the preset's sampler plus the guards (or none, for Published)
      var opts = { maxNewTokens: settings.maxNewTokens, temperature: settings.temperature, topK: settings.topK, seed: seed, untilDone: settings.untilDone };
      var guards = settings.guards ? GUARDS : NO_GUARDS;
      for (var k in guards) opts[k] = guards[k];
      if (settings.guards) opts.minP = minPFor(settings.temperature);
      opts.signal = controller.signal;
      opts.onToken = function (text) {
        tokensOut++;
        if (text) pending += text;
        if (!flushRaf) flushRaf = requestAnimationFrame(flush);
      };
      window.EmbeddingTax.generate(prompt, opts).then(finish).catch(function (err) {
        if (window.console && !(controller && controller.signal.aborted)) console.error(err);
        // the device failed mid-run: infer.js has unloaded the model, so this is a failure with a
        // retry, not a finished run (and EmbeddingTax.last would still be the previous run's)
        if (!window.EmbeddingTax.ready) {
          if (flushRaf) { cancelAnimationFrame(flushRaf); flushRaf = 0; }
          var caret = ui.cont.querySelector(".wp-caret");
          if (caret) caret.parentNode.removeChild(caret);
          controller = null;
          setState("failed");
          announce("The model stopped on this device.");
          keepFocus(ui.retry);
          return;
        }
        finish();
      });
    }
    function finish() {
      if (flushRaf) { cancelAnimationFrame(flushRaf); flushRaf = 0; }
      flush();
      // the measured count: onToken also carries the trailing UTF-8 flush, which is not a token
      var last = window.EmbeddingTax && window.EmbeddingTax.last;
      if (last && typeof last.newTokens === "number") { tokensOut = last.newTokens; ui.tokens.textContent = tokensOut + " tokens"; }
      var caret = ui.cont.querySelector(".wp-caret");
      if (caret) caret.parentNode.removeChild(caret);
      controller = null;
      setState("done");
      announce("Done, " + tokensOut + " tokens");
      keepFocus(ui.write);
    }
    function stop() { if (controller) controller.abort(); }

    /* ---- settings: a preset, then slider + numeric box on one row, the summary line always live ---- */
    function syncSummary() {
      var preset = presetOf(settings), locked = !settings.guards;
      ui.summary.textContent = (preset ? preset.name : "Custom") + " · temperature " + settings.temperature + " · top-k " + settings.topK +
        (settings.untilDone ? " · until done, at most " + UNTIL_DONE_CAP + " tokens" : " · max " + settings.maxNewTokens + " tokens") +
        (settings.guards ? " · guards on, min-p " + minPFor(settings.temperature) : " · no guards") + " · seed " + settings.seed;
      ui.modes.forEach(function (r) { r.checked = !!preset && r.value === preset.id; });
      ui.modeName.textContent = preset ? preset.name : "Custom";
      ui.modeDesc.textContent = preset ? preset.desc : (settings.guards ? CUSTOM_DESC.guarded : CUSTOM_DESC.unguarded);
      ui.guardState.textContent = settings.guards ? "on" : "off";
      ui.lockNote.hidden = !locked;
      if (locked !== wasLocked && wasLocked !== null) announce(locked ? "Published uses fixed settings; temperature, top-k and length are locked." : "Temperature, top-k and length can be changed.");
      wasLocked = locked;
      // Published is exactly its fixed sampler: its controls show the values and stand down
      ui.temp.disabled = ui.tempN.disabled = ui.topk.disabled = ui.topkN.disabled = locked;
      // Unlimited: the length controls stand down; "Not recommended" only without the guards
      ui.unlimited.checked = settings.untilDone;
      ui.unlimitedBadge.hidden = settings.guards;
      ui.max.disabled = ui.maxN.disabled = locked || settings.untilDone;
      ui.reset.setAttribute("aria-disabled", preset === DEFAULT_PRESET && settings.seed === DEFAULT_SEED ? "true" : "false");
    }
    function bindPair(range, num, key, unit) {
      var step = Number(range.step) || 1, min = Number(range.min), max = Number(range.max);
      function apply(v, from) {
        v = Number(v);
        if (!isFinite(v)) v = settings[key];
        v = Math.round(clamp(v, min, max) / step) * step;
        var shown = String(+v.toFixed(2));
        settings[key] = +shown;
        if (from !== range) range.value = shown;
        num.value = shown;   // also when the box itself changed: typed 5 must read back as the clamped 1.4
        range.setAttribute("aria-valuetext", shown + (unit ? " " + unit : ""));
        range.style.setProperty("--pct", (((v - min) / (max - min)) * 100).toFixed(1) + "%");
        syncSummary();
      }
      range.addEventListener("input", function () { apply(range.value, range); });
      num.addEventListener("change", function () { apply(num.value, num); });
      apply(range.value, null);
      return apply;
    }
    var setTemp = bindPair(ui.temp, ui.tempN, "temperature", "");
    var setTopk = bindPair(ui.topk, ui.topkN, "topK", "tokens");
    var setMax = bindPair(ui.max, ui.maxN, "maxNewTokens", "tokens");
    function setSeed(v) {
      v = Math.round(Number(v));
      if (!isFinite(v)) v = DEFAULT_SEED;
      v = Math.max(0, Math.min(2147483647, v));   // the explorer's range: infer.js wraps anything larger
      settings.seed = v; ui.seed.value = String(v); runs = 0; syncSummary();
    }
    function applyPreset(p) {   // the seed is not part of a preset
      settings.guards = p.guards;
      settings.untilDone = p.untilDone;
      setTemp(p.temperature, null); setTopk(p.topK, null); setMax(p.maxNewTokens, null);
    }
    ui.seed.addEventListener("change", function () { setSeed(ui.seed.value); });
    ui.modes.forEach(function (r) {
      r.addEventListener("change", function () {
        if (!r.checked) return;
        var p = WATCH_PRESETS.filter(function (x) { return x.id === r.value; })[0];
        if (p) applyPreset(p); else syncSummary();   // a preset this table lacks: the radios go back to what runs
      });
    });
    ui.unlimited.addEventListener("change", function () { settings.untilDone = ui.unlimited.checked; syncSummary(); });
    ui.reset.addEventListener("click", function () {
      if (ui.reset.getAttribute("aria-disabled") === "true") return;
      applyPreset(DEFAULT_PRESET); setSeed(DEFAULT_SEED);
      ui.seedLock.checked = false;
    });
    applyPreset(DEFAULT_PRESET);   // the sliders start from the table's default, not from whatever the markup says
    syncSummary();

    /* ---- wiring ---- */
    ui.presets.forEach(function (b) {
      b.addEventListener("click", function () {
        ui.prompt.value = b.textContent.trim();
        ui.presets.forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
        syncWrite();
        ui.prompt.focus();
      });
    });
    ui.prompt.addEventListener("input", function () {
      ui.presets.forEach(function (o) { if (o.textContent.trim() !== ui.prompt.value.trim()) o.removeAttribute("aria-pressed"); });
      syncWrite();
    });
    ui.prompt.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey && !ui.write.disabled && !ui.write.hidden) { e.preventDefault(); write(); }
    });
    ui.load.addEventListener("click", load);
    ui.retry.addEventListener("click", load);
    ui.write.addEventListener("click", write);
    ui.stop.addEventListener("click", stop);
    rootEl.addEventListener("keydown", function (e) { if (e.key === "Escape" && controller) { e.preventDefault(); stop(); } });
    ui.copy.addEventListener("click", function () {
      if (!navigator.clipboard || !navigator.clipboard.writeText) return;
      navigator.clipboard.writeText((ui.cont.querySelector(".wp-gen") || ui.cont).textContent).then(function () {
        ui.copy.textContent = "Copied";
        setTimeout(function () { ui.copy.textContent = "Copy"; }, 1200);
      }, function () { /* clipboard refused: the text stays selectable */ });
    });
    render();
    boot();
  }

  /* ---------- init ----------
     Every entrance state hides content under html.js, so one throw in here used to leave
     the panels, the closing copy and the footer at opacity 0. On any error the class comes
     off and the page shows everything, un-animated; the error is still thrown. */
  try {
    splitWords(document.querySelector(".intro-copy .intro-line"));
    splitWords(document.querySelector(".story-panel-bridge h2"));
    splitWords(document.querySelector(".story-panel-bazaar h2"));
    splitWords(document.getElementById("closeHead"));
    splitFootWord();
    setupSightSlider();
    measure();
    applyMode();
    updateSightSlider();
    watchReveals();
    setupWatch();
    if (scrollCue && (window.scrollY > 0 || (location.hash && location.hash !== "#cinema"))) scrollCue.classList.add("is-off");
    startIntro();
    startHero();
    requestTick();
    // Arriving from app.html with #budget or #depth: the browser has already jumped to the
    // DOM offset, so correct it once layout exists.
    if (location.hash) requestAnimationFrame(function () { measure(); goToAnchor(location.hash); });
  } catch (err) {
    root.classList.remove("js");
    throw err;
  }

  /* ---- the story opening panel: every figure comes from results.json, never from the page ---- */
  (function storyRun() {
    var bars = document.querySelector('[data-run="bars"]');
    if (!bars) return;
    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var set = function (key, text) {
      var el = document.querySelector('[data-run="' + key + '"]');
      if (el) el.textContent = text;
    };
    var fmt = function (n) { return Number(n).toLocaleString("en-US"); };
    fetch("results.json", { cache: "default" }).then(function (r) { return r.json(); }).then(function (data) {
      var curve = (data.training && data.training.val_curve) || [];
      if (!curve.length) return;
      var first = curve[0], last = curve[curve.length - 1];
      set("loss", last.loss.toFixed(4));
      set("first", first.loss.toFixed(4));
      set("firststep", fmt(first.step));
      set("drop", (-100 * (first.loss - last.loss) / first.loss).toFixed(1) + "%");
      set("tokens", fmt(data.training.tokens_seen));
      // 32 evenly spaced points of the real curve; the bar is the loss, floored just under the last one
      var N = 32, lo = last.loss * 0.6, hi = first.loss;   // a floor under the last value, so the tail still reads
      var frag = document.createDocumentFragment();
      for (var i = 0; i < N; i++) {
        var p = curve[Math.round(i * (curve.length - 1) / (N - 1))];
        var bar = document.createElement("i");
        bar.style.height = (100 * (p.loss - lo) / (hi - lo)).toFixed(1) + "%";
        if (i < 4) bar.className = "is-dim";   // the warm-up steps, dimmed as the reference card dims its edge bars
        if (!reduce) bar.style.setProperty("--d", (1100 + i * 30) + "ms");
        bar.title = "step " + fmt(p.step) + ": loss " + p.loss.toFixed(4);
        frag.appendChild(bar);
      }
      bars.appendChild(frag);
      var axis = document.querySelector('[data-run="axis"]');
      if (axis) {
        axis.textContent = "";
        [0, 0.25, 0.5, 0.75, 1].forEach(function (f) {
          var p = curve[Math.round(f * (curve.length - 1))];
          var span = document.createElement("span");
          span.textContent = fmt(p.step);
          axis.appendChild(span);
        });
      }
    }).catch(function () { /* the panel keeps its published fallback text */ });
  })();
})();

/* The opening owns the first screen: this sets .past-opening on <html> once it has been
   scrolled past, which is what turns the bar from white-on-artwork back into the glass capsule.
   The clip plays the way the reference calls for it: autoplay, muted, looping. Reduced motion
   is the one exception, since the CSS reset only reaches animations: there it sits paused on
   its first frame. */
(function () {
  var opening = document.querySelector(".ap, .nx, .vy");
  if (!opening) return;
  var video = opening.querySelector("video");
  var root = document.documentElement;
  var rm = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;

  if ("IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      root.classList.toggle("past-opening", !entries[0].isIntersecting);
    }, { rootMargin: "-72px 0px 0px 0px", threshold: 0 }).observe(opening);
  }

  if (!video) return;
  function sync() {
    if (rm && rm.matches) { video.pause(); return; }
    var p = video.play();
    if (p && p.catch) p.catch(function () {});
  }
  sync();
  if (rm && rm.addEventListener) rm.addEventListener("change", sync);
  else if (rm && rm.addListener) rm.addListener(sync);
})();

/* Retire the opening's entrance once its last tween has ended, so a later breakpoint change
   can never replay it. */
(function () {
  var last = document.querySelector(".nx-foot2");
  if (!last) return;
  var timer = 0;
  function done() {
    clearTimeout(timer);
    last.removeEventListener("animationend", done);
    document.documentElement.classList.add("is-entered");
  }
  last.addEventListener("animationend", done);
  timer = setTimeout(done, 4000);
})();
