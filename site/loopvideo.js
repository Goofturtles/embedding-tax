/* Keep the openings' clips playing, and on phones only while someone can see them.

   The clips are plain autoplaying, looping video: the markup says autoplay, loop, muted and
   playsinline, and that is what plays them. This file makes sure they keep going, because
   a browser will quietly leave one paused: autoplay can be refused until the visitor interacts
   with the page, a hidden tab suspends playback, a restored page comes back paused, and a stall
   on a slow connection can end with a paused element.

   So every clip is kicked on load, on the first touch or click, when the tab comes back, when
   the page is restored from the back/forward cache, and whenever one reports that it paused,
   stalled or ran out of data.

   Each clip's file is a <source media="(hover: hover) and (pointer: fine)"> in the markup, so a
   desktop browser picks it up while parsing, exactly as a src attribute did. Phones and tablets
   (a coarse or hover-less primary pointer, an iPad reporting itself as a Mac, or a device that
   says it has under 4 GB) are the ones that change, because a phone tab that runs out of memory
   is simply killed, and a 1080p decode is the biggest thing on these pages:
   - The file is attached only after the first paint, with preload="metadata", and only while
     its opening is on screen.
   - A clip pauses when its opening leaves the viewport (one IntersectionObserver) and picks up
     again when it comes back, so past the opening nothing decodes.
   - While the model is loading or writing (data-state on a playground root), the clips pause.

   The research opening's second copy of its clip (#vyFront, the masked turntable in front of
   the headline) only runs where it can do that job: not on touch screens and not in the narrow
   portrait layout, where it sits wholly below the type. There it never gets its file at all.
   Where it does run, it follows the back copy: a drift of more than a frame is closed by
   nudging its playback rate, a large one by a seek, so the two copies of one clip can no
   longer play apart or wrap at different moments.

   Reduced motion is the one exception to all of this. The pages hide these clips there, so they
   are paused and their download is dropped rather than left buffering for something nobody
   will see.  */
(function () {
  'use strict';

  var videos = [].slice.call(document.querySelectorAll('.ap-art, .nx-art, .vy-media video, .vy-front'));
  if (!videos.length) return;

  function mq(q) { return window.matchMedia ? window.matchMedia(q) : null; }
  var rm = mq('(prefers-reduced-motion: reduce)');
  function reduced() { return !!(rm && rm.matches); }

  var touchMq = mq('(pointer: coarse), (hover: none)');
  // the last resort: iPadOS reports itself as a Mac, and with a trackpad may report a fine pointer
  var ipad = navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1;
  var touch = !!(touchMq && touchMq.matches) || ipad;
  var lite = touch || (navigator.deviceMemory > 0 && navigator.deviceMemory < 4);
  var narrowPortrait = mq('(max-width: 900px) and (orientation: portrait)');
  var front = document.getElementById('vyFront');
  var back = document.getElementById('vyVideo');
  var painted = false, modelBusy = false;

  function frontAllowed() { return !touch && !(narrowPortrait && narrowPortrait.matches); }

  // A clip whose file waits: its <source media> did not match (so the browser skipped it while
  // parsing), or it is the front copy on a screen where that copy does not run. The file is
  // taken off the element and attached by kick() when it is wanted.
  videos.forEach(function (video) {
    video.lvSrc = null;
    var source = video.querySelector('source[src]');
    if (!source) return;
    var media = source.getAttribute('media'), m = media ? mq(media) : null;
    if (!(m && !m.matches) && !(video === front && !frontAllowed())) return;
    video.lvSrc = source.getAttribute('src');
    source.parentNode.removeChild(source);
    if (video.currentSrc) video.load();                  // a browser that ignored media: stop that download
  });

  function wanted(video) {
    if (reduced()) return false;
    if (lite && (video.lvOff || modelBusy)) return false;  // lvOff: its opening is off screen
    if (video === front && !frontAllowed()) return false;
    return true;
  }

  function attach(video) {
    if (lite) video.preload = 'metadata';
    video.src = video.lvSrc;
    video.lvSrc = null;
  }

  function kick(video) {
    if (!wanted(video)) { if (!video.paused) video.pause(); return; }
    if (video.lvSrc) {
      if (!painted) return;                               // the page paints first
      attach(video);
    }
    if (!video.paused || video.error) return;
    // an autoplay a browser will not grant answers every play() with another pause event, so a
    // clip that keeps refusing is left alone until the visitor does something
    if (video.lvTries > 8) return;
    video.lvTries = (video.lvTries || 0) + 1;
    var p = video.play();
    if (p && p.then) p.then(function () { video.lvTries = 0; }, function () { /* retried on the next gesture */ });
  }

  function kickAll() {
    videos.forEach(function (video) { video.lvTries = 0; kick(video); });
  }

  function placeFront() {
    if (!front) return;
    front.hidden = !frontAllowed();
    kick(front);
  }

  videos.forEach(function (video) {
    video.loop = true;                      // whatever the markup says, these clips repeat
    video.muted = true;                     // muted is what makes autoplay legal
    if (reduced()) {
      // the pages hide these clips entirely under reduced motion, so stop the download too
      video.pause();
      video.lvSrc = null;
      video.removeAttribute('src');
      [].slice.call(video.querySelectorAll('source')).forEach(function (s) { s.parentNode.removeChild(s); });
      video.load();
      return;
    }
    // an element that reports trouble is simply started again, that element and no other; one
    // that starts where it is not wanted (an autoplay off screen) is stopped again
    ['pause', 'play', 'stalled', 'suspend', 'waiting', 'canplay', 'loadeddata'].forEach(function (evt) {
      video.addEventListener(evt, function () { kick(video); });
    });
  });
  placeFront();
  kickAll();

  // Phones: the page gets its first paint to itself, then the clips get their files.
  requestAnimationFrame(function () { setTimeout(function () { painted = true; kickAll(); }, 0); });

  // Phones: pause what is off screen. The observer reports every target once on observe, so a
  // page restored mid-scroll starts with the right state.
  if (lite && 'IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        videos.forEach(function (video) {
          if (!entry.target.contains(video)) return;
          video.lvOff = !entry.isIntersecting;
          video.lvTries = 0;
          kick(video);
        });
      });
    }, { threshold: 0 });
    var seen = [];
    videos.forEach(function (video) {
      var opening = (video.closest && video.closest('.ap, .nx, .vy')) || video;
      if (seen.indexOf(opening) < 0) { seen.push(opening); io.observe(opening); }
    });
  }

  // Phones: the model and a 1080p decode do not share the device's memory and cores.
  if (lite && 'MutationObserver' in window) {
    var roots = [].slice.call(document.querySelectorAll('[data-playground], #lpRoot'));
    var busyNow = function () {
      return roots.some(function (r) { var s = r.getAttribute('data-state'); return s === 'loading' || s === 'writing'; });
    };
    var mo = new MutationObserver(function () {
      var busy = busyNow();
      if (busy === modelBusy) return;
      modelBusy = busy;
      kickAll();
    });
    roots.forEach(function (r) { mo.observe(r, { attributes: true, attributeFilter: ['data-state'] }); });
    modelBusy = busyNow();
  }

  // The front copy follows the back one. Checked on the back clip's own timeupdate (about four
  // times a second), so there is no loop of our own. The difference is taken round the loop, so
  // the two wrapping a few milliseconds apart does not read as a ten-second drift.
  if (front && back) {
    var FRAME = 1 / 24;
    var follow = function () {
      if (front.hidden || front.paused || back.paused || front.seeking || back.seeking) return;
      if (front.readyState < 3 || back.readyState < 3) return;
      var dur = back.duration, d = back.currentTime - front.currentTime;
      if (dur > 0) { if (d > dur / 2) d -= dur; else if (d < -dur / 2) d += dur; }
      var ad = Math.abs(d);
      if (ad > 0.5) { front.playbackRate = 1; front.currentTime = back.currentTime; }
      else if (ad > FRAME) front.playbackRate = 1 + Math.max(-0.1, Math.min(0.1, d));
      else if (front.playbackRate !== 1) front.playbackRate = 1;
    };
    ['timeupdate', 'seeked', 'playing'].forEach(function (evt) { back.addEventListener(evt, follow); });
    front.addEventListener('playing', follow);
  }

  document.addEventListener('visibilitychange', function () { if (!document.hidden) kickAll(); });
  window.addEventListener('pageshow', kickAll);
  window.addEventListener('focus', kickAll);
  // some browsers hold autoplay back until the visitor has interacted with the page
  window.addEventListener('touchstart', kickAll, { passive: true });
  window.addEventListener('click', kickAll);
  window.addEventListener('keydown', kickAll);
  if (rm && rm.addEventListener) rm.addEventListener('change', kickAll);
  if (narrowPortrait && narrowPortrait.addEventListener) narrowPortrait.addEventListener('change', placeFront);
})();
