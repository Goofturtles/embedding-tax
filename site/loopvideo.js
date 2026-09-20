/* Seamless looping for the openings' clips.

   The files are authored as loops (their first and last frames match), but a loop that rewinds
   one element always shows the rewind: the browser's own `loop`, and a seek back to zero taken
   just before the end, both froze the picture for a third of a second while the decoder started
   the file again. Measured on these pages: 330-480ms of held frame, every ten seconds.

   So nothing rewinds on screen. Each clip is given a twin. One plays while the other waits at
   the start; a few frames before the end the twin is started, it is swapped in on the frame the
   first one ends, and the one that just left rewinds out of sight and becomes the next twin. The
   visible element is always playing forward, and because the first and last frames are the same
   image the handover cannot be seen.

   Timing runs on requestVideoFrameCallback, which fires once per presented frame; where it is
   missing the work falls to requestAnimationFrame. Neither fires in a hidden tab, and the
   handover is what keeps these clips moving now that `loop` is off, so a visible tab, a restored
   page and a stalled element are all picked back up explicitly.

   A video carrying data-follow="<id>" is a second copy of the same file stacked over the page
   (the research opening puts one in front of the headline, masked to the deck, so the turntable
   occludes the type). It gets the same treatment, and is nudged back into step with its leader
   only if it drifts far, never across a handover.

   Clips pause while their opening is off screen: several autoplaying 1080p files is a lot of
   decoding to do for something nobody is looking at.  */
(function () {
  'use strict';

  var LEAD = 0.1;      // seconds before the end at which the twin is started
  var DRIFT = 0.25;    // a follower further than this from its leader is nudged back into step

  function play(video) {
    var p = video.play();
    if (p && p.catch) p.catch(function () {});
  }

  function seek(video, t) {
    try { video.currentTime = t; } catch (e) { /* ignore a refused seek */ }
  }

  // one tick per presented frame, per element
  function everyFrame(video, fn) {
    if (video.requestVideoFrameCallback) {
      var step = function () { fn(); video.requestVideoFrameCallback(step); };
      video.requestVideoFrameCallback(step);
      return;
    }
    var raf = function () { fn(); requestAnimationFrame(raf); };
    requestAnimationFrame(raf);
  }

  /* A pair built from one authored element: the element itself and a twin cloned from it. One is
     visible at a time; .lv-off hides the other. */
  function Pair(video) {
    var twin = video.cloneNode(false);
    twin.removeAttribute('id');
    twin.classList.add('lv-off');
    // the pages fade their clips in with .is-ready, so the twin must arrive already revealed or
    // the opening would drop to its poster at the first handover
    twin.classList.add('is-ready');
    twin.muted = true;                       // a cloned attribute is not always enough
    twin.loop = false;
    twin.autoplay = false;                   // the twin runs on its cue, not on arrival
    video.loop = false;                      // the handover is the loop now
    video.parentNode.insertBefore(twin, video.nextSibling);

    // the authored element is revealed the same way its twin already is, or the pair would pop
    // in at the first handover: the pages only reveal the clips they know about by id
    if (video.readyState >= 2) video.classList.add('is-ready');
    else video.addEventListener('loadeddata', function () { video.classList.add('is-ready'); }, { once: true });

    this.front = video;
    this.back = twin;
    this.armed = false;
    this.live = true;
    play(video);
  }

  Pair.prototype.clock = function () { return this.front.currentTime; };

  Pair.prototype.swap = function () {
    var front = this.front, back = this.back;
    back.classList.remove('lv-off');
    front.classList.add('lv-off');
    front.pause();
    seek(front, 0);                          // the rewind happens off screen
    this.front = back;
    this.back = front;
    this.armed = false;
  };

  Pair.prototype.tick = function () {
    if (!this.live) return;
    var front = this.front, back = this.back, d = front.duration;
    if (!isFinite(d) || d <= 0) return;      // a file that never loaded keeps its poster
    if (front.paused && !front.ended) return;
    var left = d - front.currentTime;
    if (left > LEAD && !front.ended) { this.armed = false; return; }
    if (!this.armed) {                       // start the twin so it is running before it is seen
      this.armed = true;
      if (back.currentTime > 0.05) seek(back, 0);
      play(back);
    }
    if (left > 0.03 && !front.ended) return; // hand over on the last frame, not before
    // only to something that is actually playing: a refused play would freeze the opening
    if (back.paused || back.readyState < 2) {
      if (front.ended) { seek(front, 0); play(front); this.armed = false; }   // the old way, as a backstop
      return;
    }
    this.swap();
  };

  Pair.prototype.pause = function () {
    this.live = false;
    this.armed = false;
    this.front.pause();
    this.back.pause();
  };

  Pair.prototype.resume = function () {
    this.live = true;
    play(this.front);
  };

  // nothing decodes for a visitor who asked for less motion: the pages hold their poster instead
  var rm = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
  var authored = [].slice.call(document.querySelectorAll('video[data-seamless]'));
  if (!authored.length) return;

  // The clips carry their file in data-src, so nothing is fetched until this runs: a visitor who
  // asked for less motion downloads no video at all, and keeps the poster the markup already has.
  var i;
  if (rm && rm.matches) return;
  for (i = 0; i < authored.length; i++) {
    var src = authored[i].getAttribute('data-src');
    if (src) authored[i].src = src;
  }

  var pairs = [], byId = {};
  for (i = 0; i < authored.length; i++) {
    var pair = new Pair(authored[i]);
    pairs.push(pair);
    if (authored[i].id) byId[authored[i].id] = pair;
    (function (p, el) {
      var tick = function () { p.tick(); };
      everyFrame(el, tick);
      everyFrame(p.back, tick);              // ticks continue after a handover
      el.addEventListener('ended', tick);    // a tab that was hidden gets no frame callbacks
      p.back.addEventListener('ended', tick);
      var leaderId = el.getAttribute('data-follow');
      if (!leaderId) return;
      everyFrame(el, function () {
        var lead = byId[leaderId];
        if (!lead || lead.front.paused || p.armed || lead.armed) return;   // never across a handover
        if (Math.abs(p.clock() - lead.clock()) > DRIFT) seek(p.front, lead.clock());
      });
    })(pair, authored[i]);
  }

  function kick() {
    for (var k = 0; k < pairs.length; k++) if (pairs[k].live) { play(pairs[k].front); pairs[k].tick(); }
  }
  // a hidden tab stops the frame callbacks, and a restored page starts from a paused element
  document.addEventListener('visibilitychange', function () { if (!document.hidden) kick(); });
  window.addEventListener('pageshow', kick);

  // pause while the opening is off screen, and pick it up again on the way back
  var section = authored[0].closest('section');
  if (section && 'IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      for (var k = 0; k < pairs.length; k++) {
        if (entries[0].isIntersecting) pairs[k].resume(); else pairs[k].pause();
      }
    }, { threshold: 0 }).observe(section);
  }
})();
