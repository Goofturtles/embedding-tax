/* Seamless looping for the openings' clips.

   The files are authored as loops (their first and last frames match), but a loop that rewinds
   one element always shows the rewind: the browser's own `loop`, and a seek back to zero taken
   just before the end, both freeze the picture for a third of a second while the decoder starts
   the file again. Measured on this page: 330-480ms of held frame, every ten seconds.

   So nothing rewinds on screen. Each clip is given a twin. One plays while the other waits at
   the start, and a quarter of a second before the end the twin is started and swapped in on the
   frame the first one ends; the one that just left the screen then rewinds, out of sight, and
   becomes the next twin. The visible element is therefore always playing forward, and because
   the first and last frames are the same image the handover is invisible.

   Timing runs on requestVideoFrameCallback, which fires once per presented frame, so the swap
   lands within a frame of the end. Where it is missing the work falls to requestAnimationFrame.

   A video carrying data-follow="<id>" is a second copy of the same file stacked over the page
   (the research opening puts one in front of the headline, masked to the deck, so the turntable
   occludes the type). It gets the same treatment and starts with its leader, and is nudged back
   into step only if it drifts far, never across a handover.

   Clips pause while their section is off screen: several autoplaying 1080p files is a lot of
   decoding to do for something nobody is looking at.  */
(function () {
  'use strict';

  var LEAD = 0.25;     // seconds before the end at which the twin is started
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

  /* A pair built from one authored element: the element itself and a twin cloned from it. Only
     one is visible at a time, and .lv-off is what hides the other. */
  function Pair(video) {
    var twin = video.cloneNode(false);
    twin.removeAttribute('id');
    twin.classList.add('lv-off');
    twin.muted = true;                       // a cloned attribute is not always enough
    twin.loop = false;
    video.loop = false;                      // the handover is the loop now
    video.parentNode.insertBefore(twin, video.nextSibling);

    this.front = video;
    this.back = twin;
    this.armed = false;
    play(video);
  }

  Pair.prototype.clock = function () { return this.front.currentTime; };

  Pair.prototype.tick = function () {
    var front = this.front, back = this.back, d = front.duration;
    if (!isFinite(d) || d <= 0 || front.paused) return;
    var left = d - front.currentTime;
    if (left > LEAD) { this.armed = false; return; }
    if (!this.armed) {                       // start the twin so it is running before it is seen
      this.armed = true;
      if (back.currentTime > 0.05) seek(back, 0);
      play(back);
      return;
    }
    if (left > 0.05 && !front.ended) return; // hand over on the last frame, not before
    back.classList.remove('lv-off');
    front.classList.add('lv-off');
    front.pause();
    seek(front, 0);                          // the rewind happens off screen
    this.front = back;
    this.back = front;
    this.armed = false;
  };

  Pair.prototype.pause = function () { this.front.pause(); this.back.pause(); };
  Pair.prototype.resume = function () { play(this.front); };

  // nothing decodes for a visitor who asked for less motion: the pages hold their poster instead
  var rm = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
  if (rm && rm.matches) return;

  var authored = [].slice.call(document.querySelectorAll('video[data-seamless]'));
  if (!authored.length) return;

  var pairs = [], byId = {}, i;
  for (i = 0; i < authored.length; i++) {
    var pair = new Pair(authored[i]);
    pairs.push(pair);
    if (authored[i].id) byId[authored[i].id] = pair;
    (function (p, el) {
      everyFrame(el, function () { p.tick(); });
      everyFrame(p.back, function () { p.tick(); });   // ticks continue after a handover
      var leaderId = el.getAttribute('data-follow');
      if (!leaderId) return;
      var reveal = function () { el.classList.add('is-ready'); };
      if (el.readyState >= 3) reveal(); else el.addEventListener('loadeddata', reveal, { once: true });
      everyFrame(el, function () {
        var lead = byId[leaderId];
        if (!lead || lead.front.paused || p.armed || lead.armed) return;   // never across a handover
        if (Math.abs(p.clock() - lead.clock()) > DRIFT) seek(p.front, lead.clock());
      });
    })(pair, authored[i]);
  }

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
