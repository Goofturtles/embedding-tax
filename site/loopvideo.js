/* Seamless looping for the openings' clips, and a second copy that follows the first.

   The files are authored as loops (their first and last frames match), but the browser's own
   `loop` can stall for a frame while it rewinds, which reads as a stutter. So the wrap is taken
   over here: a few frames before the end the element is seeked back to the start, which the
   decoder can do without dropping a frame, and `loop` stays on as the fallback if a seek is
   refused. Nothing fades and nothing is redrawn: the cut lands on two identical frames.

   A video carrying data-follow="<id>" is a second copy of the same file stacked over the page
   (the research opening puts one in front of the headline, masked to the deck). It is held to
   the leader's clock: it is seeked whenever it drifts by more than a frame, so the two copies
   are one image, and it wraps with the leader rather than on its own.  */
(function () {
  'use strict';

  var TAIL = 0.08;    // seconds before the end at which the wrap is taken over
  var DRIFT = 0.04;   // a follower further than this from the leader is re-seeked (about a frame)

  function play(video) {
    var p = video.play();
    if (p && p.catch) p.catch(function () {});
  }

  function seamless(video) {
    var wrapping = false;
    video.addEventListener('timeupdate', function () {
      var d = video.duration;
      if (!isFinite(d) || d <= 0) return;
      if (video.currentTime < d - TAIL) { wrapping = false; return; }
      if (wrapping) return;
      wrapping = true;
      try { video.currentTime = 0; } catch (e) { /* the browser's own loop still covers it */ }
      play(video);
    });
  }

  function follow(video) {
    var leader = document.getElementById(video.getAttribute('data-follow'));
    if (!leader) return;
    var sync = function () {
      if (leader.paused || !isFinite(leader.currentTime)) return;
      if (Math.abs(video.currentTime - leader.currentTime) > DRIFT) {
        try { video.currentTime = leader.currentTime; } catch (e) { /* ignore a refused seek */ }
      }
    };
    leader.addEventListener('timeupdate', sync);
    leader.addEventListener('seeked', sync);
    var reveal = function () { video.classList.add('is-ready'); };
    if (video.readyState >= 3) reveal();
    else video.addEventListener('loadeddata', reveal, { once: true });
    play(video);
    sync();
  }

  var rm = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
  if (rm && rm.matches) return;   // the pages pause their clips there; nothing to keep in step

  var i, list = document.querySelectorAll('video[data-seamless]');
  for (i = 0; i < list.length; i++) seamless(list[i]);
  list = document.querySelectorAll('video[data-follow]');
  for (i = 0; i < list.length; i++) follow(list[i]);
})();
