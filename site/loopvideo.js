/* Keep the openings' clips playing.

   The clips are plain autoplaying, looping video: the markup says autoplay, loop, muted and
   playsinline, and that is what plays them. This file only makes sure they keep going, because
   a browser will quietly leave one paused: autoplay can be refused until the visitor interacts
   with the page, a hidden tab suspends playback, a restored page comes back paused, and a stall
   on a slow connection can end with a paused element.

   So every clip is kicked on load, on the first touch or click, when the tab comes back, when
   the page is restored from the back/forward cache, and whenever one reports that it paused,
   stalled or ran out of data. Nothing else: no swapping, no seeking, no pausing off screen.

   Reduced motion is the one exception. The pages hide these clips there, so they are paused and
   their download is dropped rather than left buffering for something nobody will see.  */
(function () {
  'use strict';

  var videos = [].slice.call(document.querySelectorAll('.ap-art, .nx-art, .vy-media video, .vy-front'));
  if (!videos.length) return;

  var rm = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
  function reduced() { return !!(rm && rm.matches); }

  function kick(video) {
    if (reduced()) { video.pause(); return; }
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

  videos.forEach(function (video) {
    video.loop = true;                      // whatever the markup says, these clips repeat
    video.muted = true;                     // muted is what makes autoplay legal
    if (reduced()) {
      // the pages hide these clips entirely under reduced motion, so stop the download too
      video.pause();
      video.removeAttribute('src');
      video.load();
      return;
    }
    kick(video);
    // an element that reports trouble is simply started again, that element and no other
    ['pause', 'stalled', 'suspend', 'waiting', 'canplay', 'loadeddata'].forEach(function (evt) {
      video.addEventListener(evt, function () { kick(video); });
    });
  });

  document.addEventListener('visibilitychange', function () { if (!document.hidden) kickAll(); });
  window.addEventListener('pageshow', kickAll);
  window.addEventListener('focus', kickAll);
  // some browsers hold autoplay back until the visitor has interacted with the page
  window.addEventListener('touchstart', kickAll, { passive: true });
  window.addEventListener('click', kickAll);
  window.addEventListener('keydown', kickAll);
  if (rm && rm.addEventListener) rm.addEventListener('change', kickAll);
})();
