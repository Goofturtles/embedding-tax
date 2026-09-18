/* Measured results, and only measured results.

   Every value here comes from site/results.json, which is written exclusively by
   tools/publish_results.py from the training log and the lm-eval output. Nothing on
   this page is typed by hand, because the thing that kept going wrong on this project
   was prose claiming a finished run while the data file was still all nulls.

   The rule this file enforces: a null renders as "not measured yet". Never a dash that
   could read as a zero, never a blank cell, never a plausible-looking placeholder. The run
   is called complete only when results.json says run_complete === true. */
(function () {
  'use strict';

  /* ---------- tiny shared helpers (explorer.js keeps its own copies on purpose, so the two
     scripts stay independent of each other) ---------- */
  var $ = function (id) { return document.getElementById(id); };
  var UNMEASURED = 'not measured yet';
  var MINUS = '−';
  var RM = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  function reduced() { return !!(RM && RM.matches); }
  function fin(n) { return typeof n === 'number' && isFinite(n); }

  /* Formats a finite number or returns null. typeof, not Number(): "" and true both coerce
     to numbers and used to print as "0 tok/s" and "1.0". */
  function fmt(n, digits) {
    if (!fin(n)) return null;
    return n.toLocaleString('en-US', { minimumFractionDigits: digits || 0, maximumFractionDigits: digits || 0 });
  }

  /* Numeric coercion before anything reaches innerHTML. Every other value on this page goes
     in via textContent; the chart and the tables are built as strings, so a non-numeric value
     from results.json would otherwise be interpolated as markup. */
  function num(n) { return fin(n) ? n.toLocaleString('en-US') : '?'; }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }

  /* Write a value or the honest absence of one. Returns true if a number landed. */
  function put(id, value, unit) {
    var node = $(id);
    if (!node) return false;
    node.textContent = '';
    if (value === null || value === undefined) {
      node.textContent = UNMEASURED;
      node.classList.add('is-unmeasured');
      return false;
    }
    node.classList.remove('is-unmeasured');
    node.appendChild(document.createTextNode(value));
    if (unit) node.appendChild(el('span', 'c-unit', unit));
    return true;
  }

  function sub(id, text) { var node = $(id); if (node) node.textContent = text || ''; }
  function set(id, text) { var node = $(id); if (node) node.textContent = text; }
  function seg(id, on) { var node = $(id); if (node) node.hidden = !on; }
  function bn(v) { return (v / 1e9).toFixed(2) + 'B'; }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); }

  /* ---------- status ----------
     One chip in words, then the facts stated once, then a bar drawn at the real share and
     left alone. The bar is data, never a loader: no denominator, no bar. */

  function bar(pct, done) {
    var b = $('runBar');
    if (!b) return;
    if (!fin(pct)) { b.hidden = true; b.removeAttribute('aria-valuenow'); b.removeAttribute('aria-valuetext'); return; }
    b.hidden = false;
    b.classList.toggle('is-done', !!done);
    b.setAttribute('aria-valuenow', pct.toFixed(1));
    b.setAttribute('aria-valuetext', pct.toFixed(1) + '% of the planned training tokens');
    // first paint at 0, then the fill eases to its value (a transform, .5s); instant under reduce
    if (reduced()) { b.style.setProperty('--rs-p', (pct / 100).toFixed(4)); return; }
    b.style.setProperty('--rs-p', '0');
    requestAnimationFrame(function () { requestAnimationFrame(function () { b.style.setProperty('--rs-p', (pct / 100).toFixed(4)); }); });
  }

  function status(t) {
    var line = $('runStatus');
    if (!line) return;
    // typeof, not a null check: a string count passed that and printed "null of ... (NaN%)".
    if (!t || !fin(t.tokens_seen)) {
      bar(null);
      set('runState', 'No run recorded');
      seg('runPctSeg', false); seg('runStepSeg', false);
      line.textContent = 'No training run has been recorded yet.';
      return;
    }
    var complete = t.run_complete === true;
    // The denominator comes from the log (tokens_target: the run's --tokens, aligned to whole
    // steps by publish_results) or it is not shown at all.
    var target = fin(t.tokens_target) && t.tokens_target > 0 ? t.tokens_target : null;
    // Clamped: an over-run past the target should read 100%, not 104.2%.
    var pct = target ? Math.min(100, 100 * t.tokens_seen / target) : null;

    set('runState', complete ? 'Run complete' : 'In progress');
    bar(complete && !target ? null : pct, complete);
    // progress is stated once: tokens of planned tokens and the share, on this line
    if (target) {
      set('runPct', bn(t.tokens_seen) + ' of ' + bn(target) + ' planned tokens · ' + pct.toFixed(1) + '%');
      seg('runPctSeg', true);
    } else {
      set('runPct', bn(t.tokens_seen) + ' tokens seen');
      seg('runPctSeg', true);
    }
    if (fin(t.steps)) { set('runStep', num(t.steps)); seg('runStepSeg', true); }
    else seg('runStepSeg', false);

    // publish_results writes NaN/inf as null (JSON cannot carry them) and counts them, so a
    // diverged run says so here instead of quietly showing gaps.
    line.textContent = fin(t.nonfinite_values) && t.nonfinite_values > 0
      ? t.nonfinite_values + ' measurement(s) were not finite, which usually means the run ' +
        'diverged; they show as not measured.'
      : '';
  }

  /* ---------- the curve ----------
     The x axis runs to the whole planned run and the line stops where the run is, so the part
     that has not happened is empty plot rather than an invented projection. */

  var PAD = { t: 34, r: 56, b: 34, l: 8 };
  var KEEP = 3;
  var lastCurve = null, lastT = null, lastW = 0, tableOpen = false;
  var geom = null, sel = -1, armed = false, drawn = false;

  function plotWidth(host) { return Math.max(280, Math.round(host.clientWidth) || 640); }

  function niceTicks(lo, hi) {
    var span = Math.max(hi - lo, 0.01);
    var steps = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5];
    var step = steps[steps.length - 1];
    for (var i = 0; i < steps.length; i++) if (span / steps[i] <= 4) { step = steps[i]; break; }
    var y0 = Math.floor(lo / step) * step, y1 = Math.ceil(hi / step) * step;
    var out = [];
    for (var v = y0; v <= y1 + step / 2; v += step) out.push(Number(v.toFixed(4)));
    return { y0: y0, y1: y1, ticks: out };
  }

  function moreLabel(n) { return tableOpen ? 'Show fewer' : 'Show all ' + n + ' evaluations'; }

  function curve(points, svgOnly, t) {
    var host = $('curveWrap');
    if (!host) return;
    lastCurve = points;
    if (t) lastT = t;
    t = lastT || {};
    var W = plotWidth(host);
    var H = W < 640 ? 220 : 280;
    // Drop any point without a real, finite loss and token count, rather than let one bad
    // record throw on .toFixed and blank the whole card.
    points = (Array.isArray(points) ? points : []).filter(function (p) {
      return p && fin(p.loss) && fin(p.tokens);
    });
    if (points.length < 2) {
      host.textContent = '';
      host.appendChild(el('p', 'lc-empty', points.length === 1
        ? 'One validation point so far. The curve draws once there are two.'
        : 'No evaluations published yet'));
      return;
    }

    var ys = points.map(function (p) { return p.loss; });
    var scale = niceTicks(Math.min.apply(null, ys), Math.max.apply(null, ys));
    var last = points[points.length - 1], first = points[0];
    var complete = t.run_complete === true;
    var target = fin(t.tokens_target) && t.tokens_target > 0 ? t.tokens_target : null;
    var xMax = target ? Math.max(target, last.tokens) : last.tokens;

    var iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;
    var sx = function (v) { return PAD.l + (v / xMax) * iw; };
    var sy = function (v) { return PAD.t + (1 - (v - scale.y0) / (scale.y1 - scale.y0)) * ih; };

    var sxs = points.map(function (p) { return sx(p.tokens); });
    var sys = points.map(function (p) { return sy(p.loss); });
    var best = points.reduce(function (a, p) { return p.loss < a.loss ? p : a; }, points[0]);
    var bestIsLast = best === last;
    geom = { W: W, H: H, sxs: sxs, sys: sys, points: points, best: best, first: first };

    var d = points.map(function (p, i) {
      return (i ? 'L' : 'M') + sxs[i].toFixed(1) + ' ' + sys[i].toFixed(1);
    }).join(' ');
    var area = d + ' L' + sxs[sxs.length - 1].toFixed(1) + ' ' + (PAD.t + ih) + ' L' +
      sxs[0].toFixed(1) + ' ' + (PAD.t + ih) + ' Z';

    // a) gridlines with the value labels on the right edge
    var grid = scale.ticks.map(function (v) {
      var y = sy(v).toFixed(1);
      return '<line x1="' + PAD.l + '" y1="' + y + '" x2="' + (W - PAD.r) + '" y2="' + y + '" class="lc-grid"/>' +
        '<text x="' + (W - PAD.r + 8) + '" y="' + (sy(v) + 4).toFixed(1) + '" class="lc-lbl lc-lbl-r">' + v.toFixed(2) + '</text>';
    }).join('');

    // b) the baseline: solid as far as the run has come, dashed for what is still to do
    var yB = (PAD.t + ih).toFixed(1), xNow = sx(last.tokens);
    var base = '<line x1="' + PAD.l + '" y1="' + yB + '" x2="' + xNow.toFixed(1) + '" y2="' + yB + '" class="lc-base"/>';
    var unrun = !!(target && last.tokens < target && !complete);
    if (unrun) {
      base += '<line x1="' + xNow.toFixed(1) + '" y1="' + yB + '" x2="' + sx(xMax).toFixed(1) + '" y2="' + yB + '" class="lc-base-todo"/>';
    }

    // c) x labels every half billion, with the end of the plan anchored to the right
    var xl = '', stepX = W < 1000 ? 1e9 : 5e8;
    for (var v = 0; v <= xMax - stepX / 2; v += stepX) {
      if (sx(v) > sx(xMax) - 70) continue;
      xl += '<text x="' + sx(v).toFixed(1) + '" y="' + (PAD.t + ih + 20) + '" class="lc-lbl ' +
        (v === 0 ? '' : 'lc-lbl-mid') + '">' + (v === 0 ? '0' : bn(v)) + '</text>';
    }
    xl += '<text x="' + sx(xMax).toFixed(1) + '" y="' + (PAD.t + ih + 20) + '" class="lc-lbl lc-lbl-end">' +
      bn(xMax) + (unrun ? ' planned end' : '') + '</text>';

    // d) the run so far, and where it stopped: the now-marker carries the step only
    var mark = '<g class="lc-mark"><line x1="' + xNow.toFixed(1) + '" y1="' + PAD.t + '" x2="' + xNow.toFixed(1) +
      '" y2="' + yB + '" class="lc-now"/>' +
      '<text x="' + xNow.toFixed(1) + '" y="' + (PAD.t - 12) + '" class="lc-lbl lc-strong ' +
      (xNow > W - PAD.r - 60 ? 'lc-lbl-end' : 'lc-lbl-mid') + '">step ' + num(last.step) + '</text></g>';

    // e) the unrun remainder, named rather than drawn; omitted once the run is complete
    if (unrun) {
      var gapW = sx(xMax) - xNow;
      if (gapW >= 140) {
        var cx0 = (xNow + sx(xMax)) / 2;
        mark += '<g class="lc-mark"><text x="' + cx0.toFixed(1) + '" y="' + (PAD.t + ih / 2 - 4) + '" class="lc-todo">Not run yet</text>' +
          '<text x="' + cx0.toFixed(1) + '" y="' + (PAD.t + ih / 2 + 14) + '" class="lc-todo-sub">' +
          bn(target - last.tokens) + ' tokens to go</text></g>';
      }
    }

    // f) the best evaluation, clipped to the run so far, only while it is not the last point
    var bestLine = '';
    if (!bestIsLast) {
      var by = sy(best.loss).toFixed(1);
      bestLine = '<g class="lc-mark"><line x1="' + PAD.l + '" y1="' + by + '" x2="' + xNow.toFixed(1) +
        '" y2="' + by + '" class="lc-best-line"/>' +
        '<text x="' + (xNow - 6).toFixed(1) + '" y="' + (Number(by) - 6) + '" class="lc-lbl lc-strong lc-lbl-end">Best ' +
        best.loss.toFixed(4) + '</text></g>';
    }

    // The direction is read off the data, never assumed.
    var delta = last.loss - first.loss;
    var verb = delta < -0.0005 ? 'falling to' : delta > 0.0005 ? 'rising to' : 'flat at';
    var alt = 'Validation loss against tokens seen: ' + first.loss.toFixed(3) + ' at ' +
      bn(first.tokens) + ' tokens, ' + verb + ' ' + last.loss.toFixed(3) + ' at ' + bn(last.tokens) +
      ', across ' + points.length + ' measured points' +
      (target ? ', on an axis running to the planned ' + bn(target) + '.' : '.');

    var svgHtml =
      '<svg viewBox="0 0 ' + W + ' ' + H + '" class="lc-svg" role="img" aria-label="' + esc(alt) + '">' +
        grid + base + xl +
        '<path d="' + area + '" class="lc-area"/>' +
        '<path d="' + d + '" class="lc-line"/>' +
        mark + bestLine +
        '<g class="lc-end"><circle cx="' + sxs[sxs.length - 1].toFixed(1) + '" cy="' + sys[sys.length - 1].toFixed(1) +
          '" r="4.5" class="lc-dot"/></g>' +
        '<g class="lc-hover" id="lcHover">' +
          '<line y1="' + PAD.t + '" y2="' + yB + '" class="lc-cursor"/>' +
          '<circle r="4.5" class="lc-dot"/>' +
          '<g id="lcChip"><rect x="-65" y="0" width="130" height="20" rx="6" class="lc-chip"/>' +
          '<text y="14" class="lc-chip-t"></text></g>' +
        '</g>' +
        '<rect x="' + PAD.l + '" y="' + PAD.t + '" width="' + iw + '" height="' + ih + '" class="lc-hit"/>' +
      '</svg>' +
      '<div class="lc-curtain" aria-hidden="true"></div>';
    lastW = W;

    // On a resize only the plot is rebuilt. Rebuilding the table as well reset its sideways
    // scroll and dropped keyboard focus to <body> if it was inside. The scrub handlers live on
    // the wrapper, which survives, and a plot that has already drawn stays drawn.
    var plot = host.querySelector('.lc-plot');
    if (svgOnly && plot && host.querySelector('.lc-table')) {
      plot.innerHTML = svgHtml;
      plot.classList.toggle('is-armed', armed);
      plot.classList.toggle('is-drawn', armed && drawn);
      select(points.length - 1, false);
      return;
    }

    host.textContent = '';
    var wrap = el('div', 'lc-plot');
    wrap.id = 'curvePlot';
    wrap.tabIndex = 0;
    wrap.setAttribute('role', 'slider');
    wrap.setAttribute('aria-label', 'Validation loss chart. Arrow keys read each measured point; the table below lists them all.');
    wrap.setAttribute('aria-valuemin', '0');
    wrap.setAttribute('aria-valuemax', String(points.length - 1));
    wrap.setAttribute('aria-orientation', 'horizontal');
    wrap.innerHTML = svgHtml;
    host.appendChild(wrap);
    bindScrub(wrap);

    // The chart is decoration over this table, not a replacement for it. Newest first.
    var twrap = el('div', 'lc-tablewrap');
    var table = el('table', 'lc-table');
    table.innerHTML = tableHtml(points, best);
    twrap.appendChild(table);
    host.appendChild(twrap);
    var others = points.length;
    if (points.length > KEEP) {
      var btn = el('button', 'btn btn--tertiary btn--sm lc-more');
      btn.type = 'button';
      btn.setAttribute('aria-controls', 'valRows');
      btn.setAttribute('aria-expanded', String(tableOpen));
      btn.appendChild(el('span', null, moreLabel(others)));
      btn.insertAdjacentHTML('beforeend', '<svg class="chev" width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>');
      btn.addEventListener('click', function () {
        tableOpen = !tableOpen;
        var trs = $('valRows').rows;
        for (var r = KEEP; r < trs.length; r++) trs[r].hidden = !tableOpen;
        btn.setAttribute('aria-expanded', String(tableOpen));
        btn.firstChild.textContent = moreLabel(others);
      });
      host.appendChild(btn);
    }
    bindRows();

    // One-time reveal, the first time the chart is scrolled into view: the ivory curtain
    // slides off the plot and the end dot lands. Never under reduced motion.
    if (!reduced() && window.IntersectionObserver) {
      armed = true; drawn = false;
      wrap.classList.add('is-armed');
      var io = new IntersectionObserver(function (entries) {
        if (!entries[0].isIntersecting) return;
        io.disconnect();
        drawn = true;
        wrap.classList.add('is-drawn');
      }, { threshold: 0.3 });
      io.observe(wrap);
    }
    select(points.length - 1, false);
  }

  function tableHtml(points, best) {
    var desc = points.slice().reverse();
    var rows = desc.map(function (p, i) {
      var prev = desc[i + 1];
      var ch = prev ? p.loss - prev.loss : null;
      // The arrow carries the direction, so the change never rests on colour; a rise is red.
      var chCell = ch === null ? '<span class="lc-first">first evaluation</span>'
        : (ch <= 0 ? MINUS + Math.abs(ch).toFixed(4) : '<span class="lc-rise">+' + Math.abs(ch).toFixed(4) + '</span>');
      var tok = fmt(p.tokens), ppl = fmt(p.ppl, 1);
      return '<tr data-i="' + (points.length - 1 - i) + '"' + (i >= KEEP && !tableOpen ? ' hidden' : '') + '>' +
        '<th scope="row">' + num(p.step) +
          '<span class="lc-rowsub">' + (tok ? bn(p.tokens) + ' tokens' : UNMEASURED) + '</span></th>' +
        '<td class="num lc-hide">' + (tok === null ? UNMEASURED : tok) + '</td>' +
        '<td class="num lc-loss">' + p.loss.toFixed(4) +
          (p === best ? '<span class="chip chip--layers lc-tag">Best</span>' : '') +
          '<span class="lc-rowsub">' + (ppl ? 'ppl ' + ppl : UNMEASURED) + '</span></td>' +
        '<td class="num lc-hide">' + (ppl === null ? UNMEASURED : ppl) + '</td>' +
        '<td class="num">' + chCell + '</td></tr>';
    }).join('');
    return '<caption class="sr-only">Every validation evaluation: step, tokens seen, validation loss, perplexity and the change since the previous one.</caption>' +
      '<thead><tr><th scope="col">Step</th><th scope="col" class="num lc-hide">Tokens seen</th>' +
      '<th scope="col" class="num">Validation loss</th><th scope="col" class="num lc-hide">Perplexity</th>' +
      '<th scope="col" class="num">Change</th></tr></thead><tbody id="valRows">' + rows + '</tbody>';
  }

  /* The header readouts double as the chart's readout: hovering, or arrowing through the
     points, moves the cursor and rewrites them. Nothing is interpolated: the cursor only ever
     sits on a measured point. */
  function select(i, announce) {
    if (!geom || !geom.points.length) return;
    i = Math.max(0, Math.min(geom.points.length - 1, i));
    sel = i;
    var p = geom.points[i], isBest = p === geom.best;
    put('lcVal', p.loss.toFixed(4));
    var bits = ['Step ' + num(p.step), bn(p.tokens) + ' tokens'];
    if (fin(p.ppl)) bits.push('perplexity ' + fmt(p.ppl, 1) + ' per token');
    if (isBest) bits.push('best so far');
    set('lcSub', bits.join(' · '));

    var cmp = $('lcCmp');
    if (cmp && geom.first !== p) {
      cmp.hidden = false;
      var d = p.loss - geom.first.loss;
      set('lcDelta', (d <= 0 ? MINUS : '+') + Math.abs(d).toFixed(4));
      set('lcDeltaSub', 'from ' + geom.first.loss.toFixed(4) + ' at step ' + num(geom.first.step));
    } else if (cmp) {
      cmp.hidden = true;
    }

    var wrap = $('curvePlot');
    if (wrap) {
      wrap.setAttribute('aria-valuenow', String(i));
      wrap.setAttribute('aria-valuetext', 'Step ' + num(p.step) + ', validation loss ' + p.loss.toFixed(4));
    }

    var hover = $('lcHover');
    if (hover) {
      var on = i !== geom.points.length - 1 || hovering;
      hover.classList.toggle('is-on', on);
      var x = geom.sxs[i], y = geom.sys[i];
      hover.querySelector('line').setAttribute('transform', 'translate(' + x.toFixed(1) + ',0)');
      hover.querySelector('circle').setAttribute('transform', 'translate(' + x.toFixed(1) + ',' + y.toFixed(1) + ')');
      var chip = $('lcChip');
      if (chip) {
        var cx = Math.max(66, Math.min(geom.W - 66, x));
        chip.setAttribute('transform', 'translate(' + cx.toFixed(1) + ',' + (geom.H - PAD.b + 4) + ')');
        chip.querySelector('text').textContent = bn(p.tokens) + ' · step ' + num(p.step);
      }
    }

    var rows = $('valRows');
    if (rows) {
      for (var r = 0; r < rows.rows.length; r++) {
        rows.rows[r].classList.toggle('is-hot', Number(rows.rows[r].getAttribute('data-i')) === i);
      }
    }
    if (announce) liveSoon(p);
  }

  var liveTimer = 0;
  function liveSoon(p) {
    clearTimeout(liveTimer);
    liveTimer = setTimeout(function () {
      var node = $('lcLive');
      if (!node) return;
      node.textContent = 'Step ' + num(p.step) + ': validation loss ' + p.loss.toFixed(4) + ', ' +
        bn(p.tokens) + ' tokens' + (fin(p.ppl) ? ', perplexity ' + fmt(p.ppl, 1) : '') + '.';
    }, 250);
  }

  var hovering = false;
  function nearest(x) {
    var lo = 0, hi = geom.sxs.length - 1;
    while (hi - lo > 1) {
      var mid = (lo + hi) >> 1;
      if (geom.sxs[mid] < x) lo = mid; else hi = mid;
    }
    return (x - geom.sxs[lo] <= geom.sxs[hi] - x) ? lo : hi;
  }

  function bindScrub(wrap) {
    var rect = null;
    var read = function (e) {
      if (!rect) rect = wrap.getBoundingClientRect();   // read once per gesture, never per move
      if (!rect.width || !geom) return null;
      return (e.clientX - rect.left) * (geom.W / rect.width);
    };
    wrap.addEventListener('pointerenter', function (e) {
      rect = null; hovering = true;
      var x = read(e);
      if (x !== null) select(nearest(x), false);
    });
    wrap.addEventListener('pointermove', function (e) {
      var x = read(e);
      if (x !== null) select(nearest(x), false);
    });
    var off = function () {
      rect = null; hovering = false;
      if (geom) select(geom.points.length - 1, false);
    };
    wrap.addEventListener('pointerleave', off);
    wrap.addEventListener('pointercancel', off);
    wrap.addEventListener('blur', off);
    wrap.addEventListener('focus', function () { hovering = true; if (geom) select(geom.points.length - 1, false); });
    wrap.addEventListener('keydown', function (e) {
      if (!geom) return;
      var n = geom.points.length, i = sel;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') i = sel - 1;
      else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') i = sel + 1;
      else if (e.key === 'Home') i = 0;
      else if (e.key === 'End') i = n - 1;
      else if (e.key === 'PageDown') i = sel - 5;
      else if (e.key === 'PageUp') i = sel + 5;
      else if (e.key === 'Escape') { off(); wrap.blur(); return; }
      else return;
      e.preventDefault();
      hovering = true;
      select(i, true);
    });
    wrap.addEventListener('pointerdown', function (e) { rect = null; read(e); });
  }

  function bindRows() {
    var rows = $('valRows');
    if (!rows) return;
    rows.addEventListener('pointerover', function (e) {
      var tr = e.target.closest ? e.target.closest('tr[data-i]') : null;
      if (tr) { hovering = true; select(Number(tr.getAttribute('data-i')), false); }
    });
    rows.addEventListener('pointerleave', function () {
      hovering = false;
      if (geom) select(geom.points.length - 1, false);
    });
  }

  /* ---------- benchmarks ----------
     One row per task on a shared 0-100 axis: a chance tick, the GPT-2 mark, and our own mark
     only when a real score exists. A pending score is written in words, at full row weight.
     The best value in a row is bold whoever holds it. */

  var TASKS = [
    ['arc_easy', 'ARC-Easy', 'grade-school science · 4 choices'],
    ['piqa', 'PIQA', 'physical common sense · 2 choices'],
    ['hellaswag', 'HellaSwag', 'sentence endings · 4 choices'],
    ['winogrande', 'WinoGrande', 'pronoun resolution · 2 choices']
  ];

  function clamp100(v) { return Math.max(0, Math.min(100, v)); }

  function deltaText(score, chance) {
    if (!fin(score) || !fin(chance)) return '';
    var d = score - chance;
    if (Math.abs(d) < 2) return 'at chance';
    return (d > 0 ? '+' + d.toFixed(1) + ' over chance' : MINUS + Math.abs(d).toFixed(1) + ' below chance');
  }

  function valueCell(v, cls, chance, label, best, withDelta) {
    var txt = fin(v) ? v.toFixed(1) : null;
    if (txt === null) {
      return '<td role="cell" class="num is-pending" data-label="' + label + '"><span class="bm-v is-unmeasured">' + UNMEASURED + '</span></td>';
    }
    var d = withDelta ? deltaText(v, chance) : '';
    var neg = d.charAt(0) === MINUS;
    return '<td role="cell" class="num" data-label="' + label + '">' +
      '<span class="bm-v ' + cls + (best ? ' is-best' : '') + '">' + txt + '</span>' +
      (d ? '<span class="bm-d' + (neg ? ' is-neg' : '') + '">' + d + '</span>' : '') +
      '</td>';
  }

  function benchmarks(b) {
    var body = $('benchRows');
    if (!body) return;
    var any = false;
    var html = TASKS.map(function (task, i) {
      var r = b[task[0]] || {};
      var ours = fin(r.ours) ? r.ours : null;
      var ref = fin(r.gpt2_small) ? r.gpt2_small : null;
      var ch = fin(r.chance) ? r.chance : null;
      if (ours !== null) any = true;
      var bestV = Math.max(ours === null ? -Infinity : ours, ref === null ? -Infinity : ref);
      var near = ours !== null && ref !== null && Math.abs(ours - ref) <= 6;

      var plot = '<div class="bm-plot" aria-hidden="true" style="--bm-d:' + (i * 80) + 'ms">';
      if (ch !== null) plot += '<i class="bm-tick" style="--x:' + clamp100(ch) + '"></i>';
      if (ref !== null) plot += '<i class="bm-dot is-ref" style="--x:' + clamp100(ref) + '"></i>';
      if (ours !== null) plot += '<i class="bm-dot is-ours' + (near ? ' is-stemmed' : '') + '" style="--x:' + clamp100(ours) + '"></i>';
      plot += '</div>';

      return '<tr role="row"><th scope="row" role="rowheader"><span class="bm-task">' + task[1] +
        '</span><span class="bm-note">' + task[2] + '</span></th>' +
        '<td role="cell" class="bm-plotcell">' + plot + '</td>' +
        valueCell(ours, 'is-ours', ch, 'This model', ours !== null && ours >= bestV, true) +
        valueCell(ref, 'is-ref', ch, 'GPT-2 Small', ref !== null && ref >= bestV, false) +
        (ch === null
          ? '<td role="cell" class="num" data-label="Chance"><span class="bm-v is-unmeasured">' + UNMEASURED + '</span></td>'
          : '<td role="cell" class="num" data-label="Chance"><span class="bm-v is-ch">' + ch.toFixed(1) + '</span></td>') +
        '</tr>';
    }).join('');
    body.innerHTML = html;

    // The WikiText figures count as measured too.
    if (put('ppl', fmt(b.wikitext103_ppl, 2))) any = true;
    if (put('bpb', fmt(b.wikitext103_bits_per_byte, 4))) any = true;

    var note = $('benchNote');
    if (!note) return;
    if (!any) {
      note.textContent = 'No benchmark has been run against a checkpoint yet. These rows fill in ' +
        'from lm-evaluation-harness output, and from nothing else. Random chance is drawn on ' +
        'every row because at this size two of these tasks are expected to sit near chance.';
    } else if (fin(b._steps_completed)) {
      // Which weights produced a score is part of the score.
      note.textContent = 'Scored with lm-evaluation-harness on the checkpoint after ' +
        num(b._steps_completed) + ' training steps. Random chance is drawn on every row ' +
        'because at this size two of these tasks are expected to sit near chance.';
    }
  }

  /* ---------- samples ----------
     One sample at a time behind a Previous / Next pager. Built as DOM nodes throughout:
     completions are arbitrary model text and must never be parsed as markup. */

  var ICON_COPY = '<svg class="s-ic-copy" width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><rect x="5.5" y="5.5" width="8" height="8" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M10.5 5V3.5A1.5 1.5 0 0 0 9 2H4A1.5 1.5 0 0 0 2.5 3.5V9A1.5 1.5 0 0 0 4 10.5h1.5" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>';
  var ICON_DONE = '<svg class="s-ic-done" width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  function copyButton(text, n) {
    var b = el('button', 's-copy');
    b.type = 'button';
    var label = 'Copy continuation ' + n;
    b.setAttribute('aria-label', label);
    b.innerHTML = ICON_COPY + ICON_DONE;          // fixed markup; the model text stays in the closure
    var timer = 0;
    b.addEventListener('click', function () {
      navigator.clipboard.writeText(text).then(function () {
        b.classList.add('is-copied');
        b.setAttribute('aria-label', 'Copied');
        clearTimeout(timer);
        timer = setTimeout(function () {
          b.classList.remove('is-copied');
          b.setAttribute('aria-label', label);
        }, 1500);
      }, function () {});
    });
    return b;
  }

  function sampleRow(x, i, canCopy) {
    var li = el('li', 's-row');
    var left = el('div', 's-col');
    left.appendChild(el('p', 's-role', 'Prompt'));
    left.appendChild(el('p', 's-prompt', x.prompt));
    var right = el('div', 's-col');
    right.appendChild(el('p', 's-role', 'Model continues'));
    var out = el('div', 's-out');
    out.appendChild(el('pre', null, x.completion));
    if (canCopy) out.appendChild(copyButton(x.completion, i + 1));
    right.appendChild(out);
    li.appendChild(left);
    li.appendChild(right);
    return li;
  }

  function samples(s) {
    var list = $('sRows');
    var items = s && Array.isArray(s.items) ? s.items.filter(function (x) {
      return x && typeof x.prompt === 'string' && typeof x.completion === 'string';
    }) : [];
    if (!list || !items.length) return;          // keep the static "not generated yet" state
    var canCopy = !!(navigator.clipboard && window.isSecureContext);
    var page = 0, swapping = 0;

    function show(n, instant) {
      page = Math.max(0, Math.min(items.length - 1, n));
      var paint = function () {
        list.textContent = '';
        list.appendChild(sampleRow(items[page], page, canCopy));
        set('sCount', 'Sample ' + (page + 1) + ' of ' + items.length);
        set('sPage', (page + 1) + ' of ' + items.length);
        var prev = $('sPrev'), next = $('sNext');
        if (prev) prev.setAttribute('aria-disabled', String(page === 0));
        if (next) next.setAttribute('aria-disabled', String(page === items.length - 1));
      };
      var old = list.firstElementChild;
      if (instant || reduced() || !old) { paint(); return; }
      // outgoing opacity .1s, incoming .2s; the list's min-height holds the layout
      old.classList.add('is-out');
      clearTimeout(swapping);
      swapping = setTimeout(paint, 110);
    }
    var pager = $('sPager');
    if (pager && items.length > 1) {
      pager.hidden = false;
      var prev = $('sPrev'), next = $('sNext');
      if (prev) prev.addEventListener('click', function () { if (prev.getAttribute('aria-disabled') !== 'true') show(page - 1); });
      if (next) next.addEventListener('click', function () { if (next.getAttribute('aria-disabled') !== 'true') show(page + 1); });
    }
    show(0, true);

    var dl = $('sSettings');
    if (dl) {
      var st = s.settings || {}, rt = s.runtime || {};
      var n = function (v) { return fin(v) ? String(v) : null; };
      var pair = function (k, v) {
        var box = el('div');
        box.appendChild(el('dt', null, k));
        var dd = el('dd');
        if (v === null) { dd.textContent = UNMEASURED; dd.className = 'is-unmeasured'; }
        else dd.textContent = v;
        box.appendChild(dd);
        dl.appendChild(box);
      };
      dl.textContent = '';
      pair('temperature', n(st.temperature));
      pair('top-k', n(st.top_k));
      pair('max new tokens', n(st.max_new_tokens));
      pair('seed', n(st.seed));
      pair('checkpoint', fin(s._steps_completed) ? 'step ' + num(s._steps_completed) : null);
      pair('device · precision', typeof rt.device === 'string' && typeof rt.precision === 'string'
        ? rt.device + ' · ' + rt.precision : null);
      dl.hidden = false;
    }
  }

  /* ---------- boot ---------- */

  function settle() {
    document.documentElement.classList.remove('rs-loading');
    ['results', 'benchmarks', 'playground'].forEach(function (id) {
      var node = $(id);
      if (node) node.removeAttribute('aria-busy');
    });
  }

  function render(data) {
    settle();
    var t = data.training || {};
    status(t);

    // the head's meta and the crumb strip's provenance note, both from the file
    var meta = $('runMeta');
    if (meta) {
      var parts = [];
      if (fin(t.steps)) parts.push('published at step ' + num(t.steps));
      if (typeof data._run === 'string') parts.push('run ' + data._run);
      meta.textContent = '';
      if (parts.length) meta.appendChild(document.createTextNode(parts.join(' · ')));
      else meta.appendChild(el('span', 'is-unmeasured', UNMEASURED));
    }
    if (typeof data._run === 'string') set('crumbRun', 'Snapshot of run ' + data._run + ' · not live');

    put('mTokens', fin(t.tokens_seen) ? (t.tokens_seen / 1e9).toFixed(2) : null, 'B');
    put('mHours', fin(t.wall_clock_hours) ? t.wall_clock_hours.toFixed(2) : null, 'h');
    put('mTps', fmt(t.tokens_per_second), 'tok/s');
    put('mSteps', fmt(t.steps));
    put('mVram', fin(t.peak_vram_gb) ? t.peak_vram_gb.toFixed(1) : null, 'GB');
    // "Best so far" has a fixed slot, read from the published fields, never recomputed here.
    put('mBestVal', fin(t.best_val_loss) ? t.best_val_loss.toFixed(4) : null);
    put('mBestPpl', fmt(t.best_val_ppl, 1));
    put('mCorpus', typeof t.corpus === 'string' ? t.corpus : null);
    sub('mTokensSub', fin(t.tokens_seen) ? fmt(t.tokens_seen) + ' tokens' : '');
    // "idle", not "cooling pauses": since --duty, rest_hours also holds the time the GPU
    // was deliberately held idle to cap its load, and that is most of it.
    sub('mHoursSub', fin(t.rest_hours) && t.rest_hours > 0
      ? 'incl. ' + t.rest_hours.toFixed(2) + ' h idle (cooling pauses and the GPU load cap)' : '');
    sub('mTpsSub', fin(t.tokens_per_second) ? 'median over the run' : '');
    sub('mStepsSub', t.run_complete === true ? 'the whole run' : '');
    sub('mVramSub', typeof t.hardware === 'string' ? t.hardware : '');

    curve(t.val_curve, false, t);
    benchmarks(data.benchmarks || {});
    samples(data.samples);
  }

  function fail(msg) {
    settle();
    bar(null);
    set('runState', 'Results unavailable');
    seg('runPctSeg', false); seg('runStepSeg', false);
    var line = $('runStatus');
    if (line) line.textContent = msg;
    var rows = $('benchRows');
    if (rows) {
      rows.textContent = '';
      var tr = el('tr'); tr.setAttribute('role', 'row');
      var td = el('td', 'bm-empty', 'Results unavailable'); td.setAttribute('role', 'cell'); td.colSpan = 5;
      tr.appendChild(td); rows.appendChild(tr);
    }
    set('benchNote', 'Results unavailable: the scores could not be read from results.json.');
  }

  function init() {
    if (!$('runStatus')) return;              // not on a page that shows results
    // While the file is in flight the slots keep their space but claim nothing; the safety
    // timer means a hung request ends in the honest "not measured yet", never a blank page.
    document.documentElement.classList.add('rs-loading');
    ['results', 'benchmarks', 'playground'].forEach(function (id) {
      var node = $(id);
      if (node) node.setAttribute('aria-busy', 'true');
    });
    var safety = setTimeout(settle, 3000);

    // The chart is built at its container's width, so it is rebuilt when that width changes.
    var host = $('curveWrap');
    if (host && window.ResizeObserver) {
      new ResizeObserver(function () {
        // Only a change in width matters. A phone's toolbar collapsing on scroll fires a
        // resize at the same width, and redrawing then reset the table under the thumb.
        if (lastCurve === null || plotWidth(host) === lastW) return;
        curve(lastCurve, true);
      }).observe(host);
    }
    window.addEventListener('beforeprint', function () {
      var plot = $('curvePlot');
      if (plot && armed) { drawn = true; plot.classList.add('is-drawn'); }
    });

    fetch('results.json', { cache: 'no-store' })
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        // A parse failure is a data problem, not a serving problem, and is reported as one.
        return r.json().catch(function (e) {
          var err = new Error('results.json is not valid JSON (' + e.message + ')');
          err.isParse = true;
          throw err;
        });
      })
      .then(function (data) {
        clearTimeout(safety);
        try { render(data); }
        catch (e) { fail('results.json loaded but could not be drawn (' + e.message + ').'); }
      }, function (e) {
        clearTimeout(safety);
        fail(e.isParse ? e.message + '.'
          : 'Could not load results.json (' + e.message + '). If this page was opened ' +
            'straight from disk, serve it over http instead.');
      });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
