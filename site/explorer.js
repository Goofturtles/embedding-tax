/* Budget explorer: the whole cap as one live sum.
   The only claim this page makes is arithmetic, so every number here is derived
   from the four knobs. Nothing is typed twice and nothing is asserted about the
   trained model; its measured record belongs to results.js and the model itself to
   playground.js. Everything below the "Measured, not calculated" seam is never touched here.

   Also here: the page's reveal observer (the anchor's FadeIn, gated behind html.reveal so a
   scripts-off page shows everything) and the hero arch's draw-in. */
(function () {
  'use strict';

  // Scripts are running: the arch's drawn-in start state (shared.css) may apply.
  document.documentElement.classList.add('js');

  var CAP = 50000000;
  var BLOCK = 1024;                 // context length, only billed if positions are learned

  var OURS = { vocab: 16384, width: 512, ratio: 4, tie: true, rope: true };
  var PRESETS = {
    gpt2:  { vocab: 50257, width: 768, ratio: 4, tie: true, rope: true },
    gpt2w: { vocab: 50257, width: 512, ratio: 4, tie: true, rope: true },   // the story's 51.5%
    llama: { vocab: 32000, width: 512, ratio: 4, tie: true, rope: true },
    ours:  OURS,
    tiny:  { vocab: 8192, width: 512, ratio: 4, tie: true, rope: true }
  };
  var NAMES = {
    gpt2: 'GPT-2 Small', gpt2w: 'GPT-2 vocabulary, our width',
    llama: 'Llama tokenizer', ours: 'Ours', tiny: 'Tiny vocabulary'
  };

  var $ = function (id) { return document.getElementById(id); };
  var fmt = function (n) { return Math.round(n).toLocaleString('en-US'); };

  /* ---------- the sum ---------- */

  function budget(c) {
    var d = c.width;
    var embedding = c.vocab * d;
    var head = c.tie ? 0 : c.vocab * d;
    var position = c.rope ? 0 : BLOCK * d;
    var attn = 4 * d * d;
    var mlp = 2 * c.ratio * d * d;
    var norms = 2 * d;
    var perLayer = attn + mlp + norms;
    var finalNorm = d;
    var fixed = embedding + head + position + finalNorm;
    var layers = Math.max(0, Math.floor((CAP - fixed) / perLayer));
    return {
      embedding: embedding, head: head, position: position,
      attn: attn, mlp: mlp, norms: norms, perLayer: perLayer,
      finalNorm: finalNorm, layers: layers,
      layersTotal: layers * perLayer,
      total: fixed + layers * perLayer,
      free: CAP - (fixed + layers * perLayer)
    };
  }

  /* Everything spent before a single layer exists: the table, the untied head and any
     learned positions. This is the quantity the whole page argues about. */
  function spent(b) { return b.embedding + b.head + b.position; }
  function share(n) { return (100 * n / CAP).toFixed(1); }
  function plural(n, word) { return n === 1 ? word : word + 's'; }
  var MINUS = '−';

  /* ---------- reading the controls ---------- */

  function readConfig() {
    return {
      vocab: parseInt($('kVocab').value, 10),
      width: parseInt($('kWidth').value, 10),
      ratio: parseInt($('kRatio').value, 10),
      tie: $('kTie').checked,
      rope: $('kRope').checked
    };
  }

  function writeConfig(c) {
    $('kVocab').value = c.vocab;
    $('kWidth').value = c.width;
    $('kRatio').value = c.ratio;
    $('kTie').checked = c.tie;
    $('kRope').checked = c.rope;
  }

  function sameConfig(a, b) {
    return a.vocab === b.vocab && a.width === b.width && a.ratio === b.ratio &&
           a.tie === b.tie && a.rope === b.rope;
  }

  function flip(c, key) {
    var o = { vocab: c.vocab, width: c.width, ratio: c.ratio, tie: c.tie, rope: c.rope };
    o[key] = !o[key];
    return o;
  }

  /* ONE baseline for the whole page: the same settings with the vocabulary swapped for
     GPT-2's, or for ours when GPT-2's is already loaded. The readout's "vs" line, the ghost
     column, the reference line and the delta tag all read from this single result, so they
     can never disagree with each other. */
  function baseline(c) {
    var v = c.vocab === 50257 ? 16384 : 50257;
    return { vocab: v, width: c.width, ratio: c.ratio, tie: c.tie, rope: c.rope };
  }
  function baselineName(c) {
    return c.vocab === 50257 ? 'our 16,384-token vocabulary' : "GPT-2's vocabulary";
  }

  /* ---------- motion plumbing ----------
     One rAF for the page, no layout reads inside it, and everything snaps under
     prefers-reduced-motion (checked live, so the OS setting applies without a reload). */

  var RM = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  function reduced() { return !!(RM && RM.matches); }
  var ready = false;                 // true after the first paint: nothing rolls on load

  var tasks = [], frame = 0;
  function tick() {
    frame = 0;
    var run = tasks; tasks = [];
    for (var i = 0; i < run.length; i++) if (run[i]()) tasks.push(run[i]);
    if (tasks.length) frame = requestAnimationFrame(tick);
  }
  function queue(fn) {
    tasks.push(fn);
    if (!frame) frame = requestAnimationFrame(tick);
  }

  /* ---------- the odometer ----------
     Each digit is a strip of 0-9 inside a 1.25em window, moved by transform. The window uses
     overflow:clip, so the clipping probe stays clean and no descender is trimmed. */

  function roll(el, text) {
    if (!el || el._t === text) return;
    // A preset glide rewrites the figure every frame; the reels only roll on its final frame.
    var snap = !ready || reduced() || gliding;
    var shape = text.replace(/\d/g, '0');
    el.classList.toggle('is-snap', snap);
    var digits = text.replace(/\D/g, '').split('');

    if (el._shape !== shape) {
      var prev = el._digits || [];
      el.textContent = '';
      var strips = [];
      var total = digits.length, seen = 0;
      for (var i = 0; i < text.length; i++) {
        var ch = text.charAt(i);
        if (ch >= '0' && ch <= '9') {
          var win = document.createElement('span');
          win.className = 'ro-d';
          var strip = document.createElement('span');
          strip.className = 'ro-strip';
          for (var d = 0; d < 10; d++) {
            var s = document.createElement('span');
            s.textContent = String(d);
            strip.appendChild(s);
          }
          // right-aligned against the previous number, so 9 -> 10 rolls instead of jumping
          var fromRight = total - 1 - seen;
          var seed = prev[prev.length - 1 - fromRight];
          strip.style.setProperty('--i', String(fromRight));
          strip.style.setProperty('--d', snap ? digits[seen] : (seed === undefined ? digits[seen] : seed));
          win.appendChild(strip);
          el.appendChild(win);
          strips.push(strip);
          seen++;
        } else {
          var lit = document.createElement('span');
          lit.textContent = ch;
          el.appendChild(lit);
        }
      }
      el._strips = strips;
      if (!snap) {
        // two frames, so the seeded position is committed before the target is set
        queue(function () {
          queue(function () {
            for (var j = 0; j < strips.length; j++) strips[j].style.setProperty('--d', digits[j]);
          });
        });
      }
    } else if (el._strips) {
      for (var k = 0; k < el._strips.length; k++) el._strips[k].style.setProperty('--d', digits[k]);
    }
    el._t = text; el._shape = shape; el._digits = digits;
  }

  function set(id, text) { var el = $(id); if (el) el.textContent = text; }

  /* Text with one emphasised figure, built as nodes: nothing here is ever parsed as markup. */
  function vsLine(el, value, c) {
    if (!el) return;
    el.textContent = 'vs ';
    var b = document.createElement('b');
    b.textContent = value;
    el.appendChild(b);
    el.appendChild(document.createTextNode(' with ' + baselineName(c)));
  }

  /* ---------- the figure ----------
     The cap drawn as the model's own floors: the word list at the bottom, one slab per
     transformer layer above it, the unused remainder dashed, all on one scale shared with a
     ghost of the baseline configuration. */

  var plotH = 320;                   // cached by ResizeObserver; never read inside a frame

  function slabs(b, D) {
    var out = [];
    var f = spent(b);
    var y = 0;
    if (b.embedding > 0) { out.push({ k: 'emb', cls: 'is-cost', y: y / D, h: b.embedding / D, gap: 1 }); y += b.embedding; }
    if (b.head > 0) { out.push({ k: 'head', cls: 'is-cost', y: y / D, h: b.head / D, gap: 1 }); y += b.head; }
    if (b.position > 0) { out.push({ k: 'pos', cls: 'is-cost', y: y / D, h: b.position / D, gap: 1 }); y += b.position; }

    var pitch = plotH * b.perLayer / D;
    var gap = pitch >= 10 ? 2 : pitch >= 4 ? 1 : 0;
    for (var i = 1; i <= b.layers; i++) {
      var base = f + (i - 1) * b.perLayer;
      out.push({
        k: 'L' + i, cls: 'is-lay' + (gap === 0 && i % 10 === 0 ? ' is-tick' : ''),
        y: base / D, h: b.perLayer / D, gap: gap
      });
    }
    if (b.free > 0) out.push({ k: 'free', cls: 'is-free', y: b.total / D, h: b.free / D, gap: 0 });
    // Over the cap can only happen when the word list alone outgrows it, so the overflow is
    // drawn on top of the cap line rather than above a stack of layers.
    if (f + b.finalNorm > CAP) out.push({ k: 'over', cls: 'is-overflow', y: CAP / D, h: (f + b.finalNorm - CAP) / D, gap: 0 });
    return out;
  }

  /* After a glide, the slabs that did not exist before it land with the entrance stagger,
     once, diffed against the pre-glide column (intermediate frames only move geometry). */
  function staggerNew(host, before) {
    if (!host || !host._pool || reduced()) return;
    var i = 0;
    for (var key in host._pool) {
      if (before[key]) continue;
      var node = host._pool[key];
      node.classList.remove('is-in');
      node.classList.add('is-enter');
      node.style.setProperty('--sd', Math.min(i++, 12) * 16 + 'ms');
      (function (n) {
        queue(function () { queue(function () { n.classList.remove('is-enter'); n.classList.add('is-in'); }); });
      })(node);
    }
  }

  function drawColumn(host, b, D, instant) {
    if (!host) return;
    var pool = host._pool || (host._pool = {});
    var want = slabs(b, D), seen = {};
    var snap = instant || !ready || reduced() || gliding;
    for (var i = 0; i < want.length; i++) {
      var s = want[i];
      seen[s.k] = true;
      var el = pool[s.k];
      if (!el) {
        el = document.createElement('i');
        el.className = 'slab ' + s.cls + (snap ? '' : ' is-enter');
        host.appendChild(el);
        pool[s.k] = el;
        if (!snap) {
          el.style.setProperty('--sd', Math.min(i, 12) * 16 + 'ms');
          (function (node) {
            queue(function () { queue(function () { node.classList.remove('is-enter'); node.classList.add('is-in'); }); });
          })(el);
        }
      } else if (el._cls !== s.cls) {
        el.className = 'slab ' + s.cls + (el.classList.contains('is-in') ? ' is-in' : '');
      }
      el._cls = s.cls;
      el.style.setProperty('--y', s.y.toFixed(6));
      el.style.setProperty('--h', Math.max(s.h, 0).toFixed(6));
      el.style.setProperty('--gap', (s.gap || 0) + 'px');
    }
    for (var key in pool) {
      if (seen[key]) continue;
      var gone = pool[key];
      delete pool[key];
      if (snap) { if (gone.parentNode) gone.parentNode.removeChild(gone); continue; }
      gone.classList.remove('is-in');
      gone.classList.add('is-out');
      (function (node) {
        setTimeout(function () { if (node.parentNode) node.parentNode.removeChild(node); }, 220);
      })(gone);
    }
  }

  var lastD = 0;
  function gridlines(D) {
    var grid = $('capGrid');
    if (!grid || D === lastD) return;
    lastD = D;
    grid.textContent = '';
    for (var v = 0; v <= D; v += 10000000) {
      var line = document.createElement('i');
      line.className = 'cap-gline';
      line.style.setProperty('--y', (v / D).toFixed(6));
      grid.appendChild(line);
      var tick = document.createElement('b');
      tick.className = 'cap-tick';
      tick.style.setProperty('--y', (v / D).toFixed(6));
      tick.textContent = v === 0 ? '0' : (v / 1000000) + 'M';
      grid.appendChild(tick);
    }
  }

  var LEAD = 46;                     // the smallest gap two stacked labels may have, in px
  function leaders(b, D) {
    var host = $('capLeaders');
    if (!host) return;
    var f = spent(b);
    var rows = [];
    if (b.free > 0) {
      rows.push({ cls: 'is-free', mid: (b.total + b.free / 2) / D, name: 'Unused', val: fmt(b.free) + ' · ' + share(b.free) + '%' });
    } else if (b.free < 0) {
      rows.push({ cls: 'is-overflow', mid: 1, name: 'Over the cap by', val: fmt(-b.free) });
    }
    if (b.layers > 0) {
      rows.push({
        cls: 'is-lay', mid: (f + b.layersTotal / 2) / D,
        name: b.layers + ' transformer ' + plural(b.layers, 'layer'),
        val: share(b.layersTotal) + '% · ' + fmt(b.perLayer) + ' each'
      });
    }
    rows.push({ cls: 'is-cost', mid: f / 2 / D, name: 'Before any layer', val: share(f) + '% · vocabulary × width' });

    if (host.children.length !== rows.length) {
      host.textContent = '';
      for (var i = 0; i < rows.length; i++) {
        var el = document.createElement('p');
        el.className = 'cap-lead';
        el.appendChild(document.createElement('b'));
        el.appendChild(document.createElement('span'));
        host.appendChild(el);
      }
    }
    // top-down collision resolve, in pixels, against the cached plot height
    var ys = [];
    for (var j = 0; j < rows.length; j++) {
      var y = (1 - rows[j].mid) * plotH - 17;
      if (j > 0 && y < ys[j - 1] + LEAD) y = ys[j - 1] + LEAD;
      ys.push(y);
    }
    var over = ys[ys.length - 1] + 34 - plotH;
    if (over > 0) for (var k = ys.length - 1; k >= 0; k--) ys[k] = Math.max(0, ys[k] - over);
    for (var m = 0; m < rows.length; m++) {
      var node = host.children[m];
      node.className = 'cap-lead ' + rows[m].cls;
      node.style.setProperty('--ly', Math.round(ys[m]) + 'px');
      node.firstChild.textContent = rows[m].name;
      node.lastChild.textContent = rows[m].val;
    }
  }

  function legend(b) {
    var host = $('capLegend');
    if (!host) return;
    var f = spent(b);
    var rows = [
      ['is-cost', 'Before any layer', share(f) + '% of the cap', fmt(f)],
      ['is-lay', b.layers + ' transformer ' + plural(b.layers, 'layer'), fmt(b.perLayer) + ' each', fmt(b.layersTotal)],
      b.free < 0
        ? ['is-free', 'Over the cap by', 'the word list alone passes it', fmt(-b.free)]
        : ['is-free', 'Unused', share(b.free) + '% of the cap', fmt(b.free)]
    ];
    if (host.children.length !== rows.length) {
      host.textContent = '';
      for (var i = 0; i < rows.length; i++) {
        var li = document.createElement('li');
        li.appendChild(document.createElement('i'));
        var mid = document.createElement('span');
        mid.appendChild(document.createElement('b'));
        mid.appendChild(document.createElement('em'));
        li.appendChild(mid);
        var v = document.createElement('span');
        v.className = 'lg-v';
        li.appendChild(v);
        host.appendChild(li);
      }
    }
    for (var j = 0; j < rows.length; j++) {
      var node = host.children[j];
      node.firstChild.className = rows[j][0];
      node.children[1].firstChild.textContent = rows[j][1];
      node.children[1].lastChild.textContent = rows[j][2];
      node.lastChild.textContent = rows[j][3];
    }
  }

  function renderCap(b, c, g) {
    var D = Math.max(CAP, spent(b) + b.finalNorm, spent(g) + g.finalNorm);
    gridlines(D);
    drawColumn($('stLive'), b, D, false);
    drawColumn($('stGhost'), g, D, true);

    var capLine = $('capLine');
    if (capLine) capLine.style.setProperty('--y', (CAP / D).toFixed(6));

    // the ghost's ceiling, carried across the gutter, and the one delta statement on the page
    var refY = (spent(g) + g.layersTotal) / D;
    var ref = $('capRef');
    if (ref) {
      ref.style.setProperty('--y', refY.toFixed(6));
      ref.style.setProperty('--ref-on', g.layers > 0 ? '1' : '0');
    }
    var delta = $('capDelta'), d = b.layers - g.layers;
    if (delta) {
      delta.hidden = d === 0;
      delta.style.setProperty('--y', refY.toFixed(6));
      delta.textContent = (d > 0 ? '+' : MINUS) + Math.abs(d);
      delta.className = 'cap-delta chip ' + (d > 0 ? 'chip--layers' : 'chip--over');
    }

    leaders(b, D);
    legend(b);

    // The figure's one non-hidden description, rewritten with the drawing: a screen reader
    // must never hear the default state after a preset or a drag has changed it.
    var gName = c.vocab === 50257 ? 'Our 16,384-token vocabulary' : "GPT-2's 50,257-token vocabulary";
    set('figAlt',
      'At a ' + fmt(c.vocab) + '-token vocabulary and width ' + c.width + ', ' + share(spent(b)) +
      '% of the cap is spent before any layer exists, ' +
      (b.free < 0 ? 'which is over the cap by ' + fmt(-b.free) + '. '
        : b.layers === 0 ? 'and no transformer layer fits. '
        : 'leaving room for ' + b.layers + ' transformer ' + plural(b.layers, 'layer') + '. ') +
      gName + ' with the same settings spends ' + share(spent(g)) + '% and ' +
      (g.free < 0 ? 'is over the cap by ' + fmt(-g.free) + '.' : 'leaves ' + g.layers + '.'));
  }

  /* ---------- the ledger ---------- */

  var ROWS = null;
  function rowMap() {
    if (ROWS) return ROWS;
    ROWS = {};
    var all = document.querySelectorAll('#ledger [data-row]');
    for (var i = 0; i < all.length; i++) ROWS[all[i].getAttribute('data-row')] = all[i];
    return ROWS;
  }

  function cell(row, key, text) {
    if (!row) return;
    var el = row.querySelector('[data-' + key + ']');
    if (el) el.textContent = text;
  }

  function ledger(b, c) {
    var r = rowMap(), f = spent(b);
    var dims = fmt(c.vocab) + ' tokens × ' + c.width + ' wide';

    cell(r.grpEmb, 'sub', share(f) + '% of the cap');
    cell(r.grpEmb, 'params', fmt(f));

    cell(r.emb, 'sub', dims);
    cell(r.emb, 'subm', fmt(b.embedding) + ' × 1');
    cell(r.emb, 'each', fmt(b.embedding));
    cell(r.emb, 'count', '1');
    cell(r.emb, 'params', fmt(b.embedding));

    cell(r.head, 'sub', c.tie ? 'Tied to the embedding: saves ' + fmt(c.vocab * c.width) : dims);
    cell(r.head, 'subm', fmt(c.vocab * c.width) + ' × ' + (c.tie ? '0 · tied' : '1'));
    cell(r.head, 'each', fmt(c.vocab * c.width));
    cell(r.head, 'count', c.tie ? '0' : '1');
    cell(r.head, 'params', fmt(b.head));
    if (r.head) r.head.classList.toggle('is-zero', c.tie);

    cell(r.pos, 'sub', c.rope ? 'Rotary: computed, not stored, saves ' + fmt(BLOCK * c.width) : fmt(BLOCK) + ' positions × ' + c.width + ' wide');
    cell(r.pos, 'subm', fmt(BLOCK * c.width) + ' × ' + (c.rope ? '0 · rotary' : '1'));
    cell(r.pos, 'each', fmt(BLOCK * c.width));
    cell(r.pos, 'count', c.rope ? '0' : '1');
    cell(r.pos, 'params', fmt(b.position));
    if (r.pos) r.pos.classList.toggle('is-zero', c.rope);

    cell(r.grpLay, 'name', b.layers + ' ' + plural(b.layers, 'layer'));
    cell(r.grpLay, 'sub', fmt(b.perLayer) + ' each · ' + share(b.layersTotal) + '% of the cap');
    cell(r.grpLay, 'params', fmt(b.layersTotal));

    var parts = [['attn', b.attn, '4d²: query, key, value and output'],
                 ['mlp', b.mlp, (2 * c.ratio) + 'd²: two matrices at ' + c.ratio + '× the width'],
                 ['norms', b.norms, '2d: two weight-only norms']];
    for (var i = 0; i < parts.length; i++) {
      var row = r[parts[i][0]], each = parts[i][1];
      cell(row, 'sub', parts[i][2]);
      cell(row, 'subm', fmt(each) + ' × ' + b.layers);
      cell(row, 'each', fmt(each));
      cell(row, 'count', String(b.layers));
      cell(row, 'params', fmt(each * b.layers));
      if (row) row.classList.toggle('is-zero', b.layers === 0);
    }

    cell(r.final, 'sub', 'd: after the last layer');
    cell(r.final, 'subm', fmt(b.finalNorm) + ' × 1');
    cell(r.final, 'each', fmt(b.finalNorm));
    cell(r.final, 'count', '1');
    cell(r.final, 'params', fmt(b.finalNorm));

    cell(r.total, 'sub', share(b.total) + '% of the cap');
    cell(r.total, 'params', fmt(b.total));
    cell(r.free, 'name', b.free < 0 ? 'Over the cap by' : 'Unused');
    cell(r.free, 'params', fmt(Math.abs(b.free)));
  }

  /* A row whose parameter figure changed tints once, after the value settles: during a drag
     or a preset glide the figures move every frame, and flashing each frame would strobe. */
  var flashed = {};
  function flash() {
    if (reduced() || !ready) return;
    var r = rowMap(), pending = [];
    for (var key in r) {
      var td = r[key].querySelector('[data-params]');
      if (!td) continue;
      if (flashed[key] !== undefined && flashed[key] !== td.textContent) pending.push(r[key]);
      flashed[key] = td.textContent;
    }
    if (!pending.length) return;
    for (var i = 0; i < pending.length; i++) pending[i].classList.remove('is-changed');
    queue(function () {
      queue(function () {
        for (var j = 0; j < pending.length; j++) pending[j].classList.add('is-changed');
      });
    });
  }

  /* ---------- readout, helpers, dock ---------- */

  var liveTimer = 0;
  function announce(b, c) {
    clearTimeout(liveTimer);
    liveTimer = setTimeout(function () {
      var el = $('hrLive');
      if (!el) return;
      el.textContent = b.layers + ' ' + plural(b.layers, 'layer') + ' fit. ' + share(spent(b)) +
        '% of the cap is spent before any layer. ' + fmt(b.total) + ' of ' + fmt(CAP) + ': ' +
        (b.free < 0 ? 'over the cap by ' + fmt(-b.free) : 'under the cap, ' + fmt(b.free) + ' spare') + '.';
    }, 600);
  }

  function readout(b, c, g) {
    var over = b.free < 0, bad = over || b.layers === 0;
    roll($('rLayers'), String(b.layers));
    set('rLayersUnit', plural(b.layers, 'layer'));
    set('rLayersSr', b.layers + ' ' + plural(b.layers, 'layer') + ' fit');
    roll($('rShare'), share(spent(b)));
    set('rShareSr', share(spent(b)) + '% of the cap is spent before any layer');
    vsLine($('rLayersVs'), String(g.layers), c);
    vsLine($('rShareVs'), share(spent(g)) + '%', c);

    set('vTotal', fmt(b.total));
    if (over) set('vState', 'Over the cap by ' + fmt(-b.free));
    else if (b.layers === 0) set('vState', 'No layer fits: ' + fmt(b.free) + ' left is under one layer');
    else set('vState', 'Under the cap, ' + fmt(b.free) + ' spare');

    var box = $('vBox'), card = $('calcCard');
    if (box) box.classList.toggle('is-over', bad);
    if (card) card.classList.toggle('is-over', bad);

    set('dLayers', String(b.layers));
    set('dLayersUnit', plural(b.layers, 'layer'));
    set('dShare', share(spent(b)) + '%');
    set('dTotal', fmt(b.total) + ' of ' + fmt(CAP));
  }

  function helpers(b, c) {
    set('hVN', fmt(c.vocab));
    set('hVW', String(c.width));
    set('hVE', fmt(b.embedding));
    var note = $('hVNote');
    if (note) note.hidden = c.vocab >= OURS.vocab;
    set('hPL', fmt(b.perLayer));
    set('hMLP', fmt(b.mlp));

    var tieAlt = budget(flip(c, 'tie')), dTie = b.layers - tieAlt.layers;
    set('hTie', c.tie
      ? (dTie > 0 ? 'Tied: +' + dTie + ' ' + plural(dTie, 'layer') + ' over a separate head.'
                  : 'Tied: saves ' + fmt(c.vocab * c.width) + ', under one layer here.')
      : 'A separate head costs ' + fmt(c.vocab * c.width) +
        (dTie < 0 ? ', ' + Math.abs(dTie) + ' ' + plural(Math.abs(dTie), 'layer') + ' lost.' : '.'));

    var ropeAlt = budget(flip(c, 'rope')), dRope = b.layers - ropeAlt.layers;
    set('hRope', c.rope
      ? (dRope > 0 ? 'Rotary: +' + dRope + ' ' + plural(dRope, 'layer') + ' over a learned table.'
                   : 'Rotary: saves ' + fmt(BLOCK * c.width) + ', under one layer here.')
      : 'A learned table costs ' + fmt(BLOCK * c.width) + ' (1,024 × ' + c.width + ')' +
        (dRope < 0 ? ', ' + Math.abs(dRope) + ' ' + plural(Math.abs(dRope), 'layer') + ' lost.' : '.'));
  }

  /* ---------- rulers: Stripe's tier marks, built once ---------- */

  function ruler(id, marks, min, max) {
    var host = $(id);
    if (!host) return;
    host.textContent = '';
    for (var i = 0; i < marks.length; i++) {
      var p = ((marks[i][0] - min) / (max - min)).toFixed(4);
      var tick = document.createElement('i');
      tick.style.setProperty('--p', p);
      if (marks[i][2]) tick.className = 'is-minor';
      host.appendChild(tick);
      if (!marks[i][2]) {
        var lbl = document.createElement('b');
        lbl.style.setProperty('--p', p);
        lbl.textContent = marks[i][1];
        lbl.setAttribute('data-v', String(marks[i][0]));
        if (marks[i][0] === min) lbl.classList.add('is-first');
        if (marks[i][0] === max) lbl.classList.add('is-last');
        host.appendChild(lbl);
      }
    }
  }

  function buildRulers() {
    // The vocabulary stops are the presets themselves, so they cannot drift from the tiles.
    ruler('rulerVocab', [
      [PRESETS.tiny.vocab, '8K'], [OURS.vocab, '16K'],
      [PRESETS.llama.vocab, '32K'], [PRESETS.gpt2.vocab, '50K']
    ], 1024, 65536);
    var w = [];
    for (var v = 256; v <= 1024; v += 64) w.push([v, v.toLocaleString('en-US'), v % 256 !== 0]);
    ruler('rulerWidth', w, 256, 1024);
    var r = [];
    for (var k = 2; k <= 8; k++) r.push([k, k + '×', k % 2 !== 0]);
    ruler('rulerRatio', r, 2, 8);
  }

  function markRuler(id, value) {
    var host = $(id);
    if (!host) return;
    var labels = host.querySelectorAll('b');
    for (var i = 0; i < labels.length; i++) {
      labels[i].classList.toggle('is-on', Number(labels[i].getAttribute('data-v')) === value);
    }
  }

  /* ---------- tiles ---------- */

  function fillTiles() {
    var tiles = document.querySelectorAll('.pz');
    for (var i = 0; i < tiles.length; i++) {
      var input = tiles[i].querySelector('input');
      var p = input && PRESETS[input.value];
      if (!p) continue;
      var b = budget(p);
      var n = tiles[i].querySelector('[data-layers]');
      var unit = tiles[i].querySelector('[data-unit]');
      var sh = tiles[i].querySelector('[data-share]');
      var bar = tiles[i].querySelector('.pz-bar i');
      if (n) n.textContent = String(b.layers);
      if (unit) unit.textContent = plural(b.layers, 'layer');
      if (sh) sh.textContent = share(spent(b)) + '%';
      if (bar) bar.style.setProperty('--share', share(spent(b)) + '%');
    }
  }

  // Off every preset, no radio is checked: a config you dragged to is not "GPT-2".
  function markPreset(c) {
    var radios = document.querySelectorAll('input[name="preset"]');
    var hit = null;
    for (var i = 0; i < radios.length; i++) {
      var p = PRESETS[radios[i].value];
      var on = !!(p && sameConfig(p, c));
      radios[i].checked = on;
      if (on) hit = radios[i].value;
    }
    var custom = $('pzCustom');
    if (custom) custom.hidden = !!hit;
    set('lName', hit ? NAMES[hit] : 'Your settings');
    var tag = $('lTag');
    if (tag) tag.hidden = hit !== 'ours';
    set('lDims', fmt(c.vocab) + ' × ' + c.width);
    var reset = $('resetBtn');
    if (reset) reset.setAttribute('aria-disabled', String(sameConfig(c, OURS)));
    return hit;
  }

  /* ---------- the hero thesis: the two configurations the story names ---------- */

  function thesis() {
    var g = budget(PRESETS.gpt2w), o = budget(OURS);
    set('tGptPct', share(spent(g)) + '%');
    set('tGptLayers', g.layers + ' ' + plural(g.layers, 'layer'));
    set('tOursPct', share(spent(o)) + '%');
    set('tOursLayers', String(o.layers));
  }

  /* ---------- wiring ---------- */

  var gliding = false;

  function update(quiet) {
    var c = readConfig();
    var b = budget(c);
    var gc = baseline(c);          // the config, kept: budget() returns sizes, not settings
    var g = budget(gc);

    ['kVocab', 'kWidth', 'kRatio'].forEach(function (id) {
      var el = $(id), min = Number(el.min), max = Number(el.max);
      el.style.setProperty('--pct', (100 * (Number(el.value) - min) / (max - min)).toFixed(2) + '%');
    });
    $('kVocab').setAttribute('aria-valuetext', fmt(c.vocab) + ' tokens');
    $('kWidth').setAttribute('aria-valuetext', c.width + ' wide');
    $('kRatio').setAttribute('aria-valuetext', c.ratio + ' times the width');
    markRuler('rulerVocab', c.vocab);
    markRuler('rulerWidth', c.width);
    markRuler('rulerRatio', c.ratio);
    field('nVocab', fmt(c.vocab));
    field('nWidth', String(c.width));
    field('nRatio', String(c.ratio));

    helpers(b, c);
    readout(b, c, g);
    renderCap(b, c, g);
    ledger(b, c);
    set('gDims', fmt(gc.vocab) + ' × ' + gc.width);
    set('gName', c.vocab === 50257 ? 'Our vocabulary' : "GPT-2's vocabulary");
    if (!gliding) markPreset(c);
    announce(b, c);
    if (!quiet) flash();
  }

  function field(id, text) {
    var el = $(id);
    if (el && document.activeElement !== el) el.value = text;
  }

  /* A preset does not jump: it drives the real controls through real configurations, so the
     readout, the drawing and the ledger all move together through states that are true. */
  function glide(to) {
    var c = readConfig();
    $('kTie').checked = to.tie;
    $('kRope').checked = to.rope;
    if (reduced() || !ready) { writeConfig(to); update(); return; }
    // The ledger flash and the slab entrance run once, on the final frame, diffed against
    // the state before the glide; the frames between only move text and geometry.
    var live = $('stLive'), before = {};
    if (live && live._pool) for (var key in live._pool) before[key] = true;
    gliding = true;
    var t0 = 0;
    queue(function step() {
      if (!gliding) return false;
      if (!t0) t0 = Date.now();
      var k = Math.min(1, (Date.now() - t0) / 360);
      var e = 1 - Math.pow(1 - k, 3);
      $('kVocab').value = Math.round(c.vocab + (to.vocab - c.vocab) * e);
      $('kWidth').value = Math.round((c.width + (to.width - c.width) * e) / 64) * 64;
      $('kRatio').value = Math.round(c.ratio + (to.ratio - c.ratio) * e);
      update(true);
      if (k < 1) return true;
      gliding = false;
      writeConfig(to);
      update();
      staggerNew(live, before);
      return false;
    });
  }

  function stopGlide() { gliding = false; }

  /* A rejected typed value is said, not just tinted: aria-invalid plus the one-line message
     that lives in the knob's help text (WCAG 3.3.1). Both clear on the next good value or blur. */
  function commit(id, sliderId, parse) {
    var el = $(id), slider = $(sliderId), msg = $(id + 'Err');
    if (!el || !slider) return;
    var bad = function (on) {
      el.classList.toggle('is-bad', on);
      if (on) el.setAttribute('aria-invalid', 'true'); else el.removeAttribute('aria-invalid');
      if (msg) msg.hidden = !on;
    };
    var apply = function () {
      var v = parse(String(el.value).replace(/[,\s]/g, ''));
      if (v === null) { bad(true); update(); return; }
      bad(false);
      slider.value = v;
      update();
    };
    el.addEventListener('change', apply);
    el.addEventListener('keydown', function (e) { if (e.key === 'Enter') { e.preventDefault(); apply(); } });
    el.addEventListener('blur', function () { bad(false); update(); });
  }

  function clampInt(s, min, max, step) {
    var n = parseInt(s, 10);
    if (!isFinite(n)) return null;
    n = Math.max(min, Math.min(max, n));
    if (step > 1) n = Math.round(n / step) * step;
    return Math.max(min, Math.min(max, n));
  }

  /* ---------- reveals ----------
     The anchor's FadeIn (opacity + translateY(20px), .75s, 80ms stagger by --i) for every
     [data-reveal]. The hidden start state is gated behind html.reveal, added only here, once
     the observer exists, and never under reduced motion; a flip of the OS setting while the
     page is open pins everything to its final state, and a flip back re-arms what has not
     been seen yet. The hero arch draws itself once the hero's FadeIn has begun. */
  function reveals() {
    var root = document.documentElement;
    var nodes = document.querySelectorAll('[data-reveal]');
    var arch = $('exArch');
    var io = null;
    function showAll() {
      for (var i = 0; i < nodes.length; i++) nodes[i].classList.add('is-in');
    }
    function arm() {
      if (reduced() || !window.IntersectionObserver) { root.classList.remove('reveal'); showAll(); return; }
      root.classList.add('reveal');
      io = new IntersectionObserver(function (entries) {
        for (var i = 0; i < entries.length; i++) {
          if (!entries[i].isIntersecting) continue;
          entries[i].target.classList.add('is-in');
          io.unobserve(entries[i].target);
        }
      }, { threshold: 0.1, rootMargin: '0px 0px -8% 0px' });
      for (var j = 0; j < nodes.length; j++) if (!nodes[j].classList.contains('is-in')) io.observe(nodes[j]);
    }
    arm();
    // the arch: drawn after the FadeIn has started (never before the page has painted)
    if (arch && !arch.classList.contains('is-in')) {
      if (reduced()) arch.classList.add('is-in');
      else setTimeout(function () { arch.classList.add('is-in'); }, 240);
    }
    if (RM && RM.addEventListener) {
      RM.addEventListener('change', function () {
        if (io) { io.disconnect(); io = null; }
        if (reduced()) { root.classList.remove('reveal'); showAll(); }
        else arm();
      });
    }
  }

  function init() {
    if (!$('capFig')) return;           // not the explorer page

    reveals();
    thesis();
    fillTiles();
    buildRulers();

    ['kVocab', 'kWidth', 'kRatio'].forEach(function (id) {
      var el = $(id);
      if (!el) return;
      el.addEventListener('input', function () { stopGlide(); update(true); });
      el.addEventListener('change', function () { update(); });
    });
    ['kTie', 'kRope'].forEach(function (id) {
      var el = $(id);
      if (el) el.addEventListener('change', function () { stopGlide(); update(); });
    });

    // step="1" is needed so a preset can land on 50,257 exactly, but it also means 49,152
    // arrow presses to cross the range. Give the arrows a real stride.
    var vocab = $('kVocab');
    if (vocab) {
      vocab.addEventListener('keydown', function (e) {
        var big = e.key === 'PageUp' || e.key === 'PageDown';
        var stride = (e.shiftKey || big) ? 4096 : 512;
        var dir = (e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'PageUp') ? 1
                : (e.key === 'ArrowLeft' || e.key === 'ArrowDown' || e.key === 'PageDown') ? -1 : 0;
        if (!dir) return;
        e.preventDefault();
        stopGlide();
        vocab.value = Math.max(1024, Math.min(65536, parseInt(vocab.value, 10) + dir * stride));
        update();
      });
    }

    commit('nVocab', 'kVocab', function (s) { return clampInt(s, 1024, 65536, 1); });
    commit('nWidth', 'kWidth', function (s) { return clampInt(s, 256, 1024, 64); });
    commit('nRatio', 'kRatio', function (s) { return clampInt(s, 2, 8, 1); });

    var radios = document.querySelectorAll('input[name="preset"]');
    for (var i = 0; i < radios.length; i++) {
      radios[i].addEventListener('change', function () {
        var p = PRESETS[this.value];
        if (p && this.checked) glide(p);
      });
    }
    var reset = $('resetBtn');
    if (reset) {
      reset.addEventListener('click', function () {
        // aria-disabled, not disabled: a button that disables itself under the pointer drops
        // keyboard focus to <body>.
        if (reset.getAttribute('aria-disabled') === 'true') return;
        glide(OURS);
      });
    }

    // The figure's geometry is decided in fractions, but the layer pitch and the leader
    // collisions are decided in pixels, so the height is cached here and never read in a frame.
    var plot = $('capPlot');
    if (plot && window.ResizeObserver) {
      new ResizeObserver(function (entries) {
        var h = entries[0].contentRect.height;
        if (!h || Math.abs(h - plotH) < 1) return;
        plotH = h;
        update(true);
      }).observe(plot);
    } else if (plot && plot.clientHeight) {
      plotH = plot.clientHeight;
    }

    // The dock repeats the readout, so it steps aside whenever the readout itself is on screen.
    var dock = $('hrDock'), ro = $('hrReadout');
    if (dock && ro && window.IntersectionObserver) {
      new IntersectionObserver(function (entries) {
        dock.classList.toggle('is-away', entries[0].isIntersecting);
      }, { threshold: 0 }).observe(ro);
    }

    writeConfig(OURS);
    update(true);
    ready = true;
    flash();          // records the starting figures, so the first real change tints
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
