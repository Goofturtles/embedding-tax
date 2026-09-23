/* The model in use, on the visitor's device.

   One component for both pages. It mounts on every [data-playground] root (app.html's
   section#try; the home composer in index.html) and binds by data-pg hooks INSIDE that root,
   never by document ids, so the two panels can carry different ids (pg- / h-) for their
   labels and anchors. Everything it writes is
   textContent; streamed text lands in one Text node through a single rAF-batched append.

   Hooks (data-pg="..."): preset (buttons, several), prompt (textarea), load, progress
   (role=progressbar), barfill, bartext, write, stop, tokens, backendname, statusline, hint,
   status (role=status, sr-only), echo, cont, conttext, caret, empty, copy (its label: copylabel), packaging, failure,
   failuredetail, retry, preset-mode (radio inputs, value = a PRESETS id), preset-name,
   preset-desc, temp/tempN, topk/topkN, max/maxN, unlimited (switch), unlimitedbadge,
   guardstate, locknote, seed, seedlock (switch), reset, summary, card, loadnote, barmeta, the
   run readout (run, runpreset, runtokens, runspeed, runfirst, runstop, runbackend), and the
   text slots mb, mbunit, params, quant, step (several), runstate (several), runline. Every
   hook but prompt and load is optional; the slider ranges are set from RANGE here, so markup
   that still carries older min/max attributes gets the tested ranges anyway.

   Decoding presets (PRESETS, from presets.js): Balanced, Focused and Creative share the guards (GUARDS: CAD,
   a min-p that follows temperature, a repetition penalty, the two line-break rules); Published
   is tools/sample.py's temperature 0.8 / top-k 40 / 60 tokens with every guard off, and its
   sliders are locked (no unguarded sampler tested safe). Moving a slider away from a guarded
   preset reads "Custom" and keeps the guards. The label is always derived from the settings,
   so it names exactly what the next run uses.

   The contract with infer.js (window.EmbeddingTax):
     load({ onProgress }) -> Promise<{ backend, manifest }>   idempotent; downloads once
     generate(prompt, { maxNewTokens, temperature, topK, seed, untilDone, topP, minP,
       repetitionPenalty, repetitionWindow, frequencyPenalty, presencePenalty, noRepeatNgram,
       noLeadingBreak, noMidsentenceBreak, cadAlpha, bestOf, bestOfFluencyWeight, onToken, signal }) -> Promise<string>
     tokenCount(text), ready, manifest, last.
   HONESTY: the weights never download without the click (a later visit reopens the copy an
   earlier load left in Cache Storage, cacheOnly, never the network: by itself on a computer, on a
   tap on a phone or tablet, EmbeddingTax.constrained()); nothing is simulated; the checkpoint step and
   the parameter count come from model/manifest.json; the run is called complete only when the
   manifest's training_finished field is true. */
(function () {
  'use strict';

  /* The decoding presets come from presets.js (window.EmbeddingTaxPresets), the one table shared with
     the story page's panel (script.js). If that file failed to load, the panel offers Published alone:
     tools/sample.py's sampler with no guards, so nothing on screen claims a guard that is not sent. */
  var T = window.EmbeddingTaxPresets || {
    PRESETS: [{ id: 'published', name: 'Published', guards: false, temperature: 0.8, topK: 40, maxNewTokens: 60, untilDone: false,
      desc: 'The settings used for the published samples: temperature 0.8, top-k 40, 60 tokens, no other rules.' }],
    CUSTOM_DESC: { guarded: '', unguarded: 'Published\'s sampler, run until the passage ends (at most 256 tokens). None of the guards are on.' },
    DEFAULT_PRESET: 'published', DEFAULT_SEED: 1337, GUARDS: null,
    NO_GUARDS: { topP: 1, minP: 0, repetitionPenalty: 1, repetitionWindow: 0, frequencyPenalty: 0, presencePenalty: 0, noRepeatNgram: 0,
                 noLeadingBreak: 0, noMidsentenceBreak: false, cadAlpha: 0, bestOf: 1, bestOfFluencyWeight: 0 },
    RANGES: { temperature: [0.6, 1.4, 0.05], topK: [40, 200, 1], maxNewTokens: [8, 90, 1] }, UNTIL_DONE_CAP: 256,
    minPFor: function () { return 0; }
  };
  var PRESETS = T.PRESETS, CUSTOM_DESC = T.CUSTOM_DESC, DEFAULT_PRESET = T.DEFAULT_PRESET, DEFAULT_SEED = T.DEFAULT_SEED;
  var RANGE = T.RANGES, GUARDS = T.GUARDS, NO_GUARDS = T.NO_GUARDS, minPFor = T.minPFor, UNTIL_DONE_CAP = T.UNTIL_DONE_CAP;
  function presetById(id) { for (var i = 0; i < PRESETS.length; i++) if (PRESETS[i].id === id) return PRESETS[i]; return null; }
  // the preset these settings are exactly, or null (Custom)
  function presetOf(s) {
    for (var i = 0; i < PRESETS.length; i++) {
      var p = PRESETS[i];
      if (p.guards === s.guards && p.temperature === s.temperature && p.topK === s.topK && p.untilDone === s.untilDone &&
          (s.untilDone || p.maxNewTokens === s.maxNewTokens)) return p;
    }
    return null;
  }
  function fromPreset(p, seed) {
    return { guards: p.guards, temperature: p.temperature, topK: p.topK, maxNewTokens: p.maxNewTokens, untilDone: p.untilDone, seed: seed };
  }
  // everything generate() takes, spelled out
  function decoding(s) {
    var o = { temperature: s.temperature, topK: s.topK, maxNewTokens: s.maxNewTokens, untilDone: s.untilDone, seed: s.seed }, k;
    var g = s.guards ? GUARDS : NO_GUARDS;
    for (k in g) o[k] = g[k];
    if (s.guards) o.minP = minPFor(s.temperature);
    return o;
  }
  var RM = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  function reduced() { return !!(RM && RM.matches); }
  // iPhone and iPad (iPadOS calls itself a Mac with a touch screen): every browser there runs WebKit,
  // so the failure panel's "try a current Chrome, Edge or Firefox" cannot help (failCopy).
  var APPLE_TOUCH = /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  // A phone or tablet holding the weights: the button stays as short as "Load the model (50 MB)" (a
  // longer label ran out of its pill at 320px) and the note under it, which wraps, says what it costs.
  var SAVED_LABEL = 'Open the saved model';
  var SAVED_NOTE = 'Already saved on this device: opening it downloads nothing. Nothing you type leaves the page.';
  var INTERRUPTED_NOTE = 'The page reloaded before the model could open. This time it keeps the weights compressed: less memory, slower writing.';
  function fin(n) { return typeof n === 'number' && isFinite(n); }
  function num(n) { return fin(n) ? n.toLocaleString('en-US') : null; }
  function mb(bytes) { return fin(bytes) ? Math.round(bytes / 1e6) : null; }

  function q(root, key) { return root.querySelector('[data-pg="' + key + '"]'); }
  function qa(root, key) { return root.querySelectorAll('[data-pg="' + key + '"]'); }
  function show(node, on) { if (node) node.hidden = !on; }
  function text(node, s) { if (node) node.textContent = s; }
  function textAll(nodes, s) { for (var i = 0; i < nodes.length; i++) nodes[i].textContent = s; }
  function fmtT(t) { return String(Math.round(t * 100) / 100); }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function mount(root) {
    var ui = {
      card: q(root, 'card'), presets: qa(root, 'preset'), prompt: q(root, 'prompt'),
      load: q(root, 'load'), progress: q(root, 'progress'), barfill: q(root, 'barfill'), bartext: q(root, 'bartext'),
      write: q(root, 'write'), stop: q(root, 'stop'), tokens: q(root, 'tokens'),
      backendname: q(root, 'backendname'), status: q(root, 'status'), barmeta: q(root, 'barmeta'),
      echo: q(root, 'echo'), cont: q(root, 'cont'), conttext: q(root, 'conttext'), caret: q(root, 'caret'),
      empty: q(root, 'empty'), copy: q(root, 'copy'),
      packaging: q(root, 'packaging'), failure: q(root, 'failure'), failuredetail: q(root, 'failuredetail'), retry: q(root, 'retry'),
      temp: q(root, 'temp'), tempN: q(root, 'tempN'), topk: q(root, 'topk'), topkN: q(root, 'topkN'),
      max: q(root, 'max'), maxN: q(root, 'maxN'), seed: q(root, 'seed'), seedlock: q(root, 'seedlock'), unlimited: q(root, 'unlimited'),
      modes: qa(root, 'preset-mode'), modename: q(root, 'preset-name'), modedesc: q(root, 'preset-desc'),
      unlimitedbadge: q(root, 'unlimitedbadge'), guardstate: q(root, 'guardstate'), locknote: q(root, 'locknote'),
      reset: q(root, 'reset'), summary: q(root, 'summary'), loadnote: q(root, 'loadnote'),
      statusline: q(root, 'statusline'), hint: q(root, 'hint'), run: q(root, 'run'), runpreset: q(root, 'runpreset'), runtokens: q(root, 'runtokens'),
      runspeed: q(root, 'runspeed'), runfirst: q(root, 'runfirst'), runstop: q(root, 'runstop'), runbackend: q(root, 'runbackend'),
      mb: qa(root, 'mb'), mbunit: qa(root, 'mbunit'), params: qa(root, 'params'), quant: qa(root, 'quant'),
      step: qa(root, 'step'), runstate: qa(root, 'runstate'), runline: qa(root, 'runline')
    };
    if (!ui.prompt || !ui.load) return;

    var state = 'init';            // init | packaging | idle | loading | ready | writing | done | failed
    var manifest = null;
    var settings = fromPreset(presetById(DEFAULT_PRESET), DEFAULT_SEED);
    var usedName = '';             // the preset name of the run in flight / last run, for the readout
    var controller = null;
    var runs = 0;                  // generations completed for the current prompt
    var lastPrompt = null;         // the prompt of the last run: "Write again" moves the seed only for the same text
    var tokenN = 0;
    var textNode = null, pending = '', flushFrame = 0;
    var backendLabel = '';
    var resuming = false;          // reopening the copy already saved on this device, no download
    var saved = false;             // a phone or tablet holding the weights: Load opens that copy (cacheOnly) on the tap
    var interrupted = false;       // this tab died building the model (infer.js E.interrupted): the note says so
    var failTitle = ui.failure ? ui.failure.querySelector('p:not([data-pg])') : null;
    var failTitleText = failTitle ? failTitle.textContent : '';
    var loadNoteText = ui.loadnote ? ui.loadnote.textContent : '';
    var STATUS = { init: 'Not loaded', idle: 'Not loaded', packaging: 'Being packaged', loading: 'Loading',
                   ready: 'Ready', writing: 'Writing', done: 'Ready', failed: 'Could not run here' };

    /* ---------- the live region: one role=status per root, at most one message per 600ms ---------- */
    var sayAt = 0, sayTimer = 0;
    var wasLocked = null;          // the last lock state announced
    function say(msg) {
      if (!ui.status) return;
      var wait = 600 - (Date.now() - sayAt);
      clearTimeout(sayTimer);
      if (wait <= 0) { sayAt = Date.now(); ui.status.textContent = msg; return; }
      sayTimer = setTimeout(function () { sayAt = Date.now(); ui.status.textContent = msg; }, wait);
    }

    /* ---------- states ---------- */
    function setState(next) {
      state = next;
      root.setAttribute('data-state', next);
      var loaded = next === 'ready' || next === 'writing' || next === 'done';
      show(ui.load, next === 'idle' || next === 'init');   // disabled until the manifest has been read
      show(ui.progress, next === 'loading');
      show(ui.bartext, next === 'loading');
      show(ui.write, next === 'ready' || next === 'done');
      show(ui.stop, next === 'writing');
      show(ui.packaging, next === 'packaging');
      show(ui.failure, next === 'failed');
      show(ui.loadnote, next === 'idle' || next === 'loading');
      show(ui.hint, next === 'ready' || next === 'done');
      if (next === 'failed') show(ui.run, false);
      text(ui.statusline, (next === 'loading' && resuming ? 'Opening saved model' : STATUS[next]) + (loaded && backendLabel ? ' on ' + backendLabel : ''));
      for (var i = 0; i < ui.presets.length; i++) ui.presets[i].disabled = next === 'packaging';
      ui.prompt.disabled = next === 'packaging';
      if (ui.write) {
        ui.write.textContent = next === 'done' ? 'Write again' : 'Write';
        ui.write.disabled = !(loaded && ui.prompt.value.trim().length > 0);
      }
      if (ui.copy) show(ui.copy, next === 'done' && !!textNode && textNode.data.length > 0 && !!(navigator.clipboard && window.isSecureContext));
    }

    /* ---------- manifest: the small file, read on arrival; the weights wait for the click ---------- */
    function loadLabel() {
      var size = manifest ? mb(manifest.bytes) : null;
      ui.load.textContent = saved ? SAVED_LABEL : size !== null ? 'Load the model (' + size + ' MB)' : 'Load the model';
      text(ui.loadnote, interrupted ? INTERRUPTED_NOTE : saved ? SAVED_NOTE : loadNoteText);
    }
    function fillManifest(m) {
      manifest = m;
      var size = mb(m.bytes);
      if (size !== null) {
        textAll(ui.mb, String(size));
        for (var i = 0; i < ui.mbunit.length; i++) ui.mbunit[i].hidden = false;
        loadLabel();
      }
      var steps = num(m.steps_completed);
      if (steps !== null) { textAll(ui.step, steps); show(ui.barmeta, true); }
      var params = num(m.params_total);
      if (params !== null) textAll(ui.params, params);
      // The run is called complete only when the manifest says so.
      var done = m.training_finished === true;
      textAll(ui.runstate, done ? 'run complete' : 'run still in progress');
      textAll(ui.runline, done ? 'of the training run, its last step' : 'of a training run that is still going');
      if (typeof m.quantization === 'string') {
        var short = m.quantization.split(/[,;]/)[0].trim();
        textAll(ui.quant, short || m.quantization);
      }
    }

    function probe() {
      var api = window.EmbeddingTax;
      fetch('model/manifest.json', { cache: 'default' })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (m) {
          if (!m || typeof m !== 'object') throw new Error('manifest is not an object');
          fillManifest(m);
          // infer.js may still be arriving; give it two seconds after the page loaded
          var waited = 0;
          (function waitApi() {
            api = window.EmbeddingTax;
            if (api && typeof api.load === 'function') {
              if (api.ready) { onLoaded({ backend: api.backend, manifest: api.manifest || m }); }
              else {
                ui.load.disabled = false;
                // this tab died building the model (a phone reloads a tab that runs out of memory):
                // say so, and what the next try does differently (infer.js E.interrupted, stats.light)
                if (api.interrupted && isConstrained(api)) { interrupted = true; loadLabel(); }
                setState('idle');
                resume(api, m);
              }
              return;
            }
            if (waited >= 2000) { setState('packaging'); return; }
            waited += 100;
            setTimeout(waitApi, 100);
          })();
        }, function () { setState('packaging'); });
    }

    /* ---------- loading: real bytes, a bar that only grows ---------- */
    function onProgress(loaded, total) {
      var pct = total > 0 ? clamp(100 * loaded / total, 0, 100) : 0;
      if (ui.barfill) ui.barfill.style.setProperty('--p', (pct / 100).toFixed(4));
      if (ui.progress) ui.progress.setAttribute('aria-valuenow', pct.toFixed(0));
      var l = mb(loaded), t = mb(total);
      var line = (l === null || t === null) ? 'Loading' : l + ' of ' + t + ' MB';
      if (resuming) line = 'Opening the copy saved on this device';
      else if (total > 0 && loaded >= total) line += ' · starting on this device';
      text(ui.bartext, line);
      say('Loading the model, ' + line);
    }

    function onLoaded(r, quiet) {
      var backend = r && r.backend === 'webgpu' ? 'WebGPU' : 'WASM';
      text(ui.backendname, backend);
      backendLabel = backend;
      setState('ready');
      say('The model is ready on ' + backend);
      // completing the load lights the section: the preset chips 80ms apart, then the edge glow
      root.classList.add('is-lit');
      if (ui.card) ui.card.classList.add('is-lit');
      if (!reduced()) {
        for (var i = 0; i < ui.presets.length; i++) {
          (function (chip, k) {
            setTimeout(function () { chip.classList.add('is-lit'); }, 80 * k);
            setTimeout(function () { chip.classList.remove('is-lit'); }, 80 * k + 520);
          })(ui.presets[i], i);
        }
      }
      // the Load button just left the page: keep the keyboard in the panel (not when the model
      // reopened by itself on arrival: nobody pressed anything, so focus stays where the page put it)
      if (!quiet) keepFocus(ui.prompt);
    }

    // focus follows the panel only while the visitor is in it (a long load can end after they scroll on).
    // A container of the panel counts as nowhere in particular, like body: Safari does not focus a
    // button on a tap or click but the nearest focusable ancestor (<main tabindex="-1">), which left
    // Try again unfocused after a failed load on iPhone and iPad.
    function keepFocus(el) {
      var a = document.activeElement;
      if (el && (!a || a === document.body || root.contains(a) || a.contains(root))) el.focus({ preventScroll: true });
    }

    /* iPhone and iPad: every browser there runs WebKit, so the markup's "try a current Chrome, Edge or
       Firefox" is swapped for advice that can help there. Memory is named only when infer.js
       classified the error as such (err.code OUT_OF_MEMORY); the technical detail stays below. */
    function failCopy(err) {
      if (!failTitle) return;
      failTitle.textContent = !APPLE_TOUCH ? failTitleText : err && err.code === 'OUT_OF_MEMORY'
        ? 'This device ran out of memory for the model. Close other tabs, then try again.'
        : 'The model could not run on this device just now. Try again.';
    }

    function onFailed(err) {
      text(ui.failuredetail, err && err.message ? String(err.message) : '');
      failCopy(err);
      setState('failed');
      keepFocus(ui.retry);
      say('The model could not load on this device.');
    }

    function isConstrained(api) { return typeof api.constrained === 'function' && api.constrained(); }

    /* a visit after a successful load: the weights are already in Cache Storage. On a computer the
       model reopens by itself. On a phone or tablet it waits for a tap on "Open the saved model" (the
       note says it downloads nothing): building it takes a few hundred MB for seconds, which should not land on top of
       the page's own arrival, and a phone that cannot hold it would otherwise be sent into a reload
       on every visit. cacheOnly means a miss rejects instead of downloading; a failure while
       reopening by itself just leaves the Load button, as if this never ran. */
    function resume(api, m) {
      if (typeof api.isCached !== 'function') return;
      api.isCached(m).then(function (hit) {
        if (!hit || state !== 'idle' || api.ready) return;
        if (isConstrained(api)) { saved = true; loadLabel(); return; }
        openSaved(api, true);
      });
    }

    function openSaved(api, quiet) {
      resuming = true;
      if (ui.barfill) ui.barfill.style.setProperty('--p', '0');
      text(ui.bartext, 'Opening the copy saved on this device');
      setState('loading');
      api.load({ onProgress: onProgress, cacheOnly: true }).then(function (r) {
        resuming = false;
        onLoaded(r, quiet);
      }, function (err) {
        resuming = false;
        if (err && err.code === 'NOT_CACHED') {   // the saved copy is gone: back to the download button
          saved = false;
          loadLabel();
          if (!quiet) say('The saved copy is no longer on this device. Load downloads it again.');
        } else if (!quiet) { onFailed(err); return; }
        if (state === 'loading') setState(api.ready ? 'ready' : 'idle');
      });
    }

    function load() {
      if (state !== 'idle' && state !== 'failed') return;
      var api = window.EmbeddingTax;
      if (!api || typeof api.load !== 'function') { setState('packaging'); return; }
      if (saved) { openSaved(api, false); return; }
      if (ui.barfill) ui.barfill.style.setProperty('--p', '0');
      text(ui.bartext, 'Loading');
      setState('loading');
      api.load({ onProgress: onProgress }).then(onLoaded, onFailed);
    }

    /* ---------- writing: real tokens, appended as they arrive ---------- */
    function flush() {
      flushFrame = 0;
      if (!textNode || !pending) return;
      textNode.appendData(pending);
      pending = '';
      text(ui.tokens, tokenN + (tokenN === 1 ? ' token' : ' tokens'));
    }

    function write() {
      var api = window.EmbeddingTax;
      var prompt = ui.prompt.value;
      if (!api || !api.ready || !prompt.trim() || state === 'writing') return;
      // Write again: the same prompt with the seed moved by one, unless the switch holds it.
      if (state === 'done' && runs > 0 && prompt === lastPrompt && !(ui.seedlock && ui.seedlock.checked)) {
        settings.seed = settings.seed + 1;
        writeSettings();
      }
      lastPrompt = prompt;
      tokenN = 0; pending = '';
      if (ui.echo) { ui.echo.textContent = prompt; ui.echo.hidden = false; }
      if (ui.conttext) {
        ui.conttext.textContent = '';
        textNode = document.createTextNode('');
        ui.conttext.appendChild(textNode);
      }
      show(ui.cont, true);
      show(ui.empty, false);
      show(ui.caret, !reduced());
      text(ui.tokens, '0 tokens');
      controller = new AbortController();
      var used = decoding(settings);
      var named = presetOf(settings);
      usedName = named ? named.name : 'Custom';
      var fromButton = document.activeElement === ui.write;
      setState('writing');
      pendingReport();
      if (fromButton && ui.stop) ui.stop.focus({ preventScroll: true });
      say('Writing');
      used.signal = controller.signal;
      used.onToken = function (t) {
        tokenN++;
        if (t) pending += t;
        if (!flushFrame) flushFrame = requestAnimationFrame(flush);
      };
      api.generate(prompt, used).then(function () {
        if (flushFrame) { cancelAnimationFrame(flushFrame); }
        flush();
        finish();
      }, function (err) {
        if (flushFrame) { cancelAnimationFrame(flushFrame); }
        flush();
        // a stop pressed while the device failed: infer.js has unloaded the model, so it is a failure
        if (controller && controller.signal.aborted && api.ready) { finish(); return; }
        text(ui.failuredetail, err && err.message ? String(err.message) : '');
        failCopy(err);
        show(ui.caret, false);
        controller = null;
        setState('failed');
        keepFocus(ui.retry);
        say('The model could not write on this device.');
      });
    }

    /* this run, as infer.js measured it on this device; a slot with no measurement stays out */
    var STOPPED = { eot: 'end of text', maxNewTokens: 'token limit', abort: 'you', passage: 'end of passage', repeat: 'repetition', context: 'context window' };
    // while it writes the tray keeps its place (the card does not jump when the run ends): what is already true
    // (the preset sent, the backend it runs on) and a dash for every value not measured yet
    function pendingReport() {
      if (!ui.run) return;
      text(ui.runpreset, usedName);
      text(ui.runbackend, backendLabel);
      textAll([ui.runtokens, ui.runspeed, ui.runfirst, ui.runstop].filter(Boolean), '\u2014');
      var slots = ui.run.querySelectorAll('dd');
      for (var i = 0; i < slots.length; i++) slots[i].parentNode.hidden = !slots[i].textContent;
      ui.run.setAttribute('aria-busy', 'true');
      show(ui.run, true);
    }
    function report() {
      var l = window.EmbeddingTax && window.EmbeddingTax.last;
      if (ui.run) ui.run.removeAttribute('aria-busy');
      if (!ui.run || !l || !fin(l.newTokens)) { show(ui.run, false); return; }
      text(ui.runpreset, usedName);
      text(ui.runtokens, num(l.newTokens));
      text(ui.runspeed, fin(l.tokensPerSecond) && l.tokensPerSecond > 0 ? l.tokensPerSecond.toFixed(1) + ' tokens/s' : '');
      text(ui.runfirst, fin(l.msToFirstToken) && l.msToFirstToken > 0 ? Math.round(l.msToFirstToken) + ' ms' : '');
      text(ui.runstop, STOPPED[l.stoppedBy] || '');
      text(ui.runbackend, backendLabel);
      var slots = ui.run.querySelectorAll('dd');
      for (var i = 0; i < slots.length; i++) slots[i].parentNode.hidden = !slots[i].textContent;
      show(ui.run, true);
    }

    function finish() {
      show(ui.caret, false);
      runs++;
      controller = null;
      // the measured count: onToken also carries infer.js's trailing UTF-8 flush, which is not a token
      var l = window.EmbeddingTax && window.EmbeddingTax.last;
      if (l && fin(l.newTokens)) { tokenN = l.newTokens; text(ui.tokens, tokenN + (tokenN === 1 ? ' token' : ' tokens')); }
      var fromStop = document.activeElement === ui.stop;
      report();
      setState('done');
      if (fromStop && ui.write) ui.write.focus({ preventScroll: true });
      say('Done, ' + tokenN + (tokenN === 1 ? ' token' : ' tokens'));
    }

    function stop() {
      if (controller && state === 'writing') controller.abort();
    }

    /* ---------- settings: a preset, then slider and box on one row, one summary line, reset to the default preset ---------- */
    function atDefaults() {
      var p = presetOf(settings);
      return !!p && p.id === DEFAULT_PRESET && settings.seed === DEFAULT_SEED;
    }
    function pct(input) {
      var min = Number(input.min), max = Number(input.max), v = Number(input.value);
      return (100 * (v - min) / (max - min)).toFixed(2) + '%';
    }
    function setRange(input, r) {
      if (!input) return;
      input.min = String(r[0]); input.max = String(r[1]); input.step = String(r[2]);
    }
    setRange(ui.temp, RANGE.temperature);
    setRange(ui.topk, RANGE.topK);
    setRange(ui.max, RANGE.maxNewTokens);
    function writeSettings() {
      var preset = presetOf(settings), locked = !settings.guards;
      if (ui.temp) { ui.temp.value = settings.temperature; ui.temp.style.setProperty('--pct', pct(ui.temp)); ui.temp.setAttribute('aria-valuetext', fmtT(settings.temperature)); }
      if (ui.tempN && document.activeElement !== ui.tempN) ui.tempN.value = fmtT(settings.temperature);
      if (ui.topk) { ui.topk.value = settings.topK; ui.topk.style.setProperty('--pct', pct(ui.topk)); ui.topk.setAttribute('aria-valuetext', settings.topK + ' tokens'); }
      if (ui.topkN && document.activeElement !== ui.topkN) ui.topkN.value = String(settings.topK);
      if (ui.max) { ui.max.value = settings.maxNewTokens; ui.max.style.setProperty('--pct', pct(ui.max)); ui.max.setAttribute('aria-valuetext', settings.maxNewTokens + ' tokens'); }
      if (ui.maxN && document.activeElement !== ui.maxN) ui.maxN.value = String(settings.maxNewTokens);
      if (ui.seed && document.activeElement !== ui.seed) ui.seed.value = String(settings.seed);
      // the preset: its radio, its name (or Custom) and what it does
      for (var i = 0; i < ui.modes.length; i++) ui.modes[i].checked = !!preset && ui.modes[i].value === preset.id;
      text(ui.modename, preset ? preset.name : 'Custom');
      text(ui.modedesc, preset ? preset.desc : (settings.guards ? CUSTOM_DESC.guarded : CUSTOM_DESC.unguarded));
      text(ui.guardstate, settings.guards ? 'On' : 'Off');
      show(ui.locknote, locked);
      if (locked !== wasLocked && wasLocked !== null) say(locked ? 'Published uses fixed settings; temperature, top-k and length are locked.' : 'Temperature, top-k and length can be changed.');
      wasLocked = locked;
      // Published is exactly its fixed sampler: its controls show the values and stand down
      if (ui.temp) ui.temp.disabled = locked;
      if (ui.tempN) ui.tempN.disabled = locked;
      if (ui.topk) ui.topk.disabled = locked;
      if (ui.topkN) ui.topkN.disabled = locked;
      // Unlimited: the length controls stand down, and the summary says what ends the run instead.
      // "Not recommended" only without the guards: with them, this stop is what Focused tested well with.
      if (ui.unlimited) ui.unlimited.checked = settings.untilDone;
      show(ui.unlimitedbadge, !settings.guards);
      if (ui.max) ui.max.disabled = locked || settings.untilDone;
      if (ui.maxN) ui.maxN.disabled = locked || settings.untilDone;
      text(ui.summary, (preset ? preset.name : 'Custom') + ' · temperature ' + fmtT(settings.temperature) + ' · top-k ' + settings.topK +
        (settings.untilDone ? ' · until done, at most ' + UNTIL_DONE_CAP + ' tokens' : ' · max ' + settings.maxNewTokens + ' tokens') +
        (settings.guards ? ' · guards on, min-p ' + fmtT(minPFor(settings.temperature)) : ' · no guards') + ' · seed ' + settings.seed);
      if (ui.reset) ui.reset.setAttribute('aria-disabled', String(atDefaults()));
    }
    function bindRange(range, box, key, parse) {
      if (range) range.addEventListener('input', function () { settings[key] = parse(range.value); writeSettings(); });
      if (box) {
        var apply = function () {
          if (box.disabled) return;
          var v = parse(String(box.value).replace(/[,\s]/g, ''));
          if (v === null) { box.value = key === 'temperature' ? fmtT(settings[key]) : String(settings[key]); return; }
          settings[key] = v;
          writeSettings();
        };
        box.addEventListener('change', apply);
        box.addEventListener('keydown', function (e) { if (e.key === 'Enter') { e.preventDefault(); apply(); } });
      }
    }
    bindRange(ui.temp, ui.tempN, 'temperature', function (s) {
      var n = parseFloat(s); if (!isFinite(n)) return null;
      return Math.round(clamp(n, RANGE.temperature[0], RANGE.temperature[1]) * 20) / 20;
    });
    bindRange(ui.topk, ui.topkN, 'topK', function (s) {
      var n = parseInt(s, 10); if (!isFinite(n)) return null;
      return clamp(n, RANGE.topK[0], RANGE.topK[1]);
    });
    bindRange(ui.max, ui.maxN, 'maxNewTokens', function (s) {
      var n = parseInt(s, 10); if (!isFinite(n)) return null;
      return clamp(n, RANGE.maxNewTokens[0], RANGE.maxNewTokens[1]);
    });
    bindRange(null, ui.seed, 'seed', function (s) {
      var n = parseInt(s, 10); if (!isFinite(n)) return null;
      return clamp(n, 0, 2147483647);
    });
    for (var m = 0; m < ui.modes.length; m++) {
      ui.modes[m].addEventListener('change', function () {
        var p = presetById(this.value);
        if (!this.checked) return;
        if (!p) { writeSettings(); return; }       // a preset this table lacks: the radios go back to what runs
        settings = fromPreset(p, settings.seed);   // the seed is not part of a preset
        writeSettings();
      });
    }
    if (ui.reset) {
      ui.reset.addEventListener('click', function () {
        // aria-disabled, not disabled: a button that disables itself under the pointer drops focus
        if (ui.reset.getAttribute('aria-disabled') === 'true') return;
        settings = fromPreset(presetById(DEFAULT_PRESET), DEFAULT_SEED);
        if (ui.seedlock) ui.seedlock.checked = false;
        writeSettings();
      });
    }
    if (ui.unlimited) {
      ui.unlimited.addEventListener('change', function () { settings.untilDone = ui.unlimited.checked; writeSettings(); });
    }
    writeSettings();

    /* ---------- prompt, presets, buttons, keyboard ---------- */
    function promptChanged() {
      runs = 0;
      var v = ui.prompt.value;
      for (var i = 0; i < ui.presets.length; i++) {
        ui.presets[i].setAttribute('aria-pressed', String(ui.presets[i].textContent === v));
      }
      if (state === 'done') setState('ready');
      else if (ui.write) ui.write.disabled = !((state === 'ready') && v.trim().length > 0);
    }
    ui.prompt.addEventListener('input', promptChanged);
    ui.prompt.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        if (state === 'ready' || state === 'done') { e.preventDefault(); write(); }
      }
    });
    for (var p = 0; p < ui.presets.length; p++) {
      ui.presets[p].setAttribute('aria-pressed', 'false');
      ui.presets[p].addEventListener('click', function () {
        ui.prompt.value = this.textContent;
        promptChanged();
        ui.prompt.focus({ preventScroll: true });
      });
    }
    ui.load.addEventListener('click', load);
    if (ui.retry) ui.retry.addEventListener('click', function () { if (state === 'failed') { setState(window.EmbeddingTax && window.EmbeddingTax.ready ? 'ready' : 'idle'); if (!(window.EmbeddingTax && window.EmbeddingTax.ready)) load(); } });
    if (ui.write) ui.write.addEventListener('click', write);
    if (ui.stop) ui.stop.addEventListener('click', stop);
    // Escape stops a run, unless something inside already used it (home.js: closing the settings panel)
    root.addEventListener('keydown', function (e) { if (e.key === 'Escape' && state === 'writing' && !e.defaultPrevented) { e.preventDefault(); stop(); } });
    if (ui.copy) {
      // the pill keeps its icon: only its label (copylabel) changes, and data-copied swaps the glyph in CSS
      var copyLabel = q(root, 'copylabel') || ui.copy, copyTimer = 0;
      ui.copy.addEventListener('click', function () {
        if (!textNode) return;
        navigator.clipboard.writeText(textNode.data).then(function () {
          copyLabel.textContent = 'Copied';
          ui.copy.setAttribute('data-copied', '');
          say('Copied the model\'s words');
          clearTimeout(copyTimer);
          copyTimer = setTimeout(function () { copyLabel.textContent = 'Copy'; ui.copy.removeAttribute('data-copied'); }, 1200);
        }, function () {});
      });
    }
    if (RM && RM.addEventListener) RM.addEventListener('change', function () { if (state === 'writing') show(ui.caret, !reduced()); });

    /* app.html: the writing column (openings, composer, output) rides beside the long settings rail
       (playground.css .pg-write[data-stick]). While it fits the screen it sticks under the sticky header; taller,
       its top goes negative so it sticks by its bottom, 24px above the window's, and a long output never freezes
       below the fold. One continuous rule, so the column never jumps when a run makes it taller than the screen.
       Read when the column or the window changes size, never per frame. */
    var column = root.querySelector('.pg-write');
    if (column && root.querySelector('.pg-rail') && window.ResizeObserver) {
      var headTop = parseFloat(getComputedStyle(column).getPropertyValue('--sticky-top')) || 84;
      var place = function () { column.style.setProperty('--stick-top', Math.min(headTop, window.innerHeight - column.offsetHeight - 24) + 'px'); };
      column.setAttribute('data-stick', '');
      new ResizeObserver(place).observe(column);
      window.addEventListener('resize', place);
    }

    ui.load.disabled = true;
    setState('init');
    probe();
  }

  function init() {
    var roots = document.querySelectorAll('[data-playground]');
    for (var i = 0; i < roots.length; i++) mount(roots[i]);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
