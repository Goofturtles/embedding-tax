/* The Embedding Tax - run the model in the visitor's browser.
 *
 * Classic script, no modules, no build step. Loads onnxruntime-web from jsdelivr (pinned)
 * and the weights from ./model/ next to this file. Everything runs on the visitor's device;
 * nothing is sent anywhere.
 *
 *   window.EmbeddingTax = {
 *     load({ onProgress, backend })  -> Promise<{ backend: "webgpu"|"wasm", manifest }>
 *         idempotent; downloads once (also kept in Cache Storage across visits, keyed by the
 *         manifest sha256, so a re-export invalidates it); onProgress(loadedBytes, totalBytes).
 *         The bytes are checked before anything trusts them: SHA-256 against manifest.sha256
 *         (a stale or torn download is refused and never cached; a cached entry that no longer
 *         hashes right is dropped and re-downloaded), then the manifest's parity self-check (a
 *         fixed context whose argmax the exporter measured on the shipped file) on every backend
 *         candidate: a backend that gets it wrong is not used. When only one session will be built
 *         (a phone, or a browser without WebGPU), bytes whose SHA-256 matched are cached before it
 *         (a tab that dies building it reopens from the cache, no second download); otherwise, and
 *         with no SHA-256 (an insecure context), they reach Cache Storage only after a backend has
 *         passed. A download that sends nothing
 *         for 30 s of visible time, and a Worker that goes silent for 30 s mid-call, reject with a
 *         readable error instead of hanging. backend: "auto" (default) | "wasm" | "webgpu".
 *         cacheOnly: true builds the model only from weights already in Cache Storage (they get
 *         there only after an earlier load passed): nothing is downloaded, and a miss or a corrupt
 *         entry rejects with err.code "NOT_CACHED" instead of falling back to the network.
 *         "auto" benchmarks a warm-up forward on wasm and, when the browser has WebGPU, on
 *         WebGPU too, and keeps the faster one (the first call's preference wins). On a
 *         constrained() device "auto" builds wasm alone, never two sessions at once. The wasm
 *         backend always runs in a dedicated Worker with its own copy of the wasm-only
 *         onnxruntime build, so wasm compute never blocks the page, with or without WebGPU.
 *         A load that fails on memory rejects with err.code "OUT_OF_MEMORY".
 *     constrained() -> boolean
 *         a phone or tablet ((pointer: coarse) and (hover: none)), a browser reporting 4 GB of
 *         memory or less, or an iPad: see E.constrained. Pages use it to wait for a tap before
 *         reopening saved weights.
 *     interrupted -> boolean
 *         this tab died once while building the model (see BUILDING); a constrained device then
 *         builds the smaller session for the rest of the tab's visit (as it does after a load that
 *         failed with OUT_OF_MEMORY).
 *     generate(prompt, { maxNewTokens=60, temperature=0.8, topK=40, seed=1337, untilDone=false,
 *                        topP=1, minP=0, repetitionPenalty=1, repetitionWindow=0, frequencyPenalty=0,
 *                        presencePenalty=0, noRepeatNgram=0, noLeadingBreak=0, noMidsentenceBreak=false,
 *                        cadAlpha=0, bestOf=1, bestOfFluencyWeight=0,
 *                        onToken(text, tokenId), onTruncate(info), signal }) -> Promise<string>
 *         Decoding constraints, all off by default (so the defaults are exactly tools/sample.py). None
 *         of them edits model text: each one is a logit penalty, a logit mask, a stopping rule or a
 *         choice between whole unedited drafts. They follow the decode harness (harness.py) step for
 *         step, in this order, in float64:
 *           1. the forward on prompt + output; when cadAlpha != 0 or bestOf > 1 also a prompt-free
 *              forward on [<|endoftext|>] + output (its last block_size tokens): two forwards per token
 *           2. cadAlpha a (context-aware decoding): logits = (1 + a) * with_prompt - a * prompt_free
 *           3. repetitionPenalty p (CTRL): a logit v of every token in the last repetitionWindow tokens
 *              of prompt + output (0 = all of them) becomes v / p if v > 0, else v * p; the
 *              <|endoftext|> id is never penalised
 *           4. frequencyPenalty f, presencePenalty q: logit -= f * count + q for each token already in
 *              the output (counts over the output only)
 *           5. masks to -Infinity: noRepeatNgram n (a token that would complete an n-gram already in the
 *              output); noLeadingBreak N (for the first N new tokens, <|endoftext|> and every token whose
 *              bytes hold a line break); noMidsentenceBreak (outside those N, a line-break token unless
 *              prompt + output plus the token's text before its break, right-trimmed, ends a sentence:
 *              SENTENCE_END below)
 *           6. temperature <= 0: argmax. Else keep logits >= the top-k-th largest and > -Infinity;
 *              w = exp((logit - max) * (1 / temperature)); topP < 1: keep the shortest run of them by w
 *              descending (ties: lower id first) whose sum reaches topP of their total; minP > 0: keep
 *              w >= minP (the top token has w = 1); one mulberry32 draw over the survivors in id order.
 *         bestOf N runs N whole drafts, draft i seeded (seed + Math.imul(i, 0x9E3779B9)) | 0 (draft 0
 *         is the plain run), and returns the one with the highest score = mean over its tokens of
 *         (log p(token | prompt + output) - log p(token | output alone)) + bestOfFluencyWeight * mean
 *         log p(token | prompt + output), from the raw model logits; ties go to the lower draft, an
 *         empty draft never wins over a scored one. With bestOf > 1 nothing streams until every draft
 *         is done, then onToken replays the chosen draft token by token; an abort stops the draft in
 *         flight and picks among the drafts written so far, the partial one included.
 *         untilDone: no token limit, but the run ends when the passage does: at <|endoftext|>, at
 *         the first line break that follows a finished sentence once 24 tokens are written
 *         (stoppedBy "passage"; the break is not emitted as text but its token still counts in newTokens/ids), when the last 8 tokens repeat
 *         an earlier stretch (a loop, "repeat"), or when prompt + output fill the block_size
 *         context ("context"), or at 256 new tokens ("maxNewTokens"). maxNewTokens is ignored and the prompt keeps up to block_size - 64
 *         tokens.
 *         streams decoded text as it goes: onToken is called once per generated token with the
 *         text that token completed (only whole UTF-8 characters are emitted, so `text` is ""
 *         while a multi-byte character is still being assembled and the rest arrives with the
 *         token that finishes it). Stops at <|endoftext|>, at maxNewTokens, or when `signal`
 *         (an AbortSignal) aborts; on abort it resolves with the text so far, at once, even while a
 *         forward is still in flight (that forward finishes on its own; the next one waits for it). Sampling matches
 *         tools/sample.py: logits / temperature, keep the top-k (temperature <= 0 is greedy),
 *         softmax, one draw from a generator seeded with `seed` (mulberry32), so the same
 *         (prompt, settings, seed) reproduces the same text on the same backend. Prompts longer
 *         than block_size - maxNewTokens tokens are truncated from the LEFT; that is reported
 *         through the optional onTruncate({ promptTokens, keptTokens, droppedTokens,
 *         maxPromptTokens, promptChars, keptChars, droppedChars }) callback and through
 *         EmbeddingTax.last.truncated (EmbeddingTax.last also carries newTokens, stoppedBy:
 *         "eot"|"maxNewTokens"|"abort"|"passage"|"repeat"|"context", ms, tokensPerSecond, forwards,
 *         every decoding option as resolved (topP, minP, repetitionPenalty, repetitionWindow,
 *         frequencyPenalty, presencePenalty, noRepeatNgram, noLeadingBreak, noMidsentenceBreak,
 *         cadAlpha, bestOf, bestOfFluencyWeight), chosenDraft, chosenSeed, meanPmi and meanLogp (null
 *         unless the prompt-free forward ran), and drafts: [{ draft, seed, score, meanPmi, meanLogp,
 *         newTokens, stoppedBy, text }], one per draft; score is null when there was no prompt-free
 *         forward or no token). A prompt is first cut to its last
 *         maxPromptTokens x (longest token in bytes) characters, a span that already holds at
 *         least maxPromptTokens tokens, so only that much is ever tokenized; promptTokens then
 *         counts the part that was encoded. An empty prompt starts from <|endoftext|>, the token
 *         that separates documents in training. Calls are queued: one generation runs at a time,
 *         in call order. If the backend fails mid-generation (a forward rejects: WebGPU device
 *         lost while the tab sat in the background, a crashed worker; or returns non-finite
 *         logits) the promise rejects with a readable error, the backend is released and ready
 *         goes back to false: load() rebuilds it from the cached weights.
 *     unload() -> Promise
 *         aborts the generate in flight (it resolves with its text so far), releases the session,
 *         terminates the worker (its whole wasm heap, which never shrinks while it lives: ~425 MB
 *         with the folded weights, ~355 MB without prepacking, ~170 MB unfolded), resets
 *         ready/backend; tokenCount() keeps working
 *         and a later load() rebuilds everything.
 *     isCached(manifest?) -> Promise<boolean>
 *         whether this device already holds the current weights (keyed by manifest sha256), so a
 *         page can reopen the model on a later visit without asking for the download again.
 *     tokenCount(text) -> number      (available once load() has fetched the tokenizer)
 *     encode(text) -> number[], decode(ids, { skipSpecial=true }) -> string
 *     ready -> boolean; manifest -> parsed manifest or null; backend -> "webgpu"|"wasm"|null;
 *     stats -> timings gathered by load(); last -> stats of the most recent generate().
 *   }
 *
 * The tokenizer is byte-level BPE (GPT-2 style), implemented here from model/tokenizer.json:
 * no normalizer, ByteLevel pre-tokenizer (GPT-2 regex, no prefix space), BPE merges, ByteLevel
 * decoder; the single added token <|endoftext|> is matched literally in prompts, as the Python
 * tokenizers library does.
 */
(function () {
  'use strict';

  var ORT_VERSION = '1.29.0';
  var ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@' + ORT_VERSION + '/dist/';
  // ort.wasm.min.js pulls the plain wasm binary (14 MB raw, ~2.7 MB brotli); ort.min.js and the
  // WebGPU build pull the JSEP binary (28 MB raw, ~5.4 MB brotli), so only pay for it with WebGPU.
  var ORT_SCRIPT = { wasm: 'ort.wasm.min.js', webgpu: 'ort.webgpu.min.js' };
  // Subresource integrity of those two files: the page's <script> tag carries it as an integrity
  // attribute, and the Worker hashes the text it fetched before running it (importScripts has no
  // such attribute), so a CDN serving anything but the pinned build is refused. The .mjs/.wasm
  // files the runtime then pulls from ORT_CDN are not covered. Refresh after bumping ORT_VERSION:
  //   python -c "import hashlib,base64,urllib.request as u;V='1.29.0';[print(f,'sha384-'+base64.b64encode(hashlib.sha384(u.urlopen(u.Request('https://cdn.jsdelivr.net/npm/onnxruntime-web@'+V+'/dist/'+f,headers={'Accept-Encoding':'identity'})).read()).digest()).decode())for f in('ort.wasm.min.js','ort.webgpu.min.js')]"
  var ORT_INTEGRITY = {
    wasm: 'sha384-DfEnWLXuEOU4PRJhIXAuIsQhFvN453lwU7ictUMLLm3uyjMOvi5V87PJq21S1+MW',
    webgpu: 'sha384-spJihwjw5DABHdp7XVsBfqbR7hlxe8gdmFg+FQUzeWG58zV7iOKXd30heVr3CZG0'
  };
  var CACHE_NAME = 'embedding-tax-weights';
  var BENCH_T = 48;          // context length of the warm-up forward used to pick a backend
  // Stall limits, in visible time (visibleClock). A phone that loses its network mid-download, or a
  // Worker that dies without an error event, used to leave the panel on "Loading" forever.
  var DOWNLOAD_STALL_MS = 30000;   // no byte of the weights for this long: the download is cancelled
  var WORKER_SILENCE_MS = 30000;   // a busy Worker speaks once a second; silence this long means it is gone.
                                   // The longest single step measured (a 512-token forward, 1.3 s on a
                                   // desktop core) leaves a phone many times slower well inside it.
  var WORKER_CAP_MS = { init: 90000, create: 240000 };   // an op that never ends while the Worker lives (a CDN request that hangs)
  var OP_TEXT = { init: 'fetching onnxruntime', create: 'building the model', run: 'running the model' };
  // What an engine says when it runs out of memory: V8/JSC RangeErrors, Emscripten's OOM abort,
  // C++ bad_alloc, a WebAssembly.Memory that cannot grow. Only these get err.code OUT_OF_MEMORY.
  var OOM_TEXT = /out of memory|\bOOM\b|bad_alloc|maximum memory size|could not allocate|failed to allocate|cannot allocate|allocation failed/i;

  var scriptUrl = (document.currentScript && document.currentScript.src) || location.href;
  var BASE = scriptUrl.slice(0, scriptUrl.lastIndexOf('/') + 1);
  var MODEL_DIR = BASE + 'model/';

  // ---------------------------------------------------------------- byte-level BPE tokenizer

  // GPT-2's bytes_to_unicode: every byte gets a printable code point so BPE can run on strings.
  function byteUnicodeTables() {
    var bs = [], i;
    for (i = 33; i <= 126; i++) bs.push(i);
    for (i = 161; i <= 172; i++) bs.push(i);
    for (i = 174; i <= 255; i++) bs.push(i);
    var cs = bs.slice(), n = 0;
    for (var b = 0; b < 256; b++) {
      if (bs.indexOf(b) < 0) { bs.push(b); cs.push(256 + n); n++; }
    }
    var b2u = new Array(256), u2b = new Map();
    for (i = 0; i < bs.length; i++) {
      b2u[bs[i]] = String.fromCharCode(cs[i]);
      u2b.set(cs[i], bs[i]);
    }
    return { b2u: b2u, u2b: u2b };
  }

  // Oniguruma's \s (what the tokenizers crate uses): U+0009-000D, U+0085 and the Unicode
  // Space/Line/Paragraph separators. JS \s differs (has U+FEFF, lacks U+0085), so spell it out.
  var WS = '\\t\\n\\x0B\\f\\r \\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000';
  var GPT2_PATTERN = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^" + WS + "\\p{L}\\p{N}]+|[" +
    WS + "]+(?![^" + WS + "])|[" + WS + "]+";

  var utf8enc = new TextEncoder();

  function ByteLevelBPE(json) {
    if (json.normalizer) throw new Error('tokenizer: unexpected normalizer ' + JSON.stringify(json.normalizer));
    if (!json.pre_tokenizer || json.pre_tokenizer.type !== 'ByteLevel') throw new Error('tokenizer: expected ByteLevel pre_tokenizer');
    if (!json.decoder || json.decoder.type !== 'ByteLevel') throw new Error('tokenizer: expected ByteLevel decoder');
    if (!json.model || json.model.type !== 'BPE') throw new Error('tokenizer: expected BPE model');
    if (json.post_processor) throw new Error('tokenizer: unexpected post_processor');

    this.addPrefixSpace = !!json.pre_tokenizer.add_prefix_space;
    this.useRegex = json.pre_tokenizer.use_regex !== false;
    this.re = new RegExp(GPT2_PATTERN, 'gu');

    var t = byteUnicodeTables();
    this.b2u = t.b2u;
    this.u2b = t.u2b;

    var vocab = json.model.vocab, k;
    this.vocab = new Map();
    this.idToToken = [];
    for (k in vocab) {
      if (Object.prototype.hasOwnProperty.call(vocab, k)) {
        this.vocab.set(k, vocab[k]);
        this.idToToken[vocab[k]] = k;
      }
    }
    // Added tokens (here just <|endoftext|>) are matched literally on the raw text before
    // pre-tokenization, longest first.
    this.added = [];
    this.specialIds = new Set();
    var added = json.added_tokens || [], i;
    for (i = 0; i < added.length; i++) {
      var at = added[i];
      this.added.push({ content: at.content, id: at.id, special: !!at.special });
      this.idToToken[at.id] = at.content;
      if (at.special) this.specialIds.add(at.id);
    }
    this.added.sort(function (x, y) { return y.content.length - x.content.length; });
    this.size = this.idToToken.length;
    // Merges on token ids, as the tokenizers crate keeps them: pair (a, b) -> its rank and the id
    // of a+b (a merge whose result is not a token is an invalid file there too). The key packs
    // both ids into one number so the merge loop never builds a string to look a pair up.
    var K = this.size;
    this.pairBase = K;
    this.merges = new Map();
    var merges = json.model.merges;
    for (i = 0; i < merges.length; i++) {
      var m = merges[i];
      var a, b;
      if (Array.isArray(m)) { a = m[0]; b = m[1]; }
      else { var sp = m.indexOf(' '); a = m.slice(0, sp); b = m.slice(sp + 1); }
      var ia = this.vocab.get(a), ib = this.vocab.get(b), iab = this.vocab.get(a + b);
      if (ia === undefined || ib === undefined || iab === undefined) throw new Error('tokenizer: merge ' + i + ' names a token that is not in the vocab');
      this.merges.set(ia * K + ib, { rank: i, id: iab });
    }
    // Each byte's symbol as a token id; -1 (the vocab lacks it) is dropped before merging, which
    // is what HF does without an unk token.
    this.byteId = new Int32Array(256);
    for (i = 0; i < 256; i++) {
      var bid = this.vocab.get(this.b2u[i]);
      this.byteId[i] = bid === undefined ? -1 : bid;
    }
    this.cache = new Map();
    this.byteCache = new Map();
    // The longest token in bytes bounds how much of a prompt can matter (see generate()).
    this.maxTokenBytes = 1;
    for (i = 0; i < this.size; i++) this.maxTokenBytes = Math.max(this.maxTokenBytes, this.tokenBytes(i).length);
  }

  ByteLevelBPE.prototype.encode = function (text) {
    var out = [];
    text = String(text);
    if (!this.added.length) { this._encodeChunk(text, out); return out; }
    var pos = 0;
    while (pos < text.length) {
      var bestAt = -1, bestTok = null;
      for (var i = 0; i < this.added.length; i++) {
        var at = text.indexOf(this.added[i].content, pos);
        if (at >= 0 && (bestAt < 0 || at < bestAt)) { bestAt = at; bestTok = this.added[i]; }
      }
      if (bestAt < 0) { this._encodeChunk(text.slice(pos), out); break; }
      if (bestAt > pos) this._encodeChunk(text.slice(pos, bestAt), out);
      out.push(bestTok.id);
      pos = bestAt + bestTok.content.length;
    }
    return out;
  };

  ByteLevelBPE.prototype._encodeChunk = function (str, out) {
    if (!str.length) return;
    if (this.addPrefixSpace && str[0] !== ' ') str = ' ' + str;
    if (!this.useRegex) { this._word(str, out); return; }
    this.re.lastIndex = 0;
    var m;
    while ((m = this.re.exec(str)) !== null) {
      if (m[0].length === 0) { this.re.lastIndex++; continue; }
      this._word(m[0], out);
    }
  };

  ByteLevelBPE.prototype._word = function (piece, out) {
    var ids = this.cache.get(piece);
    if (ids === undefined) {
      var bytes = utf8enc.encode(piece);
      ids = [];
      for (var i = 0; i < bytes.length; i++) {
        var id = this.byteId[bytes[i]];
        if (id >= 0) ids.push(id);   // no unk token: a byte the vocab lacks is dropped, as in HF
      }
      ids = this._bpe(ids);
      if (this.cache.size > 50000) this.cache.clear();
      this.cache.set(piece, ids);
    }
    for (var j = 0; j < ids.length; j++) out.push(ids[j]);
  };

  // Merge one piece's symbols (token ids) the way the tokenizers crate's Word::merge_all does:
  // the symbols sit in a doubly linked list and every mergeable adjacent pair is in a heap
  // ordered by (rank, position), so the lowest-ranked, leftmost pair always goes next and only
  // its two new neighbours are looked at again. A piece of n symbols costs O(n log n); the old
  // "rescan everything after every merge" loop was O(n^2) and froze the tab on a 50k-digit run,
  // which \p{N}+ makes one piece. A popped entry may be stale (one of its symbols has merged
  // since): it is skipped when the pair at that position no longer yields the same token.
  ByteLevelBPE.prototype._bpe = function (ids) {
    var n = ids.length;
    if (n < 2) return ids;
    var merges = this.merges, K = this.pairBase;
    var next = new Int32Array(n), prev = new Int32Array(n), gone = new Uint8Array(n);
    var heap = [];   // {r: rank, p: position of the left symbol, id: the merged token}
    var before = function (x, y) { return x.r < y.r || (x.r === y.r && x.p < y.p); };
    var push = function (r, p, id) {
      var e = { r: r, p: p, id: id }, i = heap.length;
      heap.push(e);
      while (i > 0) {
        var j = (i - 1) >> 1;
        if (before(heap[j], e)) break;
        heap[i] = heap[j]; i = j;
      }
      heap[i] = e;
    };
    var pop = function () {
      var top = heap[0], e = heap.pop(), len = heap.length, i = 0;
      if (!len) return top;
      for (;;) {
        var l = 2 * i + 1, k = l;
        if (l >= len) break;
        if (l + 1 < len && before(heap[l + 1], heap[l])) k = l + 1;
        if (before(e, heap[k])) break;
        heap[i] = heap[k]; i = k;
      }
      heap[i] = e;
      return top;
    };
    var i, m;
    for (i = 0; i < n; i++) { prev[i] = i - 1; next[i] = i + 1 < n ? i + 1 : -1; }
    for (i = 0; i < n - 1; i++) { m = merges.get(ids[i] * K + ids[i + 1]); if (m) push(m.rank, i, m.id); }
    while (heap.length) {
      var e = pop(), p = e.p, q = next[p];
      if (gone[p] || q < 0) continue;
      m = merges.get(ids[p] * K + ids[q]);
      if (!m || m.id !== e.id) continue;
      ids[p] = e.id;
      gone[q] = 1;
      next[p] = next[q];
      if (next[p] >= 0) prev[next[p]] = p;
      if (prev[p] >= 0) { m = merges.get(ids[prev[p]] * K + ids[p]); if (m) push(m.rank, prev[p], m.id); }
      if (next[p] >= 0) { m = merges.get(ids[p] * K + ids[next[p]]); if (m) push(m.rank, p, m.id); }
    }
    var out = [];
    for (i = 0; i >= 0; i = next[i]) out.push(ids[i]);
    return out;
  };

  // Bytes of one token: every char through the byte table, else (an added token whose text is
  // outside the table) the token's own UTF-8 bytes, which is what the ByteLevel decoder does.
  ByteLevelBPE.prototype.tokenBytes = function (id) {
    var cached = this.byteCache.get(id);
    if (cached) return cached;
    var tok = this.idToToken[id];
    var bytes;
    if (tok === undefined) {
      bytes = new Uint8Array(0);
    } else {
      bytes = new Uint8Array(tok.length);
      var ok = true;
      for (var i = 0; i < tok.length; i++) {
        var b = this.u2b.get(tok.charCodeAt(i));
        if (b === undefined) { ok = false; break; }
        bytes[i] = b;
      }
      if (!ok) bytes = utf8enc.encode(tok);
    }
    this.byteCache.set(id, bytes);
    return bytes;
  };

  ByteLevelBPE.prototype.decode = function (ids, opts) {
    var skipSpecial = !opts || opts.skipSpecial !== false;
    var parts = [], total = 0, i;
    for (i = 0; i < ids.length; i++) {
      if (skipSpecial && this.specialIds.has(ids[i])) continue;
      var b = this.tokenBytes(ids[i]);
      parts.push(b);
      total += b.length;
    }
    var all = new Uint8Array(total), off = 0;
    for (i = 0; i < parts.length; i++) { all.set(parts[i], off); off += parts[i].length; }
    // lossy: invalid bytes become U+FFFD; ignoreBOM keeps a leading U+FEFF like Python does
    return new TextDecoder('utf-8', { ignoreBOM: true }).decode(all);
  };

  // ---------------------------------------------------------------- sampling (tools/sample.py)

  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function argmax(logits) {
    var best = 0, bv = logits[0];
    for (var i = 1; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
  }

  function allFinite(logits) {
    for (var i = 0; i < logits.length; i++) if (!isFinite(logits[i])) return false;
    return true;
  }

  // Scratch arrays for the topP/minP path, vocab-sized, made once and reused by every step.
  function makeScratch(V) {
    return { idx: new Int32Array(V), w: new Float64Array(V), order: new Int32Array(V), flags: new Uint8Array(V) };
  }

  // logits / temperature; keep values >= the k-th largest (ties kept, like torch.topk + `< kth`)
  // and above -Infinity (a masked token); softmax over the survivors; one multinomial draw from
  // `rng`. topP < 1 and minP > 0 narrow the survivors first (see the header, step 6). With both off
  // this is the published sample.py draw, operation for operation.
  function sample(logits, temperature, topK, rng, topP, minP, scratch) {
    var V = logits.length, i, v;
    if (!(temperature > 0)) return argmax(logits);
    var kth = -Infinity, maxv = -Infinity;
    if (topK > 0 && topK < V) {
      var top = new Float64Array(topK);
      for (i = 0; i < topK; i++) top[i] = -Infinity;
      for (i = 0; i < V; i++) {
        var v = logits[i];
        if (v > top[topK - 1]) {
          var j = topK - 1;
          while (j > 0 && top[j - 1] < v) { top[j] = top[j - 1]; j--; }
          top[j] = v;
        }
      }
      kth = top[topK - 1];
      maxv = top[0];
    } else {
      for (i = 0; i < V; i++) if (logits[i] > maxv) maxv = logits[i];
    }
    var inv = 1 / temperature, sum = 0, n = 0, u, acc = 0, j;
    if (!(topP < 1) && !(minP > 0)) {
      for (i = 0; i < V; i++) { v = logits[i]; if (v >= kth && v > -Infinity) { sum += Math.exp((v - maxv) * inv); n++; } }
      if (!n) return argmax(logits);
      u = rng() * sum;
      var lastOk = -1;
      for (i = 0; i < V; i++) {
        v = logits[i];
        if (!(v >= kth && v > -Infinity)) continue;
        lastOk = i;
        acc += Math.exp((v - maxv) * inv);
        if (acc >= u) return i;
      }
      return lastOk;
    }
    scratch = scratch || makeScratch(V);
    var idx = scratch.idx, w = scratch.w;
    for (i = 0; i < V; i++) { v = logits[i]; if (v >= kth && v > -Infinity) { idx[n] = i; w[n] = Math.exp((v - maxv) * inv); n++; } }
    if (!n) return argmax(logits);
    if (topP < 1) {
      // the shortest run by weight descending (ties: lower id first) whose running sum reaches topP x total
      var order = scratch.order.subarray(0, n), flags = scratch.flags;
      for (j = 0; j < n; j++) order[j] = j;
      order.sort(function (x, y) { return (w[y] - w[x]) || (x - y); });
      var total = 0, target, keep = n;
      for (j = 0; j < n; j++) total += w[order[j]];
      target = topP * total;
      for (j = 0; j < n; j++) { acc += w[order[j]]; if (acc >= target) { keep = j + 1; break; } }
      for (j = 0; j < keep; j++) flags[order[j]] = 1;
      var m = 0;
      for (j = 0; j < n; j++) if (flags[j]) { flags[j] = 0; idx[m] = idx[j]; w[m] = w[j]; m++; }
      n = m;
      acc = 0;
    }
    if (minP > 0) {
      var k = 0;
      for (j = 0; j < n; j++) if (w[j] >= minP) { idx[k] = idx[j]; w[k] = w[j]; k++; }
      n = k;
      if (!n) return argmax(logits);
    }
    for (j = 0; j < n; j++) sum += w[j];
    u = rng() * sum;
    for (j = 0; j < n; j++) { acc += w[j]; if (acc >= u) return idx[j]; }
    return idx[n - 1];
  }

  // log softmax(logits)[id], the way harness.py's metrics take it: max + log(sum(exp(l - max))).
  function logSoftmaxAt(logits, id) {
    var V = logits.length, m = -Infinity, s = 0, i;
    for (i = 0; i < V; i++) if (logits[i] > m) m = logits[i];
    for (i = 0; i < V; i++) s += Math.exp(logits[i] - m);
    return logits[id] - (m + Math.log(s));
  }

  // ---------------------------------------------------------------- helpers

  function loadScript(src, integrity) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src;
      s.async = true;
      // The browser hashes the response and runs nothing but the pinned build; SRI on a
      // cross-origin script needs the CORS mode (jsdelivr answers with Access-Control-Allow-Origin).
      s.integrity = integrity;
      s.crossOrigin = 'anonymous';
      s.onload = function () { resolve(); };
      s.onerror = function () { reject(new Error('failed to load ' + src + ' (network error, or the file did not match its pinned integrity hash)')); };
      document.head.appendChild(s);
    });
  }

  function fetchJSON(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error('fetch ' + url + ': HTTP ' + r.status);
      return r.json();
    });
  }

  var yieldToUI = (function () {
    if (typeof MessageChannel === 'undefined') {
      return function () { return new Promise(function (r) { setTimeout(r, 0); }); };
    }
    var ch = new MessageChannel(), waiters = [];
    ch.port1.onmessage = function () { var w = waiters; waiters = []; for (var i = 0; i < w.length; i++) w[i](); };
    return function () { return new Promise(function (r) { waiters.push(r); ch.port2.postMessage(0); }); };
  })();

  function now() { return performance.now(); }
  function noop() {}

  // Calls tick(ms) about once a second with the time that passed while the page was visible. Time
  // spent hidden, and a tick that arrives late (over 2.5 s: the tab was frozen, as iOS does to a
  // background tab), count for nothing, so a load that was only paused is never called stalled
  // when the visitor comes back. Returns stop().
  function visibleClock(tick) {
    var last = now();
    var id = setInterval(function () {
      var t = now(), dt = t - last;
      last = t;
      if (dt <= 2500 && !document.hidden) tick(dt);
    }, 1000);
    return function () { clearInterval(id); };
  }

  // ---------------------------------------------------------------- wasm worker
  //
  // The wasm backend lives in its own Worker with its own copy of the wasm-only onnxruntime
  // build; the page thread only ever receives Float32Array logits. (onnxruntime's proxy mode
  // does the same but is unavailable in the WebGPU build, which put a wasm fallback on the
  // main thread whenever navigator.gpu existed: a 2 s freeze at load and 50-110 ms tasks per
  // token.) Messages: {id, op: init|create|run|release, ...} -> {id, ok, result|error}; while an op
  // is in flight the Worker also sends {beat: 1} once a second (between its synchronous stretches),
  // so the page can tell a busy Worker from a dead one (WasmWorker._clockOn). Idle, it sends nothing.
  var WASM_WORKER_SRC = [
    "'use strict';",
    "var session = null;",
    "var busy = 0, beat = 0;",
    "function begin() { if (!busy++) beat = setInterval(function () { self.postMessage({ beat: 1 }); }, 1000); }",
    "function end() { if (!--busy) clearInterval(beat); }",
    "function errText(e) { return typeof e === 'number' ? 'native exception #' + e : String(e && e.message || e); }",
    "self.onmessage = function (ev) {",
    "  var m = ev.data;",
    "  begin();",
    "  Promise.resolve().then(function () {",
    "    if (m.op === 'init') {",
    "      // importScripts cannot carry an integrity attribute, so fetch the runtime, hash it and run",
    "      // those very bytes from a blob URL; its wasmPaths (the .mjs/.wasm) stay on the CDN.",
    "      return fetch(m.script).then(function (r) {",
    "        if (!r.ok) throw new Error('fetch ' + m.script + ': HTTP ' + r.status);",
    "        return r.arrayBuffer();",
    "      }).then(function (buf) {",
    "        var run = function () { importScripts(URL.createObjectURL(new Blob([buf], { type: 'text/javascript' }))); ort.env.wasm.wasmPaths = m.wasmPaths; };",
    "        // No crypto.subtle means an insecure context, where the page itself is unprotected too.",
    "        if (!(self.crypto && self.crypto.subtle)) { run(); return 'unavailable'; }",
    "        return self.crypto.subtle.digest('SHA-384', buf).then(function (d) {",
    "          var got = 'sha384-' + btoa(String.fromCharCode.apply(null, new Uint8Array(d)));",
    "          if (got !== m.integrity) throw new Error('integrity mismatch: ' + m.script + ' hashes to ' + got + ', the pin is ' + m.integrity);",
    "          run();",
    "          return 'verified';",
    "        });",
    "      });",
    "    }",
    "    if (m.op === 'create') {",
    "      return ort.InferenceSession.create(new Uint8Array(m.bytes), m.opts).then(function (s) {",
    "        session = s;",
    "        return { inputNames: s.inputNames, outputNames: s.outputNames, numThreads: ort.env.wasm.numThreads };",
    "      });",
    "    }",
    "    if (m.op === 'run') {",
    "      var ids = m.ids, data = new BigInt64Array(ids.length);",
    "      for (var i = 0; i < ids.length; i++) data[i] = BigInt(ids[i]);",
    "      var feeds = {};",
    "      feeds[session.inputNames[0]] = new ort.Tensor('int64', data, [1, ids.length]);",
    "      return session.run(feeds).then(function (out) { return new Float32Array(out[session.outputNames[0]].data); });",
    "    }",
    "    if (m.op === 'release') { var s = session; session = null; return s ? s.release() : null; }",
    "    throw new Error('unknown op ' + m.op);",
    "  }).then(function (r) {",
    "    end();",
    "    self.postMessage({ id: m.id, ok: true, result: r }, r instanceof Float32Array ? [r.buffer] : []);",
    "  }, function (e) {",
    "    end();",
    "    self.postMessage({ id: m.id, ok: false, error: errText(e) });",
    "  });",
    "};"
  ].join('\n');

  function WasmWorker() {
    this.url = URL.createObjectURL(new Blob([WASM_WORKER_SRC], { type: 'text/javascript' }));
    this.worker = new Worker(this.url);
    this.pending = new Map();
    this.nextId = 1;
    this.dead = null;        // once terminated: the error every later call rejects with at once
    this.quiet = 0;          // visible ms since the Worker last said anything, while calls are pending
    this.stopClock = null;
    var self = this;
    this.worker.onmessage = function (ev) {
      var m = ev.data;
      self.quiet = 0;
      if (m && m.beat) return;
      var p = self.pending.get(m.id);
      if (!p) return;
      self.pending.delete(m.id);
      if (!self.pending.size) self._clockOff();
      if (m.ok) p.resolve(m.result); else p.reject(new Error('wasm worker: ' + m.error));
    };
    this.worker.onerror = function (ev) {
      self._failAll(new Error('wasm worker: ' + (ev && ev.message || 'failed to start')));
    };
  }
  WasmWorker.prototype.call = function (msg, transfer) {
    var self = this;
    if (this.dead) return Promise.reject(this.dead);
    return new Promise(function (resolve, reject) {
      msg.id = self.nextId++;
      self.pending.set(msg.id, { resolve: resolve, reject: reject, op: msg.op, waited: 0 });
      self._clockOn();
      self.worker.postMessage(msg, transfer || []);
    });
  };
  // While calls are pending, count visible time. WORKER_SILENCE_MS without a word (a living Worker
  // beats once a second while busy), or an op past its WORKER_CAP_MS, ends the Worker: every pending
  // call and every later one rejects with what happened, which the pages show with Try again. A
  // Worker that dies without an error event (the tab's memory reclaimed, a crash) otherwise left
  // "Loading" or "Writing" on screen for good.
  WasmWorker.prototype._clockOn = function () {
    if (this.stopClock) return;
    var self = this;
    this.quiet = 0;
    this.stopClock = visibleClock(function (dt) {
      var why = null, doing = null;
      self.quiet += dt;
      self.pending.forEach(function (p) {
        p.waited += dt;
        doing = doing || OP_TEXT[p.op] || p.op;
        var cap = WORKER_CAP_MS[p.op];
        if (!why && cap && p.waited >= cap) why = (OP_TEXT[p.op] || p.op) + ' did not finish within ' + cap / 1000 + ' s';
      });
      // A build is one long blocking wasm call, so the once-a-second heartbeat cannot fire during
      // it: silence is normal there (a phone building the unfolded weights takes well over 30 s),
      // and the build's own 240 s cap is what catches a real hang.
      var building = false;
      self.pending.forEach(function (p) { if (p.op === 'create') building = true; });
      if (!why && !building && self.quiet >= WORKER_SILENCE_MS) why = 'it stopped answering while ' + doing + ' (nothing from it for ' + WORKER_SILENCE_MS / 1000 + ' s)';
      if (why) self.terminate(new Error('wasm worker: ' + why));
    });
  };
  WasmWorker.prototype._clockOff = function () {
    if (this.stopClock) { this.stopClock(); this.stopClock = null; }
  };
  WasmWorker.prototype._failAll = function (err) {
    var p = this.pending;
    this.pending = new Map();
    this._clockOff();
    p.forEach(function (w) { w.reject(err); });
  };
  WasmWorker.prototype.terminate = function (err) {
    if (!this.dead) {
      this.worker.terminate();
      URL.revokeObjectURL(this.url);
      this.dead = err || new Error('wasm worker terminated');
    }
    this._failAll(this.dead);
  };

  // A backend handle: run(ids) -> Promise<Float32Array logits>, release() -> Promise.
  function workerHandle(w) {
    return {
      ep: 'wasm',
      run: function (ids) { return w.call({ op: 'run', ids: ids }); },
      // Terminating the Worker frees everything it holds at once and, unlike a 'release' round
      // trip, cannot hang on a worker that has stopped answering (the reason it is being freed).
      release: function () { w.terminate(); return Promise.resolve(); }
    };
  }
  function sessionHandle(ort, session, ep) {
    return {
      ep: ep,
      run: function (ids) {
        var data = new BigInt64Array(ids.length);
        for (var i = 0; i < ids.length; i++) data[i] = BigInt(ids[i]);
        var feeds = {};
        feeds[session.inputNames[0] || 'input_ids'] = new ort.Tensor('int64', data, [1, ids.length]);
        return session.run(feeds).then(function (out) { return out[session.outputNames[0] || 'logits'].data; });
      },
      release: function () { return session.release(); }
    };
  }

  // untilDone ("Unlimited" in the pages): how a run with no token limit decides it is finished
  var UNTIL_DONE_MIN_TOKENS = 24;   // never call a passage done before this many tokens
  var UNTIL_DONE_MIN_ROOM = 64;     // context kept free for the output: the prompt keeps block_size - 64
  var UNTIL_DONE_CAP = 256;         // no forward-per-token run grows without bound: past this, 'maxNewTokens'
  var UNTIL_DONE_LOOP = 8;          // the last 8 tokens appearing earlier in the output = a loop
  var SENTENCE_END = /[.!?\u2026]["'\u201d\u2019)\]]*$/;   // a sentence mark, then any closing quotes or brackets
  function repeatsEarlier(ids, n) {
    var len = ids.length;
    if (len < 2 * n) return false;
    outer: for (var i = 0; i + n <= len - n; i++) {
      for (var j = 0; j < n; j++) if (ids[i + j] !== ids[len - n + j]) continue outer;
      return true;
    }
    return false;
  }

  // noMidsentenceBreak looks at the last 64 code units of prompt + output, as harness.py does
  var BREAK_TAIL = 64;
  var WS_CHAR = /\s/;
  function trimEndWS(s) {   // s.replace(/\s+$/, '') without a regex scan of a long string
    var i = s.length;
    while (i > 0 && WS_CHAR.test(s.charAt(i - 1))) i--;
    return i === s.length ? s : s.slice(0, i);
  }

  // The tokens the break rules mask, made once per tokenizer: every id but <|endoftext|> whose bytes
  // hold \n or \r, grouped by the text before the first break (right-trimmed), since that text
  // decides whether the break would follow a finished sentence.
  function breakTokens(tok, eot) {
    if (tok._breaks) return tok._breaks;
    var all = [], byPre = new Map(), dec = new TextDecoder('utf-8', { ignoreBOM: true });
    for (var id = 0; id < tok.size; id++) {
      if (id === eot) continue;
      var b = tok.tokenBytes(id), cut = -1;
      for (var j = 0; j < b.length; j++) if (b[j] === 10 || b[j] === 13) { cut = j; break; }
      if (cut < 0) continue;
      all.push(id);
      var pre = trimEndWS(dec.decode(b.subarray(0, cut)));
      if (!byPre.has(pre)) byPre.set(pre, []);
      byPre.get(pre).push(id);
    }
    var groups = [];
    byPre.forEach(function (ids, pre) { groups.push({ pre: pre, ids: Int32Array.from(ids) }); });
    tok._breaks = { all: Int32Array.from(all), groups: groups };
    return tok._breaks;
  }

  function numOr(v, dflt) {
    if (v == null || v === '') return dflt;
    var n = Number(v);
    return isFinite(n) ? n : dflt;
  }

  // ---------------------------------------------------------------- the engine

  var E = {
    ready: false,
    manifest: null,
    backend: null,
    tokenizer: null,
    stats: null,
    last: null,
    version: '1',
    ortVersion: ORT_VERSION,
    _loadPromise: null,
    _progress: [],   // onProgress listeners of the load attempt in flight
    _queue: Promise.resolve(),
    _current: null,  // AbortController of the generate in flight (unload() pulls it)
    _session: null   // the chosen backend handle (workerHandle / sessionHandle)
  };

  // ---------------------------------------------------------------- phones, tablets, small memory
  //
  // A constrained device is touch-only ((pointer: coarse) and (hover: none): a phone or a tablet),
  // reports 4 GB of memory or less (navigator.deviceMemory, Chromium only), or is an iPad (iPadOS
  // calls itself a Mac with a touch screen, and with a trackpad attached its pointer reads fine).
  // There "auto" builds one wasm session in the Worker and never the WebGPU one: both at once, with
  // the WebGPU build's own 400+ MB heap on the page, peaked at 1.35-1.45 GB on phone profiles against
  // 0.75-0.8 GB for wasm alone, past what iOS lets a tab hold (it reloads the tab instead). Desktop
  // browsers still race both and keep the faster.
  function lowMemory() { return typeof navigator.deviceMemory === 'number' && navigator.deviceMemory <= 4; }
  E.constrained = function () {
    var mm = function (q) { try { return !!(window.matchMedia && window.matchMedia(q).matches); } catch (e) { return false; } };
    return (mm('(pointer: coarse)') && mm('(hover: none)')) || lowMemory() ||
      (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  };

  // BUILDING sits in sessionStorage while sessions are being built and goes when that ends or the
  // page is left (pagehide). A tab that died mid-build (iOS reloads a tab that ran out of memory, and
  // fires no pagehide) finds it on its next arrival: E.interrupted, and LIGHT for the rest of the
  // tab's visit, so a constrained device builds the smaller session from then on (E._load).
  var BUILDING = 'embedding-tax:building', LIGHT = 'embedding-tax:light';
  var building = false;
  function tabStore() { try { return window.sessionStorage || null; } catch (e) { return null; } }
  function writeBuilding(on) {
    var s = tabStore();
    try { if (s) { if (on) s.setItem(BUILDING, '1'); else s.removeItem(BUILDING); } } catch (e) { /* storage off: nothing is remembered */ }
  }
  function markBuilding(on) { building = on; writeBuilding(on); }
  // E.interrupted: THIS arrival found the mark, so the page reloaded mid-build (the note says so,
  // once). E.lightTab: some arrival in this tab did, so every later build keeps the weights
  // compressed; that is a mode, not news, and no note is shown for it.
  E.interrupted = false;
  E.lightTab = (function () {
    var s = tabStore();
    try {
      if (s && s.getItem(BUILDING)) { s.removeItem(BUILDING); s.setItem(LIGHT, '1'); E.interrupted = true; }
      return !!(s && s.getItem(LIGHT));
    } catch (e) { return false; }
  })();
  window.addEventListener('pagehide', function () { if (building) writeBuilding(false); });
  window.addEventListener('pageshow', function (ev) { if (ev.persisted && building) writeBuilding(true); });   // back from the bfcache mid-build
  // The same fallback for a build that failed with OUT_OF_MEMORY inside the page (the Worker's heap
  // could not grow) instead of taking the tab down: its Worker is already gone, and Try again builds
  // the smaller session.
  var lightNext = false;

  function outOfMemory(err) {
    if (err instanceof Error && !err.code && OOM_TEXT.test(err.message)) err.code = 'OUT_OF_MEMORY';
    return err;
  }

  E._emitProgress = function (loaded, total) {
    for (var i = 0; i < E._progress.length; i++) {
      try { E._progress[i](loaded, total); } catch (e) { /* a listener's error must not stop the load */ }
    }
  };

  E.load = function (opts) {
    opts = opts || {};
    var onProgress = typeof opts.onProgress === 'function' ? opts.onProgress : null;
    if (E._loadPromise) {   // an attempt in flight, or done: the first call's backend preference wins
      if (onProgress) {
        // Done: one final report now. In flight: join that attempt's listeners.
        if (E.ready) { if (E.stats && E.stats.bytes) { try { onProgress(E.stats.bytes, E.stats.bytes); } catch (e) { /* ignore */ } } }
        else E._progress.push(onProgress);
      }
      return E._loadPromise;
    }
    // Listeners belong to one attempt: registered here, dropped when it settles, so a page that
    // calls load() many times does not stack callbacks for a later attempt to fire all at once.
    E._progress = onProgress ? [onProgress] : [];
    var p = E._load(opts.backend || 'auto', !!opts.cacheOnly).then(function (r) {
      E._progress = [];
      return r;
    }, function (err) {
      E._progress = [];
      if (E._loadPromise === p) E._loadPromise = null;   // a retry starts over (unload() may have reset it already)
      throw outOfMemory(err);
    });
    E._loadPromise = p;
    return p;
  };

  E._load = async function (pref, cacheOnly) {
    if (pref !== 'auto' && pref !== 'wasm' && pref !== 'webgpu') throw new Error('backend must be auto, wasm or webgpu');
    var constrained = E.constrained();
    var stats = { ortVersion: ORT_VERSION, backendPreference: pref, constrained: constrained, candidates: [] };
    if (pref === 'auto' && constrained) pref = 'wasm';   // one session, never two at once (an explicit backend is still honoured)
    E.stats = stats;
    var t0 = now();

    var manifest = await fetchJSON(MODEL_DIR + 'manifest.json');
    E.manifest = manifest;
    var tokJson = await fetchJSON(MODEL_DIR + 'tokenizer.json');
    E.tokenizer = new ByteLevelBPE(tokJson);
    var eotFromTok = E.tokenizer.vocab.get('<|endoftext|>');
    if (eotFromTok !== undefined && eotFromTok !== manifest.eot_id) {
      throw new Error('manifest eot_id ' + manifest.eot_id + ' disagrees with tokenizer (' + eotFromTok + ')');
    }
    stats.tokenizerMs = now() - t0;
    if (cacheOnly && !(await E.isCached(manifest))) throw E._notCached();

    // Backend candidates. WebGPU only when the browser exposes an adapter.
    var tryGpu = pref !== 'wasm' && typeof navigator !== 'undefined' && !!navigator.gpu;
    var adapter = null;
    if (tryGpu) {
      try { adapter = await navigator.gpu.requestAdapter(); } catch (e) { adapter = null; }
      tryGpu = !!adapter;
    }
    if (pref === 'webgpu' && !tryGpu) throw new Error('WebGPU is not available in this browser');
    stats.webgpuAdapter = tryGpu;

    // onnxruntime-web on the page thread is needed only for a WebGPU session; the wasm
    // candidate brings its own copy into its Worker (WASM_WORKER_SRC).
    var ort = null;
    if (tryGpu) {
      var tScript = now();
      if (!(window.ort && window.ort.InferenceSession)) {
        await loadScript(ORT_CDN + ORT_SCRIPT.webgpu, ORT_INTEGRITY.webgpu);
        if (!(window.ort && window.ort.InferenceSession)) throw new Error('onnxruntime-web did not initialise');
        stats.ortScriptIntegrity = 'sri';   // the tag would have run nothing but the pinned build
      } else {
        stats.ortScriptIntegrity = 'preloaded';   // someone else put ort on the page; not ours to vouch for
      }
      ort = window.ort;
      ort.env.wasm.wasmPaths = ORT_CDN;
      // 'error': the WebGPU build reports, at warning level through console.error, that a few
      // shape nodes run on the CPU. Expected for this graph, and it read as a page error.
      ort.env.logLevel = 'error';
      stats.ortScriptMs = now() - tScript;
    }

    var tDl = now();
    var dl = await E._fetchWeights(manifest, cacheOnly);   // bytes verified against manifest.bytes and .sha256
    var bytes = dl.bytes;
    stats.downloadMs = now() - tDl;
    stats.bytes = bytes.length;
    stats.sha256 = dl.sha256;   // "verified", or "unavailable" (no crypto.subtle: insecure context)

    var sessOpts = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      // Fold the DequantizeLinear nodes once at load (manifest.runtime.hint): ~10x faster
      // session creation and ~3x faster forwards with identical logits.
      extra: { session: { disable_quant_qdq: '1' } }
    };
    if (manifest.runtime && manifest.runtime.session_options) {
      var so = manifest.runtime.session_options;
      for (var k in so) if (Object.prototype.hasOwnProperty.call(so, k) && k.indexOf('session.') === 0) {
        sessOpts.extra.session[k.slice(8)] = String(so[k]);
      }
    }
    // A constrained device keeps the folded weights (unfolded, each forward ran 1.5-4x slower on a
    // CPU, and the default preset runs two per token) but skips prepacking: 71 MB less wasm heap for
    // a forward 5-15% slower, identical tokens. Unfolded (170 MB of heap instead of 354, about half
    // the speed) only where memory is known to be short: a browser reporting 4 GB or less, or a tab
    // that already failed once building the folded one (E.interrupted, lightNext). Desktop runs what
    // the manifest says.
    stats.light = false;
    if (constrained) {
      sessOpts.extra.session.disable_prepacking = '1';
      stats.light = lowMemory() || E.lightTab || lightNext;
      if (stats.light) delete sessOpts.extra.session.disable_quant_qdq;
    }
    stats.sessionOptions = JSON.parse(JSON.stringify(sessOpts.extra.session));

    // One candidate (wasm: a phone, or a browser without WebGPU) means nothing after its session
    // needs the page's copy of the weights, so the Worker gets that very buffer, transferred, not a
    // second copy. Bytes whose SHA-256 matched are the manifest's own file, so they are cached first:
    // a tab that dies building the session (iOS reloads it) reopens from Cache Storage instead of
    // downloading again. With no SHA-256 (an insecure context) the old order stands, below.
    var single = !tryGpu;
    var cachedEarly = false;
    if (single && !dl.fromCache && dl.sha256 === 'verified') { stats.cached = await E._cachePut(dl.key, bytes); cachedEarly = true; }
    var giveAway = single && (dl.fromCache || cachedEarly);
    dl.bytes = null;

    var chosen = null, loser = null;
    var wasmCand = null, gpuCand = null;
    markBuilding(true);
    try {
      if (pref !== 'webgpu') {
        wasmCand = await E._trySession(null, 'wasm', bytes, sessOpts, giveAway);
        if (giveAway) bytes = null;   // detached now: the Worker holds the only copy
        stats.candidates.push(wasmCand.report);
        if (wasmCand.error && pref === 'wasm') throw wasmCand.error;
      }
      if (tryGpu) {
        gpuCand = await E._trySession(ort, 'webgpu', bytes, sessOpts);
        stats.candidates.push(gpuCand.report);
        if (gpuCand.error && pref === 'webgpu') throw gpuCand.error;
      }
      var okW = wasmCand && !wasmCand.error, okG = gpuCand && !gpuCand.error;
      if (okW && okG) {
        if (gpuCand.report.forwardMs < wasmCand.report.forwardMs) { chosen = gpuCand; loser = wasmCand; }
        else { chosen = wasmCand; loser = gpuCand; }
      } else if (okG) chosen = gpuCand;
      else if (okW) chosen = wasmCand;
      else throw (wasmCand && wasmCand.error) || (gpuCand && gpuCand.error) || new Error('no backend could run the model');
      if (loser) { try { await loser.handle.release(); } catch (e) { /* ignore */ } }
    } catch (err) {
      // Cached bytes that no backend could run correctly are not kept for the next visit, unless
      // they hashed right: then the bytes are fine and the failure was the device (a CDN fetch, a
      // lost GPU, memory), and dropping them would only turn the next click into a 50 MB download.
      if (dl.fromCache && dl.sha256 !== 'verified') await E._cacheDelete(dl.key);
      if (constrained && outOfMemory(err).code === 'OUT_OF_MEMORY') lightNext = true;
      throw err;
    } finally {
      markBuilding(false);
    }

    // Unhashed bytes go to Cache Storage only now, after a backend has produced the expected
    // answer from them. (Caching unchecked bytes before this point once persisted a corrupt
    // download of the right length, and every later load() served it back instead of the network.)
    if (!dl.fromCache && !cachedEarly) stats.cached = await E._cachePut(dl.key, bytes);

    E._session = chosen.handle;
    E.backend = chosen.ep;
    stats.backend = chosen.ep;
    stats.wasmWorker = !!(wasmCand && !wasmCand.error);   // wasm compute is in a Worker, never on the page thread
    stats.wasmThreads = stats.wasmWorker ? wasmCand.report.wasmThreads : null;
    stats.selfCheck = chosen.report.selfCheck || null;
    stats.totalMs = now() - t0;
    E.ready = true;
    return { backend: chosen.ep, manifest: manifest };
  };

  // Build one backend, warm it up, time it, and make it prove itself on the manifest's
  // self-check. Resolves { handle, ep, report } or { error, ep, report }; never rejects.
  // For "wasm", createMs includes spawning the Worker and fetching the runtime into it. giveAway:
  // nothing after this session needs `bytes`, so the Worker gets the buffer itself (it is detached
  // here) instead of a copy.
  E._trySession = async function (ort, ep, bytes, baseOpts, giveAway) {
    var report = { backend: ep };
    var handle = null;
    try {
      // logSeverityLevel 3 (errors only): the session's own C++ logger prints, through
      // console.error, that a few shape nodes run on the CPU. Expected for this graph.
      var opts = { executionProviders: [ep], graphOptimizationLevel: baseOpts.graphOptimizationLevel, extra: baseOpts.extra, logSeverityLevel: 3 };
      var t0 = now();
      if (ep === 'wasm') {
        var w = new WasmWorker();
        handle = workerHandle(w);
        report.ortIntegrity = await w.call({ op: 'init', script: ORT_CDN + ORT_SCRIPT.wasm, wasmPaths: ORT_CDN, integrity: ORT_INTEGRITY.wasm });   // 'verified' | 'unavailable'
        // Otherwise the Worker gets its own copy: the page keeps `bytes` for WebGPU and the cache.
        var whole = giveAway && bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength;
        var copy = whole ? bytes : bytes.slice();
        var info = await w.call({ op: 'create', bytes: copy.buffer, opts: opts }, [copy.buffer]);
        report.wasmThreads = info.numThreads;
      } else {
        handle = sessionHandle(ort, await ort.InferenceSession.create(bytes, opts), ep);
      }
      report.createMs = now() - t0;
      // Warm up (shader compile / allocation) then time the forward that decides the choice.
      var ids = new Array(BENCH_T);
      for (var i = 0; i < BENCH_T; i++) ids[i] = (i * 7919 + 13) % E.manifest.vocab_size;
      var logits = await handle.run(ids);
      report.firstForwardMs = now() - t0 - report.createMs;
      var best = Infinity;
      for (var r = 0; r < 2; r++) {
        var t1 = now();
        logits = await handle.run(ids);
        best = Math.min(best, now() - t1);
      }
      report.forwardMs = best;
      report.benchTokens = BENCH_T;
      if (!allFinite(logits) || logits.length !== E.manifest.vocab_size) throw new Error(ep + ': bad logits from warm-up');
      report.argmax = argmax(logits);
      // The manifest's self-check: a fixed context whose argmax the exporter measured on the
      // shipped file. Wrong weights (a stale model.int8.onnx beside a new manifest) or a
      // backend whose kernels return finite nonsense fail here, before anything is cached.
      var sc = E.manifest.parity && E.manifest.parity.self_check;
      if (sc && sc.input_ids && typeof sc.expected_argmax === 'number') {
        var got = argmax(await handle.run(sc.input_ids));
        report.selfCheck = { expected: sc.expected_argmax, got: got, pass: got === sc.expected_argmax };
        if (!report.selfCheck.pass) {
          throw new Error(ep + ': self-check failed: argmax ' + got + ' after the manifest\'s fixed context, expected ' +
            sc.expected_argmax + ' (' + JSON.stringify(sc.expected_token || '') + '); the weights do not match ' +
            'this manifest or this backend computes wrong logits');
        }
      }
      return { handle: handle, ep: ep, report: report };
    } catch (err) {
      // The WebGPU build throws a bare number (an Emscripten exception pointer) when the
      // device cannot be set up; say so instead of printing the pointer.
      var msg = typeof err === 'number'
        ? ep + ': native exception #' + err + (ep === 'webgpu' ? ' (WebGPU device unavailable?)' : '')
        : String(err && err.message || err);
      report.error = msg;
      if (handle) { try { await handle.release(); } catch (e) { /* ignore */ } }
      return { error: err instanceof Error ? err : new Error(msg), ep: ep, report: report };
    }
  };

  E._sha256 = async function (bytes) {
    if (!(window.crypto && window.crypto.subtle && window.crypto.subtle.digest)) return null;   // insecure context
    var d = new Uint8Array(await window.crypto.subtle.digest('SHA-256', bytes));
    var hex = '';
    for (var i = 0; i < d.length; i++) hex += (d[i] < 16 ? '0' : '') + d[i].toString(16);
    return hex;
  };

  E._cacheOpen = async function () {
    try { return window.caches ? await caches.open(CACHE_NAME) : null; } catch (e) { return null; }
  };
  E._cachePut = async function (key, bytes) {
    var cache = await E._cacheOpen();
    if (!cache) return false;
    try {
      await cache.put(key, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
      var keys = await cache.keys(), keep = new Request(key).url;
      for (var i = 0; i < keys.length; i++) if (keys[i].url !== keep) await cache.delete(keys[i]);
      return true;
    } catch (e) { return false; }   // storage full or unavailable: the download still worked
  };
  E._cacheDelete = async function (key) {
    var cache = await E._cacheOpen();
    if (cache) { try { await cache.delete(key); } catch (e) { /* ignore */ } }
  };

  E._weightsKey = function (manifest) {
    return MODEL_DIR + manifest.file + '?sha256=' + (manifest.sha256 || manifest.exported_at || '');
  };
  E._notCached = function () {
    var err = new Error('the model weights are not saved on this device yet');
    err.code = 'NOT_CACHED';
    return err;
  };
  E.isCached = async function (manifest) {
    try {
      manifest = manifest || E.manifest || await fetchJSON(MODEL_DIR + 'manifest.json');
      var cache = await E._cacheOpen();
      return !!(cache && await cache.match(E._weightsKey(manifest)));
    } catch (e) { return false; }
  };

  // Resolves { bytes, fromCache, key, sha256 } with bytes of manifest.bytes length whose SHA-256
  // is manifest.sha256 (when the browser can hash: sha256 is "verified", else "unavailable").
  // Nothing is written to the cache here; _load does that (see its `single` comment for when).
  E._fetchWeights = async function (manifest, cacheOnly) {
    var url = MODEL_DIR + manifest.file;
    var key = E._weightsKey(manifest);
    var total = manifest.bytes || 0;
    var verdict = async function (bytes) {   // "verified" | "unavailable" | the digest that did not match
      if (!manifest.sha256) return 'unavailable';
      var got = await E._sha256(bytes);
      return got === null ? 'unavailable' : (got === manifest.sha256 ? 'verified' : got);
    };
    var cache = await E._cacheOpen();
    if (cache) {
      try {
        var hit = await cache.match(key);
        if (hit) {
          var buf = new Uint8Array(await hit.arrayBuffer());
          var v = (!total || buf.length === total) ? await verdict(buf) : 'wrong length';
          if (v === 'verified' || v === 'unavailable') {
            E.stats.fromCache = true;
            E._emitProgress(buf.length, buf.length);
            return { bytes: buf, fromCache: true, key: key, sha256: v };
          }
          await cache.delete(key);   // a corrupt entry: forget it and download again
        }
      } catch (e) { /* fall through to the network */ }
    }
    if (cacheOnly) throw E._notCached();   // reopening from this device never turns into a download
    E.stats.fromCache = false;
    // DOWNLOAD_STALL_MS of visible time without a byte (headers included) cancels the request, and
    // the load fails with a message the page shows beside Try again.
    var ctl = new AbortController(), stalled = false, quiet = 0;
    var stopClock = visibleClock(function (dt) {
      quiet += dt;
      if (quiet >= DOWNLOAD_STALL_MS && !stalled) { stalled = true; ctl.abort(); }
    });
    var out, loaded = 0;
    try {
      var res = await fetch(url, { signal: ctl.signal }).catch(function (e) {
        throw new Error('download of ' + url + ' failed: ' + (e && e.message || e));
      });
      quiet = 0;
      if (!res.ok) throw new Error('fetch ' + url + ': HTTP ' + res.status);
      if (!total) total = Number(res.headers.get('Content-Length')) || 0;
      var reader = res.body.getReader();
      // Straight into one buffer of the manifest's size: a list of chunks joined at the end held the
      // whole file twice at its peak. Without manifest.bytes the chunks are joined as before.
      var chunks = [];
      out = manifest.bytes ? new Uint8Array(manifest.bytes) : null;
      E._emitProgress(0, total);
      for (;;) {
        var step = await reader.read();
        if (step.done) break;
        quiet = 0;
        if (out) {
          if (loaded + step.value.length > out.length) {
            reader.cancel().catch(noop);
            throw new Error('weights: the server sent more than the ' + out.length + ' bytes the manifest says');
          }
          out.set(step.value, loaded);
        } else chunks.push(step.value);
        loaded += step.value.length;
        E._emitProgress(Math.min(loaded, total || loaded), total || loaded);
      }
    } catch (e) {
      if (stalled) throw new Error('the download of ' + manifest.file + ' stalled: no data arrived for ' + DOWNLOAD_STALL_MS / 1000 + ' s. Check the connection and try again');
      throw e;
    } finally {
      stopClock();
    }
    if (!out) {
      out = new Uint8Array(loaded);
      for (var i = 0, off = 0; i < chunks.length; i++) { out.set(chunks[i], off); off += chunks[i].length; }
    }
    if (total && loaded !== total) throw new Error('weights: got ' + loaded + ' bytes, manifest says ' + total);
    E._emitProgress(loaded, loaded);
    var sha = await verdict(out);   // ~35 ms for 50 MB
    if (sha !== 'verified' && sha !== 'unavailable') {
      throw new Error('weights checksum mismatch: ' + manifest.file + ' hashes to ' + sha.slice(0, 12) + '\u2026, the manifest says ' +
        manifest.sha256.slice(0, 12) + '\u2026; the server is serving a different model file than this manifest describes ' +
        '(a torn deploy, or an HTTP cache still holding the previous export) \u2014 try again in a few minutes');
    }
    return { bytes: out, fromCache: false, key: key, sha256: sha };
  };

  // s is a backend handle; kept as the one entry point for a forward pass. Forwards run one at a
  // time: a Stop answered while a forward was in flight (orStop) leaves that forward running, and
  // the next one starts only after it settles, so a session never gets two runs at once.
  var fwdTail = Promise.resolve();
  E._forward = function (s, ids) {
    var p = fwdTail.then(function () { return s.run(ids); });
    fwdTail = p.then(noop, noop);
    return p;
  };
  // A forward that gives way to Stop: resolves STOPPED the moment the run is stopped, even while the
  // forward is still in flight (a long context on a slow phone, or a Worker that stopped answering:
  // Stop used to wait on it, forever in that case). The forward's own result or error is dropped.
  var STOPPED = {};
  function orStop(p, stopped) {
    p.catch(noop);   // dropped after a stop: never an unhandled rejection
    return Promise.race([p, stopped]);
  }

  // Forget the loaded backend (ready goes false, the next load() starts over) and hand it back.
  E._detach = function () {
    var s = E._session;
    E._session = null; E.ready = false; E.backend = null; E._loadPromise = null;
    return s;
  };
  // Release a handle without letting its complaints through: a dead backend may refuse, and it
  // is gone either way.
  E._release = function (s) {
    return Promise.resolve().then(function () { return s.release(); }).catch(function () { /* see above */ });
  };

  // A forward that rejects (the WebGPU device lost while the tab sat in the background, a crashed
  // worker, any onnxruntime error) or hands back non-finite logits leaves the backend in a state
  // nothing can vouch for, so it is released and the engine goes back to "not loaded": the next
  // load() rebuilds it (the weights stay in Cache Storage, so that is a couple of seconds)
  // instead of every later generate() dying on the same dead session until a hard reload.
  E._runtimeFailure = function (s, err) {
    var msg = String(err && err.message || err);
    if (E._session === s) E._release(E._detach());
    return outOfMemory(new Error('EmbeddingTax: the ' + s.ep + ' backend failed mid-generation (' + msg + '); the model has been ' +
      'unloaded, call load() again to rebuild it (the weights stay cached)'));
  };

  // Free what the backend holds (the Worker's whole wasm heap: see the header). The generate in
  // flight stops at once and resolves with its text so far; ones still queued reject with "call
  // load() first". The tokenizer stays, so tokenCount() keeps working; load() rebuilds the rest
  // from the cache.
  E.unload = async function () {
    if (E._loadPromise) { try { await E._loadPromise; } catch (e) { /* a failed load left nothing to free */ } }
    var s = E._detach();
    if (E._current) E._current.abort();
    await E._queue;   // never rejects: it is the settled chain of queued generates
    // A WebGPU session is not released under a forward still running (a stopped run's, see orStop);
    // the Worker is simply terminated, which is safe mid-forward and cannot hang on a dead Worker.
    if (s && s.ep !== 'wasm') await fwdTail;
    if (s) await E._release(s);
  };

  E.generate = function (prompt, opts) {
    var run = function () { return E._generate(prompt, opts || {}); };
    var p = E._queue.then(run, run);
    E._queue = p.then(function () {}, function () {});
    return p;
  };

  E._generate = async function (prompt, o) {
    if (!E.ready) throw new Error('EmbeddingTax: call load() first');
    var manifest = E.manifest, block = manifest.block_size, eot = manifest.eot_id, tok = E.tokenizer;
    var untilDone = !!o.untilDone;
    var maxNew = o.maxNewTokens == null ? 60 : Math.floor(Number(o.maxNewTokens));
    if (!(maxNew >= 1)) maxNew = 1;
    if (maxNew > block - 1) maxNew = block - 1;
    if (untilDone) maxNew = UNTIL_DONE_MIN_ROOM;   // the prompt budget below; the real cap is set once the prompt is encoded
    var temperature = o.temperature == null ? 0.8 : Number(o.temperature);
    var topK = o.topK == null ? 40 : Math.floor(Number(o.topK));
    var seed = o.seed == null ? 1337 : (Number(o.seed) | 0);
    var onToken = typeof o.onToken === 'function' ? o.onToken : null;
    var signal = o.signal || null;
    // decoding constraints (header, steps 1-6); every default is "off"
    var repetitionPenalty = numOr(o.repetitionPenalty, 1);
    if (!(repetitionPenalty > 0)) repetitionPenalty = 1;
    var d = {
      untilDone: untilDone, maxNew: 0, temperature: temperature, topK: topK,
      topP: numOr(o.topP, 1),
      minP: Math.min(1, Math.max(0, numOr(o.minP, 0))),
      repetitionPenalty: repetitionPenalty,
      repetitionWindow: Math.max(0, Math.floor(numOr(o.repetitionWindow, 0))),
      frequencyPenalty: numOr(o.frequencyPenalty, 0),
      presencePenalty: numOr(o.presencePenalty, 0),
      noRepeatNgram: Math.max(0, Math.floor(numOr(o.noRepeatNgram, 0))),
      noLeadingBreak: Math.max(0, Math.floor(numOr(o.noLeadingBreak, 0))),
      noMidsentenceBreak: !!o.noMidsentenceBreak,
      cadAlpha: numOr(o.cadAlpha, 0),
      bestOf: Math.max(1, Math.floor(numOr(o.bestOf, 1))),
      bestOfFluencyWeight: numOr(o.bestOfFluencyWeight, 0)
    };

    var input = prompt == null ? '' : String(prompt);
    var maxPrompt = Math.max(1, block - maxNew);
    // A token covers at most maxTokenBytes bytes and a UTF-16 code unit is at least one byte, so
    // the last maxPrompt * maxTokenBytes code units already hold >= maxPrompt tokens: whatever
    // comes before them would fall to the left truncation below anyway, so it is not encoded (a
    // megabyte pasted as the prompt no longer stalls the page tokenizing text the model cannot
    // see). promptTokens then counts only the part that was encoded.
    var charLimit = maxPrompt * tok.maxTokenBytes, promptChars = input.length, droppedChars = 0;
    if (input.length > charLimit) {
      droppedChars = input.length - charLimit;
      var cc = input.charCodeAt(droppedChars);
      if (cc >= 0xDC00 && cc <= 0xDFFF) droppedChars++;   // never start on the low half of a surrogate pair
      input = input.slice(droppedChars);
    }
    var ids = tok.encode(input);
    var promptTokens = ids.length;
    var dropped = 0;
    if (ids.length > maxPrompt) { dropped = ids.length - maxPrompt; ids = ids.slice(dropped); }
    if (ids.length === 0) ids = [eot];
    if (untilDone) maxNew = Math.min(block - ids.length, UNTIL_DONE_CAP);   // never slide past the context, never past the cap
    d.maxNew = maxNew;
    var last = {
      truncated: dropped > 0 || droppedChars > 0, promptTokens: promptTokens, keptTokens: ids.length, droppedTokens: dropped,
      maxPromptTokens: maxPrompt, promptChars: promptChars, keptChars: input.length, droppedChars: droppedChars,
      newTokens: 0, stoppedBy: null, ms: 0, msToFirstToken: 0, tokensPerSecond: 0, forwards: 0,
      backend: E.backend, seed: seed, temperature: temperature, topK: topK, maxNewTokens: maxNew, untilDone: untilDone,
      topP: d.topP, minP: d.minP, repetitionPenalty: d.repetitionPenalty, repetitionWindow: d.repetitionWindow,
      frequencyPenalty: d.frequencyPenalty, presencePenalty: d.presencePenalty, noRepeatNgram: d.noRepeatNgram,
      noLeadingBreak: d.noLeadingBreak, noMidsentenceBreak: d.noMidsentenceBreak, cadAlpha: d.cadAlpha,
      bestOf: d.bestOf, bestOfFluencyWeight: d.bestOfFluencyWeight,
      chosenDraft: 0, chosenSeed: seed, meanPmi: null, meanLogp: null, drafts: [], ids: [], text: ''
    };
    E.last = last;
    if (last.truncated && typeof o.onTruncate === 'function') {
      o.onTruncate({ promptTokens: promptTokens, keptTokens: ids.length, droppedTokens: dropped, maxPromptTokens: maxPrompt,
        promptChars: promptChars, keptChars: input.length, droppedChars: droppedChars });
    }

    var s = E._session;
    var internal = new AbortController();   // unload() stops this generate through it; the caller's signal stays theirs
    E._current = internal;
    var aborted = function () { return internal.signal.aborted || !!(signal && signal.aborted); };
    var onStop = null;
    var stopped = new Promise(function (res) { onStop = function () { res(STOPPED); }; });   // settles with STOPPED on either abort
    internal.signal.addEventListener('abort', onStop);
    if (signal) { if (signal.aborted) onStop(); else signal.addEventListener('abort', onStop); }
    var t0 = now();
    var drafts = [];
    try {
      for (var i = 0; i < d.bestOf; i++) {
        // one draft streams live; best-of keeps each draft's pieces and replays the winner
        var pieces = d.bestOf > 1 ? [] : null;
        var emit = pieces ? function (p, id) { pieces.push(p, id); } : onToken;
        var r = await E._draft(s, ids, input, (seed + Math.imul(i, 0x9E3779B9)) | 0, d, emit, pieces ? null : last, aborted, stopped);
        r.draft = i;
        r.pieces = pieces;
        r.score = r.meanPmi === null ? null : r.meanPmi + d.bestOfFluencyWeight * r.meanLogp;
        drafts.push(r);
        if (r.stoppedBy === 'abort') break;
      }
    } finally {
      if (E._current === internal) E._current = null;
      if (signal) signal.removeEventListener('abort', onStop);
    }
    // the highest score wins; ties go to the lower draft; a null score (no token) never beats a number
    var best = drafts[0], bestScore = best.score === null ? -Infinity : best.score, forwards = 0, allTokens = 0, j;
    for (j = 0; j < drafts.length; j++) {
      forwards += drafts[j].forwards;
      allTokens += drafts[j].newTokens;
      var sc = drafts[j].score === null ? -Infinity : drafts[j].score;
      if (sc > bestScore) { best = drafts[j]; bestScore = sc; }
    }
    var tFirst = best.tFirst;
    if (best.pieces && onToken) {
      for (j = 0; j < best.pieces.length; j += 2) {
        if (!tFirst) tFirst = now();
        onToken(best.pieces[j], best.pieces[j + 1]);
      }
    }
    last.drafts = drafts.map(function (x) {
      return { draft: x.draft, seed: x.seed, score: x.score, meanPmi: x.meanPmi, meanLogp: x.meanLogp,
        newTokens: x.newTokens, stoppedBy: x.stoppedBy, text: x.text };
    });
    last.chosenDraft = best.draft;
    last.chosenSeed = best.seed;
    last.meanPmi = best.meanPmi;
    last.meanLogp = best.meanLogp;
    last.stoppedBy = drafts[drafts.length - 1].stoppedBy === 'abort' ? 'abort' : best.stoppedBy;
    last.newTokens = best.newTokens;
    last.forwards = forwards;
    last.ids = best.ids;
    last.text = best.text;
    last.ms = now() - t0;
    last.msToFirstToken = tFirst ? tFirst - t0 : 0;
    last.tokensPerSecond = last.ms > 0 ? allTokens / (last.ms / 1000) : 0;
    return best.text;
  };

  // Vocab-sized work arrays, made once per vocabulary size and reused by every step (generations
  // are queued, so one run at a time uses them): the processed logits, the repetition penalty's
  // seen-stamps, the output's token counts, and sample()'s topP/minP scratch.
  E._buffers = function (V) {
    var b = E._buf;
    if (!b || b.V !== V) {
      b = makeScratch(V);
      b.V = V; b.l = new Float64Array(V); b.stamp = new Uint32Array(V); b.stampId = 0; b.counts = new Int32Array(V);
      E._buf = b;
    }
    return b;
  };

  // One draft: the decoding loop of header steps 1-7 from the encoded prompt `ids` (input is the
  // prompt text the break rule reads). emit(piece, tokenId) gets the text as it is decoded; `live`
  // (E.last, or null) is kept current with newTokens and text; `stopped` settles with STOPPED when
  // the run is stopped (orStop). Resolves { ids, text, stoppedBy, newTokens, forwards, meanPmi,
  // meanLogp, tFirst, seed }.
  E._draft = async function (s, ids, input, seed, d, emit, live, aborted, stopped) {
    var manifest = E.manifest, block = manifest.block_size, eot = manifest.eot_id, tok = E.tokenizer, V = manifest.vocab_size;
    var buf = E._buffers(V), l = buf.l, stamp = buf.stamp, counts = buf.counts;
    var rng = mulberry32(seed);
    var dec = new TextDecoder('utf-8', { ignoreBOM: true });
    var text = '', ctx = ids.slice(), generated = [], promptFree = [eot], seen = [], lastId = -1;
    var a = d.cadAlpha, a1 = 1 + a;
    var needFree = a !== 0 || d.bestOf > 1;   // the prompt-free forward: CAD, and the best-of score
    var penalise = d.frequencyPenalty !== 0 || d.presencePenalty !== 0;
    var breaks = d.noLeadingBreak > 0 || d.noMidsentenceBreak ? breakTokens(tok, eot) : null;
    var inTail = input.slice(-BREAK_TAIL), inTrimTail = trimEndWS(input).slice(-BREAK_TAIL);
    var r = { ids: generated, text: '', stoppedBy: null, newTokens: 0, forwards: 0, meanPmi: null, meanLogp: null, tFirst: 0, seed: seed };
    var sumPmi = 0, sumLogp = 0, i, t, v;
    try {
      for (var step = 0; step < d.maxNew; step++) {
        if (aborted()) { r.stoppedBy = 'abort'; break; }
        var lc, lu = null;
        try {
          lc = await orStop(E._forward(s, ctx.length > block ? ctx.slice(ctx.length - block) : ctx), stopped);
          if (lc === STOPPED) { r.stoppedBy = 'abort'; break; }
          r.forwards++;
          if (needFree && !aborted()) {   // a Stop pressed during the first pass does not wait for the second
            lu = await orStop(E._forward(s, promptFree.length > block ? promptFree.slice(promptFree.length - block) : promptFree), stopped);
            if (lu === STOPPED) { r.stoppedBy = 'abort'; break; }
            r.forwards++;
          }
        } catch (err) { throw E._runtimeFailure(s, err); }
        if (aborted()) { r.stoppedBy = 'abort'; break; }
        // NaN logits would make argmax pick 0 = <|endoftext|>, a silent stop: a runtime failure too.
        if (lc.length !== V || !allFinite(lc) || (lu && (lu.length !== V || !allFinite(lu)))) {
          throw E._runtimeFailure(s, new Error('non-finite logits'));
        }
        // 2. context-aware decoding
        if (a !== 0) { for (i = 0; i < V; i++) l[i] = a1 * lc[i] - a * lu[i]; }
        else { for (i = 0; i < V; i++) l[i] = lc[i]; }
        // 3. repetition penalty over the window of prompt + output, each distinct token once
        if (d.repetitionPenalty !== 1) {
          var rp = d.repetitionPenalty, sid = ++buf.stampId;
          for (i = d.repetitionWindow > 0 ? Math.max(0, ctx.length - d.repetitionWindow) : 0; i < ctx.length; i++) {
            t = ctx[i];
            if (t === eot || stamp[t] === sid) continue;
            stamp[t] = sid;
            v = l[t];
            l[t] = v > 0 ? v / rp : v * rp;
          }
        }
        // 4. frequency / presence over the output
        if (penalise) {
          for (i = 0; i < seen.length; i++) { t = seen[i]; l[t] -= d.frequencyPenalty * counts[t] + d.presencePenalty; }
        }
        // 5. hard masks
        var n = d.noRepeatNgram, len = generated.length;
        if (n >= 1 && len >= n) {
          for (i = 0; i <= len - n; i++) {
            var k = 0;
            while (k < n - 1 && generated[i + k] === generated[len - n + 1 + k]) k++;
            if (k === n - 1) l[generated[i + n - 1]] = -Infinity;
          }
        }
        if (breaks) {
          if (len < d.noLeadingBreak) {
            for (i = 0; i < breaks.all.length; i++) l[breaks.all[i]] = -Infinity;
            l[eot] = -Infinity;
          } else if (d.noMidsentenceBreak) {
            // the tails of (prompt + text) and of its right-trimmed form, built without joining the whole prompt
            var tt = trimEndWS(text);
            var trimmedTail = tt.length >= BREAK_TAIL ? tt.slice(-BREAK_TAIL) : (tt.length ? (inTail + tt).slice(-BREAK_TAIL) : inTrimTail);
            var rawTail = text.length >= BREAK_TAIL ? text.slice(-BREAK_TAIL) : (inTail + text).slice(-BREAK_TAIL);
            for (var g = 0; g < breaks.groups.length; g++) {
              var grp = breaks.groups[g];
              if (!SENTENCE_END.test(grp.pre ? rawTail + grp.pre : trimmedTail)) {
                for (i = 0; i < grp.ids.length; i++) l[grp.ids[i]] = -Infinity;
              }
            }
          }
        }
        // 6. the draw
        var next = sample(l, d.temperature, d.topK, rng, d.topP, d.minP, buf);
        if (next === eot) { r.stoppedBy = 'eot'; break; }
        if (needFree) {
          var lpc = logSoftmaxAt(lc, next);
          sumPmi += lpc - logSoftmaxAt(lu, next);
          sumLogp += lpc;
        }
        generated.push(next);
        ctx.push(next);
        if (needFree) promptFree.push(next);
        if (penalise && counts[next]++ === 0) seen.push(next);
        lastId = next;
        if (!r.tFirst) r.tFirst = now();
        // 7. text and the untilDone stops
        var piece = dec.decode(tok.tokenBytes(next), { stream: true });
        var stopNow = null;
        if (d.untilDone) {
          // the passage is over at a line break that follows a finished sentence
          var nl = piece.indexOf('\n');
          if (nl !== -1 && generated.length >= UNTIL_DONE_MIN_TOKENS && SENTENCE_END.test((text + piece.slice(0, nl)).replace(/\s+$/, ''))) {
            piece = piece.slice(0, nl);
            stopNow = 'passage';
          } else if (repeatsEarlier(generated, UNTIL_DONE_LOOP)) {
            stopNow = 'repeat';
          }
        }
        text += piece;
        if (live) { live.newTokens = generated.length; live.text = text; }
        if (emit && piece) emit(piece, next);
        if (stopNow) { r.stoppedBy = stopNow; break; }
        await yieldToUI();
      }
    } finally {
      for (i = 0; i < seen.length; i++) counts[seen[i]] = 0;
    }
    if (!r.stoppedBy) r.stoppedBy = d.untilDone && d.maxNew < UNTIL_DONE_CAP ? 'context' : 'maxNewTokens';
    var tail = dec.decode();   // an unfinished multi-byte character at the very end -> U+FFFD
    if (tail) { text += tail; if (emit) emit(tail, lastId); }
    r.text = text;
    r.newTokens = generated.length;
    if (needFree && generated.length) { r.meanPmi = sumPmi / generated.length; r.meanLogp = sumLogp / generated.length; }
    return r;
  };

  E.tokenCount = function (text) {
    if (!E.tokenizer) throw new Error('EmbeddingTax: tokenizer not loaded yet (call load())');
    return E.tokenizer.encode(text == null ? '' : String(text)).length;
  };
  E.encode = function (text) {
    if (!E.tokenizer) throw new Error('EmbeddingTax: tokenizer not loaded yet (call load())');
    return E.tokenizer.encode(text == null ? '' : String(text));
  };
  E.decode = function (ids, opts) {
    if (!E.tokenizer) throw new Error('EmbeddingTax: tokenizer not loaded yet (call load())');
    return E.tokenizer.decode(ids, opts);
  };
  E.modelUrl = MODEL_DIR;
  E.ortIntegrity = ORT_INTEGRITY;
  E._ByteLevelBPE = ByteLevelBPE;   // for test_tokenizer_parity.py: Python tokenizers vs this encoder over a fixture of hundreds of strings
  E._sample = sample;
  E._mulberry32 = mulberry32;

  window.EmbeddingTax = E;
})();
