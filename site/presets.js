/* presets.js: the decoding presets, one table for every panel that runs the model (playground.js on
   the home composer and the explorer, script.js on the story page). Load it before either script.

   From blind tests of the decode harness (two rounds, 20 items per config). Balanced, Focused and
   Creative share the guards (GUARDS, plus a min-p that follows temperature, minPFor); Published is
   tools/sample.py's temperature 0.8 / top-k 40 / 60 tokens with every guard off (NO_GUARDS), and
   the panels lock its sliders. RANGES are the ranges tested with the guards on, [min, max, step].
   The panels spell every option out to EmbeddingTax.generate: the preset's sampler, then GUARDS or
   NO_GUARDS, then minP from minPFor when the guards are on. */
(function () {
  'use strict';

  var PRESETS = [
    { id: 'balanced', name: 'Balanced', guards: true, temperature: 0.7, topK: 40, maxNewTokens: 90, untilDone: false,
      desc: 'Scored best in blind tests: stays on your prompt and rarely repeats. About half the speed of Published, and at 90 tokens it usually stops mid-sentence.' },
    { id: 'focused', name: 'Focused', guards: true, temperature: 0.7, topK: 40, maxNewTokens: 90, untilDone: true,
      desc: 'Uses Balanced\'s sampler but stops at the end of the first finished passage (about 55 tokens). Shorter, faster, and with less room to drift off topic.' },
    { id: 'creative', name: 'Creative', guards: true, temperature: 1.2, topK: 200, maxNewTokens: 90, untilDone: false,
      desc: 'Samples from a wider set of words at temperature 1.2. Wording is less predictable and repeats less, but it drifts off topic more often than Balanced.' },
    { id: 'published', name: 'Published', guards: false, temperature: 0.8, topK: 40, maxNewTokens: 60, untilDone: false,
      desc: 'The settings used for the published samples: temperature 0.8, top-k 40, 60 tokens, no other rules. It loops and repeats the most of the four and is here for comparison.' }
  ];
  var CUSTOM_DESC = {
    guarded: 'Your own temperature, top-k or length, with the same guards as Balanced. Only tested ranges are offered.',
    unguarded: 'Published\'s sampler, run until the passage ends (at most 256 tokens). None of the guards are on.'
  };
  var RANGES = { temperature: [0.6, 1.4, 0.05], topK: [40, 200, 1], maxNewTokens: [8, 90, 1] };
  // Unlimited (untilDone) still ends at this many new tokens: infer.js UNTIL_DONE_CAP. The pages say so, never "no limit".
  var UNTIL_DONE_CAP = 256;
  // on for every preset but Published; min-p follows temperature (minPFor)
  var GUARDS = { topP: 1, repetitionPenalty: 1.2, repetitionWindow: 64, frequencyPenalty: 0, presencePenalty: 0, noRepeatNgram: 0,
                 noLeadingBreak: 3, noMidsentenceBreak: true, cadAlpha: 1, bestOf: 1, bestOfFluencyWeight: 0 };
  var NO_GUARDS = { topP: 1, minP: 0, repetitionPenalty: 1, repetitionWindow: 0, frequencyPenalty: 0, presencePenalty: 0, noRepeatNgram: 0,
                    noLeadingBreak: 0, noMidsentenceBreak: false, cadAlpha: 0, bestOf: 1, bestOfFluencyWeight: 0 };
  // min-p applies after temperature, so this keeps the cut near 0.10 x the top token's probability at every temperature
  function minPFor(t) { return Math.round(Math.max(0.10, Math.pow(0.10, 1 / t)) * 100) / 100; }

  window.EmbeddingTaxPresets = {
    PRESETS: PRESETS, GUARDS: GUARDS, NO_GUARDS: NO_GUARDS, CUSTOM_DESC: CUSTOM_DESC,
    // Focused is the default: staying on the prompt is the priority, and stopping at the end of the first
    // passage raised coherence in both blind rounds (Balanced drifts to a new subject after its first break)
    DEFAULT_PRESET: 'focused', DEFAULT_SEED: 1337,
    RANGES: RANGES, UNTIL_DONE_CAP: UNTIL_DONE_CAP, minPFor: minPFor
  };
})();
