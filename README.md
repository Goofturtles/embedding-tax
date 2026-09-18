# The Embedding Tax

A 49,296,896-parameter GPT trained from scratch on one RTX 4090, built for GIBC V2 Track 01 (Foundational LLM Development). The track caps entries at 50,000,000 trainable parameters, and the cap counts the token embeddings and the output head. This project is about that rule.

**Try it:** https://goofturtles.github.io/embedding-tax/ runs the trained model in your browser (WebGPU, with a WebAssembly fallback). Nothing you type leaves the page.

By Arjun Sharma.

## The idea

At a width of 512, the embedding table costs `vocab × 512` parameters. With GPT-2's 50,257-token vocabulary that one table is 25.7M parameters: 51.5% of the cap spent before a single transformer layer exists, leaving room for 7 layers. A 16,384-token vocabulary costs 8.4M (16.8%) and leaves room for **13 layers**. Everything else in the model is held equal; the vocabulary is the decision.

`python tools/param_count.py` prints the count and the config, and when torch is installed it also builds the real model and asserts the live count matches the arithmetic.

## Model

| | |
|---|---|
| Parameters | **49,296,896** (cap 50,000,000) |
| Token embedding | 8,388,608 (16,384 × 512), tied to the output head |
| Transformer layers | 40,907,776 (13 × 3,146,752) |
| Final norm | 512 |
| Layers / heads / width | 13 / 8 / 512 |
| Context | 1,024 tokens |
| MLP ratio | 4 |
| Positions | rotary (RoPE), no learned position table |
| Norm | RMSNorm, weight only; no biases anywhere |
| Tokenizer | byte-level BPE, 16,384 tokens, trained on the corpus by `tools/prepare_data.py` |

Config: [`configs/embedding_tax.json`](configs/embedding_tax.json). Code: [`model/`](model/).

## Training

| | |
|---|---|
| Data | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` subset, streamed |
| Data licence | ODC-By 1.0 (use is also subject to the Common Crawl terms of use) |
| Training tokens | 1,899,888,640 (14,495 steps × 131,072 tokens) |
| Held-out tokens | 100,000,000 (the last full shard of the stream; the model never trains on it) |
| Hardware | 1 × NVIDIA GeForce RTX 4090 (24 GB), bf16, PyTorch 2.6 |
| Wall clock | 7.30 h, of which 2.05 h was deliberate idle (cooling pauses and a 65% GPU duty cap), so about 5.2 h of compute |
| Throughput | 105,549 tokens/s (from the training log) |
| Approximate compute | ≈ 5.6 × 10¹⁷ FLOPs (6 · N · D with N = 49.3M, D = 1.9B) |
| Final validation loss | 3.1039 (per-token perplexity 22.29; per-token numbers only compare within this tokenizer) |

The run's own records are in [`runs/r1/`](runs/r1/): `log.jsonl` (every step and validation point), `eval.json` (the harness output), `samples.json` and `meta.json` (the corpus snapshot). The checkpoints are not in the repository because of their size (197 MB model-only, 592 MB with optimizer state).

## Results

Scored with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 0.4.9 on the final checkpoint (step 14,495). The model is registered with the harness as a `TemplateLM` subclass in `tools/eval_harness.py`, so the task code and scoring are the harness's own.

| Task | Metric | This model | GPT-2 Small (124M), reference | Chance |
|---|---|---:|---:|---:|
| ARC-Easy | acc_norm | **40.32** | 39.73 | 25.0 |
| PIQA | acc | **60.28** | 62.08 | 50.0 |
| HellaSwag | acc_norm | **28.71** | 31.38 | 25.0 |
| WinoGrande | acc | **50.99** | 50.67 | 50.0 |
| WikiText (harness `wikitext` task, the WikiText-2/-103 shared test set) | word perplexity | **65.11** | | |
| | bits per byte | **1.1267** | | |

Chance is listed beside every score because at this size HellaSwag and WinoGrande are expected to sit near it. Word perplexity and bits per byte are normalised per word and per byte, so unlike per-token perplexity they compare across tokenizers.

Every number on the site is read from [`site/results.json`](site/results.json), which `tools/publish_results.py` writes from `runs/r1/log.jsonl` and `runs/r1/eval.json`. Nothing on the site is typed in by hand.

## Running the model in the browser

`site/` is a static site with no build step. `site/infer.js` loads [`site/model/model.int8.onnx`](site/model/) with onnxruntime-web 1.29.0 (from jsDelivr, pinned with subresource integrity) and runs it on WebGPU or, failing that, WebAssembly in a worker.

- **Weights:** weight-only int8, 50,341,006 bytes, SHA-256 `f35926384a58154983651823bd0a663b2d6656c026f242c25dbf6e626b14a49a`. The browser checks the hash before using them and keeps them in Cache Storage, so a second visit opens the model without downloading it again.
- **Parity with PyTorch:** the fp32 export matches PyTorch greedy decoding on 72 of 72 tokens; the int8 file agrees on the top token for 70 of 72. Every load also runs a fixed self-check prompt and refuses a backend that gets it wrong.
- **Tokenizer:** a byte-level BPE implemented in `infer.js` from `site/model/tokenizer.json`.

To run it locally, serve the folder with any static server:

```bash
python -m http.server 3522 --directory site
```

then open http://localhost:3522/.

### Decoding presets

The weights are a base model: they continue whatever you type. The playground offers four decoding presets, chosen by two rounds of blind-judged comparisons over 20+ settings:

- **Focused** (default): temperature 0.7, top-k 40, plus decoding-time guards (context-aware decoding with α = 1.0, a repetition penalty of 1.2 over the last 64 tokens, min-p tied to temperature, no line break before the text starts or in the middle of a sentence). It stops at the end of the first finished passage, at most 256 new tokens.
- **Balanced:** the same guards, up to 90 tokens.
- **Creative:** temperature 1.2, top-k 200, the same guards.
- **Published:** temperature 0.8, top-k 40, 60 tokens, no guards. These are `tools/sample.py`'s settings, which wrote the published samples.

The guards only change which token can come next; the model's output is never edited. The benchmark scores above do not use them.

## Reproducing

```bash
pip install -r requirements.txt

python tools/prepare_data.py --tokens 2000000000   # stream FineWeb-Edu, fit the tokenizer, write 19 train + 1 val shards
python tools/train.py --run r1                      # add --resume to continue after a stop
python tools/eval_harness.py --run r1               # lm-evaluation-harness -> runs/r1/eval.json
python tools/sample.py --run r1                     # the four published samples -> runs/r1/samples.json
python tools/publish_results.py --run r1            # -> site/results.json
python tools/export_web.py --run r1                 # ONNX export, parity check, int8 -> site/model/
```

`tools/smoke.py` runs a few real training steps on random tokens and reports throughput and peak VRAM. `requirements.txt` is the exact environment the run used (a `pip freeze`, CUDA 12.4 wheels).

## Repository layout

| Path | What it holds |
|---|---|
| `model/` | the GPT (`gpt.py`) and its config and analytic parameter count (`config.py`) |
| `configs/` | the model config |
| `tools/` | data preparation, training, evaluation, sampling, publishing and the browser export |
| `runs/r1/` | the training log, evaluation output, samples and corpus snapshot of the submitted run |
| `site/` | the website: home (`index.html`), explorer and results (`app.html`), the story (`story.html`), and the in-browser runtime (`infer.js`, `playground.js`, `presets.js`) |
| `film/` | the scripts that render the project film frame by frame (`render.mjs`, `encode.mjs`) and check every frame against the parameter arithmetic (`verify_cap.mjs`) |

## How this was built

- **AI tools:** the code, the website and the tooling were written with Claude Code (Anthropic). The model itself was trained from scratch: no pretrained weights, no fine-tuning, no distillation, and no hosted model behind the demo.
- **Generated video:** `film/comfy_hero.py` and `film/comfy_bg.py` generated abstract background clips locally with LTX-Video in ComfyUI for an earlier version of the landing page. Neither the current site nor the rendered film uses them.
- **Design references** came from Mobbin; each sourced component cites its reference in a comment above its markup.

## Credits

- **Training data:** FineWeb-Edu by Hugging Face (ODC-By 1.0).
- **Runtime:** onnxruntime-web (MIT), loaded from jsDelivr.
- **Evaluation:** lm-evaluation-harness by EleutherAI (MIT).
- **Fonts:** Newsreader, Instrument Sans and JetBrains Mono via Google Fonts. Inter (SIL OFL) is used in the film.
- **Photographs:** the Stari Most (Mostar) photographs on the site are loaded from a third-party host and are not part of this repository.
