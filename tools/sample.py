"""Generate text samples from a checkpoint, for the site's playground card.

The playground is evidence inside the argument, not the front door: a 49M model writes
fluently and is often wrong, and the page says so. So every rule here exists to keep the
samples from flattering it:

* The prompts and settings are fixed in tools/sample_spec.py, every prompt is published,
  and publish_results.py refuses a samples file whose prompts or settings differ from the
  spec. Nothing is generated twenty times with the best one kept, and a reroll with a
  different seed cannot reach the page.
* Each prompt gets its own seeded generator, so a rerun on the same device and precision
  reproduces the same words; the device and precision are published with the text.
* A run that has not finished training is refused unless --partial is passed: reading its
  ckpt.pt while the trainer is live can block the trainer's next save and end the run.
  --partial then samples on the CPU unless --device says otherwise, so the model does not
  take GPU memory from the trainer. Only the spec settings on a finished run's final weights
  take the samples.json the site reads: --partial text lands in samples_partial.json, other
  settings in samples_custom.json, an earlier checkpoint (best.pt) in samples_step<N>.json,
  and a combination in both names (samples_partial_custom.json, samples_step<N>_custom.json).
* Provenance follows eval_harness.py: the run is read off where the checkpoint lives,
  and the output lands in that run's directory for publish_results.py to pick up.

Run (after training has finished -- never against a checkpoint that is still being
rewritten by a live run):
    .venv\\Scripts\\python.exe tools/sample.py --run r1          # -> runs/r1/samples.json
    .venv\\Scripts\\python.exe tools/sample.py --random --device cpu   # plumbing check
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_harness as eh  # noqa: E402  (load_model, checkpoint_for, final_steps, ROOT)
from tools.runinfo import rel_or_abs, write_atomic  # noqa: E402
from tools.sample_spec import PROMPTS, SETTINGS  # noqa: E402


@torch.no_grad()
def generate(model, tok, prompt, max_new, temperature, top_k, gen, device, eot, block):
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    use_amp = device.startswith("cuda")
    for _ in range(max_new):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, _ = model(x[:, -block:])
        logits = logits[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())      # temperature 0 means greedy, not a divide by zero
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[-1]
                logits[logits < kth] = -float("inf")
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen))
        if nxt == eot:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="the run to sample (default r1)")
    ap.add_argument("--ckpt", default=None,
                    help="default runs/<run>/ckpt.pt, the finished model. Not while that run "
                         "is still training: the trainer replaces the file every save.")
    ap.add_argument("--random", action="store_true",
                    help="sample an untrained model: proves the plumbing, never published")
    ap.add_argument("--partial", action="store_true",
                    help="allow sampling a run that has not finished training (risky while live); "
                         "it runs on the CPU unless --device is given")
    ap.add_argument("--device", default=None,
                    help="default cuda when available, but cpu for an unfinished (--partial) run, "
                         "whose trainer is using the GPU")
    # Defaults are the spec. Other values are fine for experiments, but publish_results.py
    # refuses to put them on the site.
    ap.add_argument("--max-new-tokens", type=int, default=SETTINGS["max_new_tokens"])
    ap.add_argument("--temperature", type=float, default=SETTINGS["temperature"])
    ap.add_argument("--top-k", type=int, default=SETTINGS["top_k"])
    ap.add_argument("--seed", type=int, default=SETTINGS["seed"])
    ap.add_argument("--out", default=None,
                    help="default runs/<run>/samples.json for the spec settings on a finished "
                         "run's final weights; otherwise samples_partial (unfinished run), "
                         "samples_step<N> (an earlier checkpoint), _custom (other settings), "
                         "combined as they apply, e.g. samples_partial_custom.json")
    args = ap.parse_args()
    settings = {"temperature": args.temperature, "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens, "seed": args.seed}
    # A random-init model decodes to arbitrary bytes; a cp1252 console would crash on them.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # Seed the global RNG as well: --random draws its weights from it, so without this two
    # plumbing runs sample two different models and reproducibility cannot be checked at
    # all. A real checkpoint's weights come from the file and are unaffected.
    torch.manual_seed(args.seed)

    # The checkpoint, run and device rules of eval_harness.py, from the same function.
    ckpt_run, finished = eh.checkpoint_for(args, "sample")
    if args.out is None:
        if args.random:
            args.out = os.path.join(eh.ROOT, "runs", "samples_random.json")
        elif ckpt_run is None:
            raise SystemExit(f"{args.ckpt} is not inside runs/<run>/; pass --out explicitly")
        # Otherwise named below, once load_model says which step the weights hold.

    model, cfg, tok, step, _ = eh.load_model(args.ckpt, args.device, args.random)
    if args.out is None:
        # samples.json is the file publish_results.py reads, and it takes only the spec
        # settings on a finished run's final weights. Everything else is named for what it
        # is, in every dimension that applies: not finished (samples_partial), a checkpoint
        # that is not the final one (samples_step<N>: best.pt lags the end of a run), other
        # settings (_custom). Naming one dimension only meant --partial --temperature 0
        # replaced the spec-settings samples_partial.json, the one file publish --samples
        # accepts mid-run, with text it refuses.
        stem = "samples" if finished else "samples_partial"
        if finished and step != eh.final_steps(ckpt_run):
            stem += f"_step{step}"
        if settings != SETTINGS:
            stem += "_custom"
        args.out = os.path.join(eh.ROOT, "runs", ckpt_run, stem + ".json")
    eot = tok.token_to_id("<|endoftext|>")
    samples = []
    for i, prompt in enumerate(PROMPTS):
        # One generator per prompt, seeded by position, so each sample reproduces on its
        # own rather than depending on how many tokens the prompts before it drew.
        gen = torch.Generator(device=args.device).manual_seed(args.seed + i)
        text = generate(model, tok, prompt, args.max_new_tokens, args.temperature,
                        args.top_k, gen, args.device, eot, cfg.block_size)
        samples.append({"prompt": prompt, "completion": text})
        print(f"--- {prompt}\n{text}\n")

    payload = {
        "weights": "random-init" if args.random else rel_or_abs(args.ckpt, eh.ROOT),
        "run": ckpt_run,
        "steps_completed": step,
        "settings": settings,
        # CPU and CUDA draw different random streams, and CUDA samples in bf16, so a seed
        # reproduces the text only on the same device and precision.
        "runtime": {"device": args.device,
                    "precision": "bf16" if args.device.startswith("cuda") else "fp32"},
        "note": "every prompt in tools/sample_spec.py PROMPTS is published; nothing is filtered",
        "samples": samples,
    }
    # As in eval_harness.py: abspath for a bare --out name, and never truncate samples.json
    # in place.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_atomic(args.out, json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
