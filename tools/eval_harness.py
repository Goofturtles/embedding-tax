"""Score the model with the real lm-evaluation-harness.

GIBC V2 Track 01 says entries are scored on HellaSwag, ARC-Easy, PIQA and WinoGrande
via lm-evaluation-harness, plus WikiText-103 perplexity. So this registers our model
INTO that harness rather than reimplementing the four tasks. That matters more than it
sounds: each task has a scoring convention that is easy to get subtly wrong by hand
(HellaSwag is ranked by length-normalised accuracy, WinoGrande scores two partial
contexts against a shared continuation). Reimplementing them from memory would produce
numbers that quietly disagree with the ones the judges compute.

A note on perplexity, because two different numbers carry that name in this project.
The training log's validation perplexity is per TOKEN, so it depends on the vocabulary:
with 16,384 tokens each one carries less text than GPT-2's, which lowers per-token
perplexity without the model being any better. It only compares between models that
share this tokenizer. The harness's wikitext `word_perplexity` and `bits_per_byte` are
normalised by the words and bytes of the text itself, so the tokenizer cancels out and
both compare fairly across entries (lm-eval's preprocess_wikitext.py divides by a
whitespace word count, never by tokens). lm-eval's `wikitext` task is the WikiText-2
test set, which WikiText-103 shares.

Run (after training has finished -- see --ckpt):
    .venv\\Scripts\\python.exe tools/eval_harness.py --run r1          # -> runs/r1/eval.json
    .venv\\Scripts\\python.exe tools/eval_harness.py --run r1 --limit 20
    .venv\\Scripts\\python.exe tools/eval_harness.py --random --limit 10 --device cpu
"""
import argparse
import json
import math
import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import GPTConfig  # noqa: E402
from model.gpt import GPT  # noqa: E402
from tools.runinfo import read_log, rel_or_abs, run_complete, run_dir_name, write_atomic  # noqa: E402  (imports no torch)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TOKENIZER_PATH = os.path.join(DATA, "tokenizer.json")

# The four the rules name, plus wikitext for perplexity / bits-per-byte.
TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande"]
PPL_TASKS = ["wikitext"]

# What a model that has learned nothing scores, by construction: one over the number of
# choices. Used to sanity-check the plumbing before trusting any real number.
CHANCE = {"hellaswag": 0.25, "arc_easy": 0.25, "piqa": 0.50, "winogrande": 0.50}


def load_model(ckpt_path, device, random_init=False):
    """Return (model, cfg). --random builds an untrained model on purpose: see main()."""
    from tokenizers import Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        raise SystemExit(f"no tokenizer at {TOKENIZER_PATH}; run tools/prepare_data.py")
    tok = Tokenizer.from_file(TOKENIZER_PATH)

    if random_init:
        cfg = GPTConfig()
        model = GPT(cfg)
        step, val = None, None
    else:
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"no checkpoint at {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = GPTConfig(**state["cfg"])
        model = GPT(cfg)
        model.load_state_dict(state["model"])
        # The two checkpoint files count differently: ckpt.pt stores the step to resume
        # FROM, which equals steps completed; best.pt stores the index of the step it was
        # saved at, one less. Normalised here so provenance is a single number.
        step = state.get("step", -1)
        step = step if "opt" in state else step + 1
        val = state.get("val_loss", state.get("final_val"))
    model.to(device).eval()
    return model, cfg, tok, step, val


def run_of(ckpt_path):
    """The run a checkpoint belongs to: the name of its directory when that directory
    sits directly in runs/, else None. realpath so the name comes back in the case it has
    on disk -- a typed `runs\\R1\\...` otherwise produced run "R1", which publish_results
    then refused for not being "r1". The containment test is runinfo.run_dir_name, the same
    one --run and publish_results.py go through."""
    return run_dir_name(ROOT, os.path.dirname(os.path.realpath(ckpt_path)))


def final_steps(run):
    """The steps a run's log says it completed, once that reaches its step-aligned target
    (runinfo.run_complete, the test publish_results.py uses for run_complete); None before."""
    log = os.path.join(ROOT, "runs", run, "log.jsonl")
    if not os.path.exists(log):
        return None
    recs, _ = read_log(log)
    return recs[-1]["step"] + 1 if recs and run_complete(recs[-1]) else None


def run_finished(run):
    """True once a run's log shows it reached its step-aligned target (final_steps) AND its
    final ckpt.pt is on disk. Tools that read a run's ckpt.pt refuse an unfinished run by
    default: the trainer replaces that file every save, and a long read on Windows can
    block the rename and end the run.

    The log alone is not enough. train.py writes the last line ~20 s before its final
    evaluation and save, and in that window ckpt.pt still holds an earlier periodic save.
    The final save lands after the last line, so it is checked by mtime -- never by
    opening the file the trainer is about to replace. (A periodic save also postdates the
    last line when the final step is a multiple of --save-every; r1's 14,494 is not.)"""
    log = os.path.join(ROOT, "runs", run, "log.jsonl")
    ckpt = os.path.join(ROOT, "runs", run, "ckpt.pt")
    if final_steps(run) is None or not os.path.exists(ckpt):
        return False
    return os.path.getmtime(ckpt) > os.path.getmtime(log)


def checkpoint_for(args, verb):
    """The front half eval_harness.py and sample.py share: which checkpoint, the run it
    belongs to, whether that run has finished, and the device. Sets args.ckpt and
    args.device; returns (ckpt_run, finished). verb names the work: "score" or "sample".

    The run a result belongs to is decided by where the weights live, and each tool's --out
    follows it. Deriving it from --run alone meant `--ckpt runs/r2/best.pt` with --run left
    at its default silently overwrote runs/r1/eval.json with r2's scores."""
    ckpt_run, finished = None, False
    if not args.random:
        args.ckpt = args.ckpt or os.path.join(ROOT, "runs", args.run or "r1", "ckpt.pt")
        ckpt_run = run_of(args.ckpt)
        # --run is resolved the way the checkpoint's directory is. Compared as typed, --run R1
        # was refused for a checkpoint in runs/r1/ as "belonging to run 'r1'".
        if args.run and ckpt_run and run_dir_name(ROOT, args.run) != ckpt_run:
            raise SystemExit(f"--run {args.run} but {args.ckpt} belongs to run {ckpt_run!r}; "
                             f"drop --run or point --ckpt inside runs/{args.run}/")
        finished = bool(ckpt_run) and run_finished(ckpt_run)
        if ckpt_run and not args.partial and not finished:
            raise SystemExit(f"run {ckpt_run} has not finished training. Reading its checkpoint "
                             f"now can block the trainer's next save and end the run. Wait for "
                             f"it to finish, or pass --partial to {verb} it on purpose; --partial "
                             f"runs on the CPU unless --device says otherwise, so the model does "
                             f"not take GPU memory from the trainer.")
    if args.device is None:
        # An unfinished run may still be training, and its trainer holds most of the GPU: a
        # second model loaded beside it can run the card out of memory and end the run. So a
        # --partial read defaults to the CPU; an explicit --device still wins.
        args.device = ("cpu" if ckpt_run and not finished
                       else "cuda" if torch.cuda.is_available() else "cpu")
    return ckpt_run, finished


def build_lm(model, cfg, tok, device, batch_size):
    """Define and register the lm-eval model class.

    Built inside a function so importing this module does not register a duplicate
    model name, which lm-eval treats as an error.
    """
    from lm_eval import utils
    from lm_eval.api.model import TemplateLM
    from tqdm import tqdm

    class EmbeddingTaxLM(TemplateLM):
        def __init__(self):
            super().__init__()
            self.model = model
            self.cfg = cfg
            self.tok = tok
            self._device = torch.device(device)
            self.batch_size = batch_size
            self._max_length = cfg.block_size
            self._eot = tok.token_to_id("<|endoftext|>")
            assert self._eot is not None, "tokenizer has no <|endoftext|>"

        @property
        def eot_token_id(self):
            return self._eot

        @property
        def max_length(self):
            return self._max_length

        @property
        def device(self):
            return self._device

        def tok_encode(self, string: str, **kwargs) -> List[int]:
            return self.tok.encode(string).ids

        def tok_decode(self, tokens) -> str:
            return self.tok.decode(list(tokens))

        def _forward(self, padded):
            ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if self._device.type == "cuda" else torch.autocast("cpu", enabled=False))
            with torch.no_grad(), ctx:
                logits, _ = self.model(padded)
            return logits.float()

        def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None
                                  ) -> List[Tuple[float, bool]]:
            # Right-padding is safe here: attention is causal, so a real token never
            # attends to a pad that follows it. Left-padding would need position care.
            out = [None] * len(requests)
            order = sorted(range(len(requests)),
                           key=lambda i: -(len(requests[i][1]) + len(requests[i][2])))
            bs = override_bs or self.batch_size

            for s in tqdm(range(0, len(order), bs), disable=disable_tqdm, leave=False):
                idxs = order[s:s + bs]
                seqs, inps, conts = [], [], []
                for i in idxs:
                    _, ctx_enc, cont_enc = requests[i]
                    seq = (list(ctx_enc) + list(cont_enc))[-(self.max_length + 1):]
                    inps.append(seq[:-1])
                    # Clamp against a continuation longer than the window, so the
                    # scored span can never run off the front of the input.
                    conts.append(min(len(cont_enc), len(seq) - 1))
                    seqs.append(seq)

                width = max(len(v) for v in inps)
                padded = torch.full((len(inps), width), self._eot, dtype=torch.long)
                for k, v in enumerate(inps):
                    padded[k, :len(v)] = torch.tensor(v, dtype=torch.long)
                logprobs = F.log_softmax(self._forward(padded.to(self._device)), dim=-1)

                for k, i in enumerate(idxs):
                    L, C = len(inps[k]), conts[k]
                    span = logprobs[k, L - C:L, :]
                    tgt = torch.tensor(seqs[k][-C:], dtype=torch.long, device=self._device)
                    picked = span.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    greedy = bool((span.argmax(dim=-1) == tgt).all())
                    out[i] = (float(picked.sum()), greedy)
            return out

        def loglikelihood_rolling(self, requests, disable_tqdm=False) -> List[float]:
            # Same windowing HFLM uses, so wikitext numbers are computed the harness way.
            results = []
            for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
                windows = [(None,) + x for x in map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ))]
                scored = self._loglikelihood_tokens(windows, disable_tqdm=True)
                results.append(sum(lp for lp, _ in scored))
            return results

        def generate_until(self, requests, disable_tqdm=False) -> List[str]:
            outs = []
            for req in tqdm(requests, disable=disable_tqdm):
                context, kwargs = req.args
                until = kwargs.get("until") or []
                if isinstance(until, str):
                    until = [until]
                max_new = kwargs.get("max_gen_toks", 128)
                ids = self.tok_encode(context)[-(self.max_length - 1):]
                x = torch.tensor([ids], dtype=torch.long, device=self._device)
                text = ""
                for _ in range(max_new):
                    logits = self._forward(x[:, -self.max_length:])
                    nxt = int(logits[0, -1].argmax())
                    x = torch.cat([x, torch.tensor([[nxt]], device=self._device)], dim=1)
                    text = self.tok_decode(x[0, len(ids):].tolist())
                    if any(u and u in text for u in until):
                        break
                for u in until:
                    if u and u in text:
                        text = text.split(u)[0]
                outs.append(text)
            return outs

    return EmbeddingTaxLM()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="the run to score (default r1); sets the --ckpt default. --out "
                         "follows the run the checkpoint actually lives in.")
    ap.add_argument("--ckpt", default=None,
                    help="default runs/<run>/ckpt.pt: the finished model. Not best.pt, which "
                         "only updates at eval steps and so can lag the end of a run. Do not "
                         "point this at ckpt.pt while that run is still training: the trainer "
                         "replaces that file every save and a long read can block the rename.")
    ap.add_argument("--random", action="store_true",
                    help="score an untrained model: proves the plumbing, since a wired-up "
                         "harness returns chance and a broken one returns 0 or 1")
    ap.add_argument("--partial", action="store_true",
                    help="allow scoring a run that has not finished training (risky while live); "
                         "it runs on the CPU unless --device is given")
    ap.add_argument("--device", default=None,
                    help="default cuda when available, but cpu for an unfinished (--partial) run, "
                         "whose trainer is using the GPU")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="documents per task; None = all")
    ap.add_argument("--tasks", default=",".join(TASKS + PPL_TASKS))
    ap.add_argument("--out", default=None,
                    help="default runs/<run>/eval.json for a full evaluation of a finished run's "
                         "final weights. A random-init, --limit, --tasks subset, unfinished "
                         "(eval_partial...) or earlier-checkpoint (eval_step<N>...) score writes "
                         "elsewhere, so it can never overwrite the file publish_results.py reads.")
    args = ap.parse_args()
    ckpt_run, finished = checkpoint_for(args, "score")
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.out is None:
        if args.random:
            args.out = os.path.join(ROOT, "runs", "eval_random.json")
        elif ckpt_run is None:
            raise SystemExit(f"{args.ckpt} is not inside runs/<run>/, so there is no run to "
                             f"file its scores under; pass --out explicitly")
        # Otherwise named below, once load_model says which step the weights hold.

    import lm_eval

    model, cfg, tok, step, val = load_model(args.ckpt, args.device, args.random)
    if args.out is None:
        # eval.json is the file publish_results.py reads by default, and it takes only every
        # task on a finished run's FINAL weights; each other way of scoring is named for what
        # it is, so that no two of them share a file. "Not finished" goes first (a --limit 20
        # of a live run and the same --limit 20 once it had finished used to share
        # eval_limit20.json), then the step of any checkpoint that is not the final one:
        # best.pt lags the end of a run, and its --tasks arc_easy used to replace the final
        # weights' publishable eval_tasks-arc_easy.json with a file publish refuses. --limit
        # and a --tasks subset then keep the full evaluation's name to themselves.
        stem = "eval" if finished else "eval_partial"
        if finished and step != final_steps(ckpt_run):
            stem += f"_step{step}"
        if args.limit is not None:
            name = f"{stem}_limit{args.limit}.json"
        elif set(task_list) != set(TASKS + PPL_TASKS):
            name = f"{stem}_tasks-{'+'.join(sorted(set(task_list)))}.json"
        else:
            name = f"{stem}.json"
        args.out = os.path.join(ROOT, "runs", ckpt_run, name)
    n_params = model.num_params()
    tag = ("RANDOM INIT (plumbing check)" if args.random
           else f"{args.ckpt} after {step:,} steps")
    print(f"model    {n_params:,} params, {cfg.n_layer} layers, vocab {cfg.vocab_size:,}")
    print(f"weights  {tag}")
    print(f"device   {args.device}   batch {args.batch_size}   limit {args.limit}")
    print()

    lm = build_lm(model, cfg, tok, args.device, args.batch_size)
    res = lm_eval.simple_evaluate(model=lm, tasks=task_list, limit=args.limit,
                                  bootstrap_iters=0, verbosity="WARNING")

    print(f"{'task':<16}{'metric':<22}{'value':>10}")
    print("-" * 48)
    rows = {}
    for task, metrics in sorted(res["results"].items()):
        for name, value in sorted(metrics.items()):
            if name == "alias" or name.endswith("_stderr,none"):
                continue
            key = name.split(",")[0]
            # A non-finite score is stored as null and said so here, not written as NaN:
            # publish_results.py would null it anyway but count it as a training
            # divergence, and the site would then say the run diverged when it did not.
            if isinstance(value, float) and not math.isfinite(value):
                print(f"warning: {task}.{key} came back {value}; publishing it as null")
                value = None
            rows[f"{task}.{key}"] = value
            flag = ""
            if key in ("acc_norm", "acc") and task in CHANCE:
                flag = f"   (chance {CHANCE[task]:.2f})"
            shown = "null" if value is None else f"{value:.4f}"
            print(f"{task:<16}{key:<22}{shown:>10}{flag}")

    payload = {
        "weights": "random-init" if args.random else rel_or_abs(args.ckpt, ROOT),
        "run": ckpt_run,
        "steps_completed": step,
        "val_loss": val,
        "params": n_params,
        "limit": args.limit,
        "harness": f"lm-evaluation-harness {lm_eval.__version__}",
        "note": ("wikitext word_perplexity and bits_per_byte are normalised per word and per "
                 "byte, so they compare across tokenizers; per-token perplexity would not."),
        "results": rows,
    }
    # abspath: a bare --out name has dirname '', which os.makedirs rejects -- after the
    # whole evaluation. Serialised in full, then a temp renamed over the file: opening it
    # with "w" emptied a good eval.json before an interrupted dump could replace it.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # allow_nan=False: after the pass above nothing non-finite should remain; if it does,
    # fail before the file is touched rather than write a NaN the browser cannot parse.
    write_atomic(args.out, json.dumps(payload, indent=2, allow_nan=False))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
