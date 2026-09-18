"""Fill site/results.json from measured artifacts, and from nothing else.

This exists because the same failure kept happening by hand: prose on the site claimed
a finished training run while results.json was still all nulls. The fix is structural
rather than editorial -- the only path to a number on that page is this script, and this
script only copies values that exist in runs/<id>/log.jsonl and runs/<id>/eval.json.

Rules that make that guarantee actually hold, each learned the hard way:

* **Every measured field is reset to null before anything is written.** Merging into
  whatever was already in the file meant a publish for a run with no validation points
  yet silently kept the PREVIOUS run's val curve and best loss, and showed them beside
  the new run's token count. A field is measured this publish or it is null.
* **A missing log is a hard error, not an empty result.** `--run typo` used to exit 0,
  print a success line, and leave the old numbers standing.
* **An eval belongs to one run and one checkpoint.** Scores and samples are read only from
  the run's own directory, refused if they name a different run, and published with the
  step count of the weights they scored -- an early checkpoint's scores beside the final
  curve read as the final result. Once the run is complete, anything but its final weights
  is refused, and so is a file older than the checkpoint it names (the run was trained
  again since); before then, a step count the log has not reached yet (or none at all) is
  refused, and so are eval.json and samples.json themselves, which the tools write only
  for a finished run. Complete or not, a file older than the run's meta.json -- written
  when the current training began -- came from an earlier training of the same name.
* **A file with no results rows measured nothing.** It is refused rather than reported as
  the source of four null scores.
* **JSON has no NaN.** A diverged loss is written as null and counted once, rather than as
  a bare NaN that no browser will parse. A perplexity train.py capped is null as well.

Anything not measured stays null, and the site renders null as "not measured yet". A
field is never estimated, carried over from a previous run, or filled from a comment.

Run:
    .venv\\Scripts\\python.exe tools/publish_results.py --run r1
    .venv\\Scripts\\python.exe tools/publish_results.py --run r1 --eval runs/r1/eval.json
    .venv\\Scripts\\python.exe tools/publish_results.py --run r1 --samples runs/r1/samples_partial.json
"""
import argparse
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.runinfo import (read_log, rel_or_abs, run_complete, run_dir_name,  # noqa: E402
                           step_target, write_atomic)
from tools.sample_spec import PROMPTS as SAMPLE_PROMPTS, SETTINGS as SAMPLE_SETTINGS  # noqa: E402
SITE = os.path.join(ROOT, "site", "results.json")

# lm-eval metric -> the key the site uses. HellaSwag and ARC-Easy are conventionally
# reported as length-normalised accuracy; PIQA and WinoGrande as plain accuracy.
BENCH = {
    "arc_easy": "acc_norm",
    "piqa": "acc",
    "hellaswag": "acc_norm",   # HellaSwag is always reported length-normalised
    "winogrande": "acc",       # WinoGrande has no acc_norm: both endings share a suffix
}

# Everything under "training" that this script owns. Reset to None on every publish so
# a value can only survive if this run measured it. "hardware" is deliberately absent:
# it describes the machine, not the run, and is not a measurement. peak_vram_gb is listed
# although nothing measures it yet: left out, a value typed into the file by hand would
# survive every publish.
MEASURED = ("tokens_seen", "tokens_target", "wall_clock_hours", "tokens_per_second",
            "steps", "final_train_loss", "corpus", "run_complete", "peak_vram_gb",
            "val_curve", "best_val_loss", "best_val_ppl", "nonfinite_values", "rest_hours")


def is_finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def training_block(recs, meta):
    last = recs[-1]
    tps = [r["tok_per_s"] for r in recs if "tok_per_s" in r]
    # Median, not mean: the first few steps include CUDA context setup and warm-up
    # allocation, and a mean lets those drag a number meant to describe the steady state.
    #
    # tokens_target comes from the log or not at all. It used to be hardcoded in the
    # site's JS, which made the progress percentage the one hand-typed number on a page
    # whose whole claim is that nothing is hand-typed.
    seen = last.get("tokens")
    steps = last.get("step", -1) + 1
    # Aligned to whole steps, and "complete" decided, by the rule eval_harness.py uses to
    # name a finished run's files: see runinfo.step_target.
    target = step_target(last)
    return {
        "tokens_seen": seen,
        "tokens_target": target,
        "wall_clock_hours": round(last["elapsed_s"] / 3600, 3) if "elapsed_s" in last else None,
        # Idle time is inside the wall clock: the cooling pauses, and since --duty the GPU
        # load cap, the idle train.py takes inside each step (~0.66 s a step at --duty 0.65,
        # which is most of it). Published separately so an efficiency figure can be read as
        # compute time without the idle flattering or hiding it. Still rest_hours:
        # results.js reads the field by that name.
        "rest_hours": round(last["rest_s"] / 3600, 3) if "rest_s" in last else None,
        "tokens_per_second": round(statistics.median(tps)) if tps else None,
        "steps": steps,
        "final_train_loss": last.get("train_loss"),
        "run_complete": run_complete(last),
        "corpus": (f"{meta.get('dataset')} -- {meta.get('train_tokens', 0):,} training "
                   f"tokens, {meta.get('val_tokens', 0):,} held out") if meta else None,
    }


def val_curve(recs):
    """Every measured validation point, for the site to plot. Not downsampled: at one
    point per 250 steps a full run is ~58 points, small enough to ship whole.

    A point's ppl is only as good as its loss. train.py logs exp(min(20, loss)), so from a
    loss of 20 up the logged ppl is the cap, 485,165,195, not a measurement (a loss of 23 is
    really ~9.7e9), and min(20, nan) is 20 too. Those points publish a null ppl. It is null
    outright, not a carried NaN, so a NaN loss is counted once in nonfinite_values, not twice."""
    return [{"step": r["step"], "tokens": r["tokens"], "loss": r["val_loss"],
             "ppl": r["val_ppl"] if is_finite(r["val_loss"]) and r["val_loss"] < 20 else None}
            for r in recs if "val_loss" in r]


def eval_run(ev):
    """The run an eval or samples file belongs to. An explicit "run" field wins even when it
    is null: the tool that wrote it found the checkpoint outside runs/<id>/ and said it
    belongs to no run, and guessing one from the path would overrule it. Only a file with
    no "run" field at all (written before the field existed) falls back to its weights."""
    if "run" in ev:
        return ev["run"]
    w = str(ev.get("weights", "")).replace("\\", "/")
    parts = w.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "runs" else None


def steps_refusal(f, run, log_steps, complete):
    """Why a file's steps_completed cannot be published beside this log, or None.

    Once the run is complete only its final weights count. Before then the count must still
    be one the log has reached: eval_harness.py and sample.py never write eval.json or
    samples.json for an unfinished run, so one claiming more steps (or none) is left over
    from an earlier completion of the same run name, or placed by hand."""
    sc = f.get("steps_completed")
    if not isinstance(sc, int) or isinstance(sc, bool):
        return f"it records no step count ({sc!r}), so the page could not say which weights"
    if complete and sc != log_steps:
        return f"it came from the weights after {sc} steps, but run {run} completed {log_steps}"
    if sc > log_steps:
        return f"it came from the weights after {sc} steps, but run {run}'s log has reached {log_steps}"
    return None


def place_refusal(path, run):
    """Why a file cannot be published for this run because of where it sits, or None. Only
    runs/<run>/ is read: a --eval path on another drive once reached the public results.json
    as an absolute local path, and a file outside the run's directory is not that run's by
    construction."""
    if run_dir_name(ROOT, os.path.dirname(os.path.realpath(path))) != run:
        return f"it is not in runs/{run}/, and only a run's own directory is published from"
    return None


def stale_refusal(path, f):
    """Why a complete run's file is older than the checkpoint it names, or None.

    A run name can be trained again to the same step count, and every other check then
    passes for the old file beside the new curve. The file is written after its weights were
    read, so it must be the newer of the two. Applied only once the run is complete: before
    then the trainer replaces ckpt.pt at every save, so an honest mid-run file is soon older
    than it. A checkpoint no longer on disk proves nothing either way."""
    w = f.get("weights")
    if not isinstance(w, str):
        return None
    ckpt = w if os.path.isabs(w) else os.path.join(ROOT, w)
    if os.path.exists(ckpt) and os.path.getmtime(ckpt) >= os.path.getmtime(path):
        return (f"{w} was written after it, so it scored weights that have since been "
                f"replaced (was the run trained again?)")
    return None


def start_refusal(path, run):
    """Why a file predates the training now in runs/<run>/, or None.

    A run name can be trained again from scratch: log.jsonl, ckpt.pt and best.pt deleted
    (train.py refuses a fresh start otherwise), and the old training's eval.json and
    samples.json left behind. Once the new log reaches their step count every other check
    passes them, and the page shows scores "after N steps" under a curve whose Nth step
    belongs to a different training. What records when the current training began is
    runs/<run>/meta.json: train.py copies data/meta.json over it at every start from step 0
    and leaves it alone on a resume. A file older than it scored weights this training never
    produced, complete or not. No snapshot proves nothing either way."""
    meta = os.path.join(ROOT, "runs", run, "meta.json")
    if os.path.exists(meta) and os.path.getmtime(meta) >= os.path.getmtime(path):
        return (f"runs/{run}/meta.json, which train.py writes when a training starts, is "
                f"newer than it, so it came from an earlier training of this run name")
    return None


def leftover_refusal(path, run, complete):
    """Why eval.json or samples.json cannot be published before the run is complete, or None.

    eval_harness.py and sample.py write those two names only for a finished run; every
    other score or sample goes by another name (eval_partial.json, samples_partial.json,
    ...). So before the log says complete, a file under either name was left behind by an
    earlier training of the same run name, or placed by hand -- and once the new log had
    reached its step count, the default publish took it. normcase: on Windows EVAL.JSON
    opens the same file."""
    name = os.path.basename(path)
    if complete or os.path.normcase(name) not in ("eval.json", "samples.json"):
        return None
    partial = name[:-5] + "_partial.json"
    return (f"runs/{run}/{name} is written only for a finished run, and this one has not "
            f"finished, so it was left behind by an earlier training of the name (or placed "
            f"by hand); mid-run results go by their own name, such as runs/{run}/{partial}")


def bench_block(eval_path, previous, run, log_steps, complete):
    """Copy measured scores in, keeping the published reference values already there.
    log_steps and complete are the run's, from its log: see steps_refusal."""
    out = json.loads(json.dumps(previous))          # deep copy; don't mutate the caller
    for task in BENCH:
        if task in out and isinstance(out[task], dict):
            out[task]["ours"] = None                # measured this publish, or null
    out["wikitext103_ppl"] = None
    for key in ("wikitext103_bits_per_byte", "_measured_from", "_weights", "_steps_completed"):
        out.pop(key, None)

    why = place_refusal(eval_path, run)
    if why:
        print(f"refusing to publish {eval_path}: {why}")
        return out, False
    if not os.path.exists(eval_path):
        return out, False
    with open(eval_path, encoding="utf-8") as f:
        ev = json.load(f)
    if ev.get("weights") == "random-init":
        print(f"refusing to publish {eval_path}: it scores a random-init model")
        return out, False
    if ev.get("limit") is not None:
        print(f"refusing to publish {eval_path}: it used --limit {ev['limit']}, "
              f"so the scores are on a subset, not the benchmark")
        return out, False
    scored = eval_run(ev)
    if scored != run:
        print(f"refusing to publish {eval_path}: it scored run {scored!r}, not {run!r}")
        return out, False
    # After the file has said which run it is, so a wrong-run file is refused for that, the
    # more specific reason. The trainer logs its last step ~20 s before saving the final
    # ckpt.pt, so an eval run in that window scored an earlier checkpoint while the log
    # already said complete.
    why = leftover_refusal(eval_path, run, complete)
    if not why:
        why = steps_refusal(ev, run, log_steps, complete)
    if not why:
        why = start_refusal(eval_path, run)
    if not why and complete:
        why = stale_refusal(eval_path, ev)
    if why:
        print(f"refusing to publish {eval_path}: {why}")
        return out, False
    rows = ev.get("results")
    if not isinstance(rows, dict) or not rows:
        # A samples file passed as --eval clears every gate above; it must not be reported
        # as the measurement source of four null scores.
        print(f"refusing to publish {eval_path}: it holds no results rows, so nothing was measured")
        return out, False

    for task, metric in BENCH.items():
        v = rows.get(f"{task}.{metric}")
        if v is None:
            continue
        if task not in out:
            # Loud, because silently dropping a measured score is the same class of bug
            # as inventing one.
            print(f"warning: {task} scored {v:.4f} but results.json has no row for it; "
                  f"add the row or the number is lost")
            continue
        out[task]["ours"] = round(v * 100, 2)       # site shows percentages
    wt = rows.get("wikitext.word_perplexity")
    if wt is not None:
        out["wikitext103_ppl"] = round(wt, 2)
    bpb = rows.get("wikitext.bits_per_byte")
    if bpb is not None:
        # Both wikitext figures are normalised by the text (per word, per byte), so both
        # compare across tokenizers. It is the training log's per-TOKEN val perplexity
        # that depends on the vocabulary, and the page says so beside that number.
        out["wikitext103_bits_per_byte"] = round(bpb, 4)
    # Named from the run, not the path: place_refusal kept the file inside runs/<run>/, and
    # a relpath fallback once put an absolute local path on the public page.
    out["_measured_from"] = f"runs/{run}/{os.path.basename(eval_path)}"
    out["_weights"] = ev.get("weights")
    out["_steps_completed"] = ev.get("steps_completed")
    return out, True


def samples_block(path, run, log_steps, complete):
    """The playground's text, under the same provenance rules as the scores: never from a
    random-init model, never from another run's file, always with the step count of the
    weights that wrote it, only from runs/<run>/, and only from the final weights (and a
    file newer than them) once the run is complete."""
    empty = {"items": [], "settings": None, "_steps_completed": None, "_weights": None}
    why = place_refusal(path, run)
    if why:
        print(f"refusing to publish {path}: {why}")
        return empty, False
    if not os.path.exists(path):
        return empty, False
    with open(path, encoding="utf-8") as f:
        sp = json.load(f)
    if sp.get("weights") == "random-init":
        print(f"refusing to publish {path}: samples from a random-init model")
        return empty, False
    scored = eval_run(sp)
    if scored != run:
        print(f"refusing to publish {path}: it sampled run {scored!r}, not {run!r}")
        return empty, False
    why = leftover_refusal(path, run, complete)
    if not why:
        why = steps_refusal(sp, run, log_steps, complete)
    if not why:
        why = start_refusal(path, run)
    if not why and complete:
        why = stale_refusal(path, sp)
    if why:
        print(f"refusing to publish {path}: {why}")
        return empty, False
    # The page says every prompt in the spec is shown at its fixed settings. These two
    # checks are what make that sentence true: a reroll with another seed, a tweaked
    # temperature, or an edited prompt list never reaches the site.
    samples = [s for s in sp.get("samples", []) if isinstance(s, dict)]
    if [s.get("prompt") for s in samples] != list(SAMPLE_PROMPTS):
        print(f"refusing to publish {path}: its prompts are not exactly tools/sample_spec.py "
              f"PROMPTS, so the page's 'every prompt' would not be true")
        return empty, False
    if sp.get("settings") != SAMPLE_SETTINGS:
        print(f"refusing to publish {path}: settings {sp.get('settings')} differ from the "
              f"fixed {SAMPLE_SETTINGS} in tools/sample_spec.py")
        return empty, False
    items = [{"prompt": str(s.get("prompt", "")), "completion": str(s.get("completion", ""))}
             for s in samples]
    return {"items": items, "settings": sp.get("settings"), "runtime": sp.get("runtime"),
            "_steps_completed": sp.get("steps_completed"), "_weights": sp.get("weights")}, True


def finite_or_none(o, hits):
    """Replace NaN and +-inf with None, recording each one.

    json.dump writes a bare NaN by default, which is not JSON: every browser refuses to
    parse the file, so one diverged step would blank the entire results card. Written as
    null instead, and counted in training.nonfinite_values so the page can say the run
    may have diverged rather than quietly showing gaps.
    """
    if isinstance(o, float) and not math.isfinite(o):
        hits.append(o)
        return None
    if isinstance(o, dict):
        return {k: finite_or_none(v, hits) for k, v in o.items()}
    if isinstance(o, list):
        return [finite_or_none(v, hits) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r1")
    ap.add_argument("--eval", default=None,
                    help="lm-eval output to publish, from runs/<run>/; default "
                         "runs/<run>/eval.json, which eval_harness.py writes only for a full "
                         "evaluation of a finished run. Mid-run, pass "
                         "runs/<run>/eval_partial.json (eval_harness.py --partial)")
    ap.add_argument("--samples", default=None,
                    help="sample.py output to publish, from runs/<run>/; default "
                         "runs/<run>/samples.json, which sample.py writes only for a finished "
                         "run. Mid-run, pass runs/<run>/samples_partial.json (sample.py --partial)")
    ap.add_argument("--out", default=SITE)
    args = ap.parse_args()

    # The run is named as its directory is on disk. Any spelling that still reaches
    # runs/r1/log.jsonl -- R1, r1/, .\r1 -- used to find the log, then refuse r1's own
    # eval.json and samples.json for naming "r1" rather than the typed string: every score
    # nulled, and exit 0. runinfo.run_dir_name, which eval_harness.py and sample.py use too.
    name = run_dir_name(ROOT, args.run)
    if name is None:
        raise SystemExit(f"--run {args.run!r} is not a directory directly inside runs/ -- "
                         f"refusing to publish.")
    args.run = name
    log_path = os.path.join(ROOT, "runs", args.run, "log.jsonl")
    if not os.path.exists(log_path):
        raise SystemExit(f"no log at {log_path} -- refusing to publish. A publish that "
                         f"finds nothing must not exit 0 and leave the previous run's "
                         f"numbers standing on the page.")
    recs, torn = read_log(log_path)
    if torn:
        print(f"note: skipped {torn} unparseable log line(s)")
    if not recs:
        raise SystemExit(f"{log_path} has no parseable records -- refusing to publish.")

    # The corpus is read from the run's own snapshot (train.py copies data/meta.json into
    # the run directory when training starts), never from data/ at publish time: data/
    # describes whatever it holds NOW, and a rebuild between training and publishing would
    # put the new corpus on the old run. No snapshot means the corpus stays unmeasured.
    meta_path = os.path.join(ROOT, "runs", args.run, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else None
    if meta is None:
        print(f"note: no corpus snapshot at {meta_path}; corpus left null")
    eval_path = args.eval or os.path.join(ROOT, "runs", args.run, "eval.json")

    with open(args.out, encoding="utf-8") as f:
        site = json.load(f)

    site["_comment"] = ("Written by tools/publish_results.py from runs/<id>/log.jsonl and "
                        "runs/<id>/eval.json. Every measured field is reset to null on each "
                        "publish, so a value here was measured by the run named below. "
                        "The site renders null as 'not measured yet'.")
    site["_run"] = args.run

    # Reset first, then fill. Nothing survives a publish it was not measured in.
    for key in MEASURED:
        site["training"][key] = None

    tb = training_block(recs, meta)
    site["training"].update(tb)
    curve = val_curve(recs)
    # min() over a list containing NaN returns whatever the order dictates, so only
    # finite points are eligible for "best".
    finite = [p for p in curve if isinstance(p["loss"], (int, float)) and math.isfinite(p["loss"])]
    if curve:
        site["training"]["val_curve"] = curve
    if finite:
        # The best point's own ppl, which val_curve nulls where train.py capped it.
        best = min(finite, key=lambda p: p["loss"])
        site["training"]["best_val_loss"] = best["loss"]
        site["training"]["best_val_ppl"] = best["ppl"]

    # Once the run is complete, only files made from its final weights may reach the page;
    # before then, only files from a step count the log has reached.
    site["benchmarks"], published = bench_block(eval_path, site["benchmarks"], args.run,
                                                tb["steps"], tb["run_complete"])
    # Replaced whole on every publish, like every other measured field.
    samples_path = args.samples or os.path.join(ROOT, "runs", args.run, "samples.json")
    site["samples"], sampled = samples_block(samples_path, args.run, tb["steps"],
                                             tb["run_complete"])

    hits = []
    site = finite_or_none(site, hits)
    site["training"]["nonfinite_values"] = len(hits)

    # allow_nan=False is the backstop: if anything non-finite slipped past the pass above,
    # fail here rather than write a file the browser cannot read. Serialised in full before
    # the file is touched, so that failure leaves the previous results.json intact.
    write_atomic(args.out, json.dumps(site, indent=2, allow_nan=False) + "\n")

    # Reporting happens last and guards every optional value, so a degenerate log can
    # never crash after the file has already been rewritten.
    def num(v, digits=0):
        return "not measured" if v is None else f"{v:,.{digits}f}"

    print(f"-> {rel_or_abs(args.out, ROOT)}   (run {args.run})")
    pct = ("" if not (tb["tokens_target"] and tb["tokens_seen"])
           else f" ({100 * tb['tokens_seen'] / tb['tokens_target']:.1f}% of target)")
    # The unit goes on a number only: num() of a missing elapsed_s printed "not measuredh".
    hours = "not measured" if tb["wall_clock_hours"] is None else f"{tb['wall_clock_hours']:.2f}h"
    print(f"   training   step {num(tb['steps'])}  {num(tb['tokens_seen'])} tokens{pct}  {hours}")
    print(f"   throughput {num(tb['tokens_per_second'])} tok/s (median)")
    if tb.get("rest_hours"):
        # "idle", not "cooling": train.py's rest_s also holds the deliberate load-cap idle,
        # and at --duty 0.65 that is most of it. The number is the same either way.
        print(f"   idle       {tb['rest_hours']:.2f}h of the wall clock was idle "
              f"(cooling pauses and the GPU load cap)")
    print(f"   complete   {tb['run_complete']}"
          + ("" if tb["tokens_target"] else "  (no tokens_target in the log to compare against)"))
    if finite:
        b = min(finite, key=lambda p: p["loss"])
        print(f"   val        {len(curve)} point(s), best {b['loss']:.4f} "
              f"(ppl {num(b['ppl'], 1)}) at step {b['step']:,}")
    else:
        print("   val        no finite validation points yet")
    if hits:
        print(f"   WARNING    {len(hits)} non-finite value(s) (NaN/inf) written as null -- "
              f"a NaN loss usually means the run diverged")
    # "refused", not "nothing generated yet", when a file was there and turned away: the
    # old wording told the operator no samples existed while a refusal sat ten lines up.
    n_samples = len(site["samples"]["items"])
    print("   samples    " + (f"{n_samples} published" if sampled
                              else "refused (see above)" if os.path.exists(samples_path)
                              else "none -- nothing generated yet"))
    steps_scored = site["benchmarks"].get("_steps_completed")
    print("   benchmarks " + (f"published (weights after {steps_scored:,} steps)"
                              if published and isinstance(steps_scored, int)
                              else "published" if published
                              else "refused (see above)" if os.path.exists(eval_path)
                              else "left null -- nothing measured yet"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
