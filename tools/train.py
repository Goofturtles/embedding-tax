"""Train the 49,296,896-parameter model from scratch on one RTX 4090.

Design notes worth knowing before changing anything:

* **Batch 16 is not arbitrary.** Measured on this card: batch 16 gives 107,145 tok/s at
  11 GB. Batch 24 drops to 94,334, batch 32 collapses to 1,726 tok/s at 21.3 GB and
  batch 48 asks for 31.6 GB on a 24 GB card. Raise the EFFECTIVE batch with gradient
  accumulation, never the micro batch.
* **Validation is a held-out shard**, not a slice of training data, so `val_loss` is a
  real number rather than a restatement of `train_loss`. It is also scored on the SAME
  fixed sample every time (see `evaluate`), so a change in val_loss is a change in the
  model and not a change in the draw.
* **Every metric written to runs/<id>/log.jsonl is measured**, including the ones that
  look bad. Nothing here estimates.

Run:
    .venv\\Scripts\\python.exe tools/train.py --run r1
    .venv\\Scripts\\python.exe tools/train.py --run r1 --resume
"""
import argparse
import contextlib
import ctypes
import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import GPTConfig, analytic_param_count  # noqa: E402
from model.gpt import GPT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RUNS = os.path.join(ROOT, "runs")
CAP = 50_000_000

# Hyperparameters that change what a step means, or what a val_loss means. Resuming
# with any of these altered would splice two different runs into one curve.
# Compared on the intersection of keys (see main): a checkpoint written before this
# tuple grew must still be resumable, or adding an entry here silently strands every
# checkpoint already on disk.
RUN_SHAPE = ("batch", "accum", "tokens", "lr", "warmup", "eval_batches")

# Seed of the end-of-run estimate. Deliberately not evaluate's default 99, so the larger
# final sample is an independent draw rather than a superset of the curve's (see the end
# of train_run). Named once because the checkpoint records it and the summary prints it.
FINAL_SEED = 7

# Resume liveness guard (see check_not_live). A trainer appends to log.jsonl every step,
# seconds apart even through a cooling pause, so a log written within ten minutes means
# one may still be running, and ten minutes of silence means none is.
LOCK_NAME = "train.lock"
# A file of this name in the run directory asks the trainer to stop cleanly: it saves a
# checkpoint at the end of the current step and exits, so a watchdog can take the GPU
# off the job in about two seconds without losing a step. The trainer removes the file
# once it has honoured it.
STOP_NAME = "STOP"
LIVE_WINDOW_S = 600


class Shards:
    """Flat uint16 token shards, memory-mapped and sampled at random offsets.

    Memmap rather than loading: the corpus is 4 GB and the OS page cache handles the
    working set better than we would.
    """

    def __init__(self, pattern):
        self.paths = sorted(glob.glob(os.path.join(DATA, pattern)))
        if not self.paths:
            raise SystemExit(f"no shards matching {pattern} in {DATA}; run tools/prepare_data.py")
        self.maps = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.lengths = np.array([len(m) for m in self.maps], dtype=np.int64)
        self.total = int(self.lengths.sum())
        # Pick a shard in proportion to its length. Today every shard is exactly 100M
        # tokens so this is uniform anyway; it stops a short tail shard from being
        # oversampled by orders of magnitude if the corpus is ever rebuilt unevenly.
        self.weights = self.lengths / self.lengths.sum()

    def batch(self, rng, batch_size, block, device):
        x = np.empty((batch_size, block), dtype=np.int64)
        y = np.empty((batch_size, block), dtype=np.int64)
        for i in range(batch_size):
            s = int(rng.choice(len(self.maps), p=self.weights))
            m, n = self.maps[s], int(self.lengths[s])
            j = int(rng.integers(0, n - block - 1))
            chunk = m[j:j + block + 1].astype(np.int64)
            x[i] = chunk[:-1]
            y[i] = chunk[1:]
        return (torch.from_numpy(x).pin_memory().to(device, non_blocking=True),
                torch.from_numpy(y).pin_memory().to(device, non_blocking=True))


def lr_at(step, total, base, warmup, floor_frac=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    if step >= total:
        return base * floor_frac
    t = (step - warmup) / max(1, total - warmup)
    return base * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def replace_with_retry(tmp, path, attempts=4):
    """os.replace, retried. On NTFS the rename fails with PermissionError while anything
    holds a handle to the destination -- most often a virus scanner opening the file we
    just closed. Unretried, that ends a 5-hour run at a save boundary."""
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.5 * (i + 1))


def append_with_retry(path, text, attempts=4):
    """Append to a file, retried the same way. The per-step log append is the most
    frequent open in a run, so a scanner or an editor holding log.jsonl for a moment is
    likeliest to land here, and one PermissionError would end the run between saves."""
    for i in range(attempts):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.5 * (i + 1))


def save_atomic(path, payload):
    """Write to a sibling .tmp and rename.

    torch.save straight onto ckpt.pt means a kill during the ~590 MB serialization
    leaves a truncated file where the only recovery point used to be.
    """
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    replace_with_retry(tmp, path)


def rewrite_log(log_path, keep_before):
    """Drop log lines at or past a resume point, atomically.

    log.jsonl is the one artifact here that cannot be regenerated -- the checkpoint can
    be retrained, the curve cannot -- so it gets the same tmp+rename treatment as the
    weights. Unparseable lines are dropped rather than raised on: an unclean kill can
    leave a half-written final line, which is exactly the case resume exists to handle.
    """
    if not os.path.exists(log_path):
        return
    kept, dropped = [], 0
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                if json.loads(line)["step"] < keep_before:
                    kept.append(line)
            except (json.JSONDecodeError, KeyError):
                dropped += 1
    tmp = log_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(kept)
    replace_with_retry(tmp, log_path)
    if dropped:
        print(f"dropped {dropped} unparseable log line(s) from a previous kill")


def duty_idle(dt, duty):
    """Seconds to idle after dt seconds of compute, so the GPU is busy for about `duty` of
    the wall clock: dt / (dt + idle) == duty. The loop calls it once per micro batch (see
    duty_pause) and eta_seconds once per whole step; being linear in dt, the idles of a
    step's slices add up to the idle of the whole step.

    Proportional to the compute rather than a fixed sleep, so the share holds when a step
    runs long (a throttled card, a slow batch) instead of drifting. dt comes from
    perf_counter, which cannot step backwards, but a negative length would make time.sleep
    raise and end the run, so it still idles zero rather than trust every caller.
    """
    return max(0.0, dt * (1.0 / duty - 1.0))


def duty_pause(slice_t0, duty):
    """End one slice of compute under --duty: wait for the GPU to finish it, then idle in
    proportion. Returns (seconds idled, as measured; the moment the next slice starts).

    Called after each micro batch, the last slice ending with the optimizer step, so at
    --duty 0.65 a step of ~1.2 s idles eight times for ~0.08 s rather than once for
    ~0.65 s. A load meter that samples about once a second averages the short idles into a
    steady ~65%; the single long idle read as swings between 100% and 40%. The synchronize
    comes first because kernels launch asynchronously: without it the clock would time how
    fast the CPU queued the slice, not how long the GPU spent running it.
    """
    torch.cuda.synchronize()
    busy_until = time.perf_counter()
    time.sleep(duty_idle(busy_until - slice_t0, duty))
    resumed = time.perf_counter()
    return resumed - busy_until, resumed


def duty_pattern(duty, accum):
    """The --duty line printed at startup: the idle per second of compute, and that it
    comes as one short idle per micro batch rather than one long idle per step."""
    return (f"{duty:g}: idles ~{duty_idle(1.0, duty):.2f} s after every 1 s of compute, "
            f"per micro batch ({accum} short idles a step)")


def eta_seconds(step, total_steps, dt, duty, rest_every, rest_seconds):
    """Wall-clock seconds left once `step` has computed for dt seconds, for the console.

    dt is compute time only, so an ETA built from dt alone runs short by every duty idle
    and cooling pause (at --duty 0.65 it printed 0.65 of the real time). This counts what
    the loop will actually do under the same conditions it checks: this step's cooling
    pause if one is due, then each later step's compute, duty idle and pause. The duty
    idle is taken inside each step (one short idle per micro batch, adding up to
    duty_idle(dt)), so this step's is already spent and the last step's is still to come;
    no pause follows the last step. Evals and saves are still left out, so it stays
    slightly optimistic.
    """
    left = total_steps - 1 - step
    pauses = ((total_steps - 1) // rest_every - step // rest_every) if rest_every else 0
    return left * (dt + duty_idle(dt, duty)) + pauses * rest_seconds


def cfg_diff(old, new):
    """Model-config fields that differ between a checkpoint and the code, {key: (was, now)}.

    Every key is compared, not the intersection the RUN_SHAPE check uses: a config field
    the checkpoint lacks is a change to the model, not a new bookkeeping entry. block_size
    is the one that matters most. It sets tokens per step but is not in RUN_SHAPE, and it
    leaves no trace in the weights (the rotary tables are non-persistent buffers), so an
    edited config.py would otherwise load cleanly and quietly shift the LR schedule.
    """
    return {k: (old.get(k), new.get(k)) for k in sorted(set(old) | set(new))
            if old.get(k) != new.get(k)}


def pid_alive(pid):
    """True if a process with this PID is still running.

    OpenProcess + GetExitCodeProcess, because the POSIX probe os.kill(pid, 0) is not a
    probe on Windows: signal 0 there is CTRL_C_EVENT, and it would interrupt the very run
    it was checking. Any failure (no such PID, access denied, not Windows) reads as not
    alive; the log-age check in check_not_live still stands behind it.
    """
    still_active = 259
    query_limited_information = 0x1000
    try:
        k32 = ctypes.WinDLL("kernel32")
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        k32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
        k32.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = k32.OpenProcess(query_limited_information, 0, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_uint32()
            ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == still_active
        finally:
            k32.CloseHandle(handle)
    except Exception:  # noqa: BLE001
        return False


def read_lock_pid(lock_path):
    """PID recorded in a lock file, or None if there is no lock or it does not parse."""
    try:
        with open(lock_path, encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def check_not_live(run_dir, force):
    """Refuse to start while another process may still be training this run.

    A resume rewrites log.jsonl to drop lines past the checkpoint. Against a LIVE trainer
    that deletes the lines it has logged since its last save, and from then on two
    processes append to one log and overwrite one checkpoint. Two signals, because each
    misses a case: the lock names the PID of a trainer started by this code, and the log's
    age catches a trainer started by older code that writes no lock -- which is exactly
    the run that is live as this is written. --force skips both, for an operator who has
    already confirmed the old process is gone.
    """
    if force:
        print("--force: skipping the check for a live trainer on this run")
        return
    lock_path = os.path.join(run_dir, LOCK_NAME)
    pid = read_lock_pid(lock_path)
    # Our own PID in the lock can only be a dead trainer's PID that Windows has since
    # handed to us, so it is stale rather than a second trainer.
    if pid is not None and pid != os.getpid() and pid_alive(pid):
        raise SystemExit(
            f"{lock_path} names PID {pid}, which is still running, so this run is probably "
            f"being trained right now. Stop that process first. If PID {pid} is no longer "
            f"the trainer (Windows reuses PIDs), pass --force.")
    log_path = os.path.join(run_dir, "log.jsonl")
    if os.path.exists(log_path):
        age = time.time() - os.path.getmtime(log_path)
        if age < LIVE_WINDOW_S:
            raise SystemExit(
                f"{log_path} was written {age:.0f}s ago, so a trainer may still be running "
                f"(one started before train.lock existed leaves no lock). Stop it and wait "
                f"{LIVE_WINDOW_S // 60} minutes, or pass --force once you have confirmed it "
                f"is gone.")


@contextlib.contextmanager
def run_lock(run_dir, force):
    """Hold runs/<run>/train.lock, naming this PID, for the life of the run.

    Written tmp+rename so a reader never sees a half-written PID. A process that dies
    without cleaning up leaves a lock naming a dead PID, which check_not_live treats as
    stale, so it is simply replaced. On the way out the lock is removed only if it still
    names us: after a --force takeover it belongs to the new trainer.
    """
    check_not_live(run_dir, force)
    lock_path = os.path.join(run_dir, LOCK_NAME)
    tmp = lock_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    replace_with_retry(tmp, lock_path)
    try:
        yield
    finally:
        if read_lock_pid(lock_path) == os.getpid():
            try:
                os.remove(lock_path)
            except OSError:
                # Once we exit, a lock left behind names a dead PID and the next start
                # replaces it; raising here would only hide the run's own outcome.
                pass


@torch.no_grad()
def evaluate(model, shards, cfg, device, batches, batch_size, seed=99):
    """Score the SAME held-out sample every call.

    The rng is built here rather than carried across calls on purpose: a long-lived
    generator draws a different sample each eval, so consecutive val_loss values differ
    by both the model improving and the draw changing, and a "best" val can just mean an
    easier draw. A fixed seed makes the deltas attributable.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = shards.batch(rng, batch_size, cfg.block_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r1")
    ap.add_argument("--batch", type=int, default=16, help="micro batch; 16 is the measured optimum")
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--tokens", type=int, default=1_900_000_000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--final-batches", type=int, default=320,
                    help="batches for the end-of-run estimate; larger sample, same estimator")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--rest-every", type=int, default=0,
                    help="pause the GPU after every N steps to let it cool (0 = never)")
    ap.add_argument("--rest-seconds", type=float, default=20.0,
                    help="length of each cooling pause")
    ap.add_argument("--duty", type=float, default=1.0,
                    help="share of wall-clock time the GPU computes, 0.1 to 1.0; after each "
                         "micro batch it idles in proportion, to cut heat and power "
                         "(1.0 = off)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="skip the check for a live trainer on this run; only once you have "
                         "confirmed the old process is gone")
    args = ap.parse_args(argv)
    # Written as "not inside the range" so NaN, which compares false to everything, is
    # refused too. Below 0.1 the idle is over nine times the compute: a 5-hour run would
    # take two days.
    if not 0.1 <= args.duty <= 1.0:
        raise SystemExit(f"--duty must be between 0.1 and 1.0 (got {args.duty}); 1.0 is full "
                         f"speed, 0.65 computes about 65% of the time.")
    return args


def main():
    args = parse_args()
    run_dir = os.path.join(RUNS, args.run)
    os.makedirs(run_dir, exist_ok=True)
    # Taken before anything in the run directory is read or rewritten, and held until
    # train_run returns or raises.
    with run_lock(run_dir, args.force):
        return train_run(args, run_dir)


def train_run(args, run_dir):
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    device = "cuda"
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = GPTConfig()
    budget = analytic_param_count(cfg)
    if budget["total"] > CAP:
        raise SystemExit(f"config is over the cap: {budget['total']:,} > {CAP:,}")

    log_path = os.path.join(run_dir, "log.jsonl")
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    best_path = os.path.join(run_dir, "best.pt")
    stop_path = os.path.join(run_dir, STOP_NAME)
    # A request left over from an earlier stop would halt this run on its first step. Only
    # a request OLDER than the checkpoint is stale: a stop that was honoured saved the
    # checkpoint after it. A newer one arrived in the seconds between the old process
    # leaving and this one starting, and is kept for step one.
    if os.path.exists(stop_path):
        try:
            stale = (not os.path.exists(ckpt_path)
                     or os.path.getmtime(stop_path) < os.path.getmtime(ckpt_path))
            if stale:
                os.remove(stop_path)
                print(f"note: removed a stale {STOP_NAME} file before starting")
            else:
                print(f"note: a fresh {STOP_NAME} request is waiting; this run will stop "
                      f"after its first step")
        except OSError as e:
            print(f"warning: could not handle {stop_path}: {e}")
    # Relaunching a run without --resume is the easiest way to lose it: a second curve from
    # step 0 is appended to log.jsonl, best.pt is overwritten at step 250 and ckpt.pt at
    # step 500. So a fresh start refuses a directory that already holds either.
    if not args.resume and (os.path.exists(log_path) or os.path.exists(ckpt_path)):
        raise SystemExit(f"{run_dir} already has a log or checkpoint. Pass --resume to "
                         f"continue it, or choose a new --run name.")

    train = Shards("train_*.bin")
    val = Shards("val_*.bin")

    tokens_per_step = args.batch * cfg.block_size * args.accum
    total_steps = args.tokens // tokens_per_step

    model = GPT(cfg).to(device)
    n_params = model.num_params()
    assert n_params == budget["total"], f"live {n_params:,} != analytic {budget['total']:,}"

    # dim < 2 is the norms; everything else is a matrix and gets decay. The token
    # embedding is included deliberately: it is tied to lm_head, so it IS the output
    # projection, and decaying the output projection is what nanoGPT does to reproduce
    # GPT-2. Excluding it here would be a departure from the reference recipe, not a fix.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    hp = {k: getattr(args, k) for k in RUN_SHAPE}
    start_step, prior_elapsed, best_val = 0, 0.0, float("inf")
    rest_total = 0.0         # seconds idle: duty idle + cooling pauses, carried across resumes
    if args.resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        old = state.get("hp")
        if old is not None:
            # Intersection, so a checkpoint predating a new RUN_SHAPE entry still loads.
            shared = set(old) & set(hp)
            diff = {k: (old[k], hp[k]) for k in shared if old[k] != hp[k]}
            if diff:
                raise SystemExit(
                    f"cannot resume: the checkpoint was trained with different settings "
                    f"{diff} (was, now). total_steps would change and the LR schedule "
                    f"would jump.")
            missing = sorted(set(hp) - shared)
            if missing:
                print(f"note: checkpoint predates {missing}; those cannot be verified")
        else:
            print("note: checkpoint stores no hyperparameters; resuming unverified")
        # The model config, compared in full (see cfg_diff), before the weights load, so a
        # refusal names the edited field instead of surfacing as a shape-mismatch traceback.
        old_cfg = state.get("cfg")
        if old_cfg is not None:
            changed = cfg_diff(old_cfg, cfg.__dict__)
            if changed:
                raise SystemExit(
                    f"cannot resume: model/config.py differs from the checkpoint's config "
                    f"{changed} (was, now). block_size changes tokens per step, so even an "
                    f"edit that loads cleanly would shift the schedule.")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        prior_elapsed = state.get("elapsed_s", 0.0)
        rest_total = state.get("rest_s", 0.0)
        best_val = state.get("best_val", float("inf"))
        # best.pt is written every eval but best_val is only persisted every save, so
        # after a crash the file on disk can be better than the number we just restored.
        # Trusting the number alone would let the next eval overwrite a better model.
        if os.path.exists(best_path):
            bv = torch.load(best_path, map_location="cpu").get("val_loss")
            if bv is not None and bv < best_val:
                print(f"best.pt holds a better val ({bv:.4f} < {best_val:.4f}); keeping it")
                best_val = bv
        # Drop log lines at or past the resume point: the crashed process may have
        # logged steps whose optimizer state never reached the checkpoint.
        rewrite_log(log_path, start_step)
        print(f"resumed from step {start_step:,} ({prior_elapsed/3600:.2f}h already spent)")
    elif args.resume:
        # --resume with no checkpoint means the crash landed before the first save.
        # Appending to the old log would leave duplicate step numbers in the curve.
        print(f"--resume given but no checkpoint at {ckpt_path}; starting from step 0")
        rewrite_log(log_path, 0)

    # Snapshot the corpus description into the run. publish_results.py reads the corpus
    # from here: data/meta.json describes whatever data/ holds now, and a rebuild between
    # training and publishing would otherwise put the new corpus on the old run. Decided
    # here, after the resume logic, on where training actually starts: from step 0 (a fresh
    # start, or a --resume that found no checkpoint and so retrains on today's data) takes
    # a new snapshot; a real resume keeps the one the run began with.
    run_meta = os.path.join(run_dir, "meta.json")
    data_meta = os.path.join(DATA, "meta.json")
    if os.path.exists(data_meta) and (start_step == 0 or not os.path.exists(run_meta)):
        shutil.copyfile(data_meta, run_meta)

    print(f"params          {n_params:,}  (cap {CAP:,}, {CAP - n_params:,} spare)")
    print(f"train tokens    {train.total:,} across {len(train.paths)} shards")
    print(f"val tokens      {val.total:,} held out")
    print(f"tokens/step     {tokens_per_step:,}  (micro {args.batch} x accum {args.accum})")
    print(f"total steps     {total_steps:,}  for {args.tokens:,} tokens")
    if args.rest_every:
        print(f"cooling         {args.rest_seconds:g}s pause every {args.rest_every} steps")
    if args.duty < 1.0:
        print(f"duty            {duty_pattern(args.duty, args.accum)}")
    print()

    rng = np.random.default_rng(1234 + start_step)
    model.train()
    t0 = time.time()
    tokens_done = start_step * tokens_per_step
    last_val = None          # val of the most recent in-loop eval
    stop_requested = False   # set by a STOP file (see STOP_NAME); the loop then breaks

    def elapsed():
        return prior_elapsed + (time.time() - t0)

    for step in range(start_step, total_steps):
        lr = lr_at(step, total_steps, args.lr, args.warmup)
        for group in opt.param_groups:
            group["lr"] = lr

        # perf_counter, not time.time: a Windows clock sync can step the wall clock by
        # seconds mid-step, which would inflate dt, the logged tok/s and the ETA.
        # perf_counter only moves forward.
        step_t0 = time.perf_counter()
        slice_t0, idle = step_t0, 0.0    # start of the current duty slice; idle so far
        opt.zero_grad(set_to_none=True)
        # Accumulate on the GPU and read once. A .item() inside the loop is a sync per
        # micro batch, which stops the CPU from ever running ahead of the GPU.
        loss_sum = torch.zeros((), device=device)
        for micro in range(args.accum):
            x, y = train.batch(rng, args.batch, cfg.block_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / args.accum).backward()
            loss_sum += loss.detach()
            # Duty cap, spread across the step: a short idle after each micro batch instead
            # of one long one after the step, so a once-a-second load meter reads a steady
            # --duty (see duty_pause). That does sync once per micro batch, which the note
            # above avoids, but the GPU is about to idle anyway. At --duty 1.0 neither the
            # sync nor the sleep runs and the step is exactly what it was. The last micro
            # batch is skipped here because its slice ends with the optimizer step, below.
            if args.duty < 1.0 and micro < args.accum - 1:
                took, slice_t0 = duty_pause(slice_t0, args.duty)
                idle += took
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if args.duty < 1.0:
            idle += duty_pause(slice_t0, args.duty)[0]

        torch.cuda.synchronize()
        # Compute only: the idle taken inside the step is subtracted, so tok/s measures the
        # GPU's work and eta_seconds adds the idle back itself. The idle goes to rest_total
        # like a cooling pause, before the log line, so elapsed_s - rest_s stays compute
        # time on every line. Unlike the cooling pause it runs on the last step too: that
        # costs under a second once, and every step keeps the same load pattern.
        dt = time.perf_counter() - step_t0 - idle
        rest_total += idle
        tokens_done += tokens_per_step
        train_loss = loss_sum.item() / args.accum

        if step % 10 == 0:
            tps = tokens_per_step / dt
            eta = eta_seconds(step, total_steps, dt, args.duty, args.rest_every,
                              args.rest_seconds) / 3600
            print(f"step {step:>6}/{total_steps}  loss {train_loss:.4f}  lr {lr:.2e}  "
                  f"{tps:>7,.0f} tok/s  {tokens_done/1e9:.3f}B  eta {eta:.1f}h")

        # tokens_target rides along on every record so the site can render progress
        # without anyone typing the denominator into a stylesheet or a script. It is the
        # run's own launch parameter, not the corpus size -- those are equal today and
        # would quietly diverge the first time anyone trains on a subset.
        rec = {"step": step, "train_loss": round(train_loss, 5), "lr": lr,
               "tokens": tokens_done, "tokens_target": args.tokens,
               "tok_per_s": round(tokens_per_step / dt, 1),
               "grad_norm": round(float(grad_norm), 4),
               "elapsed_s": round(elapsed(), 1),
               # cumulative idle (duty idle + cooling pauses): compute = elapsed_s - rest_s
               "rest_s": round(rest_total, 1)}

        # On the schedule AND on the very last step. Without the second clause the final
        # eval_every - 1 steps (245 on this config) never reach the curve or best.pt.
        if (step > 0 and step % args.eval_every == 0) or step == total_steps - 1:
            v = evaluate(model, val, cfg, device, args.eval_batches, args.batch)
            last_val = v
            rec["val_loss"] = round(v, 5)
            rec["val_ppl"] = round(math.exp(min(20, v)), 3)
            improved = v < best_val
            best_val = min(best_val, v)
            print(f"  -- val {v:.4f}  ppl {math.exp(min(20, v)):.2f}  (best {best_val:.4f})")
            if improved:
                # Model only: best.pt is for evaluating, not for resuming, and leaving
                # the optimizer state out keeps it a third of the size.
                save_atomic(best_path, {"model": model.state_dict(), "cfg": cfg.__dict__,
                                        "step": step, "val_loss": v})

        append_with_retry(log_path, json.dumps(rec) + "\n")

        saved_this_step = step > 0 and step % args.save_every == 0
        if saved_this_step:
            save_atomic(ckpt_path, {"model": model.state_dict(), "opt": opt.state_dict(),
                                    "step": step + 1, "cfg": cfg.__dict__, "hp": hp,
                                    "elapsed_s": elapsed(), "best_val": best_val,
                                    "rest_s": rest_total})

        # A clean stop request (see STOP_NAME), checked once the step is logged and saved
        # and before any idle, so the answer is a checkpoint two seconds later rather
        # than a kill that loses up to save_every steps. Ignored on the last step, whose
        # normal ending is already a save.
        if step < total_steps - 1 and os.path.exists(stop_path):
            if not saved_this_step:
                save_atomic(ckpt_path, {"model": model.state_dict(), "opt": opt.state_dict(),
                                        "step": step + 1, "cfg": cfg.__dict__, "hp": hp,
                                        "elapsed_s": elapsed(), "best_val": best_val,
                                        "rest_s": rest_total})
            try:
                os.remove(stop_path)
            except OSError as e:
                print(f"warning: could not remove {stop_path}: {e}")
            # --force as well: the liveness guard refuses a plain --resume until the log
            # has been quiet for ten minutes, and the operator has just watched this exit.
            print(f"\nstopped on request after step {step:,} ({step + 1:,}/{total_steps:,} "
                  f"done): checkpoint saved, resume with --resume --force to continue")
            stop_requested = True
            break

        # Cooling pause, taken after the step is logged and saved. tok/s is timed inside
        # the step so it is unaffected; elapsed_s includes the pause and rest_s records it,
        # so compute time and wall-clock time can both be reported without either being
        # flattered. The last step never pauses -- there is nothing left to cool for.
        # Timed with perf_counter, like the duty idle: a Windows clock sync that steps the
        # wall clock during the sleep would otherwise be booked as pause time, and a
        # backward step would make rest_s go down.
        if args.rest_every and (step + 1) % args.rest_every == 0 and step < total_steps - 1:
            r0 = time.perf_counter()
            time.sleep(args.rest_seconds)
            rest_total += time.perf_counter() - r0

    # A requested stop already saved its checkpoint and said so; the end-of-run
    # evaluation and final save below belong only to a finished run. SystemExit rather
    # than return: this span is lifted and run at top level by the CPU test harnesses.
    if stop_requested:
        raise SystemExit(0)

    # Two numbers, and they answer different questions. curve_val uses the curve's own
    # fixed sample, so it is the last point on the same axis as every logged val_loss.
    # final_val draws a larger sample with a DIFFERENT seed, so it is an independent
    # estimate rather than a superset of the same draw -- reporting a bigger sample from
    # seed 99 as a cross-check would be circular, since its first 40 batches are the
    # curve's 40.
    # The last step is always evaluated inside the loop, so its point is already on the
    # curve and best.pt already reflects the finished weights. Only a run resumed exactly
    # at its end (zero loop iterations) needs a fresh evaluation here.
    curve_val = (last_val if last_val is not None
                 else evaluate(model, val, cfg, device, args.eval_batches, args.batch))
    final_val = evaluate(model, val, cfg, device, args.final_batches, args.batch,
                         seed=FINAL_SEED)
    # best_val stays the best value that an eval actually wrote best.pt for. Folding
    # curve_val in here would print a "best" that no file on disk contains.
    save_atomic(ckpt_path, {"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": total_steps, "cfg": cfg.__dict__, "hp": hp,
                            "elapsed_s": elapsed(), "best_val": best_val,
                            "final_val": final_val, "final_batches": args.final_batches,
                            "final_seed": FINAL_SEED, "curve_val": curve_val,
                            "rest_s": rest_total})
    hours = elapsed() / 3600
    # "idle", not "cooling pauses": at --duty 0.65 most of rest_total is the duty idle taken
    # inside the steps and only the remainder is cooling pauses, so naming either alone would
    # misreport the other. The number is unchanged: wall clock minus it is the compute time.
    print(f"\ndone in {hours:.2f}h ({rest_total / 3600:.2f}h of it idle: duty idle "
          f"plus cooling pauses)")
    print(f"  last point on the curve  ({args.eval_batches} batches, seed 99): "
          f"{curve_val:.4f}  ppl {math.exp(min(20, curve_val)):.2f}")
    print(f"  independent estimate     ({args.final_batches} batches, seed {FINAL_SEED}):  "
          f"{final_val:.4f}  ppl {math.exp(min(20, final_val)):.2f}")
    best_str = f"{best_val:.4f}" if best_val < float("inf") else "n/a (no eval ran)"
    print(f"  best val written to best.pt: {best_str}")
    print(f"checkpoint -> {ckpt_path}")
    print(f"best       -> {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
