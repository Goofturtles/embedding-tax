"""Run bookkeeping shared by eval_harness.py, sample.py and publish_results.py.

Each rule here used to be written out in two places, and the copies had already drifted:
eval_harness.py called a run finished when its token target was smaller than one step, which
publish_results.py never did. One copy each now. The module imports no torch, so
publish_results.py can use it without loading the training stack.

Every function that looks under runs/ takes the project root as an argument instead of
keeping a global of its own: each caller has its own ROOT (the tests point it at a scratch
tree), and the rules have to follow whichever root the caller is using. train.py keeps its own
copy of the atomic writer for now, because another workflow owns that file.
"""
import json
import os
import tempfile
import time


def rel_or_abs(path, root):
    """Path relative to the project, or absolute when that is impossible. On Windows
    os.path.relpath raises for a path on another drive, which used to crash eval_harness.py
    AFTER a full evaluation had run and throw its results away."""
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return os.path.abspath(path)


def write_atomic(path, text, attempts=4):
    """Write a sibling temp file and rename it over path. Opening the file with "w" emptied
    it before a byte was serialised, so a failed dump (allow_nan, Ctrl-C, a full disk) left
    a truncated file that the site and the next publish both choke on, and an interrupted
    eval threw away hours of scoring. The rename is retried as train.py's is: NTFS refuses
    it while anything holds the destination open.

    Each writer gets its own temp name. With one fixed path + ".tmp", two overlapping
    publishes shared it: the second truncated the first's finished temp or renamed it away,
    and the first's retry then installed a partial file or crashed."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                               prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        # newline="\n": Windows would otherwise write CRLF into files that get committed.
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        for i in range(attempts):
            try:
                os.replace(tmp, path)
                return
            except PermissionError as e:
                if i == attempts - 1:
                    # The temp holds a complete, good file (hours of scoring, for an eval).
                    # Keep it and say where it is, rather than delete it with the others.
                    raise PermissionError(
                        f"{path} is held open by another program; the finished file was "
                        f"kept at {tmp}. Move it over {path} once that program lets go.") from e
                time.sleep(0.5 * (i + 1))
    except PermissionError:
        raise
    except BaseException:
        # path is untouched either way; don't leave a stale part-written copy beside it.
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def read_log(path):
    """Return (records, torn) -- torn counts lines a kill left half-written."""
    recs, torn = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                torn += 1
    return recs, torn


def step_target(last):
    """The token count a run is complete at, read off its last log record, or None when the
    record cannot say.

    tokens_target is the requested --tokens, but train.py only trains whole steps:
    total_steps = tokens // tokens_per_step, so 1,900,000,000 requested is
    14,495 x 131,072 = 1,899,888,640 actually trained. Comparing against the raw request
    meant a run that finished normally could never read as complete. tokens is always
    (step + 1) x tokens_per_step in the log, resumed or not, so the per-step size is
    recoverable from the last record."""
    requested, seen = last.get("tokens_target"), last.get("tokens")
    steps = last.get("step", -1) + 1
    if not (requested and seen and steps > 0 and seen % steps == 0):
        return None
    per_step = seen // steps
    return (requested // per_step) * per_step


def run_complete(last):
    """True once the last log record reaches its step-aligned target. A request smaller than
    one step aligns to a target of 0, which is never complete."""
    target = step_target(last)
    return bool(target and last["tokens"] >= target)


def run_dir_name(root, path):
    """The run directory `path` names, spelled as it is on disk, or None unless it resolves
    to a directory directly inside runs/. A relative path is taken from runs/, so R1, r1/,
    .\\r1 and ../runs/r1 all name run r1; an absolute path is used as it is.

    realpath returns the on-disk case on Windows and normcase makes the comparison
    Windows-safe. Compared as typed, --run R1 found runs/r1's files and then refused them
    for belonging to "r1"."""
    runs = os.path.realpath(os.path.join(root, "runs"))
    d = os.path.realpath(os.path.join(runs, path))
    if os.path.normcase(os.path.dirname(d)) != os.path.normcase(runs):
        return None
    return os.path.basename(d)
