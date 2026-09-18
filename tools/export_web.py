"""Export a checkpoint for the site's in-browser playground (site/infer.js + onnxruntime-web).

The visitor downloads the weights once and generates text on their own device, so the
export has three jobs: be small enough for GitHub Pages (hard limit 100 MB per file),
compute the same thing PyTorch computes, and say exactly what it is. Everything here is
CPU-only on purpose: a training run may be live on the GPU, and ONNX export is CPU work.

What comes out (the contract site/infer.js is written against; do not rename):
    <out>/manifest.json     provenance + sizes + parity numbers, read by the page (never typed)
    <out>/<file>            the ONNX graph the manifest's "file" names (the quantised one)
    <out>/tokenizer.json    a byte copy of data/tokenizer.json
The fp32 graph and every quantisation candidate land in --work, which defaults to the
system temp directory (<temp>/embedding-tax/export_web/<run>/), never under the project:
the fp32 file is ~200 MB, over GitHub's 100 MiB hard limit, and a `git add .` from the
project root must not be able to pick it up. The candidates that do not ship are deleted
once the pick is made (every run rebuilds them); the fp32 graph stays in temp for the record.

The graph: input "input_ids" int64 [1, T] (T dynamic, 1..block_size), output "logits"
float32 [1, vocab] for the LAST position only. No KV cache: the browser recomputes the
whole prefix per token, which at 49M parameters is a fraction of a second on wasm and
milliseconds on WebGPU. The tied embedding is one initializer used by both the input
Gather and the output head (head = MatMul(W, x^T)^T, a GEMV), so it is stored once.

Quantisation candidates, all measured, one shipped (--quant auto picks; see choose()):
    dynamic-int8        onnxruntime quantize_dynamic on the 4 x n_layer transformer MatMuls
                        (52 for r1; int8 weights, activations quantised to uint8 at run time).
                        Fastest on wasm; MatMulInteger has no WebGPU kernel.
    dynamic-int8-gather the same plus ORT's Gather quantisation, to measure what that
                        does to the tied matrix (it adds a uint8 copy: the head still
                        needs the fp32 one).
    qdq-int8            weight-only: per-channel int8 + DequantizeLinear in front of each
                        transformer MatMul, activations fp32. Embedding/head fp32.
    qdq-int8-embed      qdq-int8 plus the tied matrix as int8 with one scale per token, stored
                        once (transposed, [C, V]) and read by the Gather (columns, Cast *
                        scale) and the head (DequantizeLinear -> MatMul).
Parity is checked twice: fp32 ONNX must greedy-decode PARITY_PROMPTS token-for-token like
PyTorch, and each candidate's top-1 agreement (plus mean KL) with fp32 is measured on those
same prompts x positions, through the session options the browser should use. Both numbers
are written to the manifest; the qdq files come with a session hint (WASM_HINT) as well.

Run (interim, while r1 trains; the same command on the final checkpoint later):
    set CUDA_VISIBLE_DEVICES=-1
    .venv\\Scripts\\python.exe tools\\export_web.py --run r1
    .venv\\Scripts\\python.exe tools\\export_web.py --run r1 --ckpt runs\\r1\\ckpt.pt   # once training has finished
A run's ckpt.pt is refused until eval_harness.run_finished says the run is over (log at its
target AND the final save on disk): the trainer replaces that file every save, and a long
read on Windows can block the rename and end the run. The manifest's "run" is the directory
the checkpoint lives in (eval_harness.run_of), which must agree with --run.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time

# Before torch is imported: this tool never touches the GPU, and while a run is live the
# card belongs to the trainer. Every device below is "cpu" explicitly as well. The value
# must be "-1", not "": on Windows assigning "" DELETES the variable from the process
# environment block, so CUDA sees every card (verified: "" -> is_available() True, "-1" ->
# False). The assert below proves the guard held on every run.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

assert not torch.cuda.is_available(), "GPU visible despite CUDA_VISIBLE_DEVICES=-1; refusing to run"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import apply_rope  # noqa: E402
import tools.eval_harness as eh  # noqa: E402  (load_model: same step normalisation as sample.py)
from tools.runinfo import read_log, rel_or_abs, run_complete, run_dir_name, step_target, write_atomic  # noqa: E402

ROOT = eh.ROOT
OPSET = 17
EOT_TOKEN = "<|endoftext|>"
# Fixed, published, never tuned: three of the playground's own prompts.
PARITY_PROMPTS = [
    "The reason the sky appears blue is",
    "In 1905, Albert Einstein published",
    "A computer program is",
]
# One fixed context for the page's self-check: the shipped file's argmax after these ids is
# written to the manifest, so infer.js can prove it loaded the right weights on the right
# backend without a tokenizer in the loop. These are tok.encode(PARITY_PROMPTS[0]).ids,
# spelled out so the check does not move if the tokenizer ever does.
SELF_CHECK_IDS = [460, 1979, 262, 6044, 3883, 4119, 311]
QUANT_MODES = ["dynamic-int8", "dynamic-int8-gather", "qdq-int8", "qdq-int8-embed"]
MB = 1_000_000  # GitHub's limit is 100 MiB; sizes are reported in decimal MB, the smaller unit
# onnxruntime session option under which the qdq files are measured and should be run:
# each DequantizeLinear is folded into an fp32 initializer once at load.
FOLD_OPTION = {"session.disable_quant_qdq": "1"}
WASM_HINT = ("create the onnxruntime-web session with extra: {session: {disable_quant_qdq: '1'}} "
             "(graphOptimizationLevel 'all'): the int8 weights are dequantised once at load, "
             "which in Chrome's wasm build made session creation ~10x and each forward ~3x faster "
             "with identical logits; without it the same fp32 arithmetic is redone every run")


class WebGPT(nn.Module):
    """The trained GPT rewritten for tracing: explicit attention (no fused-kernel op), the
    causal mask from a position table, and only the last position through the final norm
    and head. Weights are the trained model's own tensors, not copies."""

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        cfg = gpt.cfg
        self.n_head, self.head_dim, self.n_embd = cfg.n_head, cfg.head_dim, cfg.n_embd
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self.register_buffer("cos", gpt.rope_cos.clone())
        self.register_buffer("sin", gpt.rope_sin.clone())
        self.register_buffer("pos", torch.arange(cfg.block_size, dtype=torch.int64))

    @staticmethod
    def rms(x, weight, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = F.embedding(input_ids, self.gpt.wte.weight)
        pos = self.pos[:T]
        future = pos.view(1, T) > pos.view(T, 1)        # [T, T] True above the diagonal
        for blk in self.gpt.blocks:
            h = self.rms(x, blk.norm1.weight, blk.norm1.eps)
            q, k, v = blk.attn.qkv(h).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            q, k = apply_rope(q, self.cos, self.sin), apply_rope(k, self.cos, self.sin)
            att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(future, -1e9)
            att = torch.softmax(att, dim=-1)
            y = torch.matmul(att, v).transpose(1, 2).reshape(B, T, self.n_embd)
            x = x + blk.attn.proj(y)
            h = self.rms(x, blk.norm2.weight, blk.norm2.eps)
            x = x + blk.mlp(h)
        x = x[:, -1, :]                                  # [1, C]: the only position sampled
        x = self.rms(x, self.gpt.norm_f.weight, self.gpt.norm_f.eps)
        # Tied head as a GEMV against the embedding matrix itself, so the exporter keeps
        # one initializer: x @ W^T would constant-fold a transposed 32 MB copy.
        return torch.matmul(self.gpt.wte.weight, x.transpose(0, 1)).transpose(0, 1)


# ----------------------------------------------------------------------------- torch side

@torch.no_grad()
def torch_last_logits(model, ids):
    logits, _ = model(torch.tensor([ids], dtype=torch.long))
    return logits[0, -1].float().numpy()


@torch.no_grad()
def torch_greedy(model, ids, n, eot, block):
    out = []
    for _ in range(n):
        nxt = int(torch_last_logits(model, (ids + out)[-block:]).argmax())
        if nxt == eot:
            break
        out.append(nxt)
    return out


def export_fp32(gpt, path):
    import onnx
    wrapper = WebGPT(gpt).eval()
    example = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    torch.onnx.export(
        wrapper, (example,), path,
        input_names=["input_ids"], output_names=["logits"],
        dynamic_axes={"input_ids": {1: "seq"}},
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )
    m = onnx.load(path)
    onnx.checker.check_model(m)
    return wrapper, m


# ------------------------------------------------------------------------------- ORT side

def ort_session(path, threads, config=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    for k, v in (config or {}).items():
        so.add_session_config_entry(k, v)
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def ort_logits(sess, ids):
    x = np.asarray([ids], dtype=np.int64)
    return sess.run(["logits"], {"input_ids": x})[0][0]


def ort_greedy(sess, ids, n, eot, block):
    out = []
    for _ in range(n):
        nxt = int(ort_logits(sess, (ids + out)[-block:]).argmax())
        if nxt == eot:
            break
        out.append(nxt)
    return out


# ----------------------------------------------------------------------------- quantisers

def tensor_size_bytes(m):
    return sum(len(t.raw_data) if t.raw_data else 0 for t in m.graph.initializer)


def matmul_weight_inits(m):
    """(node, initializer) for every MatMul whose B input is an initializer: the transformer
    matrices, four per block (qkv, proj, up, down), so 4 x n_layer in all. The tied head has
    its weight as input A and is handled apart."""
    inits = {t.name: t for t in m.graph.initializer}
    out = []
    for node in m.graph.node:
        if node.op_type == "MatMul" and node.input[1] in inits:
            out.append((node, inits[node.input[1]]))
    return out


def find_embedding(m):
    """The one initializer both a Gather and a MatMul (as input A) read: the tied matrix."""
    inits = {t.name: t for t in m.graph.initializer}
    gathers = {n.input[0] for n in m.graph.node if n.op_type == "Gather" and n.input[0] in inits}
    heads = {n.input[0] for n in m.graph.node if n.op_type == "MatMul" and n.input[0] in inits}
    shared = gathers & heads
    assert len(shared) == 1, f"expected one tied embedding initializer, found {shared}"
    return inits[shared.pop()]


def sym_int8(w, axis):
    """Symmetric per-slice int8 along `axis` (each slice gets max|w|/127)."""
    reduce_axes = tuple(i for i in range(w.ndim) if i != axis)
    amax = np.abs(w).max(axis=reduce_axes, keepdims=True)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, scale.reshape(-1)


def quant_dynamic(fp32_path, out_path, gather):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(fp32_path, out_path,
                     op_types_to_quantize=["MatMul", "Gather"] if gather else ["MatMul"],
                     per_channel=True, weight_type=QuantType.QInt8)


def quant_qdq(fp32_path, out_path, embed):
    """Weight-only int8 in QDQ form: DequantizeLinear(int8, per-channel scale) -> MatMul.
    Activations stay fp32, and every op used (DequantizeLinear, Gather, Cast, Mul, MatMul)
    has a kernel on both onnxruntime-web backends, wasm and WebGPU."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(fp32_path)
    g = m.graph
    new_inits, drop = [], set()
    new_nodes = []
    for node, init in matmul_weight_inits(m):
        w = numpy_helper.to_array(init)                     # [K, N]: per output column
        q, s = sym_int8(w, axis=1)
        qn, sn, zn = init.name + "_q", init.name + "_scale", init.name + "_zp"
        new_inits += [numpy_helper.from_array(q, qn), numpy_helper.from_array(s, sn),
                      numpy_helper.from_array(np.zeros(len(s), dtype=np.int8), zn)]
        deq = init.name + "_deq"
        new_nodes.append(helper.make_node("DequantizeLinear", [qn, sn, zn], [deq],
                                          name=node.name + "_dq", axis=1))
        node.input[1] = deq
        drop.add(init.name)
    nodes = list(g.node)
    after = {}  # node name -> nodes to insert right after it (they read its output)
    if embed:
        # One int8 copy of the tied matrix, stored transposed as [C, V] with a scale per
        # token column, so the head is a plain MatMul(x, DequantizeLinear(B, axis=1)): the
        # same shape onnxruntime fuses for the transformer matrices. The Gather reads
        # columns of the same tensor (axis 1) and rescales per token; nothing else is touched.
        emb = find_embedding(m)
        w = numpy_helper.to_array(emb)                      # [V, C]
        q, s = sym_int8(w, axis=0)                          # scale per token row
        qn, sn, zn, sn_col = emb.name + "_qT", emb.name + "_scale", emb.name + "_zp", emb.name + "_scale_col"
        new_inits += [numpy_helper.from_array(np.ascontiguousarray(q.T), qn),   # [C, V]
                      numpy_helper.from_array(s, sn),
                      numpy_helper.from_array(np.zeros(len(s), dtype=np.int8), zn),
                      numpy_helper.from_array(s.reshape(-1, 1), sn_col)]
        prod = {o: n for n in nodes for o in n.output}
        for node in nodes:
            if node.op_type == "Gather" and node.input[0] == emb.name:
                raw, out = node.output[0] + "_i8", node.output[0]
                node.input[0] = qn
                node.output[0] = raw
                keep_attrs = [a for a in node.attribute if a.name != "axis"]
                del node.attribute[:]
                node.attribute.extend(keep_attrs + [helper.make_attribute("axis", 1)])   # [C, 1, T] int8
                after[node.name] = [
                    helper.make_node("Cast", [raw], [out + "_f"], name=node.name + "_cast", to=TensorProto.FLOAT),
                    helper.make_node("Transpose", [out + "_f"], [out + "_t"], name=node.name + "_t", perm=[1, 2, 0]),
                    helper.make_node("Gather", [sn_col, node.input[1]], [out + "_rs"], name=node.name + "_rowscale"),
                    helper.make_node("Mul", [out + "_t", out + "_rs"], [out], name=node.name + "_scale"),
                ]
            elif node.op_type == "MatMul" and node.input[0] == emb.name:
                # MatMul(W, x^T)^T -> MatMul(x, W^T): drop both Transposes around it.
                x_t = prod[node.input[1]]
                assert x_t.op_type == "Transpose"
                out_t = next(n for n in nodes if n.op_type == "Transpose" and n.input[0] == node.output[0])
                deq = emb.name + "_deq"
                new_nodes.append(helper.make_node("DequantizeLinear", [qn, sn, zn], [deq],
                                                  name=node.name + "_dq", axis=1))
                node.input[0], node.input[1] = x_t.input[0], deq
                node.output[0] = out_t.output[0]
                nodes = [n for n in nodes if n is not x_t and n is not out_t]
        drop.add(emb.name)
    keep = [t for t in g.initializer if t.name not in drop]
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    # ONNX requires topological order: the DequantizeLinear nodes read only initializers,
    # so they go first; the embedding's Cast/Transpose/Mul follow the Gather they read.
    ordered = list(new_nodes)
    for n in nodes:
        ordered.append(n)
        ordered.extend(after.get(n.name, []))
    del g.node[:]
    g.node.extend(ordered)
    onnx.checker.check_model(m)
    onnx.save(m, out_path)


def size_breakdown(path):
    """Initializer bytes by dtype, in MB: the honest version of 'how big is it'."""
    import onnx
    m = onnx.load(path, load_external_data=False)
    by = {}
    for t in m.graph.initializer:
        key = onnx.TensorProto.DataType.Name(t.data_type).lower()
        by[key] = by.get(key, 0) + len(t.raw_data)
    return {k: round(v / MB, 1) for k, v in sorted(by.items(), key=lambda kv: -kv[1]) if v >= 0.05 * MB}


# ----------------------------------------------------------------------------- measuring

def agreement(sess_ref, sess_q, contexts):
    """Top-1 agreement of a candidate with fp32 on identical contexts, plus softer numbers
    (mean |dlogit|, mean KL(fp32 || q)) that separate candidates top-1 cannot."""
    hits, dl, kl = 0, [], []
    for ctx in contexts:
        a, b = ort_logits(sess_ref, ctx), ort_logits(sess_q, ctx)
        hits += int(a.argmax() == b.argmax())
        dl.append(float(np.abs(a - b).mean()))
        pa = np.exp(a - a.max()); pa /= pa.sum()
        lb = b - b.max(); lb -= np.log(np.exp(lb).sum())
        la = a - a.max(); la -= np.log(np.exp(la).sum())
        kl.append(float((pa * (la - lb)).sum()))
    return hits, float(np.mean(dl)), float(np.mean(kl))


KL_TOLERANCE = 0.005   # nats; below this the two distributions are indistinguishable at the temperatures the page samples at
HITS_TOLERANCE = 2     # positions out of 72; one flipped near-tie is noise, not a quality signal


def choose(cands, cap_bytes):
    """The smallest file that fits under the cap and loses nothing measurable: a candidate
    qualifies when its mean KL from fp32 is within KL_TOLERANCE of the best candidate's and
    its top-1 hits within HITS_TOLERANCE of the best; the smallest qualifier ships. KL is
    the gate rather than top-1 because 72 positions resolve 1.4% steps and a single flipped
    near-tie would otherwise decide 25 MB of download. The numbers are printed and written
    to the manifest, so the choice can be checked."""
    fit = [c for c in cands if c.get("error") is None and not c.get("secondary") and c["bytes"] <= cap_bytes]
    if not fit:
        return None
    best_kl, best_hits = min(c["kl"] for c in fit), max(c["hits"] for c in fit)
    ok = [c for c in fit if c["kl"] - best_kl <= KL_TOLERANCE and c["hits"] >= best_hits - HITS_TOLERANCE]
    return min(ok, key=lambda c: c["bytes"])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def training_status(run):
    """(steps_total, finished) from the run's log; the manifest must never say finished
    unless the log does."""
    log = os.path.join(ROOT, "runs", run, "log.jsonl")
    if not os.path.exists(log):
        return None, False
    recs, _ = read_log(log)
    if not recs:
        return None, False
    last = recs[-1]
    target = step_target(last)
    per_step = last["tokens"] // (last["step"] + 1)
    total = target // per_step if target and per_step else None
    return total, run_complete(last)


# ----------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="r1")
    ap.add_argument("--ckpt", default=None, help="default runs/<run>/best.pt (model-only, the "
                    "best validation loss so far). Pass runs/<run>/ckpt.pt once training has "
                    "finished, never while it is live: the trainer replaces that file every save.")
    ap.add_argument("--out", default=None, help="default site/model/")
    ap.add_argument("--work", default=None, help="fp32 graph and candidates; default <temp>/embedding-tax/export_web/<run>/ "
                    "(outside the project: the fp32 graph is over GitHub's 100 MiB limit)")
    ap.add_argument("--quant", default="auto", choices=["auto"] + QUANT_MODES)
    ap.add_argument("--size-cap-mb", type=float, default=80.0, help="a candidate over this is not shipped")
    ap.add_argument("--parity-tokens", type=int, default=24)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="CPU threads for torch and ORT; half the cores by default so a live trainer's data pipeline is not starved")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.set_num_threads(args.threads)

    ckpt = args.ckpt or os.path.join(ROOT, "runs", args.run, "best.pt")
    # The run is the directory the weights live in, resolved the way --run is (so --run R1
    # and runs/r1/ agree); the manifest carries that name, never the typed one.
    run = eh.run_of(ckpt)
    if run is None or run != run_dir_name(ROOT, args.run):
        raise SystemExit(f"--run {args.run} but {rel_or_abs(ckpt, ROOT)} belongs to run {run!r}; "
                         f"point --ckpt inside runs/{args.run}/")
    if os.path.basename(ckpt) == "ckpt.pt" and not eh.run_finished(run):
        raise SystemExit(f"run {run} has not finished training: its ckpt.pt is the trainer's live "
                         f"save file and reading it can block the next save and end the run. "
                         f"Export runs/{run}/best.pt instead, or wait for the run to finish.")
    out = args.out or os.path.join(ROOT, "site", "model")
    work = args.work or os.path.join(tempfile.gettempdir(), "embedding-tax", "export_web", run)
    os.makedirs(out, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    import onnx
    import onnxruntime as ort
    t0 = time.time()

    # 1. Weights, on the CPU, with the step normalised the way sample.py reports it.
    model, cfg, tok, step, val_loss = eh.load_model(ckpt, "cpu")
    n_params = model.num_params()
    eot = tok.token_to_id(EOT_TOKEN)
    assert eot is not None, f"tokenizer has no {EOT_TOKEN}"
    steps_total, finished = training_status(run)
    print(f"weights   {rel_or_abs(ckpt, ROOT)}: {n_params:,} params, step {step:,}"
          f"{f' of {steps_total:,}' if steps_total else ''}, val_loss {val_loss}, "
          f"training {'finished' if finished else 'NOT finished'}")
    print(f"tokenizer eot {EOT_TOKEN!r} = {eot}, vocab {tok.get_vocab_size():,}")

    # 2. fp32 ONNX, one initializer for the tied matrix.
    fp32_path = os.path.join(work, "model.fp32.onnx")
    wrapper, m = export_fp32(model, fp32_path)
    big = [t for t in m.graph.initializer if t.dims == [cfg.vocab_size, cfg.n_embd]]
    assert len(big) == 1, f"tied embedding stored {len(big)} times"
    # The quantisers work on every weight-carrying MatMul; the count is the model's, not a
    # number typed here: four Linear layers per block (qkv, proj, up, down).
    n_matmul = 4 * cfg.n_layer
    assert len(matmul_weight_inits(m)) == n_matmul, \
        f"expected {n_matmul} transformer MatMuls, found {len(matmul_weight_inits(m))}"
    fp32_bytes = os.path.getsize(fp32_path)
    print(f"fp32      {fp32_path}: {fp32_bytes / MB:.1f} MB, {len(m.graph.node)} nodes, "
          f"{len(m.graph.initializer)} initializers, opset {OPSET}")

    # 3. Parity: wrapper vs the trained module in torch, then torch vs ORT fp32 greedy.
    ids0 = tok.encode(PARITY_PROMPTS[0]).ids
    with torch.no_grad():
        d = float(np.abs(wrapper(torch.tensor([ids0])).numpy()[0] - torch_last_logits(model, ids0)).max())
    print(f"wrapper   max |dlogit| vs model.gpt forward: {d:.2e}")
    assert d < 1e-3, "export wrapper does not reproduce the model"

    sess = ort_session(fp32_path, args.threads)
    prompts_ids = [tok.encode(p).ids for p in PARITY_PROMPTS]
    n_match = n_total = 0
    contexts = []
    for p, ids in zip(PARITY_PROMPTS, prompts_ids):
        ref = torch_greedy(model, ids, args.parity_tokens, eot, cfg.block_size)
        got = ort_greedy(sess, ids, args.parity_tokens, eot, cfg.block_size)
        n_total += len(ref)
        n_match += sum(int(a == b) for a, b in zip(ref, got)) if len(ref) == len(got) else 0
        contexts += [ids + ref[:j] for j in range(len(ref))]
        print(f"parity    {p!r}\n          torch {tok.decode(ref)!r}\n          ort   {tok.decode(got)!r}")
    parity_str = f"{n_match} of {n_total}"
    print(f"parity    fp32 greedy tokens matching: {parity_str}")
    if n_match != n_total:
        raise SystemExit("fp32 ONNX does not reproduce PyTorch greedy decoding; not exporting")
    # Edge lengths: one token, and a full block, so the rope and mask tables are checked at both ends.
    edge = {}
    for T in (1, cfg.block_size):
        ids = (ids0 * (T // len(ids0) + 1))[:T]
        edge[T] = float(np.abs(ort_logits(sess, ids) - torch_last_logits(model, ids)).max())
        print(f"edge T={T:<5} max |dlogit| ORT vs torch: {edge[T]:.2e}")
    assert max(edge.values()) < 1e-2

    # 4. Candidates: build, size, agreement.
    #    The qdq files are measured with session option session.disable_quant_qdq=1, which
    #    makes onnxruntime fold each DequantizeLinear into an fp32 initializer at load and
    #    compute in fp32. That is the arithmetic onnxruntime-web's wasm build does with or
    #    without the option (checked in Chrome: identical logits), and with it the session
    #    builds ~10x faster and runs ~3x faster, so infer.js should pass it (see WASM_HINT).
    #    A default x64 session instead fuses the pair into MatMulNBits with int8 activation
    #    blocks; that row is kept as "(x64 fused kernel)" for the record, not for choosing.
    modes = QUANT_MODES if args.quant == "auto" else [args.quant]
    cands = []

    def measure(c, path, config=None):
        sq = ort_session(path, args.threads, config)
        hits, dl, kl = agreement(sess, sq, contexts)
        c.update(hits=hits, of=len(contexts), agreement=hits / len(contexts), mean_abs_dlogit=dl, kl=kl)

    for mode in modes:
        path = os.path.join(work, f"model.{mode}.onnx")
        qdq = mode.startswith("qdq")
        c = {"mode": mode, "file": os.path.basename(path)}
        if qdq:
            c["session"] = FOLD_OPTION
        try:
            {"dynamic-int8": lambda: quant_dynamic(fp32_path, path, gather=False),
             "dynamic-int8-gather": lambda: quant_dynamic(fp32_path, path, gather=True),
             "qdq-int8": lambda: quant_qdq(fp32_path, path, embed=False),
             "qdq-int8-embed": lambda: quant_qdq(fp32_path, path, embed=True)}[mode]()
            c["bytes"] = os.path.getsize(path)
            c["initializer_mb"] = size_breakdown(path)
            measure(c, path, c.get("session"))
        except Exception as e:  # a candidate that fails is reported, not hidden
            c["error"] = f"{type(e).__name__}: {e}"
        cands.append(c)
        if qdq and "hits" in c:
            c2 = {"mode": mode + " (x64 fused kernel)", "file": c["file"], "bytes": c["bytes"], "secondary": True}
            try:
                measure(c2, path)
            except Exception as e:
                c2["error"] = f"{type(e).__name__}: {e}"
            cands.append(c2)
    for c in cands:
        print(f"cand      {c['mode']:<34} " + (f"{c['bytes'] / MB:6.1f} MB  top-1 {c['hits']}/{c['of']}  "
              f"mean|dlogit| {c['mean_abs_dlogit']:.4f}  KL {c['kl']:.5f}  {c.get('initializer_mb', '')}"
              if "hits" in c else f"FAILED {c['error']}"))
    print(f"cand      {'fp32 (reference)':<32} {fp32_bytes / MB:6.1f} MB  {size_breakdown(fp32_path)}")

    pick = choose(cands, args.size_cap_mb * MB)
    if pick is None:
        raise SystemExit(f"no candidate under {args.size_cap_mb} MB; nothing shipped")
    print(f"ship      {pick['mode']} ({pick['bytes'] / MB:.1f} MB, top-1 {pick['hits']}/{pick['of']})")
    # The candidates that lost are measured and recorded above; their bytes (50-75 MB each)
    # are not kept: every run rebuilds them, and model bytes belong in site/model/ only.
    for c in cands:
        if c["file"] != pick["file"] and not c.get("secondary"):
            p = os.path.join(work, c["file"])
            if os.path.exists(p):
                os.remove(p)

    # 5. Ship: model, tokenizer copy, manifest (written last, so a torn export has no manifest).
    #    The old manifest goes first: left in place, a copy that dies halfway would describe
    #    the previous weights as if they were the new file's.
    ship_name = "model.int8.onnx"
    ship_path = os.path.join(out, ship_name)
    manifest_path = os.path.join(out, "manifest.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    shutil.copyfile(os.path.join(work, pick["file"]), ship_path)
    shutil.copyfile(eh.TOKENIZER_PATH, os.path.join(out, "tokenizer.json"))
    from tokenizers import Tokenizer
    assert Tokenizer.from_file(os.path.join(out, "tokenizer.json")).encode(PARITY_PROMPTS[0]).ids == ids0
    for stale in os.listdir(out):
        if stale.endswith(".onnx") and stale != ship_name:
            os.remove(os.path.join(out, stale))
    # Self-check, measured on the file that ships, through the session options the page is
    # told to use: the page runs SELF_CHECK_IDS once after load and compares the argmax.
    shipped = onnx.load(ship_path, load_external_data=False)
    sq = ort_session(ship_path, args.threads, pick.get("session"))
    self_argmax = int(ort_logits(sq, SELF_CHECK_IDS).argmax())
    self_fp32 = int(ort_logits(sess, SELF_CHECK_IDS).argmax())
    print(f"selfcheck ids {SELF_CHECK_IDS} -> argmax {self_argmax} {tok.decode([self_argmax])!r} "
          f"(fp32 {self_fp32}, ir_version {shipped.ir_version})")

    describe = {
        "dynamic-int8": f"int8 per-channel weights on the {n_matmul} transformer MatMuls with activations quantised to uint8 at run time (onnxruntime quantize_dynamic); tied embedding/head fp32",
        "dynamic-int8-gather": "as dynamic-int8, plus the tied matrix as uint8 with one scale for all 8.4M values (onnxruntime Gather quantisation), dequantised for the head",
        "qdq-int8": f"weight-only int8, symmetric per output channel, DequantizeLinear in front of the {n_matmul} transformer MatMuls; activations fp32; tied embedding/head fp32",
        "qdq-int8-embed": f"weight-only int8, symmetric per output channel, DequantizeLinear in front of the {n_matmul} transformer MatMuls; tied embedding/head int8 with one scale per token, stored once; activations fp32",
    }
    manifest = {
        "run": run,
        "steps_completed": int(step),
        "params_total": int(n_params),
        "vocab_size": int(cfg.vocab_size),
        "block_size": int(cfg.block_size),
        "file": ship_name,
        "bytes": os.path.getsize(ship_path),
        "quantization": describe[pick["mode"]],
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "eot_id": int(eot),
        "parity": {
            "fp32_greedy_tokens_matching": parity_str,
            "quantized_top1_agreement": round(pick["agreement"], 4),
            # Parity numbers depend on the kernels that produced them; say which.
            "measured_with": f"onnxruntime {ort.__version__} CPUExecutionProvider",
            "self_check": {
                "input_ids": SELF_CHECK_IDS,
                "expected_argmax": self_argmax,
                "expected_token": tok.decode([self_argmax]),
                "fp32_argmax": self_fp32,
            },
        },
        # Beyond the contract: provenance the page may print, never a claim it must infer.
        "checkpoint": rel_or_abs(ckpt, ROOT).replace("\\", "/"),
        "checkpoint_val_loss": val_loss,
        "steps_total": steps_total,
        "training_finished": bool(finished),
        "sha256": sha256(ship_path),
        "fp32_bytes": fp32_bytes,
        "quantization_mode": pick["mode"],
        "parity_detail": {
            "prompts": PARITY_PROMPTS,
            "tokens_per_prompt": args.parity_tokens,
            "positions_compared": len(contexts),
            "quantized_top1_hits": pick["hits"],
            "quantized_mean_abs_dlogit": round(pick["mean_abs_dlogit"], 5),
            "quantized_mean_kl_nats": round(pick["kl"], 6),
            "edge_length_max_abs_dlogit": {str(k): round(v, 6) for k, v in edge.items()},
        },
        "candidates": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items()} for c in cands],
        "graph": {"opset": OPSET, "ir_version": int(shipped.ir_version), "input": "input_ids int64 [1, T]",
                  "output": "logits float32 [1, vocab] (last position)", "kv_cache": False},
        "runtime": {"session_options": pick.get("session", {}),
                    "hint": WASM_HINT if pick.get("session") else "no special session options needed"},
        "tool": {"script": "tools/export_web.py", "torch": torch.__version__, "onnx": onnx.__version__,
                 "onnxruntime": ort.__version__},
    }
    write_atomic(manifest_path, json.dumps(manifest, indent=2))
    print(f"-> {out}: {ship_name} ({manifest['bytes'] / MB:.1f} MB), tokenizer.json, manifest.json   "
          f"[{time.time() - t0:.0f} s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
