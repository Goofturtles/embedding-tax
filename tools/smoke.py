"""Throughput smoke test: does the model train, and how fast?

Runs a handful of real forward/backward/step iterations on random tokens and reports
tokens per second and peak VRAM. The point is to turn "13 days" into an actual token
budget before committing to a corpus size.

Run:  .venv\\Scripts\\python.exe tools/smoke.py
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import GPTConfig  # noqa: E402
from model.gpt import GPT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--compile", action="store_true", help="try torch.compile (needs MSVC on Windows)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this test is meaningless on CPU.")
        return 1

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = GPTConfig()
    device = "cuda"
    model = GPT(cfg).to(device)
    print(f"params: {model.num_params():,}")

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as exc:  # noqa: BLE001
            print(f"torch.compile failed, continuing eager: {exc}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    B, T = args.batch, cfg.block_size
    tokens_per_step = B * T
    print(f"batch {B} x block {T} = {tokens_per_step:,} tokens per step")

    def one_step():
        idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        targets = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(idx, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.item()

    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    last = None
    for _ in range(args.steps):
        last = one_step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tps = args.steps * tokens_per_step / dt
    peak = torch.cuda.max_memory_allocated() / 1e9

    print("-" * 56)
    print(f"loss (random data, ~ln(vocab)={torch.log(torch.tensor(float(cfg.vocab_size))):.2f}): {last:.3f}")
    print(f"time for {args.steps} steps: {dt:.2f} s  ({dt/args.steps*1000:.0f} ms/step)")
    print(f"throughput: {tps:,.0f} tokens/sec")
    print(f"peak VRAM: {peak:.2f} GB of 24 GB")
    print("-" * 56)
    for hours in (24, 72, 120):
        print(f"  {hours:>3} h of training -> {tps*3600*hours/1e9:6.2f} B tokens")
    chinchilla = 20 * 49_296_896
    print(f"Chinchilla-optimal for 49.3M params is ~{chinchilla/1e9:.2f}B tokens "
          f"({chinchilla/tps/3600:.1f} h at this rate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
