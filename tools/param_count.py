"""Prints the parameter count and the config. Required by the Track 01 rules:
'Your repository must include a script or printed output showing the parameter count,
plus your model config.'

Run:  python tools/param_count.py
If torch is installed it also instantiates the real model and asserts the live count
matches the analytic one, so the number below is not just arithmetic on paper.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import GPTConfig, analytic_param_count  # noqa: E402

CAP = 50_000_000
GPT2_VOCAB = 50_257


def layers_that_fit(vocab: int, cfg: GPTConfig) -> int:
    """How many layers fit under the cap at this vocabulary size, all else equal."""
    probe = GPTConfig(vocab_size=vocab, n_layer=1, n_embd=cfg.n_embd, n_head=cfg.n_head,
                      mlp_ratio=cfg.mlp_ratio, tie_embeddings=cfg.tie_embeddings, rope=cfg.rope)
    b = analytic_param_count(probe)
    fixed = b["embedding"] + b["output_head"] + b["position"] + b["final_norm"]
    return (CAP - fixed) // b["per_layer"]


def main() -> int:
    cfg = GPTConfig()
    b = analytic_param_count(cfg)

    print("=" * 62)
    print("THE EMBEDDING TAX  -  parameter budget")
    print("=" * 62)
    print(f"vocab_size      {cfg.vocab_size:>12,}")
    print(f"n_layer         {cfg.n_layer:>12,}")
    print(f"n_head          {cfg.n_head:>12,}")
    print(f"n_embd          {cfg.n_embd:>12,}")
    print(f"head_dim        {cfg.head_dim:>12,}")
    print(f"block_size      {cfg.block_size:>12,}")
    print(f"tie_embeddings  {str(cfg.tie_embeddings):>12}")
    print(f"rope            {str(cfg.rope):>12}")
    print("-" * 62)
    print(f"token embedding {b['embedding']:>12,}   {100*b['embedding']/b['total']:5.1f}% of model")
    print(f"output head     {b['output_head']:>12,}   (tied to embedding)" if cfg.tie_embeddings
          else f"output head     {b['output_head']:>12,}")
    print(f"position table  {b['position']:>12,}   (RoPE: none to pay for)" if cfg.rope
          else f"position table  {b['position']:>12,}")
    print(f"per layer       {b['per_layer']:>12,}   x {cfg.n_layer} layers")
    print(f"layers total    {b['layers_total']:>12,}   {100*b['layers_total']/b['total']:5.1f}% of model")
    print(f"final norm      {b['final_norm']:>12,}")
    print("-" * 62)
    print(f"TOTAL           {b['total']:>12,}")
    print(f"CAP             {CAP:>12,}")
    print(f"headroom        {CAP - b['total']:>12,}")
    status = "UNDER CAP" if b["total"] <= CAP else "OVER CAP"
    print(f"STATUS          {status:>12}")

    print()
    print("=" * 62)
    print("WHAT THE VOCABULARY COSTS  (same width, same cap)")
    print("=" * 62)
    print(f"{'vocab':>8} {'embedding':>12} {'emb % of cap':>13} {'layers that fit':>16}")
    print("-" * 62)
    for v in (GPT2_VOCAB, 32_000, 16_384, 8_192):
        emb = v * cfg.n_embd
        mark = "  <- GPT-2" if v == GPT2_VOCAB else ("  <- ours" if v == cfg.vocab_size else "")
        print(f"{v:>8,} {emb:>12,} {100*emb/CAP:>12.1f}% {layers_that_fit(v, cfg):>16}{mark}")

    live = None
    try:
        import torch  # noqa: F401
        from model.gpt import GPT
        m = GPT(cfg)
        live = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print()
        print(f"live torch count: {live:,}")
        assert live == b["total"], f"MISMATCH: analytic {b['total']:,} vs live {live:,}"
        print("analytic and live counts agree")
    except ImportError:
        print()
        print("(torch or model/gpt.py not available yet - analytic count only)")

    cfg.to_json(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "configs", "embedding_tax.json"))
    return 0 if b["total"] <= CAP else 1


if __name__ == "__main__":
    raise SystemExit(main())
