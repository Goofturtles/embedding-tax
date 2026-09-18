"""Model configuration for The Embedding Tax.

The GIBC V2 Track 01 cap is 50,000,000 total trainable parameters, and the rules
state that count "includes token embeddings and the output head". That sentence is
the entire design constraint: with GPT-2's 50,257-token vocabulary, the embedding
table alone eats 51.5% of the cap, leaving room for only 7 transformer layers.
A 16,384-token vocabulary drops that to 16.8% and buys 13 layers instead.

Percentages here and everywhere else in this project are shares of the 50,000,000
cap, not of the built model. Both are defensible; mixing them on one page is not.
"""
from dataclasses import dataclass, asdict
import json


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 16384
    n_layer: int = 13
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 1024
    mlp_ratio: int = 4
    tie_embeddings: bool = True   # output head shares the token embedding matrix
    rope: bool = True             # rotary positions: no learned position table to pay for

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def analytic_param_count(cfg: GPTConfig) -> dict:
    """Parameter count derived from the config alone, with no model instantiated.

    Kept separate from the live torch count so the two can be cross-checked; if they
    disagree, one of them is wrong and the cap claim cannot be trusted.
    """
    d = cfg.n_embd
    embedding = cfg.vocab_size * d
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * d
    position = 0 if cfg.rope else cfg.block_size * d

    attn = 4 * d * d                    # q, k, v, out projections
    mlp = 2 * cfg.mlp_ratio * d * d     # up and down projections
    norms = 2 * d                       # two LayerNorm/RMSNorm gains per block
    per_layer = attn + mlp + norms

    total = embedding + head + position + cfg.n_layer * per_layer + d  # + final norm
    return {
        "embedding": embedding,
        "output_head": head,
        "position": position,
        "per_layer": per_layer,
        "layers_total": cfg.n_layer * per_layer,
        "final_norm": d,
        "total": total,
    }
