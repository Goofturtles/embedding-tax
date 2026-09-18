"""A decoder-only transformer sized to spend its parameter budget on depth.

Every choice here exists to keep parameters out of the embedding table and in the
layers: a 16,384-token vocabulary, an output head tied to the input embedding, RoPE
instead of a learned position table, RMSNorm without bias, and Linear layers without
bias. The result is 13 layers under a 50M cap where GPT-2's tokenizer allows 7.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class RMSNorm(nn.Module):
    """Weight-only norm. A LayerNorm would add a bias vector per norm, and at 26 norms
    that is real money against the cap."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(head_dim: int, max_len: int, base: float = 10_000.0, device=None):
    """Rotary position tables. Buffers, not parameters, so they cost nothing against the cap."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D). Rotate pairs of channels by the position angle.
    B, H, T, D = x.shape
    x = x.view(B, H, T, D // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    cos = cos[:T].view(1, 1, T, D // 2)
    sin = sin[:T].view(1, 1, T, D // 2)
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.view(B, H, T, D)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # PyTorch picks a fused/flash kernel here when the shapes and dtype allow it.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    """Two matrices, ReLU squared. Kept to two matrices deliberately: a gated variant
    would need a third and cost a layer's worth of budget."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.n_embd
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # This implementation is rotary-only: there is no learned position table to
        # fall back to. analytic_param_count() budgets for one when cfg.rope is False,
        # so allowing it here would break the analytic/live agreement the cap depends on.
        assert cfg.rope, "GPT only implements rotary positions; cfg.rope must be True"
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight  # one matrix, counted once

        cos, sin = build_rope_cache(cfg.head_dim, cfg.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale residual-path output projections by depth, as in GPT-2.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Trainable parameters, counting a tied matrix once. Buffers are excluded."""
        seen, total = set(), 0
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} exceeds block size {self.cfg.block_size}"
        x = self.wte(idx)
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
