"""
GPT-3-style model definition (~254M parameters).

A modernized decoder-only transformer that keeps GPT-2's overall shape but
adopts well-established post-GPT-2 improvements. Sized (~250M params) to train
well on the ~10B-token FineWeb-Edu sample -- roughly 40 tokens/parameter, ~2x
past the Chinchilla compute-optimal point, which gives a solid, well-converged
small model rather than a data-starved one:

  * 18 layers, n_embd 1024 (head_dim 64)
  * GQA (grouped-query attention): 16 query heads, 4 key/value heads --
    smaller KV projection + KV cache, so a larger batch fits for better GPU use
  * RoPE (rotary position embeddings) instead of a learned position table
  * RMSNorm instead of LayerNorm, and no biases anywhere
  * SwiGLU feed-forward instead of a GELU MLP
  * QK-normalization for stable from-scratch training (esp. with bf16)
  * tied input/output embeddings (kept from GPT-2)

It deliberately mirrors gpt2.py's public surface -- GPT3Config / GPT3 with a
`forward(idx, targets=None) -> (logits, loss)` contract, a `configure_optimizers`
method, and a `config.block_size` attribute -- so train.py and sample.py can
drive it through the exact same loops. block_size and vocab_size match gpt2 so
the existing tokenized FineWeb-Edu shards work unchanged.
"""

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention

# -----------------------------------------------------------------------------


@dataclass
class GPT3Config:
    """Hyperparameters for the ~254M speed-tuned GPT-3-style model."""
    block_size: int = 1024    # maximum sequence length the model supports
    vocab_size: int = 50304   # 50,257 real tokens, padded to a multiple of 64
    n_layer: int = 18         # transformer blocks (~254M params total)
    n_head: int = 16          # query heads (head_dim = n_embd // n_head = 64)
    n_kv_head: int = 4        # GQA key/value heads (must divide n_head)
    n_embd: int = 1024        # embedding / hidden dimension
    d_ff: int = 2816          # SwiGLU inner dim (~8/3 * n_embd, rounded to mult of 256)
    rope_theta: float = 10000.0  # RoPE base frequency
    norm_eps: float = 1e-5    # RMSNorm epsilon


# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no mean subtraction, no bias)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # normalize in float32 for stability, then cast back
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.type_as(self.weight) * self.weight).to(dtype)


# -----------------------------------------------------------------------------
# Rotary position embeddings (RoPE)
# -----------------------------------------------------------------------------


def build_rope_cache(block_size, head_dim, theta, device=None):
    """Precompute (cos, sin) tables of shape (block_size, head_dim) for RoPE."""
    # inverse frequencies for each pair of dimensions
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(block_size, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)        # (block_size, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)         # (block_size, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    """Rotate the two halves of the last dimension: (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply rotary embeddings to q and k, each shaped (B, n_head, T, head_dim)."""
    T = q.size(-2)
    cos = cos[:T].view(1, 1, T, -1)   # (1, 1, T, head_dim)
    sin = sin[:T].view(1, 1, T, -1)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# -----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Grouped-query causal self-attention with RoPE and QK-norm, via fused SDPA.

    Queries use n_head heads; keys/values use n_kv_head <= n_head heads (GQA),
    which shrinks the KV projection and KV-cache so a larger batch fits -- the
    main win on a big GPU. No biases; q and k are RMS-normalized per head before
    the rotary embedding, which stabilizes from-scratch training in bf16.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.q_dim = self.n_head * self.head_dim       # = n_embd
        self.kv_dim = self.n_kv_head * self.head_dim

        # one projection producing q (full) and k, v (reduced for GQA)
        self.c_attn = nn.Linear(config.n_embd, self.q_dim + 2 * self.kv_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # residual projection: scaled init

        # QK-norm: per-head RMSNorm over head_dim
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(self, x, cos, sin):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=2)
        q = q.view(B, T, self.n_head,    self.head_dim).transpose(1, 2)  # (B, nh,  T, hd)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, nkv, T, hd)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, nkv, T, hd)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)

        # SDPA repeats the n_kv_head k/v across query groups when enable_gqa=True
        y = scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.n_kv_head != self.n_head,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


# -----------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """Gated feed-forward network: (SiLU(x W_gate) * x W_up) W_down, no biases."""

    def __init__(self, config):
        super().__init__()
        self.w_gate = nn.Linear(config.n_embd, config.d_ff, bias=False)
        self.w_up   = nn.Linear(config.n_embd, config.d_ff, bias=False)
        self.w_down = nn.Linear(config.d_ff, config.n_embd, bias=False)
        self.w_down.NANOGPT_SCALE_INIT = 1  # residual projection: scaled init

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# -----------------------------------------------------------------------------


class Block(nn.Module):
    """Pre-norm transformer block with RMSNorm, RoPE attention, and SwiGLU."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


# -----------------------------------------------------------------------------


class GPT3(nn.Module):
    """GPT-3-style decoder-only language model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, eps=config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tie input and output embeddings to reduce parameter count
        self.transformer.wte.weight = self.lm_head.weight

        # RoPE tables: not persisted in the checkpoint (recomputed on build)
        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope_cache(config.block_size, head_dim, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # initialize all weights (gpt2.py defined _init_weights but never applied it)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Forward pass; returns (logits, loss). Loss is None when targets is None."""
        _, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        )
        cos = self.rope_cos.to(idx.device)
        sin = self.rope_sin.to(idx.device)

        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, cos, sin)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """Build AdamW with weight decay applied only to 2-D parameters (weights, not norms)."""
        decay_params, no_decay_params = [], []
        for _, p in self.named_parameters():
            if p.requires_grad:
                (decay_params if p.dim() >= 2 else no_decay_params).append(p)
        optim_groups = [
            {'params': decay_params,    'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused,
        )
