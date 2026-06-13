"""
A small GPT (decoder-only Transformer) language model.

Kept intentionally compact and dependency-light (just PyTorch) so it can run
and train both locally on a laptop and inside Google Colab. The architecture
uses modern, efficient components:

  * RMSNorm (cheaper than LayerNorm)               -- norm="rms"
  * Rotary position embeddings (RoPE, no wpe table) -- use_rope=True
  * Grouped-query attention (smaller KV cache)      -- n_kv_head < n_head
  * SwiGLU feed-forward                             -- mlp="swiglu"
  * A key/value cache for fast autoregressive generation
  * Optional gradient checkpointing (less activation RAM during training)

Each component can be toggled back to the classic GPT-2 variant (LayerNorm,
learned positions, GELU MLP, full multi-head attention) via the config.
"""

from dataclasses import dataclass

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    """Hyperparameters that define the model size and shape."""

    vocab_size: int = 256          # set from the tokenizer at build time
    block_size: int = 128          # maximum context length (in tokens)
    n_layer: int = 4               # number of Transformer blocks
    n_head: int = 4                # number of query attention heads
    n_kv_head: int = 0             # key/value heads (GQA); 0 -> equal to n_head
    n_embd: int = 128              # embedding / hidden dimension
    dropout: float = 0.1           # dropout probability
    bias: bool = False             # use bias in Linear / norm layers
    norm: str = "rms"              # "rms" | "layer"
    mlp: str = "swiglu"            # "swiglu" | "gelu"
    use_rope: bool = True          # rotary position embeddings

    def __post_init__(self):
        if self.n_kv_head == 0:
            self.n_kv_head = self.n_head
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        if self.use_rope:
            assert (self.n_embd // self.n_head) % 2 == 0, "head_dim must be even for RoPE"


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def make_norm(config: GPTConfig) -> nn.Module:
    if config.norm == "rms":
        return RMSNorm(config.n_embd)
    return nn.LayerNorm(config.n_embd, bias=config.bias)


# ----------------------------- RoPE helpers ------------------------------ #

def precompute_rope(head_dim: int, max_len: int, base: float = 10000.0):
    """Return (cos, sin) tables of shape (max_len, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)            # (max_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (max_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings. x: (B, nh, T, hd); cos/sin: (T, hd)."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# --------------------------- model components ---------------------------- #

class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention with optional GQA + KV cache."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rope = config.use_rope

        # Separate Q / K / V projections (K/V are narrower under GQA).
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, cos=None, sin=None, layer_cache=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Append to / read from the KV cache for fast incremental decoding.
        new_cache = None
        if layer_cache is not None:
            past_k, past_v = layer_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)

        # Grouped-query attention: replicate KV heads to match query heads.
        if self.n_kv_head != self.n_head:
            reps = self.n_head // self.n_kv_head
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        q_len, k_len = q.size(2), k.size(2)
        is_causal = q_len == k_len  # prefill/training; cached decode attends to all past

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if is_causal:
                mask = torch.tril(torch.ones(q_len, k_len, device=x.device)).bool()
                att = att.masked_fill(~mask, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        return y, new_cache


class SwiGLU(nn.Module):
    """Gated SwiGLU feed-forward network."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Size the hidden dim to ~8/3 * n_embd so the 3 matrices match a 4x
        # GELU MLP in parameter count. Round to a multiple of 32.
        hidden = int(8 * config.n_embd / 3)
        hidden = 32 * ((hidden + 31) // 32)
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)  # gate
        self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)  # up
        self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)  # down
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GELU_MLP(nn.Module):
    """Classic GPT-2 position-wise feed-forward network."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """A single pre-norm Transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config)
        self.mlp = SwiGLU(config) if config.mlp == "swiglu" else GELU_MLP(config)

    def forward(self, x, cos=None, sin=None, layer_cache=None):
        attn_out, new_cache = self.attn(self.ln_1(x), cos, sin, layer_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT(nn.Module):
    """The full GPT language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.grad_checkpoint = False

        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=make_norm(config),
        )
        if not config.use_rope:
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying between the token embedding and the output head.
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute rotary tables (registered as buffers, not parameters).
        if config.use_rope:
            head_dim = config.n_embd // config.n_head
            cos, sin = precompute_rope(head_dim, config.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init for residual output projections (GPT-2 recipe).
        scaled = ("o_proj.weight", "c_proj.weight", "w2.weight")
        for name, p in self.named_parameters():
            if name.endswith(scaled):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Count parameters. Position embeddings are excluded by default."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rope:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None, kv_caches=None, pos_offset=0):
        device = idx.device
        B, T = idx.size()
        assert pos_offset + T <= self.config.block_size, (
            f"Position {pos_offset + T} exceeds block size {self.config.block_size}"
        )

        x = self.transformer.wte(idx)
        if not self.config.use_rope:
            pos = torch.arange(pos_offset, pos_offset + T, dtype=torch.long, device=device)
            x = x + self.transformer.wpe(pos)
        x = self.transformer.drop(x)

        cos = sin = None
        if self.config.use_rope:
            cos = self.rope_cos[pos_offset:pos_offset + T]
            sin = self.rope_sin[pos_offset:pos_offset + T]

        new_caches = [] if kv_caches is not None else None
        for i, block in enumerate(self.transformer.h):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            if self.grad_checkpoint and self.training and kv_caches is None:
                # Checkpointing trades compute for activation memory.
                x, _ = checkpoint(block, x, cos, sin, None, use_reentrant=False)
            else:
                x, layer_cache = block(x, cos, sin, layer_cache)
            if new_caches is not None:
                new_caches.append(layer_cache)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss

        # Inference: only compute logits for the last position.
        logits = self.lm_head(x[:, [-1], :])
        return logits, new_caches

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             optimizer="adamw"):
        """Build an optimizer with sensible weight-decay groups.

        optimizer="adamw"      -> torch.optim.AdamW (fused on CUDA when available)
        optimizer="adamw8bit"  -> bitsandbytes 8-bit AdamW (less optimizer memory,
                                  often faster on GPU). Requires bitsandbytes + CUDA.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in param_dict.values() if p.dim() >= 2]
        no_decay = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        if optimizer == "adamw8bit":
            try:
                import bitsandbytes as bnb
            except ImportError as e:
                raise ImportError(
                    "optimizer='adamw8bit' needs bitsandbytes: pip install bitsandbytes"
                ) from e
            return bnb.optim.AdamW8bit(optim_groups, lr=learning_rate, betas=betas)

        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"
        extra = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        """Autoregressively sample, using a KV cache for speed when possible."""
        self.eval()
        if use_cache:
            return self._generate_cached(idx, max_new_tokens, temperature, top_k)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            idx = self._append(idx, logits, temperature, top_k)
        return idx

    def _generate_cached(self, idx, max_new_tokens, temperature, top_k):
        kv_caches = [(None, None) for _ in range(self.config.n_layer)]
        # Prefill: run the whole prompt once, populating the cache.
        idx_cond = idx[:, -self.config.block_size:]
        pos_offset = 0
        logits, kv_caches = self(idx_cond, kv_caches=kv_caches, pos_offset=pos_offset)
        pos_offset += idx_cond.size(1)
        idx = self._append(idx, logits, temperature, top_k)

        for _ in range(max_new_tokens - 1):
            if pos_offset >= self.config.block_size:
                # Context full: fall back to a fresh windowed prefill.
                kv_caches = [(None, None) for _ in range(self.config.n_layer)]
                idx_cond = idx[:, -self.config.block_size:]
                logits, kv_caches = self(idx_cond, kv_caches=kv_caches, pos_offset=0)
                pos_offset = idx_cond.size(1)
            else:
                last = idx[:, -1:]
                logits, kv_caches = self(last, kv_caches=kv_caches, pos_offset=pos_offset)
                pos_offset += 1
            idx = self._append(idx, logits, temperature, top_k)
        return idx

    @staticmethod
    def _append(idx, logits, temperature, top_k):
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return torch.cat((idx, idx_next), dim=1)
