"""
Build on an open-weight Llama-family model.

This repo's default architecture (RoPE + RMSNorm + SwiGLU + GQA, no biases) is
deliberately Llama-shaped, so the weights of small Llama-family models map onto
it cleanly. This module reads such a model via `transformers` and converts its
state dict into our GPT, so you can:

  * generate from it,
  * fine-tune it on your own data (train.py --init_from=...),
  * or grow it deeper (scripts/grow.py) and continue training.

Verified compatible families: pure Llama-arch models with no attention/MLP bias
and head_dim == hidden/num_heads -- e.g. HuggingFaceTB/SmolLM-135M / 360M and
TinyLlama/TinyLlama-1.1B. Models with QKV bias (Qwen2) or a custom head_dim need
extra handling; we raise a clear error rather than load them silently wrong.

The RoPE convention here matches HF Llama exactly (rotate-half with
emb = cat(freqs, freqs)), so rotary positions line up without re-permutation.
"""

import torch

from model import GPT, GPTConfig


def gpt_config_from_hf(hf_config) -> GPTConfig:
    """Translate a Llama-style HF config into our GPTConfig."""
    c = hf_config
    n_embd = c.hidden_size
    n_head = c.num_attention_heads
    n_kv_head = getattr(c, "num_key_value_heads", n_head)

    # We tie head_dim to n_embd / n_head; reject models that don't.
    explicit_head_dim = getattr(c, "head_dim", None)
    if explicit_head_dim is not None and explicit_head_dim != n_embd // n_head:
        raise ValueError(
            f"Model uses a custom head_dim ({explicit_head_dim} != {n_embd}/{n_head}); "
            f"this loader assumes head_dim == hidden/num_heads."
        )
    if getattr(c, "attention_bias", False) or getattr(c, "mlp_bias", False):
        raise ValueError(
            "Model has attention/MLP bias (e.g. Qwen2). This loader supports "
            "bias-free Llama-arch models (SmolLM, TinyLlama)."
        )

    return GPTConfig(
        vocab_size=c.vocab_size,
        block_size=getattr(c, "max_position_embeddings", 2048),
        n_layer=c.num_hidden_layers,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=False,
        norm="rms",
        mlp="swiglu",
        use_rope=True,
        norm_eps=getattr(c, "rms_norm_eps", 1e-5),
        rope_theta=float(getattr(c, "rope_theta", 10000.0)),
        mlp_hidden=c.intermediate_size,
        tie_embeddings=getattr(c, "tie_word_embeddings", False),
    )


def convert_llama_state_dict(hf_sd: dict, cfg: GPTConfig) -> dict:
    """Map Hugging Face Llama parameter names/shapes onto our GPT's."""
    sd = {}
    sd["transformer.wte.weight"] = hf_sd["model.embed_tokens.weight"]
    sd["transformer.ln_f.weight"] = hf_sd["model.norm.weight"]

    # Output head: present when untied; otherwise shares the embedding.
    lm_head = hf_sd.get("lm_head.weight", hf_sd["model.embed_tokens.weight"])
    sd["lm_head.weight"] = lm_head

    pairs = {
        "input_layernorm.weight": "ln_1.weight",
        "self_attn.q_proj.weight": "attn.q_proj.weight",
        "self_attn.k_proj.weight": "attn.k_proj.weight",
        "self_attn.v_proj.weight": "attn.v_proj.weight",
        "self_attn.o_proj.weight": "attn.o_proj.weight",
        "post_attention_layernorm.weight": "ln_2.weight",
        "mlp.gate_proj.weight": "mlp.w1.weight",
        "mlp.up_proj.weight": "mlp.w3.weight",
        "mlp.down_proj.weight": "mlp.w2.weight",
    }
    for i in range(cfg.n_layer):
        for hf_suffix, our_suffix in pairs.items():
            sd[f"transformer.h.{i}.{our_suffix}"] = hf_sd[f"model.layers.{i}.{hf_suffix}"]
    return sd


def load_pretrained(repo: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Load a Llama-family model from the Hugging Face hub into our GPT."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Loading pretrained models needs transformers: pip install transformers"
        ) from e

    hf_config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    cfg = gpt_config_from_hf(hf_config)

    hf_model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype,
                                                    trust_remote_code=True)
    our_sd = convert_llama_state_dict(hf_model.state_dict(), cfg)

    model = GPT(cfg).to(dtype)
    model.load_state_dict(our_sd, strict=True)
    return model.to(device), cfg


def model_args_from_config(cfg: GPTConfig) -> dict:
    """The dict train.py / generate.py store in a checkpoint."""
    return dict(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_kv_head=cfg.n_kv_head, n_embd=cfg.n_embd,
        dropout=cfg.dropout, bias=cfg.bias, norm=cfg.norm, mlp=cfg.mlp,
        use_rope=cfg.use_rope, norm_eps=cfg.norm_eps, rope_theta=cfg.rope_theta,
        mlp_hidden=cfg.mlp_hidden, tie_embeddings=cfg.tie_embeddings,
    )
