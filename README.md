# small-llm

A tiny, from-scratch GPT you can **train and run in Google Colab or locally** —
built on PyTorch, with modern efficiency features so it trains faster and runs
in less RAM. It can learn from a small text file, from large pre-tokenized
corpora (memory-mapped, never fully loaded into RAM), or from an effectively
**endless stream of web text** without downloading it.

## Highlights

- **Modern architecture**: RMSNorm, rotary embeddings (RoPE), grouped-query
  attention (GQA), SwiGLU MLP — better quality per parameter, smaller KV cache.
- **Fast generation**: a KV cache makes autoregressive decoding multiple times
  faster than recomputing the context each step.
- **Quantization ladder** for low-RAM / fast inference: `int8` (no extra deps),
  `int4` (torchao), and `fp4` / `nf4` 4-bit (bitsandbytes).
- **Trains faster + lighter**: mixed precision (bf16/fp16), gradient
  accumulation, gradient checkpointing, `torch.compile`, fused AdamW, TF32.
- **Train on the internet**: stream web-scale datasets (FineWeb, C4, OpenWebText)
  on the fly, or scrape specific URLs into a corpus.
- **Three tokenizers**: character-level, a from-scratch byte-level BPE, and
  GPT-2's BPE (`tiktoken`).

## Files

| File | Purpose |
|------|---------|
| `model.py` | GPT model: RMSNorm, RoPE, GQA, SwiGLU, KV-cache attention, gradient checkpointing. |
| `tokenizer.py` | Char / from-scratch BPE / GPT-2 BPE tokenizers (shared interface). |
| `dataset.py` | In-memory, memory-mapped `.bin`, and streaming web-text loaders. |
| `config.py` | All hyperparameters (overridable as `--key=value`). |
| `train.py` | Training loop: grad accumulation/checkpointing, mixed precision, all data sources. |
| `generate.py` | Sample text (KV cache + optional quantization). |
| `quantize.py` | Post-training quantization: int8 / int4 / fp4 / nf4 + size reporting. |
| `bench.py` | Measure tokens/sec, peak RAM, and model size. |
| `scripts/prepare_data.py` | Download a corpus; optionally write memmap `.bin` files. |
| `scripts/train_bpe.py` | Train + save a byte-level BPE tokenizer. |
| `scripts/fetch_web.py` | Scrape URLs into a training corpus. |
| `notebooks/small_llm_colab.ipynb` | One-click Colab notebook. |

## Quick start (local)

```bash
pip install -r requirements.txt              # just torch + requests for the basics
python scripts/prepare_data.py               # downloads data/input.txt (offline fallback included)
python train.py                              # CPU-friendly defaults, a few minutes
python generate.py --prompt="To be, or not to be"
```

With a GPU, scale up and use bf16:

```bash
python train.py --device=cuda --dtype=bf16 \
    --n_layer=6 --n_head=6 --n_embd=384 --block_size=256 \
    --batch_size=64 --grad_accum_steps=4 --max_iters=5000 --compile=true
```

## Train on Android (Termux)

You can train on a phone. PyTorch has no official Termux wheels, so a setup
script installs a CPU build from the Termux User Repository:

```bash
pkg install git
git clone https://github.com/lukas04545/llm.git && cd llm
bash scripts/termux_setup.sh            # installs python + torch (CPU) for aarch64
python scripts/prepare_data.py
# keep it small so a phone CPU stays responsive:
python train.py --device=cpu --n_layer=4 --n_embd=128 --block_size=64 \
    --batch_size=8 --max_iters=2000
python generate.py --prompt="To be"
```

Phones are CPU-only (no CUDA), so quantization/streaming extras are optional;
the char tokenizer + small model train comfortably. Bump the size up only if
your device has the RAM and patience.

## Train faster / use less RAM

| Technique | Flag | What it does |
|-----------|------|--------------|
| Mixed precision | `--dtype=bf16` (or `fp16`) | ~2× faster + less memory on GPU. `auto` picks the best. |
| Gradient accumulation | `--grad_accum_steps=4` | Large *effective* batch with small per-step memory. |
| Gradient checkpointing | `--grad_checkpoint=true` | Big cut in activation memory (fit bigger models). |
| Compilation | `--compile=true` | Fuses kernels via `torch.compile` (PyTorch 2.x + CUDA). |
| 8-bit optimizer | `--optimizer=adamw8bit` | Halves optimizer-state memory, often faster (needs bitsandbytes + CUDA). |
| Async prefetch | `--prefetch=true` (default) | Overlaps the next batch's host→GPU copy with compute. CUDA only. |
| Smaller KV cache | `--n_kv_head=2` | GQA shrinks attention memory at inference. |
| Memory-mapped data | (automatic, see below) | Large corpora never fully load into RAM. |

On CUDA, `cudnn.benchmark` is enabled automatically to autotune kernels for the
fixed batch shapes.

The KV cache is on by default for `generate.py`; compare with `bench.py --no_cache`.

## Tokenizers

```bash
python train.py --tokenizer=char                       # default, zero deps
python train.py --tokenizer=bpe --vocab_size=1024      # from-scratch byte-level BPE
python train.py --tokenizer=gpt2                        # GPT-2 BPE (pip install tiktoken)
```

BPE packs more text into each token, so sequences are shorter → less compute and
RAM per step. `python scripts/train_bpe.py` trains a BPE tokenizer standalone.

## Large corpora (memory-mapped)

Pre-tokenize once into `out/train.bin` + `out/val.bin`; training then streams
tokens from disk via `mmap` and never loads the whole corpus into RAM:

```bash
python scripts/prepare_data.py --bin --tokenizer=bpe --vocab_size=1024
python train.py        # auto-detects out/train.bin + out/val.bin
```

## Train on the internet

**Web-scale streaming** (no download — an effectively endless token stream):

```bash
pip install datasets tiktoken
python train.py --stream=fineweb --tokenizer=gpt2 --device=cuda --dtype=bf16 \
    --n_layer=6 --n_head=6 --n_embd=384 --block_size=256 --max_iters=20000
```

Presets: `fineweb`, `c4`, `openwebtext`, `wikitext`. Or pass any Hugging Face
dataset path plus `--text_field`. Documents are tokenized on the fly into a
rolling buffer, so memory stays flat regardless of corpus size.

**Scrape specific pages** into a corpus:

```bash
python scripts/fetch_web.py --urls https://www.gutenberg.org/files/1342/1342-0.txt \
    --bin --tokenizer=bpe
python train.py
```

> **Reality check:** you can't *download the whole internet*, and a small model
> can't memorize it — but streaming lets it train on an unbounded, shuffled flow
> of real web text, which is exactly how large models are trained. Make the model
> bigger (`--n_layer/--n_head/--n_embd`) to absorb more of it.

## Distill from DeepSeek (API-generated training data)

Use a strong model as a **teacher** that writes a custom corpus, then train the
small GPT on it. A remote API can't backprop into our model (no gradients, and
the tokenizers differ), but it can generate excellent, on-topic text —
"distillation by data":

```bash
export DEEPSEEK_API_KEY=sk-...           # https://platform.deepseek.com
python scripts/deepseek_gen.py --topic="bedtime stories" --num_docs=300
python train.py --data_path=data/deepseek.txt
```

Or tokenize straight to memmap `.bin` and let `train.py` auto-detect it:

```bash
python scripts/deepseek_gen.py --num_docs=500 --bin --tokenizer=bpe --vocab_size=1024
python train.py
```

Useful flags: `--prompt` (explicit instruction overriding `--topic`), `--model`
(`deepseek-chat` / `deepseek-reasoner`), `--max_tokens`, `--temperature`,
`--append` (grow a corpus over multiple runs), `--dry_run` (preview the request
without calling the API). It's OpenAI-compatible and uses only `requests`.

### Top-k logprob distillation (advanced)

For a stronger signal than text alone, match DeepSeek's tokenizer and learn from
the teacher's **top-k next-token distribution** (its `top_logprobs`) with a KL
soft-target loss. Soft targets tell the student how confident the teacher was and
what the runner-up tokens were.

```bash
pip install transformers                     # DeepSeek tokenizer
export DEEPSEEK_API_KEY=sk-...
python scripts/distill_collect.py --num_docs=300 --out=out/distill.pt
python distill.py --data=out/distill.pt --max_iters=3000 --alpha=0.5 --temp=1.0
python generate.py --prompt="Once upon a time"
```

`distill.py` minimizes `alpha * CE(hard label) + (1-alpha) * T² * KL(teacher_topk
|| student_topk)`. Caveats: the API returns only the top ~20 tokens (so the KL is
over the teacher's top-k, not the full vocab), DeepSeek's ~128k vocab inflates the
student's embedding table, and it costs more API calls than plain text generation.
`--tokenizer=deepseek` is also available for the text-only path.

## Build on an open-weight model (fine-tune & grow)

Instead of training from scratch, start from a pretrained **Llama-family** model
— this repo's default architecture (RoPE + RMSNorm + SwiGLU + GQA, bias-free) is
deliberately Llama-shaped, so the weights map on cleanly.

```bash
pip install transformers
# 1. Import an open-weight model into this repo's format
python scripts/import_hf.py --repo=HuggingFaceTB/SmolLM-135M
python generate.py --prompt="The meaning of life is"

# 2. Fine-tune it on your own data
python train.py --init_from=out/ckpt.pt --data_path=data/mydata.txt \
    --learning_rate=2e-5 --max_iters=2000

# 3. (Optional) Grow it deeper, then keep training
python scripts/grow.py --add=8 --ckpt=out/ckpt.pt   # +8 identity layers
python train.py --init_from=out/ckpt.pt --data_path=data/mydata.txt
```

- **`import_hf.py`** converts a Hugging Face Llama-arch model (verified: SmolLM
  135M/360M, TinyLlama-1.1B) into a checkpoint + tokenizer. Models with QKV bias
  (Qwen2) or a custom head_dim are rejected with a clear message rather than
  loaded incorrectly.
- **`train.py --init_from=ckpt.pt`** fine-tunes from any checkpoint, reusing its
  tokenizer so the vocab matches. Use a small learning rate (e.g. `2e-5`).
- **`grow.py --add=N`** inserts N new Transformer blocks with zeroed output
  projections (LLaMA-Pro style), so the larger model starts as the *exact same
  function* and learns to use the extra depth during further training.

> A pretrained base is the realistic way to get a capable model without
> frontier-scale compute (see the GPT-2/3/4 discussion above): borrow open
> weights, then specialize them with fine-tuning and/or distillation.

## Quantization (low-RAM, fast inference)

Quantization is applied to a **trained** model for cheaper generation:

```bash
python generate.py --quantize=int8                      # CPU, no extra deps
python generate.py --quantize=int4                      # pip install torchao  (CPU+GPU)
python generate.py --quantize=nf4 --device=cuda         # pip install bitsandbytes (GPU)
python quantize.py --quantize=int8                      # report size reduction
```

| Mode | Backend | Where | Dependency |
|------|---------|-------|------------|
| `int8` | PyTorch dynamic | CPU | none |
| `int4` | torchao weight-only | CPU + CUDA | `torchao` |
| `fp4` / `nf4` | bitsandbytes 4-bit | CUDA / Colab | `bitsandbytes` |

> 4-bit/FP4 is an **inference** technique: train in bf16/fp16, then quantize. On
> this tiny model the absolute RAM saved is small, but the speedup on CPU and the
> mechanics are identical to large models.

## Measure it

```bash
python bench.py --quantize=none
python bench.py --quantize=int8        # compare size, RAM, tokens/sec
python bench.py --no_cache             # see the KV-cache speedup
```

## Colab

Open `notebooks/small_llm_colab.ipynb`, set *Runtime → GPU*, then *Run all*. It
clones the repo, installs deps, trains (or streams from the web), generates, and
benchmarks a 4-bit quantized model.

## Configuration

Every field in `config.py` is overridable as `--key=value`. Common ones:
`--block_size`, `--n_layer/--n_head/--n_kv_head/--n_embd`, `--batch_size`,
`--grad_accum_steps`, `--max_iters`, `--learning_rate`, `--dtype`, `--device`,
`--norm` (`rms`/`layer`), `--mlp` (`swiglu`/`gelu`), `--use_rope`.

## Notes

- Checkpoints (`out/ckpt.pt`) + tokenizer (`out/tokenizer.json`) are written to
  `out_dir`; `generate.py`, `bench.py`, and `quantize.py` read from there.
- Only `torch` + `requests` are required. Streaming, GPT-2 BPE, and 4-bit quant
  use optional packages (clearly noted in `requirements.txt`).
- This is an educational model — it learns the *style* of its training text; it
  is not a chatbot.
