# small-llm

A tiny, from-scratch GPT language model you can **train and run in Google Colab
or locally** — with nothing but PyTorch. It's a compact, well-commented
implementation in the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT),
small enough to train on a laptop CPU in a few minutes and to fit comfortably on
a free Colab GPU.

## What's in here

| File | Purpose |
|------|---------|
| `model.py` | The GPT model: embeddings, causal self-attention, Transformer blocks, sampling. |
| `tokenizer.py` | A dependency-free character-level tokenizer. |
| `dataset.py` | Loads a text corpus and samples training batches. |
| `config.py` | All hyperparameters in one place (overridable from the CLI). |
| `train.py` | Training loop with eval, checkpointing, cosine LR schedule, AMP. |
| `generate.py` | Sample text from a trained checkpoint. |
| `scripts/prepare_data.py` | Download the Tiny Shakespeare corpus (or fall back to a bundled sample). |
| `data/sample.txt` | A small Shakespeare excerpt so training works fully offline. |
| `notebooks/small_llm_colab.ipynb` | One-click Colab notebook (clone → train → generate). |

## Quick start (local)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get data (downloads Tiny Shakespeare, ~1MB; offline fallback included)
python scripts/prepare_data.py            # writes data/input.txt

# 3. Train (CPU-friendly defaults; ~a few minutes)
python train.py

# 4. Generate
python generate.py --prompt="To be, or not to be" --max_new_tokens=300
```

No GPU? It still works — the defaults are sized for CPU. With a GPU, scale up:

```bash
python train.py --n_layer=6 --n_head=6 --n_embd=384 --block_size=256 \
                --batch_size=64 --max_iters=5000 --device=cuda
```

## Quick start (Google Colab)

1. Open `notebooks/small_llm_colab.ipynb` in Colab
   (or upload it via *File → Upload notebook*).
2. Set the runtime to GPU: *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all*. It clones this repo, installs deps, trains, and samples.

## Train on your own data

Point training at any UTF-8 text file:

```bash
python train.py --data_path=path/to/your.txt --out_dir=out_mydata
python generate.py --out_dir=out_mydata --prompt="Once upon a time"
```

Bigger corpus → better results. The character-level tokenizer needs no
preprocessing; it just learns the characters that appear in your file.

## Configuration

Every field in `config.py` can be overridden as `--key=value`. The most useful:

| Flag | Default | Meaning |
|------|---------|---------|
| `--block_size` | 128 | Context length (how many tokens the model sees). |
| `--n_layer` / `--n_head` / `--n_embd` | 4 / 4 / 128 | Model size. |
| `--batch_size` | 32 | Sequences per step. |
| `--max_iters` | 3000 | Training steps. |
| `--learning_rate` | 3e-4 | Peak LR (cosine-decayed with warmup). |
| `--device` | auto | `cuda`, `cpu`, `mps`, or `auto`. |
| `--compile` | false | `torch.compile` for extra speed (PyTorch 2.x + CUDA). |

Generation flags (`generate.py`): `--temperature` (randomness), `--top_k`
(restrict to top-k tokens), `--num_samples`, `--prompt`.

## How it works (the 60-second version)

The model reads a sequence of token ids, adds learned token + position
embeddings, and passes them through a stack of Transformer blocks. Each block
does masked (causal) self-attention — every position can attend only to earlier
positions — followed by an MLP. A final linear layer (weight-tied to the input
embedding) predicts the probability of the next token. Training minimizes the
cross-entropy of next-token prediction; generation samples one token at a time,
feeding each prediction back in.

## Notes

- Checkpoints (`out/ckpt.pt`) and the tokenizer (`out/tokenizer.json`) are saved
  to `out_dir`. `generate.py` reads both from there.
- The repo is `.gitignore`d to skip checkpoints and downloaded corpora; the tiny
  bundled `data/sample.txt` is kept so everything runs offline.
- This is an educational model. It learns the *style* of its training text; it
  is not a chatbot and has no knowledge beyond what it's trained on.
