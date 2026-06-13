"""
Training and model configuration.

All knobs live here so the same settings can be imported by `train.py`,
`generate.py`, `bench.py`, and the Colab notebook. Override any field on the
command line, e.g.  `python train.py --max_iters=2000 --batch_size=32`.
"""

from dataclasses import dataclass, fields


@dataclass
class TrainConfig:
    # --- data ---
    data_path: str = "data/input.txt"   # plain-text training corpus
    out_dir: str = "out"                # where checkpoints + tokenizer are saved
    init_from: str = ""                 # checkpoint to fine-tune from ("" = train from scratch)

    # --- web-scale streaming (optional) ---
    stream: str = ""                    # preset (fineweb/c4/openwebtext/wikitext)
                                        # or a Hugging Face dataset path; "" = off
    hf_config: str = ""                 # HF dataset config name (override preset)
    split: str = "train"                # dataset split to stream
    text_field: str = "text"            # field holding the document text

    # --- tokenizer ---
    tokenizer: str = "char"             # "char" | "bpe"
    vocab_size: int = 512               # target vocab size for BPE (ignored for char)

    # --- model size (small by default; bump these up if you have a GPU) ---
    block_size: int = 128               # context length
    n_layer: int = 16
    n_head: int = 4                     # query heads
    n_kv_head: int = 0                  # key/value heads (GQA); 0 -> equal to n_head
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False                  # RMSNorm + modern blocks usually drop bias
    norm: str = "rms"                   # "rms" | "layer"
    mlp: str = "swiglu"                 # "swiglu" | "gelu"
    use_rope: bool = True               # rotary embeddings (drops learned wpe table)

    # --- optimization ---
    optimizer: str = "adamw"            # "adamw" | "adamw8bit" (needs bitsandbytes)
    batch_size: int = 32                # micro-batch size (per accumulation step)
    grad_accum_steps: int = 1           # effective batch = batch_size * this
    learning_rate: float = 3e-4
    max_iters: int = 3000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0

    # --- learning-rate schedule (cosine with warmup) ---
    warmup_iters: int = 100
    lr_decay_iters: int = 3000          # usually == max_iters
    min_lr: float = 3e-5

    # --- evaluation / logging ---
    eval_interval: int = 250
    eval_iters: int = 50                # batches averaged per evaluation
    log_interval: int = 50
    val_split: float = 0.1              # fraction of data held out for validation

    # --- system / efficiency ---
    device: str = "auto"                # "auto" | "cuda" | "cpu" | "mps"
    dtype: str = "auto"                 # "auto" | "bf16" | "fp16" | "fp32"
    seed: int = 1337
    compile: bool = False               # torch.compile (PyTorch 2.x, CUDA)
    grad_checkpoint: bool = False       # gradient checkpointing (less activation RAM)
    prefetch: bool = True               # async data prefetch on CUDA (overlap I/O + compute)


def parse_overrides(config: TrainConfig, argv: list[str]) -> TrainConfig:
    """Apply simple `--key=value` command-line overrides to a config."""
    field_types = {f.name: f.type for f in fields(config)}
    for arg in argv:
        if not arg.startswith("--"):
            continue
        if "=" not in arg:
            raise ValueError(f"Expected --key=value, got: {arg}")
        key, value = arg[2:].split("=", 1)
        if key not in field_types:
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(config, key)
        # Cast the string value to the type of the existing field value.
        if isinstance(current, bool):
            cast = value.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            cast = int(value)
        elif isinstance(current, float):
            cast = float(value)
        else:
            cast = value
        setattr(config, key, cast)
    return config
