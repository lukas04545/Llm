"""
Grow a model deeper (identity-preserving block expansion), LLaMA-Pro style.

New Transformer blocks are inserted at regular intervals with their attention and
MLP *output* projections zeroed. A zeroed output projection makes the block's
residual contribution zero, so `block(x) == x` at initialization -- the expanded
model computes exactly the same function as the original, then *learns* to use
the extra depth during continued training.

    python scripts/import_hf.py --repo=HuggingFaceTB/SmolLM-135M
    python scripts/grow.py --add=8           # 30 -> 38 layers, function preserved
    python train.py --init_from=out/ckpt.pt --data_path=data/mydata.txt

Width growth is intentionally not done here (it perturbs every matrix and rarely
helps small models); depth growth is the clean, standard "expand" operation.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig  # noqa: E402


def zero_block_outputs(block) -> None:
    """Zero a block's residual contributions so it acts as the identity."""
    for name in ("o_proj",):
        layer = getattr(block.attn, name)
        torch.nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
    # MLP down-projection: w2 (SwiGLU) or c_proj (GELU).
    down = getattr(block.mlp, "w2", None) or getattr(block.mlp, "c_proj", None)
    torch.nn.init.zeros_(down.weight)
    if down.bias is not None:
        torch.nn.init.zeros_(down.bias)


def interleave_positions(n_existing: int, n_add: int) -> list[int]:
    """Indices (in the new sequence) where fresh identity blocks are inserted,
    spread roughly evenly through the stack."""
    step = n_existing / (n_add + 1)
    return [int(round((j + 1) * step)) + j for j in range(n_add)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="out/ckpt.pt")
    parser.add_argument("--add", type=int, required=True, help="number of new layers")
    parser.add_argument("--out", default=None, help="defaults to overwriting --ckpt")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    old_args = ckpt["model_args"]
    n_old = old_args["n_layer"]

    # Build the source model and load its weights.
    src = GPT(GPTConfig(**old_args))
    src.load_state_dict(ckpt["model"])

    # Build the larger target model.
    new_args = dict(old_args)
    new_args["n_layer"] = n_old + args.add
    dst = GPT(GPTConfig(**new_args))

    insert_at = set(interleave_positions(n_old, args.add))
    print(f"Growing {n_old} -> {new_args['n_layer']} layers; "
          f"identity blocks inserted at {sorted(insert_at)}")

    # Copy non-block weights (embeddings, final norm, head) verbatim.
    dst.transformer.wte.load_state_dict(src.transformer.wte.state_dict())
    dst.transformer.ln_f.load_state_dict(src.transformer.ln_f.state_dict())
    dst.lm_head.load_state_dict(src.lm_head.state_dict())
    if not new_args.get("use_rope", True):
        dst.transformer.wpe.load_state_dict(src.transformer.wpe.state_dict())

    # Walk the new stack: copy an old block, or insert a zeroed identity block.
    src_i = 0
    for dst_i in range(new_args["n_layer"]):
        if dst_i in insert_at:
            zero_block_outputs(dst.transformer.h[dst_i])
        else:
            dst.transformer.h[dst_i].load_state_dict(
                src.transformer.h[src_i].state_dict())
            src_i += 1

    out = args.out or args.ckpt
    torch.save({"model": dst.state_dict(), "model_args": new_args, "iter": 0,
                "grown_from": args.ckpt}, out)
    print(f"Saved grown model ({dst.num_params() / 1e6:.1f}M params) to {out}")
    print("Continue training: python train.py --init_from=" + out + " --data_path=...")


if __name__ == "__main__":
    main()
