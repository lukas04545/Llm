#!/data/data/com.termux/files/usr/bin/bash
#
# Set up small-llm to train inside Termux on Android (CPU-only, aarch64).
#
# PyTorch has no official Termux wheels, so we install it from the Termux User
# Repository (TUR), which ships a CPU build for aarch64. Run this once:
#
#     bash scripts/termux_setup.sh
#
# Then train a small, phone-friendly model:
#
#     python scripts/prepare_data.py
#     python train.py --device=cpu --n_layer=4 --n_embd=128 --block_size=64 \
#         --batch_size=8 --max_iters=2000
#     python generate.py --prompt="To be"
#
set -e

echo ">> Updating Termux packages ..."
pkg update -y && pkg upgrade -y

echo ">> Installing base packages (python, git, numpy, rust/clang for builds) ..."
pkg install -y python git python-numpy clang rust binutils

echo ">> Enabling the Termux User Repository (TUR) for PyTorch ..."
pkg install -y tur-repo

echo ">> Installing PyTorch (CPU build from TUR) ..."
if pkg install -y python-torch; then
    echo ">> Installed python-torch from TUR."
else
    echo ">> TUR python-torch unavailable; trying pip (may build from source) ..."
    pip install torch || {
        echo "!! Could not install torch automatically."
        echo "!! See https://github.com/termux-user-repository/tur for options."
        exit 1
    }
fi

echo ">> Installing requests (for downloading data) ..."
pip install requests

echo ">> Verifying the install ..."
python - <<'PY'
import torch
print("torch", torch.__version__, "| threads:", torch.get_num_threads())
print("OK - ready to train. CUDA is unavailable on phones; training runs on CPU.")
PY

echo
echo ">> Done. Try:"
echo "   python scripts/prepare_data.py"
echo "   python train.py --device=cpu --n_layer=4 --n_embd=128 --block_size=64 --batch_size=8 --max_iters=2000"
