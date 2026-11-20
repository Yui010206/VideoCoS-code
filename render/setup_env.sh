#!/usr/bin/env bash
# Conda env `h3render` for serving MiniMax-H3 with SGLang Diffusion and running this project's scripts.
# The check at the end prints the CUDA architectures of the installed torch: your GPU's must be among them.
set -euo pipefail
source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"     # or set CONDA_SH to your conda's profile script
conda env list | grep -q "^h3render " || conda create -y -n h3render python=3.12
conda activate h3render
pip install --upgrade pip
pip install "sglang[diffusion]==0.5.23"          # 2026-10-30 release; pins torch 2.14.1, flashinfer cu13, diffusers 0.37
python - <<'EOF'
import torch, sglang
print("torch", torch.__version__, "cuda", torch.version.cuda, "sglang", sglang.__version__)
print("arch list", torch.cuda.get_arch_list())
EOF
sglang serve --help | grep -E -- "--model-variant|--ulysses-degree|--tp-size|--encoder-parallel" | head -8
