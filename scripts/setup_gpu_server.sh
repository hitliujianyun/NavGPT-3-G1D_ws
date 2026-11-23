#!/usr/bin/env bash
# Create the three NavGPT-3 environments on a Linux + NVIDIA GPU server (uv workflow
# from code/NavGPT-3/README.md). Habitat is only needed for simulation evaluation.
#
# Blackwell GPUs (RTX 5090, compute capability 12.0) need CUDA 12.8 wheels, which
# start at PyTorch 2.7; the upstream pin (torch 2.6 + cu124) has no sm_120 kernels.
# Override with TORCH_SPEC / TORCH_INDEX if needed.
set -euo pipefail
cd "$(dirname "$0")/../code/NavGPT-3"

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || true)
if [[ "${CAP:-0}" -ge 120 ]]; then
  : "${TORCH_SPEC:=torch==2.7.1 torchvision==0.22.1}"
  : "${TORCH_INDEX:=https://download.pytorch.org/whl/cu128}"
else
  : "${TORCH_SPEC:=torch==2.6.0 torchvision==0.21.0}"
  : "${TORCH_INDEX:=https://download.pytorch.org/whl/cu124}"
fi
echo "GPU compute capability: ${CAP:-unknown} -> ${TORCH_SPEC} from ${TORCH_INDEX}"

uv venv --python 3.11 .envs/planner
uv pip install --python .envs/planner/bin/python --editable .

uv venv --python 3.11 .envs/vla
# shellcheck disable=SC2086
uv pip install --python .envs/vla/bin/python $TORCH_SPEC --index-url "$TORCH_INDEX"
# Everything else from requirements-vla.txt, without its torch pin undoing the line above.
grep -vE '^(torch|torchvision)\b' requirements-vla.txt > /tmp/navgpt3-requirements-vla.txt
uv pip install --python .envs/vla/bin/python -r /tmp/navgpt3-requirements-vla.txt
uv pip install --python .envs/vla/bin/python --no-deps --editable .

.envs/vla/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    print("device", torch.cuda.get_device_name(0), arch, "supported:", arch in torch.cuda.get_arch_list())
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    print("bf16 matmul ok:", float((x @ x).float().abs().mean()) > 0)
EOF

if [[ "${WITH_HABITAT:-0}" == "1" ]]; then
  conda create --yes --prefix "$PWD/.envs/habitat" \
    --override-channels -c aihabitat -c conda-forge \
    python=3.8 habitat-sim=0.1.7 headless 'numpy<2' quaternion pillow pyyaml
fi

cp -n configs/paths.example.yaml configs/paths.yaml || true
echo "Fill in configs/paths.yaml, then: .envs/planner/bin/navgpt check-checkpoint --config configs/experiments/vln/vla_8b_r2r.yaml"
