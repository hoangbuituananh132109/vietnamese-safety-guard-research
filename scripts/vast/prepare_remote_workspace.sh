#!/usr/bin/env bash
set -euo pipefail

cd "${1:-/workspace/safety-dataset}"
# Vast's PyTorch image keeps the CUDA-enabled framework in /venv/main. Creating
# a venv from /usr/bin/python3 would hide that installation, so install the small
# project dependency layer into the image-provided environment instead.
source /venv/main/bin/activate
uv pip install --python /venv/main/bin/python -r requirements-vast.txt

python - <<'PY'
import torch
import transformers
import peft
import gliner2

print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
print("cuda_runtime=", torch.version.cuda)
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("vram_gb=", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else 0)
print("transformers=", transformers.__version__)
print("peft=", peft.__version__)
print("gliner2=", getattr(gliner2, "__version__", "unknown"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; do not start paid training")
PY

mkdir -p reports/experiment_runs reports/remote_profile
python scripts/preflight_phase0_experiments.py
echo "Remote workspace is ready. Run the notebook or the target-GPU profiler next."
