#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/safety-dataset
mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/models" "$ROOT/hf-cache"

export HF_HOME="$ROOT/hf-cache"
export HUGGINGFACE_HUB_CACHE="$ROOT/hf-cache/hub"
export TOKENIZERS_PARALLELISM=false

uv pip install --python /venv/main/bin/python \
  "transformers==4.57.6" \
  "peft" \
  "bitsandbytes" \
  "accelerate" \
  "tensorboard" \
  "scikit-learn==1.9.0"

/venv/main/bin/python - <<'PY'
import json
import pathlib
import torch
import transformers
import peft
import accelerate
import bitsandbytes
import sklearn

out = {
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
    "bitsandbytes": bitsandbytes.__version__,
    "scikit_learn": sklearn.__version__,
}
path = pathlib.Path("/workspace/safety-dataset/results/environment.json")
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
PY

touch "$ROOT/.setup_complete"
