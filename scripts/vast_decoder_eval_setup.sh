#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/safety-dataset
VENV=/workspace/venvs/nemotron-vllm
mkdir -p "$ROOT/logs" "$ROOT/results" /workspace/venvs

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 "$VENV"
fi

uv pip install --python "$VENV/bin/python" \
  "vllm==0.17.0" \
  "transformers==4.57.6" \
  "scikit-learn==1.9.0"

"$VENV/bin/python" - <<'PY'
import json
from pathlib import Path
import sklearn
import torch
import transformers
import vllm

value = {
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "scikit_learn": sklearn.__version__,
}
Path("/workspace/safety-dataset/results/eval_environment.json").write_text(
    json.dumps(value, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(value, indent=2))
PY

touch "$ROOT/.eval_setup_complete"
