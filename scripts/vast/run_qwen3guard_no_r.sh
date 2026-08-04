#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${SAFETY_ROOT:-/workspace/safety-dataset}"
PYTHON="${QWEN_PYTHON:-/workspace/venvs/nemotron-vllm/bin/python}"
MODEL_DIR="${QWEN_MODEL_DIR:-${ROOT}/models/qwen3guard_gen_4b}"
OUTPUT_DIR="${QWEN_OUTPUT_DIR:-${ROOT}/reports/qwen3guard/Q1-QWEN3GUARD-GEN-4B-ZS-NR}"
REVISION="6ec42827da0c1ff11e7a49dc269d2e810d27e108"

mkdir -p "${OUTPUT_DIR}/provenance"
date --iso-8601=seconds > "${OUTPUT_DIR}/provenance/started_at.txt"
nvidia-smi -q > "${OUTPUT_DIR}/provenance/nvidia-smi-q.txt"
"${PYTHON}" -m pip freeze > "${OUTPUT_DIR}/provenance/pip-freeze.txt"
"${PYTHON}" - <<'PY' > "${OUTPUT_DIR}/provenance/runtime.json"
import json
import platform
import sklearn
import torch
import transformers
import vllm
print(json.dumps({
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "scikit_learn": sklearn.__version__,
}, indent=2))
PY

"${PYTHON}" "${ROOT}/scripts/vast/download_qwen3guard.py" \
  --output "${MODEL_DIR}" \
  --revision "${REVISION}"

"${PYTHON}" "${ROOT}/scripts/evaluate_qwen3guard_gen_vllm.py" \
  --model "${MODEL_DIR}" \
  --revision "${REVISION}" \
  --manifest "${ROOT}/data/no_r/decoder/test_and_sea.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --max-input-tokens 8064 \
  --max-model-len 8192 \
  --max-new-tokens 128 \
  --request-chunk-size 512 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --seed 3407 \
  --resume

date --iso-8601=seconds > "${OUTPUT_DIR}/provenance/completed_at.txt"
