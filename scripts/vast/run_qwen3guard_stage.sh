#!/usr/bin/env bash
set -Eeuo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

ROOT="${SAFETY_ROOT:-/workspace/safety-dataset}"
PYTHON="${QWEN_PYTHON:-/workspace/venvs/nemotron-vllm/bin/python}"
MODEL_DIR="${QWEN_MODEL_DIR:-${ROOT}/models/qwen3guard_gen_4b}"
RUN_ROOT="${QWEN_RUN_ROOT:-${ROOT}/reports/qwen3guard}"
STAGE_FILE="${QWEN_STAGE_FILE:-${RUN_ROOT}/stage.txt}"
REVISION="6ec42827da0c1ff11e7a49dc269d2e810d27e108"

mkdir -p "${RUN_ROOT}"
stage="$(tr -d '[:space:]' < "${STAGE_FILE}")"

case "${stage}" in
  download)
    exec "${PYTHON}" "${ROOT}/scripts/vast/download_qwen3guard.py" \
      --output "${MODEL_DIR}" \
      --revision "${REVISION}"
    ;;
  smoke)
    exec "${PYTHON}" "${ROOT}/scripts/evaluate_qwen3guard_gen_vllm.py" \
      --model "${MODEL_DIR}" \
      --revision "${REVISION}" \
      --manifest "${ROOT}/data/no_r/decoder/qwen_smoke_512.jsonl" \
      --output-dir "${RUN_ROOT}/Q1-SMOKE-512" \
      --max-input-tokens 8064 \
      --max-model-len 8192 \
      --max-new-tokens 128 \
      --request-chunk-size 512 \
      --max-num-seqs 64 \
      --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.90 \
      --seed 3407
    ;;
  full)
    exec env \
      SAFETY_ROOT="${ROOT}" \
      QWEN_PYTHON="${PYTHON}" \
      QWEN_MODEL_DIR="${MODEL_DIR}" \
      QWEN_OUTPUT_DIR="${RUN_ROOT}/Q1-QWEN3GUARD-GEN-4B-ZS-NR" \
      "${ROOT}/scripts/vast/run_qwen3guard_no_r.sh"
    ;;
  *)
    echo "Unknown Qwen stage '${stage}'. Expected download, smoke, or full." >&2
    exit 2
    ;;
esac
