#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

"$PYTHON_BIN" scripts/build_guard_smoke_manifest.py
"$PYTHON_BIN" scripts/patch_gliguard_tokenizer_compat.py
"$PYTHON_BIN" -m pytest \
  tests/test_guard_smoke_data.py tests/test_gliguard_helpers.py \
  tests/test_mmbert_schema_basic.py \
  -q -p no:cacheprovider --basetemp reports/guard_smoke/pytest_reproduction

"$PYTHON_BIN" scripts/run_mmbert_fixed_smoke.py \
  --max-length 512 --batch-size 4 --epochs 2 --max-steps 24 --lora \
  --output-dir reports/guard_smoke/mmbert_fixed_reproduction

"$PYTHON_BIN" scripts/run_mmbert_schema_basic_smoke.py \
  --max-length 512 --batch-size 2 --epochs 2 --max-steps 12 \
  --output-dir reports/guard_smoke/mmbert_schema_basic_reproduction

"$PYTHON_BIN" scripts/run_gliguard_smoke.py \
  --max-steps 8 --max-words 128 \
  --max-train-samples 24 --max-eval-samples 12 \
  --precision auto \
  --output-dir reports/guard_smoke/gliguard_lora_reproduction

echo "Guard smoke pipeline completed successfully."
