#!/bin/bash
set -Eeuo pipefail

cd /workspace/safety-dataset
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec /workspace/venvs/nemotron-train/bin/python \
  /workspace/safety-dataset/scripts/run_d2_budgeted_pipeline.py
