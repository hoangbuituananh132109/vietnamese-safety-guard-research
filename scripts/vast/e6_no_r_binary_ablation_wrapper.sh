#!/usr/bin/env bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

source /venv/main/bin/activate
cd /workspace/safety-dataset

pty /venv/main/bin/python \
  scripts/vast/run_no_r_phase0_pipeline.py \
  --root /workspace/safety-dataset \
  --config configs/e6_no_r_binary_ablation.json \
  --report-root reports/e6_no_r_binary_ablation \
  --archive-stem E6_NO_R_BINARY_ABLATION_20260724 \
  --python /venv/main/bin/python \
  --poll-seconds 15 \
  --retry-seconds 60 \
  --max-wall-hours 2.0 \
  --ack-wait-seconds 900 \
  --no-auto-stop \
  2>&1
