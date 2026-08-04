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
  --config configs/phase0_no_r_experiments.json \
  --report-root reports/no_r_phase0 \
  --python /venv/main/bin/python \
  --poll-seconds 15 \
  --retry-seconds 60 \
  --max-wall-hours 11.25 \
  --ack-wait-seconds 900 \
  2>&1
