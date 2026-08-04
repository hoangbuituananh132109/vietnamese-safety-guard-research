#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/safety-dataset
CHECKPOINT="$ROOT/results/no_r_decoder_4080s/train/nemotron/checkpoint-00000250/complete.marker"
LOG="$ROOT/logs/no_r_decoder_4080s/qwen-eval-priority-switch.log"

mkdir -p "$(dirname "$LOG")"
echo "[$(date -Is)] Waiting for complete Nemotron checkpoint 250." >>"$LOG"

while [[ ! -f "$CHECKPOINT" ]]; do
  if ! supervisorctl status no-r-decoder-pipeline | grep -q RUNNING; then
    echo "[$(date -Is)] Main pipeline stopped before checkpoint 250." >>"$LOG"
    exit 1
  fi
  sleep 10
done

echo "[$(date -Is)] Checkpoint 250 complete; pausing main pipeline." >>"$LOG"
supervisorctl stop no-r-decoder-pipeline

for _ in $(seq 1 30); do
  if ! pgrep -f "train_decoder_guard_lora.py.*model-kind nemotron" >/dev/null; then
    break
  fi
  sleep 2
done

if pgrep -f "train_decoder_guard_lora.py.*model-kind nemotron" >/dev/null; then
  echo "[$(date -Is)] Nemotron child did not stop cleanly." >>"$LOG"
  exit 1
fi

echo "[$(date -Is)] GPU released; restarting reordered pipeline." >>"$LOG"
supervisorctl start no-r-decoder-pipeline
echo "[$(date -Is)] Reordered pipeline started; Qwen evaluation is now first." >>"$LOG"
