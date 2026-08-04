#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/safety-dataset
LOG="$ROOT/logs/coordinator_events.log"

event() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"
}

wait_for_download() {
  local name="$1"
  local result="$ROOT/results/download_${name}.json"
  local program="download-${name}"
  while [[ ! -s "$result" ]]; do
    state="$(supervisorctl status "$program" 2>/dev/null | awk '{print $2}')"
    if [[ "$state" =~ ^(BACKOFF|FATAL|EXITED|STOPPED)$ ]]; then
      event "download_failed name=$name supervisor_state=$state"
      return 1
    fi
    sleep 20
  done
  event "download_ready name=$name"
}

run_profile() {
  local name="$1"
  local result="$ROOT/results/profile_${name}_5090.json"
  local program="profile-${name}"
  if [[ -s "$result" ]]; then
    event "profile_already_complete name=$name"
    return 0
  fi
  supervisorctl start "$program"
  event "profile_started name=$name"
  while true; do
    state="$(supervisorctl status "$program" 2>/dev/null | awk '{print $2}')"
    if [[ "$state" =~ ^(RUNNING|STARTING)$ ]]; then
      sleep 15
      continue
    fi
    if [[ -s "$result" ]]; then
      event "profile_complete name=$name supervisor_state=$state"
      return 0
    fi
    event "profile_failed name=$name supervisor_state=$state"
    return 1
  done
}

# Run sequentially so the two full BF16 models never compete for one GPU.
wait_for_download qwen
run_profile qwen || true
if [[ ! -s "$ROOT/results/download_nemotron.json" ]]; then
  supervisorctl start download-nemotron
  event "download_started name=nemotron"
fi
wait_for_download nemotron
run_profile nemotron || true

touch "$ROOT/results/.profile_coordinator_complete"
event "coordinator_complete"
