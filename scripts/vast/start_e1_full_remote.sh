#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/safety-dataset"
RUN_ID="E1-G-EV-512"
SESSION="e1_full"
SMOKE_SOURCE="$ROOT/reports/smoke_runs/$RUN_ID"
SMOKE_ARCHIVE="$ROOT/reports/smoke_runs/${RUN_ID}-50step-scale512"
FULL_OUTPUT="$ROOT/reports/experiment_runs/$RUN_ID"
LOG="$ROOT/reports/run_logs/$RUN_ID.log"

cd "$ROOT"
mkdir -p reports/run_logs reports/experiment_runs reports/smoke_runs

if [[ -d "$SMOKE_SOURCE" ]]; then
    if [[ -e "$SMOKE_ARCHIVE" ]]; then
        echo "Refusing to overwrite existing smoke archive: $SMOKE_ARCHIVE" >&2
        exit 2
    fi
    mv -- "$SMOKE_SOURCE" "$SMOKE_ARCHIVE"
fi
if [[ -e "$FULL_OUTPUT" ]]; then
    echo "Refusing to overwrite existing full-run output: $FULL_OUTPUT" >&2
    exit 3
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Refusing to duplicate active tmux session: $SESSION" >&2
    exit 4
fi

# 7,226 is ceil(115,608 / 32) * 2.  The pinned GLiNER2 trainer computes
# epoch steps with floor division, so an explicit max protects full coverage.
tmux new-session -d -s "$SESSION" \
    "cd '$ROOT' && export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false && /venv/main/bin/python scripts/train_gliguard_experiment.py --run-id '$RUN_ID' --micro-batch-size 8 --max-optimizer-steps 7226 --epochs 2 --eval-strategy epoch --precision auto 2>&1 | tee '$LOG'"

sleep 2
tmux list-panes -t "$SESSION" \
    -F 'session=#{session_name} pane_pid=#{pane_pid} dead=#{pane_dead} command=#{pane_current_command}'
echo "smoke_archive=$SMOKE_ARCHIVE"
echo "full_output=$FULL_OUTPUT"
echo "log=$LOG"
