#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/safety-dataset"
RUN_ID="E2-M-EV-GLI-COMPAT-8K"
SESSION="e2_eval"
OUTPUT="$ROOT/reports/evaluation_matrix/$RUN_ID"
INTERRUPTED="$ROOT/reports/evaluation_matrix/${RUN_ID}-interrupted-launch"
LOG="$ROOT/reports/run_logs/${RUN_ID}-eval.log"
INTERRUPTED_LOG="$ROOT/reports/run_logs/${RUN_ID}-eval-interrupted-launch.log"

cd "$ROOT"
mkdir -p reports/evaluation_matrix reports/run_logs

if pgrep -af "evaluate_experiment_matrix.py --run-id $RUN_ID" >/dev/null; then
    echo "Refusing to duplicate an active evaluation process" >&2
    exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Refusing to duplicate active tmux session: $SESSION" >&2
    exit 3
fi
if [[ -e "$OUTPUT" ]]; then
    if [[ -e "$INTERRUPTED" ]]; then
        echo "Refusing to overwrite interrupted output archive: $INTERRUPTED" >&2
        exit 4
    fi
    mv -- "$OUTPUT" "$INTERRUPTED"
fi
if [[ -e "$LOG" ]]; then
    if [[ -e "$INTERRUPTED_LOG" ]]; then
        echo "Refusing to overwrite interrupted log archive: $INTERRUPTED_LOG" >&2
        exit 5
    fi
    mv -- "$LOG" "$INTERRUPTED_LOG"
fi

tmux new-session -d -s "$SESSION" \
    "cd '$ROOT' && export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false && /venv/main/bin/python scripts/evaluate_experiment_matrix.py --run-id '$RUN_ID' --batch-size 32 --precision auto 2>&1 | tee '$LOG'"

sleep 2
tmux list-panes -t "$SESSION" \
    -F 'session=#{session_name} pane_pid=#{pane_pid} dead=#{pane_dead} command=#{pane_current_command}'
echo "output=$OUTPUT"
echo "log=$LOG"
