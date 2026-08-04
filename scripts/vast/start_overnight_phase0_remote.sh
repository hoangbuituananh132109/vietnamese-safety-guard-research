#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/safety-dataset"
SESSION="overnight_phase0"
LOG="$ROOT/reports/run_logs/overnight_phase0.log"
STATE="$ROOT/reports/overnight_phase0/state.json"

cd "$ROOT"
mkdir -p reports/run_logs reports/overnight_phase0

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Refusing to duplicate active tmux session: $SESSION" >&2
    exit 2
fi
if pgrep -af "scripts/vast/run_overnight_phase0.py" >/dev/null; then
    echo "Refusing to duplicate active overnight runner process" >&2
    exit 3
fi

tmux new-session -d -s "$SESSION" \
    "cd '$ROOT' && export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false && /venv/main/bin/python scripts/vast/run_overnight_phase0.py --root '$ROOT' --poll-seconds 30 2>&1 | tee '$LOG'"

sleep 2
tmux list-panes -t "$SESSION" \
    -F 'session=#{session_name} pane_pid=#{pane_pid} dead=#{pane_dead} command=#{pane_current_command}'
echo "state=$STATE"
echo "log=$LOG"
