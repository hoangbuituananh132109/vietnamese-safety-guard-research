param(
    [string]$HostName = "root@ssh8.vast.ai",
    [ValidateRange(1, 65535)]
    [int]$Port = 15873,
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_ed25519"
)

$ErrorActionPreference = "Continue"
& ssh -o BatchMode=yes -o ConnectTimeout=12 `
    -i $IdentityFile -p $Port $HostName @'
cd /workspace/safety-dataset
supervisorctl status no-r-phase0-pipeline
cat reports/no_r_phase0/pipeline_state.json 2>/dev/null
echo "== GPU =="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
echo "== latest events =="
tail -n 20 reports/no_r_phase0/events.jsonl 2>/dev/null
state_log=$(python - <<'PY'
import json
from pathlib import Path
p=Path("reports/no_r_phase0/pipeline_state.json")
if p.exists():
    print(json.loads(p.read_text()).get("stage_log",""))
PY
)
if test -n "$state_log"; then
  echo "== current stage log =="
  tail -n 30 "$state_log" 2>/dev/null
fi
'@
