param(
    [string]$HostAlias = "safety-vast",
    [ValidateSet("E1-G-EV-512", "E2-M-EV-GLI-COMPAT-8K", "E7-M-SCHEMA-EV-8K")]
    [string]$RunId = "E1-G-EV-512",
    [switch]$Watch,
    [int]$IntervalSeconds = 5
)

$ErrorActionPreference = "Stop"
if ($IntervalSeconds -lt 2) { throw "IntervalSeconds must be at least 2" }

$Remote = @"
cd /workspace/safety-dataset
/venv/main/bin/python scripts/vast/remote_experiment_status.py '$RunId'
echo tmux_sessions:
tmux list-sessions 2>/dev/null || true
echo gpu:
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
"@

do {
    if ($Watch) { Clear-Host }
    & ssh $HostAlias $Remote
    if ($LASTEXITCODE -ne 0) { throw "Could not read remote experiment status" }
    if (-not $Watch) { break }
    Start-Sleep -Seconds $IntervalSeconds
} while ($true)
