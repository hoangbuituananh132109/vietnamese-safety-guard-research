$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project '.venv\Scripts\python.exe'
$runner = Join-Path $project 'tools\luna_overnight_runner.py'
$recover = Join-Path $project 'tools\recover_overlapped_luna_run.py'
$webBuilder = Join-Path $project 'tools\build_sol_fallback_after_luna.py'
$queue = Join-Path $project 'data\luna_remaining_20260803\recovery\continuation_queue.jsonl'
$out = Join-Path $project 'data\luna_remaining_20260803\run_light_retry2_continuation'
New-Item -ItemType Directory -Force -Path $out | Out-Null

& $python $recover
if ($LASTEXITCODE -ne 0) { throw "Recovery exited with code $LASTEXITCODE" }

& $python $runner `
    --input $queue `
    --output-dir $out `
    --run-id 'luna-remaining-light-final-attempt-20260803' `
    --workers 10 `
    --model 'gpt-5.6-luna' `
    --reasoning-effort 'low' `
    --timeout-seconds 1800 `
    --max-attempts 1 `
    --infra-failure-limit 10 `
    --no-resume
if ($LASTEXITCODE -ne 0) { throw "Continuation exited with code $LASTEXITCODE" }

& $python $webBuilder
if ($LASTEXITCODE -ne 0) { throw "Sol web builder exited with code $LASTEXITCODE" }
