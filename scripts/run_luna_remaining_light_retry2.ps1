$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project '.venv\Scripts\python.exe'
$runner = Join-Path $project 'tools\luna_overnight_runner.py'
$queue = Join-Path $project 'data\luna_remaining_20260803\queue.jsonl'
$out = Join-Path $project 'data\luna_remaining_20260803\run_light_retry2'
$webBuilder = Join-Path $project 'tools\build_sol_fallback_after_luna.py'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$runnerLog = Join-Path $out 'runner.live.stdout.log'
$runnerErrorLog = Join-Path $out 'runner.live.stderr.log'

for ($pass = 1; $pass -le 20; $pass++) {
    & $python $runner `
        --input $queue `
        --output-dir $out `
        --run-id 'luna-remaining-light-retry2-20260803' `
        --workers 10 `
        --model 'gpt-5.6-luna' `
        --reasoning-effort 'low' `
        --timeout-seconds 1800 `
        --max-attempts 2 `
        --infra-failure-limit 10 2>> $runnerErrorLog | Tee-Object -FilePath $runnerLog -Append
    if ($LASTEXITCODE -ne 0) { throw "Luna runner exited with code $LASTEXITCODE" }
    $summary = Get-Content -LiteralPath (Join-Path $out 'summary.json') -Raw | ConvertFrom-Json
    if ([int]$summary.pending_records -eq 0) { break }
    Start-Sleep -Seconds 30
}

& $python $webBuilder
if ($LASTEXITCODE -ne 0) { throw "Sol web builder exited with code $LASTEXITCODE" }
