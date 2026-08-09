$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project '.venv\Scripts\python.exe'
$builder = Join-Path $project 'tools\build_sol_fallback_after_luna.py'
$continuation = Join-Path $project 'data\luna_remaining_20260803\run_light_retry2_continuation'
$summary = Join-Path $continuation 'summary.json'
$watchLog = Join-Path $continuation 'web_append_watcher.log'

for ($iteration = 1; $iteration -le 240; $iteration++) {
    if (Test-Path -LiteralPath $summary) {
        & $python $builder | Out-File -LiteralPath $watchLog -Append -Encoding utf8
        if ($LASTEXITCODE -ne 0) { throw "Incremental web builder exited with code $LASTEXITCODE" }
        break
    }
    "$(Get-Date -Format o) waiting for Luna continuation to finish before packing failures" |
        Out-File -LiteralPath $watchLog -Append -Encoding utf8
    Start-Sleep -Seconds 20
}
