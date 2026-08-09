$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = (Resolve-Path (Join-Path $projectRoot ".venv\Scripts\python.exe")).Path
$runner = (Resolve-Path (Join-Path $projectRoot "tools\luna_overnight_runner.py")).Path
$inputFile = (Resolve-Path (Join-Path $projectRoot "data\final\nemotron_train_en_vi_v10_final.jsonl")).Path
$outputDir = Join-Path $projectRoot "data\luna_overnight\nemotron_train_20260801"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
Set-Location -LiteralPath $projectRoot

& $pythonExe -u $runner `
    --input $inputFile `
    --output-dir $outputDir `
    --run-id "nemotron-train-luna-20260801" `
    --workers 10 `
    --model "gpt-5.6-luna" `
    --reasoning-effort low `
    --timeout-seconds 600 `
    --max-hours 3 `
    --max-credits 20 `
    --input-credits-per-million 5 `
    --cached-input-credits-per-million 0.5 `
    --output-credits-per-million 30 `
    --infra-failure-limit 3
