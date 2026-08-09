$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = (Resolve-Path (Join-Path $projectRoot ".venv\Scripts\python.exe")).Path
$runner = (Resolve-Path (Join-Path $projectRoot "tools\luna_overnight_runner.py")).Path
Set-Location -LiteralPath $projectRoot

foreach ($split in @("valid", "test")) {
    $inputFile = (Resolve-Path (Join-Path $projectRoot "data\final\nemotron_${split}_en_vi_v10_final.jsonl")).Path
    $outputDir = Join-Path $projectRoot "data\luna_overnight\nemotron_${split}_20260802"
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    & $pythonExe -u $runner `
        --input $inputFile `
        --output-dir $outputDir `
        --run-id "nemotron-${split}-luna-20260802" `
        --workers 10 `
        --model "gpt-5.6-luna" `
        --reasoning-effort low `
        --timeout-seconds 600 `
        --max-hours 2 `
        --max-credits 12 `
        --input-credits-per-million 5 `
        --cached-input-credits-per-million 0.5 `
        --output-credits-per-million 30 `
        --infra-failure-limit 3

    $summary = Join-Path $outputDir "summary.json"
    if (Test-Path -LiteralPath $summary) {
        $state = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
        if ($state.stop_reason -eq "infrastructure_circuit_breaker") {
            throw "Stopping eval sequence after infrastructure circuit breaker on split $split"
        }
    }
}
