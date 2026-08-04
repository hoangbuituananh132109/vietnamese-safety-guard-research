$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PidFile = Join-Path $Root "data\run\full_all_runner.pid"
$RunnerState = Join-Path $Root "data\run\runner_state.json"

$runnerPid = if (Test-Path -LiteralPath $PidFile) {
    (Get-Content -LiteralPath $PidFile -Raw).Trim()
} else { $null }
$active = $runnerPid -and [bool](Get-Process -Id ([int]$runnerPid) -ErrorAction SilentlyContinue)

Write-Output "runner_pid=$runnerPid active=$([bool]$active)"
if (Test-Path -LiteralPath $RunnerState) {
    $state = Get-Content -LiteralPath $RunnerState -Raw | ConvertFrom-Json
    Write-Output "status=$($state.status) updated_at=$($state.updated_at)"
}

foreach ($split in @("train", "valid", "test")) {
    $total = 0
    $rows = @()
    foreach ($group in 1..5) {
        $checkpoint = Join-Path $Root "data\checkpoints\nemotron_${split}_full_v10_g${group}_completed.jsonl"
        $count = if (Test-Path -LiteralPath $checkpoint) {
            ([System.IO.File]::ReadLines($checkpoint) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count
        } else { 0 }
        $total += $count
        $rows += "g${group}=${count}"
    }
    Write-Output "$split completed=$total groups=[$($rows -join ', ')]"
}
