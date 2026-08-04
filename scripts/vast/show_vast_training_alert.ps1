$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RunRoot = Join-Path $ProjectRoot "reports\vast_alert"
$PidPath = Join-Path $RunRoot "monitor.pid"
$HeartbeatPath = Join-Path $RunRoot "heartbeat.json"
$EventsPath = Join-Path $RunRoot "events.jsonl"

$monitorPid = $null
$alive = $false
if (Test-Path -LiteralPath $PidPath) {
    $pidText = (Get-Content -LiteralPath $PidPath -Raw).Trim()
    if ($pidText -match '^\d+$') {
        $monitorPid = [int]$pidText
        $alive = $null -ne (Get-Process -Id $monitorPid -ErrorAction SilentlyContinue)
    }
}

Write-Output "monitor_pid=$monitorPid active=$alive"
if (Test-Path -LiteralPath $HeartbeatPath) {
    Get-Content -LiteralPath $HeartbeatPath -Raw
}
else {
    Write-Output "heartbeat=missing"
}
if (Test-Path -LiteralPath $EventsPath) {
    Write-Output "last_events:"
    $events = @(Get-Content -LiteralPath $EventsPath | ForEach-Object {
        try { $_ | ConvertFrom-Json } catch { $null }
    })
    if ($null -ne $monitorPid) {
        $events = @($events | Where-Object { $_.monitor_pid -eq $monitorPid })
    }
    $events | Select-Object -Last 8 | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 8 }
}
