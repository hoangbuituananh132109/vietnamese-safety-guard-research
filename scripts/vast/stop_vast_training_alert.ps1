$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PidPath = Join-Path $ProjectRoot "reports\vast_alert\monitor.pid"

if (-not (Test-Path -LiteralPath $PidPath)) {
    Write-Output "Vast training alert monitor is not running (no PID file)."
    exit 0
}

$pidText = (Get-Content -LiteralPath $PidPath -Raw).Trim()
if ($pidText -notmatch '^\d+$') { throw "Invalid monitor PID file: $PidPath" }
$monitorPid = [int]$pidText
$processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $monitorPid" -ErrorAction SilentlyContinue
if ($null -eq $processInfo) {
    Remove-Item -LiteralPath $PidPath -Force
    Write-Output "Removed stale monitor PID file; process $monitorPid was not running."
    exit 0
}
if ([string]$processInfo.CommandLine -notlike '*watch_vast_training_alert.ps1*') {
    throw "Refusing to stop PID $monitorPid because it is not the Vast alert watcher."
}

Stop-Process -Id $monitorPid
Remove-Item -LiteralPath $PidPath -Force -ErrorAction SilentlyContinue
Write-Output "Stopped Vast training alert monitor PID=$monitorPid"
