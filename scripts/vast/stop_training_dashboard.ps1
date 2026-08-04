$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PidPath = Join-Path $ProjectRoot "reports\training_dashboard_local\server.pid"

if (-not (Test-Path -LiteralPath $PidPath)) {
    Write-Output "Training dashboard is not running (no PID file)."
    exit 0
}

$pidText = (Get-Content -LiteralPath $PidPath -Raw).Trim()
if ($pidText -notmatch '^\d+$') { throw "Invalid dashboard PID file: $PidPath" }
$serverPid = [int]$pidText
$processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $serverPid" -ErrorAction SilentlyContinue
if ($null -eq $processInfo) {
    Remove-Item -LiteralPath $PidPath -Force
    Write-Output "Removed stale PID file; process $serverPid was not running."
    exit 0
}
if ([string]$processInfo.CommandLine -notlike '*training_dashboard*server.js*') {
    throw "Refusing to stop PID $serverPid because it is not the training dashboard server."
}

Stop-Process -Id $serverPid
Remove-Item -LiteralPath $PidPath -Force -ErrorAction SilentlyContinue
Write-Output "Stopped training dashboard PID=$serverPid"
