param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RunRoot = Join-Path $ProjectRoot "reports\training_dashboard_local"
$PidPath = Join-Path $RunRoot "server.pid"
$serverPid = $null
$active = $false
if (Test-Path -LiteralPath $PidPath) {
    $pidText = (Get-Content -LiteralPath $PidPath -Raw).Trim()
    if ($pidText -match '^\d+$') {
        $serverPid = [int]$pidText
        $active = $null -ne (Get-Process -Id $serverPid -ErrorAction SilentlyContinue)
    }
}

Write-Output "dashboard_pid=$serverPid active=$active url=http://127.0.0.1:$Port/"
if ($active) {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 3 | ConvertTo-Json -Depth 5
    }
    catch {
        Write-Output "health_error=$($_.Exception.Message)"
    }
}
if (Test-Path -LiteralPath (Join-Path $RunRoot "server.stderr.log")) {
    $errors = @(Get-Content -LiteralPath (Join-Path $RunRoot "server.stderr.log") -Tail 5)
    if ($errors.Count) {
        Write-Output "last_stderr:"
        $errors
    }
}
