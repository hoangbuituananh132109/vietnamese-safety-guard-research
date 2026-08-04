$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StartScript = Join-Path $PSScriptRoot "start_dashboard.ps1"

try {
    $LiveDashboard = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/status" -TimeoutSec 3
} catch {
    Write-Output "No running dashboard detected."
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $StartScript
    exit $LASTEXITCODE
}

$DashboardPid = [int]$LiveDashboard.dashboard_pid
$DashboardProcess = Get-Process -Id $DashboardPid -ErrorAction Stop
if ($DashboardProcess.ProcessName -notlike "python*") {
    throw "Port 8765 returned a non-Python process; refusing to stop PID $DashboardPid."
}
Stop-Process -Id $DashboardPid -Force
Start-Sleep -Milliseconds 700
Write-Output "Stopped old dashboard PID=$DashboardPid"

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $StartScript
