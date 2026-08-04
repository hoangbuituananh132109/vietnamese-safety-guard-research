$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunDir = Join-Path $Root "data\run"
$PidFile = Join-Path $RunDir "dashboard.pid"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

try {
    $live = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/status" -TimeoutSec 2
    if ($live.dashboard_pid) {
        Set-Content -LiteralPath $PidFile -Value ([int]$live.dashboard_pid) -Encoding ascii
    }
    Write-Output "Dashboard already running: http://127.0.0.1:8765"
    Write-Output "Review Studio: http://127.0.0.1:8765/review"
    exit 0
} catch {
    # No healthy server is listening; continue with normal startup.
}

if (Test-Path -LiteralPath $PidFile) {
    $oldPid = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    if ($oldPid -match '^\d+$' -and (Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue)) {
        Write-Output "Dashboard already running: http://127.0.0.1:8765"
        Write-Output "Review Studio: http://127.0.0.1:8765/review"
        exit 0
    }
}

$arguments = "-m translator.dashboard --root `"$Root`" --host 127.0.0.1 --port 8765"
$info = [System.Diagnostics.ProcessStartInfo]::new()
$info.FileName = $Python
$info.Arguments = $arguments
$info.WorkingDirectory = $Root
$info.UseShellExecute = $true
$info.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$process = [System.Diagnostics.Process]::Start($info)
Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ascii
Write-Output "Started dashboard PID=$($process.Id)"
Write-Output "Open: http://127.0.0.1:8765"
Write-Output "Review Studio: http://127.0.0.1:8765/review"
