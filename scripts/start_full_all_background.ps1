$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunDir = Join-Path $Root "data\run"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$PidFile = Join-Path $RunDir "full_all_runner.pid"

if (Test-Path -LiteralPath $PidFile) {
    $oldPid = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    if ($oldPid -match '^\d+$' -and (Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue)) {
        Write-Output "Runner is already active with PID $oldPid"
        exit 0
    }
}

$runner = Join-Path $PSScriptRoot "run_full_all_background.ps1"
$stdout = Join-Path $RunDir "launcher_stdout.log"
$stderr = Join-Path $RunDir "launcher_stderr.log"
$process = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -PassThru `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$runner`"") `
    -WorkingDirectory $Root -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ascii
Write-Output "Started full translation runner PID=$($process.Id)"
Write-Output "State: $RunDir\runner_state.json"
Write-Output "Log: $Root\reports\full_run\runner.log"
