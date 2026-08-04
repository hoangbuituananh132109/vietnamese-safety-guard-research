param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765,
    [string]$HostAlias = "root@ssh8.vast.ai",
    [ValidateRange(1, 65535)]
    [int]$SshPort = 15873,
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$SnapshotScript = "scripts/vast/remote_no_r_decoder_dashboard_snapshot.py",
    [ValidateRange(5, 300)]
    [int]$RemoteRefreshSeconds = 15,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$DashboardRoot = Join-Path $ProjectRoot "training_dashboard"
$RunRoot = Join-Path $ProjectRoot "reports\training_dashboard_local"
$PidPath = Join-Path $RunRoot "server.pid"
$StdoutPath = Join-Path $RunRoot "server.stdout.log"
$StderrPath = Join-Path $RunRoot "server.stderr.log"
$ServerPath = Join-Path $DashboardRoot "server.js"
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null

# Codex Desktop may inject both `PATH` and `Path` into the process environment.
# Windows accepts that block, but PowerShell Start-Process materializes it into
# a case-insensitive dictionary and fails with a duplicate-key exception.
# Normalize only this short-lived script process; machine/user variables remain
# untouched and the child receives the same effective path value.
$effectivePath = $env:Path
[Environment]::SetEnvironmentVariable(
    "PATH",
    $null,
    [EnvironmentVariableTarget]::Process
)
[Environment]::SetEnvironmentVariable(
    "Path",
    $effectivePath,
    [EnvironmentVariableTarget]::Process
)

if (Test-Path -LiteralPath $PidPath) {
    $existingPidText = (Get-Content -LiteralPath $PidPath -Raw).Trim()
    if ($existingPidText -match '^\d+$') {
        $existing = Get-Process -Id ([int]$existingPidText) -ErrorAction SilentlyContinue
        if ($null -ne $existing) {
            $url = "http://127.0.0.1:$Port/"
            Write-Output "Training dashboard is already running (PID=$existingPidText)."
            Write-Output "URL: $url"
            if (-not $NoOpen) { Start-Process $url }
            exit 0
        }
    }
    Remove-Item -LiteralPath $PidPath -Force
}

$environmentBackup = @{
    DASHBOARD_PORT = $env:DASHBOARD_PORT
    VAST_SSH_HOST = $env:VAST_SSH_HOST
    VAST_SSH_PORT = $env:VAST_SSH_PORT
    VAST_SSH_IDENTITY = $env:VAST_SSH_IDENTITY
    VAST_SNAPSHOT_SCRIPT = $env:VAST_SNAPSHOT_SCRIPT
    DASHBOARD_REFRESH_MS = $env:DASHBOARD_REFRESH_MS
}
$env:DASHBOARD_PORT = [string]$Port
$env:VAST_SSH_HOST = $HostAlias
$env:VAST_SSH_PORT = [string]$SshPort
$env:VAST_SSH_IDENTITY = $IdentityFile
$env:VAST_SNAPSHOT_SCRIPT = $SnapshotScript
$env:DASHBOARD_REFRESH_MS = [string]($RemoteRefreshSeconds * 1000)
try {
    $process = Start-Process -FilePath "node.exe" `
        -ArgumentList "`"$ServerPath`"" `
        -WorkingDirectory $DashboardRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -PassThru
}
finally {
    $env:DASHBOARD_PORT = $environmentBackup.DASHBOARD_PORT
    $env:VAST_SSH_HOST = $environmentBackup.VAST_SSH_HOST
    $env:VAST_SSH_PORT = $environmentBackup.VAST_SSH_PORT
    $env:VAST_SSH_IDENTITY = $environmentBackup.VAST_SSH_IDENTITY
    $env:VAST_SNAPSHOT_SCRIPT = $environmentBackup.VAST_SNAPSHOT_SCRIPT
    $env:DASHBOARD_REFRESH_MS = $environmentBackup.DASHBOARD_REFRESH_MS
}
Set-Content -LiteralPath $PidPath -Value $process.Id -Encoding ascii

$url = "http://127.0.0.1:$Port/"
$healthy = $false
for ($attempt = 1; $attempt -le 20; $attempt++) {
    if ($process.HasExited) { break }
    try {
        $health = Invoke-RestMethod -Uri "${url}api/health" -TimeoutSec 2
        if ($health.ok) { $healthy = $true; break }
    }
    catch {}
    Start-Sleep -Milliseconds 500
}

if (-not $healthy) {
    $stderr = if (Test-Path -LiteralPath $StderrPath) { Get-Content -LiteralPath $StderrPath -Tail 20 } else { @() }
    throw "Dashboard failed to become healthy. PID=$($process.Id). $($stderr -join ' | ')"
}

Write-Output "Started training dashboard PID=$($process.Id)"
Write-Output "URL: $url"
Write-Output "Vast refresh: every $RemoteRefreshSeconds seconds via ${HostAlias}:$SshPort"
Write-Output "Snapshot: $SnapshotScript"
Write-Output "Logs: $RunRoot"
if (-not $NoOpen) { Start-Process $url }
