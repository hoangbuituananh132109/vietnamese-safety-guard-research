param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DashboardRoot = Join-Path $ProjectRoot "training_dashboard"
$RunRoot = Join-Path $ProjectRoot "reports\training_dashboard_local"
$PidPath = Join-Path $RunRoot "server.pid"
$StdoutPath = Join-Path $RunRoot "server.stdout.log"
$StderrPath = Join-Path $RunRoot "server.stderr.log"
$ServerPath = Join-Path $DashboardRoot "server.js"
$HealthUrl = "http://127.0.0.1:$Port/api/health"
$AtlasUrl = "http://127.0.0.1:$Port/safety-explorer/"
$ReviewUrl = "http://127.0.0.1:$Port/v1-review/"
$RagUrl = "http://127.0.0.1:$Port/rag-verifier/"
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null

try {
    $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
    if ($health.ok) {
        Write-Output "Atlas already running."
        Write-Output "Atlas: $AtlasUrl"
        Write-Output "Pilot V1 review: $ReviewUrl"
        Write-Output "RAG verification: $RagUrl"
        if (-not $NoOpen) { Start-Process $ReviewUrl }
        exit 0
    }
}
catch {}

if (Test-Path -LiteralPath $PidPath) {
    $existingPidText = (Get-Content -LiteralPath $PidPath -Raw).Trim()
    if ($existingPidText -match '^\d+$') {
        $existing = Get-Process -Id ([int]$existingPidText) -ErrorAction SilentlyContinue
        if ($null -ne $existing) {
            throw "PID $existingPidText is still active but Atlas is not responding. Refusing to stop an unidentified process."
        }
    }
    Remove-Item -LiteralPath $PidPath -Force
}

$effectivePath = $env:Path
[Environment]::SetEnvironmentVariable("PATH", $null, [EnvironmentVariableTarget]::Process)
[Environment]::SetEnvironmentVariable("Path", $effectivePath, [EnvironmentVariableTarget]::Process)
$oldPort = $env:DASHBOARD_PORT
$oldRemote = $env:DASHBOARD_REMOTE_DISABLED
$env:DASHBOARD_PORT = [string]$Port
$env:DASHBOARD_REMOTE_DISABLED = "1"
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
    $env:DASHBOARD_PORT = $oldPort
    $env:DASHBOARD_REMOTE_DISABLED = $oldRemote
}
Set-Content -LiteralPath $PidPath -Value $process.Id -Encoding ascii

$healthy = $false
for ($attempt = 1; $attempt -le 20; $attempt++) {
    if ($process.HasExited) { break }
    try {
        $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
        if ($health.ok -and $health.v1ReviewQueue) {
            $healthy = $true
            break
        }
    }
    catch {}
    Start-Sleep -Milliseconds 400
}

if (-not $healthy) {
    $stderr = if (Test-Path -LiteralPath $StderrPath) {
        @(Get-Content -LiteralPath $StderrPath -Tail 20)
    }
    else {
        @()
    }
    throw "Atlas failed to start. PID=$($process.Id). $($stderr -join ' | ')"
}

Write-Output "Started Atlas PID=$($process.Id)"
Write-Output "Atlas: $AtlasUrl"
Write-Output "Pilot V1 review: $ReviewUrl"
Write-Output "RAG verification: $RagUrl"
Write-Output "Logs: $RunRoot"
if (-not $NoOpen) { Start-Process $ReviewUrl }
