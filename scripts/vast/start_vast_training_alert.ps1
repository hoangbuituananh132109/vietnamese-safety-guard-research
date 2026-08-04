param(
    [string]$HostAlias = "safety-vast",
    [string]$RemoteRoot = "/workspace/safety-dataset",
    [string]$StateRelativePath = "reports/overnight_phase0/state.json",
    [string]$TmuxSession = "phase0_full",
    [int]$PollSeconds = 60,
    [int]$StaleSeconds = 240,
    [int]$ReminderMinutes = 10,
    [switch]$Visible
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RunRoot = Join-Path $ProjectRoot "reports\vast_alert"
$PidPath = Join-Path $RunRoot "monitor.pid"
$Watcher = Join-Path $PSScriptRoot "watch_vast_training_alert.ps1"
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null

if (Test-Path -LiteralPath $PidPath) {
    $existingPidText = (Get-Content -LiteralPath $PidPath -Raw).Trim()
    if ($existingPidText -match '^\d+$') {
        $existing = Get-Process -Id ([int]$existingPidText) -ErrorAction SilentlyContinue
        if ($null -ne $existing) {
            Write-Output "Vast training alert monitor is already running (PID=$existingPidText)."
            Write-Output "Status: $RunRoot\heartbeat.json"
            exit 0
        }
    }
    Remove-Item -LiteralPath $PidPath -Force
}

$argumentString = @(
    '-NoLogo'
    '-NoProfile'
    '-Sta'
    '-ExecutionPolicy Bypass'
    "-File `"$Watcher`""
    "-HostAlias `"$HostAlias`""
    "-RemoteRoot `"$RemoteRoot`""
    "-StateRelativePath `"$StateRelativePath`""
    "-TmuxSession `"$TmuxSession`""
    "-PollSeconds $PollSeconds"
    "-StaleSeconds $StaleSeconds"
    "-ReminderMinutes $ReminderMinutes"
    '-NotifyOnStart'
) -join ' '

$windowStyle = if ($Visible) { 'Normal' } else { 'Hidden' }
$process = Start-Process -FilePath "powershell.exe" `
    -ArgumentList $argumentString `
    -WindowStyle $windowStyle `
    -PassThru

Start-Sleep -Seconds 2
if ($process.HasExited) {
    throw "Vast training alert monitor exited during startup with code $($process.ExitCode)."
}

Write-Output "Started Vast training alert monitor PID=$($process.Id)"
Write-Output "State: $RemoteRoot/$StateRelativePath"
Write-Output "Polling: every $PollSeconds seconds; stale after $StaleSeconds seconds."
Write-Output "Failure reminder: every $ReminderMinutes minutes."
Write-Output "Heartbeat: $RunRoot\heartbeat.json"
Write-Output "Events: $RunRoot\events.jsonl"
Write-Output "Stop: powershell.exe -NoProfile -ExecutionPolicy Bypass -File `".\scripts\vast\stop_vast_training_alert.ps1`""
