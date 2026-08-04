param(
    [string]$HostAlias = "safety-vast",
    [int]$PollSeconds = 60,
    [int]$StaleSeconds = 240,
    [int]$ReminderMinutes = 10,
    [switch]$VisibleAlertMonitor
)

$ErrorActionPreference = "Stop"
$RemoteRoot = "/workspace/safety-dataset"
$TmuxSession = "overnight_phase0"
$remoteCommand = "cd '$RemoteRoot' && bash scripts/vast/start_overnight_phase0_remote.sh"

Write-Output "Starting the remote Phase-0 runner..."
$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & ssh -o BatchMode=yes -o ConnectTimeout=15 $HostAlias $remoteCommand
    $sshExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $savedErrorActionPreference
}
if ($sshExitCode -ne 0) {
    throw "Remote Phase-0 launcher failed with SSH exit code $sshExitCode."
}

$alertArguments = @{
    HostAlias = $HostAlias
    RemoteRoot = $RemoteRoot
    StateRelativePath = "reports/overnight_phase0/state.json"
    TmuxSession = $TmuxSession
    PollSeconds = $PollSeconds
    StaleSeconds = $StaleSeconds
    ReminderMinutes = $ReminderMinutes
}
if ($VisibleAlertMonitor) { $alertArguments["Visible"] = $true }

& (Join-Path $PSScriptRoot "start_vast_training_alert.ps1") @alertArguments
if ($LASTEXITCODE -ne 0) { throw "Remote runner started, but the local alert monitor did not." }

Write-Output "Phase-0 runner and local sound alerts are both active."
