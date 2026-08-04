param(
    [string]$HostAlias = "safety-vast",
    [string]$RemoteRoot = "/workspace/safety-dataset",
    [string]$StateRelativePath = "reports/overnight_phase0/state.json",
    [string]$TmuxSession = "phase0_full",
    [ValidateRange(15, 3600)]
    [int]$PollSeconds = 60,
    [ValidateRange(60, 86400)]
    [int]$StaleSeconds = 240,
    [ValidateRange(1, 20)]
    [int]$SshFailureThreshold = 3,
    [ValidateRange(1, 1440)]
    [int]$ReminderMinutes = 10,
    [switch]$Once,
    [switch]$NotifyOnStart,
    [switch]$NoSound,
    [switch]$NoBalloon
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RunRoot = Join-Path $ProjectRoot "reports\vast_alert"
$PidPath = Join-Path $RunRoot "monitor.pid"
$HeartbeatPath = Join-Path $RunRoot "heartbeat.json"
$EventLogPath = Join-Path $RunRoot "events.jsonl"
$SshStderrPath = Join-Path $RunRoot "last_ssh_stderr.log"
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
Set-Content -LiteralPath $PidPath -Value $PID -Encoding ascii

function Write-AlertEvent {
    param([string]$Event, [hashtable]$Payload = @{})
    $row = [ordered]@{
        at = [DateTimeOffset]::Now.ToString("o")
        event = $Event
        monitor_pid = $PID
    }
    foreach ($key in $Payload.Keys) { $row[$key] = $Payload[$key] }
    Add-Content -LiteralPath $EventLogPath -Value ($row | ConvertTo-Json -Compress -Depth 8) -Encoding utf8
}

function Show-TrainingBalloon {
    param([string]$Title, [string]$Message, [string]$Level)
    if ($NoBalloon) { return }
    try {
        Add-Type -AssemblyName System.Windows.Forms
        Add-Type -AssemblyName System.Drawing
        $notify = New-Object System.Windows.Forms.NotifyIcon
        $notify.Visible = $true
        switch ($Level) {
            "critical" {
                $notify.Icon = [System.Drawing.SystemIcons]::Error
                $notify.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Error
            }
            "warning" {
                $notify.Icon = [System.Drawing.SystemIcons]::Warning
                $notify.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Warning
            }
            default {
                $notify.Icon = [System.Drawing.SystemIcons]::Information
                $notify.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Info
            }
        }
        $notify.BalloonTipTitle = $Title
        $notify.BalloonTipText = $Message
        $notify.ShowBalloonTip(10000)
        Start-Sleep -Milliseconds 2500
        $notify.Dispose()
    }
    catch {
        Write-AlertEvent -Event "balloon_failed" -Payload @{ error = $_.Exception.Message }
    }
}

function Play-TrainingSound {
    param([string]$Level)
    if ($NoSound) { return }
    try {
        switch ($Level) {
            "critical" {
                1..3 | ForEach-Object {
                    [System.Media.SystemSounds]::Hand.Play()
                    try { [Console]::Beep(1050, 450) } catch {}
                    Start-Sleep -Milliseconds 300
                }
            }
            "warning" {
                1..2 | ForEach-Object {
                    [System.Media.SystemSounds]::Exclamation.Play()
                    try { [Console]::Beep(850, 300) } catch {}
                    Start-Sleep -Milliseconds 250
                }
            }
            "complete" {
                foreach ($frequency in @(660, 880, 1100)) {
                    [System.Media.SystemSounds]::Asterisk.Play()
                    try { [Console]::Beep($frequency, 250) } catch {}
                    Start-Sleep -Milliseconds 120
                }
            }
            default {
                [System.Media.SystemSounds]::Asterisk.Play()
                try { [Console]::Beep(700, 180) } catch {}
            }
        }
    }
    catch {
        Write-AlertEvent -Event "sound_failed" -Payload @{ error = $_.Exception.Message }
    }
}

function Send-TrainingAlert {
    param(
        [string]$Condition,
        [string]$Title,
        [string]$Message,
        [string]$Level
    )
    Write-AlertEvent -Event "alert" -Payload @{
        condition = $Condition
        title = $Title
        message = $Message
        level = $Level
    }
    Play-TrainingSound -Level $Level
    Show-TrainingBalloon -Title $Title -Message $Message -Level $Level
}

function Write-Heartbeat {
    param([string]$Condition, [object]$Probe, [int]$SshFailures)
    $heartbeat = [ordered]@{
        monitor_pid = $PID
        local_time = [DateTimeOffset]::Now.ToString("o")
        condition = $Condition
        consecutive_ssh_failures = $SshFailures
        host_alias = $HostAlias
        tmux_session = $TmuxSession
        probe = $Probe
    }
    $temporary = "$HeartbeatPath.tmp"
    Set-Content -LiteralPath $temporary -Value ($heartbeat | ConvertTo-Json -Depth 10) -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $HeartbeatPath -Force
}

$sshFailures = 0
$lastCondition = $null
$lastAlertAt = [DateTimeOffset]::MinValue
$reminderInterval = [TimeSpan]::FromMinutes($ReminderMinutes)

Write-AlertEvent -Event "monitor_started" -Payload @{
    host_alias = $HostAlias
    poll_seconds = $PollSeconds
    stale_seconds = $StaleSeconds
    tmux_session = $TmuxSession
}
if ($NotifyOnStart) {
    Send-TrainingAlert -Condition "monitor_started" `
        -Title "Vast training monitor started" `
        -Message "Watching $TmuxSession on $HostAlias every $PollSeconds seconds." `
        -Level "info"
}

try {
    while ($true) {
        $remoteCommand = "cd '$RemoteRoot' && /venv/main/bin/python scripts/vast/remote_alert_probe.py --root '$RemoteRoot' --state '$StateRelativePath' --tmux '$TmuxSession'"
        $savedErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $stdout = @(& ssh -o BatchMode=yes -o ConnectTimeout=15 $HostAlias $remoteCommand 2> $SshStderrPath)
            $sshExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $savedErrorActionPreference
        }
        $stderr = if (Test-Path -LiteralPath $SshStderrPath) {
            @(Get-Content -LiteralPath $SshStderrPath -ErrorAction SilentlyContinue)
        }
        else { @() }
        $raw = @($stdout) + @($stderr)
        $probeLine = @($stdout | ForEach-Object { $_.ToString() } | Where-Object { $_.StartsWith("ALERT_STATE=") } | Select-Object -Last 1)

        if ($sshExitCode -ne 0 -or $probeLine.Count -eq 0) {
            $sshFailures += 1
            $condition = if ($sshFailures -ge $SshFailureThreshold) { "ssh_unreachable" } else { "ssh_transient" }
            $details = ($raw | ForEach-Object { $_.ToString() } | Select-Object -Last 4) -join " | "
            Write-Heartbeat -Condition $condition -Probe @{ ssh_error = $details } -SshFailures $sshFailures
            Write-AlertEvent -Event "probe_failed" -Payload @{
                consecutive_failures = $sshFailures
                ssh_exit_code = $sshExitCode
                details = $details
            }
            if ($condition -eq "ssh_unreachable") {
                $now = [DateTimeOffset]::Now
                if ($lastCondition -ne $condition -or ($now - $lastAlertAt) -ge $reminderInterval) {
                    Send-TrainingAlert -Condition $condition `
                        -Title "Vast.ai connection lost" `
                        -Message "Could not reach $HostAlias for $sshFailures consecutive probes. Check the network and instance." `
                        -Level "critical"
                    $lastAlertAt = $now
                }
                $lastCondition = $condition
            }
            if ($Once) { break }
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        $sshFailures = 0
        $json = $probeLine[-1].Substring("ALERT_STATE=".Length)
        $probe = $json | ConvertFrom-Json
        $stateStatus = [string]$probe.state_status
        $stage = [string]$probe.current_stage
        $runId = [string]$probe.current_run_id
        $age = if ($null -eq $probe.updated_age_seconds) { $null } else { [double]$probe.updated_age_seconds }
        $condition = "healthy"
        $title = ""
        $message = ""
        $level = "info"

        if (-not [bool]$probe.state_exists -or $probe.state_read_error) {
            $condition = "state_unreadable"
            $title = "Training state is unreadable"
            $message = "The state on $HostAlias is missing or invalid: $($probe.state_read_error)"
            $level = "critical"
        }
        elseif ($stateStatus -eq "completed") {
            $condition = "completed"
            $title = "All training stages completed"
            $message = "Runner $TmuxSession completed every stage on $HostAlias."
            $level = "complete"
        }
        elseif ($stateStatus -in @("failed", "stopped_incomplete")) {
            $condition = "failed"
            $title = "Training stopped with an error"
            $message = "Stage=$stage, run=$runId. $($probe.error)"
            $level = "critical"
        }
        elseif ($stateStatus -in @("running", "waiting", "retry_wait", "initializing")) {
            if ($null -ne $age -and $age -gt $StaleSeconds) {
                $condition = "state_stale"
                $title = "Training may be stalled"
                $message = "State has not updated for $([math]::Round($age)) seconds; stage=$stage, run=$runId."
                $level = "warning"
            }
            elseif (-not [bool]$probe.tmux_active -and -not [bool]$probe.runner_active) {
                $condition = "runner_stopped"
                $title = "Training runner disappeared"
                $message = "State is still $stateStatus, but tmux and the runner process are gone; stage=$stage."
                $level = "critical"
            }
        }
        else {
            $condition = "unexpected_status"
            $title = "Training has an unexpected status"
            $message = "state_status='$stateStatus', stage=$stage, run=$runId."
            $level = "warning"
        }

        Write-Heartbeat -Condition $condition -Probe $probe -SshFailures 0
        Write-AlertEvent -Event "probe" -Payload @{
            condition = $condition
            state_status = $stateStatus
            stage = $stage
            run_id = $runId
            updated_age_seconds = $age
        }

        $now = [DateTimeOffset]::Now
        if ($condition -eq "healthy") {
            if ($null -ne $lastCondition -and $lastCondition -notin @("healthy", "ssh_transient", "monitor_started")) {
                Send-TrainingAlert -Condition "recovered" `
                    -Title "Training recovered" `
                    -Message "Runner is active again; stage=$stage, run=$runId." `
                    -Level "info"
                $lastAlertAt = $now
            }
        }
        elseif ($condition -eq "completed") {
            if ($lastCondition -ne "completed") {
                Send-TrainingAlert -Condition $condition -Title $title -Message $message -Level $level
            }
            $lastCondition = $condition
            break
        }
        elseif ($condition -ne "ssh_transient") {
            if ($lastCondition -ne $condition -or ($now - $lastAlertAt) -ge $reminderInterval) {
                Send-TrainingAlert -Condition $condition -Title $title -Message $message -Level $level
                $lastAlertAt = $now
            }
        }
        $lastCondition = $condition
        if ($Once) { break }
        Start-Sleep -Seconds $PollSeconds
    }
}
catch {
    Write-AlertEvent -Event "monitor_crashed" -Payload @{
        error = $_.Exception.Message
        stack = $_.ScriptStackTrace
    }
    Send-TrainingAlert -Condition "monitor_crashed" `
        -Title "Vast monitor crashed" `
        -Message $_.Exception.Message `
        -Level "critical"
    throw
}
finally {
    Write-AlertEvent -Event "monitor_stopped" -Payload @{ last_condition = $lastCondition }
    if (Test-Path -LiteralPath $PidPath) {
        $recordedPid = (Get-Content -LiteralPath $PidPath -Raw).Trim()
        if ($recordedPid -eq [string]$PID) { Remove-Item -LiteralPath $PidPath -Force }
    }
}
