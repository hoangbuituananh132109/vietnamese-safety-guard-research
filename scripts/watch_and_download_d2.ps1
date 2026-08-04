param(
    [string]$SshAlias = "safety-vast",
    [string]$RemoteReport = "/workspace/safety-dataset/reports/d2_nemotron",
    [string]$RemoteMatrix = "/workspace/safety-dataset/reports/evaluation_matrix/D2-NEMOTRON-GUARD-8B-VI-PILOT",
    [string]$LocalRoot = "D:\SafetyDataset_Backups\d2_nemotron_20260722",
    [double]$HardDeadlineHours = 9.0,
    [int]$PollSeconds = 30,
    [switch]$StopInstanceAfterBackup = $true
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Path $LocalRoot -Force | Out-Null
$logPath = Join-Path $LocalRoot "watcher.log"
$started = Get-Date

function Write-WatcherLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    [System.IO.File]::AppendAllText($logPath, $line + [Environment]::NewLine)
}

function Copy-RemoteReport {
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        Write-WatcherLog "backup attempt=$attempt"
        & scp -q -r -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 `
            "${SshAlias}:${RemoteReport}" $LocalRoot
        if ($LASTEXITCODE -eq 0) {
            Write-WatcherLog "backup succeeded"
            return $true
        }
        Start-Sleep -Seconds 30
    }
    Write-WatcherLog "backup failed after retries"
    return $false
}

function Copy-RemoteMatrix {
    $matrixRoot = Join-Path $LocalRoot "evaluation_matrix"
    New-Item -ItemType Directory -Path $matrixRoot -Force | Out-Null
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        Write-WatcherLog "matrix backup attempt=$attempt"
        & scp -q -r -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 `
            "${SshAlias}:${RemoteMatrix}" $matrixRoot
        if ($LASTEXITCODE -eq 0) {
            Write-WatcherLog "matrix backup succeeded"
            return $true
        }
        Start-Sleep -Seconds 30
    }
    Write-WatcherLog "matrix backup failed after retries"
    return $false
}

function Stop-RemoteInstance {
    if (-not $StopInstanceAfterBackup) {
        return
    }
    Write-WatcherLog "requesting reversible Vast stop"
    & ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 `
        -o ServerAliveCountMax=2 $SshAlias `
        'vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY'
    Write-WatcherLog "Vast stop exit=$LASTEXITCODE"
}

Write-WatcherLog "watcher started deadline_hours=$HardDeadlineHours"
while ($true) {
    $elapsed = ((Get-Date) - $started).TotalHours
    $stateText = & ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 `
        -o ServerAliveCountMax=2 $SshAlias "cat ${RemoteReport}/state.json 2>/dev/null" 2>$null
    if ($LASTEXITCODE -eq 0 -and $stateText) {
        [System.IO.File]::WriteAllText(
            (Join-Path $LocalRoot "latest_remote_state.json"),
            ($stateText -join [Environment]::NewLine) + [Environment]::NewLine
        )
        try {
            $state = ($stateText -join [Environment]::NewLine) | ConvertFrom-Json
            Write-WatcherLog "status=$($state.status) stage=$($state.stage) elapsed_hours=$([math]::Round($elapsed,3))"
            if ($state.status -eq "failed") {
                Write-WatcherLog "failure grace period started (300 seconds)"
                Start-Sleep -Seconds 300
                $recheckText = & ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 `
                    -o ServerAliveCountMax=2 $SshAlias "cat ${RemoteReport}/state.json 2>/dev/null" 2>$null
                try {
                    $recheck = ($recheckText -join [Environment]::NewLine) | ConvertFrom-Json
                    if ($recheck.status -ne "failed") {
                        Write-WatcherLog "failure was repaired; resumed status=$($recheck.status)"
                        continue
                    }
                } catch {
                    Write-WatcherLog "failed-state recheck could not be parsed"
                }
            }
            if ($state.status -in @("complete", "failed")) {
                if ($state.status -eq "complete") {
                    Write-WatcherLog "building D2 benchmark matrix before backup"
                    & ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 `
                        -o ServerAliveCountMax=2 $SshAlias `
                        "cd /workspace/safety-dataset && /workspace/venvs/nemotron-vllm/bin/python scripts/split_decoder_full_matrix.py --combined-dir reports/d2_nemotron/full_eval --run-id D2-NEMOTRON-GUARD-8B-VI-PILOT"
                    Write-WatcherLog "D2 matrix exit=$LASTEXITCODE"
                }
                $copied = Copy-RemoteReport
                if ($state.status -eq "complete") {
                    $matrixCopied = Copy-RemoteMatrix
                    $copied = $copied -and $matrixCopied
                }
                [System.Media.SystemSounds]::Exclamation.Play()
                Stop-RemoteInstance
                exit $(if ($copied) { 0 } else { 2 })
            }
        } catch {
            Write-WatcherLog "state parse error=$($_.Exception.Message)"
        }
    }
    if ($elapsed -ge $HardDeadlineHours) {
        Write-WatcherLog "hard deadline reached; stopping supervisor job"
        & ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 `
            -o ServerAliveCountMax=2 $SshAlias "supervisorctl stop d2_nemotron" | Out-Null
        $copied = Copy-RemoteReport
        [System.Media.SystemSounds]::Hand.Play()
        Stop-RemoteInstance
        exit $(if ($copied) { 3 } else { 4 })
    }
    Start-Sleep -Seconds $PollSeconds
}
