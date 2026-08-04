param(
    [string]$HostName = "ssh8.vast.ai",
    [int]$Port = 15873,
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$Destination = "D:\Downloads\Safety Dataset\reports\vast_download\d3_nemotron_no_r_4080s_20260724",
    [int]$PollSeconds = 20,
    [int]$MaxTransferAttempts = 5
)

$ErrorActionPreference = "Stop"
$remoteRoot = "/workspace/safety-dataset"
$remoteReady = "$remoteRoot/exports/D3_NEMOTRON_NO_R_4080S_20260724.ready.json"
$remoteAck = "$remoteRoot/exports/D3_NEMOTRON_NO_R_4080S_20260724.download_verified.json"
$statePath = Join-Path $Destination "local_handoff_state.json"
$logPath = Join-Path $Destination "local_handoff.log"

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class SleepControl {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS = [Convert]::ToUInt32("80000000", 16)
$ES_SYSTEM_REQUIRED = [uint32]0x00000001

function Set-Awake {
    [void][SleepControl]::SetThreadExecutionState(
        $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
    )
}

function Clear-Awake {
    [void][SleepControl]::SetThreadExecutionState($ES_CONTINUOUS)
}

function Write-AtomicJson {
    param([string]$Path, [hashtable]$Value)
    $temp = "$Path.tmp"
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temp -Encoding utf8
    Move-Item -LiteralPath $temp -Destination $Path -Force
}

function Write-Event {
    param([string]$Message, [hashtable]$Fields = @{})
    $record = @{
        at = [DateTimeOffset]::UtcNow.ToString("o")
        message = $Message
    }
    foreach ($key in $Fields.Keys) {
        $record[$key] = $Fields[$key]
    }
    $line = $record | ConvertTo-Json -Compress -Depth 12
    Add-Content -LiteralPath $logPath -Value $line -Encoding utf8
    Write-Output $line
}

function Invoke-RemoteText {
    param([string]$Command)
    $previousPreference = $ErrorActionPreference
    try {
        # Vast writes its welcome text to the native stderr stream even for a
        # successful non-interactive SSH command.  Windows PowerShell can promote
        # that stream to a terminating ErrorRecord when the script preference is
        # Stop, so temporarily handle the native exit code ourselves.
        $ErrorActionPreference = "Continue"
        $output = & ssh -i $IdentityFile -p $Port `
            -o BatchMode=yes -o ConnectTimeout=15 `
            -o ServerAliveInterval=15 -o ServerAliveCountMax=3 `
            "root@$HostName" $Command 2>$null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "SSH command failed with exit code $exitCode"
    }
    return ($output -join "`n")
}

function Download-Archive {
    param(
        [string]$RemotePath,
        [string]$FinalPath,
        [string]$ExpectedSha256,
        [int64]$ExpectedBytes
    )
    $partialPath = "$FinalPath.partial"
    $resolvedDestination = [IO.Path]::GetFullPath($Destination)
    $resolvedPartial = [IO.Path]::GetFullPath($partialPath)
    if (-not $resolvedPartial.StartsWith(
        $resolvedDestination + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to manage a partial file outside Destination: $resolvedPartial"
    }

    for ($attempt = 1; $attempt -le $MaxTransferAttempts; $attempt++) {
        Set-Awake
        if (Test-Path -LiteralPath $partialPath) {
            Remove-Item -LiteralPath $partialPath -Force
        }
        Write-Event "archive_transfer_started" @{
            attempt = $attempt
            remote_path = $RemotePath
            expected_bytes = $ExpectedBytes
        }
        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & scp -i $IdentityFile -P $Port `
                -o BatchMode=yes -o ConnectTimeout=15 `
                -o ServerAliveInterval=15 -o ServerAliveCountMax=8 `
                "root@${HostName}:$RemotePath" $partialPath 2>$null
            $transferExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousPreference
        }
        if ($transferExitCode -ne 0) {
            Write-Event "archive_transfer_failed" @{
                attempt = $attempt
                exit_code = $transferExitCode
            }
            Start-Sleep -Seconds ([Math]::Min(120, 15 * $attempt))
            continue
        }
        $actualBytes = (Get-Item -LiteralPath $partialPath).Length
        $actualSha256 = (
            Get-FileHash -LiteralPath $partialPath -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        if (
            $actualBytes -eq $ExpectedBytes -and
            $actualSha256 -eq $ExpectedSha256.ToLowerInvariant()
        ) {
            Move-Item -LiteralPath $partialPath -Destination $FinalPath -Force
            Write-Event "archive_transfer_verified" @{
                attempt = $attempt
                local_path = $FinalPath
                bytes = $actualBytes
                sha256 = $actualSha256
            }
            return
        }
        Write-Event "archive_checksum_mismatch" @{
            attempt = $attempt
            expected_bytes = $ExpectedBytes
            actual_bytes = $actualBytes
            expected_sha256 = $ExpectedSha256
            actual_sha256 = $actualSha256
        }
        Remove-Item -LiteralPath $partialPath -Force
        Start-Sleep -Seconds ([Math]::Min(120, 15 * $attempt))
    }
    throw "Archive transfer did not verify after $MaxTransferAttempts attempts"
}

New-Item -ItemType Directory -Path $Destination -Force | Out-Null
Set-Awake
Write-AtomicJson -Path $statePath -Value @{
    status = "watching"
    host = $HostName
    port = $Port
    destination = $Destination
    started_at = [DateTimeOffset]::UtcNow.ToString("o")
    prevents_windows_sleep = $true
}
Write-Event "watcher_started" @{
    host = $HostName
    port = $Port
    destination = $Destination
}

try {
    $consecutiveSshFailures = 0
    while ($true) {
        Set-Awake
        try {
            $readyText = Invoke-RemoteText `
                "test -f '$remoteReady' && cat '$remoteReady' || true"
            $consecutiveSshFailures = 0
        }
        catch {
            $consecutiveSshFailures++
            Write-Event "ssh_poll_failed" @{
                consecutive_failures = $consecutiveSshFailures
                error = $_.Exception.Message
            }
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        if ([string]::IsNullOrWhiteSpace($readyText)) {
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        $ready = $readyText | ConvertFrom-Json
        if ($ready.status -ne "ready_for_download") {
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        $archiveName = [IO.Path]::GetFileName([string]$ready.archive_name)
        $localArchive = Join-Path $Destination $archiveName
        $localReady = Join-Path $Destination "remote_ready.json"
        $readyText | Set-Content -LiteralPath $localReady -Encoding utf8
        Write-Event "remote_archive_ready" @{
            archive_path = $ready.archive_path
            bytes = [int64]$ready.archive_bytes
            sha256 = $ready.archive_sha256
            terminal_pipeline_status = $ready.terminal_pipeline_status
        }

        Download-Archive `
            -RemotePath ([string]$ready.archive_path) `
            -FinalPath $localArchive `
            -ExpectedSha256 ([string]$ready.archive_sha256) `
            -ExpectedBytes ([int64]$ready.archive_bytes)

        $remotePipelineState = Invoke-RemoteText `
            "cat '$remoteRoot/results/no_r_decoder_4080s/pipeline_state.json'"
        $remotePipelineState | Set-Content `
            -LiteralPath (Join-Path $Destination "remote_pipeline_state.json") `
            -Encoding utf8

        $ack = @{
            schema_version = 1
            status = "verified"
            archive_name = $archiveName
            archive_bytes = [int64]$ready.archive_bytes
            archive_sha256 = ([string]$ready.archive_sha256).ToLowerInvariant()
            local_path = $localArchive
            verified_at = [DateTimeOffset]::UtcNow.ToString("o")
            verified_by = "watch_no_r_d3_handoff.ps1"
        }
        $localAck = Join-Path $Destination "download_verified.json"
        Write-AtomicJson -Path $localAck -Value $ack

        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & scp -i $IdentityFile -P $Port `
                -o BatchMode=yes -o ConnectTimeout=15 `
                $localAck "root@${HostName}:$remoteAck" 2>$null
            $ackExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousPreference
        }
        if ($ackExitCode -ne 0) {
            throw "Could not upload verified ACK; instance was not deliberately stopped"
        }

        Write-AtomicJson -Path $statePath -Value @{
            status = "completed"
            host = $HostName
            port = $Port
            local_archive = $localArchive
            archive_bytes = [int64]$ready.archive_bytes
            archive_sha256 = ([string]$ready.archive_sha256).ToLowerInvariant()
            terminal_pipeline_status = $ready.terminal_pipeline_status
            completed_at = [DateTimeOffset]::UtcNow.ToString("o")
            remote_ack_uploaded = $true
            expected_remote_action = "stop_instance_preserve_filesystem"
        }
        Write-Event "verified_ack_uploaded_remote_stop_authorized" @{
            local_archive = $localArchive
            sha256 = $ready.archive_sha256
        }
        [console]::Beep(1200, 500)
        [console]::Beep(1500, 500)
        break
    }
}
catch {
    Write-AtomicJson -Path $statePath -Value @{
        status = "failed"
        host = $HostName
        port = $Port
        destination = $Destination
        failed_at = [DateTimeOffset]::UtcNow.ToString("o")
        error = $_.Exception.Message
        remote_stop_authorized = $false
    }
    Write-Event "watcher_failed" @{
        error = $_.Exception.Message
        remote_stop_authorized = $false
    }
    try {
        [console]::Beep(500, 1000)
    }
    catch {}
    exit 1
}
finally {
    Clear-Awake
}
