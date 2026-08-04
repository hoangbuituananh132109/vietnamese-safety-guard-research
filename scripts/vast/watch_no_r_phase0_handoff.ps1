param(
    [string]$HostName = "root@ssh8.vast.ai",
    [ValidateRange(1, 65535)]
    [int]$Port = 15873,
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$Destination = "D:\Downloads\Safety Dataset\reports\vast_download\phase0_no_r_20260724",
    [string]$ArchiveStem = "PHASE0_NO_R_20260724",
    [string]$RemoteReportRoot = "reports/no_r_phase0",
    [switch]$SkipDataBundle,
    [switch]$StopInstanceAfterVerified,
    [ValidateRange(10, 600)]
    [int]$PollSeconds = 45
)

$ErrorActionPreference = "Stop"
$remoteRoot = "/workspace/safety-dataset"
$remoteReady = "$remoteRoot/exports/$ArchiveStem.ready.json"
$remoteArchive = "$remoteRoot/exports/$ArchiveStem.tar.zst"
$remoteAck = "$remoteRoot/exports/$ArchiveStem.download_verified.json"
$remoteInventory = "$remoteRoot/$RemoteReportRoot/artifact_inventory.json"
$remoteDataArchive = "$remoteRoot/exports/$ArchiveStem.data.tar.zst"
$remoteDataReady = "$remoteRoot/exports/$ArchiveStem.data.ready.json"
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$workspaceRoot = [System.IO.Path]::GetFullPath(
    (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
)
if (-not $destinationPath.StartsWith($workspaceRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination must remain inside the workspace: $destinationPath"
}
New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null
$statePath = Join-Path $destinationPath "handoff_state.json"
$eventsPath = Join-Path $destinationPath "handoff_events.jsonl"
$stdoutPath = Join-Path $destinationPath "watcher.stdout.log"

function Write-AtomicJson {
    param([string]$Path, [object]$Value)
    $temporary = "$Path.tmp"
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Add-Event {
    param([string]$Event, [hashtable]$Payload = @{})
    $row = [ordered]@{
        at = [DateTimeOffset]::UtcNow.ToString("o")
        event = $Event
    }
    foreach ($key in $Payload.Keys) { $row[$key] = $Payload[$key] }
    ($row | ConvertTo-Json -Compress -Depth 12) |
        Add-Content -LiteralPath $eventsPath -Encoding utf8
}

function Invoke-SshText {
    param([string]$Command)
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & ssh -o BatchMode=yes -o ConnectTimeout=12 `
            -i $IdentityFile -p $Port $HostName $Command 2>$null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) { throw "ssh exited $exitCode" }
    return ($output -join "`n")
}

function Invoke-ScpQuiet {
    param([string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & scp @Arguments 2>$null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) { throw "scp exited $exitCode" }
}

$state = [ordered]@{
    status = "waiting_for_remote_archive"
    started_at = [DateTimeOffset]::UtcNow.ToString("o")
    updated_at = [DateTimeOffset]::UtcNow.ToString("o")
    remote = "${HostName}:$Port"
    archive_stem = $ArchiveStem
    remote_ready = $remoteReady
    destination = $destinationPath
}
Write-AtomicJson $statePath $state
Add-Event "watcher_started" @{ remote = "${HostName}:$Port" }

while ($true) {
    try {
        $readyText = Invoke-SshText "if test -f '$remoteReady'; then cat '$remoteReady'; else exit 3; fi"
        $ready = $readyText | ConvertFrom-Json
        if ($ready.status -ne "ready") {
            throw "Remote ready file has unexpected status: $($ready.status)"
        }
        $state.status = "downloading"
        $state.updated_at = [DateTimeOffset]::UtcNow.ToString("o")
        $state.remote_archive_bytes = [long]$ready.archive_bytes
        $state.remote_archive_sha256 = [string]$ready.archive_sha256
        Write-AtomicJson $statePath $state
        Add-Event "remote_archive_ready" @{
            bytes = [long]$ready.archive_bytes
            sha256 = [string]$ready.archive_sha256
        }

        if (-not $SkipDataBundle) {
        # The main pipeline archive preserves reports, code and config.  Build
        # and retrieve a second, checksum-bearing bundle containing the exact
        # JSONL manifests/benchmarks listed in its immutable inventory before
        # acknowledging the main archive.  Thus Vast cannot auto-stop before
        # both reproducibility bundles have reached this workstation.
        $null = Invoke-SshText "cd '$remoteRoot' && /venv/main/bin/python scripts/vast/package_referenced_data_bundle.py --root '$remoteRoot' --inventory '$remoteInventory' --archive '$remoteDataArchive' --ready '$remoteDataReady'"
        $dataReadyText = Invoke-SshText "cat '$remoteDataReady'"
        $dataReady = $dataReadyText | ConvertFrom-Json
        if ($dataReady.status -ne "ready") {
            throw "Remote data ready file has unexpected status: $($dataReady.status)"
        }
        $dataPartialPath = Join-Path $destinationPath "$ArchiveStem.data.tar.zst.partial"
        $dataFinalPath = Join-Path $destinationPath "$ArchiveStem.data.tar.zst"
        Invoke-ScpQuiet @(
            "-O", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-i", $IdentityFile, "-P", [string]$Port,
            "${HostName}:$remoteDataArchive", $dataPartialPath
        )
        $dataItem = Get-Item -LiteralPath $dataPartialPath
        $dataHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $dataPartialPath).Hash.ToLowerInvariant()
        if ($dataItem.Length -ne [long]$dataReady.archive_bytes) {
            throw "Data archive byte mismatch: local=$($dataItem.Length), remote=$($dataReady.archive_bytes)"
        }
        if ($dataHash -ne ([string]$dataReady.archive_sha256).ToLowerInvariant()) {
            throw "Data archive SHA256 mismatch: local=$dataHash, remote=$($dataReady.archive_sha256)"
        }
        Move-Item -LiteralPath $dataPartialPath -Destination $dataFinalPath -Force
        $state.data_archive = $dataFinalPath
        $state.data_archive_bytes = [long]$dataReady.archive_bytes
        $state.data_archive_sha256 = $dataHash
        $state.updated_at = [DateTimeOffset]::UtcNow.ToString("o")
        Write-AtomicJson $statePath $state
        Add-Event "referenced_data_bundle_verified" @{
            local_path = $dataFinalPath
            bytes = [long]$dataReady.archive_bytes
            sha256 = $dataHash
        }
        }

        $partialPath = Join-Path $destinationPath "$ArchiveStem.tar.zst.partial"
        $finalPath = Join-Path $destinationPath "$ArchiveStem.tar.zst"
        Invoke-ScpQuiet @(
            "-O", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-i", $IdentityFile, "-P", [string]$Port,
            "${HostName}:$remoteArchive", $partialPath
        )

        $item = Get-Item -LiteralPath $partialPath
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $partialPath).Hash.ToLowerInvariant()
        if ($item.Length -ne [long]$ready.archive_bytes) {
            throw "Archive byte mismatch: local=$($item.Length), remote=$($ready.archive_bytes)"
        }
        if ($hash -ne ([string]$ready.archive_sha256).ToLowerInvariant()) {
            throw "Archive SHA256 mismatch: local=$hash, remote=$($ready.archive_sha256)"
        }
        Move-Item -LiteralPath $partialPath -Destination $finalPath -Force

        $ack = [ordered]@{
            status = "verified"
            verified_at = [DateTimeOffset]::UtcNow.ToString("o")
            archive_bytes = [long]$ready.archive_bytes
            archive_sha256 = $hash
            local_path = $finalPath
            verified_by = "watch_no_r_phase0_handoff.ps1:$ArchiveStem"
        }
        $localAck = Join-Path $destinationPath "$ArchiveStem.download_verified.json"
        Write-AtomicJson $localAck $ack
        Invoke-ScpQuiet @(
            "-O", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-i", $IdentityFile, "-P", [string]$Port,
            $localAck, "${HostName}:$remoteAck"
        )

        $state.status = "download_verified_ack_uploaded"
        $state.updated_at = [DateTimeOffset]::UtcNow.ToString("o")
        $state.local_archive = $finalPath
        $state.archive_bytes = [long]$ready.archive_bytes
        $state.archive_sha256 = $hash
        Write-AtomicJson $statePath $state
        Add-Event "download_verified_ack_uploaded" @{
            local_path = $finalPath
            bytes = [long]$ready.archive_bytes
            sha256 = $hash
        }
        if ($StopInstanceAfterVerified -and $ready.terminal_status -eq "completed") {
            Add-Event "instance_stop_requested" @{ reason = "verified_completed_archive" }
            try {
                $null = Invoke-SshText 'vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"'
                Add-Event "instance_stop_command_accepted"
            }
            catch {
                Add-Event "instance_stop_command_failed" @{ error = $_.Exception.Message }
            }
        }
        exit 0
    }
    catch {
        $message = $_.Exception.Message
        if ($message -notmatch "ssh exited 3") {
            Add-Event "poll_not_ready_or_failed" @{ error = $message }
        }
        $state.status = "waiting_for_remote_archive"
        $state.updated_at = [DateTimeOffset]::UtcNow.ToString("o")
        $state.last_poll_error = $message
        Write-AtomicJson $statePath $state
        Start-Sleep -Seconds $PollSeconds
    }
}
