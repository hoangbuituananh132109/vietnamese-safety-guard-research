param(
    [string]$RemoteAlias = "safety-vast",
    [string]$RemotePackage = "/workspace/safety-backup-packages/20260722_1810",
    [Parameter(Mandatory = $true)]
    [string]$Destination,
    [int]$MaxAttempts = 6
)

$ErrorActionPreference = "Stop"
$Destination = [IO.Path]::GetFullPath($Destination)
$chunkDir = Join-Path $Destination "chunks"
$archiveDir = Join-Path $Destination "archives"
$extractRoot = Join-Path $Destination "extracted\safety-dataset"
$statePath = Join-Path $Destination "download_state.json"
$logPath = Join-Path $Destination "download.log"

New-Item -ItemType Directory -Force -Path $chunkDir, $archiveDir, $extractRoot | Out-Null

function Write-Log([string]$Message) {
    $line = "{0} {1}" -f (Get-Date).ToString("o"), $Message
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
    Write-Output $line
}

function Write-State([string]$Status, [int]$Completed, [int]$Total, [string]$Current = "") {
    [ordered]@{
        status = $Status
        updated_at = (Get-Date).ToString("o")
        completed_chunks = $Completed
        total_chunks = $Total
        current = $Current
        destination = $Destination
        remote = "${RemoteAlias}:$RemotePackage"
    } | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
}

function Invoke-ScpWithRetry([string]$RemoteFile, [string]$LocalFile, [string]$ExpectedHash = "") {
    if (Test-Path -LiteralPath $LocalFile) {
        if (-not $ExpectedHash -or (Get-FileHash -LiteralPath $LocalFile -Algorithm SHA256).Hash.ToLowerInvariant() -eq $ExpectedHash) {
            return
        }
    }
    $partial = "$LocalFile.partial"
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Write-Log "download attempt=$attempt remote=$RemoteFile"
        & scp.exe -q -O `
            -o ConnectTimeout=30 `
            -o ServerAliveInterval=15 `
            -o ServerAliveCountMax=4 `
            "${RemoteAlias}:$RemoteFile" $partial
        if ($LASTEXITCODE -eq 0) {
            if ($ExpectedHash) {
                $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($actual -ne $ExpectedHash) {
                    Write-Log "checksum mismatch remote=$RemoteFile expected=$ExpectedHash actual=$actual"
                    Start-Sleep -Seconds 5
                    continue
                }
            }
            Move-Item -LiteralPath $partial -Destination $LocalFile -Force
            return
        }
        Write-Log "scp failed exit=$LASTEXITCODE remote=$RemoteFile"
        Start-Sleep -Seconds ([Math]::Min(30, 5 * $attempt))
    }
    throw "Unable to download $RemoteFile after $MaxAttempts attempts"
}

function Join-Chunks([string]$Prefix, [string]$OutputPath) {
    $exactPartPattern = '^' + [regex]::Escape($Prefix) + '\.part-\d+$'
    $parts = @(
        Get-ChildItem -LiteralPath $chunkDir -File -Filter "$Prefix.part-*" |
            Where-Object { $_.Name -match $exactPartPattern } |
            Sort-Object Name
    )
    if (-not $parts) { throw "No chunks found for $Prefix" }
    $output = [IO.File]::Open($OutputPath, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        foreach ($part in $parts) {
            $input = [IO.File]::OpenRead($part.FullName)
            try { $input.CopyTo($output) } finally { $input.Dispose() }
        }
    } finally {
        $output.Dispose()
    }
}

try {
    Write-Log "snapshot download started"
    foreach ($name in @("archive_sha256.txt", "chunk_sha256.txt")) {
        Invoke-ScpWithRetry "$RemotePackage/$name" (Join-Path $Destination $name)
    }

    $chunkRecords = @()
    foreach ($line in Get-Content -LiteralPath (Join-Path $Destination "chunk_sha256.txt")) {
        if ($line -match '^([0-9a-fA-F]{64})\s+(.+)$') {
            $chunkRecords += [pscustomobject]@{ Hash = $Matches[1].ToLowerInvariant(); Name = $Matches[2].Trim() }
        }
    }
    if (-not $chunkRecords) { throw "Remote chunk manifest is empty" }

    $completed = 0
    Write-State "downloading" $completed $chunkRecords.Count
    foreach ($record in $chunkRecords) {
        Write-State "downloading" $completed $chunkRecords.Count $record.Name
        Invoke-ScpWithRetry "$RemotePackage/chunks/$($record.Name)" (Join-Path $chunkDir $record.Name) $record.Hash
        $completed++
        Write-State "downloading" $completed $chunkRecords.Count $record.Name
    }

    $archiveHashes = @{}
    foreach ($line in Get-Content -LiteralPath (Join-Path $Destination "archive_sha256.txt")) {
        if ($line -match '^([0-9a-fA-F]{64})\s+(.+)$') {
            $archiveHashes[$Matches[2].Trim()] = $Matches[1].ToLowerInvariant()
        }
    }

    Write-State "reassembling" $completed $chunkRecords.Count
    foreach ($archiveName in $archiveHashes.Keys) {
        $archivePath = Join-Path $archiveDir $archiveName
        Join-Chunks $archiveName $archivePath
        $actual = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $archiveHashes[$archiveName]) {
            throw "Archive checksum mismatch for $archiveName"
        }
        Write-Log "archive verified name=$archiveName sha256=$actual"
    }

    Write-State "extracting" $completed $chunkRecords.Count
    & tar.exe -xzf (Join-Path $archiveDir "project_core_no_models_reports.tar.gz") -C $extractRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed extracting project core" }
    & tar.exe -xzf (Join-Path $archiveDir "reports_complete.tar.gz") -C $extractRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed extracting reports" }
    $modelRoot = Join-Path $extractRoot "models"
    New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null
    & tar.exe -xzf (Join-Path $archiveDir "llama_recovery_no_full_weights.tar.gz") -C $modelRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed extracting Llama recovery package" }

    Write-State "complete" $completed $chunkRecords.Count
    Write-Log "snapshot download complete"
    Set-Content -LiteralPath (Join-Path $Destination "SNAPSHOT_COMPLETE.txt") -Value ((Get-Date).ToString("o")) -Encoding UTF8
    try { [console]::Beep(1000, 250); [console]::Beep(1400, 350) } catch {}
} catch {
    Write-State "failed" 0 0 $_.Exception.Message
    Write-Log "FAILED: $($_.Exception.ToString())"
    exit 1
}
