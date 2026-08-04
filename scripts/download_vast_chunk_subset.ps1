param(
    [string]$RemoteAlias = "safety-vast",
    [string]$RemotePackage = "/workspace/safety-backup-packages/20260722_1810",
    [Parameter(Mandatory = $true)]
    [string]$Destination,
    [Parameter(Mandatory = $true)]
    [string]$NameRegex,
    [int]$MaxAttempts = 6
)

$ErrorActionPreference = "Stop"
$chunkDir = Join-Path $Destination "chunks"
$manifestPath = Join-Path $Destination "chunk_sha256.txt"
$logPath = Join-Path $Destination "parallel_download.log"
New-Item -ItemType Directory -Force -Path $chunkDir | Out-Null

function Log([string]$Message) {
    Add-Content -LiteralPath $logPath -Value ("{0} {1}" -f (Get-Date).ToString("o"), $Message) -Encoding UTF8
}

$records = foreach ($line in Get-Content -LiteralPath $manifestPath) {
    if ($line -match '^([0-9a-fA-F]{64})\s+(.+)$') {
        $hash = $Matches[1].ToLowerInvariant()
        $name = $Matches[2].Trim()
        if ($name -match $NameRegex) {
            [pscustomobject]@{ Hash = $hash; Name = $name }
        }
    }
}

foreach ($record in $records) {
    $target = Join-Path $chunkDir $record.Name
    if (Test-Path -LiteralPath $target) {
        $existing = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($existing -eq $record.Hash) {
            Log "skip verified $($record.Name)"
            continue
        }
    }
    $partial = "$target.parallel.partial"
    $done = $false
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Log "attempt=$attempt $($record.Name)"
        & scp.exe -q -O `
            -o ConnectTimeout=30 `
            -o ServerAliveInterval=15 `
            -o ServerAliveCountMax=4 `
            "${RemoteAlias}:$RemotePackage/chunks/$($record.Name)" $partial
        if ($LASTEXITCODE -eq 0) {
            $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -eq $record.Hash) {
                Move-Item -LiteralPath $partial -Destination $target -Force
                Log "verified $($record.Name)"
                $done = $true
                break
            }
            Log "checksum mismatch $($record.Name)"
        } else {
            Log "scp exit=$LASTEXITCODE $($record.Name)"
        }
        Start-Sleep -Seconds ([Math]::Min(30, 5 * $attempt))
    }
    if (-not $done) { throw "Failed to download $($record.Name)" }
}

Log "parallel subset complete regex=$NameRegex"
