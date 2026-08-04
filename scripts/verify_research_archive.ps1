param(
    [string]$ProjectRoot = "D:\Downloads\Safety Dataset",
    [string]$VastSnapshot = "D:\SafetyDataset_Backups\vast_45474443_20260722_resumable",
    [string]$D2Backup = "D:\SafetyDataset_Backups\d2_nemotron_20260722"
)

$ErrorActionPreference = "Stop"
$manifestPath = Join-Path $ProjectRoot "reports\research_archive\CHECKSUMS_SHA256.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$failures = @()

foreach ($record in $manifest.records) {
    $path = Join-Path $ProjectRoot ($record.path -replace '/', '\')
    if (-not (Test-Path -LiteralPath $path)) {
        $failures += "missing: $($record.path)"
        continue
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $record.sha256) {
        $failures += "checksum: $($record.path)"
    }
}

$snapshotState = Join-Path $VastSnapshot "download_state.json"
if (Test-Path -LiteralPath $snapshotState) {
    $state = Get-Content -LiteralPath $snapshotState -Raw | ConvertFrom-Json
    if ($state.status -ne "complete") {
        $failures += "Vast snapshot is not complete: $($state.status)"
    }
} else {
    $failures += "Vast snapshot state missing"
}

$snapshotMarker = Join-Path $VastSnapshot "SNAPSHOT_COMPLETE.txt"
if (-not (Test-Path -LiteralPath $snapshotMarker -PathType Leaf)) {
    $failures += "Vast snapshot completion marker missing"
}

$archiveManifestPath = Join-Path $VastSnapshot "archive_sha256.txt"
$archiveDirectory = Join-Path $VastSnapshot "archives"
if (Test-Path -LiteralPath $archiveManifestPath -PathType Leaf) {
    foreach ($line in Get-Content -LiteralPath $archiveManifestPath) {
        if ($line -match '^([0-9a-fA-F]{64})\s+(.+)$') {
            $expectedHash = $Matches[1].ToLowerInvariant()
            $archiveName = $Matches[2].Trim()
            $archivePath = Join-Path $archiveDirectory $archiveName
            if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
                $failures += "Vast archive missing: $archiveName"
                continue
            }
            $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actualHash -ne $expectedHash) {
                $failures += "Vast archive checksum: $archiveName"
            }
        }
    }
} else {
    $failures += "Vast archive checksum manifest missing"
}

$restoredRoot = Join-Path $VastSnapshot "extracted\safety-dataset"
$criticalRestoredPaths = @(
    "reports\experiment_runs\E1-G-EV-512",
    "reports\experiment_runs\E2-M-EV-GLI-COMPAT-8K",
    "reports\experiment_runs\E3-M-E-8K",
    "reports\experiment_runs\E4-M-EV-MATCHED-8K",
    "reports\experiment_runs\E5-M-EV-FULL-8K",
    "reports\experiment_runs\E7-M-SCHEMA-EV-8K",
    "reports\evaluation_matrix",
    "data",
    "guard_train",
    "scripts",
    "configs",
    "models\llama_3_1_nemotron_safety_guard_8b_v3\lora_adapter\adapter_model.safetensors"
)
foreach ($relativePath in $criticalRestoredPaths) {
    $restoredPath = Join-Path $restoredRoot $relativePath
    if (-not (Test-Path -LiteralPath $restoredPath)) {
        $failures += "Vast restored path missing: $relativePath"
    }
}

$d2ManifestPath = Join-Path $ProjectRoot "reports\research_archive\D2_CRITICAL_SHA256_20260723.json"
if (-not (Test-Path -LiteralPath $d2ManifestPath -PathType Leaf)) {
    $failures += "D2 checksum manifest missing"
} else {
    $d2Manifest = Get-Content -LiteralPath $d2ManifestPath -Raw | ConvertFrom-Json
    foreach ($record in $d2Manifest.records) {
        $relativePath = $record.path -replace '/', '\'
        $path = Join-Path $D2Backup $relativePath
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            $failures += "D2 file missing: $($record.path)"
            continue
        }
        $item = Get-Item -LiteralPath $path
        if ($item.Length -ne [long]$record.bytes) {
            $failures += "D2 size mismatch: $($record.path)"
            continue
        }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $record.sha256) {
            $failures += "D2 checksum: $($record.path)"
        }
    }
}

if ($failures) {
    $failures | ForEach-Object { Write-Error $_ }
    exit 1
}

Write-Output "Research archive verified: $($manifest.files) files"
Write-Output "Vast snapshot state, archives, and critical restored paths verified: $VastSnapshot"
Write-Output "D2 critical files verified: $D2Backup"
