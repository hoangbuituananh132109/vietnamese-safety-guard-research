param(
    [string]$HostAlias = "safety-vast",
    [string]$RemotePath = "/workspace/safety-dataset"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# Safety boundary: this uploader intentionally transfers no dataset, benchmark,
# translation output/state, API file, credential, or user-generated report.
$CodeItems = @(
    "configs",
    "guard_smoke",
    "guard_train",
    "scripts",
    "tests",
    "notebooks",
    "requirements-vast.txt",
    "requirements-guard-smoke.txt",
    "GUARD_EXPERIMENT_EXECUTION_PLAN_LOCKED_V3.md",
    "MMBERT_GLI_SCHEMA_EXPERIMENTS.md"
)

$PublicModelItems = @(
    "models\fastino_gliguard_300m",
    "models\mmbert_small_base_smoke"
)

& ssh $HostAlias "mkdir -p '$RemotePath' '$RemotePath/models'"
if ($LASTEXITCODE -ne 0) { throw "Cannot create remote smoke workspace via $HostAlias" }

Push-Location $Root
try {
    foreach ($Item in $CodeItems) {
        if (-not (Test-Path -LiteralPath $Item)) { throw "Missing local item: $Item" }
        Write-Host "Uploading non-data item: $Item"
        & scp -r $Item "${HostAlias}:$RemotePath/"
        if ($LASTEXITCODE -ne 0) { throw "scp failed for $Item" }
    }
    foreach ($Item in $PublicModelItems) {
        if (-not (Test-Path -LiteralPath $Item)) { throw "Missing local public model: $Item" }
        Write-Host "Uploading public model: $Item"
        & scp -r $Item "${HostAlias}:$RemotePath/models/"
        if ($LASTEXITCODE -ne 0) { throw "scp failed for $Item" }
    }
}
finally {
    Pop-Location
}

Write-Host "Smoke-only upload complete. No data directory, benchmark rows, API/key,"
Write-Host "translation state, or reports were transferred."
