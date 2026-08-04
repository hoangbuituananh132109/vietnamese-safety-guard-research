param(
    [string]$HostAlias = "safety-vast",
    [string]$RemotePath = "/workspace/safety-dataset"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

& ssh $HostAlias "mkdir -p '$RemotePath' '$RemotePath/data' '$RemotePath/models'"
if ($LASTEXITCODE -ne 0) { throw "Cannot create remote workspace via $HostAlias" }

$Items = @(
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

$DataItems = @(
    @{ Source = "data\guard_full"; Destination = "$RemotePath/data/" },
    @{ Source = "data\guard_phase0_gliguard_native_512"; Destination = "$RemotePath/data/" },
    @{ Source = "data\guard_experiments_v2"; Destination = "$RemotePath/data/" },
    @{ Source = "data\eval_shared_gliguard_512"; Destination = "$RemotePath/data/" },
    @{ Source = "data\benchmarks\sea_safeguard"; Destination = "$RemotePath/data/benchmarks/" }
)

$ModelItems = @(
    "models\fastino_gliguard_300m",
    "models\mmbert_small_base_smoke"
)

Push-Location $Root
try {
    foreach ($Item in $Items) {
        if (-not (Test-Path -LiteralPath $Item)) { throw "Missing local item: $Item" }
        Write-Host "Uploading $Item"
        & scp -r $Item "${HostAlias}:$RemotePath/"
        if ($LASTEXITCODE -ne 0) { throw "scp failed for $Item" }
    }
    foreach ($Item in $DataItems) {
        if (-not (Test-Path -LiteralPath $Item.Source)) { throw "Missing local item: $($Item.Source)" }
        & ssh $HostAlias "mkdir -p '$($Item.Destination)'"
        if ($LASTEXITCODE -ne 0) { throw "Cannot create remote data destination: $($Item.Destination)" }
        Write-Host "Uploading $($Item.Source) -> $($Item.Destination)"
        & scp -r $Item.Source "${HostAlias}:$($Item.Destination)"
        if ($LASTEXITCODE -ne 0) { throw "scp failed for $($Item.Source)" }
    }
    foreach ($Item in $ModelItems) {
        if (-not (Test-Path -LiteralPath $Item)) { throw "Missing local item: $Item" }
        Write-Host "Uploading $Item"
        & scp -r $Item "${HostAlias}:$RemotePath/models/"
        if ($LASTEXITCODE -ne 0) { throw "scp failed for $Item" }
    }
}
finally {
    Pop-Location
}

Write-Host "Upload complete. API.txt and translation runner state were intentionally excluded."
Write-Host "Next: ssh $HostAlias"
Write-Host "Then: cd $RemotePath && bash scripts/vast/prepare_remote_workspace.sh $RemotePath"
