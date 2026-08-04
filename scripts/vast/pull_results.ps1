param(
    [string]$HostAlias = "safety-vast",
    [string]$RemotePath = "/workspace/safety-dataset",
    [string]$LocalPath = "reports\vast_download"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Destination = Join-Path $Root $LocalPath
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

& scp -r "${HostAlias}:$RemotePath/reports/experiment_runs" $Destination
if ($LASTEXITCODE -ne 0) { throw "Could not download experiment runs" }
& scp -r "${HostAlias}:$RemotePath/reports/remote_profile" $Destination
if ($LASTEXITCODE -ne 0) { throw "Could not download profiling reports" }

# Smoke runs are useful diagnostics but may not exist on a fresh instance.
& ssh $HostAlias "test -d '$RemotePath/reports/smoke_runs'"
if ($LASTEXITCODE -eq 0) {
    & scp -r "${HostAlias}:$RemotePath/reports/smoke_runs" $Destination
    if ($LASTEXITCODE -ne 0) { throw "Could not download smoke runs" }
}

Write-Host "Results downloaded to $Destination"
