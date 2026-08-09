$ErrorActionPreference = "Stop"

$trainWrapperPid = 24856
$trainProcess = Get-Process -Id $trainWrapperPid -ErrorAction SilentlyContinue
if ($null -ne $trainProcess) {
    Wait-Process -Id $trainWrapperPid
}

$evalScript = (Resolve-Path (Join-Path $PSScriptRoot "run_luna_eval_splits_20260802.ps1")).Path
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $evalScript
