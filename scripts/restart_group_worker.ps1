param(
    [Parameter(Mandatory=$true)][ValidateRange(1,5)][int]$Group,
    [int]$RequestIntervalSeconds = 60,
    [int]$QuotaCooldownSeconds = 60,
    [switch]$UseNewPool
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunDir = Join-Path $Root "data\run"
$statePath = Join-Path $RunDir "worker_${Group}.json"
$pidFile = Join-Path $RunDir "group_${Group}_standalone.pid"
$pids = [System.Collections.Generic.HashSet[int]]::new()

if (Test-Path -LiteralPath $statePath) {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ([int]$state.group -ne $Group) { throw "Worker state group mismatch" }
    if ($state.pid -match '^\d+$') { [void]$pids.Add([int]$state.pid) }
}
if (Test-Path -LiteralPath $pidFile) {
    $launcherPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ($launcherPid -match '^\d+$') { [void]$pids.Add([int]$launcherPid) }
}

foreach ($processId in $pids) {
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
        Write-Output "Stopping Group $Group process PID=$processId name=$($process.ProcessName)"
        Stop-Process -Id $processId -Force
    }
}
Start-Sleep -Seconds 3

$starter = Join-Path $PSScriptRoot "start_group_worker.ps1"
& $starter -Group $Group `
    -RequestIntervalSeconds $RequestIntervalSeconds `
    -QuotaCooldownSeconds $QuotaCooldownSeconds `
    -UseNewPool:$UseNewPool
