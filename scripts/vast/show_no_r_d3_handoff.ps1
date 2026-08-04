param(
    [string]$Destination = "D:\Downloads\Safety Dataset\reports\vast_download\d3_nemotron_no_r_4080s_20260724"
)

$pidFile = Join-Path $Destination "watcher.pid"
$stateFile = Join-Path $Destination "local_handoff_state.json"
$logFile = Join-Path $Destination "local_handoff.log"

if (Test-Path -LiteralPath $pidFile) {
    $watcherPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    $active = $false
    if ($watcherPid -match '^\d+$') {
        $active = $null -ne (Get-Process -Id ([int]$watcherPid) -ErrorAction SilentlyContinue)
    }
    Write-Output "watcher_pid=$watcherPid active=$active"
}
else {
    Write-Output "watcher_pid=none active=False"
}

if (Test-Path -LiteralPath $stateFile) {
    Write-Output "=== STATE ==="
    Get-Content -LiteralPath $stateFile
}
if (Test-Path -LiteralPath $logFile) {
    Write-Output "=== LOG TAIL ==="
    Get-Content -LiteralPath $logFile -Tail 20
}
