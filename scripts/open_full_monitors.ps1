$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Watch = Join-Path $Root "scripts\watch_full_status.ps1"
$OutputWatch = Join-Path $Root "scripts\watch_group_output.ps1"

$overviewArguments = "-NoExit -NoProfile -ExecutionPolicy Bypass -File `"$Watch`" -Group 0 -RefreshSeconds 5"
Start-Process -FilePath "powershell.exe" -ArgumentList $overviewArguments -WorkingDirectory $Root
Write-Output "Opened Nemotron monitor: Overview"

foreach ($group in 1..5) {
    $arguments = "-NoExit -NoProfile -ExecutionPolicy Bypass -File `"$OutputWatch`" -Group $group -RefreshSeconds 2 -ReplayAll"
    Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $Root
    Start-Sleep -Milliseconds 300
    Write-Output "Opened Nemotron full-output monitor: Group $group"
}
