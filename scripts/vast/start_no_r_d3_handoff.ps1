param(
    [string]$HostName = "ssh8.vast.ai",
    [int]$Port = 15873,
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$Destination = "D:\Downloads\Safety Dataset\reports\vast_download\d3_nemotron_no_r_4080s_20260724"
)

$ErrorActionPreference = "Stop"
$watcher = Join-Path $PSScriptRoot "watch_no_r_d3_handoff.ps1"
$stateDir = Join-Path $PSScriptRoot "..\..\reports\vast_download\d3_nemotron_no_r_4080s_20260724"
$stateDir = [IO.Path]::GetFullPath($stateDir)
New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
$stdout = Join-Path $stateDir "watcher.stdout.log"
$stderr = Join-Path $stateDir "watcher.stderr.log"
$pidFile = Join-Path $stateDir "watcher.pid"

if (Test-Path -LiteralPath $pidFile) {
    $oldPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ($oldPid -match '^\d+$' -and (Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue)) {
        Write-Output "D3 handoff watcher is already running PID=$oldPid"
        exit 0
    }
}

$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$watcher`"",
    "-HostName", "`"$HostName`"",
    "-Port", "$Port",
    "-IdentityFile", "`"$IdentityFile`"",
    "-Destination", "`"$Destination`""
)
$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList ($arguments -join " ") `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru
$process.Id | Set-Content -LiteralPath $pidFile -Encoding ascii
Write-Output "Started D3 handoff watcher PID=$($process.Id)"
Write-Output "The watcher keeps Windows awake, verifies the downloaded archive, and only then authorizes the Vast stop."
Write-Output "State: $stateDir\local_handoff_state.json"
Write-Output "Log:   $stateDir\local_handoff.log"
