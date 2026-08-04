param(
    [string]$HostName = "root@ssh8.vast.ai",
    [ValidateRange(1, 65535)]
    [int]$Port = 15873,
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$Destination = "D:\Downloads\Safety Dataset\reports\vast_download\phase0_no_r_20260724",
    [string]$ArchiveStem = "PHASE0_NO_R_20260724",
    [string]$RemoteReportRoot = "reports/no_r_phase0",
    [switch]$SkipDataBundle,
    [switch]$StopInstanceAfterVerified
)

$ErrorActionPreference = "Stop"
$watcher = Join-Path $PSScriptRoot "watch_no_r_phase0_handoff.ps1"
$runRoot = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$pidPath = Join-Path $runRoot "watcher.pid"
$stdoutPath = Join-Path $runRoot "watcher.stdout.log"
$stderrPath = Join-Path $runRoot "watcher.stderr.log"

if (Test-Path -LiteralPath $pidPath) {
    $oldPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
    if ($oldPid -match "^\d+$" -and (Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue)) {
        Write-Output "Phase-0 no-R handoff watcher is already running (PID=$oldPid)."
        exit 0
    }
}

$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$watcher`"",
    "-HostName", $HostName,
    "-Port", [string]$Port,
    "-IdentityFile", "`"$IdentityFile`"",
    "-Destination", "`"$runRoot`"",
    "-ArchiveStem", $ArchiveStem,
    "-RemoteReportRoot", $RemoteReportRoot
)
if ($SkipDataBundle) { $arguments += "-SkipDataBundle" }
if ($StopInstanceAfterVerified) { $arguments += "-StopInstanceAfterVerified" }
$processInfo = New-Object System.Diagnostics.ProcessStartInfo
$processInfo.FileName = (Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe")
$processInfo.Arguments = ($arguments -join " ")
$processInfo.WorkingDirectory = (Split-Path $watcher -Parent)
$processInfo.UseShellExecute = $false
$processInfo.CreateNoWindow = $true
$process = [System.Diagnostics.Process]::Start($processInfo)
Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Write-Output "Started handoff watcher for $ArchiveStem PID=$($process.Id)"
Write-Output "State: $(Join-Path $runRoot 'handoff_state.json')"
Write-Output "The watcher will verify SHA-256 and upload an ACK."
