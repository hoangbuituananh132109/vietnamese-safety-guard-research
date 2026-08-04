param(
    [Parameter(Mandatory=$true)][ValidateRange(1,5)][int]$Group,
    [int]$RequestIntervalSeconds = 60,
    [int]$QuotaCooldownSeconds = 60,
    [switch]$UseNewPool
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunDir = Join-Path $Root "data\run"
$LogDir = Join-Path $Root "reports\full_run"
$PidFile = Join-Path $RunDir "group_${Group}_standalone.pid"
New-Item -ItemType Directory -Force -Path $RunDir, $LogDir | Out-Null

if (Test-Path -LiteralPath $PidFile) {
    $oldPid = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    if ($oldPid -match '^\d+$' -and (Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue)) {
        Write-Output "Group $Group standalone worker is already active with PID $oldPid"
        exit 0
    }
}

$firstSlot = (($Group - 1) * 2) + 1
$secondSlot = $firstSlot + 1
$apiCount = (Get-Content -LiteralPath (Join-Path $Root "API.txt") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count
$reserveSlot = 10 + $Group
$primarySlots = "${firstSlot},${secondSlot}"
$useReserve = $true
$assignmentPath = Join-Path $Root "configs\key_assignments.json"
if (Test-Path -LiteralPath $assignmentPath) {
    $assignment = Get-Content -LiteralPath $assignmentPath -Raw | ConvertFrom-Json
    $assigned = @($assignment.groups.PSObject.Properties[[string]$Group].Value)
    if ($assigned.Count -gt 0) {
        $primarySlots = ($assigned -join ',')
        $useReserve = $false
    }
}
if ($UseNewPool) {
    if ($apiCount -lt 15) { throw "-UseNewPool requires API.txt slots 11 through 15; found only $apiCount keys" }
    $primarySlots = "11,12,13,14,15"
    $useReserve = $false
}
$groupIndex = $Group - 1
$stdout = Join-Path $LogDir "worker_${Group}_standalone.log"
$stderr = Join-Path $LogDir "worker_${Group}_standalone_error.log"
$arguments = @(
    "-m", "translator.full_run", "worker",
    "--root", "`"$Root`"",
    "--group-index", $groupIndex,
    "--group-count", 5,
    "--key-slots", $primarySlots,
    "--model", "gemini-3.1-flash-lite",
    "--max-retries", 4,
    "--quota-cooldown-seconds", $QuotaCooldownSeconds,
    "--max-quota-cooldowns", 5,
    "--request-interval-seconds", $RequestIntervalSeconds
) -join " "

if ($useReserve -and $reserveSlot -le $apiCount) {
    $arguments += " --reserve-key-slots $reserveSlot"
}

$wrapperCommand = "Set-Location -LiteralPath '$($Root.Replace("'", "''"))'; & '$($Python.Replace("'", "''"))' $arguments 1>> '$($stdout.Replace("'", "''"))' 2>> '$($stderr.Replace("'", "''"))'"
$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($wrapperCommand))
$info = [System.Diagnostics.ProcessStartInfo]::new()
$info.FileName = "powershell.exe"
$info.Arguments = "-NoProfile -ExecutionPolicy Bypass -EncodedCommand $encodedCommand"
$info.WorkingDirectory = $Root
$info.UseShellExecute = $true
$info.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$process = [System.Diagnostics.Process]::Start($info)
Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ascii
Write-Output "Started Group $Group standalone worker PID=$($process.Id) slots=$primarySlots"
if ($UseNewPool) {
    Write-Output "Fresh-pool mode: old slots $firstSlot,$secondSlot are excluded; requests rotate across slots 11-15."
} elseif (-not $useReserve) {
    Write-Output "Dashboard assignment: slots=$primarySlots"
} elseif ($reserveSlot -le $apiCount) {
    Write-Output "Reserve slot: $reserveSlot (activated only after 3 consecutive 429 errors)."
}
Write-Output "Request interval: $RequestIntervalSeconds seconds per API key (each slot has its own timer)."
Write-Output "A 429 is retried after 30 seconds."
Write-Output "On sustained 429, only this group sleeps $QuotaCooldownSeconds seconds; other groups are unaffected."
Write-Output "Log: $stdout"
