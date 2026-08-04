param(
    [int]$Group = 0,
    [int]$RefreshSeconds = 5,
    [switch]$Once
)

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunDir = Join-Path $Root "data\run"
$CheckpointDir = Join-Path $Root "data\checkpoints"

function Read-SharedLines([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return @() }
    $stream = [System.IO.FileStream]::new(
        $Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
    )
    $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
    $lines = [System.Collections.Generic.List[string]]::new()
    try {
        while (-not $reader.EndOfStream) { $lines.Add($reader.ReadLine()) }
    } finally {
        $reader.Dispose()
        $stream.Dispose()
    }
    return $lines.ToArray()
}

function Count-Lines([string]$Path) {
    return @(Read-SharedLines $Path | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count
}

function Runner-Info {
    $pidFile = Join-Path $RunDir "full_all_runner.pid"
    $runnerPid = if (Test-Path -LiteralPath $pidFile) {
        (Get-Content -LiteralPath $pidFile -Raw).Trim()
    } else { "" }
    $active = $false
    if ($runnerPid -match '^\d+$') {
        $active = [bool](Get-Process -Id ([int]$runnerPid) -ErrorAction SilentlyContinue)
    }
    return @{ Pid=$runnerPid; Active=$active }
}

function Latest-Batch([string]$Path) {
    $lines = @(Read-SharedLines $Path)
    if ($lines.Count -eq 0) { return $null }
    $last = $lines[$lines.Count - 1]
    if ([string]::IsNullOrWhiteSpace($last)) { return $null }
    try { return $last | ConvertFrom-Json } catch { return $null }
}

function Show-Overview {
    $runner = Runner-Info
    $statePath = Join-Path $RunDir "runner_state.json"
    $state = if (Test-Path -LiteralPath $statePath) {
        Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    } else { $null }

    Write-Host "NEMOTRON EN -> VI - LIVE OVERVIEW" -ForegroundColor Cyan
    Write-Host "Time       : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Runner PID : $($runner.Pid)"
    Write-Host "Runner     : $($(if ($runner.Active) {'RUNNING'} else {'STOPPED'}))" -ForegroundColor $(if ($runner.Active) {'Green'} else {'Red'})
    if ($state) {
        Write-Host "State      : $($state.status)"
        Write-Host "Updated    : $($state.updated_at)"
    }
    Write-Host ""

    $rows = foreach ($group in 1..5) {
        $counts = @{}
        $latestTime = $null
        $latest = $null
        foreach ($split in @('train','valid','test')) {
            $path = Join-Path $CheckpointDir "nemotron_${split}_full_v10_g${group}_completed.jsonl"
            $counts[$split] = Count-Lines $path
            if (Test-Path -LiteralPath $path) {
                $item = Get-Item -LiteralPath $path
                if (-not $latestTime -or $item.LastWriteTime -gt $latestTime) {
                    $latestTime = $item.LastWriteTime
                    $latest = Latest-Batch $path
                }
            }
        }
        $workerPath = Join-Path $RunDir "worker_${group}.json"
        $worker = if (Test-Path -LiteralPath $workerPath) {
            Get-Content -LiteralPath $workerPath -Raw | ConvertFrom-Json
        } else { $null }
        [pscustomobject]@{
            Group=$group
            WorkerPID=$(if ($worker) {$worker.pid} else {'-'})
            Train=$counts.train
            Valid=$counts.valid
            Test=$counts.test
            Total=$counts.train + $counts.valid + $counts.test
            LastBatch=$(if ($latest) {$latest.translation_batch_id} else {'-'})
            Tokens=$(if ($latest -and $latest.usage_metadata) {$latest.usage_metadata.total_token_count} else {'-'})
            LastWrite=$(if ($latestTime) {$latestTime.ToString('HH:mm:ss')} else {'-'})
        }
    }
    $rows | Format-Table -AutoSize
    Write-Host "Total translated: $(($rows | Measure-Object -Property Total -Sum).Sum) / 45416" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "This window is read-only. Ctrl+C closes monitoring, not the runner." -ForegroundColor DarkGray
}

function Show-Group([int]$Number) {
    $runner = Runner-Info
    $workerPath = Join-Path $RunDir "worker_${Number}.json"
    $worker = if (Test-Path -LiteralPath $workerPath) {
        Get-Content -LiteralPath $workerPath -Raw | ConvertFrom-Json
    } else { $null }

    Write-Host "NEMOTRON WORKER GROUP $Number - LIVE" -ForegroundColor Cyan
    Write-Host "Time        : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Runner      : $($(if ($runner.Active) {'RUNNING'} else {'STOPPED'}))" -ForegroundColor $(if ($runner.Active) {'Green'} else {'Red'})
    Write-Host "Worker PID  : $($(if ($worker) {$worker.pid} else {'-'}))"
    Write-Host "Key slots   : $($(if ($worker) {$worker.key_slots -join ', '} else {'-'}))"
    Write-Host "Last status : $($(if ($worker) {$worker.status} else {'starting'}))"
    if ($worker -and $worker.reason) { Write-Host "Last reason : $($worker.reason)" -ForegroundColor DarkYellow }
    Write-Host ""

    foreach ($split in @('train','valid','test')) {
        $path = Join-Path $CheckpointDir "nemotron_${split}_full_v10_g${Number}_completed.jsonl"
        $count = Count-Lines $path
        $latest = Latest-Batch $path
        Write-Host "[$split] records=$count" -ForegroundColor Yellow
        if ($latest) {
            Write-Host "  last batch : $($latest.translation_batch_id)"
            Write-Host "  translated : $($latest.translated_at)"
            Write-Host "  source char: $($latest.input_source_chars)"
            Write-Host "  output char: $($latest.output_translation_chars)"
            Write-Host "  tokens     : $($latest.usage_metadata.total_token_count)"
            Write-Host "  warnings   : $(@($latest.validation_warnings).Count)"
        }
    }
    Write-Host ""
    Write-Host "No change for several minutes can mean a 429 cooldown or a long batch." -ForegroundColor DarkGray
    Write-Host "Ctrl+C closes this monitor only." -ForegroundColor DarkGray
}

$Host.UI.RawUI.WindowTitle = if ($Group -eq 0) { "Nemotron - Overview" } else { "Nemotron - Group $Group" }
do {
    Clear-Host
    if ($Group -eq 0) { Show-Overview } else { Show-Group $Group }
    if (-not $Once) { Start-Sleep -Seconds $RefreshSeconds }
} while (-not $Once)
