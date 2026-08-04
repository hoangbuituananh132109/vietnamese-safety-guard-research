param(
    [Parameter(Mandatory=$true)][ValidateRange(1,5)][int]$Group,
    [int]$RefreshSeconds = 2,
    [switch]$ReplayAll
)

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$CheckpointDir = Join-Path $Root "data\checkpoints"
$RunDir = Join-Path $Root "data\run"
$Host.UI.RawUI.WindowTitle = "Nemotron - Group $Group - Full output"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$seen = @{ train=0; valid=0; test=0 }
$lastBatch = ""
$lastWorkerStatusKey = ""

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

foreach ($split in @('train','valid','test')) {
    $path = Join-Path $CheckpointDir "nemotron_${split}_full_v10_g${Group}_completed.jsonl"
    if (-not $ReplayAll -and (Test-Path -LiteralPath $path)) {
        $seen[$split] = @(Read-SharedLines $path).Count
    }
}

Write-Host "Nemotron Group $Group - complete API output" -ForegroundColor Cyan
Write-Host "Every translated record is printed in full after Gemini returns a batch." -ForegroundColor DarkGray
Write-Host "No text-length limit is applied by this viewer." -ForegroundColor DarkGray
Write-Host "Ctrl+C closes this viewer only; the translation runner keeps running." -ForegroundColor DarkGray
Write-Host ""

while ($true) {
    $workerStatePath = Join-Path $RunDir "worker_${Group}.json"
    if (Test-Path -LiteralPath $workerStatePath) {
        try {
            $workerState = Get-Content -LiteralPath $workerStatePath -Raw | ConvertFrom-Json
            $statusKey = "$($workerState.pid)|$($workerState.status)|$($workerState.cooldown_until)"
            if ($statusKey -ne $lastWorkerStatusKey) {
                $lastWorkerStatusKey = $statusKey
                Write-Host ""
                Write-Host "WORKER STATUS: $($workerState.status) | PID $($workerState.pid)" -ForegroundColor Magenta
                if ($workerState.cooldown_until) {
                    Write-Host "429 cooldown until: $($workerState.cooldown_until)" -ForegroundColor DarkYellow
                }
                if ($workerState.reason) {
                    Write-Host "Reason: $($workerState.reason)" -ForegroundColor DarkYellow
                }
            }
        } catch {}
    }

    foreach ($split in @('train','valid','test')) {
        $path = Join-Path $CheckpointDir "nemotron_${split}_full_v10_g${Group}_completed.jsonl"
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $lines = @(Read-SharedLines $path)
        $start = [int]$seen[$split]
        if ($lines.Count -le $start) { continue }

        for ($index = $start; $index -lt $lines.Count; $index++) {
            if ([string]::IsNullOrWhiteSpace($lines[$index])) { continue }
            try { $record = $lines[$index] | ConvertFrom-Json } catch { continue }

            if ($record.translation_batch_id -ne $lastBatch) {
                $lastBatch = $record.translation_batch_id
                Write-Host ""
                Write-Host ("=" * 110) -ForegroundColor DarkCyan
                Write-Host "GROUP $Group | SPLIT $split | BATCH $lastBatch" -ForegroundColor Cyan
                Write-Host "Completed at : $($record.translated_at)"
                Write-Host "API key slot : $($record.translation_api_key_slot)"
                if ($record.usage_metadata) {
                    Write-Host "Prompt tokens: $($record.usage_metadata.prompt_token_count)"
                    Write-Host "Output tokens: $($record.usage_metadata.candidates_token_count)"
                    Write-Host "Total tokens : $($record.usage_metadata.total_token_count)"
                }
                Write-Host ("=" * 110) -ForegroundColor DarkCyan
            }

            Write-Host ""
            Write-Host "RECORD $($record.record_uid)" -ForegroundColor Yellow
            Write-Host "prompt_vi:" -ForegroundColor Green
            if ($null -eq $record.prompt_vi) { Write-Host "<null>" } else { Write-Host $record.prompt_vi }
            Write-Host ""
            Write-Host "response_vi:" -ForegroundColor Green
            if ($null -eq $record.response_vi) { Write-Host "<null>" } elseif ($record.response_vi -eq "") { Write-Host "<empty string>" } else { Write-Host $record.response_vi }
            Write-Host ""
            Write-Host "warnings: $(@($record.validation_warnings) -join '; ')" -ForegroundColor DarkYellow
            Write-Host ("-" * 110) -ForegroundColor DarkGray
        }
        $seen[$split] = $lines.Count
    }

    $pidFile = Join-Path $RunDir "full_all_runner.pid"
    if (Test-Path -LiteralPath $pidFile) {
        $runnerPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
        $active = $runnerPid -match '^\d+$' -and [bool](Get-Process -Id ([int]$runnerPid) -ErrorAction SilentlyContinue)
        if (-not $active) {
            Write-Host ""
            Write-Host "Runner is currently stopped. Viewer remains open for inspection." -ForegroundColor Red
            Start-Sleep -Seconds 10
        }
    }
    Start-Sleep -Seconds $RefreshSeconds
}
