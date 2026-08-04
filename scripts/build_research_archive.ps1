param(
    [string]$ProjectRoot = "D:\Downloads\Safety Dataset",
    [string]$OutputDirectory = "D:\SafetyDataset_Backups"
)

$ErrorActionPreference = "Stop"

$manifestPath = Join-Path $ProjectRoot "reports\research_archive\CHECKSUMS_SHA256.json"
$bundlePath = Join-Path $OutputDirectory "research_summary_bundle_20260722.tar.gz"
$bundleHashPath = "$bundlePath.sha256"
$fileListPath = Join-Path $OutputDirectory "research_summary_bundle_20260722.filelist.txt"

if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Existing manifest not found: $manifestPath"
}

$oldManifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$paths = @($oldManifest.records.path)
$paths += @(
    "reports/research_archive/VAST_SNAPSHOT_RECOVERY_20260722.md",
    "reports/research_archive/NEXT_GPU_NO_R_QWEN_PLAN_20260723.md",
    "reports/research_archive/D2_CRITICAL_SHA256_20260723.json",
    "reports/research_archive/D2_VS_D1_NO_R_20260723.json",
    "reports/research_archive/D2_VS_D1_NO_R_20260723.md",
    "reports/research_archive/QWEN3GUARD_Q1_ANALYSIS_20260723.md",
    "reports/research_archive/QWEN3GUARD_Q1_COMPARISON_20260723.json",
    "reports/research_archive/QWEN3GUARD_Q1_COMPARISON_20260723.md",
    "reports/research_archive/QWEN3GUARD_Q1_SHA256_20260723.txt",
    "reports/research_archive/QWEN3GUARD_REPRODUCTION_RUNBOOK_20260723.md",
    "reports/qwen3guard/q1_no_r_test_length_profile.json",
    "reports/qwen3guard/BACKUP_COMPLETE_20260723.txt",
    "reports/qwen3guard/model_download_manifest.json",
    "reports/qwen3guard/qwen3guard_supervisor.log",
    "reports/qwen3guard/Q1-SMOKE-512/metrics.json",
    "reports/qwen3guard/Q1-SMOKE-512/predictions.jsonl",
    "reports/qwen3guard/Q1-SMOKE-512/progress.jsonl",
    "reports/qwen3guard/Q1-QWEN3GUARD-GEN-4B-ZS-NR/metrics.json",
    "reports/qwen3guard/Q1-QWEN3GUARD-GEN-4B-ZS-NR/predictions.jsonl",
    "reports/qwen3guard/Q1-QWEN3GUARD-GEN-4B-ZS-NR/progress.jsonl",
    "reports/qwen3guard/Q1-QWEN3GUARD-GEN-4B-ZS-NR/provenance/completed_at.txt",
    "reports/qwen3guard/Q1-QWEN3GUARD-GEN-4B-ZS-NR/provenance/nvidia-smi-q.txt",
    "reports/qwen3guard/Q1-QWEN3GUARD-GEN-4B-ZS-NR/provenance/pip-freeze.txt",
    "reports/qwen3guard/Q1-QWEN3GUARD-GEN-4B-ZS-NR/provenance/runtime.json",
    "reports/qwen3guard/Q1-QWEN3GUARD-GEN-4B-ZS-NR/provenance/started_at.txt",
    "data/no_r/phase0_gliguard_native_512/train.jsonl.audit.json",
    "data/no_r/e3_english_full/train.jsonl.audit.json",
    "data/no_r/e4_ev_matched_full/train.jsonl.audit.json",
    "data/no_r/e4_ev_matched_full/train.jsonl.matching_audit.json",
    "data/no_r/full/train.jsonl.audit.json",
    "data/no_r/full/valid.jsonl.audit.json",
    "data/no_r/full/test.jsonl.audit.json",
    "data/no_r/decoder/all_three.jsonl.audit.json",
    "data/no_r/decoder/test_and_sea.jsonl.audit.json",
    "data/no_r/decoder/test_and_sea.jsonl",
    "data/no_r/decoder/qwen_smoke_512.jsonl",
    "data/no_r/d2/paired_8192.jsonl.audit.json",
    "scripts/build_no_r_manifests.py",
    "scripts/build_decoder_sft_pilot.py",
    "scripts/compare_d2_predictions.py",
    "scripts/analyze_qwen_no_r_comparison.py",
    "scripts/evaluate_qwen3guard_gen_vllm.py",
    "scripts/vast/download_qwen3guard.py",
    "scripts/vast/run_qwen3guard_no_r.sh",
    "scripts/vast/run_qwen3guard_stage.sh",
    "scripts/vast/qwen3guard_supervisor.conf",
    "scripts/build_research_archive.ps1",
    "scripts/verify_research_archive.ps1"
)
$paths = @($paths | Sort-Object -Unique)

$records = foreach ($relativePath in $paths) {
    $nativePath = $relativePath -replace '/', '\'
    $absolutePath = Join-Path $ProjectRoot $nativePath
    if (-not (Test-Path -LiteralPath $absolutePath -PathType Leaf)) {
        throw "Manifest input missing: $relativePath"
    }
    $item = Get-Item -LiteralPath $absolutePath
    [ordered]@{
        path = $relativePath
        bytes = $item.Length
        sha256 = (Get-FileHash -LiteralPath $absolutePath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$manifest = [ordered]@{
    generated_at = (Get-Date).ToString("o")
    files = $records.Count
    records = @($records)
}

$manifestJson = $manifest | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText($manifestPath, $manifestJson + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false))

$bundlePaths = @($paths + "reports/research_archive/CHECKSUMS_SHA256.json" | Sort-Object -Unique)
[System.IO.File]::WriteAllLines($fileListPath, $bundlePaths,
    [System.Text.UTF8Encoding]::new($false))

if (-not (Test-Path -LiteralPath $OutputDirectory)) {
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
}

Push-Location $ProjectRoot
try {
    & tar.exe -czf $bundlePath -T $fileListPath
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$bundle = Get-Item -LiteralPath $bundlePath
$bundleHash = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText(
    $bundleHashPath,
    "$bundleHash  $($bundle.Name)" + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Output "Manifest records: $($records.Count)"
Write-Output "Bundle: $bundlePath"
Write-Output "Bytes: $($bundle.Length)"
Write-Output "SHA256: $bundleHash"
