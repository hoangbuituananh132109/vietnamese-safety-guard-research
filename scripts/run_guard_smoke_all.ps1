param(
    [string]$Python = "D:\Downloads\viettel_guardrail_b200_paper\.venv-rtx3050\Scripts\python.exe",
    [switch]$SkipModelRuns
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

Invoke-CheckedPython @("scripts\build_guard_smoke_manifest.py")
Invoke-CheckedPython @("scripts\patch_gliguard_tokenizer_compat.py")
Invoke-CheckedPython @(
    "-m", "pytest",
    "tests\test_guard_smoke_data.py",
    "tests\test_gliguard_helpers.py",
    "tests\test_mmbert_schema_basic.py",
    "-q", "-p", "no:cacheprovider",
    "--basetemp", "reports\guard_smoke\pytest_reproduction"
)

if (-not $SkipModelRuns) {
    Invoke-CheckedPython @(
        "scripts\run_mmbert_fixed_smoke.py",
        "--max-length", "256",
        "--batch-size", "4",
        "--epochs", "2",
        "--max-steps", "24",
        "--lora",
        "--output-dir", "reports\guard_smoke\mmbert_fixed_reproduction"
    )
    Invoke-CheckedPython @(
        "scripts\run_mmbert_schema_basic_smoke.py",
        "--max-length", "512",
        "--batch-size", "2",
        "--epochs", "2",
        "--max-steps", "12",
        "--output-dir", "reports\guard_smoke\mmbert_schema_basic_reproduction"
    )
    Invoke-CheckedPython @(
        "scripts\run_gliguard_smoke.py",
        "--max-steps", "8",
        "--max-words", "128",
        "--max-train-samples", "24",
        "--max-eval-samples", "12",
        "--precision", "auto",
        "--output-dir", "reports\guard_smoke\gliguard_lora_reproduction"
    )
}

Write-Host "Guard smoke pipeline completed successfully."
