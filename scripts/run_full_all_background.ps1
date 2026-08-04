$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m translator.full_run supervisor `
    --root $Root --model gemini-3.1-flash-lite `
    --group-count 5 --keys-per-group 2 --max-retries 4
exit $LASTEXITCODE
