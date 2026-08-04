param(
    [string]$HostName = "ssh8.vast.ai",
    [int]$Port = 15873,
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_ed25519"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Coordinator = Join-Path $ProjectRoot "scripts\run_no_r_decoder_pipeline.py"

python -m py_compile $Coordinator
if ($LASTEXITCODE -ne 0) {
    throw "Local coordinator failed py_compile."
}

$target = "root@$HostName"
& scp -i $IdentityFile -P $Port -o BatchMode=yes -o ConnectTimeout=15 `
    $Coordinator "${target}:/workspace/run_no_r_decoder_pipeline.py"
if ($LASTEXITCODE -ne 0) {
    throw "Cannot upload patched coordinator. Confirm that the Vast instance is running."
}

$remote = @'
set -Eeuo pipefail
mv /workspace/run_no_r_decoder_pipeline.py /workspace/safety-dataset/scripts/run_no_r_decoder_pipeline.py
/venv/main/bin/python -m py_compile /workspace/safety-dataset/scripts/run_no_r_decoder_pipeline.py

test -f /workspace/safety-dataset/results/no_r_decoder_4080s/train/qwen/completed.marker
test -f /workspace/safety-dataset/results/no_r_decoder_4080s/train/nemotron/checkpoint-00000250/complete.marker
test -f /workspace/safety-dataset/results/download_nemotron.json
test -f /workspace/safety-dataset/.eval_setup_complete

supervisorctl clear no-r-decoder-pipeline >/dev/null 2>&1 || true
supervisorctl clear decoder-download-nemotron >/dev/null 2>&1 || true
supervisorctl start no-r-decoder-pipeline
sleep 8
supervisorctl status no-r-decoder-pipeline
cat /workspace/safety-dataset/results/no_r_decoder_4080s/pipeline_state.json
'@

& ssh -i $IdentityFile -p $Port -o BatchMode=yes -o ConnectTimeout=15 `
    $target $remote
if ($LASTEXITCODE -ne 0) {
    throw "Remote recovery failed. Inspect supervisor and pipeline logs."
}
