param(
    [string]$HostName = "ssh8.vast.ai",
    [int]$Port = 15873,
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_ed25519",
    [switch]$Follow
)

$remote = @'
set +e
echo "=== SUPERVISOR ==="
supervisorctl status no-r-decoder-pipeline no-r-d3-postrun-guardian decoder-download-qwen decoder-download-nemotron decoder-eval-setup
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
echo
echo "=== PIPELINE ==="
cat /workspace/safety-dataset/results/no_r_decoder_4080s/pipeline_state.json 2>/dev/null
echo
echo "=== POSTRUN GUARDIAN ==="
cat /workspace/safety-dataset/results/no_r_decoder_4080s/postrun/guardian_state.json 2>/dev/null
echo
echo "=== ACTIVE TRAIN PROGRESS ==="
for file in \
  /workspace/safety-dataset/results/no_r_decoder_4080s/train/*/progress.json \
  /workspace/safety-dataset/results/no_r_decoder_4080s/smoke/*/progress.json; do
  test -f "$file" || continue
  echo "--- $file"
  cat "$file"
done
echo
echo "=== PIPELINE LOG TAIL ==="
tail -n 50 /workspace/safety-dataset/logs/no-r-decoder-pipeline.log 2>/dev/null
tail -n 50 /workspace/safety-dataset/logs/no-r-decoder-pipeline.err.log 2>/dev/null
'@

& ssh -i $IdentityFile -p $Port -o BatchMode=yes -o ConnectTimeout=15 "root@$HostName" $remote

if ($Follow) {
    & ssh -tt -i $IdentityFile -p $Port -o BatchMode=yes -o ConnectTimeout=15 `
        "root@$HostName" `
        "tail -n 100 -F /workspace/safety-dataset/logs/no-r-decoder-pipeline.log /workspace/safety-dataset/logs/no_r_decoder_4080s/*.log"
}
