$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe -m translator.cli translate `
  --input .\data\pilot\nemotron_en_train_pilot_50_diverse_v2.jsonl `
  --output .\data\translated\pilot_50_vi_dry.jsonl `
  --checkpoint .\data\checkpoints\pilot_50_dry_completed.jsonl `
  --failed-output .\data\review_queue\pilot_50_dry_failed.jsonl `
  --provider dry-run --limit 50

