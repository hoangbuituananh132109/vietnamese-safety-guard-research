$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe -m translator.cli report `
  --source .\data\pilot\nemotron_en_train_pilot_50_diverse_v2.jsonl `
  --translated .\data\translated\pilot_50_vi_machine_best.jsonl `
  --summary .\reports\pilot_50_translation_best_summary.json `
  --html .\reports\pilot_50_translation_best_review.html

