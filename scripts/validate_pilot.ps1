$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe -m translator.cli validate `
  --source .\data\pilot\nemotron_en_train_pilot_50_diverse_v2.jsonl `
  --translated .\data\translated\pilot_50_vi_machine_best.jsonl

