$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
  py -3.10 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

