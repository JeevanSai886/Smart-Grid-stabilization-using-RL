param([string]$Output = "results/final_run", [int]$Port = 8501)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$gridPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $gridPython)) { throw 'Run setup.ps1 first.' }
& $gridPython -m streamlit run dashboard.py --server.port $Port -- --run-dir $Output
if ($LASTEXITCODE -ne 0) { throw 'Dashboard stopped with an error. Read the message above.' }
