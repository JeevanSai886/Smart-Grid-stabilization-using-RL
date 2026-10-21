param([string]$Output = "results/final_run")
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$gridPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $gridPython)) { throw 'Run setup.ps1 first.' }
& $gridPython -m smart_grid run-all --config configs/final_experiment.yaml --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Pipeline stopped. Read the error above; partial artifacts were retained.' }
Write-Host "Report: $Output/evaluation/RESULTS.md"
Write-Host "Dashboard: .\run_dashboard.ps1 -Output `"$Output`""
Write-Host "Optional TensorBoard: .\.venv\Scripts\python.exe -m tensorboard.main --logdir $Output/tensorboard/agents --port 6006"

