# Compatibility entry point: uses the same complete 200,000-step experiment.
param([string]$Output = "results/final_run")
& (Join-Path $PSScriptRoot 'run_project.ps1') -Output $Output

