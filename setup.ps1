param([string]$Python = "")
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $Python) {
    foreach ($candidate in @('python', 'py')) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            $detected = & $command.Source -c "import sys; print(sys.executable if (3,11) <= sys.version_info[:2] < (3,14) else '')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $detected) { $Python = $detected.Trim(); break }
        }
    }
}
if (-not $Python) {
    $bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $bundled) { $Python = $bundled }
}
if (-not $Python) { throw 'Install 64-bit Python 3.11-3.13, then rerun setup.ps1 -Python <path-to-python.exe>.' }
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    & $Python -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creating the virtual environment failed.' }
}
& $projectPython -m pip install --disable-pip-version-check 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
& $projectPython -m pip install --disable-pip-version-check -e .
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
Write-Host 'Setup complete. Run: .\run_project.ps1'

