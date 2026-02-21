# Install project dependencies using Python 3.12.
# Use this when "python" or "py" are not on PATH.
# Run from repo root: .\scripts\install_deps.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent

# Common Python 3.12 locations on Windows
$pyPaths = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:ProgramFiles\Python312\python.exe",
    "${env:ProgramFiles(x86)}\Python312\python.exe"
)

$python = $null
foreach ($p in $pyPaths) {
    if (Test-Path $p) {
        $python = $p
        break
    }
}

if (-not $python) {
    Write-Host "Python 3.12 not found in standard locations." -ForegroundColor Red
    Write-Host "Searched: $($pyPaths -join ', ')"
    Write-Host ""
    Write-Host "Either:"
    Write-Host "  1. Install Python 3.12 from https://www.python.org/downloads/release/python-3120/"
    Write-Host "  2. Or set PYTHON312_PATH and run again, e.g.:"
    Write-Host '     $env:PYTHON312_PATH = "C:\path\to\Python312\python.exe"; .\scripts\install_deps.ps1'
    exit 1
}

if ($env:PYTHON312_PATH -and (Test-Path $env:PYTHON312_PATH)) {
    $python = $env:PYTHON312_PATH
}

Write-Host "Using: $python"
Write-Host ""

Set-Location $root

Write-Host "Installing Drone/local_backend requirements..."
& $python -m pip install -r "Drone\local_backend\requirements.txt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Installing backend requirements..."
& $python -m pip install -r "backend\requirements.txt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Done. To run the app use the same Python:"
Write-Host "  & `"$python`" -m uvicorn Drone.local_backend.app:app --host 0.0.0.0 --port 8000"
Write-Host "Or add Python to PATH (see README or run add_python_to_path.ps1)."
