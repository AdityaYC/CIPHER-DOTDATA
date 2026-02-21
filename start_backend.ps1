# Start only the Drone backend (port 8000). Use when frontend is already running (e.g. npm run dev on 5173).
# From repo root.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$pythonExe = $null
try { if (Get-Command python -ErrorAction Stop) { $pythonExe = "python" } } catch { }
if (-not $pythonExe) {
    foreach ($p in @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "$env:ProgramFiles\Python312\python.exe")) {
        if (Test-Path $p) { $pythonExe = $p; break }
    }
}
if (-not $pythonExe) {
    Write-Host "Python not found. Install Python 3.12 and add to PATH." -ForegroundColor Red
    exit 1
}
if ($pythonExe -match "\\") {
    $pyDir = Split-Path $pythonExe -Parent
    $scriptsDir = Join-Path $pyDir "Scripts"
    $env:Path = "$pyDir;$scriptsDir;$env:Path"
}

$env:PYTHONPATH = $root
$env:PHANTOM_HTTP_ONLY = "1"
Write-Host "Starting Cipher backend on http://localhost:8000 ..." -ForegroundColor Green
Write-Host "  Open Cipher at http://localhost:5173 (run 'npm run dev' in Drone/frontend if needed)." -ForegroundColor Gray
Set-Location $root
& $pythonExe -m uvicorn Drone.local_backend.app:app --host 0.0.0.0 --port 8000
