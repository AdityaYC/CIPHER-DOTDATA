# Start only the Cipher backend (port 8000). Use when frontend is already running (e.g. npm run dev on 5173).
# From repo root.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# Prefer real Python installs; "python" on Windows is often the Store stub that says "Python was not found"
$pythonExe = $null
foreach ($p in @(
    "$env:LOCALAPPDATA\Microsoft\WindowsApps\python3.12.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "$env:ProgramFiles\Python312\python.exe",
    "$env:ProgramFiles\Python311\python.exe",
    "C:\env\Scripts\python.exe"
)) {
    if (Test-Path $p) { $pythonExe = $p; break }
}
if (-not $pythonExe) {
    try {
        $pyOut = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($pyOut) { $pythonExe = $pyOut.Trim() }
    } catch { }
}
if (-not $pythonExe) {
    try {
        $ver = & python --version 2>&1
        if ($ver -notmatch "Microsoft Store|not found") {
            $pythonExe = "python"
        }
    } catch { }
}
if (-not $pythonExe) {
    Write-Host "Python not found. Install Python 3.12 from python.org (not the Store)." -ForegroundColor Red
    Write-Host "  Then run: py -3.12 -m pip install -r Drone\local_backend\requirements.txt ultralytics" -ForegroundColor Gray
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
Write-Host "  Open Cipher at http://localhost:5173 (run 'npm run dev' in frontend/ if needed)." -ForegroundColor Gray
Set-Location $root
& $pythonExe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
