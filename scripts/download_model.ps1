# Run download_model.py using the same Python as run_drone_full.ps1 (avoids "Python was not found" Store stub).
# From repo root: .\scripts\download_model.ps1
$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$root = Split-Path $scriptDir -Parent

$pythonExe = $null
foreach ($p in @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "$env:ProgramFiles\Python312\python.exe",
    "$env:ProgramFiles\Python311\python.exe"
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
        if ($ver -notmatch "Microsoft Store|not found") { $pythonExe = "python" }
    } catch { }
}
if (-not $pythonExe) {
    Write-Host "Python not found. Install Python 3.12 from python.org (not the Store)." -ForegroundColor Red
    Write-Host "  Or run: py -3.12 scripts\download_model.py" -ForegroundColor Gray
    exit 1
}

Set-Location $root
& $pythonExe (Join-Path $scriptDir "download_model.py") @args
