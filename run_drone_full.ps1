# Run Drone (main app) with Drone2 features. Backend on 8000, frontend on 5173.
# From repo root. Opens backend in new window, then frontend here. Open http://localhost:5173
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# Ensure Node is on PATH (so npm works)
if (Test-Path "${env:ProgramFiles}\nodejs") { $env:Path = "${env:ProgramFiles}\nodejs;$env:Path" }

# Free port 8000
$conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($conn) { $conn | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Start-Sleep -Seconds 1 }

# Find Python (3.12): use "python" if on PATH, else common install locations
$pythonExe = $null
try {
    if (Get-Command python -ErrorAction Stop) { $pythonExe = "python" }
} catch { }
if (-not $pythonExe) {
    foreach ($p in @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "$env:ProgramFiles\Python312\python.exe")) {
        if (Test-Path $p) { $pythonExe = $p; break }
    }
}
if (-not $pythonExe) {
    Write-Host "Python not found. Install Python 3.12 and run .\scripts\add_python_to_path.ps1 or set PATH." -ForegroundColor Red
    exit 1
}
# If we use a full path, add Python and Scripts to PATH for this session so backend and tools work
if ($pythonExe -match "\\") {
    $pyDir = Split-Path $pythonExe -Parent
    $scriptsDir = Join-Path $pyDir "Scripts"
    $env:Path = "$pyDir;$scriptsDir;$env:Path"
}

# Drone backend (includes Drone2: laptop webcam, YOLO, advisory, /api/status, Agent tactical query, etc.)
# PYTHONPATH must be repo root so "backend" (vector_db, query_agent) can be imported for Agent tab queries
$env:PYTHONPATH = $root
$env:PHANTOM_HTTP_ONLY = "1"
Write-Host "Drone backend (with Drone2 features) starting on http://localhost:8000 ..."
Start-Process -FilePath $pythonExe -ArgumentList "-m", "uvicorn", "Drone.local_backend.app:app", "--host", "0.0.0.0", "--port", "8000" -WorkingDirectory $root

Start-Sleep -Seconds 4

Write-Host ""
Write-Host "  >>> Open in browser:  http://localhost:5173  <<<" -ForegroundColor Green
Write-Host "  (Cipher: Manual = webcam + YOLO; Agent, Replay.)" -ForegroundColor Gray
Write-Host "  Ctrl+C here stops the frontend; close the backend window to stop the server." -ForegroundColor Gray
Write-Host ""
Set-Location "$root\Drone\frontend"
npm run dev
