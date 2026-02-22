# Run Drone UI + backend (with Drone2 tactical features: camera, YOLO, advisory)
# From repo root so Drone2 backend is on path for camera/YOLO/advisory.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Ensure backend (Drone2) is on path
$env:PYTHONPATH = $PSScriptRoot
$env:PHANTOM_HTTP_ONLY = "1"

# Use port 8080 if 8000 is in use (e.g. other server running)
$port = 8000
try {
    $conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction Stop
    $port = 8080
    Write-Host "Port 8000 in use; using 8080 instead. Stop the other app if you want 8000."
} catch { }

Write-Host "Starting Drone backend + Drone2 features (camera, YOLO, advisory)..."
Write-Host "  Drone UI:  http://localhost:$port  (build first: cd Drone/frontend; npm run build)"
Write-Host "  Health:    http://localhost:$port/health"
python -m uvicorn backend.app:app --host 0.0.0.0 --port $port
