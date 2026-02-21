# Run Drone (main app) with Drone2 features. Backend on 8000, frontend on 5173.
# From repo root. Opens backend in new window, then frontend here. Open http://localhost:5173
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# Ensure Node is on PATH (so npm works)
if (Test-Path "${env:ProgramFiles}\nodejs") { $env:Path = "${env:ProgramFiles}\nodejs;$env:Path" }

# Free port 8000
$conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($conn) { $conn | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Start-Sleep -Seconds 1 }

# Drone backend (includes Drone2: laptop webcam, YOLO, advisory, /api/status, etc.)
$env:PYTHONPATH = $root
$env:PHANTOM_HTTP_ONLY = "1"
Write-Host "Drone backend (with Drone2 features) starting on http://localhost:8000 ..."
Start-Process -FilePath "py" -ArgumentList "-m", "uvicorn", "Drone.local_backend.app:app", "--host", "0.0.0.0", "--port", "8000" -WorkingDirectory $root

Start-Sleep -Seconds 4

Write-Host ""
Write-Host "  >>> Open in browser:  http://localhost:5173  <<<" -ForegroundColor Green
Write-Host "  (Drone UI: Manual = webcam + tactical panel; Agent, Replay.)" -ForegroundColor Gray
Write-Host "  Ctrl+C here stops the frontend; close the backend window to stop the server." -ForegroundColor Gray
Write-Host ""
Set-Location "$root\Drone\frontend"
npm run dev
