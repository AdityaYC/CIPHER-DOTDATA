# PHANTOM CODE — Run the app (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Use venv if present
if (Test-Path ".venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

# Install deps if needed
try {
    python -c "import fastapi" 2>$null
} catch {
    Write-Host "Installing dependencies..."
    pip install -r backend/requirements.txt -q
}

# Ensure YOLO model exists
if (-not (Test-Path "models\yolov8_det.onnx")) {
    Write-Host "Downloading YOLO model..."
    python scripts/download_model.py
}

Write-Host "Starting PHANTOM CODE (HTTP — use http://localhost:8000 in browser)..."
Write-Host "  Tactical map:  http://localhost:8000"
Write-Host "  Live stream:   http://localhost:8000/live"
$env:PHANTOM_HTTP_ONLY = "1"
python backend/main.py
