# Install all dependencies, download YOLO model, and download depth model.
# Run from repo root: .\scripts\setup_yolo_and_depth.ps1
# Use this when "python" is not found: .\scripts\install_deps.ps1 first, or run with same Python as run_drone_full.ps1

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$root = Split-Path $scriptDir -Parent

# Resolve Python (same logic as run_drone_full.ps1 / download_model.ps1)
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
    exit 1
}

Write-Host "Using Python: $pythonExe" -ForegroundColor Cyan
Set-Location $root

# 1) Install all dependencies
Write-Host ""
Write-Host "Step 1/3: Installing dependencies..." -ForegroundColor Yellow
& $pythonExe -m pip install -r "Drone\local_backend\requirements.txt" -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Failed: Drone\local_backend\requirements.txt" -ForegroundColor Red
    exit 1
}
& $pythonExe -m pip install -r "backend\requirements.txt" -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Failed: backend\requirements.txt" -ForegroundColor Red
    exit 1
}
& $pythonExe -m pip install ultralytics -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Failed: ultralytics" -ForegroundColor Red
    exit 1
}
# ONNX export (for models/yolov8_det.onnx) needs onnx
& $pythonExe -m pip install "onnx>=1.12.0,<2.0.0" -q
Write-Host "  Dependencies installed." -ForegroundColor Green

# 2) Download YOLO model (yolov8n.pt and optionally export to ONNX)
Write-Host ""
Write-Host "Step 2/3: Downloading YOLO model (yolov8n.pt)..." -ForegroundColor Yellow
& $pythonExe (Join-Path $scriptDir "download_model.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "  YOLO download/export had issues; you can still run the app (ultralytics will download .pt on first use)." -ForegroundColor Gray
} else {
    Write-Host "  YOLO model ready." -ForegroundColor Green
}

# 3) Download depth model (Depth Anything V2, cached by HuggingFace)
Write-Host ""
Write-Host "Step 3/3: Downloading depth model (Depth-Anything-V2-Small)..." -ForegroundColor Yellow
& $pythonExe (Join-Path $scriptDir "download_depth_model.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Depth model download had issues; app will run without optional depth." -ForegroundColor Gray
} else {
    Write-Host "  Depth model ready." -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup complete. Run: .\run_drone_full.ps1" -ForegroundColor Cyan
