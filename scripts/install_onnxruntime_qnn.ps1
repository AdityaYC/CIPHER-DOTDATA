# Install ONNX Runtime with QNN so YOLO and Depth run on the NPU (Task Manager will show NPU utilisation).
# Run from repo root. Use the same Python as run_drone_full.ps1 (e.g. py -3.12 or full path).

$ErrorActionPreference = "Stop"
$root = if ($PSScriptRoot) { Split-Path $PSScriptRoot -Parent } else { Get-Location }
Set-Location $root

# Resolve Python
$pythonExe = $null
foreach ($p in @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:ProgramFiles\Python312\python.exe"
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
    Write-Host "Python not found. Install Python 3.12 and run again." -ForegroundColor Red
    exit 1
}

Write-Host "Using Python: $pythonExe" -ForegroundColor Cyan
Write-Host ""

# Check if QNN already available
Write-Host "Checking for existing onnxruntime..." -ForegroundColor Yellow
$hasQnn = $false
try {
    $out = & $pythonExe -c "import onnxruntime as ort; print('QNN' if 'QNNExecutionProvider' in ort.get_available_providers() else '')" 2>$null
    if ($out -eq "QNN") { $hasQnn = $true }
} catch { }
if ($hasQnn) {
    Write-Host "onnxruntime-qnn already in use (QNN provider present). Nothing to do." -ForegroundColor Green
    exit 0
}

Write-Host "IMPORTANT: Close any running backend (run_drone_full.ps1) and other Python apps that use ONNX before continuing." -ForegroundColor Yellow
Write-Host "Otherwise you may get 'Access is denied' when installing." -ForegroundColor Yellow
Write-Host ""

# Uninstall standard onnxruntime first so onnxruntime-qnn can replace it without overwriting in-use DLLs
Write-Host "Uninstalling existing onnxruntime (if any)..." -ForegroundColor Yellow
$ErrorActionPreference = "Continue"
& $pythonExe -m pip uninstall onnxruntime -y 2>&1 | Out-Null
$ErrorActionPreference = "Stop"
Start-Sleep -Seconds 1

Write-Host "Installing onnxruntime-qnn (NPU support for Snapdragon)..." -ForegroundColor Yellow
& $pythonExe -m pip install onnxruntime-qnn --upgrade
if ($LASTEXITCODE -ne 0) {
    Write-Host "First attempt failed. Retrying with --user (install to user site-packages)..." -ForegroundColor Yellow
    & $pythonExe -m pip install onnxruntime-qnn --upgrade --user
}
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Install failed (often due to files in use). Try:" -ForegroundColor Red
    Write-Host "  1. Close ALL windows running the backend or Python, then run this script again." -ForegroundColor Red
    Write-Host "  2. Or run this PowerShell window as Administrator and run the script again." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Verifying QNN provider..." -ForegroundColor Yellow
& $pythonExe -c "
try:
    import onnxruntime as ort
    p = getattr(ort, 'get_available_providers', lambda: [])()
    if 'QNNExecutionProvider' in p:
        print('  OK: QNNExecutionProvider available. YOLO/Depth will use NPU.')
    else:
        print('  Providers:', list(p) if p else '(none)')
        if not p:
            print('  If backend still uses CPU, ensure no other onnxruntime is installed and restart the backend.')
except Exception as ex:
    print('  Check failed:', ex)
"
Write-Host ""
Write-Host "Done. Restart the backend (run_drone_full.ps1) so YOLO and Depth use the NPU." -ForegroundColor Green
