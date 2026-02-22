# Run this in YOUR terminal (PowerShell): cd to this folder then .\install_deps.ps1
# Or from repo root: .\Drone\local_backend\install_deps.ps1

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$pythonExe = $null
if (Get-Command python -ErrorAction SilentlyContinue) { $pythonExe = "python" }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $pythonExe = "py" }
elseif (Get-Command pip -ErrorAction SilentlyContinue) { & pip install -r requirements.txt; exit $LASTEXITCODE }
else {
    # Try common Windows install locations
    $paths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python313\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "$env:ProgramFiles\Python311\python.exe"
    )
    foreach ($p in $paths) {
        if (Test-Path $p) { $pythonExe = $p; break }
    }
}

if ($pythonExe) {
    & $pythonExe -m pip install -r requirements.txt
    exit $LASTEXITCODE
}

Write-Host "Python not found. Do one of:"
Write-Host "  1. Install Python from https://www.python.org/downloads/ and check 'Add Python to PATH'"
Write-Host "  2. Use 'Anaconda PowerShell Prompt' if you have Anaconda/Miniconda"
Write-Host "  3. Add your Python folder to PATH (e.g. Settings > Environment variables)"
exit 1
