# Add Python 3.12 to your user PATH so "python" and "pip" work in any terminal.
# Run once (PowerShell as yourself). Restart the terminal after running.
# Run from repo root: .\scripts\add_python_to_path.ps1

$pyDir = "$env:LOCALAPPDATA\Programs\Python\Python312"
if (-not (Test-Path "$pyDir\python.exe")) {
    $pyDir = "$env:ProgramFiles\Python312"
}
if (-not (Test-Path "$pyDir\python.exe")) {
    Write-Host "Python 3.12 not found. Install from https://www.python.org/downloads/release/python-3120/"
    exit 1
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -like "*$pyDir*") {
    Write-Host "Python 3.12 is already on your PATH."
    exit 0
}

# Add Python and Scripts (for pip) to user PATH
$newPath = "$pyDir;$pyDir\Scripts;$userPath"
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")
Write-Host "Added to PATH: $pyDir and $pyDir\Scripts"
Write-Host "Close and reopen your terminal (or restart Cursor), then run: python --version"
Write-Host "After that you can use: python -m pip install ... and .\run_drone_full.ps1"
$env:Path = "$pyDir;$pyDir\Scripts;$env:Path"
Write-Host ""
Write-Host "PATH updated for this session too. Try: python --version"
