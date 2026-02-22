# Build Drone UI — run this in a terminal where Node/npm are installed
Set-Location $PSScriptRoot
npm install
npm run build
Write-Host ""
Write-Host "Done. Restart the backend (or refresh) and open http://localhost:8000 (or 8080)"
