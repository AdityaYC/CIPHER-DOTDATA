# One script to run the full Drone app (backend + frontend).
# First time: run .\scripts\install_deps.ps1 once to install Python packages.
# From repo root: .\run.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# Use run_drone_full.ps1 (it finds Python, sets PATH for session, starts backend + frontend)
& "$root\run_drone_full.ps1"
