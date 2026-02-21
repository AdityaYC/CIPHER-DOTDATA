@echo off
cd /d "%~dp0"
call npm install
call npm run build
echo.
echo Done. Restart the backend or refresh and open http://localhost:8000
pause
