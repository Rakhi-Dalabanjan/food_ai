@echo off
TITLE ShelfLife AI - Frontend Server
cd /d "%~dp0\frontend"
echo ========================================================
echo             Starting Frontend Server (Vite)
echo ========================================================
echo Frontend URL: http://localhost:5173
echo.

if not exist "node_modules" (
    echo Installing node_modules...
    call npm install
)

call npm run dev
pause
