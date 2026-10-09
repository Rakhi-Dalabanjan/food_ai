@echo off
TITLE ShelfLife AI - Launcher
echo ========================================================
echo               Starting ShelfLife AI Application
echo ========================================================
echo.

cd /d "%~dp0"

REM Ensure uploads directory exists
if not exist "uploads" mkdir uploads

REM 1. Check Python Virtual Environment Validity
set RECREATE_VENV=0

if not exist "venv\Scripts\python.exe" (
    set RECREATE_VENV=1
) else (
    venv\Scripts\python.exe -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo Virtual environment paths belong to a different computer. Re-creating venv...
        rmdir /s /q venv
        set RECREATE_VENV=1
    )
)

if "%RECREATE_VENV%"=="1" (
    echo [1/4] Setting up Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Python was not found on this system. Please install Python 3.10, 3.11, or 3.12.
        pause
        exit /b 1
    )
    echo [1/4] Installing Python requirements from requirements.txt...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\pip.exe install -r requirements.txt
) else (
    echo [1/4] Python environment verified.
)

REM 2. Check Frontend Node Modules
if not exist "frontend\node_modules" (
    echo [2/4] Frontend packages not found. Installing node_modules...
    cd frontend
    call npm install
    cd ..
) else (
    echo [2/4] Frontend packages found.
)

echo.
echo ========================================================
echo  Backend:  http://127.0.0.1:8000
echo  Frontend: http://localhost:5173
echo ========================================================
echo.

REM 3. Launch Backend Server in dedicated window
echo [3/4] Launching Backend Server (FastAPI)...
start "ShelfLife AI - Backend" cmd /k ""%~dp0run_backend.bat""

REM Wait 3 seconds for backend to initialize
timeout /t 3 /nobreak >nul

REM 4. Open Web Browser and Launch Frontend
echo [4/4] Opening Web Browser and starting Frontend...
start http://localhost:5173

cd frontend
call npm run dev

pause
