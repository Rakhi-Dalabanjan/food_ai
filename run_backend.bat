@echo off
TITLE ShelfLife AI - Backend Server
cd /d "%~dp0"
echo ========================================================
echo             Starting Backend Server (FastAPI)
echo ========================================================
echo Backend URL: http://127.0.0.1:8000
echo.

if not exist "venv\Scripts\python.exe" (
    echo Python virtual environment not found. Setting up venv...
    python -m venv venv
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\pip.exe install -r requirements.txt
)

venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
