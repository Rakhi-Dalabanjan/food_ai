# Quick Start Guide - Running ShelfLife AI on a Client Machine

This guide explains how to run the **ShelfLife AI** application on a new Windows computer after unzipping the project file.

---

## 1. Prerequisites (One-Time Setup)

Before running the application for the first time, ensure the client machine has:

1. **Python 3.10+ (3.10, 3.11, or 3.12)**:
   - Download from: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - ⚠️ **IMPORTANT**: During installation, check the box that says **"Add python.exe to PATH"**.

2. **Node.js (v18 or higher LTS)**:
   - Download from: [https://nodejs.org/](https://nodejs.org/)
   - Complete standard installation with default settings.

---

## 2. Unzipping & Preparation

1. **Extract the ZIP file**: Right-click `food-shelf-life-ai_1.0-main.zip` and choose **Extract All...**.
2. **Unblock Windows Security**:
   - Right-click `start.bat` -> Select **Properties**.
   - At the bottom of the **General** tab, check the **Unblock** box.
   - Click **Apply** and **OK**.
   *(Repeat for `run_backend.bat` and `run_frontend.bat` if necessary).*

---

## 3. How to Launch the Application

### Method A: One-Click Start (Recommended)
Double-click **`start.bat`**.

The batch script will automatically:
- Create the Python environment (`venv`) and install dependencies.
- Install frontend packages (`node_modules`).
- Start the Backend server at `http://127.0.0.1:8000`.
- Start the Frontend server at `http://localhost:5173`.
- Open your default web browser automatically.

---

### Method B: Independent Windows Launch
If you prefer to start Backend and Frontend in separate windows:

1. **Start Backend**: Double-click `run_backend.bat`
   - Keeps the backend running on `http://127.0.0.1:8000`.
2. **Start Frontend**: Double-click `run_frontend.bat`
   - Keeps the frontend running on `http://localhost:5173`.

---

## 4. Troubleshooting Common Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **"Windows protected your PC"** | SmartScreen blocking downloaded script | Click **More info** -> Click **Run anyway** |
| **"'python' is not recognized"** | Python path not added to environment variables | Re-run Python Installer -> Select **Modify** -> Check **Add Python to PATH** |
| **"'npm' is not recognized"** | Node.js not installed | Install Node.js from [nodejs.org](https://nodejs.org/) |
| **Port 8000 / 5173 in use** | Server already running in background | Close previous command prompt windows or restart computer |
