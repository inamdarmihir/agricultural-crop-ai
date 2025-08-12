@echo off
echo 🌾 Crop AI - Streamlit App Launcher
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo ✅ Virtual environment found, activating...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  No virtual environment found
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Check dependencies
echo Checking dependencies...
python run_app.py --check-only
if errorlevel 1 (
    echo ❌ Dependency check failed
    pause
    exit /b 1
)

REM Launch the app
echo.
echo 🚀 Launching Crop AI application...
echo 📱 The app will open in your browser automatically
echo 🛑 Press Ctrl+C to stop the app
echo.
python run_app.py

pause
