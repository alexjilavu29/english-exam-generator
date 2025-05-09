@echo off
echo English Exam Generator Launcher
echo ==============================

:: Change to the directory where the batch file is located
cd /d "%~dp0"

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist venv (
    echo Setting up virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

:: Check if .env file exists
if not exist .env (
    echo Creating .env file...
    echo OPENAI_API_KEY=> .env
    echo OPENAI_MODEL=gpt-4o>> .env
)

:: Run the application
echo Starting English Exam Generator...
python app.py

:: Keep the window open if errors occur
if %errorlevel% neq 0 (
    echo An error occurred while running the application.
    pause
) 