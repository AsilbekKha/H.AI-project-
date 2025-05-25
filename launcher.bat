@echo off
cd /d "%~dp0"

:: Step 1: Create venv if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Step 2: Activate venv
call venv\Scripts\activate

:: Step 3: Install dependencies
if exist requirements.txt (
    echo Installing required packages...
    pip install -r requirements.txt
) else (
    echo No requirements.txt found, skipping pip install.
)

:: Step 4: Run Streamlit
streamlit run display.py

pause
