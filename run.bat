@echo off
REM One-command setup + launch for Windows: double-click or run  run.bat
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10+ is required: https://www.python.org/downloads/  ^(tick "Add Python to PATH"^)
  pause
  exit /b 1
)

if not exist venv (
  echo ^>^> Creating virtual environment
  python -m venv venv
)
call venv\Scripts\activate.bat

echo ^>^> Installing packages ^(first run takes a few minutes^)
pip install -q -r requirements.txt

if not exist .env (
  copy .env.example .env >nul
  echo ^>^> Created .env. Open it and paste your ANTHROPIC_API_KEY ^(optional: the app also runs in search-only mode^).
)

if not exist data\lakeshore_robotics_10k.txt python generate_sample_data.py

echo ^>^> Starting the app at http://localhost:8501  ^(Ctrl+C to stop^)
streamlit run app.py
