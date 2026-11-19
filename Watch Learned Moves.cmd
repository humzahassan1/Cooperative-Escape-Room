@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" demo.py --controller free
) else (
  python demo.py --controller free
)
if errorlevel 1 pause
