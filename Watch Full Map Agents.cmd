@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" demo.py --controller map
) else (
  python demo.py --controller map
)
if errorlevel 1 pause
