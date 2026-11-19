@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" demo.py --controller manual
) else (
  python demo.py --controller manual
)
if errorlevel 1 pause
