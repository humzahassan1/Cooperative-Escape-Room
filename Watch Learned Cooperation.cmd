@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" demo.py --controller options --checkpoint runs/options/seed7_comm.json
) else (
  python demo.py --controller options --checkpoint runs/options/seed7_comm.json
)
if errorlevel 1 pause
