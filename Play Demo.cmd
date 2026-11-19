@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" demo.py --controller scripted
) else (
  python demo.py --controller scripted
)
if errorlevel 1 pause
