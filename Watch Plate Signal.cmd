@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" demo.py --controller plate --seed 42
) else (
  python demo.py --controller plate --seed 42
)
if errorlevel 1 pause

