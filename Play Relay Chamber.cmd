@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" demo.py --controller manual --relay
) else (
  python demo.py --controller manual --relay
)
if errorlevel 1 pause

