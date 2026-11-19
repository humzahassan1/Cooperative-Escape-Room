@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" demo.py --controller scripted --campaign
) else (
  python demo.py --controller scripted --campaign
)
if errorlevel 1 pause

