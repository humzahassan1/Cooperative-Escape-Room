@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" demo.py --controller ppo --checkpoint runs/communication/checkpoint.pt
) else (
  python demo.py --controller ppo --checkpoint runs/communication/checkpoint.pt
)
if errorlevel 1 pause

