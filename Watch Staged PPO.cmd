@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" demo.py --controller ppo --checkpoint runs/staged_comm/checkpoint.pt
) else (
  python demo.py --controller ppo --checkpoint runs/staged_comm/checkpoint.pt
)
if errorlevel 1 pause


