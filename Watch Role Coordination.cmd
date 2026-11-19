@echo off
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if exist "runs\role_coordination\replay.json" (
  "%PY%" demo.py --controller roles --checkpoint runs/role_coordination/seed7_comm/checkpoint.pt --replay runs/role_coordination/replay.json
) else (
  "%PY%" demo.py --controller roles --checkpoint runs/role_coordination/seed7_comm/checkpoint.pt
)
if errorlevel 1 pause
