@echo off
setlocal
cd /d "%~dp0"
set "APP=%~dp0src\roc_mcs\gui\live_scans\live_scans_app.py"
if exist "%~dp0.venv314\Scripts\python.exe" (
    "%~dp0.venv314\Scripts\python.exe" "%APP%"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%APP%"
) else if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%APP%"
) else (
    python "%APP%"
)
if errorlevel 1 pause
