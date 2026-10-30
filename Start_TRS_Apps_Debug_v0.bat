@echo off
setlocal
cd /d "%~dp0"
set "APP=%~dp0src\roc_mcs\gui\trs_apps\trs_apps_app.py"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0venv\Scripts\python.exe"
if exist "%~dp0python_path.txt" (
  for /f "usebackq eol=# delims=" %%i in ("%~dp0python_path.txt") do set "PY=%%i"
)
if not exist "%PY%" set "PY=python"
"%PY%" "%APP%" %*
if errorlevel 1 pause
